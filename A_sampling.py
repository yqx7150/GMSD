from random import betavariate
import sys
sys.path.append('..')
import functools
import matplotlib.pyplot as plt
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse
import odl
import glob
import pydicom
from cv2 import imwrite,resize
from func_test import WriteInfo
from scipy.io import loadmat,savemat
from radon_utils import (create_sinogram,bp,filter_op,
                        fbp,reade_ima,write_img,sinogram_2c_to_img,
                        padding_img,unpadding_img,indicate)
from time import sleep                        

import odl

Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-180, 180, 720)
# Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
#                             src_radius=500, det_radius=500)
Fan_geometry = odl.tomo.Parallel2dGeometry(Fan_angle_partition, Fan_detector_partition)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)

_CORRECTORS = {}
_PREDICTORS = {}


def set_predict(num):
  if num == 0:
    return 'None'
  elif num == 1:
    return 'EulerMaruyamaPredictor'
  elif num == 2:
    return 'ReverseDiffusionPredictor'

def set_correct(num):
  if num == 0:
    return 'None'
  elif num == 1:
    return 'LangevinCorrector'
  elif num == 2:
    return 'AnnealedLangevinDynamics'

def padding_img(img):
    b,w,h = img.shape
    h1 = 768
    tmp = np.zeros([b,h1,h1])
    x_start = int((h1 -w)//2)
    y_start = int((h1 -h)//2)
    tmp[:,x_start:x_start+w,y_start:y_start+h] = img
    return tmp

def unpadding_img(img):
    b,w,h = img.shape[0],720,720
    h1 = 768
    tmp = np.zeros([b,h1,h1])
    x_start = int((h1 -w)//2)
    y_start = int((h1 -h)//2)
    return img[:,x_start:x_start+w,y_start:y_start+h]

def init_ct_op(img,r):
  batch = img.shape[0]
  sinogram = np.zeros([batch,720,720])
  # 保存东西用的，占显存多，assert false一下
  # ttt = [4,6,8,12,14,16,18,10]
  # wt = [0,2,4,6,8,10]
  #
  # t_sinogram = Fan_ray_trafo(img[2,...]).data
  # for u in ttt:
  #   mask = np.zeros([720,720])
  #   tu = np.zeros([720,720])
  #   mask[::u,:] = 1
  #   for w in wt:
  #     if w >= u:
  #       continue
  #     ind = np.arange(0,720,u)
  #     for ii in range(w):
  #       # ind += ii
  #       ind_n = ind + ii
  #       ind_n = np.clip(ind_n,0,719)
  #
  #       mask[ind_n,:] = 1
  #     tu = t_sinogram * mask
  #     savemat('./sparse_sinogram_img/R_{}_W_{}.mat'.format(u,w),{"Img":tu})
  #     write_img(tu,'./sparse_sinogram_img/R_{}_W_{}.png'.format(u,w))

  sparse_sinogram = np.zeros([batch,720,720])
  ori_img = np.zeros_like(img)
  sinogram_max = np.zeros([batch,1])
  for i in range(batch):
    sinogram[i,...] = Fan_ray_trafo(img[i,...]).data
    ori_img[i,...] = Fan_FBP(sinogram[i,...]).data
    sinogram_max[i,0] = sinogram[i,...].max()
    # sinogram[i,...] /= sinogram_max[i,0]
    t = np.copy(sinogram[i,::r,:])
    sparse_sinogram[i,...] = resize(t,[720,720])
  
  return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32),sinogram_max


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method # pc
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean

#===================================================================== ReverseDiffusionPredictor 
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean
#=====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x

#================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean
#==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x

#========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)

#========================================================================================================

def get_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(img_model,check_num,predict,correct):
    # path = glob.glob("./Test_CT/*")
    path = ["../Test_CT/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA"]
    file_name = ['0319', '0160', '0001', '0354', '0405', '0292', '0462', '0137', '0521', '0062', '0248', '0361']
    for p in path[:1]:
      # file_name = p.split('/')[-1].split('.')
      # file_name = "{}_{}".format(file_name[0],file_name[3])
      psnr_list = []
      mse_list = []
      ssim_list = []
      # R = np.array([6,8,12])
      # R = np.array([12,8,6,4])
      R = np.array([6])
      views = 720//R
      for rid, r in enumerate(R):
        view = views[rid]
        with torch.no_grad():
          # dataset=pydicom.read_file(p)
          # img = dataset.pixel_array.astype(np.float32)    
          # max_deg = img.max()
          # img /= max_deg
          img = np.load('./Test_CT/batch_img.npy')
          img = img[:2,:,:]
          # img = img[None,...].repeat(2,0)
          # print(max_deg)
          # assert 0
          ori_img, sparse_sinogram, sinogram, sinogram_max = init_ct_op(img,r)
          write_img(sinogram[0,...])
          # print(sinogram.shape)
          # print(sinogram.max())
          # print(sinogram.min())
          # np.save('Par_full_sinogram.npy',sinogram)
          # assert False
          # ori_img = ori_img[2:4,...]
          # sparse_sinogram = sparse_sinogram[2:4,...]
          # sinogram = sinogram[2:4,...]
          # print(np.max(sinogram[0,...]))
          # assert False
          # sinogram_max = sinogram_max[2:4,...]
          batch_size = ori_img.shape[0]
          # write_img(ori_img[2,...])
          # assert False
          # print(ori_img[0,...].max())
          # print(ori_img[0,...].min())
          # print(sinogram[0,...].max())
          # print(sinogram[0,...].min())
          # print(sinogram_max)
          # assert False
          # np.save('FBP_img.npy',ori_img)
          # assert False
          # tt = np.copy(sinogram)
          sinogram = padding_img(sinogram).astype(np.float32)
          sparse_sinogram = padding_img(sparse_sinogram).astype(np.float32)
          sinogram = torch.from_numpy(sinogram).cuda()
          x0 = torch.from_numpy(np.array(sparse_sinogram[:,None,...])).cuda()
          # x0 = torch.from_numpy(378*np.random.random([1,1,768,768]).astype(np.float32)).cuda()

          timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
          psnr_list = np.zeros([2000,1])
          max_psnr = np.zeros(batch_size)
          max_ssim = np.zeros(batch_size)
          min_mse = 999*np.ones(batch_size)
          rec_img = np.zeros_like(ori_img)
          best_img = np.zeros_like(ori_img)

          for i in range(sde.N):
            if i>750 :
            # if i < 500 or i>510:
              continue
            t = timesteps[i]
            # print(timesteps)
            # assert False
            vec_t = torch.ones(x0.shape[0], device=t.device) * t   
            # print(x0.shape)
            # assert False

            x01, x0 = predictor_update_fn(x0, vec_t, model=img_model)



            ## DC
            # x0 = torch.clip(x0,0,300)
            # x0 += torch.mean(sinogram) - torch.mean(x0)

            x0[:,0,::r,:] = sinogram[:,::r,:]
            # x0 = torch.clip(x0,0,378)
            x01, x0 = corrector_update_fn(x0, vec_t, model=img_model)
            # x0 = torch.clip(x0,0,300)
            # x0 += torch.mean(sinogram) - torch.mean(x0)
            ## DC
            x0[:,0,::r,:] = sinogram[:,::r,:]
            # x0 = torch.clip(x0,0,378)

            # if i==0 or (i>350 and i<751 ) or (i>1800):
            if True:
              tmp = np.squeeze(x0.detach().cpu().numpy())
              tmp = unpadding_img(tmp)
              for coil in range(batch_size):
                rec_img[coil,...] = Fan_FBP(tmp[coil,...]*sinogram_max[coil,...]).data
              
              write_img(tmp[0,...],'rec_sinogram.png')
              write_img(rec_img[0,...])

              psnr0,ssim0,mse0 = indicate(rec_img,ori_img)

              # psnr_list[i,0] = psnr0
              c = max_psnr < psnr0
              # print(c)
              # print(np.sum(c))
              if np.sum(c) > 0.01:
                max_psnr = max_psnr*(1-c) + psnr0*c
                max_ssim = max_ssim*(1-c) + ssim0*c
                min_mse = min_mse*(1-c) + mse0*c
                min_mse = min_mse*(1-c) + mse0*c
                c = c[...,None,None]
                best_img = best_img * (1-c) + rec_img*c

            print("Step: {}   Views:{}  PSNR:{}  MSE:{}".format(i, view, np.round(psnr0[:4],3),np.round(1000*mse0[:4],3)))

          print("MAX:  PSNR:{} MSE:{}".format(np.round(max_psnr[:4],3),np.round(1000*min_mse[:4],3)))
          # assert False
          np.save('./batch_result/check_{}_view_{}_batch_img.npy'.format(check_num,view),best_img)


          for file_id,name in enumerate(file_name):
            WriteInfo(
              "./batch_result/result.csv", 
              Vierw=view,
              PSNR=max_psnr[file_id], SSIM=max_ssim[file_id], MSE=min_mse[file_id],
              File=name,Check_num=check_num,Predictor=set_predict(predict),Corrector=set_correct(correct))

  return pc_sampler

def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                    rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
