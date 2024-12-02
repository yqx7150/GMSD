# GMSD

**Paper**: Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction           

**Authors**: Bing Guan; Cailian Yang; Liu Zhang; Shanzhou Niu; Minghui Zhang; Yuhao Wang; Weiwen Wu; Qiegen Liu          

IEEE Transactions on Radiation and Plasma Medical Sciences       
vol. 8, no. 2, pp. 195-207, 2024.     
https://ieeexplore.ieee.org/document/10233041      

The radiation dose in computed tomography (CT) examinations is harmful for patients but can be significantly reduced by intuitively decreasing the number of projection views. Reducing projection views usually leads to severe aliasing artifacts in reconstructed images. Previous deep learning (DL) techniques with sparse-view data require sparse-view/full-view CT image pairs to train the network with supervised manners. When the number of projection view changes, the DL network should be retrained with updated sparse-view/full-view CT image pairs. To relieve this limitation, we present a fully unsupervised score-based generative model in sinogram domain for sparse-view CT reconstruction. Specifically, we first train a score-based generative model on full-view sinogram data and use multi-channel strategy to form high-dimensional tensor as the network input to capture their prior distribution. Then, at the inference stage, the stochastic differential equation (SDE) solver and data-consistency step were performed iteratively to achieve full-view projection. Filtered back-projection (FBP) algorithm was used to achieve the final image reconstruction. Qualitative and quantitative studies were implemented to evaluate the presented method with several CT data. Experimental results demonstrated that our method achieved comparable or better performance than the supervised learning counterparts. 
    
## Training
```bash
python main.py --config=aapm_sin_ncsnpp_gb.py --workdir=exp --mode=train --eval_folder=result
```

## Test
```bash
python A_PCsampling_demo.py

test默认调用exp_demo下的模型

--workdir=exp_zl
--mode=train
--eval_folder=result
--config=aapm_sin_ncsnpp_gb.py
```

## Graphical representation


<div align="center"><img width="500" alt="image" src="https://github.com/yqx7150/GMSD/assets/26964726/89bc9c85-03d2-4e59-9d19-f03ee02584eb"> </div>

<div align="center"> Linear measurement processes for sparse-view CT.

<div align="center"><img width="500" alt="image" src="https://github.com/yqx7150/GMSD/assets/26964726/6b3e8c35-b026-448f-8465-b940ec71e85a"> </div>

Visualization of the intermediate reconstruction process of GMSD. As the level of artificial noise becomes smaller, the reconstruction results tend to ground-truth data.

<div align="center"><img width="800" alt="image" src="https://github.com/yqx7150/GMSD/assets/26964726/761bf7db-8e92-4248-9f1b-715f38dc1966"> </div>

The proposed unsupervised deep learning in sinogram domain for sparse-view CT. Top: Training stage to learn the gradient distribution via denoising score matching. Bottom: Iterate between numerical SDE solver and data-consistency step to achieve reconstruction. DC stands for the data-consistency items.

<div align="center"><img width="800" alt="image" src="https://github.com/yqx7150/GMSD/assets/26964726/89d91942-0925-4571-b001-283eef235825"> </div>
    
 (a)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (e) 
 
Reconstruction results from 120 views using different methods. (a) The reference image versus the images reconstructed by (b) FBP, (c) FISTA-TV, (d) SART-TV, and (e) GMSD. The display windows are [-240,360]. The second row is residuals between the reference images and reconstruction images.

<div align="center"><img width="800" alt="image" src="https://github.com/yqx7150/GMSD/assets/26964726/531efef1-7f49-4e24-b3bc-8559565c1529"> </div>
    
 (a)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 (b)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 (c)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 (d)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 (e)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 (f)
 
Reconstruction images from 100 views using different methods. (a) The reference image versus the images reconstructed by (b) FBP, (c) U-Net, (d) FBPConvNet, (e) DRONE, and (f) GMSD. The second row is residuals between the reference images and reconstruction images.


## Test Data
In file './Test_CT', 12 sparse-view CT data from AAPM Challenge Data Study.



## Other Related Projects  
<div align="center"><img src="https://github.com/yqx7150/OSDM/blob/main/All-CT.png" >  </div>   
    
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
   
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * DP-MDM: Detail-Preserving MR Reconstruction via Multiple Diffusion Models  
[<font size=5>**[Paper]**</font>](http://arxiv.org/abs/2211.13857)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DP-MDM)
     
  * MSDiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2405.05763)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MSDiff)
    
  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)       
       
  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  
