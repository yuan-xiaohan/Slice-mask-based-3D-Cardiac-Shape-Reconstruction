# Slice-mask based 3D Cardiac Shape Reconstruction from CT volume
This repository contains the code for the ACCV'2022 paper "Slice-mask based 3D Cardiac Shape Reconstruction from CT volume". 
\[[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Yuan_Slice-mask_based_3D_Cardiac_Shape_Reconstruction_from_CT_volume_ACCV_2022_paper.pdf)\]
<img src="https://github.com/yuan-xiaohan/Slice-mask-based-3D-Cardiac-Shape-Reconstruction/blob/main/Pipeline.png" alt="drawing"/>


## Introduction
An accurate 3D ventricular model is essential for diagnosing and analyzing cardiovascular disease. It is challenging to obtain accurate patient-specific models on scarce data via widely accepted deep-learning methods. 

In this paper, To fully use the characteristics of medical volume-based images, we present a slice-mask representation to better regress the parameters of the 3D model. A data synthesis strategy is proposed to alleviate the lack of training data by sampling in the constructed statistical shape model space and obtaining the corresponding slice-masks. We train the end-to-end structure by combining the segmentation and parametric regression modules. 

## Dataset
We establish a left ventricular CT dataset of a healthy population. It consists of input images after view-planning and corresponding segmentation masks, as well as 3D models. 
The download link is: 
https://drive.google.com/drive/folders/1OxP3ciMpSlgGDOmsOe7To2ZCkg8sv4aE?usp=sharing

We also provide the data needed to generate 3D shapes from shape parameters.
The download link is: 
https://drive.google.com/drive/folders/199GekzrsF8VbNc2tNvpwbcroalTAjBC5


## Usage
### Data Layout
Each folder in the <train> and <test> represents an instance. Including:

slice: 13 slices of specific views.

slice-mask: 13 slice-masks.

shape.obj: 3D model of the shape.

theta_nor.txt: parameter theta.

### Training a Model
    train_main.py --train TRAIN_DIRECTORY --save SAVE_DIRECTORY --info PCA_INFO_DIRECTORY
                     [--pretrained PRETRAINED_DIRECTORY]

### Testing a Model
    test_main.py --test TEST_DIRECTORY --save SAVE_DIRECTORY --info PCA_INFO_DIRECTORY
    
## Citation
If you find our work is useful or want to use our dataset, please consider citing the paper.
```
@inproceedings{yuan2022slice,
  title={Slice-mask based 3D Cardiac Shape Reconstruction from CT volume},
  author={Yuan, Xiaohan and Liu, Cong and Feng, Fu and Zhu, Yinsu and Wang, Yangang},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={1909--1925},
  year={2022}
}
```
