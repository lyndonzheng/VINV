# Visiting the Invisible
[Paper]() |[ArXiv](https://arxiv.org/pdf/2104.05367.pdf) | [Project Page](https://chuanxiaz.com/vinv/) | [Video](https://www.youtube.com/watch?v=QSAYxrKgn7A)

This repository implements the training and testing for "Visiting the Invisible: Layer-by-Layer Completed Scene Decomposition" by [Chuanxia Zheng](https://www.chuanxiaz.com), Duy-Son Dao (instance segmentation),[Guoxian Song](https://guoxiansong.github.io/homepage/index.html) (data rendering), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/) and [Jianfei Cai](https://jianfei-cai.github.io/). 

## Example
<img src="images/featured.gif" align="center">

Example results of scene decomposition and recomposition. Given a single RBG image, the proposed **CSDNet** model is able to structurally decompose the scene into semantically completed instances, and background, while completing the RGB appearance for previously **invisible** regions, such as the cup. The completely decomposed instances can be used for image editing and scene recomposition, such object removal and moving without manually input annotations.

## Getting started

### Requirements

- The code architecture is based on [mmdetection](https://github.com/open-mmlab/mmdetection) (Version: 1.0rc1+621ecd2) and [mmcv](https://github.com/open-mmlab/mmcv) (Version: 0.2.15), please see [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) for the installation details. We tried to update version to the latest one, but they are failed due to many functions are different between different versions.

### Installation

- The original code was tested with Pytorch 1.4.0, CUDA 10.0, Python 3.6 and Ubuntu 16.04 (18.04 is also supported)
```
conda create -n viv python=3.6 -y
conda activate viv
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```
- Install the [mmdetection](https://github.com/open-mmlab/mmdetection) (Version: 1.0rc1+621ecd2) and [mmcv](https://github.com/open-mmlab/mmcv) (Version: 0.2.15)
```
pip install Cython==0.29.21
pip install mmcv==0.2.15
pip install -r requirements.txt
pip install -v -e .
```

## Datasets

- ``CSD``: Our rendered synthetic dataset, which contains 8,298 images, 95,030 instances for training and 1,012 images, 11,648 instances for testing. The dataset is built upon [SUNCG](https://sscnet.cs.princeton.edu/). When we built the dataset (more than half year), SUNCG dataset is publicly available.
- [COCOA](https://github.com/Wakeupbuddy/amodalAPI): is annotated from [COCO2014](https://github.com/Wakeupbuddy/amodalAPI), in which 5,000 images are selected to manually label with pairwise occlusion orders and amodal masks.
- [KINS](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset): is derived from [KITTI](http://www.cvlibs.net/datasets/kitti/), in which 14,991 images are labeled with absolute layer orders and amodal masks.

## Testing

- Test the model
```
cd tools
bash test.sh
```
- The testing and evaluation configuration can be found in ``test.py`` file.
- Please select the corresponding configuration and pre-trained model for each dataset.
- More settings needs to be modified in the code.
- Single image visualization testing (demo). Please modify the configuration for the different inputs.
```
cd demo
python predictor.py
```

## Training

- Train a model (three phases in synthetic dataset)
```
cd tools
bash tran.sh
```
- Configuration files are stored in ``configs/rgba`` directory.
- The synthetic model is trained in three phases: **decomposition**, **completion**, and **end**, which can be set in the corresponding configure file by set *mode*.
- More settings are followed as the previous works [Mask-RCNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn), [HTC](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc) in [MMdetection](https://github.com/open-mmlab/mmdetection) and [PICNet](https://github.com/lyndonzheng/Pluralistic-Inpainting).

## Pretrained Models
Download the pre-trained models using the following links and put them under ``checkpoints`` directory.

- [CSD]() | [COCOA]() | [KINS]()

## Citation
If you find our code or paper useful, please cite our paper.
```
@article{zheng2021vinv,
	title={Visiting the Invisible: Layer-by-Layer Completed Scene Decomposition},
	author={Zheng, Chuanxia and Dao, Duy-Son and Song, Guoxian and Cham, Tat-Jen and Cai, Jianfei},
	journal={International Journal of Computer Vision},
	pages={},
  year={2021},
  publisher={Springer}
}
```

