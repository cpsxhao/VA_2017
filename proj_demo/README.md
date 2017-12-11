<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


# 视听信息导论 2017Fall 课程设计  

## 文件说明
这个版本是一个仅对助教提供的特征进行分类的小型网络，数据预处理、评估、Loss 函数等内容都没有更改。

## Requirements
* [PyTorch](http://pytorch.org/)

* [torchvision](https://github.com/pytorch/vision)

## pytorch 使用说明

* 建议使用 anaconda 或 miniconda 管理 python 环境及 packages。  
* [Pytorch 官方网站](http://pytorch.org) 提供了相当好的教程用于入门，推荐之前没有使用过 Pytorch 的研究人员先阅读。

## 目录配置

* 将 [configs/train_config.yaml](https://github.com/cpsxhao/VA_2017/blob/master/proj_demo/configs/train_config.yaml) 中的 **data_dir** 和 **flist** 更改为你自己的训练集文件夹和文件名列表所在的路径

* 将 [configs/test_config.yaml](https://github.com/cpsxhao/VA_2017/blob/master/proj_demo/configs/test_config.yaml) 中的 **data_dir**, **video_flist** 和 **audio_flist**  更改为你自己的测试集文件夹和文件名列表所在的路径

* 将 [configs/test_config.yaml](https://github.com/cpsxhao/VA_2017/blob/master/proj_demo/configs/test_config.yaml) 中的 **init_model** 更改为你要评估的模型文件位置

## 数据集说明

* 数据集共包括1300对样本，每对样本为125秒的视频和对应的音频文件。

* 助教提供了使用 CNN 提取的特征，这些特征是以 1s 的采样周期采样，并对每帧图片和声音分别使用预训练好的 GoogLeNet Inception V3 和 VGGish 网络提取的。视频特征每个样本为 $1024\times120$ 维，音频特征每个样本为 $128\times120$ 维。

* 训练数据集可以在TA提供的[百度云地址](https://pan.baidu.com/s/1qY2uyhI)下载，或者向 cpsxhao@163.com 发邮件索取。

## Train your model
```
python train.py    
```

## Evaluate your model
```
python evaluate.py
```

## Self-defined model
```
modify model.py to configure your own model, and change the hyper-parameters in the config files (configs/train_config.yaml)
```

## Tested Environments  
Ubuntu16.04+python3.6+pytorch0.2.0+GPU/CPU    
