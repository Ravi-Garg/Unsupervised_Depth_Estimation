# Realtime Unsupervised Depth Estimation from an Image

This is the caffe implementation of our paper "Unsupervised CNN for single view depth estimation: Geometry to the rescue" published in ECCV 2016 with minor modifications. In this variant, we train the network end-to-end instead of in coarse to fine manner with deeper network (Resnet 50) and TVL1 loss instead of HS loss. 

With the implementation we share the sample Resnet50by2 model trained on KITTI training set:

https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation/blob/master/model/train_iter_40000.caffemodel


Shared model is a small variant of the 50 layer residual network from scratch on KITTI.
Our model is **<25 MB** and predicts depths on 160x608 resolution images **at over 30Hz** on Nvidia Geforce GTX980 (50Hz on TITAN X). It can be used with caffe without any modification and we provide a simple matlab wrapper for testing.

Click on the image to watch preview of the results on youtube:

[![Screenshot](https://img.youtube.com/vi/Fut885qvBvQ/0.jpg)](https://www.youtube.com/watch?v=Fut885qvBvQ)

If you use our model or the code for your research please cite:
```
@inproceedings{garg2016unsupervised,
  title={Unsupervised CNN for single view depth estimation: Geometry to the rescue},
  author={Garg, Ravi and Kumar, BG Vijay and Carneiro, Gustavo and Reid, Ian},
  booktitle={European Conference on Computer Vision},
  pages={740--756},
  year={2016},
  organization={Springer}
}
```

# Training Procedure
This model was trained on 23200 raw stereo pairs of KITTI taken from city, residential and road sequences. Images from other sequences of KITTI were left untouched and were used only for testing following the standard protocol.

Our model is trained end-to-end from scratch with adam solver (momentum1 = 0.9 , momentom2 = 0.999, learning rate =10e-3 ) for 40,000 iterations on 4 gpus with batchsize 14 per GPU. This model is a pre-release further tuning of hyperparameters should improve results. Only left-right flips as described in the paper were used to train the provided network. Other agumentations described in the paper and runtime shuffle were not used but should also lead to performance imrovement.

Here is the training loss recorded per 20 iterations: 

<img src="https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation/blob/master/model/train.png" alt="loss per 20 iterations " width="400">

Note: We have resized the KITTI images to 160x608 for training - which changes the aspect ratio of the images. Thus for proper evaluation on KITTI the images needs to be resized to this resolution and predicted disparities should be scaled by a factor of 608/width_of_input_image before computing depth. For ease in citing the results for further publications, we share the performance measures.

Our model gives following results on KITTI test-set without any post processing: 
---------------------------------------------------------------------

RMSE(linear):   4.400866

RMSE(log)   :   0.233548

RMSE(log10)   :   0.101441

Abs rel diff:   0.137796

Sq rel diff :   0.824861

accuracy THr 1.25 :   0.809765

accuracy THr 1.25 sq:   0.935108

accuracy THr 1.25 cube:   0.974739

---------------------------------------------------------------------


#Network Architecture

Architecture of our networks closely follow Residual networks scheme. We start from [resnet 50 by 2](https://github.com/jay-mahadeokar/pynetbuilder/tree/master/models/imagenet)
 architecture and have replaced strided convolutions with 2x2 MAX pooling layers like VGG. The first 7x7 convolution with stride 2 is replaced with the 7x7 convolution with no stride and the max-pooled output at ½ resolution is passed through an extra 3x3 convolutional (128 features)->relu->2x2 pooling block. Rest of the network followes resnet50 with half the parameters every layer.

For dense prediction we have followed the skip-connections as specified in FCN and our ECCV paper. 
We have introduced a learnable scale layer with weight decay 0.01 before every 1x1 convolution of FCN skip-connections which allows us to merge mid-level features more efficiently by:

* Adaptively selecting the mid-level features which are more correlated to depth of the scene.
* Making 1x1 convolutions for projections more stable for end to end training.

Further analysis and visualizations of learned features will be released shortly on the arxiv:
https://arxiv.org/pdf/1603.04992v2.pdf

# Using the code

To train and finetune networks on your own data, you need to compile caffe with additional:
* “AbsLoss” layer for L1 loss minimization, 

* “Warping” layer for image warpping given flow

* and modified "filler.hpp" to compute image gradient with convolutions which we share here.


# License
For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.

# Contact
Please report any known issues on this thread of to ravi.garg@adelaide.edu.au


