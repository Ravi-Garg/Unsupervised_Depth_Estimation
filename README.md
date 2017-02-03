# Unsupervised Depth Estimation

This is the caffe implementation of our paper Unsupervised CNN for single view depth estimation: Geometry to the rescue published in ECCV 2016 with minor modifications. In this varient, we train the network end-to-end instead of coarse to fine-training with deeper networks and TVL1 loss for training. Both of which leads to substancial improvements in the results.

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

# Performence on KITTI 
We have resized the KITTI images to 160x608 for training - which changes the aspect ratio of the images. Thus for evaluation the KITTI images needs to be resized to this resolution and predicted disparities should be scaled by a factor of 608/width_of_input_image before computing.

The model gives following results on KITTI test-set used in our paper.
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

# Model Description
We have trained a small variant of the residual network 50 by 2 from scratch on KITTI.
Our model is <25 MB and predicts depths on 160x608 resolution images with 50Hz on nvidia TITAN X. This is 5x the frame rate of KITTI. It can be used with caffe without any modification and we provide a simple matlab wrapper for testing.

To train and finetune networks on your own data, you need to compile caffe with additional:
* “AbsLoss” layer for L1 loss minimization, 

* “Warping” layer for image warpping given flow

* and modified "filler.hpp" to compute image gradient with convolutions which we share here.

Our model is trained end-to-end from scratch with adam solver (momentum1 = 0.9 , momentom2 = 0.999, learning rate =10e-3 ) for 40,000 iterations on 4 gpus with batchsize 14 per GPU. This model is a pre-release and results with tuning hyperparameters should improve results. Left right flips described in the paper were used to train the provided network.Other agumentations or runtime shuffle were not used to train the model.

# Changes in Network Architecture

In terms of the network architecture, our architecture have minuet differences from standard Residual networks. We have replaced resnet’s strided convolutions with 2x2 MAX pooling layers like VGG. The first 7x7 convolution with stride 2 is replaced with the 7x7 convolution with no stride and the max pooled output at ½ resolution is passed through an extra 3x3 convolutional (128 features)->relu->2x2 pooling block. Rest of the network followes resnet 50 with half the parameters every layer.

For dense prediction we have followed the skip-connections as specified in FCN and our ECCV paper. 
We have introduced a scale layer with weight decay 0.01 before every 1x1 convolution of FCN skip-connections which allows us to merge mid level features more efficiently by
* Adaptively selecting the mid level features which are more correlated to depth of the scene.
* Making 1x1 convolutions for projections more stable for end to end training.

The model as described above, was trained on 23200 raw stereo pairs of KITTI taken from city, residential and road sequences. Images from other sequences of KITTI was left untouched and is used only for testing following the standard protocol. 


# License
For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.

