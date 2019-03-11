import matplotlib
matplotlib.use('Agg') # if on remote server
import argparse
import os
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--eigen_test_files', type=str, default='./eigen_test_files', help='697 eigen split test images')
parser.add_argument('--dump_root', type=str, default='./dumps', help='directory to output disp/depth')
parser.add_argument('--output_disp', action='store_true', help='output disparity')
parser.add_argument('--output_depth', action='store_true', help='otuput depth')

args = parser.parse_args()

caffe.set_mode_gpu()
gpu_id = 0
caffe.set_device(gpu_id)

net_model = "./scripts/deploy_resnet50by2_pool.prototxt"
net_weights = "./model/train_iter_40000.caffemodel"

net = caffe.Net(net_model, net_weights, caffe.TEST)

imgs = os.listdir(args.eigen_test_files)

if not os.path.exists(args.dump_root):
    os.makedirs(args.dump_root)

for i, img in enumerate(imgs):
    im = cv2.imread(os.path.join(args.eigen_test_files, img)).astype(np.float32)
    im = cv2.resize(im, (608, 160))
    im_mean = np.mean(im)
    im -= im_mean
    net.blobs['imL'].reshape(1, im.shape[2], im.shape[0], im.shape[1]) # (1, 3, H, W)
    net.blobs['imL'].data[...] = np.transpose(im[:, :, :, np.newaxis], (3, 2, 0, 1))
    net.forward()
    A = net.blobs['h_flow'].data
    disp = A[0][0] # (H, W)
    file_name = os.path.splitext(img)[0]
    fig = plt.imshow(disp, cmap='plasma')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(args.dump_root, file_name+'_disp.png'), bbox_inches='tight', pad_inches=0)
