#!/usr/bin/env sh

TOOLS=../../../build/tools

$TOOLS/caffe train\
  --solver=solver_resnet50by2_pooladam_tvl1.prototxt\
  --gpu=0,1,2,3\
  --log_dir=logs/
  
  # run it from caffe/matlab/demo folder
