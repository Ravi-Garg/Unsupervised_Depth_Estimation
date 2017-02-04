#ifndef CAFFE_WARPING_LAYER_HPP_
#define CAFFE_WARPING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Compute forward warp image given,
 *        input image, horizontal flow, vartical flow.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class WarpingLayer : public Layer<Dtype> {
 public:
  explicit WarpingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Warping layer outputs a warp image given a input and 2 flow vectors;
  
  virtual inline const char* type() const { return "Warping";}
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //int kernel_h_, kernel_w_;
  //int stride_h_, stride_w_;
  //int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  
  //Blob<Dtype> warp_image;

  /*int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;*/
};

}  // namespace caffe

#endif  // CAFFE_WARPING_LAYER_HPP_
