#include <cfloat>
#include <vector>

#include "caffe/layers/warping_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {



template <typename Dtype>
__global__ void WarpingForward(const int nthreads,
    const Dtype* const I,const Dtype* const u,const Dtype* const v, const int num, const int channels,
    const int height, const int width, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    //const int c = (index / width / height) % channels;
    const int n = index / width / height ;
    
    float xx = float(w) + u[index];
    float yy = float(h) + v[index];
    
    int x1 = floorf(xx);
    int x2 = x1+1;

    int y1 = floorf(yy);
    int y2 = yy+1;

    
    if(x1 < 0 || x1 > width-1 || y1 < 0 || y1 > height-1 || x2 < 0 || x2 > width-1 || y2 < 0 || y2 > height-1 ){
	for (int cc = 0; cc<channels; cc++){
	     int off =  (n*channels +  cc)*height*width;
             top_data[w + h * width +off] = 0;// index of a perticular r/g/b pixel in image:  h * width + w + (n*channels +  cc)*height*width
	}
    }
    else{
	//for every channel
	for (int cc = 0; cc<channels; cc++){	
	  int off =  (n*channels +  cc)*height*width;
	  float I_in_x_y1, I_in_x_y2, I_in_xy;
          I_in_x_y1 = (float(x2)-xx) * I[x1 + y1 *width + off] +  (xx-(float)x1) * I[x2 + y1 * width + off];// I_in_x_y1 /= (x2-x1);
          I_in_x_y2 = (float(x2)-xx) * I[x1 + y2 * width + off] +  (xx-(float)x1) * I[x2 + y2 * width + off];// I_in_x_y2 /= (x2-x1);  
          I_in_xy = (float(y2)-yy) * I_in_x_y1 + (yy-float(y1)) * I_in_x_y2; //I_in_xy /= (y2-y1);
	  top_data[w + h * width +off] =  I_in_xy;
	}           
    }
  }
}



template <typename Dtype>
void WarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* I = bottom[0]->gpu_data();
  const Dtype* u = bottom[1]->gpu_data();
  const Dtype* v = bottom[2]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count()/channels_; // top is an image with n channels but the gpu thread works on the #pix*#images (same in back)
  

  WarpingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, I,u,v,
        bottom[0]->num(), channels_,
        height_, width_,  top_data);
  CUDA_POST_KERNEL_CHECK;
}




template <typename Dtype>
__global__ void WarpingBackward(const int nthreads, const Dtype* const top_diff,
    const int num,
    const int channels, const int height, const int width,
    Dtype* const u_diff, Dtype* const v_diff, const Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {  
    
    const int w = index % width;
    const int h = (index / width) % height;
    //const int c = (index / width / height) % channels;
    const int n = index / width / height ;
    
    float gradient_u = 0;
    float gradient_v = 0;

    int x1 = w+1; 
    int x2 = w-1;

    int y1 = h+1; 
    int y2 = h-1;

    //  h == v == y; w == u == x; // 
   
    	for (int cc = 0; cc<channels; cc++){
                 int off =  (n*channels +  cc)*height*width;

		 if(x1 >= 0 && x1 <= width-1 && x2 >= 0 && x2 <= width-1  )    
	        	gradient_u += top_diff[w+ h * width  + off] *
				0.5*(top_data[x1 + h * width +off] - top_data[x2 + h * width +off]);
		 if( y1 >= 0 && y1 <= height-1 &&  y2 >= 0 && y2 <= height-1 )
		 	gradient_v += top_diff[w + h * width +off] *
				0.5*(top_data[w + y1 * width +off] - top_data[w + y2 * width +off]);
	}
    

    u_diff[index] = gradient_u;
    v_diff[index] = gradient_v;
    
  }
}


template <typename Dtype>
void WarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[0])
	CHECK_EQ(1,0)<<"The input image needs to be data";

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* u_diff = bottom[1]->mutable_gpu_diff();
  Dtype* v_diff = bottom[2]->mutable_gpu_diff();
  // top_data now have the warp image
  // backprob of warp requires the darivative of this warp image
  // this derivatives are stored in bottom_diff
  const Dtype* top_data = top[0]->gpu_data();
  const int count = bottom[0]->count()/channels_;
  caffe_gpu_set(count, Dtype(0.), u_diff);
  caffe_gpu_set(count, Dtype(0.), v_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  WarpingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_,
        u_diff,v_diff,top_data);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(WarpingLayer);
}
