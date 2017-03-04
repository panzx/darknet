#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <stdio.h>

extern "C" {
#include "im2col.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                // *data_col_ptr = 0;
                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void im2col_gpu_kernel2(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    int w_out = index % width_col;
    int h_out = index / width_col % height_col;
    printf("%d %d\n", h_out, w_out);
    if ((w_out+h_out)%2 == 0)
        return;

    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void im2col_gpu_kernel3(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        const int *indicator,const int compressSize, float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        
        if (indicator[h_out*width_col+w_out] < 0) {
            return;
        }

        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out 
            - indicator[h_out*width_col+w_out] - compressSize*channel_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        // int channelDebug = channel_out;
        // int positionOldDebug = (channel_out * height_col + h_out) * width_col + w_out;
        // int positionDebug = positionOldDebug - indicator[h_out*width_col+w_out] + indicator[indicatorSize]*channel_out;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                // if (threadIdx.x == 6) {
                //     printf("col %d[%d][%d] -> %d= %.0f\n", positionOldDebug, channelDebug,
                //         h_out * width_col + w_out, positionDebug, *data_col_ptr);
                //     printf("\t%d = %d*%d+%d\n", height_col * width_col + indicator[indicatorSize], height_col, width_col ,indicator[indicatorSize]);
                // }
                // channelDebug++;
                // positionOldDebug += height_col * width_col;
                // positionDebug += height_col * width_col + indicator[indicatorSize];
                data_col_ptr += height_col * width_col - compressSize;
            }
        }
    }
}

__global__ void resizeImg_gpu_kernel(const float*im, const int channels, const int height, 
    const int width, const int *indicator, float *resizeImg)
{
    int channelIdx;
    for(channelIdx = blockIdx.x*blockDim.x+threadIdx.x; channelIdx < channels; channelIdx += blockDim.x*gridDim.x) {
        // printf("%d = %d*%d+%d  + %d(%d*%d)\n", channelIdx, blockIdx.x, blockDim.x, threadIdx.x, 
        //     blockDim.x*gridDim.x, blockDim.x, gridDim.x);
        int size = height*width;
        int baseOffset = channelIdx*indicator[size-1];
        int resizedOffset = channelIdx*size;
        for (int i = 0; i < size; ++i)
        {

            int resizedIdx = resizedOffset + i;
//            printf("%d(%d*%d+%d) %d\n", c*out_h*out_w+i, c, out_h*out_w, i, c*indicator[out_h*out_w-1]+indicator[i]);
            // if (threadIdx.x == 10)
            // {
            //     printf("%d %d*%d+%d\n", resizedIdx, channelIdx, size, i);
            // }
            if (indicator[i]<0) {
                resizeImg[resizedIdx] = 0.0;
            }
            else {
                int offset = baseOffset-indicator[i];
                resizeImg[resizedIdx] = im[resizedIdx+offset];
            }
        }
    }
}

__global__ void printImg(const float* im, const int height, const int width, const int channel)
{
    printf("----------------------------\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%f ", im[channel*width*height + y*width + x]);
        }
        printf("\n");
    }
    printf("----------------------------\n");
}

__global__ void setImg(float* im, const int height, const int width, const int channel)
{
    for (int c = 0; c < channel; c++){
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                im[c*width*height + y*height + x] = 0.0;
            }
        }
    }
}

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}

void im2col_ongpu2(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    // printImg<<<1, 1>>>(im, height, width, 0);

    // height_col: 15 = (15 + 2 * 1 - 3) / 1 + 1
    // width_col:  20 = (20 + 2 * 1 - 3) / 1 + 1
    // num_kernels:307200 = 1024 * 15 * 20
    // <<<600, 512>>>
    // printf("height_col: %d = (%d + 2 * %d - %d) / %d + 1\n", height_col, height, pad, ksize, stride);
    // printf("width_col:  %d = (%d + 2 * %d - %d) / %d + 1\n", width_col, width, pad, ksize, stride);
    // printf("num_kernels:%d = %d * %d * %d\n", num_kernels, channels, height_col, width_col);
    // printf("<<<%d, %d>>>\n",(num_kernels+BLOCK-1)/BLOCK, BLOCK);
    im2col_gpu_kernel2<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);

}

void im2col_ongpu3(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, int *indicator, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    int compressSize = indicator[height_col*width_col-1];
    if (compressSize < 0)
        compressSize = 0 - compressSize;
    // printf("%d %d %d\n", compressSize, height_col, width_col);

    im2col_gpu_kernel3<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, indicator, compressSize, data_col);
}

void resizeImg_ongpu(float *im, int channels, int height, int width, int *indicator, float *resizeImg){
    // 每条线程负责一个channel
    // printf("<<<%d %d>>>\n", (channels+BLOCK-1)/BLOCK, BLOCK);
    resizeImg_gpu_kernel<<<(channels+BLOCK-1)/BLOCK, BLOCK>>>(im, channels, height, width, indicator, resizeImg);
}


void printImg_ongpu(const float* im, const int height, const int width, const int channel)
{
     printImg<<<1, 1>>>(im, height, width, channel);
}

void setImg_ongpu(float* im, const int height, const int width, const int channel)
{
     setImg<<<1, 1>>>(im, height, width, channel);
}
