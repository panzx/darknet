#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);
void im2col_ongpu2(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);
void im2col_ongpu3(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, int *indicator, float *data_col);
void resizeImg_ongpu(float *im, int channels, int height, int width, int *indicator, float *resizeImg);
void printImg_ongpu(const float* im, const int height, const int width, const int channel);
void setImg_ongpu(float* im, const int height, const int width, const int channel);
#endif
#endif
