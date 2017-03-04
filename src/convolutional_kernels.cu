#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

char shouldCacncle(int h, int w)
{
    return 0;
    // return (h+w)%2==0;
}


void setIndicator(int height_col, int width_col, int *indicator)
{
    int h, w, cancelOffset = 0;
    for (h = 0; h < height_col; ++h) {
        for (w = 0; w < width_col; ++w) {
            if (shouldCacncle(h, w)) {
                cancelOffset++;
                indicator[h*width_col+w] = 0-cancelOffset;
            }
            else {
                indicator[h*width_col+w] = cancelOffset;
            }
            // printf("%4d ", indicator[h*width_col+w]);
        }
        // printf("\n");
    }
    // printf("\n");
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += abs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}


void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{   
    static float timings[15][10];
    double t1, t2;
    t1 = get_wall_time_us();
    double flop;
    // output:416*416*16, batch:1
    // cudaMemset(l.output_gpu, 0, l.outputs*l.batch*sizeof(float));
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    // cudaThreadSynchronize();
    t2 = get_wall_time_us();
    timings[l.idx][0] += t2-t1;
    // printf("\t %lu %d %f %f\n", &l, l.idx, timings[l.idx][0], t2-t1);
    t1 = t2;
    
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                state.input,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i;
    int m = l.n; //核数
    int k = l.size*l.size*l.c; //核大小和通道
    int n = l.out_w*l.out_h;    //输出w, h
    for(i = 0; i < l.batch; ++i){
        if (l.idx >= 11)
        // if (0)
        {
            static int *indicator = NULL;
            if (indicator == NULL){
                cudaHostAlloc((void **)&indicator, n*sizeof(int), cudaHostAllocMapped);
                setIndicator(l.out_h, l.out_w, indicator);
            }
            int compressSize = indicator[l.out_h*l.out_w-1];
            if (compressSize < 0)
                compressSize = 0-compressSize;
            im2col_ongpu3(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, indicator, state.workspace);
            // im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
            
            // cudaThreadSynchronize();
            t2 = get_wall_time_us();
            timings[l.idx][1] += t2-t1;
            t1 = t2;

            float * a = l.weights_gpu;
            float * b = state.workspace;
            float * c = l.delta_gpu;

            n -= compressSize;

            printf("\tMatrix Multiplication %dx%d * %dx<%d>\n",m,k,k,n);
            t1 = get_wall_time_us();
            gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
            // cudaThreadSynchronize();

            resizeImg_ongpu(c, m, l.out_h, l.out_w, indicator, l.output_gpu);
            // cudaThreadSynchronize();

            t2 = get_wall_time_us();
            timings[l.idx][2] += t2-t1;
            t1 = t2;
            flop = ((double)m)*n*(2.*k + 2.);
        }
        else {
            im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        
            // cudaThreadSynchronize();
            t2 = get_wall_time_us();
            timings[l.idx][1] += t2-t1;
            t1 = t2;

            float * a = l.weights_gpu;
            float * b = state.workspace;
            float * c = l.output_gpu;

            printf("\tMatrix Multiplication %dx%d * %dx%d\n",m,k,k,n);
            t1 = get_wall_time_us();
            gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
            // cudaThreadSynchronize();
            t2 = get_wall_time_us();
            timings[l.idx][2] += t2-t1;
            t1 = t2;
            flop = ((double)m)*n*(2.*k + 2.);
        }       
    }
#endif
    // cudaMemset(l.output_gpu, 0, l.out_h*l.out_w*l.out_c);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
        // cudaThreadSynchronize();
        // if (l.idx == 13) {
        //     printf("bn\n");
        //     printImg_ongpu(l.output_gpu, l.out_h, l.out_w, 0);
        // }
    }
    t2 = get_wall_time_us();
    timings[l.idx][3] += t2-t1;
    t1 = t2;

    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    // if (l.idx == 13) {
    //     printf("bias\n");
    //     printImg_ongpu(l.output_gpu, l.out_h, l.out_w, 0);
    // }
    // cudaThreadSynchronize();
    t2 = get_wall_time_us();
    timings[l.idx][4] += t2-t1;
    t1 = t2;

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    // if (l.idx == 13) {
    //     printf("activate\n");
    //     printImg_ongpu(l.output_gpu, l.out_h, l.out_w, 0);
    //     printf("hard code\n");
    //     setImg_ongpu(l.output_gpu, l.out_h, l.out_w, l.n);
    //     printImg_ongpu(l.output_gpu, l.out_h, l.out_w, 0);
    // }

    // cudaThreadSynchronize();
    t2 = get_wall_time_us();
    timings[l.idx][5] += t2-t1;
    t1 = t2;

    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor){
        swap_binary(&l);
        // cudaThreadSynchronize();
        t2 = get_wall_time_us();
        timings[l.idx][6] += t2-t1;
        t1 = t2;
    } 
    printf("\tConv %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", timings[l.idx][0]/1000, 
        timings[l.idx][1]/1000, timings[l.idx][2]/1000, timings[l.idx][3]/1000, 
        timings[l.idx][4]/1000, timings[l.idx][5]/1000, flop);
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    //constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.x_gpu, 1, l.delta_gpu, 1);
    } else {
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.output_gpu, 1, l.delta_gpu, 1);
    }
    float *original_input = state.input;

    if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(state.delta){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if(layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if(layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }else{
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
    }
}


