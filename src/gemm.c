#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "im2col.h"

char shouldCacncle(int h, int w)
{
    return (h+w)%2==0;
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

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

void printMatrix(const char *tag, float *matrix, int rows, int cols, int channels)
{
    int c, y , t, x;
    printf("%s (%d,%d,%d)\n", tag, channels, rows, cols);
    for (c = 0; c < channels; c++) {
        for (y = 0; y < rows; ++y) {
            for (t = 0; t < c; t++) printf("\t");
            for (x = 0; x < cols; ++x) {
                printf("%2.0f ", matrix[(c*rows+y)*cols + x]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

float *matrix(int rows, int cols, int channels)
{
    float *m; 
    cudaError_t status = cudaHostAlloc((void **)&m, rows*cols*channels*sizeof(float), cudaHostAllocMapped);
    check_error(status);
    status = cudaMemset((void *)m, 0, rows*cols*channels*sizeof(float));
    check_error(status);
    return m;
}

float *increMatrix(int rows, int cols, int channels)
{
    int i;
    float *m = matrix(rows, cols, channels);
    for(i = 0; i < channels*rows*cols; ++i){
        m[i] = (float)i+1;
    }
    return m;
}

float *onesMatrix(int rows, int cols, int channels)
{
    int i;
    float *m = matrix(rows, cols, channels);
    for(i = 0; i < channels*rows*cols; ++i){
        m[i] = 1.0;
    }
    return m;
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    float *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    float *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    float *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 100;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    double t1, t2;
    t1 = get_wall_time_us();
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    t2 = get_wall_time_us();
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    // end = clock();
    // double seconds = sec(end-start);
    double mySec = (t2-t1)/1000/1000;
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lfs, %lf(%f) gflops\n", 
        m, k, k, n, TA, TB, mySec, gflop, gflop/mySec);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}

void time_ongpu_compresess(int TA, int TB, int m, int k, int out_h, int out_w, int factor)
{
    int iter = 100;
    int n = out_h * out_w;
    int n2 = n / factor;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n2:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n2);
    float *c2_cl = cuda_make_array(c, m*n2);
    float *c_cl = cuda_make_array(c, m*n);
    int *indicator;
    cudaHostAlloc((void **)&indicator, out_h*out_w*sizeof(int), cudaHostAllocMapped);
    setIndicator(out_h, out_w, indicator);

    int i;
    double t1, t2;
    t1 = get_wall_time_us();
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n2,k,1,a_cl,lda,b_cl,ldb,1,c2_cl,n2);
        cudaThreadSynchronize();
        resizeImg_ongpu(c2_cl, m, out_h, out_w, indicator, c_cl);
        cudaThreadSynchronize();
    }
    t2 = get_wall_time_us();
    double flop = ((double)m)*n2*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    // end = clock();
    // double seconds = sec(end-start);
    double mySec = (t2-t1)/1000/1000;
    printf("Matrix Multiplication %dx%d * %dx%d(%d factor%d), TA=%d, TB=%d: %lfs, %lf(%f) gflops\n", 
        m, k, k, n2, n, factor, TA, TB, mySec, gflop, gflop/mySec);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c2_cl);
    cuda_free(c_cl);
    cudaFreeHost(indicator);
    free(a);
    free(b);
    free(c);
}

void time_ongpu_im2col() {
    int iter = 100;
    // int m = 3; //核数
    int size = 3; //核size
    int c = 1024; //图片通道
    int k = size*size*c; //gemm K
    int h = 15; //输入h
    int w = 20; //输入w
    int pad = size/2; //补位
    int stride = 1;
    int out_h = (h + 2*pad - size) / stride + 1; //输出h
    int out_w = (w + 2*pad - size) / stride + 1; //输出w

    float *img = increMatrix(h, w, c);
    float *imgCol = matrix(k, h*w, 1);

    int *indicator;
    cudaError_t status = cudaHostAlloc((void **)&indicator, out_h*out_w*sizeof(int), cudaHostAllocMapped);
    check_error(status);
    setIndicator(out_h, out_w, indicator);

    double t1, t2;
    int i;
    t1 = get_wall_time_us();
    for(i = 0; i<iter; ++i){
        im2col_ongpu(img, c, h, w, size, stride, pad, imgCol);
        cudaThreadSynchronize();
    }
    t2 = get_wall_time_us();
    printf("im2col_ongpu %.1fms\n", (t2-t1)/1000);

    t1 = get_wall_time_us();
    for(i = 0; i<iter; ++i){
        im2col_ongpu3(img, c, h, w, size, stride, pad, indicator, imgCol);
        cudaThreadSynchronize();
    }
    t2 = get_wall_time_us();
    printf("im2col_ongpu3 %.1fms\n", (t2-t1)/1000);
}

void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    // time_ongpu(0, 0, 16, 3*3*3, 320*240);
    // printf("\n");
    time_ongpu_im2col();
    // time_ongpu(0, 0, 512, 3*3*512, 20*15);
    // time_ongpu_compresess(0, 0, 512, 3*3*512, 20, 15, 2);
    // time_ongpu_compresess(0, 0, 512, 3*3*512, 20, 15, 5);
    // time_ongpu_compresess(0, 0, 512, 3*3*512, 20, 15, 10);
    printf("\n");

    // time_ongpu(0, 0, 1024, 3*3*1024, 20*15/2); 
    // time_ongpu(0, 0, 1024, 3*3*1024/2, 20*15);
    // time_ongpu(0, 0, 1024/2, 3*3*1024, 20*15);
    // printf("\n");

    // time_ongpu(0, 0, 1024, 3*3*1024, 20*15/10); 
    // time_ongpu(0,0,1024,3*3*1024,10*15); 
    // time_ongpu(0,0,1024,3*3*1024,2*15); 
     
    // time_ongpu(0,0,1024,3*3*512,10*15); 
    // time_ongpu(0,0,1024,3*3*512,2*15); 
    return 0;
}

int test_gpu_blas1()
{
    int m = 3; //核数
    int size = 3; //核size
    int c = 2; //图片通道
    int k = size*size*c; //gemm K
    int h = 4; //输入h
    int w = 4; //输入w
    int pad = size/2; //补位
    int stride = 1;
    int out_h = (h + 2*pad - size) / stride + 1; //输出h
    int out_w = (w + 2*pad - size) / stride + 1; //输出w
    int n = out_h*out_w; //gemm 输出大小

    float *img = increMatrix(h, w, c);
    printMatrix("img inittial", img, h, w, c);
    float *imgCol = matrix(k, out_h*out_w, 1);

    // im2col
    // im2col_ongpu(img, c, h, w, size, stride, pad, imgCol);
    // cudaThreadSynchronize();
    // printMatrix("im2col", imgCol, k, out_h*out_w, 1);

    // im2col show compress
    im2col_ongpu2(img, c, h, w, size, stride, pad, imgCol);
    cudaThreadSynchronize();
    printMatrix("im2col show compress", imgCol, k, out_h*out_w, 1);

    int *indicator;
    cudaHostAlloc((void **)&indicator, out_h*out_w*sizeof(int), cudaHostAllocMapped);
    setIndicator(out_h, out_w, indicator);

    // im2col compress
    im2col_ongpu3(img, c, h, w, size, stride, pad, indicator, imgCol);
    cudaThreadSynchronize();
    printMatrix("im2col compress", imgCol, k, out_h*out_w+indicator[out_h*out_w-1], 1);
    n += indicator[out_h*out_w-1];

    // 卷积核
    float *convKernel = onesMatrix(m, k, 1);
    printMatrix("kernel", convKernel, 1, k, m);
    
    // gemm
    float *gemm = matrix(n*m, 1, 1);
    gemm_ongpu(0, 0, m, n, k, 1, 
        convKernel, k, imgCol, n, 0., gemm, n);
    cudaThreadSynchronize();
    printMatrix("gemm", gemm, out_h, out_w, m);

    float *gemmResized = matrix(out_h*out_w*m, 1, 1);
    resizeImg_ongpu(gemm, m, out_h, out_w, indicator, gemmResized);
    cudaThreadSynchronize();
    printMatrix("gemm resized", gemmResized, out_h, out_w, m);
    return 0;
}
#endif

