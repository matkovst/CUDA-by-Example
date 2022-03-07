#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

typedef struct 
{
    int width;
    int height;
    float* data;
} Matrix;

__global__ void gpu_matmul(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; e++)
    {
        Cvalue += A.data[ row * A.width + e ] * B.data[ e * B.width + col ];
    }
    C.data[ row * C.width + col ] = Cvalue;
}

int main(int argc, char** argv)
{
    std::cout << "Hello" << std::endl;

    // CPU
    const int awidth = 4;
    const int aheight = 3;
    const int bwidth = 3;
    const int bheight = 2;
    const int cwidth = aheight;
    const int cheight = bwidth;
    const int an = awidth * aheight;
    const int bn = bwidth * bheight;
    const int cn = cwidth * cheight;
    float* h_adata = new float[an];
    float* h_bdata = new float[bn];
    float* h_cdata = new float[cn];
    for (unsigned int i = 0; i < an; i++) h_adata[i] = i+1;
    for (unsigned int i = 0; i < bn; i++) h_bdata[i] = 2*(i+1);
    Matrix A = { awidth, aheight, h_adata };
    Matrix B = { bwidth, bheight, h_bdata };
    Matrix C = { cwidth, cheight, h_cdata };

    float *d_adata, *d_bdata, *d_cdata;
    cudaMalloc((void**)&d_adata, an * sizeof(float));
    cudaMalloc((void**)&d_bdata, bn * sizeof(float));
    cudaMalloc((void**)&d_cdata, cn * sizeof(float));
    cudaMemcpy(d_adata, h_adata, an * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bdata, h_bdata, bn * sizeof(float), cudaMemcpyHostToDevice);
    Matrix d_A = { awidth, aheight, d_adata };
    Matrix d_B = { bwidth, bheight, d_bdata };
    Matrix d_C = { cwidth, cheight, d_cdata };

    clock_t gpu_begin = clock();
    dim3 blockDim(4, 2, 1);
    dim3 gridDim(awidth / blockDim.x, bheight / blockDim.y, 1);
    gpu_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    clock_t gpu_end = clock();
    double gpu_elapsed_secs = double(gpu_end - gpu_begin) / CLOCKS_PER_SEC;
    std::cout << "GPU time: " << gpu_elapsed_secs << std::endl;

    cudaMemcpy(C.data, d_C.data, cn * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < an; i++) std::cout << A.data[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < bn; i++) std::cout << B.data[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < cn; i++) std::cout << C.data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_adata);
    cudaFree(d_bdata);
    free(h_adata);
    free(h_bdata);

    std::cout << "Bye" << std::endl;
    return 0;
}