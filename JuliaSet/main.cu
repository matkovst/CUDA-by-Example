#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "../common/book.h"


// CPU implementation -------------------------------------------------------------
struct cuComplex
{
    float r;
    float i;
    cuComplex( float a, float b ) : r(a), i(b) {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) 
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) 
    {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia(int x, int y, const int DIM, const int max_iter)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < max_iter; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000) 
        {
            return 0;
        }
    }

    return 1;
}

void cpu_kernel(unsigned char* data, const int DIM, const int max_iter)
{
    for(int y = 0; y < DIM; y++)
    {
        for(int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y, DIM, max_iter);
            data[offset*3 + 0] = 200 * juliaValue;
            data[offset*3 + 1] = 200 * juliaValue;
            data[offset*3 + 2] = 0;
        }
    }
}


// GPU implementation -------------------------------------------------------------
struct gpu_cuComplex
{
    float r;
    float i;
    __device__ gpu_cuComplex( float a, float b ) : r(a), i(b) {}
    __device__ float magnitude2( void ) { return r * r + i * i; }
    __device__ gpu_cuComplex operator*(const gpu_cuComplex& a) 
    {
        return gpu_cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ gpu_cuComplex operator+(const gpu_cuComplex& a) 
    {
        return gpu_cuComplex(r+a.r, i+a.i);
    }
};

__device__ int gpu_julia(int x, int y, const int DIM, const int max_iter)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    gpu_cuComplex c(-0.8, 0.156);
    gpu_cuComplex a(jx, jy);

    for (int i = 0; i < max_iter; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000) 
        {
            return 0;
        }
    }

    return 1;
}

__global__ void gpu_kernel(unsigned char* d_data, const int DIM, const int max_iter)
{
    int y = blockIdx.y;
    int x = blockIdx.x;

    int offset = x + y * DIM;

    int juliaValue = gpu_julia(x, y, DIM, max_iter);
    d_data[offset*3 + 0] = 0;
    d_data[offset*3 + 1] = 255 * juliaValue;
    d_data[offset*3 + 2] = 0;
}

int main(int argc, char** argv)
{
    std::cout << "Hello" << std::endl;

    const int DIM = 1024;
    const int N = DIM * DIM * 3;
    const int max_iter = 200;

    // CPU
    unsigned char* data = (unsigned char*)malloc(N * sizeof(unsigned char));
    for (int i = 0; i < N; i++) data[i] = 0;

    clock_t start = clock();
    cpu_kernel(data, DIM, max_iter);
    clock_t end = clock();
    double cpu_elapsed = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "CPU time: " << cpu_elapsed << std::endl;
    cv::Mat img = cv::Mat(DIM, DIM, CV_8UC3, data);
    cv::imwrite("output/cpu_julia.png", img);
    std::cout << "CPU result saved as output/cpu_julia.png" << std::endl;


    // GPU
    unsigned char* h_data = (unsigned char*)malloc(N * sizeof(unsigned char));
    unsigned char* d_data;
    HANDLE_ERROR (cudaMalloc((void**)&d_data, N * sizeof(unsigned char)) );

    clock_t gpu_start = clock();
    dim3 grid(DIM, DIM);
    gpu_kernel<<<grid, 1>>>(d_data, DIM, max_iter);
    clock_t gpu_end = clock();
    double gpu_elapsed = double(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    HANDLE_ERROR( cudaMemcpy(h_data, d_data, N * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    std::cout << "GPU time: " << gpu_elapsed << std::endl;
    cv::Mat gpu_img = cv::Mat(DIM, DIM, CV_8UC3, h_data);
    cv::imwrite("output/gpu_julia.png", gpu_img);
    std::cout << "GPU result saved as output/gpu_julia.png" << std::endl;

    std::cout << "Bye" << std::endl;
    return 0;
}