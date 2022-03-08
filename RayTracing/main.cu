#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define INF 2e10f
#define DIM 512

#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 20

struct Sphere
{
    float r, g, b;
    float x, y, z;
    float radius;
    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius)
        {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char* ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float r = 0.f, g = 0.f, b = 0.f;
    float maxz = -INF;
    for(int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255; 
}

int main(int argc, char** argv)
{
    printf("Start\n");

    // Capture the start time
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    float elapsed_time;
    HANDLE_ERROR( cudaEventElapsedTime(&elapsed_time, start, stop) );
    printf("Time to generate: %3.1f ms\n", elapsed_time);
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );


    cv::Mat bitmapMat = cv::Mat(DIM, DIM, CV_8UC4, bitmap.get_ptr()).clone();
    cv::imwrite("output/gpu_ray_tracing.png", bitmapMat);
    std::cout << "GPU result saved as output/gpu_ray_tracing.png" << std::endl;
    
    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_bitmap));

    printf("Finish\n");
    return 0;
}