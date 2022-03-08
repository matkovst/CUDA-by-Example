#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/book.h"
#include "../common/cpu_anim.h"

#define SIZE (1024*1024*100)

__global__ void histo_kernel( unsigned char *buffer, long size, unsigned int *histo ) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size)
    {
        atomicAdd( &(histo[buffer[i]]), 1 );
        i += stride;
    }
}

__global__ void histo_kernel_shared( unsigned char *buffer, long size, unsigned int *histo ) 
{
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size)
    {
        atomicAdd( &(temp[buffer[i]]), 1 );
        i += stride;
    }
    __syncthreads();

    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main(int argc, char** argv)
{
    printf("Start\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU
    unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
    unsigned int histo[256];
    for (int i = 0; i < 256; i++)
    {
        histo[i] = 0;
    }
    clock_t cpu_start = clock();
    for (int i = 0; i < SIZE; i++)
    {
        histo[buffer[i]]++;
    }
    clock_t cpu_end = clock();
    double cpu_elapsed = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf( "CPU Time: %f\n", cpu_elapsed );

    long histoCount = 0;
    for (int i = 0; i < 256; i++)
    {
        histoCount += histo[i];
    }
    printf( "CPU Histogram Sum: %ld\n", histoCount );


    // GPU
    unsigned char* dev_buffer;
    unsigned int* dev_histo;
    cudaMalloc((void**)&dev_buffer, SIZE);
    cudaMalloc((void**)&dev_histo, 256 * sizeof(int));
    cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dev_histo, 0, 256 * sizeof(int));

    cudaEventRecord(start, 0);

    // --- CORE ---
    cudaDeviceProp prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    histo_kernel_shared<<<blocks*2, 256>>>(dev_buffer, SIZE, dev_histo);
    // --- //// ---

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time to generate: %3.1f ms\n", elapsedTime );

    unsigned int h_histo[256];
    cudaMemcpy(h_histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    long h_histoCount = 0;
    for (int i = 0; i < 256; i++)
    {
        h_histoCount += h_histo[i];
    }
    printf( "GPU Histogram Sum: %ld\n", h_histoCount );

    // verify that we have the same counts via CPU
    for (int i=0; i<SIZE; i++)
    {
        histo[buffer[i]]--;
    }
    for (int i=0; i<256; i++) 
    {
        if (histo[i] != 0) printf( "Failure at %d!\n", i );
    }

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    cudaFree( dev_histo );
    cudaFree( dev_buffer );
    free(buffer);

    printf("Finish\n");
    return 0;
}