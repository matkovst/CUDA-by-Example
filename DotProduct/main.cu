#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

static void HandleError( cudaError_t err,
    const char *file,
    int line ) {
if (err != cudaSuccess) {
printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
file, line );
exit( EXIT_FAILURE );
}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
       printf( "Host memory failed in %s at line %d\n", \
               __FILE__, __LINE__ ); \
       exit( EXIT_FAILURE );}}


#define THREADS_PER_BLOCK 512

__global__ void gpu_dot_oneblock(float *d_a, float *d_b, float *d_c, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp[];
    temp[tid] = d_a[tid] * d_b[tid];
    __syncthreads();

    if (tid == 0)
    {
        int sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += temp[i];
        }
        *d_c = sum;
    }
}

__global__ void gpu_dot_multiblock(float *d_a, float *d_b, float *d_c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float temp[THREADS_PER_BLOCK];
    temp[threadIdx.x] = d_a[tid] * d_b[tid];
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        atomicAdd(d_c, sum);
    }
}

int main(int argc, char** argv)
{
    std::cout << "Hello" << std::endl;
    
    const int N = 8;
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = (float*)malloc(sizeof(float));
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = 2*i;
    }

    float *d_a, *d_b, *d_c;
    HANDLE_ERROR( cudaMalloc((void**)&d_a, N * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&d_b, N * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&d_c, sizeof(float)) );
    HANDLE_ERROR( cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice) );

    //gpu_dot_oneblock<<<1, N, N * sizeof(float)>>>(d_a, d_b, d_c, N);
    gpu_dot_multiblock<<<(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    HANDLE_ERROR( cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost) );
    printf("Result: %f\n", *h_c);

    free(h_a);
    free(h_b);
    free(h_c);
    HANDLE_ERROR( cudaFree(d_a) );
    HANDLE_ERROR( cudaFree(d_b) );
    HANDLE_ERROR( cudaFree(d_c) );

    std::cout << "Bye" << std::endl;
    return 0;
}