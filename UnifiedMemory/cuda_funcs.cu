#include "cuda_funcs.h"

__global__ void init(unsigned char* ptr, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        ptr[i] = (unsigned char)127;
}
