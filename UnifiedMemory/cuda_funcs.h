#pragma once

#include <stdio.h>
#include <cuda.h>

__global__ void init(unsigned char* ptr, size_t n);