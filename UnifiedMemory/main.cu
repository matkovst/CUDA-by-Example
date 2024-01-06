#include <stdio.h>
#include <cctype>
#include <iostream>
#include <iomanip>

#include <cuda.h>

#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#include "../common/book.h"
#include "cuda_funcs.h"

int main(int argc, char* argv[])
{
    int dataCopyMode = 0;
    if (argc > 1)
        dataCopyMode = (1 == atoi(argv[1])) ? 1 : 0;

    int nDevices{0};
    HANDLE_ERROR( cudaGetDeviceCount(&nDevices) );
    if (0 == nDevices)
    {
        std::cout << "No CUDA device found" << std::endl;
        return 0;
    }

    cudaDeviceProp device0Prop;
    HANDLE_ERROR( cudaGetDeviceProperties(&device0Prop, 0) );
    std::cout << "Device info:" << std::endl;
    std::cout << "\tid (name)           : " << 0 << " (" << device0Prop.name << ")" << std::endl;
    std::cout << "\tcompute capability  : " << device0Prop.major << "." << device0Prop.minor << std::endl;
    std::cout << "\tunified mem support : " << std::boolalpha << bool(device0Prop.managedMemory) << std::endl;

    std::cout << "Copy mode " << ((0 == dataCopyMode) ? "OpenCV" : "cudaMemcpyDeviceToDevice") << std::endl;

    // Allocate unified memory
    const int rows = 1080;
    const int cols = 1920;
    const int chs = 3;
    const size_t n = rows * cols * chs;
    std::cout << std::left << std::setw(55) << ("Allocating " + std::to_string(n * sizeof(uint8_t)) + " bytes of unified mem ...");
    const size_t step = cols * chs;
    uint8_t* unifiedPtr{nullptr};
    HANDLE_ERROR( cudaMallocManaged(&unifiedPtr, n * sizeof(uint8_t)) );
    cudaPointerAttributes unifiedPtrProps;
    cudaError_t result = cudaPointerGetAttributes(&unifiedPtrProps, (void*)unifiedPtr);
    if (CUDA_SUCCESS != result)
    {
        std::cout << " Failed (" << cudaGetErrorString(result) << ")" << std::endl;
        if (nullptr != unifiedPtr)
            HANDLE_ERROR( cudaFree((void*)unifiedPtr) );
        return 0;
    }
    if (cudaMemoryType::cudaMemoryTypeManaged != unifiedPtrProps.type)
    {
        std::cout << " Failed ( Pointer not addressing CUDA memory. Got memory type " << int(unifiedPtrProps.type) << std::endl;
        if (nullptr != unifiedPtr)
            HANDLE_ERROR( cudaFree((void*)unifiedPtr) );
        return 0;
    }
    std::cout << " Done" << std::endl;

    // Initialize unified memory
    std::cout << std::left << std::setw(55) << "Initializing unified mem (on the GPU) ...";
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    init<<<numBlocks, blockSize>>>(unifiedPtr, n);
    cudaDeviceSynchronize();
    std::cout << " Done" << std::endl;

    // Copy unified memory to cv::cuda::GpuMat
    cv::cuda::GpuMat dataCopyOnDevice(rows, cols, CV_8UC3); // calls cudaMallocPitch(...)
    try
    {
        switch(dataCopyMode)
        {
        case 0: // OpenCV mode
        {
            std::cout << std::left << std::setw(55) << "Copy unified mem to cv::cuda::GpuMat (CV mode) ...";
            for (size_t i = 0; i < 100; ++i)
            {
                cv::Mat readOnlyDataHeader(rows, cols, CV_8UC3, (void*)unifiedPtr, step); // no modify, no copy !!!
                dataCopyOnDevice.upload(readOnlyDataHeader); // calls cudaMemcpy2D(..., cudaMemcpyHostToDevice)
            }
            break;
        }
        case 1: // CUDA memcpy DtoD mode
        {
            std::cout << std::left << std::setw(55) << "Copy unified mem to cv::cuda::GpuMat (DtoD mode) ...";
            for (size_t i = 0; i < 100; ++i)
            {
                HANDLE_ERROR( cudaMemcpy2D(
                    dataCopyOnDevice.cudaPtr(), dataCopyOnDevice.step, unifiedPtr, step, 
                    dataCopyOnDevice.cols * dataCopyOnDevice.elemSize(), dataCopyOnDevice.rows, 
                    cudaMemcpyDeviceToDevice
                ));
            }
            break;
        }
        }
        std::cout << " Done" << std::endl;
    }
    catch(...)
    {
        std::cout << " Failed" << std::endl;
    }

    // Check copied data
    std::cout << std::left << std::setw(55) << "Checking cv::cuda::GpuMat data ...";
    cv::Scalar deviceDataSum = cv::cuda::sum(dataCopyOnDevice);
    const cv::Scalar correctSum = cv::Scalar(rows * cols * 127, rows * cols * 127, rows * cols * 127);
    if (correctSum[0] == deviceDataSum[0] && correctSum[1] == deviceDataSum[1] && correctSum[2] == deviceDataSum[2])
        std::cout << " Done" << std::endl;
    else
        std::cout << " Failed (" << deviceDataSum << " != " << correctSum << ")" << std::endl;

    // Free memory
    std::cout << std::left << std::setw(55) << "Freeing unified mem ...";
    HANDLE_ERROR( cudaFree((void*)unifiedPtr) );
    dataCopyOnDevice.release();
    std::cout << " Done" << std::endl;

    return 0;
}