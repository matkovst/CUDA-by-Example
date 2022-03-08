#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "../common/book.h"

void cpu_grayscale(unsigned char *imdata, unsigned char *out, const int width, const int height)
{
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int index = w + h * width;

            out[index] = 
            0.114 * float(imdata[3 * index + 0]) + 
            0.587 * float(imdata[3 * index + 1]) + 
            0.299 * float(imdata[3 * index + 2]) ;
        }
    }
}

__global__ void gpu_grayscale(unsigned char* d_data, unsigned char* d_out, const int width, const int height)
{
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int w = threadIdx.x + blockIdx.x * blockDim.x;

    int index = w + h * width;
    if (index >= width * height)
    {
        return;
    }
    
    d_out[index] = 
    0.114 * float(d_data[3 * index + 0]) + 
    0.587 * float(d_data[3 * index + 1]) + 
    0.299 * float(d_data[3 * index + 2]) ;
}

int main(int argc, char** argv)
{
    std::cout << "Hello" << std::endl;

    std::string filepath = "me.jpg";
    if (argc > 1)
        filepath = argv[1];
    
    cv::Mat img = cv::imread(filepath);
    //cv::resize(img, img, cv::Size(360, 640));
    cv::Mat cpu_gray;
    cv::Mat gpu_gray;
    const int width = img.cols;
    const int height = img.rows;

    /* ---------- CPU ---------- */
    clock_t cpu_start = clock();

    unsigned char* data = img.data;
    unsigned char* cpu_out = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    cpu_grayscale(data, cpu_out, width, height);
    cpu_gray = cv::Mat(height, width, CV_8UC1, cpu_out);

    clock_t cpu_end = clock();
    double cpu_elapsed = double(cpu_end - cpu_start);
    /* ---------- /// ---------- */

    /* ---------- GPU ---------- */
    clock_t gpu_start = clock();

    unsigned char* gpu_out = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *d_data, *d_out;
    cudaMalloc((void**)&d_data, 3 * width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, width * height * sizeof(unsigned char));
    cudaMemcpy(d_data, data, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    int grid_x = floor((width + blockDim.x - 1) / blockDim.x);
    int grid_y = floor((height + blockDim.y - 1) / blockDim.y);
    dim3 gridDim(grid_x, grid_y);
    gpu_grayscale<<<gridDim, blockDim>>>(d_data, d_out, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_out, d_out, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    gpu_gray = cv::Mat(height, width, CV_8UC1, gpu_out);

    clock_t gpu_end = clock();
    double gpu_elapsed = double(gpu_end - gpu_start);
    /* ---------- /// ---------- */

    // Checking
    cv::Mat gtruth_gray;
    cv::cvtColor(img, gtruth_gray, cv::COLOR_BGR2GRAY);
    unsigned char* gtruth_data = gtruth_gray.data;
    int cpu_loss = 0;
    int gpu_loss = 0;
    for (int i = 0; i < width * height; i++)
    {
        cpu_loss += abs(gtruth_data[i] - cpu_out[i]);
        gpu_loss += abs(gtruth_data[i] - gpu_out[i]);
    }

    // Print info
    std::cout << "CPU time: " << cpu_elapsed << std::endl;
    std::cout << "GPU time: " << gpu_elapsed << std::endl;
    std::cout << "CPU abs loss: " << cpu_loss << std::endl;
    std::cout << "GPU abs loss: " << gpu_loss << std::endl;
    // cv::imshow("GT", gtruth_gray);
    // cv::imshow("CPU", cpu_gray);
    // cv::imshow("GPU", gpu_gray);
    // cv::waitKey();
    cv::imwrite("gtruth.jpg", gtruth_gray);
    cv::imwrite("cpu.jpg", cpu_gray);
    cv::imwrite("gpu.jpg", gpu_gray);
    std::cout << "GTruth result saved as gtruth.jpg" << std::endl;
    std::cout << "CPU result saved as cpu.jpg" << std::endl;
    std::cout << "GPU result saved as gpu.jpg" << std::endl;

    free(cpu_out);
    free(gpu_out);
    cudaFree(d_data);
    cudaFree(d_out);

    std::cout << "Bye" << std::endl;
    return 0;
}