#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

void mandelbrot_mat ( cv::Mat& out , const int width , const int height , const int max_iter )
{
    out = cv::Mat::zeros(cv::Size(width, height), CV_32FC3);

    float x_origin , y_origin , xtemp , x, y;
    int iteration ;
    for ( int i = 0; i < width ; i ++)
    {
        for ( int j = 0; j < height ; j ++)
        {
            iteration = 0;
            x = 0.0f;
            y = 0.0f;
            x_origin = (( float ) i/ width ) *3.25f -2.0f;
            y_origin = (( float ) j/ width ) *2.5f - 1.25f;
            while (x*x + y*y <= 4 && iteration < max_iter )
            {
                xtemp = x*x - y*y + x_origin ;
                y = 2*x*y + y_origin ;
                x = xtemp ;
                iteration ++;
            }

            cv::Vec3f* pixel = &out.at<cv::Vec3f>(j, i);
            if( iteration == max_iter )
            {
                *pixel = cv::Vec3f(0, 0, 0);
            }
            else
            {
                *pixel = cv::Vec3f(iteration, iteration, iteration);
            }
        }
    }
}

void cpu_mandelbrot ( float* out , const int width , const int height , const int max_iter )
{
    float x_origin , y_origin , xtemp , x, y;
    int iteration , index ;
    for ( int i = 0; i < width ; i ++)
    {
        for ( int j = 0; j < height ; j ++)
        {
            index = 3* width *j + i *3;
            iteration = 0;
            x = 0.0f;
            y = 0.0f;
            x_origin = (( float ) i/ width ) *3.25f -2.0f;
            y_origin = (( float ) j/ width ) *2.5f - 1.25f;
            while (x*x + y*y <= 4 && iteration < max_iter )
            {
                xtemp = x*x - y*y + x_origin ;
                y = 2*x*y + y_origin ;
                x = xtemp ;
                iteration ++;
            }

            if( iteration == max_iter )
            {
                out[ index ] = 0.f;
                out[ index + 1] = 0.f;
                out[ index + 2] = 0.f;
            }
            else
            {
                out[ index ] = (float)iteration ;
                out[ index + 1] = (float)iteration ;
                out[ index + 2] = (float)iteration ;
            }
        }
    }
}

__global__ void gpu_mandelbrot ( float* d_out, const int width, const int height, const int max_iter )
{
    // int i = blockIdx.x;
    // int j = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = 3 * width * j + i * 3;
    int iteration = 0;
    float x = 0.0f;
    float y = 0.0f;
    float x_origin = (( float ) i/ width ) *3.25f -2.0f;
    float y_origin = (( float ) j/ width ) *2.5f - 1.25f;
    float xtemp;
    while (x*x + y*y <= 4 && iteration < max_iter )
    {
        xtemp = x*x - y*y + x_origin ;
        y = 2*x*y + y_origin ;
        x = xtemp ;
        iteration ++;
    }

    if( iteration == max_iter )
    {
        d_out[ index ] = 0.f;
        d_out[ index + 1] = 0.f;
        d_out[ index + 2] = 0.f;
    }
    else
    {
        d_out[ index ] = (float)iteration ;
        d_out[ index + 1] = (float)iteration ;
        d_out[ index + 2] = (float)iteration ;
    }
}

void print_gpu_info()
{
	int device_Count = 0;
	cudaGetDeviceCount(&device_Count);
    //cout << "Detected " << device_Count << " CUDA Capable device(s)" << endl;
    printf(" Detected %d CUDA Capable device(s)\n", device_Count);
    
    int device = 0;
    cudaDeviceProp device_Property;
	cudaGetDeviceProperties(&device_Property, device);

	printf(" Total amount of global memory: %.0f MBytes (%llu bytes)\n",
		(float)device_Property.totalGlobalMem / 1048576.0f, (unsigned long long) device_Property.totalGlobalMem);
	if (device_Property.l2CacheSize)
	{
		printf(" L2 Cache Size: %d bytes\n", device_Property.l2CacheSize);
	}
	printf(" Total amount of constant memory: %zu bytes\n", device_Property.totalConstMem);
	printf(" Total amount of shared memory per block: %zu bytes\n", device_Property.sharedMemPerBlock);
	printf(" Total number of registers available per block: %d\n", device_Property.regsPerBlock);

	printf(" Maximum number of threads per multiprocessor: %d\n", device_Property.maxThreadsPerMultiProcessor);
	printf(" Maximum number of threads per block: %d\n", device_Property.maxThreadsPerBlock);
	printf(" Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
		device_Property.maxThreadsDim[0],
		device_Property.maxThreadsDim[1],
		device_Property.maxThreadsDim[2]);
	printf(" Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n",
		device_Property.maxGridSize[0],
		device_Property.maxGridSize[1],
        device_Property.maxGridSize[2]);
}


int main(int argc, char** argv)
{
    std::cout << "Hello" << std::endl;

    const int W = 4096;
    const int H = 4096;
    const int C = 3;
    const int N = W * H * C;
    const int max_iter = 512;

    // CPU
    float* mbrot_data = new float[N];
    for (unsigned int i = 0; i < N; i++) mbrot_data[i] = 0.f;

    clock_t begin = clock();
    cpu_mandelbrot(mbrot_data, W, H, max_iter);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "CPU time: " << elapsed_secs << std::endl;

    cv::Mat cpu_out = cv::Mat(cv::Size(W, H), CV_32FC3, mbrot_data);
    cv::Mat cpu_to_save;
    cpu_out.convertTo(cpu_to_save, CV_8UC3, 255);
    cv::imwrite("cpu_madnelbrot.jpg", cpu_to_save);
    std::cout << "CPU Mandelbrot done. Img saved as cpu_madnelbrot.jpg." << std::endl;


    // // CPU with cv::Mat
    // cv::Mat cpu_out;
    // mandelbrot_mat(cpu_out, W, H, max_iter);
    // cv::Mat cpu_to_save;
    // cpu_out.convertTo(cpu_to_save, CV_8UC3, 255);
    // cv::imwrite("cpu_madnelbrot.jpg", cpu_to_save);
    // std::cout << "CPU Mandelbrot done. Img saved as cpu_madnelbrot.jpg." << std::endl;


    // GPU
    //print_gpu_info();

    float* h_mbrot_data = new float[N];
    //for (unsigned int i = 0; i < N; i++) h_mbrot_data[i] = 0.f;

    float *d_mbrot_data;
    cudaMalloc((void**)&d_mbrot_data, N * sizeof(float));
    //cudaMemcpy(d_mbrot_data, h_mbrot_data, N * sizeof(float), cudaMemcpyHostToDevice);

    clock_t gpu_begin = clock();
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(W / blockDim.x, H / blockDim.y, 1);
    gpu_mandelbrot<<<gridDim, blockDim>>>(d_mbrot_data, W, H, max_iter);
    clock_t gpu_end = clock();
    double gpu_elapsed_secs = double(gpu_end - gpu_begin) / CLOCKS_PER_SEC;
    std::cout << "GPU time: " << gpu_elapsed_secs << std::endl;

    cudaMemcpy(h_mbrot_data, d_mbrot_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat gpu_out = cv::Mat(cv::Size(W, H), CV_32FC3, h_mbrot_data);
    cv::Mat gpu_to_save;
    gpu_out.convertTo(gpu_to_save, CV_8UC3, 255);
    cv::imwrite("gpu_madnelbrot.jpg", gpu_to_save);
    std::cout << "GPU Mandelbrot done. Img saved as gpu_madnelbrot.jpg." << std::endl;

    // Check if images are equal
    int loss = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        loss += abs(mbrot_data[i] - h_mbrot_data[i]);
    }
    std::cout << "Absolute loss btw CPU and GPU result: " << loss << std::endl;

    cudaFree(d_mbrot_data);
    free(mbrot_data);

    std::cout << "Bye" << std::endl;
    return 0;
}