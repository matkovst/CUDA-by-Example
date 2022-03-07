#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.2f

// Textures exist on GPU side
texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;

struct DataBlock
{
    unsigned char * output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

__global__ void copy_const_kernel( float* iptr, const float* cptr )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) // <- if cell contains "heater"
    {
        iptr[offset] = cptr[offset];
    }
}

__global__ void blend_kernel( float* outSrc, const float* inSrc )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0) left++;
    if (x == DIM-1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0) top += DIM;
    if (y == DIM-1) bottom -= DIM;

    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[left] + inSrc[right] + inSrc[top] + inSrc[bottom] - 4*inSrc[offset]);
}

void anim_gpu(DataBlock *d, int ticks)
{
    cudaEventRecord(d->start, 0);
    dim3 blocks(DIM/16,DIM/16);
    dim3 threads(16,16);
    CPUAnimBitmap *bitmap = d->bitmap;

    for (int i = 0; i < 90; i++)
    {
        copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap, d->dev_inSrc );

    cudaMemcpy( bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost );
    cudaEventRecord( d->stop, 0 );
    cudaEventSynchronize( d->stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, d->start, d->stop );

    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame: %3.1f ms\n", d->totalTime/d->frames ); 
}

__global__ void copy_const_kernel_tex( float* iptr )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc, x, y);
    if (c != 0) // <- if cell contains "heater"
    {
        iptr[offset] = c;
    }
}

__global__ void blend_kernel_tex(float *dst, bool dstOut)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;
    if (dstOut)
    {
        l = tex2D(texIn, x-1, y);
        r = tex2D(texIn, x+1, y);
        t = tex2D(texIn, x, y-1);
        b = tex2D(texIn, x, y+1);
        c = tex2D(texIn, x, y);
    }
    else
    {
        l = tex2D(texOut, x-1, y);
        r = tex2D(texOut, x+1, y);
        t = tex2D(texOut, x, y-1);
        b = tex2D(texOut, x, y+1);
        c = tex2D(texIn, x, y);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

void anim_gpu_tex(DataBlock *d, int ticks)
{
    cudaEventRecord(d->start, 0);
    dim3 blocks(DIM/16,DIM/16);
    dim3 threads(16,16);
    CPUAnimBitmap *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;

    for (int i = 0; i < 90; i++)
    {
        float *in, *out;
        if (dstOut)
        {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else
        {
            in = d->dev_outSrc;
            out = d->dev_inSrc;
        }

        copy_const_kernel_tex<<<blocks, threads>>>(in);
        blend_kernel_tex<<<blocks, threads>>>(out, dstOut);
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap, d->dev_inSrc );

    cudaMemcpy( bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost );
    cudaEventRecord( d->stop, 0 );
    cudaEventSynchronize( d->stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, d->start, d->stop );

    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame: %3.1f ms\n", d->totalTime/d->frames ); 
}

void anim_exit(DataBlock *d)
{
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);
    cudaEventDestroy(d->start);
    cudaEventDestroy(d->stop);
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);
}

int main(int argc, char** argv)
{
    printf("Start\n");
    
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc, imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc, imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc, imageSize ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR( cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM) );
    HANDLE_ERROR( cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM) );
    HANDLE_ERROR( cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM) );

    // intialize the constant data
    float *temp = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice ) );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice ) );
    free( temp );

    // bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu, (void (*)(void*))anim_exit );
    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu_tex, (void (*)(void*))anim_exit ); // <- texture

    printf("Finish\n");
    return 0;
}