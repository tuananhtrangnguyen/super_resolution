#include "preprocessing.h"
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "cuda_utils.h"
#include "logging.h"
// #include "preprocess/preprocess.hpp"
#define INPUT_W 640
#define INPUT_H 512
#define BATCH_SIZE 4




__global__ void preprocessKernel(const uchar3* inputFrame, float* outputData, int width, int height, int batchSize) {
    int batchIdx = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        int pixelIdx = (batchIdx * 3 * width * height) + (y * width + x);
        uchar3 pixel = inputFrame[y * width + x];

        // Chuyển đổi ảnh từ uchar3 sang float, chuẩn hóa và lưu vào output
        // outputData[pixelIdx] = (float)pixel.x / 255.0f;  // Blue channel
        // outputData[pixelIdx + width * height] = (float)pixel.y / 255.0f;  // Green channel
        // outputData[pixelIdx + 2 * width * height] = (float)pixel.z / 255.0f;  // Red channel
        outputData[pixelIdx] = (float)pixel.z / 255.0f;  // Red channel
        outputData[pixelIdx + width * height] = (float)pixel.y / 255.0f;  // Green channel
        outputData[pixelIdx + 2 * width * height] = (float)pixel.x / 255.0f;  // Blue channel

    }
}


// Kernel CUDA thực hiện hậu xử lý (Post-processing)
__global__ void postprocessKernel(float* inputData, uchar3* outputFrame, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        int pixelIdx = y * width + x;

        uchar3 pixel;
        pixel.x = static_cast<unsigned char>(fminf(fmaxf(inputData[pixelIdx+ 2 * width * height] * 255.0f, 0.0f), 255.0f));                // Blue
        pixel.y = static_cast<unsigned char>(fminf(fmaxf(inputData[pixelIdx + width * height] * 255.0f, 0.0f), 255.0f)); // Green
        pixel.z = static_cast<unsigned char>(fminf(fmaxf(inputData[pixelIdx ] * 255.0f, 0.0f), 255.0f)); // Red

        outputFrame[pixelIdx] = pixel;
    }
}

// Hàm xử lý tiền xử lý (Preprocessing)
void preprocessData(uchar3* d_inputFrame, float* d_preprocessedData, int width, int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                (height + blockDim.y - 1) / blockDim.y, BATCH_SIZE);
    preprocessKernel<<<gridDim, blockDim>>>(d_inputFrame, d_preprocessedData, width, height, BATCH_SIZE);
    
    cudaDeviceSynchronize();
}


// Hàm thực hiện hậu xử lý (Post-processing)
void postprocessData(float* d_upscaledData, uchar3* d_postprocessedFrame, int width, int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                (height + blockDim.y - 1) / blockDim.y);
    postprocessKernel<<<gridDim, blockDim>>>(d_upscaledData, d_postprocessedFrame, width, height);
    cudaDeviceSynchronize();
}

