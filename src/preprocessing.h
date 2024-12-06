// Trong preprocessing.h
#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.hpp"
#include "cuda_utils.h"
#include "logging.h"
// #include "preprocess/preprocess.hpp"

#define INPUT_W 640
#define INPUT_H 512
#define BATCH_SIZE 4

// // Struct để lưu trữ bộ nhớ GPU
// struct GPUMemoryManager {
//     float* inputBuffer;      // Buffer cho TensorRT input
//     float* outputBuffer;     // Buffer cho TensorRT output  
//     uchar3* d_inputFrame;    // Buffer cho frame input
//     float* d_preprocessedData; // Buffer cho dữ liệu đã pre-process
//     float* d_upscaledData;    // Buffer cho dữ liệu upscaled
//     uchar3* d_postprocessedFrame; // Buffer cho frame output

//     GPUMemoryManager(size_t inputSize, size_t outputSize) {
//         // Cấp phát bộ nhớ cho TensorRT buffers
//         CUDA_CHECK(cudaMalloc(&inputBuffer, inputSize));
//         CUDA_CHECK(cudaMalloc(&outputBuffer, outputSize));
        
//         // Cấp phát bộ nhớ cho CUDA buffers
//         CUDA_CHECK(cudaMalloc(&d_inputFrame, INPUT_W * INPUT_H * sizeof(uchar3)));
//         CUDA_CHECK(cudaMalloc(&d_preprocessedData, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_upscaledData, BATCH_SIZE * 3 * INPUT_H * 4 * INPUT_W * 4 * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_postprocessedFrame, INPUT_W * 4 * INPUT_H * 4 * sizeof(uchar3)));
//     }

//     ~GPUMemoryManager() {
//         cudaFree(inputBuffer);
//         cudaFree(outputBuffer);
//         cudaFree(d_inputFrame);
//         cudaFree(d_preprocessedData);
//         cudaFree(d_upscaledData);
//         cudaFree(d_postprocessedFrame);
//     }
// };
// Kernel declarations
// __global__ void preprocessKernel(const uchar3* inputFrame, float* outputData, int width, int height, int batchSize);
// __global__ void postprocessKernel(float* inputData, uchar3* outputFrame, int width, int height);

// // Processing functions
// void preprocess(cv::Mat& frame, GPUMemoryManager& gpu_mem, cudaStream_t stream);
// void postprocess(GPUMemoryManager& gpu_mem, cv::Mat& outputFrame, cudaStream_t stream);





__global__ void preprocessKernel(const uchar3* inputFrame, float* outputData, int width, int height, int batchSize);


// Kernel CUDA thực hiện hậu xử lý (Post-processing)
__global__ void postprocessKernel(float* inputData, uchar3* outputFrame, int width, int height) ;

// Hàm xử lý tiền xử lý (Preprocessing)
void preprocessData(uchar3* d_inputFrame, float* d_preprocessedData, int width, int height) ;

// Hàm thực hiện hậu xử lý (Post-processing)
void postprocessData(float* d_upscaledData, uchar3* d_postprocessedFrame, int width, int height) ;
#endif // PREPROCESSING_H