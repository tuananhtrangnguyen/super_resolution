



#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "preprocessing.h"
#include <string>
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>

#include "config.hpp"
#include "cuda_utils.h"
#include "logging.h"

#define INPUT_BLOB_NAME  "input"
#define OUTPUT_BLOB_NAME  "output"

static Logger gLogger;

using namespace nvinfer1;

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output) {
    nvtx3::scoped_range r{"doInference"};
    
    if (!context.allInputDimensionsSpecified()) {
        std::cerr << "Error: Not all input dimensions specified before inference" << std::endl;
        exit(1);
    }
    
    if(!context.enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "Error: Inference execution failed" << std::endl;
        exit(1);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


int main(int argc, char** argv) {
    std::string video_path = "/home/gremsy/Desktop/test/VID_IR_0.mp4";
    std::string output_path = "/home/gremsy/Desktop/test/out1.mp4"; 
    std::string engine_name = "/home/gremsy/Desktop/onnx/engine/vgg_best.trt"; 

    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read " << engine_name << std::endl;
        return -1;
    }

    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    std::cerr << "size: " << size << std::endl;
    
    trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();

    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    std::cout<<"input name "<<engine->getBindingName(0) <<std::endl;
    std::cout<<"output name "<<engine->getBindingName(1) <<std::endl;
    // size_t inputSize = BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    // size_t outputSize = BATCH_SIZE * OUTPUT_SIZE * sizeof(float);


    void* buffers[2];
    
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);


    Dims inputDims = engine->getBindingDimensions(inputIndex);
    inputDims.d[0] = BATCH_SIZE;  // Set batch size
    context->setBindingDimensions(inputIndex, inputDims);

    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "Error: Not all input dimensions specified" << std::endl;
        return -1;
    }

    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    uchar3* d_inputFrame;
    float* d_preprocessedData;
    uchar3* d_outputFrame;

    size_t inputSize = INPUT_H * INPUT_W * sizeof(uchar3);
    size_t preprocessedSize = INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    size_t outputSize = INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE * sizeof(uchar3);

    cudaMalloc((void**)&d_inputFrame, inputSize);
    cudaMalloc((void**)&d_preprocessedData, preprocessedSize);
    cudaMalloc((void**)&d_outputFrame, outputSize);


    std::vector<float> data(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
    std::vector<float> output(BATCH_SIZE * OUTPUT_SIZE);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << video_path << std::endl;
        return -1;
    }

    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '5');
    cv::VideoWriter writer(output_path, fourcc, cap.get(cv::CAP_PROP_FPS), cv::Size(INPUT_W * OUT_SCALE, INPUT_H * OUT_SCALE));

    int frame_count = 0;
    auto global_start_time = std::chrono::high_resolution_clock::now();
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::resizeWindow("Output", 1920, 1280);
    
    while (true) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(INPUT_W, INPUT_H));
        // std::cout << "Input frame (top-left 100 pixels):" << std::endl;
        // for (int i = 0; i < 10; ++i) {
        //     for (int j = 0; j < 10; ++j) {
        //         cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
        //         std::cout << "[" << int(pixel[2]) << ", " << int(pixel[1]) << ", " << int(pixel[0]) << "] ";
        //     }
        //     std::cout << std::endl;
        // }
        cv::Mat result(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3);
        int OUTPUT_C = 3;
        int OUTPUT_H = INPUT_H * OUT_SCALE;
        int OUTPUT_W = INPUT_W * OUT_SCALE;

        // {
        nvtxRangePushA("Preprocess");
        cudaMemcpyAsync(d_inputFrame, frame.data, inputSize, cudaMemcpyHostToDevice);
        preprocessData(d_inputFrame, d_preprocessedData, INPUT_W, INPUT_H);
        nvtxRangePop();
        // }
        // {
        //     nvtx3::scoped_range r{"Preprocessing"};
        //     cudaMemcpy(d_inputFrame, frame.data, inputSize, cudaMemcpyHostToDevice);
        //     preprocessData(d_inputFrame, d_preprocessedData, INPUT_W, INPUT_H);

        //     // Sao chép dữ liệu từ GPU về CPU
        //     cudaMemcpy(data.data(), d_preprocessedData, preprocessedSize, cudaMemcpyDeviceToHost);

        //     // In dữ liệu top-left 100 pixels
        //     std::cout << "Preprocessed data (top-left 100 pixels, multiplied by 255):" << std::endl;
        //     for (int i = 0; i < 10; ++i) {
        //         for (int j = 0; j < 10; ++j) {
        //             int index = i * INPUT_W + j;
        //             float red  = data[index] * 255.0f;
        //             float green = data[index + INPUT_H * INPUT_W] * 255.0f;
        //             float blue = data[index + 2 * INPUT_H * INPUT_W] * 255.0f;
        //             std::cout << "[" << static_cast<int>(red) << ", " 
        //                     << static_cast<int>(green) << ", " 
        //                     << static_cast<int>(blue) << "] ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }
        // {
        nvtxRangePushA("Infer");
        cudaMemcpyAsync(buffers[inputIndex], d_preprocessedData, preprocessedSize, cudaMemcpyDeviceToDevice, stream);

        // Inference TensorRT
        doInference(*context, stream, buffers, output.data());
        nvtxRangePop();  
        // }
        // {
        //     nvtx3::scoped_range r{"Inference"};

        //     // Sao chép dữ liệu từ d_preprocessedData sang buffers[inputIndex] trước khi inference
        //     CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], d_preprocessedData, preprocessedSize, cudaMemcpyDeviceToDevice, stream));

        //     // In dữ liệu trong input buffer
        //     std::vector<float> inputBufferData(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
        //     CUDA_CHECK(cudaMemcpy(inputBufferData.data(), buffers[inputIndex], preprocessedSize, cudaMemcpyDeviceToHost));

        //     // In dữ liệu top-left 100 pixels trong input buffer
        //     std::cout << "Input buffer data (top-left 100 pixels, multiplied by 255):" << std::endl;
        //     for (int i = 0; i < 10; ++i) {
        //         for (int j = 0; j < 10; ++j) {
        //             int index = i * INPUT_W + j;
        //             float red = inputBufferData[index] * 255.0f;
        //             float green = inputBufferData[index + INPUT_H * INPUT_W] * 255.0f;
        //             float blue = inputBufferData[index + 2 * INPUT_H * INPUT_W] * 255.0f;
        //             std::cout << "[" << static_cast<int>(red) << ", "
        //                     << static_cast<int>(green) << ", "
        //                     << static_cast<int>(blue) << "] ";
        //         }
        //         std::cout << std::endl;
        //     }

        //     // Tiến hành inference
        //     doInference(*context, stream, buffers, output.data());

        //     // Sau khi inference, sao chép dữ liệu kết quả từ GPU về CPU
        //     std::vector<float> outputData(OUTPUT_SIZE);
        //     CUDA_CHECK(cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));

        //     // In kết quả sau khi inference (ví dụ: top-left 100 pixels của kết quả)
        //     std::cout << "Output data (top-left 100 pixels):" << std::endl;
        //     for (int i = 0; i < 10; ++i) {
        //         for (int j = 0; j < 10; ++j) {
        //             int index = i * OUTPUT_W + j;
        //             float red = outputData[index] * 255.0f;
        //             float green = outputData[index + OUTPUT_H * OUTPUT_W] * 255.0f;
        //             float blue = outputData[index + 2 * OUTPUT_H * OUTPUT_W] * 255.0f;
        //             std::cout << "[" << static_cast<int>(red) << ", "
        //                     << static_cast<int>(green) << ", "
        //                     << static_cast<int>(blue) << "] ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }



        // cv::Mat result(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3);
        // int OUTPUT_C = 3;
        // int OUTPUT_H = INPUT_H * OUT_SCALE;
        // int OUTPUT_W = INPUT_W * OUT_SCALE;
        // {
            // postprocessData(buffers[outputIndex], d_outputFrame, INPUT_W * OUT_SCALE, INPUT_H * OUT_SCALE);
        nvtxRangePushA("Postprocess");
        postprocessData(reinterpret_cast<float*>(buffers[outputIndex]), d_outputFrame, INPUT_W * OUT_SCALE, INPUT_H * OUT_SCALE);
        cudaMemcpyAsync(result.data, d_outputFrame, outputSize, cudaMemcpyDeviceToHost);
        nvtxRangePop();  
        // }
        
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time).count();
        double fps = 1000.0 / frame_time;
        std::cout << "Frame " << frame_count << ": " << fps << " FPS" << std::endl;
        // std::cout << "Post-processed frame (top-left 100 pixels):" << std::endl;
        // for (int i = 0; i < 10; ++i) {
        //     for (int j = 0; j < 10; ++j) {
        //         cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
        //         std::cout << "[" << int(pixel[2]) << ", " << int(pixel[1]) << ", " << int(pixel[0]) << "] ";
        //     }
        //     std::cout << std::endl;
        // }


        cv::imshow("Output", result);
        writer.write(result);

        if (cv::waitKey(1) == 27) break;

        frame_count++;

    }

    auto global_end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time).count();
    std::cout << "Processed " << frame_count << " frames in " << elapsed_seconds << " seconds (" << frame_count / elapsed_seconds << " FPS overall)." << std::endl;

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    CUDA_CHECK(cudaFree(d_inputFrame));
    CUDA_CHECK(cudaFree(d_preprocessedData));
    CUDA_CHECK(cudaFree(d_outputFrame));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}