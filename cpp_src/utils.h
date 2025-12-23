#ifndef __UTILS_H__
#define __UTILS_H__

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include "NvInferPlugin.h"


#define CHECK(call)                                                                      \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        }                                                                                \
    }

    
class MyLogger : public nvinfer1::ILogger
{
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t  _start;
    cudaEvent_t  _stop;

    /// Constructor
    GpuTimer()
        : _stream_id(0)
    {
        CHECK(cudaEventCreate(&_start));
        CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CHECK(cudaEventDestroy(_start));
        CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CHECK(cudaEventSynchronize(_stop));
        CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

std::shared_ptr<float> sam_preprocess(const cv::Mat& img, int target_width, int target_height, float mean, float std);

std::vector<char> load_engine(const std::string& path);



void infer(std::string engine_path, std::string img_path, std::string prompt);

#endif   // __UTILS_H__