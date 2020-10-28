# lib_vision_gpu

A library includes some algorithm running in the GPU with the help of CUDA and OpenCV.

## How to use this library?

### 1. Requires

   - CUDA should be installed.
   - OpenCV should be download and complied with CUDA.

### 2. Using the library

The header file `gpu_algorithm_pipeline_manager.h` should be included only.
There are two stream included in the library in current version. Below is an sample code to make use of this library.
```C++
// load an image
cv::Mat image = cv::imread("../test_data/2.png");
// pre-allocate a memory to store the processed image
cv::Mat res(image.size(), image.type());
// The flag to specify whether the process has done.
std::atomic<bool> flag;
// Initialize the library first
gpu::AlgoPipelineManager::getInstance()->intialize();
// Add some algo node in LEFT_EYE stream
gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY(), gpu::LEFT_EYE);
gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma(), gpu::LEFT_EYE);
// Process the image with the added algo node
gpu::AlgoPipelineManager::getInstance()->process(image, res, gpu::LEFT_EYE, flag);
```