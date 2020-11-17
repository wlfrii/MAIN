# lib_vision_gpu

A library includes some algorithm running in the GPU with the help of CUDA and OpenCV.

## How to use this library?

### 1. Requires

   - CUDA should be installed.
   - OpenCV should be download and complied with CUDA.

### 2. Using the library

The header file `gpu_algorithm_pipeline_manager.h` need to be included only. And the `lib_vision_gpu.lib` (or `liblib_vision_gpu.o`) should be included in your project in _Win32_ (in _Linux_).

There are two stream included in the library in current version. Below is an sample code to make use of this library.
```cpp
// load an image, be processed
cv::Mat image = cv::imread("../test_data/2.png");

// The flag to specify whether the process has done.
std::atomic<bool> flag = false;

// Initialize the library first
gpu::AlgoPipelineManager::getInstance()->intialize();

// Add the specified the algo node to the AlgoPipelineManager
gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY());
// Set the corresponding property
gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::UnevenYProperty>(2, 0.99));

// Pre-allocate a memory to store the processed image
cv::Mat res1(image.size(), CV_32FC4);

// Do the process
while(!flag){
   bool ret = gpu::AlgoPipelineManager::getInstance()->process(image, res1, flag);
   // If the return is false, the processing failed.
   if(!ret) break;
}
```
If one need to do the processing with more than one node, and also want to check processed image after each algo node, the additional codes are sampled below.

```cpp
// Clear the exist algo node when we do not need the added node
gpu::AlgoPipelineManager::getInstance()->clearAlgoNode();

// Add new algo node and set the property
gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));

// Pre-allocate a memory to store the processed image so that the processed result could be checked
cv::Mat res2(image.size(), CV_32FC4);
gpu::AlgoPipelineManager::getInstance()->process(res1, res2, flag);

// Repeat the operation
gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(0, 3));
cv::Mat res3(image.size(), image.type());
gpu::AlgoPipelineManager::getInstance()->process(res2, res3, flag);
...
```

If one just want the final processed result. All the algo nodes could be added first, and then call the `process()` interface.

Note, the type of the `output` of the `process()` interface would be a <font color=blue>float</font> if 
   1. the `output` is empty (which means that a memory is not pre-allocate and the type is not specify);
   2. the `output` has not specified the type.

So only when a specified type, must be <font color=purple>CV_8UC1</font>, <font color=purple>CV_8UC3</font>, or <font color=purple>CV_8UC3</font>, the output has, an <font color=blue>uchar</font> image can be return.

