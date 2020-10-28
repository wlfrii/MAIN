#include "algo_node_image_adjust.h"

GPU_ALGO_BEGIN
void adjustImageProp_RGB(cv::cuda::GpuMat &src, float saturation, float contrast, float brightness, cudaStream_t stream);
void AdjustImageProp_RGBA(cv::cuda::GpuMat &src, float saturation, float contrast, float brightness, cudaStream_t stream);

AlgoNodeImageAdjust::AlgoNodeImageAdjust()
    : AlgoNodeBase()
{

}

void AlgoNodeImageAdjust::process(cv::cuda::GpuMat &src, cudaStream_t stream)
{
    auto image_prop = dynamic_cast<ImageAdjustProperty*>(property.get());

    AdjustImageProp_RGBA(src, image_prop->saturation, image_prop->contrast, image_prop->brightness, stream);
}
GPU_ALGO_END
