#include "algo_node_image_adjust.h"

GPU_ALGO_BEGIN
AlgoNodeImageAdjust::AlgoNodeImageAdjust()
    : AlgoNodeBase()
{
	this->setProperty(std::make_shared<ImageAdjustProperty>());
}

void AlgoNodeImageAdjust::process(cv::cuda::GpuMat &src, cudaStream_t stream)
{
    auto image_prop = dynamic_cast<ImageAdjustProperty*>(property.get());

    adjustImageProp(src, image_prop->saturation, image_prop->contrast, image_prop->brightness, stream);
}
GPU_ALGO_END
