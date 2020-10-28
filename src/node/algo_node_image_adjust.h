#ifndef ALGONODE_IMAGE_ADJUST_H
#define ALGONODE_IMAGE_ADJUST_H
#include "algo_node_base.h"

GPU_ALGO_BEGIN
struct ImageAdjustProperty : Property
{
	ImageAdjustProperty()
		: Property(IMAGE_ADJUST_NODE), saturation(0.f), brightness(0.f), contrast(0.f) {}
	float saturation;
	float brightness;
	float contrast;
};

class AlgoNodeImageAdjust : public AlgoNodeBase
{
public:
    AlgoNodeImageAdjust();
    virtual ~AlgoNodeImageAdjust(){}

    void process(cv::cuda::GpuMat &src, cudaStream_t stream = 0) override;
};
GPU_ALGO_END

#endif // ALGONODE_IMAGE_ADJUST_H
