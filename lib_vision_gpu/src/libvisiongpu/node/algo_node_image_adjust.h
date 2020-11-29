#ifndef ALGONODE_IMAGE_ADJUST_H
#define ALGONODE_IMAGE_ADJUST_H
#include "algo_node_base.h"

GPU_ALGO_BEGIN
/*@brief Adjust the Saturation, contrast, and brightness.
 * saturation [0, 100], 50 is the default value.
 * contrast [0, 100], 50 is the default value.
 * brightness [0, 100], 50 is the default value.
 */
void adjustImageProp(cv::cuda::GpuMat &src, char saturation, char contrast, char brightness, cudaStream_t stream = 0);

struct ImageAdjustProperty : Property
{
    ImageAdjustProperty(char saturation = 50, char contrast = 50, char brightness = 50)
		: Property(IMAGE_ADJUST_NODE)
		, saturation(saturation)
		, contrast(contrast)
		, brightness(brightness)
	{}
	char saturation;	//!< saturation [0, 100], 50 is the default value.
	char contrast;		//!< contrast [0, 100], 50 is the default value.
	char brightness;	//!< brightness [0, 100], 50 is the default value.
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
