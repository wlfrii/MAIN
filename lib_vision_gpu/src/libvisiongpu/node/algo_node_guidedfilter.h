#ifndef ALGONODE_GUIDED_FILTER_H
#define ALGONODE_GUIDED_FILTER_H
#include "algo_node_base.h"
#include <array>

GPU_ALGO_BEGIN
// enum{ TEMP_MAT_NUM = 5 };
void guidedFilter(cv::cuda::GpuMat &src, float eps, int radius, int scale, cudaStream_t stream, std::array<cv::cuda::GpuMat, 5> &tmp_mat);
struct GuidedFilterProperty : Property
{
    GuidedFilterProperty(float eps = 0.f, uint radius = 16, uint scale = 4)
		: Property(GUIDED_FILTER_NODE)
		, eps(eps)
		, radius(radius)
		, scale(scale)
	{}
	float eps;      //!< The regularization parameter
	int radius;     //!< The half-length of the box side length
	int scale;      //!< The ratio for 'pyrdown'(downsample) the image, try scale=box_radius/4 to scale=box_radius
};

class AlgoNodeGuidedFilter : public AlgoNodeBase
{
public:
    AlgoNodeGuidedFilter();
    ~AlgoNodeGuidedFilter();

    void process(cv::cuda::GpuMat &src, cudaStream_t stream = 0) override;

private:
    // The mat used for calculation
    // Note, these mat cannot be static temp variable.
    // Since the interface would be called in different thread.
    std::array<cv::cuda::GpuMat, 5> tmp_rgb_mat; 
	std::array<cv::cuda::GpuMat, 5> tmp_gray_mat;
};
GPU_ALGO_END
#endif // ALGONODE_GUIDED_FILTER_H
