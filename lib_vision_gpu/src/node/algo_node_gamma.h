#ifndef ALGONODE_GAMMA_H
#define ALGONODE_GAMMA_H
#include "algo_node_base.h"
#include "algo_node_guidedfilter.h"
#include <array>

GPU_ALGO_BEGIN
struct GammaProperty : Property
{
	GammaProperty(float eps = 0.2)
		: Property(GAMMA_NODE)
		, eps(eps)
	{}

	float eps; // refers to GuidedFilterProperty
};

class AlgoNodeGamma : public AlgoNodeBase
{
public:
    AlgoNodeGamma();
    ~AlgoNodeGamma();

    void process(cv::cuda::GpuMat & src, cudaStream_t stream = 0) override;

private:
	enum { NUM = 2 };
	cv::cuda::GpuMat tmp;
	cv::cuda::GpuMat gamma;

	cv::cuda::GpuMat v; // V in hsv

	AlgoNodeGuidedFilter* guided_filter_algo;
	float	tmp_eps;
};
GPU_ALGO_END
#endif // ALGONODEGAMMA_H
