#ifndef ALGONODE_GAMMA_H
#define ALGONODE_GAMMA_H
#include "algo_node_base.h"
#include "algo_node_guidedfilter.h"
#include <array>

GPU_ALGO_BEGIN
class AlgoNodeGamma : public AlgoNodeBase
{
public:
    AlgoNodeGamma();
    ~AlgoNodeGamma();

    void process(cv::cuda::GpuMat & src, cudaStream_t stream = 0) override;

private:
	enum { NUM = 2 };
	std::array<cv::cuda::GpuMat, NUM> tmp;
	cv::cuda::GpuMat gamma;

	AlgoNodeGuidedFilter* guided_filter_algo;
};
GPU_ALGO_END
#endif // ALGONODEGAMMA_H
