#ifndef ALGONODE_UNEVEN_Y_H
#define ALGONODE_UNEVEN_Y_H
#include "algo_node_base.h"
#include <array>

GPU_ALGO_BEGIN
struct UnevenYProperty : Property
{
	UnevenYProperty()
		: Property(UNEVEN_Y_NODE), distance(1000), magnify(2), magnify0(0.85) {}
	float distance;
	float magnify;	// The max magnification of pixel luminance 
	float magnify0;	// The min magnification of pixel luminance
};


void createUnevenY(int width, int height, int distance, cv::cuda::GpuMat & uneven_y, cudaStream_t stream = 0);
void reduceUnevenY(cv::cuda::GpuMat & src, cv::cuda::GpuMat & uneven_y, std::array<cv::cuda::GpuMat, 2> &tmp, float magnify, float magniy0, cudaStream_t stream = 0);


class AlgoNodeUnevenY : public AlgoNodeBase
{
public:
	AlgoNodeUnevenY();
	~AlgoNodeUnevenY();

	void process(cv::cuda::GpuMat & src, cudaStream_t stream = 0) override;

private:
	cv::cuda::GpuMat uneven_y;
	enum{ NUM = 2 };
	std::array<cv::cuda::GpuMat, NUM> tmp;
};
GPU_ALGO_END
#endif //ALGONODEUNEVENY_H