#ifndef ALGONODE_UNEVEN_Y_H
#define ALGONODE_UNEVEN_Y_H
#include "algo_node_base.h"
#include <array>

GPU_ALGO_BEGIN
struct UnevenYProperty : Property
{
	UnevenYProperty(float magnify = 2.0, float magnify0 = 1, float distance = 1000.0)
		: Property(UNEVEN_Y_NODE)
		, magnify(magnify)
		, magnify0(magnify0)
		, distance(distance)
	{}
	float magnify;	//!< The max magnification of pixel luminance 
	float magnify0;	//!< The min magnification of pixel luminance
	float distance;
};


void createUnevenY(int width, int height, int distance, cv::cuda::GpuMat & uneven_y, cudaStream_t stream = 0);
void reduceUnevenY(cv::cuda::GpuMat & src, cv::cuda::GpuMat & uneven_y, float magnify, float magniy0, cudaStream_t stream = 0);


class AlgoNodeUnevenY : public AlgoNodeBase
{
public:
	AlgoNodeUnevenY();
	~AlgoNodeUnevenY();

	void process(cv::cuda::GpuMat & src, cudaStream_t stream = 0) override;

private:
	cv::cuda::GpuMat uneven_y;
};
GPU_ALGO_END
#endif //ALGONODEUNEVENY_H