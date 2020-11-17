#include "algo_node_uneven_y.h"

GPU_ALGO_BEGIN
AlgoNodeUnevenY::AlgoNodeUnevenY()
	: AlgoNodeBase()
{
	this->setProperty(std::make_shared<UnevenYProperty>());
}

AlgoNodeUnevenY::~AlgoNodeUnevenY()
{

}

void AlgoNodeUnevenY::process(cv::cuda::GpuMat & src, cudaStream_t stream)
{
	auto uneven_prop = dynamic_cast<UnevenYProperty*>(property.get());

	if (uneven_y.empty() || src.rows != uneven_y.rows || src.cols != uneven_y.cols)
	{
		createUnevenY(src.cols, src.rows, uneven_prop->distance, uneven_y, stream);
	}

	reduceUnevenY(src, uneven_y, uneven_prop->magnify, uneven_prop->magnify0, stream);
}
GPU_ALGO_END