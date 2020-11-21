#include "algo_node_nonuniform.h"

GPU_ALGO_BEGIN
AlgoNodeNonuniform::AlgoNodeNonuniform()
{
	this->setProperty(std::make_shared<NonuniformProperty>());
}

AlgoNodeNonuniform::~AlgoNodeNonuniform()
{

}

void AlgoNodeNonuniform::process(cv::cuda::GpuMat & src, cudaStream_t stream)
{
	auto uneven_prop = dynamic_cast<NonuniformProperty*>(property.get());

	if (nonuniform_y.empty() || src.rows != nonuniform_y.rows || src.cols != nonuniform_y.cols)
	{
		createNonuniformY(src.cols, src.rows, uneven_prop->distance, nonuniform_y, stream);
	}

	reduceNonuniformY(src, nonuniform_y, uneven_prop->magnify, uneven_prop->magnify0, stream);
}
GPU_ALGO_END