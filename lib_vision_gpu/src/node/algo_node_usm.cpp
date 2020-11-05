#include "algo_node_usm.h"

GPU_ALGO_BEGIN
AlgoNodeUSM::AlgoNodeUSM()
    : AlgoNodeBase()
{
	this->setProperty(std::make_shared<USMProperty>());
}

void AlgoNodeUSM::process(cv::cuda::GpuMat &src, cudaStream_t stream)
{

}



GPU_ALGO_END
