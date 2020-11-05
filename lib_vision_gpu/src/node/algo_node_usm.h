#ifndef ALGONODE_USM_H
#define ALGONODE_USM_H
#include "../def/define.h"
#include "algo_node_base.h"

GPU_ALGO_BEGIN
struct USMProperty : Property
{
	USMProperty(float usm = 0.f, uchar radius = 4)
		: Property(USM_NODE), usm(usm), radius(radius) {}
	float usm;			//!< The extent for increasing the characters of the input image
	uchar radius;		//!< The half-length of the box side length
};

class AlgoNodeUSM : public AlgoNodeBase
{
public:
    AlgoNodeUSM();

    void process(cv::cuda::GpuMat & src, cudaStream_t stream) override;

};

GPU_ALGO_END
#endif // ALGONODEUSM_H
