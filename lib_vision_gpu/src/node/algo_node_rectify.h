#ifndef ALGONODE_RECTIFY_H
#define ALGONODE_RECTIFY_H
#include "algo_node_base.h"

GPU_ALGO_BEGIN
struct RectifyProperty : Property
{
	RectifyProperty()
		: Property(RECTIFY_NODE) {}
	RectifyProperty(cv::cuda::GpuMat mapx, cv::cuda::GpuMat mapy)
		: Property(RECTIFY_NODE), mapx(mapx), mapy(mapy) {}
	cv::cuda::GpuMat mapx;
	cv::cuda::GpuMat mapy;
};

class AlgoNodeRectify : public AlgoNodeBase
{
public:
    AlgoNodeRectify();
    ~AlgoNodeRectify();

    void process(cv::cuda::GpuMat & src, cudaStream_t stream) override;

private:
    cv::cuda::GpuMat tmp;
};
GPU_ALGO_END
#endif // ALGONODE_RECTIFY_H
