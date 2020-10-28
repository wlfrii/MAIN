#include "algo_node_rectify.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

GPU_ALGO_BEGIN
AlgoNodeRectify::AlgoNodeRectify()
    : AlgoNodeBase()
{
}

AlgoNodeRectify::~AlgoNodeRectify()
{
}

void AlgoNodeRectify::process(cv::cuda::GpuMat &src, cudaStream_t stream)
{
    if(tmp.size() != src.size() || tmp.type() != src.type())
        tmp = cv::cuda::GpuMat(src.size(),src.type());

    cv::cuda::Stream cv_stream;
    cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    auto rectify_prop = dynamic_cast<RectifyProperty*>(property.get());
    cv::cuda::remap(src, tmp, rectify_prop->mapx, rectify_prop->mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0, cv_stream);
    tmp.copyTo(src);
}



GPU_ALGO_END
