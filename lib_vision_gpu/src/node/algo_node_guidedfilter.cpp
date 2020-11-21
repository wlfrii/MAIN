#include "algo_node_guidedfilter.h"

GPU_ALGO_BEGIN
AlgoNodeGuidedFilter::AlgoNodeGuidedFilter()
{
	this->setProperty(std::make_shared<GuidedFilterProperty>());
}

AlgoNodeGuidedFilter::~AlgoNodeGuidedFilter()
{

}

void AlgoNodeGuidedFilter::process(cv::cuda::GpuMat &src, cudaStream_t stream)
{
    if(tmp_rgb_mat[0].empty())
    {
        for(auto i = 0; i < 5; i++) {
			tmp_rgb_mat[i] = cv::cuda::GpuMat(src.size(), CV_32FC3);
			tmp_gray_mat[i] = cv::cuda::GpuMat(src.size(), CV_32FC1);
        }	
    }
    auto guidedfilter_prop = dynamic_cast<GuidedFilterProperty*>(property.get());
	if (abs(guidedfilter_prop->eps < 1e-5))
		return;
	if (src.channels() == 1) {
		guidedFilter(src, guidedfilter_prop->eps, guidedfilter_prop->radius, guidedfilter_prop->scale, stream, tmp_gray_mat);
	}
	else {
		guidedFilter(src, guidedfilter_prop->eps, guidedfilter_prop->radius, guidedfilter_prop->scale, stream, tmp_rgb_mat);
	}
}

GPU_ALGO_END