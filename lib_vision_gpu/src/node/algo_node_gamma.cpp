#include "algo_node_gamma.h"
#include "../gpu_algorithm_func.h"

GPU_ALGO_BEGIN
void AdaptiveGamma(cv::cuda::GpuMat &src, cv::cuda::GpuMat &v, cudaStream_t &stream, cv::cuda::GpuMat &tmp, float alpha = 0.0, float ref_L = 0.0);

AlgoNodeGamma::AlgoNodeGamma()
    : guided_filter_algo(new AlgoNodeGuidedFilter())
	, tmp_eps(0.1)
{
	this->setProperty(std::make_shared<GammaProperty>(tmp_eps));
}

AlgoNodeGamma::~AlgoNodeGamma()
{

}

void AlgoNodeGamma::process(cv::cuda::GpuMat & src, cudaStream_t stream)
{
    if(v.empty() || v.size() != src.size()) {
        v = cv::cuda::GpuMat(src.size(), CV_32FC1);
		tmp = cv::cuda::GpuMat(src.size(), CV_32FC3);

		gamma = cv::cuda::GpuMat(src.rows, src.cols, CV_32FC1);
    }
	calcVbyHSV(src, v);
#if CU_DEBUG
	cv::Mat test_src; src.download(test_src);
	cv::Mat test_v; v.download(test_v);
#endif
		
	// guided hsv image, get the filtered v
	auto gamma_prop = dynamic_cast<GammaProperty*>(property.get());
	if (abs(gamma_prop->eps - tmp_eps) > 1e-5) {
		guided_filter_algo->setProperty(std::make_shared<GuidedFilterProperty>(gamma_prop->eps));
		tmp_eps = gamma_prop->eps;
	}
	guided_filter_algo->process(v, stream);
#if CU_DEBUG
	cv::Mat test_v2; v.download(test_v2);
#endif

	// AdaptiveGamma(src, v, stream, tmp);
	AdaptiveGamma(src, v, stream, tmp, gamma_prop->alpha, gamma_prop->ref_L);
}


GPU_ALGO_END
