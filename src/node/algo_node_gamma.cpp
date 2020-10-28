#include "algo_node_gamma.h"

GPU_ALGO_BEGIN
void AdaptiveGamma(cv::cuda::GpuMat &src, const cudaStream_t &stream, std::array<cv::cuda::GpuMat, 2> &tmp);

AlgoNodeGamma::AlgoNodeGamma()
    : AlgoNodeBase()
	, guided_filter_algo(new AlgoNodeGuidedFilter())
{
	guided_filter_algo->setProperty(std::make_shared<GuidedFilterProperty>(GuidedFilterProperty(0.05)));

	
}

AlgoNodeGamma::~AlgoNodeGamma()
{

}

void AlgoNodeGamma::process(cv::cuda::GpuMat & src, cudaStream_t stream)
{
    if(tmp[0].size() != src.size()) {
        for(auto i = 0; i < NUM; i++)
			tmp[i] = cv::cuda::GpuMat(src.size(), CV_8UC3);

		gamma = cv::cuda::GpuMat(src.rows, src.cols, CV_32FC1);
    }
	// convert image to hsv
	cv::cuda::cvtColor(src, tmp[0], cv::COLOR_BGRA2BGR);
	cv::cuda::cvtColor(tmp[0], tmp[1], cv::COLOR_BGR2HSV);

	tmp[1].copyTo(tmp[0]);

	// guided hsv image, get the filtered v
	guided_filter_algo->process(tmp[1], stream);

	/*cv::Mat t1(1080, 1920, CV_8UC3); tmp[0].download(t1);
	cv::Mat t2(1080, 1920, CV_8UC3); tmp[1].download(t2);*/

	AdaptiveGamma(src, stream, tmp);
}

GPU_ALGO_END
