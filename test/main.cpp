#include <opencv2/opencv.hpp>
#include <atomic>
#include "gpu_algorithm_pipeline_manager.h"
#include "gpu_algorithm_func.h"

void testGuidedFilterAlgo()
{
	cv::Mat image = cv::imread("../test_data/1.png");

	gpu::AlgoPipelineManager::getInstance()->intialize();
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GuidedFilterProperty>(gpu::GuidedFilterProperty(0.05)), gpu::TreeType(0));

	cv::Mat res(image.size(), image.type());
	std::atomic<bool> flag;
	gpu::AlgoPipelineManager::getInstance()->process(image, res, gpu::LEFT_EYE, flag);
	
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	cv::Mat gray_res(gray_image.size(), gray_image.type());
	gpu::AlgoPipelineManager::getInstance()->process(gray_image, gray_res, gpu::LEFT_EYE, flag);

	/*cv::imshow("DST", res);
	cv::waitKey(10);*/
	return;
}

void testGamma()
{
	cv::Mat image = cv::imread("../test_data/2.png");

	gpu::AlgoPipelineManager::getInstance()->intialize();
	cv::Mat res(image.size(), image.type());
	std::atomic<bool> flag;
	gpu::AlgoPipelineManager::getInstance()->process(image, res, gpu::LEFT_EYE, flag);

	return;
}

void testUnevenY()
{
	cv::cuda::GpuMat uneven_y;
	//gpu::createUnevenY(1920, 1080, 1000, uneven_y);

	cv::Mat test(uneven_y.size(), uneven_y.type());
	uneven_y.download(test);
}

int main()
{
	//testGuidedFilterAlgo();
	//testGamma();
	//testUnevenY();
	cv::Mat image = cv::imread("../test_data/2.png");
	cv::Mat res(image.size(), image.type());
	std::atomic<bool> flag;

	gpu::AlgoPipelineManager::getInstance()->intialize();
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY(), gpu::LEFT_EYE);
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma(), gpu::LEFT_EYE);
	gpu::AlgoPipelineManager::getInstance()->process(image, res, gpu::LEFT_EYE, flag);


	return 0;
}