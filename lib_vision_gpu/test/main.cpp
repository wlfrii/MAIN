#include <opencv2/opencv.hpp>
#include <atomic>
#include "gpu_algorithm_pipeline_manager.h"
#include "gpu_algorithm_func.h"

void testGuidedFilterAlgo()
{
	cv::Mat image = cv::imread("../test_data/1.png");

	gpu::AlgoPipelineManager::getInstance()->intialize();
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGuidedFilter());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GuidedFilterProperty>(0.005), gpu::TreeType(0));

	cv::Mat res(image.size(), image.type());
	std::atomic<bool> flag;
	gpu::AlgoPipelineManager::getInstance()->process(image, res, flag);
	
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	cv::Mat gray_res(gray_image.size(), gray_image.type());
	gpu::AlgoPipelineManager::getInstance()->process(gray_image, gray_res, flag);

	/*cv::imshow("DST", res);
	cv::waitKey(10);*/
	return;
}

void testColorSpace()
{
	cv::Mat image = cv::imread("D:/MyProjects/Vision/test_data/test_image.bmp");
	cv::cuda::GpuMat cu_image;
	cu_image.upload(image);
	gpu::convertImageFormat(cu_image);

	cv::cuda::GpuMat cu_hsv(cu_image.size(), cu_image.type());
	gpu::cvtColor(cu_image, cu_hsv, gpu::BGR2HSV);
	cv::Mat res; cu_hsv.download(res);

	gpu::cvtColor(cu_hsv, cu_image, gpu::HSV2BGR);
	cv::Mat res2; cu_image.download(res2);

	cv::cuda::GpuMat tmp(image.size(), CV_32FC4);
	gpu::cvtColor(cu_hsv, tmp, gpu::HSV2BGRA);
	cv::Mat res3; 
	tmp.download(res3);
	return;
}

int main()
{
	//testGuidedFilterAlgo();
	//testColorSpace();


	cv::Mat image = cv::imread("D:/MyProjects/Vision/test_data/test_image.bmp");
	std::atomic<bool> flag;

	gpu::AlgoPipelineManager::getInstance()->intialize();
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::UnevenYProperty>(2, 0.99));
	cv::Mat res1(image.size(), CV_32FC4);
	gpu::AlgoPipelineManager::getInstance()->process(image, res1, flag);

	
	gpu::AlgoPipelineManager::getInstance()->clearAlgoNode();
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));
	cv::Mat res2(image.size(), CV_32FC4);
	gpu::AlgoPipelineManager::getInstance()->process(res1, res2, flag);
	
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(0, 3));
	cv::Mat res3(image.size(), image.type());
	gpu::AlgoPipelineManager::getInstance()->process(res2, res3, flag);


	return 0;
}