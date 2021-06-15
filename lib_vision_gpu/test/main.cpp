#include <opencv2/opencv.hpp>
#include <atomic>
#include <libvisiongpu/gpu_algorithm_pipeline_manager.h>
#include <libvisiongpu/gpu_algorithm_func.h>
#include <vector>
#include <string>

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
	gpu::cvtImageFormat(cu_image, gpu::CVT_8U_TO_32F);

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
	struct ImageInfo {
		std::string folder;
		std::string name;
		std::string fmt;

		std::string getPath() const
		{
			return folder + name + fmt;
		}

		std::string setSavePath(const std::string &suffix) const
		{
			return folder + name + suffix + fmt;
		}
	};
	std::vector<ImageInfo> info;
	info.push_back({ "D:/MyProjects/Vision/test_data/","test_image",".bmp" });
	info.push_back({ "E:/Rii/Surgerii_ProjectReports/20201016_科技部项目-视觉相关/enhancement/", "1", ".jpg" });
	//info.push_back({ info[info.size - 1].folder, "2", ".bmp" });
	//info.push_back({ info[info.size - 1].folder, "3", ".jpg" });
	int id = 1;

	cv::Mat image = cv::imread(info[id].getPath());
	std::atomic<bool> flag;

	gpu::AlgoPipelineManager::getInstance()->intialize();
	//gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeNonuniform());
	//gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::NonuniformProperty>(2, 0.99));
	//cv::Mat res1(image.size(), CV_32FC4);
	//gpu::AlgoPipelineManager::getInstance()->process(image, res1, flag);

	//
	//gpu::AlgoPipelineManager::getInstance()->clearAlgoNode();
	//gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
	//gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));
	//cv::Mat res2(image.size(), CV_32FC4);
	//gpu::AlgoPipelineManager::getInstance()->process(res1, res2, flag);
	
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(50, 55));
	cv::Mat res3(image.size(), image.type());
	gpu::AlgoPipelineManager::getInstance()->process(image, res3, flag);
	//cv::imwrite(info[id].setSavePath("_contrast"), res3);

	return 0;
}