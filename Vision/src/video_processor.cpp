#include "video_processor.h"
#include <gpu_algorithm_pipeline_manager.h>
#include "def/mtimer.h"
#include <iostream>
#include <atomic>

namespace
{
	std::string getWriteFilename(const std::string &filename)
	{
		auto pos = filename.find_last_of('.');
		auto time_str = mtimer::getCurrentTimeStr();
		std::string res;
		res = filename.substr(0, pos - 1) + "_p" + time_str + filename.substr(pos, filename.size() - 1);

		return res;
	}
}



std::shared_ptr<VideoProcessor> VideoProcessor::getInstance()
{
	static std::shared_ptr<VideoProcessor> processor(new VideoProcessor());
	return processor;
}


bool VideoProcessor::processVideo(const std::string &filename, bool is_write/* = false*/)
{
	// Check gpu::AlgoPipelineManager
	if (!gpu::AlgoPipelineManager::getInstance()->isReady()) {
		printf("VidelProcessor: The AlgoPipelineManager should be initialzed first!\n");
		return false;
	}

	// Open video
	cv::Mat frame;
	frame = capture.open(filename);
	if (!capture.isOpened()) {
		printf("VideoProcessor: Cannot open the specified videl!\n");
		return false;
	}

	// Initialize VideoWriter is needed
	std::string write_filename = ::getWriteFilename(filename);
	if (is_write) 
	{
		cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH) * 2, capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		writer.open(write_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);
	}

	// Start process
	auto start_point = mtimer::getDurationSinceEpoch();
	std::cout << "\n\nStart processing...\n";
	std::cout << "Current time point: " << mtimer::getCurrentTimeStr() << std::endl;
	std::cout << "The processed filename is: " << filename << std::endl;
	if(is_write)
		std::cout << "The saved filename: " << write_filename << std::endl;

	std::atomic<bool> flag(true);
	while (capture.read(frame))
	{
		cv::Mat processed_frame;
		bool ret = true; // the return of the process
		// The flag for check whether the GPU process down
		flag.store(false, std::memory_order_relaxed);
		while (!flag) {
			bool ret = gpu::AlgoPipelineManager::getInstance()->process(frame, processed_frame, flag);
			
			if (!ret) break;
		}
		if (!ret) continue;
		// NOTE, the frame processed by the GPU will convert to BGRA
		cv::cvtColor(processed_frame, processed_frame, cv::COLOR_BGRA2BGR);

		if (is_write) {
			static bool st = true;
			if (st) {
				std::cout << "Video is processing and writing, please waiting......\n";
				st = false;
			}

			cv::Mat framef;
			frame.convertTo(framef, CV_32FC3);
			framef = framef / 255.f;
			cv::Mat tmp(frame.rows, frame.cols * 2, framef.type());
			framef.copyTo(tmp.colRange(0, frame.cols));
			processed_frame.copyTo(tmp.colRange(frame.cols, frame.cols * 2));

			writer.write(tmp);
		}

		char key = cv::waitKey(1);
		if (key == 'Q' || key == 'q') {
			std::cout << "Processing was interruptted.\n";
			break;
		}
	}
	std::cout << "Processing down...\n";
	std::cout << mtimer::getCurrentTimeStr() << std::endl;
	auto time = mtimer::getDurationSince(start_point, mtimer::SECOND);
	std::cout << time << " seconds elapsed.\n";

	capture.release();
	if(is_write)
		writer.release();

	return true;
}