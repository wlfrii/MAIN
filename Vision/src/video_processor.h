#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

/* @brief This class is designated to process a video stream.
 * (Next step. The display depends on @class FrameDisplayer.)
 * @author Wanglf.
 * @date 2020.10.30
 */
class VideoProcessor
{
protected:
	VideoProcessor() {}
public:
	~VideoProcessor() {}

	static std::shared_ptr<VideoProcessor> getInstance();

	/*@brief Process the specified video after initialize the lib_vision_gpu.lib.
	 *@param filename The path of the video.
	 *@param is_write Whether write the processed video.
	 */
	bool processVideo(const std::string &filename, bool is_write = false);

private:
	cv::VideoCapture capture;
	cv::VideoWriter  writer;
};

#endif //VIDEO_PROCESSOR_H