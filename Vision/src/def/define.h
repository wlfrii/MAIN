#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>

class CameraHandle;
class CameraParamsReader;

namespace vision
{
    /** @brief This struct is use to specify the size of image displaying.
      The default size is set to 1080x1920.
    */
    struct ImageSize
    {
        static uint16_t width;
        static uint16_t height;
    };


    /** @brief The interval for updating each of images from camera
	  current fps: 1000/30=33.3
	*/
	enum {
        FRAME_UPDATE_INTERVAL_MS = 16
	};

    /** @brief The max time for read an image from camera
	*/
	enum {
        MAX_CAPTURE_TIME_MS = 500
	};

    /** @brief The max number of camera in current version
	*/
	enum {
        MAX_CAMERA_NUMBER = 2
	};

    /** @brief The general size of screen.
     */
    enum {
        SCREEN_HEIGHT = 1080,
        SCREEN_WIDTH = 1920
    };

    /** @brief Specify the camera type
	*/
	enum class CameraType
	{
		NONE = 0,
		MONOCULAR = 1,
		BINOCULAR = 2
	};


    /** @brief Specify the number of images to read.
	  Since there could be just one USB or any interface used to transmitt monocular or binocular image streaming.
	  So when the one USB and two cameras were used, it should be BINOCULAR with SINGLE image.
	  Other cases is easily considered.
	*/
	enum class ReadImageNumber
	{
		SINGLE = 0,
		DOUBLE = 1
		// TRIPLE
	};


    /** @brief Used to store the image in different thread.
	*/
	struct ImageCombo
	{
		cv::Mat image;
        std::chrono::milliseconds cap_time;
		long long tag;
	};


    /** @brief Specify the zoom value when display the image streaming frame.
	*/
	enum class ZoomMode
	{
        NORMAL = 0,         //!< the display view keeps normal
		ZOOM_2X = 1,		//!< the display view will be zoomed 200%
		ZOOM_4X = 2			//!< the display view will be zoomed 400%
	};
}
