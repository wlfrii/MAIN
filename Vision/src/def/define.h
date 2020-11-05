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
		NORMAL = 0,		//!< the display view keeps normal
		ZOOM_2X = 1,		//!< the display view will be zoomed 200%
		ZOOM_4X = 2			//!< the display view will be zoomed 400%
	};


    /** @brief The parameters used in image enhancement.
	*/
	struct ImageEnhanceDetails
	{
	public:
		ImageEnhanceDetails() : m_is_enhance_image(false), m_eps(0.1f), m_usm(3.5f), m_filter_radius(16), m_scale(4), m_saturation(0), m_contrast(0), m_brightness(0), m_luma_mode(0) {}
		ImageEnhanceDetails(bool is_enhance_image, float eps, float usm, uchar filter_radius, uchar scale, char saturation, char contrast, char brightness, uchar mode) : m_is_enhance_image(is_enhance_image), m_eps(eps), m_usm(usm), m_filter_radius(filter_radius), m_scale(scale), m_saturation(saturation), m_contrast(contrast), m_brightness(brightness), m_luma_mode(mode) {}
		ImageEnhanceDetails(const ImageEnhanceDetails &) = delete;
		~ImageEnhanceDetails() {}

		void setStatus(bool status) { m_is_enhance_image = status; }
		void setEps(float value) { m_eps = value; }
		void setUsm(float value) { m_usm = value; }
		void setFilterRadius(uchar value) { m_filter_radius = value; }
		void setScale(uchar scale) { m_scale = scale; }
		void setSaturation(char value) { m_saturation = value; }
		void setContrast(char value) { m_contrast = value; }
		void setBrightness(char value) { m_brightness = value; }
		void setLumaMode(uchar mode) { m_luma_mode = mode; }

		bool  getEnhanceStatus() const { return m_is_enhance_image; }
		float getEps() const { return m_eps; }
		float getUsm() const { return m_usm; }
		uchar getFilterRadius() const { return m_filter_radius; }
		uchar getScale() const { return m_scale; }
		char  getSaturation() const { return m_saturation; }
		char  getContrast() const { return m_contrast; }
		char  getBrightness() const { return m_brightness; }
		uchar getLumaMode() const { return m_luma_mode; }

	private:
		bool  m_is_enhance_image;	//!< the flag, open the enhance mode or not

		float m_eps;				//!< the extent for increasing the noise of the image
		float m_usm;				//!< the extent for increasing the characters of the image
		uchar m_filter_radius;		//!< the filter radius for guided filter, which is aimed at reduce the noise
		uchar m_scale;				//!< the ratio for 'pyrdown'(downsample) the image, try scale=box_radius/4 to scale=box_radius

		char  m_saturation;			//!< the degree for increasing/decreasing the saturation of the image
		char  m_contrast;			//!< the degree for increasing/decreasing the contrast of the image
		char  m_brightness;			//!< the degree for increasing/decreasing the brightness of the image	

		uchar m_luma_mode;			//!< the mode for reduce the uneven luminance of the endoscopic image
	};
}
