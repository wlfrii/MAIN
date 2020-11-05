#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include "def/define.h"
#include "def/micro_define.h"

#if LINUX
class V4L2Capture;
#endif

/** @brief  the base class of monocular or binocular image reader
*/
class FrameReader
{
public:
    explicit FrameReader(uchar usb_id, uchar cam_id, uint image_width, uint image_height);
    ~FrameReader();

    /**
    * @brief  Get current frame if there is one.
    */
    bool getFrame(cv::Mat & frame);

    /**
    * @brief Set the sharpness.
    */
    void setSharpness(uchar value);

private:
    uchar   usb_id;                         //!< the usb id of camera
    uchar   cam_id;							//!< the cam id of camera
    uint	image_width;							//!< width of the image
    uint	image_height;							//!< height of the image

    uchar	sharpness;						//!< the sharpness of the image

//    cv::Mat         current_frame;          //!< store last frame before new frame receive
//    std::thread                 m_thread;			//!< used for thread capturing image
#if LINUX
    V4L2Capture*    capture;                //!< used for capturing image in LINUX
#else
    cv::VideoCapture    video_capture;    //!< used for capturing image in WINDOWS
#endif

};


#endif // IMAGE_READER_H
