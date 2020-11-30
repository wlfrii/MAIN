#include "camera.h"
#include "frame_displayer.h"
#include "frame_reader.h"
#include "camera_parameters.h"
#include "image_processor.h"
#include "def/ptr_define.h"
#include <libutility/timer/mtimer.h>


Camera::Camera(uchar usb_id, uchar cam_id, uint image_width, uint image_height)
    : index(cam_id)
    , frame_reader(new FrameReader(usb_id, cam_id, image_width, image_height))
{
    thread = std::thread(&Camera::run, this);
    thread.detach();
}

Camera::~Camera()
{
    DELETE_PIONTER(frame_reader);
}

void Camera::run()
{
    cv::Mat frame;
    cv::Mat res_frame;
    while(true)
    {
        auto start_time_point = mtimer::getCurrentTimePoint();

        // get frame
        if(frame_reader->getFrame(frame))
        {
#if DEBUG_TIME
                    start_time_point = mtimer::getDurationSinceEpoch();
#endif
            ImageProcessor::getInstance()->processImage(frame, res_frame, vision::StereoCameraID(index));
#if DEBUG_TIME
                    time = mtimer::getDurationSince(start_time_point);
                    printf("Camera id: %d, update frame time: %f ms\n", m_cam_id, time);
#endif

            FrameDisplayer::getInstance()->updateFrame(res_frame, index);
        }

        // check time
        auto time = mtimer::getDurationSince(start_time_point);
        if(time < vision::FRAME_UPDATE_INTERVAL_MS)
        {
            auto ms = vision::FRAME_UPDATE_INTERVAL_MS - time;
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }
}
