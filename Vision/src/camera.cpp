#include "camera.h"
#include "frame_displayer.h"
#include "frame_reader.h"
#include "def/micro_define.h"
#include "def/mtimer.h"
#include "camera_parameters.h"
#include "image_processor.h"


Camera::Camera(uchar usb_id, uchar cam_id, uint image_width, uint image_height, const CameraParameters & cam_params)
    : index(cam_id)
    , frame_reader(new FrameReader(usb_id, cam_id, image_width, image_height))
    , processor(new ImageProcessor(cam_id, cam_params, image_width, image_height))
{
    thread = std::thread(&Camera::run, this);
    thread.detach();
}

Camera::~Camera()
{
    DELETE_PIONTER(frame_reader);
    DELETE_PIONTER(processor);
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
            // GPU process
            if(processor->uploadImage(frame))
            {
                if(processor->downloadImage(res_frame))
                {
                    // display
#if DEBUG_TIME
                    start_time_point = mtimer::getDurationSinceEpoch();
#endif
                    FrameDisplayer::getInstance()->updateFrame(res_frame, index);
#if DEBUG_TIME
                    time = mtimer::getDurationSince(start_time_point);
                    printf("Camera id: %d, update frame time: %f ms\n", m_cam_id, time);
#endif
                }
            }
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
