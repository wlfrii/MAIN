#include "frame_reader.h"
#include "def/mtimer.h"
#if LINUX
#include "unix/v4l2_capture.h"
#endif

namespace
{
#if !LINUX
    /*
        CAP_PROP_FOURCC		4-character code of codec. see VideoWriter::fourcc .

        格式作为第二个参数，OpenCV提供的格式是未经过压缩的，目前支持的格式如下：
        CV_FOURCC('P', 'I', 'M', '1') = MPEG-1 codec
        CV_FOURCC('M', 'J', 'P', 'G') = motion-jpeg codec
        CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
        CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
        CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
        CV_FOURCC('U', '2', '6', '3') = H263 codec
        CV_FOURCC('I', '2', '6', '3') = H263I codec
        CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec

        opencv 4.0.0里面 CV_FOURCC 已经被修改了
    */

    /**@brief  Open the video frame with index, and the the size of the frame

      @param capture	the object of the VideoCapture class
      @param index		the index of camera
      @param width		the width of the video frame
      @param height		the height of the video frame
    */
    void init_video_capture(cv::VideoCapture &capture, int index, int width, int height)
    {
        capture.open(index);
        capture.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 3);
        capture.set(cv::CAP_PROP_SHARPNESS, 3);
    }
#endif

    long long img_tag = 0;

    long long getImageTag() { img_tag++; return img_tag; }
}


FrameReader::FrameReader(uchar usb_id, uchar cam_id, uint image_width, uint image_height)
    : usb_id(usb_id)
    , cam_id(cam_id)
    , image_width(image_width)
    , image_height(image_height)
    , sharpness(3)
{
#if LINUX
    capture = new V4L2Capture(image_width, image_height, 3);
    capture->openDevice(usb_id);
#else
    init_video_capture(video_capture, cam_id, image_width, image_height);
#endif

//    m_thread = std::thread(&FrameReader::runFrameStreaming, this);
//    m_thread.detach();
}


FrameReader::~FrameReader()
{
#if LINUX
    DELETE_PIONTER(capture);
#else
    video_capture.release();
#endif
}


void FrameReader::setSharpness(uchar value)
{
    value = MAX(MIN(value, 100), 0);        // the limit for sharpness is [0,100]
    if(value != sharpness)
    {
#if LINUX

#else
        if (video_capture.isOpened())
            video_capture.set(cv::CAP_PROP_SHARPNESS, value);
#endif
    }
}


bool FrameReader::getFrame(cv::Mat & frame)
{
    auto start_time_point = mtimer::getDurationSinceEpoch();
#if LINUX
    bool flag = capture->ioctlDequeueBuffers(frame);

#else
    bool flag = video_capture.read(frame);
#endif
    float time = mtimer::getDurationSince(start_time_point);
#if DEBUG_TIME
    printf("Camera id: %d, read image: %f ms.\n", m_cam_id, time);
#endif

    flag = flag && (time <= vision::MAX_CAPTURE_TIME_MS) && (!frame.empty());
    if (flag == false)
    {
        printf("ImageReader::getFrame: camera ID: %d, USB ID: %d, time: %f ms, empty: %d.\n",
               cam_id, usb_id, time, frame.empty());
#if LINUX
        while(!capture->openDevice(usb_id))
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            printf("Camera %d is retrying to connection!!!\n", cam_id);
        }
#else
        video_capture.release();
        init_video_capture(video_capture, cam_id, image_width, image_height);
        std::this_thread::sleep_for(std::chrono::seconds(1));
#endif
    }
    else
        return true;
    return false;
}
