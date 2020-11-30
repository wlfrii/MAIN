#ifndef CAMERA_H
#define CAMERA_H
#include <thread>
#include <opencv2/opencv.hpp>

class FrameReader;

class Camera
{
public:
    Camera(uchar usb_id, uchar cam_id, uint image_width, uint image_height);
    ~Camera();

private:
    void run [[noreturn]] ();

private:
    uchar           index;
    std::thread     thread;

    FrameReader*    frame_reader;
};

#endif // CAMERA_H
