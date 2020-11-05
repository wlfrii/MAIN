#ifndef CAMERA_H
#define CAMERA_H
#include <thread>
#include <opencv2/opencv.hpp>
#include <memory>

class FrameReader;
class CameraParameters;
class ImageProcessor;

class Camera
{
public:
    Camera(uchar usb_id, uchar cam_id, uint image_width, uint image_height, const CameraParameters & cam_params);
    ~Camera();

private:
    void run [[noreturn]] ();

private:
    uchar           index;
    std::thread     thread;

    FrameReader*    frame_reader;
    ImageProcessor* processor;
};

#endif // CAMERA_H
