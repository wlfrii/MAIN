#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H
#include <opencv2/opencv.hpp>
#include <atomic>
#include <memory>
#include "def/ring_buffer.h"

class CameraParameters;
class MapCalculator;

class ImageProcessor
{
public:
    ImageProcessor(uchar cam_id, const CameraParameters & cam_params, uint image_width, uint image_height);
    ~ImageProcessor();

    bool uploadImage(const cv::Mat & image);
    bool downloadImage(cv::Mat & image);

private:
    void updateRectifyProps();

private:
    uchar               cam_id;
    MapCalculator*      map_calculator;
    cv::Mat             processed_image;

    /* Atomic   ---   template<typename T> struct atomic
     * Objects of atomic types contain a value of a particular type(T)
     * The main characteristic of atomic objects is that access to this contained value from
     * different threads cannot cause data races. */
    std::atomic<bool>   read_flag;

    Ringbuffer<cv::Mat, 2> image_buffer;

    int                 disparity;
};

#endif // IMAGE_PROCESSOR_H
