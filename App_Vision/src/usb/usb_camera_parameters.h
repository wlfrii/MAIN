#ifndef USB_CAMERA_PARAMETERS_H
#define USB_CAMERA_PARAMETERS_H
#include <cstdint>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <libutility/crc/crc.h>

namespace usb
{
    struct Mat3
    {
        uint16_t rows;
        uint16_t cols;
        float data[9];

        cv::Mat to_cvMat(){
            return cv::Mat(rows, cols, CV_32F, data);
        }
    };

    struct Vec5
    {
        uint16_t rows = 5;
        uint16_t cols = 1;
        float data[5];

        cv::Mat to_cvMat(){
            return cv::Mat(rows, cols, CV_32F, data);
        }
    };

    struct Rect
    {
        uint16_t x;
        uint16_t y;
        uint16_t width;
        uint16_t height;

        cv::Rect to_cvRect(){
            return cv::Rect(x, y, width, height);
        }
    };
}

struct USBCameraParammeters
{
    uint16_t width;
    uint16_t height;
    usb::Mat3 A;
    usb::Mat3 R;
    usb::Mat3 Anew;
    usb::Vec5 D;
    usb::Rect ROI;

    float fov;
    float t;
    uint32_t ext1;
    uint32_t ext2;
    uint32_t ext3;
    uint32_t ext4;
    uint32_t ext5;
    uint32_t ext6;
    uint32_t ext7;
    uint32_t ext8;
    uint32_t ext9;
    uint32_t ext10;

    uint8_t  crc_value;

    bool crc8Check()
    {
        constexpr int size = sizeof(USBCameraParammeters) - sizeof(uint8_t);
        uint8_t data[size] = { 0 };
        std::memcpy(data, this, size);
        return crc_value == crc::crc8Check(data, sizeof(data));
    }
};

#endif // USB_CAMERA_PARAMETERS_H
