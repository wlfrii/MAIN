#ifndef USB_DEVICE_H
#define USB_DEVICE_H
#include "../def/micro.h"
#if LINUX
#include <vector>
#include <stdint.h>
#include "usb_camera_manager.h"

class USBDeviceInfo;
/**
 * @brief The USBDevice class
 */
class USBDevice
{
public:
    enum ErrorState{
        DEVICE_STATE_DUMMY,
        DEVICE_DETECT_FAILED,   // cannot detect any usb camera
        DEVICE_PID_NOT_UNIQ     // cannot distinguish the left and right camera
    };
public:
    USBDevice();
    ~USBDevice();

    int getLeftCamUSBIdx();
    int getRightCamUSBIdx();

    ErrorState getErrorState() { return error_state; }

    bool writeLeft(const uint8_t data[MAX_DATA_SIZE]);
    bool readLeft(uint8_t data[MAX_DATA_SIZE]);
    bool writeRight(const uint8_t data[MAX_DATA_SIZE]);
    bool readRight(uint8_t data[MAX_DATA_SIZE]);
    bool read(const int& index, uint8_t data[MAX_DATA_SIZE]);
    bool write(const int& index, const uint8_t data[MAX_DATA_SIZE]);

private:
    void getAndUpdate();
    bool getVideoDevices();
    void updateIndexes();

public:
    std::vector<USBDeviceInfo> usb_device_infos;
private:
    int left_usb_cam_idx;
    int right_usb_cam_idx;
    bool valid;

    USBCameraManager*   usb_cam_manager;

    ErrorState  error_state;
};

#endif
#endif // USB_DEVICE_H

