#ifndef USB_DEVICE_H
#define USB_DEVICE_H
#include <vector>
#include <stdint.h>


class USBCameraManager;
class USBDeviceInfo;

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

    ErrorState getErrorState(){
        return error_state;
    }

    std::vector<USBDeviceInfo> device_infos;
    bool writeLeft(const uint8_t data[512]);
    bool readLeft(uint8_t data[512]);
    bool writeRight(const uint8_t data[512]);
    bool readRight(uint8_t data[512]);
    bool read(const int& index,uint8_t data[512]);
    bool write(const int& index, const uint8_t data[512]);

private:
    void getAndUpdate();
    bool getVideoDevices(std::vector<USBDeviceInfo*>& usb_device_infos);
    void updateIndexes();

private:
    std::vector<USBDeviceInfo*> usb_device_infos;
    int left_usb_cam_idx;
    int right_usb_cam_idx;
    bool valid;

    USBCameraManager*   usb_cam_manager;

    ErrorState  error_state;
};

#endif // USB_DEVICE_H

