#ifndef USB_DEVICE_H
#define USB_DEVICE_H
#include <vector>


class USBDeviceInfo;

class USBDevice
{
public:
    USBDevice();
    ~USBDevice();

    int getLeftCamUSBIdx();
    int getRightCamUSBIdx();

private:
    void getAndUpdate();
    bool getVideoDevices(std::vector<USBDeviceInfo*>& usb_device_infos);
    void updateIndexes();

private:
    std::vector<USBDeviceInfo*> usb_device_infos;
    int left_cam_usb_idx;
    int right_cam_usb_idx;
    bool valid;
};

#endif // USB_DEVICE_H

