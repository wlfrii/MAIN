#include "usb_device.h"
#include "usb_device_info.h"
#include <strmif.h> // used by Microsoft TV Technologies.

USBDevice::USBDevice()
    : left_cam_usb_idx(0)
    , right_cam_usb_idx(1)
    , valid(false)
{
    getVideoDevices(usb_device_infos);
    updateIndexes();
}

USBDevice::~USBDevice()
{
    for(auto& usb_device_info : usb_device_infos)
        delete usb_device_info;
}

int USBDevice::getLeftCamUSBIdx()
{
    getAndUpdate();
    return left_cam_usb_idx;
}

int USBDevice::getRightCamUSBIdx()
{
    getAndUpdate();
    return right_cam_usb_idx;
}

void USBDevice::getAndUpdate()
{
    if(!valid){
        getVideoDevices(usb_device_infos);
        updateIndexes();
    }
}

bool USBDevice::getVideoDevices(std::vector<USBDeviceInfo *> &usb_device_infos)
{
    usb_device_infos.clear();

    // create device enumerator
    ICreateDevEnum* dev_enum = nullptr;
}

void USBDevice::updateIndexes()
{
    if(usb_device_infos.size() != 2){
        valid = false;
        return;
    }
    if(usb_device_infos[0]->getPID() % 2 == 1)
    {
        left_cam_usb_idx = 0;
        right_cam_usb_idx = 1;
    }
    else {
        left_cam_usb_idx = 1;
        right_cam_usb_idx = 0;
    }
}


