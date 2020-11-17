#ifndef USB_DEVICE_INFO_H
#define USB_DEVICE_INFO_H
#include "usb_device_desc.h"


// store both pid/vid and description of usb device
class USBDeviceInfo
{
public:
    USBDeviceInfo(USBDeviceDesc usb_device_desc, USBDeviceID usb_device_id);

    int getPID() const;

private:
    USBDeviceDesc usb_device_desc;
    USBDeviceID usb_device_id;
};

#endif // USB_DEVICE_INFO_H
