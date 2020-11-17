#include "usb_device_info.h"



USBDeviceInfo::USBDeviceInfo(USBDeviceDesc usb_device_desc, USBDeviceID usb_device_id)
    : usb_device_desc(usb_device_desc)
    , usb_device_id(usb_device_id)
{
}

int USBDeviceInfo::getPID() const
{
    return usb_device_id.getPID();
}
