#ifndef USB_DEVICE_INFO_H
#define USB_DEVICE_INFO_H
#include <vector>
#include <string.h>
/**
 * Store the pid and uid of usb device
 */
class USBDeviceID
{
public:
    USBDeviceID() : pid(-1), vid(-1) {}
    USBDeviceID(int pid, int vid)
        : pid(pid)
        , vid(vid)
    {}

    int getPID() const { return pid; }
    int getVID() const { return vid; }

private:
    int pid;
    int vid;
};


/**
 * Store the description(in this case, just friendly-name) of usb device
 */
class USBDeviceDesc
{
public:
    USBDeviceDesc()
    {
        memset(desc, 0, sizeof(desc));
    }

private:
    enum{ MAX_LEN_OF_DESC = 255 };
    char desc[MAX_LEN_OF_DESC];
};


/**
 * Store both pid/vid and description of usb device
 */
class USBDeviceInfo
{
public:
    USBDeviceInfo(USBDeviceDesc usb_device_desc, USBDeviceID usb_device_id, int index, const std::vector<int>& indices)
        : usb_device_desc(usb_device_desc)
        , usb_device_id(usb_device_id)
        , index(index)
        , indices(indices)
    {}

    int getPID() const { return usb_device_id.getPID(); }
    int getIndex() { return index; };
    const std::vector<int>& getIndices() { return indices; };

private:
    USBDeviceDesc usb_device_desc;
    USBDeviceID usb_device_id;
    int index;
    std::vector<int> indices;
};

#endif // USB_DEVICE_INFO_H
