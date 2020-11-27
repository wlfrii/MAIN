#ifndef USB_CAMERA_MANAGER_H
#define USB_CAMERA_MANAGER_H
#include "../def/micro_define.h"
#if LINUX
#include <stdint.h>
#include <vector>

// usb camera abstract
class USBCamera;

// manager USBCamera
class USBCameraManager
{
protected:
    USBCameraManager() {}
    virtual ~USBCameraManager() {}

public:
    // creat/release instance
    static USBCameraManager* getInstance();
    static void releaseInstance(USBCameraManager* instance);

    // get camera list, return value only available between getInstance() and releaseInstance()
    virtual void getCameras(std::vector<USBCamera*>& cam) = 0;

    // get camera 0-vid and 1-pid
    virtual bool getIds(USBCamera* cam, int ids[2]) = 0;

    // get camera index
    // for linux, it's /dev/video*, so non-negative number will be returned if success
    // TODO: for windows, has not been defined
    virtual int getIndex(USBCamera* cam) = 0;

    // get camera indices
    // one camera can have many indices, getIndex() return the min index, this function return all
    virtual void getIndices(USBCamera* cam, std::vector<int>& indices) = 0;

    // read 512 bytes from camera
    virtual bool read(USBCamera* cam, uint8_t data[512]) = 0;

    // write 512 bytes to camera
    virtual bool write(USBCamera* cam, const uint8_t data[512]) = 0;

    virtual bool read(const int index, uint8_t data[512]) = 0;
    virtual bool write(const int index, const uint8_t data[512]) = 0;
};


#endif
#endif // USB_CAMERA_MANAGER_H
