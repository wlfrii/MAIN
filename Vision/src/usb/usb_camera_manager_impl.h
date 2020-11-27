#ifndef USBCAMERAMANAGERIMPL_H
#define USBCAMERAMANAGERIMPL_H
#include "../def/micro_define.h"
#if LINUX
#include "usb_camera_manager.h"
#include <libuvc/libuvc.h>

/**
 * @brief The USBCamera class
 */
class USBCamera
{
public:
    USBCamera(uvc_device_t* device) : device(device) {}
    ~USBCamera() {}

    uvc_device_t *device;
};


/**
 * @brief For open/close USBCamera
 */
class USBCameraDeviceHandle
{
public:
    USBCameraDeviceHandle(USBCamera* cam)
    {
        uvc_error_t t = uvc_open(cam->device, &device_handle);
        if(t != UVC_SUCCESS){
            fprintf(stderr, "can't open camera: %p, error msg:%s", (void*) cam, uvc_strerror(t));
            device_handle = 0;
        }
    }
    USBCameraDeviceHandle(){
        if(device_handle) uvc_close(device_handle);
    }

    operator uvc_device_handle_t*() { return device_handle; }
    bool operator()() { return device_handle != nullptr; }

private:
    uvc_device_handle_t* device_handle;
};


/**
 * @brief The USBCameraManagerImpl class
 */
class USBCameraManagerImpl : public USBCameraManager
{
public:
    USBCameraManagerImpl();
    ~USBCameraManagerImpl();

    void getCameras(std::vector<USBCamera*>& cam);
    bool getIds(USBCamera* cam, int ids[2]);
    int  getIndex(USBCamera* cam);
    void getIndices(USBCamera* cam, std::vector<int>& indices);

    bool read(USBCamera* cam, uint8_t data[512]);
    bool write(USBCamera* cam, const uint8_t data[512]);
    bool read(const int index, uint8_t data[512]);
    bool write(const int index, const uint8_t data[512]);

private:
    uvc_context_t *ctx;
    uvc_device_t **devices;

    std::vector<USBCamera*> cameras;
};

#endif
#endif // USBCAMERAMANAGERIMPL_H
