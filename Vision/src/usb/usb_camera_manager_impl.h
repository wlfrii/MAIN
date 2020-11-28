#ifndef USBCAMERAMANAGERIMPL_H
#define USBCAMERAMANAGERIMPL_H
#include "../def/micro.h"
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
        uvc_error_t ret = uvc_open(cam->device, &device_handle);
        if(ret != UVC_SUCCESS){
            fprintf(stderr, "can't open camera: %p, error msg:%s", (void*) cam, uvc_strerror(ret));
            device_handle = 0;
        }
    }
    ~USBCameraDeviceHandle(){
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

    void getCameras(std::vector<USBCamera*>& usb_cam);
    bool getIds(USBCamera* usb_cam, int ids[2]);
    int  getIndex(USBCamera* usb_cam);
    void getIndices(USBCamera* usb_cam, std::vector<int>& indices);

    bool read(USBCamera* usb_cam, uint8_t data[MAX_DATA_SIZE]);
    bool write(USBCamera* usb_cam, const uint8_t data[MAX_DATA_SIZE]);
    bool read(const int index, uint8_t data[MAX_DATA_SIZE]);
    bool write(const int index, const uint8_t data[MAX_DATA_SIZE]);

private:
    uvc_context_t *context;
    uvc_device_t **devices;

    std::vector<USBCamera*> usb_cameras;
};

#endif
#endif // USBCAMERAMANAGERIMPL_H
