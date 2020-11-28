#include "usb_device.h"
#include "usb_device_info.h"
#include "../def/micro_define.h"
#if LINUX
#include "usb_camera_manager.h"
#include "usb_device_info.h"


USBDevice::USBDevice()
    : left_usb_cam_idx(0)
    , right_usb_cam_idx(1)
    , valid(false)
{
    usb_cam_manager = USBCameraManager::getInstance();
    getVideoDevices();
    updateIndexes();
}

USBDevice::~USBDevice()
{
    USBCameraManager::releaseInstance(usb_cam_manager);
}

int USBDevice::getLeftCamUSBIdx()
{
    getAndUpdate();
    return left_usb_cam_idx;
}

int USBDevice::getRightCamUSBIdx()
{
    getAndUpdate();
    return right_usb_cam_idx;
}

void USBDevice::getAndUpdate()
{
    if(!valid){
        getVideoDevices();
        updateIndexes();
    }
}

bool USBDevice::getVideoDevices()
{
    usb_device_infos.clear();

    std::vector<USBCamera*> usb_cams;

    // List devices
    usb_cam_manager->getCameras(usb_cams);

    for(auto &usb_cam : usb_cams)
    {
        int ids[2];
        if(!usb_cam_manager->getIds(usb_cam, ids)){
            return false;
        }
        int index = usb_cam_manager->getIndex(usb_cam);
        std::vector<int> indices;
        usb_cam_manager->getIndices(usb_cam, indices);
        usb_device_infos.emplace_back(USBDeviceDesc(), USBDeviceID(ids[1], ids[0]), index, indices);
    }
    return true;
}

void USBDevice::updateIndexes()
{
    if(usb_device_infos.size() != 2){
        valid = false;
        error_state = DEVICE_DETECT_FAILED;
        return;
    }

    bool diff = (usb_device_infos[0].getPID() % 2) ^ (usb_device_infos[1].getPID() % 2);
    if(!diff)
    {
        left_usb_cam_idx = right_usb_cam_idx = -1;
        valid = false;
        error_state = DEVICE_PID_NOT_UNIQ;

        for(int i = 0; i < 2; i++)
        {
            auto idx = usb_device_infos[i].getIndices();
            if(idx.size() == 0)
                continue;
            if(0x2ca3 == usb_device_infos[i].getPID()){
                left_usb_cam_idx = idx[0];
            }else if(0x2cb3 == usb_device_infos[i].getPID()){
                right_usb_cam_idx = idx[0];
            }
        }
        return;
    }
    bool ok[2] = { false, false };
    for(int i = 0; i < 2; i++)
    {
        if(usb_device_infos[i].getPID() % 2 == 1){
            right_usb_cam_idx = usb_device_infos[i].getIndex();
            ok[1] = true;
        } else {
            left_usb_cam_idx = usb_device_infos[i].getIndex();
            ok[0] = true;
        }
    }
    valid = ok[0] && ok[1];
}

bool USBDevice::writeLeft(const uint8_t data[MAX_DATA_SIZE])
{
    return write(left_usb_cam_idx, data);
}

bool USBDevice::readLeft(uint8_t data[MAX_DATA_SIZE])
{
    // printf(" USBDevice readleft left index:%d\n",left_camera_index);
    return read(left_usb_cam_idx, data);
}

bool USBDevice::writeRight(const uint8_t data[MAX_DATA_SIZE])
{
    return write(right_usb_cam_idx, data);
}

bool USBDevice::readRight(uint8_t data[MAX_DATA_SIZE])
{
    // printf("USBDevice readRight right index:%d\n",right_camera_index);
    return read(right_usb_cam_idx, data);
}

bool USBDevice::read(const int& index,uint8_t data[MAX_DATA_SIZE])
{
    return usb_cam_manager->read(index, data);
}

bool USBDevice::write(const int& index, const uint8_t data[MAX_DATA_SIZE])
{
    return usb_cam_manager->write(index, data);
}

#endif
