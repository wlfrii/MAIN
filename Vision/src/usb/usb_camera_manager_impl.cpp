#include "usb_camera_manager_impl.h"
#include "../def/micro.h"
#if LINUX
#include <assert.h>
/* Dirent is a C/C++ programming interface that allows programmers to retrieve
 * information about files and directories under Linux/Unix. This project provides
 * Linux compatible Dirent interface for Microsoft Windows.
 * Refers: https://github.com/tronkko/dirent */
#include <dirent.h>
#include <sys/types.h>
/* <sys/stat.h> contains constructs that facilitate getting information about files
 * attributes.
 * Refers: https://en.wikibooks.org/wiki/C_Programming/POSIX_Reference/sys/stat.h */
#include <sys/stat.h>
/* <limits.h> Standard header for integer sizes */
#include <limits.h> // INT_MAX
/* <string.h> contains old functions like strcpy, strlen for C style null-terminated
 * string.
 * <string> primarily contains the std::string, std::wstring and other classes.
 * <cstring> will usually import the same things as <string.h> but into the std
 * namespace. */
#include <cstring>
#include <string>
/* Input and Output operations, such as fopen, fclose, can also be performed in C++
 * using the C standard Input and Ouput Library <cstdio>, known as <stdio.h> in the
 * C language. */
#include <cstdio>
/* <algorithm> defines a collectiong of functions especially designed to be used on
 * range of elements, like find, copy, sort, merge, et.al */
#include <algorithm>


namespace
{
    // Get files in directory
    static void getFiles(const std::string &directory, std::vector<std::string> &res)
    {
        dirent *ent;
        struct stat st;

        DIR* dir = opendir(directory.c_str());
        if(!dir) return;

        res.clear();
        while((ent = readdir(dir)) != nullptr)
        {
            const std::string fname = ent->d_name;
            const std::string fname_full = directory + "/" + fname;

            if(fname[0] == '.' || !fname.compare(".") || !fname.compare("..")){
                continue;
            }

            if(stat(fname_full.c_str(), &st) == -1){
                continue;
            }

            res.push_back(fname_full);
        }
    }
}


USBCameraManager* USBCameraManager::getInstance()
{
    return new USBCameraManagerImpl;
}

void USBCameraManager::releaseInstance(USBCameraManager *instance)
{
    delete instance;
}


USBCameraManagerImpl::USBCameraManagerImpl()
    : context(0)
    , devices(0)
{
    assert(uvc_init(&context, 0) >= 0);
    assert(uvc_get_device_list(context, &devices) >= 0);

    // Get all cameras
    int i = 0;
    while(true)
    {
        if(!devices[i]){
            break;
        }
        usb_cameras.push_back(new USBCamera(devices[i]));
        i++;
    }
    printf("USBCameraManagerImpl: There are %d usb cameras\n", i - 1);
}

USBCameraManagerImpl::~USBCameraManagerImpl()
{
    // Delete all cameras
    for(auto &usb_cam : usb_cameras)
    {
        delete usb_cam;
    }
    usb_cameras.clear();

    // Free
    uvc_free_device_list(devices, 1);
    uvc_exit(context);
}

void USBCameraManagerImpl::getCameras(std::vector<USBCamera*> &usb_cam)
{
    usb_cam = this->usb_cameras;
}

bool USBCameraManagerImpl::getIds(USBCamera *usb_cam, int ids[2])
{
    uvc_device_descriptor_t* desc;
    if(uvc_get_device_descriptor(usb_cam->device, &desc) != UVC_SUCCESS){
        return false;
    }

    ids[0] = desc->idVendor;
    ids[1] = desc->idProduct;

    uvc_free_device_descriptor(desc);
    return true;
}

int USBCameraManagerImpl::getIndex(USBCamera *usb_cam)
{
    std::vector<int> indices;
    getIndices(usb_cam, indices);

    int output = INT_MAX;
    if(indices.empty()){
        output = -1;
    }else{
        output = indices[0];
    }
    return output;
}

void USBCameraManagerImpl::getIndices(USBCamera *usb_cam, std::vector<int> &indices)
{
    indices.clear();
    int ids[2] = { 0 };
    if(!getIds(usb_cam, ids)){
        printf("USBCameraManagerImpl::getIndices: cannot get ids.\n");
        return;
    }
    printf("USBCameraManagerImpl::getIndices: ids[0]:%x, ids[1]:%x.\n", ids[0], ids[1]);

    std::string directory[2];
    std::vector<std::string> res[2];
    std::vector<std::string>::iterator i[2];

    directory[0] = "/sys/class/video4linux";
    getFiles(directory[0], res[0]);
    for(auto &res0 : res[0])
    {
        directory[1] = res0 + "/device/input";
        getFiles(directory[1], res[1]);
        for(auto &res1 : res[1])
        {
            const std::string fids[2] = { res1 + "/id/vendor", res1 + "/id/product" };
            bool ok = true;
            for(int x = 0; x < 2; x++)
            {
                int val = -1;
                auto fp = fopen(fids[0].c_str(), "r");
                if(fp)
                {
                    fscanf(fp, "%x", &val);
                    fclose(fp);
                }
                if(val != ids[x])
                {
                    ok = false;
                    break;
                }
            }
            if(ok)
            {
                std::string target = *i[0];
                int res = -1;
                sscanf(target.c_str(), "/sys/class/video4linux/video%d", &res);
                // dbg("id: %x:%x, target: %s, idx: %d\n", ids[0], ids[1], target.c_str(), res);
                indices.push_back(res);
            }
        }
    }
    std::sort(indices.begin(), indices.end());
}

bool USBCameraManagerImpl::read(const int index, uint8_t data[MAX_DATA_SIZE])
{
    for(auto &usb_cam : usb_cameras){
        if(index == getIndex(usb_cam))
            return read(usb_cam, data);
    }
    return false;
}

bool USBCameraManagerImpl::write(const int index, const uint8_t data[MAX_DATA_SIZE])
{
    for(auto &usb_cam : usb_cameras){
        if(index == getIndex(usb_cam))
            return write(usb_cam, data);
    }
    return false;
}

bool USBCameraManagerImpl::read(USBCamera *usb_cam, uint8_t data[MAX_DATA_SIZE])
{
    USBCameraDeviceHandle cam_device_handle(usb_cam);
    if(!cam_device_handle){
        return false;
    }
    const uvc_extension_unit_t *eu = uvc_get_extension_units(cam_device_handle);
    const uint8_t uid = eu->bUnitID;

    int XU_ID = 0;
    constexpr int size = 32;
    uint8_t buffer[size] = { 0 };

    for(int i = 0; i < MAX_DATA_SIZE / size; ++i)
    {
        // Set the address of the data
        buffer[0] = i * size;
        buffer[1] = (i * size) >> 8;
        buffer[2] = 0;
        XU_ID = 2;
        if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)buffer, 3) < 0) {
            return false;
        }

        // Read data
        XU_ID = 3;
        if(uvc_get_ctrl(cam_device_handle, uid, XU_ID, (void*)buffer, size, UVC_GET_CUR) < 0){
            return false;
        }
        std::memcpy(data + i * size, buffer, size);
    }
    return true;
}

bool USBCameraManagerImpl::write(USBCamera *cam, const uint8_t data[MAX_DATA_SIZE])
{
    USBCameraDeviceHandle cam_device_handle(cam);
    if(!cam_device_handle){
        return false;
    }
    // Get custom processing or camera-control functions
    const uvc_extension_unit_t *eu = uvc_get_extension_units(cam_device_handle);
    // Get the index of the extension unit within the device
    const uint8_t uid = eu->bUnitID;

    int XU_ID = 0;
    constexpr int size = 32;
    uint8_t buffer[size] = { 0 };

    // First clear all the data
    XU_ID = 4;
    if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)buffer, size) < 0){
        return false;
    }

    for(int i = 0; i < MAX_DATA_SIZE / size; ++i)
    {
        // Set the address of the data
        buffer[0] = i * size;
        buffer[1] = (i * size) >> 8;
        buffer[2] = 0;
        XU_ID = 2;
        if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)buffer, 3) < 0){
            return false;
        }

        // Write data
        XU_ID = 3;
        const uint8_t* ptr = data + i * size;
        if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)ptr, size) < 0){
            return false;
        }
    }
    return true;
}
#endif
