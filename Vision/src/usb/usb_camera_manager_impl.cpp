#include "usb_camera_manager_impl.h"
#include "../def/micro_define.h"
#if LINUX
#include <assert.h>
#include <dirent.h>
//#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h> // INT_MAT
#include <string.h>
#include <string>
#include <stdio.h>
//#include <algorithm>


namespace
{
    /**
     * Get files in directory
     */
    static void getFiles(std::vector<std::string> &res, const std::string &directory)
    {
        dirent *ent;
        struct stat st;

        DIR* dir = opendir(directory.c_str());
        res.clear();
        if(!dir) return;
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
    : ctx(0)
    , devices(0)
{
    assert(uvc_init(&ctx, 0) >= 0);
    assert(uvc_get_device_list(ctx, &devices) >= 0);

    // Get all cameras
    int i = 0;
    while(true)
    {
        if(!devices[i]){
            break;
        }
        cameras.push_back(new USBCamera(devices[i]));
        i++;
    }
    printf("USBCameraManagerImpl: There are %d usb cameras\n", i - 1);
}

USBCameraManagerImpl::~USBCameraManagerImpl()
{
    // Delete all cameras
    for(auto &camera : cameras)
    {
        delete camera;
    }
    cameras.clear();

    // Free
    uvc_free_device_list(devices, 1);
    uvc_exit(ctx);
}

void USBCameraManagerImpl::getCameras(std::vector<USBCamera *> &cam)
{
    cam = this->cameras;
}

bool USBCameraManagerImpl::getIds(USBCamera *cam, int ids[2])
{
    uvc_device_descriptor_t* desc;
    if(uvc_get_device_descriptor(cam->device, &desc) != UVC_SUCCESS){
        return false;
    }

    ids[0] = desc->idVendor;
    ids[1] = desc->idProduct;

    uvc_free_device_descriptor(desc);
    return true;
}

int USBCameraManagerImpl::getIndex(USBCamera *cam)
{
    std::vector<int> idx;
    getIndices(cam, idx);

    int output = INT_MAX;
    if(idx.empty()){
        output = -1;
    }else{
        output = idx[0];
    }
    return output;
}

void USBCameraManagerImpl::getIndices(USBCamera *cam, std::vector<int> &indices)
{
    indices.clear();
    int ids[2] = { 0 };
    if(!getIds(cam, ids)){
        printf("USBCameraManagerImpl::getIndices: cannot get ids.\n");
        return;
    }
    printf("USBCameraManagerImpl::getIndices: ids[0]:%x, ids[1]:%x.\n", ids[0], ids[1]);

    std::string d[2];
    std::vector<std::string> r[2];
    std::vector<std::string>::iterator i[2];

    d[0] = "/sys/class/video4linux";
    getFiles(r[0], d[0]);
    while(i[0] != r[0].end())
    {
        d[1] = *i[0] + "/device/input";
        getFiles(r[1], d[1]);
        i[1] = r[1].begin();
        while(i[1] != r[1].end())
        {
            const std::string fids[2] = { *i[1] + "/id/vendor", *i[1] + "/id.product" };
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
                if(val != indices[x])
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
            ++i[1];
        }
        ++i[0];
    }
    std::sort(indices.begin(), indices.end());
}

bool USBCameraManagerImpl::read(const int index, uint8_t data[512])
{
    for(auto &camera : cameras){
        if(index == getIndex(camera))
            return read(camera, data);
    }
    return false;
}

bool USBCameraManagerImpl::write(const int index, const uint8_t data[512])
{
    for(auto &camera : cameras){
        if(index == getIndex(camera))
            return write(camera, data);
    }
    return false;
}

bool USBCameraManagerImpl::read(USBCamera *cam, uint8_t data[512])
{
    int XU_ID = 0;
    constexpr int size = 32;
    uint8_t buffer[size] = { 0 };
    USBCameraDeviceHandle cam_device_handle(cam);
    if(!cam_device_handle) return false;

    const uvc_extension_unit_t *eu = uvc_get_extension_units(cam_device_handle);
    const uint8_t uid = eu->bUnitID;
    for(int i = 0; i < 512 / size; ++i)
    {
        // Set the address of the data
        buffer[0] = i*size;
        buffer[1] = (i*size) >> 8;
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
        memcpy(data + i*size, buffer, size);
    }
    return true;
}

bool USBCameraManagerImpl::write(USBCamera *cam, const uint8_t data[512])
{
    USBCameraDeviceHandle cam_device_handle(cam);
    if(!cam_device_handle){
        return false;
    }

    int XU_ID = 0;
    const uvc_extension_unit_t *eu = uvc_get_extension_units(cam_device_handle);
    const uint8_t uid = eu->bUnitID;
    constexpr int size = 32;
    uint8_t buffer[size] = { 0 };

    // First clear all the data
    XU_ID = 4;
    if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)buffer, size) < 0){
        return false;
    }

    for(int i = 0; i < 512 / size; ++i)
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
        const uint8_t* ptr = data + i * 32;
        if(uvc_set_ctrl(cam_device_handle, uid, XU_ID, (void*)ptr, size) < 0){
            return false;
        }
    }
    return true;
}
#endif
