#include "camera_handle.h"
#include "camera.h"
#include "def/ptr_define.h"
#include "def/micro.h"
#include "frame_displayer.h"
#include "camera_parameters.h"
#include "image_processor.h"
#include "ui/cmd.h"
#include "usb/usb_device.h"
#include "usb/usb_camera_parameters.h"

namespace vision
{
    uint16_t ImageSize::width = 1920;
    uint16_t ImageSize::height = 1080;
}

CameraHandle::CameraHandle()
{
    for(int i = 0; i < vision::MAX_CAMERA_NUMBER; i++) {
        cameras[i] = nullptr;
    }
}

CameraHandle::~CameraHandle()
{
    for(int i = 0; i < vision::MAX_CAMERA_NUMBER; i++) {
        DELETE_PIONTER(cameras[i]);
    }
}

void CameraHandle::openCamera()
{
    while(true)
    {
        /*if(readCamParams()){
            break;
        }
        else */if(loadCamParams()){
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
#if !WITH_QT
    runCamera();
#endif
}


#if LINUX
bool CameraHandle::readCamParams()
{
    USBDevice usb_device;
    uint8_t left_data[MAX_DATA_SIZE];
    uint8_t right_data[MAX_DATA_SIZE];
    bool is_read_left = usb_device.readLeft(left_data);
    bool is_read_right = usb_device.readLeft(right_data);
    if(!is_read_left || !is_read_right)
    {
        printf("CameraHandle: Cannot read camera paramerters!\n");
        return false;
    }

    USBCameraParammeters usb_left_cam_params, usb_right_cam_params;
    memcpy(&usb_left_cam_params, left_data, sizeof(usb_left_cam_params));
    memcpy(&usb_right_cam_params, left_data, sizeof(usb_right_cam_params));
    if(!usb_left_cam_params.crc8Check() || !usb_right_cam_params.crc8Check())
    {
        printf("CameraHandle: Read failed camera paramerters!\n");
        return false;
    }
    uchar usb_id = usb_device.getLeftCamUSBIdx();
    uchar cam_id = uchar(vision::LEFT_CAMERA);
    //cameras[cam_id] = new Camera(usb_id, cam_id, usb_left_cam_params.width, usb_left_cam_params.width, CameraParameters(usb_left_cam_params));

    usb_id = usb_device.getRightCamUSBIdx();
    cam_id = uchar(vision::RIGHT_CAMERA);
    //cameras[cam_id] = new Camera(usb_id, cam_id, usb_right_cam_params.width, usb_right_cam_params.width, CameraParameters(usb_right_cam_params));

    printf("CameraHandle: Two camera initialized done.\n");

    return true;
}
#endif

bool CameraHandle::loadCamParams()
{
    auto cam_params_reader = std::make_unique<CameraParamsReader>(CMD::cam_params_path);

    // update the image size in vision_define.
    int image_width(0), image_height(0);
    cam_params_reader->getImageSize(image_width, image_height);
    if(image_width <= 0 || image_height <= 0) {
        printf("CameraHandle: Read image size failed!\n");
        return false;
    }else{
        vision::ImageSize::width = image_width;
        vision::ImageSize::height = image_height;
    }

    uchar usb_id = 4;
    uchar cam_id = uchar(vision::LEFT_CAMERA);
    cameras[cam_id] = new Camera(usb_id, cam_id, image_width, image_height);

    usb_id = 2;
    cam_id = uchar(vision::RIGHT_CAMERA);
    cameras[cam_id] = new Camera(usb_id, cam_id, image_width, image_height);
    printf("CameraHandle: Two camera initialized done.\n");

    // initialize camera parameters
    auto cam_params = cam_params_reader->getStereoCameraParameters();
    if (!cam_params) {
        printf("CameraHandle: Read camera parametes failed.\n");
    }
    //ImageProcessor::getInstance()->setMapCalculator(std::make_shared<MapCalculator>((cam_params->left.get())),vision::LEFT_CAMERA);
    //ImageProcessor::getInstance()->setMapCalculator(std::make_shared<MapCalculator>((cam_params->right.get())),vision::RIGHT_CAMERA);

    return true;
}

void CameraHandle::runCamera()
{
    //cv::startWindowThread();
    while(true)
    {
        FrameDisplayer::getInstance()->showFrame();
    }
}
