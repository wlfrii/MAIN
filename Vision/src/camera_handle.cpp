#include "camera_handle.h"
#include "camera.h"
#include "def/micro_define.h"
#include "frame_displayer.h"

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


void CameraHandle::initCamera(std::unique_ptr<CameraParamsReader> cam_params_reader)
{
    // update the image size in vision_define.
    int image_width = cam_params_reader->getImageWidth();
    int image_height = cam_params_reader->getImageHeight();
    if(image_width <= 0 || image_height <= 0) {
        printf("CameraHandle: Read image size failed!\n\nProgram exit!\n");
        exit(0);
    }
    else {
        vision::ImageSize::width = cam_params_reader->getImageWidth();
        vision::ImageSize::height = cam_params_reader->getImageHeight();
    }

    // Initialize left camera
    uchar usb_id = 1;
    uchar cam_id = uchar(vision::LEFT_CAMERA);
    const CameraParameters left_params = cam_params_reader->getCameraParameters(vision::LEFT_CAMERA);
    cameras[cam_id] = new Camera(usb_id, cam_id, image_width, image_height, left_params);
	
    // Initialize right camera
    usb_id = 2;
    cam_id = uchar(vision::RIGHT_CAMERA);
    const CameraParameters right_params = cam_params_reader->getCameraParameters(vision::RIGHT_CAMERA);
    cameras[cam_id] = new Camera(usb_id, cam_id, image_width, image_height, right_params);

    printf("CameraHandle::initCamera: Two camera initialized done.\n");
}


void CameraHandle::openCamera()
{
    while(true)
    {
        FrameDisplayer::getInstance()->showFrame();
    }
}
