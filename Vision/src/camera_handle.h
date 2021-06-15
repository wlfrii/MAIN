#ifndef CAMERA_HANDLE_H_
#define CAMERA_HANDLE_H_
#include "def/define.h"
#include <thread>
#include "./def/micro.h"

class Camera;
class CameraParameters;

class CameraHandle
{
public:
    CameraHandle();
    ~CameraHandle();

    void openCamera();
private:
    void initCamera();
#if LINUX
    bool readCamParams();
#endif
    bool loadCamParams();

    void runCamera [[noreturn]] ();

private:
    Camera* cameras[vision::MAX_CAMERA_NUMBER];
    std::thread mthread;
};

#endif //CAMERA_HANDLE_H_
