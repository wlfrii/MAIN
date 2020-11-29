#ifndef CAMERA_HANDLE_H_
#define CAMERA_HANDLE_H_
#include "def/define.h"
#include "camera_parameters.h"
#include <thread>

class Camera;

class CameraHandle
{
public:
    CameraHandle();
    ~CameraHandle();

    void initCamera();
    void openCamera();
private:
    void runCamera [[noreturn]] ();

private:
    Camera* cameras[vision::MAX_CAMERA_NUMBER];
    std::thread mthread;
};

#endif //CAMERA_HANDLE_H_
