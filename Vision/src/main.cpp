#include <string>
#include <memory>
#include "camera_handle.h"
#include "def/micro_define.h"
#include <gpu_algorithm_pipeline_manager.h>
#include "video_processor.h"
#include <memory>
#include <QApplication>
#include "ui/control_panel.h"


namespace
{
    const std::string cam_params_path = "../conf/params_NO14.yml";
    // create a params reader to read the camera parameters
    auto params_reader = std::make_unique<CameraParamsReader>(cam_params_path);

    void initGPUProcessor()
    {
        // Initialize the AlgoPipelineManager first
        gpu::AlgoPipelineManager::getInstance()->intialize();
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeNonuniform());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::NonuniformProperty>(2, 0.99));
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(50, 60));
    }
}

int main(int argc, char *argv[])
{
    ::initGPUProcessor();

#if LINUX
    CameraHandle camera_handle;
    camera_handle.initCamera(std::move(params_reader));

    camera_handle.openCamera();
#else
	std::string filename[3];
	filename[0] = "E:/Rii/Videos/20200829/M_08292020153352_00000000U2957851_1_002-1.MP4";
	filename[1] = "E:/Rii/Videos/20200829/M_08292020153352_0000-0230.MP4";
	filename[2] = "E:/Rii/Videos/20200829/M_08292020153352_0210-0230.MP4";

	//VideoProcessor::getInstance()->processVideo(filename[2], false);
#endif
	QApplication app(argc, argv);
	ControlPanel *panel = new ControlPanel();
	panel->show();
    return app.exec();
}
