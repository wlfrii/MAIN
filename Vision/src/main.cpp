#include <string>
#include <memory>
#include "camera_handle.h"
#include "def/micro_define.h"
#include <gpu_algorithm_pipeline_manager.h>
#include "video_processor.h"
#include <memory>
#if LINUX && WITH_QT
#include <QApplication>
#include "ui/control_panel.h"
#endif

namespace
{
    const std::string cam_params_path = "../conf/params_NO14.yml";
    // create a params reader to read the camera parameters
    auto params_reader = std::make_unique<CameraParamsReader>(cam_params_path);

    void initGPUProcessor()
    {
        // Initialize the AlgoPipelineManager first
        gpu::AlgoPipelineManager::getInstance()->intialize();
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::UnevenYProperty>(2, 0.99));
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));
        gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
        gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(0, 3));
    }
}

int main(int argc, char *argv[])
{
    //::initGPUProcessor();

#if LINUX
//    CameraHandle camera_handle;
//    camera_handle.initCamera(std::move(params_reader));

//    Terminal terminal;
//    terminal.run();

//    camera_handle.openCamera();
#if WITH_QT
    QApplication app(argc, argv);
    ControlPanel *panel = new ControlPanel();
    panel->show();
    app.exec();
#endif
#else
	std::string filename[3];
	filename[0] = "E:/Rii/Videos/20200829/M_08292020153352_00000000U2957851_1_002-1.MP4";
	filename[1] = "E:/Rii/Videos/20200829/M_08292020153352_0000-0230.MP4";
	filename[2] = "E:/Rii/Videos/20200829/M_08292020153352_0005-0020.MP4";

	VideoProcessor::getInstance()->processVideo(filename[1], true);

	system("pause");
#endif
    return 0;
}
