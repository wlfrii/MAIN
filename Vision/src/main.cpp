#include <string>
#include <memory>
#include "camera_handle.h"
#include "def/micro.h"
#include <libvisiongpu/gpu_algorithm_pipeline_manager.h>
#include "video_processor.h"
#include <QApplication>
#include "ui/control_panel.h"
#include "camera_parameters.h"

namespace
{
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
#if WITH_QT
	QApplication app(argc, argv);
	ControlPanel *panel = new ControlPanel();
	panel->show();
#endif

    //::initGPUProcessor();

#if LINUX
	CameraHandle camera_handle;
	camera_handle.openCamera();
#else
	std::string filename[3];
	filename[0] = "E:/Rii/Videos/20200829/M_08292020153352_00000000U2957851_1_002-1.MP4";
	filename[1] = "E:/Rii/Videos/20200829/M_08292020153352_0000-0230.MP4";
	filename[2] = "E:/Rii/Videos/20200829/M_08292020153352_0210-0230.MP4";

	//VideoProcessor::getInstance()->processVideo(filename[2], false);
#endif

#if WITH_QT
	return app.exec();
#else
    return 0;
#endif
}
