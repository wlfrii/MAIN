#include <string>
#include <memory>
#include "camera_handle.h"
#include "terminal.h"
#include "def/micro_define.h"
#include <gpu_algorithm_pipeline_manager.h>
#include "video_processor.h"
#include <memory>

namespace
{
    const std::string cam_params_path = "../conf/params_NO14.yml";
    // create a params reader to read the camera parameters
    auto params_reader = std::make_unique<CameraParamsReader>(cam_params_path);
}

int main(int argc, char *argv[])
{
	// Initialize the AlgoPipelineManager first
	gpu::AlgoPipelineManager::getInstance()->intialize();
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeUnevenY());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::UnevenYProperty>(2, 0.99));
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeGamma());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(0.005));
	gpu::AlgoPipelineManager::getInstance()->addAlgoNode(new gpu::AlgoNodeImageAdjust());
	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(0, 3));
	

#if LINUX
    CameraHandle camera_handle;
    camera_handle.initCamera(std::move(params_reader));

    Terminal terminal;
    terminal.run();

    camera_handle.openCamera();
#else
	std::string filename[3];
	filename[0] = "E:/Rii/Videos/20200829/M_08292020153352_00000000U2957851_1_002-1.MP4";
	filename[1] = "E:/Rii/Videos/20200829/M_08292020153352_0000-0230.MP4";

	VideoProcessor::getInstance()->processVideo(filename[1], true);

#endif
    return 0;
}
