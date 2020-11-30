#include "image_processor.h"
#include "camera_parameters.h"
#include "map_calculator.h"
#include "def/ptr_define.h"
#include <memory>
#include <libvisiongpu/gpu_algorithm_pipeline_manager.h>
#include "ui/cmd.h"

ImageProcessor::ImageProcessor()
{
    for(auto i = 0; i < vision::MAX_CAMERA_NUMBER; i++){
        props[i] = std::make_shared<Prop>(nullptr);
    }
}

ImageProcessor::~ImageProcessor()
{
}

ImageProcessor* ImageProcessor::getInstance()
{
    static ImageProcessor processor;
    return &processor;
}

void ImageProcessor::setMapCalculator(std::shared_ptr<MapCalculator> map_calculator, vision::StereoCameraID cam_id)
{
    if(int(cam_id) < vision::MAX_CAMERA_NUMBER){
        props[cam_id].reset();
        props[cam_id] = std::make_shared<Prop>(map_calculator);
    }
}

bool ImageProcessor::processImage(const cv::Mat &input, cv::Mat &output, vision::StereoCameraID cam_id/* = vision::LEFT_CAMERA*/)
{
    if (props[cam_id]->processed_image.empty())
        props[cam_id]->processed_image.create(input.size(), CV_8UC4);

    if(CMD::is_enhance == false)
    {
        if(props[cam_id]->map_calculator == nullptr) {
            cv::cvtColor(input, props[cam_id]->processed_image, cv::COLOR_BGR2BGRA);
        }else{
            cv::Mat temp;
            cv::remap(input, temp, props[cam_id]->map_calculator->getCPUMapx(), props[cam_id]->map_calculator->getCPUMapy(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            cv::cvtColor(temp, props[cam_id]->processed_image, cv::COLOR_BGR2BGRA);
        }
        // Put image in buffer
        if (!props[cam_id]->processed_image.empty())
            props[cam_id]->image_buffer.insert(props[cam_id]->processed_image);
    }
    else {
        if(!uploadImage(input, cam_id))
            return false;
    }

    if(!downloadImage(output, cam_id))
        return false;

    return true;
}

bool ImageProcessor::uploadImage(const cv::Mat &image, vision::StereoCameraID cam_id)
{
    /** atomic::load
     * Returns the contained value.
     * typedef enum memory_order {
     *     memory_order_relaxed,   //
     *     memory_order_consume,   // consume
     *     memory_order_acquire,   // acquire
     *     memory_order_release,   // release
     *     memory_order_acq_rel,   // acquire/release
     *     memory_order_seq_cst    // sequentially consistent
     * } memory_order;*/
    bool ret = props[cam_id]->read_flag.load(std::memory_order_relaxed);
    if (ret && !props[cam_id]->image_buffer.isFull())
    {
        if (!props[cam_id]->processed_image.empty())
            props[cam_id]->image_buffer.insert(props[cam_id]->processed_image);

        /* atomic::store
             * Replaces the contained value with new value. */
        props[cam_id]->read_flag.store(false, std::memory_order_relaxed);
        ret = gpu::AlgoPipelineManager::getInstance()->process(image, props[cam_id]->processed_image, props[cam_id]->read_flag, gpu::TreeType(cam_id));
    }
    return ret;
}

bool ImageProcessor::downloadImage(cv::Mat &image, vision::StereoCameraID cam_id)
{
    if (props[cam_id]->image_buffer.isEmpty())
        return false;
    props[cam_id]->image_buffer.remove(image);

    return true;
}





