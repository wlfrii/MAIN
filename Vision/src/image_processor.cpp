#include "image_processor.h"
#include "camera_parameters.h"
#include "map_calculator.h"
#include "def/micro_define.h"
#include <memory>

ImageProcessor::ImageProcessor(uchar cam_id, const CameraParameters &cam_params, uint image_width, uint image_height)
    : cam_id(cam_id)
    , map_calculator(new MapCalculator(cam_params, image_width, image_height))
    , read_flag(true)
    , disparity(0)
{
    updateRectifyProps();

//    gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GuidedFilterProperty>(gpu::GuidedFilterProperty(0.25)), gpu::TreeType(1));
}

ImageProcessor::~ImageProcessor()
{
    DELETE_PIONTER(map_calculator);
}

bool ImageProcessor::uploadImage(const cv::Mat &image, const int stream_id)
{
    if(processed_image.empty())
        processed_image.create(image.size(), CV_8UC4);

    /* atomic::load
     * Returns the contained value.
     * typedef enum memory_order {
     *     memory_order_relaxed,   //
     *     memory_order_consume,   // consume
     *     memory_order_acquire,   // acquire
     *     memory_order_release,   // release
     *     memory_order_acq_rel,   // acquire/release
     *     memory_order_seq_cst    // sequentially consistent
     * } memory_order;*/
    bool ret = read_flag.load(std::memory_order_relaxed);
    if(ret && !image_buffer.isFull())
    {
        if(!processed_image.empty())
            image_buffer.insert(processed_image);

        /* atomic::store
         * Replaces the contained value with new value. */
        //read_flag.store(false, std::memory_order_relaxed);
//        ret = gpu::AlgoPipelineManager::getInstance()->process(image, processed_image, stream_id, read_flag);
		cv::cvtColor(image, processed_image, cv::COLOR_BGR2BGRA);
    }

    return ret;
}

bool ImageProcessor::downloadImage(cv::Mat &image)
{
    if(image_buffer.isEmpty())
        return false;
    image_buffer.remove(image);
    return true;
}


void ImageProcessor::updateRectifyProps()
{
    // Create the map for rectification
    map_calculator->updateMap(disparity);

    // Set the map as GPU parameters
//    gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::RectifyProperty>(gpu::RectifyProperty(map_calculator->getGPUMapx(), map_calculator->getGPUMapy())), gpu::TreeType(cam_id));
}


