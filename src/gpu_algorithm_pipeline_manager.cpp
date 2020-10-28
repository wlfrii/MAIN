#include "gpu_algorithm_pipeline_manager.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

GPU_ALGO_BEGIN
void CUDART_CB AlgoPipelineManager::finishCallBack(cudaStream_t stream, cudaError_t status, void* data)
{
    std::atomic<bool>* flag = (std::atomic<bool>*) data;
    if(flag)
        flag->store(true, std::memory_order_relaxed);
}

bool AlgoPipelineManager::is_initialized = false;


AlgoPipelineManager::AlgoPipelineManager()
    : image_width(1920)
    , image_height(1080)
{
    for(auto i = 0; i < STREAM_NUM; i++)
        algo_node_tree[i].reset(new AlgoNodeTree());
}

AlgoPipelineManager::~AlgoPipelineManager()
{
    release();
}

std::shared_ptr<AlgoPipelineManager> AlgoPipelineManager::getInstance()
{
	static std::shared_ptr<AlgoPipelineManager> algo_manager(new AlgoPipelineManager());
	return algo_manager;
}

void AlgoPipelineManager::intialize(uint image_width, uint image_height)
{
    this->image_width = image_width;
    this->image_height = image_height;

    for(auto i = 0; i < int(STREAM_NUM); i++) {
        // Create streams
        cudaStreamCreate(&d_stream[i]);

        /* Allocate memory for gpu mat.
         * Linear memory is typically allocated using cudaMalloc() and freed using cudaFree()
         * and data transfer between host memory and device memory are tyypically done using
         * cudaMemcpy().
         * There, allocate memory on device to store the BGR image to be processed. */
		cudaMalloc((void **)&memory_gray[i], image_width*image_height*sizeof(uchar));
		cudaMalloc((void **)&memory_rgb[i], image_width*image_height*3*sizeof (uchar));
		cudaMalloc((void **)&memory_rgba[i], image_width*image_height*4*sizeof(uchar));
    }
    
	is_initialized = true;
}


void AlgoPipelineManager::addAlgoNode(AlgoNodeBase* algo_node, TreeType type)
{
	algo_node_tree[type]->insert(algo_node);
}


void AlgoPipelineManager::setProperty(std::shared_ptr<Property> prop, TreeType type)
{
    algo_node_tree[type]->setProperty(prop);
}


bool AlgoPipelineManager::process(const cv::Mat &src, cv::Mat &res, TreeType type, std::atomic<bool> &flag)
{
	if (!is_initialized) {
		printf("The instance should be initialized first!\n");
		return false;
	}

	ImageType imtype;
	if (src.channels() == 1) {
		imtype = gpu::GRAY;
	}
	else if(src.channels() == 3) {
		imtype = gpu::RGB;
	}
	else {
		imtype = gpu::RGBA;
	}
	if (!uploadImage((U8*)src.data, type, imtype))
		return false;
	processImage(type, imtype);
	downloadImage(res, type, imtype);

    /* The commands that are issued in a stream after a callback do not start executing before
     * the callback has completed.
     * The last parameter of cudaStreamAddCallBack() is reserved for future use. */
    cudaStreamAddCallback(d_stream[type], finishCallBack, (void*)(&flag), 0);

    return true;
}

void AlgoPipelineManager::release()
{
    for(auto i = 0; i < STREAM_NUM; i++) {
        // Release the streams
        cudaStreamDestroy(d_stream[i]);
        // Deallocate memory
		cudaFree(memory_gray[i]);
		cudaFree(memory_rgb[i]);
		cudaFree(memory_rgba[i]);
    }
}

std::shared_ptr<AlgoNodeTree> AlgoPipelineManager::getAlgoTree(TreeType type)
{
    if(int(type) < STREAM_NUM)
        return algo_node_tree[type];
    else
        return nullptr;
}


bool AlgoPipelineManager::uploadImage(uchar *src, int stream_id, ImageType imtype)
{
    // Copies data from host (src) to device (d_uc_mem)
	cudaError_t status;
	switch (imtype)
	{
	case gpu::GRAY:
		status = cudaMemcpyAsync(memory_gray[stream_id], src, image_width*image_height*sizeof(uchar), cudaMemcpyHostToDevice, d_stream[stream_id]);
		break;
	case gpu::RGB:
		status = cudaMemcpyAsync(memory_rgb[stream_id], src, image_width*image_height*3*sizeof(uchar), cudaMemcpyHostToDevice, d_stream[stream_id]);
		break;
	case gpu::RGBA:
		status = cudaMemcpyAsync(memory_rgba[stream_id], src, image_width*image_height*4*sizeof(uchar), cudaMemcpyHostToDevice, d_stream[stream_id]);
		break;
	}
	
    if(status != cudaSuccess) {
        printf("AlgoPipelineManager::uploadData: upload pipeline %d failed!\n", stream_id);
        return false;
    }
    return true;
}


void AlgoPipelineManager::processImage(int stream_id, ImageType imtype)
{
    // Get stream
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(d_stream[stream_id]);
	switch (imtype)
	{
	case gpu::GRAY:
		// Get the image in the stream
		mat_gray[stream_id] = cv::cuda::GpuMat(image_height, image_width, CV_8UC1, memory_gray[stream_id]);
		// Process the 1-channel image
		algo_node_tree[stream_id]->process(mat_gray[stream_id], d_stream[stream_id]);
		break;
	case gpu::RGB:
		// Get the image in the stream and Convert image to RGRA
		cv::cuda::cvtColor(cv::cuda::GpuMat(image_height, image_width, CV_8UC3, memory_rgb[stream_id]), mat_rgba[stream_id], cv::COLOR_BGR2BGRA);
		// Process the 4-channel image
		algo_node_tree[stream_id]->process(mat_rgba[stream_id], d_stream[stream_id]);
		break;
	case gpu::RGBA:
		// Get the image in the stream
		mat_rgba[stream_id] = cv::cuda::GpuMat(image_height, image_width, CV_8UC4, memory_rgba[stream_id]);
		// Process the 4-channel image
		algo_node_tree[stream_id]->process(mat_rgba[stream_id], d_stream[stream_id]);
		break;
	}
}


void AlgoPipelineManager::downloadImage(cv::Mat &res, int stream_id, ImageType imtype)
{
    // Class StreamAccessor enables getting cudaStream_t from cuda::Stream.
    cv::cuda::Stream cv_stream;
    cv_stream = cv::cuda::StreamAccessor::wrapStream(d_stream[stream_id]);
	switch (imtype)
	{
	case gpu::GRAY:
		mat_gray[stream_id].download(res, cv_stream);
		break;
	case gpu::RGB:
	case gpu::RGBA:
		mat_rgba[stream_id].download(res, cv_stream);
		break;
	default:
		break;
	}
	
}
GPU_ALGO_END
