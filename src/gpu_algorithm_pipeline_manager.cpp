#include "gpu_algorithm_pipeline_manager.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "gpu_algorithm_func.h"

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
	for (auto i = 0; i < STREAM_NUM; i++) {
		// Release the streams
		cudaStreamDestroy(d_stream[i]);
	}
	data.release();
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
    }
	data.initialize(image_width, image_height);

	is_initialized = true;
}


bool AlgoPipelineManager::isReady() const
{
	return is_initialized;
}


void AlgoPipelineManager::addAlgoNode(AlgoNodeBase* algo_node, TreeType type/* = TreeType(0)*/)
{
	if (!is_initialized) {
		printf("The instance should be initialized first!\n");
		return;
	}
	algo_node_tree[type]->insert(algo_node);
}


void AlgoPipelineManager::clearAlgoNode(TreeType type/* = TreeType(0)*/)
{

}

void AlgoPipelineManager::setProperty(std::shared_ptr<Property> prop, TreeType type/* = TreeType(0)*/)
{
    algo_node_tree[type]->setProperty(prop);
}


bool AlgoPipelineManager::process(const cv::Mat &src, cv::Mat &res, std::atomic<bool> &flag, TreeType type/* = TreeType(0)*/)
{
	if (!is_initialized) {
		printf("The instance should be initialized first!\n");
		return false;
	}
	
	ImageType imtype;
	if (src.channels() == 1) 		imtype = gpu::GRAY;
	else if(src.channels() == 3) 	imtype = gpu::BGR;
	else							imtype = gpu::BGRA;
	
	Fmt fmt;
	if (src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4) {
		fmt = UCHAR;
	}
	else if (src.type() == CV_32FC1 || src.type() == CV_32FC3 || src.type() == CV_32FC4) {
		fmt = FLOAT;
	}
	else return false;

	if (!uploadImage(src, type, imtype, fmt))
		return false;
	processImage(type, imtype, fmt);
	downloadImage(res, type, imtype);

    /* The commands that are issued in a stream after a callback do not start executing before
     * the callback has completed.
     * The last parameter of cudaStreamAddCallBack() is reserved for future use. */
    cudaStreamAddCallback(d_stream[type], finishCallBack, (void*)(&flag), 0);

    return true;
}


std::shared_ptr<AlgoNodeTree> AlgoPipelineManager::getAlgoTree(TreeType type/* = TreeType(0)*/)
{
    if(int(type) < STREAM_NUM)
        return algo_node_tree[type];
    else
        return nullptr;
}

bool AlgoPipelineManager::uploadImage(const cv::Mat& src, int stream_id, ImageType imtype, Fmt fmt)
{
    // Copies data from host (src) to device (d_uc_mem)
	cudaError_t status;
	if (fmt == UCHAR) {
		auto tmp = reinterpret_cast<_T<uchar>*>(data.memory(imtype, fmt));
		status = cudaMemcpyAsync(tmp->mat[stream_id], src.ptr<uchar>(0), image_width*image_height*tmp->val_size, cudaMemcpyHostToDevice, d_stream[stream_id]);
	}
	else if (fmt == FLOAT) {
		auto tmp = reinterpret_cast<_T<float>*>(data.memory(imtype, fmt));
		status = cudaMemcpyAsync(tmp->mat[stream_id], src.ptr<float>(0), image_width*image_height*tmp->val_size, cudaMemcpyHostToDevice, d_stream[stream_id]);
	}	
    if(status != cudaSuccess) {
        printf("AlgoPipelineManager::uploadData: upload pipeline %d failed!\n", stream_id);
        return false;
    }
    return true;
}


void AlgoPipelineManager::processImage(int stream_id, ImageType imtype, Fmt fmt)
{
	auto process = [&imtype, &stream_id, this](auto &tmp) {
		if (imtype == gpu::GRAY) {
			// Get the image
			mat_gray[stream_id] = cv::cuda::GpuMat(image_height, image_width, tmp->type, tmp->mat[stream_id]);
			// Project image's [0,255] to [0,1]
			convertImageFormat(mat_gray[stream_id], d_stream[stream_id]);
			// Process the 1-channel image
			algo_node_tree[stream_id]->process(mat_gray[stream_id], d_stream[stream_id]);
			return;
		}
		else if (imtype == gpu::BGR) {
			auto temp = cv::cuda::GpuMat(image_height, image_width, tmp->type, tmp->mat[stream_id]);
			cv::Mat test_temp; temp.download(test_temp);
			cv::cuda::cvtColor(temp, mat_rgba[stream_id], cv::COLOR_BGR2BGRA);
		}
		else {
			mat_rgba[stream_id] = cv::cuda::GpuMat(image_height, image_width, tmp->type, tmp->mat[stream_id]);
		}
		// Project image's [0,255] to [0,1]
		convertImageFormat(mat_rgba[stream_id], d_stream[stream_id]);
		// Process the 4-channel image
		algo_node_tree[stream_id]->process(mat_rgba[stream_id], d_stream[stream_id]);
	};
	if (fmt == UCHAR) {

		auto tmp = reinterpret_cast<_T<uchar>*>(data.memory(imtype, fmt));
		process(tmp);
	}
	else{
		auto tmp = reinterpret_cast<_T<float>*>(data.memory(imtype, fmt));
		process(tmp);
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
	case gpu::BGR:
	case gpu::BGRA:
		mat_rgba[stream_id].download(res, cv_stream);
		break;
	default:
		break;
	}
	
}

void AlgoPipelineManager::ImMemory::initialize(uint image_width, uint image_height)
{
	for (auto i = 0; i < int(STREAM_NUM); i++) {

		/* Allocate memory for gpu mat.
		 * Linear memory is typically allocated using cudaMalloc() and freed using cudaFree()
		 * and data transfer between host memory and device memory are tyypically done using
		 * cudaMemcpy().
		 * There, allocate memory on device to store the BGR image to be processed. */
		cudaMalloc((void **)&memory_gray.mat[i], image_width*image_height * sizeof(uchar));
		cudaMalloc((void **)&memory_bgr.mat[i], image_width*image_height * 3 * sizeof(uchar));
		cudaMalloc((void **)&memory_bgra.mat[i], image_width*image_height * 4 * sizeof(uchar));
		cudaMalloc((void **)&memory_grayf.mat[i], image_width*image_height * sizeof(float));
		cudaMalloc((void **)&memory_bgrf.mat[i], image_width*image_height * 3 * sizeof(float));
		cudaMalloc((void **)&memory_bgraf.mat[i], image_width*image_height * 4 * sizeof(float));
	}
	memory_gray.type = CV_8UC1;
	memory_gray.val_size = sizeof(uchar);
	memory_bgr.type = CV_8UC3;
	memory_bgr.val_size = 3 * sizeof(uchar);
	memory_bgra.type = CV_8UC4;
	memory_bgra.val_size = 4 * sizeof(uchar);

	memory_grayf.type = CV_32FC1;
	memory_grayf.val_size = sizeof(float);
	memory_bgrf.type = CV_32FC3;
	memory_bgrf.val_size = 3 * sizeof(float);
	memory_bgraf.type = CV_32FC4;
	memory_bgraf.val_size = 4 * sizeof(float);
}
void AlgoPipelineManager::ImMemory::release()
{
	for (auto i = 0; i < STREAM_NUM; i++) {
		// Deallocate memory
		cudaFree(memory_gray.mat[i]);
		cudaFree(memory_bgr.mat[i]);
		cudaFree(memory_bgra.mat[i]);
		cudaFree(memory_grayf.mat[i]);
		cudaFree(memory_bgrf.mat[i]);
		cudaFree(memory_bgraf.mat[i]);
	}
}
GPU_ALGO_END
