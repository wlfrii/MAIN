#ifndef GPU_ALGORITHM_PIPELINE_MANAGER_H
#define GPU_ALGORITHM_PIPELINE_MANAGER_H
#include "./def/define.h"
#include "gpu_algorithm_node_tree.h"
#include <memory>
#include <array>
#include <atomic>

GPU_ALGO_BEGIN
class AlgoPipelineManager
{
private:
    AlgoPipelineManager();
public:
    ~AlgoPipelineManager();

    static std::shared_ptr<AlgoPipelineManager> getInstance();

    void intialize(uint image_width = 1920, uint image_height = 1080);


	void addAlgoNode(AlgoNodeBase* algo_node, TreeType type);


    /*@brief Set property of algo node specified by TreeType
	 */
    void setProperty(std::shared_ptr<Property> prop, TreeType type);

    bool process(const cv::Mat & src, cv::Mat & res, TreeType type, std::atomic<bool>& flag);

    void release();


    // Get algo tree
    std::shared_ptr<AlgoNodeTree> getAlgoTree(TreeType type);

    static void CUDART_CB finishCallBack(cudaStream_t stream, cudaError_t status, void * data);

private:
    bool uploadImage(uchar* src, int stream_id, ImageType imtype);
    void processImage(int stream_id, ImageType imtype);
    void downloadImage(cv::Mat & res, int stream_id, ImageType imtype);

private:
	static bool is_initialized;

    uint image_width;
    uint image_height;

    enum{ STREAM_NUM = 2 };
    cudaStream_t     d_stream[STREAM_NUM];

	uchar*           memory_gray[STREAM_NUM];      //!< Used to store the GRAY image to be processed
    uchar*           memory_rgb[STREAM_NUM];       //!< Used to store the BRG image to be processed
	uchar*           memory_rgba[STREAM_NUM];      //!< Used to store the BRGA image to be processed

	std::array<cv::cuda::GpuMat, STREAM_NUM> mat_gray;
    std::array<cv::cuda::GpuMat, STREAM_NUM> mat_rgba;

    std::array<std::shared_ptr<AlgoNodeTree>, STREAM_NUM> algo_node_tree;
};
GPU_ALGO_END

#endif // GPU_ALGORITHM_PIPELINE_MANAGER_H
