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

	bool isReady() const;

	/*@brief Build the algo node tree.
	 */
	void addAlgoNode(AlgoNodeBase* algo_node, TreeType type = TreeType(0));

    /*@brief Set property of algo node specified by TreeType
	 */
    void setProperty(std::shared_ptr<Property> prop, TreeType type = TreeType(0));

	/*@brief Process the input image based on the built algo node tree.
	 *@param src The image to be processed, supporting format are CV_8UC1, CV_8UC3, CV_8UC4
	 *           CV_32FC1, CV_32FC3, CV_32FC4.
	 *@param res The processed image with same type as src.
	 */
    bool process(const cv::Mat & src, cv::Mat & res, std::atomic<bool>& flag, TreeType type = TreeType(0));


    // Get algo tree
    std::shared_ptr<AlgoNodeTree> getAlgoTree(TreeType type = TreeType(0));

    static void CUDART_CB finishCallBack(cudaStream_t stream, cudaError_t status, void * data);

private:
	// NOTE. The image will be converted to float first.
	// During the processing on GPU, a float type image is recommended.
    bool uploadImage(uchar* src, int stream_id, ImageType imtype);
	bool uploadImage(const float* src, int stream_id, ImageType imtype);
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
	float*           memory_grayf[STREAM_NUM];     //!< Used to store the GRAY image to be processed
	float*           memory_rgbf[STREAM_NUM];      //!< Used to store the BRG image to be processed
	float*           memory_rgbaf[STREAM_NUM];     //!< Used to store the BRGA image to be processed

	std::array<cv::cuda::GpuMat, STREAM_NUM> mat_gray;
    std::array<cv::cuda::GpuMat, STREAM_NUM> mat_rgba;

    std::array<std::shared_ptr<AlgoNodeTree>, STREAM_NUM> algo_node_tree;

};
GPU_ALGO_END

#endif // GPU_ALGORITHM_PIPELINE_MANAGER_H
