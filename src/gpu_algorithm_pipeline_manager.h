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
	void clearAlgoNode(TreeType type = TreeType(0));

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
	enum Fmt { UCHAR, FLOAT };
	// NOTE. The image will be converted to float first.
	// During the processing on GPU, a float type image is recommended.
    bool uploadImage(const cv::Mat& src, int stream_id, ImageType imtype, Fmt fmt);
    void processImage(int stream_id, ImageType imtype, Fmt fmt);
    void downloadImage(cv::Mat & res, int stream_id, ImageType imtype);

private:
	static bool is_initialized;

    uint image_width;
    uint image_height;

    enum{ STREAM_NUM = 2 };
    cudaStream_t     d_stream[STREAM_NUM];

	std::array<cv::cuda::GpuMat, STREAM_NUM> mat_gray;
	std::array<cv::cuda::GpuMat, STREAM_NUM> mat_rgba;

	std::array<std::shared_ptr<AlgoNodeTree>, STREAM_NUM> algo_node_tree;

	template<typename Tp, uchar N = STREAM_NUM>
	struct _T {
		std::array<Tp*, N> mat;
		int val_size;
		int type;
	};
	class ImMemory
	{
	public:
		ImMemory() {}
		void initialize(uint image_width, uint image_height);
		void release();

		//template<typename Tp>
		void* memory(ImageType imtype, Fmt fmt)
		{
			switch (imtype)
			{
			case GRAY:
				if (fmt == UCHAR)	return &memory_gray;
				else				return &memory_grayf;
			case BGR:
				if (fmt == UCHAR)	return &memory_bgr;
				else				return &memory_bgrf;
			case BGRA:
				if (fmt == UCHAR)	return &memory_bgra;
				else				return &memory_bgraf;
			}
			return nullptr;
		}

	private:
		_T<uchar> memory_gray;      //!< Used to store the GRAY uchar image to be processed
		_T<uchar> memory_bgr;       //!< Used to store the BGR uchar image to be processed
		_T<uchar> memory_bgra;      //!< Used to store the BGRA uchar image to be processed
		_T<float> memory_grayf;     //!< Used to store the GRAY float image to be processed
		_T<float> memory_bgrf;      //!< Used to store the BGR float image to be processed
		_T<float> memory_bgraf;     //!< Used to store the BGRA float image to be processed
	}data;
};
GPU_ALGO_END

#endif // GPU_ALGORITHM_PIPELINE_MANAGER_H
