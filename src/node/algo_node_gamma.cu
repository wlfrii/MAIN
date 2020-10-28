#ifndef ALGONODEGAMMA_CU
#define ALGONODEGAMMA_CU
#include "algo_node_gamma.h"
#include "../def/cu_define.h"

namespace
{
	/*__global__ void getVfromHSV(gpu::U8C3 *hsv, gpu::U8C1 *y, int width, int height)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / width;
		int col = thread_id % width;
		if (row < width && col < height)
		{
			int index = width * 3 * row + col * 3;
			gpu::U8C3 tmp = hsv[index];
			y[index].x = tmp.z;
		}
	}*/
	__global__ void getVfromHSV(cv::cuda::PtrStepSz<uchar3> hsv, cv::cuda::PtrStepSz<float1> y)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / hsv.cols;
		int col = thread_id % hsv.cols;
		if (row < hsv.rows && col < hsv.cols)
		{
			y(row, col).x = hsv(row, col).z / 255.f;
		}
	}

	/* img_hsv = rgb2hsv(src);
	 * y = img_hsv(:, :, 3);
	 * filtered_y = ipo.guidedFilter(y, y, 4, 0.01);
	 * m = mean2(filtered_y); 
	 * gamma = m. ^ ((m - filtered_y). / m);
	 * y_new = y.^gamma;
	 * img_hsv(:, :, 3) = y_new;
	 * dst = hsv2rgb(img_hsv);*/
    __global__ void calcGammaTransform(cv::cuda::PtrStepSz<uchar3> hsv, cv::cuda::PtrStepSz<float> filtered_y, float mean_y)
    {
        int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / filtered_y.cols;
		int col = thread_id % filtered_y.cols;
		if (row < filtered_y.rows && col < filtered_y.cols)
		{
			float gamma = pow(mean_y, 1 - filtered_y(row, col) / mean_y);

			float y_new = pow(hsv(row, col).z / 255.f, gamma);

			hsv(row, col).z = (uchar)MAX(MIN(y_new * 255, 255), 0);
		}
    }
}

GPU_ALGO_BEGIN
void AdaptiveGamma(cv::cuda::GpuMat &src, const cudaStream_t &stream, std::array<cv::cuda::GpuMat, 2> &tmp)
{
	int width = src.cols;
	int height = src.rows;

    // tmp[0] is the origin hsv image, tmp[1] is the filtered hsv image
	cv::cuda::GpuMat v(height, width, CV_32FC1);
	::getVfromHSV << < dim3(90, 90), 256, 0, stream >> > (tmp[1], v);

	cv::Scalar mean_rgba = cv::cuda::sum(v);
	float mean_v = mean_rgba(0) / (v.rows * v.cols);

	// calculate new v and store in tmp[0]
	::calcGammaTransform << < dim3(90, 90), 256, 0, stream >> > (tmp[0], v, mean_v);

	cv::cuda::cvtColor(tmp[0], tmp[1], cv::COLOR_HSV2BGR);

	//cv::Mat test(tmp[1].size(), tmp[1].type());
	//tmp[1].download(test);

	cv::cuda::cvtColor(tmp[1], src, cv::COLOR_BGR2BGRA);
}

//void AdaptiveGamma(cv::cuda::GpuMat &src, const cudaStream_t &stream, std::array<cv::cuda::GpuMat, 2> &tmp, cv::cuda::GpuMat &uneven_y)
//{
//
//}
GPU_ALGO_END
#endif //ALGONODEGAMMA_CU
