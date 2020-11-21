#ifndef ALGONODEGAMMA_CU
#define ALGONODEGAMMA_CU
#include "algo_node_gamma.h"
#include "../def/cu_define.h"
#include "../func/func_colorspace.h"

namespace
{
	/* img_hsv = rgb2hsv(src);
	 * y = img_hsv(:, :, 3);
	 * filtered_y = ipo.guidedFilter(y, y, 4, 0.01);
	 * m = mean2(filtered_y); 
	 * gamma = m. ^ ((m - filtered_y). / m);
	 * y_new = y.^gamma;
	 * img_hsv(:, :, 3) = y_new;
	 * dst = hsv2rgb(img_hsv);*/
    __global__ void calcGammaTransform(cv::cuda::PtrStepSz<float3> hsv, cv::cuda::PtrStepSz<float> filtered_y, float mean_y/*, cv::cuda::PtrStepSz<float> ga*/)
    {
        int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / filtered_y.cols;
		int col = thread_id % filtered_y.cols;
		if (row < filtered_y.rows && col < filtered_y.cols)
		{
			float gamma = powf(mean_y, 1.0 - filtered_y(row, col) / mean_y);
			/*ga(row, col) = gamma;*/
			float y_new = powf(hsv(row, col).z, gamma);

			hsv(row, col).z = MAX(MIN(y_new, 1), 0);
		}
    }

	__global__ void calcGammaTransform(cv::cuda::PtrStepSz<float3> hsv, cv::cuda::PtrStepSz<float> filtered_y, float alpha, float ref_L)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / filtered_y.cols;
		int col = thread_id % filtered_y.cols;
		if (row < filtered_y.rows && col < filtered_y.cols)
		{
			float gamma = powf(alpha, 1.0 - filtered_y(row, col) / ref_L);
			float y_new = powf(hsv(row, col).z, gamma);

			hsv(row, col).z = MAX(MIN(y_new, 1), 0);
		}
	}
}

GPU_ALGO_BEGIN
void AdaptiveGamma(cv::cuda::GpuMat &src, cv::cuda::GpuMat &v, cudaStream_t &stream, cv::cuda::GpuMat &tmp, float alpha, float ref_L)
{	
	gpu::cvtColor(src, tmp, BGRA2HSV);
#if CU_DEBUG
	cv::Mat test_hsv; tmp.download(test_hsv);
#endif

	if (abs(alpha) < 0.05 && abs(ref_L) < 0.05) {
		cv::Scalar mean_rgba = cv::cuda::sum(v);
		float mean_v = mean_rgba(0) / (v.rows * v.cols);

		::calcGammaTransform << < dim3(90, 90), 256, 0, stream >> > (tmp, v, mean_v);
	}
	else {
		::calcGammaTransform << < dim3(90, 90), 256, 0, stream >> > (tmp, v, alpha, ref_L);
	}

#if CU_DEBUG
	cv::Mat test_tmp; tmp.download(test_tmp);
#endif

	gpu::cvtColor(tmp, src, HSV2BGRA);

#if CU_DEBUG
	cv::Mat test_res; src.download(test_res);
#endif
}

GPU_ALGO_END
#endif //ALGONODEGAMMA_CU
