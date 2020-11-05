#include "../gpu_algorithm_func.h"
#include "../def/cu_define.h"

namespace
{
	__global__ void cvt8UC1to32FC1(cv::cuda::PtrStepSz<uchar> src, cv::cuda::PtrStepSz<float> cvt_src)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % src.cols;
		int row = thread_id / src.cols;

		if (col < src.cols && row < src.rows)
		{
			cvt_src(row, col) = float(src(row, col)) / 255.f;
		}
	}
	__global__ void cvt8UC3to32FC3(cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<float3> cvt_src)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % src.cols;
		int row = thread_id / src.cols;

		if (col < src.cols && row < src.rows)
		{
			cvt_src(row, col).x = float(src(row, col).x) / 255.f;
			cvt_src(row, col).y = float(src(row, col).y) / 255.f;
			cvt_src(row, col).z = float(src(row, col).z) / 255.f;
		}
	}
	__global__ void cvt8UC4to32FC4(cv::cuda::PtrStepSz<uchar4> src, cv::cuda::PtrStepSz<float4> cvt_src)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % src.cols;
		int row = thread_id / src.cols;

		if (col < src.cols && row < src.rows)
		{
			cvt_src(row, col).x = float(src(row, col).x) / 255.f;
			cvt_src(row, col).y = float(src(row, col).y) / 255.f;
			cvt_src(row, col).z = float(src(row, col).z) / 255.f;
			cvt_src(row, col).w = float(src(row, col).w) / 255.f;
		}
	}
}

GPU_ALGO_BEGIN
void convertImageFormat(cv::cuda::GpuMat &src, cudaStream_t stream)
{
#if CU_DEBUG
	//cv::Mat test_src; src.download(test_src);
#endif

	auto imfmt = src.type();
	if (imfmt == CV_8UC1) {
		cv::cuda::GpuMat tmp1(src.size(), CV_32FC1);
		::cvt8UC1to32FC1 << < dim3(90, 90), 256, 0, stream >> > (src, tmp1);
		src = tmp1;
	}
	else if (imfmt == CV_8UC3) {
		cv::cuda::GpuMat tmp3(src.size(), CV_32FC3);
		::cvt8UC3to32FC3 << < dim3(90, 90), 256, 0, stream >> > (src, tmp3);
		src = tmp3;
	}
	else if (imfmt == CV_8UC4) {
		cv::cuda::GpuMat tmp4(src.size(), CV_32FC4);
		::cvt8UC4to32FC4 << < dim3(90, 90), 256, 0, stream >> > (src, tmp4);
		src = tmp4;
	}
#if CU_DEBUG
	//cv::Mat test_res; src.download(test_res);
#endif
}

GPU_ALGO_END