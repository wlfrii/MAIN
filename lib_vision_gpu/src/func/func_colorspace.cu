#include "func_colorspace.h"
#include "../def/cu_define.h"

namespace
{
	__constant__ float ZERO_LIMIT = 1e-3;
	__device__ float3 rgb2hsv(float3& rgb)
	{
		float max_rgb = MAX(MAX(rgb.x, rgb.y), rgb.z);
		float min_rgb = MIN(MIN(rgb.x, rgb.y), rgb.z);
		// Calculate Saturation
		float S = 0;
		if (abs(max_rgb) >= ZERO_LIMIT)
			S = 1 - min_rgb / max_rgb;
		// Calcuate Hue
		float H = 0;
		float C = max_rgb - min_rgb;
		bool isvalid = C > ZERO_LIMIT;
		if (isvalid)
		{
			if (abs(max_rgb - rgb.x) < ZERO_LIMIT) // max = r
			{
				float tmp = (rgb.y - rgb.z) / C;
				if (tmp < 0) H = 6 + tmp;
				else H = tmp;
			}
			else if (abs(max_rgb - rgb.y) < ZERO_LIMIT) { // max = g
				H = (rgb.z - rgb.x) / C + 2;
			}
			else { // max = b
				H = (rgb.x - rgb.y) / C + 4;
			}
		}
		else
			H = 0;
		// Project H to [0,1]
		return make_float3(H / 6, S, max_rgb);
	}
	__device__ float3 hsv2rgb(float3& hsv)
	{
		auto H = hsv.x * 6; // Since the H has been projected to [0,1]
		auto S = hsv.y;
		auto V = hsv.z;
		float C = 0;
		if (abs(S) > ZERO_LIMIT) C = V * S;

		float m = V - C; // the min value
		float X = C * (1 - abs(fmod(H, 2.f) - 1));
		float R = 0, G = 0, B = 0;
		if (H >= 0 && H < 1) {
			R = V; G = X + m; B = m;
		}
		else if (H >= 1 && H < 2) {
			R = X + m; G = V; B = m;
		}
		else if (H >= 2 && H < 3) {
			R = m; G = V; B = X + m;
		}
		else if (H >= 3 && H < 4) {
			R = m; G = X + m; B = V;
		}
		else if (H >= 4 && H < 5) {
			R = X + m; G = m; B = V;
		}
		else { // (H >= 5 && H <= 6) 
			R = V; G = m; B = X + m;
		}
		return make_float3(R, G, B);
	}
	__global__ void bgr2hsv(cv::cuda::PtrStepSz<float3> bgr, cv::cuda::PtrStepSz<float3> hsv)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / bgr.cols;
		int col = thread_id % bgr.cols;
		if (row < bgr.rows && col < bgr.cols)
		{
			auto rgb = make_float3(bgr(row, col).z, bgr(row, col).y, bgr(row, col).x);	
			hsv(row, col) = rgb2hsv(rgb);
		}
	}
	__global__ void bgra2hsv(cv::cuda::PtrStepSz<float4> bgra, cv::cuda::PtrStepSz<float3> hsv)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / bgra.cols;
		int col = thread_id % bgra.cols;
		if (row < bgra.rows && col < bgra.cols)
		{
			auto rgb = make_float3(bgra(row, col).z, bgra(row, col).y, bgra(row, col).x);
			hsv(row, col) = rgb2hsv(rgb);
		}
	}

	__global__ void hsv2bgr(cv::cuda::PtrStepSz<float3> hsv, cv::cuda::PtrStepSz<float3> bgr)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / hsv.cols;
		int col = thread_id % hsv.cols;
		if (row < hsv.rows && col < hsv.cols)
		{
			auto rgb = hsv2rgb(hsv(row, col));
			bgr(row, col) = make_float3(rgb.z, rgb.y, rgb.x);
		}
	}
	__global__ void hsv2bgra(cv::cuda::PtrStepSz<float3> hsv, cv::cuda::PtrStepSz<float4> bgra)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / hsv.cols;
		int col = thread_id % hsv.cols;
		if (row < hsv.rows && col < hsv.cols)
		{
			auto rgb = hsv2rgb(hsv(row, col));
			bgra(row, col) = make_float4(rgb.z, rgb.y, rgb.x, 1);
		}
	}

	__global__ void calcVbyHSV(cv::cuda::PtrStepSz<float4> src, cv::cuda::PtrStepSz<float> v)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / src.cols;
		int col = thread_id % src.cols;
		if (row < src.rows && col < src.cols)
		{
			v(row, col) = MAX(MAX(src(row, col).x, src(row, col).y), src(row, col).z);
		}
	}
}


GPU_ALGO_BEGIN
void cvtColor(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, ColorConvertType cvttype, cudaStream_t stream)
{
	switch (cvttype)
	{
	case BGR2HSV:
		::bgr2hsv << < dim3(90, 90), 256, 0, stream >> > (src, dst);
		break;
	case HSV2BGR:
		::hsv2bgr << < dim3(90, 90), 256, 0, stream >> > (src, dst);
		break;
	case BGRA2HSV:
		::bgra2hsv << < dim3(90, 90), 256, 0, stream >> > (src, dst);
		break;
	case HSV2BGRA:
		::hsv2bgra << < dim3(90, 90), 256, 0, stream >> > (src, dst);
		break;
	default:
		break;
	}
}

void calcVbyHSV(cv::cuda::GpuMat &src, cv::cuda::GpuMat &v, cudaStream_t stream)
{
	::calcVbyHSV << < dim3(90, 90), 256, 0, stream >> > (src, v);
}
GPU_ALGO_END