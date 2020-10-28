#include "gpu_algorithm_colorspace.h"
#include "./def/cu_define.h"

namespace
{
	__global__ void rgb2hsv(gpu::U8C3 *src, gpu::U8C3 *dst, int width, int height)
	{
		int thread_id = _get_threadId_grid2D_block1D();

		if (thread_id < width*height)
		{
			gpu::U8C3 tmp_src = src[thread_id];
			gpu::U8C3 tmp_dst = dst[thread_id];

			tmp_dst.z = MAX(tmp_src.x, MAX(tmp_src.y, tmp_src.z));
		}

		/*float max = 0, min = 0;
		R = R / 100;
		G = G / 100;
		B = B / 100;

		max = retmax(R, G, B);
		min = retmin(R, G, B);
		*v = max;
		if (max == 0)
			*s = 0;
		else
			*s = 1 - (min / max);

		if (max == min)
			*h = 0;
		else if (max == R && G >= B)
			*h = 60 * ((G - B) / (max - min));
		else if (max == R && G < B)
			*h = 60 * ((G - B) / (max - min)) + 360;
		else if (max == G)
			*h = 60 * ((B - R) / (max - min)) + 120;
		else if (max == B)
			*h = 60 * ((R - G) / (max - min)) + 240;

		*v = *v * 100;
		*s = *s * 100;*/
	}

	__global__ void hsv2rgb(gpu::U8C3 *src, gpu::U8C3 *dst)
	{

	}
}


GPU_ALGO_BEGIN
void cvtColor(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, ColorConvert cvt)
{
	switch (cvt)
	{
	case gpu::BGR2HSV:

		break;
	default:
		break;
	}
}
GPU_ALGO_END