#ifndef ALGONODEIMAGEADJUST_CU
#define ALGONODEIMAGEADJUST_CU
#include "algo_node_image_adjust.h"
#include "../def/cu_define.h"

//kernel func
namespace
{
	/* Some constant value */
	__constant__ float NORMALIZE_RGB = 1.f / 255.f;

	__device__ void checkBGR(float3& bgr)
	{
		bgr.x = MAX(MIN(bgr.x, 255), 0);
		bgr.y = MAX(MIN(bgr.y, 255), 0);
		bgr.z = MAX(MIN(bgr.z, 255), 0);
	}

	__device__ float3& adjustSaturation(float3& bgr, float saturation)
	{
		int rgb_max = MAX(bgr.x, MAX(bgr.y, bgr.z));
		int rgb_min = MIN(bgr.x, MIN(bgr.y, bgr.z));
		int delta = rgb_max - rgb_min;
		if (delta != 0)
		{
			float s = 0;
			int value = rgb_max + rgb_min;
			if (value < 255)
				s = float(delta) / float(value);
			else
				s = float(delta) / (2 * 255.f - float(value));

			float alpha = 0;
			float temp = 0;
			if (saturation >= 0)
			{
				alpha = MAX(s, 1.f - saturation);
				if (abs(alpha - 1.f) < 1e-3)  // to avoid divide a zeros
					alpha = 0;
				else
					alpha = 1.f / alpha - 1.f;
				temp = (value >> 1) * alpha;
				alpha = alpha + 1;
			}
			else
			{
				alpha = 1 + saturation;
				temp = (value >> 1) * saturation;
			}
			bgr.x = alpha * bgr.x - temp;
			bgr.y = alpha * bgr.y - temp;
			bgr.z = alpha * bgr.z - temp;
			checkBGR(bgr);
		}
		return bgr;
	}

	__device__ float3& adjustContrast(float3& bgr, float contrast)
	{
		bgr.x = (bgr.x - 128) * contrast + 128;
		bgr.y = (bgr.y - 128) * contrast + 128;
		bgr.z = (bgr.z - 128) * contrast + 128;
		checkBGR(bgr);
		return bgr;
	}

	__constant__ float NORM_3 = 1.f / 3.f;
	__constant__ float NORM_128 = 1.f / 128.f;
	__device__ float3& adjustBrightness(float3& bgr, char brightness)
	{
		float L = (bgr.x + bgr.y + bgr.z) * NORM_3;

		if (L < 1e-3) // to avoid divide a zeros
			L = 1;

		if (L > 128)
		{
			bgr.x = (bgr.x * 128 - (L - 128) * 256) / (256 - L);
			bgr.y = (bgr.y * 128 - (L - 128) * 256) / (256 - L);
			bgr.z = (bgr.z * 128 - (L - 128) * 256) / (256 - L);
		}
		else {
			bgr.x = bgr.x * 128 / L;
			bgr.y = bgr.y * 128 / L;
			bgr.z = bgr.z * 128 / L;
		}

		L = L + brightness - 128;
		if (L > 0)
		{
			bgr.x = bgr.x + (256 - bgr.x) * L * NORM_128;
			bgr.y = bgr.y + (256 - bgr.y) * L * NORM_128;
			bgr.z = bgr.z + (256 - bgr.z) * L * NORM_128;
		}
		else
		{
			bgr.x = bgr.x * (1 + L * NORM_128);
			bgr.y = bgr.y * (1 + L * NORM_128);
			bgr.z = bgr.z * (1 + L * NORM_128);
		}
		checkBGR(bgr);
		return bgr;
	}

	__global__ void adjustImageProp_RGB(cv::cuda::PtrStepSz<float3> src, float saturation, float contrast, char brightness)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % src.cols;
		int row = thread_id / src.cols;

		//if (col < src.cols && row < src.rows)
		//{
		//	uchar3 bgr = src(row, col);
		//	// saturation
		//	if (abs(saturation) > 1e-2) {
		//		bgr = adjustSaturation(bgr, saturation);
		//	};
		//	// constrast
		//	if (abs(contrast) > 1e-2) {
		//		bgr = adjustContrast(bgr, brightness);
		//	}
		//	// brightness
		//	if (brightness != 0) {
		//		bgr = adjustBrightness(bgr, brightness);
		//	}
		//	src(row, col) = bgr;
		//}
	}


    __global__ void adjustImageProp_RGBA(cv::cuda::PtrStepSz<float4> src, float saturation, float contrast, char brightness)
    {
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % src.cols;
		int row = thread_id / src.cols;

		if (col < src.cols && row < src.rows)
		{
			float3 bgr = make_float3(src(row, col).x * 255, src(row, col).y * 255, src(row, col).z * 255);
			// saturation
			if (abs(saturation) > 1e-2) {
				bgr = adjustSaturation(bgr, saturation);
			};
			// constrast
			if (abs(contrast) > 1e-2) {
				bgr = adjustContrast(bgr, contrast);
			}
			// brightness
			if (brightness != 0) {
				bgr = adjustBrightness(bgr, brightness);
			}
			src(row, col).x = bgr.x * NORMALIZE_RGB;
			src(row, col).y = bgr.y * NORMALIZE_RGB;
			src(row, col).z = bgr.z * NORMALIZE_RGB;
		}
    }
}

GPU_ALGO_BEGIN
void adjustImageProp(cv::cuda::GpuMat &src, char saturation, char contrast, char brightness, cudaStream_t stream)
{
	float s = MAX(MIN(saturation - 50, 50.f), -50.f) / 50.f;
	float c = powf((50.f + MAX(MIN(contrast - 50, 50.f), -50.f)) / 50.f, 2);
	char b = MAX(MIN(brightness - 50, 50), -50);

	if (src.channels() == 4) {
		::adjustImageProp_RGBA << <dim3(30, 270), 256, 0, stream >> > (src, s, c, b);
	}
	else if (src.channels() == 3) {
		::adjustImageProp_RGB << <dim3(30, 270), 256, 0, stream >> > (src, s, c, b);
	}
}
GPU_ALGO_END


namespace
{
	// adjust the saturation/contrast/brightness of the input image
	template<typename Tp>
	__global__ void adjustImage(cv::cuda::PtrStepSz<Tp> src, cv::cuda::PtrStepSz<uchar3> dst, float saturation, float contrast, char brightness)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int i = thread_id % src.cols;
		int j = thread_id / src.cols;

		if (i < src.cols && j < src.rows)
		{
			int BGR[3];
			BGR[0] = src(j, i).x;
			BGR[1] = src(j, i).y;
			BGR[2] = src(j, i).z;

			// saturation
			if (abs(saturation) >= 1e-2)
			{
				int maxRGB = MAX(src(j, i).x, MAX(src(j, i).y, src(j, i).z));
				int minRGB = MIN(src(j, i).x, MIN(src(j, i).y, src(j, i).z));
				int delta = maxRGB - minRGB;
				if (delta != 0)
				{
					float s = 0;
					int value = maxRGB + minRGB;
					if (value < 255)
						s = float(delta) / float(value);
					else
						s = float(delta) / (2 * 255.f - float(value));

					float alpha = 0;
					float temp = 0;
					if (saturation >= 0)
					{
						alpha = MAX(s, 1.f - saturation);
						if (abs(alpha - 1.f) < 1e-3)  // to avoid divide a zeros
							alpha = 0;
						else
							alpha = 1.f / alpha - 1.f;
						temp = (value >> 1) * alpha;
						alpha = alpha + 1;
					}
					else
					{
						alpha = 1 + saturation;
						temp = (value >> 1) * saturation;
					}
					for (int k = 0; k < 3; k++)
					{
						BGR[k] = BGR[k] * alpha - temp;
						BGR[k] = MAX(MIN(BGR[k], 255), 0);
					}
				}
			}
			// contrast
			if (abs(contrast) >= 1e-2)
			{
				for (int k = 0; k < 3; k++)
				{
					BGR[k] = (BGR[k] - 128) * contrast + 128;
					BGR[k] = MAX(MIN(BGR[k], 255), 0);
				}
			}
			// brightness
			if (brightness != 0)
			{
				float L = (BGR[0] + BGR[1] + BGR[2]) / 3.f;

				if (L < 1e-3) // to avoid divide a zeros
					L = 1;

				if (L > 128)
					for (int k = 0; k < 3; k++)
						BGR[k] = (BGR[k] * 128 - (L - 128) * 256) / (256 - L);
				else
					for (int k = 0; k < 3; k++)
						BGR[k] = BGR[k] * 128 / L;

				L = L + brightness - 128;
				if (L > 0)
					for (int k = 0; k < 3; k++)
					{
						BGR[k] = BGR[k] + (256 - BGR[k]) * L / 128;
						BGR[k] = MAX(MIN(BGR[k], 255), 0);
					}
				else
					for (int k = 0; k < 3; k++)
					{
						BGR[k] = BGR[k] + BGR[k] * L / 128;
						BGR[k] = MAX(MIN(BGR[k], 255), 0);
					}
			}
			dst(j, i) = make_uchar3(BGR[0], BGR[1], BGR[2]);
		}
	}
}
void adjustImage(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, char saturation /*= 0*/, char contrast /*= 0*/, char brightness /*= 0*/)
{
	assert(src.type() == CV_8UC3 && dst.type() == CV_8UC3);

	if (saturation == 0 && contrast == 0 && brightness == 0)
		return;

	float satur = MAX(MIN(saturation, 100.f), -100.f) / 100.f;
	float contra = powf((100.f + MAX(MIN(contrast, 100.f), -100.f)) / 100.f, 2);
	int bright = MAX(MIN(brightness, 100), -100);

	static int threads = CUDA_THREAD96;
	static dim3 block_per_grid(std::ceil(1.f * src.cols / threads * 10), std::ceil(1.f * src.rows / 10));

	//::adjustImage<uchar3> << <block_per_grid, threads >> > (src, dst, satur, contra, bright);
}

#endif // ALGONODEIMAGEADJUST_CU
