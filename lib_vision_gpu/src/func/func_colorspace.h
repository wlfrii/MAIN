#pragma once
#include "../def/define.h"

GPU_ALGO_BEGIN

/* Color Space - HSV
 * The HSV representation models the way paints of different colors mix together, with
 * the saturation dimension resembling various tints of brightly colored paint, and
 * the value dimension resembling the mixture of those paints with varying amounts of
 * black or white paint.
 */

enum ColorConvertType
{
	BGR2HSV,
	HSV2BGR,
	BGRA2HSV,
	HSV2BGRA,
};

void cvtColor(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, ColorConvertType cvttype, cudaStream_t stream = 0);


/*@brief Calculate V in HSV space.
 */
void calcVbyHSV(cv::cuda::GpuMat &src, cv::cuda::GpuMat &v, cudaStream_t stream = 0);
GPU_ALGO_END