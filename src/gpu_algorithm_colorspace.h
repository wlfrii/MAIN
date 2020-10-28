#pragma once
#include "./def/define.h"

GPU_ALGO_BEGIN

/* Color Space - HSV
 * The HSV representation models the way paints of different colors mix together, with
 * the saturation dimension resembling various tints of brightly colored paint, and
 * the value dimension resembling the mixture of those paints with varying amounts of
 * black or white paint.
 */

enum ColorConvert
{
	BGR2HSV
};

void cvtColor(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, ColorConvert cvt);

GPU_ALGO_END