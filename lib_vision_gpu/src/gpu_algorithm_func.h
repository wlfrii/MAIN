#ifndef GPU_ALGORITHM_FUNC_H
#define GPU_ALGORITHM_FUNC_H
#include "./def/define.h"
/* The other func algorithm */
#include "./func/func_colorspace.h"


GPU_ALGO_BEGIN
enum cvtType
{
	T_8U_2_32F,
	T_32F_2_8U
};
/*@brief Convert uchar image to float image, while convert [0,255] to [0,1].
 */
void cvtImageFormat(cv::cuda::GpuMat &src, cvtType cvttype, cudaStream_t stream = 0);



GPU_ALGO_END
#endif //GPU_ALGORITHM_FUNC_H

