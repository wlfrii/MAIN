#ifndef GPU_ALGORITHM_FUNC_H
#define GPU_ALGORITHM_FUNC_H
#include "./def/define.h"
/* The other func algorithm */
#include "./func/func_colorspace.h"


GPU_ALGO_BEGIN
/*@brief Convert uchar image to float image, while convert [0,255] to [0,1].
 */
void convertImageFormat(cv::cuda::GpuMat &src, cudaStream_t stream = 0);



GPU_ALGO_END
#endif //GPU_ALGORITHM_FUNC_H

