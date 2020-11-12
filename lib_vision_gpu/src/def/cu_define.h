#ifndef GPU_ALGORITHM_CU_DEFINE_H
#define GPU_ALGORITHM_CU_DEFINE_H
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

/* Define some constant variables */
#define CUDA_THREAD16	16
#define CUDA_THREAD32	32
#define CUDA_THREAD64	64
#define CUDA_THREAD96	96
#define CUDA_THREAD128	128
#define CUDA_THREAD256	256
#define CUDA_THREAD512	512
#define CUDA_THREAD1024 1024

#define CUDA_WARP 32


/*get threadId: 2D grid and 1D block */
#define _get_threadId_grid2D_block1D() \
    (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/*get blockId: 2D grid */
#define _get_blockId_grid2() \
    (blockIdx.x + blockIdx.y * gridDim.x)


/*get threadId.x: 2D grid and 2D block */
#define _get_threadId_x_grid2D_block2D() \
    (threadIdx.x + blockDim.x * blockIdx.x)

/*get threadId.y: 2D grid and 2D block */
#define _get_threadId_y_grid2D_block2D() \
    (threadIdx.y + blockDim.y * blockIdx.y)


/* Bellow is some usefull type definition.
 * U8C3 corresponds to RGB
 * U8C4 corresponds to RGBA
 */

using U8 = unsigned char;
struct U8C1
{
	__device__ __host__ U8C1()
		: x(0) {}
	unsigned char x;
};
struct U8C3
{
	__device__ __host__ U8C3()
		: x(0), y(0), z(0)
	{}
	unsigned char x;
	unsigned char y;
	unsigned char z;
};
struct U8C4
{
	__device__ __host__ U8C4()
		: x(0), y(0), z(0), w(255)
	{}
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
};

#endif // GPU_ALGORITHM_CU_DEFINE_H
