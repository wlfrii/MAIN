#ifndef GPU_ALGORITHM_CU_DEFINE_H
#define GPU_ALGORITHM_CU_DEFINE_H

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


#endif // GPU_ALGORITHM_CU_DEFINE_H
