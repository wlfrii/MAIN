#ifndef ALGONODEGUIDEDFILTER_CU
#define ALGONODEGUIDEDFILTER_CU
#include "algo_node_guidedfilter.h"
#include "../def/cu_define.h"


namespace
{
	/** \b Downsampling the input gray-scale image
	 * \p src The address of input gray-scale image
	 * \p dst The output
	 * \p width The width of input image
	 * \p height The height of input image
	 * \p channel The channel of input image
	 * \p scale The scale for downsampling
	 */
	__global__ void downSampling_C1(float *src, float *dst, int scale, int width, int height)
	{
		int thread_id = _get_threadId_grid2D_block1D();

		int down_width = width / scale;
		int down_height = height / scale;

		int row = thread_id / down_width;
		int col = thread_id % down_width;
		
		if (row < down_height && col < down_width)
		{
			// index = width * channel * (row * scale + id) + (col * scale + id) * channel
			int index = width * (row * scale + scale - 1) + (col * scale + scale - 1);

			dst[thread_id] = src[index];
		}
	}
	template <typename T>
	__global__ void downSampling_C3C4(T *src, float *dst, int scale, int width, int height, unsigned char channel)
	{
		int thread_id = _get_threadId_grid2D_block1D();

		int down_width = width / scale;
		int down_height = height / scale;

		int row = thread_id / down_width;
		int col = thread_id % down_width;

		if (row < down_height && col < down_width)
		{
			// index = width * channel * (row * scale + id) + (col * scale + id) * channel
			int index = width * (row * scale + scale - 1) + (col * scale + scale - 1);
			T temp = src[index];

			// Disgard the 4th channel if a RGBA image input
			dst[3 * thread_id] = temp.x;
			dst[3 * thread_id + 1] = temp.y;
			dst[3 * thread_id + 2] = temp.z;
		}
	}
	
	/** \b Upsampling the input gray-scale image
	 * \p src The address of input gray-scale image
	 * \p dst The output
	 * \p width The width of input image
	 * \p height The height of input image
	 * \p channel The channel of input image
	 * \p scale The scale for downsampling
	 */
	__global__ void upSampling(float *src, float *dst, int width, int height, unsigned char channel, int scale = 4)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		extern __shared__ float data[];
		//__shared__ float data[480 * 2 * 3];
		int w = width * channel;

		int data_index = col % width * channel;

		for (unsigned char k = 0; k < channel; k++)
		{
			// Get current point
			data[data_index + k] = src[row * w + col * channel + k];
			// Get the point at next row
			data[data_index + w + k] = src[MIN((row + 1), height - 1) * w + col * channel + k];
		}

		//__syncthreads();
		int row2, col2;
		float row3, col3;
		for (int i = 0; i < scale; ++i)
		{
			for (int j = 0; j < scale; ++j)
			{
				// The coordinate in output image
				row2 = scale * row + i;
				col2 = scale * col + j;

				row3 = row2 * (1.f * (height - 1) / (height * scale - 1));
				col3 = col2 * (1.f * (width - 1) / (width * scale - 1));

				int index_row = row - row3;
				int index_col = col - col3;

				int dst_index = row2 * w * scale + col2 * channel;
				int index_bound = MIN((col % width + 1), width - 1) * channel;

				for (unsigned char k = 0; k < channel; k++)
				{
					dst[dst_index + k] = data[data_index + k] * (index_row + 1)*(index_col + 1)
						+ data[data_index + w + k] * (-index_row)*(index_col + 1)
						+ data[index_bound + k] * (index_row + 1)*(-index_col)
						+ data[index_bound + w + k] * (index_row)*(index_col);
				}
			}
		}
	}
	__global__ void upSampling_C1(float *src, float *dst, int width, int height, unsigned char channel, int scale = 4)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		extern __shared__ float data[];
		//__shared__ float data[480 * 2 * 3];
		int w = width * channel;

		int data_index = col % width;

		// Get current point
		data[data_index] = src[row * w + col * channel];
		// Get the point at next row
		data[data_index + w] = src[MIN((row + 1), height - 1) * w + col * channel];
		
		//__syncthreads();
		int row2, col2;
		float row3, col3;
		for (int i = 0; i < scale; ++i)
		{
			for (int j = 0; j < scale; ++j)
			{
				// The coordinate in output image
				row2 = scale * row + i;
				col2 = scale * col + j;

				row3 = row2 * (1.f * (height - 1) / (height * scale - 1));
				col3 = col2 * (1.f * (width - 1) / (width * scale - 1));

				int index_row = row - row3;
				int index_col = col - col3;

				int dst_index = row2 * w * scale + col2 * channel;
				int index_bound = MIN((col % width + 1), width - 1) * channel;

				for (unsigned char k = 0; k < channel; k++)
				{
					dst[dst_index + k] = data[data_index + k] * (index_row + 1)*(index_col + 1)
						+ data[data_index + w + k] * (-index_row)*(index_col + 1)
						+ data[index_bound + k] * (index_row + 1)*(-index_col)
						+ data[index_bound + w + k] * (index_row)*(index_col);
				}
			}
		}
	}

	// boxFilter along x direction (top to down)
	__global__ void boxfilterX(float *src, float *dst, int radius, int width, int height, unsigned char channel)
	{
		unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
		// there will be channel * height threads
		if (thread_id >= channel * width)
			return;

		float coef = 1.0f / ((radius << 1) + 1);

		int index = thread_id / channel * width * channel + thread_id % channel;
		src = src + index;
		dst = dst + index;

		// do left edge
		float tmp = src[0] * radius;// / s;
		for (int c = 0; c < (radius + 1); ++c)
		{
			tmp += src[c * channel];// / s;
		}
		dst[0] = tmp * coef;

		for (int c = 1; c < (radius + 1); ++c)
		{
			tmp += src[(c + radius) * channel] - src[0];// / s;
			dst[c * channel] = tmp * coef;
		}

		// main loop
		for (int c = (radius + 1); c < (width - radius); ++c)
		{
			tmp += src[(c + radius) * channel] - src[(c - radius - 1) * channel];// / s;
			dst[c * channel] = tmp * coef;
		}

		// do right edge
		for (int c = width - radius; c < width; ++c)
		{
			tmp += src[(width - 1) * channel] - src[(c - radius - 1) * channel];// / s;
			dst[c * channel] = tmp * coef;
		}
	}

	// boxFilter along y direction (top to down)
	__global__ void boxfilterY(float *src, float *dst, int radius, int width, int height, unsigned char channel)
	{
		int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
		if (thread_id >= channel * width)
			return;

		src = src + thread_id;
		dst = dst + thread_id;

		int w = width * channel;
		float coef = 1.0f / ((radius << 1) + 1);

		// do left edge
		float tmp = src[0] * radius;
		for (int r = 0; r < (radius + 1); ++r)
		{
			tmp += src[r * w];
		}
		dst[0] = tmp * coef;

		for (int r = 1; r < (radius + 1); ++r)
		{
			tmp += src[(r + radius) * w] - src[0];
			dst[r * w] = tmp * coef;
		}

		// main loop
		for (int r = (radius + 1); r < (height - radius); ++r)
		{
			tmp += src[(r + radius) * w] - src[(r - radius - 1) * w];
			dst[r * w] = tmp * coef;
		}

		// do right edge
		for (int r = height - radius; r < height; ++r)
		{
			tmp += src[(height - 1) * w] - src[(r - radius - 1) * w];
			dst[r * w] = tmp * coef;
		}
	}

	// Matrix dot
	__global__ void dotMatrix(float *input1, float *input2, float* output, int width, int height, unsigned char channel)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		if (thread_id < width * height * channel)
		{
			output[thread_id] = input1[thread_id] * input2[thread_id];
		}
	}
	/* Equation:
	 * var(I,I) = var(I,p) = cov(I,p) = mean_II - meanI*meanI
	 * The first input a should be mean_II, while the second input should be mean_I
	 */
	__global__ void calc_ab(float *a, float *b, float eps, int width, int height, unsigned char channel)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		if (thread_id < width * height * channel)
		{
			a[thread_id] = 1 - eps / abs(a[thread_id] - b[thread_id] * b[thread_id] + eps);
			b[thread_id] = b[thread_id] - a[thread_id] * b[thread_id];
		}
	}
	
	// res = a .* src + b
	__global__ void calc_q_C1(float *src, float *a, float *b, float *res, int width, int height)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / width;
		int col = thread_id % width;

		if (col < width && row < height)
		{
			int index = row * width + col;

			float res_temp = a[thread_id] * float(src[thread_id]) + b[thread_id];
			res[thread_id] = (MIN(MAX(res_temp, 0), 1));
		}
	}
	template<typename T>
	__global__ void calc_q_C3C4(T *src, float *a, float *b, T *res, int width, int height, unsigned char channel)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int row = thread_id / width;
		int col = thread_id % width;

		if (col < width && row < height)
		{
			T temp = src[thread_id];

			int index = row * width + col;

			float res_temp_x = a[channel * thread_id] * float(temp.x) + b[channel * thread_id];
			temp.x = (MIN(MAX(res_temp_x, 0), 1));
			float res_temp_y = a[channel * thread_id + 1] * float(temp.y) + b[channel * thread_id + 1];
			temp.y = (MIN(MAX(res_temp_y, 0), 1));
			float res_temp_z = a[channel * thread_id + 2] * float(temp.z) + b[channel * thread_id + 2];
			temp.z = (MIN(MAX(res_temp_z, 0), 1));

			res[thread_id] = temp;
		}
	}

	void d_guidedFilter(float *p1, float *p2, float *p3, float *pa, float *pb, float eps, int scale, int scaled_radius, int scaled_width, int scaled_height, int channel, cudaStream_t stream)
	{
		//step2: box filter I
		::boxfilterX << < 51, 16, 0, stream >> > (p1, p2, scaled_radius, scaled_width, scaled_height, channel);
		::boxfilterY << < 45, 32, 0, stream >> > (p2, p3, scaled_radius, scaled_width, scaled_height, channel);  // p3 ---> mean_I mean_p
		//step3: cal mat I*I
		::dotMatrix << < dim3(81, 75), 64, 0, stream >> > (p1, p1, p1, scaled_width, scaled_height, channel);    // p1 ---> I*I
		//step4: box filter I*I
		::boxfilterX << < 51, 16, 0, stream >> > (p1, p2, scaled_radius, scaled_width, scaled_height, channel);
		::boxfilterY << < 45, 32, 0, stream >> > (p2, p1, scaled_radius, scaled_width, scaled_height, channel);  // p1 ---> corr_II corr_Ip
		//step5: cal a,b
		::calc_ab << < dim3(81, 75), 64, 0, stream >> > (p1, p3, eps, scaled_width, scaled_height, channel); // p1 ---> a, p3 ---> b
		//step6: box filter a,b
		::boxfilterX << < 51, 16, 0, stream >> > (p1, p2, scaled_radius, scaled_width, scaled_height, channel);
		::boxfilterY << < 45, 32, 0, stream >> > (p2, p1, scaled_radius, scaled_width, scaled_height, channel);  // p1 ---> mean_a
		::boxfilterX << < 51, 16, 0, stream >> > (p3, p2, scaled_radius, scaled_width, scaled_height, channel);
		::boxfilterY << < 45, 32, 0, stream >> > (p2, p3, scaled_radius, scaled_width, scaled_height, channel);  // p3 ---> mean_b
		//step7: interpolation a,b
		::upSampling << < dim3(1, 270), dim3(480, 1), 2*scaled_width*channel*sizeof(float), stream >> > (p1, pa, scaled_width, scaled_height, channel, scale);
		::upSampling << < dim3(1, 270), dim3(480, 1), 2*scaled_width*channel*sizeof(float), stream >> > (p3, pb, scaled_width, scaled_height, channel, scale);
	}
}

GPU_ALGO_BEGIN
void guidedFilter(cv::cuda::GpuMat &src,  float eps, int radius, int scale, cudaStream_t stream, std::array<cv::cuda::GpuMat, 5> &tmp_mat)
{
    int height = src.rows;
	int width = src.cols;
	int scaled_width = width / scale;
	int scaled_height = height / scale;
	int scaled_radius = radius / scale;

	float* p1 = tmp_mat[0].ptr<float>(0);
	float* p2 = tmp_mat[1].ptr<float>(0);
	float* p3 = tmp_mat[2].ptr<float>(0);

	float* pa = tmp_mat[3].ptr<float>(0);
	float* pb = tmp_mat[4].ptr<float>(0);

	// For current version, we consider RGBA image as RGB image there.
	int C = src.channels();
	int channel = MIN(3, src.channels());

	if (C == 1) {
		float* psrc = src.ptr<float>(0);
		::downSampling_C1 << < dim3(45, 45), 64, 0, stream >> > (psrc, p1, scale, width, height);  // p1 ---> I
		::d_guidedFilter(p1, p2, p3, pa, pb, eps, scale, scaled_radius, scaled_width, scaled_height, channel, stream);
		::calc_q_C1 << < dim3(90, 90), 256, 0, stream >> > (psrc, pa, pb, psrc, width, height);
	}
	else if (C == 3) {
		float3* psrc = src.ptr<float3>(0);
		::downSampling_C3C4 << < dim3(45, 45), 64, 0, stream >> > (psrc, p1, scale, width, height, channel);  // p1 ---> I
		::d_guidedFilter(p1, p2, p3, pa, pb, eps, scale, scaled_radius, scaled_width, scaled_height, channel, stream);
		::calc_q_C3C4 << < dim3(90, 90), 256, 0, stream >> > (psrc, pa, pb, psrc, width, height, channel);
	}
	else {
		float4* psrc = src.ptr<float4>(0);
		::downSampling_C3C4 << < dim3(45, 45), 64, 0, stream >> > (psrc, p1, scale, width, height, channel);  // p1 ---> I
		::d_guidedFilter(p1, p2, p3, pa, pb, eps, scale, scaled_radius, scaled_width, scaled_height, channel, stream);
		::calc_q_C3C4 << < dim3(90, 90), 256, 0, stream >> > (psrc, pa, pb, psrc, width, height, channel);
	}
}

GPU_ALGO_END
#endif // ALGONODEGUIDEDFILTER_CU
