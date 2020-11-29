#include "algo_node_nonuniform.h"
#include "../def/cu_define.h"


namespace
{
	/*function uneven_y = lumalaws(width, height, distance, I)
		% calculate the distance between pixel point and image center
		[X, Y] = meshgrid(1:width, 1 : height);
		radius_square = (X - floor(width / 2)). ^ 2 + (Y - floor(height / 2)). ^ 2;
		% calculate the angle
		theta = atan2(sqrt(radius_square), distance);
		% calculate L
		L_square = radius_square + distance ^ 2;
		% calculate intensity of pixel
		uneven_y = I * cos(theta). / L_square;

		% remap the y_image
		max_y = max(max(uneven_y));
		min_y = min(min(uneven_y));
		uneven_y = (uneven_y - min_y). / (max_y - min_y);
	end*/
	__device__ float lumalaws2(int col, int row, int width, int height, int distance)
	{
		__shared__ float tmp[3];
		tmp[0] = 1.f / (distance*distance); // max_y
		tmp[1] = (width / 2.f)*(width / 2.f) + (height / 2.f)*(height / 2.f); // radius_square_max
		tmp[2] = cosf(atan2f(sqrtf(tmp[1]), distance)) / (tmp[1] + distance * distance); // min_y

		float radius_square = (col - width / 2)*(col - width / 2) + (row - height / 2)*(row - height / 2);
		float theta = atan2f(sqrtf(radius_square), distance);
		float uneven_y = cosf(theta) / (radius_square + distance * distance);

		// Projecting the luminance value to [0,1]
		uneven_y = (uneven_y - tmp[2]) / (tmp[0] - tmp[2]);
		return uneven_y;
	}

	__constant__ float NORM_2 = 1.f / 2.f;
	__global__ void createNonuniformY(int distance, cv::cuda::PtrStepSz<float> uneven_y)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % uneven_y.cols;
		int row = thread_id / uneven_y.cols;

		if (col < uneven_y.cols && row < uneven_y.rows)
		{
			uneven_y(row, col) = lumalaws2(col + 1, row + 1, uneven_y.cols, uneven_y.rows, distance);
		}
	}

	//__global__ void reduceUnevenY(cv::cuda::PtrStepSz<uchar4> src, cv::cuda::PtrStepSz<uchar3> hsv, cv::cuda::PtrStepSz<float> uneven_y, float magnify, float magniy0)
	//{
	//	int thread_id = _get_threadId_grid2D_block1D();
	//	int col = thread_id % uneven_y.cols;
	//	int row = thread_id / uneven_y.cols;

	//	if (col < uneven_y.cols && row < uneven_y.rows)
	//	{
	//		float p = magnify * (1 - uneven_y(row, col)) * float(hsv(row, col).y) / 255.f;

	//		src(row, col).x = (uchar)MIN(float(src(row, col).x)*(magniy0 + p), 255);
	//		src(row, col).y = (uchar)MIN(float(src(row, col).y)*(magniy0 + p), 255);
	//		src(row, col).z = (uchar)MIN(float(src(row, col).z)*(magniy0 + p), 255);
	//	}
	//}

	__global__ void reduceNonuniformY(cv::cuda::PtrStepSz<float4> src, cv::cuda::PtrStepSz<float> uneven_y, float magnify, float magnify0)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int col = thread_id % uneven_y.cols;
		int row = thread_id / uneven_y.cols;

		if (col < uneven_y.cols && row < uneven_y.rows)
		{
			// Calcuate the pixel saturation first
			float max_rgb = MAX(MAX(src(row, col).x, src(row, col).y), src(row, col).z);
			float min_rgb = MIN(MIN(src(row, col).x, src(row, col).y), src(row, col).z);
			float saturation = abs(max_rgb) < 1e-3 ? 0 : 1 - min_rgb / max_rgb;

			float p = magnify * (1 - uneven_y(row, col)) * saturation;

			src(row, col).x = src(row, col).x * (magnify0 + p);
			src(row, col).y = src(row, col).y * (magnify0 + p);
			src(row, col).z = src(row, col).z * (magnify0 + p);
		}
	}
}

GPU_ALGO_BEGIN
void createNonuniformY(int width, int height, int distance, cv::cuda::GpuMat & uneven_y, cudaStream_t stream)
{
	if (uneven_y.empty())
		uneven_y = cv::cuda::GpuMat(height, width, CV_32FC1);

	::createNonuniformY << < dim3(90, 90), 256, 0, stream >> > (distance, uneven_y);
}

void reduceNonuniformY(cv::cuda::GpuMat & src, cv::cuda::GpuMat & uneven_y, float magnify, float magnify0, cudaStream_t stream)
{
	::reduceNonuniformY << < dim3(90, 90), 256, 0, stream >> > (src, uneven_y, magnify, magnify0);

#if CU_DEBUG
	cv::Mat test_uneven_y;	uneven_y.download(test_uneven_y);
	cv::Mat test_res; src.download(test_res);
#endif
}
GPU_ALGO_END

namespace
{

	__global__ void calcUnevenY(float *Y, int width, int height, int distance)
	{
		int thread_id = _get_threadId_grid2D_block1D();

		int row = thread_id / width;
		int col = thread_id % width;
		if (row < width && col < height)
		{
			float radius_square = (col - width / 2.f) * (col - width / 2.f) + (row - height / 2.f) * (row - height / 2.f);
			float theta = atan2f(sqrtf(radius_square), distance);
			float L_square = radius_square + distance * distance;
			Y[thread_id] = 1 * cosf(theta) / L_square;
		}
	}

	__global__ void reduceUevenY_UUC3(cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> dst, int degree = 2, int distance = 1000)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int i = thread_id % src.cols;
		int j = thread_id / src.cols;

		if (i < src.cols && j < src.rows)
		{
			float max_y = 1.f / (distance*distance);
			float radius_square_max = (src.cols / 2.f)*(src.cols / 2.f) + (src.rows / 2.f)*(src.rows / 2.f);
			float min_y = cosf(atan2f(sqrtf(radius_square_max), distance)) / (radius_square_max + distance * distance);

			float y = 0.2126*src(j, i).z / 255.f + 0.7152*src(j, i).y / 255.f + 0.0722 *src(j, i).x / 255.f;
			float u = -0.09991*src(j, i).z / 255.f - 0.33609*src(j, i).y / 255.f + 0.436 *src(j, i).x / 255.f;
			float v = 0.615*src(j, i).z / 255.f - 0.55861*src(j, i).y / 255.f - 0.05639 *src(j, i).x / 255.f;

			float radius_square = (i - src.cols / 2)*(i - src.cols / 2) + (j - src.rows / 2)*(j - src.rows / 2);
			float theta = atan2f(sqrtf(radius_square), distance);
			float uneven_y = cosf(theta) / (radius_square + distance * distance);
			uneven_y = (uneven_y - min_y) / (max_y - min_y);

			float y_new = ((1.f - degree)*uneven_y + degree) * y;
			y_new = y_new > 1.f ? 1.f : y_new;

			float R = (1.f * y_new + 1.28033*v) * 255.f;
			float G = (1.f * y_new - 0.21482*u - 0.38059*v) * 255.f;
			float B = (1.f * y_new + 2.12798*u) * 255.f;
			dst(j, i).x = uchar(B > 255 ? 255 : B);
			dst(j, i).y = uchar(G > 255 ? 255 : G);
			dst(j, i).z = uchar(R > 255 ? 255 : R);
		}
	}
	__global__ void reduceUevenY_FFC1(cv::cuda::PtrStepSz<float1> src, cv::cuda::PtrStepSz<float1> dst, int degree = 2, int distance = 1000)
	{
		int thread_id = _get_threadId_grid2D_block1D();
		int i = thread_id % src.cols;
		int j = thread_id / src.cols;

		if (i < src.cols && j < src.rows)
		{
			float max_y = 1.f / (distance*distance);
			float radius_square_max = (src.cols / 2.f)*(src.cols / 2.f) + (src.rows / 2.f)*(src.rows / 2.f);
			float min_y = cosf(atan2f(sqrtf(radius_square_max), distance)) / (radius_square_max + distance * distance);

			float radius_square = (i - src.cols / 2)*(i - src.cols / 2) + (j - src.rows / 2)*(j - src.rows / 2);
			float theta = atan2f(sqrtf(radius_square), distance);
			float uneven_y = cosf(theta) / (radius_square + distance * distance);
			uneven_y = (uneven_y - min_y) / (max_y - min_y);

			float y_new = ((1.f - degree)*uneven_y + degree) * src(j, i).x;
			// NOTE: the luminance value should not be limited here, but should be limited before convert th RGB
			//y_new = y_new > 1.f ? 1.f : y_new;
			//sy_new = y_new < 0 ? 0 : y_new;
			dst(j, i).x = y_new;
		}
	}

}