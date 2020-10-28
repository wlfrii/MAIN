#ifndef ALGONODEIMAGEADJUST_CU
#define ALGONODEIMAGEADJUST_CU
#include "algo_node_image_adjust.h"
#include "../def/cu_define.h"

//kernel func
namespace
{
    __global__ void adjustImageProp_RGB(gpu::U8 *src, float saturation, float constrast, float brightness)
    {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        int threadId = blockId * blockDim.x + threadIdx.x;

        if(threadId >= 2073600)
            return;
        int bgr[3];

        //saturation
        int row = threadId / 1920;
        int col = threadId % 1920;
        int sub_index = row * 5760 + col * 3;
        bgr[0] = src[sub_index ];
        bgr[1] = src[sub_index  + 1];
        bgr[2] = src[sub_index  + 2];

        do
        {
            int rgb_max = MAX(bgr[0], MAX(bgr[1], bgr[2]));
            int rgb_min = MIN(bgr[0], MIN(bgr[1], bgr[2]));
            int delta = rgb_max - rgb_min;
            if (delta == 0)
            {
                break;
            }
            int illuminance = rgb_max + rgb_min;
            float s;
            if (illuminance < 255)
                s = delta * 1.f / illuminance;
            else
                s = delta / (510.f - illuminance);

            float alpha = 0.f;
            float tmp = 0.f;
            if (saturation >= 0)
            {
                alpha = MAX(s, 1.f - saturation);
                alpha = 1.f / alpha - 1.f;
                tmp = (illuminance >> 1) * alpha;
                alpha = alpha + 1;
            }
            else
            {
                alpha = saturation + 1;
                tmp = (illuminance >> 1) * saturation;
            }
            bgr[0] = alpha * src[sub_index] - tmp;
            bgr[1] = alpha * src[sub_index + 1] - tmp;
            bgr[2] = alpha * src[sub_index + 2] - tmp;
            bgr[0] = MAX(MIN(bgr[0], 255), 0);
            bgr[1] = MAX(MIN(bgr[1], 255), 0);
            bgr[2] = MAX(MIN(bgr[2], 255), 0);
        } while(0);

        //brightness
        //brightness = brightness * 100;
        do
        {
            int rgb_max = MAX(bgr[0], MAX(bgr[1], bgr[2]));
            int rgb_min = MIN(bgr[0], MIN(bgr[1], bgr[2]));
            int illuminance = (rgb_max + rgb_min) >> 1;
            if (illuminance == 0){
                break;
            }

            int temp1, temp2;
            if (illuminance > 128)
            {
                temp1 = (illuminance - 128) * 256;
                temp2 = (256 - illuminance);
                bgr[0] = (bgr[0] * 128 - temp1) / temp2;
                bgr[1] = (bgr[1] * 128 - temp1) / temp2;
                bgr[2] = (bgr[2] * 128 - temp1) / temp2;
            }
            else
            {
                bgr[0] = bgr[0] * 128 / illuminance;
                bgr[1] = bgr[1] * 128 / illuminance;
                bgr[2] = bgr[2] * 128 / illuminance;
            }

            int illuminance_new = illuminance + brightness - 128;
            float temp3 = illuminance_new / 128.f;
            if (illuminance_new > 0)
            {
                bgr[0] = bgr[0] + (256 - bgr[0]) * temp3;
                bgr[1] = bgr[1] + (256 - bgr[1]) * temp3;
                bgr[2] = bgr[2] + (256 - bgr[2]) * temp3;
            }
            else
            {
                bgr[0] = bgr[0] * (1 + temp3);
                bgr[1] = bgr[1] * (1 + temp3);
                bgr[2] = bgr[2] * (1 + temp3);
            }

            bgr[0] = MAX(MIN(bgr[0], 255), 0);
            bgr[1] = MAX(MIN(bgr[1], 255), 0);
            bgr[2] = MAX(MIN(bgr[2], 255), 0);
        } while(0);

        // constrast

        do
        {
            bgr[0] = (bgr[0] - 128) * constrast + 128;
            bgr[1] = (bgr[1] - 128) * constrast + 128;
            bgr[2] = (bgr[2] - 128) * constrast + 128;

        } while(0);


        src[sub_index] = (uchar)MAX(MIN(bgr[0], 255), 0);
        src[sub_index + 1] = (uchar)MAX(MIN(bgr[1], 255), 0);
        src[sub_index + 2] = (uchar)MAX(MIN(bgr[2], 255), 0);
    }


    __global__ void adjustImageProp_RGBA(gpu::U8C4 *src, float saturation, float constrast, float brightness)
    {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        int threadId = blockId * blockDim.x + threadIdx.x;

        if(threadId >= 2073600)
            return;

        gpu::U8C4 tempBGR = src[threadId];
        int bgr[3];
        bgr[0] = tempBGR.x;
        bgr[1] = tempBGR.y;
        bgr[2] = tempBGR.z;

        do
        {
            int rgb_max = MAX(bgr[0], MAX(bgr[1], bgr[2]));
            int rgb_min = MIN(bgr[0], MIN(bgr[1], bgr[2]));
            int delta = rgb_max - rgb_min;
            if (delta == 0)
            {
                break;
            }
            int illuminance = rgb_max + rgb_min;
            float s;
            if (illuminance < 255)
                s = delta * 1.f / illuminance;
            else
                s = delta / (510.f - illuminance);

            float alpha = 0.f;
            float tmp = 0.f;
            if (saturation >= 0)
            {
                alpha = MAX(s, 1.f - saturation);
                alpha = 1.f / alpha - 1.f;
                tmp = (illuminance >> 1) * alpha;
                alpha = alpha + 1;
            }
            else
            {
                alpha = saturation + 1;
                tmp = (illuminance >> 1) * saturation;
            }
            bgr[0] = alpha * tempBGR.x - tmp;
            bgr[1] = alpha * tempBGR.y - tmp;
            bgr[2] = alpha * tempBGR.z - tmp;
        } while(0);

        //brightness
        do
        {
            int rgb_max = MAX(bgr[0], MAX(bgr[1], bgr[2]));
            int rgb_min = MIN(bgr[0], MIN(bgr[1], bgr[2]));
            int illuminance = (rgb_max + rgb_min) >> 1;
            if (illuminance == 0){
                break;
            }

            int temp1, temp2;
            if (illuminance > 128)
            {
                temp1 = (illuminance - 128) * 256;
                temp2 = (256 - illuminance);
                bgr[0] = (bgr[0] * 128 - temp1) / temp2;
                bgr[1] = (bgr[1] * 128 - temp1) / temp2;
                bgr[2] = (bgr[2] * 128 - temp1) / temp2;
            }
            else
            {
                bgr[0] = bgr[0] * 128 / illuminance;
                bgr[1] = bgr[1] * 128 / illuminance;
                bgr[2] = bgr[2] * 128 / illuminance;
            }

            int illuminance_new = illuminance + brightness - 128;
            float temp3 = illuminance_new / 128.f;
            if (illuminance_new > 0)
            {
                bgr[0] = bgr[0] + (256 - bgr[0]) * temp3;
                bgr[1] = bgr[1] + (256 - bgr[1]) * temp3;
                bgr[2] = bgr[2] + (256 - bgr[2]) * temp3;
            }
            else
            {
                bgr[0] = bgr[0] * (1 + temp3);
                bgr[1] = bgr[1] * (1 + temp3);
                bgr[2] = bgr[2] * (1 + temp3);
            }

        } while(0);

        // constrast
        do
        {
            bgr[0] = (bgr[0] - 128) * constrast + 128;
            bgr[1] = (bgr[1] - 128) * constrast + 128;
            bgr[2] = (bgr[2] - 128) * constrast + 128;

        } while(0);


        tempBGR.x = MAX(MIN(bgr[0], 255), 0);
        tempBGR.y = MAX(MIN(bgr[1], 255), 0);
        tempBGR.z = MAX(MIN(bgr[2], 255), 0);

        src[threadId] = tempBGR;
    }
}

GPU_ALGO_BEGIN
void AdjustImageProp_RGB(cv::cuda::GpuMat &src, float saturation, float contrast, float brightness, cudaStream_t stream)
{
    U8* psrc = src.ptr<U8>(0);
    ::adjustImageProp_RGB << <dim3(30, 270), 256,0,stream >> >(psrc, saturation, contrast, brightness);
}

void AdjustImageProp_RGBA(cv::cuda::GpuMat &src, float saturation, float contrast, float brightness, cudaStream_t stream)
{
    U8C4* psrc = src.ptr<U8C4>(0);
    ::adjustImageProp_RGBA << <dim3(30, 270), 256,0,stream >> >(psrc, saturation, contrast, brightness);
}
GPU_ALGO_END

#endif // ALGONODEIMAGEADJUST_CU
