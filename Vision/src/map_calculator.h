#ifndef MAP_CALCULATOR_H
#define MAP_CALCULATOR_H
#include <opencv2/opencv.hpp>
#include "camera_parameters.h"


/** @brief Create the map tor rectify the image
  This class is designed as a virtual class
*/
class MapCalculator
{
public:
    MapCalculator(const CameraParameters& cam_params, uint16_t image_width, uint16_t image_height);

    ~MapCalculator() {}

    /** @brief Update the map for rectification.
    */
    void updateMap(int disparity);

    const cv::cuda::GpuMat& getGPUMapx() const { return gpu_mapx; }
    const cv::cuda::GpuMat& getGPUMapy() const { return gpu_mapy; }
    const cv::cuda::GpuMat& getGPUMapx2X() const { return gpu_mapx_2X; }
    const cv::cuda::GpuMat& getGPUMapy2X() const { return gpu_mapy_2X; }
    const cv::cuda::GpuMat& getGPUMapx4X() const { return gpu_mapx_4X; }
    const cv::cuda::GpuMat& getGPUMapy4X() const { return gpu_mapy_4X; }

    const cv::Mat& getCPUMapx() const { return cpu_mapx; }
    const cv::Mat& getCPUMapy() const { return cpu_mapy; }
    const cv::Mat& getCPUMapx2X() const { return cpu_mapx_2X; }
    const cv::Mat& getCPUMapy2X() const { return cpu_mapy_2X; }
    const cv::Mat& getCPUMapx4X() const { return cpu_mapx_4X; }
    const cv::Mat& getCPUMapy4X() const { return cpu_mapy_4X; }

protected:
    /** @brief Initialize all the Mat
    */
    void initMat();

    /** @brief Calculate the image map.
      the map include map(x&y) in both 2D and 3D, however 3D map is same as 2D actually.
    */
    void calcImageMap(const CameraParameters& cam_params, cv::Mat& mapx, cv::Mat& mapy);

protected:
    const CameraParameters  camera_params;
    cv::Mat					double_mapx;	//!< the map in x dimension
    cv::Mat					double_mapy;	//!< the map in y dimension

    uint16_t    image_width;
    uint16_t    image_height;

    cv::cuda::GpuMat		gpu_mapx;		//!< the roi region in 'm_double_mapx'
    cv::cuda::GpuMat		gpu_mapy;		//!< the roi region in 'm_double_mapy'
    cv::cuda::GpuMat        gpu_mapx_2X;	//!< zoomed roi region from 'm_mapx'
    cv::cuda::GpuMat        gpu_mapy_2X;	//!< zoomed roi region from 'm_mapy'
    cv::cuda::GpuMat        gpu_mapx_4X;	//!< zoomed roi region from 'm_mapx'
    cv::cuda::GpuMat        gpu_mapy_4X;	//!< zoomed roi region from 'm_mapy'

    cv::Mat		cpu_mapx;			//!< the roi region in 'm_double_mapx'
    cv::Mat		cpu_mapy;			//!< the roi region in 'm_double_mapy'
    cv::Mat     cpu_mapx_2X;		//!< zoomed roi region from 'm_mapx'
    cv::Mat     cpu_mapy_2X;		//!< zoomed roi region from 'm_mapy'
    cv::Mat     cpu_mapx_4X;		//!< zoomed roi region from 'm_mapx'
    cv::Mat     cpu_mapy_4X;		//!< zoomed roi region from 'm_mapy'
};

#endif // MAP_CALCULATOR_H
