#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "def/define.h"
#include "def/micro.h"
#if LINUX
#include "usb/usb_camera_parameters.h"
#endif


/** @brief Store the parameters for monocular.
*/
class CameraParameters
{
public:
    CameraParameters() = default;

    CameraParameters(const cv::Mat &intrinsic, const cv::Mat &distortion,
        const cv::Mat &rectification, const cv::Rect &roi, const cv::Mat &new_intrinsic);
    CameraParameters(const CameraParameters& rhs);
#if LINUX
    CameraParameters(USBCameraParammeters &params);
#endif

    const float& getFX() const;	//!< the focal length in x direction
    const float& getFY() const;	//!< the focal length in y direction
    const float& getCX() const;	//!< the x position of camera optical center in the image
    const float& getCY() const;	//!< the y position of camera optical center in the image
    const float& getNewFX() const;	//!< the new focal length in x direction
    const float& getNewFY() const;	//!< the new focal length in y direction
    const float& getNewCX() const;	//!< the new x position of camera optical center in the image
    const float& getNewCY() const;  //!< the new y position of camera optical center in the image
    const float& getK1() const;     //!< radial distortion k1
    const float& getK2() const;     //!< radial distortion k2
    const float& getK3() const;     //!< radial distortion k3
    const float& getP1() const;     //!< tangentia distortion p1
    const float& getP2() const;     //!< tangentia distortion p2

    const cv::Mat& intrinsic() const;       //!< the intrinsic mat 3x3
    const cv::Mat& distCoeffs() const;      //!< the dist_coeffs mat 1x5
    const cv::Mat& rectifyMat() const;      //!< the rectification mat 3x3
    const cv::Mat& newIntrinsic() const;    //!< the new_intrinsic mat 3x3
    const cv::Rect& roi() const;            //!< the roi in the image map [x,y,width,height]

private:
    cv::Mat         A;                      //!< intrinsic parameters 3x3
    cv::Mat         D;                      //!< distortion parameters 1x5
    cv::Mat         R;                      //!< rectification parameters 3x3
    cv::Rect        ROI;                    //!< region of interest [x,y,width,height]
    cv::Mat         Anew;                   //!< new intrinsic parameters 3x3
};



/** @brief Store the parameters for monocular.
*/
class StereoCameraParameters
{
public:
    StereoCameraParameters(){}
    StereoCameraParameters(std::shared_ptr<CameraParameters> left, std::shared_ptr<CameraParameters> right)
        : left(left), right(right)
    {}
	std::shared_ptr<CameraParameters> left;
	std::shared_ptr<CameraParameters> right;
};


/** @brief The tool for read parameters of monocular or binocular.
*/
class CameraParamsReader
{
public:
    CameraParamsReader(const std::string &cam_params_path);
    ~CameraParamsReader();

    //static CameraParamsReader* getInstance();

	/** Set camera parameters path, return true means the path is valid. */
    //bool setParamsPath(const std::string &filename);

    std::shared_ptr<CameraParameters> getCameraParameters(vision::StereoCameraID index = vision::LEFT_CAMERA) const;
    std::shared_ptr<StereoCameraParameters> getStereoCameraParameters() const;

	/* these fucntions must be called */
    void	getImageSize(int &width, int &height) const;

private:
	/* warning!!!, this initialization must be done in the intializer list,
	  or there maybe exist a situation that fs is initialized twice */
    cv::FileStorage fs;

    bool is_valid_path;
    //std::string path;
};

