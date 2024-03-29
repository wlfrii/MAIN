#include "camera_parameters.h"

// matrix_name  expected_type  width  height
#define ASSERT_MATRIX_SIZE_TYPE(matrix, width, height)   \
    assert(matrix.size() == cv::Size(width, height));

CameraParameters::CameraParameters(const cv::Mat &intrinsic, const cv::Mat &dist_coeffs,
    const cv::Mat &R_rectify, const cv::Rect &roi, const cv::Mat &new_intrinsic)
    : A(intrinsic)
    , D(dist_coeffs)
    , R(R_rectify)
    , ROI(roi)
    , Anew(new_intrinsic)
{
    ASSERT_MATRIX_SIZE_TYPE(intrinsic, 3, 3);
    ASSERT_MATRIX_SIZE_TYPE(dist_coeffs, 1, 5);
    ASSERT_MATRIX_SIZE_TYPE(new_intrinsic, 3, 3);
    ASSERT_MATRIX_SIZE_TYPE(R_rectify, 3, 3);
    assert(roi.width > 0 && roi.height > 0);
}

CameraParameters::CameraParameters(const CameraParameters& rhs)
{
    A = rhs.A;
    D = rhs.D;
    R = rhs.R;
    ROI = rhs.ROI;
    Anew = rhs.Anew;
}

#if LINUX
CameraParameters::CameraParameters(USBCameraParammeters &params)
    : A(params.A.to_cvMat())
    , D(params.D.to_cvMat())
    , R(params.R.to_cvMat())
    , ROI(params.ROI.to_cvRect())
    , Anew(params.Anew.to_cvMat())
{
}
#endif

const float& CameraParameters::getFX() const
{
    return A.at<float>(0, 0);
}
const float& CameraParameters::getFY() const
{
    return A.at<float>(1, 1);
}
const float& CameraParameters::getCX() const
{
    return A.at<float>(0, 2);
}
const float& CameraParameters::getCY() const
{
    return A.at<float>(1, 2);
}
const float& CameraParameters::getNewFX() const
{
    return Anew.at<float>(0, 0);
}
const float& CameraParameters::getNewFY() const
{
    return Anew.at<float>(1, 1);
}
const float& CameraParameters::getNewCX() const
{
    return Anew.at<float>(0, 2);
}
const float& CameraParameters::getNewCY() const
{
    return Anew.at<float>(1, 2);
}
const float& CameraParameters::getK1() const
{
    return D.at<float>(0, 0);
}
const float& CameraParameters::getK2() const
{
    return D.at<float>(1, 0);
}
const float& CameraParameters::getK3() const
{
    return D.at<float>(2, 0);
}
const float& CameraParameters::getP1() const
{
    return D.at<float>(3, 0);
}
const float& CameraParameters::getP2() const
{
    return D.at<float>(4, 0);
}
const cv::Mat& CameraParameters::intrinsic() const
{
    return A;
}
const cv::Mat& CameraParameters::distCoeffs() const
{
    return D;
}
const cv::Mat& CameraParameters::rectifyMat() const
{
    return R;
}
const cv::Mat& CameraParameters::newIntrinsic() const
{
    return Anew;
}
const cv::Rect& CameraParameters::roi() const
{
    return ROI;
}


CameraParamsReader::CameraParamsReader(const std::string &cam_params_path)
    : fs(cv::FileStorage(cam_params_path, cv::FileStorage::READ))
{
    if (!fs.isOpened())
    {
        printf("CameraParamsReader: Cannot open the camera parameters file!\n");
        fs.release();
        is_valid_path = false;
    }
    is_valid_path = true;
}

CameraParamsReader::~CameraParamsReader()
{
    fs.release();
}

#define GET_DATA_BY_NAME(file, variable, index)				\
{															\
	std::string tmp = #variable + std::to_string(index);	\
    file[tmp] >> variable;									\
}

std::shared_ptr<CameraParameters> CameraParamsReader::getCameraParameters(vision::StereoCameraID index /* = vision::LEFT_CAMERA */) const
{
    if (!is_valid_path)
        return nullptr;

    cv::Mat A;
    cv::Mat D;
    cv::Mat R;
    cv::Rect roi;
    cv::Mat Anew;
    int camera_index = (index == vision::LEFT_CAMERA ? 1 : 2);
    GET_DATA_BY_NAME(fs, A, camera_index);
    GET_DATA_BY_NAME(fs, D, camera_index);
    GET_DATA_BY_NAME(fs, R, camera_index);
    GET_DATA_BY_NAME(fs, roi, camera_index);

    if (D.rows == 1)
        D = D.t();
    fs["Anew"] >> Anew;
    if (Anew.empty())
        Anew = A;
    if (roi.empty())
        roi = cv::Rect(0, 0, 1, 1);

	auto check = [](cv::Mat& mat) {
		if (mat.type() != CV_32F) {
			cv::Mat tmp(mat.size(), CV_32F);
			mat.convertTo(tmp, CV_32F);
			mat = tmp;
		}
	}; 
	check(A);
	check(D);
	check(R);
	check(Anew);

    return std::make_shared<CameraParameters>(A, D, R, roi, Anew);
}

std::shared_ptr<StereoCameraParameters> CameraParamsReader::getStereoCameraParameters() const
{
	if (!is_valid_path)
		return nullptr;

    std::shared_ptr<CameraParameters> params[2];
    for(auto i = 0; i < 2;i++)
    {
        params[i] = getCameraParameters(vision::StereoCameraID(i));
    }
    return std::make_shared<StereoCameraParameters>(params[0], params[1]);
}

void CameraParamsReader::getImageSize(int &width, int &height) const
{
	if (!is_valid_path)
        return;

    fs["image_width"] >> width;
    fs["image_height"] >> height;
}

