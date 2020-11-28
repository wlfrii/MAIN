#include "matrix_operation.h"

namespace math
{
	Eigen::Matrix3f skewSymmetric(Eigen::Vector3f &vec)
	{
		Eigen::Matrix3f mat;
		mat << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
		return mat;
	}
	Eigen::Matrix3f skewSymmetric(Eigen::RowVector3f &vec)
	{
		Eigen::Matrix3f mat;
		mat << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
		return mat;
	}
}