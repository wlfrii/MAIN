#ifndef _MATH_MATRIX_OPERATION_H_
#define _MATH_MATRIX_OPERATION_H_
#include <Eigen/Dense>
#include <type_traits>
#include <cmath>

namespace math
{
	/** @brief Return a rotation matrix with rotated by x-axis by radian.
	 */
	template<typename T = float, typename T1 = double>
	Eigen::Matrix<T, 3, 3> rotByX(const T1 radian)
	{
		if (!std::is_arithmetic<T1>::value)
			std::abort();

		Eigen::Matrix<T, 3, 3> rot;
		rot << static_cast<T>(1), static_cast<T>(0), static_cast<T>(0),
			static_cast<T>(0), static_cast<T>(cos(radian)), static_cast<T>(-sin(radian)),
			static_cast<T>(0), static_cast<T>(sin(radian)), static_cast<T>(cos(radian));
		return rot;
	}

	/** @brief Return a rotation matrix with rotated by y-axis by radian.
	 */
	template<typename T = float, typename T1 = double>
	Eigen::Matrix<T, 3, 3> rotByY(const T1 radian)
	{
		if (!std::is_arithmetic<T1>::value)
			std::abort();

		Eigen::Matrix<T, 3, 3> rot;
		rot << static_cast<T>(cos(radian)), static_cast<T>(0), static_cast<T>(sin(radian)),
			static_cast<T>(0), static_cast<T>(1), static_cast<T>(0),
			static_cast<T>(-sin(radian)), static_cast<T>(0), static_cast<T>(cos(radian));
		return rot;
	}


	/** @biref Return a rotation matrix with rotated by z-axis by radian.
	 */
	template<typename T = float, typename T1 = double>
	Eigen::Matrix<T, 3, 3> rotByZ(const T1 radian)
	{
		if (!std::is_arithmetic<T1>::value)
			std::abort();

		Eigen::Matrix<T, 3, 3> rot;
		rot << static_cast<T>(cos(radian)), static_cast<T>(-sin(radian)), static_cast<T>(0),
			static_cast<T>(sin(radian)), static_cast<T>(cos(radian)), static_cast<T>(0),
			static_cast<T>(0), static_cast<T>(0), static_cast<T>(1);
		return rot;
	}


	/** @brief Return the absoluted Matrix.
	* M  represents the row of the matrix.
	* N  represents the column of the matrix.
	*/
	template<typename T, int M, int N>
	Eigen::Matrix<T, M, N> absMat(const Eigen::Matrix<T, M, N>& mat)
	{
		Eigen::Matrix<T, M, N> abs_mat = mat;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				abs_mat(i, j) = std::abs(abs_mat(i, j));
			}
		}
		return abs_mat;
	}


	/** Return the skew-symmetric matrix based on the input vector.
	*/
	Eigen::Matrix3f skewSymmetric(Eigen::Vector3f &vec);
	Eigen::Matrix3f skewSymmetric(Eigen::RowVector3f &vec);
}

#endif // _MATH_MATRIX_OPERATION_H_