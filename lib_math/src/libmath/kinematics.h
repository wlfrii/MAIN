#ifndef _MATH_KINEMATICS_H_
#define _MATH_KINEMATICS_H_
#include <Eigen/Dense>

namespace math
{
	class RT
	{
	public:
		RT()
			: R(Eigen::Matrix3f::Identity())
			, t(0, 0, 0)
		{}
		explicit RT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t)
			: R(R)
			, t(t)
		{}

		RT inverse()
		{
			RT inv;
			inv.R = R.transpose();
			inv.t = -R.transpose() * t;
			return inv;
		}

		Eigen::Matrix3f R;
		Eigen::Vector3f t;
	};


	/** @brief Calculation the rotation of the end frame of single continuum segment
	* with respect to its base frame.
	* @param L  The length of the segment.
	* @param theta  The bending angle of the segment.
	* @param delta  The bending direction of the segment.
	* @return The rotaion matrix of single continuum segment with respect to its
	* base frame.
	*/
	Eigen::Matrix3f getSingleSegmentR(int L, int theta, int delta);


	/** @brief Calculation the position of the end frame of single continuum segment
	* with respect to its base frame.
	* @param L  The length of the segment.
	* @param theta  The bending angle of the segment.
	* @param delta  The bending direction of the segment.
	* @return The position matrix of single continuum segment with respect to its
	* base frame.
	*/
	Eigen::Vector3f getSingleSegmentP(int L, int theta, int delta);


	/** @brief Calculation the rotation and position of the end frame of single 
	* continuum segment with respect to its base frame.
	* @param L  The length of the segment.
	* @param theta  The bending angle of the segment.
	* @param delta  The bending direction of the segment.
	* @return The rotation and position matrix of single continuum segment with 
	* respect to its base frame.
	*/
	RT getSingleSegmentRT(int L, int theta, int delta);

}

#endif // _MATH_KINEMATICS_H_