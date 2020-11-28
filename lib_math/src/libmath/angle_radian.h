#ifndef _MATH_ANGLE_RADIAN_H_
#define _MATH_ANGLE_RADIAN_H_
#include <type_traits>

extern const double PI;

namespace math
{
	/** @biref Convert the angle with unit of degree to angle with unit of radian.
	 */
	template<typename T1 = double, typename T2 = double>
	inline T1 deg2rad(const T2 degree)
	{
		if (!std::is_arithmetic<T2>::value)
			std::abort();

		return static_cast<T1>(degree / 180.0 * PI);
	}

	/** @brief Convert the angle with unit of radian to angle with unit of degree.
	 */
	template<typename T1 = double, typename T2 = double>
	inline T1 rad2deg(const T2 radian)
	{
		if (!std::is_arithmetic<T2>::value)
			std::abort();

		return static_cast<T1>(radian / PI * 180.0);
	}
}


#endif // _MATH_ANGLE_RADIAN_H_