#ifndef _MATH_OTHERS_H_
#define _MATH_OTHERS_H_
#include <vector>

namespace math
{
	/**
	 * Return a vector contain 'num' numeral between 'start' and 'end'.
	 */
	template<typename T1, typename T2, typename T3>
	std::vector<T3>& linespace(const T1 start, const T2 end, const int num, std::vector<T3> &output)
	{
		if (!std::is_arithmetic<T1>::value || !std::is_arithmetic<T2>::value)
			std::abort();

		if (num == 1)
		{
			output.push_back(static_cast<T3>(start));
			return output;
		}
		else if (num == 2)
		{
			output.push_back(static_cast<T3>(start));
			output.push_back(static_cast<T3>(end));
			return output;
		}

		float step = (static_cast<T3>(end) - static_cast<T3>(start)) / static_cast<T3>(num - 1);
		for (int i = 0; i < num; ++i)
			output.push_back(static_cast<T3>(start + i * step));

		return output;
	}
}

#endif // _MATH_OTHERS_H_