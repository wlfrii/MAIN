#include "mtimer.h"

namespace mtimer
{
	::std::chrono::steady_clock::time_point getCurrentTimePoint()
	{
		return ::std::chrono::steady_clock::now();
	}

	long long getDurationSince(const ::std::chrono::steady_clock::time_point &start_time_point, TimeUnit unit /* = MILLISECOND */)
	{
		::std::chrono::steady_clock::time_point now = ::std::chrono::steady_clock::now();
		switch (unit)
		{
		case NANOSECOND:
			return ::std::chrono::duration_cast<::std::chrono::nanoseconds>(now - start_time_point).count();
		case MICROSECOND:
			return ::std::chrono::duration_cast<::std::chrono::microseconds>(now - start_time_point).count();
		case MILLISECOND:
			return ::std::chrono::duration_cast<::std::chrono::milliseconds>(now - start_time_point).count();
		case SECOND:
			return ::std::chrono::duration_cast<::std::chrono::seconds>(now - start_time_point).count();
		case MINUTE:
			return ::std::chrono::duration_cast<::std::chrono::minutes>(now - start_time_point).count();
		}
		return ::std::chrono::duration_cast<::std::chrono::milliseconds>(now - start_time_point).count();
	}

	::std::string getCurrentTimeStr()
	{
		time_t timep;
		time(&timep);
		char tmp[64];
		strftime(tmp, sizeof(tmp), "%Y%m%d_%H%M%S", localtime(&timep));

		return std::string(tmp);
	}
}