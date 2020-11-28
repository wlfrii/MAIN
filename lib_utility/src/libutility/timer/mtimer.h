#ifndef _MTIMER_H_
#define _MTIMER_H_
#include <typeinfo>
#include <string>
#include <ctime>
/** Ignorethe waring in Windows. */
#pragma warning( disable : 4996 ) // disable the error of localtime
/* Include the other similiar header file so that the user could call all of this interfaces by include 
*  this header file only. 
*/
#include "timer_micro.h" 

namespace mtimer
{
	/** @brief The flags used for time count.
	*/
	enum TimeUnit
	{
		NANOSECOND = 0,
		MICROSECOND = 1,
		MILLISECOND = 2,
		SECOND = 3,
		MINUTE = 4
		// HOUR
		// DAY
	};


	/** @brief Return current time point
	*/
    inline ::std::chrono::steady_clock::time_point getCurrentTimePoint()
    {
        return ::std::chrono::steady_clock::now();
    }


	/** @brief Return the duration time from start_time_point, default unit is milliseconds
	*/
	long long getDurationSince(const ::std::chrono::steady_clock::time_point &start_time_point, TimeUnit unit = MILLISECOND);


	/** @brief Return time duration in milliseconds since epoch
		*/
	template<typename Tp = std::chrono::microseconds>
	Tp getDurationSinceEpoch()
	{
		return ::std::chrono::duration_cast<Tp>(::std::chrono::system_clock::now().time_since_epoch());
	}

	/** @brief Return the duration time from start_time_point, the time unit is milliseconds
	*/
	template<typename Tp = std::chrono::microseconds>
	float getDurationSince(const Tp &start_time_point, TimeUnit unit = TimeUnit::MILLISECOND)
	{
		auto current_time_point = ::std::chrono::duration_cast<Tp>(::std::chrono::system_clock::now().time_since_epoch());

		float k[3] = { 0 };
		if (typeid(Tp) == typeid(std::chrono::microseconds))
		{
			k[0] = 1.f;
			k[1] = 1.f / 1000.f;
			k[2] = 1.f / 1000000.f;
		}
		else if (typeid(Tp) == typeid(std::chrono::milliseconds))
		{
			k[0] = 1000.f;
			k[1] = 1.f;
			k[2] = 1.f / 1000.f;
		}
		else if (typeid(Tp) == typeid(std::chrono::seconds))
		{
			k[0] = 1000000.f;
			k[1] = 1000.f;
			k[2] = 1.f;
		}
		switch (unit)
		{
		case TimeUnit::MICROSECOND:
			return (current_time_point - start_time_point).count() * k[0];
		case TimeUnit::MILLISECOND:
			return (current_time_point - start_time_point).count() * k[1];
		case TimeUnit::SECOND:
			return (current_time_point - start_time_point).count() * k[2];
		}
		return (current_time_point - start_time_point).count();
	}


    /** @brief Return current time string.
    */
    inline ::std::string getCurrentTimeStr()
    {
        time_t timep;
        time(&timep);
        char tmp[64];
        strftime(tmp, sizeof(tmp), "%Y%m%d_%H%M%S", localtime(&timep));

        return std::string(tmp);
    }
}

#endif // _MTIMER_H_
