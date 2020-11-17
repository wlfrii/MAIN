#ifndef MTIMER_H
#define MTIMER_H
#pragma warning( disable : 4996 ) // disable the error of localtime
#include <chrono>
#include <typeinfo>
#include <time.h>
#include <string>

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
	::std::chrono::steady_clock::time_point getCurrentTimePoint();

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

        float k[3] = {0};
        if(typeid(Tp) == typeid(std::chrono::microseconds))
        {
            k[0] = 1.f;
            k[1] = 1.f / 1000.f;
            k[2] = 1.f / 1000000.f;
        }
        else if(typeid(Tp) == typeid(std::chrono::milliseconds))
        {
            k[0] = 1000.f;
            k[1] = 1.f;
            k[2] = 1.f / 1000.f;
        }
        else if(typeid(Tp) == typeid(std::chrono::seconds))
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
	::std::string getCurrentTimeStr();
}


/** @brief Return current time point.
  Class std::chrono::steady_clock represents a monotonic clock. The time points of this clock cannot decrease as physical time moves forward and the time between ticks of this clock is constant. This clock is not related to wall clock time (for example, it can be time since last reboot), and is most suitable for measuring intervals.
*/
#define GET_CURRENT_TIME_POINT(TIME_POINT) \
	::std::chrono::steady_clock::time_point TIME_POINT = ::std::chrono::steady_clock::now()


#define ASSERT_FINISH_IN_VOID(FUNC, DURATION, INFO)  \
	do{  \
		  	GET_CURRENT_TIME_POINT(start);\
		    FUNC;\
		    GET_CURRENT_TIME_POINT(end);\
			long long ms = ::std::chrono::duration_cast<::std::chrono::microseconds>(end - start).count(); \
			if (ms > DURATION) { printf("%s, [%.4f] ms elapsed:\t", INFO, ms / 1000.f); } \
	 } while (0)

#define ASSERT_FINISH_IN(FUNC, DURATION, INFO, RC)  \
	do{  \
		  	GET_CURRENT_TIME_POINT(start);\
		    RC = FUNC;\
		    GET_CURRENT_TIME_POINT(end);\
			long long ms = ::std::chrono::duration_cast<::std::chrono::microseconds>(end - start).count(); \
			if (ms > DURATION) { printf("%s, [%.4f] ms elapsed:\t", INFO, ms / 1000.f);} \
	 } while (0)

#define ASSERT_FINISH_IN_INFO(FUNC, DURATION, RC, FMT, ...)  \
	do{  \
		  	GET_CURRENT_TIME_POINT(start);\
		    RC = FUNC;\
		    GET_CURRENT_TIME_POINT(end);\
			long long ms = ::std::chrono::duration_cast<::std::chrono::microseconds>(end - start).count(); \
			if (ms > DURATION) { printf("[%.4f] ms elapsed:\t " #FMT, ms / 1000.f, ## __VA_ARGS__);} \
	 } while (0)

#endif
