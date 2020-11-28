#ifndef _TIMER_MICRO_H_
#define _TIMER_MICRO_H_
#include <chrono>

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


#endif // _TIMER_MICRO_H_