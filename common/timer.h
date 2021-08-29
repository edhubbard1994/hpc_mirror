#ifndef __timer_h
#define __timer_h

#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define HighResTimer
#ifdef HighResTimer

   typedef struct timespec TimerType;

#else

   typedef struct timeval TimerType;

#endif

TimerType getTimeStamp(void);
uint64_t getClockTicks(void);
double getElapsedTime (TimerType t0, TimerType t1);
double getTicksPerSecond (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __timer_h
