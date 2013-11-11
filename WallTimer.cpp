#include "WallTimer.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#ifndef NULL
#define NULL 0
#endif

WallTimer::WallTimer()
{
}

WallTimer::~WallTimer()
{
}

/*
 * http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
 */
double WallTimer::getTime() const
{
#ifdef _WIN32
	LARGE_INTEGER time,freq;
    if( !QueryPerformanceFrequency( &freq ) )
	{
        //  Handle error
        return 0;
    }
    if( !QueryPerformanceCounter( &time ) )
	{
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
#else
	struct timeval time;
    if( gettimeofday( &time, NULL ) )
	{
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}
