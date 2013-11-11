#include "CpuTimer.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <ctime>
#endif

#ifndef NULL
#define NULL 0
#endif

CpuTimer::CpuTimer()
{
}

CpuTimer::~CpuTimer()
{
}

/*
 * http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
 */
double CpuTimer::getTime() const
{
#ifdef _WIN32
	FILETIME a,b,c,d;
    if( GetProcessTimes( GetCurrentProcess(), &a, &b, &c, &d ) != 0 )
	{
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return (double)( d.dwLowDateTime |
				( (unsigned long long)d.dwHighDateTime << 32 ) ) * 0.0000001;
    }
	else
	{
        //  Handle error
        return 0;
    }
#else
	return (double)clock() / CLOCKS_PER_SEC;
#endif
}
