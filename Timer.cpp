#include "Timer.h"

Timer::Timer()
	: m_dStartTime( 0.0 ), m_dStopTime( 0.0 ), m_bRunning( false )
{
}

Timer::~Timer()
{
}


void Timer::start()
{
	if( !m_bRunning )
	{
		m_dStartTime = getTime();
		m_bRunning = true;
	}
}

double Timer::stop()
{
	if( !m_bRunning )
	{
		return 0.0;
	}

	m_dStopTime = getTime();
	m_bRunning = false;
	return ( m_dStopTime - m_dStartTime );
}

double Timer::getElapsedTime()
{
	return ( ( m_bRunning ? getTime() : m_dStopTime ) - m_dStartTime );
}
