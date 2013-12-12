#include "ThreadPool.h"

ThreadPool::ThreadPool()
	: m_pFunction( NULL ), m_uiCurJobIndex( 0 )
{
}

ThreadPool::ThreadPool( void( *func )( void* ), unsigned int numThreads )
	: m_pFunction( func ), m_uiCurJobIndex( 0 )
{
	if( numThreads == 0 )
	{
		// choose optimal number of threads
		numThreads = std::thread::hardware_concurrency();
		if( numThreads == 0 )
		{
			// unable to determine optimal number of threads, default to 4
			numThreads = 4;
		}
	}

	m_vecThreads.resize( numThreads );
}

ThreadPool::~ThreadPool()
{
}

void ThreadPool::addThreadArgs( void* param )
{
	m_vecParams.push_back( param );
}

void ThreadPool::run( bool clearJobs )
{
	if( m_pFunction == NULL )
	{
		return;
	}
	for( unsigned int i = 0; i < m_vecThreads.size(); ++i )
	{
		m_vecThreads[ i ] = std::thread( &ThreadPool::thread, this );
	}
	for( unsigned int i = 0; i < m_vecThreads.size(); ++i )
	{
		m_vecThreads[ i ].join();
	}
	if( clearJobs )
	{
		m_vecParams.clear();
	}
	m_uiCurJobIndex = 0;
}

void ThreadPool::thread()
{
	unsigned int jobIndex;
	while( true )
	{
		m_Mutex.lock();
		jobIndex = m_uiCurJobIndex++;
		m_Mutex.unlock();
		if( jobIndex >= m_vecParams.size() )
		{
			break;
		}

		m_pFunction( m_vecParams[ jobIndex ] );
	}
}
