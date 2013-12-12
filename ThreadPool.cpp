#include "ThreadPool.h"

ThreadPool::ThreadPool()
{
}

ThreadPool::ThreadPool( unsigned int numThreads, void( *func )( void* ) )
	: m_pFunction( func ), m_vecThreads( numThreads )
{
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
	unsigned int jobsRemaining = m_vecParams.size();
	for( unsigned int k = 0; k < m_vecParams.size(); k += m_vecThreads.size() )
	{
		unsigned int imax = MIN( jobsRemaining, m_vecThreads.size() );
		for( unsigned int i = 0; i < imax; ++i )
		{
			m_vecThreads[ i ] = std::thread( m_pFunction, m_vecParams[ i ] );
		}
		for( unsigned int i = 0; i < imax; ++i )
		{
			m_vecThreads[ i ].join();
		}
	}
}
