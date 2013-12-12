#pragma once

#include <vector>
#include <thread>

class ThreadPool
{
public:
	ThreadPool();
	ThreadPool( unsigned int numThreads, void( *func )( void* ) );
	~ThreadPool();

	void addThreadArgs( void* param );
	void run( bool clearJobs = true );

private:
	void( *m_pFunction )( void* );
	std::vector<void*> m_vecParams;
	std::vector<std::thread*> m_vecThreads;
};
