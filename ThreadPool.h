#pragma once

#include <vector>
#include <thread>
#include <mutex>

/*
 * Queue up different parameters that a function is to be called with on a given
 * number of threads.
 */
class ThreadPool
{
public:
	ThreadPool();
	ThreadPool( void( *func )( void* ), unsigned int numThreads = 0 );
	~ThreadPool();

	/*
	 * Add parameters for an additional call to the function.
	 */
	void addThreadArgs( void* param );

	/*
	 * Run the function for all parameters on multiple threads and wait for all
	 * threads to finish.
	 */
	void run( bool clearJobs = true );

private:
	void thread();

private:
	void( *m_pFunction )( void* );
	std::vector<void*> m_vecParams;
	std::vector<std::thread> m_vecThreads;
	unsigned int m_uiCurJobIndex;
	std::mutex m_Mutex;
};
