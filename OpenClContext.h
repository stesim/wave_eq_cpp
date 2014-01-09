#pragma once
#ifndef NO_CL

#include <CL/cl.h>
#include <iostream>

class OpenClContext
{
public:
	OpenClContext();
	~OpenClContext();

	void initialize();
	void finalize();

	bool isInitialized();

	/*
	 * Allocate device memory.
	 */
	template<typename T>
	cl_mem allocDevMem( size_t numElem )
	{
		return clCreateBuffer(
				context,
				CL_MEM_READ_WRITE,
				numElem * sizeof( T ),
				nullptr,
				nullptr );
	}

	/*
	 * Free device memory.
	 */
	void freeDevMem( cl_mem mem );

	/*
	 * Copy host memory to device memory.
	 */
	template<typename T>
	void copyHostToDevMem( const T* hostMem, cl_mem devMem, size_t numElem )
	{
		cl_int err = clEnqueueWriteBuffer(
				queue,
				devMem,
				true,
				0,
				numElem * sizeof( T ),
				const_cast<T*>( hostMem ),
				0,
				nullptr,
				nullptr );

		if( err != CL_SUCCESS )
		{
			std::cout << "copyHostToDevMem() error " << err << std::endl;
		}
	}

	/*
	 * Copy device memory to host memory.
	 */
	template<typename T>
	void copyDevToHostMem( cl_mem devMem, T* hostMem, size_t numElem )
	{
		cl_int err = clEnqueueReadBuffer(
				queue,
				devMem,
				true,
				0,
				numElem * sizeof( T ),
				hostMem,
				0,
				nullptr,
				nullptr );

		if( err != CL_SUCCESS )
		{
			std::cout << "copyDevToHostMem() error " << err << std::endl;
		}
	}

public:
	cl_context context;
	cl_device_id device;
	cl_program program;
	cl_command_queue queue;

private:
	bool m_bInitialized;
};

#endif
