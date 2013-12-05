#pragma once
#ifndef NO_CL

#include <CL/cl.h>

class OpenClContext
{
public:
	OpenClContext();
	~OpenClContext();

	void initialize();
	void finalize();

	bool isInitialized();

public:
	cl_context context;
	cl_device_id device;
	cl_program program;
	cl_command_queue queue;

private:
	bool m_bInitialized;
};

#endif
