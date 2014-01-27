#ifndef NO_CL
#include "OpenClContext.h"
#include "InputHelper.h"

OpenClContext::OpenClContext()
	: context( NULL ),
	device( NULL ),
	queue( NULL ),
	m_bInitialized( false )
{
}

OpenClContext::~OpenClContext()
{
	if( m_bInitialized )
	{
		finalize();
	}
}

void CL_CALLBACK err_callback( const char* err, const void*, size_t, void* )
{
	std::cout << "OpenCL Error: " << err << std::endl;
}

void OpenClContext::initialize()
{
	// get number of OpenCL platforms
	cl_uint numPlatforms;
	clGetPlatformIDs( 0, NULL, &numPlatforms );

	// let user choose in case multiple platforms are found
	cl_platform_id platform;
	if( numPlatforms > 1 )
	{
		cl_platform_id* platforms = new cl_platform_id[ numPlatforms ];
		clGetPlatformIDs( numPlatforms, platforms, NULL );

		std::cout << "Select OpenCL platform:" << std::endl;
		for( unsigned int i = 0; i < numPlatforms; ++i )
		{
			char name[ 256 ];
			clGetPlatformInfo(
					platforms[ i ],
					CL_PLATFORM_NAME,
					sizeof( name ),
					name,
					NULL );

			std::cout << i << ") " << name << std::endl;
		}
		unsigned int i = inputParam<unsigned int>( "platform", 0 );
		while( i >= numPlatforms )
		{
			std::cout << "Index out of range." << std::endl;
			i = inputParam<unsigned int>( "platform", 0 );
		}

		platform = platforms[ i ];
		delete[] platforms;
	}
	else
	{
		clGetPlatformIDs( 1, &platform, NULL );

		char name[ 256 ];
		clGetPlatformInfo(
				platform,
				CL_PLATFORM_NAME,
				sizeof( name ),
				name,
				NULL );
	}

	// get number of devices supported by the selected platform
	cl_uint numDevices;
	clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );

	// let user choose in case multiple devices are found
	if( numDevices > 1 )
	{
		cl_device_id* devices = new cl_device_id[ numDevices ];
		clGetDeviceIDs(
				platform,
				CL_DEVICE_TYPE_ALL,
				numDevices,
				devices,
				NULL );

		std::cout << "Select OpenCL device:" << std::endl;
		for( unsigned int i = 0; i < numPlatforms; ++i )
		{
			char name[ 256 ];
			clGetDeviceInfo(
					devices[ i ],
					CL_DEVICE_NAME,
					sizeof( name ),
					name,
					NULL );

			std::cout << i << ") " << name << std::endl;
		}
		unsigned int i = inputParam<unsigned int>( "platform", 0 );
		while( i >= numPlatforms )
		{
			std::cout << "Index out of range." << std::endl;
			i = inputParam<unsigned int>( "platform", 0 );
		}

		device = devices[ i ];
		delete[] devices;
	}
	else
	{
		clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );

		char name[ 256 ];
		clGetDeviceInfo(
				device,
				CL_DEVICE_NAME,
				sizeof( name ),
				name,
				NULL );
	}

	// create OpenCL context
	context = clCreateContext( NULL, 1, &device, err_callback, NULL, NULL );

	/*
	// read kernel file
	ifstream sourceFile( "kernel.cl" );
	string sourceCode(
			istreambuf_iterator<char>( sourceFile ),
			( istreambuf_iterator<char>() ) );
	size_t sourceLength = sourceCode.length() + 1;
	const char* source = sourceCode.c_str();
	*/

	// HACK: use single #include statement instead of manually reading the
	// kernel source file
	const char* source = "#include \"kernel.cl\"";

	program = clCreateProgramWithSource(
			context,
			1,
			&source,
			NULL,
			NULL );

	// compile OpenCL program for the selected device
	cl_int err = clBuildProgram( program, 1, &device, NULL/*"-I."*/, NULL, NULL );
	// print build log in case of failure
	if( err != CL_SUCCESS )
	{
		size_t logLength;
		clGetProgramBuildInfo(
				program,
				device,
				CL_PROGRAM_BUILD_LOG,
				0,
				NULL,
				&logLength );

		char* log = new char[ logLength ];
		clGetProgramBuildInfo(
				program,
				device,
				CL_PROGRAM_BUILD_LOG,
				logLength,
				log,
				NULL );

		std::cout << "OpenCL program compilation failed:" << std::endl
			<< log << std::endl;

		delete[] log;
		finalize();
		return;
	}


	queue = clCreateCommandQueue( context, device, 0, NULL );
}

void OpenClContext::finalize()
{
	if( !m_bInitialized )
	{
		return;
	}

	if( queue != NULL )
	{
		clReleaseCommandQueue( queue );
		queue = NULL;
	}
	if( program != NULL )
	{
		clReleaseProgram( program );
		program = NULL;
	}
	if( device != NULL )
	{
		device = NULL;
	}
	if( context != NULL )
	{
		clReleaseContext( context );
		context = NULL;
	}
	m_bInitialized = false;
}

bool OpenClContext::isInitialized()
{
	return m_bInitialized;
}

void OpenClContext::freeDevMem( cl_mem mem )
{
	clReleaseMemObject( mem );
}

#endif
