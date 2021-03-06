#ifndef NO_CL
#include "OpenClSolver.h"
#include <string>

using namespace arma;

OpenClContext OpenClSolver::s_clContext;

OpenClSolver::OpenClSolver()
{
	if( !s_clContext.isInitialized() )
	{
		s_clContext.initialize();
	}

	m_KernelInit = createClKernel( "kernelInit" );
	m_KernelMain = createClKernel( "kernelMain" );
	m_KernelSync = createClKernel( "kernelSync" );
	m_KernelReassociate = createClKernel( "kernelReassociate" );
}

OpenClSolver::~OpenClSolver()
{
}

void OpenClSolver::solve(
		double L,
		unsigned int N,
		unsigned int n,
		double T,
		SpacialFunction funu0,
		SpacialFunction funu1,
		SpaciotemporalFunction funsol,
		arma::vec& x,
		arma::vec& numSol,
		arma::vec* exactSol,
		arma::vec* error )
{
	// number of points
	unsigned int np = 1 << N;
	// number of disjoint subdomains (= number of processors/threads)
	unsigned int ns = 1 << n;
	// number of overlapping subdomains
	unsigned int ndom = 2 * ns;
	// points per subdomain
	unsigned int ip = np / ns;
	// time steps between solution reassociations
	unsigned int nsteps = ip / 2;

	// spacial discretization
	x = arma::linspace( -L, L, np );
	// discretization spacing
	double h = 2.0 * L / ( np - 1 );

	// temporal discretization step size
	double dt = h / 4.0;
	// helper variables
	double dt2 = dt * dt;
	double l = dt / h;
	double l2 = l * l;

	// total number of reassociations
	unsigned int kmax =
		static_cast<unsigned int>( ceil( T / ( nsteps * dt ) ) );

	// generate finite differences matrix
	std::vector<int> fdOffsets;
	std::vector<double> fdValues;
	createSparseDiagFDMatrix( ip, l2, fdOffsets, fdValues );
	unsigned int fdDiags = fdOffsets.size();

	// allocate memory fot the numerical solution
	// numSol = arma::vec( np );
	numSol = arma::zeros( np );
	// allocate memory for the analytical solution
	arma::vec exSol( np );
	// allocate memory for error values
	arma::vec errorL2( kmax );

	// allocate arrays required by the kernels on device memory
	cl_mem d_fdOffsets = s_clContext.allocDevMem<int>( fdOffsets.size() );
	cl_mem d_fdValues = s_clContext.allocDevMem<double>( fdValues.size() );
	cl_mem d_Z = s_clContext.allocDevMem<double>( ndom * ip );
	cl_mem d_W = s_clContext.allocDevMem<double>( ndom * ip );
	cl_mem d_U = s_clContext.allocDevMem<double>( ndom * ip );
	cl_mem d_numSol = s_clContext.allocDevMem<double>( numSol.n_elem );

	// copy FD matrix to device memory
	s_clContext.copyHostToDevMem( &fdOffsets[ 0 ], d_fdOffsets, fdOffsets.size() );
	s_clContext.copyHostToDevMem( &fdValues[ 0 ], d_fdValues, fdValues.size() );
	// free FD matrix host memory
	fdOffsets.clear();
	fdValues.clear();

	errorL2( 0 ) = 0.0;
	arma::vec errorVec( np );

	// TODO: find optimal values
	unsigned int threads = 4; // GPU: 4; CPU: 8
	unsigned int blocks = ndom / threads;

	// calculate initial values on the CUDA device
	callInitKernel(
			blocks,
			threads,
			ip,
			L,
			h,
			dt,
			fdDiags,
			d_fdOffsets,
			d_fdValues,
			d_Z,
			d_W,
			d_U );

	// synchronize the solutions (exchange numerically exact parts)
	callSyncKernel(
			blocks,
			threads,
			ip,
			d_Z,
			d_W );

	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate 'nsteps' iterations
		callMainKernel(
				blocks,
				threads,
				ip,
				nsteps,
				l2,
				fdDiags,
				d_fdOffsets,
				d_fdValues,
				d_Z,
				d_W,
				d_U );

		// shuffle the buffers to mirror the shuffling inside the kernel
		cl_mem swap;
		switch( nsteps % 3 )
		{
			case 1:
				swap = d_W;
				d_W = d_Z;
				d_Z = d_U;
				d_U = swap;
				break;
			case 2:
				swap = d_Z;
				d_Z = d_W;
				d_W = d_U;
				d_U = swap;
				break;
		}

		// synchronize the solutions (exchange numerically exact parts)
		callSyncKernel(
				blocks,
				threads,
				ip,
				d_Z,
				d_W );

		if( k == kmax - 1 )
		{
			// reassociate the solutions
			callReassociationKernel(
					blocks,
					threads,
					ip,
					d_Z,
					d_numSol );

			// copy the solution to host memory
			s_clContext.copyDevToHostMem(
					d_numSol,
					numSol.memptr(),
					numSol.n_elem );
		}

		// calculate exact solution
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );
		//arrayfun2( funsol, x, 0.0, exSol );

		// calculate current L2 error
		double err = 0.0;
		for( unsigned int i = 0; i < exSol.size(); ++i )
		{
			double locErr = numSol( i ) - exSol( i );
			err += locErr * locErr;
		}
		errorL2( k ) = sqrt( h * err );
		//errorVec = numSol - exSol;
		//errorL2( k ) = sqrt( h * arma::dot( errorVec, errorVec ) );

		if( m_funOnReassociation != NULL )
		{
			m_funOnReassociation( k, kmax, x, numSol, exSol, errorL2( k ) );
		}
	}

	// return analytical solution and L2 error if necessary
	if( exactSol != NULL )
	{
		*exactSol = exSol;
	}
	if( error != NULL )
	{
		*error = errorL2;
	}

	// free device resources
	s_clContext.freeDevMem( d_fdOffsets );
	s_clContext.freeDevMem( d_fdValues );
	s_clContext.freeDevMem( d_Z );
	s_clContext.freeDevMem( d_W );
	s_clContext.freeDevMem( d_U );
	s_clContext.freeDevMem( d_numSol );


/*
	// number of points
	unsigned int np = 1 << N;
	// number of (overlapping) subdomains (= number of processors/threads)
	unsigned int ns = 1 << n;
	// number of overlapping subdomains
	unsigned int ndom = 2 * ns;
	// points per subdomain
	unsigned int ip = np / ns;
	// time steps between solution reassociations
	unsigned int nsteps = ip / 2;

	// spacial discretization
	x = linspace( -L, L, np );
	// discretization spacing
	double h = 2.0 * L / ( np - 1 );

	// temporal discretization step size
	double dt = h / 4.0;
	// helper variables
	double dt2 = dt * dt;
	double l = dt / h;
	double l2 = l * l;

	// total number of reassociations
	unsigned int kmax =
		static_cast<unsigned int>( ceil( T / ( nsteps * dt ) ) );

	if( !checkClCapabilities( np, ns ) )
	{
		std::cout << "Insufficient OpenCL device capabilities." << std::endl;
		return;
	}

	// allocate memory for error values
	arma::vec errorL2( kmax );

	// initial values / boundary conditions
	std::vector<double> Z( ndom * ip );
	std::vector<double> W( ndom * ip );
	std::vector<double> U( ndom * ip );
	// complete solution
	//std::vector<double> s( np );
	numSol = arma::vec( np );
	// exact solution
	//std::vector<double> e( np );
	arma::vec exSol( np );
	// sparse diagonal FD matrix
	std::vector<int> matOffsets;
	std::vector<double> matValues;
	createSparseDiagFDMatrix( ip, l2, matOffsets, matValues );

	// create OpenCL buffers
	cl_mem _clZ = createClBufferFromStdVector( Z );
	cl_mem _clW = createClBufferFromStdVector( W );
	cl_mem _clU = createClBufferFromStdVector( U );
	cl_mem _clNumSol = createClBufferFromArmaVec( numSol );
	cl_mem _clExSol = createClBufferFromArmaVec( exSol );
	cl_mem _clMatOffsets = copyClBufferFromStdVector( matOffsets );
	cl_mem _clMatValues = copyClBufferFromStdVector( matValues );
	if( _clZ == NULL || _clW == NULL || _clU == NULL
			|| _clNumSol == NULL || _clExSol == NULL
			|| _clMatOffsets == NULL || _clMatValues == NULL )
	{
		std::cout << "OpenCL buffer creation failed." << std::endl;
		return;
	}

	// create OpenCL kernels
	cl_kernel kernelInit = createClKernel( "wave_eq_init" );
	cl_kernel kernelMain = createClKernel( "wave_eq_cl" );
	cl_kernel kernelReassociate = createClKernel( "wave_eq_reassociate" );
	if( kernelInit == NULL || kernelMain == NULL || kernelReassociate == NULL )
	{
		std::cout << "OpenCL kernel creation failed." << std::endl;
		std::cout << kernelInit << std::endl << kernelMain << std::endl << kernelReassociate << std::endl;
		return;
	}

	// ND ranges
	size_t globalWorkSize = ndom;
	size_t localWorkSize = 1;

	// calculate initial values
	double a = 1 - l2;
	unsigned int numDiag = matOffsets.size();
	clSetKernelArg( kernelInit, 0, sizeof( unsigned int ), &ip );
	clSetKernelArg( kernelInit, 1, sizeof( double ), &L );
	clSetKernelArg( kernelInit, 2, sizeof( double ), &h );
	clSetKernelArg( kernelInit, 3, sizeof( double ), &a );
	clSetKernelArg( kernelInit, 4, sizeof( double ), &dt );
	clSetKernelArg( kernelInit, 5, sizeof( cl_mem ), &_clZ );
	clSetKernelArg( kernelInit, 6, sizeof( cl_mem ), &_clW );
	clSetKernelArg( kernelInit, 7, sizeof( cl_mem ), &_clU );
	clSetKernelArg( kernelInit, 8, sizeof( unsigned int ), &numDiag );
	clSetKernelArg( kernelInit, 9, sizeof( cl_mem ), &_clMatOffsets );
	clSetKernelArg( kernelInit, 10, sizeof( cl_mem ), &_clMatValues );
	clEnqueueNDRangeKernel(
			s_clContext.queue,
			kernelInit,
			1,
			NULL,
			&globalWorkSize,
			&localWorkSize,
			0,
			NULL,
			NULL );
	clFinish( s_clContext.queue );

	// set (constant) main kernel arguments
	a *= 2.0;
	clSetKernelArg( kernelMain, 0, sizeof( unsigned int ), &ip );
	clSetKernelArg( kernelMain, 1, sizeof( unsigned int ), &nsteps );
	clSetKernelArg( kernelMain, 2, sizeof( double ), &a );
	//clSetKernelArg( kernelMain, 3, sizeof( cl_mem ), &_clZ );
	//clSetKernelArg( kernelMain, 4, sizeof( cl_mem ), &_clW );
	//clSetKernelArg( kernelMain, 5, sizeof( cl_mem ), &_clU );
	clSetKernelArg( kernelMain, 6, sizeof( unsigned int ), &numDiag );
	clSetKernelArg( kernelMain, 7, sizeof( cl_mem ), &_clMatOffsets );
	clSetKernelArg( kernelMain, 8, sizeof( cl_mem ), &_clMatValues );

	// set (constant) reassociation kernel arguments
	clSetKernelArg( kernelReassociate, 0, sizeof( unsigned int ), &ip );
	//clSetKernelArg( kernelReassociate, 1, sizeof( cl_mem ), &_clZ );
	//clSetKernelArg( kernelReassociate, 2, sizeof( cl_mem ), &_clW );
	clSetKernelArg( kernelReassociate, 3, sizeof( cl_mem ), &_clNumSol );
	clSetKernelArg( kernelReassociate, 4, sizeof( double ), &L );
	clSetKernelArg( kernelReassociate, 5, sizeof( double ), &h );
	//clSetKernelArg( kernelReassociate, 6, sizeof( double ), &t );
	clSetKernelArg( kernelReassociate, 7, sizeof( cl_mem ), &_clExSol );
	
	// use buffer pointer to account for the shuffling inside the main kernel
	cl_mem* p_clZ = &_clZ;
	cl_mem* p_clW = &_clW;
	cl_mem* p_clU = &_clU;
	for( unsigned int k = 0; k < kmax; ++k )
	{
		// set shuffled buffer arguments
		clSetKernelArg( kernelMain, 3, sizeof( cl_mem ), p_clZ );
		clSetKernelArg( kernelMain, 4, sizeof( cl_mem ), p_clW );
		clSetKernelArg( kernelMain, 5, sizeof( cl_mem ), p_clU );
		// run main kernel
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				kernelMain,
				1,
				NULL,
				&globalWorkSize,
				&localWorkSize,
				0,
				NULL,
				NULL );
		clFinish( s_clContext.queue );

		// determine buffers after shuffling
		switch( ( ( k + 1 ) * nsteps ) % 3 )
		{
			case 0:
				p_clZ = &_clZ;
				p_clW = &_clW;
				p_clU = &_clU;
				break;
			case 1:
				p_clZ = &_clU;
				p_clW = &_clZ;
				p_clU = &_clW;
				break;
			case 2:
				p_clZ = &_clW;
				p_clW = &_clU;
				p_clU = &_clZ;
				break;
		}

		// set reassociation arguments
		double t = ( k + 1 ) * dt;
		clSetKernelArg( kernelReassociate, 1, sizeof( cl_mem ), p_clZ );
		clSetKernelArg( kernelReassociate, 2, sizeof( cl_mem ), p_clW );
		clSetKernelArg( kernelReassociate, 6, sizeof( double ), &t );

		// reassociate solutions
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				kernelReassociate,
				1,
				NULL,
				&globalWorkSize,
				&localWorkSize,
				0,
				NULL,
				NULL );
		clFinish( s_clContext.queue );

		// copy numerical and exact solution to host memory
		if( !copyClBufferToArmaVec( _clNumSol, numSol )
				|| !copyClBufferToArmaVec( _clExSol, exSol ) )
		{
			std::cout << "Copying buffers to host memory failed." << std::endl;
			return;
		}
		clFinish( s_clContext.queue );

		// calculate L2 error
		arma::vec err = exSol - numSol;
		errorL2( k ) = sqrt( h * dot( err, err ) );
		
		if( m_funOnReassociation != NULL )
		{
			m_funOnReassociation( k, kmax, x, numSol, exSol, errorL2( k ) );
		}
	}

	// return numerical solution and optionally analytical solution and L2 error
	if( exactSol != NULL )
	{
		*exactSol = exSol;
	}
	if( error != NULL )
	{
		*error = errorL2;
	}
*/
}

void OpenClSolver::createSparseDiagFDMatrix(
		unsigned int n,
		double l2,
		std::vector<int>& offsets,
		std::vector<double>& values )
{
	offsets = std::vector<int>( { -1, 1 } );
	values = std::vector<double>( 2 * ( n - 1 ), l2 );
}

bool OpenClSolver::checkClCapabilities( unsigned int np, unsigned int ns )
{
	cl_uint computeUnits;
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof( cl_uint ),
			&computeUnits,
			NULL );

	cl_uint workItemDimensions;
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			sizeof( cl_uint ),
			&workItemDimensions,
			NULL );

	std::vector<size_t> workItemSizes( workItemDimensions );
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_MAX_WORK_ITEM_SIZES,
			workItemDimensions * sizeof( size_t ),
			&workItemSizes[ 0 ],
			NULL );

	size_t extensionStringLength;
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_EXTENSIONS,
			0,
			NULL,
			&extensionStringLength );

	std::string extensionString( extensionStringLength, '\0' );
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_EXTENSIONS,
			extensionStringLength * sizeof( char ),
			&extensionString[ 0 ],
			NULL );

	std::string extStr( extensionString );
	bool doublePrecisionSupported =
		( extStr.find( "cl_khr_fp64" ) != std::string::npos );
	
	/*
	std::cout << "Compute units: " << computeUnits << std::endl;
	std::cout << "1D work item size: " << workItemSizes[ 0 ] << std::endl;
	std::cout << "Double precision: "
		<< ( doublePrecisionSupported ? "supported" : "not supported" )
		<< std::endl;
	*/

	return ( workItemSizes[ 0 ] >= ns && doublePrecisionSupported );
}

cl_kernel OpenClSolver::createClKernel( const char* name )
{
	cl_int err;
	cl_kernel ker = clCreateKernel( s_clContext.program, name, &err );
	if( err != CL_SUCCESS )
	{
		std::cout << "createClKernel() error " << err << std::endl;
	}
	return ker;
}

#endif
