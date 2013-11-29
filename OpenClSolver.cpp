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
	// number of (overlapping) subdomains (= number of processors/threads)
	unsigned int ns = 1 << n;
	// points per subdomain
	unsigned int ip = 2 * np / ns;
	// time steps between solution reassociations
	unsigned int nsteps = ip / 4;

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

	// allocate memory for the analytical solution
	vec exSol( np );
	// allocate memory for error values
	vec errorL2( kmax );

	// initial values / boundary conditions
	vec* F = new vec[ ns ];
	vec* G = new vec[ ns ];

	vec* W = new vec[ ns ];
	vec* Z = new vec[ ns ];
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

	size_t* workItemSizes = new size_t[ workItemDimensions ];
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_MAX_WORK_ITEM_SIZES,
			workItemDimensions * sizeof( size_t ),
			workItemSizes,
			NULL );

	size_t extensionStringLength;
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_EXTENSIONS,
			0,
			NULL,
			&extensionStringLength );

	char* extensionString = new char[ extensionStringLength ];
	clGetDeviceInfo(
			s_clContext.device,
			CL_DEVICE_EXTENSIONS,
			extensionStringLength * sizeof( char ),
			extensionString,
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

	delete[] extensionString;
	delete[] workItemSizes;
	return ( workItemSizes[ 0 ] >= ns && doublePrecisionSupported );
}

#endif
