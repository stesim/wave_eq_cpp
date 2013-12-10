#include "CudaSolver.h"
#include "CudaHelper.h"
#include "wave_eq_func.h"

CudaSolver::CudaSolver()
{
}

CudaSolver::~CudaSolver()
{
}

void CudaSolver::solve(
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
	numSol = arma::vec( np );
	// allocate memory for the analytical solution
	arma::vec exSol( np );
	// allocate memory for error values
	arma::vec errorL2( kmax );

	// allocate arrays required by the kernels on device memory
	int* d_fdOffsets = CudaHelper::allocDevMem<int>( fdOffsets.size() );
	double* d_fdValues = CudaHelper::allocDevMem<double>( fdValues.size() );
	double* d_Z = CudaHelper::allocDevMem<double>( ndom * ip );
	double* d_W = CudaHelper::allocDevMem<double>( ndom * ip );
	double* d_U = CudaHelper::allocDevMem<double>( ndom * ip );
	double* d_numSol = CudaHelper::allocDevMem<double>( numSol.n_elem );

	// copy FD matrix to device memory
	CudaHelper::copyHostToDevMem( &fdOffsets[ 0 ], d_fdOffsets, fdOffsets.size() );
	CudaHelper::copyHostToDevMem( &fdValues[ 0 ], d_fdValues, fdValues.size() );
	// free FD matrix host memory
	fdOffsets.clear();
	fdValues.clear();

	// TODO: find optimal values
	unsigned int blocks = 32;
	unsigned int threads = ndom / blocks;

	// calculate initial values on the CUDA device
	CudaHelper::callInitKernel(
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
	errorL2( 0 ) = 0.0;

	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate 'nsteps' iterations
		CudaHelper::callMainKernel(
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
		double* swap;
		switch( nsteps % 3 )
		{
			case 1:
				swap = d_W;
				d_Z = d_U;
				d_W = d_Z;
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
		CudaHelper::callSyncKernel(
				blocks,
				threads,
				ip,
				d_Z,
				d_W );

		// reassociate the solutions
		CudaHelper::callReassociationKernel(
				blocks,
				threads,
				ip,
				d_Z,
				d_numSol );

		// copy the solution to host memory
		CudaHelper::copyDevToHostMem(
				d_numSol,
				numSol.memptr(),
				numSol.n_elem );

		// calculate exact solution
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );

		// calculate current L2 error
		double err = 0.0;
		for( unsigned int i = 0; i < exSol.size(); ++i )
		{
			double locErr = numSol( i ) - exSol( i );
			err += locErr * locErr;
		}
		errorL2( k ) = sqrt( h * err );

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
	CudaHelper::freeDevMem( d_fdOffsets );
	CudaHelper::freeDevMem( d_fdValues );
	CudaHelper::freeDevMem( d_Z );
	CudaHelper::freeDevMem( d_W );
	CudaHelper::freeDevMem( d_U );
	CudaHelper::freeDevMem( d_numSol );
}

void CudaSolver::createSparseDiagFDMatrix(
		unsigned int n,
		double l2,
		std::vector<int>& offsets,
		std::vector<double>& values )
{
	offsets = std::vector<int>( { -1, 1 } );
	values = std::vector<double>( 2 * ( n - 1 ), l2 );
}

