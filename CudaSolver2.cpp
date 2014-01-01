#ifdef BLUB

#include "CudaSolver2.h"
//#include "wave_eq_func.h"

#include <iostream>

#include "wave_eq_func.h"

struct Idx
{
	unsigned int x;
};
static Idx threadIdx;
static Idx blockIdx;
static Idx blockDim;
static Idx gridDim;
#define max(a,b) ((a>b)?a:b)
#define call(b,t,k) { gridDim.x = b; blockDim.x = t; for(unsigned int _b = 0; _b < b; ++_b) { blockIdx.x = _b; for(unsigned int _t = 0; _t < t; ++_t) { threadIdx.x = _t; k; } } }

double* diaMulVec(
		unsigned int n,
		unsigned int diaDiags,
		const int* diaOffsets,
		const double* diaValues,
		const double* vec,
		double* res )
{
	// set result vector to zero
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = 0.0;
	}
	unsigned int totalDiagOffset = 0;
	for( unsigned int k = 0; k < diaDiags; ++k )
	{
		// determine first vector index involved in the diagonal multiplication
		unsigned int vecIndex = max( 0, diaOffsets[ k ] );
		// determine first result vector index
		unsigned int resIndex = max( 0, -diaOffsets[ k ] );
		// detemine the number of elements in the diagonal
		unsigned int diagSize = n - abs( diaOffsets[ k ] );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			res[ resIndex + i ] += diaValues[ totalDiagOffset + i ] * vec[ vecIndex + i ];
		}
		// increment values pointer to next diagonal
		totalDiagOffset += diagSize;
	}
	return res;
}

double* vecAddScaledVec(
		unsigned int n,
		double* u,
		double c,
		const double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] += c * v[ i ];
	}
	return u;
}

double* vecAddScaledVecs(
		unsigned int n,
		double c,
		const double* u,
		double d,
		const double* v,
		double* res )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = c * u[ i ] + d * v[ i ];
	}
	return res;
}

void mainKernel(
		unsigned int ip,
		unsigned int nsteps,
		double l2,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	// determine index of the first element of the subdomain
	unsigned int vecIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];
	double* u = &U[ vecIndex ];

	double a = 2.0 * ( 1.0 - l2 );
	double* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		// u = M * z
		diaMulVec( ip, fdDiags, fdOffsets, fdValues, z, u );
		// u = u + a * z = M * z + a * z
		vecAddScaledVec( ip, u, a, z );
		// u = u + (-w) = M * z + a * z - w
		vecAddScaledVec( ip, u, -1.0, w );

		// shuffle buffers to avoid copying
		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

#define temppr if( id == (1<<7) ) \
	{ \
		std::cout << "w:\t"; \
		for( unsigned int i = 0; i < ip / 16; ++i ) \
		{ \
			std::cout << w[ i ] << '\t'; \
		} \
		std::cout << "--------" << std::endl; \
		std::cout << "z:\t"; \
		for( unsigned int i = 0; i < ip / 16; ++i ) \
		{ \
			std::cout << z[ i ] << '\t'; \
		} \
		std::cout << "--------" << std::endl; \
	}

void initKernel(
		unsigned int ip,
		double L,
		double h,
		double dt,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	// determine index of the first element of the subdomain
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIndex = id * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];
	double* u = &U[ vecIndex ];

	double x0 = h * ip * id / 2 - L;
	for( unsigned int i = 0; i < ip; ++i )
	{
		double x = x0 + i * h;
		w[ i ] = funu0( x );
		u[ i ] = funu1( x );
	}

	double a = 1.0 - dt / h * dt / h;
	// z = M * w
	diaMulVec( ip, fdDiags, fdOffsets, fdValues, w, z );
	temppr
	// z = 0.5 * z + a * w = 0.5 * M * w + a * w
	vecAddScaledVecs( ip, 0.5, z, a, w, z );
	temppr
	// z = z + dt * u
	vecAddScaledVec( ip, z, dt, u );
}

void syncKernel(
		unsigned int ip,
		double* Z,
		double* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int n_1 = gridDim.x * blockDim.x - 1;
	unsigned int vecIndex = id * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];

	// determine indices of left and right neighbor subdomains, considering a
	// periodical continuation
	unsigned int leftNeighbor;
	unsigned int rightNeighbor;
	if( id == 0 )
	{
		leftNeighbor = n_1;
		rightNeighbor = 1;
	}
	else if( id == n_1 )
	{
		leftNeighbor = n_1 - 1;
		rightNeighbor = 0;
	}
	else
	{
		leftNeighbor = id - 1;
		rightNeighbor = id + 1;
	}

	// copy exact data from left neighbors
	double* nz = &Z[ leftNeighbor * ip ];
	double* nw = &W[ leftNeighbor * ip ];
	memcpy( &z[ 0 ], &nz[ ip / 2 ], ip / 4 * sizeof( double ) );
	memcpy( &w[ 0 ], &nw[ ip / 2 ], ip / 4 * sizeof( double ) );
	/*
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ i ] = nz[ ip / 2 + i ];
		w[ i ] = nw[ ip / 2 + i ];
	}
	*/
	// copy exact data from right neighbors
	nz = &Z[ rightNeighbor * ip ];
	nw = &W[ rightNeighbor * ip ];
	memcpy( &z[ ip * 3 / 4 ], &nz[ ip / 4 ], ip / 4 * sizeof( double ) );
	memcpy( &w[ ip * 3 / 4 ], &nw[ ip / 4 ], ip / 4 * sizeof( double ) );
	/*
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ ip * 3 / 4 + i ] = nz[ ip / 4 + i ];
		w[ ip * 3 / 4 + i ] = nw[ ip / 4 + i ];
	}
	*/
}

void reassociationKernel(
		unsigned int ip,
		const double* Z,
		double* S )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIndex = id * ip;
	const double* z = &Z[ vecIndex ];
	double* s = &S[ vecIndex / 2 ];

	// copy left half of the subdomain solution
	memcpy( s, z, ip / 2 * sizeof( double ) );
	/*
	for( unsigned int i = 0; i < ip / 2; ++i )
	{
		s[ ip * id / 2 + i ] = z[ ip * id / 2 + i ];
	}
	*/
}

CudaSolver2::CudaSolver2()
{
}

CudaSolver2::~CudaSolver2()
{
}

// FIXME
#include "ParallelSolver2.h"

void CudaSolver2::solve(
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

	std::vector<double> Z( ndom * ip );
	std::vector<double> W( ndom * ip );
	std::vector<double> U( ndom * ip );
	double* PZ = &Z[ 0 ];
	double* PW = &W[ 0 ];
	double* PU = &U[ 0 ];

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

	// TODO: find optimal values
	unsigned int blocks = 32;
	unsigned int threads = ndom / blocks;

	// calculate initial values on the CUDA device
	call( blocks, threads, initKernel(
			ip,
			L,
			h,
			dt,
			fdDiags,
			&fdOffsets[ 0 ],
			&fdValues[ 0 ],
			PZ,
			PW,
			PU ) );

	// synchronize the solutions (exchange numerically exact parts)
	call( blocks, threads, syncKernel(
			ip,
			PZ,
			PW ) );

	/*

	arma::sp_mat fdArma, _, __;
	ParallelSolver2::genFDMatrices( ip, l2, _, fdArma, __ );

	arma::vec v( &Z[ ( 1 << 7 ) * ip ], ip );
	arma::vec resA = fdArma * v;
	arma::vec resC( ip );
	diaMulVec( ip, fdDiags, &fdOffsets[ 0 ], &fdValues[ 0 ], &v[ 0 ], &resC[ 0 ] );

	//std::cout.precision( 8 );
	std::cout.setf( std::ios::fixed );

#define PRINTVEC(v, n) std::cout << #v ":\t"; std::cout.width( 8 ); for( unsigned int i = 0; i < n; ++i ) { std::cout << v[ i ] << '\t'; } std::cout << std::endl; std::cout.width( 0 );

	std::cout << "l2:\t" << l2 << std::endl;
	PRINTVEC(v, 16);
	PRINTVEC(resA, 16);
	PRINTVEC(resC, 16);

	*/

	errorL2( 0 ) = 0.0;
	arma::vec errorVec( np );

		
		for( unsigned int i = 0; i < ip / 16; ++i )
		{
			std::cout << W[ (1<<7) * ip + i ] << '\t';
		}
		std::cout << "--------" << std::endl;
		for( unsigned int i = 0; i < ip / 16; ++i )
		{
			std::cout << Z[ (1<<7) * ip + i ] << '\t';
		}
		std::cout << "--------" << std::endl;
		

	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate 'nsteps' iterations
		call( blocks, threads, mainKernel(
				ip,
				nsteps,
				l2,
				fdDiags,
				&fdOffsets[ 0 ],
				&fdValues[ 0 ],
				PZ,
				PW,
				PU ) );

		// shuffle the buffers to mirror the shuffling inside the kernel
		double* swap;
		switch( nsteps % 3 )
		{
			case 1:
				swap = PW;
				PW = PZ;
				PZ = PU;
				PU = swap;
				break;
			case 2:
				swap = PZ;
				PZ = PW;
				PW = PU;
				PU = swap;
				break;
		}

		// synchronize the solutions (exchange numerically exact parts)
		call( blocks, threads, syncKernel(
				ip,
				PZ,
				PW ) );

		// reassociate the solutions
		call( blocks, threads, reassociationKernel(
				ip,
				PZ,
				&numSol[ 0 ] ) );

		// calculate exact solution
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );
		// arrayfun2( funsol, x, 0.0, exSol );

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
}

void CudaSolver2::createSparseDiagFDMatrix(
		unsigned int n,
		double l2,
		std::vector<int>& offsets,
		std::vector<double>& values )
{
	offsets = std::vector<int>( { -1, 1 } );
	values = std::vector<double>( 2 * ( n - 1 ), l2 );
}

#endif
