#include "CudaSolver.h"

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

	// allocate memory for the analytical solution
	arma::vec exSol( np );
	// allocate memory for error values
	arma::vec errorL2( kmax );

	// allocate memory for Z, W, U vectors, used for the solution
	std::vector<double> Z( ndom * ip );
	std::vector<double> W( ndom * ip );
	std::vector<double> U( ndom * ip );

	// initialize arrays allocated above
	for( unsigned int i = 0; i < ndom; ++i )
	{
		// calculate values for F, G
		F[ i ] = vec( ip );
		arrayfun( funu0, X[ i ], F[ i ] );
		G[ i ] = vec( ip );
		arrayfun( funu1, X[ i ], G[ i ] );

		// assign initial values to W, Z
		W[ i ] = F[ i ];
		Z[ i ] = vec( ip );

		// get pointers to solution vectors
		PW[ i ] = &W[ i ];
		PZ[ i ] = &Z[ i ];
	}

	// calculate one step back in time
	evalRhs( F[ 0 ], G[ 0 ], Ml, l2, dt2, 1.0, 0.5, -dt, Z[ 0 ] );
	for( unsigned int i = 1; i < ndom - 1; ++i )
	{
		evalRhs( F[ i ], G[ i ], Mi, l2, dt2, 1.0, 0.5, -dt, Z[ i ] );
	}
	evalRhs(
			F[ ndom - 1 ],
			G[ ndom - 1 ],
			Mr,
			l2,
			dt2,
			1.0,
			0.5,
			-dt,
			Z[ ndom - 1 ] );

	numSol = vec( np );
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

