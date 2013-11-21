#include "ParallelSolver.h"
#include <vector>
#include <thread>

using namespace arma;

ParallelSolver::ParallelSolver()
{
}


ParallelSolver::~ParallelSolver()
{
}

void ParallelSolver::solve(
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
	unsigned int ndom = 2 * ns - 1;
	// points per subdomain
	unsigned int ip = np / ns;
	// time steps between solution reassociations
	unsigned int nsteps = ip / 2;

	unsigned int threadsSupported = std::thread::hardware_concurrency();
	if( threadsSupported > 0 && ndom > threadsSupported )
	{
		std::cout << "WARNING: Number of used threads is greater than "
			"physically supported (" << threadsSupported << ")." << std::endl;
	}

	// spacial discretization
	x = linspace( -L, L, np );
	// discretization spacing
	double h = 2.0 * L / ( np - 1 );
	// overlapping intervals
	vec* X = new vec[ ndom ];
	for( unsigned int i = 0; i < ndom; ++i )
	{
		X[ i ] = x.subvec( ip * i / 2, ip * ( i + 2 ) / 2 - 1 );
	}

	// temporal discretization step size
	double dt = h / 4.0;
	// helper variables
	double dt2 = dt * dt;
	double l = dt / h;
	double l2 = l * l;

	// total number of reassociations
	unsigned int kmax =
		static_cast<unsigned int>( ceil( T / ( nsteps * dt ) ) );

	// generate finite differences matrices
	sp_mat Ml;
	sp_mat Mi;
	sp_mat Mr;
	genFDMatrices( ip, l2, Ml, Mi, Mr );

	// allocate memory for the analytical solution
	vec exSol( np );
	// allocate memory for error values
	vec errorL2( kmax );

	// initial values / boundary conditions
	vec* F = new vec[ ndom ];
	vec* G = new vec[ ndom ];

	vec* W = new vec[ ndom ];
	vec* Z = new vec[ ndom ];

	// use pointers to vectors to be able to swap efficiently!
	vec** PW = new vec*[ ndom ];
	vec** PZ = new vec*[ ndom ];

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

	// allocate thread array
	std::thread* threads = new std::thread[ ndom ];

	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate as many time steps as possible before reassociating
		// start a thread for each calculation
		threads[ 0 ] = std::thread(
				evalMulti,
				std::ref( PZ[ 0 ] ),
				std::ref( PW[ 0 ] ),
				std::ref( Ml ),
				l2,
				dt2,
				nsteps );
		for( unsigned int i = 1; i < ndom - 1; ++i )
		{
			threads[ i ] = std::thread(
					evalMulti,
					std::ref( PZ[ i ] ),
					std::ref( PW[ i ] ),
					std::ref( Mi ),
					l2,
					dt2,
					nsteps );
		}
		threads[ ndom - 1 ] = std::thread(
				evalMulti,
				std::ref( PZ[ ndom - 1 ] ),
				std::ref( PW[ ndom - 1 ] ),
				std::ref( Mr ),
				l2,
				dt2,
				nsteps );

		// wait for all threads to complete
		for( unsigned int i = 0; i < ndom; ++i )
		{
			threads[ i ].join();
		}

		// reassociate numerical solution
		numSol.subvec( 0, ip * 3 / 4 - 1 ) =
			PZ[ 0 ]->subvec( 0, ip * 3 / 4 - 1 );
		for( unsigned int i = 1; i < ndom - 1; ++i )
		{
			numSol.subvec( ip * ( 2 * i + 1 ) / 4, ip * ( 2 * i + 3 ) / 4 - 1 ) =
				PZ[ i ]->subvec( ip / 4, ip * 3  / 4 - 1 );
		}
		numSol.subvec( ip * ( 2 * ndom - 1 ) / 4, ip * ( ndom + 1 ) / 2 - 1 ) =
			PZ[ ndom - 1 ]->subvec( ip / 4, ip - 1 );

		// copy numerically exact solution to boundaries of the subintervals
		PZ[ 0 ]->subvec( ip * 3 / 4, ip - 1 ) =
			PZ[ 1 ]->subvec( ip / 4, ip / 2 - 1 );
		PW[ 0 ]->subvec( ip * 3 / 4, ip - 1 ) =
			PW[ 1 ]->subvec( ip / 4, ip / 2 - 1 );
		for( unsigned int i = 1; i < ndom - 1; ++i )
		{
			PZ[ i ]->subvec( 0, ip / 4 - 1 ) =
				PZ[ i - 1 ]->subvec( ip / 2, ip * 3 / 4 - 1  );
			PW[ i ]->subvec( 0, ip / 4 - 1 ) =
				PW[ i - 1 ]->subvec( ip / 2, ip * 3 / 4 - 1  );
			PZ[ i ]->subvec( ip * 3 / 4, ip - 1 ) =
				PZ[ i + 1 ]->subvec( ip / 4, ip / 2 - 1 );
			PW[ i ]->subvec( ip * 3 / 4, ip - 1 ) =
				PW[ i + 1 ]->subvec( ip / 4, ip / 2 - 1 );
		}
		PZ[ ndom - 1 ]->subvec( 0, ip / 4 - 1 ) =
			PZ[ ndom - 2 ]->subvec( ip / 2, ip * 3 / 4 - 1 );
		PW[ ndom - 1 ]->subvec( 0, ip / 4 - 1 ) =
			PW[ ndom - 2 ]->subvec( ip / 2, ip * 3 / 4 - 1 );

		// calculate exact solution
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );
		// calculate L2 error
		vec error = exSol - numSol;
		errorL2( k ) = sqrt( h * dot( error, error ) );

		if( m_funOnReassociation != NULL )
		{
			m_funOnReassociation( k, kmax, x, numSol, exSol, errorL2( k ) );
		}
	}

	// optionally return analytical solution and L2 error
	if( exactSol != NULL )
	{
		*exactSol = exSol;
	}
	if( error != NULL )
	{
		*error = errorL2;
	}

	delete[] PW;
	delete[] PZ;
	delete[] Z;
	delete[] W;
	delete[] F;
	delete[] G;
	delete[] X;
	delete[] threads;
}

void ParallelSolver::genFDMatrices(
		unsigned int np,
		double l2,
		arma::sp_mat& left,
		arma::sp_mat& inner,
		arma::sp_mat& right )
{
	// number of entries (two secondary diagonals)
	unsigned int numEntries = 2 * ( np - 1 );
	// locations of entries
	umat locations( 2, numEntries );
	// values of entries
	vec values( numEntries );
	
	// left boundary matrix
	for( unsigned int i = 0; i < np - 1; ++i )
	{
		// upper diagonal
		locations( 0, 2 * i ) = i;
		locations( 1, 2 * i ) = i + 1;
		values( 2 * i ) = ( i == 0 ) ? 2 * l2 : l2;
		// lower diagonal
		locations( 0, 2 * i + 1 ) = i + 1;
		locations( 1, 2 * i + 1 ) = i;
		values( 2 * i + 1 ) = l2;
	}
	left = sp_mat( locations, values, np, np, true );

	// inner matrix (no boundary considerations)
	for( unsigned int i = 0; i < np - 1; ++i )
	{
		// upper diagonal
		locations( 0, 2 * i ) = i;
		locations( 1, 2 * i ) = i + 1;
		values( 2 * i ) = l2;
		// lower diagonal
		locations( 0, 2 * i + 1 ) = i + 1;
		locations( 1, 2 * i + 1 ) = i;
		values( 2 * i + 1 ) = l2;
	}
	inner = sp_mat( locations, values, np, np, true );

	// right boundary matrix
	for( unsigned int i = 0; i < np - 1; ++i )
	{
		// upper diagonal
		locations( 0, 2 * i ) = i;
		locations( 1, 2 * i ) = i + 1;
		values( 2 * i ) = l2;
		// lower diagonal
		locations( 0, 2 * i + 1 ) = i + 1;
		locations( 1, 2 * i + 1 ) = i;
		values( 2 * i + 1 ) = ( i == np - 2 ) ? 2 * l2 : l2;
	}
	right = sp_mat( locations, values, np, np, true );
}
