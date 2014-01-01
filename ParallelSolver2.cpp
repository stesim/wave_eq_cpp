#include "ParallelSolver2.h"
#include "ThreadPool.h"
#include <vector>

using namespace arma;

ParallelSolver2::ParallelSolver2()
{
}

ParallelSolver2::~ParallelSolver2()
{
}


void ParallelSolver2::solve(
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
	// overlapping intervals
	std::vector<arma::vec> X( ndom );
	for( unsigned int i = 0; i < ndom - 1; ++i )
	{
		X[ i ] = x.subvec( ip * i / 2, ip * ( i + 2 ) / 2 - 1 );
	}
	X[ ndom - 1 ] = arma::vec( ip );
	X[ ndom - 1 ].subvec( 0, ip / 2 - 1 ) = x.subvec( np - ip / 2, np - 1 );
	X[ ndom - 1 ].subvec( ip / 2, ip - 1 ) = x.subvec( 0, ip / 2 - 1 );

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
	arma::sp_mat Ml;
	arma::sp_mat Mi;
	arma::sp_mat Mr;
	genFDMatrices( ip, l2, Ml, Mi, Mr );

	// allocate memory for the analytical solution
	arma::vec exSol( np );
	// allocate memory for error values
	arma::vec errorL2( kmax );

	// initial values / boundary conditions
	std::vector<arma::vec> F( ndom );
	std::vector<arma::vec> G( ndom );

	std::vector<arma::vec> W( ndom );
	std::vector<arma::vec> Z( ndom );

	// use pointers to vectors to be able to swap efficiently!
	std::vector<arma::vec*> PW( ndom );
	std::vector<arma::vec*> PZ( ndom );

	// initialize arrays allocated above
	for( unsigned int i = 0; i < ndom; ++i )
	{
		// calculate values for F, G
		F[ i ] = arma::vec( ip );
		arrayfun( funu0, X[ i ], F[ i ] );
		G[ i ] = arma::vec( ip );
		arrayfun( funu1, X[ i ], G[ i ] );

		// assign initial values to W, Z
		W[ i ] = F[ i ];
		Z[ i ] = arma::vec( ip );

		// get pointers to solution vectors
		PW[ i ] = &W[ i ];
		PZ[ i ] = &Z[ i ];
	}

	// calculate one step back in time
	for( unsigned int i = 0; i < ndom; ++i )
	{
		evalRhs( F[ i ], G[ i ], Mi, l2, dt2, 1.0, 0.5, -dt, Z[ i ] );
	}

	
		for( unsigned int i = 0; i < ip / 16; ++i )
		{
			std::cout << W[ (1<<7) ][ i ] << '\t';
		}
		std::cout << "--------" << std::endl;
		for( unsigned int i = 0; i < ip / 16; ++i )
		{
			std::cout << Z[ (1<<7) ][ i ] << '\t';
		}
		std::cout << "--------" << std::endl;

	numSol = arma::vec( np );

	// create thread pool for subdomain calculations
	ThreadPool pool( &ParallelSolver2::solveSubdomain );

	// generate parameters and add them to the thread pool
	std::vector<Params> params( ndom );
	for( unsigned int i = 0; i < params.size(); ++i )
	{
		Params& p = params[ i ];
		p.pz = &PZ[ i ];
		p.pw = &PW[ i ];
		p.M = &Mi;
		p.l2 = l2;
		p.dt2 = dt2;
		p.nsteps = nsteps;

		pool.addThreadArgs( &p );
	}

	for( unsigned int k = 0; k < kmax; ++k )
	{
		// calculate as many time steps as possible before reassociating
		pool.run( false );

		// reassociate numerical solution
		numSol.subvec( 0, ip / 4 - 1 ) =
			PZ[ ndom - 1 ]->subvec( ip / 2, ip * 3 / 4 - 1 );
		for( unsigned int i = 0; i < ndom - 1; ++i )
		{
			numSol.subvec( ip * ( 2 * i + 1 ) / 4, ip * ( 2 * i + 3 ) / 4 - 1 ) =
				PZ[ i ]->subvec( ip / 4, ip * 3  / 4 - 1 );
		}
		numSol.subvec( np - ip / 2, np - 1 ) =
			PZ[ ndom - 1 ]->subvec( ip / 4, ip / 2 - 1 );

		// copy numerically exact solution to boundaries of the subintervals
		PZ[ 0 ]->subvec( 0, ip / 4 - 1 ) =
			PZ[ ndom - 1 ]->subvec( ip / 2, ip * 3 / 4 - 1 );
		PZ[ 0 ]->subvec( ip * 3 / 4, ip - 1 ) =
			PZ[ 1 ]->subvec( ip / 4, ip / 2 - 1 );
		PW[ 0 ]->subvec( 0, ip / 4 - 1 ) =
			PW[ ndom - 1 ]->subvec( ip / 2, ip * 3 / 4 - 1 );
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
		PZ[ ndom - 1 ]->subvec( ip * 3 / 4, ip - 1 ) =
			PZ[ 0 ]->subvec( ip / 4, ip / 2 - 1 );
		PW[ ndom - 1 ]->subvec( 0, ip / 4 - 1 ) =
			PW[ ndom - 2 ]->subvec( ip / 2, ip * 3 / 4 - 1 );
		PW[ ndom - 1 ]->subvec( ip * 3 / 4, ip - 1 ) =
			PW[ 0 ]->subvec( ip / 4, ip / 2 - 1 );

		// calculate exact solution
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );
		//arrayfun2( funsol, x, 0.0, exSol );
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
}

void ParallelSolver2::genFDMatrices(
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

void ParallelSolver2::solveSubdomain( void* args )
{
	Params& params = *static_cast<Params*>( args );
	Solver::evalMulti(
			*params.pz,
			*params.pw,
			*params.M,
			params.l2,
			params.dt2,
			params.nsteps );
}
