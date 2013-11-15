#include "SerialSolver.h"

using namespace arma;

SerialSolver::SerialSolver()
{
}


SerialSolver::~SerialSolver()
{
}

void SerialSolver::solve(
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
	// number of (disjoint) subdomains (= number of processors)
	unsigned int ns = 1 << n;

	// spacial discretization
	x = linspace( -L, L, np );
	// discretization spacing
	double h = x( 1 ) - x( 0 );

	// temporal discretization step size
	double dt = h / 4.0;
	// helper variables
	double dt2 = dt * dt;
	double l = dt / h;
	double l2 = l * l;

	// points per subdomain
	unsigned int ip = np / ns;
	// time steps between solution reassociations
	unsigned int nsteps = ip / 2;
	// total number of reassociations
	unsigned int kmax =
		static_cast<unsigned int>( ceil( T / ( nsteps * dt ) ) );

	// generate finite differences matrix
	sp_mat M = l2 * genFDMatrix( np );
	// calculate initial values / boundary conditions
	vec f( np );
	arrayfun( funu0, x, f );
	vec g( np );
	arrayfun( funu1, x, g );
	// allocate memory for the analytical solution
	vec exSol( np );
	// allocate memory for error values
	vec errorL2( kmax );

	vec w( f );
	// calculate one step back in time
	vec z( np );
	evalRhs( f, g, M, l2, dt2, 1.0, 0.5, -dt, z );

	// use ONLY pointer to vectors from now on to be able to swap efficiently!
	vec* pw = &w;
	vec* pz = &z;
	for( unsigned int k = 0; k < kmax; ++k )
	{
		evalMulti( pz, pw, M, l2, dt2, nsteps );
		arrayfun2( funsol, x, ( k + 1 ) * nsteps * dt, exSol );
		// *pz == z is not necessarily true, thus the solution vector is *pz
		vec error = *pz - exSol;
		errorL2( k ) = sqrt( h * dot( error, error ) );

		if( m_funOnReassociation != NULL )
		{
			m_funOnReassociation( k, kmax, x, *pz, exSol, errorL2( k ) );
		}
	}

	// return numerical solution and optionally analytical solution and L2 error
	numSol = *pz;
	if( exactSol != NULL )
	{
		*exactSol = exSol;
	}
	if( error != NULL )
	{
		*error = errorL2;
	}
}
