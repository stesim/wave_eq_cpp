#include "Solver.h"

#define EVAL_RHS(z, w, M, l2, dt2, a, b, c, u ) u = ( 1 - l2 ) * z * a + M * z * b - w * c

using namespace arma;

Solver::Solver()
	: m_funOnReassociation( NULL )
{
}


Solver::~Solver()
{
}


void Solver::onReassociation( ReassociationCallback func )
{
	m_funOnReassociation = func;
}

sp_mat Solver::genFDMatrix( unsigned int np )
{
	// number of entries (two secondary diagonals)
	unsigned int numEntries = 2 * ( np - 1 );
	// locations of entries
	umat locations( 2, numEntries );
	// values of entries
	vec values( numEntries );
	for( unsigned int i = 0; i < np - 1; ++i )
	{
		// upper diagonal
		locations( 0, 2 * i ) = i;
		locations( 1, 2 * i ) = i + 1;
		values( 2 * i ) = ( i == 0 ) ? 2 : 1;
		// lower diagonal
		locations( 0, 2 * i + 1 ) = i + 1;
		locations( 1, 2 * i + 1 ) = i;
		values( 2 * i + 1 ) = ( i == np - 2 ) ? 2 : 1;
	}
	return sp_mat( locations, values, np, np, true );
}

void Solver::arrayfun( double( *f )( double ), const vec& x, vec& val )
{
	for( unsigned int i = 0; i < x.size( ); ++i )
	{
		val( i ) = f( x( i ) );
	}
}

void Solver::arrayfun2(
	double( *f )( double, double ),
	const vec& x,
	double t,
	vec& val )
{
	for( unsigned int i = 0; i < x.size( ); ++i )
	{
		val( i ) = f( x( i ), t );
	}
}

/*
* Calculate one time step.
*/
void Solver::evalRhs(
	const vec& z,
	const vec& w,
	const sp_mat& M,
	double l2,
	double dt2,
	double a,
	double b,
	double c,
	vec& u )
{
	u = ( 1 - l2 ) * z * a + M * z * b - w * c;
}

/*
* Calculate all time steps until next reassociation.
*/
void Solver::evalMulti(
	vec*& pz,
	vec*& pw,
	sp_mat& M,
	double l2,
	double dt2,
	unsigned int nsteps )
{
	vec* swap;
	for( unsigned int k = 0; k < nsteps; ++k )
	{
#ifdef OPTI_MAX
		EVAL_RHS( *pz, *pw, M, l2, dt2, 2.0, 1, 1, *pw );
#else
		evalRhs( *pz, *pw, M, l2, dt2, 2.0, 1, 1, *pw );
#endif
		swap = pw;
		pw = pz;
		pz = swap;
	}
}
