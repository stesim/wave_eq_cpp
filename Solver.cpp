#include "Solver.h"
#include <iostream>

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
