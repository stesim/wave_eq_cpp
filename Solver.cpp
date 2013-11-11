#include "Solver.h"

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
