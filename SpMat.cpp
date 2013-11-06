/*
 * SpMat.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#include "SpMat.h"
#include "Vec.h"
#include <stdexcept>
#include <cstdlib>

SpMat::SpMat( unsigned int m, unsigned int n )
	: m( m ), n( n )
{
}

SpMat::~SpMat()
{
}

void SpMat::addEntry( unsigned int i, unsigned int j, double val )
{
	if( i >= m || j >= n )
	{
		throw std::invalid_argument( "i or j out of range" );
	}

	entry e;
	e.i = i;
	e.j = j;
	e.val = val;
	entries.push_back( e );
}

const std::vector<SpMat::entry>& SpMat::getEntries() const
{
	return entries;
}

void SpMat::multiplyVector( const Vec& vec, Vec& res ) const
{
	const double* vecVal = vec.getValues();
	double* resVal = res.getValues();
	res.zeros();

	const entry* ptr = &entries[ 0 ];
	unsigned int size = entries.size();
	for( unsigned int i = 0; i < size; ++i )
	{
		resVal[ ptr->i ] += ptr->val * vecVal[ ptr->j ];
		++ptr;
	}
}

void SpMat::diag( double val, int offset )
{
	entries.clear();

	unsigned int minDim = ( m < n ) ? m : n;
	unsigned int numEntries = minDim - abs( offset );

	unsigned int i;
	unsigned int j;

	if( offset >= 0 )
	{
		i = 0;
		j = offset;
	}
	else
	{
		i = -offset;
		j = 0;
	}

	for( unsigned int k = 0; k < numEntries; ++k )
	{
		addEntry( i++, j++, val );
	}
}

