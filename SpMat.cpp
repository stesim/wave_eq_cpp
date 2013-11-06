/*
 * SpMat.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#include "SpMat.h"

SpMat::SpMat()
	: m( 0 ), n( 0 )
{
}

SpMat::~SpMat()
{
}

void SpMat::addEntry( int i, int j, double val )
{
	entry e;
	e.i = i;
	e.j = j;
	e.val = val;
	entries.push_back( e );

	if( i >= m )
	{
		m = i + 1;
	}
	if( j >= n )
	{
		n = j + 1;
	}
}

const std::vector<SpMat::entry>& SpMat::getEntries()
{
	return entries;
}

