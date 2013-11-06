/*
 * Vec.cpp
 *
 *  Created on: Nov 6, 2013
 *      Author: stefan
 */

#include "Vec.h"
#include <cstring>
#include <iostream>

Vec::Vec( unsigned int n )
	: val( new double[ n ] ), n( n )
{
}

Vec::~Vec()
{
	delete[] val;
}

double* Vec::getValues()
{
	return val;
}

const double* Vec::getValues() const
{
	return val;
}

unsigned int Vec::getSize() const
{
	return n;
}

void Vec::zeros()
{
	memset( val, 0, n * sizeof( double ) );
}

void Vec::print()
{
	for( unsigned int i = 0; i < n; ++i )
	{
		std::cout << val[ i ] << std::endl;
	}
}

