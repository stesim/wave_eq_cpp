/*
 * main.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#include <iostream>
//#include <armadillo>
#include "SpMat.h"
#include "Vec.h"

/*
void evalRhs(
		arma::vec z,
		arma::vec w,
		arma::SpMat<double>& M,
		double l2,
		double dt2,
		double a,
		double b,
		double c,
		double d
		arma::vec u )
{
}
*/

template<typename T>
T inputParam( const char* name, T defVal )
{
	std::cout << name << " (" << defVal << "): ";
	T val;
	while( true )
	{
	    if( std::cin.peek() == '\n' )
	    {
	    	std::cin.get();

	        val = defVal;
	        break;
	    }
	    else if ( std::cin >> val )
	    {
	    	break;
	    }
	    else
	    {
	    	std::cout << "Invalid input." << std::endl;
			std::cin.sync();
			std::cin.clear();
	    }
	}
	return val;
}

int main( int argc, char* argv[] )
{
	std::cout << "Enter input parameters." << std::endl;
	double L = inputParam<double>( "L", 150.0 );
	int N = inputParam<int>( "N", 14 );
	int n = inputParam<int>( "n", 7 );
	double T = inputParam<double>( "T", 100.0 );

	SpMat M( 3, 3 );
	M.diag( 2.0, 1 );
	Vec v( 3 );
	v.getValues()[ 0 ] = 1.0;
	v.getValues()[ 1 ] = 2.0;
	v.getValues()[ 2 ] = 3.0;

	v.print();

	Vec w( 3 );
	M.multiplyVector( v, w );

	w.print();

	return 0;
}

