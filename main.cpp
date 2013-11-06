/*
 * main.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#include <iostream>
#include <armadillo>

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

	return 0;
}

