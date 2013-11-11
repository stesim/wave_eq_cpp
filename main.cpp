/*
 * main.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: stefan
 */

#include <iostream>
#include <sstream>
#include <string>
#include <armadillo>
#include <Python.h>
#include "SerialSolver.h"
#include "Plotter.h"

using namespace arma;

/*
* Input helper function.
*/
template<typename T>
T inputParam( const char* name, T defVal )
{
	std::cout << name << " [" << defVal << "]: ";
	T val;
	std::string input;
	std::getline( std::cin, input );
	if( !input.empty() )
	{
		std::istringstream stream( input );
		stream >> val;
	}
	else
	{
		val = defVal;
	}
	/*
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
	*/
	return val;
}

/*
* Initial value function.
*/
double funu0( double x )
{
	return 1 / ( 1 + x * x );
}

/*
* Neumann boundary condition.
*/
double funu1( double x )
{
	return 0.0;
}

/*
* Exact solution.
*/
double funsol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1 / ( 1 + a * a ) + 1 / ( 1 + b * b ) ) / 2.0;
}

void onReassociate(
	unsigned int step,
	unsigned int numSteps,
	const arma::vec& x,
	const arma::vec& numSol,
	const arma::vec& exSol,
	double error )
{
	std::cout << step + 1 << " / " << numSteps << " : " << error << std::endl;
}

int main( int argc, char* argv[] )
{
	Py_SetProgramName( (wchar_t*)argv[ 0 ] );
	Py_Initialize();
	Plotter::initialize();

	//PyRun_SimpleString( "import numpy\n" );
	//PyRun_SimpleString( "import matplotlib.pyplot as plt\nplt.plot([0,1,2],[2,0,5])\nplt.show()\n" );
	//PyRun_SimpleString( "import sys\nprint(sys.getfilesystemencoding())" );

	std::cout << "Enter input parameters." << std::endl;

	// spacial domain size (spanning from -L to L)
	double L = inputParam<double>( "L", 150.0 );
	// spacial discretization point count exponent (number of points = 2^N)
	unsigned int N = inputParam<unsigned int>( "N", 14 );
	// processor count exponent (number of processes = 2^n)
	unsigned int n = inputParam<unsigned int>( "n", 7 );
	// temporal domain length
	double T = inputParam<double>( "T", 100.0 );

	Solver& solver = *new SerialSolver();
	// assign a function to be called on each reassociation (e.g. for plotting)
	solver.onReassociation( onReassociate );

	arma::vec x;
	arma::vec numSol;
	arma::vec exSol;
	arma::vec error;
	solver.solve( L, N, n, T, funu0, funu1, funsol, x, numSol, &exSol, &error );

	Plotter::plot( x, numSol );

	// require key-press to exit
	std::cout << "Press Enter to exit." << std::endl;
	std::cin.get();

	Plotter::finalize();
	Py_Finalize( );
	return 0;
}

