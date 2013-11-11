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
#include "WallTimer.h"
#include "CpuTimer.h"

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
	std::cout << "k / kmax: " << step + 1 << " / " << numSteps
		<< " (" << 100 * ( step + 1 ) / numSteps << "%) "
		<< "; Current L2 error: " << error << std::endl;
}

int main( int argc, char* argv[] )
{
	Py_SetProgramName( (wchar_t*)argv[ 0 ] );
	Py_Initialize();
	Plotter::initialize();

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

	WallTimer wallTimer;
	CpuTimer cpuTimer;
	
	arma::vec x;
	arma::vec numSol;
	arma::vec exSol;
	arma::vec error;

	wallTimer.start();
	cpuTimer.start();

	solver.solve( L, N, n, T, funu0, funu1, funsol, x, numSol, &exSol, &error );

	wallTimer.stop();
	cpuTimer.stop();

	std::cout << "Calculation completed in " << wallTimer.getElapsedTime()
		<< "s wall time, or " << cpuTimer.getElapsedTime()
		<< "s cpu time respectively." << std::endl;

	Plotter::plot( x, numSol );

#ifdef WIN32
	// require key-press to exit (Windows only)
	std::cout << "Press Enter to exit." << std::endl;
	std::cin.get();
#endif

	Plotter::finalize();
	Py_Finalize( );
	return 0;
}

