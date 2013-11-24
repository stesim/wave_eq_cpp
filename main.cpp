#include <iostream>
#include <string>
#include <armadillo>
#include <Python.h>
#include "SerialSolver.h"
#include "ParallelSolver.h"
#include "WallTimer.h"
#include "CpuTimer.h"
#include <thread>
#include "InputHelper.h"

using namespace arma;

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

PyObject* pyListFromArmaVec( const arma::vec& vec )
{
	PyObject* list = PyList_New( vec.size() );
	for( unsigned int i = 0; i < vec.size(); ++i )
	{
		PyList_SetItem( list, i, PyFloat_FromDouble( vec( i ) ) );
	}
	return list;
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
		<< " (" << 100 * ( step + 1 ) / numSteps << "%)"
		<< "; Current L2 error: " << error << std::endl;
}

void plotResults(
		const arma::vec& x,
		const arma::vec& numSol,
		const arma::vec& exSol,
		const arma::vec& error )
{
	PyObject* px = pyListFromArmaVec( x );
	PyObject* pnumSol = pyListFromArmaVec( numSol );
	PyObject* pexSol = pyListFromArmaVec( exSol );
	PyObject* perror = pyListFromArmaVec( error );
	
	PyObject* pyplotName = PyUnicode_FromString( "plot" );
	PyObject* modPyplot = PyImport_Import( pyplotName );
	if( PyErr_Occurred() != NULL )
	{
		PyErr_Print();
		return;
	}
	Py_DECREF( pyplotName );

	PyObject* funPlot = PyObject_GetAttrString( modPyplot, "plot" );
	
	PyObject* plotArgs = PyTuple_New( 6 );
	PyTuple_SetItem( plotArgs, 0, PyLong_FromLong( 0 ) );
	PyTuple_SetItem( plotArgs, 1, PyLong_FromLong( 0 ) );
	PyTuple_SetItem( plotArgs, 2, px );
	PyTuple_SetItem( plotArgs, 3, pnumSol );
	PyTuple_SetItem( plotArgs, 4, pexSol );
	PyTuple_SetItem( plotArgs, 5, perror );

	PyObject_CallObject( funPlot, plotArgs );
	Py_DECREF( plotArgs );
	Py_DECREF( funPlot );
	Py_DECREF( modPyplot );
}

int main( int argc, char* argv[] )
{
	// Python initialization
	Py_SetProgramName( (wchar_t*)argv[ 0 ] );
	Py_Initialize();
	PyRun_SimpleString( "import sys\nsys.path.append(\".\")\n" );

	std::cout << "Enter input parameters." << std::endl;

	// spacial domain size (spanning from -L to L)
	double L = inputParam<double>( "L", 150.0 );
	// spacial discretization point count exponent (number of points = 2^N)
	unsigned int N = inputParam<unsigned int>( "N", 14 );
	// processor count exponent (number of processes = 2^n)
	unsigned int n = inputParam<unsigned int>( "n", 7 );
	// temporal domain length
	double T = inputParam<double>( "T", 100.0 );
	// solver type
	bool useParallelSolver = inputParam<bool>( "useParallelSolver", true );

	// initialize solver
	Solver& solver = *( useParallelSolver
			? (Solver*)new ParallelSolver() : (Solver*)new SerialSolver() );
	// assign a function to be called on each reassociation
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

	// plot results using python
	plotResults( x, numSol, exSol, error );

#ifdef _WIN32
	// require key-press to exit (Windows only)
	std::cout << "Press Enter to exit." << std::endl;
	std::cin.get();
#endif

	Py_Finalize();

	delete &solver;
	return 0;
}

