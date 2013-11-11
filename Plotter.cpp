#include "Plotter.h"
#include <iostream>
#include <cstdlib>

PyObject* Plotter::s_pymodPyplot = NULL;
PyObject* Plotter::s_pyfunPlot = NULL;
PyObject* Plotter::s_pyfunShow = NULL;

Plotter::Plotter()
{
}


Plotter::~Plotter()
{
}

void Plotter::initialize()
{
	PyObject* pyplotName = PyUnicode_FromString( "matplotlib.pyplot" );
	s_pymodPyplot = PyImport_Import( pyplotName );
	if( PyErr_Occurred() != NULL )
	{
		PyErr_Print();
		return;
	}
	Py_DECREF( pyplotName );

	s_pyfunPlot = PyObject_GetAttrString( s_pymodPyplot, "plot" );
	s_pyfunShow = PyObject_GetAttrString( s_pymodPyplot, "show" );
}

void Plotter::finalize()
{
	Py_XDECREF( s_pyfunShow );
	Py_XDECREF( s_pyfunPlot );
	Py_XDECREF( s_pymodPyplot );
}

void Plotter::plot( arma::vec& x, arma::mat& y )
{
	/*
	x.save( "plot_data_x.dat", arma::raw_ascii );
	y.save( "plot_data_y.dat", arma::raw_ascii );

	std::system( "python plot.py" );
	*/

	PyObject* py_x = PyList_New( x.size() );
	for( unsigned int i = 0; i < x.size(); ++i )
	{
		PyList_SetItem( py_x, i, PyFloat_FromDouble( x( i ) ) );
	}
	PyObject* py_y;
	if( y.n_cols > 1 )
	{
		py_y = PyList_New( y.n_cols );
		for( unsigned int j = 0; j < y.n_cols; ++j )
		{
			PyObject* sublist = PyList_New( y.n_rows );
			for( unsigned int i = 0; i < y.n_rows; ++i )
			{
				PyList_SetItem( sublist, i, PyFloat_FromDouble( y( i, j ) ) );
			}
			PyList_SetItem( py_y, j, sublist );
		}
	}
	else
	{
		py_y = PyList_New( y.n_rows );
		for( unsigned int i = 0; i < y.n_rows; ++i )
		{
			PyList_SetItem( py_y, i, PyFloat_FromDouble( y( i, 1 ) ) );
		}
	}

	
	PyObject* plotArgs = PyTuple_New( 2 );
	PyTuple_SetItem( plotArgs, 0, py_x );
	PyTuple_SetItem( plotArgs, 1, py_y );
	
	PyObject_CallObject( s_pyfunPlot, plotArgs );
	Py_DECREF( plotArgs );

	PyObject_CallObject( s_pyfunShow, NULL );
}
