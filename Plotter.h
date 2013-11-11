#pragma once

#include <armadillo>
#include <Python.h>

class Plotter
{
public:
	~Plotter();

	static void initialize();
	static void finalize();

	static void plot( arma::vec& x, arma::mat& y );

private:
	Plotter( );

private:
	static PyObject* s_pymodPyplot;
	static PyObject* s_pyfunPlot;
	static PyObject* s_pyfunShow;
};

