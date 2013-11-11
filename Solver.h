#pragma once

#include <armadillo>

class Solver
{
public:
	typedef void( *ReassociationCallback )(
		unsigned int step,
		unsigned int numSteps,
		const arma::vec& x,
		const arma::vec& numSol,
		const arma::vec& exSol,
		double error );
	typedef double( *SpacialFunction )( double );
	typedef double( *SpaciotemporalFunction )( double, double );

public:
	Solver();
	virtual ~Solver();

	virtual void solve(
		double L,
		unsigned int N,
		unsigned int n,
		double T,
		SpacialFunction funu0,
		SpacialFunction funu1,
		SpaciotemporalFunction funsol,
		arma::vec& x,
		arma::vec& numSol,
		arma::vec* exactSol,
		arma::vec* error ) = 0;

	virtual void onReassociation( ReassociationCallback func );

protected:
	/*
	* Generate the finite differences matrix with split off main diagonal.
	*/
	static arma::sp_mat genFDMatrix( unsigned int np );

	/*
	* Evaluate function at multiple points.
	*/
	static void arrayfun(
		SpacialFunction f,
		const arma::vec& x,
		arma::vec& val );

	/*
	* Evaluate time dependant function at multiple points.
	*/
	static void arrayfun2(
		SpaciotemporalFunction f,
		const arma::vec& x,
		double t,
		arma::vec& val );

protected:
	ReassociationCallback m_funOnReassociation;
};

