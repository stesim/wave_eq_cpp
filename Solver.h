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

	/*
	* Calculate one time step.
	*/
	static void evalRhs(
		const arma::vec& z,
		const arma::vec& w,
		const arma::sp_mat& M,
		double l2,
		double dt2,
		double a,
		double b,
		double c,
		arma::vec& u );

	/*
	* Calculate all time steps until next reassociation.
	*/
	static void evalMulti(
		arma::vec*& pz,
		arma::vec*& pw,
		arma::sp_mat& M,
		double l2,
		double dt2,
		unsigned int nsteps );

protected:
	ReassociationCallback m_funOnReassociation;
};

