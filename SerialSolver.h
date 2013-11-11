#pragma once

#include "Solver.h"

class SerialSolver : public Solver
{
public:
	SerialSolver();
	virtual ~SerialSolver();

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
		arma::vec* error );

private:
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
		double d,
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
};

