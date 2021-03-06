#pragma once

#include "Solver.h"

class ParallelSolver : public Solver
{
private:
	struct Params
	{
		arma::vec** pz;
		arma::vec** pw;
		arma::sp_mat* M;
		double l2;
		double dt2;
		unsigned int nsteps;
	};

public:
	ParallelSolver();
	virtual ~ParallelSolver();

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
	* Generate the finite differences matrix with split off main diagonal.
	*/
	static void genFDMatrices(
			unsigned int np,
			double l2,
			arma::sp_mat& left,
			arma::sp_mat& center,
			arma::sp_mat& right );

	static void solveSubdomain( void* args );
};
