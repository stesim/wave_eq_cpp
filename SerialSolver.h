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
};

