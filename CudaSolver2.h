#pragma once
#ifdef BLUB

#include "Solver.h"

class CudaSolver2 : public Solver
{
public:
	CudaSolver2();
	virtual ~CudaSolver2();

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
	 * Create FD matrix in sparse diagonal form.
	 */
	static void createSparseDiagFDMatrix(
			unsigned int n,
			double l2,
			std::vector<int>& offsets,
			std::vector<double>& values );
};

#endif
