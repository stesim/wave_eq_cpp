#pragma once

#include "Solver.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

class CudaSolver : public Solver
{
public:
	CudaSolver();
	virtual ~CudaSolver();

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
	 * Call the CUDA kernel, calculating 'nsteps' iterations.
	 */
	static void callKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// points per subdomain
			unsigned int nsteps,	// number of iterations before reassociation
			double a,				// 2*(1-l^2) = 2*(1-dt^2/h^2)
			unsigned int fdDiags,	// number of diagonals of the FD-matrix
			int* fdOffsets,			// diagonal offsets
			double* fdValues,		// values of the diagonals
			double* Z,				// solution vector
			double* W,				// solution vector
			double* U );			// solution vector

	/*
	 * Create FD matrix in sparse diagonal form.
	 */
	static void createSparseDiagFDMatrix(
			unsigned int n,
			double l2,
			std::vector<int>& offsets,
			std::vector<double>& values );
};
