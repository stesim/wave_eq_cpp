#pragma once
#ifndef NO_CL

#include "Solver.h"
#include "OpenClContext.h"

class OpenClSolver : public Solver
{
private:

public:
	OpenClSolver();
	virtual ~OpenClSolver();

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
	 * Check OpenCL device capabilities.
	 */
	static bool checkClCapabilities( unsigned int np, unsigned int ns );

	/*
	 * Create OpenCL buffer.
	 */
	static cl_mem createClBuffer(
			cl_mem_flags flags,
			size_t size,
			void* host_ptr );

private:
	static OpenClContext s_clContext;
};

#endif
