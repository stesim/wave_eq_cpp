#include "CudaSolver.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__
void kernel(
		unsigned int ip,
		unsigned int nsteps,
		double a,
		unsigned int fdDiags,
		int* fdOffsets,
		double* fdValues,
		double* Z,
		double* W,
		double* U )
{
}

void CudaSolver::callKernel(
		unsigned int blocks,
		unsigned int threads,
		unsigned int ip,
		unsigned int nsteps,
		double a,
		unsigned int fdDiags,
		int* fdOffsets,
		double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	kernel<<<blocks, threads>>>(
			ip,
			nsteps,
			a,
			fdDiags,
			fdOffsets,
			fdValues,
			Z,
			W,
			U );
}
