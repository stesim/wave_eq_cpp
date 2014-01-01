#pragma once
#include <cstddef>

class CudaHelper
{
public:
	/*
	 * Call the initialization CUDA kernel, calculating the initial values.
	 */
	static void callInitKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			double L,				// spacial domain radius
			double h,				// spacial discretization step size
			double dt,				// temporal discretization step size
			unsigned int fdDiags,	// number of diagonals of the FD-matrix
			const int* fdOffsets,	// diagonal offsets
			const double* fdValues,	// values of the diagonals
			double* Z,				// solution vector
			double* W,				// solution vector
			double* U );			// solution vector

	/*
	 * Call the main CUDA kernel, calculating 'nsteps' iterations.
	 */
	static void callMainKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			unsigned int nsteps,	// number of iterations before reassociation
			double l2,				// l^2 = dt^2/h^2
			unsigned int fdDiags,	// number of diagonals of the FD-matrix
			const int* fdOffsets,	// diagonal offsets
			const double* fdValues,	// values of the diagonals
			double* Z,				// solution vector
			double* W,				// solution vector
			double* U );			// solution vector

	/*
	 * Call the synchronization CUDA kernel, exchanging numerically exact
	 * solution parts between subdomains.
	 */
	static void callSyncKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			double* Z,				// solution vector
			double* W );			// solution vector

	/*
	 * Call the reassociation CUDA kernel, constructing the complete numerical
	 * solution from the solutions on the subdomains.
	 */
	static void callReassociationKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			const double* Z,		// subdomain solutions vector
			double* S );			// complete solution vector

	/*
	 * Allocate device memory.
	 */
	template<typename T>
	static T* allocDevMem( size_t numElem );

	/*
	 * Free device memory.
	 */
	template<typename T>
	static void freeDevMem( T* mem );

	/*
	 * Copy host memory to device memory.
	 */
	template<typename T>
	static void copyHostToDevMem( const T* hostMem, T* devMem, size_t numElem );

	/*
	 * Copy device memory to host memory.
	 */
	template<typename T>
	static void copyDevToHostMem( const T* devMem, T* hostMem, size_t numElem );

private:
	CudaHelper();
	~CudaHelper();
};
