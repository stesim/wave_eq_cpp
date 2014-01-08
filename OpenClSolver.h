#pragma once
#ifndef NO_CL

#include "Solver.h"
#include "OpenClContext.h"
#include <vector>

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
	 * Create FD matrix in sparse diagonal form.
	 */
	static void createSparseDiagFDMatrix(
			unsigned int n,
			double l2,
			std::vector<int>& offsets,
			std::vector<double>& values );

	/*
	 * Check OpenCL device capabilities.
	 */
	static bool checkClCapabilities( unsigned int np, unsigned int ns );

	/*
	 * Create OpenCL buffer from std::vector.
	 */
	template<typename T>
	inline static cl_mem createClBufferFromStdVector( std::vector<T>& vec )
	{
		return clCreateBuffer(
				s_clContext.context,
				CL_MEM_READ_WRITE,
				vec.size() * sizeof( T ),
				NULL,
				NULL );
	}

	/*
	 * Create OpenCL buffer from arma::vec.
	 */
	inline static cl_mem createClBufferFromArmaVec( arma::vec& vec )
	{
		return clCreateBuffer(
				s_clContext.context,
				CL_MEM_READ_WRITE,
				vec.n_elem * sizeof( double ),
				NULL,
				NULL );
	}

	/*
	 * Create OpenCL buffer and copy contents from std::vector.
	 */
	template<typename T>
	inline static cl_mem copyClBufferFromStdVector( std::vector<T>& vec )
	{
		return clCreateBuffer(
				s_clContext.context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				vec.size() * sizeof( T ),
				&vec[ 0 ],
				NULL );
	}

	/*
	 * Copy OpenCL buffer contents to std::vector.
	 */
	template<typename T>
	inline static bool copyClBufferToStdVector(
			cl_mem buf,
			std::vector<T>& vec,
			bool blocking = false )
	{
		cl_int err = clEnqueueReadBuffer(
				s_clContext.queue,
				buf,
				blocking,
				0,
				vec.size() * sizeof( T ),
				&vec[ 0 ],
				0,
				NULL,
				NULL );
		return ( err == CL_SUCCESS );
	}

	/*
	 * Copy OpenCL buffer contents to arma::vec.
	 */
	inline static bool copyClBufferToArmaVec(
			cl_mem buf,
			arma::vec& vec,
			bool blocking = false )
	{
		cl_int err = clEnqueueReadBuffer(
				s_clContext.queue,
				buf,
				blocking,
				0,
				vec.n_elem * sizeof( double ),
				&vec[ 0 ],
				0,
				NULL,
				NULL );
		return ( err == CL_SUCCESS );
	}

	/*
	 * Create OpenCL kernel.
	 */
	inline static cl_kernel createClKernel( const char* name );

	/*
	 * Call the initialization CUDA kernel, calculating the initial values.
	 */
	void callInitKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			double L,				// spacial domain radius
			double h,				// spacial discretization step size
			double dt,				// temporal discretization step size
			unsigned int fdDiags,	// number of diagonals of the FD-matrix
			cl_mem fdOffsets,	// diagonal offsets
			cl_mem fdValues,	// values of the diagonals
			cl_mem Z,				// solution vector
			cl_mem W,				// solution vector
			cl_mem U )				// solution vector
	{
		size_t globalWorkSize = blocks * threads;
		size_t localWorkSize = threads;

		clSetKernelArg( m_KernelInit, 0, sizeof( unsigned int ), &ip );
		clSetKernelArg( m_KernelInit, 1, sizeof( double ), &L );
		clSetKernelArg( m_KernelInit, 2, sizeof( double ), &h );
		clSetKernelArg( m_KernelInit, 3, sizeof( double ), &dt );
		clSetKernelArg( m_KernelInit, 4, sizeof( unsigned int ), &fdDiags );
		clSetKernelArg( m_KernelInit, 5, sizeof( cl_mem ), &fdOffsets );
		clSetKernelArg( m_KernelInit, 6, sizeof( cl_mem ), &fdValues );
		clSetKernelArg( m_KernelInit, 7, sizeof( cl_mem ), &Z );
		clSetKernelArg( m_KernelInit, 8, sizeof( cl_mem ), &W );
		clSetKernelArg( m_KernelInit, 9, sizeof( cl_mem ), &U );
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				m_KernelInit,
				1,
				nullptr,
				&globalWorkSize,
				&localWorkSize,
				0,
				nullptr,
				nullptr );
	}

	/*
	 * Call the main CUDA kernel, calculating 'nsteps' iterations.
	 */
	void callMainKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			unsigned int nsteps,	// number of iterations before reassociation
			double l2,				// l^2 = dt^2/h^2
			unsigned int fdDiags,	// number of diagonals of the FD-matrix
			cl_mem fdOffsets,	// diagonal offsets
			cl_mem fdValues,	// values of the diagonals
			cl_mem Z,				// solution vector
			cl_mem W,				// solution vector
			cl_mem U )				// solution vector
	{
		size_t globalWorkSize = blocks * threads;
		size_t localWorkSize = threads;

		clSetKernelArg( m_KernelMain, 0, sizeof( unsigned int ), &ip );
		clSetKernelArg( m_KernelMain, 1, sizeof( unsigned int ), &nsteps );
		clSetKernelArg( m_KernelMain, 2, sizeof( double ), &l2 );
		clSetKernelArg( m_KernelMain, 3, sizeof( unsigned int ), &fdDiags );
		clSetKernelArg( m_KernelMain, 4, sizeof( cl_mem ), &fdOffsets );
		clSetKernelArg( m_KernelMain, 5, sizeof( cl_mem ), &fdValues );
		clSetKernelArg( m_KernelMain, 6, sizeof( cl_mem ), &Z );
		clSetKernelArg( m_KernelMain, 7, sizeof( cl_mem ), &W );
		clSetKernelArg( m_KernelMain, 8, sizeof( cl_mem ), &U );
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				m_KernelMain,
				1,
				nullptr,
				&globalWorkSize,
				&localWorkSize,
				0,
				nullptr,
				nullptr );
	}

	/*
	 * Call the synchronization CUDA kernel, exchanging numerically exact
	 * solution parts between subdomains.
	 */
	void callSyncKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			cl_mem Z,				// solution vector
			cl_mem W )				// solution vector
	{
		size_t globalWorkSize = blocks * threads;
		size_t localWorkSize = threads;

		clSetKernelArg( m_KernelSync, 0, sizeof( unsigned int ), &ip );
		clSetKernelArg( m_KernelSync, 1, sizeof( cl_mem ), &Z );
		clSetKernelArg( m_KernelSync, 2, sizeof( cl_mem ), &W );
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				m_KernelSync,
				1,
				nullptr,
				&globalWorkSize,
				&localWorkSize,
				0,
				nullptr,
				nullptr );
	}

	/*
	 * Call the reassociation CUDA kernel, constructing the complete numerical
	 * solution from the solutions on the subdomains.
	 */
	void callReassociationKernel(
			unsigned int blocks,	// number of CUDA thread blocks
			unsigned int threads,	// number of CUDA threads per block
			unsigned int ip,		// number of points per subdomain
			cl_mem Z,		// subdomain solutions vector
			cl_mem S )				// complete solution vector
	{
		size_t globalWorkSize = blocks * threads;
		size_t localWorkSize = threads;

		clSetKernelArg( m_KernelSync, 0, sizeof( unsigned int ), &ip );
		clSetKernelArg( m_KernelSync, 1, sizeof( cl_mem ), &Z );
		clSetKernelArg( m_KernelSync, 2, sizeof( cl_mem ), &S );
		clEnqueueNDRangeKernel(
				s_clContext.queue,
				m_KernelReassociate,
				1,
				nullptr,
				&globalWorkSize,
				&localWorkSize,
				0,
				nullptr,
				nullptr );
	}

private:
	static OpenClContext s_clContext;

	cl_kernel m_KernelInit;
	cl_kernel m_KernelMain;
	cl_kernel m_KernelSync;
	cl_kernel m_KernelReassociate;
};

#endif
