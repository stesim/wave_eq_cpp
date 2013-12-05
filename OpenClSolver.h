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

private:
	static OpenClContext s_clContext;
};

#endif
