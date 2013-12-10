#include "CudaHelper.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define RUNTIME_CUDA
#include "wave_eq_func.h"

__device__
double* diaMulVec(
		unsigned int n,
		unsigned int diaDiags,
		const int* diaOffsets,
		const double* diaValues,
		const double* vec,
		double* res )
{
	// set result vector to zero
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = 0.0;
	}
	for( unsigned int k = 0; k < diaDiags; ++k )
	{
		// determine first vector index involved in the diagonal multiplication
		unsigned int vecIndex = fmin( 0.0, diaOffsets[ k ] );
		// detemine the number of elements in the diagonal
		unsigned int diagSize = n - abs( diaOffsets[ k ] );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			res[ vecIndex + i ] += diaValues[ i ] * vec[ vecIndex + i ];
		}
		// increment values pointer to next diagonal
		diaValues += diagSize;
	}
	return res;
}

__device__
double* vecAddScaledVec(
		unsigned int n,
		double* u,
		double c,
		const double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] += c * v[ i ];
	}
	return u;
}

__device__
double* vecAddScaledVecs(
		unsigned int n,
		double c,
		const double* u,
		double d,
		const double* v,
		double* res )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = c * u[ i ] + d * v[ i ];
	}
	return res;
}

__global__
void mainKernel(
		unsigned int ip,
		unsigned int nsteps,
		double l2,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	// determine index of the first element of the subdomain
	unsigned int vecIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];
	double* u = &U[ vecIndex ];

	double a = 2 * ( 1 - l2 );
	double* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		// u = M * z
		diaMulVec( ip, fdDiags, fdOffsets, fdValues, z, u );
		// u = u + a * z = M * z + a * z
		vecAddScaledVec( ip, u, a, z );
		// u = u + (-w) = M * z + a * z - w
		vecAddScaledVec( ip, u, -1.0, z );

		// shuffle buffers to avoid copying
		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__global__
void initKernel(
		unsigned int ip,
		double L,
		double h,
		double dt,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	// determine index of the first element of the subdomain
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIndex = id * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];
	double* u = &U[ vecIndex ];

	double x0 = h * ip * id / 2 - L;
	for( unsigned int i = 0; i < ip; ++i )
	{
		double x = x0 + i * h;
		w[ i ] = funu0( x );
		u[ i ] = funu1( x );
	}

	double a = 1 - dt / h * dt / h;
	// z = M * w
	diaMulVec( ip, fdDiags, fdOffsets, fdValues, w, z );
	// z = 0.5 * z + a * w = 0.5 * M * w + a * w
	vecAddScaledVecs( ip, 0.5, z, a, w, z );
	// z = z + dt * u
	vecAddScaledVec( ip, z, dt, u );
}

__global__
void syncKernel(
		unsigned int ip,
		double* Z,
		double* W )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int n_1 = gridDim.x * blockDim.x - 1;
	unsigned int vecIndex = id * ip;
	double* z = &Z[ vecIndex ];
	double* w = &W[ vecIndex ];

	// determine indices of left and right neighbor subdomains, considering a
	// periodical continuation
	unsigned int leftNeighbor;
	unsigned int rightNeighbor;
	if( id == 0 )
	{
		leftNeighbor = n_1;
		rightNeighbor = 1;
	}
	else if( id == n_1 )
	{
		leftNeighbor = n_1 - 1;
		rightNeighbor = 0;
	}
	else
	{
		leftNeighbor = id - 1;
		rightNeighbor = id + 1;
	}

	// copy exact data from left neighbors
	double* nz = &Z[ leftNeighbor * ip ];
	double* nw = &W[ leftNeighbor * ip ];
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ i ] = nz[ ip / 2 + i ];
		w[ i ] = nw[ ip / 2 + i ];
	}
	// copy exact data from right neighbors
	nz = &Z[ rightNeighbor * ip ];
	nw = &W[ rightNeighbor * ip ];
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ ip * 3 / 4 + i ] = nz[ ip / 4 + i ];
		w[ ip * 3 / 4 + i ] = nw[ ip / 4 + i ];
	}
}

__global__
void reassociationKernel(
		unsigned int ip,
		const double* Z,
		double* S )
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int vecIndex = id * ip;
	const double* z = &Z[ vecIndex ];
	double* s = &S[ vecIndex / 2 ];

	// copy left half of the subdomain solution
	for( unsigned int i = 0; i < ip / 2; ++i )
	{
		s[ i ] = z[ i ];
	}
}

void CudaHelper::callMainKernel(
		unsigned int blocks,
		unsigned int threads,
		unsigned int ip,
		unsigned int nsteps,
		double l2,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	mainKernel<<<blocks, threads>>>(
			ip,
			nsteps,
			l2,
			fdDiags,
			fdOffsets,
			fdValues,
			Z,
			W,
			U );
}

void CudaHelper::callInitKernel(
		unsigned int blocks,
		unsigned int threads,
		unsigned int ip,
		double L,
		double h,
		double dt,
		unsigned int fdDiags,
		const int* fdOffsets,
		const double* fdValues,
		double* Z,
		double* W,
		double* U )
{
	initKernel<<<blocks, threads>>>(
			ip,
			L,
			h,
			dt,
			fdDiags,
			fdOffsets,
			fdValues,
			Z,
			W,
			U );
}

void CudaHelper::callSyncKernel(
		unsigned int blocks,
		unsigned int threads,
		unsigned int ip,
		double* Z,
		double* W )
{
	syncKernel<<<blocks, threads>>>( ip, Z, W );
}

void CudaHelper::callReassociationKernel(
		unsigned int blocks,
		unsigned int threads,
		unsigned int ip,
		const double* Z,
		double* S )
{
	reassociationKernel<<<blocks, threads>>>( ip, Z, S );
}

template<typename T>
T* CudaHelper::allocDevMem( size_t numElem )
{
	T* mem;
	cudaMalloc( &mem, numElem * sizeof( T ) );
	return mem;
}

template<typename T>
void CudaHelper::freeDevMem( T* mem )
{
	cudaFree( mem );
}

template<typename T>
void CudaHelper::copyHostToDevMem( const T* hostMem, T* devMem, size_t numElem )
{
	cudaMemcpy(
			devMem,
			hostMem,
			numElem * sizeof( T ),
			cudaMemcpyHostToDevice );
}

template<typename T>
void CudaHelper::copyDevToHostMem( const T* devMem, T* hostMem, size_t numElem )
{
	cudaMemcpy(
			hostMem,
			devMem,
			numElem * sizeof( T ),
			cudaMemcpyDeviceToHost );
}

/* explicitly define template instantiations used in the program (otherwise they
   won't be compiled, as nvcc doesn't know about their use, since all but this
   class is compiled by g++ */

template int* CudaHelper::allocDevMem<int>( size_t );
template double* CudaHelper::allocDevMem<double>( size_t );

template void CudaHelper::freeDevMem<int>( int* );
template void CudaHelper::freeDevMem<double>( double* );

template void CudaHelper::copyHostToDevMem<int>( const int*, int*, size_t );
template void CudaHelper::copyHostToDevMem<double>( const double*, double*, size_t );

template void CudaHelper::copyDevToHostMem<int>( const int*, int*, size_t );
template void CudaHelper::copyDevToHostMem<double>( const double*, double*, size_t );

