#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __device__
#define __global__ __kernel
#define GLOBAL_ID ( get_global_id( 0 ) )
#define GLOBAL_SIZE ( get_global_size( 0 ) )

#include "wave_eq_func.h"

// FIXME: implement generic version (non-double specific)
void memcpy(
	__global double* dst,
	__global const double* src,
	unsigned int size )
{
	unsigned int numElem = size / sizeof( double );
	for( unsigned int i = 0; i < numElem; ++i )
	{
		dst[ i ] = src[ i ];
	}
}

__device__
__global double* diaMulVec(
		unsigned int n,
		unsigned int diaDiags,
		__global const int* diaOffsets,
		__global const double* diaValues,
		__global const double* vec,
		__global double* res )
{
	// set result vector to zero
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = 0.0;
	}
	unsigned int totalDiagOffset = 0;
	for( unsigned int k = 0; k < diaDiags; ++k )
	{
		// determine first vector index involved in the diagonal multiplication
		unsigned int vecIndex = max( 0, diaOffsets[ k ] );
		// determine first result vector index
		unsigned int resIndex = max( 0, -diaOffsets[ k ] );
		// detemine the number of elements in the diagonal
		unsigned int diagSize = n - abs( diaOffsets[ k ] );
		for( unsigned int i = 0; i < diagSize; ++i )
		{
			res[ resIndex + i ] += diaValues[ totalDiagOffset + i ] * vec[ vecIndex + i ];
		}
		// increment values pointer to next diagonal
		totalDiagOffset += diagSize;
	}
	return res;
}

__device__
__global double* vecAddScaledVec(
		unsigned int n,
		__global double* u,
		double c,
		__global const double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] += c * v[ i ];
	}
	return u;
}

__device__
__global double* vecAddScaledVecs(
		unsigned int n,
		double c,
		__global const double* u,
		double d,
		__global const double* v,
		__global double* res )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = c * u[ i ] + d * v[ i ];
	}
	return res;
}

__global__
void kernelMain(
		unsigned int ip,
		unsigned int nsteps,
		double l2,
		unsigned int fdDiags,
		__global const int* fdOffsets,
		__global const double* fdValues,
		__global double* Z,
		__global double* W,
		__global double* U )
{
	// determine index of the first element of the subdomain
	unsigned int vecIndex = GLOBAL_ID * ip;
	__global double* z = &Z[ vecIndex ];
	__global double* w = &W[ vecIndex ];
	__global double* u = &U[ vecIndex ];

	double a = 2.0 * ( 1.0 - l2 );
	__global double* swap;
	for( unsigned int i = 0; i < nsteps; ++i )
	{
		// u = M * z
		diaMulVec( ip, fdDiags, fdOffsets, fdValues, z, u );
		// u = u + a * z = M * z + a * z
		vecAddScaledVec( ip, u, a, z );
		// u = u + (-w) = M * z + a * z - w
		vecAddScaledVec( ip, u, -1.0, w );

		// shuffle buffers to avoid copying
		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

__global__
void kernelInit(
		unsigned int ip,
		double L,
		double h,
		double dt,
		unsigned int fdDiags,
		__global const int* fdOffsets,
		__global const double* fdValues,
		__global double* Z,
		__global double* W,
		__global double* U )
{
	// determine index of the first element of the subdomain
	unsigned int id = GLOBAL_ID;
	unsigned int vecIndex = id * ip;
	__global double* z = &Z[ vecIndex ];
	__global double* w = &W[ vecIndex ];
	__global double* u = &U[ vecIndex ];

	double x0 = h * ip * id / 2 - L;
	for( unsigned int i = 0; i < ip; ++i )
	{
		double x = x0 + i * h;
		w[ i ] = funu0( x );
		u[ i ] = funu1( x );
	}

	double a = 1.0 - dt / h * dt / h;
	// z = M * w
	diaMulVec( ip, fdDiags, fdOffsets, fdValues, w, z );
	// z = 0.5 * z + a * w = 0.5 * M * w + a * w
	vecAddScaledVecs( ip, 0.5, z, a, w, z );
	// z = z + dt * u
	vecAddScaledVec( ip, z, dt, u );
}

__global__
void kernelSync(
		unsigned int ip,
		__global double* Z,
		__global double* W )
{
	unsigned int id = GLOBAL_ID;
	unsigned int n_1 = GLOBAL_SIZE - 1;
	unsigned int vecIndex = id * ip;
	__global double* z = &Z[ vecIndex ];
	__global double* w = &W[ vecIndex ];

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
	__global double* nz = &Z[ leftNeighbor * ip ];
	__global double* nw = &W[ leftNeighbor * ip ];
	memcpy( &z[ 0 ], &nz[ ip / 2 ], ip / 4 * sizeof( double ) );
	memcpy( &w[ 0 ], &nw[ ip / 2 ], ip / 4 * sizeof( double ) );
	/*
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ i ] = nz[ ip / 2 + i ];
		w[ i ] = nw[ ip / 2 + i ];
	}
	*/
	// copy exact data from right neighbors
	nz = &Z[ rightNeighbor * ip ];
	nw = &W[ rightNeighbor * ip ];
	memcpy( &z[ ip * 3 / 4 ], &nz[ ip / 4 ], ip / 4 * sizeof( double ) );
	memcpy( &w[ ip * 3 / 4 ], &nw[ ip / 4 ], ip / 4 * sizeof( double ) );
	/*
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ ip * 3 / 4 + i ] = nz[ ip / 4 + i ];
		w[ ip * 3 / 4 + i ] = nw[ ip / 4 + i ];
	}
	*/
}

__global__
void kernelReassociate(
		unsigned int ip,
		__global const double* Z,
		__global double* S )
{
	unsigned int id = GLOBAL_ID;
	unsigned int vecIndex = id * ip;
	__global const double* z = &Z[ vecIndex ];
	__global double* s = &S[ vecIndex / 2 ];

	// copy left half of the subdomain solution
	// memcpy( s, z, ip / 2 * sizeof( double ) );
	
	for( unsigned int i = 0; i < ip / 2; ++i )
	{
		s[ i ] = z[ i ];
	}
	
}
