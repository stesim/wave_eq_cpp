#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "wave_eq_func.h"

/*
 * Matrix vector multiplication for square (multi-)diagonal sparse matrices.
 */
inline __global double* mul_mat_vec(
	unsigned int n,
	unsigned int num_diag,
	const __global int* diag_offset,
	const __global double* values,
	const __global double* vec,
	__global double* res )
{
	// set result vector to zero
	for( unsigned int i = 0; i < n; ++i )
	{
		res[ i ] = 0.0;
	}
	for( unsigned int d = 0; d < num_diag; ++d )
	{
		// first vector index involved in the multiplication
		unsigned int vec_index = min( 0, diag_offset[ d ] );
		// number of elements in the current diagonal
		unsigned int diag_size = n - abs( diag_offset[ d ] );
		for( unsigned int i = 0; i < diag_size; ++i )
		{
			res[ vec_index ] += values[ i ] * vec[ vec_index ];
			// increment vector index after each multiplication
			++vec_index;
		}
		// increment 'values' pointer to next diagonal
		values += diag_size;
	}
	return res;
}

/*
 * Calculate "u = u + s * v" where u,v are vectors and s is scalar.
 */
inline __global double* add_scaled_vec(
	unsigned int n,
	__global double* u,
	double s,
	const __global double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] += s * v[ i ];
	}
	return u;
}

/*
 * Calculate "u = u - v" for vectors u,v.
 */
inline __global double* sub_vec(
	unsigned int n,
	__global double* u,
	const __global double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] -= v[ i ];
	}
	return u;
}

/*
 * Solve on each interval.
 */
__kernel void wave_eq_cl(
	unsigned int ip,
	unsigned int nsteps,
	double a,
	__global double* Z,
	__global double* W,
	__global double* U,
	unsigned int mat_num_diag,
	__global const int* mat_diag_offset,
	__global const double* mat_values )
{
	// calculate beginning of vector for current interval based on work-item ID
	unsigned int vec_index = get_global_id( 0 ) * ip;
	__global double* z = &Z[ vec_index ];
	__global double* w = &W[ vec_index ];
	__global double* u = &U[ vec_index ];

	__global double* swap;

	for( unsigned int i = 0; i < nsteps; ++i )
	{
		// u = M * z + 2 * ( 1 - l2 ) * z - w;	with a = 2 * ( 1 - l2 )
		sub_vec( ip,
			add_scaled_vec( ip,
				mul_mat_vec( ip,
					mat_num_diag, mat_diag_offset, mat_values,
					z, u ),		// u = M * z
				a, z ),		// u = u + a * z = M * z + a * z
			w );		// u = u - w = M * z + a * z - w

		swap = w;
		w = z;
		z = u;
		u = swap;
	}
}

/*
 * Calculate initial values.
 */
__kernel void wave_eq_init(
	unsigned int ip,
	double L,

