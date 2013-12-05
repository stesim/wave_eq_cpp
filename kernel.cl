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
 * Calculate "u = p * u + s * v" where u,v are vectors and p,s are scalar.
 */
inline __global double* add_scaled_vecs(
	unsigned int n,
	double p,
	__global double* u,
	double s,
	const __global double* v )
{
	for( unsigned int i = 0; i < n; ++i )
	{
		u[ i ] *= p;
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
	// calculate beginning of vector for current interval based on work-item id
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
	double h,
	double a,
	double dt,
	__global double* Z,
	__global double* W,
	__global double* U,
	unsigned int mat_num_diag,
	__global const int* mat_diag_offset,
	__global const double* mat_values )
{
	unsigned int id = get_global_id( 0 );
	unsigned int vec_index = id * ip;
	__global double* z = &Z[ vec_index ];
	__global double* w = &W[ vec_index ];
	__global double* u = &U[ vec_index ];

	double x0 = h * ip * id / 2 - L;
	for( unsigned int i = 0; i < ip; ++i )
	{
		double x = x0 + i * h;
		w[ i ] = funu0( x );
		u[ i ] = funu1( x );
	}

	// Calculate one step back in time using:
	// z = 0.5 * M * w + a * w + dt * u;	with a = ( 1 - l2 )
	add_scaled_vec( ip,
		add_scaled_vecs( ip,
			0.5, mul_mat_vec( ip,
					mat_num_diag, mat_diag_offset, mat_values,
					w, z ),		// z = M * w
			a, w ),		// z = 0.5 * z + a * w = 0.5 * M * w + a * w
		dt, u );		// z = z + dt * u = 0.5 * M * w + a * w + dt * u
}

/*
 * Reassociate the solutions.
 */
__kernel void wave_eq_reassociate(
	unsigned int ip,
	__global double* Z,
	__global double* W,
	__global double* s,
	double L,
	double h,
	double t,
	__global double* e )
{
	unsigned int id = get_global_id( 0 );
	unsigned int n_1 = get_global_size( 0 ) - 1;
	unsigned int vec_index = id * ip;

	__global double* z = &Z[ vec_index ];
	__global double* w = &W[ vec_index ];

	// Determine indices of left and right neighbors of the current interval.
	unsigned int left_index;
	unsigned int right_index;
	if( id == 0 )
	{
		left_index = n_1;
		right_index = 1;
	}
	else if( id == n_1 )
	{
		left_index = n_1 - 1;
		right_index = 0;
	}
	else
	{
		left_index = vec_index - 1;
		right_index = vec_index + 1;
	}

	// Copy exact data from left neighbors
	__global double* nz = &Z[ left_index ];
	__global double* nw = &W[ left_index ];
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ i ] = nz[ ip / 2 + i ];
		w[ i ] = nw[ ip / 2 + i ];
	}
	// Copy exact data from right neighbors
	nz = &Z[ right_index ];
	nw = &W[ right_index ];
	for( unsigned int i = 0; i < ip / 4; ++i )
	{
		z[ ip * 3 / 4 + i ] = nz[ ip / 4 + i ];
		w[ ip * 3 / 4 + i ] = nz[ ip / 4 + i ];
	}

	// Copy originally exact part to the complete solution.
	if( id != n_1 )
	{
		for( unsigned int i = 0; i < ip / 2; ++i )
		{
			s[ id * ip + ip / 4 + i ] = z[ ip / 4 + i ];
		}
	}
	else
	{
		for( unsigned int i = 0; i < ip / 4; ++i )
		{
			s[ ( n_1 + 1 ) / 2 * ip - ip / 4 + i ] = z[ ip / 4 + i ];
			s[ i ] = z[ ip / 2 + i ];
		}
	}

	// Calculate equal parts of the exact solution
	double x0 = id * ip / 2 * h - L;
	unsigned int i0 = id * ip / 2;
	for( unsigned int i = 0; i < ip / 2; ++i )
	{
		e[ i0 + i ] = funsol( x0 + i * h, t );
	}
}

