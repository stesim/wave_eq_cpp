#pragma once

#ifndef RUNTIME_CUDA
#define __device__
#endif

/*
* Initial value function.
*/
__device__
inline double funu0( double x )
{
	return 1 / ( 1 + x * x );
}

/*
* Neumann boundary condition.
*/
__device__
inline double funu1( double x )
{
	return 0.0;
}

/*
* Exact solution.
*/
__device__
inline double funsol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1 / ( 1 + a * a ) + 1 / ( 1 + b * b ) ) / 2.0;
}
