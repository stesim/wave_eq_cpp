#pragma once

#ifdef RUNTIME_CUDA
#define __FUNC_MODIFIER__ __device__
#else
#define __FUNC_MODIFIER__ inline
#endif

/*
* Initial value function.
*/
__FUNC_MODIFIER__
double funu0( double x )
{
	return 1 / ( 1 + x * x );
}

/*
* Neumann boundary condition.
*/
__FUNC_MODIFIER__
double funu1( double x )
{
	return 0.0;
}

/*
* Exact solution.
*/
__FUNC_MODIFIER__
double funsol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1 / ( 1 + a * a ) + 1 / ( 1 + b * b ) ) / 2.0;
}
