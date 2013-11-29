#pragma once

/*
* Initial value function.
*/
inline double funu0( double x )
{
	return 1 / ( 1 + x * x );
}

/*
* Neumann boundary condition.
*/
inline double funu1( double x )
{
	return 0.0;
}

/*
* Exact solution.
*/
inline double funsol( double x, double t )
{
	double a = x - t;
	double b = x + t;
	return ( 1 / ( 1 + a * a ) + 1 / ( 1 + b * b ) ) / 2.0;
}
