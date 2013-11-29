#ifndef _DIAG_SP_MAT_H_
#define _DIAG_SP_MAT_H_

typedef struct diag_sp_mat
{
	unsigned int n;
	double* v;
	unsigned int d;
	unsigned int* o;
} diag_sp_mat;

#endif
