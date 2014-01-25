#ifndef _KF_
#define _KF_

#include <R.h>
#include <Rinternals.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

//from /usr/share/R/include/Rmath.h
#ifndef M_2PI
#define M_2PI		6.283185307179586476925286766559	/* 2*pi */
#endif

#define isNotNA(x) x != -9999.99

extern void KF_C (const int *dim, const double *y, const double *sZ, const double *sT, 
  const double *H, //const double *sV, 
  const double *sQ, const double *sa0, const double *sP0, 
  const double *convtol, const int *convmaxiter, double *mll);

#endif
