#ifndef _KF_KS_DS_
#define _KF_KS_DS_

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
//#include <assert.h>

//#include <R_ext/Rdynload.h>

#include <vector>
#include <numeric> //accumulate

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

//from /usr/share/R/include/Rmath.h
#ifndef M_2PI
#define M_2PI		6.283185307179586476925286766559	/* 2*pi */
#endif

#define isNotNA(x) x != -9999.99
#define isNA(x) x == -9999.99

extern void KF_deriv_aux_C (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0, 
  std::vector<double> *invf, std::vector<double> *vof, 
  double *dvof, std::vector<double> *dfinvfsq,
  gsl_matrix *a_pred, std::vector<gsl_matrix*> *P_pred,
  gsl_matrix *K, std::vector<gsl_matrix*> *L,  
  std::vector<gsl_matrix*> *da_pred,
  std::vector< std::vector<gsl_matrix*> > *dP_pred,
  std::vector<gsl_matrix*> *dK);

extern void KF_deriv_steady_C (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0, 
  double *tol, int *maxiter,
  std::vector<double> *invf, std::vector<double> *vof, 
  double *dvof, std::vector<double> *dfinvfsq,
  gsl_matrix *a_pred, std::vector<gsl_matrix*> *P_pred,
  gsl_matrix *K, std::vector<gsl_matrix*> *L,  
  std::vector<gsl_matrix*> *da_pred,
  std::vector< std::vector<gsl_matrix*> > *dP_pred,
  std::vector<gsl_matrix*> *dK);

#endif
