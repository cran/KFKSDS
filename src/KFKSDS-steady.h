#ifndef _KF_KS_DS_steady_
#define _KF_KS_DS_steady_

#include <R.h>
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

void KF_steady (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0, 
  double *mll, 
  std::vector<double> *v, std::vector<double> *f, 
  std::vector<double> *invf, std::vector<double> *vof,
  gsl_matrix *K, std::vector<gsl_matrix*> *L,
  double *tol, int *maxiter);

extern "C" void KFKSDS_steady (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0,
  double *tol, int *maxiter, double *ksconvfactor,
  double *mll, double *epshat, double *vareps,
  double *etahat, double *vareta, 
  double *sumepsmisc, double *sumetamisc);

#endif
