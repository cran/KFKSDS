#include "KF.h"

void KF_C (const int *dim, const double *y, const double *sZ, const double *sT, 
  const double *H, //const double *sV, 
  const double *sQ, const double *sa0, const double *sP0, 
  const double *convtol, const int *convmaxiter, double *mll)
{
  int i, n = dim[0], m = dim[1], t0 = dim[3] - 1, 
    checkconv = dim[4], notconv = 1, counter = 0, convit = 0; 
    //r = dim[2], sUP = dim[5] - 1;
  double v, f, invf, fprev;

  mll[0] = 0.0;

  // data and state space model matrices

  gsl_vector_const_view vZ = gsl_vector_const_view_array(sZ, m);
  //gsl_matrix_const_view mZ = gsl_matrix_const_view_array(sZ, m, 1);
  gsl_matrix_const_view T = gsl_matrix_const_view_array(sT, m, m);
  gsl_matrix_const_view Q = gsl_matrix_const_view_array(sQ, m, m);
  gsl_vector_const_view a0 = gsl_vector_const_view_array(sa0, m);
  gsl_matrix_const_view P0 = gsl_matrix_const_view_array(sP0, m, m);

  gsl_vector *a_upd_init = gsl_vector_alloc(m);
  gsl_matrix *P_upd_init = gsl_matrix_alloc(m, m);
  gsl_vector_memcpy(a_upd_init, &a0.vector);
  gsl_matrix_memcpy(P_upd_init, &P0.matrix);

  // storage vectors and matrices

  gsl_vector *a_pred = gsl_vector_alloc(m);
  gsl_matrix *P_pred = gsl_matrix_alloc(m, m);
  gsl_vector *Vm = gsl_vector_alloc(m);
  gsl_matrix *Mmm = gsl_matrix_alloc(m, m);
  gsl_matrix *Mpm = gsl_matrix_alloc(1, m);
  gsl_matrix_view Mmp;

  // filtering recursions

  for (i = 0; i < n; i++)
  {
    // prediction

    gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, a_upd_init, 0.0, a_pred);

    if (notconv == 1)
    {
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, P_upd_init, 0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Mmm, &T.matrix, 0.0, P_pred);
      gsl_matrix_add(P_pred, &Q.matrix);
    } //else last value of 'P_pred' (steady state value) is kept
    //}

    if (isNotNA(y[i]))
    {
      gsl_blas_ddot(&vZ.vector, a_pred, &v);
      v = y[i] - v;

      if (notconv == 1)
      {
        gsl_blas_dgemv(CblasNoTrans, 1.0, P_pred, &vZ.vector, 0.0, Vm);
        gsl_blas_ddot(&vZ.vector, Vm, &f);
        f += H[0];
        invf = 1.0 / f;
      } // else values from previous iteration are kept

      // contribution to the minus log-likelihood function
      // (less constant added below)

      if (i >= t0)
      {
        mll[0] += log(f) + pow(v, 2) / f;
      }

      // updating

      gsl_vector_memcpy(a_upd_init, Vm);
      gsl_vector_scale(a_upd_init, v * invf);
      gsl_vector_add(a_upd_init, a_pred); 

      if (notconv == 1)
      {    
        // outer product of 'Vm'
        Mmp = gsl_matrix_view_array(Vm->data, m, 1);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, 
          &Mmp.matrix, &Mmp.matrix, 0.0, P_upd_init);
        gsl_matrix_scale(P_upd_init, -1.0 * invf);
        gsl_matrix_add(P_upd_init, P_pred);
      } // else P_upd_init from previous iteration is kept

      // check convergence of the filter

      if (checkconv == 1 && notconv == 1)
      {
        if (i == 0) {
          fprev = f + convtol[0] + 1.0;
        }
      
        if (fabs(f - fprev) < convtol[0])
        {
          // remain steady over 'maxiter' consecutive iterations
          if (convit == i - 1)
          {
            counter = counter + 1;
          } else
            counter = 1;
          convit = i;
        }

        fprev = f;

        if (counter == convmaxiter[0]) {
          notconv = 0; // the filter has converged to a steady state
          convit = i;
        }
      }      
    } else { // y[i] is NA
      gsl_vector_memcpy(a_upd_init, a_pred);
      gsl_matrix_memcpy(P_upd_init, P_pred);
      
      // NOTE reset if NA is found after the filter converged
      // for safety and also because it is a way to deal with the NA in f[i]
      if (notconv == 0) {
        notconv = 1;
        counter = 1;
      }
    }
  }

  mll[0] = 0.5 * (n - t0) * log(M_2PI) + 0.5 * mll[0];
  
  // deallocate memory

  gsl_vector_free(a_upd_init);
  gsl_matrix_free(P_upd_init);
  gsl_vector_free(a_pred);
  gsl_matrix_free(P_pred);
  gsl_vector_free(Vm);
  gsl_matrix_free(Mmm);
  gsl_matrix_free(Mpm);
}
