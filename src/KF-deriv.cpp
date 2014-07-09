#include "KFKSDS.h"

void KF_deriv_aux2_C (const int *dim, const double *y, const double *xreg, 
  const double *sZ, const double *sT, const double *sH, const double *sR, 
  const double *sV, const double *sQ, const double *sa0, const double *sP0, 
  const double *convtol, const int *convmaxiter,
  int *conv, double *mll, double *invf, double *vof, double *dvof, 
  double *dfinvfsq, double *dv, double *df,
  gsl_matrix *a_pred, std::vector<gsl_matrix*> *P_pred,
  gsl_matrix *K, std::vector<gsl_matrix*> *L,  
  std::vector<gsl_matrix*> *da_pred,
  std::vector< std::vector<gsl_matrix*> > *dP_pred,
  std::vector<gsl_matrix*> *dK)
{
  //int s, mp1 = m + 1;
  int i, j, k, n = dim[0], m = dim[1], r = dim[2], rp1 = r + 1, 
    t0 = dim[3] - 1, checkconv = dim[4], ncxreg = dim[5],
    notconv = 1, counter = 0, convit = 0, ncoldv = rp1 + ncxreg, jm1; 
  double v, f, invfsq, dtmp, fprev;

  conv[0] = notconv;
  conv[1] = 0;
  mll[0] = 0.0;

  // data and state space model matrices

  gsl_vector_const_view Z = gsl_vector_const_view_array(sZ, m);
  gsl_matrix_const_view T = gsl_matrix_const_view_array(sT, m, m);
  gsl_matrix_const_view Q = gsl_matrix_const_view_array(sQ, m, m);

  // storage vectors and matrices
  
  gsl_vector *Vm = gsl_vector_alloc(m);
  gsl_vector *Vm_cp = gsl_vector_alloc(m);
  gsl_vector *Vm_cp2 = gsl_vector_alloc(m);
  gsl_vector *Vm3 = gsl_vector_alloc(m);
  gsl_matrix *Mmm = gsl_matrix_alloc(m, m);
  gsl_matrix *M1m = gsl_matrix_alloc(1, m);
  gsl_matrix *Mm1 = gsl_matrix_alloc(m, 1);

  gsl_vector_const_view a0 = gsl_vector_const_view_array(sa0, m);
  gsl_vector *a_upd = gsl_vector_alloc(m);
  gsl_vector_memcpy(a_upd, &a0.vector);

  gsl_matrix_const_view P0 = gsl_matrix_const_view_array(sP0, m, m);
  gsl_matrix *P_upd = gsl_matrix_alloc(m, m);
  gsl_matrix_memcpy(P_upd, &P0.matrix);

  gsl_vector_view K_irow, m_irow, m2_irow, m3_irow;
  gsl_matrix_view maux1;
  gsl_matrix_const_view Zm = gsl_matrix_const_view_array(gsl_vector_const_ptr(&Z.vector, 0), 1, m);
  gsl_vector *mZ = gsl_vector_alloc(m);
  gsl_vector_memcpy(mZ, &Z.vector);
  gsl_vector_scale(mZ, -1.0);

  std::vector<gsl_matrix*> dP_upd(rp1);

  for (j = 0; j < ncoldv; j++)
  {
    da_pred[0].at(j) = gsl_matrix_calloc(n, m);
    if (j < rp1)
      dP_upd.at(j) = gsl_matrix_calloc(m, m);
  }

  gsl_matrix *da_upd = gsl_matrix_calloc(ncoldv, m);

  // filtering recursions

  for (i = 0; i < n; i++)
  {
    // prediction

    m_irow = gsl_matrix_row(a_pred, i);
    gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, a_upd, 0.0, &m_irow.vector);

    P_pred[0].at(i) = gsl_matrix_calloc(m, m);
    
    if (notconv == 1)
    {
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, P_upd, 0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Mmm, &T.matrix, 0.0, P_pred[0].at(i));
      gsl_matrix_add(P_pred[0].at(i), &Q.matrix);
    } else {
      gsl_matrix_memcpy(P_pred[0].at(i), P_pred[0].at(i-1));
    }

    if (isNotNA(y[i]))
    {
      gsl_blas_ddot(&Z.vector, &m_irow.vector, &v);
      v = y[i] - v;

      if (notconv == 1)
      {
        gsl_blas_dgemv(CblasNoTrans, 1.0, P_pred[0].at(i), &Z.vector, 0.0, Vm);
        gsl_blas_ddot(&Z.vector, Vm, &f);
        f += *sH;

        gsl_vector_memcpy(Vm_cp, Vm);
        gsl_vector_memcpy(Vm_cp2, Vm);

        invf[i] = 1.0 / f;

        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm, 0), m, 1);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &maux1.matrix, &maux1.matrix, 0.0, Mmm);
        gsl_matrix_scale(Mmm, invf[i]);

      } else { // f, Vm, Vm_cp, Vm_cp2, Mmm from previous iteration are kept
        gsl_blas_dgemv(CblasNoTrans, 1.0, P_pred[0].at(i), &Z.vector, 0.0, Vm);
        gsl_blas_ddot(&Z.vector, Vm, &f);
        f += *sH;

        gsl_vector_memcpy(Vm_cp, Vm);
        gsl_vector_memcpy(Vm_cp2, Vm);

        invf[i] = invf[i-1];

        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm, 0), m, 1);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &maux1.matrix, &maux1.matrix, 0.0, Mmm);
        gsl_matrix_scale(Mmm, invf[i]);
      }

      vof[i] = v * invf[i];

      // contribution to the minus log-likelihood function
      // (less constant added below)

      if (i >= t0)
      {
        mll[0] += log(f) + pow(v, 2) * invf[i];
      }

      // updating

      gsl_vector_memcpy(a_upd, &m_irow.vector);
      gsl_vector_scale(Vm, vof[i]);
      gsl_vector_add(a_upd, Vm);

      K_irow = gsl_matrix_row(K, i);
      L[0].at(i) = gsl_matrix_calloc(m, m);

      if (notconv == 1)
      {
        gsl_vector_scale(Vm_cp, invf[i]);
        gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, Vm_cp, 0.0, &K_irow.vector);

        gsl_matrix_memcpy(P_upd, P_pred[0].at(i));
        gsl_matrix_sub(P_upd, Mmm);

        maux1 = gsl_matrix_view_array(gsl_vector_ptr(&K_irow.vector, 0), m, 1);
        gsl_matrix_memcpy(L[0].at(i), &T.matrix);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, &maux1.matrix, 
          &Zm.matrix, 1.0, L[0].at(i));

      } else { // keep the values from previous iteration for 'P_upd'
        gsl_vector_scale(Vm_cp, invf[i]);
        gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, Vm_cp, 0.0, &K_irow.vector);
        gsl_matrix_memcpy(P_upd, P_pred[0].at(i));
        gsl_matrix_sub(P_upd, Mmm);

        gsl_matrix_memcpy(L[0].at(i), L[0].at(i-1));
      }      

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
    } else { // if (!isNotNA(y[i]))
      gsl_vector_memcpy(a_upd, &m_irow.vector);
      gsl_matrix_memcpy(P_upd, P_pred[0].at(i));
    }

    // derivatives

    dK[0].at(i) = gsl_matrix_calloc(rp1, m);
    invfsq = pow(invf[i], 2);
    
    for (j = 0; j < rp1; j++)
    {
      k = i + j * n;

      m_irow = gsl_matrix_row(da_upd, j);
      m2_irow = gsl_matrix_row(da_pred[0].at(j), i);
      gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, &m_irow.vector, 
          0.0, &m2_irow.vector);
      gsl_blas_ddot(mZ, &m2_irow.vector, &dv[k]);
      (dP_pred[0].at(i)).at(j) = gsl_matrix_calloc(m, m);

      if (notconv == 1)
      {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, dP_upd.at(j), 0.0, Mmm);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Mmm, &T.matrix, 0.0, (dP_pred[0].at(i)).at(j));

        if (j != 0)
        {
          jm1 = j - 1;
          dtmp = gsl_matrix_get((dP_pred[0].at(i)).at(j), jm1, jm1);
          gsl_matrix_set((dP_pred[0].at(i)).at(j), jm1, jm1, dtmp + 1.0);
        }

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Zm.matrix, 
          (dP_pred[0].at(i)).at(j), 0.0, M1m);
        m_irow = gsl_matrix_row(M1m, 0);
        gsl_blas_ddot(&m_irow.vector, &Z.vector, &df[k]);
        if (j == 0) {
          df[k] += 1.0;
        }
      } else {
        gsl_matrix_memcpy((dP_pred[0].at(i)).at(j), (dP_pred[0].at(i-1)).at(j));
        gsl_matrix_memcpy(dK[0].at(i), dK[0].at(i-1));
        df[k] = df[i - 1 + j * n];
      }

      dvof[k] = (dv[k] * f - v * df[k]) * invfsq;

      m_irow = gsl_matrix_row(da_upd, j);
      gsl_blas_dgemv(CblasNoTrans, vof[i], (dP_pred[0].at(i)).at(j), &Z.vector, 
        0.0, &m_irow.vector);
      gsl_vector_add(&m_irow.vector, &m2_irow.vector);
      dtmp = -1.0 * df[k] * invf[i];
      gsl_blas_daxpy(dtmp, Vm, &m_irow.vector);
      gsl_blas_daxpy(dv[k], Vm_cp, &m_irow.vector);

      if (notconv == 1 && isNotNA(y[i]))
      {
        gsl_matrix_memcpy(dP_upd.at(j), (dP_pred[0].at(i)).at(j));   
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, (dP_pred[0].at(i)).at(j), 
          &Zm.matrix, 0.0, Mm1);
        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp, 0), 1, m);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, Mm1, &maux1.matrix, 1.0, dP_upd.at(j));

        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp2, 0), m, 1);
        gsl_matrix_memcpy(Mm1, &maux1.matrix);
        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp2, 0), 1, m);
        dfinvfsq[k] = df[k] * invfsq;
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, dfinvfsq[k], Mm1, 
          &maux1.matrix, 1.0, dP_upd.at(j));

        maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp, 0), m, 1);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &maux1.matrix, &Zm.matrix, 0.0, Mmm);

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, Mmm, 
          (dP_pred[0].at(i)).at(j), 1.0, dP_upd.at(j));

        m3_irow = gsl_matrix_row(dK[0].at(i), j);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, 
          (dP_pred[0].at(i)).at(j), 0.0, Mmm);
        gsl_blas_dgemv(CblasNoTrans, 1.0, Mmm, &Z.vector, 0.0, &m3_irow.vector);
        gsl_vector_scale(&m3_irow.vector, invf[i]);

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, P_pred[0].at(i), 0.0, Mmm);
        gsl_blas_dgemv(CblasNoTrans, 1.0, Mmm, &Z.vector, 0.0, Vm3);
        gsl_vector_scale(Vm3, dfinvfsq[k]);
        gsl_vector_sub(&m3_irow.vector, Vm3);
      } //else {
        //gsl_matrix_memcpy(dP_upd.at(j), dP_upd.at(j));
        ////dK outside the loop over 'j'
      //}
/*
if(j==0){
//printf("\ni = %d\n", i+1);
//for (int kk = 0; kk < m; kk++)
//{
//  printf("%g  ", gsl_vector_get(Vm_cp, kk));
//}

//m4_irow = gsl_matrix_row(da_upd, 3);
//gsl_vector_add(&m4_irow.vector, &m5_irow.vector);
//gsl_blas_daxpy(dv[k3], Vm_cp, &m4_irow.vector);

gsl_blas_dgemv(CblasNoTrans, 0.0, (dP_pred[0].at(0)).at(0), &Z.vector, 
  0.0, &m4_irow.vector);
gsl_vector_add(&m4_irow.vector, &m5_irow.vector);
gsl_blas_daxpy(dv[k3], Vm_cp, &m4_irow.vector);
}
*/
      if (isNA(y[i]))
      {
        gsl_vector_memcpy(&m_irow.vector, &m2_irow.vector);
        gsl_matrix_memcpy(dP_upd.at(j), (dP_pred[0].at(i)).at(j));
      }
    }

    if (notconv == 0) {
      gsl_matrix_memcpy(dK[0].at(i), dK[0].at(i-1));
    }
    
    //if (ncxreg > 0 && j == r)
    if (ncxreg > 0)
    {
      int k2 = 0;
      
      for (int j2 = rp1; j2 < ncoldv; j2++)
      {
        int k3 = i + j2 * n;

        m_irow = gsl_matrix_row(da_upd, j2);
        m2_irow = gsl_matrix_row(da_pred[0].at(j2), i);
        gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, &m_irow.vector, 
          0.0, &m2_irow.vector);
        gsl_blas_ddot(mZ, &m2_irow.vector, &dv[k3]);
        dv[k3] -= xreg[i + k2 * n];

        gsl_vector_set_zero(&m_irow.vector); 
        gsl_vector_add(&m_irow.vector, &m2_irow.vector);
        gsl_blas_daxpy(dv[k3], Vm_cp, &m_irow.vector);

        k2 += 1;
      }
    }

    if (isNA(y[i]))
    {
      if (notconv == 0) {
        notconv = 1;
        counter = 1;
      }

      for (j = 0; j < ncoldv; j++)
      {
        k = i + j * n;
        dv[k] = 0.0;
        if (j < rp1)
          df[k] = 0.0;
      }
    }
  }
  
  conv[0] = notconv;
  conv[1] = convit;

  mll[0] = 0.5 * n * log(M_2PI) + 0.5 * mll[0];
  
  // deallocate memory

  for (j = 0; j < rp1; j++)
  {
    gsl_matrix_free(dP_upd.at(j));
  }
  
  gsl_vector_free(mZ);
  gsl_vector_free(a_upd);
  gsl_matrix_free(P_upd);
  gsl_vector_free(Vm);
  gsl_vector_free(Vm_cp);
  gsl_vector_free(Vm_cp2);
  gsl_vector_free(Vm3);
  gsl_matrix_free(Mmm);
  gsl_matrix_free(M1m);
  gsl_matrix_free(Mm1);
  gsl_matrix_free(da_upd);
}

extern "C" {
void KF_deriv_C (const int *dim, const double *y, const double *xreg,
  const double *sZ, const double *sT, 
  const double *sH, const double *sR, const double *sV, const double *sQ, 
  const double *sa0, const double *sP0, 
  const double *convtol, const int *convmaxiter,
  int *conv, double *mll, double *invf, double *vof, double *dvof, 
  double *dfinvfsq, double *dv, double *df,
  double *a_pred0, double *P_pred0, double *K0, double *L0,  
  double *da_pred0, double *dP_pred0, double *dK0)
{
  //int ip1, k, s, mp1 = m + 1,  nrp1 = n * rp1, iaux, irp1m,
  //  irsod = ir * sizeof(double);
  int i, j, n = dim[0], m = dim[1], 
    mm = m*m, nm = n*m,
    ir = dim[2], rp1 = ir + 1, ncxreg = dim[5],
    rp1m = rp1 * m, 
    msod = m * sizeof(double), 
    mmsod = m * msod, nsod = n * sizeof(double), nmsod = m*nsod,
    rp1msod = rp1 * msod;

  gsl_matrix * a_pred = gsl_matrix_alloc(n, m);
  std::vector<gsl_matrix*> P_pred(n);
    
  gsl_matrix * K = gsl_matrix_alloc(n, m);  
  std::vector<gsl_matrix*> L(n);
  std::vector<gsl_matrix*> da_pred(rp1 + ncxreg);
  std::vector< std::vector<gsl_matrix*> > dP_pred(n, std::vector<gsl_matrix*>(rp1));
  std::vector<gsl_matrix*> dK(n);

  // filtering

  KF_deriv_aux2_C(dim, y, xreg, sZ, sT, sH, sR, sV, sQ, sa0, sP0, 
    convtol, convmaxiter,
    conv, mll, invf, vof, dvof, dfinvfsq, dv, df, a_pred, &P_pred, 
    K, &L, &da_pred, &dP_pred, &dK);

  memcpy(&a_pred0[0], a_pred->data, nmsod);
  memcpy(&K0[0], K->data, nmsod);

  // copy to output variables and free

  for (i = 0; i < n; i++)
  {
    if (isNotNA(y[i]))
    {
      memcpy(&P_pred0[i*mm], (P_pred.at(i))->data, mmsod);
      memcpy(&L0[i*mm], (L.at(i))->data, mmsod);
      memcpy(&dK0[i*rp1m], (dK.at(i))->data, rp1msod);

      gsl_matrix_free(P_pred.at(i));
      gsl_matrix_free(L.at(i));
      gsl_matrix_free(dK.at(i));
    }

    for (j = 0; j < rp1 + ncxreg; j++)
    {
      if (j < rp1)
      {
        memcpy(&dP_pred0[i*rp1*mm+j*mm], ((dP_pred.at(i)).at(j))->data, mmsod);
        gsl_matrix_free((dP_pred.at(i)).at(j));
      }

      if (i == 0)
      {
        memcpy(&da_pred0[j*nm], (da_pred.at(j))->data, nmsod);
        gsl_matrix_free(da_pred.at(j));
      }
    }
  }

  gsl_matrix_free(a_pred);
  gsl_matrix_free(K);
}}
