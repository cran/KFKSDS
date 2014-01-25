#include "KFKSDS.h"

void KF_deriv_steady_C (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0, 
  double *tol, int *maxiter,
  std::vector<double> *invf, std::vector<double> *vof, 
  double *dvof, std::vector<double> *dfinvfsq,
  gsl_matrix *a_pred, std::vector<gsl_matrix*> *P_pred,
  gsl_matrix *K, std::vector<gsl_matrix*> *L,  
  std::vector<gsl_matrix*> *da_pred,
  std::vector< std::vector<gsl_matrix*> > *dP_pred,
  std::vector<gsl_matrix*> *dK)
{
  int i, j, k, s, n = dim[0], p = dim[1], m = dim[2], 
    jm1, mp1 = m + 1, r = dim[3], rp1 = r + 1,
    conv = 0, counter = 0;

  //double v, f, fim1, df[rp1], dv, dtmp; //Kisum, Kim1sum;
  double v, f, fim1, dv, dtmp; //Kisum, Kim1sum;
  std::vector<double> df(rp1); 

  double mll = 0.0;  // for debugging

  // data and state space model matrices

  gsl_vector_view Z = gsl_vector_view_array(sZ, m);
  gsl_matrix_view T = gsl_matrix_view_array(sT, m, m);
  gsl_matrix_view Q = gsl_matrix_view_array(sQ, m, m);

  // storage vectors and matrices
  
  gsl_vector *Vm = gsl_vector_alloc(m);
  gsl_vector *Vm_cp = gsl_vector_alloc(m);
  gsl_vector *Vm_cp2 = gsl_vector_alloc(m);
  gsl_vector *Vm_cp3 = gsl_vector_alloc(m);
  gsl_vector *Vm3 = gsl_vector_alloc(m);
  gsl_matrix *Mmm = gsl_matrix_alloc(m, m);
  gsl_matrix *M1m = gsl_matrix_alloc(1, m);
  gsl_matrix *Mm1 = gsl_matrix_alloc(m, 1);

  gsl_vector_view a0 = gsl_vector_view_array(sa0, m);
  gsl_vector *a_upd = gsl_vector_alloc(m);
  gsl_vector_memcpy(a_upd, &a0.vector);

  gsl_matrix_view P0 = gsl_matrix_view_array(sP0, m, m);
  gsl_matrix *P_upd = gsl_matrix_alloc(m, m);
  gsl_matrix_memcpy(P_upd, &P0.matrix);

  gsl_vector_view K_irow, m_irow, m2_irow, m3_irow, K_im1row; //Kri;
  gsl_matrix_view maux1;
  gsl_matrix_view Zm = gsl_matrix_view_array(gsl_vector_ptr(&Z.vector, 0), 1, m);
  gsl_vector *mZ = gsl_vector_alloc(m);
  gsl_vector_memcpy(mZ, &Z.vector);
  gsl_vector_scale(mZ, -1.0);
  
  //std::vector<std::vector<gsl_matrix*> *> *da_pred;

  std::vector<gsl_matrix*> dP_upd(rp1);

  for (j = 0; j < rp1; j++)
  {
    da_pred[0].at(j) = gsl_matrix_alloc(n, m);
    dP_upd.at(j) = gsl_matrix_calloc(m, m);
  }

  gsl_matrix *da_upd = gsl_matrix_calloc(rp1, m);

  // filtering recursions

  for (i = 0; i < n; i++)
  {
    m_irow = gsl_matrix_row(a_pred, i);
    gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, a_upd, 0.0, &m_irow.vector);

    P_pred[0].at(i) = gsl_matrix_alloc(m, m);
if (conv == 0) {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, P_upd,
      0.0, Mmm);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Mmm, &T.matrix, 
      0.0, P_pred[0].at(i));
    gsl_matrix_add(P_pred[0].at(i), &Q.matrix);
} else {
    gsl_matrix_memcpy(P_pred[0].at(i), P_pred[0].at(i-1));
}

    gsl_blas_ddot(&Z.vector, &m_irow.vector, &v);
    v = sy[i] - v;

if (conv == 0) {
    gsl_blas_dgemv(CblasNoTrans, 1.0, P_pred[0].at(i), &Z.vector, 
      0.0, Vm);
    gsl_blas_ddot(&Z.vector, Vm, &f);
    f += *sH;
    
    invf->at(i) = 1.0 / f;    
    
} else {
    invf->at(i) = invf->at(i-1);
}

    gsl_vector_memcpy(Vm_cp, Vm);
    gsl_vector_memcpy(Vm_cp2, Vm);
    gsl_vector_memcpy(Vm_cp3, Vm);

    vof->at(i) = v * invf->at(i); // v[i]/f[i];

if (conv == 0) {
    maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm, 0), m, 1);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &maux1.matrix, 
      &maux1.matrix, 0.0, Mmm);
    gsl_matrix_scale(Mmm, invf->at(i));

    gsl_matrix_memcpy(P_upd, P_pred[0].at(i));
    gsl_matrix_sub(P_upd, Mmm);
}
    gsl_vector_memcpy(a_upd, &m_irow.vector);
    gsl_vector_scale(Vm_cp3, vof->at(i));
    gsl_vector_add(a_upd, Vm_cp3);

    K_irow = gsl_matrix_row(K, i);
    gsl_vector_scale(Vm_cp, invf->at(i));
if (conv == 0) {
    gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, Vm_cp, 0.0, &K_irow.vector);
} else {
    K_im1row = gsl_matrix_row(K, i-1);
    gsl_vector_memcpy(&K_irow.vector, &K_im1row.vector);
}

    L[0].at(i) = gsl_matrix_alloc(m, m);
if (conv == 0) {
    maux1 = gsl_matrix_view_array(gsl_vector_ptr(&K_irow.vector, 0), m, 1);
    gsl_matrix_memcpy(L[0].at(i), &T.matrix);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, &maux1.matrix, 
      &Zm.matrix, 1.0, L[0].at(i));
} else {
    gsl_matrix_memcpy(L[0].at(i), L[0].at(i-1));
}  
    // derivatives

    dK[0].at(i) = gsl_matrix_alloc(rp1, m);
    
    for (j = 0; j < rp1; j++)
    {
      k = i + j * n;

      m_irow = gsl_matrix_row(da_upd, j);
      m2_irow = gsl_matrix_row(da_pred[0].at(j), i);
      gsl_blas_dgemv(CblasNoTrans, 1.0, &T.matrix, &m_irow.vector, 
        0.0, &m2_irow.vector);

      gsl_blas_ddot(mZ, &m2_irow.vector, &dv);

      (dP_pred[0].at(i)).at(j) = gsl_matrix_alloc(m, m);
if (conv == 0) {
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, dP_upd.at(j),
        0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Mmm, &T.matrix, 
        0.0, (dP_pred[0].at(i)).at(j));
      if (j != 0)
      {
        jm1 = j - 1;
        dtmp = gsl_matrix_get((dP_pred[0].at(i)).at(j), jm1, jm1);
        gsl_matrix_set((dP_pred[0].at(i)).at(j), jm1, jm1, dtmp + 1.0);
      }
} else {
    gsl_matrix_memcpy((dP_pred[0].at(i)).at(j), (dP_pred[0].at(i-1)).at(j));
}

if (conv == 0) {
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Zm.matrix, 
        (dP_pred[0].at(i)).at(j), 0.0, M1m);
      m_irow = gsl_matrix_row(M1m, 0);
      gsl_blas_ddot(&m_irow.vector, &Z.vector, &df[j]);
      if (j == 0) {
        df[j] += 1.0;
      }
}

      dvof[k] = (dv * f - v * df[j]) * pow(invf->at(i), 2); 

      m_irow = gsl_matrix_row(da_upd, j);
      gsl_blas_dgemv(CblasNoTrans, vof->at(i), (dP_pred[0].at(i)).at(j), &Z.vector, 
        0.0, &m_irow.vector);
      gsl_vector_add(&m_irow.vector, &m2_irow.vector);
      dtmp = -1.0 * df[j] * invf->at(i);
      gsl_blas_daxpy(dtmp, Vm_cp3, &m_irow.vector);
      gsl_blas_daxpy(dv, Vm_cp, &m_irow.vector);

      dfinvfsq->at(k) = df[j] * pow(invf->at(i), 2);
if (conv == 0) {
      gsl_matrix_memcpy(dP_upd.at(j), (dP_pred[0].at(i)).at(j));   

      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, (dP_pred[0].at(i)).at(j), 
        &Zm.matrix, 0.0, Mm1);

      maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp, 0), 1, m);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, Mm1, &maux1.matrix, 
        1.0, dP_upd.at(j));

      maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp2, 0), m, 1);
      gsl_matrix_memcpy(Mm1, &maux1.matrix);
      maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp2, 0), 1, m);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, dfinvfsq->at(k), Mm1, 
        &maux1.matrix, 1.0, dP_upd.at(j));

      maux1 = gsl_matrix_view_array(gsl_vector_ptr(Vm_cp, 0), m, 1);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &maux1.matrix, 
        &Zm.matrix, 0.0, Mmm);

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, Mmm, 
        (dP_pred[0].at(i)).at(j), 1.0, dP_upd.at(j));
}

      m3_irow = gsl_matrix_row(dK[0].at(i), j);
if (conv == 0) {
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, 
        (dP_pred[0].at(i)).at(j), 0.0, Mmm);
      gsl_blas_dgemv(CblasNoTrans, 1.0, Mmm, &Z.vector, 0.0, &m3_irow.vector);
      gsl_vector_scale(&m3_irow.vector, invf->at(i));

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &T.matrix, 
        P_pred[0].at(i), 0.0, Mmm);
      gsl_blas_dgemv(CblasNoTrans, 1.0, Mmm, &Z.vector, 0.0, Vm3);
      gsl_vector_scale(Vm3, dfinvfsq->at(k));
      gsl_vector_sub(&m3_irow.vector, Vm3);
} else {
      K_im1row = gsl_matrix_row(dK[0].at(i-1), j);
      gsl_vector_memcpy(&m3_irow.vector, &K_im1row.vector);
}
    }
    
    // check if convergence to the steady state has been reached

    if (i > 0 & conv == 0)
    {
      if (i == 1)
      {
        fim1 = f + 1.0;

      }
      if (fabs(f - fim1) < *tol)
      {
        counter += 1;
      }
      fim1 = f;
      
      if (counter == *maxiter) {
        conv = 1;
        dim[5] = i;
      }
    }
  }

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
  gsl_vector_free(Vm_cp3);
  gsl_vector_free(Vm3);
  gsl_matrix_free(Mmm);
  gsl_matrix_free(M1m);
  gsl_matrix_free(Mm1);
  gsl_matrix_free(da_upd);
}

extern "C" {
void KFKSDS_deriv_steady_C (int *dim, double *sy, double *sZ, double *sT, double *sH, 
  double *sR, double *sV, double *sQ, double *sa0, double *sP0, 
  double *tol, int *maxiter, double *ksconvfactor,
  double *dvof, double *epshat, double *vareps, 
  double *etahat, double *vareta, 
  double *r, double *N, double *dr, double *dN, 
  double *dahat, double *dvareps)
{
  int i, ip1, j, k, s, n = dim[0], p = dim[1], m = dim[2], 
    mp1 = m + 1, ir = dim[3], rp1 = ir + 1, nrp1 = n * rp1,
    rp1m = rp1 * m, iaux, irp1m,
    convref, nmconvref, nm1 = n-1,
    irsod = ir * sizeof(double), msod = m * sizeof(double), 
    nsod = n * sizeof(double), rp1msod = rp1 * msod;

  //double invf[n], vof[n], msHsq, dfinvfsq[nrp1], dvareps0;
  double msHsq, dvareps0;
  std::vector<double> invf(n);
  std::vector<double> vof(n);
  std::vector<double> dfinvfsq(nrp1);

  gsl_matrix_view Q = gsl_matrix_view_array(sQ, m, m);
  
  gsl_vector_view Z = gsl_vector_view_array(sZ, m);  
  gsl_vector * Z_cp = gsl_vector_alloc(m);

  gsl_matrix * ZtZ = gsl_matrix_alloc(m, m);
  gsl_matrix_view maux1, maux2;
  maux1 = gsl_matrix_view_array(gsl_vector_ptr(&Z.vector, 0), m, 1);
  gsl_vector_memcpy(Z_cp, &Z.vector);
  maux2 = gsl_matrix_view_array(gsl_vector_ptr(Z_cp, 0), 1, m);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &maux1.matrix, 
    &maux2.matrix, 0.0, ZtZ);
  
  gsl_matrix * a_pred = gsl_matrix_alloc(n, m);
  std::vector<gsl_matrix*> P_pred(n);
    
  gsl_matrix * K = gsl_matrix_alloc(n, m);
  gsl_vector_view K_irow;
  
  std::vector<gsl_matrix*> L(n);
  
  gsl_vector_view Qdiag = gsl_matrix_diagonal(&Q.matrix);
  gsl_vector * Qdiag_msq = gsl_vector_alloc(m);
  gsl_vector_memcpy(Qdiag_msq, &Qdiag.vector);
  gsl_vector_mul(Qdiag_msq, &Qdiag.vector);
  gsl_vector_scale(Qdiag_msq, -1.0);
  
  std::vector<gsl_matrix*> da_pred(rp1);

  std::vector< std::vector<gsl_matrix*> > dP_pred(n, std::vector<gsl_matrix*>(rp1));

  std::vector<gsl_matrix*> dK(n);
  
  // filtering
  
  KF_deriv_steady_C(dim, sy, sZ, sT, sH, sR, sV, sQ, sa0, sP0, 
    tol, maxiter, &invf, &vof, dvof, &dfinvfsq, a_pred, &P_pred, K, 
    &L, &da_pred, &dP_pred, &dK);

  convref = dim[5];
  if (convref == -1) {
    convref = n;    
  } else
    convref = ceil(convref * ksconvfactor[0]);
  nmconvref = n - convref;

  // state vector smoothing and disturbances smoothing

  gsl_matrix_view V = gsl_matrix_view_array(sV, ir, ir);  
  gsl_matrix_view R = gsl_matrix_view_array(sR, m, ir);
  
  gsl_vector_view vaux;
  gsl_vector *vaux2 = gsl_vector_alloc(m);
  
  gsl_matrix *Mmm = gsl_matrix_alloc(m, m);
  gsl_matrix *Mmm2 = gsl_matrix_alloc(m, m);
  gsl_matrix *Mmm3 = gsl_matrix_alloc(m, m);
  gsl_matrix *Mrm = gsl_matrix_alloc(ir, m);

  gsl_vector_memcpy(Z_cp, &Z.vector);
  
  gsl_matrix *r0 = gsl_matrix_alloc(n + 1, m);
  gsl_vector_view r_row_t;
  gsl_vector_view r_row_tp1 = gsl_matrix_row(r0, n);
  gsl_vector_set_zero(&r_row_tp1.vector);

  std::vector<gsl_matrix*> N0(n + 1);
  N0.at(n) = gsl_matrix_calloc(m, m);
  gsl_vector_view Ndiag;

  gsl_vector *var_eps = gsl_vector_alloc(n);  
  msHsq = -1.0 * pow(*sH, 2);
  //vaux = gsl_vector_view_array(invf, n);
  vaux = gsl_vector_view_array(&invf[0], n);
  gsl_vector_set_all(var_eps, msHsq);
  gsl_vector_mul(var_eps, &vaux.vector);
  gsl_vector_add_constant(var_eps, *sH);

  gsl_vector *vr = gsl_vector_alloc(ir);
  gsl_vector *vr2 = gsl_vector_alloc(ir);

  gsl_matrix *dL = gsl_matrix_alloc(m, m);

  std::vector<gsl_matrix*> dr0(n + 1);
  dr0.at(n) = gsl_matrix_calloc(rp1, m);
  gsl_vector_view dr_row_t, dr_row_tp1;

  std::vector< std::vector<gsl_matrix*> > dN0(n + 1, std::vector<gsl_matrix*>(rp1));
  
  for (j = 0; j < rp1; j++)
  {
    (dN0.at(n)).at(j) = gsl_matrix_calloc(m, m);
  }

  for (i = nm1; i > -1; i--)
  {
    ip1 = i + 1;
    iaux = (i-1) * rp1m;
    irp1m = i * rp1m;

    if (i != n-1)
      r_row_tp1 = gsl_matrix_row(r0, ip1);
    r_row_t = gsl_matrix_row(r0, i);

    gsl_blas_dgemv(CblasTrans, 1.0, L.at(i), &r_row_tp1.vector, 
      0.0, &r_row_t.vector);
    gsl_vector_memcpy(Z_cp, &Z.vector);
    gsl_vector_scale(Z_cp, vof.at(i));
    gsl_vector_add(&r_row_t.vector, Z_cp);

    gsl_vector_memcpy(vaux2, &r_row_tp1.vector);
    memcpy(&r[i * m], vaux2->data, msod);
    
    N0.at(i) = gsl_matrix_alloc(m, m);
if (i < convref || i > nmconvref)
{    
    gsl_matrix_memcpy(N0.at(i), ZtZ);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, L.at(i), N0.at(ip1), 0.0, Mmm);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm, L.at(i), invf.at(i), N0.at(i));
} else {
    gsl_matrix_memcpy(N0.at(i), N0.at(ip1));
}

    vaux = gsl_matrix_diagonal(N0.at(ip1));
    gsl_vector_memcpy(vaux2, &vaux.vector);
    memcpy(&N[i * m], vaux2->data, msod);

if (i < convref || i == nm1) {
    K_irow = gsl_matrix_row(K, i);
}

    gsl_blas_ddot(&K_irow.vector, &r_row_tp1.vector, &epshat[i]);
    epshat[i] -= vof.at(i);
    epshat[i] *= -*sH;

if (i < convref || i > nmconvref)
{    
    maux1 = gsl_matrix_view_array(gsl_vector_ptr(&K_irow.vector, 0), 1, m);
    maux2 = gsl_matrix_view_array(gsl_vector_ptr(Z_cp, 0), 1, m);    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &maux1.matrix, N0.at(ip1),
      0.0, &maux2.matrix);
    vaux = gsl_vector_view_array(gsl_vector_ptr(var_eps, i), 1);
    gsl_blas_dgemv(CblasNoTrans, msHsq, &maux2.matrix, &K_irow.vector, 
      1.0, &vaux.vector);
} else {
    gsl_vector_set(var_eps, i, gsl_vector_get(var_eps, ip1));
}

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &V.matrix, &R.matrix,
      0.0, Mrm);
    gsl_blas_dgemv(CblasNoTrans, 1.0, Mrm, &r_row_tp1.vector,
      0.0, vr);
    memcpy(&etahat[i*ir], vr->data, irsod);

if (i < convref || i > nmconvref)
{
    Ndiag = gsl_matrix_diagonal(N0.at(ip1));
    gsl_vector_memcpy(Z_cp, &Ndiag.vector); 
    gsl_vector_mul(Z_cp, Qdiag_msq);
    gsl_vector_add(Z_cp, &Qdiag.vector);
    gsl_blas_dgemv(CblasTrans, 1.0, &R.matrix, Z_cp, 0.0, vr2);
}    
    memcpy(&vareta[i*ir], vr2->data, irsod);

    // derivatives 

    dr0.at(i) = gsl_matrix_alloc(rp1, m);
    
    for (j = 0; j < rp1; j++)
    {
      k = i + j * n;
      
      gsl_vector_memcpy(Z_cp, &Z.vector);
      gsl_vector_scale(Z_cp, dvof[k]);      
      
if (i < convref || i == nm1)
{      
      vaux = gsl_matrix_row(dK.at(i), j);
      maux1 = gsl_matrix_view_array(gsl_vector_ptr(&vaux.vector, 0), m, 1);
      maux2 = gsl_matrix_view_array(gsl_vector_ptr(&Z.vector, 0), 1, m);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, &maux1.matrix, 
        &maux2.matrix, 0.0, dL);
}

      dr_row_t = gsl_matrix_row(dr0.at(i), j);
      dr_row_tp1 = gsl_matrix_row(dr0.at(ip1), j);
      gsl_blas_dgemv(CblasTrans, 1.0, dL, &r_row_tp1.vector, 0.0, &dr_row_t.vector);
      gsl_vector_add(&dr_row_t.vector, Z_cp);
      gsl_blas_dgemv(CblasTrans, 1.0, L.at(i), &dr_row_tp1.vector, 1.0, &dr_row_t.vector);

      (dN0.at(i)).at(j) = gsl_matrix_alloc(m, m);
if (i < convref || i > nmconvref)
{      
      gsl_matrix_memcpy((dN0.at(i)).at(j), ZtZ);
      gsl_matrix_scale((dN0.at(i)).at(j), -1.0 * dfinvfsq[k]);
      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, dL, N0.at(ip1), 0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm, L.at(i), 
        1.0, (dN0.at(i)).at(j));
      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, L.at(i), 
        (dN0.at(ip1)).at(j), 0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm, L.at(i), 
        1.0, (dN0.at(i)).at(j));
      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, L.at(i), 
        N0.at(ip1), 0.0, Mmm);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm, dL, 
        1.0, (dN0.at(i)).at(j));
} else {
      gsl_matrix_memcpy((dN0.at(i)).at(j), (dN0.at(ip1)).at(j));
}
  
      if (i != 0)
      {
        vaux = gsl_matrix_diagonal((dN0.at(i)).at(j));
        gsl_vector_memcpy(vaux2, &vaux.vector);
        memcpy(&dN[iaux + j * m], vaux2->data, msod);
      }

      vaux = gsl_matrix_row(da_pred.at(j), i);
      gsl_blas_dgemv(CblasNoTrans, 1.0, (dP_pred.at(i)).at(j) , &r_row_t.vector, 
        1.0, &vaux.vector);
      gsl_blas_dgemv(CblasNoTrans, 1.0, P_pred.at(i), &dr_row_t.vector, 
        1.0, &vaux.vector);
      gsl_vector_memcpy(vaux2, &vaux.vector);
      memcpy(&dahat[irp1m + j * m], vaux2->data, msod);

if (i < convref || i > nmconvref)
{
      gsl_matrix_memcpy(Mmm3, (dP_pred.at(i)).at(j));
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, (dP_pred.at(i)).at(j), 
        N0.at(i), 0.0, Mmm2);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm2, P_pred.at(i),
        1.0, Mmm3);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, P_pred.at(i), 
        (dN0.at(i)).at(j), 0.0, Mmm2);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm2, P_pred.at(i),
        1.0, Mmm3);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, P_pred.at(i), 
        N0.at(i), 0.0, Mmm2);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mmm2,
        (dP_pred.at(i)).at(j), 1.0, Mmm3);
      
      gsl_matrix_mul_elements(Mmm3, ZtZ);
      std::vector<double> vmm(Mmm3->data, Mmm3->data + m*m);
dvareps0 = accumulate(vmm.begin(), vmm.end(), 0.0);

}// else {
//      gsl_matrix_memcpy((dP_pred.at(i)).at(j), (dP_pred.at(ip1)).at(j));
//}

dvareps[i*rp1 + j] = dvareps0;

      gsl_matrix_free((dN0.at(ip1)).at(j));    
      gsl_matrix_free((dP_pred.at(i)).at(j));
    }

    if (i != 0)
    {
      memcpy(&dr[iaux], (dr0.at(i))->data, rp1msod);
    }

    gsl_matrix_free(dr0.at(ip1));
    
    gsl_matrix_free(dK.at(i));
    gsl_matrix_free(P_pred.at(i));
    gsl_matrix_free(L.at(i));
    gsl_matrix_free(N0.at(ip1));
  }

  gsl_matrix_free(N0.at(0));
  gsl_matrix_free(dr0.at(0));
  for (j = 0; j < rp1; j++)
  {
    gsl_matrix_free((dN0.at(0)).at(j));
    gsl_matrix_free(da_pred.at(j));
  }
  
  memcpy(&vareps[0], var_eps->data, nsod);
    
  gsl_matrix_free(Mmm);
  gsl_matrix_free(Mmm2);
  gsl_matrix_free(Mmm3);
  gsl_matrix_free(Mrm);
  
  gsl_matrix_free(r0);
  gsl_matrix_free(K);
  gsl_matrix_free(dL);
  
  gsl_matrix_free(a_pred);
  
  gsl_vector_free(Z_cp);
  gsl_matrix_free(ZtZ);
  gsl_vector_free(var_eps);
  gsl_vector_free(vr);
  gsl_vector_free(vr2);
  gsl_vector_free(Qdiag_msq);
  gsl_vector_free(vaux2);
}}
