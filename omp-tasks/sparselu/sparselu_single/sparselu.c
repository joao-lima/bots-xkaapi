/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <libgen.h>
#include "bots.h"
#include "sparselu.h"

#if defined(BOTS_KAAPI)
#include "kaapic.h"

#include <cblas.h>

#include <cuda_runtime_api.h>
#include "cublas_v2.h"

extern cublasHandle_t kaapi_cuda_cublas_handle( void );
#endif

#define   __COL_MAJOR

/***********************************************************************
 * checkmat: 
 **********************************************************************/
#if defined(__COL_MAJOR)
#define M(i,j)     (m[j*bots_arg_size_1+i])
#define N(i,j)     (n[j*bots_arg_size_1+i])
#else
#define M(i,j)     (m[i*bots_arg_size_1+j])
#define N(i,j)     (n[i*bots_arg_size_1+j])
#endif
int checkmat (float *m, float *n)
{
  int i, j;
  float r_err;
  
  for (i = 0; i < bots_arg_size_1; i++)
  {
    for (j = 0; j < bots_arg_size_1; j++)
    {
      r_err = M(i,j) - N(i,j);
      if ( r_err == 0.0 ) continue;
      
      if (r_err < 0.0 ) r_err = -r_err;
      
      if ( M(i,j) == 0 )
      {
        bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n",
                     i,j, M(i,j), i,j, N(i,j) );
        return FALSE;
      }  
      r_err = r_err / M(i,j);
      if(r_err > EPSILON)
      {
        bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
                     i,j, M(i,j), i,j, N(i,j), r_err);
        return FALSE;
      }
    }
  }
  return TRUE;
}
#undef M
#undef N

/***********************************************************************
 * genmat:
 **********************************************************************/
#if defined(__COL_MAJOR)
#define   M(i,j)   (m[j*bots_arg_size+i])
#else
#define   M(i,j)   (m[i*bots_arg_size+j])
#endif
void genmat (float *m[])
{
  int null_entry, init_val, i, j, ii, jj;
  float *p;
  
  init_val = 1325;
  
  /* generating the structure */
  for (ii=0; ii < bots_arg_size; ii++)
  {
    for (jj=0; jj < bots_arg_size; jj++)
    {
      /* computing null entries */
      null_entry=FALSE;
      if ((ii<jj) && (ii%3 !=0)) null_entry = TRUE;
      if ((ii>jj) && (jj%3 !=0)) null_entry = TRUE;
      if (ii%2==1) null_entry = TRUE;
      if (jj%2==1) null_entry = TRUE;
      if (ii==jj) null_entry = FALSE;
      if (ii==jj-1) null_entry = FALSE;
      if (ii-1 == jj) null_entry = FALSE; 
      /* allocating matrix */
      if (null_entry == FALSE){
        M(ii,jj) = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
        if(M(ii,jj) == NULL)
        {
          bots_message("Error: Out of memory\n");
          exit(101);
        }
        /* initializing matrix */
        p = M(ii,jj);
        for (i = 0; i < bots_arg_size_1; i++) 
        {
          for (j = 0; j < bots_arg_size_1; j++)
          {
            init_val = (3125 * init_val) % 65536;
            (*p) = (float)((init_val - 32768.0) / 16384.0);
            p++;
          }
        }
      }
      else
      {
        M(ii,jj) = NULL;
      }
    }
  }
}
#undef M

/***********************************************************************
 * print_structure: 
 **********************************************************************/
#if defined(__COL_MAJOR)
#define M(i,j)   (m[j*bots_arg_size+i])
#else
#define M(i,j)   (m[i*bots_arg_size+j])
#endif
void print_structure(char *name, float *m[])
{
  int ii, jj;
  bots_message("Structure for matrix %s @ 0x%p\n",name, m);
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      if (M(ii,jj) != NULL) {bots_message("x");}
      else bots_message(" ");
    }
    bots_message("\n");
  }
  bots_message("\n");
}
#undef M

/***********************************************************************
 * allocate_clean_block: 
 **********************************************************************/
float * allocate_clean_block()
{
  int i,j;
  float *p, *q;
  
  p = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
  q=p;
  if (p!=NULL){
    for (i = 0; i < bots_arg_size_1; i++) 
      for (j = 0; j < bots_arg_size_1; j++){(*p)=0.0; p++;}
    
  }
  else
  {
    bots_message("Error: Out of memory\n");
    exit (101);
  }
  return (q);
}

/***********************************************************************
 * lu0: 
 **********************************************************************/
#if defined(__COL_MAJOR)
#define A(i,j)    (diag[j*bots_arg_size_1+i])
#else
#define A(i,j)    (diag[i*bots_arg_size_1+j])
#endif
void lu0(float *diag)
{
  int i, j, k;
  
  for (k=0; k<bots_arg_size_1; k++)
    for (i=k+1; i<bots_arg_size_1; i++)
    {
      A(i,k) = A(i,k) / A(k,k);
      for (j= k+1; j<bots_arg_size_1; j++)
        A(i,j) = A(i,j) - A(i,k) * A(k,j);
    }
}
#undef A

/***********************************************************************
 * bdiv: 
 **********************************************************************/
#if defined(__COL_MAJOR)
#define   A(i,j)    (diag[j*bots_arg_size_1+i])
#define   B(i,j)    (row[j*bots_arg_size_1+i])
#else
#define   A(i,j)    (diag[i*bots_arg_size_1+j])
#define   B(i,j)    (row[i*bots_arg_size_1+j])
#endif
void bdiv(float *diag, float *row)
{
  int i, j, k;
  for (i=0; i<bots_arg_size_1; i++)
    for (k=0; k<bots_arg_size_1; k++)
    {
      B(i,k) = B(i,k) / A(k,k);
      for (j=k+1; j<bots_arg_size_1; j++)
        B(i,j) = B(i,j) - B(i,k)*A(k,j);
    }
}
#undef A
#undef B

/***********************************************************************
 * bmod: 
 **********************************************************************/
void bmod(float *row, float *col, float *inner)
{
#if defined(BOTS_KAAPI)
  cblas_sgemm(
#if defined(__COL_MAJOR)
              CblasColMajor,
#else
              CblasRowMajor,
#endif
              CblasNoTrans, CblasNoTrans,
              bots_arg_size_1, bots_arg_size_1, bots_arg_size_1,
              -1.0f, row, bots_arg_size_1,
              col, bots_arg_size_1,
              1.0f, inner, bots_arg_size_1
              );
#else
  int i, j, k;
  for (i=0; i<bots_arg_size_1; i++)
    for (j=0; j<bots_arg_size_1; j++)
      for (k=0; k<bots_arg_size_1; k++)
        inner[i*bots_arg_size_1+j] = inner[i*bots_arg_size_1+j] - row[i*bots_arg_size_1+k]*col[k*bots_arg_size_1+j];
#endif
}
/***********************************************************************
 * fwd: 
 **********************************************************************/
void fwd(float *diag, float *col)
{
#if defined(BOTS_KAAPI)
  cblas_sgemm(
#if defined(__COL_MAJOR)
              CblasColMajor,
#else
              CblasRowMajor,
#endif
              CblasNoTrans, CblasNoTrans,
              bots_arg_size_1, bots_arg_size_1, bots_arg_size_1,
              -1.0f, diag, bots_arg_size_1,
              col, bots_arg_size_1,
              1.0f, col, bots_arg_size_1
              );
#else
  int i, j, k;
  for (j=0; j<bots_arg_size_1; j++)
    for (k=0; k<bots_arg_size_1; k++) 
      for (i=k+1; i<bots_arg_size_1; i++)
        col[i*bots_arg_size_1+j] = col[i*bots_arg_size_1+j] - diag[i*bots_arg_size_1+k]*col[k*bots_arg_size_1+j];
#endif
}

#if defined(BOTS_KAAPI)
void fwd_cuda(float *diag, float *col)
{
  cublasStatus_t res;
  float alpha = -1.0f;
  float beta = 1.0f;
#if 1
  fprintf(stdout, "TaskGPU FWD diag=%p col=%p\n", (void*)diag, (void*)col);
  fflush(stdout);
#endif
  res= cublasSgemm_v2(
    kaapi_cuda_cublas_handle(),
    CUBLAS_OP_N, CUBLAS_OP_N,
    bots_arg_size_1, bots_arg_size_1, bots_arg_size_1,
    &alpha,
    diag, bots_arg_size_1,
    col, bots_arg_size_1,
    &beta,
    col, bots_arg_size_1 );
  
  if( res != CUBLAS_STATUS_SUCCESS ) {
    fprintf(stdout, "CUBLAS error: %d\n", res );
    fflush(stdout);
  }
}

void bmod_cuda(float *row, float *col, float *inner)
{
  cublasStatus_t res;
  float alpha = -1.0f;
  float beta = 1.0f;
#if 1
  fprintf(stdout, "TaskGPU BMOD row=%p col=%p inner=%p\n", (void*)row, (void*)col, (void*)inner);
  fflush(stdout);
#endif  
//  cblas_sgemm(
//              CblasColMajor,
//              CblasNoTrans, CblasNoTrans,
//              bots_arg_size_1, bots_arg_size_1, bots_arg_size_1,
//              -1.0f, row, bots_arg_size_1,
//              col, bots_arg_size_1,
//              1.0f, inner, bots_arg_size_1
//              );
  res= cublasSgemm_v2(
                      kaapi_cuda_cublas_handle(),
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      bots_arg_size_1, bots_arg_size_1, bots_arg_size_1,
                      &alpha, row, bots_arg_size_1,
                      col, bots_arg_size_1,
                      &beta, inner, bots_arg_size_1
                      );
  
  if( res != CUBLAS_STATUS_SUCCESS ) {
    fprintf(stdout, "CUBLAS error: %d\n", res );
    fflush(stdout);
  }
}
#endif

#if defined(__COL_MAJOR)
#define   A(i,j)    (BENCH[j*bots_arg_size+i])
#else
#define   A(i,j)    (BENCH[i*bots_arg_size+j])
#endif
void sparselu_init_seq (float ***pBENCH, char *pass)
{
  *pBENCH = (float **) malloc(bots_arg_size*bots_arg_size*sizeof(float *));
  genmat(*pBENCH);

//  print_structure(pass, *pBENCH);

  int ii, jj, kk;
  float** BENCH=*pBENCH;
  for (kk=0; kk<bots_arg_size; kk++) 
  {
    for (ii=kk+1; ii<bots_arg_size; ii++)
      if (A(ii,kk) != NULL)
        for (jj=kk+1; jj<bots_arg_size; jj++)
          if (A(kk,jj) != NULL)
          {
            if (A(ii,jj) == NULL)
              A(ii,jj) = allocate_clean_block();
          }
  }
}
#undef A

void sparselu_init_par (float ***pBENCH, char *pass)
{
#if defined(BOTS_KAAPI)
  kaapic_init( 0 );
//  kaapic_init( KAAPIC_START_ONLY_MAIN );
#endif
  sparselu_init_seq(pBENCH, pass);
}

#if defined(__COL_MAJOR)
#define   A(i,j)    (BENCH[j*bots_arg_size+i])
#else
#define   A(i,j)    (BENCH[i*bots_arg_size+j])
#endif
void sparselu_par_call(float **BENCH)
{
  int ii, jj, kk;
#if defined(BOTS_KAAPI)
  kaapic_spawn_attr_t lu0_attr, fwd_attr, bdiv_attr, bmod_attr;
#endif
  
  bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
               bots_arg_size,bots_arg_size,bots_arg_size_1,bots_arg_size_1);

#if defined(BOTS_KAAPI)
  kaapic_spawn_attr_init(&lu0_attr);
  kaapic_spawn_attr_init(&fwd_attr);
  kaapic_spawn_attr_init(&bdiv_attr);
  kaapic_spawn_attr_init(&bmod_attr);
  
  kaapic_spawn_attr_set_arch(&lu0_attr, KAAPIC_ARCH_CPU_ONLY);
  kaapic_spawn_attr_set_arch(&fwd_attr, KAAPIC_ARCH_CPU_ONLY);
  kaapic_spawn_attr_set_arch(&bdiv_attr, KAAPIC_ARCH_CPU_ONLY);
  kaapic_spawn_attr_set_arch(&bmod_attr, KAAPIC_ARCH_DEFAULT);
  
//  kaapic_spawn_attr_set_alternative_body(&fwd_attr, (void (*)())fwd_cuda);
  kaapic_spawn_attr_set_alternative_body(&bmod_attr, (void (*)())bmod_cuda);
#endif

#if defined(BOTS_KAAPI)
  kaapic_begin_parallel( KAAPIC_FLAG_STATIC_SCHED );
#endif

  for (kk=0; kk<bots_arg_size; kk++)
  {
#if defined(BOTS_KAAPI)
    kaapic_spawn(&lu0_attr,  /* attribut */
      1, lu0,
      KAAPIC_MODE_RW,                 /* RW access */
      KAAPIC_TYPE_FLT,
      bots_arg_size_1*bots_arg_size_1,
      A(kk,kk)
    );
#else
    lu0(A(kk,kk));
#endif
    
    for (jj=kk+1; jj<bots_arg_size; jj++)
    {
      if (A(kk,jj) != NULL)
      {
#if defined(BOTS_KAAPI)
        kaapic_spawn(&fwd_attr,  /* attribut */
          2, fwd, 
          /* diag is read */
          KAAPIC_MODE_R,
          KAAPIC_TYPE_FLT,
          bots_arg_size_1*bots_arg_size_1,
          A(kk,kk),
          
          /* col is rw */
          KAAPIC_MODE_RW,
          KAAPIC_TYPE_FLT,
          bots_arg_size_1*bots_arg_size_1,
          A(kk,jj)
        );
#else
#pragma omp task untied firstprivate(kk, jj) shared(BENCH)
        fwd(A(kk,kk), A(kk,jj));
#endif
      }
    }
    
    for (ii=kk+1; ii<bots_arg_size; ii++)
      if (A(ii,kk) != NULL)
      {
#if defined(BOTS_KAAPI)
        kaapic_spawn(&bdiv_attr,  /* attribut */
          2, bdiv, 
          /* diag is readwrite */
          KAAPIC_MODE_R,
          KAAPIC_TYPE_FLT,
          bots_arg_size_1*bots_arg_size_1,
          A(kk,kk),

          /* row is read write */
          KAAPIC_MODE_RW,
          KAAPIC_TYPE_FLT,
          bots_arg_size_1*bots_arg_size_1,
          A(ii,kk)
        );
#else
#pragma omp task untied firstprivate(kk, ii) shared(BENCH)
        bdiv(A(kk,kk), A(ii,kk));
#endif
      }
    
#if defined(BOTS_KAAPI)
//    kaapic_sync();
#else
  #pragma omp taskwait
#endif
    
    for (ii=kk+1; ii<bots_arg_size; ii++)
    {
      if (A(ii,kk) != NULL)
      {
        for (jj=kk+1; jj<bots_arg_size; jj++)
        {
          if(A(kk,jj) != NULL)
#if defined(BOTS_KAAPI)
          {
            //if (BENCH[ii*bots_arg_size+jj]==NULL) BENCH[ii*bots_arg_size+jj] = allocate_clean_block();
            kaapic_spawn(&bmod_attr,  /* attribut */
              3, bmod, 
              /* */
              KAAPIC_MODE_R,
              KAAPIC_TYPE_FLT,
              bots_arg_size_1*bots_arg_size_1,
              A(ii,kk),

              /* */
              KAAPIC_MODE_R,
              KAAPIC_TYPE_FLT,
              bots_arg_size_1*bots_arg_size_1,
              A(kk,jj),

              /* inner: */
              KAAPIC_MODE_RW,
              KAAPIC_TYPE_FLT,
              bots_arg_size_1*bots_arg_size_1,
              A(ii,jj)
            );
          }
#else
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
          {
            if (A(ii,jj) == NULL)
              A(ii,jj) = allocate_clean_block();
            bmod(A(ii,kk), A(kk,jj), A(ii,jj));
          }
#endif
        } // for(jj...)
      } // if
    } //for(ii...)

#if defined(BOTS_KAAPI)
//    kaapic_sync();
#else
  #pragma omp taskwait
#endif
  } // for(kk...)
  
#if defined(BOTS_KAAPI)
  kaapic_end_parallel( KAAPIC_FLAG_STATIC_SCHED );
#endif
  bots_message(" completed!\n");
}
#undef A

#if defined(__COL_MAJOR)
#define A(i,j)    (BENCH[j*bots_arg_size+i])
#else
#define A(i,j)    (BENCH[i*bots_arg_size+j])
#endif
void sparselu_seq_call(float **BENCH)
{
  int ii, jj, kk;
  
  for (kk=0; kk<bots_arg_size; kk++)
  {
    lu0(A(kk,kk));
    for (jj=kk+1; jj<bots_arg_size; jj++)
      if (A(kk,jj) != NULL)
      {
        fwd(A(kk,kk), A(kk,jj));
      }
    for (ii=kk+1; ii<bots_arg_size; ii++)
      if (A(ii,kk) != NULL)
      {
        bdiv(A(kk,kk), A(ii,kk));
      }
    
    for (ii=kk+1; ii<bots_arg_size; ii++)
      if (A(ii,kk) != NULL)
        for (jj=kk+1; jj<bots_arg_size; jj++)
          if (A(kk,jj) != NULL)
          {
            if(A(ii,jj) == NULL)
              A(ii,jj) = allocate_clean_block();
            bmod(A(ii,kk), A(kk,jj), A(ii,jj));
          }
    
  }
}
#undef A

void sparselu_fini_seq (float **BENCH, char *pass)
{
//  print_structure(pass, BENCH);
}

void sparselu_fini_par (float **BENCH, char *pass)
{
  sparselu_fini_seq(BENCH, pass);
#if defined(BOTS_KAAPI)
  kaapic_finalize( );
#endif
}

#if defined(__COL_MAJOR)
#define A(i,j)    (BENCH[j*bots_arg_size+i])
#define B(i,j)    (SEQ[j*bots_arg_size+i])
#else
#define A(i,j)    (BENCH[i*bots_arg_size+j])
#define B(i,j)    (SEQ[i*bots_arg_size+j])
#endif
int sparselu_check(float **SEQ, float **BENCH)
{
  int ii,jj,ok=1;
  
  for (ii=0; ((ii<bots_arg_size) && ok); ii++)
  {
    for (jj=0; ((jj<bots_arg_size) && ok); jj++)
    {
      if ((B(ii,jj) == NULL) && (A(ii,jj) != NULL)) ok = FALSE;
      if ((B(ii,jj) != NULL) && (A(ii,jj) == NULL)) ok = FALSE;
      if ((B(ii,jj) != NULL) && (A(ii,jj) != NULL))
        ok = checkmat(B(ii,jj), A(ii,jj));
    }
  }
  if (ok) return BOTS_RESULT_SUCCESSFUL;
  else return BOTS_RESULT_UNSUCCESSFUL;
}
#undef A
#undef B

