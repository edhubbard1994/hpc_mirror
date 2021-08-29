#ifndef __matmul_h
#define __matmul_h

#ifndef RESTRICT
#define RESTRICT
#endif

#ifndef UNROLL
# define UNROLL (8)
#endif
#ifndef BLOCKSIZE
# define BLOCKSIZE (64)
#endif

#ifdef WITH_PAPI
#include <papi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

typedef double ValueType;

#ifdef __cplusplus
extern "C"
{
#endif

typedef void (*matmul_ptr) (const int m, const int n, const int k, const ValueType alpha,
                            ValueType A[], const int lda,
                            ValueType B[], const int ldb,
                            const ValueType beta,
                            ValueType C[], const int ldc);

void matmul_blas (const int m, const int n, const int k, const double alpha,
                  double A[], const int lda,
                  double B[], const int ldb,
                  const double beta,
                  double C[], const int ldc);

void matmul_naive (const int m, const int n, const int k, const ValueType alpha,
                   ValueType A[], const int lda,
                   ValueType B[], const int ldb,
                   const ValueType beta,
                   ValueType C[], const int ldc);

void matmul_vect (const int m, const int n, const int k, const ValueType alpha,
                   ValueType A[], const int lda,
                   ValueType B[], const int ldb,
                   const ValueType beta,
                   ValueType C[], const int ldc);

void matmul_unroll (const int m, const int n, const int k, const ValueType alpha,
                    ValueType A[], const int lda,
                    ValueType B[], const int ldb,
                    const ValueType beta,
                    ValueType C[], const int ldc);

void matmul_blocked (const int m, const int n, const int k, const ValueType alpha,
                    ValueType A[], const int lda,
                    ValueType B[], const int ldb,
                    const ValueType beta,
                    ValueType C[], const int ldc);

void dummy( double *x );

#ifdef __cplusplus
}
#endif

#endif
