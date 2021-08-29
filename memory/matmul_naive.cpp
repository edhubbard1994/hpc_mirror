#include <matmul.h>

/* This routine performs a matmul operation using a straightforward
 * "naive", three-loop method.
 *    C := beta*C + alpha*A * B => C_i,j = Sum_l A_i,l * B_l,j
 *    where A (mxk), B (kxn), and C (mxn) are matrices stored in column-major format.
 *    On exit, A and B maintain their input values. */
void matmul_naive (const int m, const int n, const int k, const ValueType alpha,
                   ValueType A[], const int lda,
                   ValueType B[], const int ldb
                    const ValueType beta,
                   ValueType C[], const int ldc)
{
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
         C[i + j*ldc] *= beta;
         for (int l = 0; l < k; ++l)
            C[i + j*ldc] += alpha * A[i + l*lda] * B[l + j*ldb];
      }
}
