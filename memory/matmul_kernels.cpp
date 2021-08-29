#include <matmul.h>

//#include <algorithm>

/* This routine performs a matmul operation using a straightforward
 * "naive", three-loop method.
 *    C := beta*C + alpha*A * B => C_i,j = Sum_l A_i,l * B_l,j
 *    where A (mxk), B (kxn), and C (mxn) are matrices stored in column-major format.
 *    On exit, A and B maintain their input values. */
void matmul_naive (const int m, const int n, const int k, const ValueType alpha, ValueType A[], const int lda, ValueType B[], const int ldb, const ValueType beta, ValueType C[], const int ldc)
{
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
         C[i + j*ldc] *= beta;
         for (int l = 0; l < k; ++l)
            C[i + j*ldc] += alpha * A[i + l*lda] * B[l + j*ldb];
      }
}

void matmul_vect (const int m, const int n, const int k, const ValueType alpha, ValueType A[], const int lda, ValueType B[], const int ldb, const ValueType beta, ValueType C[], const int ldc)
{
   // Restructed to improved vectorization.
   for (int j = 0; j < n; ++j)
   {
      if (beta == 0.0)
         for (int i = 0; i < m; ++i) C[i + j*ldc] = 0.0;
      else if (beta != 1.0)
         for (int i = 0; i < m; ++i) C[i + j*ldc] *= beta;

      for (int l = 0; l < k; ++l)
      {
         //if (B[l + j*ldb] != 0.0)
         {
            ValueType temp = alpha * B[l + j*ldb];
            #pragma ivdep
            for (int i = 0; i < m; ++i)
               C[i + j*ldc] += temp * A[i + l*lda];
         }
      }
   }
}

void matmul_unroll (const int m, const int n, const int k, const ValueType alpha, ValueType A[], const int lda, ValueType B[], const int ldb, const ValueType beta, ValueType C[], const int ldc)
{
   ValueType *RESTRICT _A = A;
   ValueType *RESTRICT _B = B;
   ValueType *RESTRICT _C = C;

#if (UNROLL > 8)
#error "Unroll > 8 not supported."
#endif
   //static_assert( unroll <= 8, "Unroll > 8 not supported.");
   const int unroll = UNROLL;

#ifdef __INTEL_COMPILER
__assume_aligned( _A, 64 );
__assume_aligned( _B, 64 );
__assume_aligned( _C, 64 );
__assume( lda % 8 == 0 );
__assume( ldb % 8 == 0 );
__assume( ldc % 8 == 0 );
#endif

   // Restructed to improved vectorization.
   for (int j = 0; j < n; ++j)
   {
      if (beta == 0.0)
         for (int i = 0; i < m; ++i) _C[i + j*ldc] = 0.0;
      else if (beta != 1.0)
         for (int i = 0; i < m; ++i) _C[i + j*ldc] *= beta;

      int l = 0;
      const int k_stop = k - unroll;
      for (; l < k_stop; l += unroll)
      {
         ValueType t0 = alpha * _B[l    + j*ldb];
         ValueType t1,t2,t3,t4,t5,t6,t7;
         if (unroll > 1) t1 = alpha * _B[l+1  + j*ldb];
         if (unroll > 2) t2 = alpha * _B[l+2  + j*ldb];
         if (unroll > 3) t3 = alpha * _B[l+3  + j*ldb];
         if (unroll > 4) t4 = alpha * _B[l+4  + j*ldb];
         if (unroll > 5) t5 = alpha * _B[l+5  + j*ldb];
         if (unroll > 6) t6 = alpha * _B[l+6  + j*ldb];
         if (unroll > 7) t7 = alpha * _B[l+7  + j*ldb];

         #pragma ivdep
//       #pragma omp simd safelen(8)// aligned( _A, _B, _C : 64 )
         for (int i = 0; i < m; ++i)
         {
            ValueType cij = _C[i + j*ldc];
            cij += t0 * _A[i + (l  )*lda];
            if (unroll > 1) cij += t1 * _A[i + (l+1)*lda];
            if (unroll > 2) cij += t2 * _A[i + (l+2)*lda];
            if (unroll > 3) cij += t3 * _A[i + (l+3)*lda];
            if (unroll > 4) cij += t4 * _A[i + (l+4)*lda];
            if (unroll > 5) cij += t5 * _A[i + (l+5)*lda];
            if (unroll > 6) cij += t6 * _A[i + (l+6)*lda];
            if (unroll > 7) cij += t7 * _A[i + (l+7)*lda];

            _C[i + j*ldc] = cij;
         }
      }

      for (; l < k; ++l)
      {
         //if (B[l + j*ldb] != 0.0)
         {
            ValueType temp = alpha * _B[l + j*ldb];
            for (int i = 0; i < m; ++i)
               _C[i + j*ldc] += temp * _A[i + l*lda];
         }
      }
   }
}

#define MIN(a,b) ( a < b ? a : b )
#define MAX(a,b) ( a > b ? a : b )

void matmul_blocked (const int m, const int n, const int k, const ValueType alpha, ValueType A[], const int lda, ValueType B[], const int ldb, const ValueType beta, ValueType C[], const int ldc)
{
   const int blockSize = BLOCKSIZE;

   #pragma omp parallel for collapse(2)
   for (int i = 0; i < m; i += blockSize)
      for (int j = 0; j < n; j += blockSize)
      {
         const int m_blk = MIN( blockSize, m-i);
         const int n_blk = MIN( blockSize, n-j);

         ValueType *C_blk = C + i + j*ldc;

         if (beta == 0.0)
            for (int jj = 0; jj < n_blk; ++jj)
               for (int ii = 0; ii < m_blk; ++ii)
                  C_blk[ii + jj*ldc] = 0.0;
         else if (beta != 1.0)
            for (int jj = 0; jj < n_blk; ++jj)
               for (int ii = 0; ii < m_blk; ++ii)
                  C_blk[ii + jj*ldc] *= beta;

         for (int l = 0; l < k; l += blockSize)
         {
            const int k_blk = MIN( blockSize, k-l);

            ValueType *A_blk = A + i + l*lda;
            ValueType *B_blk = B + l + j*ldb;

            const ValueType one = 1.0;
            //matmul_vect (m_blk, n_blk, k_blk, alpha, A_blk, lda, B_blk, ldb, one, C_blk, ldc);
            matmul_unroll (m_blk, n_blk, k_blk, alpha, A_blk, lda, B_blk, ldb, one, C_blk, ldc);
         }
      }
}
