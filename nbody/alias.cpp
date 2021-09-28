#ifndef __RESTRICT
#define __RESTRICT
#endif

void alias_kernel (const double alpha, const double beta, const double *__RESTRICT x, double *__RESTRICT y, const int n)
{
   for (int i = 0; i < n; ++i)
      y[i] = beta * y[i] + alpha * x[i];
}
