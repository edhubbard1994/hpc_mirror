#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

#ifndef __RESTRICT
# define __RESTRICT
#endif

static bool verbose = false;

template <typename ValueType>
void invSqrt (const int n)
{
   ValueType *x = aligned_alloc<ValueType>(n);
   ValueType *y = aligned_alloc<ValueType>(n);
   int *count = aligned_alloc<int>(n);

   const ValueType half(0.5);
   const ValueType three(3);

   for (int i = 0; i < n; ++i) {
      x[i] = 0.1 + ValueType( rand() ) / RAND_MAX;
      count[i] = 0;
   }

   int nloops = 1;

   int sum = 0, min = 10000, max = 0;
   ValueType ysum = 0;

   double runtime = 0;

   while (1)
   {
      for (int i = 0; i < n; ++i) count[i] = 0;

      TimerType t_start = getTimeStamp();

      for (int loop = 0; loop < nloops; ++loop)
      {
         for (int i = 0; i < n; ++i)
         {
            ValueType yi = 1.0;
            //ValueType yi = x[i];

            ValueType delta = 1.0;
            int niters = 0;
            while ( delta > 10*std::numeric_limits<ValueType>::epsilon() and niters++ < 20 )
            {
               ValueType ynext = half * yi * (three - x[i] * yi * yi );
               delta = fabs( ynext - yi );
               //printf("%d %e %e %e %e\n", niters, yi, delta, ynext, 1.0/sqrt(x[i]));
               yi = ynext;
            }

            y[i] = yi;
            count[i] += niters;
         }

         dummy_function( n, x, y, count );
      }

      TimerType t_stop = getTimeStamp();

      sum = 0; min = 10000; max = 0;

      for (int i = 0; i < n; ++i) {
         sum += count[i];
         min = std::min(min, count[i]);
         max = std::max(max, count[i]);
      }

      min /= nloops;
      max /= nloops;

      runtime = getElapsedTime( t_start, t_stop );
      if (runtime > 0.1)
         break;
      else
         nloops *= 2;
   }

   ValueType err2 = 0, ref2 = 0, errmax = 0;
   for (int i = 0; i < n; ++i)
   {
      if (i < 10 and verbose) printf("%e %e\n", y[i], 1.0/sqrt(x[i]));
      ValueType ref = 1.0/sqrt(x[i]);
      ValueType diff = y[i] - ref;
      err2 += diff*diff;
      ref2 += ref*ref;
      errmax = fmax( errmax, fabs(diff) );
   }

   printf("invSqrt:     %10.5f (ns) %.2f %d %d %e %e\n", 1e9*runtime/(n*nloops), float(sum)/(n*nloops), min, max, sqrt(err2/ref2), errmax);

   free(x);
   free(y);
   free(count);
}

#ifdef __ENABLE_VCL_SIMD

#if   (__ENABLE_VCL_SIMD == 128)
# define MAX_VECTOR_SIZE 128
#elif (__ENABLE_VCL_SIMD == 256)
# define MAX_VECTOR_SIZE 256
#elif (__ENABLE_VCL_SIMD == 512)
# define MAX_VECTOR_SIZE 512
#else
// Using default from available instruction set.
#endif

#include "vcl_helper.h"
const int VL = SIMD_Vector_Length<double>();

//#include <immintrin.h>

void invSqrtSimd (const int n)
{
   typedef double ValueType;

   ValueType *x = aligned_alloc<ValueType>(n);
   ValueType *y = aligned_alloc<ValueType>(n);
   int *count = aligned_alloc<int>(n);

   for (int i = 0; i < n; ++i) {
      x[i] = 0.1 + ValueType( rand() ) / RAND_MAX;
      count[i] = 0;
   }

   using vector_type  = vcl_type<double, VL>::type;
   using counter_type = vcl_type<long, VL>::type;  // This must match the vector type.
   using mask_type    = vcl_mask_type< vector_type >::type;

   int nloops = 1;

   int min = 10000, max = 0, sum = 0;
   int total_iters = 0;
   double runtime = 0;

   while (1)
   {
      min = 10000;
      max = 0;
      total_iters = 0;
      sum = 0;
      for (int i = 0; i < n; ++i) count[i] = 0;

      TimerType t_start = getTimeStamp();

      for (int loop = 0; loop < nloops; ++loop)
      {
         for (int i = 0; i < n; i += VL)
         {
            vector_type yi = 1.0;
            vector_type xi;

            if (i + VL <= n)
               xi.load_a( &x[i] );
            else
            {
               xi = x[i]; // Pack all with the same problem;
               xi.load_partial( n-i, &x[i] ); // Update just the value terms.
            }

            vector_type delta = 1.0;

            mask_type not_done = true;

            const vector_type half(0.5);
            const vector_type three(3);

            counter_type niters_i ( 0);
            while ( any( not_done ) )
            {
               vector_type ynext = half * yi * (three - xi * yi * yi );
               delta = abs( ynext - yi );
               //printf("%d %e %e %e %e %e %e %e\n", niters, yi[0], yi[1], delta[0], delta[1], ynext[0], ynext[1], 1.0/sqrt(x[i]));
               yi = where( not_done, ynext, yi );
               niters_i = where( not_done, niters_i + 1, niters_i );

               not_done = delta > 10*std::numeric_limits<ValueType>::epsilon();

               total_iters ++;
            }

            if (i + VL <= n )
               yi.store_a( &y[i] );
            else
               yi.store_partial( n-i, &y[i] );

            for (int ii = 0; ii < VL && i+ii < n; ++ii)
               count[i+ii] += niters_i[ii];
         }

         dummy_function( n, x );
         dummy_function( n, y );
         dummy_function( n, count );
      }

      TimerType t_stop = getTimeStamp();

      for (int i = 0; i < n; ++i) {
         min = std::min(min, count[i]);
         max = std::max(max, count[i]);
         sum += count[i];
      }

      min /= nloops;
      max /= nloops;

      runtime = getElapsedTime( t_start, t_stop );
      if (runtime > 0.1)
         break;
      else
         nloops *= 2;
   }

   ValueType err2 = 0, ref2 = 0, errmax = 0;
   for (int i = 0; i < n; ++i)
   {
      if (i < 10 and verbose) printf("%e %e\n", y[i], 1.0/sqrt(x[i]));
      ValueType ref = 1.0/sqrt(x[i]);
      ValueType diff = y[i] - ref;
      err2 += diff*diff;
      ref2 += ref*ref;
      errmax = fmax( errmax, fabs(diff) );
   }

   printf("invSqrtSimd: %10.5f (ns) %.2f %d %d %e %e %d %.2f\n", 1e9*runtime/(n*nloops), float(sum)/(n*nloops), min, max, sqrt(err2/ref2), errmax, VL, float(total_iters*VL)/(n*nloops));

   free(x);
   free(y);
   free(count);

/*
   {
      __m512d x{0,1,2,3,4,5,6,7};
      __m512d y{1,1,2,3,4,5,6,7};
      __m512d z = _mm512_set1_pd(0);

      auto mask = _mm512_cmplt_pd_mask(x,y);
      printf("mask: %x %d\n", mask, bool(mask));

      auto mask2 = _mm512_cmplt_pd_mask(x,z);
      printf("mask2: %x %d\n", mask2, bool(mask2));
   }*/
}

#endif

#ifndef __RESTRICT
#define __RESTRICT
#endif

static double beta = 1.0, alpha = 0.5;
static int    flip = 0;

//external
void alias_kernel (const double alpha, const double beta, const double *__RESTRICT x, double *__RESTRICT y, double *__RESTRICT z, const int n);

void test_alias (const int n)
{
   typedef double ValueType;

   ValueType *x = aligned_alloc<ValueType>(n);
   ValueType *y = aligned_alloc<ValueType>(n);
   ValueType *y_ref = aligned_alloc<ValueType>(n);

   for (int i = 0; i < n; ++i) {
      x[i] = ValueType( rand() ) / RAND_MAX;
      y[i] = x[i];
      y_ref[i] = x[i];
   }

   int offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
   const int noffsets = sizeof(offsets)/sizeof(offsets[0]);

   dummy_function( noffsets, offsets );

   for (int e = 0; e < noffsets; ++e)
   {
      const int offset = offsets[e];

      int ntests = 10;
      double runtime = 0;
      while (1)
      {
         for (int i = 0; i < n; ++i)
            y[i] = x[i];

         TimerType t_start = getTimeStamp();

         for (int t = 0; t < ntests; ++t)
         {
            double *z = y;
            alias_kernel ( alpha, beta, x+offset, y, z, n-offset );

            dummy_function( n, x, y, z );
         }

         TimerType t_stop = getTimeStamp();

         runtime = getElapsedTime( t_start, t_stop );
         if (runtime < 0.01) {
            ntests *= 2;
            dummy_function( n, x, y );
         }
         else
            break;
      }

      for (int i = 0; i < n; ++i)
         y_ref[i] = x[i];

      for (int t = 0; t < ntests; ++t)
         for (int i = 0; i < n-offset; ++i) {
            double *z = y_ref;
            z[i] = beta * y_ref[i] + alpha * x[i+offset];
         }

      double err2 = 0, ref2 = 0, err0 = 0;
      for (int i = 0; i < n; ++i) {
         double diff = fabs( y_ref[i] - y[i] );
         err2 += diff*diff;
         ref2 += y_ref[i]*y_ref[i];
         err0 = std::max( err0, diff );
      }

      err2 = sqrt(err2 / ref2);
      bool success = ( err2 < 10*DBL_EPSILON );

      printf("alias: %10.5f (ns) offset %2d %s\n", 1e9*runtime/((n-offset)*ntests), offset, (success) ? "Passed" : "Failed");
   }

   free(x);
   free(y);
   free(y_ref);
}

template <typename T>
void histogram (const T *x, const int n, int *count, const int nbins)
{
   const T dx = T(1) / nbins;
   const T one_over_dx = T(1) / dx;

   //__assume_aligned(x, 64);
   //__assume_aligned(count, 64);

   #pragma omp simd aligned( x, count : 64 )
   for (int i = 0; i < n; ++i)
   {
     // int bid = floor( x[i] / dx );
     // int bid = floor( x[i] * one_over_dx );
      int bid = x[i] * one_over_dx;
      ++ count[ bid ];
   }
}

template <typename ValueType>
void histogram (const int n, const int nbins)
{
   ValueType *x = aligned_alloc<ValueType>(n);

   int *count = aligned_alloc<int>(nbins);

   for (int i = 0; i < nbins; ++i)
      count[i] = 0;

   for (int i = 0; i < n; ++i) {
      ValueType r = ValueType( rand() ) / RAND_MAX; // [0,1)
      //ValueType r = i / ValueType(n);
      x[i] = r;
   }

   int jloops = 1;
   double runtime = 0;

   while (1)
   {
      for (int i = 0; i < nbins; ++i)
         count[i] = 0;

      TimerType t_start = getTimeStamp();

      for (int j = 0; j < jloops; ++j)
      {
         histogram( x, n, count, nbins );

         dummy_function( n, x, count );
      }

      TimerType t_stop = getTimeStamp();

      runtime = getElapsedTime( t_start, t_stop );
      if (runtime > 0.1)
         break;
      else
         jloops *= 2;
   }

   printf("histogram: %f (ns) nbins= %d\n", 1e9*runtime/(n*jloops), nbins);

   free(x);
   free(count);
}

template <typename ValueType>
void run_tests (const int n)
{
   invSqrt<ValueType>(n);

#ifdef __ENABLE_VCL_SIMD
   invSqrtSimd(n);
#endif

   test_alias(n);

   int bins[] = {n, 10000, 2000, 1000, 200, 100, 20, 10, 8, 4, 2, 1};

   printf("float\n");
   for (int i = 0; i < sizeof(bins)/sizeof(bins[0]); i++) {
      histogram<float>(n, bins[i]);
   }
   printf("double\n");
   for (int i = 0; i < sizeof(bins)/sizeof(bins[0]); i++) {
      histogram<double>(n, bins[i]);
   }
}

void help (FILE *fp)
{
   fprintf(fp,"--help    | -h   : Print help message.\n");
   fprintf(fp,"--nelems  | -n   : # of elements (100).\n");
   fprintf(fp,"--float   | -f   : Use 32-bit floats.\n");
   fprintf(fp,"--double  | -d   : Use 64-bit doubles. (default)\n");
   fprintf(fp,"--verbose | -v\n");
}

int main (int argc, char* argv[])
{
   /* Define the number of particles. The default is 100. */
   int n = 100;

   /* ValueType? (float or double) */
   bool useDouble = true;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); help(stderr); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help(stdout);
         return 0;
      }
      else if (strcmp(argv[i],"--nelems") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--nelems|-n");
         i++;
         if (not(isdigit(*argv[i])))
            { fprintf(stderr,"Invalid value for option \"--nelems\" %s\n", argv[i]); help(stderr); return 1; }
         n = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--alpha") == 0 || strcmp(argv[i],"-a") == 0)
      {
         check_index(i+1,"--alpha");
         i++;
         alpha = atof( argv[i] );
      }
      else if (strcmp(argv[i],"--beta") == 0 || strcmp(argv[i],"-b") == 0)
      {
         check_index(i+1,"--beta");
         i++;
         beta = atof( argv[i] );
      }
      else if (strcmp(argv[i],"--double") == 0 || strcmp(argv[i],"-d") == 0)
      {
         useDouble = true;
      }
      else if (strcmp(argv[i],"--float") == 0 || strcmp(argv[i],"-f") == 0)
      {
         useDouble = false;
      }
      else if (strcmp(argv[i],"--alias") == 0)
      {
         flip = true;
      }
      else if (strcmp(argv[i],"--verbose") == 0 || strcmp(argv[i],"-v") == 0)
      {
         verbose = true;
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         help(stderr);
         return 1;
      }
   }

   if (useDouble)
      run_tests<double>( n );
   else
      run_tests<float >( n );

   return 0;
}
