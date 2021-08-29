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

#include <my_timer.h>
#include <aligned_allocator.h>

#ifndef __RESTRICT
#  define __RESTRICT
#endif

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

   const int nloops = 10;

   int sum = 0, min = 10000, max = 0;
   ValueType ysum = 0;

   myTimer_t t_start = getTimeStamp();

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
         count[i] = niters;
      }

      for (int i = 0; i < n; ++i) {
         sum += count[i];
         min = std::min(min, count[i]);
         max = std::max(max, count[i]);
         ysum += y[i];
      }

      //memcpy( &y[loop], &y[0], sizeof(ValueType) );
      srand( y[loop] );
   }

   myTimer_t t_stop = getTimeStamp();

   ValueType err2 = 0, ref2 = 0, errmax = 0;
   for (int i = 0; i < n; ++i)
   {
      if (i < 10) printf("%e %e\n", y[i], 1.0/sqrt(x[i]));
      ValueType ref = 1.0/sqrt(x[i]);
      ValueType diff = y[i] - ref;
      err2 += diff*diff;
      ref2 += ref*ref;
      errmax = fmax( errmax, fabs(diff) );
   }

   printf("invSqrt: %f (ns) %.2f %d %d %e %e\n", 1e9*getElapsedTime( t_start, t_stop )/(n*nloops), float(sum)/(n*nloops), min, max, sqrt(err2/ref2), errmax);

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

#include <vcl/vectorclass.h>
#include <vcl/vectormath_exp.h>

#if (MAX_VECTOR_SIZE == 128)
# define VCL_DBL_TYPE Vec2d
# define VCL_BOOL_TYPE Vec2db
# define VCL_LONG_TYPE Vec2q
#elif (MAX_VECTOR_SIZE == 256)
# define VCL_DBL_TYPE Vec4d
# define VCL_BOOL_TYPE Vec4db
# define VCL_LONG_TYPE Vec4q
#elif (MAX_VECTOR_SIZE == 512)
# define VCL_DBL_TYPE Vec8d
# define VCL_BOOL_TYPE Vec8db
# define VCL_LONG_TYPE Vec8q
#else
# error 'Unknown MAX_VECTOR_SIZE in VCL'
#endif

#include <immintrin.h>
#include <vcl/vectorclass.h>
#include <vcl/vectormath_exp.h>

template <typename BoolSimdType>
bool any( const BoolSimdType &x ) { return horizontal_or(x); }

template <typename BoolSimdType>
bool all( const BoolSimdType &x ) { return horizontal_and(x); }

template <typename BoolSimdType, typename SimdType>
SimdType where( const BoolSimdType &mask, const SimdType &a, const SimdType &b) {
   return select( mask, a, b );
}

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

   typedef VCL_DBL_TYPE simd_type;
   typedef VCL_LONG_TYPE int_simd_type;
   typedef VCL_BOOL_TYPE bool_type;

   const int VL = sizeof(simd_type) / sizeof(double);

   const int nloops = 10;

   int sum = 0, min = 10000, max = 0, total_iters = 0;
   ValueType ysum = 0;

   myTimer_t t_start = getTimeStamp();

   for (int loop = 0; loop < nloops; ++loop)
   {
      for (int i = 0; i < n; i += VL)
      {
         simd_type yi = 1.0;
         simd_type xi;

         if (i + VL <= n)
            xi.load_a( &x[i] );
         else
         {
            xi = x[i]; // Pack all with the same problem;
            xi.load_partial( n-i, &x[i] ); // Update just the value terms.
         }

         simd_type delta = 1.0;

         bool_type notDone = true;

         const simd_type half(0.5);
         const simd_type three(3);

         int_simd_type niters_i ( 0);
         while ( any( notDone ) )
         {
            simd_type ynext = half * yi * (three - xi * yi * yi );
            delta = abs( ynext - yi );
            //printf("%d %e %e %e %e %e %e %e\n", niters, yi[0], yi[1], delta[0], delta[1], ynext[0], ynext[1], 1.0/sqrt(x[i]));
            yi = where( notDone, ynext, yi );
            niters_i = where( notDone, niters_i + 1, niters_i );

            notDone = delta > 10*std::numeric_limits<ValueType>::epsilon();

            total_iters ++;
         }

         if (i + VL <= n )
            yi.store_a( &y[i] );
         else
            yi.store_partial( n-i, &y[i] );

         for (int ii = 0; ii < VL && i+ii < n; ++ii)
            count[i+ii] = niters_i[ii];
      }

      for (int i = 0; i < n; ++i) {
         sum += count[i];
         min = std::min(min, count[i]);
         max = std::max(max, count[i]);
         ysum += y[i];
      }

      //memcpy( &y[loop], &y[0], sizeof(ValueType) );
      srand( y[loop] );
   }

   myTimer_t t_stop = getTimeStamp();

   ValueType err2 = 0, ref2 = 0, errmax = 0;
   for (int i = 0; i < n; ++i)
   {
      if (i < 10) printf("%e %e\n", y[i], 1.0/sqrt(x[i]));
      ValueType ref = 1.0/sqrt(x[i]);
      ValueType diff = y[i] - ref;
      err2 += diff*diff;
      ref2 += ref*ref;
      errmax = fmax( errmax, fabs(diff) );
   }

   printf("invSqrtSimd: %f (ns) %.2f %d %d %e %e %d %.2f\n", 1e9*getElapsedTime( t_start, t_stop )/(n*nloops), float(sum)/(n*nloops), min, max, sqrt(err2/ref2), errmax, VL, float(total_iters*VL)/(n*nloops));

   free(x);
   free(y);
   free(count);

   {
      __m512d x{0,1,2,3,4,5,6,7};
      __m512d y{1,1,2,3,4,5,6,7};
      __m512d z = _mm512_set1_pd(0);

      auto mask = _mm512_cmplt_pd_mask(x,y);
      printf("mask: %x %d\n", mask, bool(mask));

      auto mask2 = _mm512_cmplt_pd_mask(x,z);
      printf("mask2: %x %d\n", mask2, bool(mask2));
   }
}

#endif

#ifndef __RESTRICT
#define __RESTRICT
#endif

void alias_1 (const double *__RESTRICT x, double *__RESTRICT y, const int n)
{
   //#pragma vector aligned
   for (int i = 0; i < n; ++i)
      y[i] = sqrt( x[i] );
}

void alias (const int n)
{
   typedef double ValueType;

   ValueType *x = aligned_alloc<ValueType>(n);
   ValueType *y = aligned_alloc<ValueType>(n);

   for (int i = 0; i < n; ++i) {
      x[i] = ValueType( rand() ) / RAND_MAX;
      y[i] = x[i];
   }

   for (int i = 0; i < 12; ++i)
      if (i % 3 == 0)
      {
         const int j = i % n;
         const int offset = x[j] * j; // [0,8)

         myTimer_t t_start = getTimeStamp();

         alias_1( x+offset, y, n-offset );

         myTimer_t t_stop = getTimeStamp();

         printf("alias: %f (ns) 0 offset %d\n", 1e9*getElapsedTime( t_start, t_stop )/(n-offset), offset);
      }
      else if (i % 3 == 1)
      {
         const int offset = 2;

         myTimer_t t_start = getTimeStamp();

         alias_1( x+offset, y, n-offset );

         myTimer_t t_stop = getTimeStamp();

         printf("alias: %f (ns) 1 offset=%d\n", 1e9*getElapsedTime( t_start, t_stop )/(n-offset), offset);
      }
      else if (i % 3 == 2)
      {
         const int j = i % n;
         const int offset = std::min(n, int(x[j] * 32)); // [0,32)

         myTimer_t t_start = getTimeStamp();

         alias_1( x, x + offset, n-offset );

         myTimer_t t_stop = getTimeStamp();

         printf("alias: %f (ns) 2 n-%d\n", 1e9*getElapsedTime( t_start, t_stop )/(n-offset), j);
      }
}

void histogram (const double *x, const int n, int *count, const int nbins)
{
   const double dx = 1.0 / nbins;
   const double one_over_dx = 1.0 / dx;

   //#pragma omp simd aligned( x, count : 64 )
   for (int i = 0; i < n; ++i)
   {
      int bid = floor( x[i] / dx );
    //int bid = floor( x[i] * one_over_dx );
      ++ count[ bid ];
   }
}

void histogram (const int n, const int nbins)
{
   typedef double ValueType;

   ValueType *x = aligned_alloc<ValueType>(n);

   int *count = aligned_alloc<int>(nbins);

   for (int i = 0; i < nbins; ++i)
      count[i] = 0;

   for (int i = 0; i < n; ++i) {
      ValueType r = ValueType( rand() ) / RAND_MAX; // [0,1)
      x[i] = r;
   }

   myTimer_t t_start = getTimeStamp();

   histogram( x, n, count, nbins );

   myTimer_t t_stop = getTimeStamp();
   printf("histogram: %f (ns) nbins= %d\n", 1e9*getElapsedTime( t_start, t_stop )/n, nbins);
}

template <typename ValueType>
void run_tests (const int n)
{
   invSqrt<ValueType>(n);

#ifdef __ENABLE_VCL_SIMD
   invSqrtSimd(n);
#endif

   alias(n);

   int bins[] = {1000, 100, 10, 8, 4, 2, 1};

   for (int i = 0; i < sizeof(bins)/sizeof(bins[0]); i++)
      histogram(n, bins[i]);
}

void help(const char* prg)
{
   if (prg) fprintf(stderr,"%s:\n", prg);
   fprintf(stderr,"\t--help | -h     : Print help message.\n");
   fprintf(stderr,"\t--nelems | -n   : # of particles (100).\n");
   fprintf(stderr,"\t--float | -f    : Use 32-bit floats.\n");
   fprintf(stderr,"\t--double | -d   : Use 64-bit doubles. (default)\n");
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
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); help(argv[0]); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help(argv[0]);
         return 0;
      }
      else if (strcmp(argv[i],"--nelems") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--nelems|-n");
         i++;
         if (not(isdigit(*argv[i])))
            { fprintf(stderr,"Invalid value for option \"--nelems\" %s\n", argv[i]); help(argv[0]); return 1; }
         n = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--double") == 0 || strcmp(argv[i],"-d") == 0)
      {
         useDouble = true;
      }
      else if (strcmp(argv[i],"--float") == 0 || strcmp(argv[i],"-f") == 0)
      {
         useDouble = false;
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         help(argv[0]);
         return 1;
      }
   }

   if (useDouble)
      run_tests<double>( n );
   else
      run_tests<float >( n );

   return 0;
}
