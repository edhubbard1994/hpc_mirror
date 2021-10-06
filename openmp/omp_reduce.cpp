#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <cfloat>

#include <algorithm>
#include <string>
#include <vector>

#include "timer.h"
#include "omp_helper.h"
#include "dummy.h"

void help(const char *prg, FILE *fp = stderr)
{
   fprintf(fp,"%s: --help|-h --nelems|-n <#> --niters|-i --method|-m <0,1,2>\n", prg);

   return;
}

/* Test if the input value is not a power-of-2.
 * Any power-of-2 minus 1 is all 1's with a right shift.
 * Ex: 4 = 100
 *     3 = 011
 */
inline bool isPower2( const unsigned int x )
{
   return (x & (x - 1)) == 0;
}

/* Find a power-of-2 that's less than the input value.
 */
inline unsigned int lowerPower2( const unsigned int x )
{
   unsigned int pwr2 = 1;
   while ( (pwr2 << 1) < x )
      pwr2 <<= 1;

   return pwr2;
}

double f( const double x)
{
   return sqrt(x/3.0);
}

double reduce_serial( const int n, double x[] )
{
   double xsum(0);

   for (int i = 0; i < n; ++i)
      xsum += f( x[i] );

   return xsum;
}

double reduce_auto( const int n, double x[] )
{
   double xsum(0);

   #pragma omp parallel for reduction(+:xsum)
   for (int i = 0; i < n; ++i)
      xsum += f( x[i] );

   return xsum;
}

double reduce_critical( const int n, double x[] )
{
   double xsum(0);

   #pragma omp parallel shared(xsum)
   {
      double my_sum(0);

      #pragma omp for nowait
      for (int i = 0; i < n; ++i)
         my_sum += f( x[i] );

      #pragma omp critical
      xsum += my_sum;
   }

   return xsum;
}

double reduce_recursion( const int n, double x[] )
{
#define MAX_THREADS (256+1)
   static double thread_sums[MAX_THREADS][1];

   #pragma omp parallel shared(thread_sums)
   {
      double my_sum(0);

      #pragma omp for nowait
      for (int i = 0; i < n; ++i)
         my_sum += f( x[i] );

      // Put my sum onto the shared list.
      const int thread_id = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      thread_sums[thread_id][0] = my_sum;
      #pragma omp barrier

      // Recursively add with the neighbor on the upper half of the list.
      const unsigned int active_threads = lowerPower2( nthreads );
      for (unsigned int stride = active_threads; stride > 0; stride >>= 1)
      {
         if (thread_id < stride and thread_id + stride < nthreads)
            thread_sums[thread_id][0] += thread_sums[thread_id+stride][0];

         #pragma omp barrier
      }
   }

   return thread_sums[0][0];
}

int main(int argc, char* argv[])
{
   int nelems = 1000; // Length of the array
   int niters = 100;
   int method = 0;
   bool check = false;

   const char* method_names[] = {"serial","omp-auto","omp-recusive"};

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"--nelems") == 0 || strcmp(argv[i],"-n") == 0 )
      {
         check_index(i+1,"--nelems|-n");
         i++;
         if (isdigit(*argv[i]))
            nelems = atoi( argv[i] );
      }
      else if ( strcmp(argv[i],"--niters") == 0 || strcmp(argv[i],"-i") == 0 )
      {
         check_index(i+1,"--niters|-n");
         i++;
         if (isdigit(*argv[i]))
            niters = atoi( argv[i] );
      }
      else if ( strcmp(argv[i],"--method") == 0 || strcmp(argv[i],"-m") == 0 )
      {
         check_index(i+1,"--method|-m");
         i++;
         if (isdigit(*argv[i]))
            method = atoi( argv[i] );
      }
      else if ( strcmp(argv[i],"--check") == 0 || strcmp(argv[i],"-c") == 0 )
      {
         check = true;
      }
      else
      {
         help(argv[0], stderr);
         return 0;
      }
   }

   int num_threads = 1;
#ifdef _OPENMP
   #pragma omp parallel
   #pragma omp single
   num_threads = omp_get_num_threads();
#endif

   if (num_threads > MAX_THREADS)
   {
      fprintf(stderr,"too many threads requested: %d %d\n", num_threads, MAX_THREADS);
      return 1;
   }

   if ( not(method >= 0 and method <= 3) )
   {
      fprintf(stderr,"invalid method %d\n", method);
      return 3;
   }

   fprintf(stderr,"OpenMP Parallel reduce example: nelems= %d, niters= %d, num_threads= %d, method= %d\n", nelems, niters, num_threads, method);

   double xmin = DBL_EPSILON, xmax = 1;
   double *x = new double [nelems];

   srand(100);
   for (int i = 0; i < nelems; ++i)
   {
      double r = double( rand() ) / RAND_MAX;
      x[i] = xmin + r * (xmax - xmin);
   }

   double sum_ref = 0;
   double serial_runtime = 0;

   if (method == 0 or check)
   {
      TimerType t0 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
      {
         dummy_function( nelems, (void*)x );
         sum_ref += reduce_serial( nelems, x );
      }

      TimerType t1 = getTimeStamp();

      serial_runtime = getElapsedTime(t0,t1);

      printf("serial reduce took %f (ms)\n", 1000*serial_runtime/niters);
   }

   if (method == 1)
   {
      double sum(0);

      TimerType t0 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
      {
         dummy_function( nelems, (void*)x );
         sum += reduce_auto( nelems, x );
      }

      TimerType t1 = getTimeStamp();

      double runtime = getElapsedTime(t0,t1);
      bool passed = ( fabs( sum - sum_ref ) / sum_ref ) < (100*DBL_EPSILON);

      printf("built-in reduce took %f (ms)", 1000*runtime/niters);
      if (check)
         printf(" %s %e %e %e", passed ? "True" : "False", sum, sum_ref, ( fabs( sum - sum_ref ) / sum_ref ));
      printf("\n");
   }
   else if (method == 2)
   {
      double sum(0);

      TimerType t0 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
         sum += reduce_recursion( nelems, x );

      TimerType t1 = getTimeStamp();

      double runtime = getElapsedTime(t0,t1);
      bool passed = ( fabs( sum - sum_ref ) / sum_ref ) < (100*DBL_EPSILON);

      printf("recursive reduce took %f (ms)", 1000*runtime/niters);
      if (check)
         printf(" %s %e %e %e", passed ? "True" : "False", sum, sum_ref, ( fabs( sum - sum_ref ) / sum_ref ));
      printf("\n");
   }
   else if (method == 3)
   {
      double sum(0);

      TimerType t0 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
         sum += reduce_critical( nelems, x );

      TimerType t1 = getTimeStamp();

      double runtime = getElapsedTime(t0,t1);
      bool passed = ( fabs( sum - sum_ref ) / sum_ref ) < (100*DBL_EPSILON);

      printf("critical reduce took %f (ms)", 1000*runtime/niters);
      if (check)
         printf(" %s %e %e %e", passed ? "True" : "False", sum, sum_ref, ( fabs( sum - sum_ref ) / sum_ref ));
      printf("\n");
   }

   delete [] x;

   return 0;
}
