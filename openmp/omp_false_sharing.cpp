#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include "timer.h"
#include "omp_helper.h"
#include "aligned_allocator.h"

double drand (void)
{
   return drand48();
}

// Estimate PI by integration: Int{ dx / (1 + x^2) } ~= pi/4
template <typename ValueType>
ValueType calcPiInt (const int n, const int numThreads)
{
   TimerType t0 = getTimeStamp();

   ValueType step = 1.0 / ValueType(n);
   ValueType sum = 0;

   #pragma omp parallel default(shared)
   {
      #pragma omp for reduction(+:sum) nowait
      for (int i = 0; i < n; ++i)
      {
         ValueType x = (i + 0.5) * step;
         sum += 1.0 / (1.0 + x*x);
      }
   }

   ValueType pi = sum * (4 * step);
   TimerType t1 = getTimeStamp();

   printf("Integration: pi = %e %f(%%) %d %f (ms) %d\n", pi, 100*fabs(M_PI-pi)/M_PI, n, 1000.*getElapsedTime(t0, t1), numThreads);

   return pi;
}

template <typename ValueType, int len>
ValueType calcPiInt_FS (const int n, const int numThreads)
{
   TimerType t0 = getTimeStamp();

   ValueType step = 1.0 / ValueType(n);

   typedef ValueType ThreadSumType[len];
   ThreadSumType *threadSums = new ThreadSumType[numThreads];

   //for (int k = 0; k < numThreads; ++k)
   //   threadSums[k][0] = 0;

   #pragma omp parallel default(shared) shared(threadSums)
   {
      int thread_id = 0;
#ifdef _OPENMP
      thread_id = omp_get_thread_num();
#endif

      threadSums[thread_id][0] = 0;

      #pragma omp for nowait
      for (int i = 0; i < n; ++i)
      {
         ValueType x = (i + 0.5) * step;
         threadSums[thread_id][0] += 1.0 / (1.0 + x*x);
      }
   }

   // Sum up the private accumulators.
   ValueType sum = 0;
   for (int k = 0; k < numThreads; ++k)
      sum += threadSums[k][0];

   ValueType pi = sum * (4 * step);
   TimerType t1 = getTimeStamp();

   delete [] threadSums;

   printf("IntegrationFS: pi = %e %f(%%) %d %f (ms) %d %lu\n", pi, 100*fabs(M_PI-pi)/M_PI, n, 1000.*getElapsedTime(t0, t1), numThreads, sizeof(ThreadSumType));

   return pi;
}

// Estimate PI by integration: Int{ dx / (1 + x^2) } ~= pi/4
template <typename ValueType>
ValueType calcPiIntAtomic (const int n, const int numThreads)
{
   TimerType t0 = getTimeStamp();

   ValueType step = 1.0 / ValueType(n);
   ValueType sum = 0;

   #pragma omp parallel default(shared)
   {
      #pragma omp for nowait
      for (int i = 0; i < n; ++i)
      {
         ValueType x = (i + 0.5) * step;
         #pragma omp atomic
         sum += 1.0 / (1.0 + x*x);
      }
   }

   ValueType pi = sum * (4 * step);
   TimerType t1 = getTimeStamp();

   printf("Integration: pi = %e %f(%%) %d %f (ms) %d\n", pi, 100*fabs(M_PI-pi)/M_PI, n, 1000.*getElapsedTime(t0, t1), numThreads);

   return pi;
}

// Estimate PI by Monte Carlo method.
void calcPiMC (const int n, const int numThreads)
{
   TimerType t0 = getTimeStamp();
   int counter = 0;

   #pragma omp parallel for default(shared) reduction(+:counter)
   for (int i = 0; i < n; ++i)
   {
      double x = drand();
      double y = drand();
      if ((x*x + y*y) <= 1.0)
         counter ++;
   }

   double pi = 4.0 * (double(counter) / n);

   TimerType t1 = getTimeStamp();

   printf("Monte-Carlo: pi = %e %f(%%) %d %f (ms) %d\n", pi, 100*fabs(M_PI-pi)/M_PI, n, 1000.*getElapsedTime(t0, t1), numThreads);

   return;
}

int main(int argc, char *argv[])
{
   int len = 1024; // Length of the array.
   int method = 0; // Which method?

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         fprintf(stderr,"omp_false_sharing --help|-h --length|-n --method|-m\n");
         return 1;
      }
      else if (strcmp(argv[i],"--length") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--length|-n");
         i++;
         if (isdigit(*argv[i]))
            len = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--method") == 0 || strcmp(argv[i],"-m") == 0)
      {
         check_index(i+1,"--method|-m");
         i++;
         if (isdigit(*argv[i]))
            method = atoi( argv[i] );
      }
   }

   int max_threads = 1;
#ifdef _OPENMP
   max_threads = omp_get_max_threads();
#endif

   int num_threads = 1;
#ifdef _OPENMP
   #pragma omp parallel
   #pragma omp single
   {
      num_threads = omp_get_num_threads();
   }
#endif

   printf("OpenMP Parallel False-sharing example: length= %d, num_threads= %d, method= %d\n", len, num_threads, method);
   double pi_sum = 0;

   for (int iter = 0; iter < 20; iter++)
   {
      if (method == 1)
         pi_sum += calcPiInt<double>(len, num_threads);
      else if (method == 2)
         if (iter % 2 == 0)
            pi_sum += calcPiInt_FS<double,1>(len, num_threads);
         else
            pi_sum += calcPiInt_FS<double,8>(len, num_threads);
      else if (method == 3)
         pi_sum += calcPiIntAtomic<double>(len, num_threads);
      //else
      //   calcPiMC(len, num_threads);
   }

   return pi_sum < 0;
}
