#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "timer.h"
#include "omp_helper.h"

int global_counter = 0; // Shared (or global) variable.
double global_sum = 0;

int doSomeThing (int i)
{
   double x = sqrt(i*2);
   return int(x);
}

int test_critical(const int n)
{
   TimerType t0 = getTimeStamp();
   global_counter = 0;
   global_sum = 0;

   #pragma omp parallel for default(shared) 
   for (int i = 0; i < n; ++i)
   {
      double myValue = doSomeThing(i);
      #pragma omp critical
      {
         global_sum += myValue;
         if (myValue > 10)
            global_counter++;
      }
   }

   TimerType t1 = getTimeStamp();
   printf("critical: global_sum = %e %d %f\n", global_sum, global_counter, 1000*getElapsedTime(t0,t1));

   return 0;
}

int test_manual_reduction(const int n)
{
   TimerType t0 = getTimeStamp();
   global_counter = 0;
   global_sum = 0;

   #pragma omp parallel default(shared)
   {

   int my_counter = 0; // Private to each thread.
   double my_sum = 0;

   #pragma omp for nowait
   for (int i = 0; i < n; ++i)
   {
      double myValue = doSomeThing(i);
      my_sum += myValue;
      if (myValue > 10)
         my_counter++;
   }

   #pragma omp critical
   {
      global_counter += my_counter;
      global_sum += my_sum;
   }

   } // end parallel

   TimerType t1 = getTimeStamp();
   printf("manual: global_sum = %e %d %f\n", global_sum, global_counter, 1000*getElapsedTime(t0,t1));

   return 0;
}

int test_reduction(const int n)
{
   TimerType t0 = getTimeStamp();
   global_counter = 0;
   global_sum = 0;

   #pragma omp parallel for default(shared) reduction(+:global_sum,global_counter)
   for (int i = 0; i < n; ++i)
   {
      double myValue = doSomeThing(i);
      global_sum += myValue;
      if (myValue > 10)
         global_counter++;
   }

   TimerType t1 = getTimeStamp();
   printf("reduction: global_sum = %e %d %f\n", global_sum, global_counter, 1000*getElapsedTime(t0,t1));

   return 0;
}

#ifdef __has_openmp_atomics
int test_atomic(const int n)
{
   TimerType t0 = getTimeStamp();
   global_counter = 0;
   global_sum = 0;

   #pragma omp parallel for default(shared)
   for (int i = 0; i < n; ++i)
   {
      double myValue = doSomeThing(i);
      #pragma omp atomic
      global_sum += myValue;
      if (myValue > 10)
      {
         #pragma omp atomic
         global_counter++;
      }
   }

   TimerType t1 = getTimeStamp();
   printf("atomic: global_sum = %e %d %f\n", global_sum, global_counter, 1000*getElapsedTime(t0,t1));

   return 0;
}
#endif

void help(const char *prg, FILE *fp = stderr)
{
   fprintf(fp,"%s:\n", prg);
   fprintf(fp,"\t--help|-h\n");
   fprintf(fp,"\t--length|-n <int> : array length\n");
   fprintf(fp,"\t--niters|-i <int> : number of times to run tests.\n");
}

int main (int argc, char* argv[])
{
#ifdef _OPENMP
   fprintf(stderr,"OpenMP specification %d.%d (%d)\n", openmp_version_major(), openmp_version_minor(), openmp_version());

   int num_threads = 1;

   #pragma omp parallel default(shared)
   {
      #pragma omp master
      num_threads = omp_get_num_threads();
   }

   printf("num_threads = %d\n", num_threads);
#endif

   int n = 100000;
   int niters = 10;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help(argv[0], stdout); return 0;
      }
      else if (strcmp(argv[i],"--length") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--length|-n");
         i++;
         if (isdigit(*argv[i]))
            n = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--niters") == 0 || strcmp(argv[i],"-i") == 0)
      {
         check_index(i+1,"--niters|-i");
         i++;
         if (isdigit(*argv[i]))
            niters = atoi( argv[i] );
      }
      else
      {
         fprintf(stderr,"Unknown CLI argument: %s\n", argv[i]);
         help(argv[0], stderr); return 1;
      }
   }

   // Try to demonstrate unpredictable outcome of a race condition.
   {
      int shared_int = 0;
      double shared_double = 0;
      #pragma omp parallel shared( shared_int, shared_double )
      {
         if (omp_get_thread_num() < 100)
            shared_int++;

         for (int i = 0; i < n; ++i)
            shared_double += sqrt(double(i)/n);

         shared_int++;
      }

      printf("shared_int= %d %e\n", shared_int, shared_double);
   }

   printf("length: %d\n", n);
   printf("niters: %d\n", niters);

   for (int iter = 0; iter < niters; iter++)
   {
      test_critical(n);

#ifdef __has_openmp_atomics
      test_atomic(n);
#endif

      test_manual_reduction(n);
      test_reduction(n);
   }

   return 0;
}
