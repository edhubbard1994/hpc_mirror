#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <algorithm>
#include <string>
#include <vector>

#include "timer.h"
#include "omp_helper.h"

void help(const char *prg, FILE *fp = stderr)
{
   fprintf(fp,"%s: --help|-h --length|-n <#> --segments|-s <#> --atomic|-a --critical|-c\n", prg);

   return;
}

typedef enum { use_atomics, use_critical, use_locks } MethodType;

int main(int argc, char* argv[])
{
   int nSamples = 1000; // Length of the array ... # of samples.
   int nSegments = 10; // # of entries in the histogram.
   MethodType method = use_atomics;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if (strcmp(argv[i],"--samples") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--samples|-n");
         i++;
         if (isdigit(*argv[i]))
            nSamples = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--segments") == 0 || strcmp(argv[i],"-s") == 0)
      {
         check_index(i+1,"--segments|-s");
         i++;
         if (isdigit(*argv[i]))
            nSegments = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--atomic") == 0 || strcmp(argv[i],"-a") == 0)
      {
         method = use_atomics;
      }
      else if (strcmp(argv[i],"--critical") == 0 || strcmp(argv[i],"-c") == 0)
      {
         method = use_critical;
      }
      else if (strcmp(argv[i],"--locks") == 0 || strcmp(argv[i],"-l") == 0)
      {
         method = use_locks;
      }
      else
      {
         help(argv[0], stderr);
         return 0;
      }
   }

   int max_threads = 1;
#ifdef _OPENMP
   max_threads = omp_get_max_threads();
#endif

   std::vector< std::string > method_names;
   method_names.push_back( "Atomics" );
   method_names.push_back( "Critical" );
   method_names.push_back( "Locks" );

   printf("OpenMP Parallel atomic example: nSamples= %d, nSegments= %d, num_threads= %d %s\n", nSamples, nSegments, max_threads, method_names[method].c_str());

   float xmin = 0, xmax = 1;
   float *x = new float [nSamples];
   int *count = new int [nSegments];

#ifdef _OPENMP
   omp_lock_t *lock = new omp_lock_t [nSegments];
   for (int i = 0; i < nSegments; ++i)
      omp_init_lock( &lock[i] );
#endif

   srand(100);
   for (int i = 0; i < nSamples; ++i)
   {
      float r = float( rand() ) / RAND_MAX;
      x[i] = xmin + r * (xmax - xmin);
   }

   double runtime = 0;
   const int niters = 100;
   int iter = 0;

   for (; iter < niters; iter++)
   {

   for (int i = 0; i < nSegments; ++i)
      count[i] = 0;

   TimerType t0 = getTimeStamp();

   #pragma omp parallel for default(shared)
   for (int i = 0; i < nSamples; ++i)
   {
      float segment_width = (xmax - xmin) / nSegments;
      int segment_index = floorf( (x[i] - xmin) / segment_width );

      if (method == use_atomics)
      {
         #pragma omp atomic update
         count[ segment_index ] ++;
      }
      else if (method == use_critical)
      {
         #pragma omp critical
         count[ segment_index ] ++;
      }
      else if (method == use_locks)
      {
#ifdef _OPENMP
         omp_set_lock( &lock[segment_index] );
#endif
         count[ segment_index ] ++;
#ifdef _OPENMP
         omp_unset_lock( &lock[segment_index] );
#endif
      }
      //printf("%f %d\n", x[i], segment_index);
   }

   TimerType t1 = getTimeStamp();
   runtime += getElapsedTime(t0,t1);
   if (runtime > 5.0)
      break;

   }

   printf("histogram took %f (ms) %s %d\n", 1000*runtime/iter, method_names[method].c_str(), iter);

   for (int i = 0; i < std::min(10,nSegments); ++i)
      printf("count[%d]= %d\n", i, count[i]);

   delete [] x;
   delete [] count;
#ifdef _OPENMP
   for (int i = 0; i < nSegments; ++i)
      omp_destroy_lock( &lock[i] );
   delete [] lock;
#endif

   return 0;
}
