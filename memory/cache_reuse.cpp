#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm> // std::min
#include <string>
#include <vector>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

#ifdef WITH_PAPI
# include "papi_helper.h"
  std::vector<int> papi_events{ PAPI_TOT_CYC, PAPI_L1_DCM, PAPI_L2_DCM };
#endif

#include "omp_helper.h"

#ifdef _OPENMP
# if (_OPENMP_MAJOR < 4)
#  error "OpenMP < 4: simd statement not supported."
# endif
#endif

#if defined(L1SIZE)
const int L1size = L1SIZE;
#else
const int L1size = 32*1024; // kb
#endif

#if defined(L1CACHELINE)
const int L1cacheline = L1CACHELINE;
#else
const int L1cacheline = 64;
#endif

template <typename ValueType, int Pad>
int cache_test (const int miters, const int length, const ValueType alpha, const ValueType beta)
{
   const auto nticks_per_sec = getTicksPerSecond();

   constexpr int Padding = (Pad == 0) ? 0 : ( L1cacheline / sizeof(ValueType) ); // an extra cacheline.

   const int N = length + Padding;
   ValueType *x = (ValueType *) aligned_alloc<ValueType>(N, 64 /* alignment bytes*/);
   ValueType *y = (ValueType *) aligned_alloc<ValueType>(N, 64 /* alignment bytes*/);

   // Flush the array and clear the cache.
   for (int i = 0; i < N; ++i)
      x[i] = y[i] = ValueType(i);

   ValueType sum(0);

   int niters = (miters == 0) ? 1 : miters;

   while (1)
   {
      auto kernel = [&] () {
            // Touch the vector elements.
            for (int iter = 0; iter < niters; ++iter)
            {
               #pragma omp simd aligned(x: 64)
               for (int i = 0; i < length; ++i)
                  x[i] = alpha * x[i] + beta;

               //sum += x[iter % length];
               dummy_function( length, x, y );

               #pragma omp simd aligned(y: 64)
               for (int i = 0; i < length; ++i)
                  y[i] = alpha * y[i] + beta * x[i];

               dummy_function( length, x, y );
            }
         };

      auto ticks_start = getClockTicks();

      kernel();

      auto ticks_stop = getClockTicks();

      auto clock_ticks = ticks_stop - ticks_start;
      double clock_time = double(clock_ticks) / nticks_per_sec; // ns

      if (clock_time > 0.1 or miters > 0)
      {
         printf("%10d, %10.3f, %10.3f, %10d, %10d", length, double(clock_ticks)/(length*niters), 1e9*clock_time/(niters*length), niters, N*sizeof(double) / 1024);

#ifdef WITH_PAPI
         PAPI_CMD( PAPI_start_counters( papi_events.data(), papi_events.size() ) );
         kernel(); // run again
         std::vector<long long> papi_counters( papi_events.size(), 0 );
         PAPI_CMD( PAPI_stop_counters( papi_counters.data(), papi_events.size() ) );
         for (int i = 0; i < papi_events.size(); ++i) {
            auto avg = double(papi_counters[i]) / niters; // avg per iteration.
            auto val = length / avg;
            printf(", %15.5f", val);
         }
#endif
         printf("\n");
         break;
      }
      else
         niters *= 2;
   }

   free(x);
   free(y);

   return (sum > 0) ? 0 : 1;
}

void show_usage( FILE *f )
{
   fprintf(f, "Usage:\n");
   fprintf(f, "\t--niters | -i <int value> : Number of iterations. (1000)\n");
   fprintf(f, "\t--offset | -o             : Force non-aligned allocation.\n");
   fprintf(f, "\t--min                     : Minimum array length. (128)\n");
   fprintf(f, "\t--max                     : Maximum array length. (16 * L1)\n");
}

int main (int argc, char * argv[])
{
   int niters = 0; // Number of samples for each test.
   int padding = 0;
   bool use_double = true;
   int min_length = 128;
   int max_length = 16 * (L1size / sizeof(double));

   double alpha = 1.1;
   double beta = 2.2;

   // Get user inputs.
   {
      #define check_index(_i) { if ((_i) >= argc){ fprintf(stderr,"Missing value for argument %s\n", for
      for (int i = 1; i < argc;)
      {
         std::string key = argv[i++];
         if (key == "--niters" || key == "-i")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            niters = atoi( argv[i++] );
         }
         else if (key == "--padding" || key == "-p")
         {
            padding = 1;
         }
         else if (key == "--float" || key == "--single" || key == "-s")
         {
            use_double = false;
         }
         else if (key == "--min" || key == "-m")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            min_length = atoi( argv[i++] );
         }
         else if (key == "--max" || key == "-m")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            max_length = atoi( argv[i++] );
         }
         else if (key == "--alpha" || key == "-a")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            alpha = atof( argv[i++] );
         }
         else if (key == "--beta" || key == "-b")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            beta = atof( argv[i++] );
         }
         else if (key == "--help" || key == "-h")
         {
            show_usage(stdout); return 0;
         }
      }
   }

#ifdef WITH_PAPI
   papi_start();

   if (PAPI_num_counters() < papi_events.size()) {
      fprintf(stderr,"PAPI: not enough hardware counters available %d %d!\n", PAPI_num_counters(), papi_events.size());
      return 1;
   }
#endif

   // Warm up the CPU
   {
      const auto t_start = getTimeStamp();

      int n = max_length;

      while (1)
      {
         std::vector<double> y(n);

         for (int i = 0; i < n; ++i)
            y[i] = double(rand()) / RAND_MAX;

         double minval = 1.0, maxval = 0.0;
         for (int i = 0; i < n; ++i)
         {
            if (y[i] > maxval) maxval = y[i];
            if (y[i] < minval) minval = y[i];
         }

         dummy_function( 1, &maxval );
         dummy_function( 1, &minval );

         auto t_stop = getTimeStamp();
         if (getElapsedTime( t_start, t_stop ) > 1.0) break;
         n *= 2;
      }

      dummy_function( 1, &n );
   }

   // Check the timer accuracy.
   double tDelta = 1.0e50; // really big value.
   {
      const int size = 1000000;
      std::vector<double> y(size);

      for (int i = 0; i < size; ++i)
         y[i] = 0.0;

      TimerType t0 = getTimeStamp();
      for (int i = 0; i < size; ++i)
      {
         TimerType t1 = getTimeStamp();
         y[i] = getElapsedTime(t0, t1);
         t0 = t1;
      }

      for (int i = 0; i < size-1; ++i)
         if (std::abs(y[i+1] - y[i]) > 0.0)
            tDelta = std::min(tDelta, std::abs(y[i+1]-y[i]));

      fprintf(stderr, "Smallest detectable time = %e (ms)\n", tDelta*1000.0);

      fprintf(stderr, "getTicksPerSecond = %e\n", getTicksPerSecond());
   }

   fprintf(stderr, "L1 Data cache size %d\n", L1size);

   fprintf(stderr, "%10s, %10s, %10s, %10s, %10s", "length", "ticks/elem", "time", "loops", "size");
#ifdef WITH_PAPI
   for (int i = 0; i < papi_events.size(); ++i)
   {
      char str[PAPI_MAX_STR_LEN];
      PAPI_CMD( PAPI_event_code_to_name( papi_events[i], str ) );
      printf(", %15s", str);
   }
#endif
   printf("\n");

   int success = 0;

   for (int length = min_length; length <= max_length; length *= 2)
      if (use_double) {
         if (padding) success += cache_test<double, 1>( niters, length, alpha, beta );
         else         success += cache_test<double, 0>( niters, length, alpha, beta );
      }
      else {
         if (padding) success += cache_test< float, 1>( niters, length, alpha, beta );
         else         success += cache_test< float, 0>( niters, length, alpha, beta );
      }

#ifdef WITH_PAPI
   papi_stop();
#endif

   return success;
}
