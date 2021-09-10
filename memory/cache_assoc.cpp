#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <algorithm> // std::min
#include <string>
#include <vector>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

#ifdef WITH_PAPI
# include "papi_helper.h"
  //std::vector<int> papi_events{ PAPI_L1_DCM, PAPI_L2_DCM };
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
int associativity_test (const int rows, const int cols, const ValueType beta, bool report = true)
{
   static_assert( Pad == 0 or Pad == 1, "" );
   const auto nticks_per_sec = getTicksPerSecond();

   const int len = (rows + Pad) * cols;
   ValueType *x = (ValueType *) aligned_alloc<ValueType>(len, L1cacheline /* alignment bytes*/);

#define X(i,j) x[ (i) + (j) * (rows + Pad) ]

   // Flush the array and clear the cache.
   for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
         X(i,j) = ValueType(i + j);

   int iters = 128;

   while (1)
   {
      auto kernel = [&]() {
            for (int k = 0; k < iters; ++k)
            {
               for (int i = 0; i < rows; ++i)
                  for (int j = 1; j < cols; ++j)
                     X(i,0) += beta * X(i,j);

               dummy_function( len, x );
            }
         };

      auto ticks_start = getClockTicks();

      kernel();

      auto ticks_stop = getClockTicks();

      dummy_function( len, x );

      auto clock_ticks = ticks_stop - ticks_start;
      double clock_time = double(clock_ticks) / nticks_per_sec;

      if (clock_time > 0.1) {
         if (report) {
            printf("%10d, %10d, %10d, %10.3f", rows, cols, (rows * cols * sizeof(ValueType)) / 1024,
                               double(clock_ticks)/(rows * cols * iters));
#ifdef WITH_PAPI
            PAPI_CMD( PAPI_start_counters( papi_events.data(), papi_events.size() ) );
            kernel(); // run again
            std::vector<long long> papi_counters( papi_events.size(), 0 );
            PAPI_CMD( PAPI_stop_counters( papi_counters.data(), papi_events.size() ) );
            for (int i = 0; i < papi_events.size(); ++i) {
               auto avg = double(papi_counters[i]) / iters; // avg per iteration.
               auto val = avg / (rows * cols);
               //auto val = (rows * cols) / avg;
               printf(", %15.5f", val);
            }
#endif
            printf("\n");
         }
         break;
      }
      else {
         iters *= 2;
      }
   }

   free(x);

   return 0;
}

void show_usage( FILE *f )
{
   fprintf(f, "Usage:\n");
   fprintf(f, "\t--niters | -i <int value> : Number of iterations. (1000)\n");
   fprintf(f, "\t--pad    | -p             : Pad the rows with an extra cacheline.\n");
   fprintf(f, "\t--cols   | -c <int value> : Number of columns. (17)\n");
   fprintf(f, "\t--rows   | -r <int value> : Number of rows. (2048)\n");
}

int main (int argc, char * argv[])
{
   int niters = 1000; // Number of samples for each test.
   bool padding = 0;
   bool use_double = true;

   int rows = L1size / sizeof(double);
   int cols = 17;

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
         else if (key == "--pad" || key == "-p")
         {
            padding = 1;
         }
         else if (key == "--float" || key == "--single" || key == "-s")
         {
            use_double = false;
         }
         else if (key == "--cols" || key == "-c")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            cols = atoi( argv[i++] );
         }
         else if (key == "--rows" || key == "-r")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            rows = atoi( argv[i++] );
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

      int n = cols * rows;

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
         if (std::fabs(y[i+1] - y[i]) > 0.0)
            tDelta = std::fmin(tDelta, std::fabs(y[i+1]-y[i]));

      fprintf(stderr, "Smallest detectable time = %e (ms)\n", tDelta*1000.0);

      fprintf(stderr, "getTicksPerSecond = %e\n", getTicksPerSecond());
   }

   fprintf(stderr, "L1 Data cacheline length %d\n", L1cacheline);
   fprintf(stderr, "L1 Data cache size %d\n", L1size);
   fprintf(stderr, "Padding: %d\n", (padding) ? 1 : 0);
   fprintf(stderr, "Rows length: %5.2f (kB)\n", (rows * ((use_double) ? sizeof(double) : sizeof(float))) / 1024.0);

   fprintf(stderr, "%10s, %10s, %10s, %10s", "Rows", "Cols", "Size (kb)", "Ticks");
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

   {
      if (use_double) {
         if (padding) success += associativity_test<double, 1>( rows, cols, beta, false );
         else         success += associativity_test<double, 0>( rows, cols, beta, false );
      }
      else {
         if (padding) success += associativity_test< float, 1>( rows, cols, beta, false );
         else         success += associativity_test< float, 0>( rows, cols, beta, false );
      }
   }

   for (int m = 2; m <= cols; ++m)
      if (use_double) {
         if (padding) success += associativity_test<double, 1>( rows, m, beta );
         else         success += associativity_test<double, 0>( rows, m, beta );
      }
      else {
         if (padding) success += associativity_test< float, 1>( rows, m, beta );
         else         success += associativity_test< float, 0>( rows, m, beta );
      }

#ifdef WITH_PAPI
   papi_stop();
#endif

   return success;
}
