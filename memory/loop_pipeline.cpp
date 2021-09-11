#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm> // std::min
#include <string>
#include <vector>
#include <limits>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

#ifdef WITH_PAPI
# include "papi_helper.h"
  std::vector<int> papi_events{ PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_DP_OPS };
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

#define MAX(a,b) ( (a) < (b) ? (b) : (a) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

template <typename T>
T random ( const T lo = std::numeric_limits<T>::min(), const T hi = std::numeric_limits<T>::max() )
{
    auto r = double( rand() ) / RAND_MAX;
    T val = lo + r * (hi - lo);
    return MAX( lo, MIN( hi, val ));
}

template <typename T>
int loop_test (const int miters, const int length, const int offset, const T alpha, const T beta)
{
   const auto nticks_per_sec = getTicksPerSecond();

   std::vector<T> x(length);
   std::vector<T> y(length);
   std::vector<T> z(length);
   std::vector<int> idx(length);

   // Flush the array and clear the cache.
   for (int i = 0; i < length; ++i) {
      x[i] = random<T>(0, 1);
      y[i] = random<T>(0, 100);
      z[i] = random<T>(0, 10);
      idx[i] = random<T>(0, length - 1);
   }

   assert (length > 2*offset);

   T sum(0);

   int niters = miters;

   while (1)
   {
      auto kernel = [&] () {
            // Touch the vector elements.
            for (int iter = 0; iter < niters; ++iter)
            {
               for (int i = offset; i < length - offset; ++i) {
                  if (y[i-offset] > x[i]) z[i] = sqrt( y[i] * x[i] );
                  y[i+offset] *= 2;
               }

               dummy_function( length, x.data() );
               dummy_function( length, y.data() );
               dummy_function( length, z.data() );
            }
         };

      auto ticks_start = getClockTicks();

      kernel();

      auto ticks_stop = getClockTicks();

      auto clock_ticks = ticks_stop - ticks_start;
      double clock_time = double(clock_ticks) / nticks_per_sec; // ns

      if (clock_time > 0.1)
      {
         printf("%10d, %10.3f, %10.3f, %10d, %10d", offset, double(clock_ticks)/(niters*length), 1e9*clock_time/(niters*length), niters, length*3*sizeof(double) / 1024);

#ifdef WITH_PAPI
         PAPI_CMD( PAPI_start_counters( papi_events.data(), papi_events.size() ) );
         kernel(); // run again
         std::vector<long long> papi_counters( papi_events.size(), 0 );
         PAPI_CMD( PAPI_stop_counters( papi_counters.data(), papi_events.size() ) );
         for (int i = 0; i < papi_events.size(); ++i) {
            auto avg = double(papi_counters[i]) / niters; // avg per iteration.
            auto n_cachelines = (length + L1cacheline - 1) / L1cacheline;
            //auto val = avg / n_cachelines;
            auto val = avg / length;
            //auto val = length / avg;
            printf(", %15.5f", val);
         }
#endif
         printf("\n");
         break;
      }
      else
         niters *= 2;
   }

   return 0;
}

void show_usage( FILE *f )
{
   fprintf(f, "Usage:\n");
   fprintf(f, "\t--iters  | -i <int value> : Number of iterations. (1000)\n");
   fprintf(f, "\t--offset | -o             : Force non-aligned allocation.\n");
   fprintf(f, "\t--length | -n <int value> : Array length. (50%% of L1 data cache)\n");
}

int main (int argc, char * argv[])
{
   int niters = 1000; // Number of samples for each test.
   int offset = 0;
   bool use_double = true;

   int length = (L1size / sizeof(double)) / 2;

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
         else if (key == "--offset" || key == "-o")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            offset = atoi( argv[i++] );
         }
         else if (key == "--float" || key == "--single" || key == "-s")
         {
            use_double = false;
         }
         else if (key == "--length" || key == "-n")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            length = atoi( argv[i++] );
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

      int n = length;

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
            tDelta = std::min(tDelta, std::abs(y[i+1]-y[i]));

      fprintf(stderr, "Smallest detectable time = %e (ms)\n", tDelta*1000.0);

      fprintf(stderr, "getTicksPerSecond = %e\n", getTicksPerSecond());
   }

   fprintf(stderr, "L1 Data cache size %d\n", L1size);
   fprintf(stderr, "Length: %d\n", length);

   fprintf(stderr, "%10s, %10s, %10s, %10s, %10s", "offset", "ticks/elem", "time", "iters", "size");
#ifdef WITH_PAPI
   for (int i = 0; i < papi_events.size(); ++i)
   {
      char str[PAPI_MAX_STR_LEN];
      PAPI_CMD( PAPI_event_code_to_name( papi_events[i], str ) );
      printf(", %15s", str);
   }
#endif
   printf("\n");

   for (int o = 0; o < 20; ++o)
      loop_test< double > ( niters, length, o, alpha, beta );

#ifdef WITH_PAPI
   papi_stop();
#endif

   return 0;
}
