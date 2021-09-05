#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm> // std::min
#include <string>
#include <vector>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

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

template <typename ValueType, int Offset>
int stride_test (const int miters, const int stride, const int length, const ValueType alpha, const ValueType beta)
{
   const auto nticks_per_sec = getTicksPerSecond();

   const int N = length * stride + Offset;
   const int N0 = std::max(N, int(L1size / sizeof(ValueType)) + 1 );
   ValueType *array = (ValueType *) aligned_alloc<ValueType>(N0, 64 /* alignment bytes*/);

   // Flush the array and clear the cache.
   for (int i = 0; i < N0; ++i)
      array[i] = ValueType(i);

   ValueType sum(0);

   int niters = miters;

   while (1)
   {
      auto ticks_start = getClockTicks();

      // Touch the vector elements.
      for (int iter = 0; iter < niters; ++iter)
      {
         #pragma omp simd aligned(array:64)
         for (int i = 0; i < length; ++i)
            array[i*stride + Offset] = alpha * array[i*stride + Offset] + beta;

         sum += array[iter % length];
         dummy_function( N, array );
      }

      auto ticks_stop = getClockTicks();
      auto clock_ticks = ticks_stop - ticks_start;
      double clock_time = double(clock_ticks) / nticks_per_sec; // ns

      if (clock_time > 0.1)
      {
         printf("%5d, %10.3f, %10.3f, %10d, %10d\n", stride, double(clock_ticks)/(niters*length), 1e9*clock_time/(niters*length), niters, N*sizeof(double) / 1024);
         break;
      }
      else
         niters *= 2;
   }

   free(array);

   return (sum > 0) ? 0 : 1;
}

void show_usage( FILE *f )
{
   fprintf(f, "Usage:\n");
   fprintf(f, "\t--iters  | -i <int value> : Number of iterations. (1000)\n");
   fprintf(f, "\t--offset | -o             : Force non-aligned allocation.\n");
   fprintf(f, "\t--length | -n <int value> : Array length. (50\% of L1 data cache)\n");
}

int main (int argc, char * argv[])
{
   int niters = 1000; // Number of samples for each test.
   int offset = 0;
   int max_stride = 33;
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
         if (key == "--iters" || key == "-i")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            niters = atoi( argv[i++] );
         }
         else if (key == "--offset" || key == "-o")
         {
            offset = 1;
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
         else if (key == "--max" || key == "-m")
         {
            if (i >= argc) { fprintf(stderr,"Missing value for %s\n", key); show_usage(stderr); return 1; }
            max_stride = atoi( argv[i++] );
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
         if (std::abs(y[i+1] - y[i]) > 0.0)
            tDelta = std::min(tDelta, std::abs(y[i+1]-y[i]));

      fprintf(stderr, "Smallest detectable time = %e (ms)\n", tDelta*1000.0);

      fprintf(stderr, "getTicksPerSecond = %e\n", getTicksPerSecond());
   }

   fprintf(stderr, "L1 Data cache size %d\n", L1size);
   fprintf(stderr, "Length: %d\n", length);

   fprintf(stderr, "stride, ticks, time, #iters, size\n");

   int success = 0;

   for (int stride = 1; stride <= max_stride; ++stride)
      if (use_double) {
         if (offset) success += stride_test<double, 1>( niters, stride, length, alpha, beta );
         else        success += stride_test<double, 0>( niters, stride, length, alpha, beta );
      }
      else {
         if (offset) success += stride_test< float, 1>( niters, stride, length, alpha, beta );
         else        success += stride_test< float, 0>( niters, stride, length, alpha, beta );
      }

   return success;
}
