#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cassert>

#include <algorithm> // std::min
#include <string>
#include <vector>

#include "timer.h"
#include "aligned_allocator.h"
#include "dummy.h"

#ifdef WITH_PAPI
# include "papi_helper.h"
  //std::vector<int> papi_events{ PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM };
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

template <typename T, int Size>
struct Array
{
     enum : int { N = Size };
     T m_dat[N];
           T& operator[](const int i)       { return m_dat[i]; }
     const T& operator[](const int i) const { return m_dat[i]; }
};

template <typename T, int Stride>
int stride_test (const int miters, const int idx, const int length, const T alpha, const T beta)
{
    const auto nticks_per_sec = getTicksPerSecond();

    using AT = Array<T, Stride>;

    const int N = length;
    AT *array = (AT *) aligned_alloc<AT>(N, 64 /* alignment bytes*/);

    // Flush the array and clear the cache.
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < Stride; ++j)
            array[i][j] = T(i+j);

    int niters = miters;

    while (1)
    {
       auto kernel = [&] () {
               // Touch the vector elements.
               for (int iter = 0; iter < niters; ++iter)
               {
                   #pragma omp simd aligned(array: 64)
                   for (int i = 0; i < length; ++i)
                       array[i][idx] = alpha * array[i][idx] + beta;

                   dummy_function( N, array );
               }
           };

       auto ticks_start = getClockTicks();

       kernel();

       auto ticks_stop = getClockTicks();

       auto clock_ticks = ticks_stop - ticks_start;
       double clock_time = double(clock_ticks) / nticks_per_sec; // ns

       if (clock_time > 0.1)
       {
          printf("%10d, %10.3f, %10.3f, %10d, %10d", Stride, double(clock_ticks)/(niters*length), 1e9*clock_time/(niters*length), niters, N*sizeof(double) / 1024);

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

    free(array);

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

    fprintf(stderr, "%10s, %10s, %10s, %10s, %10s", "stride", "ticks/elem", "time", "loops", "size");
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

#define func(s,type) if ( (s) <= max_stride ) { success += stride_test<type,(s)>( niters, (s)-1, length, alpha, beta ); }

    func ( 1, double )
    func ( 2, double )
    func ( 3, double )
    func ( 4, double )
    func ( 5, double )
    func ( 6, double )
    func ( 7, double )
    func ( 8, double )
    func ( 9, double )
    func (10, double )
    func (11, double )
    func (12, double )
    func (13, double )
    func (14, double )
    func (15, double )
    func (16, double )
    func (17, double )
    func (18, double )
    func (19, double )
    func (20, double )
    func (21, double )
    func (22, double )
    func (23, double )
    func (24, double )
    func (25, double )
    func (26, double )
    func (27, double )
    func (28, double )
    func (29, double )
    func (30, double )
    func (31, double )
    func (32, double )
    func (33, double )
    func (34, double )
    func (35, double )
    func (36, double )
    func (37, double )
    func (38, double )
    func (39, double )
    assert( max_stride < 40 );

#ifdef WITH_PAPI
    papi_stop();
#endif

    return success;
}
