#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string>
#include <cstring>
#include <vector>

#include <cmath>
#include <algorithm>

#include <aligned_allocator.h>
#include <timer.h>

#include <matmul.h>

#ifndef L3SIZE
#define L3SIZE ( 32 * (1<<20) ) /* 32 MB */
#endif

void flush_cache (void)
{
   size_t llc_size = L3SIZE;
   size_t nelems = 2 * llc_size / sizeof(double); // 60 MB as an array of doubles.

   double *x = new double [nelems];

   for (int i = 0; i < nelems; ++i)
      x[i] = rand() % nelems;

   std::sort( x, x + nelems );

   dummy( x );

   delete [] x;
}

int run_matmul (int n, int niters, const double tDelta, matmul_ptr matmul, const bool doCheck = false)
{
   int npad = 8; // Array padding.

   // Allocate arrays with a little padding.
   //ValueType *a = new ValueType[n*n+npad];
   //ValueType *b = new ValueType[n*n+npad];
   //ValueType *c = new ValueType[n*n+npad];
   //ValueType *cref = new ValueType[n*n+npad];
   ValueType *a = (ValueType*) aligned_alloc<ValueType>(n*n+npad, 64);
   ValueType *b = (ValueType*) aligned_alloc<ValueType>(n*n+npad, 64);
   ValueType *c = (ValueType*) aligned_alloc<ValueType>(n*n+npad, 64);
   ValueType *cref = (ValueType*) aligned_alloc<ValueType>(n*n+npad, 64);

   const ValueType cval = 1e-5;

   // Seed A[] with random #'s, zero C[], and set B[] = I.
   srand(n);
   ValueType invRandMax = ValueType(1) / ValueType(RAND_MAX);
   ValueType invN = 1.0 / n;
   for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
      {
         //a[i+j*n] = ValueType(rand()) * invRandMax;
         a[i+j*n] = i * invN + j * invN;
         //b[i+j*n] = 0;
         //if (i==j) b[i+j*n] = 1;
         b[i+j*n] = 0.1;
         cref[i+j*n] = cval;
      }

   // Interesting scaling factors (not 0 or 1).
   ValueType alpha = 0.1, beta = 0.21;

   // Run the BLAS version to get the correct answer.
   matmul_blas(n, n, n, alpha, a, n, b, n, beta, cref, n);

   // Run a few iterations to warm up the system.
   for (int iter = 0; iter < std::min(1,niters); iter++)
   {
      std::fill ( c, c + n*n, cval );
      matmul(n, n, n, alpha, a, n, b, n, beta, c, n);
   }

   double err2 = 0.0, ref2 = 1.0;
   if (doCheck)
   {
      for (int i = 0; i < n; ++i)
         for (int j = 0; j < n; ++j)
            c[i + j*n] = cval;

      matmul(n, n, n, alpha, a, n, b, n, beta, c, n);

      ref2 = 0.0;
      for (int j = 0; j < n; ++j)
         for (int i = 0; i < n; ++i)
         {
            double diff = c[i + j*n] - cref[i + j*n];
            ref2 += cref[i + j*n] * cref[i + j*n];
            err2 += diff*diff;
         }
   }

   // Run the test for 'a long time.'
   double tCalc = 0;
   const double min_time = 0.1;

   while( tCalc < min_time )
   {
      int c_index = rand() % n;

      TimerType t1 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
      {
         ValueType val = iter/3.0;
         for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
               c[i + j*n] = val;

//       flush_cache();

         // Trick the compiler to keep the results.
         dummy( c );
      }

      TimerType t2 = getTimeStamp();

      for (int iter = 0; iter < niters; iter++)
      {
         for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
               c[i + j*n] = cval;

//       flush_cache();

         matmul(n, n, n, alpha, a, n, b, n, beta, c, n);

         // Trick the compiler to keep the results.
         dummy( c );
      }

      TimerType t3 = getTimeStamp();

      double t_total = getElapsedTime(t1,t3);
      double t_waste = getElapsedTime(t1,t2);
      tCalc = t_total - t_waste;
//    printf("%f %f %f %d\n", tCalc, t_total, t_waste, niters);
      if (tCalc < min_time)
         niters *= 2;
   }
   tCalc /= niters;

#ifdef WITH_PAPI

   std::vector<int> papi_events{ PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM, PAPI_DP_OPS };

   //std::vector< std::string > papi_native_event_names{ "MEM_UOPS_RETIRED.ALL_STORES",
   //                                                    "MEM_UOPS_RETIRED.ALL_LOADS",
   //                                                    "MEM_LOAD_UOPS_RETIRED.L3_MISS",
   //                                                    "MEM_LOAD_UOPS_RETIRED.L3_HIT",
   //                                                    "MEM_LOAD_UOPS_RETIRED.L2_HIT",
   //                                                    "MEM_LOAD_UOPS_RETIRED.L1_HIT" };

   //std::vector<int> papi_events;
   std::vector< std::string > papi_event_names;

   char *papi_env = getenv("PAPI_EVENTS");
   if (papi_env)
   {
      papi_events.clear();
      printf("papi_env %s\n", papi_env);
      char *str = papi_env;
      size_t len = strlen(papi_env);
      while (str < papi_env + len)
      {
         char *next = strchr( str, ',' );
         if (next) {
            papi_event_names.push_back( std::string( str, next-str ) );
            str = next + 1;
         }
         else {
            papi_event_names.push_back( std::string( str ) );
            break;
         }
      }
   }

   for (int i = 0; i < papi_event_names.size(); ++i)
   {
      int this_event = 0 | PAPI_NATIVE_MASK;
      int retval = PAPI_event_name_to_code( const_cast<char*>(papi_event_names[i].c_str()), &this_event );
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_event_code_to_name %d\n", retval);
         return 1;
      }
      //printf("PAPI event %s %x\n", papi_event_names[i].c_str(), this_event);
      papi_events.push_back( this_event );
   }

   const int num_papi_events = papi_events.size();
   std::vector<long long> papi_event_counters( num_papi_events );
   //const int num_hw_counters = 1; //PAPI_num_counters();
   const int num_hw_counters = 2; //PAPI_num_counters();
//   const int num_hw_counters = PAPI_num_counters();
   //printf("events= %d, hw= %d %d\n", num_papi_events, num_hw_counters, PAPI_num_counters());

   {
      for (int i = 0; i < n; ++i)
         for (int j = 0; j < n; ++j)
            c[i + j*n] = cval;

      matmul(n, n, n, alpha, a, n, b, n, beta, c, n);

      srand((int)c[0]);
   }

   for (int iter = 0; iter < num_papi_events; iter += num_hw_counters)
   {
      for (int i = 0; i < n; ++i)
         for (int j = 0; j < n; ++j)
            c[i + j*n] = cval;

      int n_events = std::min( num_papi_events - iter, num_hw_counters );

      int retval = PAPI_start_counters( &papi_events[iter], n_events );
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_start_counters %d\n", retval);
         return 1;
      }

      matmul(n, n, n, alpha, a, n, b, n, beta, c, n);

      retval = PAPI_stop_counters( &papi_event_counters[iter], n_events );
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_stop_counters %d\n", retval);
         return 1;
      }

      srand((int)c[(iter%n)]);

      for (int i = iter; i < (iter+n_events); ++i)
      {
         char papi_event_str[PAPI_MAX_STR_LEN];
         retval = PAPI_event_code_to_name( papi_events[i], papi_event_str );
         if (retval != PAPI_OK) {
            fprintf(stderr,"PAPI: Error calling PAPI_event_code_to_name %d\n", retval);
            return 1;
         }
         //printf("PAPI event %d %d %s %.2f\n", i, papi_events[i], papi_event_str, 1e-3*papi_event_counters[i]);
      }
   }
#endif

   double Gflops = 1e-9 * ((((2.0*n)*n)*n) + ((3.0*n)*n)) / tCalc;
   //printf("%5d, %10.4f, %10.4f, %.2f%%, %10d, %e", n, tCalc*1000, Gflops, 100*tDelta/(niters*tCalc), niters, sqrt(err2 / ref2));
   printf("%10d, %10.2f, %10.4f, %10.4f, %10d, %10.4e", n, double(3*n*n)*sizeof(ValueType) / 1024., tCalc*1000, Gflops, niters, sqrt(err2 / ref2));
#ifdef WITH_PAPI
   for (int i = 0; i < num_papi_events; ++i)
      printf(", %10.2f", papi_event_counters[i]*1e-3);
#endif
   printf("\n");

   if (n<=5)
   {
      for (int i = 0; i < n; ++i)
      {
         for (int j = 0; j < n; ++j)
            printf("%f ", c[i + j*n]);
         printf("\n");
      }
   }

   //delete [] a;
   //delete [] b;
   //delete [] c;
   free( a );
   free( b );
   free( c );
   free( cref );

   return 0;
}

void show_usage( const char* prog )
{
   printf("Usage for %s\n", prog);
   printf("\t--minsize            <int value> : Minimum matrix size to start. (0)\n");
   printf("\t--maxsize            <int value> : Maximum matrix size to start. (0)\n");
   printf("\t--stepsize           <flt value> : Growth rate of matrix size.   (2)\n");
   printf("\t--size     | -s      <int value> : Maximum matrix size to start. (1000)\n");
   printf("\t--check    | -c                  : Verify the solution against DGEMM. (off)\n");
   printf("\t--method   | -m      <int value> : Function choices are ... (0)\n");
   printf("\t\t0: blas (vendor)\n");
   printf("\t\t1: naive\n");
   printf("\t\t2: vectorized\n");
   printf("\t\t3: cache blocked\n");
   printf("\t\t4: unrolled\n");
}

int main (int argc, char * argv[])
{
   int min_size = 0;
   int max_size = 0;
   int mat_size = 1000;
   int niters = 2; // Number of samples for each test.
   int method = 0; // blas
   double stepSize = 2;
   bool doCheck = false;

   // Get user inputs.
   {
      #define check_index(_i) { if ((_i) >= argc){ fprintf(stderr,"Missing value for argument %s\n", for
      for (int i = 1; i < argc; i++)
      {
         std::string arg = argv[i];
         if (arg == "--minsize")
         {
            if ((i+1) >= argc) { fprintf(stderr,"Missing value for --minsize\n"); show_usage(argv[0]); return 1; }
            min_size = atoi( argv[i+1] );
            i++;
         }
         else if (arg == "--maxsize")
         {
            if ((i+1) >= argc) { fprintf(stderr,"Missing value for --maxsize\n"); show_usage(argv[0]); return 1; }
            max_size = atoi( argv[i+1] );
            i++;
         }
         else if (arg == "--stepsize")
         {
            if ((i+1) >= argc) { fprintf(stderr,"Missing value for --stepsize\n"); show_usage(argv[0]); return 1; }
            stepSize = atof( argv[i+1] );
            i++;
         }
         else if (arg == "--size" || arg == "-s")
         {
            if ((i+1) >= argc) { fprintf(stderr,"Missing value for --maxsize\n"); show_usage(argv[0]); return 1; }
            mat_size = atoi( argv[i+1] );
            i++;
         }
         else if (arg == "--method" || arg == "-m")
         {
            if ((i+1) >= argc) { fprintf(stderr,"Missing value for --method\n"); show_usage(argv[0]); return 1; }
            method = atoi( argv[i+1] );
            i++;
         }
         else if (arg == "--check" || arg == "-c")
         {
            doCheck = true;
         }
         else if (arg == "--help" || arg == "-h")
         {
            show_usage(argv[0]); return 0;
         }
      }
   }

   matmul_ptr methods[] = {matmul_blas, matmul_naive, matmul_vect, matmul_blocked, matmul_unroll};
   const char *method_names[] = {"matmul_blas", "matmul_naive", "matmul_vect", "matmul_blocked", "matmul_unroll"};

   if (method < 0 || method > sizeof(methods)/sizeof(methods[0]))
   {
      fprintf(stderr,"Invalid method selected %d\n", method);
      return 1;
   }
   else
      fprintf(stderr,"Using method[%d]=%s, min/max=%d/%d, step=%f\n", method, method_names[method], min_size, max_size, stepSize);

   printf("unroll = %d\n", UNROLL);
   printf("blockSize = %d\n", BLOCKSIZE);

#ifdef WITH_PAPI
   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
      fprintf(stderr,"PAPI: version mismatch!\n");
      return 1;
   }
   if (PAPI_num_counters() < 2) {
      fprintf(stderr,"PAPI: no hardware counters available!\n");
      return 1;
   }
#endif

   // Check the timer accuracy.
   double tDelta = 1e50;
   {
      //int nmax = std::min(10000,max_size);
      int nmax = 10000;

      ValueType *a = new ValueType[nmax];

      for (int i = 0; i < nmax; ++i)
         a[i] = 0.0;

      TimerType t0 = getTimeStamp();
      for (int i = 0; i < nmax; ++i)
      {
         TimerType t1 = getTimeStamp();
         a[i] = getElapsedTime(t0,t1);
         t0 = t1;
      }

      for (int i = 0; i < nmax-1; ++i)
         if (std::abs(a[i+1] - a[i]) > 0.0)
            tDelta = std::min(tDelta, std::abs(a[i+1]-a[i]));

      printf("Smallest detectable time = %e (ms)\n", tDelta*1000);

      printf("getTicksPerSecond = %e\n", getTicksPerSecond());

      delete [] a;
   }

#ifdef _OPENMP
   #pragma omp parallel
   {
      #pragma omp master
      printf("OpenMP # threads: %d\n", omp_get_num_threads());
   }
#endif

   printf("%10s, %10s, %10s, %10s, %10s, %10s", "N", "Size (kb)", "time (ms)", "GFLOP/s", "Runs", "Error");
#ifdef WITH_PAPI
   printf(", PAPI TOT_CYC, L1, L2, L3 misses, DP_OPS");
#endif
   printf("\n");

   if (max_size > min_size)
      for (int size = min_size; size <= max_size; size *= stepSize)
         run_matmul (size, niters, tDelta, methods[method], doCheck );
   else
      run_matmul (mat_size, niters, tDelta, methods[method], doCheck );

#ifdef WITH_PAPI
   PAPI_shutdown();
#endif

   return 0;
}
