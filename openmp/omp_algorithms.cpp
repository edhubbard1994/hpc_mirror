#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>
#include <cmath>

#include "omp_helper.h"
#include "timer.h"

void help(const char *prg)
{
   fprintf(stderr,"%s --help|-h --length|-n\n", prg);

   return;
}

// Partition the range specified as [start,stop) into N (nearly) equal, continugous chunks.
// Returns the i^th sub-range [left,right) where i is [0,N).
void partition_range (const int start, const int stop, const int N, const int i, int &left, int &right)
{
   int chunk_size = (stop - start) / N;
   int remainder  = (stop - start) % N;
   left = i * chunk_size;
   right = left + chunk_size;
   if (i < remainder) { left++; right++; }
}

int main(int argc, char* argv[])
{
   int len = 1024; // Length of the array to process.

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help(argv[0]);
         return 0;
      }
      else if (strcmp(argv[i],"--length") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--length|-n");
         i++;
         if (isdigit(*argv[i]))
            len = atoi( argv[i] );
      }
      else
      {
         fprintf(stderr,"Unknown CLI option %s\n", argv[i]);
         help(argv[0]);
         return 1;
      }
   }

   int max_threads = 1;
#ifdef _OPENMP
   max_threads = omp_get_max_threads();
#endif

   printf("Common array algorithms using OpenMP parallel threads: length= %d, num_threads= %d\n", len, max_threads);

   typedef float value_type;

   value_type *x = new value_type [len];
   value_type *y = new value_type [len];

   srand( len );
   for (int i = 0; i < len; ++i)
   {
      x[i] = value_type( rand() ) / RAND_MAX;
      y[i] = value_type( rand() ) / RAND_MAX;
   }

   // We'll use a sorted array later.
   std::sort( y, y + len );

   if (len <= 10)
      for (int i = 0; i < len; ++i)
         printf("%2d: %f %f\n", i, x[i], y[i]);

   // Max value over the array.
   {
      value_type xmax = 0;
      #pragma omp parallel for reduction( max : xmax )
      for (int i = 0; i < len; ++i)
         xmax = std::max( xmax, x[i] );

      printf("maxval( x ) = %f\n", xmax);
   }

   // Average (mean) value over the array.
   {
      value_type xsum = 0;
      #pragma omp parallel for reduction( + : xsum )
      for (int i = 0; i < len; ++i)
         xsum += x[i];

      printf("average( x ) = %f\n", xsum / len);
   }

   // Location (index) of maximum value.
   // ... there isn't an implicit algorithm in OpenMP for this
   //     type of algorithm. But we have a few choices.
   {
      // 1) 2-steps ... loop over to get the max via reduction.
      //                then loop again and atomically write the matching location.
      //                ... this is only necessary if multiple locations may have the same value.
      value_type xmax = 0;
      int imax = len;
      #pragma omp parallel shared( xmax, imax )
      {
         #pragma omp for reduction( max : xmax )
         for (int i = 0; i < len; ++i)
            xmax = std::max( xmax, x[i] );

         #pragma omp for
         for (int i = 0; i < len; ++i)
            if ( x[i] == xmax )
            {
               #pragma omp atomic write
               imax = i;
            }
      }

      printf("maxloc( x ) = %d %f\n", imax, xmax);

      // 2) Find values per-thread and save into a shared data structure.
      //  ... then inspect manually afterwards.
      value_type *xmax_thr = new value_type[max_threads];
      int *imax_thr = new int[max_threads];
      for (int j = 0; j < max_threads; ++j)
         xmax_thr[j] = 0;

      #pragma omp parallel shared( xmax_thr, imax_thr )
      {
         value_type my_xmax = 0;
         int my_imax = len;
         const int thread_id = omp_get_thread_num();
         const int chunk = len / max_threads;

         int istart = thread_id * chunk;
         int iend   = istart + chunk;
         if (thread_id < chunk) { istart++; iend++; }
 
         for (int i = istart; i < iend; ++i)
            if (x[i] > my_xmax)
            {
               my_xmax = x[i];
               my_imax = i;
            }

         xmax_thr[ thread_id ] = my_xmax;
         imax_thr[ thread_id ] = my_imax;
      }

      xmax = 0;
      for (int j = 0; j < max_threads; ++j)
         if (xmax_thr[j] > xmax)
         {
            xmax = xmax_thr[j];
            imax = imax_thr[j];
         }

      delete [] xmax_thr;
      delete [] imax_thr;

      printf("maxloc( x ) = %d %f\n", imax, xmax);
   }

   // Location (index) of some logic condition in a sorted array.
   // ... there isn't an implicit algorithm in OpenMP for this
   //     type of algorithm. We can do this in parallel but it may
   //     not be worth the effort.
   {
      // Find index where y[] > 0.5

      int index = len;
      for (int i = 0; i < len; ++i)
         if (y[i] > value_type(0.5))
         {
            index = i;
            break;
         }

      if (index == len)
         printf("index( y > 0.5 ) never occured\n");
      else
         printf("index( y > 0.5 ) = %d %f\n", index, y[index]);

      int *index_thr = new int[max_threads];
      for (int j = 0; j < max_threads; ++j)
         index_thr[j] = len;

      #pragma omp parallel shared( index_thr )
      {
         const int thread_id = omp_get_thread_num();
         int start, stop;
         partition_range( 0, len, max_threads, thread_id, start, stop );

         for (int i = start; i < stop; ++i)
            if (y[i] > value_type(0.5))
            {
               index_thr[thread_id] = i;
               break;
            }
      }

      for (int j = 0; j < max_threads; ++j)
         if (index_thr[j] != len)
         {
            index = index_thr[j];
            break;
         }

      delete [] index_thr;

      if (index == len)
         printf("index( y > 0.5 ) never occured\n");
      else
         printf("index( y > 0.5 ) = %d %f\n", index, y[index]);

      // The parallel algorithm has linear cost. But we can do this
      // in O(log2(N)) time with a binary search.

      index = len;
      if ( !(y[0] > 0.5) and y[len-1] > 0.5 ) // sanity check.
      {
         int left = 0;
         int right = len-1;

         while (right - left > 0)
         {
            if (right - left <= 3)
            {
               // We know !(y[left] > 0.5) and y[right] > 0.5.
               if ( y[left+1] > 0.5 )
                  index = left+1;
               else
                  index = right;

               break;
            }
            int middle = left + (right - left) / 2;
            if (y[middle] > 0.5)
               right = middle;
            else
               left = middle;
         }
      }

      if (index == len)
         printf("index( y > 0.5 ) never occured\n");
      else
         printf("index( y > 0.5 ) = %d %f\n", index, y[index]);

   }

   // Histogram of values in x[] across 20 uniform bins.
   // ... This is where atomics can really shine in terms of simplicity but what about performance?
   {
      // Find counts of x[] in [0,1)/N

      const int nbins = 10;
      const value_type width = (1.0 - 0.0) / value_type(nbins);
      int *counts = new int[nbins];

      auto t0 = getTimeStamp();
      for (int i = 0; i < nbins; ++i)
         counts[i] = 0;

      #pragma omp parallel for
      for (int i = 0; i < len; ++i)
      {
         int bin = std::floor( x[i] / width );
         #pragma omp atomic update
         ++counts[bin];
      }

      auto t1 = getTimeStamp();

      printf("Parallel histogram %f (atomics)\n", getElapsedTime(t0, t1));
      for (int i = 0; i < nbins; ++i)
         printf("counts[%d]= %d %f\n", i, counts[i], 100*value_type(counts[i])/len);

      // Here's an older approach that can be better for the collision rates is high.

      auto t2 = getTimeStamp();

      for (int i = 0; i < nbins; ++i)
         counts[i] = 0;

      #pragma omp parallel default(shared)
      {
         int *my_counts = new int[nbins];

         for (int i = 0; i < nbins; ++i)
            my_counts[i] = 0;

         #pragma omp for
         for (int i = 0; i < len; ++i)
         {
            int bin = std::floor( x[i] / width );
            ++my_counts[bin];
         }

         #pragma omp critical
         {
            for (int i = 0; i < nbins; ++i)
               counts[i] += my_counts[i];
         }

         delete [] my_counts;
      }

      auto t3 = getTimeStamp();

      printf("Parallel histogram %f (private)\n", getElapsedTime(t2, t3));
      for (int i = 0; i < nbins; ++i)
         printf("counts[%d]= %d %f\n", i, counts[i], 100*value_type(counts[i])/len);

      delete [] counts;

   }

   delete [] x;
   delete [] y;

   return 0;
}
