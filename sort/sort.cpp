#include <iostream>     // std::cout
#include <functional>   // std::less
#include <algorithm>    // std::sort
#include <vector>	// std::vector
#include <string.h>	// strcmp

#include <ctype.h>	// isdigit

#include <sys/time.h>	// gettimeofday, struct timeval

#include <stdio.h>	// printf
#include <stdlib.h>	// srand, rand, RAND_MAX

#include <timer.h>
#include <aligned_allocator.h>
#include "partitioner.hpp"
#include "dummy.h"

#ifdef ENABLE_PARALLEL_CXX
#include <parallel/algorithm>
#endif
//#ifdef ENABLE_CXX17
//#include <algorithm>
//#include <execution>
//#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

template <typename T, class BinaryCompOp>
void selectionSort (const int n, T *arr, const BinaryCompOp &comp)
{
   for (int j = 0; j < (n-1); ++j) // Don't need to sort to j=(n-1) since it'll be in place already.
   {
      // Find the index of the minimum value in the active list.
      // Initialize with the first element in the active list (j).
      int idx = j;
      for (int i = j+1; i < n; ++i)
         if ( comp( arr[i], arr[idx] ) )
            idx = i;

      // Swap the minimum value at [idx] with the 1st element in the active list.
      if (j != idx)
         std::swap( arr[j], arr[idx] );
   }
}
template <typename T, class BinaryCompOp>
void selectionSort (T *first, T *last, const BinaryCompOp &comp)
{
   const size_t len = last - first;
   selectionSort (len, first, comp);
}
template <typename ValueType, typename PointerType, class BinaryCompOp>
void selectionSortPtr (const int n, ValueType *arr, PointerType *ptr, const BinaryCompOp &comp)
{
   // Notice the test for (n-1).
   // ... Don't need to sort to j=(n-1) since it'll be in place already.
   for (int j = 0; j < (n-1); ++j)
   {
      // Find the index of the minimum value in the active list.
      // Initialize with the first element in the active list (j).
      int idx = j;
      for (int i = j+1; i < n; ++i)
         if ( comp( arr[ptr[i]], arr[ptr[idx]] ) )
            idx = i;

      // Swap the minimum value at [idx] with the 1st element in the active list.
      if (j != idx)
         std::swap( ptr[j], ptr[idx] );
   }
}
template <typename ValueType, typename PointerType, class BinaryCompOp>
void selectionSortPtr (ValueType *first, ValueType *last, PointerType *ptr, const BinaryCompOp &comp)
{
   const size_t len = last - first;
   selectionSortPtr (len, first, ptr, comp);
}

template <typename T, class BinaryCompOp>
T* myPartition (T *first, T *last, const T &pivot, const BinaryCompOp &comp)
{
   while (first != last)
   {
      while (comp(*first, pivot))
      {
         ++first;
         if (first == last)
            return first;
      }
      do
      {
         --last;
         if (first == last)
            return first;
      }
      while (!comp(*last, pivot));

      std::swap (*first, *last);
      ++first;
   }

   return first;
}

#ifdef _OPENMP
template <typename T, class BinaryCompOp>
T* ParallelPartition ( T *data, const size_t len, const T &pivot, const BinaryCompOp &comp )
{
   // The steps:
   // a) each parallel partition scans the input list and counts the # of elements that 
   //    will be left of the pivot.
   // b) for each partition, save the left count and the partition range.
   // c) do a scan of the left counts to find the offsets in the output array both
   //    left and right of the pivot.

   const int maxThreads = omp_get_max_threads(); // This works outside of the parallel region.

   if ( maxThreads == 1 )
      return myPartition( data, data + len, pivot, comp );

   // Shared array to hold the counts of all thread's lengths below the pivot.
   std::vector<size_t> shared_low_counts( maxThreads, 0 );

   // a) partition the range
   auto shared_parts = partition_range( len, maxThreads );

   size_t shared_split_index = 0;

   #pragma omp parallel default(shared)
   {
    //const int nThreads = omp_get_num_threads();
      const int threadID = omp_get_thread_num();

      const size_t my_start = shared_parts[threadID];
      const size_t my_stop  = shared_parts[threadID+1];
      const size_t my_size  = my_stop - my_start;

      // b) Each thread copies their partition into a temporary local array.
      //    ... we can't do this in-place.
      std::vector<T> local_data( data + my_start, data + my_stop );

      T* my_data = local_data.data();

      // c) Partition my local data.
      T* my_split = myPartition( my_data, my_data + my_size, pivot, comp );

      // d) How many below the pivot?
      const size_t my_count = std::distance( my_data, my_split );

      // e) All threads write their local counts into the shared arrays.
      shared_low_counts[threadID] = my_count;

      #pragma omp barrier

      // f) Sum up the low count array to see where I need to write
      //    my left and right data.
      size_t global_split_index = 0, my_left_offset = 0, my_right_offset = 0;
      for (int tid = 0; tid < maxThreads; ++tid)
      {
         global_split_index += shared_low_counts[tid];

         if ( tid < threadID )
         {
            my_left_offset  += shared_low_counts[tid];

            const size_t thread_size = shared_parts[tid+1] - shared_parts[tid];
            my_right_offset += (thread_size - shared_low_counts[tid]);
         }
      }

      #pragma omp master
      shared_split_index = global_split_index;

      my_right_offset += global_split_index;

      // g) Copy my two pivot arrays into the original array.
      for (size_t i = 0; i < my_count; ++i)
         data[ i + my_left_offset ] = my_data[i];

      const size_t my_right_count = my_size - my_count;
      for (size_t i = 0; i < my_right_count; ++i)
         data[ i + my_right_offset ] = my_split[i];
   }

   return data + shared_split_index;
}
#endif

#ifdef ENABLE_SIMD
#include <immintrin.h>
template <typename Comp>
int* myPartitionSimd ( int *first, int *last, const int &pivot, const Comp&)
{
   std::less<int> comp;

   int *orig_first = first;
   int *orig_last = last;

   const int v_len = sizeof(__m512i) / sizeof(int);

   const __m512i v_pivot = _mm512_set1_epi32( pivot );

   while (first != last)
   {
      //printf("first: %x %x %d\n", first, last, (int)std::distance( orig_first, first ));

      // If we have enough room, use simd version to jump forward quickly.
      while ( std::distance( first, last ) > v_len )
      {
         //__m512i v = _mm512_loadu_epi32( first );
         __m512i v = isAligned( first, 64u ) ? _mm512_load_si512( first) : _mm512_loadu_si512( first );
         __mmask16 mask = _mm512_cmplt_epi32_mask( v, v_pivot );
         //printf("forward mask %x\n", mask);
         if ( mask == 0 ) // any
            first += v_len;
         else
            break;
      }

      // And now walk element by element.
      while (comp(*first, pivot))
      {
         ++first;
         if (first == last)
            return first;
      }

      // Do the same from the back end ...
      while ( std::distance( first, last ) > v_len )
      {
         //__m512i v = _mm512_loadu_si512( last - v_len );
         int *ptr = last - v_len;
         __m512i v = isAligned( ptr, 64u ) ? _mm512_load_si512( ptr) : _mm512_loadu_si512( ptr );
         __mmask16 mask = _mm512_cmplt_epi32_mask( v, v_pivot );
         //printf("backward mask %x\n", mask);
         if ( mask == 0 )
            last -= v_len;
         else
            break;
      }

      do
      {
         --last;
         if (first == last)
            return first;
      }
      while (!comp(*last, pivot));

      std::swap (*first, *last);
      ++first;
   }

   return first;
}
#endif

template <typename T, class BinaryComparisonOp>
bool isSorted (T *first, T *last, const BinaryComparisonOp &comp, T *failure = NULL)
{
   T *next = first+1;
   for (; next != last; ++first, ++next)
      if (comp(*next, *first))
         break;
         //return first;

   if (failure != NULL) failure = next;
   return (next == last);
}

template <typename T>
inline T medianOfThree (const T x, const T y, const T z)
{
   if (x < y) {
      if (y < z) return y; // [x, y, z] <-
      if (x < z) return z; // [x, z, y] ->
   }
   if (y < z) {
      if (z < x) return z; // [y, z, x]
      if (x < z) return x; // [y, x, z]
   }
   if (z < x) {
      if (z < y) return y; // [z, y, x]
      if (x < y) return x; // [z, x, y]
   }
   return y;
}

template <typename T>
inline T selectPivot (T *first, T *last)
{
   //const int idx = (last - first)/2;
   //const int idx = rand() % (last - first);
   //if (first + idx >= last)
   //{  fprintf(stderr,"first + idx >= last %d\n", idx); exit(-1);
   //return first[idx];

   if (last - first > 3)
   {
      const T lo  = *first;
      const T mid = first[(last - first)/2];
      const T hi  = *(last-1);
      return medianOfThree(lo, mid, hi);
   }
   else
      return first[0];
}

template <typename T>
void printList (T *first, T *last)
{
   const int n = last - first;
   printf("[");
   for(int i = 0; i < (n-1); ++i)
      std::cout << first[i] << ",";
   if (n>0) std::cout << first[n-1];
   printf("]\n");
}

template <typename T, class BinaryComparisonOp>
void quickSortSerial (T *first, T *last, const BinaryComparisonOp &comp, int level)
{
   const size_t cutOff = 10;
   const size_t n = last - first;
   //level ++;
   //if (level > 32) {
   //   fprintf(stderr,"level>32\n");
   //   exit(-1);
   //}
   if (n > 1)
      if (n < cutOff)
         // Switch to simple sorting algorithm.
         selectionSort (n, first, comp);
      else
      {
         // Take middle value.
         const T pivotValue = selectPivot(first,last);
         //const T pivotValue = first[n/2];
         T *middle = myPartition(first, last, pivotValue, comp);
         //std::cout << "pivot = " << pivotValue << "," << n << "," << *middle << "," << middle-first <<std::endl;
         //printList(first,middle);
         //printList(middle,last);

         // Tricky case when the pivot falls at the head.
         if (middle == first || middle == last)
         {
            while (middle < last and !comp(pivotValue,*middle))
               middle++;

            if (middle == last)
            {
               //std::cout << "sorted" << std::endl;
               return;
            }

            //std::cout << "middle updated to " << *middle << "," << middle-first << std::endl;

            const T newPivot = *middle;
            middle = myPartition(first, last, newPivot, comp);

            //printList(first,middle);
            //printList(middle,last);
         }

         // Do small side first.
         if (middle - first <= last - middle)
         {
            if (middle != first)
               quickSortSerial (first, middle, comp, level);
            //if (middle != last)
               quickSortSerial (middle, last, comp, level);
         }
         else
         {
            if (middle != last)
               quickSortSerial (middle, last, comp, level);
            //if (middle != first)
               quickSortSerial (first, middle, comp, level);
         }
         //if (middle != first)
         //   quickSort (first, middle, comp, level);
         //if (middle != last)
         //   quickSort (middle, last, comp, level);
      }
}

template <typename T, class BinaryComparisonOp>
void quickSortParallel (T *first, T *last, int level)
{
   const int maxParallelDepth = 8;
   BinaryComparisonOp comp;

   const size_t cutOff = 10;
   const size_t n = last - first;

   if (n > 1)
      if (n < cutOff)
         // Switch to simple sorting algorithm.
         selectionSort (n, first, comp);
      else
      {
         // Take middle value.
         const T pivotValue = selectPivot(first,last);
         T *middle = myPartition(first, last, pivotValue, comp);

         // Tricky case when the pivot falls at the head.
         if (middle == first || middle == last)
         {
            while (middle < last and !comp(pivotValue,*middle))
               middle++;

            if (middle == last)
            {
               //std::cout << "sorted" << std::endl;
               return;
            }

            const T newPivot = *middle;
            middle = myPartition(first, last, newPivot, comp);
         }

         // Do small side first.
         if (middle - first <= last - middle)
         {
            if (middle != first)
            {
               #pragma omp task if (level < maxParallelDepth)
               quickSortParallel<T,BinaryComparisonOp> (first, middle, level);
            }
            //if (middle != last)
               quickSortParallel<T,BinaryComparisonOp> (middle, last, level);
         }
         else
         {
            if (middle != last)
            {
               #pragma omp task if (level < maxParallelDepth)
               quickSortParallel<T,BinaryComparisonOp> (middle, last, level);
            }
            //if (middle != first)
               quickSortParallel<T,BinaryComparisonOp> (first, middle, level);
         }
      }
}

template <typename T, class BinaryComparisonOp>
void quickSort (T *first, T *last, const BinaryComparisonOp &comp)
{
   if (first >= last)
      return;

   //quickSortSerial (first, last, comp, 0);
   #pragma omp parallel
   {
      #pragma omp single nowait
      quickSortParallel<T,BinaryComparisonOp> (first, last, 0);
   }
}

template <typename T, class BinaryComparisonOp>
void mergeSort_Merge (T *a, T *a_last, T *b, T *b_last, T *out, BinaryComparisonOp &comp)
{
   // Merge A[] and B[] into Out[]. len(Out[]) >= len(A[]) + len(B[])

   // 1. Both lists have elements.
   while (a < a_last && b < b_last)
   {
      //*out = comp(*a, *b) ? *a++ : *b++;
      //out++;
      *out++ = comp(*a, *b) ? *a++ : *b++;
   }

   // 2. Only A[] or B[] has elements left. Directly copy over.
   while (a < a_last)
   {
      //*out = *a++;
      //out++;
      *out++ = *a++;
   }
   while (b < b_last)
   {
      //*out = *b++;
      //out++;
      *out++ = *b++;
   }
}

template <typename T>
void mergeSort_Copy (T *a_first, T *a_last, T *b)
{
   const size_t len = a_last - a_first;
   T *a = a_first;
   for (size_t i = 0; i < len; ++i)
      b[i] = a[i];
}
template <typename T, class BinaryComparisonOp>
T * BinarySearch (const T value, T *lo, T *hi, const BinaryComparisonOp &comp)
{
   while (lo < hi)
   {
      T *mid = lo + (hi - lo) / 2;
      if (comp(value,*mid))
         hi = mid;
      else
         lo = mid+1;
   }
   return hi;
}
template <typename T, class BinaryComparisonOp>
void mergeSort_ParallelMerge (T *a, T *a_last, T *b, T *b_last, T *out, const BinaryComparisonOp &comp, int level = 0, const int maxParallelDepth = 8)
{
//   const int maxDepth = 8;

   // Merge A[] and B[] into Out[]. len(Out[]) >= len(A[]) + len(B[])

   level++;

   // Early return.
   if (a == a_last)
   {
      mergeSort_Copy (b, b_last, out);
      return;
   }
   if (b == b_last)
   {
      mergeSort_Copy (a, a_last, out);
      return;
   }

   // Make A[] the longest run.
   if ((a_last - a) < (b_last - b))
   {
      std::swap(a, b);
      std::swap(a_last, b_last);
   }

   const size_t cutOff = 128;
   if ((a_last - a) < cutOff)
      mergeSort_Merge (a, a_last, b, b_last, out, comp);
   else
   {
      // Midpoint through A[].
      T *p = a + (a_last - a) / 2;

      // Find location 'q' within B[] where A[p] falls.
      T *q = BinarySearch (*p, b, b_last, comp);

      // Find location 'r' in Out[] where A[p] should go.
      T *r = out + (p - a) + (q - b);
      *r = *p;

      // Recurse down again.
      //mergeSort_ParallelMerge (a,   p,      b, q,      out, comp, level);
      //mergeSort_ParallelMerge (p+1, a_last, q, b_last, r+1, comp, level);
      #pragma omp task if (level < maxParallelDepth)
      mergeSort_ParallelMerge (a,   p,      b, q,      out, BinaryComparisonOp(), level, maxParallelDepth);
      //#pragma omp task
      mergeSort_ParallelMerge (p+1, a_last, q, b_last, r+1, BinaryComparisonOp(), level, maxParallelDepth);
      #pragma omp taskwait
   }

   return;
}

template <typename T, class BinaryComparisonOp>
void mergeSort_TopDownSplit (T *a_first, T *a_last, T *b_first, const BinaryComparisonOp &comp, int level = 0)
{
   const size_t maxParallelDepth = 8;
   const size_t cutOff = 16;
   const size_t len = a_last - a_first;

#ifdef _OPENMP
   const int thread_id = omp_get_thread_num();
#else
   const int thread_id(0);
#endif

   level++;

   // Empty list. Return promptly.
   if (len > 1)
      if (len < cutOff)
         selectionSort (len, a_first, comp);
      else
      {
         // Pick middle of A[] and B[].
         const size_t half_len = len / 2;
         T *a_middle = a_first + half_len;
         T *b_middle = b_first + half_len;
         T *b_last   = b_first + len;

         // Initiate Top-down recursion.
         #pragma omp task if (level < maxParallelDepth)
         mergeSort_TopDownSplit (a_first,  a_middle, b_first,  BinaryComparisonOp(), level); // Low
         //#pragma omp task
         mergeSort_TopDownSplit (a_middle, a_last,   b_middle, BinaryComparisonOp(), level); // High
         //mergeSort_TopDownSplit (a_first,  a_middle, b_first,  comp, level); // Low
         //mergeSort_TopDownSplit (a_middle, a_last,   b_middle, comp, level); // High

         #pragma omp taskwait

         // A[first,middle) and A[middle,last) are both sorted.
         // Merge them the low and high partitions into B[].
         //if (len < 100)
         if (level < maxParallelDepth)
            mergeSort_ParallelMerge (a_first,  a_middle, a_middle, a_last, b_first, comp);
         else
            mergeSort_Merge (a_first,  a_middle, a_middle, a_last, b_first, comp);
         //else
         //   mergeSort_ParallelMerge (a_first,  a_middle, a_middle, a_last, b_first, comp);
         //std::merge(a_first, a_middle, a_middle, a_last, b_first);

         // Copy solution back into A[].
         //std::copy(b_first, b_last, a_first);
         mergeSort_Copy(b_first, b_last, a_first);
      }
}
template <typename T, class BinaryComparisonOp>
void mergeSort (T *first, T *last, const BinaryComparisonOp &comp)
{
   // Empty list. Return promptly.
   if (first >= last)
      return;

   const size_t n = last - first;

   // Need scratch buffer.
   T *buffer = NULL;
   Allocate(buffer, n);

   // Initiate Top-down recursion.
   #pragma omp parallel default(shared)
   {
      #pragma omp single
      mergeSort_TopDownSplit (first, last, buffer, comp);
   }

   Deallocate(buffer);
}

template <typename T, class BinaryComparisonOp>
void hybridSort (T *first, T *last, const BinaryComparisonOp &comp)
{
   if (first >= last)
      return;

   // 1) Split the input range into chunks and partition each independently.
   // 2) Recursively merge the sorted lists into a single sorted list.

   int nChunks = 4;
#ifdef _OPENMP
   nChunks = omp_get_max_threads();
#endif
   size_t n = last - first;
   size_t chunkSize = n / nChunks;
   //printf("nChunks = %d, n=%d, chunkSize=%d\n", nChunks, n, chunkSize);

   TimerType t0 = getTimeStamp();

   std::vector<size_t> idx(nChunks+1);

   for (int k = 0; k < nChunks; ++k)
      idx[k] = k * chunkSize;
   idx[nChunks] = n;

   #pragma omp parallel for
   for (int k = 0; k < nChunks; ++k)
   {
      T *ptr0 = first + idx[k];
      T *ptr1 = first + idx[k+1];
      //quickSortSerial (ptr0, ptr1, comp, 0);
      std::sort (ptr0, ptr1, comp);
   }

   TimerType t1 = getTimeStamp();
   //printf("sort time = %g\n", 1000*getElapsedTime(t0,t1));

   TimerType t2 = getTimeStamp();

   //std::vector<T> buf(n);
   T *buf = NULL;
   Allocate( buf, n );

   T *in = first;
   T *out = buf;

   for (int stride = 1; stride < nChunks; stride *= 2)
   {
      //printf("stride %d %x %x\n", stride, in, first);
      #pragma omp parallel for
      for (int k = 0; k < nChunks; k += 2*stride)
      {
         // Merge in[k] & in[k+stride] -> out[k]
         T *k_first = &in[idx[k]];
         T *k_mid   = &in[idx[k+stride]];
         T *k_last  = &in[idx[k+2*stride]];
         T *k_out   = &out[idx[k]];
         mergeSort_Merge(k_first, k_mid, k_mid, k_last, k_out, comp);
         //printf("merged %d %d %d %d %d\n", stride, idx[k], idx[k+stride], idx[k+2*stride], idx[k]);
      }
      std::swap(in,out);
   }

   if (out == first)
   {
      //printf("copied buf -> first\n");
      std::copy(buf, buf + n, first);
   }

   Deallocate( buf );

   TimerType t3 = getTimeStamp();
   //printf("merge time = %g\n", 1000*getElapsedTime(t2,t3));
}

template <typename T>
struct random_value
{
   T operator()(void) const;
   T operator()(const T, const T) const;
};

template <> float random_value<float>::
   operator()(const float lo, const float hi) const
      { return (float(rand()) / RAND_MAX) * (hi-lo) + lo; }
template <> float random_value<float>::
   operator()(void) const
      { return random_value<float>::operator()(0.f, 1.f); }

template <> double random_value<double>::
   operator()(const double lo, const double hi) const
      { return (double(rand()) / RAND_MAX) * (hi-lo) + lo; }
template <> double random_value<double>::
   operator()(void) const
      { return random_value<double>::operator()(0., 1.); }

template <> int random_value<int>::
   operator()(const int lo, const int hi) const
      { return lo + rand() % (hi - lo + 1); }
template <> int random_value<int>::
   operator()(void) const
      { return rand(); }

template <> long random_value<long>::
   operator()(const long lo, const long hi) const
      { return lo + rand() % (hi - lo + 1); }
template <> long random_value<long>::
   operator()(void) const
      { return rand(); }

template <> char random_value<char>::
   operator()(const char _lo, const char _hi) const
      {
         int lo = int(_lo);
         int hi = int(_hi);
         random_value<int> op;
         return char(op(lo,hi));
      }
template <> char random_value<char>::
   operator()(void) const
      { return random_value<char>::operator()('a','z'); }
         //int val = 97 + rand() % (122 - 97 + 1);
         //return char(val);
      //}

typedef int ValueType;

template <typename ValueType, typename IndexType>//, class CompType>
struct IndexSortHelper
{
   typedef ValueType value_type;
   typedef IndexType index_type;
   //typedef CompType  comp_type;

   value_type *val;
   index_type *idx;

   IndexSortHelper(value_type *val, index_type *idx) : val(val), idx(idx) {}

   bool operator()(const IndexType &left, const IndexType &right) const
   {
      return val[left] < val[right];
      //comp_type comp;
      //return comp_type( val[left], val[right] );
   }
};

enum algorithmTagType
{
   stdSortTag,
   selectSortTag,
   mergeSortTag,
   quickSortTag,
   hybridSortTag,
   partitionOnlyTag,
   numberOfAlgorithms // Always last.
};

std::string getAlgorithmName (int tag)
{
   switch (tag)
   {
      case stdSortTag: return std::string("std::qsort");
      case selectSortTag: return std::string("selectSort");
      case mergeSortTag: return std::string("mergeSort");
      case quickSortTag: return std::string("quickSort");
      case hybridSortTag: return std::string("hybridSort");
      case partitionOnlyTag: return std::string("partitionOnly");
      default:
         fprintf(stderr,"Unknown sorting tag\n");
         exit(-1);
   }
}
algorithmTagType getAlgorithmTag (int tag)
{
   if      (tag == int(stdSortTag) ) return stdSortTag;
   else if (tag == int(selectSortTag) ) return selectSortTag;
   else if (tag == int(mergeSortTag) ) return mergeSortTag;
   else if (tag == int(quickSortTag) ) return quickSortTag;
   else if (tag == int(hybridSortTag) ) return hybridSortTag;
   else if (tag == int(partitionOnlyTag) ) return partitionOnlyTag;
   else
   {
      fprintf(stderr,"Unknown sorting tag %d\n", tag);
      exit(-1);
   }
}


void run_test (const int n, int numTests, const algorithmTagType algorithmTag, const bool doIndexSort = false)
{
   ValueType *a = NULL, *b = NULL;
   Allocate(a, n);
   Allocate(b, n);

   int *idx = NULL;
   if (doIndexSort)
      Allocate(idx, n);

   typedef std::less<ValueType> comp;

   typedef IndexSortHelper<ValueType,int> index_comp_type;

   index_comp_type index_comp( b, idx );

   // Initialize a[] with random numbers (0,1].
   {
      // 1. Seed the pseudo-random generator.
      srand(n);

      random_value<ValueType> random;

      // 2. Loop over a[] and set each value.
      for (int i = 0; i < n; ++i)
      {
         // rand() returns an integer [0,RAND_MAX]. Normalize to [0,1].
         //a[i] = ValueType(rand()) / RAND_MAX;
         //a[i] = rand() % (n+1);
         a[i] = random(0,n);
         //a[i] = random();
         //a[i] = random('A','Z');
         //std::cout << i << "," << a[i] << std::endl;
      }
   }

   int idummy = 1;
   ValueType dummy = 2;

   // Run the test several times.
   double tSort = 0;
   int niters = (numTests > 1) ? numTests : 1;

   for(;;)
   {
      TimerType t_copy_start = getTimeStamp();
      for (int k = 0; k < niters; ++k)
      {
         // 1. Copy a[] to b[].
         std::copy(a, a + n, b);

         int a_index = rand() % n;
         int b_index = rand() % n;
         dummy += a[a_index] + b[b_index];

         dummy_function( n, (void*)a, (void*)b );

         if (doIndexSort) {
            for (int i = 0; i < n; ++i)
               idx[i] = i;

            idummy = std::min( idummy, idx[a_index] );
         }
      }
      TimerType t_copy_stop = getTimeStamp();
      double copy_time = getElapsedTime(t_copy_start, t_copy_stop );

      TimerType t_start = getTimeStamp();
      for (int k = 0; k < niters; ++k)
      {
         // 1. Copy a[] to b[].
         std::copy(a, a + n, b);

         dummy_function( n, (void*)a, (void*)b );

         int a_index = rand() % n;
         int b_index = rand() % n;
         dummy += a[a_index] + b[b_index];

         if (doIndexSort)
         {
            for (int i = 0; i < n; ++i)
               idx[i] = i;

            idummy = std::min( idummy, idx[a_index] );

            if (algorithmTag == selectSortTag)
               selectionSort (n, idx, index_comp);
            else if (algorithmTag == stdSortTag)
               std::sort(idx, idx+n, index_comp);
            else
            {
               fprintf(stderr,"Index sort is only support with selection- or quick-sort\n");
               return;
            }
         }
         else
         {
            if (algorithmTag == partitionOnlyTag) // just run the partition kernel
            {
               const ValueType piv = selectPivot( b, b + n );
               //const ValueType piv = 1;

               auto print_list = [&n]( ValueType *u )
                  {
                     for (int j = 0; j < n; j++)
                     {
                        std::cout << u[j];
                        if ( j % 30 == 29 or j == (n-1) )
                           std::cout << std::endl;
                        else
                           std::cout << ", ";
                     }
                  };

               if ( n < 100 ) {
                  printf("piv = %d\n", piv);
                  print_list(b);
               }

               ValueType *mid = NULL;

#ifdef _OPENMP
#  ifdef ENABLE_PARALLEL_CXX
               {
                  auto op = comp();
                  auto unary_op = [&]( const ValueType& el ) { return op(el, piv); };
                //auto it = std::partition( std::parallel::par, b, b+n, unary_op );
                  auto it = __gnu_parallel::partition( b, b+n, unary_op );
                  mid = &*it;
               }
#  else
               mid = ParallelPartition( b, n, piv, comp() );
#  endif
#else
#ifdef ENABLE_SIMD
#warning "Calling SIMD Parallel Partitioner"
               mid = myPartitionSimd( b, b + n, piv, comp() );
#else
               mid = myPartition( b, b + n, piv, comp() );
#endif
#endif

               if ( n < 100 ) {
                  printf("partitioned = %d\n", std::distance(b, mid));
                  print_list(b);

                  ValueType *vmid = myPartition( a, a + n, piv, comp() );

                  printf("parallel = %d\n", std::distance(a, vmid));
                  print_list(a);

                  return;
               }
            }
            else {
            // 2. Sort b[].
            if (algorithmTag == selectSortTag)
               selectionSort (n, b, comp());
            else if (algorithmTag == mergeSortTag)
               mergeSort (b, b + n, comp());
            else if (algorithmTag == quickSortTag)
               quickSort (b, b + n, comp());
            else if (algorithmTag == hybridSortTag)
               hybridSort (b, b + n, comp());
            else
               std::sort (b, b + n, comp());
            }
         }
      }

      tSort = getElapsedTime(t_start, getTimeStamp());
      //printf("tSort: %e %e %e\n", tSort, copy_time, tSort - copy_time);
      if (copy_time > 0) tSort -= copy_time;

      if (tSort > 0.1 or numTests > 1)
         break;
      else
         niters *= 2;
   }
   tSort /= niters;

   srand( (unsigned int) dummy );
   srand( (unsigned int) idummy );

   bool listIsSorted = false;
   if (algorithmTag != partitionOnlyTag)
   {
      if (doIndexSort)
         listIsSorted = isSorted(idx, idx+n, index_comp);
      else
         listIsSorted = isSorted(b, b+n, comp());

      if (not(listIsSorted))
         fprintf(stderr, "Error: output list is not sorted\n");
   }

   printf("%10d, %20.6f, %10d, %20.4f, %20.4f\n", n, tSort*1000.0, niters, sizeof(ValueType)*size_t(n)/1024., (1e-6 * n) / (tSort));

   // Print a[] and b[] (if not too large).
   if (1 && n < 50)
   {
      for (int i = 0; i < n; ++i)
         if (doIndexSort)
            std::cout << i << "," << a[i] << "," << b[idx[i]] << "," << idx[i] << std::endl;
         else
            std::cout << i << "," << a[i] << "," << b[i] << std::endl;

      if (listIsSorted)
      {
         ValueType *p = BinarySearch (b[n/2], b, b+n, comp());
         std::cout << "Search: " << b[n/2] << ", " << p-b << ", " << *p << std::endl;
      }
   }

   Deallocate( a );
   Deallocate( b );
   if (doIndexSort)
      Deallocate( idx );
}


int main (int argc, char* argv[])
{
   // Define the smallest size of the array to be sorted. The default is 10.
   int minSize = 10;

   // Define the largest size of the array to be sorted. The default is 1000000.
   int maxSize = 1000000;

   // Define the number of tests to run for statistics.
   int num_tests = 0;

   int algorithm = 1;

   int doIndexSort = 0;

   float stepSize = 1.5;

   algorithmTagType algorithmTag = stdSortTag;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

#define print_help() \
      { fprintf(stderr,"sort --help|-h --nelems|-n # -min # -max # --ntests|-t # --algorithm|-a # --index|-i --stepsize|-s #\n"); \
        fprintf(stderr,"\talgorithms: default=0\n"); \
        for (int k = 0; k < numberOfAlgorithms; ++k) \
           fprintf(stderr,"\t%d) %s\n", k, getAlgorithmName( getAlgorithmTag(k) ).c_str()); \
      }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         print_help();
         return 1;
      }
      else if (strcmp(argv[i],"--nelems") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--nelems|-n");
         i++;
         if (isdigit(*argv[i]))
            minSize = maxSize = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"-min") == 0)
      {
         check_index(i+1,"-min");
         i++;
         if (isdigit(*argv[i]))
            minSize = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"-max") == 0)
      {
         check_index(i+1,"-max");
         i++;
         if (isdigit(*argv[i]))
            maxSize = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--ntests") == 0 || strcmp(argv[i],"-t") == 0)
      {
         check_index(i+1,"--ntests|-t");
         i++;
         if (isdigit(*argv[i]) || *argv[i] == '-')
            num_tests = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--algorithm") == 0 || strcmp(argv[i],"-a") == 0)
      {
         check_index(i+1,"--algorithm|-a");
         i++;
         if (isdigit(*argv[i]))
         {
            algorithmTag = getAlgorithmTag( atoi( argv[i] ) );
         }
      }
      else if (strcmp(argv[i],"--stepsize") == 0 || strcmp(argv[i],"-s") == 0)
      {
         check_index(i+1,"--stepsize|-s");
         i++;
         if (isdigit(*argv[i]) || *argv[i] == '.')
            stepSize = atof( argv[i] );
      }
      else if (strcmp(argv[i],"--index") == 0 || strcmp(argv[i],"-i") == 0)
      {
         doIndexSort = true;
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         print_help();
         return 1;
      }
   }

   fprintf(stderr,"algorithm = %d %s\n", algorithmTag, getAlgorithmName(algorithmTag).c_str());

   #ifdef _OPENMP
   fprintf(stderr,"num_threads = %d\n", omp_get_max_threads());
   #endif

   fprintf(stderr,"minSize = %d, maxSize = %d, stepSize = %f, numTests = %d, indexSort = %d\n", minSize, maxSize, stepSize, num_tests, doIndexSort);

   fprintf(stderr, "%10s, %20s, %10s, %20s, %20s\n", "Length", "Time (ms)", "Ntests", "Size (kb)", "TPut (/us)");

   for (int size = minSize; size <= maxSize; size *= stepSize)
      run_test(size, num_tests, algorithmTag, doIndexSort);

   return 0;
}
