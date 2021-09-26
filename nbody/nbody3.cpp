#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <typeinfo>

#include <timer.h>
#include <aligned_allocator.h>

#ifndef __RESTRICT
#  define __RESTRICT
#endif

#define NDIM (3)

//#define Enable_ArrayOfStructures
#if defined(Enable_ArrayOfStructures) || defined(__AOS)
#  ifndef Enable_ArrayOfStructures
#    define Enable_ArrayOfStructures
#  endif
   /* Array-of-structures (like) format. */
#  define _index(i,j) (NDIM*(i) + (j))
#else
   /* Structure-of-arrays (like) format. */
//#  define _index(i,j) ((i) + (j)*n)
#  define _index(i,j) ((i) + (j)*n_pad)
#endif

#define acc_array(i,j) acc[ _index((i),(j)) ]
#define pos_array(i,j) pos[ _index((i),(j)) ]
#define vel_array(i,j) vel[ _index((i),(j)) ]

template <typename ValueType>
constexpr ValueType square(const ValueType& x) { return x*x; }

#define G ( ValueType(1) )
#define TINY ( std::numeric_limits<ValueType>::epsilon() )
#define TINY2 ( TINY * TINY )

/* Generate a random double between 0,1. */
template <typename ValueType>
ValueType frand(void) { return ValueType( rand() ) / RAND_MAX; }

template <typename ValueType>
void accel_naive (ValueType pos[], ValueType vel[], ValueType mass[], ValueType acc[], const int n, const int n_pad)
{
   for (int i = 0; i < n; ++i)
      for (int k = 0; k < NDIM; ++k)
         acc_array(i,k) = ValueType(0);

   for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
      {
         /* Position vector from i to j and the distance^2. */
         ValueType rx = pos_array(j,0) - pos_array(i,0);
         ValueType ry = pos_array(j,1) - pos_array(i,1);
         ValueType rz = pos_array(j,2) - pos_array(i,2);
         ValueType dsq = rx*rx + ry*ry + rz*rz + TINY2;
         ValueType dcu = dsq * std::sqrt(dsq);

         acc_array(i,0) += G * mass[j] * rx / dcu;
         acc_array(i,1) += G * mass[j] * ry / dcu;
         acc_array(i,2) += G * mass[j] * rz / dcu;
      }
}

// Operator strength reduction: replace equivalent math with cheaper operations.
template <typename ValueType>
void accel_strength (ValueType pos[], ValueType vel[], ValueType mass[], ValueType acc[], const int n, const int n_pad)
{
   for (int i = 0; i < n; ++i)
      for (int k = 0; k < NDIM; ++k)
         acc_array(i,k) = ValueType(0);

   for (int i = 0; i < n; ++i)
   {
      for (int j = 0; j < n; ++j)
      {
         /* Position vector from i to j and the distance^2. */
         ValueType rx = pos_array(j,0) - pos_array(i,0);
         ValueType ry = pos_array(j,1) - pos_array(i,1);
         ValueType rz = pos_array(j,2) - pos_array(i,2);
         ValueType dsq = rx*rx + ry*ry + rz*rz + TINY2;
         // Store 1/r^3 once instead of 3x divisions.
         ValueType m_invR3 = mass[j] / (dsq * std::sqrt(dsq));

         acc_array(i,0) += rx * m_invR3;
         acc_array(i,1) += ry * m_invR3;
         acc_array(i,2) += rz * m_invR3;
      }

      // Pull G out of repeated calculation.
      for (int k = 0; k < NDIM; ++k)
         acc_array(i,k) *= G;
   }
}

// Store target data in registers: Compiler "may" do this automatically but
// it often helps with cache efficiency. This can be especially helpfule
// by avoiding repeated writes which are several times slower than reads.
template <typename ValueType>
void accel_register (ValueType * __RESTRICT pos, ValueType * __RESTRICT vel, ValueType * __RESTRICT mass, ValueType * __RESTRICT acc, const int n, const int n_pad)
{
   for (int i = 0; i < n; ++i)
   {
      ValueType ax = 0, ay = 0, az = 0;
      const ValueType xi = pos_array(i,0);
      const ValueType yi = pos_array(i,1);
      const ValueType zi = pos_array(i,2);

      for (int j = 0; j < n; ++j)
      {
         /* Position vector from i to j and the distance^2. */
         ValueType rx = pos_array(j,0) - xi;
         ValueType ry = pos_array(j,1) - yi;
         ValueType rz = pos_array(j,2) - zi;
         ValueType dsq = rx*rx + ry*ry + rz*rz + TINY2;
         ValueType m_invR3 = mass[j] / (dsq * std::sqrt(dsq));

         ax += rx * m_invR3;
         ay += ry * m_invR3;
         az += rz * m_invR3;
      }

      acc_array(i,0) = G * ax;
      acc_array(i,1) = G * ay;
      acc_array(i,2) = G * az;
   }
}

//#pragma omp declare simd simdlen(32) uniform(xi,yi,zi) notinbranch
template <typename ValueType>
inline
void accel_ij_interaction (const ValueType xi, const ValueType yi, const ValueType zi,
                           const ValueType xj, const ValueType yj, const ValueType zj, const ValueType mj,
                           ValueType &ax, ValueType &ay, ValueType &az)
{
   /* Position vector from i to j and the distance^2. */
   const ValueType rx = xj - xi;
   const ValueType ry = yj - yi;
   const ValueType rz = zj - zi;
   const ValueType r2 = rx*rx + ry*ry + rz*rz + TINY2;
   const ValueType m_invR3 = mj / (r2 * std::sqrt(r2));

   ax += rx * m_invR3;
   ay += ry * m_invR3;
   az += rz * m_invR3;
}

// SIMD (vector) processing: Compute several inner interactions at once with SIMD operations.
template <typename ValueType>
void accel_inner_simd (ValueType * __RESTRICT pos, ValueType * __RESTRICT vel, ValueType * __RESTRICT mass, ValueType * __RESTRICT acc, const int n, const int n_pad)
{
   // Outer loop vectorization.
   for (int i = 0; i < n; ++i)
   {
      ValueType ax = 0, ay = 0, az = 0;
      const ValueType xi = pos_array(i,0);
      const ValueType yi = pos_array(i,1);
      const ValueType zi = pos_array(i,2);

      // Inner loop vectorization.
      #pragma omp simd aligned(pos, mass : __ALIGNMENT )
      for (int j = 0; j < n; ++j)
         accel_ij_interaction( xi, yi, zi,
                               pos_array(j,0), pos_array(j,1), pos_array(j,2), mass[j],
                               ax, ay, az );

      acc_array(i,0) = G * ax;
      acc_array(i,1) = G * ay;
      acc_array(i,2) = G * az;
   }
}

// SIMD (vector) processing: Compute several outer interactions at once with SIMD operations.
template <typename ValueType>
void accel_outer_simd (ValueType * __RESTRICT pos, ValueType * __RESTRICT vel, ValueType * __RESTRICT mass, ValueType * __RESTRICT acc, const int n, const int n_pad)
{
   // Outer loop vectorization.
   #pragma omp simd aligned(pos, mass : __ALIGNMENT )
   for (int i = 0; i < n; ++i)
   {
      ValueType ax = 0, ay = 0, az = 0;
      const ValueType xi = pos_array(i,0);
      const ValueType yi = pos_array(i,1);
      const ValueType zi = pos_array(i,2);

      // Inner loop vectorization.
      for (int j = 0; j < n; ++j)
         accel_ij_interaction( xi, yi, zi,
                               pos_array(j,0), pos_array(j,1), pos_array(j,2), mass[j],
                               ax, ay, az );

      acc_array(i,0) = G * ax;
      acc_array(i,1) = G * ay;
      acc_array(i,2) = G * az;
   }
}

#ifdef __ENABLE_VCL_SIMD

#if   (__ENABLE_VCL_SIMD == 128)
# define MAX_VECTOR_SIZE 128
#elif (__ENABLE_VCL_SIMD == 256)
# define MAX_VECTOR_SIZE 256
#elif (__ENABLE_VCL_SIMD == 512)
# define MAX_VECTOR_SIZE 512
#else
// Using default from available instruction set.
#endif

#include "vcl_helper.h"

//#define __ENABLE_VCL_RSQRT

/*template <int VL>
typename vcl_type<double, VL>::type rsqrt ( const typename vcl_type<double, VL>::type& v )
{
   typedef typename vcl_type<double, VL>::type T;
   return T(1) / sqrt(v);
}

template <int VL>
typename vcl_type<float, VL>::type rsqrt ( const typename vcl_type<float, VL>::type& v )
{
   typedef typename vcl_type<float, VL>::type T;
#ifdef __ENABLE_VCL_RSQRT
   return approx_rsqrt(v);
#else
   return T(1.f) / sqrt(v);
#endif
}*/

#if (MAX_VECTOR_SIZE >= 128)
Vec2d  rsqrt (const Vec2d& v) { return 1.0 / sqrt(v); }
Vec4f  rsqrt (const Vec4f& v) {
#ifdef __ENABLE_VCL_RSQRT
   return approx_rsqrt(v);
#else
   return float(1) / sqrt(v);
#endif
}
#endif
#if (MAX_VECTOR_SIZE >= 256)
Vec4d  rsqrt (const Vec4d& v) { return 1.0 / sqrt(v); }

Vec8f  rsqrt (const Vec8f& v) {
#ifdef __ENABLE_VCL_RSQRT
   return approx_rsqrt(v);
#else
   return float(1) / sqrt(v);
#endif
}
#endif
#if (MAX_VECTOR_SIZE >= 512)
//Vec8d  rsqrt (const Vec8d& v) { return 1.0 / sqrt(v); }
Vec8d  rsqrt (const Vec8d& v) {
#ifdef __ENABLE_VCL_RSQRT
   return approx_rsqrt(v);
#else
   return 1.0 / sqrt(v);
#endif
}
Vec16f  rsqrt (const Vec16f& v) {
#ifdef __ENABLE_VCL_RSQRT
   return approx_rsqrt(v);
#else
   return float(1) / sqrt(v);
#endif
}
#endif

// Explicit SIMD (vector) processing: Compute several interactions at once with SIMD operations.
// Using the outer (target) particles into a SIMD word.
template <typename ValueType>
void accel_vcl_simd (ValueType * __RESTRICT pos, ValueType * __RESTRICT vel, ValueType * __RESTRICT mass, ValueType * __RESTRICT acc, const int n, const int n_pad)
{
   const int VL = SIMD_Vector_Length< ValueType >();
   typedef typename vcl_type< ValueType, VL >::type simdType;

   // Must be aligned to wordsize (preferrably cacheline) boundary.
   if ( isAligned(     pos, sizeof(simdType)) == false ||
        isAligned(     vel, sizeof(simdType)) == false ||
        isAligned(     acc, sizeof(simdType)) == false ||
        isAligned(    mass, sizeof(simdType)) == false ||
        isAligned( &pos_array(0,1), sizeof(simdType)) == false ||
        isAligned( &pos_array(0,2), sizeof(simdType)) == false )
   {
      fprintf(stderr,"Data is not aligned to %u-byte boundary: %p %p %p %p %d %d\n", sizeof(simdType), pos, vel, acc, mass, n, n_pad);
      exit(1);
   }

   // And we must be using SoA
#ifdef Enable_ArrayOfStructures
   if (1)
   {
      fprintf(stderr,"explicit simd method requires SoA format\n");
      exit(1);
   }
#endif

   const simdType vTINY2( TINY2 );
   const simdType vG( G );

   // Outer loop vectorization.
   for (int i = 0; i < n; i += VL)
   {
      simdType ax(0), ay(0), az(0);

      simdType xi, yi, zi;

      if (i + VL < n)
      {
         xi.load_a( &pos_array(i,0) );
         yi.load_a( &pos_array(i,1) );
         zi.load_a( &pos_array(i,2) );
      }
      else
      {
         xi = yi = zi = 0;
         xi.load_partial( n-i, &pos_array(i,0) );
         yi.load_partial( n-i, &pos_array(i,1) );
         zi.load_partial( n-i, &pos_array(i,2) );
      }

      for (int j = 0; j < n; ++j)
      {
         // Set all lanes to the same j values.
         const simdType xj( pos_array(j,0) );
         const simdType yj( pos_array(j,1) );
         const simdType zj( pos_array(j,2) );
         const simdType mj( mass[j] );

         // Position vector from i to j and the distance^3.
         const simdType rx = xj - xi;
         const simdType ry = yj - yi;
         const simdType rz = zj - zi;

         const simdType r2 = rx*rx + ry*ry + rz*rz + vTINY2;

         simdType m_invR3;
         if ( 1 )
         {
            const simdType invR = rsqrt( r2 );
            m_invR3 = mj * (invR * invR * invR);
         }
         else
            m_invR3 = mj / (r2 * sqrt(r2));

         ax += (rx * m_invR3);
         ay += (ry * m_invR3);
         az += (rz * m_invR3);
      }

      ax *= vG;
      ay *= vG;
      az *= vG;

      if (i + VL < n)
      {
         ax.store_a( &acc_array(i,0) );
         ay.store_a( &acc_array(i,1) );
         az.store_a( &acc_array(i,2) );
      }
      else
      {
         ax.store_partial( n-i, &acc_array(i,0) );
         ay.store_partial( n-i, &acc_array(i,1) );
         az.store_partial( n-i, &acc_array(i,2) );
      }
   }
}

#if 0
// Shuffle the lanes to the right.
Vec2d  rotate_right( const Vec2d  x ) { return permute2d<1,0>(x); }
Vec4f  rotate_right( const Vec4f  x ) { return permute4f<3,0,1,2>(x); }
#if MAX_VECTOR_SIZE >= 256
Vec4d  rotate_right( const Vec4d  x ) { return permute4d<3,0,1,2>(x); }
Vec8f  rotate_right( const Vec8f  x ) { return permute8f<7,0,1,2,3,4,5,6>(x); }
#endif
#if MAX_VECTOR_SIZE >= 512
Vec8d  rotate_right( const Vec8d  x ) { return permute8d<7,0,1,2,3,4,5,6>(x); }
Vec16f rotate_right( const Vec16f x ) { return permute16f<15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x); }
#endif

// Explicit SIMD (vector) processing: Compute several interactions at once with SIMD operations.
// Load the outer (target) particles into a SIMD word and do the same for the inner loop. Careful!
// still need to interact with all lanes of the inner loop word.
template <typename ValueType>
void accel_vcl_simd_rotate (ValueType * __RESTRICT pos, ValueType * __RESTRICT vel, ValueType * __RESTRICT mass, ValueType * __RESTRICT acc, const int n)
{
   typedef typename set_simdType<ValueType>::simdType simdType;
   const int simd_length = sizeof(simdType) / sizeof(ValueType);

   // Must be aligned to wordsize (preferrably cacheline) boundary.
   if ( isAligned(  pos, sizeof(simdType )) == false ||
        isAligned(  vel, sizeof(simdType )) == false ||
        isAligned(  acc, sizeof(simdType )) == false ||
        isAligned( mass, sizeof(simdType )) == false )
   {
      fprintf(stderr,"Data is not aligned to %u-byte boundary: %p %p %p %p\n", pos, vel, acc, mass, sizeof(simdType)*8);
      exit(1);
   }

   // And we must be using SoA
#ifdef Enable_ArrayOfStructures
   if (1)
   {
      fprintf(stderr,"explicit simd method requires SoA format\n");
      exit(1);
   }
#endif

   const simdType vTINY2( TINY2 );
   const simdType vG( G );

   // Outer loop vectorization.
   for (int i = 0; i < n; i += simd_length)
   {
      simdType ax(0), ay(0), az(0);

      simdType xi, yi, zi;

      if (i + simd_length < n)
      {
         xi.load_a( &pos_array(i,0) );
         yi.load_a( &pos_array(i,1) );
         zi.load_a( &pos_array(i,2) );
      }
      else
      {
         xi = yi = zi = 0;
         xi.load_partial( n-i, &pos_array(i,0) );
         yi.load_partial( n-i, &pos_array(i,1) );
         zi.load_partial( n-i, &pos_array(i,2) );
      }

      for (int j = 0; j < n; j += simd_length)
      {
         simdType xj, yj, zj, mj;

         if (j + simd_length < n)
         {
            xj.load_a( &pos_array(j,0) );
            yj.load_a( &pos_array(j,1) );
            zj.load_a( &pos_array(j,2) );
            mj.load_a( &mass[j] );
         }
         else
         {
            mj = 0;
            xj.load_partial( n - j, &pos_array(j,0) );
            yj.load_partial( n - j, &pos_array(j,1) );
            zj.load_partial( n - j, &pos_array(j,2) );
            mj.load_partial( n - j, &mass[j] );
         }

         // Now loop over the lanes.
         for (int lane = 0; lane < simd_length; lane++)
         {
            // Position vector from i to j and the distance^3.
            const simdType rx = xj - xi;
            const simdType ry = yj - yi;
            const simdType rz = zj - zi;

            const simdType r2 = rx*rx + ry*ry + rz*rz + vTINY2;

            simdType m_invR3;
            if ( sizeof(ValueType) == sizeof(float) )
            {
               const simdType invR = rsqrt( r2 );
               m_invR3 = mj * (invR * invR * invR);
            }
            else
               m_invR3 = mj / (r2 * sqrt(r2));

            ax += (rx * m_invR3);
            ay += (ry * m_invR3);
            az += (rz * m_invR3);

            // Now rotate the data in the lanes.
            if (lane < simd_length-1)
            {
               xj = rotate_right( xj );
               yj = rotate_right( yj );
               zj = rotate_right( zj );
               mj = rotate_right( mj );
            }
         }
      }

      ax *= vG;
      ay *= vG;
      az *= vG;

      if (i + simd_length < n)
      {
         ax.store_a( &acc_array(i,0) );
         ay.store_a( &acc_array(i,1) );
         az.store_a( &acc_array(i,2) );
      }
      else
      {
         ax.store_partial( n-i, &acc_array(i,0) );
         ay.store_partial( n-i, &acc_array(i,1) );
         az.store_partial( n-i, &acc_array(i,2) );
      }
   }
}
#endif

#endif

template <typename ValueType>
void update (ValueType pos[], ValueType vel[], ValueType mass[], ValueType acc[], const int n, ValueType h, const int n_pad)
{
   for (int i = 0; i < n; ++i)
      for (int k = 0; k < NDIM; ++k)
      {
         pos_array(i,k) += vel_array(i,k)*h + acc_array(i,k)*h*h/2;
         vel_array(i,k) += acc_array(i,k)*h;
      }
}

template <typename ValueType>
void output (ValueType pos[], ValueType vel[], ValueType mass[], ValueType acc[], const int n, int flnum, const int n_pad)
{
   char flname[20];
   sprintf (flname, "pos_%d.out", flnum);
   FILE *fp = fopen(flname,"w");
   if (!fp)
   {
      fprintf(stderr,"Error opening file %s\n", flname);
      exit(-1);
   }

   fwrite (&n, sizeof(int), 1, fp);
   for (int i = 0; i < n; ++i)
   {
      for (int k = 0; k < NDIM; ++k)
      {
         fwrite (&pos_array(i,k), sizeof(ValueType), 1, fp);
      }
      fwrite (&mass[i], sizeof(ValueType), 1, fp);
   }

   fclose(fp);
}

template <typename ValueType>
void search (ValueType pos[], ValueType vel[], ValueType mass[], ValueType acc[], const int n, const int n_pad)
{
   ValueType minv = 1e10, maxv = 0, ave = 0;
   for (int i = 0; i < n; ++i)
   {
      ValueType vmag = 0;
      for (int k = 0; k < NDIM; ++k)
         vmag += (vel_array(i,k) * vel_array(i,k));

      vmag = sqrt(vmag);

      maxv = std::max(maxv, vmag);
      minv = std::min(minv, vmag);
      ave += vmag;
   }
   printf("min/max/ave velocity = %e, %e, %e\n", minv, maxv, ave / n);
}

void help(const char* prg)
{
   if (prg) fprintf(stderr,"%s:\n", prg);
   fprintf(stderr,"\t--help | -h       : Print help message.\n");
   fprintf(stderr,"\t--nparticles | -n : # of particles (100).\n");
   fprintf(stderr,"\t--nsteps | -s     : # of steps to take (100).\n");
   fprintf(stderr,"\t--stepsize | -dt  : Delta-t step-size in seconds (0.01).\n");
   fprintf(stderr,"\t--float | -f      : Use 32-bit floats.\n");
   fprintf(stderr,"\t--double | -d     : Use 64-bit doubles. (default)\n");
   fprintf(stderr,"\t--method | -m     : Select acceleration evaluation method (1).\n");
   fprintf(stderr,"\t\tNaive              (0) : Direct equation translation.\n");
   fprintf(stderr,"\t\tStrength Reduction (1) : Factor out constants from inner loop.\n");
   fprintf(stderr,"\t\tRegister Storage   (2) : Use scalars (registers) to store outer loop data.\n");
   fprintf(stderr,"\t\tInner SIMD         (3) : Use OpenMP v4+ directive to vectorize inner j loop.\n");
   fprintf(stderr,"\t\tOuter SIMD         (4) : Use OpenMP v4+ directive to vectorize outer i loop.\n");
#ifdef __ENABLE_VCL_SIMD
   fprintf(stderr,"\t\tVCL SIMD           (5) : Use the VCL C++ library to explicitly vectorize.\n");
//   fprintf(stderr,"\t\tVCL SIMD-ROTATE    (6) : Use the VCL C++ library to explicitly vectorize and rotate the inner loop.\n");
#endif
}

enum { ACCEL_NAIVE = 0,
       ACCEL_STRENGTH,
       ACCEL_REGISTER,
       ACCEL_INNER_SIMD,
       ACCEL_OUTER_SIMD
#ifdef __ENABLE_VCL_SIMD
      ,ACCEL_VCL_SIMD
//      ,ACCEL_VCL_SIMD_ROTATE
#endif
};

std::vector< std::string > methods;

template <typename ValueType, int Method>
int run_tests( const int n, const int num_steps, const ValueType dt)
{
   fprintf(stderr,"Number Objects = %d\n", n);
   fprintf(stderr,"Number Steps   = %d\n", num_steps);
   fprintf(stderr,"Timestep size  = %g\n", dt);
   fprintf(stderr,"Alignment      = %lu bytes\n", Alignment());
   fprintf(stderr,"ValueType      = %s\n", (sizeof(ValueType)==sizeof(double)) ? "double" : "float");
#ifdef __ENABLE_VCL_SIMD
   const int VL = SIMD_Vector_Length< ValueType >();
   typedef vcl_type< ValueType, VL > simdType;
   fprintf(stderr,"SIMD Length    = %d (%d)\n", VL, MAX_VECTOR_SIZE);
   fprintf(stderr,"SIMD Type      = %s\n", typeid(simdType).name());
# ifdef __ENABLE_VCL_RSQRT
   fprintf(stderr,"SIMD rsqrt     = enabled\n");
# endif
#endif
#ifdef Enable_ArrayOfStructures
   fprintf(stderr,"Format         = ArrayOfStructures\n");
#else
   fprintf(stderr,"Format         = StructureOfArrays\n");
#endif

   fprintf(stderr,"Accel function = %s\n", methods[Method].c_str() );

   ValueType *pos = NULL;
   ValueType *vel = NULL;
   ValueType *acc = NULL;
   ValueType *mass = NULL;

   int n_pad = n;
#ifndef Enable_ArrayOfStructures
   const int cacheline_size = ( Alignment() / sizeof(ValueType) );
   if (n % cacheline_size != 0) {
      n_pad = n + (cacheline_size - n % cacheline_size);
   }
#endif
   //printf("%d %d\n", n, n_pad);
   fprintf(stderr,"Padding        = %d %d\n", (n_pad-n), n_pad );

   Allocate(pos, n_pad*NDIM);
   Allocate(vel, n_pad*NDIM);
   Allocate(acc, n_pad*NDIM);
   Allocate(mass, n);

   if (1 && n == 2)
   {
      /* Initialize a 2-body problem with large mass ratio and tangential
       * velocity for the small body. */

      pos_array(0,0) = 0.0; pos_array(0,1) = 0.0; pos_array(0,2) = 0.0;
      vel_array(0,0) = 0.0; vel_array(0,1) = 0.0; vel_array(0,2) = 0.0;
      mass[0] = 1000.0;

      ValueType vy = std::sqrt(G*mass[0]);
      pos_array(1,0) = 1.0; pos_array(1,1) = 0.0; pos_array(1,2) = 0.0;
      vel_array(1,0) = 0.0; vel_array(1,1) =  vy; vel_array(1,2) = 0.0;
      mass[1] = 1.0;
   }
   else
   {
      /* Initialize the positions and velocities with random numbers (0,1]. */

      /* 1. Seed the pseudo-random generator. */
      srand(n);

      for (int i = 0; i < n; ++i)
      {
         /* 2. Set some random positions for each object {-1,1}. */
         for (int k = 0; k < NDIM; ++k)
            pos_array(i,k) = 2*(frand<ValueType>() - 0.5);

         /* 3. Set some random velocity (or zero). */
         for (int k = 0; k < NDIM; ++k)
            vel_array(i,k) = 0;
            //vel_array(i,k) = frand();

         /* 4. Set a random mass (> 0). */
         mass[i] = frand<ValueType>() + TINY;

         for (int k = 0; k < NDIM; ++k)
            acc_array(i,k) = 0;
      }
   }

   /* Run the step several times. */
   TimerType t_start = getTimeStamp();
   double t_accel = 0, t_update = 0, t_search = 0;
   int flnum = 0;
   for (int step = 0; step < num_steps; ++step)
   {
      /* 1. Compute the acceleration on each object. */
      TimerType t0 = getTimeStamp();

      if (Method == ACCEL_NAIVE)
         accel_naive( pos, vel, mass, acc, n, n_pad );
      else if (Method == ACCEL_STRENGTH)
         accel_strength( pos, vel, mass, acc, n, n_pad );
      else if (Method == ACCEL_REGISTER)
         accel_register( pos, vel, mass, acc, n, n_pad );
      else if (Method == ACCEL_INNER_SIMD)
         accel_inner_simd( pos, vel, mass, acc, n, n_pad );
      else if (Method == ACCEL_OUTER_SIMD)
         accel_outer_simd( pos, vel, mass, acc, n, n_pad );
#ifdef __ENABLE_VCL_SIMD
      else if (Method == ACCEL_VCL_SIMD)
         accel_vcl_simd( pos, vel, mass, acc, n, n_pad );
      //else if (Method == ACCEL_VCL_SIMD_ROTATE)
      //   accel_vcl_simd_rotate( pos, vel, mass, acc, n, n_pad );
#endif

      TimerType t1 = getTimeStamp();

      /* 2. Advance the position and velocities. */
      update( pos, vel, mass, acc, n, dt, n_pad );

      TimerType t2 = getTimeStamp();

      /* 3. Find the faster moving object. */
      if (step % 10 == 0)
         search( pos, vel, mass, acc, n, n_pad );

      TimerType t3 = getTimeStamp();

      t_accel += getElapsedTime(t0,t1);
      t_update += getElapsedTime(t1,t2);
      t_search += getElapsedTime(t2,t3);

      /* 4. Write positions. */
      if (false && (step % 1 == 0))
      {
         for (int i = 0; i < n; ++i)
         {
            for (int k = 0; k < NDIM; ++k)
               fprintf(stderr,"%f ", pos_array(i,k));
            fprintf(stderr,"%f ", mass[i]);
         }
         fprintf(stderr,"\n");
         //output (pos, vel, mass, acc, n, flnum); flnum++;
      }
   }
   double t_calc = getElapsedTime( t_start, getTimeStamp());

   float nkbytes = (float)((size_t)7 * sizeof(ValueType) * (size_t)n) / 1024.0f;
   printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %f %f %f\n", t_calc*1000.0/num_steps, n, nkbytes, num_steps, t_accel*1000/num_steps, t_update*1000/num_steps, t_search*1000/num_steps);

   /*fclose(fp);*/

   /* Print out the positions (if not too large). */
   if (n < 50)
   {
      for (int i = 0; i < n; ++i)
      {
         for (int k = 0; k < NDIM; ++k)
            fprintf(stderr,"%f ", pos_array(i,k));
         for (int k = 0; k < NDIM; ++k)
            fprintf(stderr,"%f ", vel_array(i,k));

         fprintf(stderr,"%f\n", mass[i]);
      }
   }

   Deallocate(pos);
   Deallocate(vel);
   Deallocate(acc);
   Deallocate(mass);

   return 0;
}

template <typename ValueType>
int run_tests_1( const int n, const int num_steps, const ValueType dt, const int method)
{
   if (method == ACCEL_NAIVE)
      return run_tests<ValueType,ACCEL_NAIVE>( n, num_steps, dt );
   else if (method == ACCEL_STRENGTH)
      return run_tests<ValueType,ACCEL_STRENGTH>( n, num_steps, dt );
   else if (method == ACCEL_REGISTER)
      return run_tests<ValueType,ACCEL_REGISTER>( n, num_steps, dt );
   else if (method == ACCEL_INNER_SIMD)
      return run_tests<ValueType,ACCEL_INNER_SIMD>( n, num_steps, dt );
   else if (method == ACCEL_OUTER_SIMD)
      return run_tests<ValueType,ACCEL_OUTER_SIMD>( n, num_steps, dt );
#ifdef __ENABLE_VCL_SIMD
   else if (method == ACCEL_VCL_SIMD)
      return run_tests<ValueType,ACCEL_VCL_SIMD>( n, num_steps, dt );
   //else if (method == ACCEL_VCL_SIMD_ROTATE)
   //   return run_tests<ValueType,ACCEL_VCL_SIMD_ROTATE>( n, num_steps, dt );
#endif
   else
      return 1;
}

int main (int argc, char* argv[])
{
   /* Define the number of particles. The default is 100. */
   int n = 100;

   /* Define the number of steps to run. The default is 100. */
   int num_steps = 100;

   /* Pick the timestep size. */
   double dt = 0.01;

   /* ValueType? (float or double) */
   bool useDouble = true;

   int method = 2;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); help(argv[0]); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help(argv[0]);
         return 0;
      }
      else if (strcmp(argv[i],"--nparticles") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--nparticles|-n");
         i++;
         if (not(isdigit(*argv[i])))
            { fprintf(stderr,"Invalid value for option \"--particles\" %s\n", argv[i]); help(argv[0]); return 1; }
         n = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--nsteps") == 0 || strcmp(argv[i],"-s") == 0)
      {
         if (not(isdigit(*argv[i])))
            { fprintf(stderr,"Invalid value for option \"--stepsize\" %s\n", argv[i]); help(argv[0]); return 1; }
         dt = atof( argv[i] );
      }
      else if (strcmp(argv[i],"--method") == 0 || strcmp(argv[i],"-m") == 0)
      {
         check_index(i+1,"--method|-m");
         i++;
         if (not(isdigit(*argv[i])))
            { fprintf(stderr,"Invalid value for option \"method\" %s\n", argv[i]); help(argv[0]); return 1; }
         method = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--double") == 0 || strcmp(argv[i],"-d") == 0)
      {
         useDouble = true;
      }
      else if (strcmp(argv[i],"--float") == 0 || strcmp(argv[i],"-f") == 0)
      {
         useDouble = false;
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         help(argv[0]);
         return 1;
      }
   }

   methods.push_back( "accel_naive" );
   methods.push_back( "accel_strength" );
   methods.push_back( "accel_register" );
   methods.push_back( "accel_inner_simd" );
   methods.push_back( "accel_outer_simd" );
#ifdef __ENABLE_VCL_SIMD
   methods.push_back( "accel_vcl_simd" );
   //methods.push_back( "accel_vcl_simd_rotate" );
#endif

   if (method < 0 or method >= methods.size())
   {
      fprintf(stderr,"Error: method is out of range %d\n", method);
      help(argv[0]);
      return 1;
   }

   if (useDouble)
      return run_tests_1<double>( n, num_steps, dt, method );
   else
      return run_tests_1<float >( n, num_steps, dt, method );

   return 0;
}
