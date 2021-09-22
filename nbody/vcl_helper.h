#ifndef __vcl_helper_h
#define __vcl_helper_h

#ifndef MAX_VECTOR_SIZE
# define MAX_VECTOR_SIZE 256
#endif

#if !((MAX_VECTOR_SIZE == 128) || \
      (MAX_VECTOR_SIZE == 256) || \
      (MAX_VECTOR_SIZE == 512))
# error "MAX_VECTOR_SIZE is not 128/256/512"
#endif

#include <vcl/vectorclass.h>
#include <vcl/vectormath_exp.h>

template <typename T, int width> struct vcl_type;
//template <typename V> struct vcl_mask_type;

#if (MAX_VECTOR_SIZE >= 128)

template <> struct vcl_type<double, 2> { using type = Vec2d; };
template <> struct vcl_type<   int, 4> { using type = Vec4i; };
template <> struct vcl_type< float, 4> { using type = Vec4f; };
template <> struct vcl_type<  long, 2> { using type = Vec2q; };

#endif
#if (MAX_VECTOR_SIZE >= 256)

template <> struct vcl_type<double, 4> { using type = Vec4d; };
template <> struct vcl_type<   int, 8> { using type = Vec8i; };
template <> struct vcl_type< float, 8> { using type = Vec8f; };
template <> struct vcl_type<  long, 4> { using type = Vec4q; };

#endif
#if (MAX_VECTOR_SIZE >= 512)

template <> struct vcl_type<double,  8> { using type = Vec8d; };
template <> struct vcl_type<   int, 16> { using type = Vec16i; };
template <> struct vcl_type< float, 16> { using type = Vec16f; };
template <> struct vcl_type<  long,  8> { using type = Vec8q; };

#endif

template <typename V>
struct vcl_mask_type
{
   typedef decltype( V(0) < V(1) ) type;
};

template <typename SimdMask>
bool any( const SimdMask &x ) { return horizontal_or(x); }

template <typename SimdMask>
bool all( const SimdMask &x ) { return horizontal_and(x); }

template <typename SimdMask, typename SimdType>
SimdType where( const SimdMask &mask, const SimdType &a, const SimdType &b) {
   return select( mask, a, b );
}

template <typename T>
constexpr int SIMD_Vector_Length(const T) { return (MAX_VECTOR_SIZE / 8) / sizeof(T); }

template <typename T>
constexpr int SIMD_Vector_Length(void) { return (MAX_VECTOR_SIZE / 8) / sizeof(T); }

#endif
