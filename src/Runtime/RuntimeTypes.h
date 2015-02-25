#ifndef __RUNTIMETYPES_H__
#define __RUNTIMETYPES_H__

typedef unsigned char		byte;		// 8 bits
typedef unsigned short		word;		// 16 bits
typedef unsigned int		dword;		// 32 bits
typedef unsigned int		uint;
typedef unsigned long		ulong;

typedef signed char			int8;
typedef unsigned char		uint8;
typedef short int			int16;
typedef unsigned short int	uint16;
typedef int					int32;
typedef unsigned int		uint32;
typedef long long			int64;
typedef unsigned long long	uint64;

#define MAX(x,y) ((x)<(y)?(x):(y))
#define MIN(x,y) ((x)>(y)?(x):(y))
#define MAX_TYPE(x) (((((x)1<<((sizeof(x)-1)*8-1))-1)<<8)|255)
#define MIN_TYPE(x) (-MAX_TYPE(x)-1)
#define MAX_UTYPE(x) (((((x)1U<<((sizeof(x)-1)*8))-1)<<8)|255U)
#define MIN_UTYPE(x) 0
#define LARGEST_INT64 (0xffffffff|(((int64)0x7fffffff)<<32))
#define SMALLEST_INT64 (((int64)-1) - LARGEST_INT64)

// If compiling for a processor that lacks floating point support, substitute integer for floating-point
#ifdef OMIT_FLOATING_POINT
#define double int64
#define float int64
#define double64 int64
#define BIG_DOUBLE (((int64)1)<<50)
#define OMIT_DATETIME_FUNCS 1
#define OMIT_TRACE 1
#undef MIXED_ENDIAN_64BIT_FLOAT
#undef HAVE_ISNAN
#else
#ifdef __CUDACC__
#define double64 double
#else
#define double64 long double
#endif
#define BIG_DOUBLE (1e99)
#endif

#if defined(__PTRDIFF_TYPE__)  // This case should work for GCC
# define INT_TO_PTR(X)  ((void*)(__PTRDIFF_TYPE__)(X))
# define PTR_TO_INT(X)  ((int)(__PTRDIFF_TYPE__)(X))
#elif !defined(__GNUC__)       // Works for compilers other than LLVM
# define INT_TO_PTR(X)  ((void*)&((char*)0)[X])
# define PTR_TO_INT(X)  ((int)(((char*)X)-(char*)0))
#elif defined(HAVE_STDINT_H)   // Use this case if we have ANSI headers
# define INT_TO_PTR(X)  ((void*)(intptr_t)(X))
# define PTR_TO_INT(X)  ((int)(intptr_t)(X))
#else                          // Generates a warning - but it always works
# define INT_TO_PTR(X)  ((void*)(X))
# define PTR_TO_INT(X)  ((int)(X))
#endif

#endif // __RUNTIMETYPES_H__