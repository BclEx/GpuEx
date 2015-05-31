///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

#ifndef __RUNTIME_CPU_H__
#define __RUNTIME_CPU_H__

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_ISNAN
#include <math.h>
#endif
//
#ifndef __global__
#define __host__
#define __global__
#define __device__
#define __constant__
#define __shared__
#define __inline__ __inline
#define __restrict__ __restrict
#define cudaError_t int
#endif

//////////////////////
// ASSERT
#pragma region ASSERT

#include <assert.h>
#undef _assert
#ifndef NDEBUG
#define _assert(X) assert(X)
#define ASSERTONLY(X) X
__device__ __forceinline void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif
#define _ALWAYS(X) (X)
#define _NEVER(X) (X)

#pragma endregion

//////////////////////
// HEAP
#pragma region HEAP

#define CURT_UNRESTRICTED -1

extern "C" __device__ inline static void _runtimeSetHeap(void *heap) { }
extern "C" cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host);
extern "C" __device__ inline static void runtimeRestrict(int threadid, int blockid) { }
// embed
extern "C" const unsigned char __curtUpperToLower[256];
extern "C" const unsigned char __curtCtypeMap[256];

#ifdef __EMBED__
cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host) { return 0; }
#endif

#pragma endregion

//////////////////////
// STDARG
#pragma region STDARG

#include <stdarg.h>
#define _va_list va_list
#define _va_start va_start
#define _va_arg va_arg
#define _va_end va_end

#pragma endregion

//////////////////////
// PRINTF
#pragma region PRINTF

#define _printf printf
#if 0
__device__ __forceinline  int _printf(const char *fmt) { return printf(fmt); }
template <typename T1> __device__ __forceinline  int _printf(const char *fmt, T1 arg1) { return printf(fmt, arg1); }
template <typename T1, typename T2> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2) { return printf(fmt, arg1, arg2); }
template <typename T1, typename T2, typename T3> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return printf(fmt, arg1, arg2, arg3); }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return printf(fmt, arg1, arg2, arg3, arg4); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return printf(fmt, arg1, arg2, arg3, arg4, arg5); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline  int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); }
#endif

#pragma endregion

//////////////////////
// TRANSFER
#pragma region TRANSFER

__device__ __forceinline  int _transfer(const char *fmt) { return 0; }
template <typename T1> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1) { return 0; }
template <typename T1, typename T2> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2) { return 0; }
template <typename T1, typename T2, typename T3> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return 0; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline  int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return 0; }

#pragma endregion

//////////////////////
// THROW
#pragma region THROW

__device__ __forceinline  void _throw(const char *fmt) { printf(fmt); exit(0); }
template <typename T1> __device__ __forceinline  void _throw(const char *fmt, T1 arg1) { printf(fmt, arg1); exit(0); }
template <typename T1, typename T2> __device__ __forceinline  void _throw(const char *fmt, T1 arg1, T2 arg2) { printf(fmt, arg1, arg2); exit(0); }
template <typename T1, typename T2, typename T3> __device__ __forceinline  void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { printf(fmt, arg1, arg2, arg3); exit(0); }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline  void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { printf(fmt, arg1, arg2, arg3, arg4); exit(0); }

#pragma endregion

#endif // __RUNTIME_CPU_H__
