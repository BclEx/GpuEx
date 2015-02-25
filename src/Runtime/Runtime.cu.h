#ifndef __RUNTIME_CU_H__
#define __RUNTIME_CU_H__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#ifdef HAVE_ISNAN
#include <math.h>
#endif
#include <assert.h>
#include "RuntimeHost.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

//////////////////////
// ASSERT
#pragma region ASSERT

#undef _assert
#ifndef NDEBUG
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
extern "C" __device__ static void __assertWrite(const char *message, const char *file, unsigned int line);
#undef assert
#define assert(X) _assert(X)
#else
extern "C" __device__ void __assertWrite(const char *message, const char *file, unsigned int line);
#endif
//
#define _assert(X) (void)((!!(X))||(__assertWrite(#X, __FILE__, __LINE__), 0))
#define ASSERTONLY(X) X
__device__ __forceinline__ void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif
#define _ALWAYS(X) (X)
#define _NEVER(X) (X)

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define CURT_UNRESTRICTED -1
#define __HEAP_ALIGNSIZE sizeof(long long)
#define __HEAP_HEADER_RAW 0
#define __HEAP_HEADER_PRINTF 1
#define __HEAP_HEADER_TRANSFER 2
#define __HEAP_HEADER_ASSERT 4
#define __HEAP_HEADER_THROW 5

#pragma endregion

#if defined(__EMBED__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#ifndef __EMBED__
#define __static__ static
#endif
#include "Runtime.cu+native.h"
#else

// Heap
extern "C" __device__ char *__heap_movenext(char *&end, char *&bufptr);
extern "C" __device__ void __heap_writeheader(unsigned short type, char *ptr, char *fmtptr);
extern "C" __device__ char *__heap_write(char *dest, const char *src, int maxLength, char *end);
extern "C" __device__ void _runtimeSetHeap(void *heap);
extern "C" __device__ void runtimeRestrict(int threadid, int blockid);

// Embed
extern __constant__ unsigned char __curtUpperToLower[256];
extern __constant__ unsigned char __curtCtypeMap[256];

#endif // __CUDA_ARCH__

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

__device__ static char *__copyArg(char *ptr, const char *arg, char *end)
{
	// initialization check
	if (!ptr) // || !arg)
		return nullptr;
	// strncpy does all our work. We just terminate.
	if ((ptr = __heap_write(ptr, arg, 0, end)) != nullptr)
		*ptr = 0;
	return ptr;
}

template <typename T> __device__ static char *__copyArg(char *ptr, T &arg, char *end)
{
	// initialization and overflow check. Alignment rules mean that we're at least RUNTIME_ALIGNSIZE away from "end", so we only need to check that one offset.
	if (!ptr || (ptr + __HEAP_ALIGNSIZE) >= end)
		return nullptr;
	// write the length and argument
	*(int *)(void *)ptr = sizeof(arg);
	ptr += __HEAP_ALIGNSIZE;
	*(T *)(void *)ptr = arg;
	ptr += __HEAP_ALIGNSIZE;
	*ptr = 0;
	return ptr;
}

#pragma endregion

//////////////////////
// STDARG
#pragma region STDARG

#undef va_start
#undef va_arg
#undef va_end

typedef char *va_list;
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
//#define va_start(ap, v)  (ap = (va_list)_ADDRESSOF(v) + _INTSIZEOF(v))
#define va_arg(ap, t) (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
#define va_end(ap) (ap = (va_list)0)

#define STDARG_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return;
#define STDARG_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define STDARG_POSTAMBLE \
	fmtstart = bufptr; \
	__heap_writeheader(__HEAP_HEADER_RAW, start, (end ? fmtstart : nullptr)); \
	args = (end ? (va_list)(end - start) : 0);

__device__ static void va_start(va_list &args)
{
	STDARG_PREAMBLE;
	STDARG_POSTAMBLE;
}
template <typename T1> __device__ static void va_start(va_list &args, T1 arg1)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA);
	STDARG_POSTAMBLE;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA); STDARG_ARG(argB);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA); STDARG_ARG(argB); STDARG_ARG(argC);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA); STDARG_ARG(argB); STDARG_ARG(argC); STDARG_ARG(argD);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA); STDARG_ARG(argB); STDARG_ARG(argC); STDARG_ARG(argD); STDARG_ARG(argE);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA); STDARG_ARG(argB); STDARG_ARG(argC); STDARG_ARG(argD); STDARG_ARG(argE); STDARG_ARG(argF);
	STDARG_POSTAMBLE;
}

#undef STDARG_PREAMBLE
#undef STDARG_ARG
#undef STDARG_POSTAMBLE

#pragma endregion

//////////////////////
// PRINTF
#pragma region PRINTF

#define PRINTF_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return 0;
#define PRINTF_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define PRINTF_POSTAMBLE \
	fmtstart = bufptr; end = __heap_write(bufptr, fmt, 0, end); \
	__heap_writeheader(__HEAP_HEADER_PRINTF, start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ static int _printf(const char *fmt)
{
	PRINTF_PREAMBLE;
	PRINTF_POSTAMBLE;
}
template <typename T1> __device__ static int _printf(const char *fmt, T1 arg1)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA);
	PRINTF_POSTAMBLE;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD); PRINTF_ARG(argE);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD); PRINTF_ARG(argE); PRINTF_ARG(argF);
	PRINTF_POSTAMBLE;
}

#undef PRINTF_PREAMBLE
#undef PRINTF_ARG
#undef PRINTF_POSTAMBLE

#pragma endregion

//////////////////////
// TRANSFER
#pragma region TRANSFER

#define TRANSFER_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return 0;
#define TRANSFER_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define TRANSFER_POSTAMBLE \
	fmtstart = bufptr; end = __heap_write(bufptr, fmt, 0, end); \
	__heap_writeheader(__HEAP_HEADER_TRANSFER, start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ static int _transfer(const char *fmt)
{
	TRANSFER_PREAMBLE;
	TRANSFER_POSTAMBLE;
}
template <typename T1> __device__ static int _transfer(const char *fmt, T1 arg1)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8); TRANSFER_ARG(arg9);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8); TRANSFER_ARG(arg9); TRANSFER_ARG(argA);
	TRANSFER_POSTAMBLE;
}

#undef TRANSFER_PREAMBLE
#undef TRANSFER_ARG
#undef TRANSFER_POSTAMBLE

#pragma endregion

//////////////////////
// THROW
#pragma region THROW

#define THROW_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return;
#define THROW_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define THROW_POSTAMBLE \
	fmtstart = bufptr; end = __heap_write(bufptr, fmt, 0, end); \
	__heap_writeheader(__HEAP_HEADER_THROW, start, (end ? fmtstart : nullptr)); \
	__THROW;

__device__ static void _throw(const char *fmt)
{
	THROW_PREAMBLE;
	THROW_POSTAMBLE;
}
template <typename T1> __device__ static void _throw(const char *fmt, T1 arg1)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2); THROW_ARG(arg3);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2); THROW_ARG(arg3); THROW_ARG(arg4);
	THROW_POSTAMBLE;
}

#undef THROW_PREAMBLE
#undef THROW_ARG
#undef THROW_POSTAMBLE

#pragma endregion

#endif // __RUNTIME_CU_H__
