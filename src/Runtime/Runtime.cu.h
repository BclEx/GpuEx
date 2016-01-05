///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

#ifndef __RUNTIME_CU_H__
#define __RUNTIME_CU_H__

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif

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
#define __HEAP_ALIGNSIZE sizeof(long long)
#define __HEAP_HEADER_RAW 0
#define __HEAP_HEADER_PRINTF 1
#define __HEAP_HEADER_TRANSFER 2
#define __HEAP_HEADER_ASSERT 4
#define __HEAP_HEADER_THROW 5

#if defined(__EMBED__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#ifndef __EMBED__
#define __static__ static
#endif
#include "Runtime.cu+native.h"
#else
extern "C" __device__ char *__heap_movenext(char *&end, char *&bufptr); 
extern "C" __device__ void __heap_writeheader(unsigned short type, char *ptr, char *fmtptr);
extern "C" __device__ char *__heap_write(char *dest, const char *src, int maxLength, char *end);
extern "C" __device__ void _runtimeSetHeap(void *heap);
extern "C" __device__ void runtimeRestrict(int threadid, int blockid);
// embed
extern __host_constant__ unsigned char __curtUpperToLower[256];
extern __host_constant__ unsigned char __curtCtypeMap[256]; 
#endif

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

//#undef _va_start
//#undef _va_arg
//#undef _va_end
#define COMMA ,
//__device__ void name##_(__VA_ARGS__, _va_list &args);
#define STDARGvoid(name, paramN, params1, ...) \
	__device__ __forceinline void name(__VA_ARGS__) { _va_list args; _va_start(args); name##_(params1, args); _va_end(args); } \
	template <typename T1> __device__ __forceinline void name(__VA_ARGS__, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); name##_(params1, args); _va_end(args); }
#define STDARG(ret, name, paramN, params1, ...) \
	__device__ __forceinline void name(__VA_ARGS__) { _va_list args; _va_start(args); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1> __device__ __forceinline void name(__VA_ARGS__, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); ret r = name##_(params1, args); _va_end(args); return r; }
#define STDARGEXvoid(name, paramN, params1, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); name##_(params1, args); _va_end(args); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); name##_(params1, args); _va_end(args); }
#define STDARGEX(ret, name, paramN, params1, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); ret r = name##_(params1, args); _va_end(args); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); ret r = name##_(params1, args); _va_end(args); return r; }

struct va_list0 { char *b; char *i; };
template <typename T1> struct va_list1 : va_list0 { T1 v1; };
template <typename T1, typename T2> struct va_list2 : va_list0 { T1 v1; T2 v2; };
template <typename T1, typename T2, typename T3> struct va_list3 : va_list0 { T1 v1; T2 v2; T3 v3; };
template <typename T1, typename T2, typename T3, typename T4> struct va_list4 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; };
template <typename T1, typename T2, typename T3, typename T4, typename T5> struct va_list5 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> struct va_list6 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> struct va_list7 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> struct va_list8 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> struct va_list9 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> struct va_listA : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; };
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> struct va_listB : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> struct va_listC : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> struct va_listD : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> struct va_listE : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> struct va_listF : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; };
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> struct va_list11 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> struct va_list12 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> struct va_list13 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> struct va_list14 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> struct va_list15 : va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; T15 v15; };

#ifndef _INTSIZEOF
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
#endif
#define _va_list va_list0 
#define _va_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT(t, (unsigned long long)(ap.i + _INTSIZEOF(t)))) - _INTSIZEOF(t)))
#define _va_end(ap) (ap.i = nullptr);

//#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
//#define _va_start(ap, v, ...) (ap = (_va_list)_ADDRESSOF(v) + _INTSIZEOF(v))
//#define _va_arg(ap, t) (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
//#define _va_end(ap) (ap = (_va_list)0)

__device__ __forceinline static void _va_restart(_va_list &args)
{
	args.i = args.b;
}
__device__ __forceinline static void _va_start(_va_list &args)
{
	args.b = args.i = nullptr;
}
template <typename T1> __device__ __forceinline static void _va_start(va_list1<T1> &args, T1 arg1)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1;
}
template <typename T1, typename T2> __device__ __forceinline static void _va_start(va_list2<T1,T2> &args, T1 arg1, T2 arg2)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _va_start(va_list3<T1,T2,T3> &args, T1 arg1, T2 arg2, T3 arg3)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _va_start(va_list4<T1,T2,T3,T4> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void _va_start(va_list5<T1,T2,T3,T4,T5> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void _va_start(va_list6<T1,T2,T3,T4,T5,T6> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void _va_start(va_list7<T1,T2,T3,T4,T5,T6,T7> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void _va_start(va_list8<T1,T2,T3,T4,T5,T6,T7,T8> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void _va_start(va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void _va_start(va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static void _va_start(va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static void _va_start(va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static void _va_start(va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static void _va_start(va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static void _va_start(va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF;
}
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __device__ __forceinline static void _va_start(va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __device__ __forceinline static void _va_start(va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __device__ __forceinline static void _va_start(va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __device__ __forceinline static void _va_start(va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13; args.v14 = arg14;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __device__ __forceinline static void _va_start(va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15)
{
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13; args.v14 = arg14; args.v15 = arg15;
}

#pragma endregion

//////////////////////
// PRINTF
#pragma region PRINTF

#define _printf printf
#define _putchar(c) printf("%c", c)
#if 0
#define PRINTF_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return 0;
#define PRINTF_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define PRINTF_POSTAMBLE \
	fmtstart = bufptr; end = __heap_write(bufptr, fmt, 0, end); \
	__heap_writeheader(__HEAP_HEADER_PRINTF, start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ __forceinline static int _printf(const char *fmt)
{
	PRINTF_PREAMBLE;
	PRINTF_POSTAMBLE;
}
template <typename T1> __device__ __forceinline static int _printf(const char *fmt, T1 arg1)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA);
	PRINTF_POSTAMBLE;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD); PRINTF_ARG(argE);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA); PRINTF_ARG(argB); PRINTF_ARG(argC); PRINTF_ARG(argD); PRINTF_ARG(argE); PRINTF_ARG(argF);
	PRINTF_POSTAMBLE;
}

#undef PRINTF_PREAMBLE
#undef PRINTF_ARG
#undef PRINTF_POSTAMBLE
#endif

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

__device__ __forceinline static int _transfer(const char *fmt)
{
	TRANSFER_PREAMBLE;
	TRANSFER_POSTAMBLE;
}
template <typename T1> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8); TRANSFER_ARG(arg9);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
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

__device__ __forceinline static void _throw(const char *fmt)
{
	THROW_PREAMBLE;
	THROW_POSTAMBLE;
}
template <typename T1> __device__ __forceinline static void _throw(const char *fmt, T1 arg1)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2> __device__ __forceinline static void _throw(const char *fmt, T1 arg1, T2 arg2)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2); THROW_ARG(arg3);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
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
