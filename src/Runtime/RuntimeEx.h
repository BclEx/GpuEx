#ifndef __RUNTIMEEX_H__
#define __RUNTIMEEX_H__
#include "Runtime.h"

#include <stdint.h>
#define u_quad_t uint64_t
#define quad_t int64_t
#define QUAD_MAX INT64_MAX
#define QUAD_MIN INT64_MIN
#define UQUAD_MAX UINT64_MAX

//////////////////////
// FUNC
#pragma region FUNC

// sleep
#if __CUDACC__
__device__ void __sleep(unsigned long milliseconds);
#else
#define __sleep(n) _sleep(n);
#endif

// errno
#if __CUDACC__
extern __device__ int __errno;
#else
#define __errno errno
#endif

// strtol, strtoul, strtod
#if __CUDACC__
__device__ unsigned long _strtol_(const char *str, char **endptr, register int base, bool signed_);
#define _strtol(str, endptr, base) (long)_strtol_(str, endptr, base, true)
#define _strtoul(str, endptr, base) (unsigned long)_strtol_(str, endptr, base, false)
__device__ double _strtod(const char *str, char **endptr);
__device__ u_quad_t _strtoq_(const char *str, char **endptr, register int base, bool signed_);
#define _strtoq(str, endptr, base) (long)_strtoq_(str, endptr, base, true)
#define _strtouq(str, endptr, base) (unsigned long)_strtoq_(str, endptr, base, false)
#else
#define _strtol(str, endptr, base) strtol(str, endptr, base)
#define _strtoul(str, endptr, base) strtoul(str, endptr, base)
#define _strtod(str, endptr) strtod(str, endptr)
#define _strtoq(str, endptr) strtod(str, endptr)
#define _strtouq(str, endptr) strtod(str, endptr)
#endif

// strrchr
#if __CUDACC__
__device__ char *_strrchr(char *str, int ch);
#else
#define _strrchr(str, ch) strrchr(str, ch)
#endif

// sscanf
#if __CUDACC__
__device__ int __sscanf(const char *str, const char *fmt, _va_list *args);
__device__ __forceinline static int _sscanf(const char *str, const char *fmt) { _va_list args; _va_start(args); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); int z = __sscanf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) { va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) { va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) { va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) { va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __device__ __forceinline static int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) { va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14, arg15); int z = __sscanf(str, fmt, &args); _va_end(args); return z; }
#else
#define _sscanf(str, fmt, ...) sscanf(str, fmt, __VA_ARGS__ )
#endif

// qsort
#if __CUDACC__
//__device__ void _qsort(void *base, size_t num, size_t size, int (*compar)(const void*,const void*));
__device__ void _qsort(char *base, size_t num, size_t size, int (*compar)(const void*,const void*));
#else
#define _qsort(base, num, size, compar) qsort(base, num, size, compar)
#endif

#pragma endregion

//////////////////////
// OS
#pragma region OS

#if __CUDACC__
typedef int mode_t;
__device__ int _chmod(const char *, mode_t);
__device__ int _mkdir(const char *, mode_t);
__device__ int _mkfifo(const char *, mode_t);
__device__ int __stat(const char *, struct stat *);
#else
#define _chmod(a, m) chmod(a, m)
#define _mkdir(a, m) mkdir(a, m)
#define _mkfifo(a, m) mkfifo(a, m)
#define __stat(a, b) stat(a, b)
#endif

#if OS_GPU
__device__ char *_getenv(const char *name);
#else
#define _getenv(name) getenv(name)
#endif

#if OS_UNIX
#else
typedef struct DIR DIR;
struct dirent
{
	char *d_name;
};
__device__ DIR *_opendir(const char *);
__device__ int _closedir(DIR *);
__device__ struct dirent *_readdir(DIR *);
__device__ void _rewinddir(DIR *);
#endif

#ifndef S_ISDIR
# ifdef S_IFDIR
# define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
# else
# define S_ISDIR(m) 0
# endif
#endif

#pragma endregion

#endif // __RUNTIMEEX_H__