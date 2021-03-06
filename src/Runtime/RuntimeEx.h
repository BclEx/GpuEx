#ifndef __RUNTIMEEX_H__
#define __RUNTIMEEX_H__
#ifndef __CUDACC__
#include <setjmp.h>
#endif
#include "Runtime.h"
#include <stdint.h>
//#include <float.h>
#define u_quad_t uint64_t
#define quad_t int64_t
#define QUAD_MAX INT64_MAX
#define QUAD_MIN INT64_MIN
#define UQUAD_MAX UINT64_MAX

//////////////////////
// FUNC
#pragma region FUNC

// setjmp/longjmp
#if __CUDACC__
struct __jmp_buf_tag {
	unsigned char used;
};
typedef struct __jmp_buf_tag jmp_buf[1];
__device__ int _setjmp(jmp_buf xxenv);
__device__ void _longjmp(jmp_buf yyenv, int zzval);
#else
#define _setjmp(a) setjmp(a)
#define _longjmp(a, b) longjmp(a, b)
#endif

// rand
#if __CUDACC__
__device__ int _rand();
#else
#define _rand() rand()
#endif

// time
#if __CUDACC__
__device__ time_t _time(time_t *timer);
#else
#define _time(a) time(a)
#endif

// gettimeofday
#if !defined(_WINSOCKAPI_)
struct timeval {
	long tv_sec;
	long tv_usec;
};
#endif
__device__ int _gettimeofday(struct timeval *tp, void *tz);

// sleep
#if __CUDACC__
__device__ void __sleep(unsigned long milliseconds);
#else
#define __sleep(n) _sleep(n)
#endif

// errno
#if __CUDACC__
extern __device__ int __errno;
__device__ char *__strerror(int errno_);
#else
#define __errno errno
#define __strerror(e) strerror(e)
#endif

// strtol, strtoul
#if __CUDACC__
__device__ unsigned long _strtol_(const char *str, char **endptr, register int base, bool signed_);
__device__ unsigned long long _strtoll_(const char *str, char **endptr, register int base, bool signed_);
__device__ u_quad_t _strtoq_(const char *str, char **endptr, register int base, bool signed_);
#define _strtol(str, endptr, base) (long)_strtol_(str, endptr, base, true)
#define _strtoul(str, endptr, base) (unsigned long)_strtol_(str, endptr, base, false)
#define _strtoll(str, endptr, base) (long long)_strtoll_(str, endptr, base, true)
#define _strtoull(str, endptr, base) (unsigned long long)_strtoll_(str, endptr, base, false)
#define _strtoq(str, endptr, base) (long)_strtoq_(str, endptr, base, true)
#define _strtouq(str, endptr, base) (unsigned long)_strtoq_(str, endptr, base, false)
#else
#define _strtol(str, endptr, base) strtol(str, endptr, base)
#define _strtoul(str, endptr, base) strtoul(str, endptr, base)
#define _strtoll(str, endptr, base) strtoll(str, endptr, base)
#define _strtoull(str, endptr, base) _strtoui64(str, endptr, base)
#define _strtoq(str, endptr) strtod(str, endptr)
#define _strtouq(str, endptr) strtod(str, endptr)
#endif

// strtod
#if __CUDACC__
#define _HUGE_VAL 1.7976931348623158e+308
__device__ double _strtod(const char *str, char **endptr);
#else
#define _HUGE_VAL HUGE_VAL
#define _strtod(str, endptr) strtod(str, endptr)
#endif

// strrchr
#if __CUDACC__
__device__ char *_strrchr(const char *str, int ch);
#else
#define _strrchr(str, ch) strrchr(str, ch)
#endif

// strpbrk
#if __CUDACC__
__device__ char *_strpbrk(register const char *s1, register const char *s2);
#else
#define _strpbrk(s1, s2) strpbrk(s1, s2)
#endif

// sscanf
#if __CUDACC__
__device__ int _sscanf_(const char *str, const char *fmt, _va_list &args);
__device__ __forceinline int _sscanf(const char *str, const char *fmt) { _va_list args; _va_start(args); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline  int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) { va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) { va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) { va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) { va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __device__ __forceinline int _sscanf(const char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) { va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14, arg15); int z = _sscanf_(str, fmt, args); _va_end(args); return z; }
#else
#define _sscanf(str, fmt, ...) sscanf(str, fmt, __VA_ARGS__)
#endif

// qsort
#if __CUDACC__
__device__ void _qsort(void *base, size_t num, size_t size, int (*compar)(const void*,const void*));
#else
#define _qsort(base, num, size, compar) qsort(base, num, size, compar)
#endif

// div
#if __CUDACC__
__device__ div_t _div(int num, int denom);
#else
#define _div(num, denom) div(num, denom)
#endif

#if __CUDACC__
#ifdef __cplusplus
extern "C++" {
	__device__ __forceinline char *_strpbrk(char *src, const char *c) { return (char *)_strpbrk((const char*)src, c); }
	__device__ __forceinline char *_strrchr(char *src, int ch) { return (char *)_strrchr((const char *)src, ch); }
}
#endif
#endif

#pragma endregion

//////////////////////
// OS
#pragma region OS

#include <sys/stat.h>
//#define WIN32_LEAN_AND_MEAN
//#include <windows.h>
//#undef WIN32_LEAN_AND_MEAN

#if __CUDACC__
typedef int mode_t;
__device__ int __chmod(const char *a, mode_t m);
__device__ int __rmdir(const char *a);
__device__ int __mkdir(const char *a, mode_t m = 0755);
__device__ int __mkfifo(const char *a, mode_t m);
__device__ int __stat(const char *a, struct stat *b);
__device__ char *__getcwd(char *b, int l);
__device__ int __chdir(const char *p);
__device__ int __access(const char *p, int flags);
#else
#define __chmod(a, m) chmod(a, m)
#define __rmdir(a) rmdir(a)
#define __mkdir(a, m) mkdir(a, m)
#define __mkfifo(a, m) mkfifo(a, m)
#define __stat(a, b) stat(a, b)
#define __getcwd(b, l) getcwd(b, l)
#define __chdir(p) chdir(p)
#define __access(p, f) access(p, f)
#endif

#if OS_GPU
extern __device__ char *__environ[];
__device__ char *_getenv(const char *name);
__device__ void _setenv(const char *name, const char *value);
#define _unsetenv(name) _setenv(name, "")
#elif OS_WIN
#define __environ environ
#define _getenv(name) getenv(name)
#define _setenv(name, value, overwrite) _putenv_s(name, value)
#define _unsetenv(name) _putenv_s(name, "")
#else OS_UNIX
#define __environ environ
#define _getenv(name) getenv(name)
#define _setenv(name, value, overwrite) setenv(name, value, overwrite)
#define _unsetenv(name) unsetenv(name)
#endif

#if OS_UNIX
#else
typedef struct DIR DIR;
struct _dirent
{
	char *d_name;
};
__device__ DIR *_opendir(const char *);
__device__ int _closedir(DIR *);
__device__ struct _dirent *_readdir(DIR *);
__device__ void _rewinddir(DIR *);
#endif

//#ifndef STDIN_FILENO
//# define STDIN_FILENO 0
//#endif
//
//#ifndef STDOUT_FILENO
//# define STDOUT_FILENO 1
//#endif
//
//#ifndef STDERR_FILENO
//# define STDERR_FILENO 2
//#endif

#if !defined SEEK_SET
# define SEEK_SET 0
# define SEEK_CUR 1
# define SEEK_END 2
#endif
#ifndef F_OK
# define F_OK 0
# define X_OK 1
# define W_OK 2
# define R_OK 4
#endif

// Define macros to query file type bits, if they're not already defined.
#ifndef S_ISREG
# ifdef S_IFREG
# define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
# else
# define S_ISREG(m) 0
# endif
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