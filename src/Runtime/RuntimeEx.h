#ifndef __RUNTIMEEX_H__
#define __RUNTIMEEX_H__
#include "Runtime.h"

//////////////////////
// FUNC
#pragma region FUNC

// panic
#define _panic(fmt, ...) printf(fmt, __VA_ARGS__)

// strtol, strtoul, strtod
#if __CUDACC__
__device__ unsigned long _strtol_(const char *str, char **endptr, register int base, bool signed_);
#define _strtol(str, endptr, base) (long)_strtol_(str, endptr, base, true)
#define _strtoul(str, endptr, base) (unsigned long)_strtol_(str, endptr, base, false)
__device__ double _strtod(const char *str, char **endptr);
#else
#define _strtol(str, endptr, base) strtol(str, endptr, base)
#define _strtoul(str, endptr, base) strtoul(str, endptr, base)
#define _strtod(str, endptr) strtod(str, endptr)
#endif

// strrchr
#if __CUDACC__
__device__ char *_strrchr(char *str, int ch);
#else
#define _strrchr(str, ch) strrchr(str, ch)
#endif

// qsort
#if __CUDACC__
__device__ void _qsort(void *base, size_t num, size_t size, int (*compar)(const void*,const void*));
#else
#define _qsort(base, num, size, compar) qsort(base, num, size, compar)
#endif

#pragma endregion

//////////////////////
// OS
#pragma region OS

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