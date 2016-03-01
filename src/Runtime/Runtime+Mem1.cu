#include "Runtime.h"
#include "RuntimeTypes.h"
#define RUNTIME_ALLOC_SYSTEM
RUNTIME_NAMEBEGIN

	// This file contains low-level memory allocation drivers for when SQLite will use the standard C-library malloc/realloc/free interface
	// to obtain the memory it needs.
	//
	// This file contains implementations of the low-level memory allocation routines specified in the sqlite3_mem_methods object.  The content of
	// this file is only used if SQLITE_SYSTEM_MALLOC is defined.  The SQLITE_SYSTEM_MALLOC macro is defined automatically if neither the
	// SQLITE_MEMDEBUG nor the SQLITE_WIN32_MALLOC macros are defined.  The default configuration is to use memory allocation routines in this file.
	//
	// C-preprocessor macro summary:
	//
	//    HAVE_MALLOC_USABLE_SIZE     The configure script sets this symbol if the malloc_usable_size() interface exists
	//                                on the target platform.  Or, this symbol can be set manually, if desired.
	//                                If an equivalent interface exists by a different name, using a separate -D option to rename it.
	//
	//    RUNTIME_WITHOUT_ZONEMALLOC   Some older macs lack support for the zone memory allocator.  Set this symbol to enable building on older macs.
	//
	//    RUNTIME_WITHOUT_MSIZE        Set this symbol to disable the use of _msize() on windows systems.  This might be necessary when compiling for Delphi, for example.

#ifdef RUNTIME_ALLOC_SYSTEM

	// The MSVCRT has malloc_usable_size() but it is called _msize(). The use of _msize() is automatic, but can be disabled by compiling with -DSQLITE_WITHOUT_MSIZE
#if defined(_MSC_VER) && !defined(RUNTIME_WITHOUT_MSIZE) && !defined(__CUDACC__)
#define RUNTIME_ALLOCSIZE _msize
#endif

#if __CUDACC__

	// Use standard C library malloc and free on non-Apple systems. Also used by Apple systems if SQLITE_WITHOUT_ZONEMALLOC is defined.
#define RUNTIME_ALLOC(x) malloc(x)
#define RUNTIME_FREE(x) free(x)
#define RUNTIME_REALLOC(x,y) _cudarealloc((x),(y))

	__device__ inline static void *_cudarealloc(void *old, size_t newSize)
{ 
	void *new_ = malloc(newSize + 8);
	if (old)
	{ 
		int64 *p = (int64 *)old;
		size_t oldSize = (size_t)p[0];
		if (oldSize != 0) _memcpy(new_, old, oldSize + 8);
		free(old);
	}
	return new_;
}

#elif defined(__APPLE__) && !defined(RUNTIME_WITHOUT_ZONEMALLOC)

	// Use the zone allocator available on apple products unless the SQLITE_WITHOUT_ZONEMALLOC symbol is defined.
#include <sys/sysctl.h>
#include <malloc/malloc.h>
#include <libkern/OSAtomic.h>
	static malloc_zone_t * _zoneMalloc;
#define RUNTIME_ALLOC(x) malloc_zone_malloc(_zoneMalloc,(x))
#define RUNTIME_FREE(x) malloc_zone_free(_zoneMalloc,(x));
#define RUNTIME_REALLOC(x,y) malloc_zone_realloc(_zoneMalloc,(x),(y))
#define RUNTIME_ALLOCSIZE(x) (_zoneMalloc ? _zoneMalloc->size(_sqliteZone_,x) : malloc_size(x))

#else /* if not __APPLE__ */

	// Use standard C library malloc and free on non-Apple systems. Also used by Apple systems if SQLITE_WITHOUT_ZONEMALLOC is defined.
#define RUNTIME_ALLOC(x) malloc(x)
#define RUNTIME_FREE(x) free(x)
#define RUNTIME_REALLOC(x,y) realloc((x),(y))

#if (defined(_MSC_VER) && !defined(RUNTIME_WITHOUT_MSIZE)) || (defined(HAVE_MALLOC_H) && defined(HAVE_MALLOC_USABLE_SIZE))
#include <malloc.h> // Needed for malloc_usable_size on linux
#endif

	//#ifdef HAVE_MALLOC_USABLE_SIZE
	//#ifndef RUNTIME_ALLOCSIZE
	//#define RUNTIME_ALLOCSIZE(x) malloc_usable_size(x)
	//#endif
	//#else
	//#undef RUNTIME_ALLOCSIZE
	//#endif

#endif /* __APPLE__ or not __APPLE__ */

	// Like malloc(), but remember the size of the allocation so that we can find it later using sqlite3MemSize().
	//
	// For this low-level routine, we are guaranteed that nByte>0 because cases of nByte<=0 will be intercepted and dealt with by higher level routines.
	__device__ void *MemAlloc(size_t size)
{
#ifdef RUNTIME_ALLOCSIZE
	void *p = RUNTIME_ALLOC(size);
	if (!p)
	{
		//ASSERTCOVERAGE(sqlite3GlobalConfig.xLog!=0);
		_printf("failed to allocate %u bytes of memory", size);
	}
	return p;
#else
	_assert(size > 0);
	size = _ROUND8(size);
	int64 *p = (int64 *)RUNTIME_ALLOC(size + 8);
	if (p)
	{
		p[0] = size;
		p++;
	}
	else
	{
		//ASSERTCOVERAGE(sqlite3GlobalConfig.xLog!=0);
		_printf("failed to allocate %u bytes of memory", size);
	}
	return (void *)p;
#endif
}

// Like free() but works for allocations obtained from sqlite3MemMalloc() or sqlite3MemRealloc().
//
// For this low-level routine, we already know that pPrior!=0 since cases where pPrior==0 will have been intecepted and dealt with
// by higher-level routines.
__device__ void MemFree(void *prior)
{
#ifdef RUNTIME_ALLOCSIZE
	RUNTIME_FREE(prior);
#else
	_assert(prior);
	int64 *p = (int64 *)prior;
	p--;
	RUNTIME_FREE(p);
#endif
}

// Report the allocated size of a prior return from xMalloc() or xRealloc().
__device__ size_t MemSize(void *prior)
{
#ifdef RUNTIME_ALLOCSIZE
	return (prior ? (int)RUNTIME_ALLOCSIZE(prior) : 0);
#else
	if (!prior) return 0;
	int64 *p = (int64 *)prior;
	p--;
	return (size_t)p[0];
#endif
}

// Like realloc().  Resize an allocation previously obtained from sqlite3MemMalloc().
//
// For this low-level interface, we know that pPrior!=0.  Cases where pPrior==0 while have been intercepted by higher-level routine and
// redirected to xMalloc.  Similarly, we know that nByte>0 becauses cases where nByte<=0 will have been intercepted by higher-level
// routines and redirected to xFree.
__device__ void *MemRealloc(void *prior, size_t size)
{
#ifdef RUNTIME_ALLOCSIZE
	void *p = RUNTIME_REALLOC(prior, size);
	if (!p)
	{
		//ASSERTCOVERAGE(sqlite3GlobalConfig.xLog != 0);
		_printf("failed memory resize %u to %u bytes", RUNTIME_ALLOCSIZE(prior), size);
	}
	return p;
#else
	_assert(prior && size > 0);
	_assert(size == _ROUND8(size)); /* EV: R-46199-30249 */
	int64 *p = (int64 *)prior;
	p--;
	p = (int64 *)RUNTIME_REALLOC(p, size);
	if (p)
	{
		p[0] = size;
		p++;
	}
	else
	{
		//ASSERTCOVERAGE(sqlite3GlobalConfig.xLog != 0);
		_printf("failed memory resize %u to %u bytes", MemSize(prior), size);
	}
	return (void *)p;
#endif
}

// Round up a request size to the next valid allocation size.
__device__ size_t MemRoundup(size_t size) { return _ROUND8(size); }

// Initialize this module.
__device__ int MemInit(void *notUsed1)
{
#if defined(__APPLE__) && !defined(RUNTIME_WITHOUT_ZONEMALLOC)
	if (_zoneMalloc)
		return 0;
	int cpuCount;
	size_t len = sizeof(cpuCount);
	// One usually wants to use hw.acctivecpu for MT decisions, but not here
	sysctlbyname("hw.ncpu", &cpuCount, &len, NULL, 0);
	if (cpuCount > 1)
		_zoneMalloc = malloc_default_zone(); // defer MT decisions to system malloc
	else
	{
		// only 1 core, use our own zone to contention over global locks, e.g. we have our own dedicated locks
		malloc_zone_t *newzone = malloc_create_zone(4096, 0);
		malloc_set_zone_name(newzone, "_heap");
		bool success;
		do { success = OSAtomicCompareAndSwapPtrBarrier(NULL, newzone, (void *volatile*)&_zoneMalloc); }
		while (!_zoneMalloc);
		if (!success)
			malloc_destroy_zone(newzone); // somebody registered a zone first
	}
#endif
	UNUSED_PARAMETER(notUsed1);
	return 0;
}

// Deinitialize this module.
__device__ void MemShutdown(void *notUsed1)
{
	UNUSED_PARAMETER(notUsed1);
	return;
}

// This routine is the only routine in this file with external linkage.
// Populate the low-level memory allocation function pointers in sqlite3GlobalConfig.m with pointers to the routines in this file.
__constant__ const _mem_methods _mem1DefaultMethods = {
	MemAlloc,
	MemFree,
	MemRealloc,
	MemSize,
	MemRoundup,
	MemInit,
	MemShutdown,
	nullptr
};

__device__ void __allocsystem_setdefault()
{
	__allocsystem = _mem1DefaultMethods;
}

#endif /* RUNTIME_ALLOC_SYSTEM */

RUNTIME_NAMEEND