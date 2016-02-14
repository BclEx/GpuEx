#include "Runtime.h"
RUNTIME_NAMEBEGIN

#ifdef RUNTIME_ALLOC_ZERO
	__device__ static void *MemAlloc(size_t size) { return 0; }
__device__ static void MemFree(void *prior) { }
__device__ static void *MemRealloc(void *prior, size_t size) { return 0; }
__device__ static size_t MemSize(void *prior ){ return 0; }
__device__ static size_t MemRoundup(size_t size) { return n; }
__device__ static int MemInit(void *notUsed1) { return 1; }
__device__ static void MemShutdown(void *notUsed1) { }

// This routine is the only routine in this file with external linkage.
//
// Populate the low-level memory allocation function pointers in sqlite3GlobalConfig.m with pointers to the routines in this file.
__constant__ static const _mem_methods _defaultMethods = {
	MemMalloc,
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
	__allocsystem = _defaultMethods;
}
#endif /* RUNTIME_ALLOC_ZERO */

RUNTIME_NAMEEND