#include "Runtime.h"

#ifndef OMIT_BUILTIN_TEST

typedef struct BenignMallocHooks BenignMallocHooks;
__device__ static _WSD struct BenignMallocHooks
{
	void (*BenignBegin)();
	void (*BenignEnd)();
} BenignMallocHooks_ = { nullptr, nullptr };

#ifdef OMIT_WSD
#define g_BenignMallocHooksInit BenignMallocHooks *x = &GLOBAL(BenignMallocHooks, BenignMallocHooks_)
#define g_BenignMallocHooks x[0]
#else
#define g_BenignMallocHooksInit
#define g_BenignMallocHooks BenignMallocHooks_
#endif

__device__ void _benignalloc_hook(void (*benignBegin)(), void (*benignEnd)())
{
	g_BenignMallocHooksInit;
	g_BenignMallocHooks.BenignBegin = benignBegin;
	g_BenignMallocHooks.BenignEnd = benignEnd;
}

__device__ void _benignalloc_begin()
{
	g_BenignMallocHooksInit;
	if (g_BenignMallocHooks.BenignBegin)
		g_BenignMallocHooks.BenignBegin();
}

__device__ void _benignalloc_end()
{
	g_BenignMallocHooksInit;
	if (g_BenignMallocHooks.BenignEnd)
		g_BenignMallocHooks.BenignEnd();
}

#endif