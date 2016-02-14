#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "Runtime.h"
RUNTIME_NAMEBEGIN

#ifdef MUTEX_WIN
#pragma region MUTEX_WIN

struct _mutex_obj
{
	CRITICAL_SECTION Mutex;		// Mutex controlling the lock
	MUTEX Id;						// Mutex type
#ifdef _DEBUG
	volatile int Refs;			// Number of enterances
	volatile DWORD Owner;		// Thread holding this mutex
	bool Trace;					// True to trace changes
#endif
};
#ifdef _DEBUG
#define MUTEX_INIT { 0, 0, 0L, (DWORD)0, 0 }
#else
#define MUTEX_INIT { 0, 0 }
#endif
static _mutex_obj MutexStatics[6] = { MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT };
#undef MUTEX_INIT
static bool MutexIsInit = false;
static long MutexLock = 0;

#ifdef _DEBUG
bool MutexHeld(MutexEx p) { return (!p || (p->Refs != 0 && p->Owner == GetCurrentThreadId())); }
bool MutexNotHeld2(MutexEx p, DWORD tid) { return (!p || p->Refs == 0 || p->Owner != tid); }
bool MutexNotHeld(MutexEx p) { DWORD tid = GetCurrentThreadId(); return (!p || MutexNotHeld2(p, tid)); }
#endif

int MutexInit()
{ 
	// The first to increment to 1 does actual initialization
	if (InterlockedCompareExchange(&MutexLock, 1, 0) == 0)
	{
		for (int i = 0; i < _lengthof(MutexStatics); i++)
		{
#if OS_WINRT
			InitializeCriticalSectionEx(&MutexStatics[i].Mutex, 0, 0);
#else
			InitializeCriticalSection(&MutexStatics[i].Mutex);
#endif
		}
		MutexIsInit = true;
	}
	else while (!MutexIsInit) // Someone else is in the process of initing the static mutexes
		Sleep(1);
	return 0; 
}

void MutexShutdown()
{
	// The first to decrement to 0 does actual shutdown (which should be the last to shutdown.)
	if (InterlockedCompareExchange(&MutexLock, 0, 1) == 1)
	{
		if (MutexIsInit)
		{
			for (int i =0 ; i < _lengthof(MutexStatics); i++)
				DeleteCriticalSection(&MutexStatics[i].Mutex);
			MutexIsInit = false;
		}
	}
}

MutexEx MutexAlloc(MUTEX id)
{
	if (!g_RuntimeStatics.CoreMutex)
		return nullptr;
	_mutex_obj *p;
	switch (id)
	{
	case MUTEX_FAST:
	case MUTEX_RECURSIVE: {
		p = (_mutex_obj *)_allocZero(sizeof(*p));
		if (p)
		{  
#ifdef _DEBUG
			p->Id = id;
#endif
#if OS_WINRT
			InitializeCriticalSectionEx(&p->Mutex, 0, 0);
#else
			InitializeCriticalSection(&p->Mutex);
#endif
		}
		break; }
	default: {
		_assert(MutexIsInit);
		_assert(id-2 >= 0);
		_assert(id-2 < _lengthof(MutexStatics));
		p = &MutexStatics[id-2];
#ifdef _DEBUG
		p->Id = id;
#endif
		break; }
	}
	return p;
}

void MutexFree(MutexEx p)
{
	if (!p) return;
	_assert(p->Refs == 0 && p->Owner == 0);
	_assert(p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE);
	DeleteCriticalSection(&p->Mutex);
	_free(p);
}

void MutexEnter(MutexEx p)
{
	if (!p) return;
#ifdef _DEBUG
	DWORD tid = GetCurrentThreadId(); 
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p, tid));
#endif
	EnterCriticalSection(&p->Mutex);
#ifdef _DEBUG
	_assert(p->Refs > 0 || p->Owner == 0);
	p->Owner = tid; 
	p->Refs++;
	if (p->Trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
}

bool MutexTryEnter(MutexEx p)
{
	if (!p) return true;
#ifndef NDEBUG
	DWORD tid = GetCurrentThreadId(); 
#endif
	bool rc = false;
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p, tid));
	// The sqlite3_mutex_try() routine is very rarely used, and when it is used it is merely an optimization.  So it is OK for it to always fail.  
	//
	// The TryEnterCriticalSection() interface is only available on WinNT. And some windows compilers complain if you try to use it without
	// first doing some #defines that prevent SQLite from building on Win98. For that reason, we will omit this optimization for now.  See ticket #2685.
	if (TryEnterCriticalSection(&p->Mutex))
	{
#ifdef _DEBUG
		p->Owner = tid;
		p->Refs++;
#endif
		rc = true;
	}
#ifdef _DEBUG
	if (rc && p->Trace)
		printf("try mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
	return rc;
}

void MutexLeave(MutexEx p)
{
	if (!p) return;
#ifndef NDEBUG
	DWORD tid = GetCurrentThreadId();
	_assert(p->Refs > 0);
	_assert(p->Owner == tid);
	p->Refs--;
	if (p->Refs == 0) p->Owner = 0;
	_assert(p->Refs == 0 || p->Id == MUTEX_RECURSIVE);
#endif
	LeaveCriticalSection(&p->Mutex);
#ifdef _DEBUG
	if (p->Trace)
		printf("leave mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
}

static const _mutex_methods _winDefaultMethods = {
	MutexInit,
	MutexShutdown,
	MutexAlloc,
	MutexFree,
	MutexEnter,
	MutexTryEnter,
	MutexLeave,
#ifdef _DEBUG
	MutexHeld,
	MutexNotHeld
#else
	nullptr,
	nullptr
#endif
};
void __mutexsystem_setdefault()
{
	__mutexsystem = _winDefaultMethods;
}

#pragma endregion
#endif

RUNTIME_NAMEEND