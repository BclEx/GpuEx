#include "Runtime.h"
RUNTIME_NAMEBEGIN

#define MUTEX_NOOP
#ifdef MUTEX_NOOP
#pragma region MUTEX_NOOP

#ifndef _DEBUG

struct _mutex_obj { };
__device__ int MutexInit() { return 0; }
__device__ void MutexShutdown() { }

__device__ _mutex_obj *MutexAlloc(MUTEX id) { return (_mutex_obj *)8;  }
__device__ void MutexFree(_mutex_obj *p) { }
__device__ void MutexEnter(_mutex_obj *p) { }
__device__ bool MutexTryEnter(_mutex_obj *p) { return true; }
__device__ void MutexLeave(_mutex_obj *p) { }

__constant__ static const _mutex_methods _noopDefaultMethods = {
	(int (*)())MutexInit,
	(void (*)())MutexShutdown,
	(MutexEx (*)(MUTEX))MutexAlloc,
	(void (*)(MutexEx))MutexFree,
	(void (*)(MutexEx))MutexEnter,
	(bool (*)(MutexEx))MutexTryEnter,
	(void (*)(MutexEx))MutexLeave,
	nullptr,
	nullptr,
};

#else

struct _mutex_obj
{
	MUTEX Id;	// Mutex type
	int Refs;	// Number of entries without a matching leave
};
#define MUTEX_INIT { (MUTEX)0, 0 }
__device__ static _mutex_obj MutexStatics[6] = { MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT };
#undef MUTEX_INIT

__device__ bool MutexHeld(_mutex_obj *p) { return (!p || p->Refs != 0); }
__device__ bool MutexNotHeld(_mutex_obj *p) { return (!p || p->Refs == 0); }

__device__ int MutexInit() { return 0; }
__device__ void MutexShutdown() { }

__device__ _mutex_obj *MutexAlloc(MUTEX id)
{
	_mutex_obj *p;
	switch (id)
	{
	case MUTEX_FAST:
	case MUTEX_RECURSIVE: {
		p = (_mutex_obj *)_alloc(sizeof(*p));
		if (p)
		{  
			p->Id = id;
			p->Refs = 0;
		}
		break; }
	default: {
		_assert(id-2 >= 0);
		_assert(id-2 < _lengthof(MutexStatics));
		p = &MutexStatics[id-2];
		p->Id = id;
		break; }
	}
	return p;
}

__device__ void MutexFree(_mutex_obj *p)
{
	if (!p) return;
	_assert(p);
	_assert(p->Refs == 0);
	_assert(p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE);
	_free(p);
}

__device__ void MutexEnter(_mutex_obj *p)
{
	if (!p) return;
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
}

__device__ bool MutexTryEnter(_mutex_obj *p)
{
	if (!p) return true;
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
	return true;
}

__device__ void MutexLeave(_mutex_obj *p)
{
	if (!p) return;
	_assert(p->Refs > 0);
	p->Refs--;
	_assert(p->Refs == 0 || p->Id == MUTEX_RECURSIVE);
}

__constant__ static const _mutex_methods _noopDefaultMethods = {
	(int (*)())MutexInit,
	(void (*)())MutexShutdown,
	(MutexEx (*)(MUTEX))MutexAlloc,
	(void (*)(MutexEx))MutexFree,
	(void (*)(MutexEx))MutexEnter,
	(bool (*)(MutexEx))MutexTryEnter,
	(void (*)(MutexEx))MutexLeave,
	(bool (*)(MutexEx))MutexHeld,
	(bool (*)(MutexEx))MutexNotHeld
};

#endif

__device__ void __mutexsystem_setdefault()
{
	__mutexsystem = _noopDefaultMethods;
}

#pragma endregion
#endif

RUNTIME_NAMEEND