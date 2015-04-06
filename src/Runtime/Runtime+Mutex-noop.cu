#include "Runtime.h"

#ifdef MUTEX_NOOP
#pragma region MUTEX_NOOP

#if !_DEBUG
__device__ int _mutex_init() { return 0; }
__device__ int _mutex_shutdown() { }

__device__ _mutex_obj *_mutex_alloc(int id)  {  return (_mutex_obj *)1;  }
__device__ void _mutex_free(_mutex_obj *p) { }
__device__ void _mutex_enter(_mutex_obj *p) { }
__device__ bool _mutex_tryenter(_mutex_obj *p) { return true; }
__device__ void noopMutexLeave(_mutex_obj *p) { }
#else
struct _mutex_obj
{
	MUTEX Id;	// Mutex type
	int Refs;	// Number of entries without a matching leave
};
#define MUTEX_INIT { (MUTEX)0, 0 }
__device__ static _mutex_obj g_mutex_Statics[6] = { MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT };
#undef MUTEX_INIT

__device__ bool _mutex_held(_mutex_obj *p) { return (!p || p->Refs != 0); }
__device__ bool _mutex_notheld(_mutex_obj *p) { return (!p || p->Refs == 0); }

__device__ int _mutex_init() {  return 0;  }
__device__ void _mutex_shutdown() { }

__device__ _mutex_obj *_mutex_alloc(MUTEX id)
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
		_assert(id-2 < _lengthof(g_mutex_Statics));
		p = &g_mutex_Statics[id-2];
		p->Id = id;
		break; }
	}
	return p;
}

__device__ void _mutex_free(_mutex_obj *p)
{
	if (!p) return;
	_assert(p);
	_assert(p->Refs == 0);
	_assert(p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE);
	_free(p);
}

__device__ void _mutex_enter(_mutex_obj *p)
{
	if (!p) return;
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
}

__device__ bool _mutex_tryenter(_mutex_obj *p)
{
	if (!p) return true;
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
	return true;
}

__device__ void _mutex_leave(_mutex_obj *p)
{
	if (!p) return;
	_assert(p->Refs > 0);
	p->Refs--;
	_assert(p->Refs == 0 || p->Id == MUTEX_RECURSIVE);
}
#endif

#pragma endregion
#endif