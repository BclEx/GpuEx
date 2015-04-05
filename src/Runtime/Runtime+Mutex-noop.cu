#include "Runtime.h"

#ifdef MUTEX_NOOP
#pragma region MUTEX_NOOP

#if !_DEBUG
int _mutex_init() { return 0; }
int _mutex_shutdown() { }

_mutex_obj *_mutex_alloc(int id)  {  return (_mutex_obj *)1;  }
void _mutex_free(_mutex_obj *p) { }
void _mutex_enter(_mutex_obj *p) { }
bool _mutex_tryenter(_mutex_obj *p) { return true; }
void noopMutexLeave(_mutex_obj *p) { }
#else
struct _mutex_obj
{
	MUTEX Id;	// Mutex type
	int Refs;	// Number of entries without a matching leave
};
#define MUTEX_INIT { (MUTEX)0, 0 }
static _mutex_obj g_mutex_Statics[6] = { MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT };
#undef MUTEX_INIT

bool _mutex_held(_mutex_obj *p) { return (p->Refs != 0); }
bool _mutex_notheld(_mutex_obj *p) { return (p->Refs == 0); }

int _mutex_init() {  return 0;  }
void _mutex_shutdown() { }

_mutex_obj *_mutex_alloc(MUTEX id)
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

void _mutex_free(_mutex_obj *p)
{
	_assert(p);
	_assert(p->Refs == 0);
	_assert(p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE);
	_free(p);
}

void _mutex_enter(_mutex_obj *p)
{
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
}

bool _mutex_tryenter(_mutex_obj *p)
{
	_assert(p->Id == MUTEX_RECURSIVE || _mutex_notheld(p));
	p->Refs++;
	return true;
}

void _mutex_leave(_mutex_obj *p)
{
	_assert(p->Refs > 0);
	p->Refs--;
	_assert(p->Refs == 0 || p->Id == MUTEX_RECURSIVE);
}
#endif

#pragma endregion
#endif