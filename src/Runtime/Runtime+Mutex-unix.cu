#include "Runtime.h"
RUNTIME_NAMEBEGIN

#ifdef MUTEX_PTHREADS
#pragma region MUTEX_PTHREADS
#include <pthread.h>

#if defined(_DEBUG) || defined(HOMEGROWN_RECURSIVE_MUTEX)
#define MUTEX_NREF 1
#else
#define MUTEX_NREF 0
#endif

struct _mutex_obj
{
	pthread_mutex_t Mutex;     // Mutex controlling the lock
#if MUTEX_NREF
	int Id;                    // Mutex type
	volatile int Refs;         // Number of entrances
	volatile pthread_t Owner;  // Thread that is within this mutex
	bool Trace;                 // True to trace changes
#endif
};
#if MUTEX_NREF
#define MUTEX_INIT { PTHREAD_MUTEX_INITIALIZER, 0, 0, (pthread_t)0, 0 }
#else
#define MUTEX_INIT { PTHREAD_MUTEX_INITIALIZER }
#endif
static _mutex_obj g_mutex_Statics[] = { MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT, MUTEX_INIT };
#undef MUTEX_INIT

#if !defined(NDEBUG) || defined(_DEBUG)
bool MutexHeld(MutexEx p) { return (p->Refs != 0 && pthread_equal(p->Owner, pthread_self())); }
bool MutexNotHeld(MutexEx p) { return p->Refs == 0 || pthread_equal(p->Owner, pthread_self()) == 0; }
#endif

int MutexInit() { return 0; }
void MutexShutdown() { }

MutexEx MutexAlloc(MUTEX id)
{
	if (!g_RuntimeStatics.CoreMutex)
		return nullptr;
	_mutex_obj *p;
	switch (id)
	{
	case MUTEX_RECURSIVE: {
		p = (_mutex_obj *)_allocZero(sizeof(*p));
		if (p)
		{
#ifdef HOMEGROWN_RECURSIVE_MUTEX
			// If recursive mutexes are not available, we will have to build our own.  See below.
			pthread_mutex_init(&p->Mutex, 0);
#else
			// Use a recursive mutex if it is available
			pthread_mutexattr_t recursiveAttr;
			pthread_mutexattr_init(&recursiveAttr);
			pthread_mutexattr_settype(&recursiveAttr, PTHREAD_MUTEX_RECURSIVE);
			pthread_mutex_init(&p->mutex, &recursiveAttr);
			pthread_mutexattr_destroy(&recursiveAttr);
#endif
#if MUTEX_NREF
			p->Id = id;
#endif
		}
		break; }
	case MUTEX_FAST: {
		p = (_mutex_obj *)_allocZero(sizeof(*p));
		if (p)
		{
#if MUTEX_NREF
			p->Id = id;
#endif
			pthread_mutex_init(&p->Mutex, 0);
		}
		break; }
	default: {
		_assert(id-2 >= 0);
		_assert(id-2 < _lengthof(g_mutex_Statics));
		p = &g_mutex_Statics[id-2];
#if MUTEX_NREF
		p->Id = id;
#endif
		break; }
	}
	return p;
}

void MutexFree(MutexEx p)
{
	if (!p) return;
	_assert(p->Refs == 0);
	_assert(p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE);
	pthread_mutex_destroy(&p->Mutex);
	_free(p);
}

void MutexEnter(MutexEx p)
{
	if (!p) return;
	_assert(p->Id == MUTEX_RECURSIVE || pthreadMutexNotheld(p));
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and p->owner are equal if p->owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (p->Refs > 0 && pthread_equal(p->Owner, self))
			p->Refs++;
		else
		{
			pthread_mutex_lock(&p->Mutex);
			_assert(p->Refs == 0);
			p->Owner = self;
			p->Refs = 1;
		}
	}
#else
	// Use the built-in recursive mutexes if they are available.
	pthread_mutex_lock(&p->Mutex);
#if MUTEX_NREF
	_assert(p->Refs > 0 || p->Owner == 0);
	p->Owner = pthread_self();
	p->Refs++;
#endif
#endif
#ifdef _DEBUG
	if (p->Trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
}

bool MutexTryEnter(MutexEx p)
{
	if (!p) return true;
	_assert(p->Id == MUTEX_RECURSIVE || pthreadMutexNotheld(p));
	bool rc;
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and p->owner are equal if p->owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (p->Refs > 0 && pthread_equal(p->Owner, self))
		{
			p->Refs++;
			rc = true;
		}
		else if (pthread_mutex_trylock(&p->Mutex) == 0)
		{
			_assert(p->Refs == 0);
			p->Owner = self;
			p->Refs = 1;
			rc = true;
		}
		else
			rc = false;
	}
#else
	// Use the built-in recursive mutexes if they are available.
	if (pthread_mutex_trylock(&p->Mutex) == 0)
	{
#if MUTEX_NREF
		p->Owner = pthread_self();
		p->Refs++;
#endif
		rc = true;
	}
	else
		rc = false;
#endif
#ifdef _DEBUG
	if (rc && p->Trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
	return rc;
}

void MutexLeave(MutexEx p)
{
	if (!p) return;
	_assert(pthreadMutexHeld(p));
#if MUTEX_NREF
	p->Refs--;
	if (p->Refs == 0) p->Owner = 0;
#endif
	_assert(p->Refs == 0 || p->Id == MUTEX_RECURSIVE);
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	if (p->Refs == 0)
		pthread_mutex_unlock(&p->Mutex);
#else
	pthread_mutex_unlock(&p->Mutex);
#endif
#ifdef _DEBUG
	if (p->Trace)
		printf("leave mutex %p (%d) with nRef=%d\n", p, p->Trace, p->Refs);
#endif
}

#pragma endregion
#endif

RUNTIME_NAMEEND