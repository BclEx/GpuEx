// btmutex.c
#include "Core+Btree.cu.h"
#include "BtreeInt.cu.h"
#ifndef OMIT_SHARED_CACHE
#if THREADSAFE

__device__ static void LockBtreeMutex(Btree *p)
{
	_assert(!p->Locked);
	_assert(_mutex_notheld(p->Bt->Mutex));
	_assert(_mutex_held(p->Ctx->Mutex));

	_mutex_enter(p->Bt->Mutex);
	p->Bt->Ctx = p->Ctx;
	p->Locked = 1;
}

__device__ static void UnlockBtreeMutex(Btree *p)
{
	BtShared *bt = p->Bt;
	_assert(p->Locked);
	_assert(_mutex_held(bt->Mutex));
	_assert(_mutex_held(p->Ctx->Mutex));
	_assert(p->Ctx == bt->Ctx);

	_mutex_leave(bt->Mutex);
	p->Locked = false;
}

__device__ void Btree::Enter()
{
	// Some basic sanity checking on the Btree.  The list of Btrees connected by pNext and pPrev should be in sorted order by
	// Btree.pBt value. All elements of the list should belong to the same connection. Only shared Btrees are on the list.
	_assert(!Next || Next->Bt > Bt);
	_assert(!Prev || Prev->Bt < Bt);
	_assert(!Next || Next->Ctx == Ctx);
	_assert(!Prev || Prev->Ctx == Ctx);
	_assert(Sharable_ || (!Next && !Prev));

	// Check for locking consistency
	_assert(!Locked || WantToLock > 0);
	_assert(Sharable_ || WantToLock == 0);

	// We should already hold a lock on the database connection
	_assert(_mutex_held(Ctx->Mutex));

	/// Unless the database is sharable and unlocked, then BtShared.db should already be set correctly.
	_assert((!Locked && Sharable_) || Bt->Ctx == Ctx);

	if (!Sharable_) return;
	WantToLock++;
	if (Locked) return;

	// In most cases, we should be able to acquire the lock we want without having to go throught the ascending lock
	// procedure that follows.  Just be sure not to block.
	if (_mutex_tryenter(Bt->Mutex))
	{
		Bt->Ctx = Ctx;
		Locked = true;
		return;
	}

	// To avoid deadlock, first release all locks with a larger BtShared address.  Then acquire our lock.  Then reacquire
	// the other BtShared locks that we used to hold in ascending order.
	Btree *later;
	for (later = Next; later; later = later->Next)
	{
		_assert(later->Sharable_);
		_assert(!later->Next || later->Next->Bt > later->Bt);
		_assert(!later->Locked || later->WantToLock > 0);
		if (later->Locked)
			UnlockBtreeMutex(later);
	}
	LockBtreeMutex(this);
	for (later = Next; later; later = later->Next)
		if (later->WantToLock)
			LockBtreeMutex(later);
}

__device__ void Btree::Leave()
{
	if (Sharable_)
	{
		_assert(WantToLock > 0);
		WantToLock--;
		if (!WantToLock)
			UnlockBtreeMutex(this);
	}
}

#ifndef NDEBUG
__device__ bool Btree::HoldsMutex()
{
	_assert(!Sharable_ || !Locked || WantToLock > 0);
	_assert(!Sharable_ || !Locked || Ctx == Bt->Ctx);
	_assert(!Sharable_ || !Locked || _mutex_held(Bt->Mutex));
	_assert(!Sharable_ || !Locked || _mutex_held(Ctx->Mutex));
	return (!Sharable_ || Locked);
}
#endif

#ifndef OMIT_INCRBLOB
//__device__ void Btree::EnterCursor(BtCursor *cur) { cur->Btree->Enter(); }
//__device__ void Btree::LeaveCursor(BtCursor *cur) { cur->Btree->Leave(); }
#endif

__device__ void Btree::EnterAll(BContext *ctx)
{
	_assert(_mutex_held(ctx->Mutex));
	for (int i = 0; i < ctx->DBs.length; i++)
	{
		Btree *p = ctx->DBs[i].Bt;
		if (p) p->Enter();
	}
}
__device__ void Btree::LeaveAll(BContext *ctx)
{
	_assert(_mutex_held(ctx->Mutex));
	for (int i = 0; i < ctx->DBs.length; i++)
	{
		Btree *p = ctx->DBs[i].Bt;
		if (p) p->Leave();
	}
}

__device__ bool Btree::Sharable() { return Sharable_; }

#ifndef NDEBUG
__device__ bool Btree::HoldsAllMutexes(BContext *ctx)
{
	if (!_mutex_held(ctx->Mutex))
		return false;
	for (int i = 0; i < ctx->DBs.length; i++)
	{
		Btree *p = ctx->DBs[i].Bt;
		if (p && p->Sharable_ && (!p->WantToLock || !_mutex_held(p->Bt->Mutex)))
			return false;
	}
	return true;
}

__device__ bool Btree::SchemaMutexHeld(BContext *ctx, int db, Core::Schema *schema)
{
	_assert(ctx != nullptr);
	if (schema) db = Schema::ToIndex(ctx, schema);
	_assert(db >= 0 && db < ctx->DBs.length);
	if (!_mutex_held(ctx->Mutex)) return false;
	if (db == 1) return true;
	Btree *p = ctx->DBs[db].Bt;
	_assert(p != nullptr);
	return (!p->Sharable_ || p->Locked);
}
#endif

#else

__device__ void Btree::Enter()
{
	Bt->Ctx = Ctx;
}

__device__ void Btree::EnterAll(BContext *ctx)
{
	for (int i = 0; i < ctx->DBs.length; i++)
	{
		Btree *p = ctx->DBs[i].Bt;
		if (p)
			p->Bt->Ctx = p->Ctx;
	}
}

#endif
#endif
