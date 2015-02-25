// notify.c
#include "Core+Btree.cu.h"
#include "BtreeInt.cu.h"

#ifdef ENABLE_UNLOCK_NOTIFY

#define AssertMutexHeld() _assert(MutexEx::Held(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER)))

__device__ static BContext *_WSD g_blockedList = nullptr;

#ifndef NDEBUG
__device__ static void CheckListProperties(BContext *ctx)
{
	for (BContext *p = g_blockedList; p; p = p->NextBlocked)
	{
		// Verify property (1)
		_assert(p->UnlockConnection || p->BlockingConnection);

		// Verify property (2)
		bool seen = false;
		for (BContext *p2 = g_blockedList; p2 != p; p2 = p2->NextBlocked)
		{
			if (p2->UnlockNotify == p->UnlockNotify) seen = true;
			_assert(p2->UnlockNotify == p->UnlockNotify || !seen);
			_assert(!ctx || p->UnlockConnection != ctx);
			_assert(!ctx || p->BlockingConnection != ctx);
		}
	}
}
#else
#define CheckListProperties(x)
#endif

__device__ static void RemoveFromBlockedList(BContext *ctx)
{
	AssertMutexHeld();
	for (BContext **pp = &g_blockedList; *pp; pp = &(*pp)->NextBlocked)
		if (*pp == ctx)
		{
			*pp = (*pp)->NextBlocked;
			break;
		}
}

__device__ static void AddToBlockedList(BContext *ctx)
{
	AssertMutexHeld();
	BContext **pp;
	for (pp = &g_blockedList; *pp && (*pp)->UnlockNotify != ctx->UnlockNotify; pp = &(*pp)->NextBlocked) ;
	ctx->NextBlocked = *pp;
	*pp = ctx;
}

__device__ static void EnterMutex()
{
	MutexEx::Enter(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER));
	CheckListProperties(nullptr);
}

__device__ static void LeaveMutex()
{
	AssertMutexHeld();
	CheckListProperties(nullptr);
	MutexEx::Leave(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER));
}

__device__ RC BContext::UnlockNotify_(void (*notify)(void **, int), void *arg, void (*error)(BContext *, RC, const char *))
{
	RC rc = RC_OK;
	MutexEx::Enter(Mutex);
	EnterMutex();

	if (!notify)
	{
		RemoveFromBlockedList(this);
		BlockingConnection = nullptr;
		UnlockConnection = nullptr;
		UnlockNotify = nullptr;
		UnlockArg = nullptr;
	}
	else if (!BlockingConnection)
		notify(&arg, 1); // The blocking transaction has been concluded. Or there never was a blocking transaction. In either case, invoke the notify callback immediately. 
	else
	{
		BContext *p;
		for (p = BlockingConnection; p && p != this; p = p->UnlockConnection) { }
		if (p)
			rc = RC_LOCKED; // Deadlock detected.
		else
		{
			UnlockConnection = BlockingConnection;
			UnlockNotify = notify;
			UnlockArg = arg;
			RemoveFromBlockedList(this);
			AddToBlockedList(this);
		}
	}

	LeaveMutex();
	_assert(!MallocFailed);
	if (error)
		error(this, rc, (rc ? "database is deadlocked" : nullptr));
	MutexEx::Leave(Mutex);
	return rc;
}

__device__ void BContext::ConnectionBlocked(BContext *blocker)
{
	EnterMutex();
	if (!BlockingConnection && !UnlockConnection)
		AddToBlockedList(this);
	BlockingConnection = blocker;
	LeaveMutex();
}

__device__ void BContext::ConnectionUnlocked()
{
	void (*unlockNotify)(void **, int) = nullptr; // Unlock-notify cb to invoke
	int argsLength = 0; // Number of entries in args[]
	BContext **pp; // Iterator variable
	void **args; // Arguments to the unlock callback
	void **dyns = nullptr; // Dynamically allocated space for args[]
	void *ArgsStatic[16]; // Starter space for args[].  No malloc required

	args = ArgsStatic;
	EnterMutex(); // Enter STATIC_MASTER mutex

	// This loop runs once for each entry in the blocked-connections list.
	for (pp = &g_blockedList; *pp;)
	{
		BContext *p = *pp;

		// Step 1.
		if (p->BlockingConnection == this)
			p->BlockingConnection = nullptr;

		// Step 2.
		if (p->UnlockConnection == this)
		{
			_assert(p->UnlockNotify);
			if (p->UnlockNotify != unlockNotify && argsLength != 0)
			{
				unlockNotify(args, argsLength);
				argsLength = 0;
			}

			_benignalloc_begin();
			_assert(args == dyns || (!dyns && args == ArgsStatic));
			_assert(argsLength <= (int)_lengthof(ArgsStatic) || args == dyns);
			if ((!dyns && argsLength == (int)_lengthof(ArgsStatic)) || (dyns && argsLength == (int)(_allocsize(dyns) / sizeof(void*))))
			{
				// The args[] array needs to grow.
				void **newArgs = (void **)_alloc(argsLength*sizeof(void *)*2);
				if (newArgs)
				{
					_memcpy(newArgs, args, argsLength*sizeof(void *));
					_free(dyns);
					dyns = args = newArgs;
				}
				else
				{
					// This occurs when the array of context pointers that need to be passed to the unlock-notify callback is larger than the
					// ArgsStatic[] array allocated on the stack and the attempt to allocate a larger array from the heap has failed.
					//
					// This is a difficult situation to handle. Returning an error code to the caller is insufficient, as even if an error code
					// is returned the transaction on connection db will still be closed and the unlock-notify callbacks on blocked connections
					// will go unissued. This might cause the application to wait indefinitely for an unlock-notify callback that will never arrive.
					//
					// Instead, invoke the unlock-notify callback with the context array already accumulated. We can then clear the array and
					// begin accumulating any further context pointers without requiring any dynamic allocation. This is sub-optimal because
					// it means that instead of one callback with a large array of context pointers the application will receive two or more
					// callbacks with smaller arrays of context pointers, which will reduce the applications ability to prioritize multiple 
					// connections. But it is the best that can be done under the circumstances.
					unlockNotify(args, argsLength);
					argsLength = 0;
				}
			}
			_benignalloc_end();

			args[argsLength++] = p->UnlockArg;
			UnlockNotify = p->UnlockNotify;
			p->UnlockConnection = nullptr;
			p->UnlockNotify = nullptr;
			p->UnlockArg = nullptr;
		}

		// Step 3.
		if (!p->BlockingConnection && !p->UnlockConnection)
		{
			// Remove connection p from the blocked connections list.
			*pp = p->NextBlocked;
			p->NextBlocked = nullptr;
		}
		else
			pp = &p->NextBlocked;
	}

	if (argsLength != 0)
		unlockNotify(args, argsLength);
	_free(dyns);
	LeaveMutex(); // Leave STATIC_MASTER mutex
}

__device__ void BContext::ConnectionClosed()
{
	ConnectionUnlocked();
	EnterMutex();
	RemoveFromBlockedList(this);
	CheckListProperties(this);
	LeaveMutex();
}

#endif
