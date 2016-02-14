#include "Runtime.h"
#include "RuntimeTypes.h"
#include <stdarg.h>
RUNTIME_NAMEBEGIN

	// Attempt to release up to n bytes of non-essential memory currently held by SQLite. An example of non-essential memory is memory used to
	// cache database pages that are not currently in use.
	// SoftHeapLimitEnforcer::This routine runs when the memory allocator sees that the total memory allocation is about to exceed the soft heap limit.
	__device__ static int __alloc_releasememory(void *arg, int64 used, int allocSize)
{
#ifdef ENABLE_MEMORY_MANAGEMENT
	return sqlite3PcacheReleaseMemory(allocSize);
#else
	// IMPLEMENTATION-OF: R-34391-24921 The sqlite3_release_memory() routine is a no-op returning zero if SQLite is not compiled with SQLITE_ENABLE_MEMORY_MANAGEMENT.
	UNUSED_PARAMETER(allocSize);
	return 0;
#endif
}

// An instance of the following object records the location of each unused scratch buffer.
typedef struct ScratchFreeslot
{
	struct ScratchFreeslot *Next; // Next unused scratch buffer
} ScratchFreeslot;

// State information local to the memory allocation subsystem.
__device__ static _WSD struct Mem0Global
{
	MutexEx Mutex; // Mutex to serialize access

	// The alarm callback and its arguments.  The mem0.mutex lock will be held while the callback is running.  Recursive calls into
	// the memory subsystem are allowed, but no new callbacks will be issued.
	int64 AlarmThreshold;
	int (*AlarmCallback)(void*,int64,int);
	void *AlarmArg;

	// Pointers to the end of sqlite3GlobalConfig.pScratch memory (so that a range test can be used to determine if an allocation
	// being freed came from pScratch) and a pointer to the list of unused scratch allocations.
	void *ScratchEnd;
	ScratchFreeslot *ScratchFree;
	uint32 ScratchFreeLength;

	// True if heap is nearly "full" where "full" is defined by the sqlite3_soft_heap_limit() setting.
	bool NearlyFull;
} g_mem0 = { 0, 0, 0, 0, 0, 0, 0, 0 };
#define mem0 _GLOBAL(struct Mem0Global, g_mem0)

// Change the alarm callback
__device__ void __alloc_setmemoryalarm(int (*callback)(void*,int64,int), void *arg, int64 threshold)
{
	_mutex_enter(mem0.Mutex);
	mem0.AlarmCallback = callback;
	mem0.AlarmArg = arg;
	mem0.AlarmThreshold = threshold;
	int used = _status_value(STATUS_MEMORY_USED);
	mem0.NearlyFull = (threshold > 0 && threshold <= used);
	_mutex_leave(mem0.Mutex);
}

// Set the soft heap-size limit for the library. Passing a zero or negative value indicates no limit.
__device__ int64 __alloc_memoryused();
__device__ int64 __alloc_softheaplimit64(int64 size)
{
	_mutex_enter(mem0.Mutex);
	int64 priorLimit = mem0.AlarmThreshold;
	_mutex_leave(mem0.Mutex);
	if (size < 0) return priorLimit;
	if (size > 0)
		__alloc_setmemoryalarm(__alloc_releasememory, 0, size);
	else
		__alloc_setmemoryalarm(nullptr, 0, 0);
	int64 excess = __alloc_memoryused() - size;
	if (excess > 0) __alloc_releasememory(nullptr, 0, (int)(excess & 0x7fffffff));
	return priorLimit;
}

__device__ void __alloc_softheaplimit(int size)
{
	if (size < 0) size = 0;
	__alloc_softheaplimit64(size);
}

// Initialize the memory allocation subsystem.
__device__ int _alloc_initG()
{
	if (!__allocsystem.Alloc)
	{
		//printf("__allocsystem_setdefault\n");
		__allocsystem_setdefault();
	}
	memset(&mem0, 0, sizeof(mem0));
	if (TagBase_RuntimeStatics.RuntimeMutex)
		mem0.Mutex = _mutex_alloc(MUTEX_STATIC_MEM);
	if (TagBase_RuntimeStatics.Scratch && TagBase_RuntimeStatics.ScratchSize >= 100 && TagBase_RuntimeStatics.Scratchs > 0)
	{
		int size = _ROUNDDOWN8(TagBase_RuntimeStatics.ScratchSize);
		TagBase_RuntimeStatics.ScratchSize = size;
		ScratchFreeslot *slot = (ScratchFreeslot*)TagBase_RuntimeStatics.Scratch;
		int n = TagBase_RuntimeStatics.Scratchs;
		mem0.ScratchFree = slot;
		mem0.ScratchFreeLength = n;
		for (int i = 0; i < n-1; i++)
		{
			slot->Next = (ScratchFreeslot *)(size + (char *)slot);
			slot = slot->Next;
		}
		slot->Next = nullptr;
		mem0.ScratchEnd = (void *)&slot[1];
	}
	else
	{
		mem0.ScratchEnd = 0;
		TagBase_RuntimeStatics.Scratch = 0;
		TagBase_RuntimeStatics.ScratchSize = 0;
		TagBase_RuntimeStatics.Scratchs = 0;
	}
	//if (!TagBase_RuntimeStatics.Page || TagBase_RuntimeStatics.PageSize < 512 || TagBase_RuntimeStatics.Pages < 1)
	//{
	//	TagBase_RuntimeStatics.Page = 0;
	//	TagBase_RuntimeStatics.PageSize = 0;
	//	TagBase_RuntimeStatics.Pages = 0;
	//}
	return __allocsystem.Init(nullptr);
}

// Return true if the heap is currently under memory pressure - in other words if the amount of heap used is close to the limit set by sqlite3_soft_heap_limit().
__device__ bool _alloc_heapnearlyfull()
{
	return mem0.NearlyFull;
}

// Deinitialize the memory allocation subsystem.
__device__ void _alloc_shutdownG()
{
	__allocsystem.Shutdown(nullptr);
	_memset(&mem0, 0, sizeof(mem0));
}

// Return the amount of memory currently checked out.
__device__ int64 __alloc_memoryused()
{
	int n, max;
	_status(STATUS_MEMORY_USED, &n, &max, false);
	return (int64)n; // Work around bug in Borland C. Ticket #3216
}

// Return the maximum amount of memory that has ever been checked out since either the beginning of this process or since the most recent reset.
__device__ int64 __alloc_memoryhighwater(bool resetFlag)
{
	int n, max;
	_status(STATUS_MEMORY_USED, &n, &max, resetFlag);
	return (int64)max; // Work around bug in Borland C. Ticket #3216
}

// Trigger the alarm 
__device__ static void __alloc_triggermemoryalarm(size_t size)
{
	if (!mem0.AlarmCallback) return;
	int (*callback)(void*,int64,int) = mem0.AlarmCallback;
	int64 nowUsed = _status_value(STATUS_MEMORY_USED);
	void *arg = mem0.AlarmArg;
	mem0.AlarmCallback = 0;
	_mutex_leave(mem0.Mutex);
	callback(arg, nowUsed, (int)size);
	_mutex_enter(mem0.Mutex);
	mem0.AlarmCallback = callback;
	mem0.AlarmArg = arg;
}

// Do a memory allocation with statistics and alarms.  Assume the lock is already held.
__device__ static size_t AllocWithAlarm(size_t size, void **pp)
{
	_assert(_mutex_held(mem0.Mutex));
	size_t fullSize = __allocsystem.Roundup(size);
	_status_set(STATUS_MALLOC_SIZE, (int)size);
	if (mem0.AlarmCallback)
	{
		int used = _status_value(STATUS_MEMORY_USED);
		if (used >= mem0.AlarmThreshold - fullSize)
		{
			mem0.NearlyFull = true;
			__alloc_triggermemoryalarm(fullSize);
		}
		else
			mem0.NearlyFull = false;
	}
	void *p = __allocsystem.Alloc(fullSize);
#ifdef ENABLE_MEMORY_MANAGEMENT
	if (!p && mem0.AlarmCallback)
	{
		sqlite3MallocAlarm(fullSize);
		p = __allocsystem_alloc(fullSize);
	}
#endif
	if (p)
	{
		fullSize = _allocsize(p);
		_status_add(STATUS_MEMORY_USED, (int)fullSize);
		_status_add(STATUS_MALLOC_COUNT, 1);
	}
	*pp = p;
	return fullSize;
}

// Allocate memory.  This routine is like sqlite3_malloc() except that it assumes the memory subsystem has already been initialized.
__device__ void *_allocG(size_t size)
{
	// A memory allocation of a number of bytes which is near the maximum signed integer value might cause an integer overflow inside of the
	// xMalloc().  Hence we limit the maximum size to 0x7fffff00, giving 255 bytes of overhead.  SQLite itself will never use anything near
	// this amount.  The only way to reach the limit is with sqlite3_malloc()
	void *p;
	if (size <= 0 || size >= 0x7fffff00)
		p = nullptr;
	else if (TagBase_RuntimeStatics.Memstat)
	{
		_mutex_enter(mem0.Mutex);
		AllocWithAlarm(size, &p);
		_mutex_leave(mem0.Mutex);
	}
	else
		p = __allocsystem.Alloc(size);
	_assert(_HASALIGNMENT8(p)); // IMP: R-04675-44850
	return p;
}

// Each thread may only have a single outstanding allocation from xScratchMalloc().  We verify this constraint in the single-threaded
// case by setting scratchAllocOut to 1 when an allocation is outstanding clearing it when the allocation is freed.
#if THREADSAFE == 0 && defined(_DEBUG)
__device__ static int g_scratchAllocOut = 0;
#endif

// Allocate memory that is to be used and released right away. This routine is similar to alloca() in that it is not intended
// for situations where the memory might be held long-term.  This routine is intended to get memory to old large transient data
// structures that would not normally fit on the stack of an embedded processor.
__device__ void *_scratchallocG(size_t size)
{
	_assert(size > 0);
	_mutex_enter(mem0.Mutex);
	void *p;
	if (mem0.ScratchFreeLength && TagBase_RuntimeStatics.ScratchSize >= size)
	{
		p = mem0.ScratchFree;
		mem0.ScratchFree = mem0.ScratchFree->Next;
		mem0.ScratchFreeLength--;
		_status_add(STATUS_SCRATCH_USED, 1);
		_status_set(STATUS_SCRATCH_SIZE, (int)size);
		_mutex_leave(mem0.Mutex);
	}
	else
	{
		if (TagBase_RuntimeStatics.Memstat)
		{
			_status_set(STATUS_SCRATCH_SIZE, (int)size);
			size = AllocWithAlarm(size, &p);
			if (p) _status_add(STATUS_SCRATCH_OVERFLOW, (int)size);
			_mutex_leave(mem0.Mutex);
		}
		else
		{
			_mutex_leave(mem0.Mutex);
			p = __allocsystem.Alloc(size);
		}
		_memdbg_settype(p, MEMTYPE_LRATCH);
	}
	_assert(_mutex_notheld(mem0.Mutex));
#if THREADSAFE == 0 && defined(_DEBUG)
	// Verify that no more than two scratch allocations per thread are outstanding at one time.  (This is only checked in the
	// single-threaded case since checking in the multi-threaded case would be much more complicated.)
	_assert(g_scratchAllocOut <= 1);
	if (p) g_scratchAllocOut++;
#endif
	return p;
}

__device__ void _scratchfreeG(void *p)
{
	if (p)
	{
#if THREADSAFE == 0 && defined(_DEBUG)
		// Verify that no more than two scratch allocation per thread is outstanding at one time.  (This is only checked in the
		// single-threaded case since checking in the multi-threaded case would be much more complicated.)
		_assert(g_scratchAllocOut >= 1 && g_scratchAllocOut <= 2);
		g_scratchAllocOut--;
#endif
		if (p >= TagBase_RuntimeStatics.Scratch && p < mem0.ScratchEnd)
		{
			// Release memory from the SQLITE_CONFIG_LRATCH allocation
			ScratchFreeslot *slot = (ScratchFreeslot *)p;
			_mutex_enter(mem0.Mutex);
			slot->Next = mem0.ScratchFree;
			mem0.ScratchFree = slot;
			mem0.ScratchFreeLength++;
			_assert(mem0.ScratchFreeLength <= (uint32)TagBase_RuntimeStatics.Scratchs);
			_status_add(STATUS_SCRATCH_USED, -1);
			_mutex_leave(mem0.Mutex);
		}
		else
		{
			// Release memory back to the heap
			_assert(_memdbg_hastype(p, MEMTYPE_LRATCH));
			_assert(_memdbg_nottype(p, ~MEMTYPE_LRATCH));
			_memdbg_settype(p, MEMTYPE_HEAP);
			if (TagBase_RuntimeStatics.Memstat)
			{
				size_t size2 = _allocsize(p);
				_mutex_enter(mem0.Mutex);
				_status_add(STATUS_SCRATCH_OVERFLOW, -(int)size2);
				_status_add(STATUS_MEMORY_USED, -(int)size2);
				_status_add(STATUS_MALLOC_COUNT, -1);
				__allocsystem.Free(p);
				_mutex_leave(mem0.Mutex);
			}
			else
				__allocsystem.Free(p);
		}
	}
}

// TRUE if p is a lookaside memory allocation from db
#ifndef OMIT_LOOKASIDE
__device__ static bool IsLookaside(TagBase *tag, void *p) { return (p && p >= tag->Lookaside.Start && p < tag->Lookaside.End); }
#else
#define IsLookaside(A,B) false
#endif

// Return the size of a memory allocation previously obtained from sqlite3Malloc() or sqlite3_malloc().
__device__ size_t _allocsize(void *p)
{
	_assert(_memdbg_hastype(p, MEMTYPE_HEAP));
	_assert(_memdbg_nottype(p, MEMTYPE_DB));
	return __allocsystem.Size(p);
}

__device__ size_t _tagallocsize(TagBase *tag, void *p)
{
	_assert(!tag || _mutex_held(tag->Mutex));
	if (tag && IsLookaside(tag, p))
		return tag->Lookaside.Size;
	_assert(_memdbg_hastype(p, MEMTYPE_DB));
	_assert(_memdbg_hastype(p, (MEMTYPE)(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
	_assert(tag || _memdbg_nottype(p, MEMTYPE_LOOKASIDE));
	return __allocsystem.Size(p);
}

// Free memory previously obtained from sqlite3Malloc().
__device__ void _freeG(void *p)
{
	if (!p) return; // IMP: R-49053-54554
	_assert(_memdbg_nottype(p, MEMTYPE_DB));
	_assert(_memdbg_hastype(p, MEMTYPE_HEAP));
	if (TagBase_RuntimeStatics.Memstat)
	{
		_mutex_enter(mem0.Mutex);
		_status_add(STATUS_MEMORY_USED, -(int)_allocsize(p));
		_status_add(STATUS_MALLOC_COUNT, -1);
		__allocsystem.Free(p);
		_mutex_leave(mem0.Mutex);
	}
	else
		__allocsystem.Free(p);
}

// Free memory that might be associated with a particular database connection.
__device__ void _tagfreeG(TagBase *tag, void *p)
{
	_assert(!tag || _mutex_held(tag->Mutex));
	if (tag)
	{
		if (tag->BytesFreed)
		{
			*tag->BytesFreed += (int)_tagallocsize(tag, p);
			return;
		}
		if (IsLookaside(tag, p))
		{
			TagBase::LookasideSlot *b = (TagBase::LookasideSlot *)p;
#if _DEBUG
			memset(p, (char)0xaa, tag->Lookaside.Size); // Trash all content in the buffer being freed
#endif
			b->Next = tag->Lookaside.Free;
			tag->Lookaside.Free = b;
			tag->Lookaside.Outs--;
			return;
		}
	}
	_assert(_memdbg_hastype(p, MEMTYPE_DB));
	_assert(_memdbg_hastype(p, (MEMTYPE)(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
	_assert(tag || _memdbg_nottype(p, MEMTYPE_LOOKASIDE));
	_memdbg_settype(p, MEMTYPE_HEAP);
	_freeG(p);
}

// Change the size of an existing memory allocation
__device__ void *_reallocG(void *old, size_t newSize)
{
	if (!old) return _allocG(newSize); /* IMP: R-28354-25769 */
	if (newSize <= 0) { _freeG(old); return nullptr; } // IMP: R-31593-10574
	if (newSize >= 0x7fffff00) return nullptr; // The 0x7ffff00 limit term is explained in comments on sqlite3Malloc()
	size_t oldSize = _allocsize(old);
	// IMPLEMENTATION-OF: R-46199-30249 SQLite guarantees that the second argument to xRealloc is always a value returned by a prior call to xRoundup.
	void *p;
	size_t newSize2 = __allocsystem.Roundup(newSize);
	if (oldSize == newSize2)
		p = old;
	else if (TagBase_RuntimeStatics.Memstat)
	{
		_mutex_enter(mem0.Mutex);
		_status_set(STATUS_MALLOC_SIZE, (int)newSize);
		size_t sizeDiff = newSize2 - oldSize;
		if (_status_value(STATUS_MEMORY_USED) >= mem0.AlarmThreshold-sizeDiff)
			__alloc_triggermemoryalarm(sizeDiff);
		_assert(_memdbg_hastype(old, MEMTYPE_HEAP));
		_assert(_memdbg_nottype(old, ~MEMTYPE_HEAP));
		p = __allocsystem.Realloc(old, newSize2);
		if (!p && mem0.AlarmCallback)
		{
			__alloc_triggermemoryalarm(newSize);
			p = __allocsystem.Realloc(old, newSize2);
		}
		if (p)
		{
			newSize2 = _allocsize(p);
			_status_add(STATUS_MEMORY_USED, (int)(newSize2-oldSize));
		}
		_mutex_leave(mem0.Mutex);
	}
	else
		p = __allocsystem.Realloc(old, newSize2);
	_assert(_HASALIGNMENT8(p)); // IMP: R-04675-44850
	return p;
}

// Allocate and zero memory.
__device__ void *_allocZeroG(size_t size)
{
	void *p = _allocG(size);
	if (p) memset(p, 0, size);
	return p;
}

// Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer.
__device__ void *_tagallocZeroG(TagBase *tag, size_t size)
{
	void *p = _tagallocG(tag, size);
	if (p) memset(p, 0, size);
	return p;
}

// Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer.
//
// If db!=0 and db->mallocFailed is true (indicating a prior malloc failure on the same database connection) then always return 0.
// Hence for a particular database connection, once malloc starts failing, it fails consistently until mallocFailed is reset.
// This is an important assumption.  There are many places in the code that do things like this:
//
//         int *a = (int*)sqlite3DbMallocRaw(db, 100);
//         int *b = (int*)sqlite3DbMallocRaw(db, 200);
//         if( b ) a[10] = 9;
//
// In other words, if a subsequent malloc (ex: "b") worked, it is assumed that all prior mallocs (ex: "a") worked too.
__device__ void *_tagallocG(TagBase *tag, size_t size)
{
	_assert(!tag || _mutex_held(tag->Mutex));
	_assert(!tag || tag->BytesFreed == 0);
#ifndef OMIT_LOOKASIDE
	if (tag)
	{
		TagBase::LookasideSlot *b;
		if (tag->MallocFailed) return nullptr;
		if (tag->Lookaside.Enabled)
		{
			if (size > tag->Lookaside.Size)
				tag->Lookaside.Stats[1]++;
			else if (!(b = tag->Lookaside.Free))
				tag->Lookaside.Stats[2]++;
			else
			{
				tag->Lookaside.Free = b->Next;
				tag->Lookaside.Outs++;
				tag->Lookaside.Stats[0]++;
				if (tag->Lookaside.Outs > tag->Lookaside.MaxOuts)
					tag->Lookaside.MaxOuts = tag->Lookaside.Outs;
				return (void *)b;
			}
		}
	}
#else
	if (tag && tag->MallocFailed) return nullptr;
#endif
	void *p = _allocG(size);
	if (!p && tag)
		tag->MallocFailed = true;
	_memdbg_settype(p, (MEMTYPE)(MEMTYPE_DB|(tag && tag->Lookaside.Enabled ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP)));
	return p;
}

// Resize the block of memory pointed to by old to size bytes. If the resize fails, set the mallocFailed flag in the connection object.
__device__ void *_tagreallocG(TagBase *tag, void *old, size_t size)
{
	void *p = nullptr;
	_assert(tag != nullptr);
	_assert(_mutex_held(tag->Mutex));
	if (!tag->MallocFailed)
	{
		if (!old) return _tagallocG(tag, size);
		if (IsLookaside(tag, old))
		{
			if (size <= tag->Lookaside.Size) return old;
			p = _tagallocG(tag, size);
			if (p)
			{
				memcpy(p, old, tag->Lookaside.Size);
				_tagfreeG(tag, old);
			}
		}
		else
		{
			_assert(_memdbg_hastype(old, MEMTYPE_DB));
			_assert(_memdbg_hastype(old, (MEMTYPE)(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
			_memdbg_settype(old, MEMTYPE_HEAP);
			p = _reallocG(old, size);
			if (!p)
			{
				_memdbg_settype(old, (MEMTYPE)(MEMTYPE_DB|MEMTYPE_HEAP));
				tag->MallocFailed = true;
			}
			_memdbg_settype(p, (MEMTYPE)(MEMTYPE_DB|(tag->Lookaside.Enabled ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP)));
		}
	}
	return p;
}

#ifdef OMIT_INLINEMEM
#if 0
// Attempt to reallocate p.  If the reallocation fails, then free p and set the mallocFailed flag in the database connection.
__device__ void *_tagrealloc_or_freeG(TagBase *tag, void *old, size_t newSize)
{
	void *p = _tagreallocG(tag, old, newSize);
	if (!p) _tagfreeG(tag, old);
	return p;
}
#endif

__device__ char *__strdupG(const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG((int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ char *_strndupG(const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG(n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}

__device__ char *_tagstrdupG(TagBase *tag, const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, (int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ char *_tagstrndupG(TagBase *tag, const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}
#endif

RUNTIME_NAMEEND