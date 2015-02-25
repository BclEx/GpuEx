#include <stdarg.h>
#include "Core.cu.h"

namespace Core
{
	bool IOTrace;

#pragma region Initialize/Shutdown/Config

#ifndef CORE_DEFAULT_MEMSTATUS
#define CORE_DEFAULT_MEMSTATUS true
#endif

#ifndef CORE_USE_URI
#define CORE_USE_URI false
#endif

	// The following singleton contains the global configuration for the SQLite library.
	__device__ _WSD SysEx::GlobalStatics g_GlobalStatics =
	{
		CORE_DEFAULT_MEMSTATUS,		// Memstat
		true,						// CoreMutex
#ifdef THREADSAFE
		true,						// FullMutex
#else
		false,						// FullMutex
#endif
		CORE_USE_URI,				// OpenUri
		// Main::UseCis
		0x7ffffffe,					// MaxStrlen
		128,						// LookasideSize
		500,						// Lookasides
		//{0,0,0,0,0,0,0,0},			// m
		//{0,0,0,0,0,0,0,0,0},		// mutex
		// pcache2
		//array_t(void *)nullptr, 0)// Heap
		//0, 0,						// MinHeap, MaxHeap
		(void *)nullptr,			// Scratch
		0,							// ScratchSize
		0,							// Scratchs
		// Main::Page
		// Main::PageSize
		// Main::Pages
		// Main::MaxParserStack
		false,						// SharedCacheEnabled
		// All the rest should always be initialized to zero
		false,						// IsInit
		false,						// InProgress
		false,						// IsMutexInit
		false,						// IsMallocInit
		// Main::IsPCacheInit
		false,						// InitMutex
		0,							// InitMutexRefs
		nullptr,					// Log
		0,							// LogArg
		false,						// LocaltimeFault
#ifdef ENABLE_SQLLOG
		nullptr,					// Sqllog
		0							// SqllogArg
#endif
	};

	__device__ RC SysEx::PreInitialize(MutexEx &masterMutex)
	{
		IOTrace = true;
		// If SQLite is already completely initialized, then this call to sqlite3_initialize() should be a no-op.  But the initialization
		// must be complete.  So isInit must not be set until the very end of this routine.
		if (SysEx_GlobalStatics.IsInit) return RC_OK;

		// The following is just a sanity check to make sure SQLite has been compiled correctly.  It is important to run this code, but
		// we don't want to run it too often and soak up CPU cycles for no reason.  So we run it once during initialization.
#if !defined(NDEBUG) && !defined(OMIT_FLOATING_POINT)
		// This section of code's only "output" is via assert() statements.
		uint64 x = (((uint64)1)<<63)-1;
		double y;
		_assert(sizeof(x) == 8);
		_assert(sizeof(x) == sizeof(y));
		_memcpy<void>(&y, &x, 8);
		_assert(_isnan(y));
#endif

		RC rc;
#ifdef OMIT_WSD
		rc = __wsdinit(4096, 24);
		if (rc != RC_OK) return rc;
#endif

#ifdef ENABLE_SQLLOG
		{
			extern void Init_Sqllog();
			Init_Sqllog();
		}
#endif

		// Make sure the mutex subsystem is initialized.  If unable to initialize the mutex subsystem, return early with the error.
		// If the system is so sick that we are unable to allocate a mutex, there is not much SQLite is going to be able to do.
		// The mutex subsystem must take care of serializing its own initialization.
		rc = MutexEx::Init();
		if (rc) return rc;

		// Initialize the malloc() system and the recursive pInitMutex mutex. This operation is protected by the STATIC_MASTER mutex.  Note that
		// MutexAlloc() is called for a static mutex prior to initializing the malloc subsystem - this implies that the allocation of a static
		// mutex must not require support from the malloc subsystem.
		masterMutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER); // The main static mutex
		MutexEx::Enter(masterMutex);
		SysEx_GlobalStatics.IsMutexInit = true;
		//if (!SysEx_GlobalStatics.IsMallocInit)
		//	rc = sqlite3MallocInit();
		if (rc == RC_OK)
		{
			SysEx_GlobalStatics.IsMallocInit = true;
			if (!SysEx_GlobalStatics.InitMutex.Tag)
			{
				SysEx_GlobalStatics.InitMutex = MutexEx::Alloc(MutexEx::MUTEX_RECURSIVE);
				if (SysEx_GlobalStatics.CoreMutex && !SysEx_GlobalStatics.InitMutex.Tag)
					rc = RC_NOMEM;
			}
		}
		if (rc == RC_OK)
			SysEx_GlobalStatics.InitMutexRefs++;
		MutexEx::Leave(masterMutex);

		// If rc is not SQLITE_OK at this point, then either the malloc subsystem could not be initialized or the system failed to allocate
		// the pInitMutex mutex. Return an error in either case.
		if (rc != RC_OK)
			return rc;

		// Do the rest of the initialization under the recursive mutex so that we will be able to handle recursive calls into
		// sqlite3_initialize().  The recursive calls normally come through sqlite3_os_init() when it invokes sqlite3_vfs_register(), but other
		// recursive calls might also be possible.
		//
		// IMPLEMENTATION-OF: R-00140-37445 SQLite automatically serializes calls to the xInit method, so the xInit method need not be threadsafe.
		//
		// The following mutex is what serializes access to the appdef pcache xInit methods.  The sqlite3_pcache_methods.xInit() all is embedded in the
		// call to sqlite3PcacheInitialize().
		MutexEx::Enter(SysEx_GlobalStatics.InitMutex);
		if (!SysEx_GlobalStatics.IsInit && !SysEx_GlobalStatics.InProgress)
		{
			SysEx_GlobalStatics.InProgress = true;
			rc = VSystem::Initialize();
		}
		if (rc != RC_OK)
			MutexEx::Leave(SysEx_GlobalStatics.InitMutex);
		return rc;
	}

	__device__ void SysEx::PostInitialize(MutexEx masterMutex)
	{
		MutexEx::Leave(SysEx_GlobalStatics.InitMutex);

		// Go back under the static mutex and clean up the recursive mutex to prevent a resource leak.
		MutexEx::Enter(masterMutex);
		SysEx_GlobalStatics.InitMutexRefs--;
		if (SysEx_GlobalStatics.InitMutexRefs <= 0)
		{
			_assert(SysEx_GlobalStatics.InitMutexRefs == 0);
			MutexEx::Free(SysEx_GlobalStatics.InitMutex);
			SysEx_GlobalStatics.InitMutex.Tag = nullptr;
		}
		MutexEx::Leave(masterMutex);
	}

	__device__ RC SysEx::Shutdown()
	{
		if (SysEx_GlobalStatics.IsInit)
		{
			VSystem::Shutdown();
			//sqlite3_reset_auto_extension();
			SysEx_GlobalStatics.IsInit = false;
		}
		//if (SysEx_GlobalStatics.IsMallocInit)
		//{
		//	sqlite3MallocEnd();
		//	SysEx_GlobalStatics.IsMallocInit = false;
		//}
		if (SysEx_GlobalStatics.IsMutexInit)
		{
			MutexEx::End();
			SysEx_GlobalStatics.IsMutexInit = false;
		}
		return RC_OK;
	}

	__device__ RC SysEx::Config(CONFIG op, va_list *args)
	{
		// sqlite3_config() shall return SQLITE_MISUSE if it is invoked while the SQLite library is in use.
		if (SysEx_GlobalStatics.IsInit) return SysEx_MISUSE_BKPT;
		RC rc = RC_OK;
		switch (op)
		{
#ifdef THREADSAFE
			// Mutex configuration options are only available in a threadsafe compile. 
		case CONFIG_SINGLETHREAD: { // Disable all mutexing
			SysEx_GlobalStatics.CoreMutex = false;
			SysEx_GlobalStatics.FullMutex = false;
			break; }
		case CONFIG_MULTITHREAD: { // Disable mutexing of database connections, Enable mutexing of core data structures
			SysEx_GlobalStatics.CoreMutex = true;
			SysEx_GlobalStatics.FullMutex = false;
			break; }
		case CONFIG_SERIALIZED: { // Enable all mutexing
			SysEx_GlobalStatics.CoreMutex = true;
			SysEx_GlobalStatics.FullMutex = true;
			break; }
		case CONFIG_MUTEX: { // Specify an alternative mutex implementation
			SysEx_GlobalStatics.Mutex = *va_arg(args, sqlite3_mutex_methods*);
			break; }
		case CONFIG_GETMUTEX: { // Retrieve the current mutex implementation
			*va_arg(args, sqlite3_mutex_methods*) = SysEx_GlobalStatics.Mutex;
			break; }
#endif
		case CONFIG_MALLOC: { // Specify an alternative malloc implementation
			//SysEx_GlobalStatics.m = *va_arg(args, sqlite3_mem_methods*);
			break; }
		case CONFIG_GETMALLOC: { // Retrieve the current malloc() implementation
			//if (SysEx_GlobalStatics.m.xMalloc==0) sqlite3MemSetDefault();
			//*va_arg(args, sqlite3_mem_methods*) = SysEx_GlobalStatics.m;
			break; }
		case CONFIG_MEMSTATUS: { // Enable or disable the malloc status collection
			SysEx_GlobalStatics.Memstat = va_arg(args, bool);
			break; }
		case CONFIG_SCRATCH: { // Designate a buffer for scratch memory space
			SysEx_GlobalStatics.Scratch = va_arg(args, void*);
			SysEx_GlobalStatics.ScratchSize = va_arg(args, int);
			SysEx_GlobalStatics.Scratchs = va_arg(args, int);
			break; }
#if defined(ENABLE_MEMSYS3) || defined(ENABLE_MEMSYS5)
		case CONFIG_HEAP: {
			// Designate a buffer for heap memory space
			SysEx_GlobalStatics.Heap.data = va_arg(args, void*);
			SysEx_GlobalStatics.Heap.length = va_arg(args, int);
			SysEx_GlobalStatics.MinReq = va_arg(ap, int);
			if (SysEx_GlobalStatics.MinReq < 1)
				SysEx_GlobalStatics.MinReq = 1;
			else if (SysEx_GlobalStatics.MinReq > (1<<12)) // cap min request size at 2^12
				SysEx_GlobalStatics.MinReq = (1<<12);
			if (!SysEx_GlobalStatics.Heap.data)
				// If the heap pointer is NULL, then restore the malloc implementation back to NULL pointers too.  This will cause the malloc to go back to its default implementation when sqlite3_initialize() is run.
					memset(&SysEx_GlobalStatics.m, 0, sizeof(SysEx_GlobalStatics.m));
			else
				// The heap pointer is not NULL, then install one of the mem5.c/mem3.c methods. If neither ENABLE_MEMSYS3 nor ENABLE_MEMSYS5 is defined, return an error.
#ifdef ENABLE_MEMSYS3
				SysEx_GlobalStatics.m = *sqlite3MemGetMemsys3();
#endif
#ifdef ENABLE_MEMSYS5
			SysEx_GlobalStatics.m = *sqlite3MemGetMemsys5();
#endif
			break; }
#endif
		case CONFIG_LOOKASIDE: {
			SysEx_GlobalStatics.LookasideSize = va_arg(args, int);
			SysEx_GlobalStatics.Lookasides = va_arg(args, int);
			break; }
		case CONFIG_LOG: { // Record a pointer to the logger function and its first argument. The default is NULL.  Logging is disabled if the function pointer is NULL.
			// MSVC is picky about pulling func ptrs from va lists.
			// http://support.microsoft.com/kb/47961
			// SysEx_GlobalStatics.xLog = va_arg(ap, void(*)(void*,int,const char*));
			typedef void(*LOGFUNC_t)(void*,int,const char*);
			SysEx_GlobalStatics.Log = va_arg(args, LOGFUNC_t);
			SysEx_GlobalStatics.LogArg = va_arg(args, void*);
			break; }
		case CONFIG_URI: {
			SysEx_GlobalStatics.OpenUri = va_arg(args, bool);
			break; }
#ifdef ENABLE_SQLLOG
		case CONFIG_SQLLOG: {
			typedef void (*SQLLOGFUNC_t)(void*,TagBase*,const char*,int);
			SysEx_GlobalStatics.Sqllog = va_arg(args, SQLLOGFUNC_t);
			SysEx_GlobalStatics.SqllogArg = va_arg(args, void*);
			break; }
#endif
		default: {
			rc = RC_ERROR;
			break; }
		}
		return rc;
	}

#pragma endregion

	__device__ RC SysEx::SetupLookaside(TagBase *tag, void *buf, int size, int count)
	{
		if (tag->Lookaside.Outs)
			return RC_BUSY;
		// Free any existing lookaside buffer for this handle before allocating a new one so we don't have to have space for both at the same time.
		if (tag->Lookaside.Malloced)
			_free(tag->Lookaside.Start);
		// The size of a lookaside slot after ROUNDDOWN8 needs to be larger than a pointer to be useful.
		size = _ROUNDDOWN8(size); // IMP: R-33038-09382
		if (size <= (int)sizeof(TagBase::LookasideSlot *)) size = 0;
		if (count < 0) count = 0;
		void *start;
		if (size == 0 || count == 0)
		{
			size = 0;
			start = nullptr;
		}
		else if (!buf)
		{
			_benignalloc_begin();
			start = _alloc(size * count); // IMP: R-61949-35727
			_benignalloc_end();
			if (start) count = (int)_allocsize(start) / size;
		}
		else
			start = buf;
		tag->Lookaside.Start = start;
		tag->Lookaside.Free = nullptr;
		tag->Lookaside.Size = (uint16)size;
		if (start)
		{
			_assert(size > (int)sizeof(TagBase::LookasideSlot *));
			TagBase::LookasideSlot *p = (TagBase::LookasideSlot *)start;
			for (int i = count - 1; i >= 0; i--)
			{
				p->Next = tag->Lookaside.Free;
				tag->Lookaside.Free = p;
				p = (TagBase::LookasideSlot *)&((uint8 *)p)[size];
			}
			tag->Lookaside.End = p;
			tag->Lookaside.Enabled = true;
			tag->Lookaside.Malloced = (!buf);
		}
		else
		{
			tag->Lookaside.End = nullptr;
			tag->Lookaside.Enabled = false;
			tag->Lookaside.Malloced = false;
		}
		return RC_OK;
	}
}
