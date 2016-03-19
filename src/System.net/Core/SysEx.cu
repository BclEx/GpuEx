#include <stdarg.h>
#include "Core.cu.h"

namespace CORE_NAME
{
	__device__ bool SysTrace = false;
	__device__ bool IOTrace = false;

#pragma region Initialize/Shutdown/Config

#ifndef CORE_USE_URI
#define CORE_USE_URI false
#endif

	// The following singleton contains the global configuration for the SQLite library.
	__device__ _WSD SysEx::GlobalStatics g_GlobalStatics =
	{
		// TAG::CoreMutex
		// TAG::FullMutex
		CORE_USE_URI,				// OpenUri
		// DataEx::UseCis
		0x7ffffffe,					// MaxStrlen
		//{0,0,0,0,0,0,0,0,0},		// mutex
		// pcache2
		//array_t(void *)nullptr, 0)// Heap
		//0, 0,						// MinHeap, MaxHeap
		// DataEx::Page
		// DataEx::PageSize
		// DataEx::Pages
		// DataEx::MaxParserStack
		false,						// SharedCacheEnabled
		// All the rest should always be initialized to zero
		false,						// IsInit
		false,						// InProgress
		false,						// IsMutexInit
		false,						// IsMallocInit
		// DataEx::IsPCacheInit
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
#if _DEBUG
		//SysTrace = true;
		//IOTrace = true;
#endif
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
		memcpy(&y, &x, 8);
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
		rc = (RC)_mutex_init();
		if (rc) return rc;

		// Initialize the malloc() system and the recursive pInitMutex mutex. This operation is protected by the STATIC_MASTER mutex.  Note that
		// MutexAlloc() is called for a static mutex prior to initializing the malloc subsystem - this implies that the allocation of a static
		// mutex must not require support from the malloc subsystem.
		masterMutex = _mutex_alloc(MUTEX_STATIC_MASTER); // The main static mutex
		_mutex_enter(masterMutex);
		SysEx_GlobalStatics.IsMutexInit = true;
		if (!SysEx_GlobalStatics.IsMallocInit)
			rc = (RC)_alloc_init();
		if (rc == RC_OK)
		{
			SysEx_GlobalStatics.IsMallocInit = true;
			if (!SysEx_GlobalStatics.InitMutex)
			{
				SysEx_GlobalStatics.InitMutex = _mutex_alloc(MUTEX_RECURSIVE);
				if (TagBase_RuntimeStatics.CoreMutex && !SysEx_GlobalStatics.InitMutex)
					rc = RC_NOMEM;
			}
		}
		if (rc == RC_OK)
			SysEx_GlobalStatics.InitMutexRefs++;
		_mutex_leave(masterMutex);

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
		_mutex_enter(SysEx_GlobalStatics.InitMutex);
		if (!SysEx_GlobalStatics.IsInit && !SysEx_GlobalStatics.InProgress)
		{
			SysEx_GlobalStatics.InProgress = true;
			rc = VSystem::Initialize();
		}
		if (rc != RC_OK) { _mutex_leave(SysEx_GlobalStatics.InitMutex); }
		return rc;
	}

	__device__ void SysEx::PostInitialize(MutexEx masterMutex)
	{
		_mutex_leave(SysEx_GlobalStatics.InitMutex);

		// Go back under the static mutex and clean up the recursive mutex to prevent a resource leak.
		_mutex_enter(masterMutex);
		SysEx_GlobalStatics.InitMutexRefs--;
		if (SysEx_GlobalStatics.InitMutexRefs <= 0)
		{
			_assert(SysEx_GlobalStatics.InitMutexRefs == 0);
			_mutex_free(SysEx_GlobalStatics.InitMutex);
			SysEx_GlobalStatics.InitMutex = nullptr;
		}
		_mutex_leave(masterMutex);
	}

	__device__ RC SysEx::Shutdown()
	{
		if (SysEx_GlobalStatics.IsInit)
		{
			VSystem::Shutdown();
			//sqlite3_reset_auto_extension();
			SysEx_GlobalStatics.IsInit = false;
		}
		if (SysEx_GlobalStatics.IsMallocInit)
		{
			_alloc_shutdown();
			SysEx_GlobalStatics.IsMallocInit = false;
		}
		if (SysEx_GlobalStatics.IsMutexInit)
		{
			_mutex_shutdown();
			SysEx_GlobalStatics.IsMutexInit = false;
		}
		return RC_OK;
	}

	__device__ RC SysEx::Config_(CONFIG op, _va_list &args)
	{
		// sqlite3_config() shall return SQLITE_MISUSE if it is invoked while the SQLite library is in use.
		if (SysEx_GlobalStatics.IsInit) return SysEx_MISUSE_BKPT;
		RC rc = RC_OK;
		switch (op)
		{
#ifdef THREADSAFE
			// Mutex configuration options are only available in a threadsafe compile. 
		case CONFIG_SINGLETHREAD: { // Disable all mutexing
			TagBase_RuntimeStatics.CoreMutex = false;
			TagBase_RuntimeStatics.FullMutex = false;
			break; }
		case CONFIG_MULTITHREAD: { // Disable mutexing of database connections, Enable mutexing of core data structures
			TagBase_RuntimeStatics.CoreMutex = true;
			TagBase_RuntimeStatics.FullMutex = false;
			break; }
		case CONFIG_SERIALIZED: { // Enable all mutexing
			TagBase_RuntimeStatics.CoreMutex = true;
			TagBase_RuntimeStatics.FullMutex = true;
			break; }
		case CONFIG_MUTEX: { // Specify an alternative mutex implementation
			__mutexsystem = *_va_arg(args, _mutex_methods *);
			break; }
		case CONFIG_GETMUTEX: { // Retrieve the current mutex implementation
			*_va_arg(args, _mutex_methods *) = __mutexsystem;
			break; }
#endif
		case CONFIG_MALLOC: { // Specify an alternative malloc implementation
			__allocsystem = *_va_arg(args, _mem_methods *);
			break; }
		case CONFIG_GETMALLOC: { // Retrieve the current malloc() implementation
			if (!__allocsystem.Alloc) __allocsystem_setdefault();
			*_va_arg(args, _mem_methods *) = __allocsystem;
			break; }
		case CONFIG_MEMSTATUS: { // Enable or disable the malloc status collection
			TagBase_RuntimeStatics.Memstat = _va_arg(args, bool);
			break; }
		case CONFIG_SCRATCH: { // Designate a buffer for scratch memory space
			TagBase_RuntimeStatics.Scratch = _va_arg(args, void*);
			TagBase_RuntimeStatics.ScratchSize = _va_arg(args, int);
			TagBase_RuntimeStatics.Scratchs = _va_arg(args, int);
			break; }
#if defined(ENABLE_MEMSYS3) || defined(ENABLE_MEMSYS5)
		case CONFIG_HEAP: {
			// Designate a buffer for heap memory space
			SysEx_GlobalStatics.Heap.data = _va_arg(args, void*);
			SysEx_GlobalStatics.Heap.length = _va_arg(args, int);
			SysEx_GlobalStatics.MinReq = _va_arg(ap, int);
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
			TagBase_RuntimeStatics.LookasideSize = _va_arg(args, int);
			TagBase_RuntimeStatics.Lookasides = _va_arg(args, int);
			break; }
		case CONFIG_LOG: { // Record a pointer to the logger function and its first argument. The default is NULL.  Logging is disabled if the function pointer is NULL.
			// MSVC is picky about pulling func ptrs from va lists.
			// http://support.microsoft.com/kb/47961
			// SysEx_GlobalStatics.xLog = _va_arg(ap, void(*)(void*,int,const char*));
			typedef void(*LOGFUNC_t)(void*,int,const char*);
			SysEx_GlobalStatics.Log = _va_arg(args, LOGFUNC_t);
			SysEx_GlobalStatics.LogArg = _va_arg(args, void*);
			break; }
		case CONFIG_URI: {
			SysEx_GlobalStatics.OpenUri = _va_arg(args, bool);
			break; }
#ifdef ENABLE_SQLLOG
		case CONFIG_SQLLOG: {
			typedef void (*SQLLOGFUNC_t)(void*,TagBase*,const char*,int);
			SysEx_GlobalStatics.Sqllog = _va_arg(args, SQLLOGFUNC_t);
			SysEx_GlobalStatics.SqllogArg = _va_arg(args, void*);
			break; }
#endif
		default: {
			rc = RC_ERROR;
			break; }
		}
		return rc;
	}

#pragma endregion

#pragma	region Tests
#ifdef _TEST

#define SETBIT(V,I) V[(I) >> 3] |= (1 << ((I) & 7))
#define CLEARBIT(V,I) V[(I) >> 3] &= ~(1 << ((I) & 7))
#define TESTBIT(V,I) ((V[(I) >> 3] & (1 << ((I) & 7))) != 0)

	__device__ int Bitvec_BuiltinTest(int size, int *ops)
	{
		int rc = -1;
		// Allocate the Bitvec to be tested and a linear array of bits to act as the reference
		Bitvec *bitvec = Bitvec::New(size);
		unsigned char *v = (unsigned char *)_allocZero((size + 7) / 8 + 1);
		void *tmpSpace = _alloc(BITVEC_SZ);
		int pc = 0;
		int i, nx, op;
		if (!bitvec || !v || !tmpSpace)
			goto bitvec_end;

		// Run the program
		while ((op = ops[pc]))
		{
			switch (op)
			{
			case 1:
			case 2:
			case 5:
				{
					nx = 4;
					i = ops[pc + 2] - 1;
					ops[pc + 2] += ops[pc + 3];
					break;
				}
			case 3:
			case 4: 
			default:
				{
					nx = 2;
					SysEx::PutRandom(sizeof(i), &i);
					break;
				}
			}
			if ((--ops[pc + 1]) > 0) nx = 0;
			pc += nx;
			i = (i & 0x7fffffff) % size;
			if (op & 1)
			{
				SETBIT(v, i + 1);
				if (op != 5)
					if (bitvec->Set(i + 1)) goto bitvec_end;
			}
			else
			{
				CLEARBIT(v, i + 1);
				bitvec->Clear(i + 1, tmpSpace);
			}
		}

		// Test to make sure the linear array exactly matches the Bitvec object.  Start with the assumption that they do
		// match (rc==0).  Change rc to non-zero if a discrepancy is found.
		rc = bitvec->Get(size + 1)
			+ bitvec->Get(0)
			+ (bitvec->get_Length() - size);
		for (i = 1; i <= size; i++)
		{
			if (TESTBIT(v, i) != bitvec->Get(i))
			{
				rc = i;
				break;
			}
		}

		// Free allocated structure
bitvec_end:
		_free(tmpSpace);
		_free(v);
		Bitvec::Destroy(bitvec);
		return rc;
	}

#endif
#pragma endregion
}
