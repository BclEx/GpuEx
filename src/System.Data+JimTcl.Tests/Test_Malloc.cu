// This file contains code used to implement test interfaces to the memory allocation subsystem.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>

// This structure is used to encapsulate the global state variables used by malloc() fault simulation.
static struct MemFault
{
	int Countdown;			// Number of pending successes before a failure
	int Repeats;			// Number of times to repeat the failure
	int Benigns;			// Number of benign failures seen since last config
	int Fails;				// Number of failures seen since last config
	bool Enable;            // True if enabled
	bool IsInstalled;		// True if the fault simulation layer is installed
	int IsBenignMode;		// True if malloc failures are considered benign
	_mem_methods m;			// 'Real' malloc implementation
} _memfault;

// This routine exists as a place to set a breakpoint that will fire on any simulated malloc() failure.
__device__ static int _fault_cnt = 0;
__device__ static void onfault() { _fault_cnt++; }

// Check to see if a fault should be simulated.  Return true to simulate the fault.  Return false if the fault should not be simulated.
__device__ static bool faultsimStep()
{
	if (likely(!_memfault.Enable))
		return false;
	if (_memfault.Countdown > 0)
	{
		_memfault.Countdown--;
		return false;
	}
	onfault();
	_memfault.Fails++;
	if (_memfault.IsBenignMode > 0)
		_memfault.Benigns++;
	_memfault.Repeats--;
	if (_memfault.Repeats <= 0)
		_memfault.Enable = false;
	return true;  
}

// A version of sqlite3_mem_methods.xMalloc() that includes fault simulation logic.
__device__ static void *faultsimAlloc(size_t n) { return (!faultsimStep() ? _memfault.m.Alloc(n): nullptr); }

// A version of sqlite3_mem_methods.xRealloc() that includes fault simulation logic.
__device__ static void *faultsimRealloc(void *old, size_t n) { return (!faultsimStep() ? _memfault.m.Realloc(old, n) : nullptr); }

// The following method calls are passed directly through to the underlying malloc system:
//     xFree
//     xSize
//     xRoundup
//     xInit
//     xShutdown
__device__ static void faultsimFree(void *p) { _memfault.m.Free(p); }
__device__ static size_t faultsimSize(void *p) { return _memfault.m.Size(p); }
__device__ static size_t faultsimRoundup(size_t n) { return _memfault.m.Roundup(n); }
__device__ static int faultsimInit(void *p) { return _memfault.m.Init(_memfault.m.AppData); }
__device__ static void faultsimShutdown(void *p) { _memfault.m.Shutdown(_memfault.m.AppData); }

// This routine configures the malloc failure simulation.  After calling this routine, the next nDelay mallocs will succeed, followed
// by a block of nRepeat failures, after which malloc() calls will begin to succeed again.
__device__ static void faultsimConfig(int delay, int repeats)
{
	_memfault.Countdown = delay;
	_memfault.Repeats = repeats;
	_memfault.Benigns = 0;
	_memfault.Fails = 0;
	_memfault.Enable = (delay >= 0);
	// Sometimes, when running multi-threaded tests, the isBenignMode variable is not properly incremented/decremented so that it is
	// 0 when not inside a benign malloc block. This doesn't affect the multi-threaded tests, as they do not use this system. But
	// it does affect OOM tests run later in the same process. So zero the variable here, just to be sure.
	_memfault.IsBenignMode = 0;
}

// Return the number of faults (both hard and benign faults) that have occurred since the injector was last configured.
__device__ static int faultsimFailures() { return _memfault.Fails; }
// Return the number of benign faults that have occurred since the injector was last configured.
__device__ static int faultsimBenignFailures() { return _memfault.Benigns; }
// Return the number of successes that will occur before the next failure. If no failures are scheduled, return -1.
__device__ static int faultsimPending() { return (_memfault.Enable ? _memfault.Countdown : -1); }
__device__ static void faultsimBeginBenign() { _memfault.IsBenignMode++;}
__device__ static void faultsimEndBenign() { _memfault.IsBenignMode--; }

// Add or remove the fault-simulation layer using sqlite3_config(). If the argument is non-zero, the 
__device__ static _mem_methods _m = {
	faultsimAlloc,					// xMalloc
	faultsimFree,					// xFree
	faultsimRealloc,				// xRealloc
	faultsimSize,					// xSize
	faultsimRoundup,				// xRoundup
	faultsimInit,					// xInit
	faultsimShutdown,				// xShutdown
	nullptr                         // pAppData
};

__device__ static RC faultsimInstall(bool install)
{
	if (install == _memfault.IsInstalled)
		return RC_ERROR;
	RC rc;
	if (install)
	{
		rc = SysEx::Config(SysEx::CONFIG_GETMALLOC, &_memfault.m);
		_assert(_memfault.m.Alloc);
		if (rc == RC_OK)
			rc = SysEx::Config(SysEx::CONFIG_MALLOC, &_m);
		DataEx::TestControl(DataEx::TESTCTRL_BENIGN_MALLOC_HOOKS, faultsimBeginBenign, faultsimEndBenign);
	}
	else
	{
		_mem_methods m;
		_assert(_memfault.m.Alloc);
		// One should be able to reset the default memory allocator by storing a zeroed allocator then calling GETMALLOC. */
		_memset(&m, 0, sizeof(m));
		SysEx::Config(SysEx::CONFIG_MALLOC, &m);
		SysEx::Config(SysEx::CONFIG_GETMALLOC, &m);
		_assert(!_memcmp(&m, &_memfault.m, sizeof(m)));
		rc = SysEx::Config(SysEx::CONFIG_MALLOC, &_memfault.m);
		DataEx::TestControl(DataEx::TESTCTRL_BENIGN_MALLOC_HOOKS, nullptr, nullptr);
	}
	if (rc == RC_OK)
		_memfault.IsInstalled = true;
	return rc;
}

#ifdef _TEST

// This function is implemented in test1.c. Returns a pointer to a static buffer containing the symbolic SQLite error code that corresponds to
// the least-significant 8-bits of the integer passed as an argument.
// For example:
//   sqlite3TestErrorName(1) -> "SQLITE_ERROR"
__device__ const char *sqlite3TestErrorName(int);

// Transform pointers to text and back again
__device__ static void pointerToText(void *p, char *z)
{
	const char hexs[] = "0123456789abcdef";
	if (p == 0)
	{
		_strcpy(z, "0");
		return;
	}
	unsigned int u;
	uint64 n;
	if (sizeof(n) == sizeof(p)) _memcpy(&n, &p, sizeof(p));
	else if (sizeof(u) == sizeof(p)) { memcpy(&u, &p, sizeof(u)); n = u; }
	else _assert(false);
	int i, k;
	for (i = 0, k = sizeof(p)*2-1; i < sizeof(p)*2; i++, k--) { z[k] = hexs[n&0xf]; n >>= 4; }
	z[sizeof(p)*2] = 0;
}
__device__ static int hexToInt(int h)
{
	if (h >= '0' && h <= '9') return h - '0';
	else if (h >= 'a' && h <= 'f') return h - 'a' + 10;
	else return -1;
}
__device__ static int textToPointer(const char *z, void **p)
{
	uint64 n = 0;
	unsigned int u;
	for (int i = 0; i < sizeof(void*)*2 && z[0]; i++)
	{
		int v = hexToInt(*z++);
		if (v < 0) return JIM_ERROR;
		n = n*16 + v;
	}
	if (*z != 0) return JIM_ERROR;
	if (sizeof(n) == sizeof(*p)) _memcpy(p, &n, sizeof(n));
	else if (sizeof(u) == sizeof(*p)) { u = (unsigned int)n; _memcpy(p, &u, sizeof(u)); }
	else _assert(false);
	return JIM_OK;
}

// Usage:    sqlite3_malloc  NBYTES
// Raw test interface for sqlite3_malloc().
__device__ static int test_malloc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "NBYTES");
		return JIM_ERROR;
	}
	int bytes;
	if (Jim_GetInt(interp, args[1], &bytes)) return JIM_ERROR;
	void *p = _alloc((unsigned)bytes);
	char out[100];
	pointerToText(p, out);
	Jim_AppendResult(interp, out, nullptr);
	return JIM_OK;
}

// Usage:    sqlite3_realloc  PRIOR  NBYTES
// Raw test interface for sqlite3_realloc().
__device__ static int test_realloc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "PRIOR NBYTES");
		return JIM_ERROR;
	}
	int bytes;
	void *prior;
	if (Jim_GetInt(interp, args[2], &bytes)) return JIM_ERROR;
	if (textToPointer(Jim_String(args[1]), &prior))
	{
		Jim_AppendResult(interp, "bad pointer: ", Jim_String(args[1]), nullptr);
		return JIM_ERROR;
	}
	void *p = _realloc(prior, (unsigned)bytes);
	char out[100];
	pointerToText(p, out);
	Jim_AppendResult(interp, out, nullptr);
	return JIM_OK;
}

// Usage:    sqlite3_free  PRIOR
// Raw test interface for sqlite3_free().
__device__ static int test_free(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "PRIOR");
		return JIM_ERROR;
	}
	void *prior;
	if (textToPointer(Jim_String(args[1]), &prior))
	{
		Jim_AppendResult(interp, "bad pointer: ", Jim_String(args[1]), nullptr);
		return JIM_ERROR;
	}
	_free(prior);
	return JIM_OK;
}

// These routines are in test_hexio.c
__device__ void sqlite3TestBinToHex(char *buf, int value);
__device__ int sqlite3TestHexToBin(const char *in_, int value, char *out_);

// Usage:    memset  ADDRESS  SIZE  HEX
// Set a chunk of memory (obtained from malloc, probably) to a specified hex pattern.
__device__ static int test_memset(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 4)
	{
		Jim_WrongNumArgs(interp, 1, args, "ADDRESS SIZE HEX");
		return JIM_ERROR;
	}
	void *p;
	if (textToPointer(Jim_String(args[1]), &p))
	{
		Jim_AppendResult(interp, "bad pointer: ", Jim_String(args[1]), nullptr);
		return JIM_ERROR;
	}
	int size;
	if (Jim_GetInt(interp, args[2], &size))
		return JIM_ERROR;
	if (size <= 0)
	{
		Jim_AppendResult(interp, "size must be positive", nullptr);
		return JIM_ERROR;
	}
	int n;
	const char *hex = Jim_GetString(args[3], &n);
	char bin[100];
	if (n > sizeof(bin)*2) n = sizeof(bin)*2;
	n = sqlite3TestHexToBin(hex, n, bin);
	if (n == 0)
	{
		Jim_AppendResult(interp, "no data", nullptr);
		return JIM_ERROR;
	}
	char *out = (char *)p;
	for (int i = 0; i < size; i++)
		out[i] = bin[i%n];
	return JIM_OK;
}

// Usage:    memget  ADDRESS  SIZE
// Return memory as hexadecimal text.
__device__ static int test_memget(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "ADDRESS SIZE");
		return JIM_ERROR;
	}
	void *p;
	if (textToPointer(Jim_String(args[1]), &p))
	{
		Jim_AppendResult(interp, "bad pointer: ", Jim_String(args[1]), nullptr);
		return JIM_ERROR;
	}
	int size;
	if (Jim_GetInt(interp, args[2], &size))
		return JIM_ERROR;
	if (size <= 0)
	{
		Jim_AppendResult(interp, "size must be positive", nullptr);
		return JIM_ERROR;
	}
	char *bin = (char *)p;
	char hex[100];
	int n;
	while (size > 0)
	{
		n = (size > (sizeof(hex)-1)/2 ? (sizeof(hex)-1)/2 : size);
		_memcpy(hex, bin, n);
		bin += n;
		size -= n;
		sqlite3TestBinToHex(hex, n);
		Jim_AppendResult(interp, hex, nullptr);
	}
	return JIM_OK;
}

// Usage:    sqlite3_memory_used
// Raw test interface for sqlite3_memory_used().
__device__ static int test_memory_used(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	Jim_SetResult(interp, Jim_NewWideObj(interp, __alloc_memoryused()));
	return JIM_OK;
}

// Usage:    sqlite3_memory_highwater ?RESETFLAG?
// Raw test interface for sqlite3_memory_highwater().
__device__ static int test_memory_highwater(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1 && argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "?RESET?");
		return JIM_ERROR;
	}
	bool resetFlag = false;
	if (argc == 2)
	{
		if (Jim_GetBoolean(interp, args[1], &resetFlag)) return JIM_ERROR;
	} 
	Jim_SetResult(interp, Jim_NewWideObj(interp, __alloc_memoryhighwater(resetFlag)));
	return JIM_OK;
}

// Usage:    sqlite3_memdebug_backtrace DEPTH
// Set the depth of backtracing.  If SQLITE_MEMDEBUG is not defined then this routine is a no-op.
__device__ static int test_memdebug_backtrace(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DEPT");
		return JIM_ERROR;
	}
	int depth;
	if (Jim_GetInt(interp, args[1], &depth)) return JIM_ERROR;
#ifdef MEMDEBUG
	{
		extern void sqlite3MemdebugBacktrace(int);
		sqlite3MemdebugBacktrace(depth);
	}
#endif
	return JIM_OK;
}

// Usage:    sqlite3_memdebug_dump  FILENAME
// Write a summary of unfreed memory to FILENAME.
__device__ static int test_memdebug_dump(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "FILENAME");
		return JIM_ERROR;
	}
#if defined(MEMDEBUG) || defined(MEMORY_SIZE) || defined(POW2_MEMORY_SIZE)
	{
		extern void sqlite3MemdebugDump(const char*);
		sqlite3MemdebugDump(Jim_String(args[1]));
	}
#endif
	return JIM_OK;
}

// Usage:    sqlite3_memdebug_malloc_count
// Return the total number of times malloc() has been called.
__device__ static int test_memdebug_malloc_count(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	int malloc = -1;
#if defined(MEMDEBUG)
	{
		extern int sqlite3MemdebugMallocCount();
		malloc = sqlite3MemdebugMallocCount();
	}
#endif
	Jim_SetResult(interp, Jim_NewIntObj(interp, malloc));
	return JIM_OK;
}


// Usage:    sqlite3_memdebug_fail  COUNTER  ?OPTIONS?
// where options are:
//     -repeat    <count>
//     -benigncnt <varname>
//
// Arrange for a simulated malloc() failure after COUNTER successes. If a repeat count is specified, the fault is repeated that many times.
//
// Each call to this routine overrides the prior counter value. This routine returns the number of simulated failures that have
// happened since the previous call to this routine.
//
// To disable simulated failures, use a COUNTER of -1.
__device__ static int test_memdebug_fail(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc < 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "COUNTER ?OPTIONS?");
		return JIM_ERROR;
	}
	int fail;
	if (Jim_GetInt(interp, args[1], &fail)) return JIM_ERROR;
	int repeats = 1;
	Jim_Obj *benignCnt = nullptr;
	for (int ii = 2; ii < argc; ii +=2 )
	{
		int optionLength;
		const char *option = Jim_GetString(args[ii], &optionLength);
		char *err = nullptr;
		if (optionLength>1 && !_strncmp(option, "-repeat", optionLength))
		{
			if (ii == (argc-1))
				err = "option requires an argument: ";
			else if (Jim_GetInt(interp, args[ii+1], &repeats))
				return JIM_ERROR;
		}
		else if (optionLength > 1 && !_strncmp(option, "-benigncnt", optionLength))
		{
			if (ii == (argc-1))
				err = "option requires an argument: ";
			else
				benignCnt = args[ii+1];
		}
		else
			err = "unknown option: ";
		if (err)
		{
			Jim_AppendResult(interp, err, option, nullptr);
			return JIM_ERROR;
		}
	}
	int benigns = faultsimBenignFailures();
	int fails = faultsimFailures();
	faultsimConfig(fail, repeats);
	if (benignCnt)
		Jim_SetVariable(interp, benignCnt, Jim_NewIntObj(interp, benigns));
	Jim_SetResult(interp, Jim_NewIntObj(interp, fails));
	return JIM_OK;
}

// Usage:    sqlite3_memdebug_pending
// Return the number of malloc() calls that will succeed before a simulated failure occurs. A negative return value indicates that
// no malloc() failure is scheduled.
__device__ static int test_memdebug_pending(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	int pending = faultsimPending();
	Jim_SetResult(interp, Jim_NewIntObj(interp, pending));
	return JIM_OK;
}


// Usage:    sqlite3_memdebug_settitle TITLE
// Set a title string stored with each allocation.  The TITLE is typically the name of the test that was running when the
// allocation occurred.  The TITLE is stored with the allocation and can be used to figure out which tests are leaking memory.
// Each title overwrite the previous.
__device__ static int test_memdebug_settitle(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "TITLE");
		return JIM_ERROR;
	}
#ifdef MEMDEBUG
	{
		__device__ extern int sqlite3MemdebugSettitle(const char*);
		const char *title = Jim_String(args[1]);
		sqlite3MemdebugSettitle(title);
	}
#endif
	return JIM_OK;
}

#define MALLOC_LOG_FRAMES  10 
#define MALLOC_LOG_KEYINTS (10*((sizeof(int)>=sizeof(void*))?1:sizeof(void*)/sizeof(int)))
__device__ static Jim_HashTable _mallocLog;
__device__ static bool _mallocLogEnabled = false;

typedef struct MallocLog MallocLog;
struct MallocLog
{
	int Calls;
	int Bytes;
};

#ifdef MEMDEBUG
__device__ static void test_memdebug_callback(int bytes, int framesLength, void **frames)
{
	if (_mallocLogEnabled)
	{
		int keys[MALLOC_LOG_KEYINTS];
		int keysLength = sizeof(int)*MALLOC_LOG_KEYINTS;
		_memset(keys, 0, keysLength);
		if ((sizeof(void*)*framesLength) < keysLength)
			keysLength = framesLength*sizeof(void*);
		_memcpy(keys, frames, keysLength);
		int isNew;
		Jim_HashEntry *entry = Jim_CreateHashEntry(&_mallocLog, (const char *)keys, &isNew);
		MallocLog *log;
		if (isNew)
		{
			log = (MallocLog *)Jim_Alloc(sizeof(MallocLog));
			_memset(log, 0, sizeof(MallocLog));
			Jim_SetHashVal(&_mallocLog, entry, (ClientData)log);
		}
		else
			log = (MallocLog *)Jim_GetHashEntryVal(entry);
		log->Calls++;
		log->Bytes += bytes;
	}
}
#endif

__device__ static void test_memdebug_log_clear()
{
	Jim_HashTableIterator *iter = Jim_GetHashTableIterator(&_mallocLog);
	for (Jim_HashEntry *entry = Jim_NextHashEntry(iter); entry; entry = Jim_NextHashEntry(iter))
	{
		MallocLog *log = (MallocLog *)Jim_GetHashEntryVal(entry);
		Jim_Free((char *)log);
	}
	Jim_FreeHashTable(&_mallocLog);
	Jim_InitHashTable(&_mallocLog, nullptr, nullptr); //MALLOC_LOG_KEYINTS);
}

__device__ static bool _log_isInit = false;
__device__ static const char *_log_MB_strs[] = { "start", "stop", "dump", "clear", "sync" };
__device__ static int test_memdebug_log(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	enum MB_enum { 
		MB_LOG_START, MB_LOG_STOP, MB_LOG_DUMP, MB_LOG_CLEAR, MB_LOG_SYNC 
	};
	if (!_log_isInit)
	{
#ifdef MEMDEBUG
		__device__ extern void sqlite3MemdebugBacktraceCallback(void (*backtrace)(int,int,void **));
		sqlite3MemdebugBacktraceCallback(test_memdebug_callback);
#endif
		Jim_InitHashTable(&_mallocLog, nullptr, nullptr); //, MALLOC_LOG_KEYINTS);
		_log_isInit = true;
	}
	if (argc < 2)
		Jim_WrongNumArgs(interp, 1, args, "SUB-COMMAND ...");
	int sub;
	if (Jim_GetEnum(interp, args[1], _log_MB_strs, &sub, "sub-command", 0))
		return JIM_ERROR;
	switch ((enum MB_enum)sub)
	{
	case MB_LOG_START:
		_mallocLogEnabled = true;
		break;
	case MB_LOG_STOP:
		_mallocLogEnabled = false;
		break;
	case MB_LOG_DUMP: {
		_assert(sizeof(jim_wide) >= sizeof(void*));
		Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
		Jim_HashTableIterator *iter = Jim_GetHashTableIterator(&_mallocLog);
		for (Jim_HashEntry *entry = Jim_NextHashEntry(iter); entry; entry = Jim_NextHashEntry(iter))
		{
			Jim_Obj *elems[MALLOC_LOG_FRAMES+2];
			MallocLog *log = (MallocLog *)Jim_GetHashEntryVal(entry);
			jim_wide *keys = (jim_wide *)Jim_GetHashEntryKey(entry);
			elems[0] = Jim_NewIntObj(interp, log->Calls);
			elems[1] = Jim_NewIntObj(interp, log->Bytes);
			for (int ii = 0; ii < MALLOC_LOG_FRAMES; ii++)
				elems[ii+2] = Jim_NewWideObj(interp, keys[ii]);
			Jim_ListAppendElement(interp, r, Jim_NewListObj(interp, elems, MALLOC_LOG_FRAMES+2));
		}
		Jim_SetResult(interp, r);
		break; }
	case MB_LOG_CLEAR: {
		test_memdebug_log_clear();
		break; }
	case MB_LOG_SYNC: {
#ifdef MEMDEBUG
		extern void sqlite3MemdebugSync();
		test_memdebug_log_clear();
		mallocLogEnabled = 1;
		sqlite3MemdebugSync();
#endif
		break; }
	}
	return JIM_OK;
}

// Usage:    sqlite3_config_scratch SIZE N
// Set the scratch memory buffer using SQLITE_CONFIG_SCRATCH. The buffer is static and is of limited size.  N might be
// adjusted downward as needed to accomodate the requested size. The revised value of N is returned.
// A negative SIZE causes the buffer pointer to be NULL.
__device__ static char *_scratch_buf = nullptr;
__device__ static int test_config_scratch(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "SIZE N");
		return JIM_ERROR;
	}
	int size, n;
	if (Jim_GetInt(interp, args[1], &size)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], &n)) return JIM_ERROR;
	free(_scratch_buf);
	RC rc;
	if (size < 0)
	{
		_scratch_buf = nullptr;
		rc = SysEx::Config(SysEx::CONFIG_SCRATCH, nullptr, 0, 0);
	}
	else
	{
		_scratch_buf = (char *)malloc(size*n + 1);
		rc = SysEx::Config(SysEx::CONFIG_SCRATCH, _scratch_buf, size, n);
	}
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, rc));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, n));
	Jim_SetResult(interp, r);
	return JIM_OK;
}

// Usage:    sqlite3_config_pagecache SIZE N
// Set the page-cache memory buffer using SQLITE_CONFIG_PAGECACHE. The buffer is static and is of limited size.  N might be
// adjusted downward as needed to accomodate the requested size. The revised value of N is returned.
// A negative SIZE causes the buffer pointer to be NULL.
__device__ static char *_pagecache_buf = nullptr;
__device__ static int test_config_pagecache(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "SIZE N");
		return JIM_ERROR;
	}
	int size, n;
	if (Jim_GetInt(interp, args[1], &size)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], &n)) return JIM_ERROR;
	_free(_pagecache_buf);
	RC rc;
	if (size < 0)
	{
		_pagecache_buf = nullptr;
		rc = DataEx::Config(DataEx::CONFIG_PAGECACHE, nullptr, 0, 0);
	}
	else
	{
		_pagecache_buf = (char *)malloc(size*n);
		rc = DataEx::Config(DataEx::CONFIG_PAGECACHE, _pagecache_buf, size, n);
	}
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, rc));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, n));
	Jim_SetResult(interp, r);
	return JIM_OK;
}

// Usage:    sqlite3_config_alt_pcache INSTALL_FLAG DISCARD_CHANCE PRNG_SEED
// Set up the alternative test page cache.  Install if INSTALL_FLAG is true and uninstall (reverting to the default page cache) if INSTALL_FLAG
// is false.  DISCARD_CHANGE is an integer between 0 and 100 inclusive which determines the chance of discarding a page when unpinned.  100
// is certainty.  0 is never.  PRNG_SEED is the pseudo-random number generator seed.
__device__ static int test_alt_pcache(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern void installTestPCache(bool installFlag, unsigned discardChance, unsigned prngSeed, unsigned highStress);
	if (argc < 2 || argc > 5)
	{
		Jim_WrongNumArgs(interp, 1, args, "INSTALLFLAG DISCARDCHANCE PRNGSEEED HIGHSTRESS");
		return JIM_ERROR;
	}
	int installFlag;
	int discardChance = 0;
	int prngSeed = 0;
	int highStress = 0;
	if (Jim_GetInt(interp, args[1], &installFlag)) return JIM_ERROR;
	if (argc >= 3 && Jim_GetInt(interp, args[2], &discardChance)) return JIM_ERROR;
	if (argc >= 4 && Jim_GetInt(interp, args[3], &prngSeed)) return JIM_ERROR;
	if (argc >= 5 && Jim_GetInt(interp, args[4], &highStress)) return JIM_ERROR;
	if (discardChance < 0 || discardChance > 100)
	{
		Jim_AppendResult(interp, "discard-chance should be between 0 and 100", nullptr);
		return JIM_ERROR;
	}
	installTestPCache(installFlag != 0, (unsigned)discardChance, (unsigned)prngSeed, (unsigned)highStress);
	return JIM_OK;
}

// Usage:    sqlite3_config_memstatus BOOLEAN
// Enable or disable memory status reporting using SQLITE_CONFIG_MEMSTATUS.
__device__ static int test_config_memstatus(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "BOOLEAN");
		return JIM_ERROR;
	}
	bool enable;
	if (Jim_GetBoolean(interp, args[1], &enable)) return JIM_ERROR;
	RC rc = SysEx::Config(SysEx::CONFIG_MEMSTATUS, enable);
	Jim_SetResult(interp, Jim_NewIntObj(interp, rc));
	return JIM_OK;
}

// Usage:    sqlite3_config_lookaside  SIZE  COUNT
__device__ static int test_config_lookaside(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "SIZE COUNT");
		return JIM_ERROR;
	}
	int size, count;
	if (Jim_GetInt(interp, args[1], &size)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], &count)) return JIM_ERROR;
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_ListAppendElement(interp, r, Jim_NewIntObj(interp, TagBase_RuntimeStatics.LookasideSize));
	Jim_ListAppendElement(interp, r, Jim_NewIntObj(interp, TagBase_RuntimeStatics.Lookasides));
	SysEx::Config(SysEx::CONFIG_LOOKASIDE, size, count);
	Jim_SetResult(interp, r);
	return JIM_OK;
}


// Usage:    sqlite3_db_config_lookaside  CONNECTION  BUFID  SIZE  COUNT
// There are two static buffers with BUFID 1 and 2.   Each static buffer is 10KB in size.  A BUFID of 0 indicates that the buffer should be NULL
// which will cause sqlite3_db_config() to allocate space on its own.
__device__ static char _lookaside_buf[2][10000];
__device__ static int test_db_config_lookaside(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);
	if (argc != 5)
	{
		Jim_WrongNumArgs(interp, 1, args, "BUFID SIZE COUNT");
		return JIM_ERROR;
	}
	Context *ctx;
	int bufid, size, count;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], &bufid)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[3], &size)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[4], &count)) return JIM_ERROR;
	RC rc;
	if (bufid == 0) rc = DataEx::CtxConfig(ctx, DataEx::CTXCONFIG_LOOKASIDE, nullptr, size, count);
	else if (bufid >= 1 && bufid <= 2 && size*count <= sizeof(_lookaside_buf[0]) ) rc = DataEx::CtxConfig(ctx, DataEx::CTXCONFIG_LOOKASIDE, _lookaside_buf[bufid], size, count);
	else
	{
		Jim_AppendResult(interp, "illegal arguments - see documentation", nullptr);
		return JIM_ERROR;
	}
	Jim_SetResult(interp, Jim_NewIntObj(interp, rc));
	return JIM_OK;
}

// Usage:    sqlite3_config_heap NBYTE NMINALLOC
__device__ static char *_heap_buf; // Use this memory
__device__ static int test_config_heap(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	Jim_Obj *const *args2 = &args[1];
	int argc2 = argc-1;
	if (argc2 != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "NBYTE NMINALLOC");
		return JIM_ERROR;
	}
	int bytes; // Size of buffer to pass to sqlite3_config() 
	int minAlloc; // Size of minimum allocation
	if (Jim_GetInt(interp, args2[0], &bytes)) return JIM_ERROR;
	if (Jim_GetInt(interp, args2[1], &minAlloc)) return JIM_ERROR;
	RC rc;
	if (bytes == 0)
	{
		_free(_heap_buf);
		_heap_buf = nullptr;
		rc = SysEx::Config(SysEx::CONFIG_HEAP, nullptr, 0, 0);
	}
	else
	{
		_heap_buf = (char *)_realloc(_heap_buf, bytes);
		rc = SysEx::Config(SysEx::CONFIG_HEAP, _heap_buf, bytes, minAlloc);
	}
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// Usage:    sqlite3_config_error  [DB]
// Invoke sqlite3_config() or sqlite3_db_config() with invalid opcodes and verify that they return errors.
__device__ static int test_config_error(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);
	if (argc != 2 && argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "[DB]");
		return JIM_ERROR;
	}
	Context *ctx;
	if (argc == 2)
	{
		if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
		if (DataEx::CtxConfig(ctx, (DataEx::CTXCONFIG)99999) != RC_ERROR)
		{
			Jim_AppendResult(interp, "sqlite3_db_config(db, 99999) does not return SQLITE_ERROR", nullptr);
			return JIM_ERROR;
		}
	}
	else if (DataEx::Config((DataEx::CONFIG)99999) != RC_ERROR)
	{
		Jim_AppendResult(interp, "sqlite3_config(99999) does not return SQLITE_ERROR", nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:    sqlite3_config_uri  BOOLEAN
// Enables or disables interpretation of URI parameters by default using SQLITE_CONFIG_URI.
__device__ static int test_config_uri(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "BOOL");
		return JIM_ERROR;
	}
	bool openUri;
	if (Jim_GetBoolean(interp, args[1], &openUri)) return JIM_ERROR;
	RC rc = SysEx::Config(SysEx::CONFIG_URI, openUri);
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// Usage:    sqlite3_config_cis  BOOLEAN
// Enables or disables the use of the covering-index scan optimization. SQLITE_CONFIG_COVERING_INDEX_SCAN.
__device__ static int test_config_cis(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "BOOL");
		return JIM_ERROR;
	}
	bool useCis;
	if (Jim_GetBoolean(interp, args[1], &useCis)) return JIM_ERROR;
	RC rc = DataEx::Config(DataEx::CONFIG_COVERING_INDEX_SCAN, useCis);
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// Usage:    sqlite3_dump_memsys3  FILENAME
//           sqlite3_dump_memsys5  FILENAME
// Write a summary of unfreed memsys3 allocations to FILENAME.
__device__ static int test_dump_memsys3(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "FILENAME");
		return JIM_ERROR;
	}
	switch (PTR_TO_INT(clientData))
	{
	case 3: {
#ifdef ENABLE_MEMSYS3
		__device__ extern void sqlite3Memsys3Dump(const char *);
		sqlite3Memsys3Dump(Jim_String(args[1]));
		break;
#endif
			}
	case 5: {
#ifdef ENABLE_MEMSYS5
		__device__ extern void sqlite3Memsys5Dump(const char *);
		sqlite3Memsys5Dump(Jim_String(args[1]));
		break;
#endif
			}
	}
	return JIM_OK;
}

// Usage:    sqlite3_status  OPCODE  RESETFLAG
// Return a list of three elements which are the sqlite3_status() return code, the current value, and the high-water mark value.
__constant__ static const struct {
	const char *Name;
	STATUS OP;
} _status_ops[] = {
	{ "SQLITE_STATUS_MEMORY_USED",         STATUS_MEMORY_USED         },
	{ "SQLITE_STATUS_MALLOC_SIZE",         STATUS_MALLOC_SIZE         },
	{ "SQLITE_STATUS_PAGECACHE_USED",      STATUS_PAGECACHE_USED      },
	{ "SQLITE_STATUS_PAGECACHE_OVERFLOW",  STATUS_PAGECACHE_OVERFLOW  },
	{ "SQLITE_STATUS_PAGECACHE_SIZE",      STATUS_PAGECACHE_SIZE      },
	{ "SQLITE_STATUS_SCRATCH_USED",        STATUS_SCRATCH_USED        },
	{ "SQLITE_STATUS_SCRATCH_OVERFLOW",    STATUS_SCRATCH_OVERFLOW    },
	{ "SQLITE_STATUS_SCRATCH_SIZE",        STATUS_SCRATCH_SIZE        },
	{ "SQLITE_STATUS_PARSER_STACK",        STATUS_PARSER_STACK        },
	{ "SQLITE_STATUS_MALLOC_COUNT",        STATUS_MALLOC_COUNT        },
};
__device__ static int test_status(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "PARAMETER RESETFLAG");
		return JIM_ERROR;
	}
	const char *opName = Jim_String(args[1]);
	int i;
	STATUS op;
	for (i = 0; i < _lengthof(_status_ops); i++)
	{
		if (!_strcmp(_status_ops[i].Name, opName))
		{
			op = _status_ops[i].OP;
			break;
		}
	}
	if (i >= _lengthof(_status_ops))
		if (Jim_GetInt(interp, args[1], (int *)&op)) return JIM_ERROR;
	bool resetFlag;
	if (Jim_GetBoolean(interp, args[2], &resetFlag)) return JIM_ERROR;
	int value = 0, maxValue = 0;
	bool rc = _status(op, &value, &maxValue, resetFlag);
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, rc?1:0));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, value));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, maxValue));
	Jim_SetResult(interp, r);
	return JIM_OK;
}

// Usage:    sqlite3_db_status  DATABASE  OPCODE  RESETFLAG
// Return a list of three elements which are the sqlite3_db_status() return code, the current value, and the high-water mark value.
__constant__ static const struct {
	const char *Name;
	Context::CTXSTATUS OP;
} _ctxstatus_ops[] = {
	{ "LOOKASIDE_USED",      Context::CTXSTATUS_LOOKASIDE_USED      },
	{ "CACHE_USED",          Context::CTXSTATUS_CACHE_USED          },
	{ "SCHEMA_USED",         Context::CTXSTATUS_SCHEMA_USED         },
	{ "STMT_USED",           Context::CTXSTATUS_STMT_USED           },
	{ "LOOKASIDE_HIT",       Context::CTXSTATUS_LOOKASIDE_HIT       },
	{ "LOOKASIDE_MISS_SIZE", Context::CTXSTATUS_LOOKASIDE_MISS_SIZE },
	{ "LOOKASIDE_MISS_FULL", Context::CTXSTATUS_LOOKASIDE_MISS_FULL },
	{ "CACHE_HIT",           Context::CTXSTATUS_CACHE_HIT           },
	{ "CACHE_MISS",          Context::CTXSTATUS_CACHE_MISS          },
	{ "CACHE_WRITE",         Context::CTXSTATUS_CACHE_WRITE         }
};
__device__ static int test_db_status(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);
	if (argc != 4)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB PARAMETER RESETFLAG");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	const char *opName = Jim_String(args[2]);
	if (!_memcmp(opName, "SQLITE_", 7)) opName += 7;
	if (!_memcmp(opName, "DBSTATUS_", 9)) opName += 9;
	int i;
	Context::CTXSTATUS op;
	for (i = 0; i < _lengthof(_ctxstatus_ops); i++)
	{
		if (!_strcmp(_ctxstatus_ops[i].Name, opName))
		{
			op = _ctxstatus_ops[i].OP;
			break;
		}
	}
	if (i >= _lengthof(_ctxstatus_ops))
		if (Jim_GetInt(interp, args[2], (int *)&op)) return JIM_ERROR;
	bool resetFlag;
	if (Jim_GetBoolean(interp, args[3], &resetFlag)) return JIM_ERROR;
	int value = 0, maxValue = 0;
	RC rc = ctx->Status(op, &value, &maxValue, resetFlag);
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, rc));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, value));
	Jim_ListAppendElement(nullptr, r, Jim_NewIntObj(interp, maxValue));
	Jim_SetResult(interp, r);
	return JIM_OK;
}

// install_malloc_faultsim BOOLEAN
__device__ static int test_install_malloc_faultsim(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "BOOLEAN");
		return JIM_ERROR;
	}
	bool isInstall;
	if (Jim_GetBoolean(interp, args[1], &isInstall) != JIM_OK) return JIM_ERROR;
	RC rc = faultsimInstall(isInstall);
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// sqlite3_install_memsys3
__device__ static int test_install_memsys3(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	RC rc = RC_MISUSE;
#ifdef ENABLE_MEMSYS3
	const sqlite3_mem_methods *sqlite3MemGetMemsys3();
	rc = sqlite3_config(SQLITE_CONFIG_MALLOC, sqlite3MemGetMemsys3());
#endif
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

__device__ static int test_vfs_oom_test(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	//__device__ extern int _memdebug_vfs_oom_test;
	int _memdebug_vfs_oom_test = 0;
	if (argc > 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "?INTEGER?");
		return JIM_ERROR;
	}
	else if (argc == 2)
	{
		int new_;
		if (Jim_GetInt(interp, args[1], &new_)) return JIM_ERROR;
		_memdebug_vfs_oom_test = new_;
	}
	Jim_SetResult(interp, Jim_NewIntObj(interp, _memdebug_vfs_oom_test));
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "sqlite3_malloc",             test_malloc                   , nullptr },
	{ "sqlite3_realloc",            test_realloc                  , nullptr },
	{ "sqlite3_free",               test_free                     , nullptr },
	{ "memset",                     test_memset                   , nullptr },
	{ "memget",                     test_memget                   , nullptr },
	{ "sqlite3_memory_used",        test_memory_used              , nullptr },
	{ "sqlite3_memory_highwater",   test_memory_highwater         , nullptr },
	{ "sqlite3_memdebug_backtrace", test_memdebug_backtrace       , nullptr },
	{ "sqlite3_memdebug_dump",      test_memdebug_dump            , nullptr },
	{ "sqlite3_memdebug_fail",      test_memdebug_fail            , nullptr },
	{ "sqlite3_memdebug_pending",   test_memdebug_pending         , nullptr },
	{ "sqlite3_memdebug_settitle",  test_memdebug_settitle        , nullptr },
	{ "sqlite3_memdebug_malloc_count", test_memdebug_malloc_count , nullptr },
	{ "sqlite3_memdebug_log",       test_memdebug_log             , nullptr },
	{ "sqlite3_config_scratch",     test_config_scratch           , nullptr },
	{ "sqlite3_config_pagecache",   test_config_pagecache         , nullptr },
	{ "sqlite3_config_alt_pcache",  test_alt_pcache               , nullptr },
	{ "sqlite3_status",             test_status                   , nullptr },
	{ "sqlite3_db_status",          test_db_status                , nullptr },
	{ "install_malloc_faultsim",    test_install_malloc_faultsim  , nullptr },
	{ "sqlite3_config_heap",        test_config_heap              , nullptr },
	{ "sqlite3_config_memstatus",   test_config_memstatus         , nullptr },
	{ "sqlite3_config_lookaside",   test_config_lookaside         , nullptr },
	{ "sqlite3_config_error",       test_config_error             , nullptr },
	{ "sqlite3_config_uri",         test_config_uri               , nullptr },
	{ "sqlite3_config_cis",         test_config_cis               , nullptr },
	{ "sqlite3_db_config_lookaside",test_db_config_lookaside      , nullptr },
	{ "sqlite3_dump_memsys3",       test_dump_memsys3             , (void *)3 },
	{ "sqlite3_dump_memsys5",       test_dump_memsys3             , (void *)5 },
	{ "sqlite3_install_memsys3",    test_install_memsys3          , nullptr },
	{ "sqlite3_memdebug_vfs_oom_test", test_vfs_oom_test          , nullptr },
};
__device__ int Sqlitetest_malloc_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
	{
		ClientData c = _objCmds[i].ClientData;
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, c, nullptr);
	}
	return JIM_OK;
}

#endif
