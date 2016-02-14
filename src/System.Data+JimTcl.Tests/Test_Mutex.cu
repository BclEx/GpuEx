// This file contains test logic for the sqlite3_mutex interfaces.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>

// defined in test1.c
void *sqlite3TestTextToPtr(const char *);
const char *sqlite3TestErrorName(int);

// A countable mutex
struct _mutex_obj
{
	MutexEx Real;
	MUTEX Type;
};

// State variables
__device__ static struct test_mutex_globals
{
	bool IsInstalled;           // True if installed
	bool DisableInit;           // True to cause sqlite3_initalize() to fail
	bool DisableTry;            // True to force sqlite3_mutex_try() to fail
	bool IsInit;                // True if initialized
	_mutex_methods m;			// Interface to "real" mutex system
	int Counters[8];            // Number of grabs of each type of mutex
	_mutex_obj Statics[6];		// The six static mutexes
} g_ = {0};

// Return true if the countable mutex is currently held
__device__ static bool counterMutexHeld(_mutex_obj *p) { return g_.m.Held(p->Real); }

// Return true if the countable mutex is not currently held 
__device__ static bool counterMutexNotHeld(_mutex_obj *p) { return g_.m.NotHeld(p->Real); }

// Initialize the countable mutex interface
// Or, if g.disableInit is non-zero, then do not initialize but instead return the value of g.disableInit as the result code.  This can be used
// to simulate an initialization failure.
__device__ static int counterMutexInit()
{ 
	int rc;
	if (g_.DisableInit) return g_.DisableInit;
	rc = g_.m.Init();
	g_.IsInit = true;
	return rc;
}

// Uninitialize the mutex subsystem
__device__ static void counterMutexShutdown()
{ 
	g_.IsInit = false;
	g_.m.Shutdown();
}

// Allocate a countable mutex
__device__ static MutexEx counterMutexAlloc(MUTEX type)
{
	_assert(g_.IsInit);
	_assert(type < 8 && type >= 0);
	MutexEx real = g_.m.Alloc(type);
	if (!real) return nullptr;
	_mutex_obj *r = (type == MUTEX_FAST || type == MUTEX_RECURSIVE ? (_mutex_obj *)_alloc(sizeof(_mutex_obj)) : &g_.Statics[(int)type-2]);
	r->Type = type;
	r->Real = real;
	return r;
}

// Free a countable mutex
__device__ static void counterMutexFree(_mutex_obj *p)
{
	_assert(g_.IsInit);
	g_.m.Free(p->Real);
	if (p->Type == MUTEX_FAST || p->Type == MUTEX_RECURSIVE)
		_free(p);
}

// Enter a countable mutex.  Block until entry is safe.
__device__ static void counterMutexEnter(_mutex_obj *p)
{
	_assert(g_.IsInit);
	g_.Counters[(int)p->Type]++;
	g_.m.Enter(p->Real);
}

// Try to enter a mutex.  Return true on success.
__device__ static bool counterMutexTryEnter(_mutex_obj *p)
{
	_assert(g_.IsInit);
	g_.Counters[(int)p->Type]++;
	if (g_.DisableTry) return RC_BUSY;
	return g_.m.TryEnter(p->Real);
}

// Leave a mutex
__device__ static void counterMutexLeave(_mutex_obj *p)
{
	_assert(g_.IsInit);
	g_.m.Leave(p->Real);
}

// sqlite3_shutdown
__device__ static int test_shutdown(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	RC rc = DataEx::Shutdown();
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// sqlite3_initialize
__device__ static int test_initialize(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	RC rc = DataEx::Initialize();
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// install_mutex_counters BOOLEAN
_mutex_methods _counter_methods = {
	counterMutexInit,
	counterMutexShutdown,
	(MutexEx (*)(MUTEX))counterMutexAlloc,
	(void (*)(MutexEx))counterMutexFree,
	(void (*)(MutexEx))counterMutexEnter,
	(bool (*)(MutexEx))counterMutexTryEnter,
	(void (*)(MutexEx))counterMutexLeave,
	(bool (*)(MutexEx))counterMutexHeld,
	(bool (*)(MutexEx))counterMutexNotHeld
};

__device__ static int test_install_mutex_counters(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "BOOLEAN");
		return JIM_ERROR;
	}
	bool isInstall;
	if (Jim_GetBoolean(interp, args[1], &isInstall) != JIM_OK)
		return JIM_ERROR;
	if (isInstall == g_.IsInstalled)
	{
		Jim_AppendResult(interp, "mutex counters are ", nullptr);
		Jim_AppendResult(interp, isInstall ? "already installed" : "not installed", nullptr);
		return JIM_ERROR;
	}
	RC rc = RC_OK;
	if (isInstall)
	{
		_assert(!g_.m.Alloc);
		rc = SysEx::Config(SysEx::CONFIG_GETMUTEX, &g_.m);
		if (rc == RC_OK)
			SysEx::Config(SysEx::CONFIG_MUTEX, &_counter_methods);
		g_.DisableTry = false;
	}
	else
	{
		_assert(g_.m.Alloc);
		rc = SysEx::Config(SysEx::CONFIG_MUTEX, &g_.m);
		memset(&g_.m, 0, sizeof(_mutex_methods));
	}
	if (rc == RC_OK)
		g_.IsInstalled = isInstall;
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// read_mutex_counters
__device__ static int test_read_mutex_counters(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	char *names[8] = {
		"fast",        "recursive",   "static_master", "static_mem", 
		"static_open", "static_prng", "static_lru",    "static_pmem"
	};
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	Jim_IncrRefCount(r);
	for (int ii = 0; ii < 8; ii++)
	{
		Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, names[ii], -1));
		Jim_ListAppendElement(interp, r, Jim_NewIntObj(interp, g_.Counters[ii]));
	}
	Jim_SetResult(interp, r);
	Jim_DecrRefCount(interp, r);
	return JIM_OK;
}

// clear_mutex_counters
__device__ static int test_clear_mutex_counters(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	for (int ii = 0; ii < 8; ii++)
		g_.Counters[ii] = 0;
	return JIM_OK;
}

// Create and free a mutex.  Return the mutex pointer.  The pointer will be invalid since the mutex has already been freed.  The
// return pointer just checks to see if the mutex really was allocated.
__device__ static int test_alloc_mutex(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
#if THREADSAFE
	MutexEx p = _mutex_alloc(MUTEX_FAST);
	char buf[100];
	_mutex_free(p);
	__snprintf(buf, sizeof(buf), "%p", p);
	Jim_AppendResult(interp, buf, nullptr);
#endif
	return JIM_OK;
}

// sqlite3_config OPTION
//
// OPTION can be either one of the keywords:
//            SQLITE_CONFIG_SINGLETHREAD
//            SQLITE_CONFIG_MULTITHREAD
//            SQLITE_CONFIG_SERIALIZED
// Or OPTION can be an raw integer.
__device__ static int test_config(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	struct ConfigOption {
		const char *Name;
		int Value;
	} opts[] = {
		{"singlethread", SysEx::CONFIG_SINGLETHREAD},
		{"multithread",  SysEx::CONFIG_MULTITHREAD},
		{"serialized",   SysEx::CONFIG_SERIALIZED},
		{nullptr, (SysEx::CONFIG)0}
	};
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	int i;
	if (Jim_GetEnumFromStruct(interp, args[1], (const void **)opts, sizeof(struct ConfigOption), &i, "flag", 0))
	{
		if (Jim_GetInt(interp, args[1], &i))
			return JIM_ERROR;
	}
	else
		i = opts[i].Value;
	RC rc = SysEx::Config((SysEx::CONFIG)i);
	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

__device__ static Context *GetDbPointer(Jim_Interp *interp, Jim_Obj *obj)
{
	const char *cmd = Jim_String(obj);
	Jim_CmdInfo info;
	Context *ctx = (Jim_GetCommandInfo(interp, obj, &info) ? *((Context **)info.objClientData) : (Context *)sqlite3TestTextToPtr(cmd));
	_assert(ctx);
	return ctx;
}

__device__ static int test_enter_db_mutex(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx = GetDbPointer(interp, args[1]);
	if (!ctx)
		return JIM_ERROR;
	_mutex_enter(sqlite3_db_mutex(ctx));
	return JIM_OK;
}

__device__ static int test_leave_db_mutex(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx = GetDbPointer(interp, args[1]);
	if (!ctx)
		return JIM_ERROR;
	_mutex_leave(sqlite3_db_mutex(ctx));
	return JIM_OK;
}

__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "sqlite3_shutdown",        test_shutdown },
	{ "sqlite3_initialize",      test_initialize },
	{ "sqlite3_config",          test_config },
	{ "enter_db_mutex",          test_enter_db_mutex },
	{ "leave_db_mutex",          test_leave_db_mutex },
	{ "alloc_dealloc_mutex",     test_alloc_mutex },
	{ "install_mutex_counters",  test_install_mutex_counters },
	{ "read_mutex_counters",     test_read_mutex_counters },
	{ "clear_mutex_counters",    test_clear_mutex_counters }
};
__device__ int Sqlitetest_mutex_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	Jim_LinkVar(interp, "disable_mutex_init", (char *)&g_.DisableInit, JIM_LINK_INT);
	Jim_LinkVar(interp, "disable_mutex_try", (char *)&g_.DisableTry, JIM_LINK_INT);
	return JIM_OK;
}
