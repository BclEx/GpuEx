// The code in this file is used for testing SQLite. It is not part of the source code used in production systems.
//
// Specifically, this file tests the effect of errors while initializing the various pluggable sub-systems from within sqlite3_initialize().
// If an error occurs in sqlite3_initialize() the following should be true:
//
//   1) An error code is returned to the user, and
//   2) A subsequent call to sqlite3_shutdown() calls the shutdown method of those subsystems that were initialized, and
//   3) A subsequent call to sqlite3_initialize() attempts to initialize the remaining, uninitialized, subsystems.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <Jim.h>

__device__ static struct Wrapped
{
	sqlite3_pcache_methods2 pcache;
	sqlite3_mem_methods     mem;
	sqlite3_mutex_methods   mutex;
	bool mem_init;                // True if mem subsystem is initalized
	bool mem_fail;                // True to fail mem subsystem inialization
	bool mutex_init;              // True if mutex subsystem is initalized
	bool mutex_fail;              // True to fail mutex subsystem inialization
	bool pcache_init;             // True if pcache subsystem is initalized
	bool pcache_fail;             // True to fail pcache subsystem inialization
} _wrapped;

__device__ static int wrMemInit(void *appData)
{
	int rc = (_wrapped.mem_fail ? RC_ERROR : _wrapped.mem.xInit(_wrapped.mem.pAppData));
	if (rc == RC_OK)
		_wrapped.mem_init = true;
	return rc;
}
__device__ static void wrMemShutdown(void *appData) { _wrapped.mem.xShutdown(wrapped.mem.pAppData); _wrapped.mem_init = false; }
__device__ static void *wrMemMalloc(int n) { return _wrapped.mem.xMalloc(n); }
__device__ static void wrMemFree(void *p) { _wrapped.mem.xFree(p); }
__device__ static void *wrMemRealloc(void *p, int n) { return _wrapped.mem.xRealloc(p, n); }
__device__ static int wrMemSize(void *p) { return _wrapped.mem.xSize(p); }
__device__ static int wrMemRoundup(int n) { return _wrapped.mem.xRoundup(n); }

__device__ static int wrMutexInit()
{
	int rc = (_wrapped.mutex_fail ? RC_ERROR : _wrapped.mutex.xMutexInit());
	if (rc == RC_OK)
		_wrapped.mutex_init = true;
	return rc;
}
static int wrMutexEnd() { _wrapped.mutex.xMutexEnd(); _wrapped.mutex_init = false; return RC_OK; }

__device__ static sqlite3_mutex *wrMutexAlloc(int e) { return _wrapped.mutex.xMutexAlloc(e); }
__device__ static void wrMutexFree(sqlite3_mutex *p) { _wrapped.mutex.xMutexFree(p); }
__device__ static void wrMutexEnter(sqlite3_mutex *p) { _wrapped.mutex.xMutexEnter(p); }
__device__ static int wrMutexTry(sqlite3_mutex *p) { return _wrapped.mutex.xMutexTry(p); }
__device__ static void wrMutexLeave(sqlite3_mutex *p) { _wrapped.mutex.xMutexLeave(p); }
__device__ static int wrMutexHeld(sqlite3_mutex *p) { return _wrapped.mutex.xMutexHeld(p); }
__device__ static int wrMutexNotheld(sqlite3_mutex *p) { return _wrapped.mutex.xMutexNotheld(p); }

__device__ static int wrPCacheInit(void *arg)
{
	int rc = (_wrapped.pcache_fail ? RC_ERROR : _wrapped.pcache.xInit(_wrapped.pcache.pArg));
	if (rc == RC_OK)
		_wrapped.pcache_init = true;
	return rc;
}
__device__ static void wrPCacheShutdown(void *arg) { _wrapped.pcache.xShutdown(_wrapped.pcache.pArg); _wrapped.pcache_init = false; }

__device__ static sqlite3_pcache *wrPCacheCreate(int a, int b, int c) { return _wrapped.pcache.xCreate(a, b, c); }
__device__ static void wrPCacheCachesize(sqlite3_pcache *p, int n) { _wrapped.pcache.xCachesize(p, n); }
__device__ static int wrPCachePagecount(sqlite3_pcache *p) { return _wrapped.pcache.xPagecount(p); }
__device__ static sqlite3_pcache_page *wrPCacheFetch(sqlite3_pcache *p, unsigned a, int b) { return _wrapped.pcache.xFetch(p, a, b); }
__device__ static void wrPCacheUnpin(sqlite3_pcache *p, sqlite3_pcache_page *a, int b) { _wrapped.pcache.xUnpin(p, a, b); }
__device__ static void wrPCacheRekey(sqlite3_pcache *p, sqlite3_pcache_page *a, unsigned int b, unsigned int c) { _wrapped.pcache.xRekey(p, a, b, c); }
__device__ static void wrPCacheTruncate(sqlite3_pcache *p, unsigned int a) { _wrapped.pcache.xTruncate(p, a); }
__device__ static void wrPCacheDestroy(sqlite3_pcache *p) { _wrapped.pcache.xDestroy(p); }

__device__ static void installInitWrappers()
{
	sqlite3_mutex_methods mutexmethods = {
		wrMutexInit,  wrMutexEnd,   wrMutexAlloc,
		wrMutexFree,  wrMutexEnter, wrMutexTry,
		wrMutexLeave, wrMutexHeld,  wrMutexNotheld
	};
	sqlite3_pcache_methods2 pcachemethods = {
		1, 0,
		wrPCacheInit,      wrPCacheShutdown,  wrPCacheCreate, 
		wrPCacheCachesize, wrPCachePagecount, wrPCacheFetch,
		wrPCacheUnpin,     wrPCacheRekey,     wrPCacheTruncate,  
		wrPCacheDestroy
	};
	sqlite3_mem_methods memmethods = {
		wrMemMalloc,   wrMemFree,    wrMemRealloc,
		wrMemSize,     wrMemRoundup, wrMemInit,
		wrMemShutdown,
		0
	};
	_memset(&_wrapped, 0, sizeof(_wrapped));
	sqlite3_shutdown();
	sqlite3_config(SQLITE_CONFIG_GETMUTEX, &_wrapped.mutex);
	sqlite3_config(SQLITE_CONFIG_GETMALLOC, &_wrapped.mem);
	sqlite3_config(SQLITE_CONFIG_GETPCACHE2, &_wrapped.pcache);
	sqlite3_config(SQLITE_CONFIG_MUTEX, &_mutexmethods);
	sqlite3_config(SQLITE_CONFIG_MALLOC, &_memmethods);
	sqlite3_config(SQLITE_CONFIG_PCACHE2, &_pcachemethods);
}

__device__ static int init_wrapper_install(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	installInitWrappers();
	for (int i = 1; i < argc; i++)
	{
		const char *z = Jim_String(args[i]);
		if (!_strcmp(z, "mem")) _wrapped.mem_fail = true;
		else if (!_strcmp(z, "mutex")) _wrapped.mutex_fail = true ;
		else if (!_strcmp(z, "pcache")) _wrapped.pcache_fail = true;
		else
		{
			Tcl_AppendResult(interp, "Unknown argument: \"", z, "\"");
			return JIM_ERROR;
		}
	}
	return JIM_OK;
}

__device__ static int init_wrapper_uninstall(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Tcl_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	_memset(&_wrapped, 0, sizeof(&_wrapped));
	sqlite3_shutdown();
	sqlite3_config(SQLITE_CONFIG_MUTEX, &_wrapped.mutex);
	sqlite3_config(SQLITE_CONFIG_MALLOC, &_wrapped.mem);
	sqlite3_config(SQLITE_CONFIG_PCACHE2, &_wrapped.pcache);
	return JIM_OK;
}

__device__ static int init_wrapper_clear(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Tcl_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	_wrapped.mem_fail = false;
	_wrapped.mutex_fail = false;
	_wrapped.pcache_fail = false;
	return JIM_OK;
}

__device__ static int init_wrapper_query(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Tcl_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	Jim_Obj *r = Tcl_NewListObj(interp, nullptr, 0);
	if (_wrapped.mutex_init) Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, "mutex", -1));
	if (_wrapped.mem_init) Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, "mem", -1));
	if (_wrapped.pcache_init) Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, "pcache", -1));
	Jim_SetResult(interp, r);
	return JIM_OK;
}

__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "init_wrapper_install",   init_wrapper_install },
	{ "init_wrapper_query",     init_wrapper_query },
	{ "init_wrapper_uninstall", init_wrapper_uninstall },
	{ "init_wrapper_clear",     init_wrapper_clear }
};
__device__ int Sqlitetest_init_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateObjCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	return JIM_OK;
}
