// This file contains the implementation of some Tcl commands used to test that sqlite3 database handles may be concurrently accessed by 
// multiple threads. Right now this only works on unix.

#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <Jim.h>
#include <Jim+EventLoop.h>

#ifdef _THREADSAFE
#include <errno.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif

// One of these is allocated for each thread created by [sqlthread spawn].
typedef struct SqlThread SqlThread;
struct SqlThread
{
	Jim_ThreadId parent;     // Thread id of parent thread
	Jim_Interp *interp;      // Parent interpreter
	char *Script;           // The script to execute.
	char *Varname;          // Varname in parent script
};

// A custom Tcl_Event type used by this module. When the event is handled, script zScript is evaluated in interpreter interp. If
// the evaluation throws an exception (returns JIM_ERROR), then the error is handled by Tcl_BackgroundError(). If no error occurs,
// the result is simply discarded.
typedef struct EvalEvent EvalEvent;
struct EvalEvent
{
	Jim_Event base;          // Base class of type Tcl_Event
	char *Script;           // The script to execute.
	Jim_Interp *interp;      // The interpreter to execute it in.
};

__device__ static Jim_CmdProc _sqlthread_proc;
__device__ static Jim_CmdProc _clock_seconds_proc;
#if OS_UNIX && defined(ENABLE_UNLOCK_NOTIFY)
__device__ static Jim_CmdProc _blocking_step_proc;
__device__ static Jim_CmdProc _blocking_prepare_v2_proc;
#endif
__device__ int Sqlitetest1_Init(Jim_Interp *);
__device__ int Sqlite3_Init(Jim_Interp *);

// Functions from test1.c
__device__ void *sqlite3TestTextToPtr(const char *);
__device__ const char *sqlite3TestErrorName(int);
__device__ int GetDbPointer(Jim_Interp *, const char *, Context **);
__device__ int sqlite3TestMakePointerStr(Jim_Interp *, char *, void *);
__device__ int sqlite3TestErrCode(Jim_Interp *, Context *, int);

// Handler for events of type EvalEvent.
__device__ static int tclScriptEvent(Jim_Event *evPtr, int flags)
{
	EvalEvent *p = (EvalEvent *)evPtr;
	int rc = Jim_Eval(p->interp, p->Script);
	if (rc != JIM_OK)
		Tcl_BackgroundError(p->interp);
	return 1;
}

// Register an EvalEvent to evaluate the script pScript in the parent interpreter/thread of SqlThread p.
__device__ static void postToParent(SqlThread *p, Jim_Obj *script)
{
	int msgLength;
	char *msg = Jim_GetString(script, &msgLength); 
	EvalEvent *event_ = (EvalEvent *)_alloc(sizeof(EvalEvent)+msgLength+1);
	event_->base.nextPtr = 0;
	event_->base.proc = tclScriptEvent;
	event_->Script = (char *)&event_[1];
	memcpy(event_->Script, msg, msgLength+1);
	event_->interp = p->interp;
	Jim_ThreadQueueEvent(p->parent, (Jim_Event *)event_, JIM_QUEUE_TAIL);
	Jim_ThreadAlert(p->parent);
}

// The main function for threads created with [sqlthread spawn].
__device__ static Tcl_ThreadCreateType tclScriptThread(ClientData sqlThread)
{
	__device__ extern int Sqlitetest_mutex_Init(Jim_Interp *);
	SqlThread *p = (SqlThread *)sqlThread;
	Jim_Interp *interp = Jim_CreateInterp();
	Jim_CreateCommand(interp, "clock_seconds", clock_seconds_proc, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlthread", sqlthread_proc, sqlThread, nullptr);
#if OS_UNIX && defined(ENABLE_UNLOCK_NOTIFY)
	Jim_CreateCommand(interp, "sqlite3_blocking_step", blocking_step_proc, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_blocking_prepare_v2", blocking_prepare_v2_proc, (void *)1, nullptr);
	Jim_CreateCommand(interp, "sqlite3_nonblocking_prepare_v2", blocking_prepare_v2_proc, nullptr, nullptr);
#endif
	Sqlitetest1_Init(interp);
	Sqlitetest_mutex_Init(interp);
	Sqlite3_Init(interp);

	int rc = Jim_Eval(interp, p->Script);
	Jim_Obj *res = Jim_GetResult(interp);
	Jim_Obj *list = Jim_NewListObj(interp, nullptr, 0);
	Jim_IncrRefCount(list);
	Jim_IncrRefCount(res);
	if (rc != JIM_OK)
	{
		Jim_ListAppendElement(interp, list, Jim_NewStringObj(interp, "error", -1));
		Jim_ListAppendElement(interp, list, res);
		postToParent(p, list);
		Jim_DecrRefCount(interp, list);
		list = Jim_NewListObj(interp, nullptr, 0);
	}
	Jim_ListAppendElement(interp, list, Jim_NewStringObj(interp, "set", -1));
	Jim_ListAppendElement(interp, list, Jim_NewStringObj(interp, p->Varname, -1));
	Jim_ListAppendElement(interp, list, res);
	postToParent(p, list);

	_free((void *)p);
	Jim_DecrRefCount(interp, list);
	Jim_DecrRefCount(interp, res);
	Jim_DeleteInterp(interp);
	while (Tcl_DoOneEvent(JIM_ALL_EVENTS | JIM_DONT_WAIT));
	Jim_ExitThread(0);
	JIM_THREAD_CREATE_RETURN;
}

// sqlthread spawn VARNAME SCRIPT
//     Spawn a new thread with its own Tcl interpreter and run the specified SCRIPT(s) in it. The thread terminates after running
//     the script. The result of the script is stored in the variable VARNAME.
//     The caller can wait for the script to terminate using [vwait VARNAME].
__device__ static int sqlthread_spawn(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	// Parameters for thread creation
	const int stack = JIM_THREAD_STACK_DEFAULT;
	const int flags = JIM_THREAD_NOFLAGS;
	_assert(argc == 4);
	int varnameLength, scriptLength;
	const char *varname = Jim_GetString(args[2], &varnameLength);
	const char *script = Jim_GetString(args[3], &scriptLength);

	SqlThread *new_ = (SqlThread *)ckalloc(sizeof(SqlThread)+nVarname+nScript+2);
	new_->zVarname = (char *)&new_[1];
	new_->zScript = (char *)&new_->zVarname[nVarname+1];
	memcpy(new_->zVarname, zVarname, nVarname+1);
	memcpy(new_->zScript, zScript, nScript+1);
	new_->parent = Tcl_GetCurrentThread();
	new_->interp = interp;

	Tcl_ThreadId x;
	int rc = Jim_CreateThread(&x, tclScriptThread, (void *)new_, stack, flags);
	if (rc != JIM_OK)
	{
		Jim_AppendResult(interp, "Error in Tcl_CreateThread()", nullptr);
		_free((char *)new_);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// sqlthread parent SCRIPT
//     This can be called by spawned threads only. It sends the specified script back to the parent thread for execution. The result of
//     evaluating the SCRIPT is returned. The parent thread must enter the event loop for this to work - otherwise the caller will
//     block indefinitely.
//
//     NOTE: At the moment, this doesn't work. FIXME.
__device__ static int sqlthread_parent(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	_assert(argc == 3);
	SqlThread *p = (SqlThread *)clientData;
	if (!p)
	{
		Jim_AppendResult(interp, "no parent thread", nullptr);
		return JIM_ERROR;
	}
	int msgLength;
	char *msg = Jim_GetString(args[2], &nMsg);
	EvalEvent *event_ = (EvalEvent *)_alloc(sizeof(EvalEvent)+msgLength+1);
	event_->base.nextPtr = 0;
	event_->base.proc = tclScriptEvent;
	event_->Script = (char *)&event_[1];
	_memcpy(event_->Script, msg, msgLength+1);
	event_->interp = p->interp;
	Jim_ThreadQueueEvent(p->parent, (Jim_Event *)event_, JIM_QUEUE_TAIL);
	Jim_ThreadAlert(p->parent);
	return JIM_OK;
}

__device__ static int xBusy(void *arg, int busy)
{
	_sleep(50);
	return 1; // Try again...
}

// sqlthread open
//     Open a database handle and return the string representation of the pointer value.
__device__ static int sqlthread_open(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern int sqlite3TestMakePointerStr(Jim_Interp *interp, char *ptr, void *p);
	__device__ extern void Md5_Register(Context *);
	const char *filename = Jim_String(args[2]);
	Context *ctx;
	Main::Open(filename, &ctx);
#ifdef HAS_CODEC
	if (ctx && argc >= 4)
	{
		int keyLength;
		const char *key = Jim_GetString(objv[3], &keyLength);
		int rc = sqlite3_key(ctx, key, keyLength);
		if (rc != RC_OK)
		{
			char *errMsg = sqlite3_mprintf("error %d: %s", rc, Main::Errmsg(ctx));
			Main::Close(ctx);
			Jim_AppendResult(interp, errMsg, nullptr);
			_free(errMsg);
			return JIM_ERROR;
		}
	}
#endif
	Md5_Register(ctx);
	Main::BusyHandler(ctx, xBusy, nullptr);
	char buf[100];
	if (sqlite3TestMakePointerStr(interp, buf, ctx)) return JIM_ERROR;
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}


// sqlthread open
//     Return the current thread-id (Tcl_GetCurrentThread()) cast to an integer.
__device__ static int sqlthread_id(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	Jim_ThreadId id = Jim_GetCurrentThread();
	Jim_SetResult(interp, Jim_NewIntObj(PTR_TO_INT(id)));
	return JIM_OK;
}

// Dispatch routine for the sub-commands of [sqlthread].
__device__ static int sqlthread_proc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	struct SubCommand {
		char *Name;
		Jim_CmdProc *Proc;
		int Args;
		char *Usage;
	} subs[] = {
		{ "parent", sqlthread_parent, 1, "SCRIPT" },
		{ "spawn",  sqlthread_spawn,  2, "VARNAME SCRIPT" },
		{ "open",   sqlthread_open,   1, "DBNAME" },
		{ "id",     sqlthread_id,     0, "" },
		{ nullptr, nullptr, 0, nullptr }
	};
	if (argc < 2)
	{
		Tcl_WrongNumArgs(interp, 1, args, "SUB-COMMAND");
		return JIM_ERROR;
	}
	int index;
	int rc = Jim_GetEnumFromStruct(interp, args[1], (const void **)subs, sizeof(subs[0]), &index, "sub-command", 0);
	if (rc != JIM_OK) return rc;
	struct SubCommand *sub = &subs[index];
	if (argc < (sub->Args+2))
	{
		Jim_WrongNumArgs(interp, 2, args, sub->Usage);
		return JIM_ERROR;
	}
	return sub->Proc(clientData, interp, argc, args);
}

// The [clock_seconds] command. This is more or less the same as the regular tcl [clock seconds], except that it is available in testfixture
// when linked against both Tcl 8.4 and 8.5. Because [clock seconds] is implemented as a script in Tcl 8.5, it is not usually available to testfixture.
__device__ static int clock_seconds_proc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	Jim_Time now;
	Jim_GetTime(&now);
	Jim_SetResult(interp, Jim_NewIntObj(interp, now.sec));
	return JIM_OK;
}


#pragma region blocking step
// This block contains the implementation of the [sqlite3_blocking_step] command available to threads created by [sqlthread spawn] commands. It
// is only available on UNIX for now. This is because pthread condition variables are used.
//
// The source code for the C functions sqlite3_blocking_step(), blocking_step_notify() and the structure UnlockNotification is
// automatically extracted from this file and used as part of the documentation for the sqlite3_unlock_notify() API function. This
// should be considered if these functions are to be extended (i.e. to support windows) in the future.
#if OS_UNIX && defined(ENABLE_UNLOCK_NOTIFY)

// This example uses the pthreads API
#include <pthread.h>

// A pointer to an instance of this structure is passed as the user-context pointer when registering for an unlock-notify callback.
typedef struct UnlockNotification UnlockNotification;
struct UnlockNotification
{
	bool Fired;                         // True after unlock event has occurred
	pthread_cond_t cond;				// Condition variable to wait on
	pthread_mutex_t mutex;				// Mutex to protect structure
};

// This function is an unlock-notify callback registered with SQLite.
__device__ static void unlock_notify_cb(void **args, int argc)
{
	for (int i = 0; i < argc; i++)
	{
		UnlockNotification *p = (UnlockNotification *)args[i];
		pthread_mutex_lock(&p->mutex);
		p->Fired = 1;
		pthread_cond_signal(&p->cond);
		pthread_mutex_unlock(&p->mutex);
	}
}

// This function assumes that an SQLite API call (either sqlite3_prepare_v2() or sqlite3_step()) has just returned SQLITE_LOCKED. The argument is the
// associated database connection.
//
// This function calls sqlite3_unlock_notify() to register for an unlock-notify callback, then blocks until that callback is delivered 
// and returns SQLITE_OK. The caller should then retry the failed operation.
//
// Or, if sqlite3_unlock_notify() indicates that to block would deadlock the system, then this function returns SQLITE_LOCKED immediately. In 
// this case the caller should not retry the operation and should roll back the current transaction (if any).
__device__ static int wait_for_unlock_notify(Context *ctx)
{
	// Initialize the UnlockNotification structure.
	UnlockNotification un;
	un.fired = 0;
	pthread_mutex_init(&un.mutex, 0);
	pthread_cond_init(&un.cond, 0);
	// Register for an unlock-notify callback.
	RC rc = Main::UnlockNotify(ctx, unlock_notify_cb, (void *)&un);
	_assert(rc == RC_LOCKED || rc == RC_OK);
	// The call to sqlite3_unlock_notify() always returns either SQLITE_LOCKED or SQLITE_OK. 
	//
	// If SQLITE_LOCKED was returned, then the system is deadlocked. In this case this function needs to return SQLITE_LOCKED to the caller so 
	// that the current transaction can be rolled back. Otherwise, block until the unlock-notify callback is invoked, then return SQLITE_OK.
	if (rc == RC_OK)
	{
		pthread_mutex_lock(&un.mutex);
		if (!un.Fired)
			pthread_cond_wait(&un.cond, &un.mutex);
		pthread_mutex_unlock(&un.mutex);
	}
	// Destroy the mutex and condition variables.
	pthread_cond_destroy(&un.cond);
	pthread_mutex_destroy(&un.mutex);
	return rc;
}

// This function is a wrapper around the SQLite function sqlite3_step(). It functions in the same way as step(), except that if a required
// shared-cache lock cannot be obtained, this function may block waiting for the lock to become available. In this scenario the normal API step()
// function always returns SQLITE_LOCKED.
//
// If this function returns SQLITE_LOCKED, the caller should rollback the current transaction (if any) and try again later. Otherwise, the
// system may become deadlocked.
__device__ int sqlite3_blocking_step(Vdbe *stmt)
{
	RC rc;
	while ((rc = sqlite3_step(pStmt)) == RC_LOCKED)
	{
		rc = wait_for_unlock_notify(sqlite3_db_handle(stmt));
		if (rc != RC_OK) break;
		stmt->Reset();
	}
	return rc;
}

// This function is a wrapper around the SQLite function sqlite3_prepare_v2(). It functions in the same way as prepare_v2(), except that if a required
// shared-cache lock cannot be obtained, this function may block waiting for the lock to become available. In this scenario the normal API prepare_v2()
// function always returns SQLITE_LOCKED.
//
// If this function returns SQLITE_LOCKED, the caller should rollback the current transaction (if any) and try again later. Otherwise, the
// system may become deadlocked.
__device__ int sqlite3_blocking_prepare_v2(Context *ctx, const char *sql, int sqlLength, Vdbe **stmt, const char **z)
{
	RC rc;
	while ((rc = Prepare::Prepare_v2(ctx, sql, sqlLength, stmt, z)) == RC_LOCKED)
	{
		rc = wait_for_unlock_notify(ctx);
		if (rc != RC_OK) break;
	}
	return rc;
}

// Usage: sqlite3_blocking_step STMT
// Advance the statement to the next row.
__device__ static int blocking_step_proc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "STMT");
		return JIM_ERROR;
	}
	Vdbe *stmt = (Vdbe *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = sqlite3_blocking_step(stmt);
	Jim_SetResult(interp, (char *)sqlite3TestErrorName(rc), nullptr);
	return JIM_OK;
}

// Usage: sqlite3_blocking_prepare_v2 DB sql bytes ?tailvar?
// Usage: sqlite3_nonblocking_prepare_v2 DB sql bytes ?tailvar?
__device__ static int blocking_prepare_v2_proc(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 5 && argc != 4)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " DB sql bytes tailvar", nullptr);
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	const char *sql = Jim_String(args[2]);
	int bytes;
	if (Jim_GetInt(interp, args[3], &bytes)) return JIM_ERROR;
	RC rc;
	const char *tail = nullptr;
	Vdbe *stmt = nullptr;
	int isBlocking = !(clientData == 0);
	if (isBlocking)
		rc = sqlite3_blocking_prepare_v2(ctx, sql, bytes, &stmt, &tail);
	else
		rc = sqlite3_prepare_v2(ctx, sql, bytes, &stmt, &tail);
	_assert(rc == RC_OK || !stmt);
	if (tail && argc >= 5)
	{
		if (bytes >= 0)
			bytes = bytes - (int)(tail-sql);
		Jim_SetVar2(interp, args[4], 0, Jim_NewStringObj(interp, tail, bytes), 0);
	}
	char buf[50];
	if (rc != RC_OK)
	{
		_assert(!stmt);
		_sprintf(buf, "%s ", (char *)sqlite3TestErrorName(rc));
		Jim_AppendResult(interp, buf, Main::ErrMsg(ctx), nullptr);
		return JIM_ERROR;
	}
	if (stmt)
	{
		if (sqlite3TestMakePointerStr(interp, buf, stmt)) return JIM_ERROR;
		Jim_AppendResult(interp, buf, nullptr);
	}
	return JIM_OK;
}

#endif
#pragma endregion 

// Register commands with the TCL interpreter.
__device__ int SqlitetestThread_Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "sqlthread", sqlthread_proc, nullptr, nullptr);
	Jim_CreateCommand(interp, "clock_seconds", clock_seconds_proc, nullptr, nullptr);
#if OS_UNIX && defined(ENABLE_UNLOCK_NOTIFY)
	Jim_CreateCommand(interp, "sqlite3_blocking_step", blocking_step_proc, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_blocking_prepare_v2", blocking_prepare_v2_proc, (void *)1, nullptr);
	Tcl_CreateCommand(interp, "sqlite3_nonblocking_prepare_v2", blocking_prepare_v2_proc, nullptr, nullptr);
#endif
	return JIM_OK;
}
#else
__device__ int SqlitetestThread_Init(Jim_Interp *interp)
{
	return JIM_OK;
}
#endif
