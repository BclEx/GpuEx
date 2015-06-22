
// A TCL Interface to SQLite.  Append this file to sqlite3.c and compile the whole thing to build a TCL-enabled version of SQLite.
//
// Compile-time options:
//
//  -DTCLSH=1             Add a "main()" routine that works as a tclsh.
//
//  -DSQLITE_TCLMD5       When used in conjuction with -DTCLSH=1, add four new commands to the TCL interpreter for
//                        generating MD5 checksums:  md5, md5file, md5-10x8, and md5file-10x8.
//
//  -DSQLITE_TEST         When used in conjuction with -DTCLSH=1, add hundreds of new commands used for testing
//                        SQLite.  This option implies -DSQLITE_TCLMD5.
//#include <errno.h>

// Some additional include files are needed if this file is not appended to the amalgamation.
#include "TclContext.cu.h"
#include <string.h>
#include <stdlib.h>

#define NUM_PREPARED_STMTS 10
#define MAX_PREPARED_STMTS 100

#pragma region BLOB
#ifdef OMIT_INCRBLOB

// Close all incrblob channels opened using database connection pDb.
// This is called when shutting down the database connection.
__device__ static void CloseIncrblobChannels(TclContext *tctx)
{
	IncrblobChannel *next;
	for (IncrblobChannel *p = tctx->Incrblob; p; p = next)
	{
		next = p->Next;
		// Note: Calling unregister here call Tcl_Close on the incrblob channel, which deletes the IncrblobChannel structure at *p. So do not call Tcl_Free() here.
		Tcl_UnregisterChannel(tctx->Interp, p->Channel);
	}
}

// Close an incremental blob channel.
__device__ static int IncrblobClose(ClientData instanceData, Tcl_Interp *interp)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int rc = Vdbe::Blob_Close(p->Blob);
	Context *ctx = p->Ctx->Ctx;

	// Remove the channel from the SqliteDb.pIncrblob list.
	if (p->Next)
		p->Next->Prev = p->Prev;
	if (p->Prev)
		p->Prev->Next = p->Next;
	if (p->Ctx->Incrblobs == p)
		p->Ctx->Incrblobs = p->Next;

	// Free the IncrblobChannel structure
	Tcl_Free((char *)p);

	if (rc != RC_OK)
	{
		Tcl_SetResult(interp, (char *)Main::ErrMsg(ctx), TCL_VOLATILE);
		return TCL_ERROR;
	}
	return TCL_OK;
}

// Read data from an incremental blob channel.
__device__ static int IncrblobInput(ClientData instanceData, char *buf, int bufSize, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int read = bufSize; // Number of bytes to read
	int blob = Vdbe::Blob_Bytes(p->Blob); // Total size of the blob
	if ((p->Seek + read) > blob)
		read = blob - p->Seek;
	if (read <= 0)
		return 0;
	RC rc = Vdbe::Blob_Read(p->Blob, (void *)buf, read, p->Seek);
	if (rc != RC_OK)
	{
		*errorCodePtr = rc;
		return -1;
	}
	p->Seek += read;
	return read;
}

//  Write data to an incremental blob channel.
__device__ static int IncrblobOutput(ClientData instanceData, const char *buf, int toWrite, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int write = toWrite; // Number of bytes to write
	int blob = Vdbe::Blob_Bytes(p->Blob); // Total size of the blob
	if ((p->Seek + write) > blob)
	{
		*errorCodePtr = EINVAL;
		return -1;
	}
	if (write <= 0)
		return 0;
	RC rc = Vdbe::Blob_Write(p->Blob, (void *)buf, write, p->Seek);
	if (rc != RC_OK)
	{
		*errorCodePtr = EIO;
		return -1;
	}
	p->Seek += write;
	return write;
}

// Seek an incremental blob channel.
__device__ static int IncrblobSeek(ClientData instanceData, long offset, int seekMode, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	switch (seekMode)
	{
	case SEEK_SET:
		p->Seek = offset;
		break;
	case SEEK_CUR:
		p->Seek += offset;
		break;
	case SEEK_END:
		p->Seek = Vdbe::Blob_Bytes(p->Blob) + offset;
		break;
	default: assert(!"Bad seekMode");
	}
	return p->Seek;
}

__device__ static void IncrblobWatch(ClientData instanceData, int mode) { } // NO-OP
__device__ static int IncrblobHandle(ClientData instanceData, int dir, ClientData *ptr)
{
	return TCL_ERROR;
}

__constant__ static Tcl_ChannelType IncrblobChannelType = {
	"incrblob",					// typeName                            
	TCL_CHANNEL_VERSION_2,		// version                             
	IncrblobClose,				// closeProc                           
	IncrblobInput,				// inputProc                           
	IncrblobOutput,				// outputProc                          
	IncrblobSeek,				// seekProc                            
	nullptr,                    // setOptionProc                       
	nullptr,                    // getOptionProc                       
	IncrblobWatch,				// watchProc (this is a no-op)         
	IncrblobHandle,				// getHandleProc (always returns error)
	nullptr,					// close2Proc                          
	nullptr,					// blockModeProc                       
	nullptr,					// flushProc                           
	nullptr,					// handlerProc                         
	nullptr,					// wideSeekProc                        
};

// Create a new incrblob channel.
__device__ static int CreateIncrblobChannel(Tcl_Interp *interp, TclContext *pDb, const char *dbName, const char *tableName, const char *columnName, int64 row, bool isReadonly)
{
	IncrblobChannel *p;
	Context *ctx = tctx->Ctx;
	int rc;
	int flags = TCL_READABLE | (isReadonly ? 0 : TCL_WRITABLE);

	Blob *blob;
	RC rc = Vdbe::Blob_Open(ctx, dbName, tableName, columnName, row, !isReadonly, &blob);
	if (rc != RC_OK)
	{
		Tcl_SetResult(interp, (char *)Main::ErrMsg(ctx), TCL_VOLATILE);
		return TCL_ERROR;
	}

	p = (IncrblobChannel *)Tcl_Alloc(sizeof(IncrblobChannel));
	p->Seek = 0;
	p->Blob = blob;

	static int count = 0; // This variable is used to name the channels: "incrblob_[incr count]"
	char channelName[64];
	__snprintf(channelName, sizeof(channelName), "incrblob_%d", ++count);
	p->Channel = Tcl_CreateChannel(&IncrblobChannelType, channelName, p, flags);
	Tcl_RegisterChannel(interp, p->Channel);

	// Link the new channel into the SqliteDb.pIncrblob list.
	p->Next = tctx->Incrblob;
	p->Prev = nullptr;
	if (p->Next)
		p->Next->Prev = p;
	tctx->Incrblob = p;
	p->Ctx = tctx;

	Tcl_SetResult(interp, (char *)Tcl_GetChannelName(p->Channel), TCL_VOLATILE);
	return TCL_OK;
}
#else
#define CloseIncrblobChannels(tctx)
#endif
#pragma endregion

#pragma region Stmt

// Look at the script prefix in pCmd.  We will be executing this script after first appending one or more arguments.  This routine analyzes
// the script to see if it is safe to use Tcl_EvalObjv() on the script rather than the more general Tcl_EvalEx().  Tcl_EvalObjv() is much faster.
//
// Scripts that are safe to use with Tcl_EvalObjv() consists of a command name followed by zero or more arguments with no [...] or $
// or {...} or ; to be seen anywhere.  Most callback scripts consist of just a single procedure name and they meet this requirement.
__device__ static bool SafeToUseEvalObjv(Tcl_Interp *interp, Tcl_Obj *cmd)
{
	// We could try to do something with Tcl_Parse().  But we will instead just do a search for forbidden characters.  If any of the forbidden
	// characters appear in pCmd, we will report the string as unsafe.
	int n;
	const char *z = Tcl_GetString(interp, cmd, &n);
	while (n-- > 0)
	{
		int c = *(z++);
		if (c == '$' || c == '[' || c == ';') return false;
	}
	return true;
}

// Find an SqlFunc structure with the given name.  Or create a new one if an existing one cannot be found.  Return a pointer to the structure.
__device__ static SqlFunc *FindSqlFunc(TclContext *tctx, const char *name)
{
	SqlFunc *newFunc = (SqlFunc *)Tcl_Alloc(sizeof(*newFunc) + _strlen(name) + 1);
	newFunc->Name = (char *)&newFunc[1];
	int i;
	for (i = 0; name[i]; i++) { newFunc->Name[i] = __tolower(name[i]); }
	newFunc->Name[i] = 0;
	for (SqlFunc *p = tctx->Funcs; p; p = p->Next)
	{ 
		if (!_strcmp(p->Name, newFunc->Name))
		{
			Tcl_Free((char *)newFunc);
			return p;
		}
	}
	newFunc->Interp = tctx->Interp;
	newFunc->Ctx = tctx;
	newFunc->Script = nullptr;
	newFunc->Next = tctx->Funcs;
	tctx->Funcs = newFunc;
	return newFunc;
}

// Free a single SqlPreparedStmt object.
__device__ static void DbFreeStmt(SqlPreparedStmt *stmt)
{
#ifdef _TEST
	if (!Vdbe::Sql(stmt->Stmt))
		Tcl_Free((char *)stmt->Sql);
#endif
	Vdbe::Finalize(stmt->Stmt);
	Tcl_Free(stmt);
}

// Finalize and free a list of prepared statements
__device__ static void FlushStmtCache(TclContext *tctx)
{
	SqlPreparedStmt *next;
	for (SqlPreparedStmt *p = tctx->Stmts.data; p; p = next)
	{
		next = p->Next;
		DbFreeStmt(p);
	}
	tctx->Stmts.length = 0;
	tctx->StmtLast = nullptr;
	tctx->Stmts.data = nullptr;
}

// TCL calls this procedure when an sqlite3 database command is deleted.
__device__ static void DbDeleteCmd(void *db)
{
	TclContext *tctx = (TclContext *)db;
	FlushStmtCache(tctx);
	CloseIncrblobChannels(tctx);
	Main::Close(tctx->Ctx);
	while (tctx->Funcs)
	{
		SqlFunc *func = tctx->Funcs;
		tctx->Funcs = func->Next;
		_assert(func->Ctx == tctx);
		Tcl_DecrRefCount(func->Script);
		Tcl_Free(func);
	}
	while (tctx->Collates)
	{
		SqlCollate *collate = tctx->Collates;
		tctx->Collates = collate->Next;
		Tcl_Free(collate);
	}
	if (tctx->Busy)
		Tcl_Free(tctx->Busy);
	if (tctx->Trace)
		Tcl_Free(tctx->Trace);
	if (tctx->Profile)
		Tcl_Free(tctx->Profile);
	if (tctx->Auth)
		Tcl_Free(tctx->Auth);
	if (tctx->NullText)
		Tcl_Free(tctx->NullText);
	if (tctx->UpdateHook)
		Tcl_DecrRefCount(tctx->UpdateHook);
	if (tctx->RollbackHook)
		Tcl_DecrRefCount(tctx->RollbackHook);
	if (tctx->WalHook)
		Tcl_DecrRefCount(tctx->WalHook);
	if (tctx->CollateNeeded)
		Tcl_DecrRefCount(tctx->CollateNeeded);
	Tcl_Free(tctx);
}

#pragma endregion

#pragma region Hooks

// This routine is called when a database file is locked while trying to execute SQL.
__device__ static int DbBusyHandler(void *cd, int tries)
{
	TclContext *tctx = (TclContext *)cd;
	char b[30];
	__snprintf(b, sizeof(b), "%d", tries);
	int rc = Tcl_VarEval(tctx->Interp, tctx->Busy, " ", b, (char *)0);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return 0;
	return 1;
}

#ifndef OMIT_PROGRESS_CALLBACK
// This routine is invoked as the 'progress callback' for the database.
__device__ static bool DbProgressHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	_assert(tctx->Progress);
	int rc = Tcl_Eval(tctx->Interp, tctx->Progress);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return true;
	return false;
}
#endif

#ifndef OMIT_TRACE
// This routine is called by the SQLite trace handler whenever a new block of SQL is executed.  The TCL script in pDb->zTrace is executed.
__device__ static void DbTraceHandler(void *cd, const char *sql)
{
	TclContext *tctx = (TclContext *)cd;
	char *cmd = _mprintf("%s%s", tctx->Trace, sql);
	Tcl_Eval(tctx->Interp, cmd, 0, nullptr);
	_free(cmd);
	Tcl_ResetResult(tctx->Interp);
}

// This routine is called by the SQLite profile handler after a statement SQL has executed.  The TCL script in pDb->zProfile is evaluated.
__device__ static void DbProfileHandler(void *cd, const char *sql, uint64 tm)
{
	TclContext *tctx = (TclContext *)cd;
	char tmAsString[100];
	__snprintf(tmAsString, sizeof(tmAsString)-1, "%lld", tm);
	char *cmd = _mprintf("%s%s%s", tctx->Profile, sql, tmAsString);
	Tcl_Eval(tctx->Interp, cmd, 0, nullptr);
	_free(cmd);
	Tcl_ResetResult(tctx->Interp);
}
#endif

// This routine is called when a transaction is committed.  The TCL script in pDb->zCommit is executed.  If it returns non-zero or
// if it throws an exception, the transaction is rolled back instead of being committed.
__device__ static bool DbCommitHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	int rc = Tcl_Eval(tctx->Interp, tctx->Commit, 0, nullptr);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return true;
	return false;
}

__device__ static void DbRollbackHandler(void *clientData)
{
	TclContext *tctx = (TclContext *)clientData;
	_assert(tctx->RollbackHook);
	if (Tcl_Eval(tctx->Interp, tctx->RollbackHook, 0, nullptr) != TCL_OK)
		Tcl_BackgroundError(tctx->Interp);
}

// This procedure handles wal_hook callbacks.
__device__ static int DbWalHandler(void *clientData, Context *ctx, const char *dbName,  int entrys)
{
	TclContext *tctx = (TclContext *)clientData;
	Tcl_Interp *interp = tctx->Interp;
	_assert(tctx->WalHook);
	char b[50];
	Tcl_SetResult(interp, tctx->WalHook, nullptr);
	Tcl_AppendElement(interp, dbName, false);
	Tcl_AppendElement(interp, _itoa(entrys, b), false);
	char *cmd = interp->result;
	int rc = RC_OK;
	if (Tcl_Eval(interp, cmd, 0, nullptr) != TCL_OK || Tcl_GetInt(interp, interp->result, &rc) != TCL_OK)
		Tcl_BackgroundError(tctx->Interp);
	return rc;
}

#if defined(_TEST) && defined(ENABLE_UNLOCK_NOTIFY)
__device__ static void SetTestUnlockNotifyVars(Tcl_Interp *interp, int argId, int argsLength)
{
	char b[64];
	__snprintf(b, sizeof(b), "%d", argId);
	Tcl_SetVar(interp, "sqlite_unlock_notify_arg", b, TCL_GLOBAL_ONLY);
	__snprintf(b, sizeof(b), "%d", argsLength);
	Tcl_SetVar(interp, "sqlite_unlock_notify_argcount", b, TCL_GLOBAL_ONLY);
}
#else
#define SetTestUnlockNotifyVars(x,y,z)
#endif

#ifdef ENABLE_UNLOCK_NOTIFY
__device__ static void DbUnlockNotify(void **args, int argsLength)
{
	for (int i = 0; i < argsLength; i++)
	{
		const int flags = (TCL_EVAL_GLOBAL | TCL_EVAL_DIRECT);
		TclContext *tctx = (TclContext *)args[i];
		Tcl_Interp *interp = tctx->Interp;
		SetTestUnlockNotifyVars(interp, i, argsLength);
		_assert(tctx->UnlockNotify);
		Tcl_Eval(interp, tctx->UnlockNotify, flags);
		tctx->UnlockNotify = nullptr;
	}
}
#endif

__device__ static void DbUpdateHandler(void *p, TK op, const char *dbName, const char *tableName,  int64 rowid)
{
	TclContext *tctx = (TclContext *)p;
	Tcl_Interp *interp = tctx->Interp;
	_assert(tctx->UpdateHook);
	_assert(op == TK_INSERT || op == TK_UPDATE || op == TK_DELETE);
	char b[50];
	Tcl_SetResult(interp, tctx->UpdateHook, nullptr);
	Tcl_AppendElement(interp, (op == TK_INSERT?"INSERT":(op == TK_UPDATE?"UPDATE":"DELETE"), false);
	Tcl_AppendElement(interp, dbName, false);
	Tcl_AppendElement(interp, tableName, false);
	Tcl_AppendElement(interp, _itoa(rowid, b), false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
}

__device__ static void TclCollateNeeded(void *p, Context *ctx, TEXTENCODE encode, const char *name)
{
	TclContext *tctx = (TclContext *)p;
	Tcl_Interp *interp = tctx->Interp;
	Tcl_SetResult(interp, tctx->CollateNeeded, nullptr);
	Tcl_AppendElement(interp, name, false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
}

// This routine is called to evaluate an SQL collation function implemented using TCL script.
__device__ static int TclSqlCollate(void *p1, int aLength, const void *a, int bLength, const void *b)
{
	SqlCollate *p = (SqlCollate *)p1;
	Tcl_Interp *interp = p->Interp;
	Tcl_SetResult(interp, p->Script, nullptr);
	Tcl_AppendElement(p->Interp, (const char *)a, false); //Tcl_AppendElement(p->Interp, Tcl_Obj::NewStringObj((const char *)a, aLength), false);
	Tcl_AppendElement(p->Interp, (const char *)b, false); //Tcl_AppendElement(p->Interp, Tcl_Obj::NewStringObj((const char *)b, bLength), false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
	return _atoi(interp->result);
}

// This routine is called to evaluate an SQL function implemented using TCL script.
__device__ static void TclSqlFunc(FuncContext *fctx, int argc, Mem **argv)
{
	SqlFunc *p = (SqlFunc *)Vdbe::User_Data(fctx);
	Tcl_Interp *interp = p->Interp;
	Tcl_Obj *cmd;
	int rc;
	if (argc == 0)
	{
		// If there are no arguments to the function, call Tcl_EvalObjEx on the script object directly.  This allows the TCL compiler to generate
		// bytecode for the command on the first invocation and thus make subsequent invocations much faster.
		cmd = p->Script;
		rc = Tcl_Eval(interp, cmd, 0, nullptr);
	}
	else
	{
		// If there are arguments to the function, make a shallow copy of the script object, lappend the arguments, then evaluate the copy.
		//
		// By "shallow" copy, we mean a only the outer list Tcl_Obj is duplicated. The new Tcl_Obj contains pointers to the original list elements. 
		// That way, when Tcl_EvalObjv() is run and shimmers the first element of the list to tclCmdNameType, that alternate representation will
		// be preserved and reused on the next invocation.
		Tcl_Obj **args;
		int argsLength;
		if (Tcl_SplitList(interp, p->Script, &argsLength, &args))
		{
			Vdbe::Result_Error(fctx, interp->result, -1); 
			return;
		}     
		cmd = Tcl_Obj::NewListObj(argsLength, args);
		cmd->IncrRefCount();
		for (int i = 0; i < argc; i++)
		{
			Mem *in = argv[i];
			Tcl_Obj *val;

			// Set pVal to contain the i'th column of this row.
			char b[50];
			switch (Vdbe::Value_Type(in))
			{
			case TYPE_BLOB: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Tcl_Obj::NewByteArrayObj(Vdbe::Value_Blob(in), bytes);
				break; }
			case TYPE_INTEGER: {
				int64 v = Vdbe::Value_Int64(in);
				val = (v >= -2147483647 && v <= 2147483647 ? _itoa((int)v, b) : _itoa(v, b));
				break; }
			case TYPE_FLOAT: {
				double r = Vdbe::Value_Double(in);
				val = Tcl_Obj::NewDoubleObj(r);
				break; }
			case TYPE_NULL: {
				val = p->Ctx->NullText;
				break; }
			default: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Tcl_Obj::NewStringObj((char *)Vdbe::Value_Text(in), bytes);
				break; }
			}
			rc = cmd->ListObjAppendElement(p->Interp, val);
			if (rc)
			{
				Tcl_DecrRefCount(cmd);
				Vdbe::Result_Error(fctx, interp->result, -1); 
				return;
			}
		}
		// Tcl_EvalObjEx() will automatically call Tcl_EvalObjv() if pCmd is a list without a string representation.  To prevent this from happening, make sure pCmd has a valid string representation
		if (!p->UseEvalObjv)
			Tcl_GetString(cmd);
		rc = Tcl_Eval(interp, cmd, 0, nullptr);
		Tcl_DecrRefCount(cmd);
	}

	if (rc && rc != RC_BUSY)
		Vdbe::Result_Error(fctx, interp->result, -1); 
	else
	{
		Tcl_Obj *var = interp->result;
		int n;
		uint8 *data;
		const char *typeName = (var->TypePtr ? var->TypePtr->Name : "");
		char c = typeName[0];
		if (c == 'b' && !_strcmp(typeName, "bytearray") && var->Bytes == 0)
		{
			// Only return a BLOB type if the Tcl variable is a bytearray and has no string representation.
			data = Tcl_GetByteArray(interp, var, &n);
			Vdbe::Result_Blob(fctx, data, n, DESTRUCTOR_TRANSIENT);
		}
		else if (c == 'b' && !_strcmp(typeName, "boolean"))
		{
			Tcl_GetInt(interp, var, &n);
			Vdbe::Result_Int(fctx, n);
		}
		else if (c == 'd' && !_strcmp(typeName, "double"))
		{
			double r;
			Tcl_GetDouble(interp, var, &r);
			Vdbe::Result_Double(fctx, r);
		}
		else if ((c == 'w' && !_strcmp(typeName, "wideInt")) || (c == 'i' && !_strcmp(typeName, "int")))
		{
			int64 v;
			Tcl_GetWideInt(nullptr, var, &v);
			Vdbe::Result_Int64(fctx, v);
		}
		else
		{
			data = (unsigned char *)Tcl_GetString(interp, var, &n);
			Vdbe::Result_Text(fctx, (char *)data, n, DESTRUCTOR_TRANSIENT);
		}
	}
}

#ifndef OMIT_AUTHORIZATION
// This is the authentication function.  It appends the authentication type code and the two arguments to zCmd[] then invokes the result
// on the interpreter.  The reply is examined to determine if the authentication fails or succeeds.
__device__ static ARC AuthCallback(void *arg, int code, const char *arg1, const char *arg2, const char *arg3, const char *arg4)
{
	TclContext *tctx = (TclContext *)arg;
	Tcl_Interp *interp = tctx->Interp;
	if (tctx->DisableAuth) return ARC_OK;

	char *codeName;
	switch (code)
	{
	case AUTH_COPY              : codeName="AUTH_COPY"; break;
	case AUTH_CREATE_INDEX      : codeName="AUTH_CREATE_INDEX"; break;
	case AUTH_CREATE_TABLE      : codeName="AUTH_CREATE_TABLE"; break;
	case AUTH_CREATE_TEMP_INDEX : codeName="AUTH_CREATE_TEMP_INDEX"; break;
	case AUTH_CREATE_TEMP_TABLE : codeName="AUTH_CREATE_TEMP_TABLE"; break;
	case AUTH_CREATE_TEMP_TRIGGER: codeName="AUTH_CREATE_TEMP_TRIGGER"; break;
	case AUTH_CREATE_TEMP_VIEW  : codeName="AUTH_CREATE_TEMP_VIEW"; break;
	case AUTH_CREATE_TRIGGER    : codeName="AUTH_CREATE_TRIGGER"; break;
	case AUTH_CREATE_VIEW       : codeName="AUTH_CREATE_VIEW"; break;
	case AUTH_DELETE            : codeName="AUTH_DELETE"; break;
	case AUTH_DROP_INDEX        : codeName="AUTH_DROP_INDEX"; break;
	case AUTH_DROP_TABLE        : codeName="AUTH_DROP_TABLE"; break;
	case AUTH_DROP_TEMP_INDEX   : codeName="AUTH_DROP_TEMP_INDEX"; break;
	case AUTH_DROP_TEMP_TABLE   : codeName="AUTH_DROP_TEMP_TABLE"; break;
	case AUTH_DROP_TEMP_TRIGGER : codeName="AUTH_DROP_TEMP_TRIGGER"; break;
	case AUTH_DROP_TEMP_VIEW    : codeName="AUTH_DROP_TEMP_VIEW"; break;
	case AUTH_DROP_TRIGGER      : codeName="AUTH_DROP_TRIGGER"; break;
	case AUTH_DROP_VIEW         : codeName="AUTH_DROP_VIEW"; break;
	case AUTH_INSERT            : codeName="AUTH_INSERT"; break;
	case AUTH_PRAGMA            : codeName="AUTH_PRAGMA"; break;
	case AUTH_READ              : codeName="AUTH_READ"; break;
	case AUTH_SELECT            : codeName="AUTH_SELECT"; break;
	case AUTH_TRANSACTION       : codeName="AUTH_TRANSACTION"; break;
	case AUTH_UPDATE            : codeName="AUTH_UPDATE"; break;
	case AUTH_ATTACH            : codeName="AUTH_ATTACH"; break;
	case AUTH_DETACH            : codeName="AUTH_DETACH"; break;
	case AUTH_ALTER_TABLE       : codeName="AUTH_ALTER_TABLE"; break;
	case AUTH_REINDEX           : codeName="AUTH_REINDEX"; break;
	case AUTH_ANALYZE           : codeName="AUTH_ANALYZE"; break;
	case AUTH_CREATE_VTABLE     : codeName="AUTH_CREATE_VTABLE"; break;
	case AUTH_DROP_VTABLE       : codeName="AUTH_DROP_VTABLE"; break;
	case AUTH_FUNCTION          : codeName="AUTH_FUNCTION"; break;
	case AUTH_SAVEPOINT         : codeName="AUTH_SAVEPOINT"; break;
	default                     : codeName="????"; break;
	}
	char *str = _mprintf("%s%s%s%s%s", tctx->Auth, codeName, (arg1?arg1:""), (arg2?arg2:""), (arg3?arg3:""), (arg4?arg4:""));
	int rc2 = Tcl_GlobalEval(interp, str);
	_free(str);
	ARC rc = ARC_OK;
	const char *reply = (rc == RC_OK ? interp->result : "ARC_DENY");
	if (!_strcmp(reply, "ARC_OK")) rc = ARC_OK;
	else if (!_strcmp(reply, "ARC_DENY")) rc = ARC_DENY;
	else if (!_strcmp(reply, "SQLITE_IGNORE")) rc = ARC_IGNORE;
	else rc = (ARC)999;
	return rc;
}
#endif

#pragma endregion

#pragma region DB

__constant__ static const char *_ends[] = {
	"RELEASE _tcl_transaction",        // rc==TCL_ERROR, nTransaction!=0
	"COMMIT",                          // rc!=TCL_ERROR, nTransaction==0
	"ROLLBACK TO _tcl_transaction ; RELEASE _tcl_transaction",
	"ROLLBACK"                         // rc==TCL_ERROR, nTransaction==0
};
__device__ static int DbTransPostCmd(ClientData data[], Tcl_Interp *interp, int result)
{
	TclContext *tctx = (TclContext *)data[0];
	Context *ctx = tctx->Ctx;
	int rc = result;

	tctx->Transactions--;
	const char *end = _ends[(rc == RC_ERROR) * 2 + (tctx->Transactions == 0)];

	tctx->DisableAuth++;
	if (Main::Exec(ctx, end, nullptr, nullptr, nullptr))
	{
		// This is a tricky scenario to handle. The most likely cause of an error is that the exec() above was an attempt to commit the 
		// top-level transaction that returned SQLITE_BUSY. Or, less likely, that an IO-error has occurred. In either case, throw a Tcl exception
		// and try to rollback the transaction.
		//
		// But it could also be that the user executed one or more BEGIN, COMMIT, SAVEPOINT, RELEASE or ROLLBACK commands that are confusing
		// this method's logic. Not clear how this would be best handled.
		if (rc != RC_ERROR)
		{
			Tcl_AppendResult(interp, Main::ErrMsg(ctx), -1, 0);
			rc = RC_ERROR;
		}
		Main::Exec(ctx, "ROLLBACK", nullptr, nullptr, nullptr);
	}
	tctx->DisableAuth--;
	return rc;
}

__device__ static RC DbPrepare(TclContext *tctx, const char *sql, Vdbe **stmt, const char **out)
{
#ifdef _TEST
	if (tctx->LegacyPrepare)
		return Prepare::Prepare_(tctx->Ctx, sql, -1, stmt, out);
#endif
	return Prepare::Prepare_v2(tctx->Ctx, sql, -1, stmt, out);
}

__device__ static RC DbPrepareAndBind(TclContext *tctx, char const *sql, char const **out, SqlPreparedStmt **preStmt)
{
	Context *ctx = tctx->Ctx;
	Tcl_Interp *interp = tctx->Interp;
	*preStmt = nullptr;

	// Trim spaces from the start of zSql and calculate the remaining length.
	while (_isspace(sql[0])) sql++;
	int sqlLength = _strlen(sql);

	SqlPreparedStmt *p;
	Vdbe *stmt;
	int vars;
	for (p = tctx->Stmts; p; p = p->Next)
	{
		int n = p->SqlLength;
		if (sqlLength >= n && !_memcmp(p->Sql, sql, n) && (sql[n] == 0 || sql[n-1] == ';'))
		{
			stmt = p->Stmt;
			*out = &sql[p->SqlLength];

			// When a prepared statement is found, unlink it from the cache list.  It will later be added back to the beginning
			// of the cache list in order to implement LRU replacement.
			if (p->Prev)
				p->Prev->Next = p->Next;
			else
				tctx->Stmts = p->Next;
			if (p->Next)
				p->Next->Prev = p->Prev;
			else
				tctx->StmtLast = p->Prev;
			tctx->Stmts.length--;
			vars = Vdbe::Bind_ParameterCount(stmt);
			break;
		}
	}

	// If no prepared statement was found. Compile the SQL text. Also allocate a new SqlPreparedStmt structure.
	if (!p)
	{
		if (DbPrepare(tctx, sql, &stmt, out) != RC_OK)
		{
			Tcl_SetResult(interp, (char *)Main::ErrMsg(ctx), nullptr);
			return RC_ERROR;
		}
		if (!stmt)
		{
			if (Main::ErrCode(ctx) != RC_OK)
			{
				Tcl_SetResult(interp, (char *)Main::ErrMsg(ctx), nullptr);
				return RC_ERROR; // A compile-time error in the statement.
			}
			else
				return RC_OK; // The statement was a no-op.  Continue to the next statement in the SQL string.
		}

		_assert(!p);
		vars = Vdbe::Bind_ParameterCount(stmt);
		int bytes = sizeof(SqlPreparedStmt) + vars * sizeof(Tcl_Obj *);
		p = (SqlPreparedStmt *)_alloc(bytes);
		_memset(p, 0, bytes);

		p->Stmt = stmt;
		p->SqlLength = (int)(*out - sql);
		p->Sql = Vdbe::Sql(stmt);
		p->Parms.data = (Tcl_Obj **)&p[1];
#ifdef _TEST
		if (!p->Sql)
		{
			char *copy = (char *)_alloc(p->SqlLength + 1);
			_memcpy(copy, sql, p->SqlLength);
			copy[p->SqlLength] = '\0';
			p->Sql = copy;
		}
#endif
	}
	_assert(p);
	_assert(_strlen(p->Sql) == p->SqlLength);
	_assert(!_memcmp(p->Sql, sql, p->SqlLength));

	// Bind values to parameters that begin with $ or :
	int parmsLength = 0;
	for (int i = 1; i <= vars; i++)
	{
		const char *varName = Vdbe::Bind_ParameterName(stmt, i);
		if (varName && (varName[0] == '$' || varName[0] == ':' || varName[0] == '@'))
		{
			Tcl_Obj *var = Tcl_Obj::GetVar2Ex(interp, &varName[1], 0, 0);
			if (var)
			{
				int n;
				uint8 *data;
				const char *typeName = (var->TypePtr ? var->TypePtr->Name : "");
				char c = typeName[0];
				if (varName[0] == '@' || (c == 'b' && !_strcmp(typeName, "bytearray") && var->Bytes == 0))
				{
					// Load a BLOB type if the Tcl variable is a bytearray and it has no string representation or the host parameter name begins with "@".
					data = Tcl_GetByteArray(interp, var, &n);
					Vdbe::Bind_Blob(stmt, i, data, n, DESTRUCTOR_STATIC);
					Tcl_IncrRefCount(var);
					p->Parms[parmsLength++] = var;
				}
				else if (c == 'b' && !_strcmp(typeName, "boolean"))
				{
					Tcl_GetInt(interp, var, &n);
					Vdbe::Bind_Int(stmt, i, n);
				}
				else if (c == 'd' && !_strcmp(typeName, "double"))
				{
					double r;
					Tcl_GetDouble(interp, var, &r);
					Vdbe::Bind_Double(stmt, i, r);
				}
				else if ((c == 'w' && !_strcmp(typeName, "wideInt")) || (c == 'i' && !_strcmp(typeName, "int")))
				{
					int64 v;
					Tcl_GetWideInt(interp, var, &v);
					Vdbe::Bind_Int64(stmt, i, v);
				}
				else
				{
					data = (unsigned char *)Tcl_GetString(interp, var, &n);
					Vdbe::Bind_Text(stmt, i, (char *)data, n, DESTRUCTOR_STATIC);
					Tcl_IncrRefCount(var);
					p->Parms[parmsLength++] = var;
				}
			}
			else
				Vdbe::Bind_Null(stmt, i);
		}
	}
	p->Parms.length = parmsLength;
	*preStmt = p;

	return RC_OK;
}

__device__ static void DbReleaseStmt(TclContext *tctx, SqlPreparedStmt *preStmt, bool discard)
{
	// Free the bound string and blob parameters
	for (int i = 0; i < preStmt->Parms.length; i++)
		Tcl_DecrRefCount(preStmt->Parms[i]);
	preStmt->Parms.length = 0;

	if (tctx->MaxStmt <= 0 || discard)
		DbFreeStmt(preStmt); // If the cache is turned off, deallocated the statement
	else
	{
		// Add the prepared statement to the beginning of the cache list.
		preStmt->Next = tctx->Stmts.data;
		preStmt->Prev = nullptr;
		if (tctx->Stmts.data)
			tctx->Stmts.data->Prev = preStmt;
		tctx->Stmts.data = preStmt;
		if (!tctx->StmtLast)
		{
			_assert(tctx->Stmts.length == 0);
			tctx->StmtLast = preStmt;
		}
		else
			_assert(tctx->Stmts.length > 0);
		tctx->Stmts.length++;

		// If we have too many statement in cache, remove the surplus from the end of the cache list.
		while (tctx->Stmts.length > tctx->MaxStmt)
		{
			SqlPreparedStmt *last = tctx->StmtLast;
			tctx->StmtLast = last->Prev;
			tctx->StmtLast->Next = nullptr;
			tctx->Stmts.length--;
			DbFreeStmt(last);
		}
	}
}

#pragma endregion

#pragma region EVAL

// dbEvalInit()
// dbEvalStep()
// dbEvalFinalize()
// dbEvalRowInfo()
// dbEvalColumnValue()

typedef struct DbEvalContext DbEvalContext;
struct DbEvalContext
{
	TclContext *Ctx;               // Database handle
	Tcl_Obj *Sql;               // Object holding string zSql
	const char *SqlAsString;    // Remaining SQL to execute
	SqlPreparedStmt *PreStmt;   // Current statement
	int Cols;                   // Number of columns returned by pStmt
	Tcl_Obj *Array;             // Name of array variable
	Tcl_Obj **ColNames;         // Array of column names
};

// Release any cache of column names currently held as part of the DbEvalContext structure passed as the first argument.
__device__ static void DbReleaseColumnNames(DbEvalContext *p)
{
	if (p->ColNames)
	{
		for (int i = 0; i < p->Cols; i++)
			p->ColNames[i]->DecrRefCount();
		_free(p->ColNames);
		p->ColNames = nullptr;
	}
	p->Cols = 0;
}

// Initialize a DbEvalContext structure.
//
// If pArray is not NULL, then it contains the name of a Tcl array variable. The "*" member of this array is set to a list containing
// the names of the columns returned by the statement as part of each call to dbEvalStep(), in order from left to right. e.g. if the names 
// of the returned columns are a, b and c, it does the equivalent of the tcl command:
//
//     set ${pArray}(*) {a b c}
__device__ static void DbEvalInit(DbEvalContext *p, TclContext *tctx, Tcl_Obj *sql, Tcl_Obj *array)
{
	_memset(p, 0, sizeof(DbEvalContext));
	p->Ctx = tctx;
	p->SqlAsString = (char *)sql->GetString();
	p->Sql = sql;
	sql->IncrRefCount();
	if (array)
	{
		p->Array = array;
		array->IncrRefCount();
	}
}

// Obtain information about the row that the DbEvalContext passed as the first argument currently points to.
__device__ static void DbEvalRowInfo(DbEvalContext *p, int *colsOut, Tcl_Obj ***colNamesOut)
{
	// Compute column names
	if (!p->ColNames)
	{
		Vdbe *stmt = p->PreStmt->Stmt;
		int i;
		Tcl_Obj **colNames = nullptr; // Array of column names
		int cols = p->Cols = Vdbe::Column_Count(stmt); // Number of columns returned by pStmt
		if (cols > 0 && (colNamesOut || p->Array))
		{
			colNames = (Tcl_Obj **)_alloc(sizeof(Tcl_Obj *) * cols);
			for (i = 0; i < cols; i++)
			{
				colNames[i] = Tcl_Obj::NewStringObj(Vdbe::Column_Name(stmt, i), -1);
				colNames[i]->IncrRefCount();
			}
			p->ColNames = colNames;
		}

		// If results are being stored in an array variable, then create the array(*) entry for that array
		if (p->Array)
		{
			Tcl_Interp *interp = p->Ctx->Interp;
			Tcl_Obj *colList = Tcl_Obj::NewObj();
			Tcl_Obj *star = Tcl_Obj::NewStringObj("*", -1);
			for (i = 0; i < cols; i++)
				colList->ListObjAppendElement(interp, colNames[i]);
			star->IncrRefCount();
			p->Array->ObjSetVar2(interp, star, colList, 0);
			star->DecrRefCount();
		}
	}

	if (colNamesOut)
		*colNamesOut = p->ColNames;
	if (colsOut)
		*colsOut = p->Cols;
}

// Return one of TCL_OK, TCL_BREAK or TCL_ERROR. If TCL_ERROR is returned, then an error message is stored in the interpreter before
// returning.
//
// A return value of TCL_OK means there is a row of data available. The data may be accessed using dbEvalRowInfo() and dbEvalColumnValue(). This
// is analogous to a return of SQLITE_ROW from sqlite3_step(). If TCL_BREAK is returned, then the SQL script has finished executing and there are
// no further rows available. This is similar to SQLITE_DONE.
__device__ static RC DbEvalStep(DbEvalContext *p)
{
	const char *prevSql = nullptr; // Previous value of p->zSql
	while (p->SqlAsString[0] || p->PreStmt)
	{
		RC rc;
		if (!p->PreStmt)
		{
			prevSql = (p->SqlAsString == prevSql ? nullptr : p->SqlAsString);
			rc = DbPrepareAndBind(p->Ctx, p->SqlAsString, &p->SqlAsString, &p->PreStmt);
			if (rc != RC_OK) return rc;
		}
		else
		{
			TclContext *tctx = p->Ctx;
			SqlPreparedStmt *preStmt = p->PreStmt;
			Vdbe *stmt = preStmt->Stmt;

			rc = stmt->Step();
			if (rc == RC_ROW)
				return RC_OK;
			if (p->Array)
				DbEvalRowInfo(p, 0, 0);
			rc = Vdbe::Reset(stmt);

			tctx->Steps = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_FULLSCAN_STEP, true);
			tctx->Sorts = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_SORT, true);
			tctx->Indexs = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_AUTOINDEX, true);
			DbReleaseColumnNames(p);
			p->PreStmt = nullptr;

			if (rc != RC_OK)
			{
				// If a run-time error occurs, report the error and stop reading the SQL.
				DbReleaseStmt(tctx, preStmt, true);
#if _TEST
				if (p->Ctx->LegacyPrepare && rc == RC_SCHEMA && prevSql)
				{
					// If the runtime error was an SQLITE_SCHEMA, and the database handle is configured to use the legacy sqlite3_prepare() 
					// interface, retry prepare()/step() on the same SQL statement. This only happens once. If there is a second SQLITE_SCHEMA
					// error, the error will be returned to the caller.
					p->SqlAsString = prevSql;
					continue;
				}
#endif
				tctx->Interp->SetObjResult(Tcl_Obj::NewStringObj(Main::ErrMsg(tctx->Ctx), -1));
				return RC_ERROR;
			}
			else
				DbReleaseStmt(tctx, preStmt, false);
		}
	}

	// Finished
	return RC_DONE;
}

// Free all resources currently held by the DbEvalContext structure passed as the first argument. There should be exactly one call to this function
// for each call to dbEvalInit().
__device__ static void DbEvalFinalize(DbEvalContext *p)
{
	if (p->PreStmt)
	{
		Vdbe::Reset(p->PreStmt->Stmt);
		DbReleaseStmt(p->Ctx, p->PreStmt, false);
		p->PreStmt = nullptr;
	}
	if (p->Array)
	{
		p->Array->DecrRefCount();
		p->Array = nullptr;
	}
	p->Sql->DecrRefCount();
	DbReleaseColumnNames(p);
}

// Return a pointer to a Tcl_Obj structure with ref-count 0 that contains the value for the iCol'th column of the row currently pointed to by
// the DbEvalContext structure passed as the first argument.
__device__ static Tcl_Obj *DbEvalColumnValue(DbEvalContext *p, int colId)
{
	Vdbe *stmt = p->PreStmt->Stmt;
	switch (Vdbe::Column_Type(stmt, colId))
	{
	case TYPE_BLOB: {
		int bytes = Vdbe::Column_Bytes(stmt, colId);
		const void *blob = Vdbe::Column_Blob(stmt, colId);
		if (!blob) bytes = 0;
		return Tcl_Obj::NewByteArrayObj((uint8 *)blob, bytes); }
	case TYPE_INTEGER: {
		int64 v = Vdbe::Column_Int64(stmt, colId);
		return (v >= -2147483647 && v <= 2147483647 ? Tcl_Obj::NewIntObj((int)v) : Tcl_Obj::NewWideIntObj(v)); }
	case TYPE_FLOAT: {
		return Tcl_Obj::NewDoubleObj(Vdbe::Column_Double(stmt, colId)); }
	case TYPE_NULL: {
		return Tcl_Obj::NewStringObj(p->Ctx->NullText, -1); }
	}
	return Tcl_Obj::NewStringObj((char *)Vdbe::Column_Text(stmt, colId), -1);
}

// This function is part of the implementation of the command:
//
//   $db eval SQL ?ARRAYNAME? SCRIPT
__device__ static RC DbEvalNextCmd(ClientData data[], Tcl_Interp *interp, RC result)
{
	RC rc = result;

	// The first element of the data[] array is a pointer to a DbEvalContext structure allocated using Tcl_Alloc(). The second element of data[]
	// is a pointer to a Tcl_Obj containing the script to run for each row returned by the queries encapsulated in data[0].
	DbEvalContext *p = (DbEvalContext *)data[0];
	Tcl_Obj *script = (Tcl_Obj *)data[1];
	Tcl_Obj *array = p->Array;

	while ((rc == RC_OK || rc == RC_ROW) && (rc = DbEvalStep(p)) == RC_OK)
	{
		int cols;
		Tcl_Obj **colNames;
		DbEvalRowInfo(p, &cols, &colNames);
		for (int i =0 ; i < cols; i++)
		{
			Tcl_Obj *val = DbEvalColumnValue(p, i);
			if (!array)
				colNames[i]->ObjSetVar2(interp, nullptr, val, false);
			else
				array->ObjSetVar2(interp, colNames[i], val, false);
		}

		rc = interp->EvalObjEx(script, false);
	}

	script->DecrRefCount();
	DbEvalFinalize(p);
	_free(p);

	if (rc == RC_OK || rc == RC_DONE)
	{
		interp->ResetResult();
		rc = RC_OK;
	}
	return rc;
}

#pragma endregion

#pragma region DbObjCmd

//    $db authorizer ?CALLBACK?
//
// Invoke the given callback to authorize each SQL operation as it is compiled.  5 arguments are appended to the callback before it is invoked:
//
//   (1) The authorization type (ex: SQLITE_CREATE_TABLE, SQLITE_INSERT, ...)
//   (2) First descriptive name (depends on authorization type)
//   (3) Second descriptive name
//   (4) Name of the database (ex: "main", "temp")
//   (5) Name of trigger that is doing the access
//
// The callback should return on of the following strings: SQLITE_OK, SQLITE_IGNORE, or SQLITE_DENY.  Any other return value is an error.
//
// If this method is invoked with no arguments, the current authorization callback string is returned.
__device__ RC TclContext::AUTHORIZER(array_t<Tcl_Obj *> objv)
{
#ifdef OMIT_AUTHORIZATION
	interp->AppendResult("authorization not available in this build", 0);
	return RC_ERROR;
#else
	if (objv.length > 3)
	{
		Interp->WrongNumArgs(2, objv, "?CALLBACK?");
		return RC_ERROR;
	}
	else if (objv.length == 2)
	{
		if (Auth)
			Interp->AppendResult(Auth, nullptr);
	}
	else
	{
		if (Auth)
			_free(Auth);
		int len;
		char *auth = objv[2]->GetStringFromObj(&len);
		if (auth && len > 0)
		{
			Auth = (char *)_alloc(len + 1);
			_memcpy(Auth, auth, len + 1);
		}
		else
			Auth = nullptr;
		if (Auth)
			Auth::SetAuthorizer(Ctx, AuthCallback, this);
		else
			Auth::SetAuthorizer(Ctx, nullptr, nullptr);
	}
#endif
	return RC_OK;
}

//    $db backup ?DATABASE? FILENAME
//
// Open or create a database file named FILENAME.  Transfer the content of local database DATABASE (default: "main") into the FILENAME database.
__device__ RC TclContext::BACKUP(array_t<Tcl_Obj *> objv)
{
	const char *srcDb;
	const char *destFile;
	if (objv.length == 3)
	{
		srcDb = "main";
		destFile = objv[2]->GetString();
	}
	else if (objv.length == 4)
	{
		srcDb = objv[2]->GetString();
		destFile = objv[3]->GetString();
	}
	else
	{
		Interp->WrongNumArgs(2, objv, "?DATABASE? FILENAME");
		return RC_ERROR;
	}

	Context *destCtx;
	::RC rc = Main::Open(destFile, &destCtx);
	if (rc != RC_OK)
	{
		Interp->AppendResult("cannot open target database: ", Main::ErrMsg(destCtx), (char *)nullptr);
		Main::Close(destCtx);
		return RC_ERROR;
	}
	Backup *backup = Backup::Init(destCtx, "main", Ctx, srcDb);
	if (!backup)
	{
		Interp->AppendResult("backup failed: ", Main::ErrMsg(destCtx), (char *)nullptr);
		Main::Close(destCtx);
		return RC_ERROR;
	}
	while ((rc = backup->Step(100)) == RC_OK ) { }
	Backup::Finish(backup);
	if (rc == RC_DONE)
		rc = RC_OK;
	else
	{
		Interp->AppendResult("backup failed: ", Main::ErrMsg(destCtx), (char *)nullptr);
		rc = RC_ERROR;
	}
	Main::Close(destCtx);
	return RC_OK;
}

//    $db busy ?CALLBACK?
//
// Invoke the given callback if an SQL statement attempts to open a locked database file.
__device__ RC TclContext::BUSY(array_t<Tcl_Obj *> objv)
{
	if (objv.length > 3)
	{
		Interp->WrongNumArgs(2, objv, "CALLBACK");
		return RC_ERROR;
	}
	else if (objv.length == 2)
	{
		if (Busy)
			Interp->AppendResult(Busy, 0);
	}
	else
	{
		int len;
		if (Busy)
			_free(Busy);
		char *busy = objv[2]->GetStringFromObj(&len);
		if (busy && len > 0)
		{
			Busy = (char *)_alloc(len + 1);
			_memcpy(Busy, busy, len + 1);
		}
		else
			Busy = nullptr;
		if (Busy)
			Main::BusyHandler(Ctx, DbBusyHandler, this);
		else
			Main::BusyHandler(Ctx, nullptr, nullptr);
	}
	return RC_OK;
}

//     $db cache flush
//     $db cache size n
//
// Flush the prepared statement cache, or set the maximum number of cached statements.
__device__ RC TclContext::CACHE(array_t<Tcl_Obj *> objv)
{
	if (objv.length <= 2)
	{
		Interp->WrongNumArgs(1, objv, "cache option ?arg?");
		return RC_ERROR;
	}
	char *subCmd = objv[2]->GetStringFromObj(nullptr);
	if (subCmd[0] == 'f' && !_strcmp(subCmd,"flush"))
	{
		if (objv.length != 3)
		{
			Interp->WrongNumArgs(2, objv, "flush");
			return RC_ERROR;
		}
		else
			FlushStmtCache(this);
	}
	else if (subCmd[0] == 's' && !_strcmp(subCmd, "size"))
	{
		if (objv.length != 4)
		{
			Interp->WrongNumArgs(2, objv, "size n");
			return RC_ERROR;
		}
		else
		{
			int n;
			if (objv[3]->GetIntFromObj(Interp, &n) == RC_ERROR)
			{
				Interp->AppendResult("cannot convert \"", objv[3]->GetStringFromObj(nullptr), "\" to integer", nullptr);
				return RC_ERROR;
			}
			else
			{
				if (n < 0)
				{
					FlushStmtCache(this);
					n = 0;
				}
				else if (n > MAX_PREPARED_STMTS)
					n = MAX_PREPARED_STMTS;
				MaxStmt = n;
			}
		}
	}
	else
	{
		Interp->AppendResult("bad option \"", objv[2]->GetStringFromObj(nullptr), "\": must be flush or size", nullptr);
		return RC_ERROR;
	}
	return RC_OK;
}

//     $db changes
//
// Return the number of rows that were modified, inserted, or deleted by the most recent INSERT, UPDATE or DELETE statement, not including 
// any changes made by trigger programs.
__device__ RC TclContext::CHANGES(array_t<Tcl_Obj *> objv)
{
	if (objv.length != 2){
		Interp->WrongNumArgs(2, objv, "");
		return RC_ERROR;
	}
	Tcl_Obj *result = Interp->GetObjResult();
	result->SetIntObj(Main::CtxChanges(Ctx));
	return RC_OK;
}

//    $db close
//
// Shutdown the database
__device__ RC TclContext::CLOSE(array_t<Tcl_Obj *> objv)
{
	Interp->DeleteCommand(objv[0]->GetStringFromObj(nullptr));
	return RC_OK;
}

#if 0
//     $db collate NAME SCRIPT
//
// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
__device__ RC DB_COLLATE()
{
	SqlCollate *pCollate;
	char *zName;
	char *zScript;
	int nScript;
	if( objc!=4 ){
		Tcl_WrongNumArgs(interp, 2, objv, "NAME SCRIPT");
		return TCL_ERROR;
	}
	zName = Tcl_GetStringFromObj(objv[2], 0);
	zScript = Tcl_GetStringFromObj(objv[3], &nScript);
	pCollate = (SqlCollate*)Tcl_Alloc( sizeof(*pCollate) + nScript + 1 );
	if( pCollate==0 ) return TCL_ERROR;
	pCollate->interp = interp;
	pCollate->pNext = pDb->pCollate;
	pCollate->zScript = (char*)&pCollate[1];
	pDb->pCollate = pCollate;
	memcpy(pCollate->zScript, zScript, nScript+1);
	if( sqlite3_create_collation(pDb->db, zName, SQLITE_UTF8, 
		pCollate, tclSqlCollate) ){
			Tcl_SetResult(interp, (char *)sqlite3_errmsg(pDb->db), TCL_VOLATILE);
			return TCL_ERROR;
	}
}

//     $db collation_needed SCRIPT
//
// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
__device__ RC DB_COLLATION_NEEDED()
{
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "SCRIPT");
		return TCL_ERROR;
	}
	if( pDb->pCollateNeeded ){
		Tcl_DecrRefCount(pDb->pCollateNeeded);
	}
	pDb->pCollateNeeded = Tcl_DuplicateObj(objv[2]);
	Tcl_IncrRefCount(pDb->pCollateNeeded);
	sqlite3_collation_needed(pDb->db, pDb, tclCollateNeeded);
}

//    $db commit_hook ?CALLBACK?
//
// Invoke the given callback just before committing every SQL transaction. If the callback throws an exception or returns non-zero, then the
// transaction is aborted.  If CALLBACK is an empty string, the callback is disabled.
__device__ RC DB_COMMIT_HOOK()
{
	if( objc>3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "?CALLBACK?");
		return TCL_ERROR;
	}else if( objc==2 ){
		if( pDb->zCommit ){
			Tcl_AppendResult(interp, pDb->zCommit, 0);
		}
	}else{
		char *zCommit;
		int len;
		if( pDb->zCommit ){
			Tcl_Free(pDb->zCommit);
		}
		zCommit = Tcl_GetStringFromObj(objv[2], &len);
		if( zCommit && len>0 ){
			pDb->zCommit = Tcl_Alloc( len + 1 );
			memcpy(pDb->zCommit, zCommit, len+1);
		}else{
			pDb->zCommit = 0;
		}
		if( pDb->zCommit ){
			pDb->interp = interp;
			sqlite3_commit_hook(pDb->db, DbCommitHandler, pDb);
		}else{
			sqlite3_commit_hook(pDb->db, 0, 0);
		}
	}
}

//    $db complete SQL
//
// Return TRUE if SQL is a complete SQL statement.  Return FALSE if additional lines of input are needed.  This is similar to the
// built-in "info complete" command of Tcl.
__device__ RC DB_COMPLETE()
{
#ifndef OMIT_COMPLETE
	Tcl_Obj *pResult;
	int isComplete;
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "SQL");
		return TCL_ERROR;
	}
	isComplete = sqlite3_complete( Tcl_GetStringFromObj(objv[2], 0) );
	pResult = Tcl_GetObjResult(interp);
	Tcl_SetBooleanObj(pResult, isComplete);
#endif
}

//    $db copy conflict-algorithm table filename ?SEPARATOR? ?NULLINDICATOR?
//
// Copy data into table from filename, optionally using SEPARATOR as column separators.  If a column contains a null string, or the
// value of NULLINDICATOR, a NULL is inserted for the column. conflict-algorithm is one of the sqlite conflict algorithms:
//    rollback, abort, fail, ignore, replace
// On success, return the number of lines processed, not necessarily same as 'db changes' due to conflict-algorithm selected.
//
// This code is basically an implementation/enhancement of the sqlite3 shell.c ".import" command.
//
// This command usage is equivalent to the sqlite2.x COPY statement, which imports file data into a table using the PostgreSQL COPY file format:
//   $db copy $conflit_algo $table_name $filename \t \\N
__device__ RC DB_COPY()
{
	char *zTable;               /* Insert data into this table */
	char *zFile;                /* The file from which to extract data */
	char *zConflict;            /* The conflict algorithm to use */
	sqlite3_stmt *pStmt;        /* A statement */
	int nCol;                   /* Number of columns in the table */
	int nByte;                  /* Number of bytes in an SQL string */
	int i, j;                   /* Loop counters */
	int nSep;                   /* Number of bytes in zSep[] */
	int nNull;                  /* Number of bytes in zNull[] */
	char *zSql;                 /* An SQL statement */
	char *zLine;                /* A single line of input from the file */
	char **azCol;               /* zLine[] broken up into columns */
	char *zCommit;              /* How to commit changes */
	FILE *in;                   /* The input file */
	int lineno = 0;             /* Line number of input file */
	char zLineNum[80];          /* Line number print buffer */
	Tcl_Obj *pResult;           /* interp result */

	char *zSep;
	char *zNull;
	if( objc<5 || objc>7 ){
		Tcl_WrongNumArgs(interp, 2, objv, 
			"CONFLICT-ALGORITHM TABLE FILENAME ?SEPARATOR? ?NULLINDICATOR?");
		return TCL_ERROR;
	}
	if( objc>=6 ){
		zSep = Tcl_GetStringFromObj(objv[5], 0);
	}else{
		zSep = "\t";
	}
	if( objc>=7 ){
		zNull = Tcl_GetStringFromObj(objv[6], 0);
	}else{
		zNull = "";
	}
	zConflict = Tcl_GetStringFromObj(objv[2], 0);
	zTable = Tcl_GetStringFromObj(objv[3], 0);
	zFile = Tcl_GetStringFromObj(objv[4], 0);
	nSep = strlen30(zSep);
	nNull = strlen30(zNull);
	if( nSep==0 ){
		Tcl_AppendResult(interp,"Error: non-null separator required for copy",0);
		return TCL_ERROR;
	}
	if(strcmp(zConflict, "rollback") != 0 &&
		strcmp(zConflict, "abort"   ) != 0 &&
		strcmp(zConflict, "fail"    ) != 0 &&
		strcmp(zConflict, "ignore"  ) != 0 &&
		strcmp(zConflict, "replace" ) != 0 ) {
			Tcl_AppendResult(interp, "Error: \"", zConflict, 
				"\", conflict-algorithm must be one of: rollback, "
				"abort, fail, ignore, or replace", 0);
			return TCL_ERROR;
	}
	zSql = sqlite3_mprintf("SELECT * FROM '%q'", zTable);
	if( zSql==0 ){
		Tcl_AppendResult(interp, "Error: no such table: ", zTable, 0);
		return TCL_ERROR;
	}
	nByte = strlen30(zSql);
	rc = sqlite3_prepare(pDb->db, zSql, -1, &pStmt, 0);
	sqlite3_free(zSql);
	if( rc ){
		Tcl_AppendResult(interp, "Error: ", sqlite3_errmsg(pDb->db), 0);
		nCol = 0;
	}else{
		nCol = sqlite3_column_count(pStmt);
	}
	sqlite3_finalize(pStmt);
	if( nCol==0 ) {
		return TCL_ERROR;
	}
	zSql = malloc( nByte + 50 + nCol*2 );
	if( zSql==0 ) {
		Tcl_AppendResult(interp, "Error: can't malloc()", 0);
		return TCL_ERROR;
	}
	sqlite3_snprintf(nByte+50, zSql, "INSERT OR %q INTO '%q' VALUES(?",
		zConflict, zTable);
	j = strlen30(zSql);
	for(i=1; i<nCol; i++){
		zSql[j++] = ',';
		zSql[j++] = '?';
	}
	zSql[j++] = ')';
	zSql[j] = 0;
	rc = sqlite3_prepare(pDb->db, zSql, -1, &pStmt, 0);
	free(zSql);
	if( rc ){
		Tcl_AppendResult(interp, "Error: ", sqlite3_errmsg(pDb->db), 0);
		sqlite3_finalize(pStmt);
		return TCL_ERROR;
	}
	in = fopen(zFile, "rb");
	if( in==0 ){
		Tcl_AppendResult(interp, "Error: cannot open file: ", zFile, NULL);
		sqlite3_finalize(pStmt);
		return TCL_ERROR;
	}
	azCol = malloc( sizeof(azCol[0])*(nCol+1) );
	if( azCol==0 ) {
		Tcl_AppendResult(interp, "Error: can't malloc()", 0);
		fclose(in);
		return TCL_ERROR;
	}
	(void)sqlite3_exec(pDb->db, "BEGIN", 0, 0, 0);
	zCommit = "COMMIT";
	while( (zLine = local_getline(0, in))!=0 ){
		char *z;
		lineno++;
		azCol[0] = zLine;
		for(i=0, z=zLine; *z; z++){
			if( *z==zSep[0] && strncmp(z, zSep, nSep)==0 ){
				*z = 0;
				i++;
				if( i<nCol ){
					azCol[i] = &z[nSep];
					z += nSep-1;
				}
			}
		}
		if( i+1!=nCol ){
			char *zErr;
			int nErr = strlen30(zFile) + 200;
			zErr = malloc(nErr);
			if( zErr ){
				sqlite3_snprintf(nErr, zErr,
					"Error: %s line %d: expected %d columns of data but found %d",
					zFile, lineno, nCol, i+1);
				Tcl_AppendResult(interp, zErr, 0);
				free(zErr);
			}
			zCommit = "ROLLBACK";
			break;
		}
		for(i=0; i<nCol; i++){
			/* check for null data, if so, bind as null */
			if( (nNull>0 && strcmp(azCol[i], zNull)==0)
				|| strlen30(azCol[i])==0 
				){
					sqlite3_bind_null(pStmt, i+1);
			}else{
				sqlite3_bind_text(pStmt, i+1, azCol[i], -1, SQLITE_STATIC);
			}
		}
		sqlite3_step(pStmt);
		rc = sqlite3_reset(pStmt);
		free(zLine);
		if( rc!=SQLITE_OK ){
			Tcl_AppendResult(interp,"Error: ", sqlite3_errmsg(pDb->db), 0);
			zCommit = "ROLLBACK";
			break;
		}
	}
	free(azCol);
	fclose(in);
	sqlite3_finalize(pStmt);
	(void)sqlite3_exec(pDb->db, zCommit, 0, 0, 0);

	if( zCommit[0] == 'C' ){
		/* success, set result as number of lines processed */
		pResult = Tcl_GetObjResult(interp);
		Tcl_SetIntObj(pResult, lineno);
		rc = TCL_OK;
	}else{
		/* failure, append lineno where failed */
		sqlite3_snprintf(sizeof(zLineNum), zLineNum,"%d",lineno);
		Tcl_AppendResult(interp,", failed while processing line: ",zLineNum,0);
		rc = TCL_ERROR;
	}
}

//    $db enable_load_extension BOOLEAN
//
// Turn the extension loading feature on or off.  It if off by default.
__device__ RC DB_ENABLE_LOAD_EXTENSION()
{
#ifndef OMIT_LOAD_EXTENSION
	int onoff;
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "BOOLEAN");
		return TCL_ERROR;
	}
	if( Tcl_GetBooleanFromObj(interp, objv[2], &onoff) ){
		return TCL_ERROR;
	}
	sqlite3_enable_load_extension(pDb->db, onoff);
#else
	interp-AppendResult("extension loading is turned off at compile-time", 0);
	return RC_ERROR;
#endif
}

//    $db errorcode
//
// Return the numeric error code that was returned by the most recent call to sqlite3_exec().
__device__ RC DB_ERRORCODE()
{
	interp->SetObjResult(Tcl_Obj::NewIntObj(Main::ErrCode(tctx->Ctx)));
}

//    $db exists $sql
//    $db onecolumn $sql
//
// The onecolumn method is the equivalent of:
//     lindex [$db eval $sql] 0
__device__ RC DB_EXISTS() { return DB_ONECOLUMN(); }
__device__ RC DB_ONECOLUMN()
{
	DbEvalContext sEval;
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "SQL");
		return TCL_ERROR;
	}

	dbEvalInit(&sEval, pDb, objv[2], 0);
	rc = dbEvalStep(&sEval);
	if( choice==DB_ONECOLUMN ){
		if( rc==TCL_OK ){
			Tcl_SetObjResult(interp, dbEvalColumnValue(&sEval, 0));
		}else if( rc==TCL_BREAK ){
			Tcl_ResetResult(interp);
		}
	}else if( rc==TCL_BREAK || rc==TCL_OK ){
		Tcl_SetObjResult(interp, Tcl_NewBooleanObj(rc==TCL_OK));
	}
	dbEvalFinalize(&sEval);

	if( rc==TCL_BREAK ){
		rc = TCL_OK;
	}
}

#endif

//    $db eval $sql ?array? ?{  ...code... }?
//
// The SQL statement in $sql is evaluated.  For each row, the values are placed in elements of the array named "array" and ...code... is executed.
// If "array" and "code" are omitted, then no callback is every invoked. If "array" is an empty string, then the values are placed in variables
// that have the same name as the fields extracted by the query.
__device__ RC TclContext::EVAL(array_t<Tcl_Obj *> objv)
{
	if (objv.length < 3 || objv.length > 5)
	{
		Interp->WrongNumArgs(2, objv, "SQL ?ARRAY-NAME? ?SCRIPT?");
		return RC_ERROR;
	}

	::RC rc;
	if (objv.length == 3)
	{
		DbEvalContext sEval;
		Tcl_Obj *ret = Tcl_Obj::NewObj();
		ret->IncrRefCount();
		DbEvalInit(&sEval, this, objv[2], nullptr);

		while ((rc = DbEvalStep(&sEval)) == RC_OK)
		{
			int cols;
			DbEvalRowInfo(&sEval, &cols, nullptr);
			for (int i = 0; i < cols; i++)
				ret->ListObjAppendElement(Interp, DbEvalColumnValue(&sEval, i));
		}
		DbEvalFinalize(&sEval);
		if (rc == RC_DONE)
		{
			Interp->SetObjResult(ret);
			rc = RC_OK;
		}
		ret->DecrRefCount();
	}
	else
	{
		ClientData cd[2];
		Tcl_Obj *array = (objv.length == 5 && *(char *)objv[3]->GetString() ? objv[3] : nullptr);
		Tcl_Obj *script = objv[objv.length-1];
		script->IncrRefCount();

		DbEvalContext *p = (DbEvalContext *)_alloc(sizeof(DbEvalContext));
		DbEvalInit(p, this, objv[2], array);

		cd[0] = (void *)p;
		cd[1] = (void *)script;
		rc = DbEvalNextCmd(cd, Interp, RC_OK);
	}
	return RC_OK;
}

#if 0

//     $db function NAME [-argcount N] SCRIPT
//
// Create a new SQL function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
__device__ RC DB_FUNCTION()
{
	SqlFunc *pFunc;
	Tcl_Obj *pScript;
	char *zName;
	int nArg = -1;
	if( objc==6 ){
		const char *z = Tcl_GetString(objv[3]);
		int n = strlen30(z);
		if( n>2 && strncmp(z, "-argcount",n)==0 ){
			if( Tcl_GetIntFromObj(interp, objv[4], &nArg) ) return TCL_ERROR;
			if( nArg<0 ){
				Tcl_AppendResult(interp, "number of arguments must be non-negative",
					(char*)0);
				return TCL_ERROR;
			}
		}
		pScript = objv[5];
	}else if( objc!=4 ){
		Tcl_WrongNumArgs(interp, 2, objv, "NAME [-argcount N] SCRIPT");
		return TCL_ERROR;
	}else{
		pScript = objv[3];
	}
	zName = Tcl_GetStringFromObj(objv[2], 0);
	pFunc = findSqlFunc(pDb, zName);
	if( pFunc==0 ) return TCL_ERROR;
	if( pFunc->pScript ){
		Tcl_DecrRefCount(pFunc->pScript);
	}
	pFunc->pScript = pScript;
	Tcl_IncrRefCount(pScript);
	pFunc->useEvalObjv = safeToUseEvalObjv(interp, pScript);
	rc = sqlite3_create_function(pDb->db, zName, nArg, SQLITE_UTF8,
		pFunc, tclSqlFunc, 0, 0);
	if( rc!=SQLITE_OK ){
		rc = TCL_ERROR;
		Tcl_SetResult(interp, (char *)sqlite3_errmsg(pDb->db), TCL_VOLATILE);
	}
}

//     $db incrblob ?-readonly? ?DB? TABLE COLUMN ROWID
__device__ RC DB_INCRBLOB()
{
#ifdef OMIT_INCRBLOB
	interp->AppendResult("incrblob not available in this build", 0);
	return RC_ERROR;
#else
	int isReadonly = 0;
	const char *zDb = "main";
	const char *zTable;
	const char *zColumn;
	Tcl_WideInt iRow;

	/* Check for the -readonly option */
	if( objc>3 && strcmp(Tcl_GetString(objv[2]), "-readonly")==0 ){
		isReadonly = 1;
	}

	if( objc!=(5+isReadonly) && objc!=(6+isReadonly) ){
		Tcl_WrongNumArgs(interp, 2, objv, "?-readonly? ?DB? TABLE COLUMN ROWID");
		return TCL_ERROR;
	}

	if( objc==(6+isReadonly) ){
		zDb = Tcl_GetString(objv[2]);
	}
	zTable = Tcl_GetString(objv[objc-3]);
	zColumn = Tcl_GetString(objv[objc-2]);
	rc = Tcl_GetWideIntFromObj(interp, objv[objc-1], &iRow);

	if( rc==TCL_OK ){
		rc = createIncrblobChannel(
			interp, pDb, zDb, zTable, zColumn, iRow, isReadonly
			);
	}
#endif
}

//     $db interrupt
//
// Interrupt the execution of the inner-most SQL interpreter.  This causes the SQL statement to return an error of SQLITE_INTERRUPT.
__device__ RC DB_INTERRUPT()
{
	Main::Interrupt(tctx->Ctx);
}

//     $db nullvalue ?STRING?
//
// Change text used when a NULL comes back from the database. If ?STRING? is not present, then the current string used for NULL is returned.
// If STRING is present, then STRING is returned.
__device__ RC DB_NULLVALUE()
{
	if( objc!=2 && objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "NULLVALUE");
		return TCL_ERROR;
	}
	if( objc==3 ){
		int len;
		char *zNull = Tcl_GetStringFromObj(objv[2], &len);
		if( pDb->zNull ){
			Tcl_Free(pDb->zNull);
		}
		if( zNull && len>0 ){
			pDb->zNull = Tcl_Alloc( len + 1 );
			memcpy(pDb->zNull, zNull, len);
			pDb->zNull[len] = '\0';
		}else{
			pDb->zNull = 0;
		}
	}
	Tcl_SetObjResult(interp, Tcl_NewStringObj(pDb->zNull, -1));
}

//     $db last_insert_rowid 
//
// Return an integer which is the ROWID for the most recent insert.
__device__ RC DB_LAST_INSERT_ROWID()
{
	Tcl_Obj *pResult;
	Tcl_WideInt rowid;
	if( objc!=2 ){
		Tcl_WrongNumArgs(interp, 2, objv, "");
		return TCL_ERROR;
	}
	rowid = sqlite3_last_insert_rowid(pDb->db);
	pResult = Tcl_GetObjResult(interp);
	Tcl_SetWideIntObj(pResult, rowid);
}

// The DB_ONECOLUMN method is implemented together with DB_EXISTS.

//    $db progress ?N CALLBACK?
// 
// Invoke the given callback every N virtual machine opcodes while executing queries.
__device__ RC DB_PROGRESS()
{
	if( objc==2 ){
		if( pDb->zProgress ){
			Tcl_AppendResult(interp, pDb->zProgress, 0);
		}
	}else if( objc==4 ){
		char *zProgress;
		int len;
		int N;
		if( TCL_OK!=Tcl_GetIntFromObj(interp, objv[2], &N) ){
			return TCL_ERROR;
		};
		if( pDb->zProgress ){
			Tcl_Free(pDb->zProgress);
		}
		zProgress = Tcl_GetStringFromObj(objv[3], &len);
		if( zProgress && len>0 ){
			pDb->zProgress = Tcl_Alloc( len + 1 );
			memcpy(pDb->zProgress, zProgress, len+1);
		}else{
			pDb->zProgress = 0;
		}
#ifndef OMIT_PROGRESS_CALLBACK
		if( pDb->zProgress ){
			pDb->interp = interp;
			sqlite3_progress_handler(pDb->db, N, DbProgressHandler, pDb);
		}else{
			sqlite3_progress_handler(pDb->db, 0, 0, 0);
		}
#endif
	}else{
		Tcl_WrongNumArgs(interp, 2, objv, "N CALLBACK");
		return TCL_ERROR;
	}
}

//    $db profile ?CALLBACK?
//
// Make arrangements to invoke the CALLBACK routine after each SQL statement that has run.  The text of the SQL and the amount of elapse time are
// appended to CALLBACK before the script is run.
__device__ RC DB_PROFILE()
{
	if( objc>3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "?CALLBACK?");
		return TCL_ERROR;
	}else if( objc==2 ){
		if( pDb->zProfile ){
			Tcl_AppendResult(interp, pDb->zProfile, 0);
		}
	}else{
		char *zProfile;
		int len;
		if( pDb->zProfile ){
			Tcl_Free(pDb->zProfile);
		}
		zProfile = Tcl_GetStringFromObj(objv[2], &len);
		if( zProfile && len>0 ){
			pDb->zProfile = Tcl_Alloc( len + 1 );
			memcpy(pDb->zProfile, zProfile, len+1);
		}else{
			pDb->zProfile = 0;
		}
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
		if( pDb->zProfile ){
			pDb->interp = interp;
			sqlite3_profile(pDb->db, DbProfileHandler, pDb);
		}else{
			sqlite3_profile(pDb->db, 0, 0);
		}
#endif
	}
}

//     $db rekey KEY
//
// Change the encryption key on the currently open database.
__device__ RC DB_REKEY()
{
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "KEY");
		return TCL_ERROR;
	}
#ifdef HAS_CODEC
	int nKey;
	void *pKey;
	pKey = Tcl_GetByteArrayFromObj(objv[2], &nKey);
	rc = sqlite3_rekey(pDb->db, pKey, nKey);
	if( rc ){
		Tcl_AppendResult(interp, sqlite3_errstr(rc), 0);
		rc = TCL_ERROR;
	}
#endif
}

//    $db restore ?DATABASE? FILENAME
//
// Open a database file named FILENAME.  Transfer the content  of FILENAME into the local database DATABASE (default: "main").
__device__ RC DB_RESTORE()
{
	const char *zSrcFile;
	const char *zDestDb;
	sqlite3 *pSrc;
	sqlite3_backup *pBackup;
	int nTimeout = 0;

	if( objc==3 ){
		zDestDb = "main";
		zSrcFile = Tcl_GetString(objv[2]);
	}else if( objc==4 ){
		zDestDb = Tcl_GetString(objv[2]);
		zSrcFile = Tcl_GetString(objv[3]);
	}else{
		Tcl_WrongNumArgs(interp, 2, objv, "?DATABASE? FILENAME");
		return TCL_ERROR;
	}
	rc = sqlite3_open_v2(zSrcFile, &pSrc, SQLITE_OPEN_READONLY, 0);
	if( rc!=SQLITE_OK ){
		Tcl_AppendResult(interp, "cannot open source database: ",
			sqlite3_errmsg(pSrc), (char*)0);
		sqlite3_close(pSrc);
		return TCL_ERROR;
	}
	pBackup = sqlite3_backup_init(pDb->db, zDestDb, pSrc, "main");
	if( pBackup==0 ){
		Tcl_AppendResult(interp, "restore failed: ",
			sqlite3_errmsg(pDb->db), (char*)0);
		sqlite3_close(pSrc);
		return TCL_ERROR;
	}
	while( (rc = sqlite3_backup_step(pBackup,100))==SQLITE_OK
		|| rc==SQLITE_BUSY ){
			if( rc==SQLITE_BUSY ){
				if( nTimeout++ >= 3 ) break;
				sqlite3_sleep(100);
			}
	}
	sqlite3_backup_finish(pBackup);
	if( rc==SQLITE_DONE ){
		rc = TCL_OK;
	}else if( rc==SQLITE_BUSY || rc==SQLITE_LOCKED ){
		Tcl_AppendResult(interp, "restore failed: source database busy",
			(char*)0);
		rc = TCL_ERROR;
	}else{
		Tcl_AppendResult(interp, "restore failed: ",
			sqlite3_errmsg(pDb->db), (char*)0);
		rc = TCL_ERROR;
	}
	sqlite3_close(pSrc);
}

//     $db status (step|sort|autoindex)
//
// Display SQLITE_STMTSTATUS_FULLSCAN_STEP or SQLITE_STMTSTATUS_SORT for the most recent eval.
__device__ RC DB_STATUS()
{
	int v;
	const char *zOp;
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "(step|sort|autoindex)");
		return TCL_ERROR;
	}
	zOp = Tcl_GetString(objv[2]);
	if( strcmp(zOp, "step")==0 ){
		v = pDb->nStep;
	}else if( strcmp(zOp, "sort")==0 ){
		v = pDb->nSort;
	}else if( strcmp(zOp, "autoindex")==0 ){
		v = pDb->nIndex;
	}else{
		Tcl_AppendResult(interp, 
			"bad argument: should be autoindex, step, or sort", 
			(char*)0);
		return TCL_ERROR;
	}
	Tcl_SetObjResult(interp, Tcl_NewIntObj(v));
}

//     $db timeout MILLESECONDS
//
// Delay for the number of milliseconds specified when a file is locked.
__device__ RC DB_TIMEOUT()
{
	int ms;
	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "MILLISECONDS");
		return TCL_ERROR;
	}
	if( Tcl_GetIntFromObj(interp, objv[2], &ms) ) return TCL_ERROR;
	sqlite3_busy_timeout(pDb->db, ms);
}

//     $db total_changes
//
// Return the number of rows that were modified, inserted, or deleted since the database handle was created.
__device__ RC DB_TOTAL_CHANGES()
{
	Tcl_Obj *pResult;
	if( objc!=2 ){
		Tcl_WrongNumArgs(interp, 2, objv, "");
		return TCL_ERROR;
	}
	pResult = Tcl_GetObjResult(interp);
	Tcl_SetIntObj(pResult, sqlite3_total_changes(pDb->db));
}

//    $db trace ?CALLBACK?
//
// Make arrangements to invoke the CALLBACK routine for each SQL statement that is executed.  The text of the SQL is appended to CALLBACK before
// it is executed.
__device__ RC DB_TRACE()
{
	if( objc>3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "?CALLBACK?");
		return TCL_ERROR;
	}else if( objc==2 ){
		if( pDb->zTrace ){
			Tcl_AppendResult(interp, pDb->zTrace, 0);
		}
	}else{
		char *zTrace;
		int len;
		if( pDb->zTrace ){
			Tcl_Free(pDb->zTrace);
		}
		zTrace = Tcl_GetStringFromObj(objv[2], &len);
		if( zTrace && len>0 ){
			pDb->zTrace = Tcl_Alloc( len + 1 );
			memcpy(pDb->zTrace, zTrace, len+1);
		}else{
			pDb->zTrace = 0;
		}
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
		if( pDb->zTrace ){
			pDb->interp = interp;
			sqlite3_trace(pDb->db, DbTraceHandler, pDb);
		}else{
			sqlite3_trace(pDb->db, 0, 0);
		}
#endif
	}
}

//    $db transaction [-deferred|-immediate|-exclusive] SCRIPT
//
// Start a new transaction (if we are not already in the midst of a transaction) and execute the TCL script SCRIPT.  After SCRIPT
// completes, either commit the transaction or roll it back if SCRIPT throws an exception.  Or if no new transation was started, do nothing.
// pass the exception on up the stack.
//
// This command was inspired by Dave Thomas's talk on Ruby at the 2005 O'Reilly Open Source Convention (OSCON).
__device__ RC DB_TRANSACTION()
{
	Tcl_Obj *pScript;
	const char *zBegin = "SAVEPOINT _tcl_transaction";
	if( objc!=3 && objc!=4 ){
		Tcl_WrongNumArgs(interp, 2, objv, "[TYPE] SCRIPT");
		return TCL_ERROR;
	}

	if( pDb->nTransaction==0 && objc==4 ){
		static const char *TTYPE_strs[] = {
			"deferred",   "exclusive",  "immediate", 0
		};
		enum TTYPE_enum {
			TTYPE_DEFERRED, TTYPE_EXCLUSIVE, TTYPE_IMMEDIATE
		};
		int ttype;
		if( Tcl_GetIndexFromObj(interp, objv[2], TTYPE_strs, "transaction type",
			0, &ttype) ){
				return TCL_ERROR;
		}
		switch( (enum TTYPE_enum)ttype ){
		case TTYPE_DEFERRED:    /* no-op */;                 break;
		case TTYPE_EXCLUSIVE:   zBegin = "BEGIN EXCLUSIVE";  break;
		case TTYPE_IMMEDIATE:   zBegin = "BEGIN IMMEDIATE";  break;
		}
	}
	pScript = objv[objc-1];

	/* Run the SQLite BEGIN command to open a transaction or savepoint. */
	pDb->disableAuth++;
	rc = sqlite3_exec(pDb->db, zBegin, 0, 0, 0);
	pDb->disableAuth--;
	if( rc!=SQLITE_OK ){
		Tcl_AppendResult(interp, sqlite3_errmsg(pDb->db), 0);
		return TCL_ERROR;
	}
	pDb->nTransaction++;

	/* If using NRE, schedule a callback to invoke the script pScript, then
	** a second callback to commit (or rollback) the transaction or savepoint
	** opened above. If not using NRE, evaluate the script directly, then
	** call function DbTransPostCmd() to commit (or rollback) the transaction 
	** or savepoint.  */
	if( DbUseNre() ){
		Tcl_NRAddCallback(interp, DbTransPostCmd, cd, 0, 0, 0);
		Tcl_NREvalObj(interp, pScript, 0);
	}else{
		rc = DbTransPostCmd(&cd, interp, Tcl_EvalObjEx(interp, pScript, 0));
	}
}

//    $db unlock_notify ?script?
__device__ RC DB_UNLOCK_NOTIFY()
{
#ifndef ENABLE_UNLOCK_NOTIFY
	interp->AppendResult("unlock_notify not available in this build", 0);
	rc = RC_ERROR;
#else
	if( objc!=2 && objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "?SCRIPT?");
		rc = TCL_ERROR;
	}else{
		void (*xNotify)(void **, int) = 0;
		void *pNotifyArg = 0;

		if( pDb->pUnlockNotify ){
			Tcl_DecrRefCount(pDb->pUnlockNotify);
			pDb->pUnlockNotify = 0;
		}

		if( objc==3 ){
			xNotify = DbUnlockNotify;
			pNotifyArg = (void *)pDb;
			pDb->pUnlockNotify = objv[2];
			Tcl_IncrRefCount(pDb->pUnlockNotify);
		}

		if( sqlite3_unlock_notify(pDb->db, xNotify, pNotifyArg) ){
			Tcl_AppendResult(interp, sqlite3_errmsg(pDb->db), 0);
			rc = TCL_ERROR;
		}
	}
#endif
}

//    $db wal_hook ?script?
//    $db update_hook ?script?
//    $db rollback_hook ?script?
__device__ RC DB_WAL_HOOK() { return DB_ROLLBACK_HOOK(); }
__device__ RC DB_UPDATE_HOOK() { return DB_ROLLBACK_HOOK(); }
__device__ RC DB_ROLLBACK_HOOK()
{
	/* set ppHook to point at pUpdateHook or pRollbackHook, depending on 
	** whether [$db update_hook] or [$db rollback_hook] was invoked.
	*/
	Tcl_Obj **ppHook; 
	if( choice==DB_UPDATE_HOOK ){
		ppHook = &pDb->pUpdateHook;
	}else if( choice==DB_WAL_HOOK ){
		ppHook = &pDb->pWalHook;
	}else{
		ppHook = &pDb->pRollbackHook;
	}

	if( objc!=2 && objc!=3 ){
		Tcl_WrongNumArgs(interp, 2, objv, "?SCRIPT?");
		return TCL_ERROR;
	}
	if( *ppHook ){
		Tcl_SetObjResult(interp, *ppHook);
		if( objc==3 ){
			Tcl_DecrRefCount(*ppHook);
			*ppHook = 0;
		}
	}
	if( objc==3 ){
		assert( !(*ppHook) );
		if( Tcl_GetCharLength(objv[2])>0 ){
			*ppHook = objv[2];
			Tcl_IncrRefCount(*ppHook);
		}
	}

	sqlite3_update_hook(pDb->db, (pDb->pUpdateHook?DbUpdateHandler:0), pDb);
	sqlite3_rollback_hook(pDb->db,(pDb->pRollbackHook?DbRollbackHandler:0),pDb);
	sqlite3_wal_hook(pDb->db,(pDb->pWalHook?DbWalHandler:0),pDb);

	break;
}

//    $db version
//
// Return the version string for this database.
__device__ RC DB_VERSION()
{
	Tcl_SetResult(interp, (char *)sqlite3_libversion(), TCL_STATIC);
}

#endif
#pragma endregion

//   sqlite3 DBNAME FILENAME ?-vfs VFSNAME? ?-key KEY? ?-readonly BOOLEAN?
//                           ?-create BOOLEAN? ?-nomutex BOOLEAN?
//
// This is the main Tcl command.  When the "sqlite" Tcl command is invoked, this routine runs to process that command.
//
// The first argument, DBNAME, is an arbitrary name for a new database connection.  This command creates a new command named
// DBNAME that is used to control that connection.  The database connection is deleted when the DBNAME command is deleted.
//
// The second argument is the name of the database file.
__device__ static int DbMain(void *cd, Tcl_Interp *interp, array_t<Tcl_Obj *> objv)
{
	// In normal use, each TCL interpreter runs in a single thread.  So by default, we can turn of mutexing on SQLite database connections.
	// However, for testing purposes it is useful to have mutexes turned on.  So, by default, mutexes default off.  But if compiled with
	// SQLITE_TCL_DEFAULT_FULLMUTEX then mutexes default on.
#ifdef TCL_DEFAULT_FULLMUTEX
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_FULLMUTEX);
#else
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_NOMUTEX);
#endif

	const char *arg;
	if (objv.length == 2)
	{
		arg = objv[1]->GetStringFromObj(nullptr);
		if (!_strcmp(arg, "-version"))
		{
			interp->AppendResult(CORE_VERSION, 0);
			return RC_OK;
		}
		if (!_strcmp(arg, "-has-codec"))
		{
#ifdef HAS_CODEC
			interp->AppendResult("1", 0);
#else
			interp->AppendResult("0", 0);
#endif
			return RC_OK;
		}
	}
#ifdef HAS_CODEC
	void *key = nullptr;
	int keyLength = 0;
#endif
	const char *vfsName = nullptr;
	for (int i = 3; i + 1 < objv.length; i += 2)
	{
		arg = objv[i]->GetString();
		bool b;
		if (!_strcmp(arg, "-key"))
		{
#ifdef HAS_CODEC
			key = objv[i+1]->GetByteArrayFromObj(&keyLength);
#endif
		}
		else if (!_strcmp(arg, "-vfs"))
			vfsName = objv[i+1]->GetString();
		else if (!_strcmp(arg, "-readonly"))
		{
			if (objv[i+1]->GetBooleanFromObj(interp, &b)) return RC_ERROR;
			if (b)
			{
				flags &= ~(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE);
				flags |= VSystem::OPEN_READONLY;
			}
			else
			{
				flags &= ~VSystem::OPEN_READONLY;
				flags |= VSystem::OPEN_READWRITE;
			}
		}
		else if (!_strcmp(arg, "-create"))
		{
			if (objv[i+1]->GetBooleanFromObj(interp, &b)) return RC_ERROR;
			if (b && (flags & VSystem::OPEN_READONLY) == 0)
				flags |= VSystem::OPEN_CREATE;
			else
				flags &= ~VSystem::OPEN_CREATE;
		}
		else if (!_strcmp(arg, "-nomutex"))
		{
			if (objv[i+1]->GetBooleanFromObj(interp, &b)) return RC_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_NOMUTEX;
				flags &= ~VSystem::OPEN_FULLMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_NOMUTEX;
		}
		else if (!_strcmp(arg, "-fullmutex"))
		{
			if (objv[i+1]->GetBooleanFromObj(interp, &b)) return RC_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_FULLMUTEX;
				flags &= ~VSystem::OPEN_NOMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_FULLMUTEX;
		}
		else if (!_strcmp(arg, "-uri"))
		{

			if (objv[i+1]->GetBooleanFromObj(interp, &b)) return RC_ERROR;
			if (b)
				flags |= VSystem::OPEN_URI;
			else
				flags &= ~VSystem::OPEN_URI;
		}
		else
		{
			interp->AppendResult("unknown option: ", arg, nullptr);
			return RC_ERROR;
		}
	}
	if (objv.length < 3 || (objv.length & 1) != 1)
	{
		interp->WrongNumArgs(1, objv, "HANDLE FILENAME ?-vfs VFSNAME? ?-readonly BOOLEAN? ?-create BOOLEAN?"
			" ?-nomutex BOOLEAN? ?-fullmutex BOOLEAN? ?-uri BOOLEAN?"
#ifdef HAS_CODEC
			" ?-key CODECKEY?"
#endif
			);
		return RC_ERROR;
	}
	char *errMsg = nullptr;
	TclContext *p = (TclContext *)_alloc(sizeof(*p));
	if (!p)
	{
		interp->SetResult("malloc failed", DESTRUCTOR_STATIC);
		return RC_ERROR;
	}
	_memset(p, 0, sizeof(*p));
	char *fileName = objv[2]->GetStringFromObj(0);
	RC rc = Main::Open_v2(fileName, &p->Ctx, flags, vfsName);
	if (p->Ctx)
	{
		if (Main::ErrCode(p->Ctx) != RC_OK)
		{
			errMsg = _mprintf("%s", Main::ErrMsg(p->Ctx));
			Main::Close(p->Ctx);
			p->Ctx = nullptr;
		}
	}
	else
		errMsg = _mprintf("%s", Main::ErrStr(rc));
#ifdef HAS_CODEC
	if (p->Ctx)
		sqlite3_key(p->Ctx, key, keyLength);
#endif
	if (!p->Ctx)
	{
		interp->SetResult(errMsg, DESTRUCTOR_TRANSIENT);
		_free(p);
		_free(errMsg);
		return RC_ERROR;
	}
	p->MaxStmt = NUM_PREPARED_STMTS;
	p->Interp = interp;
	arg = objv[1]->GetStringFromObj(nullptr);
	interp->CreateObjCommand(arg, nullptr, (ClientData)p, DbDeleteCmd);
	return RC_OK;
}
