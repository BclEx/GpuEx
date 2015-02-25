#include "Core+Vdbe.cu.h"
#ifdef ENABLE_FTS3
#include "fts3.h"
#endif
#ifdef ENABLE_RTREE
#include "rtree.h"
#endif
#ifdef ENABLE_ICU
#include "sqliteicu.h"
#endif

namespace Core
{

#pragma region From: Util_c

	__device__ void Parse::ErrorMsg(const char *fmt, va_list *args)
	{
		Context *ctx = Ctx;
		char *msg = _vmtagprintf(ctx, fmt, args, nullptr);
		if (ctx->SuppressErr)
			_tagfree(ctx, msg);
		else
		{
			Errs++;
			_tagfree(ctx, ErrMsg);
			ErrMsg = msg;
			RC = RC_ERROR;
		}
	}

	__device__ void Main::Error(Context *ctx, RC errCode, const char *fmt, va_list *args)
	{
		if (ctx && (ctx->Err || (ctx->Err = Vdbe::ValueNew(ctx)) != nullptr))
		{
			ctx->ErrCode = errCode;
			if (fmt)
			{
				char *z = _vmtagprintf(ctx, fmt, args, nullptr);
				Vdbe::ValueSetStr(ctx->Err, -1, z, TEXTENCODE_UTF8, DESTRUCTOR_DYNAMIC);
			}
			else
				Vdbe::ValueSetStr(ctx->Err, 0, nullptr, TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
		}
	}

	__device__ inline static void LogBadConnection(const char *type)
	{
		SysEx_LOG(RC_MISUSE, "API call with %s database connection pointer", type);
	}

	__device__ bool Main::SafetyCheckOk(Context *ctx)
	{
		if (!ctx)
		{
			LogBadConnection("NULL");
			return false;
		}
		MAGIC magic = ctx->Magic;
		if (magic != MAGIC_OPEN)
		{
			if (SafetyCheckSickOrOk(ctx))
			{
				ASSERTCOVERAGE(SysEx_GlobalStatics.Log != nullptr);
				LogBadConnection("unopened");
			}
			return false;
		}
		return true;
	}

	__device__ bool Main::SafetyCheckSickOrOk(Context *ctx)
	{
		MAGIC magic = ctx->Magic;
		if (magic != MAGIC_SICK && magic != MAGIC_OPEN && magic != MAGIC_BUSY)
		{
			ASSERTCOVERAGE(SysEx_GlobalStatics.Log != nullptr);
			LogBadConnection("invalid");
			return false;
		}
		return true;
	}

#pragma endregion

	//#if !defined(OMIT_TRACE) && defined(ENABLE_IOTRACE)
	//	// If the following function pointer is not NULL and if SQLITE_ENABLE_IOTRACE is enabled, then messages describing
	//	// I/O active are written using this function.  These messages are intended for debugging activity only.
	//	void (*sqlite3IoTrace)(const char*, ...) = nullptr;
	//#endif

#pragma region Initialize/Shutdown/Config

	// If the following global variable points to a string which is the name of a directory, then that directory will be used to store temporary files.
	// See also the "PRAGMA temp_store_directory" SQL command.
	__device__ char *g_temp_directory = nullptr;
	// If the following global variable points to a string which is the name of a directory, then that directory will be used to store
	// all database files specified with a relative pathname.
	// See also the "PRAGMA data_store_directory" SQL command.
	__device__ char *g_data_directory = nullptr;

#ifndef ALLOW_COVERING_INDEX_SCAN
#define ALLOW_COVERING_INDEX_SCAN true
#endif

	// The following singleton contains the global configuration for the SQLite library.
	__device__ _WSD Main::GlobalStatics g_globalStatics =
	{
		ALLOW_COVERING_INDEX_SCAN,		// UseCis
		//{0,0,0,0,0,0,0,0,0,0,0,0,0},	// pcache2
		nullptr,						// Page
		0,								// PageSize
		0,								// Pages
		0,								// MaxParserStack
		// All the rest should always be initialized to zero
		false							// IsPCacheInit
	};

	__device__ _WSD FuncDefHash g_globalFunctions;

	__device__ RC Main::Initialize()
	{
		if (SysEx_GlobalStatics.IsInit) return RC_OK;

		MutexEx masterMutex;
		RC rc = SysEx::PreInitialize(masterMutex);
		if (rc != RC_OK) return rc;

		FuncDefHash *hash = &Main_GlobalFunctions;
		_memset(hash, 0, sizeof(g_globalFunctions));
		Func::RegisterGlobalFunctions();
		if (!Main_GlobalStatics.IsPCacheInit)
			rc = PCache::Initialize();
		if (rc == RC_OK)
		{
			Main_GlobalStatics.IsPCacheInit = true;
			PCache::PageBufferSetup(Main_GlobalStatics.Page, Main_GlobalStatics.PageSize, Main_GlobalStatics.Pages);
			SysEx_GlobalStatics.IsInit = true;
		}
		SysEx_GlobalStatics.InProgress = false;

		// Do extra initialization steps requested by the EXTRA_INIT compile-time option.
#ifdef EXTRA_INIT
		if (rc == RC_OK && SysEx_GlobalStatics.IsInit)
		{
			RC EXTRA_INIT(const char *);
			rc = EXTRA_INIT(nullptr);
		}
#endif

		SysEx::PostInitialize(masterMutex);
		return rc;
	}

	__device__ RC Main::Shutdown()
	{
		if (!SysEx_GlobalStatics.IsInit) return RC_OK;

		// Do extra shutdown steps requested by the EXTRA_SHUTDOWN compile-time option.
#ifdef EXTRA_SHUTDOWN
		void EXTRA_SHUTDOWN();
		EXTRA_SHUTDOWN();
#endif

		if (Main_GlobalStatics.IsPCacheInit)
		{
			PCache::Shutdown();
			Main_GlobalStatics.IsPCacheInit = false;
		}
#ifndef OMIT_SHUTDOWN_DIRECTORIES
		// The heap subsystem has now been shutdown and these values are supposed to be NULL or point to memory that was obtained from sqlite3_malloc(),
		// which would rely on that heap subsystem; therefore, make sure these values cannot refer to heap memory that was just invalidated when the
		// heap subsystem was shutdown.  This is only done if the current call to this function resulted in the heap subsystem actually being shutdown.
		g_data_directory = nullptr;
		g_temp_directory = nullptr;
#endif

		SysEx::Shutdown();
		return RC_OK;
	}

	__device__ RC Main::Config(CONFIG op, va_list *args)
	{
		if (op < CONFIG_PAGECACHE) return SysEx::Config((SysEx::CONFIG)op, args);
		RC rc = RC_OK;
		switch (op)
		{
		case CONFIG_PAGECACHE: { // Designate a buffer for page cache memory space
			Main_GlobalStatics.Page = va_arg(args, void*);
			Main_GlobalStatics.PageSize = va_arg(args, int);
			Main_GlobalStatics.Pages = va_arg(args, int);
			break; }
		case CONFIG_PCACHE: { // no-op
			break; }
		case CONFIG_GETPCACHE: { // now an error
			rc = RC_ERROR;
			break; }
		case CONFIG_PCACHE2: { // Specify an alternative page cache implementation
			//Main_GlobalStatics.PCache2 = *va_arg(args, sqlite3_pcache_methods2*);
			break; }
		case CONFIG_GETPCACHE2: {
			//if (Main_GlobalStatics.Pcache2.Init == 0)
			//	PCacheSetDefault();
			//*va_arg(args, sqlite3_pcache_methods2*) = SysEx_GlobalStatics.pcache2;
			break; }
		case CONFIG_COVERING_INDEX_SCAN: {
			Main_GlobalStatics.UseCis = va_arg(args, bool);
			break; }
		default: {
			rc = RC_ERROR;
			break; }
		}
		return rc;
	}

	__constant__ static const struct {
		int OP;      // The opcode
		Context::FLAG Mask; // Mask of the bit in sqlite3.flags to set/clear
	} _flagOps[] = {
		{ Main::CTXCONFIG_ENABLE_FKEY,    Context::FLAG_ForeignKeys   },
		{ Main::CTXCONFIG_ENABLE_TRIGGER, Context::FLAG_EnableTrigger },
	};
	__device__ RC Main::CtxConfig(Context *ctx, CTXCONFIG op, va_list *args)
	{
		RC rc;
		switch (op)
		{
		case CTXCONFIG_LOOKASIDE: {
			void *buf = va_arg(args, void*); // IMP: R-26835-10964
			int size = va_arg(args, int);       // IMP: R-47871-25994
			int count = va_arg(args, int);      // IMP: R-04460-53386
			rc = SysEx::SetupLookaside(ctx, buf, size, count);
			break; }
		default: {
			rc = RC_ERROR; // IMP: R-42790-23372
			for (int i = 0; i < _lengthof(_flagOps); i++)
			{
				if (_flagOps[i].OP == op)
				{
					bool set = va_arg(args, bool);
					bool *r = va_arg(args, bool*);
					Context::FLAG oldFlags = ctx->Flags;
					if (set)
						ctx->Flags |= _flagOps[i].Mask;
					else
						ctx->Flags &= ~_flagOps[i].Mask;
					if (oldFlags != ctx->Flags)
						Vdbe::ExpirePreparedStatements(ctx);
					if (r)
						*r = ((ctx->Flags & _flagOps[i].Mask) != 0);
					rc = RC_OK;
					break;
				}
			}
			break; }
		}
		return rc;
	}

#pragma endregion

	//__device__ MutexEx *Main::CtxMutex(Context *ctx) { return ctx->Mutex; }

	__device__ RC Main::CtxReleaseMemory(Context *ctx)
	{
		MutexEx::Enter(ctx->Mutex);
		Btree::EnterAll(ctx);
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
			{
				Pager *pager = bt->get_Pager();
				pager->Shrink();
			}
		}
		Btree::LeaveAll(ctx);
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

	__device__ static bool AllSpaces(const char *z, int n)
	{
		while (n > 0 && z[n-1] == ' ' ) { n--; }
		return (n == 0);
	}

	__device__ static int BinCollFunc(void *padFlag, int key1Length, const void *key1, int key2Length, const void *key2)
	{
		int n = (key1Length < key2Length ? key1Length : key2Length);
		int rc = _memcmp(key1, key2, n);
		if (rc == 0)
		{
			if (padFlag && AllSpaces(((char *)key1)+n, key1Length-n) && AllSpaces(((char *)key2)+n, key2Length-n)) { } // Leave rc unchanged at 0
			else rc = key1Length - key2Length;
		}
		return rc;
	}

	__device__ static int NocaseCollatingFunc(void *dummy1, int key1Length, const void *key1, int key2Length, const void *key2)
	{
		int r = _strncmp((const char *)key1, (const char *)key2, (key1Length < key2Length ? key1Length : key2Length));
		if (r == 0)
			r = key1Length - key2Length;
		return r;
	}

	//__device__ int64 Main::CtxLastInsertRowid(Context *ctx) { return ctx->LastRowid; }
	//__device__ int Main::CtxChanges(Context *ctx) { return ctx->Changes; }
	//__device__ int Main::CtxTotalChanges(Context *ctx) { return ctx->TotalChanges; }

	__device__ void Main::CloseSavepoints(Context *ctx)
	{
		while (ctx->Savepoints)
		{
			Savepoint *t = ctx->Savepoints;
			ctx->Savepoints = t->Next;
			_tagfree(ctx, t);
		}
		ctx->SavepointsLength = 0;
		ctx->Statements = 0;
		ctx->IsTransactionSavepoint = false;
	}

	__device__ static void FunctionDestroy(Context *ctx, FuncDef *p)
	{
		FuncDestructor *destructor = p->Destructor;
		if (destructor)
		{
			destructor->Refs--;
			if (destructor->Refs == 0)
			{
				destructor->Destroy(destructor->UserData);
				_tagfree(ctx, destructor);
			}
		}
	}

	__device__ static void DisconnectAllVtab(Context *ctx)
	{
#ifndef OMIT_VIRTUALTABLE
		Btree::EnterAll(ctx);
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Schema *schema = ctx->DBs[i].Schema;
			if (ctx->DBs[i].Schema)
				for (HashElem *p = schema->TableHash.First; p; p = p->Next)
				{
					Table *table = (Table *)p->Data;
					if (IsVirtual(table)) VTable::Disconnect(ctx, table);
				}
		}
		Btree::LeaveAll(ctx);
#endif
	}

	__device__ static bool ConnectionIsBusy(Context *ctx)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		if (ctx->Vdbes) return true;
		for (int j = 0; j < ctx->DBs.length; j++)
		{
			Btree *bt = ctx->DBs[j].Bt;
			if (bt && bt->IsInBackup()) return true;
		}
		return false;
	}

#pragma region Close/Rollback

	__device__ RC Main::Close(Context *ctx) { return Close(ctx, false); }
	__device__ RC Main::Close_v2(Context *ctx) { return Close(ctx, true); }
	__device__ RC Main::Close(Context *ctx, bool forceZombie)
	{
		if (!ctx)
			return RC_OK;
		if (!SafetyCheckSickOrOk(ctx))
			return SysEx_MISUSE_BKPT;
		MutexEx::Enter(ctx->Mutex);

		// Force xDisconnect calls on all virtual tables
		DisconnectAllVtab(ctx);

		// If a transaction is open, the disconnectAllVtab() call above will not have called the xDisconnect() method on any virtual
		// tables in the ctx->aVTrans[] array. The following sqlite3VtabRollback() call will do so. We need to do this before the check for active
		// SQL statements below, as the v-table implementation may be storing some prepared statements internally.
		VTable::Rollback(ctx);

		// Legacy behavior (sqlite3_close() behavior) is to return SQLITE_BUSY if the connection can not be closed immediately.
		if (!forceZombie && ConnectionIsBusy(ctx))
		{
			Error(ctx, RC_BUSY, "unable to close due to unfinalized statements or unfinished backups");
			MutexEx::Leave(ctx->Mutex);
			return RC_BUSY;
		}

#ifdef ENABLE_SQLLOG
		if (SysEx_GlobalStatics.Sqllog)
			SysEx_GlobalStatics.Sqllog(SysEx_GlobalStatics.SqllogArg, ctx, 0, 2); // Closing the handle. Fourth parameter is passed the value 2.
#endif

		// Convert the connection into a zombie and then close it.
		ctx->Magic = MAGIC_ZOMBIE;
		LeaveMutexAndCloseZombie(ctx);
		return RC_OK;
	}

	__device__ void Main::LeaveMutexAndCloseZombie(Context *ctx)
	{
		// If there are outstanding sqlite3_stmt or sqlite3_backup objects or if the connection has not yet been closed by sqlite3_close_v2(),
		// then just leave the mutex and return.
		if (ctx->Magic != MAGIC_ZOMBIE || ConnectionIsBusy(ctx))
		{
			MutexEx::Leave(ctx->Mutex);
			return;
		}

		// If we reach this point, it means that the database connection has closed all sqlite3_stmt and sqlite3_backup objects and has been
		// passed to sqlite3_close (meaning that it is a zombie).  Therefore, go ahead and free all resources.

		// Free any outstanding Savepoint structures.
		CloseSavepoints(ctx);

		// Close all database connections
		int j;
		for (j = 0; j < ctx->DBs.length; j++)
		{
			Context::DB *dbAsObj = &ctx->DBs[j];
			if (dbAsObj->Bt)
			{
				dbAsObj->Bt->Close();
				dbAsObj->Bt = nullptr;
				if (j != 1)
					dbAsObj->Schema = nullptr;
			}
		}

		// Clear the TEMP schema separately and last
		if (ctx->DBs[1].Schema)
			Callback::SchemaClear(ctx->DBs[1].Schema);
		VTable::UnlockList(ctx);

		// Free up the array of auxiliary databases
		Parse::CollapseDatabaseArray(ctx);
		_assert(ctx->DBs.length <= 2);
		_assert(ctx->DBs.data == ctx->DBStatics);

		// Tell the code in notify.c that the connection no longer holds any locks and does not require any further unlock-notify callbacks.
		ctx->ConnectionClosed();

		for (j = 0; j < _lengthof(ctx->Funcs.data); j++)
		{
			FuncDef *next, *hash;
			for (FuncDef *p = ctx->Funcs.data[j]; p; p = hash)
			{
				hash = p->Hash;
				while (p)
				{
					FunctionDestroy(ctx, p);
					next = p->Next;
					_tagfree(ctx, p);
					p = next;
				}
			}
		}
		HashElem *i;
		for (i = ctx->CollSeqs.First; i; i = i->Next) // Hash table iterator
		{ 
			CollSeq *coll = (CollSeq *)i->Data;
			// Invoke any destructors registered for collation sequence user data.
			for (j = 0; j < 3; j++)
				if (coll[j].Del)
					coll[j].Del(coll[j].User);
			_tagfree(ctx, coll);
		}
		ctx->CollSeqs.Clear();
#ifndef OMIT_VIRTUALTABLE
		for (i = ctx->Modules.First; i; i = i->Next)
		{
			TableModule *mod = (TableModule *)i->Data;
			if (mod->Destroy)
				mod->Destroy(mod->Aux);
			_tagfree(ctx, mod);
		}
		ctx->Modules.Clear();
#endif

		Error(ctx, RC_OK, nullptr); // Deallocates any cached error strings.
		if (ctx->Err)
			Vdbe::ValueFree(ctx->Err);
		//LoadExt::CloseExtensions(ctx);

		ctx->Magic = MAGIC_ERROR;

		// The temp-database schema is allocated differently from the other schema objects (using sqliteMalloc() directly, instead of sqlite3BtreeSchema()).
		// So it needs to be freed here. Todo: Why not roll the temp schema into the same sqliteMalloc() as the one that allocates the database structure?
		_tagfree(ctx, ctx->DBs[1].Schema);
		MutexEx::Leave(ctx->Mutex);
		ctx->Magic = MAGIC_CLOSED;
		MutexEx::Free(ctx->Mutex);
		_assert(ctx->Lookaside.Outs == 0); // Fails on a lookaside memory leak
		if (ctx->Lookaside.Malloced)
			_free(ctx->Lookaside.Start);
		_free(ctx);
	}

	__device__ void Main::RollbackAll(Context *ctx, RC tripCode)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		_benignalloc_begin();
		bool inTrans = false;
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Btree *p = ctx->DBs[i].Bt;
			if (p)
			{
				if (p->IsInTrans())
					inTrans = true;
				p->Rollback(tripCode);
				ctx->DBs[i].InTrans = false;
			}
		}
		VTable::Rollback(ctx);
		_benignalloc_end();

		if ((ctx->Flags & Context::FLAG_InternChanges) != 0 && !ctx->Init.Busy)
		{
			Vdbe::ExpirePreparedStatements(ctx);
			Parse::ResetAllSchemasOfConnection(ctx);
		}

		// Any deferred constraint violations have now been resolved.
		ctx->DeferredCons = 0;

		// If one has been configured, invoke the rollback-hook callback
		if (ctx->RollbackCallback && (inTrans || !ctx->AutoCommit))
			ctx->RollbackCallback(ctx->RollbackArg);
	}

#pragma endregion

	__constant__ static const char* const _msgs[] =
	{
		/* RC_OK          */ "not an error",
		/* RC_ERROR       */ "SQL logic error or missing database",
		/* RC_INTERNAL    */ nullptr,
		/* RC_PERM        */ "access permission denied",
		/* RC_ABORT       */ "callback requested query abort",
		/* RC_BUSY        */ "database is locked",
		/* RC_LOCKED      */ "database table is locked",
		/* RC_NOMEM       */ "out of memory",
		/* RC_READONLY    */ "attempt to write a readonly database",
		/* RC_INTERRUPT   */ "interrupted",
		/* RC_IOERR       */ "disk I/O error",
		/* RC_CORRUPT     */ "database disk image is malformed",
		/* RC_NOTFOUND    */ "unknown operation",
		/* RC_FULL        */ "database or disk is full",
		/* RC_CANTOPEN    */ "unable to open database file",
		/* RC_PROTOCOL    */ "locking protocol",
		/* RC_EMPTY       */ "table contains no data",
		/* RC_SCHEMA      */ "database schema has changed",
		/* RC_TOOBIG      */ "string or blob too big",
		/* RC_CONSTRAINT  */ "constraint failed",
		/* RC_MISMATCH    */ "datatype mismatch",
		/* RC_MISUSE      */ "library routine called out of sequence",
		/* RC_NOLFS       */ "large file support is disabled",
		/* RC_AUTH        */ "authorization denied",
		/* RC_FORMAT      */ "auxiliary database format error",
		/* RC_RANGE       */ "bind or column index out of range",
		/* RC_NOTADB      */ "file is encrypted or is not a database",
	};
	__device__ const char *Main::ErrStr(RC rc)
	{
		const char *err = "unknown error";
		switch (rc)
		{
		case RC_ABORT_ROLLBACK: {
			err = "abort due to ROLLBACK";
			break; }
		default: {
			rc &= 0xff;
			if (_ALWAYS(rc >= 0) && rc < _lengthof(_msgs) && _msgs[rc] != nullptr)
				err = _msgs[rc];
			break; }
		}
		return err;
	}

#pragma region Busy Handler

#if OS_WIN || (defined(HAVE_USLEEP) && HAVE_USLEEP)
	__constant__ const uint8 _delays[] = { 1, 2, 5, 10, 15, 20, 25, 25,  25,  50,  50, 100 };
	__constant__ static const uint8 _totals[] = { 0, 1, 3,  8, 18, 33, 53, 78, 103, 128, 178, 228 };
#define NDELAY _lengthof(_delays)
#endif
	__device__ int Main::DefaultBusyCallback(void *ptr, int count)
	{
		Context *ctx = (Context *)ptr;
		int timeout = ctx->BusyTimeout;
		_assert(count >= 0);
#if OS_WIN || (defined(HAVE_USLEEP) && HAVE_USLEEP)
		int delay, prior;
		if (count < NDELAY)
		{
			delay = _delays[count];
			prior = _totals[count];
		}
		else
		{
			delay = _delays[NDELAY-1];
			prior = _totals[NDELAY-1] + delay*(count-(NDELAY-1));
		}
		if (prior + delay > timeout)
		{
			delay = timeout - prior;
			if (delay <= 0) return 0;
		}
		ctx->Vfs->Sleep(delay*1000);
		return 1;
#else
		if ((count+1)*1000 > timeout)
			return 0;
		ctx->Vfs->Sleep(1000000);
		return 1;
#endif
	}

	// Moved to BContext
	//__device__ int Main::InvokeBusyHandler(Context::BusyHandlerType *p)
	//{
	//	if (_NEVER(p == nullptr) || p->Func == nullptr || p->Busys < 0) return 0;
	//	int rc = p->Func(p->Arg, p->Busys);
	//	if (rc == 0)
	//		p->Busys = -1;
	//	else
	//		p->Busys++;
	//	return rc; 
	//}

	__device__ RC Main::BusyHandler(Context *ctx, int (*busy)(void *, int), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		ctx->BusyHandler.Func = busy;
		ctx->BusyHandler.Arg = arg;
		ctx->BusyHandler.Busys = 0;
		ctx->BusyTimeout = 0;
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

#ifndef OMIT_PROGRESS_CALLBACK
	__device__ void Main::ProgressHandler(Context *ctx,  int ops, int (*progress)(void *), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		if (ops > 0)
		{
			ctx->Progress = progress;
			ctx->ProgressOps = ops;
			ctx->ProgressArg = arg;
		}
		else
		{
			ctx->Progress = nullptr;
			ctx->ProgressOps = 0;
			ctx->ProgressArg = nullptr;
		}
		MutexEx::Leave(ctx->Mutex);
	}
#endif

	__device__ RC Main::BusyTmeout(Context *ctx, int ms)
	{
		if (ms > 0)
		{
			BusyHandler(ctx, DefaultBusyCallback, (void *)ctx);
			ctx->BusyTimeout = ms;
		}
		else
			BusyHandler(ctx, nullptr, nullptr);
		return RC_OK;
	}

	__device__ void Main::Interrupt(Context *ctx)
	{
		ctx->u1.IsInterrupted = true;
	}

#pragma endregion

#pragma region Function

	__device__ RC Main::CreateFunc(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *userData, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*), FuncDestructor *destructor)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		int funcNameLength;
		if (!funcName ||
			(func && (final_ || step)) || 
			(!func && (final_ && !step)) ||
			(!func && (!final_ && step)) ||
			(args < -1 || args > MAX_FUNCTION_ARG) ||
			(255 < (funcNameLength = _strlen30(funcName))))
			return SysEx_MISUSE_BKPT;

#ifndef OMIT_UTF16
		// If SQLITE_UTF16 is specified as the encoding type, transform this to one of SQLITE_UTF16LE or SQLITE_UTF16BE using the
		// SQLITE_UTF16NATIVE macro. SQLITE_UTF16 is not used internally.
		//
		// If SQLITE_ANY is specified, add three versions of the function to the hash table.
		if (encode == TEXTENCODE_UTF16)
			encode = TEXTENCODE_UTF16NATIVE;
		else if (encode == TEXTENCODE_ANY)
		{
			RC rc = CreateFunc(ctx, funcName, args, TEXTENCODE_UTF8, userData, func, step, final_, destructor);
			if (rc == RC_OK)
				rc = CreateFunc(ctx, funcName, args, TEXTENCODE_UTF16LE, userData, func, step, final_, destructor);
			if (rc != RC_OK)
				return rc;
			encode = TEXTENCODE_UTF16BE;
		}
#else
		encode = TEXTENCODE_UTF8;
#endif

		// Check if an existing function is being overridden or deleted. If so, and there are active VMs, then return SQLITE_BUSY. If a function
		// is being overridden/deleted but there are no active VMs, allow the operation to continue but invalidate all precompiled statements.
		FuncDef *p = Callback::FindFunction(ctx, funcName, funcNameLength, args, encode, false);
		if (p && p->PrefEncode == encode && p->Args == args)
		{
			if (ctx->ActiveVdbeCnt)
			{
				Error(ctx, RC_BUSY, "unable to delete/modify user-function due to active statements");
				_assert(!ctx->MallocFailed);
				return RC_BUSY;
			}
			Vdbe::ExpirePreparedStatements(ctx);
		}

		p = Callback::FindFunction(ctx, funcName, funcNameLength, args, encode, true);
		_assert(p || ctx->MallocFailed);
		if (!p)
			return RC_NOMEM;

		// If an older version of the function with a configured destructor is being replaced invoke the destructor function here.
		FunctionDestroy(ctx, p);

		if (destructor)
			destructor->Refs++;
		p->Destructor = destructor;
		p->Flags = (FUNC)0;
		p->Func = func;
		p->Step = step;
		p->Finalize = final_;
		p->UserData = userData;
		p->Args = (int16)args;
		return RC_OK;
	}

	__device__ RC Main::CreateFunction(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*)) { return CreateFunction_v2(ctx, funcName, args, encode, p, func, step, final_, nullptr); }
	__device__ RC Main::CreateFunction_v2(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*), void (*destroy)(void*))
	{
		RC rc = RC_ERROR;
		FuncDestructor *arg = nullptr;
		MutexEx::Enter(ctx->Mutex);
		if (destroy)
		{
			arg = (FuncDestructor *)_tagalloc2(ctx, sizeof(FuncDestructor), true);
			if (!arg)
			{
				destroy(p);
				goto _out;
			}
			arg->Destroy = destroy;
			arg->UserData = p;
		}
		rc = CreateFunc(ctx, funcName, args, encode, p, func, step, final_, arg);
		if (arg && arg->Refs == 0)
		{
			_assert(rc != RC_OK);
			destroy(p);
			_tagfree(ctx, arg);
		}
_out:
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

#ifndef OMIT_UTF16
	__device__ RC Main::CreateFunction16(Context *ctx, const void *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*))
	{
		MutexEx::Enter(ctx->Mutex);
		_assert(!ctx->MallocFailed);
		char *funcName8 = Vdbe::Utf16to8(ctx, funcName, -1, TEXTENCODE_UTF16NATIVE);
		RC rc = CreateFunc(ctx, funcName8, args, encode, p, func, step, final_, nullptr);
		_tagfree(ctx, funcName8);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}
#endif

	__device__ RC Main::OverloadFunction(Context *ctx, const char *funcName, int args)
	{
		int funcNameLength = _strlen30(funcName);
		RC rc;
		MutexEx::Enter(ctx->Mutex);
		if (!Callback::FindFunction(ctx, funcName, funcNameLength, args, TEXTENCODE_UTF8, false))
			rc = CreateFunc(ctx, funcName, args, TEXTENCODE_UTF8, nullptr, Vdbe::InvalidFunction, nullptr, nullptr, nullptr);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

#pragma endregion

#pragma region Callback

#ifndef OMIT_TRACE
	__device__ void *Main::Trace(Context *ctx, void (*trace)(void*,const char*), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->TraceArg;
		ctx->Trace = trace;
		ctx->TraceArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
	}

	__device__ void *Main::Profile(Context *ctx, void (*profile)(void*,const char*,uint64), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->ProfileArg;
		ctx->Profile = profile;
		ctx->ProfileArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
	}
#endif

	__device__ void *Main::CommitHook(Context *ctx, RC (*callback)(void*), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->CommitArg;
		ctx->CommitCallback = callback;
		ctx->CommitArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
	}

	__device__ void *Main::UpdateHook(Context *ctx, void (*callback)(void*,int,char const*,char const*,int64), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->UpdateArg;
		ctx->UpdateCallback = callback;
		ctx->UpdateArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
	}

	__device__ void *Main::RollbackHook(Context *ctx, void (*callback)(void*), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->RollbackArg;
		ctx->RollbackCallback = callback;
		ctx->RollbackArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
	}

#ifndef OMIT_WAL
	__device__ RC Main::WalDefaultHook(void *clientData, Context *ctx, const char *dbName, int frames)
	{
		if (frames >= PTR_TO_INT(clientData))
		{
			_benignalloc_begin();
			WalCheckpoint(ctx, dbName);
			_benignalloc_end();
		}
		return RC_OK;
	}
#endif

	__device__ RC Main::WalAutocheckpoint(Context *ctx, int frames)
	{
#ifdef OMIT_WAL
		return RC_OK;
#else
		if (frames > 0)
			WalHook(ctx, WalDefaultHook, INT_TO_PTR(frames));
		else
			WalHook(ctx, nullptr, 0);
		return RC_OK;
#endif
	}

	__device__ void *Main::WalHook(Context *ctx, int (*callback)(void*,Context*,const char*,int), void *arg)
	{
#ifdef OMIT_WAL
		return nullptr;
#else
		MutexEx::Enter(ctx->Mutex);
		void *oldArg = ctx->WalArg;
		ctx->WalCallback = callback;
		ctx->WalArg = arg;
		MutexEx::Leave(ctx->Mutex);
		return oldArg;
#endif
	}

#pragma endregion

#pragma region Checkpoint

	__device__ RC Main::WalCheckpoint(Context *ctx, const char *dbName) { return WalCheckpoint_v2(ctx, dbName, IPager::CHECKPOINT_PASSIVE, nullptr, nullptr); }
	__device__ RC Main::WalCheckpoint_v2(Context *ctx, const char *dbName, IPager::CHECKPOINT mode, int *logsOut, int *ckptsOut)
	{
#ifdef OMIT_WAL
		return RC_OK;
#else
		int db = CORE_MAX_ATTACHED; // sqlite3.aDb[] index of ctx to checkpoint

		// Initialize the output variables to -1 in case an error occurs. */
		if (logsOut) *logsOut = -1;
		if (ckptsOut) *ckptsOut = -1;

		_assert(IPager::CHECKPOINT_FULL > IPager::CHECKPOINT_PASSIVE);
		_assert(IPager::CHECKPOINT_FULL < IPager::CHECKPOINT_RESTART);
		_assert(IPager::CHECKPOINT_PASSIVE+2 == IPager::CHECKPOINT_RESTART);
		if (mode < IPager::CHECKPOINT_PASSIVE || mode > IPager::CHECKPOINT_RESTART)
			return RC_MISUSE;

		MutexEx::Enter(ctx->Mutex);
		if (dbName && dbName[0])
			db = Parse::FindDbName(ctx, dbName);
		RC rc;
		if (db < 0)
		{
			rc = RC_ERROR;
			Error(ctx, RC_ERROR, "unknown database: %s", dbName);
		}
		else
		{
			rc = Checkpoint(ctx, db, mode, logsOut, ckptsOut);
			Error(ctx, rc, nullptr);
		}
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
#endif
	}

#ifndef OMIT_WAL
	__device__ RC Main::Checkpoint(Context *ctx, int db, IPager::CHECKPOINT mode, int *logsOut, int *ckptsOut)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		_assert(!logsOut || *logsOut == -1);
		_assert(!ckptsOut || *ckptsOut == -1);

		RC rc = RC_OK;
		bool busy = false; // True if SQLITE_BUSY has been encountered
		for (int i = 0; i < ctx->DBs.length && rc == RC_OK; i++)
		{
			if (i == db || db == MAX_ATTACHED)
			{
				rc = Btree::Checkpoint(ctx->DBs[i].Bt, mode, logsOut, ckptsOut);
				logsOut = nullptr;
				ckptsOut = nullptr;
				if (rc == RC_BUSY)
				{
					busy = true;
					rc = RC_OK;
				}
			}
		}
		return (rc == RC_OK && busy ? RC_BUSY : rc);
	}
#endif

#pragma endregion

	// Moved to BContext
	//	__device__ bool Main::TempInMemory(Context *ctx)
	//	{
	//#if TEMP_STORE == 1
	//		return (ctx->TempStore == 2);
	//#endif
	//#if TEMP_STORE == 2
	//		return (ctx->TempStore != 1);
	//#endif
	//#if TEMP_STORE == 3
	//		return true;
	//#endif
	//		return false;
	//	}

#pragma region Error Message

	__device__ const char *Main::ErrMsg(Context *ctx)
	{
		const char *z;
		if (!ctx)
			return ErrStr(RC_NOMEM);
		if (!SafetyCheckSickOrOk(ctx))
			return ErrStr(SysEx_MISUSE_BKPT);
		MutexEx::Enter(ctx->Mutex);
		if (ctx->MallocFailed)
			z = ErrStr(RC_NOMEM);
		else
		{
			z = (char *)Vdbe::Value_Text(ctx->Err);
			_assert(!ctx->MallocFailed);
			if (!z)
				z = ErrStr(ctx->ErrCode);
		}
		MutexEx::Leave(ctx->Mutex);
		return z;
	}

#ifndef OMIT_UTF16
	__constant__ static const uint16 _outOfMem[] =
	{
		'o', 'u', 't', ' ', 'o', 'f', ' ', 'm', 'e', 'm', 'o', 'r', 'y', 0
	};
	__constant__ static const uint16 _misuse[] =
	{
		'l', 'i', 'b', 'r', 'a', 'r', 'y', ' ', 
		'r', 'o', 'u', 't', 'i', 'n', 'e', ' ', 
		'c', 'a', 'l', 'l', 'e', 'd', ' ', 
		'o', 'u', 't', ' ', 
		'o', 'f', ' ', 
		's', 'e', 'q', 'u', 'e', 'n', 'c', 'e', 0
	};

	__device__ const void *Main::ErrMsg16(Context *ctx)
	{
		if (!ctx)
			return (void *)_outOfMem;
		if (!SafetyCheckSickOrOk(ctx))
			return (void *)_misuse;
		MutexEx::Enter(ctx->Mutex);
		const void *z;
		if (ctx->MallocFailed)
			z = (void *)_outOfMem;
		else
		{
			z = Vdbe::Value_Text16(ctx->Err);
			if (!z)
			{
				Vdbe::ValueSetStr(ctx->Err, -1, ErrStr(ctx->ErrCode), TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
				z = Vdbe::Value_Text16(ctx->Err);
			}
			// A malloc() may have failed within the call to sqlite3_value_text16() above. If this is the case, then the ctx->mallocFailed flag needs to
			// be cleared before returning. Do this directly, instead of via sqlite3ApiExit(), to avoid setting the database handle error message.
			ctx->MallocFailed = false;
		}
		MutexEx::Leave(ctx->Mutex);
		return z;
	}
#endif

	__device__ RC Main::ErrCode(Context *ctx)
	{
		if (ctx && !SafetyCheckSickOrOk(ctx))
			return SysEx_MISUSE_BKPT;
		if (!ctx || ctx->MallocFailed)
			return RC_NOMEM;
		return (RC)(ctx->ErrCode & ctx->ErrMask);
	}

	__device__ RC Main::ExtendedErrCode(Context *ctx)
	{
		if (ctx && !SafetyCheckSickOrOk(ctx))
			return SysEx_MISUSE_BKPT;
		if (!ctx || ctx->MallocFailed)
			return RC_NOMEM;
		return ctx->ErrCode;
	}

	//__device__ const char *Main::Errstr(int rc) { return ErrStr(rc); }

#pragma endregion

	__device__ static RC CreateCollation(Context *ctx, const char *name, TEXTENCODE encode, void *ctx2, int (*compare)(void*,int,const void*,int,const void*), void (*del)(void*))
	{
		int nameLength = _strlen30(name);
		_assert(MutexEx::Held(ctx->Mutex));

		// If SQLITE_UTF16 is specified as the encoding type, transform this to one of SQLITE_UTF16LE or SQLITE_UTF16BE using the
		// SQLITE_UTF16NATIVE macro. SQLITE_UTF16 is not used internally.
		TEXTENCODE encode2 = encode;
		ASSERTCOVERAGE(encode2 == TEXTENCODE_UTF16);
		ASSERTCOVERAGE(encode2 == TEXTENCODE_UTF16_ALIGNED);
		if (encode2 == TEXTENCODE_UTF16 || encode2 == TEXTENCODE_UTF16_ALIGNED)
			encode2 = TEXTENCODE_UTF16NATIVE;
		if (encode2 < TEXTENCODE_UTF8 || encode2 > TEXTENCODE_UTF16BE)
			return SysEx_MISUSE_BKPT;

		// Check if this call is removing or replacing an existing collation sequence. If so, and there are active VMs, return busy. If there
		// are no active VMs, invalidate any pre-compiled statements.
		CollSeq *coll = Callback::FindCollSeq(ctx, encode2, name, false);
		if (coll && coll->Cmp)
		{
			if (ctx->ActiveVdbeCnt)
			{
				Main::Error(ctx, RC_BUSY, "unable to delete/modify collation sequence due to active statements");
				return RC_BUSY;
			}
			Vdbe::ExpirePreparedStatements(ctx);

			// If collation sequence pColl was created directly by a call to sqlite3_create_collation, and not generated by synthCollSeq(),
			// then any copies made by synthCollSeq() need to be invalidated. Also, collation destructor - CollSeq.xDel() - function may need
			// to be called.
			if ((coll->Encode & ~TEXTENCODE_UTF16_ALIGNED) == encode2)
			{
				CollSeq *colls = (CollSeq *)ctx->CollSeqs.Find(name, nameLength);
				for (int j = 0; j < 3; j++)
				{
					CollSeq *p = &colls[j];
					if (p->Encode == coll->Encode)
					{
						if (p->Del)
							p->Del(p->User);
						p->Cmp = nullptr;
					}
				}
			}
		}

		coll = Callback::FindCollSeq(ctx, encode2, name, true);
		if (!coll) return RC_NOMEM;
		coll->Cmp = compare;
		coll->User = ctx2;
		coll->Del = del;
		coll->Encode = (encode2 | (encode & TEXTENCODE_UTF16_ALIGNED));
		Main::Error(ctx, RC_OK, nullptr);
		return RC_OK;
	}

#pragma region Limit

	// Make sure the hard limits are set to reasonable values
#if CORE_MAX_LENGTH < 100
#error CORE_MAX_LENGTH must be at least 100
#endif
#if MAX_SQL_LENGTH < 100
#error MAX_SQL_LENGTH must be at least 100
#endif
#if MAX_SQL_LENGTH > CORE_MAX_LENGTH
#error MAX_SQL_LENGTH must not be greater than CORE_MAX_LENGTH
#endif
#if MAX_COMPOUND_SELECT < 2
#error MAX_COMPOUND_SELECT must be at least 2
#endif
#if MAX_VDBE_OP < 40
#error MAX_VDBE_OP must be at least 40
#endif
#if MAX_FUNCTION_ARG < 0 || MAX_FUNCTION_ARG > 1000
#error MAX_FUNCTION_ARG must be between 0 and 1000
#endif
#if MAX_ATTACHED<0 || MAX_ATTACHED > 62
#error MAX_ATTACHED must be between 0 and 62
#endif
#if MAX_LIKE_PATTERN_LENGTH < 1
#error MAX_LIKE_PATTERN_LENGTH must be at least 1
#endif
#if MAX_COLUMN > 32767
#error MAX_COLUMN must not exceed 32767
#endif
#if MAX_TRIGGER_DEPTH < 1
#error MAX_TRIGGER_DEPTH must be at least 1
#endif

	__constant__ static const int _hardLimits[] = // kept in sync with the LIMIT_*
	{
		CORE_MAX_LENGTH,
		MAX_SQL_LENGTH,
		MAX_COLUMN,
		MAX_EXPR_DEPTH,
		MAX_COMPOUND_SELECT,
		MAX_VDBE_OP,
		MAX_FUNCTION_ARG,
		MAX_ATTACHED,
		MAX_LIKE_PATTERN_LENGTH,
		MAX_VARIABLE_NUMBER,
		MAX_TRIGGER_DEPTH,
	};

	__device__ int Main::Limit(Context *ctx, LIMIT limit, int newLimit)
	{
		// EVIDENCE-OF: R-30189-54097 For each limit category SQLITE_LIMIT_NAME there is a hard upper bound set at compile-time by a C preprocessor
		// macro called SQLITE_MAX_NAME. (The "_LIMIT_" in the name is changed to "_MAX_".)
		_assert(_hardLimits[LIMIT_LENGTH] == CORE_MAX_LENGTH);
		_assert(_hardLimits[LIMIT_SQL_LENGTH] == MAX_SQL_LENGTH);
		_assert(_hardLimits[LIMIT_COLUMN] == MAX_COLUMN);
		_assert(_hardLimits[LIMIT_EXPR_DEPTH] == MAX_EXPR_DEPTH);
		_assert(_hardLimits[LIMIT_COMPOUND_SELECT] == MAX_COMPOUND_SELECT);
		_assert(_hardLimits[LIMIT_VDBE_OP] == MAX_VDBE_OP);
		_assert(_hardLimits[LIMIT_FUNCTION_ARG] == MAX_FUNCTION_ARG);
		_assert(_hardLimits[LIMIT_ATTACHED] == MAX_ATTACHED);
		_assert(_hardLimits[LIMIT_LIKE_PATTERN_LENGTH] == MAX_LIKE_PATTERN_LENGTH);
		_assert(_hardLimits[LIMIT_VARIABLE_NUMBER] == MAX_VARIABLE_NUMBER);
		_assert(_hardLimits[LIMIT_TRIGGER_DEPTH] == MAX_TRIGGER_DEPTH);
		_assert(LIMIT_TRIGGER_DEPTH == (LIMIT_MAX_-1));

		if (limit >= LIMIT_MAX_)
			return -1;
		int oldLimit = ctx->Limits[limit];
		if (newLimit >= 0) // IMP: R-52476-28732
		{                   
			if (newLimit > _hardLimits[limit])
				newLimit = _hardLimits[limit]; // IMP: R-51463-25634
			ctx->Limits[limit] = newLimit;
		}
		return oldLimit; // IMP: R-53341-35419
	}

#pragma endregion

#pragma region Open Database

	__device__ static RC OpenDatabase(const char *fileName, Context **ctxOut, VSystem::OPEN flags, const char *vfsName)
	{
		*ctxOut = nullptr;
		RC rc;

#ifndef OMIT_AUTOINIT
		rc = SysEx::AutoInitialize();
		if (rc) return rc;
#endif

		// Only allow sensible combinations of bits in the flags argument.  Throw an error if any non-sense combination is used.  If we
		// do not block illegal combinations here, it could trigger assert() statements in deeper layers.  Sensible combinations are:
		//
		//  1:  VSystem::OPEN_READONLY
		//  2:  VSystem::OPEN_READWRITE
		//  6:  VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE
		_assert(VSystem::OPEN_READONLY  == 0x01);
		_assert(VSystem::OPEN_READWRITE == 0x02);
		_assert(VSystem::OPEN_CREATE    == 0x04);
		ASSERTCOVERAGE((1<<(flags&7)) == 0x02); // READONLY
		ASSERTCOVERAGE((1<<(flags&7)) == 0x04); // READWRITE
		ASSERTCOVERAGE((1<<(flags&7)) == 0x40); // READWRITE | CREATE
		if (((1<<(flags&7)) & 0x46) == 0) return SysEx_MISUSE_BKPT;

		bool isThreadsafe; // True for threadsafe connections
		if (!SysEx_GlobalStatics.CoreMutex) isThreadsafe = false;
		else if (flags & VSystem::OPEN_NOMUTEX) isThreadsafe = false;
		else if (flags & VSystem::OPEN_FULLMUTEX) isThreadsafe = true;
		else isThreadsafe = SysEx_GlobalStatics.FullMutex;
		if (flags & VSystem::OPEN_PRIVATECACHE) flags &= ~VSystem::OPEN_SHAREDCACHE;
		else if (SysEx_GlobalStatics.SharedCacheEnabled) flags |= VSystem::OPEN_SHAREDCACHE;

		// Remove harmful bits from the flags parameter
		//
		// The SQLITE_OPEN_NOMUTEX and SQLITE_OPEN_FULLMUTEX flags were dealt with in the previous code block.  Besides these, the only
		// valid input flags for sqlite3_open_v2() are SQLITE_OPEN_READONLY, SQLITE_OPEN_READWRITE, SQLITE_OPEN_CREATE, SQLITE_OPEN_SHAREDCACHE,
		// SQLITE_OPEN_PRIVATECACHE, and some reserved bits.  Silently mask off all other flags.
		flags &= ~(
			VSystem::OPEN_DELETEONCLOSE |
			VSystem::OPEN_EXCLUSIVE |
			VSystem::OPEN_MAIN_DB |
			VSystem::OPEN_TEMP_DB | 
			VSystem::OPEN_TRANSIENT_DB | 
			VSystem::OPEN_MAIN_JOURNAL | 
			VSystem::OPEN_TEMP_JOURNAL | 
			VSystem::OPEN_SUBJOURNAL | 
			VSystem::OPEN_MASTER_JOURNAL |
			VSystem::OPEN_NOMUTEX |
			VSystem::OPEN_FULLMUTEX |
			VSystem::OPEN_WAL);

		// Allocate the sqlite data structure
		Context *ctx = (Context *)_alloc2(sizeof(Context), true); // Store allocated handle here
		if (!ctx) goto opendb_out;
		if (isThreadsafe)
		{
			ctx->Mutex = MutexEx::Alloc(MutexEx::MUTEX_RECURSIVE);
			if (!ctx->Mutex.Tag)
			{
				_free(ctx);
				ctx = nullptr;
				goto opendb_out;
			}
		}
		MutexEx::Enter(ctx->Mutex);
		ctx->ErrMask = 0xff;
		ctx->Magic = MAGIC_BUSY;
		ctx->DBs.length = 2;
		ctx->DBs.data = ctx->DBStatics;

		_assert(sizeof(ctx->Limits) == sizeof(_hardLimits));
		_memcpy(ctx->Limits, _hardLimits, sizeof(ctx->Limits));
		ctx->AutoCommit = 1;
		ctx->NextAutovac = (Btree::AUTOVACUUM)-1;
		ctx->NextPagesize = 0;
		ctx->Flags |= Context::FLAG_ShortColNames | Context::FLAG_AutoIndex | Context::FLAG_EnableTrigger
#if DEFAULT_FILE_FORMAT<4
			| Context::FLAG_LegacyFileFmt
#endif
#ifdef ENABLE_LOAD_EXTENSION
			| Context::FLAG_LoadExtension
#endif
#if DEFAULT_RECURSIVE_TRIGGERS
			| Context::FLAG_RecTriggers
#endif
#if defined(DEFAULT_FOREIGN_KEYS) && DEFAULT_FOREIGN_KEYS
			| Context::FLAG_ForeignKeys
#endif
			;
		ctx->CollSeqs.Init();
#ifndef OMIT_VIRTUALTABLE
		ctx->Modules.Init();
#endif

		// Add the default collation sequence BINARY. BINARY works for both UTF-8 and UTF-16, so add a version for each to avoid any unnecessary
		// conversions. The only error that can occur here is a malloc() failure.
		CreateCollation(ctx, "BINARY", TEXTENCODE_UTF8, nullptr, BinCollFunc, nullptr);
		CreateCollation(ctx, "BINARY", TEXTENCODE_UTF16BE, nullptr, BinCollFunc, nullptr);
		CreateCollation(ctx, "BINARY", TEXTENCODE_UTF16LE, nullptr, BinCollFunc, nullptr);
		CreateCollation(ctx, "RTRIM", TEXTENCODE_UTF8, (void *)1, BinCollFunc, nullptr);
		if (ctx->MallocFailed)
			goto opendb_out;
		ctx->DefaultColl = Callback::FindCollSeq(ctx, TEXTENCODE_UTF8, "BINARY", false);
		_assert(ctx->DefaultColl);

		// Also add a UTF-8 case-insensitive collation sequence.
		CreateCollation(ctx, "NOCASE", TEXTENCODE_UTF8, nullptr, NocaseCollatingFunc, nullptr);

		// Parse the filename/URI argument.
		ctx->OpenFlags = flags;
		char *open = nullptr; // Filename argument to pass to BtreeOpen()
		char *errMsg = nullptr; // Error message from sqlite3ParseUri()
		rc = VSystem::ParseUri(vfsName, fileName, &flags, &ctx->Vfs, &open, &errMsg);
		if (rc != RC_OK)
		{
			if (rc == RC_NOMEM) ctx->MallocFailed = true;
			Main::Error(ctx, rc, (errMsg ? "%s" : nullptr), errMsg);
			_free(errMsg);
			goto opendb_out;
		}

		// Open the backend database driver
		rc = Btree::Open(ctx->Vfs, open, ctx, &ctx->DBs[0].Bt, (Btree::OPEN)0, flags | VSystem::OPEN_MAIN_DB);
		if (rc != RC_OK)
		{
			if (rc == RC_IOERR_NOMEM)
				rc = RC_NOMEM;
			Main::Error(ctx, rc, nullptr);
			goto opendb_out;
		}
		ctx->DBs[0].Schema = Callback::SchemaGet(ctx, ctx->DBs[0].Bt);
		ctx->DBs[1].Schema = Callback::SchemaGet(ctx, nullptr);

		// The default safety_level for the main database is 'full'; for the temp database it is 'NONE'. This matches the pager layer defaults.  
		ctx->DBs[0].Name = "main";
		ctx->DBs[0].SafetyLevel = 3;
		ctx->DBs[1].Name = "temp";
		ctx->DBs[1].SafetyLevel = 1;

		ctx->Magic = MAGIC_OPEN;
		if (ctx->MallocFailed)
			goto opendb_out;

		// Register all built-in functions, but do not attempt to read the database schema yet. This is delayed until the first time the database is accessed.
		Main::Error(ctx, RC_OK, nullptr);
		Func::RegisterBuiltinFunctions(ctx);

		// Load automatic extensions - extensions that have been registered using the sqlite3_automatic_extension() API.
		rc = Main::ErrCode(ctx);
		if (rc == RC_OK)
		{
			Main::AutoLoadExtensions(ctx);
			rc = Main::ErrCode(ctx);
			if (rc != RC_OK)
				goto opendb_out;
		}

#ifdef ENABLE_FTS1
		extern int sqlite3Fts1Init(Context *);
		if (!ctx->MallocFailed)
			rc = sqlite3Fts1Init(ctx);
#endif

#ifdef ENABLE_FTS2
		extern int sqlite3Fts2Init(Context *);
		if (!ctx->MallocFailed && rc == RC_OK)
			rc = sqlite3Fts2Init(ctx);
#endif

#ifdef ENABLE_FTS3
		if (!ctx->MallocFailed && rc == RC_OK)
			rc = sqlite3Fts3Init(ctx);
#endif

#ifdef ENABLE_ICU
		if (!ctx->MallocFailed && rc == RC_OK)
			rc = sqlite3IcuInit(ctx);
#endif

#ifdef ENABLE_RTREE
		if (!ctx->MallocFailed && rc == RC_OK)
			rc = sqlite3RtreeInit(ctx);
#endif

		Main::Error(ctx, rc, nullptr);

		// -DSQLITE_DEFAULT_LOCKING_MODE=1 makes EXCLUSIVE the default locking
		// mode.  -DSQLITE_DEFAULT_LOCKING_MODE=0 make NORMAL the default locking
		// mode.  Doing nothing at all also makes NORMAL the default.
#ifdef DEFAULT_LOCKING_MODE
		ctx->DefaultLockMode = DEFAULT_LOCKING_MODE;
		ctx->DBs[0].Bt->get_Pager()->LockingMode(DEFAULT_LOCKING_MODE);
#endif

		// Enable the lookaside-malloc subsystem
		SysEx::SetupLookaside(ctx, nullptr, SysEx_GlobalStatics.LookasideSize, SysEx_GlobalStatics.Lookasides);

		Main::WalAutocheckpoint(ctx, DEFAULT_WAL_AUTOCHECKPOINT);

opendb_out:
		_free(open);
		if (ctx)
		{
			_assert(ctx->Mutex.Tag || !isThreadsafe || !SysEx_GlobalStatics.FullMutex);
			MutexEx::Leave(ctx->Mutex);
		}
		rc = Main::ErrCode(ctx);
		_assert(ctx || rc == RC_NOMEM);
		if (rc == RC_NOMEM)
		{
			Main::Close(ctx);
			ctx = nullptr;
		}
		else if (rc != RC_OK)
			ctx->Magic = MAGIC_SICK;
		*ctxOut = ctx;
#ifdef ENABLE_SQLLOG
		if (SysEx_GlobalStatics.Sqllog)
			SysEx_GlobalStatics.Sqllog(SysEx_GlobalStatics.SqllogArg, ctx, fileName, 0); // Opening a ctx handle. Fourth parameter is passed 0.
#endif
		return Main::ApiExit(nullptr, rc);
	}

	__device__ RC Main::Open(const char *fileName, Context **ctxOut) { return OpenDatabase(fileName, ctxOut, VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE, nullptr); }
	__device__ RC Main::Open_v2(const char *fileName, Context **ctxOut, VSystem::OPEN flags, const char *vfsName) { return OpenDatabase(fileName, ctxOut, flags, vfsName); }

#ifndef OMIT_UTF16
	__device__ RC Main::Open16(const void *fileName,  Context **ctxOut)
	{
		_assert(fileName);
		_assert(ctxOut);
		*ctxOut = nullptr;
		RC rc;
#ifndef OMIT_AUTOINIT
		rc = SysEx::AutoInitialize();
		if (rc) return rc;
#endif
		Mem *val = Vdbe::ValueNew(nullptr);
		Vdbe::ValueSetStr(val, -1, fileName, TEXTENCODE_UTF16NATIVE, DESTRUCTOR_STATIC);
		const char *fileName8 = (const char *)Vdbe::ValueText(val, TEXTENCODE_UTF8); // filename encoded in UTF-8 instead of UTF-16
		if (fileName8)
		{
			rc = OpenDatabase(fileName8, ctxOut, VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE, nullptr);
			_assert(*ctxOut || rc == RC_NOMEM);
			if (rc == RC_OK && !DbHasProperty(*ctxOut, 0, SCHEMA_SchemaLoaded))
				CTXENCODE(*ctxOut) = TEXTENCODE_UTF16NATIVE;
		}
		else
			rc = RC_NOMEM;
		Vdbe::ValueFree(val);
		return ApiExit(nullptr, rc);
	}
#endif

#pragma endregion

#pragma region Create Collation

	__device__ RC Main::CreateCollation(Context *ctx, const char *name, TEXTENCODE encode, void *ctx2, int (*compare)(void*,int,const void*,int,const void*))
	{
		MutexEx::Enter(ctx->Mutex);
		_assert(!ctx->MallocFailed);
		RC rc = ::CreateCollation(ctx, name, encode, ctx2, compare, nullptr);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ RC Main::CreateCollation_v2(Context *ctx, const char *name, TEXTENCODE encode, void *ctx2, int (*compare)(void*,int,const void*,int,const void*), void (*del)(void*))
	{
		MutexEx::Enter(ctx->Mutex);
		_assert(!ctx->MallocFailed);
		RC rc = ::CreateCollation(ctx, name, encode, ctx2, compare, del);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

#ifndef OMIT_UTF16
	__device__ RC Main::CreateCollation16(Context *ctx, const void *name, TEXTENCODE encode,  void *ctx2, int (*compare)(void*,int,const void*,int,const void*))
	{
		RC rc = RC_OK;
		MutexEx::Enter(ctx->Mutex);
		_assert(!ctx->MallocFailed);
		char *name8 = Vdbe::Utf16to8(ctx, name, -1, TEXTENCODE_UTF16NATIVE);
		if (name8)
		{
			rc = ::CreateCollation(ctx, name8, encode, ctx2, compare, nullptr);
			_tagfree(ctx, name8);
		}
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}
#endif

	__device__ RC Main::CollationNeeded(Context *ctx, void *collNeededArg, void (*collNeeded)(void*,Context*,TEXTENCODE,const char*))
	{
		MutexEx::Enter(ctx->Mutex);
		ctx->CollNeeded = collNeeded;
		ctx->CollNeeded16 = nullptr;
		ctx->CollNeededArg = collNeededArg;
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

#ifndef OMIT_UTF16
	__device__ RC Main::CollationNeeded16(Context *ctx, void *collNeededArg, void (*collNeeded16)(void*,Context*,TEXTENCODE,const void*))
	{
		MutexEx::Enter(ctx->Mutex);
		ctx->CollNeeded = nullptr;
		ctx->CollNeeded16 = collNeeded16;
		ctx->CollNeededArg = collNeededArg;
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}
#endif

#pragma endregion

	// THIS IS AN EXPERIMENTAL API AND IS SUBJECT TO CHANGE

	//__device__ int Main::GetAutocommit(Context *ctx) { return ctx->AutoCommit; }
	//__device__ RC Main::CorruptError(int lineno)
	//{
	//	ASSERTCOVERAGE(SysEx_GlobalStatics.Log != nullptr);
	//	SysEx_LOG(RC_CORRUPT, "database corruption at line %d of [%.10s]", lineno, 20+sqlite3_sourceid());
	//	return RC_CORRUPT;
	//}
	//__device__ RC Main::MisuseError(int lineno)
	//{
	//	ASSERTCOVERAGE(SysEx_GlobalStatics.Log != nullptr);
	//	SysEx_LOG(RC_MISUSE, "misuse at line %d of [%.10s]", lineno, 20+sqlite3_sourceid());
	//	return RC_MISUSE;
	//}
	//__device__ RC Main::CantopenError(int lineno)
	//{
	//	ASSERTCOVERAGE(SysEx_GlobalStatics.Log != nullptr);
	//	SysEx_LOG(RC_CANTOPEN, "cannot open file at line %d of [%.10s]", lineno, 20+sqlite3_sourceid());
	//	return RC_CANTOPEN;
	//}

#pragma region Column Metadata
#ifdef ENABLE_COLUMN_METADATA
	__device__ RC Main::TableColumnMetadata(Context *ctx, const char *dbName, const char *tableName, const char *columnName, char const **dataTypeOut, char const **collSeqNameOut, bool *notNullOut, bool *primaryKeyOut, bool *autoincOut)
	{
		// Ensure the database schema has been loaded
		MutexEx::Enter(ctx->Mutex);
		Btree::EnterAll(ctx);
		char *errMsg = nullptr;
		RC rc = Prepare::Init(ctx, &errMsg);
		if (rc != RC_OK)
			goto error_out;

		// Locate the table in question
		Table *table = Parse::FindTable(ctx, tableName, dbName);
		if (!table || table->Select)
		{
			table = nullptr;
			goto error_out;
		}

		// Find the column for which info is requested
		int colId;
		Column *col = nullptr;
		if (Expr::IsRowid(columnName))
		{
			colId = table->PKey;
			if (colId >= 0)
				col = &table->Cols[colId];
		}
		else
		{
			for (colId = 0; colId < table->Cols.length; colId++)
			{
				col = &table->Cols[colId];
				if (!_strcmp(col->Name, columnName))
					break;
			}
			if (colId == table->Cols.length)
			{
				table = nullptr;
				goto error_out;
			}
		}

		// The following block stores the meta information that will be returned to the caller in local variables zDataType, zCollSeq, notnull, primarykey
		// and autoinc. At this point there are two possibilities:
		//     1. The specified column name was rowid", "oid" or "_rowid_" and there is no explicitly declared IPK column. 
		//     2. The table is not a view and the column name identified an explicitly declared column. Copy meta information from *col.
		char const *dataType = nullptr;
		char const *collSeqName = nullptr;
		bool notnull = false;
		bool primarykey = false;
		bool autoinc = false;
		if (col)
		{
			dataType = col->Type;
			collSeqName = col->Coll;
			notnull = (col->NotNull != 0);
			primarykey  = ((col->ColFlags & COLFLAG_PRIMKEY) != 0);
			autoinc = (table->PKey == colId && (table->TabFlags & TF_Autoincrement) != 0);
		}
		else
		{
			dataType = "INTEGER";
			primarykey = true;
		}
		if (!collSeqName)
			collSeqName = "BINARY";

error_out:
		Btree::LeaveAll(ctx);

		// Whether the function call succeeded or failed, set the output parameters to whatever their local counterparts contain. If an error did occur,
		// this has the effect of zeroing all output parameters.
		if (dataTypeOut) *dataTypeOut = dataType;
		if (collSeqNameOut) *collSeqNameOut = collSeqName;
		if (notNullOut) *notNullOut = notnull;
		if (primaryKeyOut) *primaryKeyOut = primarykey;
		if (autoincOut) *autoincOut = autoinc;

		if (rc == RC_OK && !table)
		{
			_tagfree(ctx, errMsg);
			errMsg = _mtagprintf(ctx, "no such table column: %s.%s", tableName, columnName);
			rc = RC_ERROR;
		}
		Error(ctx, rc, (errMsg ? "%s" : nullptr), errMsg);
		_tagfree(ctx, errMsg);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}
#endif
#pragma endregion

	__device__ int Main::Sleep(int ms)
	{
		VSystem *vfs = VSystem::FindVfs(nullptr);
		if (!vfs) return 0;
		// This function works in milliseconds, but the underlying OsSleep() API uses microseconds. Hence the 1000's.
		return (vfs->Sleep(1000*ms)/1000);
	}

	__device__ RC Main::ExtendedResultCodes(Context *ctx, bool onoff)
	{
		MutexEx::Enter(ctx->Mutex);
		ctx->ErrMask = (onoff ? 0xffffffff : 0xff);
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

	__device__ RC Main::FileControl(Context *ctx, const char *dbName, VFile::FCNTL op, void *arg)
	{
		RC rc = RC_ERROR;
		MutexEx::Enter(ctx->Mutex);
		Btree *bt = DbNameToBtree(ctx, dbName);
		if (bt)
		{
			bt->Enter();
			Pager *pager = bt->get_Pager();
			_assert(pager != nullptr);
			VFile *fd = pager->get_File();
			_assert(fd != nullptr);
			if (op == VFile::FCNTL_FILE_POINTER)
			{
				*(VFile **)arg = fd;
				rc = RC_OK;
			}
			else if (fd->Type)
				rc = fd->FileControl(op, arg);
			else
				rc = RC_NOTFOUND;
			bt->Leave();
		}
		MutexEx::Leave(ctx->Mutex);
		return rc;   
	}

#pragma region TEST
	extern __device__ void Random_PrngSaveState();
	extern __device__ void Random_PrngRestoreState();
	extern __device__ void Random_PrngResetState();
	extern __device__ int Bitvec_BuiltinTest(int size, int *ops);
	__device__ RC Main::TestControl(TESTCTRL op, va_list *args)
	{
		int rc = 0;
#ifndef OMIT_BUILTIN_TEST
		switch (op)
		{
		case TESTCTRL_PRNG_SAVE: {
			// Save the current state of the PRNG.
			Random_PrngSaveState();
			break; }
		case TESTCTRL_PRNG_RESTORE: {
			// Restore the state of the PRNG to the last state saved using PRNG_SAVE.  If PRNG_SAVE has never before been called, then
			// this verb acts like PRNG_RESET.
			Random_PrngRestoreState();
			break; }
		case TESTCTRL_PRNG_RESET: {
			// Reset the PRNG back to its uninitialized state.  The next call to sqlite3_randomness() will reseed the PRNG using a single call
			// to the xRandomness method of the default VFS.
			Random_PrngResetState();
			break; }
		case TESTCTRL_BITVEC_TEST: {
			// sqlite3_test_control(BITVEC_TEST, size, program)
			//
			// Run a test against a Bitvec object of size.  The program argument is an array of integers that defines the test.  Return -1 on a
			// memory allocation error, 0 on success, or non-zero for an error. See the sqlite3BitvecBuiltinTest() for additional information.
			int sz = va_arg(args, int);
			int *progs = va_arg(args, int*);
			rc = Bitvec_BuiltinTest(sz, progs);
			break; }
		case TESTCTRL_BENIGN_MALLOC_HOOKS: {
			// sqlite3_test_control(BENIGN_MALLOC_HOOKS, xBegin, xEnd)
			//
			// Register hooks to call to indicate which malloc() failures are benign.
			typedef void (*action)();
			action benignBegin = va_arg(args, action);
			action benignEnd = va_arg(args, action);
			_benignalloc_hook(benignBegin, benignEnd);
			break; }
		case TESTCTRL_PENDING_BYTE: {
			// sqlite3_test_control(SQLITE_TESTCTRL_PENDING_BYTE, unsigned int X)
			//
			// Set the PENDING byte to the value in the argument, if X>0. Make no changes if X==0.  Return the value of the pending byte
			// as it existing before this routine was called.
			//
			// IMPORTANT:  Changing the PENDING byte from 0x40000000 results in an incompatible database file format.  Changing the PENDING byte
			// while any database connection is open results in undefined and dileterious behavior.
			rc = PENDING_BYTE;
#ifndef OMIT_WSD
			{
				//uint32 newVal = va_arg(args, uint32);
				//if (newVal) Pager::PendingByte = newVal;
			}
#endif
			break; }
		case TESTCTRL_ASSERT: {
			// sqlite3_test_control(SQLITE_TESTCTRL_ASSERT, int X)
			//
			// This action provides a run-time test to see whether or not assert() was enabled at compile-time.  If X is true and assert()
			// is enabled, then the return value is true.  If X is true and assert() is disabled, then the return value is zero.  If X is
			// false and assert() is enabled, then the assertion fires and the process aborts.  If X is false and assert() is disabled, then the
			// return value is zero.
			volatile int x = 0;
			_assert((x = va_arg(args, int)) != 0);
			rc = x;
			break; }
		case TESTCTRL_ALWAYS: {
			// sqlite3_test_control(SQLITE_TESTCTRL_ALWAYS, int X)
			//
			// This action provides a run-time test to see how the ALWAYS and NEVER macros were defined at compile-time.
			//
			// The return value is ALWAYS(X).  
			//
			// The recommended test is X==2.  If the return value is 2, that means ALWAYS() and NEVER() are both no-op pass-through macros, which is the
			// default setting.  If the return value is 1, then ALWAYS() is either hard-coded to true or else it asserts if its argument is false.
			// The first behavior (hard-coded to true) is the case if SQLITE_TESTCTRL_ASSERT shows that assert() is disabled and the second
			// behavior (assert if the argument to ALWAYS() is false) is the case if SQLITE_TESTCTRL_ASSERT shows that assert() is enabled.
			//
			// The run-time test procedure might look something like this:
			//
			//    if( sqlite3_test_control(SQLITE_TESTCTRL_ALWAYS, 2)==2 ){
			//      // ALWAYS() and NEVER() are no-op pass-through macros
			//    }else if( sqlite3_test_control(SQLITE_TESTCTRL_ASSERT, 1) ){
			//      // ALWAYS(x) asserts that x is true. NEVER(x) asserts x is false.
			//    }else{
			//      // ALWAYS(x) is a constant 1.  NEVER(x) is a constant 0.
			//    }
			int x = va_arg(args, int);
			rc = _ALWAYS(x);
			break; }
		case TESTCTRL_RESERVE: {
			// sqlite3_test_control(SQLITE_TESTCTRL_RESERVE, sqlite3 *ctx, int N)
			//
			// Set the nReserve size to N for the main database on the database connection ctx.
			Context *ctx = va_arg(args, Context*);
			int x = va_arg(args, int);
			MutexEx::Enter(ctx->Mutex);
			ctx->DBs[0].Bt->SetPageSize(0, x, false);
			MutexEx::Leave(ctx->Mutex);
			break; }
		case TESTCTRL_OPTIMIZATIONS: {
			// sqlite3_test_control(SQLITE_TESTCTRL_OPTIMIZATIONS, sqlite3 *ctx, int N)
			//
			// Enable or disable various optimizations for testing purposes.  The argument N is a bitmask of optimizations to be disabled.  For normal
			// operation N should be 0.  The idea is that a test program (like the SQL Logic Test or SLT test module) can run the same SQL multiple times
			// with various optimizations disabled to verify that the same answer is obtained in every case.
			Context *ctx = va_arg(args, Context*);
			ctx->OptFlags = (OPTFLAG)(va_arg(args, int) & 0xffff);
			break; }
#ifdef N_KEYWORD
		case TESTCTRL_ISKEYWORD: {
			// sqlite3_test_control(SQLITE_TESTCTRL_ISKEYWORD, const char *zWord)
			//
			// If zWord is a keyword recognized by the parser, then return the number of keywords.  Or if zWord is not a keyword, return 0.
			// 
			// This test feature is only available in the amalgamation since the SQLITE_N_KEYWORD macro is not defined in this file if SQLite
			// is built using separate source files.
			const char *word = va_arg(args, const char*);
			int n = _strlen30(word);
			rc = (KeywordCode((uint8 *)word, n) != TK_ID ? N_KEYWORD : 0);
			break; }
#endif 
		case TESTCTRL_SCRATCHMALLOC: {
			// sqlite3_test_control(SQLITE_TESTCTRL_SCRATCHMALLOC, sz, &pNew, pFree);
			//
			// Pass pFree into sqlite3ScratchFree(). If sz>0 then allocate a scratch buffer into pNew.
			int size = va_arg(args, int);
			void **new_ = va_arg(args, void**);
			void *free = va_arg(args, void*);
			if (size) *new_ = _stackalloc(nullptr, size, false);
			_stackfree(nullptr, free);
			break; }
		case TESTCTRL_LOCALTIME_FAULT: {
			// sqlite3_test_control(SQLITE_TESTCTRL_LOCALTIME_FAULT, int onoff);
			//
			// If parameter onoff is non-zero, configure the wrappers so that all subsequent calls to localtime() and variants fail. If onoff is zero,
			// undo this setting.
			SysEx_GlobalStatics.LocaltimeFault = va_arg(args, bool);
			break; }
#if defined(ENABLE_TREE_EXPLAIN)
		case TESTCTRL_EXPLAIN_STMT: {
			// sqlite3_test_control(SQLITE_TESTCTRL_EXPLAIN_STMT,sqlite3_stmt*,const char**);
			//
			// If compiled with SQLITE_ENABLE_TREE_EXPLAIN, each sqlite3_stmt holds a string that describes the optimized parse tree.  This test-control
			// returns a pointer to that string.
			Vdbe *stmt = va_arg(args, Vdbe*);
			const char **r = va_arg(args, const char**);
			*r = Vdbe::Explanation(stmt);
			break;
									}
#endif
		}
#endif
		return (RC)rc;
	}
#pragma endregion

	__device__ Btree *Main::DbNameToBtree(Context *ctx, const char *dbName)
	{
		for (int i = 0; i < ctx->DBs.length; i++)
			if (ctx->DBs[i].Bt && (!dbName || !_strcmp(dbName, ctx->DBs[i].Name)))
				return ctx->DBs[i].Bt;
		return nullptr;
	}

	__device__ const char *Main::CtxFilename(Context *ctx, const char *dbName)
	{
		Btree *bt = DbNameToBtree(ctx, dbName);
		return (bt ? bt->get_Filename() : nullptr);
	}

	__device__ int Main::CtxReadonly(Context *ctx, const char *dbName)
	{
		Btree *bt = DbNameToBtree(ctx, dbName);
		return (bt ? bt->get_Pager()->get_Readonly() : -1);
	}
}