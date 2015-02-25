#include "VdbeInt.cu.h"

namespace Core
{
	__device__ static void CorruptSchema(InitData *data, const char *obj, const char *extra)
	{
		Context *ctx = data->Ctx;
		if (!ctx->MallocFailed && (ctx->Flags & Context::FLAG_RecoveryMode) == 0)
		{
			if (!obj) obj = "?";
			_setstring(data->ErrMsg, ctx, "malformed database schema (%s)", obj);
			if (extra)
				*data->ErrMsg = _mtagappendf(ctx, *data->ErrMsg, "%s - %s", *data->ErrMsg, extra);
		}
		data->RC = (ctx->MallocFailed ? RC_NOMEM : SysEx_CORRUPT_BKPT);
	}

	__device__ bool Prepare::InitCallback(void *init, int argc, char **argv, char **notUsed1)
	{
		InitData *data = (InitData *)init;
		Context *ctx = data->Ctx;
		int db = data->Db;

		_assert(argc == 3);
		_assert(MutexEx::Held(ctx->Mutex));
		DbClearProperty(ctx, db, SCHEMA_Empty);
		if (ctx->MallocFailed)
		{
			CorruptSchema(data, argv[0], nullptr);
			return true;
		}

		_assert(db >= 0 && db < ctx->DBs.length);
		if (argv == nullptr) return false; // Might happen if EMPTY_RESULT_CALLBACKS are on
		if (argv[1] == 0)
			CorruptSchema(data, argv[0], nullptr);
		else if (argv[2] && argv[2][0])
		{
			// Call the parser to process a CREATE TABLE, INDEX or VIEW. But because ctx->init.busy is set to 1, no VDBE code is generated
			// or executed.  All the parser does is build the internal data structures that describe the table, index, or view.
			_assert(ctx->Init.Busy);
			ctx->Init.DB = db;
			ctx->Init.NewTid = ConvertEx::Atoi(argv[1]);
			ctx->Init.OrphanTrigger = false;
			Vdbe *stmt;
#if _DEBUG
			int rcp = Prepare_(ctx, argv[2], -1, &stmt, nullptr);
#else
			Prepare(ctx, argv[2], -1, &stmt, nullptr);
#endif
			RC rc = ctx->ErrCode;
			_assert((rc&0xFF) == (rcp&0xFF));
			ctx->Init.DB = 0;
			if (rc != RC_OK)
			{
				if (ctx->Init.OrphanTrigger)
					_assert(db == 1);
				else
				{
					data->RC = rc;
					if (rc == RC_NOMEM)
						ctx->MallocFailed = true;
					else if (rc != RC_INTERRUPT && (rc&0xFF) != RC_LOCKED)
						CorruptSchema(data, argv[0], Main::ErrMsg(ctx));
				}
			}
			stmt->Finalize();
		}
		else if (argv[0] == 0)
			CorruptSchema(data, nullptr, nullptr);
		else
		{
			// If the SQL column is blank it means this is an index that was created to be the PRIMARY KEY or to fulfill a UNIQUE
			// constraint for a CREATE TABLE.  The index should have already been created when we processed the CREATE TABLE.  All we have
			// to do here is record the root page number for that index.
			Index *index = Parse::FindIndex(ctx, argv[0], ctx->DBs[db].Name);
			if (!index)
			{
				// This can occur if there exists an index on a TEMP table which has the same name as another index on a permanent index.  Since
				// the permanent table is hidden by the TEMP table, we can also safely ignore the index on the permanent table.
				// Do Nothing
			}
			else if (!ConvertEx::Atoi(argv[1], &index->Id))
				CorruptSchema(data, argv[0], "invalid rootpage");
		}
		return false;
	}

	// The master database table has a structure like this
	__constant__ static const char master_schema[] = 
		"CREATE TABLE sqlite_master(\n"
		"  type text,\n"
		"  name text,\n"
		"  tbl_name text,\n"
		"  rootpage integer,\n"
		"  sql text\n"
		")";
#ifndef OMIT_TEMPDB
	__constant__ static const char temp_master_schema[] = 
		"CREATE TEMP TABLE sqlite_temp_master(\n"
		"  type text,\n"
		"  name text,\n"
		"  tbl_name text,\n"
		"  rootpage integer,\n"
		"  sql text\n"
		")";
#else
#define temp_master_schema nullptr
#endif

	__device__ RC Prepare::InitOne(Context *ctx, int db, char **errMsg)
	{
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(ctx->DBs[db].Schema);
		_assert(MutexEx::Held(ctx->Mutex));
		_assert(db == 1 || ctx->DBs[db].Bt->HoldsMutex());
		RC rc;
		int i;

		// zMasterSchema and zInitScript are set to point at the master schema and initialisation script appropriate for the database being
		// initialized. zMasterName is the name of the master table.
		char const *masterSchema = (!E_OMIT_TEMPDB && db == 1 ? temp_master_schema : master_schema);
		char const *masterName = SCHEMA_TABLE(db);

		// Construct the schema tables.
		char const *args[4];
		args[0] = masterName;
		args[1] = "1";
		args[2] = masterSchema;
		args[3] = nullptr;
		InitData initData;
		initData.Ctx = ctx;
		initData.Db = db;
		initData.RC = RC_OK;
		initData.ErrMsg = errMsg;
		InitCallback(&initData, 3, (char **)args, nullptr);
		if (initData.RC)
		{
			rc = initData.RC;
			goto error_out;
		}
		Table *table = Parse::FindTable(ctx, masterName, ctx->DBs[db].Name);
		if (_ALWAYS(table))
			table->TabFlags |= TF_Readonly;

		// Create a cursor to hold the database open
		Context::DB *dbAsObj = &ctx->DBs[db];
		if (!dbAsObj->Bt)
		{
			if (!E_OMIT_TEMPDB && _ALWAYS(db == 1))
				DbSetProperty(ctx, 1, SCHEMA_SchemaLoaded);
			return RC_OK;
		}

		// If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
		// will be closed before this function returns.
		bool openedTransaction = false;
		dbAsObj->Bt->Enter();
		if (!dbAsObj->Bt->IsInReadTrans())
		{
			rc = dbAsObj->Bt->BeginTrans(0);
			if (rc != RC_OK)
			{
				_setstring(errMsg, ctx, "%s", Main::ErrStr(rc));
				goto initone_error_out;
			}
			openedTransaction = true;
		}

		// Get the database meta information.
		// Meta values are as follows:
		//    meta[0]   Schema cookie.  Changes with each schema change.
		//    meta[1]   File format of schema layer.
		//    meta[2]   Size of the page cache.
		//    meta[3]   Largest rootpage (auto/incr_vacuum mode)
		//    meta[4]   Db text encoding. 1:UTF-8 2:UTF-16LE 3:UTF-16BE
		//    meta[5]   User version
		//    meta[6]   Incremental vacuum mode
		//    meta[7]   unused
		//    meta[8]   unused
		//    meta[9]   unused
		// Note: The #defined SQLITE_UTF* symbols in sqliteInt.h correspond to the possible values of meta[4].
		int meta[5];
		for (i = 0; i < _lengthof(meta); i++)
			dbAsObj->Bt->GetMeta((Btree::META)(i+1), (uint32 *)&meta[i]);
		dbAsObj->Schema->SchemaCookie = meta[Btree::META_SCHEMA_VERSION-1];

		// If opening a non-empty database, check the text encoding. For the main database, set sqlite3.enc to the encoding of the main database.
		// For an attached ctx, it is an error if the encoding is not the same as sqlite3.enc.
		if (meta[Btree::META_TEXT_ENCODING-1]) // text encoding
		{  
			if (db == 0)
			{
#ifndef OMIT_UTF16
				// If opening the main database, set ENC(ctx).
				TEXTENCODE encoding = (TEXTENCODE)(meta[Btree::META_TEXT_ENCODING-1] & 3);
				if (encoding == 0) encoding = TEXTENCODE_UTF8;
				CTXENCODE(ctx) = encoding;
#else
				CTXENCODE(ctx) = TEXTENCODE_UTF8;
#endif
			}
			else
			{
				// If opening an attached database, the encoding much match CTXENCODE(ctx)
				if (meta[Btree::META_TEXT_ENCODING-1] != CTXENCODE(ctx))
				{
					_setstring(errMsg, ctx, "attached databases must use the same text encoding as main database");
					rc = RC_ERROR;
					goto initone_error_out;
				}
			}
		}
		else
			DbSetProperty(ctx, db, SCHEMA_Empty);
		dbAsObj->Schema->Encode = CTXENCODE(ctx);

		if (dbAsObj->Schema->CacheSize == 0)
		{
			dbAsObj->Schema->CacheSize = DEFAULT_CACHE_SIZE;
			dbAsObj->Bt->SetCacheSize(dbAsObj->Schema->CacheSize);
		}

		// file_format==1    Version 3.0.0.
		// file_format==2    Version 3.1.3.  // ALTER TABLE ADD COLUMN
		// file_format==3    Version 3.1.4.  // ditto but with non-NULL defaults
		// file_format==4    Version 3.3.0.  // DESC indices.  Boolean constants
		dbAsObj->Schema->FileFormat = (uint8)meta[Btree::META_FILE_FORMAT-1];
		if (dbAsObj->Schema->FileFormat == 0)
			dbAsObj->Schema->FileFormat = 1;
		if (dbAsObj->Schema->FileFormat > MAX_FILE_FORMAT)
		{
			_setstring(errMsg, ctx, "unsupported file format");
			rc = RC_ERROR;
			goto initone_error_out;
		}

		// Ticket #2804:  When we open a database in the newer file format, clear the legacy_file_format pragma flag so that a VACUUM will
		// not downgrade the database and thus invalidate any descending indices that the user might have created.
		if (db == 0 && meta[Btree::META_FILE_FORMAT-1] >= 4)
			ctx->Flags &= ~Context::FLAG_LegacyFileFmt;

		// Read the schema information out of the schema tables
		_assert(ctx->Init.Busy);
		{
			char *sql = _mtagprintf(ctx,  "SELECT name, rootpage, sql FROM '%q'.%s ORDER BY rowid", ctx->DBs[db].Name, masterName);
#ifndef OMIT_AUTHORIZATION
			{
				ARC (*auth)(void*,int,const char*,const char*,const char*,const char*) = ctx->Auth;
				ctx->Auth = nullptr;
#endif
				rc = Main::Exec(ctx, sql, InitCallback, &initData, 0);
#ifndef OMIT_AUTHORIZATION
				ctx->Auth = auth;
			}
#endif
			if (rc == RC_OK) rc = initData.RC;
			_tagfree(ctx, sql);
#ifndef OMIT_ANALYZE
			if (rc == RC_OK)
				Analyze::AnalysisLoad(ctx, db);
#endif
		}
		if (ctx->MallocFailed)
		{
			rc = RC_NOMEM;
			Parse::ResetAllSchemasOfConnection(ctx);
		}
		if (rc == RC_OK || (ctx->Flags&Context::FLAG_RecoveryMode))
		{
			// Black magic: If the SQLITE_RecoveryMode flag is set, then consider the schema loaded, even if errors occurred. In this situation the 
			// current sqlite3_prepare() operation will fail, but the following one will attempt to compile the supplied statement against whatever subset
			// of the schema was loaded before the error occurred. The primary purpose of this is to allow access to the sqlite_master table
			// even when its contents have been corrupted.
			DbSetProperty(ctx, db, SCHEMA_SchemaLoaded);
			rc = RC_OK;
		}

		// Jump here for an error that occurs after successfully allocating curMain and calling sqlite3BtreeEnter(). For an error that occurs
		// before that point, jump to error_out.
initone_error_out:
		if (openedTransaction)
			dbAsObj->Bt->Commit();
		dbAsObj->Bt->Leave();

error_out:
		if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM)
			ctx->MallocFailed = true;
		return rc;
	}

	__device__ RC Prepare::Init(Context *ctx, char **errMsg)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		RC rc = RC_OK;
		ctx->Init.Busy = true;
		for (int i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
		{
			if (DbHasProperty(ctx, i, SCHEMA_SchemaLoaded) || i == 1) continue;
			rc = InitOne(ctx, i, errMsg);
			if (rc)
				Parse::ResetOneSchema(ctx, i);
		}

		// Once all the other databases have been initialized, load the schema for the TEMP database. This is loaded last, as the TEMP database
		// schema may contain references to objects in other databases.
#ifndef OMIT_TEMPDB
		if (rc == RC_OK && _ALWAYS(ctx->DBs.length > 1) && !DbHasProperty(ctx, 1, SCHEMA_SchemaLoaded))
		{
			rc = InitOne(ctx, 1, errMsg);
			if (rc)
				Parse::ResetOneSchema(ctx, 1);
		}
#endif

		bool commitInternal = !(ctx->Flags & Context::FLAG_InternChanges);
		ctx->Init.Busy = false;
		if (rc == RC_OK && commitInternal)
			Parse::CommitInternalChanges(ctx);
		return rc; 
	}

	__device__ RC Prepare::ReadSchema(Parse *parse)
	{
		RC rc = RC_OK;
		Context *ctx = parse->Ctx;
		_assert(MutexEx::Held(ctx->Mutex));
		if (!ctx->Init.Busy)
			rc = Init(ctx, &parse->ErrMsg);
		if (rc != RC_OK)
		{
			parse->RC = rc;
			parse->Errs++;
		}
		return rc;
	}

	__device__ static void SchemaIsValid(Parse *parse)
	{
		Context *ctx = parse->Ctx;
		_assert(parse->CheckSchema);
		_assert(MutexEx::Held(ctx->Mutex));
		for (int db = 0; db <ctx->DBs.length; db++)
		{
			bool openedTransaction = false; // True if a transaction is opened
			Btree *bt = ctx->DBs[db].Bt; // Btree database to read cookie from
			if (!bt) continue;

			// If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
			// will be closed immediately after reading the meta-value.
			if (!bt->IsInReadTrans())
			{
				RC rc = bt->BeginTrans(0);
				if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM)
					ctx->MallocFailed = true;
				if (rc != RC_OK) return;
				openedTransaction = true;
			}

			// Read the schema cookie from the database. If it does not match the value stored as part of the in-memory schema representation,
			// set Parse.rc to SQLITE_SCHEMA.
			uint cookie;
			bt->GetMeta(Btree::META_SCHEMA_VERSION, (uint32 *)&cookie);
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			if (cookie != ctx->DBs[db].Schema->SchemaCookie)
			{
				Parse::ResetOneSchema(ctx, db);
				parse->RC = RC_SCHEMA;
			}

			// Close the transaction, if one was opened.
			if (openedTransaction)
				bt->Commit();
		}
	}

	__device__ int Prepare::SchemaToIndex(Context *ctx, Schema *schema)
	{
		// If pSchema is NULL, then return -1000000. This happens when code in expr.c is trying to resolve a reference to a transient table (i.e. one
		// created by a sub-select). In this case the return value of this function should never be used.
		//
		// We return -1000000 instead of the more usual -1 simply because using -1000000 as the incorrect index into ctx->aDb[] is much 
		// more likely to cause a segfault than -1 (of course there are assert() statements too, but it never hurts to play the odds).
		_assert(MutexEx::Held(ctx->Mutex));
		int i = -1000000;
		if (schema)
		{
			for (i = 0; _ALWAYS(i < ctx->DBs.length); i++)
				if (ctx->DBs[i].Schema == schema)
					break;
			_assert(i >= 0 && i < ctx->DBs.length);
		}
		return i;
	}

#ifndef OMIT_EXPLAIN
	__constant__ static const char * const _colName[] = {
		"addr", "opcode", "p1", "p2", "p3", "p4", "p5", "comment",
		"selectid", "order", "from", "detail" };
#endif
	__device__ RC Prepare::Prepare_(Context *ctx, const char *sql, int bytes, bool isPrepareV2, Vdbe *reprepare, Vdbe **stmtOut, const char **tailOut)
	{
		char *errMsg = nullptr; // Error message
		int i;
		RC rc = RC_OK;

		// Allocate the parsing context
		Parse *parse = (Parse *)_stackalloc(ctx, sizeof(*parse), true); // Parsing context
		if (!parse)
		{
			rc = RC_NOMEM;
			goto end_prepare;
		}
		parse->Reprepare = reprepare;
		_assert(stmtOut && *stmtOut == nullptr);
		_assert(!ctx->MallocFailed);
		_assert(MutexEx::Held(ctx->Mutex));

		// Check to verify that it is possible to get a read lock on all database schemas.  The inability to get a read lock indicates that
		// some other database connection is holding a write-lock, which in turn means that the other connection has made uncommitted changes
		// to the schema.
		//
		// Were we to proceed and prepare the statement against the uncommitted schema changes and if those schema changes are subsequently rolled
		// back and different changes are made in their place, then when this prepared statement goes to run the schema cookie would fail to detect
		// the schema change.  Disaster would follow.
		//
		// This thread is currently holding mutexes on all Btrees (because of the sqlite3BtreeEnterAll() in sqlite3LockAndPrepare()) so it
		// is not possible for another thread to start a new schema change while this routine is running.  Hence, we do not need to hold 
		// locks on the schema, we just need to make sure nobody else is holding them.
		//
		// Note that setting READ_UNCOMMITTED overrides most lock detection, but it does *not* override schema lock detection, so this all still
		// works even if READ_UNCOMMITTED is set.
		for (i = 0; i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
			{
				_assert(bt->HoldsMutex());
				rc = bt->SchemaLocked();
				if (rc)
				{
					const char *dbName = ctx->DBs[i].Name;
					Main::Error(ctx, rc, "database schema is locked: %s", dbName);
					ASSERTCOVERAGE(ctx->Flags & Context::FLAG_ReadUncommitted);
					goto end_prepare;
				}
			}
		}

		VTable::UnlockList(ctx);

		parse->Ctx = ctx;
		parse->QueryLoops = (double)1;
		if (bytes >= 0 && (bytes == 0 || sql[bytes-1] != 0))
		{
			int maxLen = ctx->Limits[LIMIT_SQL_LENGTH];
			ASSERTCOVERAGE(bytes == maxLen);
			ASSERTCOVERAGE(bytes == maxLen+1);
			if (bytes > maxLen)
			{
				Main::Error(ctx, RC_TOOBIG, "statement too long");
				rc = Main::ApiExit(ctx, RC_TOOBIG);
				goto end_prepare;
			}
			char *sqlCopy = _tagstrndup(ctx, sql, bytes);
			if (sqlCopy)
			{
				parse->RunParser(sqlCopy, &errMsg);
				_tagfree(ctx, sqlCopy);
				parse->Tail = &sql[parse->Tail - sqlCopy];
			}
			else
				parse->Tail = &sql[bytes];
		}
		else
			parse->RunParser(sql, &errMsg);
		_assert((int)parse->QueryLoops == 1);

		if (ctx->MallocFailed)
			parse->RC = RC_NOMEM;
		if (parse->RC == RC_DONE) parse->RC = RC_OK;
		if (parse->CheckSchema)
			SchemaIsValid(parse);
		if (ctx->MallocFailed)
			parse->RC = RC_NOMEM;
		if (tailOut)
			*tailOut = parse->Tail;
		rc = parse->RC;

		Vdbe *v = parse->V;
#ifndef OMIT_EXPLAIN
		if (rc == RC_OK && parse->V && parse->Explain)
		{
			int first, max;
			if (parse->Explain == 2)
			{
				v->SetNumCols(4);
				first = 8;
				max = 12;
			}
			else
			{
				v->SetNumCols(8);
				first = 0;
				max = 8;
			}
			for (i = first; i < max; i++)
				v->SetColName(i-first, COLNAME_NAME, _colName[i], DESTRUCTOR_STATIC);
		}
#endif

		_assert(!ctx->Init.Busy || !isPrepareV2);
		if (!ctx->Init.Busy)
			Vdbe::SetSql(v, sql, (int)(parse->Tail - sql), isPrepareV2);
		if (v && (rc != RC_OK || ctx->MallocFailed))
		{
			v->Finalize();
			_assert(!(*stmtOut));
		}
		else
			*stmtOut = v;

		if (errMsg)
		{
			Main::Error(ctx, rc, "%s", errMsg);
			_tagfree(ctx, errMsg);
		}
		else
			Main::Error(ctx, rc, nullptr);

		// Delete any TriggerPrg structures allocated while parsing this statement.
		while (parse->TriggerPrg)
		{
			TriggerPrg *t = parse->TriggerPrg;
			parse->TriggerPrg = t->Next;
			_tagfree(ctx, t);
		}

end_prepare:
		_stackfree(ctx, parse);
		rc = Main::ApiExit(ctx, rc);
		_assert((rc&ctx->ErrMask) == rc);
		return rc;
	}

	__device__ RC Prepare::LockAndPrepare(Context *ctx, const char *sql, int bytes, bool isPrepareV2, Vdbe *reprepare, Vdbe **stmtOut, const char **tailOut)
	{
		_assert(stmtOut != 0);
		*stmtOut = nullptr;
		if (!Main::SafetyCheckOk(ctx))
			return SysEx_MISUSE_BKPT;
		MutexEx::Enter(ctx->Mutex);
		Btree::EnterAll(ctx);
		RC rc = Prepare_(ctx, sql, bytes, isPrepareV2, reprepare, stmtOut, tailOut);
		if (rc == RC_SCHEMA)
		{
			(*stmtOut)->Finalize();
			rc = Prepare_(ctx, sql, bytes, isPrepareV2, reprepare, stmtOut, tailOut);
		}
		Btree::LeaveAll(ctx);
		MutexEx::Leave(ctx->Mutex);
		_assert(rc == RC_OK || !*stmtOut);
		return rc;
	}

	__device__ RC Prepare::Reprepare(Vdbe *p)
	{
		Context *ctx = p->Ctx;
		_assert(MutexEx::Held(ctx->Mutex));
		const char *sql = Vdbe::Sql(p);
		_assert(sql != nullptr); // Reprepare only called for prepare_v2() statements
		Vdbe *newVdbe;
		RC rc = LockAndPrepare(ctx, sql, -1, false, p, &newVdbe, nullptr);
		if (rc)
		{
			if (rc == RC_NOMEM)
				ctx->MallocFailed = true;
			_assert(newVdbe == nullptr);
			return rc;
		}
		else
			_assert(newVdbe != nullptr);
		Vdbe::Swap(newVdbe, p);
		Vdbe::TransferBindings(newVdbe, p);
		newVdbe->ResetStepResult();
		newVdbe->Finalize();
		return RC_OK;
	}

	__device__ RC Prepare::Prepare_(Context *ctx, const char *sql, int bytes, Vdbe **stmtOut, const char **tailOut)
	{
		RC rc = LockAndPrepare(ctx, sql, bytes, false, nullptr, stmtOut, tailOut);
		_assert(rc == RC_OK || stmtOut || !*stmtOut);  // VERIFY: F13021
		return rc;
	}

	__device__ RC Prepare::Prepare_v2(Context *ctx, const char *sql, int bytes, Vdbe **stmtOut, const char **tailOut)
	{
		RC rc = LockAndPrepare(ctx, sql, bytes, true, nullptr, stmtOut, tailOut);
		_assert(rc == RC_OK || !stmtOut || !*stmtOut); // VERIFY: F13021
		return rc;
	}

#ifndef OMIT_UTF16
	__device__ RC Prepare::Prepare16(Context *ctx, const void *sql, int bytes, bool isPrepareV2, Vdbe **stmtOut, const void **tailOut)
	{
		// This function currently works by first transforming the UTF-16 encoded string to UTF-8, then invoking sqlite3_prepare(). The
		// tricky bit is figuring out the pointer to return in *pzTail.
		_assert(stmtOut);
		*stmtOut = nullptr;
		if (!Main::SafetyCheckOk(ctx))
			return SysEx_MISUSE_BKPT;
		MutexEx::Enter(ctx->Mutex);
		RC rc = RC_OK;
		const char *tail8 = nullptr;
		char *sql8 = Vdbe::Utf16to8(ctx, sql, bytes, TEXTENCODE_UTF16NATIVE);
		if (sql8)
			rc = LockAndPrepare(ctx, sql8, -1, isPrepareV2, 0, stmtOut, &tail8);
		if (tail8 && tailOut)
		{
			// If sqlite3_prepare returns a tail pointer, we calculate the equivalent pointer into the UTF-16 string by counting the unicode
			// characters between zSql8 and zTail8, and then returning a pointer the same number of characters into the UTF-16 string.
			int charsParsed = _utf8charlength(sql8, (int)(tail8 - sql8));
			*tailOut = (uint8 *)sql + _utf16bytelength(sql, charsParsed);
		}
		_tagfree(ctx, sql8); 
		rc = Main::ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ RC Prepare::Prepare16(Context *ctx, const void *sql, int bytes, Vdbe **stmtOut, const void **tailOut)
	{
		RC rc = Prepare16(ctx, sql, bytes, 0, stmtOut, tailOut);
		_assert(rc == RC_OK || !stmtOut || !*stmtOut); // VERIFY: F13021
		return rc;
	}

	__device__ RC Prepare::Prepare16_v2(Context *ctx, const void *sql, int bytes, Vdbe **stmtOut, const void **tailOut)
	{
		RC rc = Prepare16(ctx, sql, bytes, 1, stmtOut, tailOut);
		_assert(rc == RC_OK || !stmtOut || !*stmtOut); // VERIFY: F13021
		return rc;
	}
#endif
}