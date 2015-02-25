// pragma.c
#include "..\VdbeInt.cu.h"

namespace Core {
	__device__ extern char *g_temp_directory;
	__device__ extern char *g_data_directory;
#ifndef OMIT_COMPILEOPTION_DIAGS
	__device__ extern const char *CompileTimeGet(int id);
#endif
}

namespace Core { namespace Command
{
	// moved to ConvertEx
	//__device__ static uint8 GetSafetyLevel(const char *z, int omitFull, uint8 dflt);
	//__device__ bool ConvertEx::GetBoolean(const char *z, uint8 dflt);

#if !defined(OMIT_PRAGMA)

	__device__ static IPager::LOCKINGMODE GetLockingMode(const char *z)
	{
		if (z)
		{
			if (!_strcmp(z, "exclusive")) return IPager::LOCKINGMODE_EXCLUSIVE;
			if (!_strcmp(z, "normal")) return IPager::LOCKINGMODE_NORMAL;
		}
		return IPager::LOCKINGMODE_QUERY;
	}

#ifndef OMIT_AUTOVACUUM
	__device__ static Btree::AUTOVACUUM GetAutoVacuum(const char *z)
	{
		if (!_strcmp(z, "none")) return Btree::AUTOVACUUM_NONE;
		if (!_strcmp(z, "full")) return Btree::AUTOVACUUM_FULL;
		if (!_strcmp(z, "incremental")) return Btree::AUTOVACUUM_INCR;
		int i = ConvertEx::Atoi(z);
		return (Btree::AUTOVACUUM)(i>=0 && i<=2 ? i : 0);
	}
#endif

#ifndef OMIT_PAGER_PRAGMAS
	__device__ static int GetTempStore(const char *z)
	{
		if (z[0]>='0' && z[0]<='2') return z[0] - '0';
		else if (!_strcmp(z, "file")) return 1;
		else if (!_strcmp(z, "memory")) return 2;
		else return 0;
	}
#endif

#ifndef OMIT_PAGER_PRAGMAS
	__device__ static RC InvalidateTempStorage(Parse *parse)
	{
		Context *ctx = parse->Ctx;
		if (ctx->DBs[1].Bt != nullptr)
		{
			if (!ctx->AutoCommit || ctx->DBs[1].Bt->IsInReadTrans())
			{
				parse->ErrorMsg("temporary storage cannot be changed from within a transaction");
				return RC_ERROR;
			}
			ctx->DBs[1].Bt->Close();
			ctx->DBs[1].Bt = nullptr;
			Parse::ResetAllSchemasOfConnection(ctx);
		}
		return RC_OK;
	}
#endif

#ifndef OMIT_PAGER_PRAGMAS
	__device__ static RC ChangeTempStorage(Parse *parse, const char *storageType)
	{
		int ts = GetTempStore(storageType);
		Context *ctx = parse->Ctx;
		if (ctx->TempStore == ts) return RC_OK;
		if (InvalidateTempStorage(parse) != RC_OK)
			return RC_ERROR;
		ctx->TempStore = (uint8)ts;
		return RC_OK;
	}
#endif

	__device__ static void ReturnSingleInt(Parse *parse, const char *label, int64 value)
	{
		Vdbe *v = parse->GetVdbe();
		int mem = ++parse->Mems;
		int64 *i64 = (int64 *)_tagalloc(parse->Ctx, sizeof(value));
		if (i64)
			_memcpy(i64, &value, sizeof(value));
		v->AddOp4(OP_Int64, 0, mem, 0, (char *)i64, Vdbe::P4T_INT64);
		v->SetNumCols(1);
		v->SetColName(0, COLNAME_NAME, label, DESTRUCTOR_STATIC);
		v->AddOp2(OP_ResultRow, mem, 1);
	}

#ifndef OMIT_FLAG_PRAGMAS
	struct sPragmaType
	{
		const char *Name; // Name of the pragma
		Context::FLAG Mask; // Mask for the db->flags value
	};
#if __CUDACC__
	__constant__ static sPragmaType _pragmas[19];
	static const sPragmaType h_pragmas[19] =
#else
	static const sPragmaType _pragmas[19] =
#endif
	{
		{"full_column_names",        Context::FLAG_FullColNames},
		{"short_column_names",       Context::FLAG_ShortColNames},
		{"count_changes",            Context::FLAG_CountRows},
		{"empty_result_callbacks",   Context::FLAG_NullCallback},
		{"legacy_file_format",       Context::FLAG_LegacyFileFmt},
		{"fullfsync",                Context::FLAG_FullFSync},
		{"checkpoint_fullfsync",     Context::FLAG_CkptFullFSync},
		{"reverse_unordered_selects", Context::FLAG_ReverseOrder},
#ifndef OMIT_AUTOMATIC_INDEX
		{"automatic_index",          Context::FLAG_AutoIndex},
#endif
#ifdef _DEBUG
		{"sql_trace",                Context::FLAG_SqlTrace},
		{"vdbe_listing",             Context::FLAG_VdbeListing},
		{"vdbe_trace",               Context::FLAG_VdbeTrace},
		{"vdbe_addoptrace",          Context::FLAG_VdbeAddopTrace},
		{"vdbe_debug",               Context::FLAG_SqlTrace|Context::FLAG_VdbeListing|Context::FLAG_VdbeTrace},
#endif
#ifndef OMIT_CHECK
		{"ignore_check_constraints", Context::FLAG_IgnoreChecks},
#endif
		// The following is VERY experimental
		{"writable_schema",          Context::FLAG_WriteSchema|Context::FLAG_RecoveryMode},

		// TODO: Maybe it shouldn't be possible to change the ReadUncommitted flag if there are any active statements.
		{"read_uncommitted",         Context::FLAG_ReadUncommitted},
		{"recursive_triggers",       Context::FLAG_RecTriggers},

		// This flag may only be set if both foreign-key and trigger support are present in the build.
#if !defined(OMIT_FOREIGN_KEY) && !defined(OMIT_TRIGGER)
		{"foreign_keys",             Context::FLAG_ForeignKeys},
#endif
	};

	__device__ static bool FlagPragma(Parse *parse, const char *left, const char *right)
	{
		int i;
		const sPragmaType *p;
		for (i = 0, p = _pragmas; i < _lengthof(_pragmas); i++, p++)
		{
			if (!_strcmp(left, p->Name))
			{
				Context *ctx = parse->Ctx;
				Vdbe *v = parse->GetVdbe();
				_assert(v != nullptr); // Already allocated by sqlite3Pragma()
				if (_ALWAYS(v))
				{
					if (!right)
						ReturnSingleInt(parse, p->Name, (ctx->Flags & p->Mask) != 0);
					else
					{
						Context::FLAG mask = p->Mask; // Mask of bits to set or clear.
						if (ctx->AutoCommit == 0)
							mask &= ~Context::FLAG_ForeignKeys; // Foreign key support may not be enabled or disabled while not in auto-commit mode.
						if (ConvertEx::GetBoolean(right, 0))
							ctx->Flags |= mask;
						else
							ctx->Flags &= ~mask;
						// Many of the flag-pragmas modify the code generated by the SQL compiler (eg. count_changes). So add an opcode to expire all
						// compiled SQL statements after modifying a pragma value.
						v->AddOp2(OP_Expire, 0, 0);
					}
				}
				return true;
			}
		}
		return false;
	}
#endif

#ifndef OMIT_FOREIGN_KEY
	__device__ static const char *ActionName(OE action)
	{
		switch (action)
		{
		case OE_SetNull: return "SET NULL";
		case OE_SetDflt: return "SET DEFAULT";
		case OE_Cascade: return "CASCADE";
		case OE_Restrict: return "RESTRICT";
		default: _assert(action == OE_None); return "NO ACTION";
		}
	}
#endif

	__constant__ static char *const _modeName[] = {
		"delete", "persist", "off", "truncate", "memory"
#ifndef OMIT_WAL
		, "wal"
#endif
	};
	__device__ const char *Pragma::JournalModename(IPager::JOURNALMODE mode)
	{
		_assert(IPager::JOURNALMODE_DELETE == 0);
		_assert(IPager::JOURNALMODE_PERSIST == 1);
		_assert(IPager::JOURNALMODE_OFF == 2);
		_assert(IPager::JOURNALMODE_TRUNCATE == 3);
		_assert(IPager::JOURNALMODE_JMEMORY == 4);
		_assert(IPager::JOURNALMODE_WAL == 5);
		_assert(mode>=0 && mode<=_lengthof(_modeName));
		if (mode == _lengthof(_modeName)) return nullptr;
		return _modeName[mode];
	}

#if !defined(ENABLE_LOCKING_STYLE)
#if defined(__APPLE__)
#define ENABLE_LOCKING_STYLE 1
#else
#define ENABLE_LOCKING_STYLE 0
#endif
#endif

#ifndef INTEGRITY_CHECK_ERROR_MAX
#define INTEGRITY_CHECK_ERROR_MAX 100
#endif

	__constant__ static const Vdbe::VdbeOpList _getCacheSize[] =
	{
		{OP_Transaction, 0, 0,        0},                         // 0
		{OP_ReadCookie,  0, 1,        Btree::META_DEFAULT_CACHE_SIZE},  // 1
		{OP_IfPos,       1, 7,        0},
		{OP_Integer,     0, 2,        0},
		{OP_Subtract,    1, 2,        1},
		{OP_IfPos,       1, 7,        0},
		{OP_Integer,     0, 1,        0},                         // 6
		{OP_ResultRow,   1, 1,        0},
	};
	__constant__ static const Vdbe::VdbeOpList _setMeta6[] =
	{
		{OP_Transaction,    0,         1,                 0},    // 0
		{OP_ReadCookie,     0,         1,         Btree::META_LARGEST_ROOT_PAGE},
		{OP_If,             1,         0,                 0},    // 2
		{OP_Halt,           RC_OK, OE_Abort,          0},    // 3
		{OP_Integer,        0,         1,                 0},    // 4
		{OP_SetCookie,      0,         Btree::META_INCR_VACUUM, 1},    // 5
	};
	// Code that appears at the end of the integrity check.  If no error messages have been generated, output OK.  Otherwise output the error message
	__constant__ static const Vdbe::VdbeOpList _endCode[] =
	{
		{OP_AddImm,      1, 0,        0},    // 0
		{OP_IfNeg,       1, 0,        0},    // 1
		{OP_String8,     0, 3,        0},    // 2
		{OP_ResultRow,   3, 1,        0},
	};
	__constant__ static const Vdbe::VdbeOpList _idxErr[] =
	{
		{OP_AddImm,      1, -1,  0},
		{OP_String8,     0,  3,  0},    // 1
		{OP_Rowid,       1,  4,  0},
		{OP_String8,     0,  5,  0},    // 3
		{OP_String8,     0,  6,  0},    // 4
		{OP_Concat,      4,  3,  3},
		{OP_Concat,      5,  3,  3},
		{OP_Concat,      6,  3,  3},
		{OP_ResultRow,   3,  1,  0},
		{OP_IfPos,       1,  0,  0},    // 9
		{OP_Halt,        0,  0,  0},
	};
	__constant__ static const Vdbe::VdbeOpList _cntIdx[] =
	{
		{OP_Integer,      0,  3,  0},
		{OP_Rewind,       0,  0,  0},  // 1
		{OP_AddImm,       3,  1,  0},
		{OP_Next,         0,  0,  0},  // 3
		{OP_Eq,           2,  0,  3},  // 4
		{OP_AddImm,       1, -1,  0},
		{OP_String8,      0,  2,  0},  // 6
		{OP_String8,      0,  3,  0},  // 7
		{OP_Concat,       3,  2,  2},
		{OP_ResultRow,    2,  1,  0},
	};
	struct EncodeName
	{
		char *Name;
		TEXTENCODE Encode;
	};
	__constant__ static const EncodeName _encodeNames[] =
	{
		{"UTF8",     TEXTENCODE_UTF8},
		{"UTF-8",    TEXTENCODE_UTF8},  // Must be element [1]
		{"UTF-16le", TEXTENCODE_UTF16LE},  // Must be element [2]
		{"UTF-16be", TEXTENCODE_UTF16BE},  // Must be element [3]
		{"UTF16le",  TEXTENCODE_UTF16LE},
		{"UTF16be",  TEXTENCODE_UTF16BE},
		{"UTF-16",   (TEXTENCODE)0}, // TEXTENCODE_UTF16NATIVE
		{"UTF16",    (TEXTENCODE)0}, // TEXTENCODE_UTF16NATIVE
		{nullptr, (TEXTENCODE)0}
	};
	// Write the specified cookie value
	__constant__ static const Vdbe::VdbeOpList _setCookie[] =
	{
		{OP_Transaction,    0,  1,  0},    // 0
		{OP_Integer,        0,  1,  0},    // 1
		{OP_SetCookie,      0,  0,  1},    // 2
	};
	// Read the specified cookie value
	__constant__ static const Vdbe::VdbeOpList _readCookie[] =
	{
		{OP_Transaction,     0,  0,  0},    // 0
		{OP_ReadCookie,      0,  1,  0},    // 1
		{OP_ResultRow,       1,  1,  0}
	};
	__constant__ static const char *const _lockNames[] =
	{
		"unlocked", "shared", "reserved", "pending", "exclusive"
	};

	__device__ void Pragma::Pragma_(Parse *parse, Token *id1, Token *id2, Token *value, bool minusFlag)
	{
		Context *ctx = parse->Ctx; // The database connection
		Vdbe *v = parse->V = Vdbe::Create(ctx); // Prepared statement
		if (!v) return;
		v->set_RunOnlyOnce();
		parse->Mems = 2;

		// Interpret the [database.] part of the pragma statement. db is the index of the database this pragma is being applied to in db.aDb[].
		Token *id; // Pointer to <id> token
		int db = parse->TwoPartName(id1, id2, &id); // Database index for <database>
		if (db < 0) return;
		Context::DB *dbAsObj = &ctx->DBs[db]; // The specific database being pragmaed

		// If the temp database has been explicitly named as part of the pragma, make sure it is open.
		if (db == 1 && parse->OpenTempDatabase())
			return;

		char *left = Parse::NameFromToken(ctx, id); // Nul-terminated UTF-8 string <id>
		if (!left) return;
		char *right = (minusFlag ? _mtagprintf(ctx, "-%T", value) : Parse::NameFromToken(ctx, value)); // Nul-terminated UTF-8 string <value>, or NULL

		_assert(id2 != nullptr);
		const char *dbName = (id2->length > 0 ? dbAsObj->Name : nullptr); // The database name
		if (Auth::Check(parse, AUTH_PRAGMA, left, right, dbName))
			goto pragma_out;

		// Send an SQLITE_FCNTL_PRAGMA file-control to the underlying VFS connection.  If it returns SQLITE_OK, then assume that the VFS
		// handled the pragma and generate a no-op prepared statement.
		char *fcntls[4]; // Argument to FCNTL_PRAGMA
		fcntls[0] = nullptr;
		fcntls[1] = left;
		fcntls[2] = right;
		fcntls[3] = nullptr;
		ctx->BusyHandler.Busys = 0;
		RC rc = Main::FileControl(ctx, dbName, VFile::FCNTL_PRAGMA, (void *)fcntls); // return value form SQLITE_FCNTL_PRAGMA
		if (rc == RC_OK)
		{
			if (fcntls[0])
			{
				int mem = ++parse->Mems;
				v->AddOp4(OP_String8, 0, mem, 0, fcntls[0], 0);
				v->SetNumCols(1);
				v->SetColName(0, COLNAME_NAME, "result", DESTRUCTOR_STATIC);
				v->AddOp2(OP_ResultRow, mem, 1);
				_free(fcntls[0]);
			}
		}
		else if (rc != RC_NOTFOUND)
		{
			if (fcntls[0])
			{
				parse->ErrorMsg("%s", fcntls[0]);
				_free(fcntls[0]);
			}
			parse->Errs++;
			parse->RC = rc;
		}

#if !defined(OMIT_PAGER_PRAGMAS) && !defined(OMIT_DEPRECATED)
		//  PRAGMA [database.]default_cache_size
		//  PRAGMA [database.]default_cache_size=N
		//
		// The first form reports the current persistent setting for the page cache size.  The value returned is the maximum number of
		// pages in the page cache.  The second form sets both the current page cache size value and the persistent page cache size value
		// stored in the database file.
		//
		// Older versions of SQLite would set the default cache size to a negative number to indicate synchronous=OFF.  These days, synchronous
		// is always on by default regardless of the sign of the default cache size.  But continue to take the absolute value of the default cache
		// size of historical compatibility.
		else if (!_strcmp(left, "default_cache_size"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			v->UsesBtree(db);
			if (!right)
			{
				v->SetNumCols(1);
				v->SetColName(0, COLNAME_NAME, "cache_size", DESTRUCTOR_STATIC);
				parse->Mems += 2;
				int addr = v->AddOpList(_lengthof(_getCacheSize), _getCacheSize);
				v->ChangeP1(addr, db);
				v->ChangeP1(addr+1, db);
				v->ChangeP1(addr+6, DEFAULT_CACHE_SIZE);
			}
			else
			{
				int size = MathEx::Abs(ConvertEx::Atoi(right));
				parse->BeginWriteOperation(0, db);
				v->AddOp2(OP_Integer, size, 1);
				v->AddOp3(OP_SetCookie, db, Btree::META_DEFAULT_CACHE_SIZE, 1);
				_assert(Btree::SchemaMutexHeld(ctx, db, 0));
				dbAsObj->Schema->CacheSize = size;
				dbAsObj->Bt->SetCacheSize(dbAsObj->Schema->CacheSize);
			}
		}
#endif

#if !defined(OMIT_PAGER_PRAGMAS)
		//  PRAGMA [database.]page_size
		//  PRAGMA [database.]page_size=N
		//
		// The first form reports the current setting for the database page size in bytes.  The second form sets the
		// database page size value.  The value can only be set if the database has not yet been created.
		else if (!_strcmp(left, "page_size"))
		{
			Btree *bt = dbAsObj->Bt;
			_assert(bt != nullptr);
			if (!right)
			{
				int size = (_ALWAYS(bt) ? bt->GetPageSize() : 0);
				ReturnSingleInt(parse, "page_size", size);
			}
			else
			{
				// Malloc may fail when setting the page-size, as there is an internal buffer that the pager module resizes using sqlite3_realloc().
				ctx->NextPagesize = ConvertEx::Atoi(right);
				if (bt->SetPageSize(ctx->NextPagesize, -1, false) == RC_NOMEM)
					ctx->MallocFailed = true;
			}
		}

		//  PRAGMA [database.]secure_delete
		//  PRAGMA [database.]secure_delete=ON/OFF
		//
		// The first form reports the current setting for the secure_delete flag.  The second form changes the secure_delete
		// flag setting and reports thenew value.
		else if (!_strcmp(left, "secure_delete"))
		{
			Btree *bt = dbAsObj->Bt;
			_assert(bt != nullptr);
			int b = (right ? (int)ConvertEx::GetBoolean(right, 0) : -1);
			if (id2->length == 0 && b >= 0)
				for (int ii = 0; ii < ctx->DBs.length; ii++)
					ctx->DBs[ii].Bt->SecureDelete(b != 0);
			b = (bt->SecureDelete(b) ? 1 : 0);
			ReturnSingleInt(parse, "secure_delete", b);
		}

		//  PRAGMA [database.]max_page_count
		//  PRAGMA [database.]max_page_count=N
		//
		// The first form reports the current setting for the maximum number of pages in the database file.  The 
		// second form attempts to change this setting.  Both forms return the current setting.
		//
		// The absolute value of N is used.  This is undocumented and might change.  The only purpose is to provide an easy way to test
		// the sqlite3AbsInt32() function.
		//
		//  PRAGMA [database.]page_count
		//
		// Return the number of pages in the specified database.
		else if (!_strcmp(left, "page_count") || !_strcmp(left, "max_page_count"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			parse->CodeVerifySchema(db);
			int regId = ++parse->Mems;
			if (_tolower(left[0]) == 'p')
				v->AddOp2(OP_Pagecount, db, regId);
			else
				v->AddOp3(OP_MaxPgcnt, db, regId, MathEx::Abs(ConvertEx::Atoi(right)));
			v->AddOp2(OP_ResultRow, regId, 1);
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, left, DESTRUCTOR_TRANSIENT);
		}

		//  PRAGMA [database.]locking_mode
		//  PRAGMA [database.]locking_mode = (normal|exclusive)
		else if (!_strcmp(left, "locking_mode"))
		{
			const char *ret = "normal";
			IPager::LOCKINGMODE mode = GetLockingMode(right);
			if (id2->length == 0 && mode == IPager::LOCKINGMODE_QUERY)
				mode = ctx->DefaultLockMode; // Simple "PRAGMA locking_mode;" statement. This is a query for the current default locking mode (which may be different to the locking-mode of the main database).
			else
			{
				Pager *pager;
				if (id2->length == 0)
				{
					// This indicates that no database name was specified as part of the PRAGMA command. In this case the locking-mode must be
					// set on all attached databases, as well as the main db file.
					//
					// Also, the sqlite3.dfltLockMode variable is set so that any subsequently attached databases also use the specified
					// locking mode.
					_assert(dbAsObj == &ctx->DBs[0]);
					for (int ii = 2; ii < ctx->DBs.length; ii++)
					{
						pager = ctx->DBs[ii].Bt->get_Pager();
						pager->LockingMode(mode);
					}
					ctx->DefaultLockMode = mode;
				}
				pager = dbAsObj->Bt->get_Pager();
				mode = pager->LockingMode(mode);
			}

			_assert(mode == IPager::LOCKINGMODE_NORMAL || mode == IPager::LOCKINGMODE_EXCLUSIVE);
			if (mode == IPager::LOCKINGMODE_EXCLUSIVE)
				ret = "exclusive";
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "locking_mode", DESTRUCTOR_STATIC);
			v->AddOp4(OP_String8, 0, 1, 0, ret, 0);
			v->AddOp2(OP_ResultRow, 1, 1);
		}

		//  PRAGMA [database.]journal_mode
		//  PRAGMA [database.]journal_mode = (delete|persist|off|truncate|memory|wal|off)
		else if (!_strcmp(left, "journal_mode"))
		{
			// Force the schema to be loaded on all databases.  This causes all database files to be opened and the journal_modes set.  This is
			// necessary because subsequent processing must know if the databases are in WAL mode.
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "journal_mode", DESTRUCTOR_STATIC);

			int mode; // One of the PAGER_JOURNALMODE_XXX symbols
			if (!right)
				mode = IPager::JOURNALMODE_JQUERY; // If there is no "=MODE" part of the pragma, do a query for the current mode
			else
			{
				const char *modeName;
				int n = _strlen30(right);
				for (mode = 0; (modeName = JournalModename((IPager::JOURNALMODE)mode)) != 0; mode++)
					if (!_strncmp(right, modeName, n)) break;
				if (!modeName)
					mode = IPager::JOURNALMODE_JQUERY; // If the "=MODE" part does not match any known journal mode, then do a query
			}
			if (mode == IPager::JOURNALMODE_JQUERY && id2->length == 0) // Convert "PRAGMA journal_mode" into "PRAGMA main.journal_mode"
			{
				db = 0;
				id2->length = 1;
			}
			for (int ii = ctx->DBs.length-1; ii >= 0; ii--)
			{
				if (ctx->DBs[ii].Bt && (ii == db || id2->length == 0))
				{
					v->UsesBtree(ii);
					v->AddOp3(OP_JournalMode, ii, 1, mode);
				}
			}
			v->AddOp2(OP_ResultRow, 1, 1);
		}

		//  PRAGMA [database.]journal_size_limit
		//  PRAGMA [database.]journal_size_limit=N
		//
		// Get or set the size limit on rollback journal files.
		else if (!_strcmp(left, "journal_size_limit"))
		{
			Pager *pager = dbAsObj->Bt->get_Pager();
			int64 limit = -2;
			if (right)
			{
				ConvertEx::Atoi64(right, &limit, 1000000, TEXTENCODE_UTF8);
				if (limit < -1) limit = -1;
			}
			limit = pager->SetJournalSizeLimit(limit);
			ReturnSingleInt(parse, "journal_size_limit", limit);
		}
#endif

#ifndef OMIT_AUTOVACUUM
		//  PRAGMA [database.]auto_vacuum
		//  PRAGMA [database.]auto_vacuum=N
		//
		// Get or set the value of the database 'auto-vacuum' parameter. The value is one of:  0 NONE 1 FULL 2 INCREMENTAL
		else if (!_strcmp(left, "auto_vacuum"))
		{
			Btree *bt = dbAsObj->Bt;
			_assert(bt != nullptr);
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			if (!right)
			{
				Btree::AUTOVACUUM auto_ = (_ALWAYS(bt) ? bt->GetAutoVacuum() : DEFAULT_AUTOVACUUM);
				ReturnSingleInt(parse, "auto_vacuum", (int)auto_);
			}
			else
			{
				Btree::AUTOVACUUM auto_ = GetAutoVacuum(right);
				_assert(auto_ <= 2);
				ctx->NextAutovac = auto_;
				//if (_ALWAYS(auto_ >= 0))
				{
					// Call SetAutoVacuum() to set initialize the internal auto and incr-vacuum flags. This is required in case this connection
					// creates the database file. It is important that it is created as an auto-vacuum capable db.
					rc = bt->SetAutoVacuum(auto_);
					if (rc == RC_OK && (auto_==Btree::AUTOVACUUM_FULL || auto_==Btree::AUTOVACUUM_INCR))
					{
						// When setting the auto_vacuum mode to either "full" or  "incremental", write the value of meta[6] in the database
						// file. Before writing to meta[6], check that meta[3] indicates that this really is an auto-vacuum capable database.
						int addr = v->AddOpList(_lengthof(_setMeta6), _setMeta6);
						v->ChangeP1(addr, db);
						v->ChangeP1(addr+1, db);
						v->ChangeP2(addr+2, addr+4);
						v->ChangeP1(addr+4, auto_-1);
						v->ChangeP1(addr+5, db);
						v->UsesBtree(db);
					}
				}
			}
		}

		//  PRAGMA [database.]incremental_vacuum(N)
		//
		// Do N steps of incremental vacuuming on a database.
		else if (!_strcmp(left, "incremental_vacuum"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			int limit;
			if (right == 0 || !ConvertEx::Atoi(right, &limit) || limit <= 0)
				limit = 0x7fffffff;
			parse->BeginWriteOperation(0, db);
			v->AddOp2(OP_Integer, limit, 1);
			int addr = v->AddOp1(OP_IncrVacuum, db);
			v->AddOp1(OP_ResultRow, 1);
			v->AddOp2(OP_AddImm, 1, -1);
			v->AddOp2(OP_IfPos, 1, addr);
			v->JumpHere(addr);
		}
#endif

#ifndef OMIT_PAGER_PRAGMAS
		//  PRAGMA [database.]cache_size
		//  PRAGMA [database.]cache_size=N
		//
		// The first form reports the current local setting for the page cache size. The second form sets the local
		// page cache size value.  If N is positive then that is the number of pages in the cache.  If N is negative, then the
		// number of pages is adjusted so that the cache uses -N kibibytes of memory.
		else if (!_strcmp(left, "cache_size"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			_assert(Btree::SchemaMutexHeld(ctx, db, 0));
			if (!right)
				ReturnSingleInt(parse, "cache_size", dbAsObj->Schema->CacheSize);
			else
			{
				int size = ConvertEx::Atoi(right);
				dbAsObj->Schema->CacheSize = size;
				dbAsObj->Bt->SetCacheSize(dbAsObj->Schema->CacheSize);
			}
		}

		//   PRAGMA temp_store
		//   PRAGMA temp_store = "default"|"memory"|"file"
		//
		// Return or set the local value of the temp_store flag.  Changing the local value does not make changes to the disk file and the default
		// value will be restored the next time the database is opened.
		//
		// Note that it is possible for the library compile-time options to override this setting
		else if (!_strcmp(left, "temp_store"))
		{
			if (!right)
				ReturnSingleInt(parse, "temp_store", ctx->TempStore);
			else
				ChangeTempStorage(parse, right);
		}

		//   PRAGMA temp_store_directory
		//   PRAGMA temp_store_directory = ""|"directory_name"
		//
		// Return or set the local value of the temp_store_directory flag.  Changing the value sets a specific directory to be used for temporary files.
		// Setting to a null string reverts to the default temporary directory search. If temporary directory is changed, then invalidateTempStorage.

		else if (!_strcmp(left, "temp_store_directory"))
		{
			if (!right)
			{
				if (g_temp_directory)
				{
					v->SetNumCols(1);
					v->SetColName(0, COLNAME_NAME, "temp_store_directory", DESTRUCTOR_STATIC);
					v->AddOp4(OP_String8, 0, 1, 0, g_temp_directory, 0);
					v->AddOp2(OP_ResultRow, 1, 1);
				}
			}
			else
			{
#ifndef OMIT_WSD
				if (right[0])
				{
					int res;
					rc = ctx->Vfs->Access(right, VSystem::ACCESS_READWRITE, &res);
					if (rc != RC_OK || res == 0)
					{
						parse->ErrorMsg("not a writable directory");
						goto pragma_out;
					}
				}
				if (TEMP_STORE == 0 || (TEMP_STORE == 1 && ctx->TempStore <= 1) || (TEMP_STORE == 2 && ctx->TempStore == 1))
					InvalidateTempStorage(parse);
				_free(g_temp_directory);
				g_temp_directory = (right[0] ? _mprintf("%s", right) : nullptr);
#endif
			}
		}

#if OS_WIN
		//   PRAGMA data_store_directory
		//   PRAGMA data_store_directory = ""|"directory_name"
		//
		// Return or set the local value of the data_store_directory flag.  Changing the value sets a specific directory to be used for database files that
		// were specified with a relative pathname.  Setting to a null string reverts to the default database directory, which for database files specified with
		// a relative path will probably be based on the current directory for the process.  Database file specified with an absolute path are not impacted
		// by this setting, regardless of its value.
		else if (!_strcmp(left, "data_store_directory"))
		{
			if (!right)
			{
				if (g_data_directory)
				{
					v->SetNumCols(1);
					v->SetColName(0, COLNAME_NAME, "data_store_directory", DESTRUCTOR_STATIC);
					v->AddOp4(OP_String8, 0, 1, 0, g_data_directory, 0);
					v->AddOp2(OP_ResultRow, 1, 1);
				}
			}
			else
			{
#ifndef OMIT_WSD
				if (right[0])
				{
					int res;
					rc = ctx->Vfs->Access(right, VSystem::ACCESS_READWRITE, &res);
					if (rc!=RC_OK || res==0)
					{
						parse->ErrorMsg("not a writable directory");
						goto pragma_out;
					}
				}
				_free(g_data_directory);
				g_data_directory = (right[0] ? _mprintf("%s", right) : nullptr);
#endif
			}
		}
#endif

#if ENABLE_LOCKING_STYLE
		//   PRAGMA [database.]lock_proxy_file
		//   PRAGMA [database.]lock_proxy_file = ":auto:"|"lock_file_path"
		//
		// Return or set the value of the lock_proxy_file flag.  Changing the value sets a specific file to be used for database access locks.
		else if (!_strcmp(left, "lock_proxy_file"))
		{
			if (!right)
			{
				Pager *pager = dbAsObj->Bt->get_Pager();
				VFile *file = pager->get_File();
				char *proxy_file_path = NULL;
				file->FileControl(VFile::GET_LOCKPROXYFILE, &proxy_file_path);
				if (proxy_file_path)
				{
					v->SetNumCols(1);
					v->SetColName(0, COLNAME_NAME, "lock_proxy_file", DESTRUCTOR_STATIC);
					v->AddOp4(OP_String8, 0, 1, 0, proxy_file_path, 0);
					v->AddOp2(OP_ResultRow, 1, 1);
				}
			}
			else
			{
				Pager *pager = dbAsObj->Bt->get_Pager();
				VFile *file = pager->get_File();
				rc = file->FileControl(SQLITE_SET_LOCKPROXYFILE, (right[0] ? right : nullptr));
				if (rc != RC_OK)
				{
					parse->ErrorMsg("failed to set lock proxy file");
					goto pragma_out;
				}
			}
		}
#endif

		//   PRAGMA [database.]synchronous
		//   PRAGMA [database.]synchronous=OFF|ON|NORMAL|FULL
		//
		// Return or set the local value of the synchronous flag.  Changing the local value does not make changes to the disk file and the
		// default value will be restored the next time the database is opened.
		else if (!_strcmp(left, "synchronous"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			if (!right)
				ReturnSingleInt(parse, "synchronous", dbAsObj->SafetyLevel-1);
			else
			{
				if (!ctx->AutoCommit)
					parse->ErrorMsg("Safety level may not be changed inside a transaction");
				else
					dbAsObj->SafetyLevel = ConvertEx::GetSafetyLevel(right, 0, 1) + 1;
			}
		}
#endif

#ifndef OMIT_FLAG_PRAGMAS
		// The FlagPragma() subroutine also generates any necessary code there is nothing more to do here
		else if (FlagPragma(parse, left, right)) { }	
#endif

#ifndef OMIT_SCHEMA_PRAGMAS
		//   PRAGMA table_info(<table>)
		//
		// Return a single row for each column of the named table. The columns of the returned data set are:
		//
		// cid:        Column id (numbered from left to right, starting at 0)
		// name:       Column name
		// type:       Column declaration type.
		// notnull:    True if 'NOT NULL' is part of column declaration
		// dflt_value: The default value for the column, if any.
		else if (!_strcmp(left, "table_info") && right)
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			Table *table = Parse::FindTable(ctx, right, dbName);
			if (table)
			{
				Index *pk;
				for (pk = table->Index; pk && pk->AutoIndex != 2; pk = pk->Next) { }
				v->SetNumCols(6);
				parse->Mems = 6;
				parse->CodeVerifySchema(db);
				v->SetColName(0, COLNAME_NAME, "cid", DESTRUCTOR_STATIC);
				v->SetColName(1, COLNAME_NAME, "name", DESTRUCTOR_STATIC);
				v->SetColName(2, COLNAME_NAME, "type", DESTRUCTOR_STATIC);
				v->SetColName(3, COLNAME_NAME, "notnull", DESTRUCTOR_STATIC);
				v->SetColName(4, COLNAME_NAME, "dflt_value", DESTRUCTOR_STATIC);
				v->SetColName(5, COLNAME_NAME, "pk", DESTRUCTOR_STATIC);
				parse->ViewGetColumnNames(table);
				int i;
				Column *col;
				int hidden = 0;
				for (i = 0, col = table->Cols.data; i < table->Cols.length; i++, col++)
				{
					if (IsHiddenColumn(col))
					{
						hidden++;
						continue;
					}
					v->AddOp2(OP_Integer, i-hidden, 1);
					v->AddOp4(OP_String8, 0, 2, 0, col->Name, 0);
					v->AddOp4(OP_String8, 0, 3, 0, (col->Type ? col->Type : ""), 0);
					v->AddOp2(OP_Integer, (col->NotNull ? 1 : 0), 4);
					if (col->DfltName)
						v->AddOp4(OP_String8, 0, 5, 0, (char *)col->DfltName, 0);
					else
						v->AddOp2(OP_Null, 0, 5);
					int k;
					if ((col->ColFlags & COLFLAG_PRIMKEY) == 0)
						k = 0;
					else if (!pk)
						k = 1;
					else
					{
						for (k = 1; _ALWAYS(k <= table->Cols.length) && pk->Columns[k-1] != i; k++) { }
					}
					v->AddOp2(OP_Integer, k, 6);
					v->AddOp2(OP_ResultRow, 1, 6);
				}
			}
		}
		else if (!_strcmp(left, "index_info") && right)
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			Index *index = Parse::FindIndex(ctx, right, dbName);
			if (index)
			{
				Table *table = index->Table;
				v->SetNumCols(3);
				parse->Mems = 3;
				parse->CodeVerifySchema(db);
				v->SetColName(0, COLNAME_NAME, "seqno", DESTRUCTOR_STATIC);
				v->SetColName(1, COLNAME_NAME, "cid", DESTRUCTOR_STATIC);
				v->SetColName(2, COLNAME_NAME, "name", DESTRUCTOR_STATIC);
				for (int i = 0; i < index->Columns.length; i++)
				{
					int cid = index->Columns[i];
					v->AddOp2(OP_Integer, i, 1);
					v->AddOp2(OP_Integer, cid, 2);
					_assert(table->Cols.length > cid);
					v->AddOp4(OP_String8, 0, 3, 0, table->Cols[cid].Name, 0);
					v->AddOp2(OP_ResultRow, 1, 3);
				}
			}
		}
		else if (!_strcmp(left, "index_list") && right)
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			Table *table = Parse::FindTable(ctx, right, dbName);
			if (table)
			{
				v = parse->GetVdbe();
				Index *index = table->Index;
				if (index)
				{
					v->SetNumCols(3);
					parse->Mems = 3;
					parse->CodeVerifySchema(db);
					v->SetColName(0, COLNAME_NAME, "seq", DESTRUCTOR_STATIC);
					v->SetColName(1, COLNAME_NAME, "name", DESTRUCTOR_STATIC);
					v->SetColName(2, COLNAME_NAME, "unique", DESTRUCTOR_STATIC);
					int i = 0; 
					while (index)
					{
						v->AddOp2(OP_Integer, i, 1);
						v->AddOp4(OP_String8, 0, 2, 0, index->Name, 0);
						v->AddOp2(OP_Integer, index->OnError!=OE_None, 3);
						v->AddOp2(OP_ResultRow, 1, 3);
						++i;
						index = index->Next;
					}
				}
			}
		}
		else if (!_strcmp(left, "database_list"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			v->SetNumCols(3);
			parse->Mems = 3;
			v->SetColName(0, COLNAME_NAME, "seq", DESTRUCTOR_STATIC);
			v->SetColName(1, COLNAME_NAME, "name", DESTRUCTOR_STATIC);
			v->SetColName(2, COLNAME_NAME, "file", DESTRUCTOR_STATIC);
			for (int i = 0; i < ctx->DBs.length; i++)
			{
				if (!ctx->DBs[i].Bt) continue;
				_assert(ctx->DBs[i].Name != nullptr);
				v->AddOp2(OP_Integer, i, 1);
				v->AddOp4(OP_String8, 0, 2, 0, ctx->DBs[i].Name, 0);
				v->AddOp4(OP_String8, 0, 3, 0, ctx->DBs[i].Bt->get_Filename(), 0);
				v->AddOp2(OP_ResultRow, 1, 3);
			}
		}
		else if (!_strcmp(left, "collation_list"))
		{
			v->SetNumCols(2);
			parse->Mems = 2;
			v->SetColName(0, COLNAME_NAME, "seq", DESTRUCTOR_STATIC);
			v->SetColName(1, COLNAME_NAME, "name", DESTRUCTOR_STATIC);
			int i = 0;
			for (HashElem *p = ctx->CollSeqs.First; p; p = p->Next)
			{
				CollSeq *coll = (CollSeq *)p->Data;
				v->AddOp2(OP_Integer, i++, 1);
				v->AddOp4(OP_String8, 0, 2, 0, coll->Name, 0);
				v->AddOp2(OP_ResultRow, 1, 2);
			}
		}
#endif
#ifndef OMIT_FOREIGN_KEY
		else if (!_strcmp(left, "foreign_key_list") && right)
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			Table *table = Parse::FindTable(ctx, right, dbName);
			if (table)
			{
				v = parse->GetVdbe();
				FKey *fk = table->FKeys;
				if (fk)
				{
					v->SetNumCols(8);
					parse->Mems = 8;
					parse->CodeVerifySchema(db);
					v->SetColName(0, COLNAME_NAME, "id", DESTRUCTOR_STATIC);
					v->SetColName(1, COLNAME_NAME, "seq", DESTRUCTOR_STATIC);
					v->SetColName(2, COLNAME_NAME, "table", DESTRUCTOR_STATIC);
					v->SetColName(3, COLNAME_NAME, "from", DESTRUCTOR_STATIC);
					v->SetColName(4, COLNAME_NAME, "to", DESTRUCTOR_STATIC);
					v->SetColName(5, COLNAME_NAME, "on_update", DESTRUCTOR_STATIC);
					v->SetColName(6, COLNAME_NAME, "on_delete", DESTRUCTOR_STATIC);
					v->SetColName(7, COLNAME_NAME, "match", DESTRUCTOR_STATIC);
					int i = 0; 
					while (fk)
					{
						for (int j = 0; j < fk->Cols.length; j++)
						{
							char *colName = fk->Cols[j].Col;
							char *onDelete = (char *)ActionName(fk->Actions[0]);
							char *onUpdate = (char *)ActionName(fk->Actions[1]);
							v->AddOp2(OP_Integer, i, 1);
							v->AddOp2(OP_Integer, j, 2);
							v->AddOp4(OP_String8, 0, 3, 0, fk->To, 0);
							v->AddOp4(OP_String8, 0, 4, 0, table->Cols[fk->Cols[j].From].Name, 0);
							v->AddOp4(colName ? OP_String8 : OP_Null, 0, 5, 0, colName, 0);
							v->AddOp4(OP_String8, 0, 6, 0, onUpdate, 0);
							v->AddOp4(OP_String8, 0, 7, 0, onDelete, 0);
							v->AddOp4(OP_String8, 0, 8, 0, "NONE", 0);
							v->AddOp2(OP_ResultRow, 1, 8);
						}
						++i;
						fk = fk->NextFrom;
					}
				}
			}
		}
#ifndef OMIT_TRIGGER
		else if (!_strcmp(left, "foreign_key_check"))
		{
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			int regResult = parse->Mems+1; // 3 registers to hold a result row
			parse->Mems += 4;
			int regKey = ++parse->Mems; // Register to hold key for checking the FK
			int regRow = ++parse->Mems; // Registers to hold a row from pTab
			v = parse->GetVdbe();
			v->SetNumCols(4);
			v->SetColName(0, COLNAME_NAME, "table", DESTRUCTOR_STATIC);
			v->SetColName(1, COLNAME_NAME, "rowid", DESTRUCTOR_STATIC);
			v->SetColName(2, COLNAME_NAME, "parent", DESTRUCTOR_STATIC);
			v->SetColName(3, COLNAME_NAME, "fkid", DESTRUCTOR_STATIC);
			parse->CodeVerifySchema(db);
			HashElem *k = ctx->DBs[db].Schema->TableHash.First; // Loop counter:  Next table in schema
			int i; // Loop counter:  Foreign key number for pTab
			while (k)
			{
				Table *table; // Child table contain "REFERENCES" keyword
				if (right)
				{
					table = parse->LocateTable(false, right, dbName);
					k = nullptr;
				}
				else
				{
					table = (Table *)k->Data;
					k = k->Next;
				}
				if (!table || !table->FKeys) continue;
				parse->TableLock(db, table->Id, false, table->Name);
				if (table->Cols.length+regRow > parse->Mems) parse->Mems = table->Cols.length + regRow;
				Insert::OpenTable(parse, 0, db, table, OP_OpenRead);
				v->AddOp4(OP_String8, 0, regResult, 0, table->Name, Vdbe::P4T_TRANSIENT);
				Table *parent; // Parent table that child points to
				Index *index; // Index in the parent table
				int *cols; // child to parent column mapping
				int x; // result variable
				FKey *fk; // A foreign key constraint
				for (i = 1, fk = table->FKeys; fk; i++, fk = fk->NextFrom)
				{
					parent = parse->LocateTable(false, fk->To, dbName);
					if (!parent) break;
					index = nullptr;
					parse->TableLock(db, parent->Id, false, parent->Name);
					x = parse->FKLocateIndex(parent, fk, &index, nullptr);
					if (x == 0)
					{
						if (!index)
							Insert::OpenTable(parse, i, db, parent, OP_OpenRead);
						else
						{
							KeyInfo *key = parse->IndexKeyinfo(index);
							v->AddOp3(OP_OpenRead, i, index->Id, db);
							v->ChangeP4(-1, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
						}
					}
					else
					{
						k = nullptr;
						break;
					}
				}
				if (fk) break;
				if (parse->Tabs < i) parse->Tabs = i;
				int addrTop = v->AddOp1(OP_Rewind, 0); // Top of a loop checking foreign keys
				for (i = 1, fk = table->FKeys; fk; i++, fk = fk->NextFrom)
				{
					parent = parse->LocateTable(false, fk->To, dbName);
					_assert(parent != nullptr);
					index = nullptr;
					cols = nullptr;
					x = parse->FKLocateIndex(parent, fk, &index, &cols);
					_assert(x == 0);
					int addrOk = v->MakeLabel(); // Jump here if the key is OK
					if (!index)
					{
						int keyId = fk->Cols[0].From;
						_assert(keyId>=0 && keyId < table->Cols.length);
						if (keyId != table->PKey)
						{
							v->AddOp3(OP_Column, 0, keyId, regRow);
							Update::ColumnDefault(v, table, keyId, regRow);
							v->AddOp2(OP_IsNull, regRow, addrOk);
							v->AddOp2(OP_MustBeInt, regRow, v->CurrentAddr()+3);
						}
						else
							v->AddOp2(OP_Rowid, 0, regRow);
						v->AddOp3(OP_NotExists, i, 0, regRow);
						v->AddOp2(OP_Goto, 0, addrOk);
						v->JumpHere(v->CurrentAddr()-2);
					}
					else
					{
						for (int j = 0; j < fk->Cols.length; j++)
						{
							Expr::CodeGetColumnOfTable(v, table, 0, (cols ? cols[j] : fk->Cols[0].From), regRow+j);
							v->AddOp2(OP_IsNull, regRow+j, addrOk);
						}
						v->AddOp3(OP_MakeRecord, regRow, fk->Cols.length, regKey);
						v->ChangeP4(-1, Insert::IndexAffinityStr(v, index), Vdbe::P4T_TRANSIENT);
						v->AddOp4Int(OP_Found, i, addrOk, regKey, 0);
					}
					v->AddOp2(OP_Rowid, 0, regResult+1);
					v->AddOp4(OP_String8, 0, regResult+2, 0, fk->To, Vdbe::P4T_TRANSIENT);
					v->AddOp2(OP_Integer, i-1, regResult+3);
					v->AddOp2(OP_ResultRow, regResult, 4);
					v->ResolveLabel(addrOk);
					_tagfree(ctx, cols);
				}
				v->AddOp2(OP_Next, 0, addrTop+1);
				v->JumpHere(addrTop);
			}
		}
#endif
#endif

#ifndef NDEBUG
		else if (!_strcmp(left, "parser_trace"))
		{
			if (right)
			{
				if (ConvertEx::GetBoolean(right, 0))
#if __CUDACC__
					ParserTrace(nullptr, "parser: ");
#else
					ParserTrace(stderr, "parser: ");
#endif
				else
					ParserTrace(nullptr, nullptr);
			}
		}
#endif

		// Reinstall the LIKE and GLOB functions.  The variant of LIKE used will be case sensitive or not depending on the RHS.
		else if (!_strcmp(left, "case_sensitive_like"))
		{
			if (right)
				Func::RegisterLikeFunctions(ctx, ConvertEx::GetBoolean(right, 0));
		}

#ifndef OMIT_INTEGRITY_CHECK
		// Pragma "quick_check" is an experimental reduced version of integrity_check designed to detect most database corruption
		// without most of the overhead of a full integrity-check.
		else if (!_strcmp(left, "integrity_check") || !_strcmp(left, "quick_check"))
		{
			bool isQuick = (_tolower(left[0]) == 'q');

			// If the PRAGMA command was of the form "PRAGMA <ctx>.integrity_check", then db is set to the index of the database identified by <ctx>.
			// In this case, the integrity of database db only is verified by the VDBE created below.
			//
			// Otherwise, if the command was simply "PRAGMA integrity_check" (or "PRAGMA quick_check"), then db is set to 0. In this case, set db
			// to -1 here, to indicate that the VDBE should verify the integrity of all attached databases.
			_assert(db >= 0);
			_assert(db == 0 || id2->data);
			if (!id2->data) db = -1;

			// Initialize the VDBE program
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			parse->Mems = 6;
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "integrity_check", DESTRUCTOR_STATIC);

			// Set the maximum error count
			int maxErr = INTEGRITY_CHECK_ERROR_MAX;
			if (right)
			{
				ConvertEx::Atoi(right, &maxErr);
				if (maxErr <= 0)
					maxErr = INTEGRITY_CHECK_ERROR_MAX;
			}
			v->AddOp2(OP_Integer, maxErr, 1); // reg[1] holds errors left

			// Do an integrity check on each database file
			int addr;
			for (int i = 0; i < ctx->DBs.length; i++)
			{
				if (E_OMIT_TEMPDB && i == 1) continue;
				if (db >= 0 && i != db) continue;

				parse->CodeVerifySchema(i);
				addr = v->AddOp1(OP_IfPos, 1); // Halt if out of errors
				v->AddOp2(OP_Halt, 0, 0);
				v->JumpHere(addr);

				// Do an integrity check of the B-Tree
				//
				// Begin by filling registers 2, 3, ... with the root pages numbers for all tables and indices in the database.
				int cnt = 0;
				_assert(Btree::SchemaMutexHeld(ctx, i, nullptr));
				Hash *tables = &ctx->DBs[i].Schema->TableHash;
				HashElem *x;
				for (x = tables->First; x; x = x->Next)
				{
					Table *table = (Table *)x->Data;
					v->AddOp2(OP_Integer, table->Id, 2+cnt);
					cnt++;
					for (Index *index = table->Index; index; index = index->Next)
					{
						v->AddOp2(OP_Integer, index->Id, 2+cnt);
						cnt++;
					}
				}

				// Make sure sufficient number of registers have been allocated
				if (parse->Mems < cnt+4)
					parse->Mems = cnt+4;

				// Do the b-tree integrity checks
				v->AddOp3(OP_IntegrityCk, 2, cnt, 1);
				v->ChangeP5((uint8)i);
				addr = v->AddOp1(OP_IsNull, 2);
				v->AddOp4(OP_String8, 0, 3, 0, _mtagprintf(ctx, "*** in database %s ***\n", ctx->DBs[i].Name), Vdbe::P4T_DYNAMIC);
				v->AddOp2(OP_Move, 2, 4);
				v->AddOp3(OP_Concat, 4, 3, 2);
				v->AddOp2(OP_ResultRow, 2, 1);
				v->JumpHere(addr);

				// Make sure all the indices are constructed correctly.
				for (x = tables->First; x && !isQuick; x = x->Next)
				{
					Table *table = (Table *)x->Data;
					if (!table->Index) continue;
					addr = v->AddOp1(OP_IfPos, 1); // Stop if out of errors
					v->AddOp2(OP_Halt, 0, 0);
					v->JumpHere(addr);
					Insert::OpenTableAndIndices(parse, table, 1, OP_OpenRead);
					v->AddOp2(OP_Integer, 0, 2); // reg(2) will count entries
					int loopTop = v->AddOp2(OP_Rewind, 1, 0);
					v->AddOp2(OP_AddImm, 2, 1); // increment entry count
					int j;
					Index *index;
					for (j = 0, index = table->Index; index; index = index->Next, j++)
					{
						int r1 = Delete::GenerateIndexKey(parse, index, 1, 3, 0);
						int jmp2 = v->AddOp4Int(OP_Found, j+2, 0, r1, index->Columns.length+1);
						addr = v->AddOpList(_lengthof(_idxErr), _idxErr);
						v->ChangeP4(addr+1, "rowid ", Vdbe::P4T_STATIC);
						v->ChangeP4(addr+3, " missing from index ", Vdbe::P4T_STATIC);
						v->ChangeP4(addr+4, index->Name, Vdbe::P4T_TRANSIENT);
						v->JumpHere(addr+9);
						v->JumpHere(jmp2);
					}
					v->AddOp2(OP_Next, 1, loopTop+1);
					v->JumpHere(loopTop);
					for (j = 0, index = table->Index; index; index = index->Next, j++)
					{
						addr = v->AddOp1(OP_IfPos, 1);
						v->AddOp2(OP_Halt, 0, 0);
						v->JumpHere(addr);
						addr = v->AddOpList(_lengthof(_cntIdx), _cntIdx);
						v->ChangeP1(addr+1, j+2);
						v->ChangeP2(addr+1, addr+4);
						v->ChangeP1(addr+3, j+2);
						v->ChangeP2(addr+3, addr+2);
						v->JumpHere(addr+4);
						v->ChangeP4(addr+6, "wrong # of entries in index ", Vdbe::P4T_STATIC);
						v->ChangeP4(addr+7, index->Name, Vdbe::P4T_TRANSIENT);
					}
				} 
			}
			addr = v->AddOpList(_lengthof(_endCode), _endCode);
			v->ChangeP2(addr, -maxErr);
			v->JumpHere(addr+1);
			v->ChangeP4(addr+2, "ok", Vdbe::P4T_STATIC);
		}
#endif

#ifndef OMIT_UTF16
		//   PRAGMA encoding
		//   PRAGMA encoding = "utf-8"|"utf-16"|"utf-16le"|"utf-16be"
		//
		// In its first form, this pragma returns the encoding of the main database. If the database is not initialized, it is initialized now.
		//
		// The second form of this pragma is a no-op if the main database file has not already been initialized. In this case it sets the default
		// encoding that will be used for the main database file if a new file is created. If an existing main database file is opened, then the
		// default text encoding for the existing database is used.
		// 
		// In all cases new databases created using the ATTACH command are created to use the same default text encoding as the main database. If
		// the main database has not been initialized and/or created when ATTACH is executed, this is done before the ATTACH operation.
		//
		// In the second form this pragma sets the text encoding to be used in new database files created using this database handle. It is only
		// useful if invoked immediately after the main database i
		else if (!_strcmp(left, "encoding"))
		{
			if (!right) // "PRAGMA encoding"
			{    
				if (Prepare::ReadSchema(parse)) goto pragma_out;
				v->SetNumCols(1);
				v->SetColName(0, COLNAME_NAME, "encoding", DESTRUCTOR_STATIC);
				v->AddOp2(OP_String8, 0, 1);
				_assert(_encodeNames[TEXTENCODE_UTF8].Encode == TEXTENCODE_UTF8);
				_assert(_encodeNames[TEXTENCODE_UTF16LE].Encode == TEXTENCODE_UTF16LE);
				_assert(_encodeNames[TEXTENCODE_UTF16BE].Encode == TEXTENCODE_UTF16BE);
				v->ChangeP4(-1, _encodeNames[CTXENCODE(parse->Ctx)].Name, Vdbe::P4T_STATIC);
				v->AddOp2(OP_ResultRow, 1, 1);
			}
			else // "PRAGMA encoding = XXX"
			{
				// Only change the value of sqlite.enc if the database handle is not initialized. If the main database exists, the new sqlite.enc value
				// will be overwritten when the schema is next loaded. If it does not already exists, it will be created to use the new encoding value.
				if (!(DbHasProperty(ctx, 0, SCHEMA_SchemaLoaded)) || DbHasProperty(ctx, 0, SCHEMA_Empty))
				{
					const EncodeName *encode;
					for (encode = &_encodeNames[0]; encode->Name; encode++)
					{
						if (!_strcmp(right, encode->Name))
						{
							CTXENCODE(parse->Ctx) = (encode->Encode ? encode->Encode : TEXTENCODE_UTF16NATIVE);
							break;
						}
					}
					if (!encode->Name)
						parse->ErrorMsg("unsupported encoding: %s", right);
				}
			}
		}
#endif

#ifndef OMIT_SCHEMA_VERSION_PRAGMAS
		//   PRAGMA [database.]schema_version
		//   PRAGMA [database.]schema_version = <integer>
		//
		//   PRAGMA [database.]user_version
		//   PRAGMA [database.]user_version = <integer>
		//
		// The pragma's schema_version and user_version are used to set or get the value of the schema-version and user-version, respectively. Both
		// the schema-version and the user-version are 32-bit signed integers stored in the database header.
		//
		// The schema-cookie is usually only manipulated internally by SQLite. It is incremented by SQLite whenever the database schema is modified (by
		// creating or dropping a table or index). The schema version is used by SQLite each time a query is executed to ensure that the internal cache
		// of the schema used when compiling the SQL query matches the schema of the database against which the compiled query is actually executed.
		// Subverting this mechanism by using "PRAGMA schema_version" to modify the schema-version is potentially dangerous and may lead to program
		// crashes or database corruption. Use with caution!
		//
		// The user-version is not used internally by SQLite. It may be used by applications for any purpose.
		else if (!_strcmp(left, "schema_version") || !_strcmp(left, "user_version") || !_strcmp(left, "freelist_count"))
		{
			v->UsesBtree(db);
			int cookie; // Cookie index. 1 for schema-cookie, 6 for user-cookie.
			switch (left[0])
			{
			case 'f': case 'F': cookie = Btree::META_FREE_PAGE_COUNT; break;
			case 's': case 'S': cookie = Btree::META_SCHEMA_VERSION; break;
			default: cookie = Btree::META_USER_VERSION; break;
			}
			if (right && cookie != Btree::META_FREE_PAGE_COUNT)
			{
				int addr = v->AddOpList(_lengthof(_setCookie), _setCookie);
				v->ChangeP1(addr, db);
				v->ChangeP1(addr+1, ConvertEx::Atoi(right));
				v->ChangeP1(addr+2, db);
				v->ChangeP2(addr+2, cookie);
			}
			else
			{
				int addr = v->AddOpList(_lengthof(_readCookie), _readCookie);
				v->ChangeP1(addr, db);
				v->ChangeP1(addr+1, db);
				v->ChangeP3(addr+1, cookie);
				v->SetNumCols(1);
				v->SetColName(0, COLNAME_NAME, left, DESTRUCTOR_TRANSIENT);
			}
		}
#endif

#ifndef OMIT_COMPILEOPTION_DIAGS
		//   PRAGMA compile_options
		//
		// Return the names of all compile-time options used in this build, one option per row.
		else if (!_strcmp(left, "compile_options"))
		{
			v->SetNumCols(1);
			parse->Mems = 1;
			v->SetColName(0, COLNAME_NAME, "compile_option", DESTRUCTOR_STATIC);
			const char *opt;
			int i = 0;
			while ((opt = CompileTimeGet(i++)) != nullptr)
			{
				v->AddOp4(OP_String8, 0, 1, 0, opt, 0);
				v->AddOp2(OP_ResultRow, 1, 1);
			}
		}
#endif

#ifndef OMIT_WAL
		//   PRAGMA [database.]wal_checkpoint = passive|full|restart
		//
		// Checkpoint the database.
		else if (!_strcmp(left, "wal_checkpoint"))
		{
			int bt = (id2->data ? db : MAX_ATTACHED);
			IPager::CHECKPOINT mode = IPager::CHECKPOINT_PASSIVE;
			if (right)
			{
				if (!_strcmp(right, "full")) mode = IPager::CHECKPOINT_FULL;
				else if (!_strcmp(right, "restart")) mode = IPager::CHECKPOINT_RESTART;
			}
			if (Prepare::ReadSchema(parse)) goto pragma_out;
			v->SetNumCols(3);
			parse->Mems = 3;
			v->SetColName(0, COLNAME_NAME, "busy", DESTRUCTOR_STATIC);
			v->SetColName(1, COLNAME_NAME, "log", DESTRUCTOR_STATIC);
			v->SetColName(2, COLNAME_NAME, "checkpointed", DESTRUCTOR_STATIC);
			v->AddOp3(OP_Checkpoint, bt, mode, 1);
			v->AddOp2(OP_ResultRow, 1, 3);
		}

		//   PRAGMA wal_autocheckpoint
		//   PRAGMA wal_autocheckpoint = N
		//
		// Configure a database connection to automatically checkpoint a database after accumulating N frames in the log. Or query for the current value of N.
		else if (!_strcmp(left, "wal_autocheckpoint"))
		{
			if (right)
				sqlite3_wal_autocheckpoint(ctx, ConvertEx::Atoi(right));
			ReturnSingleInt(parse, "wal_autocheckpoint", ctx->WalCallback == sqlite3WalDefaultHook ? PTR_TO_INT(ctx->WalArg) : 0);
		}
#endif

		//  PRAGMA shrink_memory
		//
		// This pragma attempts to free as much memory as possible from the current database connection.
		else if (!_strcmp(left, "shrink_memory"))
			Main::CtxReleaseMemory(ctx);

		//   PRAGMA busy_timeout
		//   PRAGMA busy_timeout = N
		//
		// Call sqlite3_busy_timeout(ctx, N).  Return the current timeout value if one is set.  If no busy handler or a different busy handler is set
		// then 0 is returned.  Setting the busy_timeout to 0 or negative disables the timeout.
		else if (!_strcmp(left, "busy_timeout"))
		{
			if (right)
				Main::BusyTmeout(ctx, ConvertEx::Atoi(right));
			ReturnSingleInt(parse, "timeout",  ctx->BusyTimeout);
		}

#if defined(DEBUG) || defined(TEST)
		// Report the current state of file logs for all databases
		else if (!_strcmp(left, "lock_status"))
		{
			v->SetNumCols(2);
			parse->Mems = 2;
			v->SetColName(0, COLNAME_NAME, "database", DESTRUCTOR_STATIC);
			v->SetColName(1, COLNAME_NAME, "status", DESTRUCTOR_STATIC);
			for (int i =0 ; i < ctx->DBs.length; i++)
			{
				if (!ctx->DBs[i].Name) continue;
				v->AddOp4(OP_String8, 0, 1, 0, ctx->DBs[i].Name, Vdbe::P4T_STATIC);
				Btree *bt = ctx->DBs[i].Bt;
				int j;
				const char *state = "unknown";
				if (!bt || !bt->get_Pager()) state = "closed";
				else if (Main::FileControl(ctx, i ? ctx->DBs[i].Name : 0, VFile::FCNTL_LOCKSTATE, &j) == RC_OK) state = _lockNames[j];
				v->AddOp4(OP_String8, 0, 2, 0, state, Vdbe::P4T_STATIC);
				v->AddOp2(OP_ResultRow, 1, 2);
			}
		}
#endif

#ifdef HAS_CODEC
		else if (!_strcmp(left, "key") && right)
			Codec::Key(ctx, right, _strlen30(right));
		else if (!_strcmp(left, "rekey") && right)
			Codec::Rekey(ctx, right, _strlen30(right));
		else if (right && (!_strcmp(left, "hexkey") || !_strcmp(left, "hexrekey")))
		{
			int i, h1, h2;
			char key[40];
			for (i = 0; (h1 = right[i]) != 0 && (h2 = right[i+1]) != 0; i += 2)
			{
				h1 += 9*(1&(h1>>6));
				h2 += 9*(1&(h2>>6));
				key[i/2] = (h2 & 0x0f) | ((h1 & 0xf)<<4);
			}
			if ((left[3] & 0xf) == 0xb)
				Codec::Key(ctx, key, i/2);
			else
				Codec::Rekey(ctx, key, i/2);
		}
#endif

#if defined(HAS_CODEC) || defined(ENABLE_CEROD)
		else if (!_strcmp(left, "activate_extensions") && right)
		{
#ifdef HAS_CODEC
			if (!_strncmp(right, "see-", 4))
				Codec::ActivateSee(&right[4]);
#endif
#ifdef ENABLE_CEROD
			if (!_strncmp(right, "cerod-", 6))
				Codec::ActivateCerod(&right[6]);
#endif
		}
#endif

#ifndef OMIT_PAGER_PRAGMAS
		// Reset the safety level, in case the fullfsync flag or synchronous setting changed.
		else if (ctx->AutoCommit)
		{
			dbAsObj->Bt->SetSafetyLevel(dbAsObj->SafetyLevel, (ctx->Flags & Context::FLAG_FullFSync) != 0, (ctx->Flags & Context::FLAG_CkptFullFSync) != 0);
		}
#endif

pragma_out:
		_tagfree(ctx, left);
		_tagfree(ctx, right);
	}

#endif
}}
