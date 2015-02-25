#include "VdbeInt.cu.h"
#include "..\KeywordHash.h"
//#include <stddef.h>

namespace Core
{
	__device__ void Parse::BeginParse(int explainFlag)
	{
		Explain = explainFlag;
		VarsSeen = 0;
	}

#ifndef OMIT_SHARED_CACHE
	struct TableLock
	{
		int DB;					// The database containing the table to be locked
		int Table;				// The root page of the table to be locked
		bool IsWriteLock;		// True for write lock.  False for a read lock
		const char *Name;		// Name of the table
	};

	__device__ void Parse::TableLock(int db, int table, bool isWriteLock, const char *name)
	{
		_assert(db >= 0);
		Parse *toplevel = Parse_Toplevel(this);
		Core::TableLock *tableLock;
		for (int i = 0; i < toplevel->TableLocks.length; i++)
		{
			tableLock = &toplevel->TableLocks[i];
			if (tableLock->DB == db && tableLock->Table == table)
			{
				tableLock->IsWriteLock |= isWriteLock;
				return;
			}
		}
		int bytes = sizeof(Core::TableLock) * (toplevel->TableLocks.length + 1);
		toplevel->TableLocks.data = (Core::TableLock *)_tagrealloc_or_free(toplevel->Ctx, toplevel->TableLocks.data, bytes);
		if (toplevel->TableLocks)
		{
			tableLock = &toplevel->TableLocks[toplevel->TableLocks.length++];
			tableLock->DB = db;
			tableLock->Table = table;
			tableLock->IsWriteLock = isWriteLock;
			tableLock->Name = name;
		}
		else
		{
			toplevel->TableLocks.length = 0;
			toplevel->Ctx->MallocFailed = true;
		}
	}

	__device__ static void CodeTableLocks(Parse *parse)
	{
		Vdbe *v = parse->GetVdbe();
		_assert(v); // sqlite3GetVdbe cannot fail: VDBE already allocated
		for (int i = 0; i < parse->TableLocks.length; i++)
		{
			TableLock *tableLock = &parse->TableLocks[i];
			int p1 = tableLock->DB;
			v->AddOp4(OP_TableLock, p1, tableLock->Table, tableLock->IsWriteLock, tableLock->Name, Vdbe::P4T_STATIC);
		}
	}
#else
#define CodeTableLocks(x)
#endif

	__device__ void Parse::FinishCoding()
	{
		_assert(!Toplevel);
		Context *ctx = Ctx;
		if (ctx->MallocFailed || Nested || Errs)
			return;

		// Begin by generating some termination code at the end of the vdbe program
		Vdbe *v = GetVdbe();
		_assert(!IsMultiWrite || v->AssertMayAbort(_MayAbort));
		if (v)
		{
			v->AddOp0(OP_Halt);

			// The cookie mask contains one bit for each database file open. (Bit 0 is for main, bit 1 is for temp, and so forth.)  Bits are
			// set for each database that is used.  Generate code to start a transaction on each used database and to verify the schema cookie
			// on each used database.
			if (CookieGoto > 0)
			{
				v->JumpHere(CookieGoto - 1);
				int db; yDbMask mask; 
				for (db = 0, mask = 1; db < ctx->DBs.length; mask <<= 1, db++)
				{
					if ((mask & CookieMask) == 0)
						continue;
					v->UsesBtree(db);
					v->AddOp2(OP_Transaction, db, (mask & WriteMask) != 0);
					if (!ctx->Init.Busy)
					{
						_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
						v->AddOp3(OP_VerifyCookie, db, CookieValue[db], ctx->DBs[db].Schema->Generation);
					}
				}
#ifndef OMIT_VIRTUALTABLE
				{
					for (int i = 0; i < VTableLocks.length; i++)
					{
						char *vtable = (char *)VTable::GetVTable(ctx, VTableLocks[i]);
						v->AddOp4(OP_VBegin, 0, 0, 0, vtable, Vdbe::P4T_VTAB);
					}
					VTableLocks.length = 0;
				}
#endif

				// Once all the cookies have been verified and transactions opened, obtain the required table-locks. This is a no-op unless the 
				// shared-cache feature is enabled.
				CodeTableLocks(this);
				// Initialize any AUTOINCREMENT data structures required.
				Insert::AutoincrementBegin(this);
				// Finally, jump back to the beginning of the executable code.
				v->AddOp2(OP_Goto, 0, CookieGoto);
			}
		}

		// Get the VDBE program ready for execution
		if (v && _ALWAYS(Errs == 0) && !ctx->MallocFailed)
		{
#ifdef DEBUG
			FILE *trace = ((ctx->Flags & BContext::FLAG_VdbeTrace) != 0 ? stdout : 0);
			v.Trace(trace);
#endif
			_assert(CacheLevel == 0);  // Disables and re-enables match
			// A minimum of one cursor is required if autoincrement is used See ticket [a696379c1f08866]
			if (Ainc != 0 && Tabs == 0)
				Tabs = 1;
			v->MakeReady(this);
			RC = RC_DONE;
			ColNamesSet = 0;
		}
		else
			RC = RC_ERROR;
		Tabs = 0;
		Mems = 0;
		Sets = 0;
		VarsSeen = 0;
		CookieMask = 0;
		CookieGoto = 0;
	}

	__device__ void Parse::NestedParse(const char *format, va_list *args)
	{
		if (Errs)
			return;
		_assert(Nested < 10); // Nesting should only be of limited depth
		char *sql = nullptr; //char *sql = sqlite3VMPrintf(ctx, format, args);
		if (!sql)
			return; // A malloc must have failed
		Nested++;
#define SAVE_SZ (sizeof(Parse) - offsetof(Parse, VarsSeen))
		char saveBuf[SAVE_SZ];
		_memcpy(saveBuf, (const char *)&VarsSeen, SAVE_SZ);
		_memset(&VarsSeen, 0, SAVE_SZ);
		char *errMsg = nullptr;
		RunParser(sql, &errMsg);
		Context *ctx = Ctx;
		_tagfree(ctx, errMsg);
		_tagfree(ctx, sql);
		_memcpy((char *)&VarsSeen, saveBuf, SAVE_SZ);
		Nested--;
	}

	__device__ Table *Parse::FindTable(Context *ctx, const char *name, const char *dbName)
	{
		_assert(name);
		int nameLength = _strlen30(name);
		// All mutexes are required for schema access.  Make sure we hold them. */
		_assert(dbName || Btree::HoldsAllMutexes(ctx));
		Table *table = nullptr;
		for (int i = E_OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
			if (dbName && _strcmp(dbName, ctx->DBs[j].Name))
				continue;
			_assert(Btree::SchemaMutexHeld(ctx, j, 0));
			table = (Table *)ctx->DBs[j].Schema->TableHash.Find(name, nameLength);
			if (table)
				break;
		}
		return table;
	}

	__device__ Table *Parse::LocateTable(bool isView, const char *name, const char *dbName)
	{
		// Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
		if (Prepare::ReadSchema(this) != RC_OK)
			return nullptr;

		Table *table = FindTable(Ctx, name, dbName);
		if (!table)
		{
			const char *msg = (isView ? "no such view" : "no such table");
			if (dbName)
				ErrorMsg("%s: %s.%s", msg, dbName, name);
			else
				ErrorMsg("%s: %s", msg, name);
			CheckSchema = 1;
		}
		return table;
	}

	__device__ Table *Parse::LocateTableItem(bool isView,  SrcList::SrcListItem *item)
	{
		_assert(!item->Schema || !item->Database);
		const char *dbName;
		if (item->Schema)
		{
			int db = Prepare::SchemaToIndex(Ctx, item->Schema);
			dbName = Ctx->DBs[db].Name;
		}
		else
			dbName = item->Database;
		return LocateTable(isView, item->Name, dbName);
	}

	__device__ Index *Parse::FindIndex(Context *ctx, const char *name, const char *dbName)
	{
		// All mutexes are required for schema access.  Make sure we hold them.
		_assert(Btree::HoldsAllMutexes(ctx));
		Index *p = nullptr;
		int nameLength = _strlen30(name);
		for (int i = E_OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
			Schema *schema = ctx->DBs[j].Schema;
			_assert(schema);
			if (dbName && _strcmp(dbName, ctx->DBs[j].Name))
				continue;
			_assert(Btree::SchemaMutexHeld(ctx, j, 0));
			p = (Index *)schema->IndexHash.Find(name, nameLength);
			if (p)
				break;
		}
		return p;
	}

	__device__ static void FreeIndex(Context *ctx, Index *index)
	{
#ifndef OMIT_ANALYZE
		Analyze::DeleteIndexSamples(ctx, index);
#endif
		_tagfree(ctx, index->ColAff);
		_tagfree(ctx, index);
	}

	__device__ void Parse::UnlinkAndDeleteIndex(Context *ctx, int db, const char *indexName)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		Hash *hash = &ctx->DBs[db].Schema->IndexHash;
		int indexNameLength = _strlen30(indexName);
		Index *index = (Index *)hash->Insert(indexName, indexNameLength, 0);
		if (_ALWAYS(index))
		{
			if (index->Table->Index == index)
				index->Table->Index = index->Next;
			else
			{
				// Justification of ALWAYS();  The index must be on the list of indices.
				Index *p = index->Table->Index;
				while (_ALWAYS(p) && p->Next != index)
					p = p->Next;
				if (_ALWAYS(p && p->Next == index))
					p->Next = index->Next;
			}
			FreeIndex(ctx, index);
		}
		ctx->Flags |= Context::FLAG_InternChanges;
	}

	__device__ void Parse::CollapseDatabaseArray(Context *ctx)
	{
		int i, j;
		for (i = j = 2; i < ctx->DBs.length; i++)
		{
			Context::DB *db = &ctx->DBs[i];
			if (!db->Bt)
			{
				_tagfree(ctx, db->Name);
				db->Name = nullptr;
				continue;
			}
			if (j < i)
				ctx->DBs[j] = ctx->DBs[i];
			j++;
		}
		_memset(&ctx->DBs[j], 0, (ctx->DBs.length - j) * sizeof(ctx->DBs[0]));
		ctx->DBs.length = j;
		if (ctx->DBs.length <= 2 && ctx->DBs.data != ctx->DBStatics)
		{
			_memcpy(ctx->DBStatics, ctx->DBs.data, 2 * sizeof(ctx->DBs[0]));
			_tagfree(ctx, ctx->DBs);
			ctx->DBs = ctx->DBStatics;
		}
	}

	__device__ void Parse::ResetOneSchema(Context *ctx, int db)
	{
		_assert(db < ctx->DBs.length);
		// Case 1:  Reset the single schema identified by iDb
		Context::DB *dbobj = &ctx->DBs[db];
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		_assert(dbobj->Schema);
		Callback::SchemaClear(dbobj->Schema);
		// If any database other than TEMP is reset, then also reset TEMP since TEMP might be holding triggers that reference tables in the other database.
		if (db != 1)
		{
			dbobj = &ctx->DBs[1];
			_assert(dbobj->Schema);
			Callback::SchemaClear(dbobj->Schema);
		}
	}

	__device__ void Parse::ResetAllSchemasOfConnection(Context *ctx)
	{
		// Case 2 (from here to the end): Reset all schemas for all attached databases.
		Btree::EnterAll(ctx);
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Context::DB *db = &ctx->DBs[i];
			if (db->Schema)
				Callback::SchemaClear(db->Schema);
		}
		ctx->Flags &= ~Context::FLAG_InternChanges;
		VTable::UnlockList(ctx);
		Btree::LeaveAll(ctx);
		CollapseDatabaseArray(ctx);
	}

	__device__ void Parse::CommitInternalChanges(Context *ctx)
	{
		ctx->Flags &= ~Context::FLAG_InternChanges;
	}

	__device__ static void DeleteColumnNames(Context *ctx, Table *table)
	{
		_assert(table);
		Column *col;
		if ((col = table->Cols) != nullptr)
		{
			for (int i = 0; i < table->Cols.length; i++, col++)
			{
				_tagfree(ctx, col->Name);
				Expr::Delete(ctx, col->Dflt);
				_tagfree(ctx, col->DfltName);
				_tagfree(ctx, col->Type);
				_tagfree(ctx, col->Coll);
			}
			_tagfree(ctx, table->Cols);
		}
	}

	__device__ void Parse::DeleteTable(Context *ctx, Table *table)
	{
		_assert(!table || table->Refs > 0);

		// Do not delete the table until the reference count reaches zero.
		if (!table) return;
		if ((!ctx || ctx->BytesFreed == 0) && --table->Refs > 0) return;

		// Record the number of outstanding lookaside allocations in schema Tables prior to doing any free() operations.  Since schema Tables do not use
		// lookaside, this number should not change.
#if !DEBUG
		int lookaside = (ctx && (table->TabFlags & TF_Ephemeral) == 0 ? ctx->Lookaside.Outs : 0); // Used to verify lookaside not used for schema 
#endif

		// Delete all indices associated with this table.
		Index *index, *next;
		for (index = table->Index; index; index = next)
		{
			next = index->Next;
			_assert(index->Schema == table->Schema);
			if (!ctx || ctx->BytesFreed == 0)
			{
				char *name = index->Name; 
				ASSERTONLY(Index *oldIndex = (Index *)) index->Schema->IndexHash.Insert(name, _strlen30(name), nullptr);
				_assert(!ctx || Btree::SchemaMutexHeld(ctx, 0, index->Schema));
				_assert(!oldIndex || oldIndex == index);
			}
			FreeIndex(ctx, index);
		}

		// Delete any foreign keys attached to this table.
		FKDelete(ctx, table);

		// Delete the Table structure itself.
		DeleteColumnNames(ctx, table);
		_tagfree(ctx, table->Name);
		_tagfree(ctx, table->ColAff);
		Select::Delete(ctx, table->Select);
#ifndef OMIT_CHECK
		Expr::ListDelete(ctx, table->Check);
#endif
#ifndef OMIT_VIRTUALTABLE
		VTable::Clear(ctx, table);
#endif
		_tagfree(ctx, table);

		// Verify that no lookaside memory was used by schema tables
		_assert(!lookaside || lookaside == ctx->Lookaside.Outs);
	}

	__device__ void Parse::UnlinkAndDeleteTable(Context *ctx, int db, const char *tableName)
	{
		_assert(ctx);
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(tableName);
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		ASSERTCOVERAGE(tableName[0] == 0);  // Zero-length table names are allowed
		Context::DB *db2 = &ctx->DBs[db];
		Table *table = (Table *)db2->Schema->TableHash.Insert(tableName, _strlen30(tableName), nullptr);
		DeleteTable(ctx, table);
		ctx->Flags |= Context::FLAG_InternChanges;
	}

	__device__ char *Parse::NameFromToken(Context *ctx, Token *name)
	{
		if (name)
		{
			char *nameAsString = _tagstrndup(ctx, name->data, name->length);
			Parse::Dequote(nameAsString);
			return nameAsString;
		}
		return nullptr;
	}

	__device__ void Parse::OpenMasterTable(int db)
	{
		Vdbe *v = GetVdbe();
		TableLock(db, MASTER_ROOT, true, SCHEMA_TABLE(db));
		v->AddOp3(OP_OpenWrite, 0, MASTER_ROOT, db);
		v->ChangeP4(-1, (char *)5, Vdbe::P4T_INT32);  // 5 column table
		if (Tabs == 0)
			Tabs = 1;
	}

	__device__ int Parse::FindDbName(Context *ctx, const char *name)
	{
		int db = -1; // Database number
		if (name)
		{
			int nameLength = _strlen30(name);
			Context::DB *db2;
			for (db = (ctx->DBs.length - 1), db2 = &ctx->DBs[db]; db >= 0; db--, db2--)
				if ((!E_OMIT_TEMPDB || db != 1) && _strlen30(db2->Name) == nameLength && !_strcmp(db2->Name, name))
					break;
		}
		return db;
	}

	__device__ int Parse::FindDb(Context *ctx, Token *name)
	{
		char *nameAsString = NameFromToken(ctx, name); // Name we are searching for
		int db = FindDbName(ctx, nameAsString); // Database number                    
		_tagfree(ctx, nameAsString);
		return db;
	}

	__device__ int Parse::TwoPartName(Token *name1, Token *name2, Token **unqual)
	{
		Context *ctx = Ctx;
		int db; // Database holding the object
		if (_ALWAYS(name2 != nullptr) && name2->length > 0)
		{
			if (ctx->Init.Busy)
			{
				ErrorMsg("corrupt database");
				Errs++;
				return -1;
			}
			*unqual = name2;
			db = FindDb(ctx, name1);
			if (db < 0)
			{
				ErrorMsg("unknown database %T", name1);
				Errs++;
				return -1;
			}
		}
		else
		{
			_assert(ctx->Init.DB == 0 || ctx->Init.Busy);
			db = (int)ctx->Init.DB;
			*unqual = name1;
		}
		return db;
	}

	__device__ RC Parse::CheckObjectName(const char *name)
	{
		if (!Ctx->Init.Busy && Nested == 0 && (Ctx->Flags & Context::FLAG_WriteSchema) == 0 && !_strncmp(name, "sqlite_", 7))
		{
			ErrorMsg("object name reserved for internal use: %s", name);
			return RC_ERROR;
		}
		return RC_OK;
	}

	__device__ void Parse::StartTable(Token *name1, Token *name2, bool isTemp, bool isView, bool isVirtual, bool noErr)
	{
		// The table or view name to create is passed to this routine via tokens pName1 and pName2. If the table name was fully qualified, for example:
		//
		// CREATE TABLE xxx.yyy (...);
		// 
		// Then pName1 is set to "xxx" and pName2 "yyy". On the other hand if the table name is not fully qualified, i.e.:
		//
		// CREATE TABLE yyy(...);
		//
		// Then pName1 is set to "yyy" and pName2 is "".
		//
		// The call below sets the unqual pointer to point at the token (pName1 or pName2) that stores the unqualified table name. The variable iDb is
		// set to the index of the database that the table or view is to be created in.
		Token *unqual; // Unqualified name of the table to create
		int db = TwoPartName(name1, name2, &unqual); // Database number to create the table in
		if (db < 0)
			return;
		if (!E_OMIT_TEMPDB && isTemp && name2->length > 0 && db != 1)
		{
			// If creating a temp table, the name may not be qualified. Unless the database name is "temp" anyway. 
			ErrorMsg("temporary table name must be unqualified");
			return;
		}
		if (!E_OMIT_TEMPDB && isTemp)
			db = 1;
		NameToken = *unqual;
		Context *ctx = Ctx;
		char *name = NameFromToken(ctx, unqual); // The name of the new table
		if (!name)
			return;
		Vdbe *v;
		Table *table;
		if (CheckObjectName(name) != RC_OK)
			goto begin_table_error;
		if (ctx->Init.DB == 1)
			isTemp = true;
#ifndef OMIT_AUTHORIZATION
		//_assert((isTemp & 1) == isTemp);
		{
			AUTH code;
			char *dbName = ctx->DBs[db].Name;
			if (Auth::Check(this, AUTH_INSERT, SCHEMA_TABLE(isTemp), 0, dbName))
				goto begin_table_error;
			if (isView)
				code = (!E_OMIT_TEMPDB && isTemp ? AUTH_CREATE_TEMP_VIEW : AUTH_CREATE_VIEW);
			else
				code = (!E_OMIT_TEMPDB && isTemp ? AUTH_CREATE_TEMP_TABLE : AUTH_CREATE_TABLE);
			if (!isVirtual && Auth::Check(this, code, name, nullptr, dbName))
				goto begin_table_error;
		}
#endif

		// Make sure the new table name does not collide with an existing index or table name in the same database.  Issue an error message if
		// it does. The exception is if the statement being parsed was passed to an sqlite3_declare_vtab() call. In that case only the column names
		// and types will be used, so there is no need to test for namespace collisions.
		if (!INDECLARE_VTABLE(this))
		{
			char *dbName = ctx->DBs[db].Name;
			if (Prepare::ReadSchema(this) != RC_OK)
				goto begin_table_error;
			table = FindTable(ctx, name, dbName);
			if (table)
			{
				if (!noErr)
					ErrorMsg("table %T already exists", unqual);
				else
				{
					_assert(!ctx->Init.Busy);
					CodeVerifySchema(db);
				}
				goto begin_table_error;
			}
			if (FindIndex(ctx, name, dbName))
			{
				ErrorMsg("there is already an index named %s", name);
				goto begin_table_error;
			}
		}

		table = (Table *)_tagalloc2(ctx, sizeof(Table), true);
		if (!table)
		{
			ctx->MallocFailed = true;
			RC = RC_NOMEM;
			Errs++;
			goto begin_table_error;
		}
		table->Name = name;
		table->PKey = -1;
		table->Schema = ctx->DBs[db].Schema;
		table->Refs = 1;
		table->RowEst = 1000000;
		_assert(!NewTable);
		NewTable = table;

		// If this is the magic sqlite_sequence table used by autoincrement, then record a pointer to this table in the main database structure
		// so that INSERT can find the table easily.
#ifndef OMIT_AUTOINCREMENT
		if (!Nested && !_strcmp(name, "sqlite_sequence"))
		{
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			table->Schema->SeqTable = table;
		}
#endif

		// Begin generating the code that will insert the table record into the SQLITE_MASTER table.  Note in particular that we must go ahead
		// and allocate the record number for the table entry now.  Before any PRIMARY KEY or UNIQUE keywords are parsed.  Those keywords will cause
		// indices to be created and the table record must come before the indices.  Hence, the record number for the table must be allocated now.
		if (!ctx->Init.Busy && (v = GetVdbe()) != nullptr)
		{
			BeginWriteOperation(0, db);

#ifndef OMIT_VIRTUALTABLE
			if (isVirtual)
				v->AddOp0(OP_VBegin);
#endif

			// If the file format and encoding in the database have not been set, set them now.
			int reg1 = RegRowid = ++Mems;
			int reg2 = RegRoot = ++Mems;
			int reg3 = ++Mems;
			v->AddOp3(OP_ReadCookie, db, reg3, Btree::META_FILE_FORMAT);
			v->UsesBtree(db);
			int j1 = v->AddOp1(OP_If, reg3);
			int fileFormat = ((ctx->Flags & Context::FLAG_LegacyFileFmt) != 0 ? 1 : MAX_FILE_FORMAT);
			v->AddOp2(OP_Integer, fileFormat, reg3);
			v->AddOp3(OP_SetCookie, db, Btree::META_FILE_FORMAT, reg3);
			v->AddOp2(OP_Integer, CTXENCODE(ctx), reg3);
			v->AddOp3(OP_SetCookie, db, Btree::META_TEXT_ENCODING, reg3);
			v->JumpHere(j1);

			// This just creates a place-holder record in the sqlite_master table. The record created does not contain anything yet.  It will be replaced
			// by the real entry in code generated at sqlite3EndTable().
			//
			// The rowid for the new entry is left in register pParse->regRowid. The root page number of the new table is left in reg pParse->regRoot.
			// The rowid and root page number values are needed by the code that sqlite3EndTable will generate.
#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
			if (isView || isVirtual)
				v->AddOp2(OP_Integer, 0, reg2);
			else
#endif
				v->AddOp2(OP_CreateTable, db, reg2);
			OpenMasterTable(db);
			v->AddOp2(OP_NewRowid, 0, reg1);
			v->AddOp2(OP_Null, 0, reg3);
			v->AddOp3(OP_Insert, 0, reg3, reg1);
			v->ChangeP5(Vdbe::OPFLAG_APPEND);
			v->AddOp0(OP_Close);
		}
		return;

begin_table_error:
		_tagfree(ctx, name);
		return;
	}

	#define FSTRCMP(x, y) (__curtCtypeMap[*(unsigned char *)(x)] == __curtCtypeMap[*(unsigned char *)(y)] && !_strcmp((x)+1,(y)+1))
	__device__ void Parse::AddColumn(Token *name)
	{
		Table *table;
		if ((table = NewTable) == nullptr)
			return;
		Context *ctx = Ctx;
#if MAX_COLUMN
		if (table->Cols.length + 1 > ctx->Limits[LIMIT_COLUMN])
		{
			ErrorMsg("too many columns on %s", table->Name);
			return;
		}
#endif
		char *nameAsString = NameFromToken(ctx, name);
		if (!nameAsString)
			return;
		for (int i = 0; i < table->Cols.length; i++)
		{
			if (FSTRCMP(nameAsString, table->Cols[i].Name))
			{
				ErrorMsg("duplicate column name: %s", nameAsString);
				_tagfree(ctx, nameAsString);
				return;
			}
		}
		if ((table->Cols.length & 0x7) == 0)
		{
			Column *newCols = (Column *)_tagrealloc(ctx, table->Cols, (table->Cols.length + 8) * sizeof(table->Cols[0]));
			if (!newCols)
			{
				_tagfree(ctx, nameAsString);
				return;
			}
			table->Cols = newCols;
		}
		Column *col = &table->Cols[table->Cols.length];
		_memset(col, 0, sizeof(table->Cols[0]));
		col->Name = nameAsString;

		// If there is no type specified, columns have the default affinity 'NONE'. If there is a type specified, then sqlite3AddColumnType() will
		// be called next to set pCol->affinity correctly.
		col->Affinity = AFF_NONE;
		table->Cols.length++;
	}

	__device__ void Parse::AddNotNull(OE onError)
	{
		Table *table = NewTable;
		if (!table || _NEVER(table->Cols.length < 1))
			return;
		table->Cols[table->Cols.length - 1].NotNull = onError;
	}

	__device__ AFF Parse::AffinityType(const char *data)
	{
		uint32 h = 0;
		AFF aff = AFF_NUMERIC;
		if (data)
			while (data[0])
			{
				h = (h << 8) + _tolower((*data) & 0xff);
				data++;
				if (h == (('c'<<24)+('h'<<16)+('a'<<8)+'r')) aff = AFF_TEXT; // CHAR
				else if (h == (('c'<<24)+('l'<<16)+('o'<<8)+'b')) aff = AFF_TEXT; // CLOB
				else if (h == (('t'<<24)+('e'<<16)+('x'<<8)+'t')) aff = AFF_TEXT; // TEXT
				else if (h == (('b'<<24)+('l'<<16)+('o'<<8)+'b') && (aff == AFF_NUMERIC || aff == AFF_REAL)) aff = AFF_NONE; // BLOB
#ifndef OMIT_FLOATING_POINT
				else if (h == (('r'<<24)+('e'<<16)+('a'<<8)+'l') && aff == AFF_NUMERIC) aff = AFF_REAL; // REAL
				else if (h == (('f'<<24)+('l'<<16)+('o'<<8)+'a') && aff == AFF_NUMERIC) aff = AFF_REAL; // FLOA
				else if (h == (('d'<<24)+('o'<<16)+('u'<<8)+'b') && aff == AFF_NUMERIC) aff = AFF_REAL; // DOUB
#endif
				else if ((h & 0x00FFFFFF) == (('i'<<16)+('n'<<8)+'t')) { aff = AFF_INTEGER; break; }  // INT
			}
			return aff;
	}

	__device__ void Parse::AddColumnType(Token *type)
	{
		Table *table = NewTable;
		if (!table || _NEVER(table->Cols.length < 1))
			return;
		Column *col = &table->Cols[table->Cols.length - 1];
		_assert(col->Type == nullptr);
		col->Type = NameFromToken(Ctx, type);
		col->Affinity = AffinityType(col->Type);
	}

	__device__ void Parse::AddDefaultValue(ExprSpan *span)
	{
		Context *ctx = Ctx;
		Table *table = NewTable;
		if (table)
		{
			Column *col = &(table->Cols[table->Cols.length - 1]);
			if (!span->Expr->IsConstantOrFunction())
				ErrorMsg("default value of column [%s] is not constant", col->Name);
			else
			{
				// A copy of pExpr is used instead of the original, as pExpr contains tokens that point to volatile memory. The 'span' of the expression
				// is required by pragma table_info.
				Expr::Delete(ctx, col->Dflt);
				col->Dflt = Expr::Dup(ctx, span->Expr, EXPRDUP_REDUCE);
				_tagfree(ctx, col->DfltName);
				col->DfltName = _tagstrndup(ctx, (char *)span->Start, (int)(span->End - span->Start));
			}
		}
		Expr::Delete(ctx, span->Expr);
	}

	__device__ void Parse::AddPrimaryKey(ExprList *list, OE onError, bool autoInc, SO sortOrder)
	{
		Table *table = NewTable;
		if (!table || INDECLARE_VTABLE(this))
			goto primary_key_exit;
		if ((table->TabFlags & TF_HasPrimaryKey) != 0)
		{
			ErrorMsg("table \"%s\" has more than one primary key", table->Name);
			goto primary_key_exit;
		}
		table->TabFlags |= TF_HasPrimaryKey;
		int col = -1;
		if (!list)
		{
			col = table->Cols.length - 1;
			table->Cols[col].ColFlags |= COLFLAG_PRIMKEY;
		}
		else
		{
			for (int i = 0; i < list->Exprs; i++)
			{
				for (col = 0; col < table->Cols.length; col++)
					if (!_strcmp(list->Ids[i].Name, table->Cols[col].Name))
						break;
				if (col < table->Cols.length)
					table->Cols[col].ColFlags |= COLFLAG_PRIMKEY;
			}
			if (list->Exprs > 1)
				col = -1;
		}
		char *type = nullptr;
		if (col >= 0 && col < table->Cols.length)
			type = table->Cols[col].Type;
		if (type && !_strcmp(type, "INTEGER") && sortOrder == SO_ASC)
		{
			table->PKey = col;
			table->KeyConf = onError;
			table->TabFlags |= (autoInc ? TF_Autoincrement : 0);
		}
		else if (autoInc)
		{
#ifndef OMIT_AUTOINCREMENT
			ErrorMsg("AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY");
#endif
		}
		else
		{
			Index *index = CreateIndex(0, 0, 0, list, onError, 0, 0, sortOrder, 0);
			if (index)
				index->AutoIndex = 2;
			list = nullptr;
		}
primary_key_exit:
		Expr::ListDelete(Ctx, list);
		return;
	}

	__device__ void Parse::AddCheckConstraint(Expr *checkExpr)
	{
#ifndef OMIT_CHECK
		Table *table = NewTable;
		if (table && !INDECLARE_VTABLE(this))
		{
			table->Check = Expr::ListAppend(this, table->Check, checkExpr);
			if (ConstraintName.length)
				Expr::ListSetName(this, table->Check, &ConstraintName, true);
		}
		else
#endif
			Expr::Delete(Ctx, checkExpr);
	}

	__device__ void Parse::AddCollateType(Token *token)
	{
		Table *table;
		if (!(table = NewTable))
			return;
		int col = table->Cols.length - 1;
		Context *ctx = Ctx;
		char *collName = NameFromToken(ctx, token); // Dequoted name of collation sequence
		if (!collName)
			return;
		if (LocateCollSeq(collName))
		{
			table->Cols[col].Coll = collName;
			// If the column is declared as "<name> PRIMARY KEY COLLATE <type>", then an index may have been created on this column before the
			// collation type was added. Correct this if it is the case.
			for (Index *index = table->Index; index; index = index->Next)
			{
				_assert(index->Columns.length == 1);
				if (index->Columns[0] == col)
					index->CollNames[0] = table->Cols[col].Coll;
			}
		}
		else
			_tagfree(ctx, collName);
	}

	__device__ CollSeq *Parse::LocateCollSeq(const char *name)
	{
		Context *ctx = Ctx;
		TEXTENCODE encode = CTXENCODE(ctx);
		bool initBusy = ctx->Init.Busy;
		CollSeq *coll = Callback::FindCollSeq(ctx, encode, name, initBusy);
		if (!initBusy && (!coll || !coll->Cmp))
			coll = Callback::GetCollSeq(this, encode, coll, name);
		return coll;
	}

	__device__ void Parse::ChangeCookie(int db)
	{
		int r1 = Expr::GetTempReg(this);
		Context *ctx = Ctx;
		Vdbe *v = V;
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		v->AddOp2(OP_Integer, ctx->DBs[db].Schema->SchemaCookie + 1, r1);
		v->AddOp3(OP_SetCookie, db, Btree::META_SCHEMA_VERSION, r1);
		Expr::ReleaseTempReg(this, r1);
	}

	__device__ static int IdentLength(const char *z)
	{
		int size;
		for (size = 0; *z; size++, z++)
			if (*z == '"')
				size++;
		return size + 2;
	}

	__device__ static void IdentPut(char *z, int *idx, char *signedIdent)
	{
		unsigned char *ident = (unsigned char *)signedIdent;
		int i = *idx;
		int j;
		for (j = 0; ident[j]; j++)
			if (!_isalnum(ident[j]) && ident[j] != '_')
				break;
		bool needQuote = (_isdigit(ident[0]) || KeywordCode((const char *)ident, j) != TK_ID);
		if (!needQuote) needQuote = (ident[j] != 0);
		else z[i++] = '"';
		for (j = 0; ident[j]; j++)
		{
			z[i++] = ident[j];
			if (ident[j] == '"') z[i++] = '"';
		}
		if (needQuote) z[i++] = '"';
		z[i] = 0;
		*idx = i;
	}

	__constant__ static const char * const _createTableStmt_Types[] = {
		" TEXT",	// AFF_TEXT
		"",			// AFF_NONE
		" NUM",		// AFF_NUMERIC
		" INT",		// AFF_INTEGER
		" REAL"};	// AFF_REAL   
	__device__ static char *CreateTableStmt(Context *ctx, Table *table)
	{		
		Column *col;
		int i, n = 0;
		for (col = table->Cols, i = 0; i < table->Cols.length; i++, col++)
			n += IdentLength(col->Name) + 5;
		n += IdentLength(table->Name);
		char *sep, *sep2, *end;
		if (n < 50)
		{ 
			sep = "";
			sep2 = ",";
			end = ")";
		}
		else
		{
			sep = "\n  ";
			sep2 = ",\n  ";
			end = "\n)";
		}
		n += 35 + 6 * table->Cols.length;
		char *stmt = (char *)_tagalloc(0, n);
		if (!stmt)
		{
			ctx->MallocFailed = true;
			return nullptr;
		}
		__snprintf(stmt, n, "CREATE TABLE ");
		int k = _strlen30(stmt);
		IdentPut(stmt, &k, table->Name);
		stmt[k++] = '(';
		for (col = table->Cols, i = 0; i < table->Cols.length; i++, col++)
		{
			__snprintf(&stmt[k], n - k, sep);
			k += _strlen30(&stmt[k]);
			sep = sep2;
			IdentPut(stmt, &k, col->Name);
			_assert(col->Affinity - AFF_TEXT >= 0);
			_assert(col->Affinity - AFF_TEXT < _lengthof(_createTableStmt_Types));
			ASSERTCOVERAGE(col->Affinity == AFF_TEXT);
			ASSERTCOVERAGE(col->Affinity == AFF_NONE);
			ASSERTCOVERAGE(col->Affinity == AFF_NUMERIC);
			ASSERTCOVERAGE(col->Affinity == AFF_INTEGER);
			ASSERTCOVERAGE(col->Affinity == AFF_REAL);

			const char *type = _createTableStmt_Types[col->Affinity - AFF_TEXT];
			int typeLength = _strlen30(type);
			_assert(col->Affinity == AFF_NONE || col->Affinity == Parse::AffinityType(type));
			_memcpy(&stmt[k], type, typeLength);
			k += typeLength;
			_assert(k <= n);
		}
		__snprintf(&stmt[k], n - k, "%s", end);
		return stmt;
	}

	__device__ void Parse::EndTable(Token *cons, Token *end, Select *select)
	{
		Context *ctx = Ctx;
		if ((!end && !select) || ctx->MallocFailed)
			return;
		Table *table = NewTable;
		if (!table)
			return;
		_assert(!ctx->Init.Busy || !select);

		int db = Prepare::SchemaToIndex(ctx, table->Schema);
#ifndef OMIT_CHECK
		// Resolve names in all CHECK constraint expressions.
		if (table->Check)
		{
			SrcList src; // Fake SrcList for pParse->pNewTable
			NameContext nc; // Name context for pParse->pNewTable
			_memset(&nc, 0, sizeof(nc));
			_memset(&src, 0, sizeof(src));
			src.Srcs = 1;
			src.Ids[0].Name = table->Name;
			src.Ids[0].Table = table;
			src.Ids[0].Cursor = -1;
			nc.Parse = this;
			nc.SrcList = &src;
			nc.NCFlags = NC_IsCheck;
			ExprList *list = table->Check; // List of all CHECK constraints
			for (int i = 0; i < list->Exprs; i++)
				if (Walker::ResolveExprNames(&nc, list->Ids[i].Expr))
					return;
		}
#endif

		// If the db->init.busy is 1 it means we are reading the SQL off the "sqlite_master" or "sqlite_temp_master" table on the disk.
		// So do not write to the disk again.  Extract the root page number for the table from the db->init.newTnum field.  (The page number
		// should have been put there by the sqliteOpenCb routine.)
		if (ctx->Init.Busy)
			table->Id = ctx->Init.NewTid;

		// If not initializing, then create a record for the new table in the SQLITE_MASTER table of the database.
		// If this is a TEMPORARY table, write the entry into the auxiliary file instead of into the main database file.
		if (!ctx->Init.Busy)
		{
			Vdbe *v = GetVdbe();
			if (_NEVER(v == 0))
				return;
			v->AddOp1(OP_Close, 0);
			// Initialize zType for the new view or table.
			char *type; // "view" or "table" */
			char *type2; // "VIEW" or "TABLE" */
			if (!table->Select)
			{
				// A regular table
				type = "table";
				type2 = "TABLE";
			}
#ifndef OMIT_VIEW
			else
			{
				// A view
				type = "view";
				type2 = "VIEW";

			}
#endif

			// If this is a CREATE TABLE xx AS SELECT ..., execute the SELECT statement to populate the new table. The root-page number for the
			// new table is in register pParse->regRoot.
			//
			// Once the SELECT has been coded by sqlite3Select(), it is in a suitable state to query for the column names and types to be used
			// by the new table.
			//
			// A shared-cache write-lock is not required to write to the new table, as a schema-lock must have already been obtained to create it. Since
			// a schema-lock excludes all other database users, the write-lock would be redundant.
			if (select)
			{
				_assert(Tabs == 1);
				v->AddOp3(OP_OpenWrite, 1, RegRoot, db);
				v->ChangeP5(Vdbe::OPFLAG_P2ISREG);
				Tabs = 2;
				SelectDest dest;
				Select::DestInit(&dest, SRT_Table, 1);
				Select::Select_(this, select, &dest);
				v->AddOp1(OP_Close, 1);
				if (Errs == 0)
				{
					Table *selectTable = select->ResultSetOfSelect(this);
					if (!selectTable)
						return;
					_assert(!table->Cols);
					table->Cols.length = selectTable->Cols.length;
					table->Cols = selectTable->Cols;
					selectTable->Cols.length = 0;
					selectTable->Cols = nullptr;
					DeleteTable(ctx, selectTable);
				}
			}

			// Compute the complete text of the CREATE statement
			int n;
			char *stmt; // Text of the CREATE TABLE or CREATE VIEW statement
			if (select)
				stmt = CreateTableStmt(ctx, table);
			else
			{
				n = (int)(end->data - NameToken.data) + 1;
				stmt = _mtagprintf(ctx, "CREATE %s %.*s", type2, n, NameToken.data);
			}

			// A slot for the record has already been allocated in the SQLITE_MASTER table.  We just need to update that slot with all
			// the information we've collected.
			NestedParse(
				"UPDATE %Q.%s "
				"SET type='%s', name=%Q, tbl_name=%Q, rootpage=#%d, sql=%Q "
				"WHERE rowid=#%d", ctx->DBs[db].Name, SCHEMA_TABLE(db),
				type, table->Name, table->Name, RegRoot, stmt,
				RegRowid);
			_tagfree(ctx, stmt);
			ChangeCookie(db);

#ifndef OMIT_AUTOINCREMENT
			// Check to see if we need to create an sqlite_sequence table for keeping track of autoincrement keys.
			if (table->TabFlags & TF_Autoincrement)
			{
				Context::DB *dbobj = &ctx->DBs[db];
				_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
				if (!dbobj->Schema->SeqTable)
					NestedParse("CREATE TABLE %Q.sqlite_sequence(name,seq)", dbobj->Name);
			}
#endif

			// Reparse everything to update our internal data structures
			v->AddParseSchemaOp(db, _mtagprintf(ctx, "tbl_name='%q'", table->Name));
		}

		// Add the table to the in-memory representation of the database.
		if (ctx->Init.Busy)
		{
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			Schema *schema = table->Schema;
			Table *oldTable = (Table *)schema->TableHash.Insert(table->Name, _strlen30(table->Name), table);
			if (oldTable)
			{
				_assert(table == oldTable);  // Malloc must have failed inside HashInsert()
				ctx->MallocFailed = true;
				return;
			}
			NewTable = nullptr;
			ctx->Flags |= Context::FLAG_InternChanges;
#ifndef OMIT_ALTERTABLE
			if (!table->Select)
			{
				const char *name = (const char *)NameToken.data;
				_assert(!select && cons && end); // SKY::THIS MIGHT FAIL BECAUSE TESTING DATA NOT THE OBJ
				if (!cons->data)
					cons = end;
				int nameLength = (int)((const char *)cons->data - name);
				table->AddColOffset = 13 + _utf8charlength(name, nameLength);
			}
#endif
		}
	}

#ifndef OMIT_VIEW
	__device__ void Parse::CreateView(Token *begin, Token *name1, Token *name2, Select *select, bool isTemp, bool noErr)
	{
		Context *ctx = Ctx;
		if (Vars.length > 0)
		{
			ErrorMsg("parameters are not allowed in views");
			Select::Delete(ctx, select);
			return;
		}
		StartTable(name1, name2, isTemp, 1, 0, noErr);
		Table *table = NewTable;
		if (!table || Errs)
		{
			Select::Delete(ctx, select);
			return;
		}
		Token *name = nullptr;
		TwoPartName(name1, name2, &name);
		int db = Prepare::SchemaToIndex(ctx, table->Schema);
		DbFixer fix;
		if (fix.FixInit(this, db, "view", name) && fix.FixSelect(select))
		{
			Select::Delete(ctx, select);
			return;
		}

		// Make a copy of the entire SELECT statement that defines the view. This will force all the Expr.token.z values to be dynamically
		// allocated rather than point to the input string - which means that they will persist after the current sqlite3_exec() call returns.
		table->Select = Expr::SelectDup(ctx, select, EXPRDUP_REDUCE);
		Select::Delete(ctx, select);
		if (ctx->MallocFailed)
			return;
		if (!ctx->Init.Busy)
			ViewGetColumnNames(table);

		// Locate the end of the CREATE VIEW statement.  Make sEnd point to the end.
		Token sEnd = LastToken;
		if (_ALWAYS(sEnd.data[0] != 0) && sEnd.data[0] != ';')
			sEnd.data += sEnd.length;
		sEnd.length = 0;

		int n = (int)(sEnd.data - begin->data);
		const char *z = begin->data;
		while (_ALWAYS(n > 0) && _isspace(z[n - 1])) { n--; }
		sEnd.data = &z[n - 1];
		sEnd.length = 1;

		// Use sqlite3EndTable() to add the view to the SQLITE_MASTER table
		EndTable(nullptr, &sEnd, nullptr);
		return;
	}
#endif

#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
	__device__ int Parse::ViewGetColumnNames(Table *table)
	{
		_assert(table);
		Context *ctx = Ctx;  // Database connection for malloc errors
#ifndef OMIT_VIRTUALTABLE
		if (VTable::CallConnect(this, table))
			return RC_ERROR;
		if (IsVirtual(table))
			return 0;
#endif
#ifndef OMIT_VIEW
		// A positive nCol means the columns names for this view are already known.
		if (table->Cols.length > 0)
			return 0;

		// A negative nCol is a special marker meaning that we are currently trying to compute the column names.  If we enter this routine with
		// a negative nCol, it means two or more views form a loop, like this:
		//     CREATE VIEW one AS SELECT * FROM two;
		//     CREATE VIEW two AS SELECT * FROM one;
		// Actually, the error above is now caught prior to reaching this point. But the following test is still important as it does come up
		// in the following:
		//     CREATE TABLE main.ex1(a);
		//     CREATE TEMP VIEW ex1 AS SELECT a FROM ex1;
		//     SELECT * FROM temp.ex1;
		if (table->Cols.length < 0)
		{
			ErrorMsg("view %s is circularly defined", table->Name);
			return 1;
		}
		_assert(table->Cols.length >= 0);

		// If we get this far, it means we need to compute the table names. Note that the call to sqlite3ResultSetOfSelect() will expand any
		// "*" elements in the results set of the view and will assign cursors to the elements of the FROM clause.  But we do not want these changes
		// to be permanent.  So the computation is done on a copy of the SELECT statement that defines the view.
		_assert(table->Select);
		int errs = 0; // Number of errors encountered
		Select *select = Expr::SelectDup(ctx, table->Select, 0); // Copy of the SELECT that implements the view
		if (select)
		{
			bool enableLookaside = ctx->Lookaside.Enabled;
			int n = Tabs; // Temporarily holds the number of cursors assigned
			SrcListAssignCursors(select->Src);
			table->Cols.length = -1;
			ctx->Lookaside.Enabled = false;
#ifndef OMIT_AUTHORIZATION
			ARC (*auth)(void*,int,const char*,const char*,const char*,const char*) = ctx->Auth;
			ctx->Auth = nullptr;
			Table *selectTable = select->ResultSetOfSelect(this); // A fake table from which we get the result set
			ctx->Auth = auth;
#else
			Table *selectTable = ResultSetOfSelect(select); // A fake table from which we get the result set
#endif
			ctx->Lookaside.Enabled = enableLookaside;
			Tabs = n;
			if (selectTable)
			{
				_assert(table->Cols.data == nullptr);
				table->Cols.length = selectTable->Cols.length;
				table->Cols.data = selectTable->Cols.data;
				selectTable->Cols.length = 0;
				selectTable->Cols.data = nullptr;
				DeleteTable(ctx, selectTable);
				_assert(Btree::SchemaMutexHeld(ctx, 0, table->Schema));
				table->Schema->Flags |= SCHEMA_UnresetViews;
			}
			else
			{
				table->Cols.length = 0;
				errs++;
			}
			Select::Delete(ctx, select);
		}
		else
			errs++;
#endif
		return errs;  
	}
#endif

#ifndef OMIT_VIEW
	__device__ void Parse::ViewResetAll(Context *ctx, int db)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		if (!DbHasProperty(ctx, db, SCHEMA_UnresetViews))
			return;
		for (HashElem *i = (HashElem *)ctx->DBs[db].Schema->TableHash.First; i; i = i->Next)
		{
			Table *table = (Table *)i->Data;
			if (table->Select)
			{
				DeleteColumnNames(ctx, table);
				table->Cols.data = nullptr;
				table->Cols.length = 0;
			}
		}
		DbClearProperty(ctx, db, SCHEMA_UnresetViews);
	}
#else
	__device__ void Parse::ViewResetAll(Context *ctx, int db) { }
#endif

#ifndef OMIT_AUTOVACUUM
	__device__ void Parse::RootPageMoved(Context *ctx, int db, int from, int to)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		Context::DB *dbobj = &ctx->DBs[db];
		Hash *hash = &dbobj->Schema->TableHash;
		for (HashElem *elem = (HashElem *)hash->First; elem; elem = elem->Next)
		{
			Table *table = (Table *)elem->Data;
			if (table->Id == from)
				table->Id = to;
		}
		hash = &dbobj->Schema->IndexHash;
		for (HashElem *elem = (HashElem *)hash->First; elem; elem = elem->Next)
		{
			Index *index = (Index *)elem->Data;
			if (index->Id == from)
				index->Id = to;
		}
	}
#endif

	__device__ static void DestroyRootPage(Parse *parse, int table, int db)
	{
		Vdbe *v = parse->GetVdbe();
		int r1 = Expr::GetTempReg(parse);
		v->AddOp3(OP_Destroy, table, r1, db);
		parse->MayAbort();
#ifndef OMIT_AUTOVACUUM
		// OP_Destroy stores an in integer r1. If this integer is non-zero, then it is the root page number of a table moved to
		// location iTable. The following code modifies the sqlite_master table to reflect this.
		//
		// The "#NNN" in the SQL is a special constant that means whatever value is in register NNN.  See grammar rules associated with the TK_REGISTER
		// token for additional information.
		parse->NestedParse("UPDATE %Q.%s SET rootpage=%d WHERE #%d AND rootpage=#%d", parse->Ctx->DBs[db].Name, SCHEMA_TABLE(db), table, r1, r1);
#endif
		Expr::ReleaseTempReg(parse, r1);
	}

	__device__ static void DestroyTable(Parse *parse, Table *table)
	{
#ifdef OMIT_AUTOVACUUM
		int db = SchemaToIndex(parse->Ctx, table->Schema);
		DestroyRootPage(parse, table->Id, db);
		for (Index *index = table->Index; index; index = index->Next)
			DestroyRootPage(parse, index->Id, db);
#else
		// If the database may be auto-vacuum capable (if SQLITE_OMIT_AUTOVACUUM is not defined), then it is important to call OP_Destroy on the
		// table and index root-pages in order, starting with the numerically largest root-page number. This guarantees that none of the root-pages
		// to be destroyed is relocated by an earlier OP_Destroy. i.e. if the following were coded:
		//
		// OP_Destroy 4 0
		// ...
		// OP_Destroy 5 0
		//
		// and root page 5 happened to be the largest root-page number in the database, then root page 5 would be moved to page 4 by the 
		// "OP_Destroy 4 0" opcode. The subsequent "OP_Destroy 5 0" would hit a free-list page.
		int tableId = table->Id;
		int destroyedId = 0;
		while (true)
		{
			int largestId = 0;
			if (destroyedId == 0 || tableId < destroyedId)
				largestId = tableId;
			for (Index *index = table->Index; index; index = index->Next)
			{
				int indexId = index->Id;
				_assert(index->Schema == table->Schema);
				if ((destroyedId == 0 || indexId < destroyedId) && indexId > largestId)
					largestId = indexId;
			}
			if (largestId == 0)
				return;
			else
			{
				int db = Prepare::SchemaToIndex(parse->Ctx, table->Schema);
				_assert(db >= 0 && db < parse->Ctx->DBs.length);
				DestroyRootPage(parse, largestId, db);
				destroyedId = largestId;
			}
		}
#endif
	}

	__device__ void Parse::ClearStatTables(int db, const char *type, const char *name)
	{
		const char *dbName = Ctx->DBs[db].Name;
		for (int i = 1; i <= 3; i++)
		{
			char tableName[24];
			__snprintf(tableName, sizeof(tableName), "sqlite_stat%d", i);
			if (FindTable(Ctx, tableName, dbName))
				NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", dbName, tableName, type, name);
		}
	}

	__device__ void Parse::CodeDropTable(Table *table, int db, bool isView)
	{
		Context *ctx = Ctx;
		Context::DB *dbobj = &ctx->DBs[db];

		Vdbe *v = GetVdbe();
		_assert(v);
		BeginWriteOperation(1, db);

#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(table))
			v->AddOp0(OP_VBegin);
#endif

		// Drop all triggers associated with the table being dropped. Code is generated to remove entries from sqlite_master and/or
		// sqlite_temp_master if required.
		Trigger *trigger = Trigger::List(this, table);
		while (trigger)
		{
			_assert(trigger->Schema == table->Schema || trigger->Schema == ctx->DBs[1].Schema);
			Trigger::DropTriggerPtr(this, trigger);
			trigger = trigger->Next;
		}

#ifndef OMIT_AUTOINCREMENT
		// Remove any entries of the sqlite_sequence table associated with the table being dropped. This is done before the table is dropped
		// at the btree level, in case the sqlite_sequence table needs to move as a result of the drop (can happen in auto-vacuum mode).
		if (table->TabFlags & TF_Autoincrement)
			NestedParse("DELETE FROM %Q.sqlite_sequence WHERE name=%Q", dbobj->Name, table->Name);
#endif

		// Drop all SQLITE_MASTER table and index entries that refer to the table. The program name loops through the master table and deletes
		// every row that refers to a table of the same name as the one being dropped. Triggers are handled separately because a trigger can be
		// created in the temp database that refers to a table in another database.
		NestedParse("DELETE FROM %Q.%s WHERE tbl_name=%Q and type!='trigger'", dbobj->Name, SCHEMA_TABLE(db), table->Name);
		if (!isView && !IsVirtual(table))
			DestroyTable(this, table);

		// Remove the table entry from SQLite's internal schema and modify the schema cookie.
		if (IsVirtual(table))
			v->AddOp4(OP_VDestroy, db, 0, 0, table->Name, 0);
		v->AddOp4(OP_DropTable, db, 0, 0, table->Name, 0);
		ChangeCookie(db);
		ViewResetAll(ctx, db);
	}

	__device__ void Parse::DropTable(SrcList *name, bool isView, bool noErr)
	{
		Context *ctx = Ctx;
		if (ctx->MallocFailed)
			goto exit_drop_table;

		_assert(Errs == 0);
		_assert(name->Srcs == 1);
		if (noErr)
			ctx->SuppressErr++;
		Table *table = LocateTableItem(isView, &name->Ids[0]);
		if (noErr)
			ctx->SuppressErr--;

		if (!table)
		{
			if (noErr)
				CodeVerifyNamedSchema(name->Ids[0].Database);
			goto exit_drop_table;
		}
		int db = Prepare::SchemaToIndex(ctx, table->Schema);
		_assert(db >= 0 && db < ctx->DBs.length);

		// If pTab is a virtual table, call ViewGetColumnNames() to ensure it is initialized.
		if (IsVirtual(table) && ViewGetColumnNames(table))
			goto exit_drop_table;
#ifndef OMIT_AUTHORIZATION
		{
			const char *tableName = SCHEMA_TABLE(db);
			const char *dbName = ctx->DBs[db].Name;
			if (Auth::Check(this, AUTH_DELETE, tableName, 0, dbName))
				goto exit_drop_table;
			AUTH code;
			const char *arg2;
			if (isView)
			{
				code = (!E_OMIT_TEMPDB && db == 1 ? AUTH_DROP_TEMP_VIEW : AUTH_DROP_VIEW);
				arg2 = 0;
			}
#ifndef OMIT_VIRTUALTABLE
			else if (IsVirtual(table))
			{
				code = AUTH_DROP_VTABLE;
				arg2 = VTable::GetVTable(ctx, table)->Module->Name;
			}
#endif
			else
			{
				code = (!E_OMIT_TEMPDB && db == 1 ? AUTH_DROP_TEMP_TABLE : AUTH_DROP_TABLE);
				arg2 = 0;
			}
			if (Auth::Check(this, code, table->Name, arg2, dbName) || Auth::Check(this, AUTH_DELETE, table->Name, nullptr, dbName))
				goto exit_drop_table;
		}
#endif
		if (!_strncmp(table->Name, "sqlite_", 7) && _strncmp(table->Name, "sqlite_stat", 11))
		{
			ErrorMsg("table %s may not be dropped", table->Name);
			goto exit_drop_table;
		}

#ifndef OMIT_VIEW
		// Ensure DROP TABLE is not used on a view, and DROP VIEW is not used on a table.
		if (isView && !table->Select)
		{
			ErrorMsg("use DROP TABLE to delete table %s", table->Name);
			goto exit_drop_table;
		}
		if (!isView && table->Select)
		{
			ErrorMsg("use DROP VIEW to delete view %s", table->Name);
			goto exit_drop_table;
		}
#endif

		// Generate code to remove the table from the master table on disk.
		Vdbe *v = GetVdbe();
		if (v)
		{
			BeginWriteOperation(1, db);
			ClearStatTables(db, "tbl", table->Name);
			FKDropTable(name, table);
			CodeDropTable(table, db, isView);
		}

exit_drop_table:
		SrcListDelete(ctx, name);
	}

	__device__ void Parse::CreateForeignKey(ExprList *fromCol, Token *to, ExprList *toCol, int flags)
	{
		Context *ctx = Ctx;
#ifndef OMIT_FOREIGN_KEY
		FKey *nextTo;
		Table *table = NewTable;
		int i;

		_assert(!to);
		FKey *fkey = nullptr;
		if (!table || INDECLARE_VTABLE(this))
			goto fk_end;
		int cols;
		if (!fromCol)
		{
			int col = table->Cols.length-1;
			if (_NEVER(col < 0))
				goto fk_end;
			if (toCol && toCol->Exprs != 1)
			{
				ErrorMsg("foreign key on %s should reference only one column of table %T", table->Cols[col].Name, to);
				goto fk_end;
			}
			cols = 1;
		}
		else if (toCol && toCol->Exprs != fromCol->Exprs)
		{
			ErrorMsg("number of columns in foreign key does not match the number of columns in the referenced table");
			goto fk_end;
		}
		else
			cols = fromCol->Exprs;
		int bytes = sizeof(*fkey) + (cols-1)*sizeof(fkey->Cols[0]) + to->length + 1;
		if (toCol)
			for (i = 0; i < toCol->Exprs; i++)
				bytes += _strlen30(toCol->Ids[i].Name) + 1;
		fkey = (FKey *)_tagalloc2(ctx, bytes, true);
		if (!fkey)
			goto fk_end;
		fkey->From = table;
		fkey->NextFrom = table->FKeys;
		char *z = (char *)&fkey->Cols[cols];
		fkey->To = z;
		_memcpy(z, to->data, to->length);
		z[to->length] = 0;
		Dequote(z);
		z += to->length+1;
		fkey->Cols.length = cols;
		if (!fromCol)
			fkey->Cols[0].From = table->Cols.length-1;
		else
		{
			for (i = 0; i < cols; i++)
			{
				int j;
				for (j = 0; j < table->Cols.length; j++)
				{
					if (!_strcmp(table->Cols[j].Name, fromCol->Ids[i].Name))
					{
						fkey->Cols[i].From = j;
						break;
					}
				}
				if (j >= table->Cols.length)
				{
					ErrorMsg("unknown column \"%s\" in foreign key definition", fromCol->Ids[i].Name);
					goto fk_end;
				}
			}
		}
		if (toCol)
		{
			for (i = 0; i < cols; i++)
			{
				int n = _strlen30(toCol->Ids[i].Name);
				fkey->Cols[i].Col = z;
				_memcpy(z, toCol->Ids[i].Name, n);
				z[n] = 0;
				z += n+1;
			}
		}
		fkey->IsDeferred = false;
		fkey->Actions[0] = (OE)(flags & 0xff);            // ON DELETE action
		fkey->Actions[1] = (OE)((flags >> 8 ) & 0xff);    // ON UPDATE action

		_assert(Btree::SchemaMutexHeld(ctx, 0, table->Schema));
		nextTo = (FKey *)table->Schema->FKeyHash.Insert(fkey->To, _strlen30(fkey->To), (void *)fkey);
		if (nextTo == fkey)
		{
			ctx->MallocFailed = true;
			goto fk_end;
		}
		if (nextTo)
		{
			_assert(nextTo->PrevTo == nullptr);
			fkey->NextTo = nextTo;
			nextTo->PrevTo = fkey;
		}

		// Link the foreign key to the table as the last step.
		table->FKeys = fkey;
		fkey = nullptr;

fk_end:
		_tagfree(ctx, fkey);
#endif
		Expr::ListDelete(ctx, fromCol);
		Expr::ListDelete(ctx, toCol);
	}

	__device__ void Parse::DeferForeignKey(bool isDeferred)
	{
#ifndef OMIT_FOREIGN_KEY
		Table *table;
		FKey *fkey;
		if (!(table = NewTable) || !(fkey = table->FKeys)) return;
		fkey->IsDeferred = isDeferred;
#endif
	}

	__device__ void Parse::RefillIndex(Index *index, int memRootPage)
	{
		Context *ctx = Ctx; // The database connection
		int db = Prepare::SchemaToIndex(ctx, index->Schema);

#ifndef OMIT_AUTHORIZATION
		if (Auth::Check(this, AUTH_REINDEX, index->Name, 0, ctx->DBs[db].Name))
			return;
#endif

		// Require a write-lock on the table to perform this operation
		Table *table = index->Table; // The table that is indexed
		TableLock(db, table->Id, 1, table->Name);

		Vdbe *v = GetVdbe(); // Generate code into this virtual machine
		if (!v) return;
		int tid; // Root page of index
		if (memRootPage >= 0)
			tid = memRootPage;
		else
		{
			tid = index->Id;
			v->AddOp2(OP_Clear, tid, db);
		}
		int indexIdx = Tabs++; // Btree cursor used for pIndex
		KeyInfo *key = IndexKeyinfo(index); // KeyInfo for index
		v->AddOp4(OP_OpenWrite, indexIdx, tid, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
		v->ChangeP5(Vdbe::OPFLAG_BULKCSR | (memRootPage >= 0 ? Vdbe::OPFLAG_P2ISREG : 0));

		// Open the sorter cursor if we are to use one.
		int sorterIdx = Tabs++; // Cursor opened by OpenSorter (if in use)
		v->AddOp4(OP_SorterOpen, sorterIdx, 0, 0, (char *)key, Vdbe::P4T_KEYINFO);

		// Open the table. Loop through all rows of the table, inserting index records into the sorter.
		int tableIdx = Tabs++; // Btree cursor used for pTab
		Insert::OpenTable(this, tableIdx, db, table, OP_OpenRead);
		int addr1 = v->AddOp2(OP_Rewind, tableIdx, 0); // Address of top of loop
		int regRecord = Expr::GetTempReg(this); // Register holding assemblied index record

		Delete::GenerateIndexKey(this, index, tableIdx, regRecord, true);
		v->AddOp2(OP_SorterInsert, sorterIdx, regRecord);
		v->AddOp2(OP_Next, tableIdx, addr1+1);
		v->JumpHere(addr1);
		addr1 = v->AddOp2(OP_SorterSort, sorterIdx, 0);
		int addr2; // Address to jump to for next iteration
		if (index->OnError != OE_None)
		{
			int j2 = v->CurrentAddr() + 3;
			v->AddOp2(OP_Goto, 0, j2);
			addr2 = v->CurrentAddr();
			v->AddOp3(OP_SorterCompare, sorterIdx, j2, regRecord);
			HaltConstraint(RC_CONSTRAINT_UNIQUE, OE_Abort, "indexed columns are not unique", Vdbe::P4T_STATIC);
		}
		else
			addr2 = v->CurrentAddr();
		v->AddOp2(OP_SorterData, sorterIdx, regRecord);
		v->AddOp3(OP_IdxInsert, indexIdx, regRecord, 1);
		v->ChangeP5(Vdbe::OPFLAG_USESEEKRESULT);
		Expr::ReleaseTempReg(this, regRecord);
		v->AddOp2(OP_SorterNext, sorterIdx, addr2);
		v->JumpHere(addr1);
		v->AddOp1(OP_Close, tableIdx);
		v->AddOp1(OP_Close, indexIdx);
		v->AddOp1(OP_Close, sorterIdx);
	}

	__device__ Index *Parse::CreateIndex(Token *name1, Token *name2, SrcList *tableName, ExprList *list, OE onError, Token *start, Token *end, int sortOrder, bool ifNotExist)
	{
		_assert(start || !end); // pEnd must be non-NULL if pStart is
		_assert(Errs == 0); // Never called with prior errors
		Index *r = nullptr; // Pointer to return
		Context *ctx = Ctx;
		if (ctx->MallocFailed || INDECLARE_VTABLE(this) || Prepare::ReadSchema(this) != RC_OK)
			goto exit_create_index;

		// Find the table that is to be indexed.  Return early if not found.
		int db; // Index of the database that is being written
		Table *table = nullptr; // Table to be indexed
		DbFixer fix; // For assigning database names to pTable
		Token *nameAsToken = nullptr; // Unqualified name of the index to create
		if (tableName)
		{
			// Use the two-part index name to determine the database to search for the table. 'Fix' the table name to this db
			// before looking up the table.
			_assert(name1 && name2);
			db = TwoPartName(name1, name2, &nameAsToken);
			if (db < 0)
				goto exit_create_index;
			_assert(nameAsToken && nameAsToken->data);
#ifndef OMIT_TEMPDB
			// If the index name was unqualified, check if the table is a temp table. If so, set the database to 1. Do not do this
			// if initialising a database schema.
			if (!ctx->Init.Busy)
			{
				table = Delete::SrcListLookup(this, tableName);
				if (name2->length == 0 && table && table->Schema == ctx->DBs[1].Schema)
					db = 1;
			}
#endif
			// Because the parser constructs pTblName from a single identifier, sqlite3FixSrcList can never fail.
			if (fix.FixInit(this, db, "index", nameAsToken) && fix.FixSrcList(tableName))
				_assert(0);
			table = LocateTableItem(false, &tableName->Ids[0]);
			_assert(!ctx->MallocFailed || !table);
			if (!table)
				goto exit_create_index;
			_assert(ctx->DBs[db].Schema == table->Schema);
		}
		else
		{
			_assert(!nameAsToken);
			_assert(!start);
			table = NewTable;
			if (!table)
				goto exit_create_index;
			db = Prepare::SchemaToIndex(ctx, table->Schema);
		}
		Context::DB *dbobj = &ctx->DBs[db]; // The specific table containing the indexed database
		_assert(table);
		_assert(Errs == 0);
		if (!_strncmp(table->Name, "sqlite_", 7) && _strncmp(&table->Name[7], "altertab_", 9) != 0)
		{
			ErrorMsg("table %s may not be indexed", table->Name);
			goto exit_create_index;
		}
#ifndef OMIT_VIEW
		if (table->Select)
		{
			ErrorMsg("views may not be indexed");
			goto exit_create_index;
		}
#endif
#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(table))
		{
			ErrorMsg("virtual tables may not be indexed");
			goto exit_create_index;
		}
#endif

		// Find the name of the index.  Make sure there is not already another index or table with the same name.  
		//
		// Exception:  If we are reading the names of permanent indices from the sqlite_master table (because some other process changed the schema) and
		// one of the index names collides with the name of a temporary table or index, then we will continue to process this index.
		//
		// If pName==0 it means that we are dealing with a primary key or UNIQUE constraint.  We have to invent our own name.
		char *name = nullptr; // Name of the index
		if (nameAsToken)
		{
			name = NameFromToken(ctx, nameAsToken);
			if (!name)
				goto exit_create_index;
			_assert(nameAsToken->data != nullptr);
			if (CheckObjectName(name) != RC_OK)
				goto exit_create_index;
			if (!ctx->Init.Busy)
			{
				if (FindTable(ctx, name, 0))
				{
					ErrorMsg("there is already a table named %s", name);
					goto exit_create_index;
				}
			}
			if (FindIndex(ctx, name, dbobj->Name))
			{
				if (!ifNotExist)
					ErrorMsg("index %s already exists", name);
				else
				{
					_assert(!ctx->Init.Busy);
					CodeVerifySchema(db);
				}
				goto exit_create_index;
			}
		}
		else
		{
			int n;
			Index *loop;
			for (loop = table->Index, n = 1; loop; loop = loop->Next, n++) { }
			name = _mtagprintf(ctx, "sqlite_autoindex_%s_%d", table->Name, n);
			if (!name)
				goto exit_create_index;
		}

		// Check for authorization to create an index.
		int i;
#ifndef OMIT_AUTHORIZATION
		{
			const char *dbName = dbobj->Name;
			if (Auth::Check(this, AUTH_INSERT, SCHEMA_TABLE(db), 0, dbName))
				goto exit_create_index;
			i = (!E_OMIT_TEMPDB && db == 1 ? AUTH_CREATE_TEMP_INDEX : AUTH_CREATE_INDEX);
			if (Auth::Check(this, (AUTH)i, name, table->Name, dbName))
				goto exit_create_index;
		}
#endif

		// If pList==0, it means this routine was called to make a primary key out of the last column added to the table under construction.
		// So create a fake list to simulate this.
		if (!list)
		{
			Token nullId; // Fake token for an empty ID list
			nullId.data = table->Cols[table->Cols.length-1].Name;
			nullId.length = _strlen30((char *)nullId.data);
			list = Expr::ListAppend(this, nullptr, nullptr);
			if (!list)
				goto exit_create_index;
			Expr::ListSetName(this, list, &nullId, 0);
			list->Ids[0].SortOrder = (SO)sortOrder;
		}

		// Figure out how many bytes of space are required to store explicitly specified collation sequence names.
		int extraLength = 0;
		for (i = 0; i < list->Exprs; i++)
		{
			Expr *expr = list->Ids[i].Expr;
			if (expr)
			{
				CollSeq *coll = expr->CollSeq(this);
				if (coll)
					extraLength += (1 + _strlen30(coll->Name));
			}
		}

		// Allocate the index structure. 
		int nameLength = _strlen30(name); // Number of characters in zName
		int cols = list->Exprs;
		Index *index = (Index *)_tagalloc2(ctx, 
			_ROUND8(sizeof(Index)) +			// Index structure
			_ROUND8(sizeof(tRowcnt)*(cols+1)) +// Index.aiRowEst
			sizeof(char *)*cols +					// Index.azColl
			sizeof(int)*cols +						// Index.aiColumn
			sizeof(uint8)*cols +                    // Index.aSortOrder
			nameLength + 1 +                        // Index.zName
			extraLength, true);						// Collation sequence names
		if (ctx->MallocFailed)
			goto exit_create_index;
		char *extra = (char *)index;
		index->RowEsts = (tRowcnt *)&extra[_ROUND8(sizeof(Index))];
		index->CollNames = (char **)((char *)index->RowEsts + _ROUND8(sizeof(tRowcnt)*cols+1));
		_assert(_HASALIGNMENT8(index->RowEsts) );
		_assert(_HASALIGNMENT8(index->CollNames) );
		index->Columns = (int *)(&index->CollNames[cols]);
		index->SortOrders = (SO *)(&index->Columns[cols]);
		index->Name = (char *)(&index->SortOrders[cols]);
		extra = (char *)(&index->Name[nameLength+1]);
		_memcpy(index->Name, name, nameLength+1);
		index->Table = table;
		index->Columns.length = list->Exprs;
		index->OnError = (OE)onError;
		index->AutoIndex = (byte)(name == nullptr);
		index->Schema = ctx->DBs[db].Schema;
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));

		// Check to see if we should honor DESC requests on index columns
		int sortOrderMask = (dbobj->Schema->FileFormat >= 4 ? -1 : 0); // Honor/Ignore DESC - 1 to honor DESC in index.  0 to ignore.

		// Scan the names of the columns of the table to be indexed and load the column indices into the Index structure.  Report an error
		// if any column is not found.
		//
		// TODO:  Add a test to make sure that the same column is not named more than once within the same index.  Only the first instance of
		// the column will ever be used by the optimizer.  Note that using the same column more than once cannot be an error because that would 
		// break backwards compatibility - it needs to be a warning.
		int j;
		ExprList::ExprListItem *listItem; // For looping over pList
		for (i = 0, listItem = list->Ids; i < list->Exprs; i++, listItem++)
		{
			const char *colName = listItem->Name;
			Column *tableCol;
			for (j = 0, tableCol = table->Cols; j < table->Cols.length; j++, tableCol++)
				if (!_strcmp(colName, tableCol->Name)) break;
			if (j >= table->Cols.length)
			{
				ErrorMsg("table %s has no column named %s", table->Name, colName);
				CheckSchema = 1;
				goto exit_create_index;
			}
			index->Columns[i] = j;
			char *collName; // Collation sequence name
			CollSeq *coll; // Collating sequence
			if (listItem->Expr && (coll = listItem->Expr->CollSeq(this)) != nullptr)
			{
				collName = coll->Name;
				int collNameLength = _strlen30(collName) + 1;
				_assert(extraLength >= collNameLength);
				_memcpy(extra, collName, collNameLength);
				collName = extra;
				extra += collNameLength;
				extraLength -= collNameLength;
			}
			else
			{
				collName = table->Cols[j].Coll;
				if (!collName)
					collName = "BINARY";
			}
			if (!ctx->Init.Busy && !LocateCollSeq(collName))
				goto exit_create_index;
			index->CollNames[i] = collName;
			SO requestedSortOrder = (SO)(listItem->SortOrder & sortOrderMask);
			index->SortOrders[i] = requestedSortOrder;
		}
		DefaultRowEst(index);

		if (table == NewTable)
		{
			// This routine has been called to create an automatic index as a result of a PRIMARY KEY or UNIQUE clause on a column definition, or
			// a PRIMARY KEY or UNIQUE clause following the column definitions. i.e. one of:
			//
			// CREATE TABLE t(x PRIMARY KEY, y);
			// CREATE TABLE t(x, y, UNIQUE(x, y));
			//
			// Either way, check to see if the table already has such an index. If so, don't bother creating this one. This only applies to
			// automatically created indices. Users can do as they wish with explicit indices.
			//
			// Two UNIQUE or PRIMARY KEY constraints are considered equivalent (and thus suppressing the second one) even if they have different sort orders.
			//
			// If there are different collating sequences or if the columns of the constraint occur in different orders, then the constraints are
			// considered distinct and both result in separate indices.
			for (Index *index2 = table->Index; index2; index2 = index2->Next)
			{
				_assert(index2->OnError != OE_None);
				_assert(index2->AutoIndex);
				_assert(index->OnError != OE_None);
				if (index2->Columns.length != index->Columns.length) continue;
				int k;
				for (k = 0; k < index2->Columns.length; k++)
				{
					if (index2->Columns[k] != index->Columns[k]) break;
					const char *z1 = index2->CollNames[k];
					const char *z2 = index->CollNames[k];
					if (z1 != z2 && _strcmp(z1, z2)) break;
				}
				if (k == index2->Columns.length)
				{
					if (index2->OnError != index->OnError)
					{
						// This constraint creates the same index as a previous constraint specified somewhere in the CREATE TABLE statement.
						// However the ON CONFLICT clauses are different. If both this constraint and the previous equivalent constraint have explicit
						// ON CONFLICT clauses this is an error. Otherwise, use the explicitly specified behavior for the index.
						if (!(index2->OnError == OE_Default || index->OnError == OE_Default))
							ErrorMsg("conflicting ON CONFLICT clauses specified");
						if (index2->OnError == OE_Default)
							index2->OnError = index->OnError;
					}
					goto exit_create_index;
				}
			}
		}

		// Link the new Index structure to its table and to the other in-memory database structures. 
		if (!ctx->Init.Busy)
		{
			_assert(Btree::SchemaMutexHeld(ctx, 0, index->Schema));
			Index *p = (Index *)index->Schema->IndexHash.Insert(index->Name, _strlen30(index->Name), index);
			if (p)
			{
				_assert(p == index); // Malloc must have failed
				ctx->MallocFailed = true;
				goto exit_create_index;
			}
			ctx->Flags |= Context::FLAG_InternChanges;
			if (tableName)
				index->Id = ctx->Init.NewTid;
		}

		// If the db->init.busy is 0 then create the index on disk.  This involves writing the index into the master table and filling in the
		// index with the current table contents.
		//
		// The db->init.busy is 0 when the user first enters a CREATE INDEX command.  db->init.busy is 1 when a database is opened and 
		// CREATE INDEX statements are read out of the master table.  In the latter case the index already exists on disk, which is why
		// we don't want to recreate it.
		//
		// If pTblName==0 it means this index is generated as a primary key or UNIQUE constraint of a CREATE TABLE statement.  Since the table
		// has just been created, it contains no data and the index initialization step can be skipped.
		else
		{
			Vdbe *v = GetVdbe();
			if (!v) goto exit_create_index;

			// Create the rootpage for the index
			BeginWriteOperation(1, db);
			int memId = ++Mems;
			v->AddOp2(OP_CreateIndex, db, memId);

			// Gather the complete text of the CREATE INDEX statement into the zStmt variable
			char *stmt;
			if (start)
			{
				_assert(end != nullptr);
				// A named index with an explicit CREATE INDEX statement
				stmt = _mtagprintf(ctx, "CREATE%s INDEX %.*s", (onError == OE_None ? "" : " UNIQUE"), (int)(end->data - nameAsToken->data) + 1, nameAsToken->data);
			}
			else
				stmt = nullptr; // An automatic index created by a PRIMARY KEY or UNIQUE constraint zStmt = sqlite3MPrintf("");

			// Add an entry in sqlite_master for this index
			NestedParse("INSERT INTO %Q.%s VALUES('index',%Q,%Q,#%d,%Q);", ctx->DBs[db].Name, SCHEMA_TABLE(db),
				index->Name,
				table->Name,
				(void *)memId,
				stmt);
			_tagfree(ctx, stmt);

			// Fill the index with data and reparse the schema. Code an OP_Expire to invalidate all pre-compiled statements.
			if (tableName)
			{
				RefillIndex(index, memId);
				ChangeCookie(db);
				v->AddParseSchemaOp(db, _mtagprintf(ctx, "name='%q' AND type='index'", index->Name));
				v->AddOp1(OP_Expire, 0);
			}
		}

		// When adding an index to the list of indices for a table, make sure all indices labeled OE_Replace come after all those labeled
		// OE_Ignore.  This is necessary for the correct constraint check processing (in sqlite3GenerateConstraintChecks()) as part of
		// UPDATE and INSERT statements.  
		if (ctx->Init.Busy || !tableName)
		{
			if (onError != OE_Replace || !table->Index || table->Index->OnError == OE_Replace)
			{
				index->Next = table->Index;
				table->Index = index;
			}
			else
			{
				Index *otherIndex = table->Index;
				while (otherIndex->Next && otherIndex->Next->OnError != OE_Replace)
					otherIndex = otherIndex->Next;
				index->Next = otherIndex->Next;
				otherIndex->Next = index;
			}
			r = index;
			index = nullptr;
		}

		// Clean up before exiting
exit_create_index:
		if (index)
		{
			_tagfree(ctx, index->ColAff);
			_tagfree(ctx, index);
		}
		Expr::ListDelete(ctx, list);
		SrcListDelete(ctx, tableName);
		_tagfree(ctx, name);
		return r;
	}

	__device__ void Parse::DefaultRowEst(Index *index)
	{
		tRowcnt *rowEsts = index->RowEsts;
		_assert(rowEsts != nullptr);
		rowEsts[0] = index->Table->RowEst;
		if (rowEsts[0] < 10) rowEsts[0] = 10;
		tRowcnt n = 10;
		for(int i = 1; i <= index->Columns.length; i++)
		{
			rowEsts[i] = n;
			if (n > 5) n--;
		}
		if (index->OnError != OE_None)
			rowEsts[index->Columns.length] = 1;
	}

	__device__ void Parse::DropIndex(SrcList *name, bool ifExists)
	{
		_assert(Errs == 0); // Never called with prior errors
		Context *ctx = Ctx;
		if (ctx->MallocFailed)
			goto exit_drop_index;
		_assert(name->Srcs == 1);
		if (Prepare::ReadSchema(this) != RC_OK)
			goto exit_drop_index;
		Index *index = FindIndex(ctx, name->Ids[0].Name, name->Ids[0].Database);
		if (!index)
		{
			if (!ifExists)
				ErrorMsg("no such index: %S", name);
			else
				CodeVerifyNamedSchema(name->Ids[0].Database);
			CheckSchema = 1;
			goto exit_drop_index;
		}
		if (index->AutoIndex)
		{
			ErrorMsg("index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped");
			goto exit_drop_index;
		}
		int db = Prepare::SchemaToIndex(ctx, index->Schema);
#ifndef OMIT_AUTHORIZATION
		{
			Table *table = index->Table;
			const char *dbName = ctx->DBs[db].Name;
			const char *tableName = SCHEMA_TABLE(db);
			AUTH code = (!E_OMIT_TEMPDB && db ? AUTH_DROP_TEMP_INDEX : AUTH_DROP_INDEX);
			if (Auth::Check(this, AUTH_DELETE, tableName, nullptr, dbName) || Auth::Check(this, code, index->Name, table->Name, dbName))
				goto exit_drop_index;
		}
#endif

		// Generate code to remove the index and from the master table
		Vdbe *v = GetVdbe();
		if (v)
		{
			BeginWriteOperation(1, db);
			NestedParse("DELETE FROM %Q.%s WHERE name=%Q AND type='index'", ctx->DBs[db].Name, SCHEMA_TABLE(db), index->Name);
			ClearStatTables(db, "idx", index->Name);
			ChangeCookie(db);
			DestroyRootPage(this, index->Id, db);
			v->AddOp4(OP_DropIndex, db, 0, 0, index->Name, 0);
		}

exit_drop_index:
		SrcListDelete(ctx, name);
	}

	__device__ void *Parse::ArrayAllocate(Context *ctx, void *array_, int entrySize, int *entryLength, int *index)
	{
		int n = *entryLength;
		if ((n & (n-1)) == 0)
		{
			int newSize = (n == 0 ? 1 : 2*n);
			void *newArray = _tagrealloc(ctx, array_, newSize*entrySize);
			if (!newArray)
			{
				*index = -1;
				return array_;
			}
			array_ = newArray;
		}
		char *z = (char *)array_;
		_memset(&z[n * entrySize], 0, entrySize);
		*index = n;
		++*entryLength;
		return array_;
	}

	__device__ IdList *Parse::IdListAppend(Context *ctx, IdList *list, Token *token)
	{
		if (!list)
		{
			list = (IdList *)_tagalloc2(ctx, sizeof(IdList), true);
			if (!list) return nullptr;
		}
		int i;
		list->Ids.data = (IdList::IdListItem *)ArrayAllocate(ctx, list->Ids, sizeof(list->Ids[0]), &list->Ids.length, &i);
		if (i < 0)
		{
			IdListDelete(ctx, list);
			return nullptr;
		}
		list->Ids[i].Name = NameFromToken(ctx, token);
		return list;
	}

	__device__ void Parse::IdListDelete(Context *ctx, IdList *list)
	{
		if (!list) return;
		for (int i = 0; i < list->Ids.length; i++)
			_tagfree(ctx, list->Ids[i].Name);
		_tagfree(ctx, list->Ids);
		_tagfree(ctx, list);
	}

	__device__ int Parse::IdListIndex(IdList *list, const char *name)
	{
		if (!list) return -1;
		for (int i = 0; i < list->Ids.length; i++)
			if (!_strcmp(list->Ids[i].Name, name)) return i;
		return -1;
	}

	__device__ SrcList *Parse::SrcListEnlarge(Context *ctx, SrcList *src, int extra, int start)
	{
		// Sanity checking on calling parameters
		_assert(start >= 0);
		_assert(extra >= 1);
		_assert(src != 0);
		_assert(start <= src->Srcs);
		// Allocate additional space if needed
		if (src->Srcs + extra > src->Allocs)
		{
			int allocs = src->Srcs + extra;
			SrcList *newSrc = (SrcList *)_tagrealloc(ctx, src, sizeof(*src) + (allocs - 1)*sizeof(src->Ids[0]));
			if (!newSrc)
			{
				_assert(ctx->MallocFailed);
				return src;
			}
			src = newSrc;
			int got = ((int)_tagallocsize(ctx, newSrc) - sizeof(*src))/sizeof(src->Ids[0])+1;
			src->Allocs = (uint16)got;
		}
		// Move existing slots that come after the newly inserted slots out of the way
		for (int i = src->Srcs - 1; i >= start; i--)
			src->Ids[i + extra] = src->Ids[i];
		src->Srcs += (int16)extra;
		// Zero the newly allocated slots
		_memset(&src->Ids[start], 0, sizeof(src->Ids[0])*extra);
		for (int i = start; i < start + extra; i++)
			src->Ids[i].Cursor = -1;
		// Return a pointer to the enlarged SrcList
		return src;
	}

	__device__ SrcList *Parse::SrcListAppend(Context *ctx, SrcList *list, Token *table, Token *database)
	{
		_assert(!database || table); // Cannot have C without B
		if (!list)
		{
			list = (SrcList *)_tagalloc2(ctx, sizeof(SrcList), true);
			if (!list) return nullptr;
			list->Allocs = 1;
		}
		list = SrcListEnlarge(ctx, list, 1, list->Srcs);
		if (ctx->MallocFailed)
		{
			SrcListDelete(ctx, list);
			return nullptr;
		}
		SrcList::SrcListItem *item = &list->Ids[list->Srcs - 1];
		if (database && !database->data)
			database = nullptr;
		if (database)
		{
			Token *tempToken = database;
			database = table;
			table = tempToken;
		}
		item->Name = NameFromToken(ctx, table);
		item->Database = NameFromToken(ctx, database);
		return list;
	}

	__device__ void Parse::SrcListAssignCursors(SrcList *list)
	{
		_assert(list || Ctx->MallocFailed);
		int i;
		SrcList::SrcListItem *item;
		if (list)
			for (i = 0, item = list->Ids; i < list->Srcs; i++, item++)
			{
				if (item->Cursor >= 0) break;
				item->Cursor = Tabs++;
				if (item->Select)
					SrcListAssignCursors(item->Select->Src);
			}
	}

	__device__ void Parse::SrcListDelete(Context *ctx, SrcList *list)
	{
		if (!list) return;
		int i;
		SrcList::SrcListItem *item;
		for (item = list->Ids, i = 0; i < list->Srcs; i++, item++)
		{
			_tagfree(ctx, item->Database);
			_tagfree(ctx, item->Name);
			_tagfree(ctx, item->Alias);
			_tagfree(ctx, item->Index);
			DeleteTable(ctx, item->Table);
			Select::Delete(ctx, item->Select);
			Expr::Delete(ctx, item->On);
			IdListDelete(ctx, item->Using);
		}
		_tagfree(ctx, list);
	}

	__device__ SrcList *Parse::SrcListAppendFromTerm(SrcList *list, Token *table, Token *database, Token *alias, Select *subquery, Expr *on, IdList *using_)
	{
		Context *ctx = Ctx;
		if (!list && (on || using_))
		{
			ErrorMsg("a JOIN clause is required before %s", (on ? "ON" : "USING"));
			goto append_from_error;
		}
		list = SrcListAppend(ctx, list, table, database);
		if (!list || _NEVER(list->Srcs == 0))
			goto append_from_error;
		SrcList::SrcListItem *item = &list->Ids[list->Srcs-1];
		_assert(alias != nullptr);
		if (alias->length != 0)
			item->Alias = NameFromToken(ctx, alias);
		item->Select = subquery;
		item->On = on;
		item->Using = using_;
		return list;

append_from_error:
		_assert(list == nullptr);
		Expr::Delete(ctx, on);
		IdListDelete(Ctx, using_);
		Select::Delete(ctx, subquery);
		return nullptr;
	}

	__device__ void Parse::SrcListIndexedBy(SrcList *list, Token *indexedBy)
	{
		_assert(indexedBy != nullptr);
		if (list && _ALWAYS(list->Srcs > 0))
		{
			SrcList::SrcListItem *item = &list->Ids[list->Srcs-1];
			_assert(!item->NotIndexed && item->Index == nullptr);
			if (indexedBy->length == 1 && !indexedBy->data)
				item->NotIndexed = 1; // A "NOT INDEXED" clause was supplied. See parse.y construct "indexed_opt" for details.
			else
				item->IndexName = NameFromToken(Ctx, indexedBy);
		}
	}

	__device__ void Parse::SrcListShiftJoinType(SrcList *list)
	{
		if (list)
		{
			_assert(list->Ids || list->Srcs == 0);
			for (int i = list->Srcs - 1; i > 0; i--)
				list->Ids[i].Jointype = list->Ids[i - 1].Jointype;
			list->Ids[0].Jointype = (JT)0;
		}
	}

	__device__ void Parse::BeginTransaction(int type)
	{
		Context *ctx = Ctx;
		_assert(ctx);
#ifndef OMIT_AUTHORIZATION
		if (Auth::Check(this, AUTH_TRANSACTION, "BEGIN", 0, 0))
			return;
#endif
		Vdbe *v = GetVdbe();
		if (!v) return;
		if (type != TK_DEFERRED)
		{
			for (int i = 0; i < ctx->DBs.length; i++)
			{
				v->AddOp2(OP_Transaction, i, (type == TK_EXCLUSIVE ? 2 : 1));
				v->UsesBtree(i);
			}
		}
		v->AddOp2(OP_AutoCommit, 0, 0);
	}

	__device__ void Parse::CommitTransaction()
	{
		_assert(Ctx);
#ifndef OMIT_AUTHORIZATION
		if (Auth::Check(this, AUTH_TRANSACTION, "COMMIT", 0, 0))
			return;
#endif
		Vdbe *v = GetVdbe();
		if (v)
			v->AddOp2(OP_AutoCommit, 1, 0);
	}

	__device__ void Parse::RollbackTransaction()
	{
		_assert(Ctx);
#ifndef OMIT_AUTHORIZATION
		if (Auth::Check(this, AUTH_TRANSACTION, "ROLLBACK", 0, 0))
			return;
#endif
		Vdbe *v = GetVdbe();
		if (v)
			v->AddOp2(OP_AutoCommit, 1, 1);
	}

#if !OMIT_AUTHORIZATION
	__constant__ static const char *const _savepoint_Names[] = { "BEGIN", "RELEASE", "ROLLBACK" };
#endif
	__device__ void Parse::Savepoint(IPager::SAVEPOINT op, Token *name)
	{
#if !OMIT_AUTHORIZATION
		_assert(IPager::SAVEPOINT_BEGIN == 0 && IPager::SAVEPOINT_RELEASE == 1 && IPager::SAVEPOINT_ROLLBACK == 2);
#endif
		char *nameAsString = NameFromToken(Ctx, name);
		if (nameAsString)
		{
			Vdbe *v = GetVdbe();
			if (!v
#ifndef OMIT_AUTHORIZATION
				|| Auth::Check(this, AUTH_SAVEPOINT, _savepoint_Names[op], nameAsString, 0)
#endif
				)
			{
				_tagfree(Ctx, nameAsString);
				return;
			}
			v->AddOp4(OP_Savepoint, op, 0, 0, nameAsString, Vdbe::P4T_DYNAMIC);
		}
	}

	__device__ int Parse::OpenTempDatabase()
	{
		Context *ctx = Ctx;
		if (!ctx->DBs[1].Bt && !Explain)
		{
			const VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_EXCLUSIVE | VSystem::OPEN_DELETEONCLOSE | VSystem::OPEN_TEMP_DB);
			Btree *bt;
			Core::RC rc = Btree::Open(ctx->Vfs, 0, ctx, &bt, (Btree::OPEN)0, flags);
			if (rc != RC_OK)
			{
				ErrorMsg("unable to open a temporary database file for storing temporary tables");
				RC = rc;
				return 1;
			}
			ctx->DBs[1].Bt = bt;
			_assert(ctx->DBs[1].Schema);
			if (bt->SetPageSize(ctx->NextPagesize, -1, 0) != RC_NOMEM)
			{
				ctx->MallocFailed = true;
				return 1;
			}
		}
		return 0;
	}

	__device__ void Parse::CodeVerifySchema(int db)
	{
		Parse *toplevel = Parse_Toplevel(this);
#ifndef OMIT_TRIGGER
		if (toplevel != this)
		{
			// This branch is taken if a trigger is currently being coded. In this case, set cookieGoto to a non-zero value to show that this function
			// has been called. This is used by the sqlite3ExprCodeConstants() function.
			CookieGoto = -1;
		}
#endif
		if (toplevel->CookieGoto == 0)
		{
			Vdbe *v = toplevel->GetVdbe();
			if (!v) return; // This only happens if there was a prior error
			toplevel->CookieGoto = v->AddOp2(OP_Goto, 0, 0) + 1;
		}
		if (db >= 0)
		{
			Context *ctx = toplevel->Ctx;
			_assert(db < ctx->DBs.length);
			_assert(ctx->DBs[db].Bt || db == 1);
			_assert(db < MAX_ATTACHED + 2);
			_assert(Btree::SchemaMutexHeld(ctx, db, 0));
			yDbMask mask = ((yDbMask)1) << db;
			if ((toplevel->CookieMask & mask) == 0)
			{
				toplevel->CookieMask |= mask;
				toplevel->CookieValue[db] = ctx->DBs[db].Schema->SchemaCookie;
				if (!E_OMIT_TEMPDB && db == 1)
					toplevel->OpenTempDatabase();
			}
		}
	}

	__device__ void Parse::CodeVerifyNamedSchema(const char *dbName)
	{
		Context *ctx = Ctx;
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			Context::DB *dbObj = &ctx->DBs[i];
			if (dbObj->Bt && (!dbName || !_strcmp(dbName, dbObj->Name)))
				CodeVerifySchema(i);
		}
	}

	__device__ void Parse::BeginWriteOperation(int setStatement, int db)
	{
		Parse *toplevel = Parse_Toplevel(this);
		CodeVerifySchema(db);
		toplevel->WriteMask |= ((yDbMask)1) << db;
		toplevel->IsMultiWrite |= setStatement;
	}

	__device__ void Parse::MultiWrite()
	{
		Parse *toplevel = Parse_Toplevel(this);
		toplevel->IsMultiWrite = 1;
	}

	__device__ void Parse::MayAbort()
	{
		Parse *toplevel = Parse_Toplevel(this);
		toplevel->_MayAbort = true;
	}

	__device__ void Parse::HaltConstraint(Core::RC errCode, int onError, char *p4, int p4type)
	{
		Vdbe *v = GetVdbe();
		_assert((errCode&0xff) == RC_CONSTRAINT);
		if (onError == OE_Abort)
			MayAbort();
		v->AddOp4(OP_Halt, errCode, onError, 0, p4, p4type);
	}

#ifndef OMIT_REINDEX
	__device__ static bool CollationMatch(const char *collName, Index *index)
	{
		_assert(collName);
		for(int i = 0; i < index->Columns.length; i++)
		{
			const char *z = index->CollNames[i];
			_assert(z);
			if (!_strcmp(z, collName))
				return true;
		}
		return false;
	}

	__device__ static void ReindexTable(Parse *parse, Table *table, char const *collName)
	{
		for (Index *index = table->Index; index; index = index->Next)
			if (!collName || CollationMatch(collName, index))
			{
				int db = Prepare::SchemaToIndex(parse->Ctx, table->Schema);
				parse->BeginWriteOperation(0, db);
				parse->RefillIndex(index, -1);
			}
	}

	__device__ static void ReindexDatabases(Parse *parse, char const *collName)
	{
		Context *ctx = parse->Ctx; // The database connection
		_assert(Btree::HoldsAllMutexes(ctx)); // Needed for schema access
		int db; // The database index number
		Context::DB *dbObj; // A single database
		for (db = 0, dbObj = ctx->DBs; db < ctx->DBs.length; db++, dbObj++)
		{
			_assert(dbObj);
			for (HashElem *k = dbObj->Schema->TableHash.First;  k; k = k->Next)
			{
				Table *table = (Table *)k->Data; // A table in the database
				ReindexTable(parse, table, collName);
			}
		}
	}

	__device__ void Parse::Reindex(Token *name1, Token *name2)
	{
		Context *ctx = Ctx; // The database connection
		// Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
		if (Prepare::ReadSchema(this) != RC_OK)
			return;
		if (!name1)
		{
			ReindexDatabases(this, nullptr);
			return;
		}
		else if (_NEVER(!name2) || !name2->data)
		{
			_assert(name1->data);
			char *collName = NameFromToken(ctx, name1);
			if (!collName) return;
			CollSeq *coll = Callback::FindCollSeq(ctx, CTXENCODE(ctx), collName, 0); // Collating sequence to be reindexed, or NULL
			if (coll)
			{
				ReindexDatabases(this, collName);
				_tagfree(ctx, collName);
				return;
			}
			_tagfree(ctx, collName);
		}
		Token *objName; // Name of the table or index to be reindexed
		int db = TwoPartName(name1, name2, &objName); // The database index number
		if (db < 0) return;
		char *z = NameFromToken(ctx, objName); // Name of a table or index
		if (!z) return;
		const char *dbName = ctx->DBs[db].Name; // Name of the database
		Table *table = FindTable(ctx, z, dbName); // A table in the database
		if (table)
		{
			ReindexTable(this, table, 0);
			_tagfree(ctx, z);
			return;
		}
		Index *index = FindIndex(ctx, z, dbName); // An index associated with pTab
		_tagfree(ctx, z);
		if (index)
		{
			BeginWriteOperation(0, db);
			RefillIndex(index, -1);
			return;
		}
		ErrorMsg("unable to identify the object to be reindexed");
	}
#endif

	__device__ KeyInfo *Parse::IndexKeyinfo(Index *index)
	{
		int colLength = index->Columns.length;
		int bytes = sizeof(KeyInfo) + (colLength-1)*sizeof(CollSeq*) + colLength;
		Context *ctx = Ctx;
		KeyInfo *key = (KeyInfo *)_tagalloc2(ctx, bytes, true);
		if (key)
		{
			key->Ctx = Ctx;
			key->SortOrders = (SO *)&(key->Colls[colLength]);
			_assert(&key->SortOrders[colLength] == &(((SO *)key)[bytes]));
			for (int i = 0; i < colLength; i++)
			{
				char *collName = index->CollNames[i];
				_assert(collName);
				key->Colls[i] = LocateCollSeq(collName);
				key->SortOrders[i] = index->SortOrders[i];
			}
			key->Fields = (uint16)colLength;
		}
		if (Errs)
		{
			_tagfree(ctx, key);
			key = nullptr;
		}
		return key;
	}
}