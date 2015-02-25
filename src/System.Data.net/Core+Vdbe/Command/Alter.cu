// alter.c
#pragma region OMIT_ALTERTABLE
#ifndef OMIT_ALTERTABLE
#include "..\VdbeInt.cu.h"

namespace Core { namespace Command
{
	__device__ static void RenameTableFunc(FuncContext *fctx, int notUsed, Mem **argv)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		unsigned char const *sql = Vdbe::Value_Text(argv[0]);
		unsigned char const *tableName = Vdbe::Value_Text(argv[1]);
		if (!sql)
			return;
		unsigned char const *z = sql;
		int length = 0;
		int token;
		Token tname;
		do
		{
			if (!*z)
				return; // Ran out of input before finding an opening bracket. Return NULL.
			// Store the token that zCsr points to in tname.
			tname.data = (char *)z;
			tname.length = length;
			// Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
			do
			{
				z += length;
				length = Parse::GetToken(z, &token);
			} while (token == TK_SPACE);
			_assert(length > 0);
		} while (token != TK_LP && token != TK_USING);

		char *r = _mtagprintf(ctx, "%.*s\"%w\"%s", ((uint8 *)tname.data) - sql, sql, tableName, tname.data + tname.length);
		Vdbe::Result_Text(fctx, r, -1, DESTRUCTOR_DYNAMIC);
	}

#ifndef OMIT_FOREIGN_KEY
	__device__ static void RenameParentFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		unsigned char const *input = Vdbe::Value_Text(argv[0]);
		unsigned char const *oldName = Vdbe::Value_Text(argv[1]);
		unsigned char const *newName = Vdbe::Value_Text(argv[2]);

		char *output = nullptr;
		int n; // Length of token z
		for (unsigned const char *z = input; *z; z = z + n)
		{
			int token; // Type of token
			n = Parse::GetToken(z, &token);
			if (token == TK_REFERENCES)
			{
				char *parent;
				do
				{
					z += n;
					n = Parse::GetToken(z, &token);
				} while (token == TK_SPACE);

				parent = _tagstrndup(ctx, (const char *)z, n);
				if (parent == 0) break;
				Parse::Dequote(parent);
				if (!_strcmp((const char *)oldName, parent))
				{
					char *out_ = _mtagprintf(ctx, "%s%.*s\"%w\"", (output ? output : ""), z - input, input, (const char *)newName);
					_tagfree(ctx, output);
					output = out_;
					input = &z[n];
				}
				_tagfree(ctx, parent);
			}
		}

		char *r = _mtagprintf(ctx, "%s%s", (output ? output : ""), input);
		Vdbe::Result_Text(fctx, r, -1, DESTRUCTOR_DYNAMIC);
		_tagfree(ctx, output);
	}
#endif

#ifndef OMIT_TRIGGER
	__device__ static void RenameTriggerFunc(FuncContext *fctx, int notUsed, Mem **argv)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		unsigned char const *sql = Vdbe::Value_Text(argv[0]);
		unsigned char const *tableName = Vdbe::Value_Text(argv[1]);

		unsigned char const *z = sql;
		int length = 0;
		int token;
		Token tname;
		int dist = 3;

		// The principle used to locate the table name in the CREATE TRIGGER statement is that the table name is the first token that is immediatedly
		// preceded by either TK_ON or TK_DOT and immediatedly followed by one of TK_WHEN, TK_BEGIN or TK_FOR.
		if (sql)
		{
			do
			{
				if (!*z)
					return; // Ran out of input before finding the table name. Return NULL.
				// Store the token that zCsr points to in tname.
				tname.data = (char *)z;
				tname.length = length;

				// Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
				do
				{
					z += length;
					length = Parse::GetToken(z, &token);
				} while (token == TK_SPACE);
				_assert(length > 0);

				// Variable 'dist' stores the number of tokens read since the most recent TK_DOT or TK_ON. This means that when a WHEN, FOR or BEGIN 
				// token is read and 'dist' equals 2, the condition stated above to be met.
				//
				// Note that ON cannot be a database, table or column name, so there is no need to worry about syntax like 
				// "CREATE TRIGGER ... ON ON.ON BEGIN ..." etc.
				dist++;
				if (token == TK_DOT || token == TK_ON)
					dist = 0;
			} while (dist != 2 || (token != TK_WHEN && token != TK_FOR && token != TK_BEGIN));

			// Variable tname now contains the token that is the old table-name in the CREATE TRIGGER statement.
			char *r = _mtagprintf(ctx, "%.*s\"%w\"%s", ((uint8 *)tname.data) - sql, sql, tableName, tname.data+tname.length);
			Vdbe::Result_Text(fctx, r, -1, DESTRUCTOR_DYNAMIC);
		}
	}
#endif

	__constant__ static _WSD FuncDef g_alterTableFuncs[] = {
		FUNCTION(sqlite_rename_table,   2, 0, 0, RenameTableFunc),
#ifndef OMIT_TRIGGER
		FUNCTION(sqlite_rename_trigger, 2, 0, 0, RenameTriggerFunc),
#endif
#ifndef OMIT_FOREIGN_KEY
		FUNCTION(sqlite_rename_parent,  3, 0, 0, RenameParentFunc),
#endif
	};
	__device__ void Alter::Functions()
	{
		FuncDefHash *hash = &Main_GlobalFunctions;
		FuncDef *funcs = (FuncDef *)&_GLOBAL(FuncDef, g_alterTableFuncs);
		for (int i = 0; i < _lengthof(g_alterTableFuncs); i++)
			Callback::FuncDefInsert(hash, &funcs[i]);
	}

	__device__ static char *WhereOrName(Context *ctx, char *where_, char *constant)
	{
		char *newExpr;
		if (!where_)
			newExpr = _mtagprintf(ctx, "name=%Q", constant);
		else
		{
			newExpr = _mtagprintf(ctx, "%s OR name=%Q", where_, constant);
			_tagfree(ctx, where_);
		}
		return newExpr;
	}

#if !defined(OMIT_FOREIGN_KEY) && !defined(OMIT_TRIGGER)
	__device__ static char *WhereForeignKeys(Parse *parse, Table *table)
	{
		char *where_ = nullptr;
		for (FKey *p = Parse::FKReferences(table); p; p = p->NextTo)
			where_ = WhereOrName(parse->Ctx, where_, p->From->Name);
		return where_;
	}
#endif

	__device__ static char *WhereTempTriggers(Parse *parse, Table *table)
	{
		Context *ctx = parse->Ctx;
		char *where_ = nullptr;
		const Schema *tempSchema = ctx->DBs[1].Schema; // Temp db schema
		// If the table is not located in the temp-db (in which case NULL is returned, loop through the tables list of triggers. For each trigger
		// that is not part of the temp-db schema, add a clause to the WHERE expression being built up in zWhere.
		if (table->Schema != tempSchema)
		{
			for (Trigger *trig = Trigger::List(parse, table); trig; trig = trig->Next)
				if (trig->Schema == tempSchema)
					where_ = WhereOrName(ctx, where_, trig->Name);
		}
		if (where_)
		{
			char *newWhere = _mtagprintf(ctx, "type='trigger' AND (%s)", where_);
			_tagfree(ctx, where_);
			where_ = newWhere;
		}
		return where_;
	}

	__device__ static void ReloadTableSchema(Parse *parse, Table *table, const char *name)
	{
		Context *ctx = parse->Ctx;
		Vdbe *v = parse->GetVdbe();
		if (_NEVER(v == nullptr)) return;
		_assert(Btree::HoldsAllMutexes(ctx));
		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Index of database containing pTab
		_assert(db >= 0);

#ifndef OMIT_TRIGGER
		// Drop any table triggers from the internal schema.
		for (Trigger *trig = Trigger::List(parse, table); trig; trig = trig->Next)
		{
			int trigDb = Prepare::SchemaToIndex(ctx, trig->Schema);
			_assert(trigDb == db || trigDb == 1);
			v->AddOp4(OP_DropTrigger, trigDb, 0, 0, trig->Name, 0);
		}
#endif

		// Drop the table and index from the internal schema.
		v->AddOp4(OP_DropTable, db, 0, 0, table->Name, 0);

		// Reload the table, index and permanent trigger schemas.
		char *where_ = _mtagprintf(ctx, "tbl_name=%Q", name);
		if (!where_) return;
		v->AddParseSchemaOp(db, where_);

#ifndef OMIT_TRIGGER
		// Now, if the table is not stored in the temp database, reload any temp triggers. Don't use IN(...) in case SQLITE_OMIT_SUBQUERY is defined. 
		if ((where_ = WhereTempTriggers(parse, table)) != nullptr)
			v->AddParseSchemaOp(1, where_);
#endif
	}

	__device__ static bool IsSystemTable(Parse *parse, const char *name)
	{
		if (_strlen30(name) > 6 && !_strncmp(name, "sqlite_", 7))
		{
			parse->ErrorMsg("table %s may not be altered", name);
			return true;
		}
		return false;
	}

	__device__ void Alter::RenameTable(Parse *parse, SrcList *src, Token *name)
	{
		Context *ctx = parse->Ctx; // Database connection

		Context::FLAG savedDbFlags = ctx->Flags;  // Saved value of db->flags
		if (_NEVER(ctx->MallocFailed)) goto exit_rename_table;
		_assert(src->Srcs == 1);
		_assert(Btree::HoldsAllMutexes(ctx));

		Table *table = parse->LocateTableItem(false, &src->Ids[0]); // Table being renamed
		if (!table) goto exit_rename_table;
		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Database that contains the table
		char *dbName = ctx->DBs[db].Name; // Name of database iDb
		ctx->Flags |= BContext::FLAG_PreferBuiltin;

		// Get a NULL terminated version of the new table name.
		char *nameAsString = Parse::NameFromToken(ctx, name); // NULL-terminated version of pName
		if (!nameAsString) goto exit_rename_table;

		// Check that a table or index named 'zName' does not already exist in database iDb. If so, this is an error.
		if (Parse::FindTable(ctx, nameAsString, dbName) || Parse::FindIndex(ctx, nameAsString, dbName))
		{
			parse->ErrorMsg("there is already another table or index with this name: %s", nameAsString);
			goto exit_rename_table;
		}

		// Make sure it is not a system table being altered, or a reserved name that the table is being renamed to.
		if (IsSystemTable(parse, table->Name) || parse->CheckObjectName(nameAsString) != RC_OK)
			goto exit_rename_table;

#ifndef OMIT_VIEW
		if (table->Select)
		{
			parse->ErrorMsg("view %s may not be altered", table->Name);
			goto exit_rename_table;
		}
#endif

#ifndef OMIT_AUTHORIZATION
		// Invoke the authorization callback.
		if (Auth::Check(parse, AUTH_ALTER_TABLE, dbName, table->Name, nullptr))
			goto exit_rename_table;
#endif

		VTable *vtable = nullptr; // Non-zero if this is a v-tab with an xRename()
#ifndef OMIT_VIRTUALTABLE
		if (parse->ViewGetColumnNames(table))
			goto exit_rename_table;
		if (IsVirtual(table))
		{
			vtable = VTable::GetVTable(ctx, table);
			if (!vtable->IVTable->IModule->Rename)
				vtable  = nullptr;
		}
#endif

		// Begin a transaction and code the VerifyCookie for database iDb. Then modify the schema cookie (since the ALTER TABLE modifies the
		// schema). Open a statement transaction if the table is a virtual table.
		Vdbe *v = parse->GetVdbe();
		if (!v)
			goto exit_rename_table;
		parse->BeginWriteOperation(vtable != nullptr, db);
		parse->ChangeCookie(db);

		// If this is a virtual table, invoke the xRename() function if one is defined. The xRename() callback will modify the names
		// of any resources used by the v-table implementation (including other SQLite tables) that are identified by the name of the virtual table.
#ifndef OMIT_VIRTUALTABLE
		if (vtable)
		{
			int i = ++parse->Mems;
			v->AddOp4(OP_String8, 0, i, 0, nameAsString, 0);
			v->AddOp4(OP_VRename, i, 0, 0,(const char *)vtable, Vdbe::P4T_VTAB);
			parse->MayAbort();
		}
#endif

		// figure out how many UTF-8 characters are in zName
		const char *tableName = table->Name; // Original name of the table
		int tableNameLength = _utf8charlength(tableName, -1); // Number of UTF-8 characters in zTabName

#ifndef OMIT_TRIGGER
		char *where_ = nullptr; // Where clause to locate temp triggers
#endif

#if !defined(OMIT_FOREIGN_KEY) && !defined(OMIT_TRIGGER)
		if (ctx->Flags & Context::FLAG_ForeignKeys)
		{
			// If foreign-key support is enabled, rewrite the CREATE TABLE statements corresponding to all child tables of foreign key constraints
			// for which the renamed table is the parent table.
			if ((where_ = WhereForeignKeys(parse, table)) != nullptr)
			{
				parse->NestedParse(
					"UPDATE \"%w\".%s SET "
					"sql = sqlite_rename_parent(sql, %Q, %Q) "
					"WHERE %s;", dbName, SCHEMA_TABLE(db), tableName, nameAsString, where_);
				_tagfree(ctx, where_);
			}
		}
#endif

		// Modify the sqlite_master table to use the new table name.
		parse->NestedParse(
			"UPDATE %Q.%s SET "
#ifdef OMIT_TRIGGER
			"sql = sqlite_rename_table(sql, %Q), "
#else
			"sql = CASE "
			"WHEN type = 'trigger' THEN sqlite_rename_trigger(sql, %Q)"
			"ELSE sqlite_rename_table(sql, %Q) END, "
#endif
			"tbl_name = %Q, "
			"name = CASE "
			"WHEN type='table' THEN %Q "
			"WHEN name LIKE 'sqlite_autoindex%%' AND type='index' THEN "
			"'sqlite_autoindex_' || %Q || substr(name,%d+18) "
			"ELSE name END "
			"WHERE tbl_name=%Q COLLATE nocase AND "
			"(type='table' OR type='index' OR type='trigger');", 
			dbName, SCHEMA_TABLE(db), nameAsString, nameAsString, nameAsString, 
#ifndef OMIT_TRIGGER
			nameAsString,
#endif
			nameAsString, tableNameLength, tableName);

#ifndef OMIT_AUTOINCREMENT
		// If the sqlite_sequence table exists in this database, then update it with the new table name.
		if (Parse::FindTable(ctx, "sqlite_sequence", dbName))
			parse->NestedParse(
			"UPDATE \"%w\".sqlite_sequence set name = %Q WHERE name = %Q",
			dbName, nameAsString, table->Name);
#endif

#ifndef OMIT_TRIGGER
		// If there are TEMP triggers on this table, modify the sqlite_temp_master table. Don't do this if the table being ALTERed is itself located in the temp database.
		if ((where_ = WhereTempTriggers(parse, table)) != nullptr)
		{
			parse->NestedParse(
				"UPDATE sqlite_temp_master SET "
				"sql = sqlite_rename_trigger(sql, %Q), "
				"tbl_name = %Q "
				"WHERE %s;", nameAsString, nameAsString, where_);
			_tagfree(ctx, where_);
		}
#endif

#if !defined(OMIT_FOREIGN_KEY) && !defined(OMIT_TRIGGER)
		if (ctx->Flags & Context::FLAG_ForeignKeys)
		{
			for (FKey *p = Parse::FKReferences(table); p; p = p->NextTo)
			{
				Table *from = p->From;
				if (from != table)
					ReloadTableSchema(parse, p->From, from->Name);
			}
		}
#endif

		// Drop and reload the internal table schema.
		ReloadTableSchema(parse, table, nameAsString);

exit_rename_table:
		Parse::SrcListDelete(ctx, src);
		_tagfree(ctx, nameAsString);
		ctx->Flags = savedDbFlags;
	}

	__device__ void Alter::MinimumFileFormat(Parse *parse, int db, int minFormat)
	{
		Vdbe *v = parse->GetVdbe();
		// The VDBE should have been allocated before this routine is called. If that allocation failed, we would have quit before reaching this point
		if (_ALWAYS(v))
		{
			int r1 = Expr::GetTempReg(parse);
			int r2 = Expr::GetTempReg(parse);
			v->AddOp3(OP_ReadCookie, db, r1, Btree::META_FILE_FORMAT);
			v->UsesBtree(db);
			v->AddOp2(OP_Integer, minFormat, r2);
			int j1 = v->AddOp3( OP_Ge, r2, 0, r1);
			v->AddOp3(OP_SetCookie, db, Btree::META_FILE_FORMAT, r2);
			v->JumpHere(j1);
			Expr::ReleaseTempReg(parse, r1);
			Expr::ReleaseTempReg(parse, r2);
		}
	}

	__device__ void Alter::FinishAddColumn(Parse *parse, Token *colDef)
	{
		Context *ctx = parse->Ctx; // The database connection
		if (parse->Errs || ctx->MallocFailed) return;
		Table *newTable = parse->NewTable; // Copy of pParse->pNewTable
		_assert(newTable);
		_assert(Btree::HoldsAllMutexes(ctx));
		int db = Prepare::SchemaToIndex(ctx, newTable->Schema); // Database number
		const char *dbName = ctx->DBs[db].Name; // Database name
		const char *tableName = &newTable->Name[16];  // Table name: Skip the "sqlite_altertab_" prefix on the name
		Column *col = &newTable->Cols[newTable->Cols.length-1]; // The new column
		Expr *dflt = col->Dflt; // Default value for the new column
		Table *table = Parse::FindTable(ctx, tableName, dbName); // Table being altered
		_assert(table);

#ifndef OMIT_AUTHORIZATION
		// Invoke the authorization callback.
		if (Auth::Check(parse, AUTH_ALTER_TABLE, dbName, table->Name, nullptr))
			return;
#endif

		// If the default value for the new column was specified with a literal NULL, then set pDflt to 0. This simplifies checking
		// for an SQL NULL default below.
		if (dflt && dflt->OP == TK_NULL)
			dflt = nullptr;

		// Check that the new column is not specified as PRIMARY KEY or UNIQUE. If there is a NOT NULL constraint, then the default value for the
		// column must not be NULL.
		if (col->ColFlags & COLFLAG_PRIMKEY)
		{
			parse->ErrorMsg("Cannot add a PRIMARY KEY column");
			return;
		}
		if (newTable->Index)
		{
			parse->ErrorMsg("Cannot add a UNIQUE column");
			return;
		}
		if ((ctx->Flags&Context::FLAG_ForeignKeys) && newTable->FKeys && dflt)
		{
			parse->ErrorMsg("Cannot add a REFERENCES column with non-NULL default value");
			return;
		}
		if (col->NotNull && !dflt)
		{
			parse->ErrorMsg("Cannot add a NOT NULL column with default value NULL");
			return;
		}

		// Ensure the default expression is something that sqlite3ValueFromExpr() can handle (i.e. not CURRENT_TIME etc.)
		if (dflt)
		{
			Mem *val;
			if (Vdbe::ValueFromExpr(ctx, dflt, TEXTENCODE_UTF8, AFF_NONE, &val))
			{
				ctx->MallocFailed = true;
				return;
			}
			if (!val)
			{
				parse->ErrorMsg("Cannot add a column with non-constant default");
				return;
			}
			Vdbe::ValueFree(val);
		}

		// Modify the CREATE TABLE statement.
		char *colDefAsString = _tagstrndup(ctx, (char *)colDef->data, colDef->length); // Null-terminated column definition
		if (colDefAsString)
		{
			char *end = &colDefAsString[colDef->length-1];
			while (end > colDefAsString && (*end == ';' || _isspace(*end))) *end-- = '\0';
			Context::FLAG savedDbFlags = ctx->Flags;
			ctx->Flags |= BContext::FLAG_PreferBuiltin;
			parse->NestedParse( 
				"UPDATE \"%w\".%s SET "
				"sql = substr(sql,1,%d) || ', ' || %Q || substr(sql,%d) "
				"WHERE type = 'table' AND name = %Q", 
				dbName, SCHEMA_TABLE(db), newTable->AddColOffset, colDefAsString, newTable->AddColOffset+1,
				tableName);
			_tagfree(ctx, colDefAsString);
			ctx->Flags = savedDbFlags;
		}

		// If the default value of the new column is NULL, then set the file format to 2. If the default value of the new column is not NULL,
		// the file format becomes 3.
		MinimumFileFormat(parse, db, (dflt ? 3 : 2));

		// Reload the schema of the modified table.
		ReloadTableSchema(parse, table, table->Name);
	}

	__device__ void Alter::BeginAddColumn(Parse *parse, SrcList *src)
	{
		// Look up the table being altered.
		Context *ctx = parse->Ctx;
		_assert(!parse->NewTable);
		_assert(Btree::HoldsAllMutexes(ctx));
		if (ctx->MallocFailed) goto exit_begin_add_column;
		Table *table = parse->LocateTableItem(false, &src->Ids[0]);
		if (!table) goto exit_begin_add_column;

#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(table))
		{
			parse->ErrorMsg( "virtual tables may not be altered");
			goto exit_begin_add_column;
		}
#endif

		// Make sure this is not an attempt to ALTER a view.
		if (table->Select)
		{
			parse->ErrorMsg("Cannot add a column to a view");
			goto exit_begin_add_column;
		}
		if (IsSystemTable(parse, table->Name))
			goto exit_begin_add_column;

		_assert(table->AddColOffset > 0);
		int db = Prepare::SchemaToIndex(ctx, table->Schema);

		// Put a copy of the Table struct in Parse.pNewTable for the sqlite3AddColumn() function and friends to modify.  But modify
		// the name by adding an "sqlite_altertab_" prefix.  By adding this prefix, we insure that the name will not collide with an existing
		// table because user table are not allowed to have the "sqlite_" prefix on their name.
		Table *newTable = (Table *)_tagalloc2(ctx, sizeof(Table), true);
		if (!newTable) goto exit_begin_add_column;
		parse->NewTable = newTable;
		newTable->Refs = 1;
		newTable->Cols.length = table->Cols.length;
		_assert(newTable->Cols.length > 0);
		int allocs = (((newTable->Cols.length-1)/8)*8)+8;
		_assert(allocs >= newTable->Cols.length && allocs%8 == 0 && allocs - newTable->Cols.length < 8);
		newTable->Cols.data = (Column *)_tagalloc2(ctx, sizeof(Column)*allocs, true);
		newTable->Name = _mtagprintf(ctx, "sqlite_altertab_%s", table->Name);
		if (!newTable->Cols || !newTable->Name)
		{
			ctx->MallocFailed = true;
			goto exit_begin_add_column;
		}
		_memcpy(newTable->Cols.data, table->Cols.data, sizeof(Column)*newTable->Cols.length);
		for (int i = 0; i < newTable->Cols.length; i++)
		{
			Column *col = &newTable->Cols[i];
			col->Name = _tagstrdup(ctx, col->Name);
			col->Coll = nullptr;
			col->Type = nullptr;
			col->Dflt = nullptr;
			col->DfltName = nullptr;
		}
		newTable->Schema = ctx->DBs[db].Schema;
		newTable->AddColOffset = table->AddColOffset;
		newTable->Refs = 1;

		// Begin a transaction and increment the schema cookie.
		parse->BeginWriteOperation(0, db);
		Vdbe *v = parse->GetVdbe();
		if (!v) goto exit_begin_add_column;
		parse->ChangeCookie(db);

exit_begin_add_column:
		Parse::SrcListDelete(ctx, src);
		return;
	}

} }
#endif
#pragma endregion