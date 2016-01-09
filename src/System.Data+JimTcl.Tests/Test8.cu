// Code for testing the virtual table interfaces.  This code is not included in the SQLite library.  It is used for automated testing of the SQLite library.
#include "Test.cu.h"

#ifndef OMIT_VIRTUALTABLE

typedef struct echo_vtab echo_vtab;
typedef struct echo_cursor echo_cursor;

// The test module defined in this file uses four global Tcl variables to commicate with test-scripts:
//     $::echo_module
//     $::echo_module_sync_fail
//     $::echo_module_begin_fail
//     $::echo_module_cost
//
// The variable ::echo_module is a list. Each time one of the following methods is called, one or more elements are appended to the list.
// This is used for automated testing of virtual table modules.
//
// The ::echo_module_sync_fail variable is set by test scripts and read by code in this file. If it is set to the name of a real table in the
// the database, then all xSync operations on echo virtual tables that use the named table as a backing store will fail.

// Errors can be provoked within the following echo virtual table methods:
//   xBestIndex   xOpen     xFilter   xNext   
//   xColumn      xRowid    xUpdate   xSync   
//   xBegin       xRename
//
// This is done by setting the global tcl variable:
//   echo_module_fail($method,$tbl)
//
// where $method is set to the name of the virtual table method to fail (i.e. "xBestIndex") and $tbl is the name of the table being echoed (not
// the name of the virtual table, the name of the underlying real table).

// An echo virtual-table object.
//
// echo.vtab.aIndex is an array of booleans. The nth entry is true if the nth column of the real table is the left-most column of an index
// (implicit or otherwise). In other words, if SQLite can optimize a query like "SELECT * FROM real_table WHERE col = ?".
//
// Member variable aCol[] contains copies of the column names of the real table.
struct echo_vtab
{
	IVTable base;
	Jim_Interp *Interp;     // Tcl interpreter containing debug variables
	Context *Ctx;           // Database connection

	bool IsPattern;
	bool InTransaction;      // True if within a transaction
	char *ThisName;            // Name of the echo table
	char *TableName;       // Name of the real table
	char *LogName;         // Name of the log table
	int Cols;               // Number of columns in the real table
	bool *Indexs;            // Array of size nCol. True if column has an index
	char **ColNames;            // Array of size nCol. Column names
};

// An echo cursor object
struct echo_cursor
{
	IVTableCursor base;
	Vdbe *Stmt;
};

__device__ static int SimulateVtabError(echo_vtab *p, const char *methodName)
{
	char varname[128];
	varname[127] = '\0';
	__snprintf(varname, sizeof(varname), "echo_module_fail(%s,%s)", methodName, p->TableName);
	const char *err = Jim_String(Jim_GetGlobalVariableStr(p->Interp, varname, 0));
	if (err)
		p->base.ErrMsg = _mprintf("echo-vtab-error: %s", err);
	return (err != nullptr);
}

// Convert an SQL-style quoted string into a normal string by removing the quote characters.  The conversion is done in-place.  If the
// input does not begin with a quote character, then this routine is a no-op.
//
// Examples:
//     "abc"   becomes   abc
//     'xyz'   becomes   xyz
//     [pqr]   becomes   pqr
//     `mno`   becomes   mno
__device__ static void DequoteString(char *z)
{
	if (!z) return;
	int quote = z[0];
	switch (quote)
	{
	case '\'':  break;
	case '"':   break;
	case '`':   break;                // For MySQL compatibility
	case '[':   quote = ']';  break;  // For MS SqlServer compatibility
	default:    return;
	}
	int i, j;
	for (i = 1, j = 0; z[i]; i++)
		if (z[i] == quote)
		{
			if (z[i+1] == quote)
			{
				z[j++] = quote;
				i++;
			}
			else
			{
				z[j++] = 0;
				break;
			}
		}
		else
			z[j++] = z[i];
}

// Retrieve the column names for the table named zTab via database connection db. SQLITE_OK is returned on success, or an sqlite error
// code otherwise.
//
// If successful, the number of columns is written to *pnCol. *paCol is set to point at sqlite3_malloc()'d space containing the array of
// nCol column names. The caller is responsible for calling sqlite3_free on *paCol.
__device__ static RC GetColumnNames(Context *ctx, const char *table, char ***colsOut, int *colsLengthOut)
{
	char **cols = nullptr;
	int colsLength = 0;

	// Prepare the statement "SELECT * FROM <tbl>". The column names of the result set of the compiled SELECT will be the same as
	// the column names of table <tbl>.
	RC rc = RC_OK;
	char *sql = _mprintf("SELECT * FROM %Q", table);
	if (!sql) { rc = RC_NOMEM; goto out; }
	Vdbe *stmt = nullptr;
	rc = Prepare::Prepare_(ctx, sql, -1, &stmt, nullptr);
	_free(sql);

	if (rc == RC_OK)
	{
		colsLength = Vdbe::Column_Count(stmt);

		// Figure out how much space to allocate for the array of column names (including space for the strings themselves). Then allocate it.
		int bytes = (sizeof(char *) * colsLength);
		int ii;
		for (ii = 0; ii < colsLength; ii++)
		{
			const char *name = Vdbe::Column_Name(stmt, ii);
			if (!name) { rc = RC_NOMEM; goto out; }
			bytes += (int)_strlen(name)+1;
		}
		cols = (char **)_allocZero(bytes);
		if (!cols) { rc = RC_NOMEM; goto out; }

		// Copy the column names into the allocated space and set up the pointers in the aCol[] array.
		char *space = (char *)(&cols[colsLength]);
		for (ii = 0; ii < colsLength; ii++)
		{
			cols[ii] = space;
			space += _sprintf(space, "%s", Vdbe::Column_Name(stmt, ii));
			space++;
		}
		_assert((space-bytes) == (char *)cols);
	}

	*colsOut = cols;
	*colsLengthOut = colsLength;

out:
	Vdbe::Finalize(stmt);
	return rc;
}

// Parameter zTab is the name of a table in database db with nCol columns. This function allocates an array of integers nCol in 
// size and populates it according to any implicit or explicit indices on table zTab.
//
// If successful, SQLITE_OK is returned and *paIndex set to point at the allocated array. Otherwise, an error code is returned.
//
// See comments associated with the member variable aIndex above "struct echo_vtab" for details of the contents of the array.
__device__ static RC GetIndexArray(Context *ctx, const char *table, int cols, bool **indexsOut)
{
	RC rc;

	// Allocate space for the index array
	bool *indexs = (bool *)_allocZero(sizeof(bool) * cols);
	if (!indexs) { rc = RC_NOMEM; goto get_index_array_out; }

	// Compile an sqlite pragma to loop through all indices on table zTab
	char *sql = _mprintf("PRAGMA index_list(%s)", table);
	if (!sql) { rc = RC_NOMEM; goto get_index_array_out; }
	Vdbe *stmt = nullptr;
	rc = Prepare::Prepare_(ctx, sql, -1, &stmt, nullptr);
	_free(sql);

	// For each index, figure out the left-most column and set the corresponding entry in indexs[] to 1.
	while (stmt && stmt->Step() == RC_ROW)
	{
		const char *idx = (const char *)Vdbe::Column_Text(stmt, 1);
		Vdbe *stmt2 = nullptr;
		sql = _mprintf("PRAGMA index_info(%s)", idx);
		if (!sql) { rc = RC_NOMEM; goto get_index_array_out; }
		rc = Prepare::Prepare_(ctx, sql, -1, &stmt2, nullptr);
		_free(sql);
		if (stmt2 && stmt2->Step() == RC_ROW)
		{
			int cid = Vdbe::Column_Int(stmt2, 1);
			_assert(cid >= 0 && cid < cols);
			indexs[cid] = 1;
		}
		if (stmt2)
			rc = Vdbe::Finalize(stmt2);
		if (rc != RC_OK)
			goto get_index_array_out;
	}

get_index_array_out:
	if (stmt)
	{
		RC rc2 = Vdbe::Finalize(stmt);
		if (rc == RC_OK)
			rc = rc2;
	}
	if (rc != RC_OK)
	{
		_free(indexs);
		indexs = 0;
	}
	*indexsOut = indexs;
	return rc;
}

// Global Tcl variable $echo_module is a list. This routine appends the string element zArg to that list in interpreter interp.
__device__ static void AppendToEchoModule(Jim_Interp *interp, const char *arg)
{
	Jim_Obj *list = Jim_GetGlobalVariableStr(interp, "echo_module", JIM_ERRMSG);
	Jim_ListAppendElement(interp, list, Jim_NewStringObj(interp, (arg ? arg : ""), -1));
}

// This function is called from within the echo-modules xCreate and xConnect methods. The argc and argv arguments are copies of those 
// passed to the calling method. This function is responsible for calling sqlite3_declare_vtab() to declare the schema of the virtual
// table being created or connected.
//
// If the constructor was passed just one argument, i.e.:
//   CREATE TABLE t1 AS echo(t2);
//
// Then t2 is assumed to be the name of a *real* database table. The schema of the virtual table is declared by passing a copy of the 
// CREATE TABLE statement for the real table to sqlite3_declare_vtab(). Hence, the virtual table should have exactly the same column names and 
// types as the real table.
__device__ static RC EchoDeclareVtab(echo_vtab *vtab, Context *ctx)
{
	RC rc = RC_OK;
	if (vtab->TableName)
	{
		Vdbe *stmt = nullptr;
		rc = Prepare::Prepare_(ctx, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", -1, &stmt, nullptr);
		if (rc == RC_OK)
		{
			Vdbe::Bind_Text(stmt, 1, vtab->TableName, -1, 0);
			if (stmt->Step() == RC_ROW)
			{
				const char *createTable = (const char *)Vdbe::Column_Text(stmt, 0);
				rc = VTable::DeclareVTable(ctx, createTable);
				RC rc2 = Vdbe::Finalize(stmt);
				if (rc == RC_OK)
					rc = rc2;
			} else {
				rc = Vdbe::Finalize(stmt);
				if (rc == RC_OK)
					rc = RC_ERROR;
			}
			if (rc == RC_OK)
				rc = GetColumnNames(ctx, vtab->TableName, &vtab->ColNames, &vtab->Cols);
			if (rc == RC_OK)
				rc = GetIndexArray(ctx, vtab->TableName, vtab->Cols, &vtab->Indexs);
		}
	}

	return rc;
}

// This function frees all runtime structures associated with the virtual table pVtab.
__device__ static RC EchoDestructor(IVTable *vtab)
{
	echo_vtab *p = (echo_vtab *)vtab;
	_free(p->Indexs);
	_free(p->ColNames);
	_free(p->ThisName);
	_free(p->TableName);
	_free(p->LogName);
	_free(p);
	return RC_OK;
}

typedef struct EchoModule EchoModule;
struct EchoModule
{
	Jim_Interp *Interp;
};

// This function is called to do the work of the xConnect() method - to allocate the required in-memory structures for a newly connected
// virtual table.
__device__ static RC EchoConstructor(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtable, char **err)
{
	// Allocate the sqlite3_vtab/echo_vtab structure itself
	echo_vtab *vtab = (echo_vtab *)_allocZero(sizeof(*vtab));
	if (!vtab)
		return RC_NOMEM;
	vtab->Interp = ((EchoModule *)aux)->Interp;
	vtab->Ctx = ctx;

	// Allocate echo_vtab.zThis
	vtab->ThisName = _mprintf("%s", args[2]);
	if (!vtab->ThisName)
	{
		EchoDestructor((IVTable *)vtab);
		return RC_NOMEM;
	}

	// Allocate echo_vtab.zTableName
	if (argc > 3)
	{
		vtab->TableName = _mprintf("%s", args[3]);
		DequoteString(vtab->TableName);
		if (vtab->TableName && vtab->TableName[0] == '*'){
			char *z = _mprintf("%s%s", args[2], &(vtab->TableName[1]));
			_free(vtab->TableName);
			vtab->TableName = z;
			vtab->IsPattern = true;
		}
		if (!vtab->TableName)
		{
			EchoDestructor((IVTable *)vtab);
			return RC_NOMEM;
		}
	}

	// Log the arguments to this function to Tcl var ::echo_module
	for (int i = 0; i < argc; i++)
		AppendToEchoModule(vtab->Interp, args[i]);

	// Invoke sqlite3_declare_vtab and set up other members of the echo_vtab structure. If an error occurs, delete the sqlite3_vtab structure and return an error code.
	RC rc = EchoDeclareVtab(vtab, ctx);
	if (rc != RC_OK)
	{
		EchoDestructor((IVTable *)vtab);
		return rc;
	}

	// Success. Set *ppVtab and return
	*vtable = &vtab->base;
	return RC_OK;
}

// Echo virtual table module xCreate method.
__device__ static RC EchoCreate(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtable, char **err)
{
	AppendToEchoModule(((EchoModule *)aux)->Interp, "xCreate");
	RC rc = EchoConstructor(ctx, aux, argc, args, vtable, err);

	// If there were two arguments passed to the module at the SQL level (i.e. "CREATE VIRTUAL TABLE tbl USING echo(arg1, arg2)"), then 
	// the second argument is used as a table name. Attempt to create such a table with a single column, "logmsg". This table will
	// be used to log calls to the xUpdate method. It will be deleted when the virtual table is DROPed.
	//
	// Note: The main point of this is to test that we can drop tables from within an xDestroy method call.
	if (rc == RC_OK && argc == 5)
	{
		echo_vtab *vtab = *(echo_vtab **)vtable;
		vtab->LogName = _mprintf("%s", args[4]);
		char *sql = _mprintf("CREATE TABLE %Q(logmsg)", vtab->LogName);
		rc = Main::Exec(ctx, sql, nullptr, nullptr, nullptr);
		_free(sql);
		if (rc != RC_OK)
			*err = _mprintf("%s", Main::ErrMsg(ctx));
	}

	if (*vtable && rc != RC_OK)
	{
		EchoDestructor(*vtable);
		*vtable = nullptr;
	}
	if (rc == RC_OK)
		(*(echo_vtab**)vtable)->InTransaction = true;
	return rc;
}

// Echo virtual table module xConnect method.
__device__ static RC EchoConnect(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtable, char **err)
{
	AppendToEchoModule(((EchoModule *)aux)->Interp, "xConnect");
	return EchoConstructor(ctx, aux, argc, args, vtable, err);
}

// Echo virtual table module xDisconnect method.
__device__ static RC EchoDisconnect(IVTable *vtab)
{
	AppendToEchoModule(((echo_vtab *)vtab)->Interp, "xDisconnect");
	return EchoDestructor(vtab);
}

// Echo virtual table module xDestroy method.
__device__ static RC EchoDestroy(IVTable *vtab)
{
	RC rc = RC_OK;
	echo_vtab *p = (echo_vtab *)vtab;
	AppendToEchoModule(p->Interp, "xDestroy");

	// Drop the "log" table, if one exists (see echoCreate() for details)
	if (p && p->LogName)
	{
		char *sql = _mprintf("DROP TABLE %Q", p->LogName);
		rc = Main::Exec(p->Ctx, sql, nullptr, nullptr, nullptr);
		_free(sql);
	}

	if (rc == RC_OK)
		rc = EchoDestructor(vtab);
	return rc;
}

// Echo virtual table module xOpen method.
__device__ static RC EchoOpen(IVTable *vtab, IVTableCursor **cursor)
{
	if (SimulateVtabError((echo_vtab *)vtab, "xOpen"))
		return RC_ERROR;
	echo_cursor *cur = (echo_cursor *)_allocZero(sizeof(echo_cursor));
	*cursor = (IVTableCursor *)cur;
	return (cur ? RC_OK : RC_NOMEM);
}

// Echo virtual table module xClose method.
__device__ static RC EchoClose(IVTableCursor *cur)
{
	echo_cursor *cur2 = (echo_cursor *)cur;
	Vdbe *stmt = cur2->Stmt;
	cur2->Stmt = nullptr;
	_free(cur2);
	return Vdbe::Finalize(stmt);
}

// Return non-zero if the cursor does not currently point to a valid record (i.e if the scan has finished), or zero otherwise.
__device__ static bool EchoEof(IVTableCursor *cur)
{
	return (((echo_cursor *)cur)->Stmt ? false : true);
}

// Echo virtual table module xNext method.
__device__ static RC EchoNext(IVTableCursor *cur)
{
	echo_cursor *cur2 = (echo_cursor *)cur;
	if (SimulateVtabError((echo_vtab *)(cur->IVTable), "xNext"))
		return RC_ERROR;

	RC rc = RC_OK;
	if (cur2->Stmt)
	{
		rc = cur2->Stmt->Step();
		if (rc == RC_ROW)
			rc = RC_OK;
		else
		{
			rc = Vdbe::Finalize(cur2->Stmt);
			cur2->Stmt = nullptr;
		}
	}
	return rc;
}

// Echo virtual table module xColumn method.
__device__ static RC EchoColumn(IVTableCursor *cur, FuncContext *fctx, int i)
{
	Vdbe *stmt = ((echo_cursor *)cur)->Stmt;
	if (SimulateVtabError((echo_vtab *)(cur->IVTable), "xColumn"))
		return RC_ERROR;

	if (!stmt)
		Vdbe::Result_Null(fctx);
	else
	{
		int col = i + 1;
		_assert(Vdbe::Data_Count(stmt) > col);
		Vdbe::Result_Value(fctx, Vdbe::Column_Value(stmt, col));
	}
	return RC_OK;
}

// Echo virtual table module xRowid method.
__device__ static RC EchoRowid(IVTableCursor *cur, int64 *rowid)
{
	Vdbe *stmt = ((echo_cursor *)cur)->Stmt;
	if (SimulateVtabError((echo_vtab *)(cur->IVTable), "xRowid"))
		return RC_ERROR;

	*rowid = Vdbe::Column_Int64(stmt, 0);
	return RC_OK;
}

// Compute a simple hash of the null terminated string zString.
//
// This module uses only sqlite3_index_info.idxStr, not sqlite3_index_info.idxNum. So to test idxNum, when idxStr is set
// in echoBestIndex(), idxNum is set to the corresponding hash value. In echoFilter(), code assert()s that the supplied idxNum value is
// indeed the hash of the supplied idxStr.
__device__ static int HashString(const char *string)
{
	int val = 0;
	for (int ii = 0; string[ii]; ii++)
		val = (val << 3) + (int)string[ii];
	return val;
}

// Echo virtual table module xFilter method.
__device__ static RC EchoFilter(IVTableCursor *vtabCursor, int idxNum, const char *idxStr, int argc, Mem **args)
{
	echo_cursor *cur = (echo_cursor *)vtabCursor;
	echo_vtab *vtab = (echo_vtab *)vtabCursor->IVTable;
	Context *ctx = vtab->Ctx;
	if (SimulateVtabError(vtab, "xFilter"))
		return RC_ERROR;

	// Check that idxNum matches idxStr
	_assert( idxNum == HashString(idxStr));

	// Log arguments to the ::echo_module Tcl variable
	AppendToEchoModule(vtab->Interp, "xFilter");
	AppendToEchoModule(vtab->Interp, idxStr);
	int i;
	for (i = 0; i < argc; i++)
		AppendToEchoModule(vtab->Interp, (const char *)Vdbe::Value_Text(args[i]));

	Vdbe::Finalize(cur->Stmt);
	cur->Stmt = nullptr;

	// Prepare the SQL statement created by echoBestIndex and bind the runtime parameters passed to this function to it.
	RC rc = Prepare::Prepare_(ctx, idxStr, -1, &cur->Stmt, nullptr);
	_assert(cur->Stmt || rc != RC_OK);
	for (i = 0; rc == RC_OK && i < argc; i++)
		rc = Vdbe::Bind_Value(cur->Stmt, i+1, args[i]);

	// If everything was successful, advance to the first row of the scan
	if (rc == RC_OK)
		rc = EchoNext(vtabCursor);
	return rc;
}

// A helper function used by echoUpdate() and echoBestIndex() for manipulating strings in concert with the sqlite3_mprintf() function.
//
// Parameter pzStr points to a pointer to a string allocated with sqlite3_mprintf. The second parameter, zAppend, points to another
// string. The two strings are concatenated together and *pzStr set to point at the result. The initial buffer pointed to by *pzStr
// is deallocated via sqlite3_free().
//
// If the third argument, doFree, is true, then sqlite3_free() is also called to free the buffer pointed to by zAppend.
__device__ static void String_Concat(char **str, char *append, bool doFree, RC *rc)
{
	char *in_ = *str;
	if (!append && doFree && *rc == RC_OK)
		*rc = RC_NOMEM;
	if (*rc != RC_OK)
	{
		_free(in_);
		in_ = nullptr;
	}
	else
	{
		if (in_)
		{
			char *temp = in_;
			in_ = _mprintf("%s%s", in_, append);
			_free(temp);
		}
		else
			in_ = _mprintf("%s", append);
		if (!in_)
			*rc = RC_NOMEM;
	}
	*str = in_;
	if (doFree)
		_free(append);
}

// The echo module implements the subset of query constraints and sort orders that may take advantage of SQLite indices on the underlying
// real table. For example, if the real table is declared as:
//     CREATE TABLE real(a, b, c);
//     CREATE INDEX real_index ON real(b);
//
// then the echo module handles WHERE or ORDER BY clauses that refer to the column "b", but not "a" or "c". If a multi-column index is
// present, only its left most column is considered. 
//
// This xBestIndex method encodes the proposed search strategy as an SQL query on the real table underlying the virtual echo module 
// table and stores the query in sqlite3_index_info.idxStr. The SQL
// statement is of the form:
//   SELECT rowid, * FROM <real-table> ?<where-clause>? ?<order-by-clause>?
//
// where the <where-clause> and <order-by-clause> are determined by the contents of the structure pointed to by the pIdxInfo argument.
__device__ static RC EchoBestIndex(IVTable *tab, IIndexInfo *idxInfo)
{
	const char *sep = "WHERE";
	echo_vtab *vtab = (echo_vtab *)tab;
	Vdbe *stmt = nullptr;
	Jim_Interp *interp = vtab->Interp;

	bool isIgnoreUsable = (Jim_String(Jim_GetGlobalVariableStr(interp, "echo_module_ignore_usable", 0)) != nullptr);
	if (SimulateVtabError(vtab, "xBestIndex"))
		return RC_ERROR;

	RC rc = RC_OK;
	int args = 0;
	int useIdx = 0;
	// Determine the number of rows in the table and store this value in local variable rows. The 'estimated-cost' of the scan will be the number of
	// rows in the table for a linear scan, or the log (base 2) of the number of rows if the proposed scan uses an index.  
	int rows;
	bool useCost = false;
	double cost;
	char *query = nullptr;
	const char *costAsString = Jim_String(Jim_GetGlobalVariableStr(interp, "echo_module_cost", 0));
	if (costAsString)
	{
		cost = _atof(costAsString);
		useCost = true;
	}
	else
	{
		query = _mprintf("SELECT count(*) FROM %Q", vtab->TableName);
		if (!query)
			return RC_NOMEM;
		rc = Prepare::Prepare_(vtab->Ctx, query, -1, &stmt, nullptr);
		_free(query);
		if (rc != RC_OK)
			return rc;
		stmt->Step();
		rows = Vdbe::Column_Int(stmt, 0);
		rc = Vdbe::Finalize(stmt);
		if (rc != RC_OK)
			return rc;
	}

	query = _mprintf("SELECT rowid, * FROM %Q", vtab->TableName);
	if (!query)
		return RC_NOMEM;
	int ii;
	char *new_;
	for (ii = 0; ii < idxInfo->Constraints.length; ii++)
	{
		const IIndexInfo::Constraint *constraint = &idxInfo->Constraints[ii];
		IIndexInfo::ConstraintUsage *usage = &idxInfo->ConstraintUsages[ii];
		if (!isIgnoreUsable && !constraint->Usable) continue;

		int col = constraint->Column;
		if (col < 0 || vtab->Indexs[col])
		{
			char *colName = (col >= 0 ? vtab->ColNames[col] : "rowid");
			useIdx = true;
			char *op = nullptr;
			switch (constraint->OP)
			{
			case INDEX_CONSTRAINT_EQ: op = "="; break;
			case INDEX_CONSTRAINT_LT: op = "<"; break;
			case INDEX_CONSTRAINT_GT: op = ">"; break;
			case INDEX_CONSTRAINT_LE: op = "<="; break;
			case INDEX_CONSTRAINT_GE: op = ">="; break;
			case INDEX_CONSTRAINT_MATCH: op = "LIKE"; break;
			}
			if (op[0] == 'L')
				new_ = _mprintf(" %s %s LIKE (SELECT '%%'||?||'%%')", sep, colName);
			else
				new_ = _mprintf(" %s %s %s ?", sep, colName, op);
			String_Concat(&query, new_, true, &rc);

			sep = "AND";
			usage->ArgvIndex = ++args;
			usage->Omit = 1;
		}
	}

	// If there is only one term in the ORDER BY clause, and it is on a column that this virtual table has an index for, then consume 
	// the ORDER BY clause.
	if (idxInfo->OrderBys.length == 1 && (idxInfo->OrderBys.data->Column < 0 || vtab->Indexs[idxInfo->OrderBys.data->Column]))
	{
		int col = idxInfo->OrderBys.data->Column;
		char *colName = (col >= 0 ? vtab->ColNames[col] : "rowid");
		char *dir = (idxInfo->OrderBys.data->Desc ? "DESC" : "ASC");
		new_ = _mprintf(" ORDER BY %s %s", colName, dir);
		String_Concat(&query, new_, true, &rc);
		idxInfo->OrderByConsumed = true;
	}

	AppendToEchoModule(vtab->Interp, "xBestIndex");;
	AppendToEchoModule(vtab->Interp, query);

	if (!query)
		return rc;
	idxInfo->IdxNum = HashString(query);
	idxInfo->IdxStr = query;
	idxInfo->NeedToFreeIdxStr = 1;
	if (useCost)
		idxInfo->EstimatedCost = cost;
	else if (useIdx)
	{
		for (ii = 0; ii < (sizeof(int) * 8); ii++) // Approximation of log2(rows).
			if (rows & (1 << ii))
				idxInfo->EstimatedCost = (double)ii;
	}
	else
		idxInfo->EstimatedCost = (double)rows;
	return rc;
}

//The xUpdate method for echo module virtual tables.
//   apData[0]  apData[1]  apData[2..]
//
//   INTEGER                              DELETE            
//
//   INTEGER    NULL       (nCol args)    UPDATE (do not set rowid)
//   INTEGER    INTEGER    (nCol args)    UPDATE (with SET rowid = <arg1>)
//
//   NULL       NULL       (nCol args)    INSERT INTO (automatic rowid value)
//   NULL       INTEGER    (nCol args)    INSERT (incl. rowid value)
__device__ RC EchoUpdate(IVTable *tab, int dataLength, Mem **datas, int64 *rowid)
{
	echo_vtab *vtab = (echo_vtab *)tab;
	Context *ctx = vtab->Ctx;
	RC rc = RC_OK;
	Vdbe *stmt;
	char *z = nullptr;			// SQL statement to execute
	bool bindArgZero = false;   // True to bind apData[0] to sql var no. nData
	bool bindArgOne = false;    // True to bind apData[1] to sql var no. 1
	int i;
	_assert(dataLength == vtab->Cols+2 || dataLength == 1);
	// Ticket #3083 - make sure we always start a transaction prior to making any changes to a virtual table
	_assert(vtab->InTransaction);
	if (SimulateVtabError(vtab, "xUpdate"))
		return RC_ERROR;

	// If apData[0] is an integer and nData>1 then do an UPDATE
	if (dataLength > 1 && Vdbe::Value_Type(datas[0]) == TYPE_INTEGER)
	{
		char *sep = " SET";
		z = _mprintf("UPDATE %Q", vtab->TableName);
		if (!z)
			rc = RC_NOMEM;

		bindArgOne = (datas[1] && Vdbe::Value_Type(datas[1]) == TYPE_INTEGER);
		bindArgZero = true;

		if (bindArgOne)
		{
			String_Concat(&z, " SET rowid=?1 ", false, &rc);
			sep = ",";
		}
		for (i = 2; i < dataLength; i++)
		{
			if (!datas[i]) continue;
			String_Concat(&z, _mprintf("%s %Q=?%d", sep, vtab->ColNames[i-2], i), true, &rc);
			sep = ",";
		}
		String_Concat(&z, _mprintf(" WHERE rowid=?%d", dataLength), true, &rc);
	}

	// If apData[0] is an integer and nData==1 then do a DELETE
	else if (dataLength == 1 && Vdbe::Value_Type(datas[0]) == TYPE_INTEGER)
	{
		z = _mprintf("DELETE FROM %Q WHERE rowid = ?1", vtab->TableName);
		if (!z)
			rc = RC_NOMEM;
		bindArgZero = true;
	}

	// If the first argument is NULL and there are more than two args, INSERT
	else if (dataLength > 2 && Vdbe::Value_Type(datas[0]) == TYPE_NULL)
	{
		char *values = nullptr;
		char *insert = _mprintf("INSERT INTO %Q (", vtab->TableName);
		if (!insert)
			rc = RC_NOMEM;
		if (Vdbe::Value_Type(datas[1]) == TYPE_INTEGER)
		{
			bindArgOne = true;
			values = _mprintf("?");
			String_Concat(&insert, "rowid", false, &rc);
		}
		_assert((vtab->Cols+2) == dataLength);
		for (int ii = 2; ii < dataLength; ii++)
		{
			String_Concat(&insert, _mprintf("%s%Q", (values ? ", " : ""), vtab->ColNames[ii-2]), true, &rc);
			String_Concat(&values, _mprintf("%s?%d", (values ? ", " : ""), ii), true, &rc);
		}
		String_Concat(&z, insert, true, &rc);
		String_Concat(&z, ") VALUES(", false, &rc);
		String_Concat(&z, values, true, &rc);
		String_Concat(&z, ")", false, &rc);
	}

	// Anything else is an error
	else
	{
		_assert(0);
		return RC_ERROR;
	}

	if (rc == RC_OK)
		rc = Prepare::Prepare_(ctx, z, -1, &stmt, nullptr);
	_assert(rc != RC_OK || stmt);
	_free(z);
	if (rc == RC_OK)
	{
		if (bindArgZero)
			Vdbe::Bind_Value(stmt, dataLength, datas[0]);
		if (bindArgOne)
			Vdbe::Bind_Value(stmt, 1, datas[1]);
		for (i = 2; i < dataLength && rc == RC_OK; i++)
			if (datas[i]) rc = Vdbe::Bind_Value(stmt, i, datas[i]);
		if (rc == RC_OK)
		{
			stmt->Step();
			rc = Vdbe::Finalize(stmt);
		}
		else
			Vdbe::Finalize(stmt);
	}

	if (rowid && rc == RC_OK)
		*rowid = Main::CtxLastInsertRowid(ctx);
	if (rc != RC_OK)
		tab->ErrMsg = _mprintf("echo-vtab-error: %s", Main::ErrMsg(ctx));
	return rc;
}

// xBegin, xSync, xCommit and xRollback callbacks for echo module virtual tables. Do nothing other than add the name of the callback
// to the $::echo_module Tcl variable.
__device__ static RC EchoTransactionCall(IVTable *tab, const char *call)
{
	echo_vtab *vtab = (echo_vtab *)tab;
	char *z = _mprintf("echo(%s)", vtab->TableName);
	if (!z) return RC_NOMEM;
	AppendToEchoModule(vtab->Interp, call);
	AppendToEchoModule(vtab->Interp, z);
	_free(z);
	return RC_OK;
}

__device__ static RC EchoBegin(IVTable *tab)
{
	echo_vtab *vtab = (echo_vtab *)tab;
	Jim_Interp *interp = vtab->Interp;
	// Ticket #3083 - do not start a transaction if we are already in a transaction
	_assert(!vtab->InTransaction);
	if (SimulateVtabError(vtab, "xBegin"))
		return RC_ERROR;

	RC rc = EchoTransactionCall(tab, "xBegin");
	if (rc == RC_OK)
	{
		// Check if the $::echo_module_begin_fail variable is defined. If it is, and it is set to the name of the real table underlying this virtual
		// echo module table, then cause this xSync operation to fail.
		const char *val = Jim_String(Jim_GetGlobalVariableStr(interp, "echo_module_begin_fail", 0));
		if (val && !_strcmp(val, vtab->TableName))
			rc = RC_ERROR;
	}
	if (rc == RC_OK)
		vtab->InTransaction = true;
	return rc;
}

__device__ static RC EchoSync(IVTable *tab)
{
	echo_vtab *vtab = (echo_vtab *)tab;
	Jim_Interp *interp = vtab->Interp;
	// Ticket #3083 - Only call xSync if we have previously started a transaction
	_assert(vtab->InTransaction );
	if (SimulateVtabError(vtab, "xSync"))
		return RC_ERROR;

	RC rc = EchoTransactionCall(tab, "xSync");
	if (rc == RC_OK)
	{
		// Check if the $::echo_module_sync_fail variable is defined. If it is, and it is set to the name of the real table underlying this virtual
		// echo module table, then cause this xSync operation to fail.
		const char *val = Jim_String(Jim_GetGlobalVariableStr(interp, "echo_module_sync_fail", 0));
		if (val && !_strcmp(val, vtab->TableName))
			rc = RC_INVALID;
	}
	return rc;
}

__device__ static RC EchoCommit(IVTable *tab)
{
	echo_vtab *vtab = (echo_vtab*)tab;
	// Ticket #3083 - Only call xCommit if we have previously started a transaction
	_assert(vtab->InTransaction);
	if (SimulateVtabError(vtab, "xCommit"))
		return RC_ERROR;

	_benignalloc_begin();
	RC rc = EchoTransactionCall(tab, "xCommit");
	_benignalloc_end();
	vtab->InTransaction = false;
	return rc;
}

__device__ static RC EchoRollback(IVTable *tab)
{
	echo_vtab *vtab = (echo_vtab*)tab;
	// Ticket #3083 - Only call xRollback if we have previously started a transaction
	_assert(vtab->InTransaction);

	RC rc = EchoTransactionCall(tab, "xRollback");
	vtab->InTransaction = false;
	return rc;
}

// Implementation of "GLOB" function on the echo module.  Pass all arguments to the ::echo_glob_overload procedure of TCL
// and return the result of that procedure as a string.
__device__ static void OverloadedGlobFunction(FuncContext *fctx, int argc, Mem **args)
{
	Jim_Interp *interp = (Jim_Interp *)Vdbe::User_Data(fctx);
	TextBuilder str;
	TextBuilder::Init(&str);
	str.AppendElement("::echo_glob_overload");
	for (int i = 0; i < argc; i++)
		str.AppendElement((char *)Vdbe::Value_Text(args[i]));
	int rc = Jim_Eval(interp, str.ToString());
	str.Reset();
	if (rc)
		Vdbe::Result_Error(fctx, Jim_String(Jim_GetResult(interp)), -1);
	else
		Vdbe::Result_Text(fctx, Jim_String(Jim_GetResult(interp)), -1, DESTRUCTOR_TRANSIENT);
	Jim_ResetResult(interp);
}

// This is the xFindFunction implementation for the echo module. SQLite calls this routine when the first argument of a function
// is a column of an echo virtual table.  This routine can optionally override the implementation of that function.  It will choose to
// do so if the function is named "glob", and a TCL command named ::echo_glob_overload exists.
__device__ static bool EchoFindFunction(IVTable *tab, int args, const char *funcName, void (**funcOut)(FuncContext*,int,Mem**), void **argOut)
{
	echo_vtab *vtab = (echo_vtab *)tab;
	Jim_Interp *interp = vtab->Interp;
	if (_strcmp(funcName, "glob"))
		return false;
	Jim_CmdInfo info;
	if (!Jim_GetCommandInfo(interp, "::echo_glob_overload", &info))
		return false;
	*funcOut = OverloadedGlobFunction;
	*argOut = interp;
	return true;
}

__device__ static RC EchoRename(IVTable *vtab, const char *newName)
{
	echo_vtab *p = (echo_vtab *)vtab;
	if (SimulateVtabError(p, "xRename"))
		return RC_ERROR;

	RC rc = RC_OK;
	if (p->IsPattern)
	{
		int thisLength = (int)_strlen(p->ThisName);
		char *sql = _mprintf("ALTER TABLE %s RENAME TO %s%s", p->TableName, newName, &p->TableName[thisLength]);
		rc = Main::Exec(p->Ctx, sql, nullptr, nullptr, nullptr);
		_free(sql);
	}
	return rc;
}

__device__ static RC EchoSavepoint(IVTable *vtab, int savepoint)
{
	_assert(vtab);
	return RC_OK;
}

__device__ static RC EchoRelease(IVTable *vtab, int savepoint)
{
	_assert(vtab);
	return RC_OK;
}

__device__ static RC EchoRollbackTo(IVTable *vtab, int savepoint)
{
	_assert(vtab);
	return RC_OK;
}

// A virtual table module that merely "echos" the contents of another table (like an SQL VIEW).
__device__ static ITableModule _echoModule =
{
	1,                         // iVersion
	EchoCreate,
	EchoConnect,
	EchoBestIndex,
	EchoDisconnect, 
	EchoDestroy,
	EchoOpen,                  // xOpen - open a cursor
	EchoClose,                 // xClose - close a cursor
	EchoFilter,                // xFilter - configure scan constraints
	EchoNext,                  // xNext - advance a cursor
	EchoEof,                   // xEof */
	EchoColumn,                // xColumn - read data
	EchoRowid,                 // xRowid - read data
	EchoUpdate,                // xUpdate - write data
	EchoBegin,                 // xBegin - begin transaction
	EchoSync,                  // xSync - sync transaction
	EchoCommit,                // xCommit - commit transaction
	EchoRollback,              // xRollback - rollback transaction
	EchoFindFunction,          // xFindFunction - function overloading
	EchoRename                 // xRename - rename the table
};

__device__ static ITableModule _echoModuleV2 =
{
	2,                         // iVersion
	EchoCreate,
	EchoConnect,
	EchoBestIndex,
	EchoDisconnect, 
	EchoDestroy,
	EchoOpen,                  // xOpen - open a cursor
	EchoClose,                 // xClose - close a cursor
	EchoFilter,                // xFilter - configure scan constraints
	EchoNext,                  // xNext - advance a cursor
	EchoEof,                   // xEof
	EchoColumn,                // xColumn - read data
	EchoRowid,                 // xRowid - read data
	EchoUpdate,                // xUpdate - write data
	EchoBegin,                 // xBegin - begin transaction
	EchoSync,                  // xSync - sync transaction
	EchoCommit,                // xCommit - commit transaction
	EchoRollback,              // xRollback - rollback transaction
	EchoFindFunction,          // xFindFunction - function overloading
	EchoRename,                // xRename - rename the table
	EchoSavepoint,
	EchoRelease,
	EchoRollbackTo
};

// Decode a pointer to an sqlite3 object.
__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctxOut);
__device__ extern const char *sqlite3TestErrorName(int rc);

__device__ static void ModuleDestroy(void *p)
{
	_free(p);
}

// Register the echo virtual table module.
__device__ static int register_echo_module(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;

	// Virtual table module "echo"
	EchoModule *mod = (EchoModule *)_alloc(sizeof(EchoModule));
	mod->Interp = interp;
	RC rc = VTable::CreateModule(ctx, "echo", &_echoModule, (void *)mod, ModuleDestroy);

	// Virtual table module "echo_v2"
	if (rc == RC_OK)
	{
		mod = (EchoModule *)_alloc(sizeof(EchoModule));
		mod->Interp = interp;
		rc = VTable::CreateModule(ctx, "echo_v2", &_echoModuleV2, (void *)mod, ModuleDestroy);
	}

	Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
	return JIM_OK;
}

// Tcl interface to sqlite3_declare_vtab, invoked as follows from Tcl:
//
// sqlite3_declare_vtab DB SQL
__device__ static int declare_vtab(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB SQL");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	RC rc = VTable::DeclareVTable(ctx, Jim_String(args[2]));
	if (rc != RC_OK)
	{
		Jim_SetResultString(interp, (char *)Main::ErrMsg(ctx), -1);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Register the spellfix virtual table module.
//#include "Test_SpellFix.cu.inc"
__device__ static int register_spellfix_module(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	//sqlite3_spellfix1_register(ctx);
	return JIM_OK;
}

#endif

// Register commands with the TCL interpreter.
#ifndef OMIT_VIRTUALTABLE
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_echo_module",       register_echo_module, nullptr },
	{ "register_spellfix_module",   register_spellfix_module, nullptr },
	{ "sqlite3_declare_vtab",       declare_vtab, nullptr },
};
#endif
__device__ int Sqlitetest8_Init(Jim_Interp *interp)
{
#ifndef OMIT_VIRTUALTABLE
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
#endif
	return JIM_OK;
}
