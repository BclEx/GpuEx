
// Code for testing the virtual table interfaces.  This code is not included in the SQLite library.  It is used for automated
// testing of the SQLite library.

// The code in this file defines a sqlite3 virtual-table module that provides a read-only view of the current database schema. There is one
// row in the schema table for each column in the database schema.
#define _SCHEMA_ \
	"CREATE TABLE x("                                                            \
	"database,"          /* Name of database (i.e. main, temp etc.) */         \
	"tablename,"         /* Name of table */                                   \
	"cid,"               /* Column number (from left-to-right, 0 upward) */    \
	"name,"              /* Column name */                                     \
	"type,"              /* Specified type (i.e. VARCHAR(32)) */               \
	"not_null,"          /* Boolean. True if NOT NULL was specified */         \
	"dflt_value,"        /* Default value for this column */                   \
	"pk"                 /* True if this column is part of the primary key */  \
	")"

// If SQLITE_TEST is defined this code is preprocessed for use as part of the sqlite test binary "testfixture". Otherwise it is preprocessed
// to be compiled into an sqlite dynamic extension.
//#ifdef _TEST
#include <Core+Vdbe\VdbeInt.cu.h>
#include <Tcl.h>
//#else
//#include "sqlite3ext.h"
//SQLITE_EXTENSION_INIT1
//#endif

#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct schema_vtab schema_vtab;
typedef struct schema_cursor schema_cursor;

// A schema table object
struct schema_vtab
{
	IVTable base;
	Context *Ctx;
};

// A schema table cursor object
struct schema_cursor
{
	IVTableCursor base;
	Vdbe *DbList;
	Vdbe *TableList;
	Vdbe *ColumnList;
	int Rowid;
};

// None of this works unless we have virtual tables.
#ifndef OMIT_VIRTUALTABLE

// Table destructor for the schema module.
__device__ static RC SchemaDestroy(IVTable *vtab)
{
	_free(vtab);
	return RC_OK;
}

// Table constructor for the schema module.
__device__ static RC SchemaCreate(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtable, char **err)
{
	RC rc = RC_NOMEM;
	schema_vtab *vtab = (schema_vtab *)_alloc(sizeof(schema_vtab));
	if (vtab)
	{
		_memset(vtab, 0, sizeof(schema_vtab));
		vtab->Ctx = ctx;
#ifndef OMIT_VIRTUALTABLE
		rc = VTable::DeclareVTable(ctx, _SCHEMA_);
#endif
	}
	*vtable = (IVTable *)vtab;
	return rc;
}

// Open a new cursor on the schema table.
__device__ static RC SchemaOpen(IVTable *vTab, IVTableCursor **cursor)
{
	RC rc = RC_NOMEM;
	schema_cursor *cur = (schema_cursor *)_alloc(sizeof(schema_cursor));
	if (cur)
	{
		_memset(cur, 0, sizeof(schema_cursor));
		*cursor = (IVTableCursor *)cur;
		rc = RC_OK;
	}
	return rc;
}

// Close a schema table cursor.
__device__ static RC SchemaClose(IVTableCursor *cur2)
{
	schema_cursor *cur = (schema_cursor *)cur2;
	Vdbe::Finalize(cur->DbList);
	Vdbe::Finalize(cur->TableList);
	Vdbe::Finalize(cur->ColumnList);
	_free(cur);
	return RC_OK;
}

// Retrieve a column of data.
__device__ static RC SchemaColumn(IVTableCursor *cur2, FuncContext *fctx, int i)
{
	schema_cursor *cur = (schema_cursor *)cur2;
	switch (i)
	{
	case 0: Vdbe::Result_Value(fctx, Vdbe::Column_Value(cur->DbList, 1)); break;
	case 1: Vdbe::Result_Value(fctx, Vdbe::Column_Value(cur->TableList, 0)); break;
	default: Vdbe::Result_Value(fctx, Vdbe::Column_Value(cur->ColumnList, i-2)); break;
	}
	return RC_OK;
}

// Retrieve the current rowid.
__device__ static RC SchemaRowid(IVTableCursor *cur2, int64 *rowid)
{
	schema_cursor *cur = (schema_cursor *)cur2;
	*rowid = cur->Rowid;
	return RC_OK;
}

__device__ static RC Finalize(Vdbe **stmt)
{
	RC rc = Vdbe::Finalize(*stmt);
	*stmt = nullptr;
	return rc;
}

__device__ static bool SchemaEof(IVTableCursor *cur2)
{
	schema_cursor *cur = (schema_cursor *)cur2;
	return (cur->DbList ? false : true);
}

// Advance the cursor to the next row.
__device__ static RC SchemaNext(IVTableCursor *cur2)
{
	RC rc = RC_OK;
	schema_cursor *cur = (schema_cursor *)cur2;
	schema_vtab *vtab = (schema_vtab *)(cur2->IVTable);
	char *sql = nullptr;
	while (!cur->ColumnList || cur->ColumnList->Step() != RC_ROW)
	{
		if ((rc = Finalize(&cur->ColumnList)) != RC_OK) goto next_exit;
		while (!cur->TableList || cur->TableList->Step() != RC_ROW)
		{
			if ((rc = Finalize(&cur->TableList)) != RC_OK) goto next_exit;
			_assert(cur->DbList);
			while (cur->DbList->Step() != RC_ROW)
			{
				rc = Finalize(&cur->DbList);
				goto next_exit;
			}
			// Set zSql to the SQL to pull the list of tables from the sqlite_master (or sqlite_temp_master) table of the database
			// identfied by the row pointed to by the SQL statement pCur->pDbList (iterating through a "PRAGMA database_list;" statement).
			if (Vdbe::Column_Int(cur->DbList, 0) == 1)
				sql = _mprintf("SELECT name FROM sqlite_temp_master WHERE type='table'");
			else
			{
				Vdbe *dbList = cur->DbList;
				sql = _mprintf("SELECT name FROM %Q.sqlite_master WHERE type='table'", Vdbe::Column_Text(dbList, 1));
			}
			if (!sql)
			{
				rc = RC_NOMEM;
				goto next_exit;
			}
			rc = Prepare::Prepare_(vtab->Ctx, sql, -1, &cur->TableList, 0);
			_free(sql);
			if (rc != RC_OK) goto next_exit;
		}

		// Set zSql to the SQL to the table_info pragma for the table currently identified by the rows pointed to by statements pCur->pDbList and
		// pCur->pTableList.
		sql = _mprintf("PRAGMA %Q.table_info(%Q)", Vdbe::Column_Text(cur->DbList, 1), Vdbe::Column_Text(cur->TableList, 0));

		if (!sql)
		{
			rc = RC_NOMEM;
			goto next_exit;
		}
		rc = Prepare::Prepare_(vtab->Ctx, sql, -1, &cur->ColumnList, 0);
		_free(sql);
		if (rc != RC_OK) goto next_exit;
	}
	cur->Rowid++;

next_exit:
	// TODO: Handle rc
	return rc;
}

// Reset a schema table cursor.
__device__ static RC SchemaFilter(IVTableCursor *vtabCursor, int idxNum, const char *idxStr, int argc, Mem **args)
{
	schema_vtab *vtab = (schema_vtab *)(vtabCursor->IVTable);
	schema_cursor *cur = (schema_cursor *)vtabCursor;
	cur->Rowid = 0;
	Finalize(&cur->TableList);
	Finalize(&cur->ColumnList);
	Finalize(&cur->DbList);
	RC rc = Prepare::Prepare_(vtab->Ctx, "PRAGMA database_list", -1, &cur->DbList, 0);
	return (rc == RC_OK ? SchemaNext(vtabCursor) : rc);
}

// Analyse the WHERE condition.
__device__ static RC SchemaBestIndex(IVTable *vtab, IIndexInfo *idxInfo)
{
	return RC_OK;
}

// A virtual table module that merely echos method calls into TCL variables.
__constant__ static ITableModule _schemaModule =
{
	0,							// iVersion
	SchemaCreate,
	SchemaCreate,
	SchemaBestIndex,
	SchemaDestroy,
	SchemaDestroy,
	SchemaOpen,					// xOpen - open a cursor
	SchemaClose,				// xClose - close a cursor
	SchemaFilter,				// xFilter - configure scan constraints
	SchemaNext,					// xNext - advance a cursor
	SchemaEof,					// xEof
	SchemaColumn,				// xColumn - read data
	SchemaRowid,				// xRowid - read data
	nullptr,					// xUpdate
	nullptr,					// xBegin
	nullptr,					// xSync
	nullptr,					// xCommit
	nullptr,					// xRollback
	nullptr,					// xFindMethod
	nullptr,                    // xRename
};

#endif

#ifdef _TEST

// Decode a pointer to an sqlite3 object.
__device__ extern int GetDbPointer(Tcl_Interp *interp, char *a, Context **ctx);

// Register the schema virtual table module.
__device__ static int register_schema_module(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2)
	{
		Tcl_WrongNumArgs(interp, 1, args, "DB");
		return TCL_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, (char *)args[1], &ctx)) return TCL_ERROR;
#ifndef OMIT_VIRTUALTABLE
	VTable::CreateModule(ctx, "schema", &_schemaModule, nullptr, nullptr);
#endif
	return TCL_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Tcl_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_schema_module", register_schema_module, nullptr },
};
__device__ int Sqlitetestschema_Init(Tcl_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Tcl_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
	return TCL_OK;
}

#else

// Extension load function.
__device__ int sqlite3_extension_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	EXTENSION_INIT2(api);
#ifndef OMIT_VIRTUALTABLE
	VTable::CreateModule(ctx, "schema", &_schemaModule, nullptr, nullptr);
#endif
	return 0;
}

#endif
