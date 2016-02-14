// This file implements a read-only VIRTUAL TABLE that contains the content of a C-language array of integer values.  See the corresponding
// header file for full details.
#include "test_intarray.h"

// Definition of the sqlite3_intarray object.
//
// The internal representation of an intarray object is subject to change, is not externally visible, and should be used by
// the implementation of intarray only.  This object is opaque to users.
struct sqlite3_intarray
{
	int n;                  // Number of elements in the array
	int64 *a;				// Contents of the array
	void (*Free)(void*);    // Function used to free a[]
};

// Objects used internally by the virtual table implementation
typedef struct intarray_vtab intarray_vtab;
typedef struct intarray_cursor intarray_cursor;

// A intarray table object
struct intarray_vtab
{
	IVTable base;				// Base class
	sqlite3_intarray *Content;	// Content of the integer array
};

// A intarray cursor object
struct intarray_cursor
{
	IVTableCursor base;			// Base class
	int i;                      // Current cursor position
};

// None of this works unless we have virtual tables.
#ifndef OMIT_VIRTUALTABLE

// Free an sqlite3_intarray object.
__device__ static void intarrayFree(sqlite3_intarray *p)
{
	if (p->Free)
		p->Free(p->a);
	_free(p);
}

// Table destructor for the intarray module.
__device__ static RC intarrayDestroy(IVTable *p)
{
	intarray_vtab *vtab = (intarray_vtab *)p;
	_free(vtab);
	return RC_OK;
}

// Table constructor for the intarray module.
__device__ static RC intarrayCreate(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtab, char **err){
	RC rc = RC_NOMEM;
	intarray_vtab *p = (intarray_vtab *)_alloc(sizeof(intarray_vtab));
	if (p)
	{
		_memset(p, 0, sizeof(intarray_vtab));
		p->Content = (sqlite3_intarray *)aux;
		rc = VTable::DeclareVTable(ctx, "CREATE TABLE x(value INTEGER PRIMARY KEY)");
	}
	*vtab = (IVTable *)p;
	return rc;
}

// Open a new cursor on the intarray table.
__device__ static RC intarrayOpen(IVTable *vTab, IVTableCursor **cursor)
{
	RC rc = RC_NOMEM;
	intarray_cursor *cur = (intarray_cursor *)_alloc(sizeof(intarray_cursor));
	if (!cur)
	{
		memset(cur, 0, sizeof(intarray_cursor));
		*cursor = (IVTableCursor *)cur;
		rc = RC_OK;
	}
	return rc;
}

// Close a intarray table cursor.
__device__ static RC intarrayClose(IVTableCursor *cur_)
{
	intarray_cursor *cur = (intarray_cursor *)cur_;
	_free(cur);
	return RC_OK;
}

// Retrieve a column of data.
__device__ static RC intarrayColumn(IVTableCursor *cur_, FuncContext *fctx, int i)
{
	intarray_cursor *cur = (intarray_cursor*)cur_;
	intarray_vtab *vtab = (intarray_vtab*)cur_->IVTable;
	if (cur->i >= 0 && cur->i < vtab->Content->n)
		Vdbe::Result_Int64(fctx, vtab->Content->a[cur->i]);
	return RC_OK;
}

// Retrieve the current rowid.
__device__ static RC intarrayRowid(IVTableCursor *cur_, int64 *rowid)
{
	intarray_cursor *cur = (intarray_cursor *)cur_;
	*rowid = cur->i;
	return RC_OK;
}

__device__ static bool intarrayEof(IVTableCursor *cur_)
{
	intarray_cursor *cur = (intarray_cursor *)cur_;
	intarray_vtab *vtab = (intarray_vtab *)cur_->IVTable;
	return cur->i >= vtab->Content->n;
}

// Advance the cursor to the next row.
__device__ static RC intarrayNext(IVTableCursor *cur_)
{
	intarray_cursor *cur = (intarray_cursor *)cur_;
	cur->i++;
	return RC_OK;
}

// Reset a intarray table cursor.
__device__ static RC intarrayFilter(IVTableCursor *cur_, int idxNum, const char *idxStr, int argc, Mem **args)
{
	intarray_cursor *cur = (intarray_cursor *)cur_;
	cur->i = 0;
	return RC_OK;
}

// Analyse the WHERE condition.
__device__ static RC intarrayBestIndex(IVTable *tab, IIndexInfo *idxInfo)
{
	return RC_OK;
}

// A virtual table module that merely echos method calls into TCL variables.
__constant__ static ITableModule _intarrayModule = {
	0,							// Version
	intarrayCreate,				// Create - create a new virtual table
	intarrayCreate,				// Connect - connect to an existing vtab
	intarrayBestIndex,			// BestIndex - find the best query index
	intarrayDestroy,			// Disconnect - disconnect a vtab
	intarrayDestroy,			// Destroy - destroy a vtab
	intarrayOpen,				// Open - open a cursor
	intarrayClose,				// Close - close a cursor
	intarrayFilter,				// Filter - configure scan constraints
	intarrayNext,				// Next - advance a cursor
	intarrayEof,				// Eof
	intarrayColumn,				// Column - read data
	intarrayRowid,				// Rowid - read data
	nullptr,					// Update
	nullptr,					// Begin
	nullptr,					// Sync
	nullptr,					// Commit
	nullptr,					// Rollback
	nullptr,					// FindMethod
	nullptr,					// Rename
};

#endif

// Invoke this routine to create a specific instance of an intarray object. The new intarray object is returned by the 3rd parameter.
//
// Each intarray object corresponds to a virtual table in the TEMP table with a name of zName.
//
// Destroy the intarray object by dropping the virtual table.  If not done explicitly by the application, the virtual table will be dropped implicitly
// by the system when the database connection is closed.
__device__ RC sqlite3_intarray_create(Context *ctx, const char *name, sqlite3_intarray **ret)
{
	RC rc = RC_OK;
#ifndef OMIT_VIRTUALTABLE
	sqlite3_intarray *p = *ret = (sqlite3_intarray *)_alloc(sizeof(*p));
	if (!p)
		return RC_NOMEM;
	_memset(p, 0, sizeof(*p));
	rc = VTable::CreateModule(ctx, name, &_intarrayModule, p, (void(*)(void*))intarrayFree);
	if (rc == RC_OK)
	{
		char *sql = _mprintf("CREATE VIRTUAL TABLE temp.%Q USING %Q", name, name);
		rc = DataEx::Exec(ctx, sql, nullptr, nullptr, nullptr);
		_free(sql);
	}
#endif
	return rc;
}

// Bind a new array array of integers to a specific intarray object.
//
// The array of integers bound must be unchanged for the duration of any query against the corresponding virtual table.  If the integer
// array does change or is deallocated undefined behavior will result.
__device__ RC sqlite3_intarray_bind(sqlite3_intarray *array_, int elementsLength, int64 *elements, void (*free)(void*))
{
	if (array_->Free)
		array_->Free(array_->a);
	array_->n = elementsLength;
	array_->a = elements;
	array_->Free = free;
	return RC_OK;
}

// Everything below is interface for testing this module.
#ifdef _TEST
#include <JimEx.h>

// Routines to encode and decode pointers
__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);
__device__ extern void *sqlite3TestTextToPtr(const char *);
__device__ extern int sqlite3TestMakePointerStr(Jim_Interp*, char *, void *);
__device__ extern const char *sqlite3TestErrorName(int);

// sqlite3_intarray_create  DB  NAME
//
// Invoke the sqlite3_intarray_create interface.  A string that becomes the first parameter to sqlite3_intarray_bind.
__device__ static int test_intarray_create(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	const char *name = Jim_String(args[2]);
	sqlite3_intarray *array_;
	RC rc = RC_OK;
#ifndef OMIT_VIRTUALTABLE
	rc = sqlite3_intarray_create(ctx, name, &array_);
#endif
	if (rc != RC_OK)
	{
		_assert(array_ == nullptr);
		Jim_AppendResult(interp, sqlite3TestErrorName(rc), nullptr);
		return JIM_ERROR;
	}
	char ptr[100];
	sqlite3TestMakePointerStr(interp, ptr, array_);
	Jim_AppendResult(interp, ptr, (char*)0);
	return JIM_OK;
}

// sqlite3_intarray_bind  INTARRAY  ?VALUE ...?
//
// Invoke the sqlite3_intarray_bind interface on the given array of integers.
__device__ static int test_intarray_bind(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc < 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "INTARRAY");
		return JIM_ERROR;
	}
	sqlite3_intarray *array_ = (sqlite3_intarray *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int n = argc - 2;
#ifndef OMIT_VIRTUALTABLE
	int64 *a = (int64 *)_alloc(sizeof(a[0])*n);
	if (!a)
	{
		Jim_AppendResult(interp, "SQLITE_NOMEM", nullptr);
		return JIM_ERROR;
	}
	for (int i = 0; i < n; i++)
	{
		long long x = 0;
		Jim_GetWide(interp, args[i+2], &x);
		a[i] = x;
	}
	RC rc = sqlite3_intarray_bind(array_, n, a, _free);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, sqlite3TestErrorName(rc), nullptr);
		return JIM_ERROR;
	}
#endif
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "sqlite3_intarray_create", test_intarray_create, nullptr },
	{ "sqlite3_intarray_bind", test_intarray_bind, nullptr },
};
__device__ int Sqlitetestintarray_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
	return JIM_OK;
}

#endif
