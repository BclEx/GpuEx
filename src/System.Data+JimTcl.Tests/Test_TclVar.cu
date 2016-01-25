
// Code for testing the virtual table interfaces.  This code is not included in the SQLite library.  It is used for automated
// testing of the SQLite library.
//
// The emphasis of this file is a virtual table that provides access to TCL variables.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>

#ifndef OMIT_VIRTUALTABLE

typedef struct tclvar_vtab tclvar_vtab;
typedef struct tclvar_cursor tclvar_cursor;

// A tclvar virtual-table object 
struct tclvar_vtab
{
	IVTable base;
	Jim_Interp *interp;
};

// A tclvar cursor object 
struct tclvar_cursor
{
	IVTableCursor base;
	Jim_Obj *List1;     // Result of [info vars ?pattern?]
	Jim_Obj *List2;     // Result of [array names [lindex $pList1 $i1]]
	int I1;             // Current item in pList1
	int I2;             // Current item (if any) in pList2
};

// Methods for the tclvar module
__constant__ static const char _schema[] = "CREATE TABLE whatever(name TEXT, arrayname TEXT, value TEXT)";
// Note that for this virtual table, the xCreate and xConnect methods are identical.
__device__ static RC tclvarConnect(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtabOut, char **err)
{
	tclvar_vtab *vtab = (tclvar_vtab *)_allocZero(sizeof(*vtab));
	if (!vtab) return RC_NOMEM;
	*vtabOut = &vtab->base;
	vtab->interp = (Jim_Interp *)aux;
	VTable::DeclareVTable(ctx, _schema);
	return RC_OK;
}

// The xDisconnect and xDestroy methods are also the same
__device__ static RC tclvarDisconnect(IVTable *vtab)
{
	_free(vtab);
	return RC_OK;
}

// Open a new tclvar cursor.
__device__ static RC tclvarOpen(IVTable *vtab, IVTableCursor **cursor)
{
	tclvar_cursor *cur = (tclvar_cursor *)_allocZero(sizeof(tclvar_cursor));
	*cursor = &cur->base;
	return RC_OK;
}

// Close a tclvar cursor.
__device__ static RC tclvarClose(IVTableCursor *cur_)
{
	tclvar_cursor *cur = (tclvar_cursor *)cur_;
	Jim_Interp *interp = ((tclvar_vtab *)(cur_->IVTable))->interp;
	if (cur->List1)
		Jim_DecrRefCount(interp, cur->List1);
	if (cur->List2)
		Jim_DecrRefCount(interp, cur->List2);
	_free(cur);
	return RC_OK;
}

// Returns 1 if data is ready, or 0 if not.
__device__ static bool next2(Jim_Interp *interp, tclvar_cursor *cur, Jim_Obj *obj)
{
	if (obj)
	{
		if (!cur->List2)
		{
			Jim_Obj *p = Jim_NewStringObj(interp, "array names", -1);
			Jim_IncrRefCount(p);
			Jim_ListAppendElement(nullptr, p, obj);
			Jim_EvalObj(interp, p);
			Jim_DecrRefCount(interp, p);
			cur->List2 = Jim_GetResult(interp);
			Jim_IncrRefCount(cur->List2);
			_assert(cur->I2 == 0);
		}
		else
		{
			cur->I2++;
			int n = Jim_ListLength(nullptr, cur->List2);
			if (cur->I2 >= n)
			{
				Jim_DecrRefCount(interp, cur->List2);
				cur->List2 = nullptr;
				cur->I2 = 0;
				return false;
			}
		}
	}
	return true;
}

__device__ static RC tclvarNext(IVTableCursor *cur_)
{
	tclvar_cursor *cur = (tclvar_cursor *)cur_;
	Jim_Interp *interp = ((tclvar_vtab *)(cur_->IVTable))->interp;
	int n = Jim_ListLength(nullptr, cur->List1);
	bool ok = false;
	while (!ok && cur->I1 < n)
	{
		Jim_Obj *obj = Jim_ListGetIndex(interp, cur->List1, cur->I1);
		ok = next2(interp, cur, obj);
		if (!ok)
			cur->I1++;
	}
	return RC_OK;
}

__device__ static RC tclvarFilter(IVTableCursor *vtabCursor, int idxNum, const char *idxStr, int argc, Mem **args)
{
	tclvar_cursor *cur = (tclvar_cursor *)vtabCursor;
	Jim_Interp *interp = ((tclvar_vtab *)(vtabCursor->IVTable))->interp;
	Jim_Obj *p = Jim_NewStringObj(interp, "info vars", -1);
	Jim_IncrRefCount(p);
	_assert(argc == 0 || argc == 1);
	if (argc == 1)
	{
		Jim_Obj *arg = Jim_NewStringObj(interp, (char *)Vdbe::Value_Text(args[0]), -1);
		Jim_ListAppendElement(nullptr, p, arg);
	}
	Jim_EvalObj(interp, p);
	if (cur->List1)
		Jim_DecrRefCount(interp, cur->List1);
	if (cur->List2)
	{
		Jim_DecrRefCount(interp, cur->List2);
		cur->List2 = nullptr;
	}
	cur->I1 = 0;
	cur->I2 = 0;
	cur->List1 = Jim_GetResult(interp);
	Jim_IncrRefCount(cur->List1);
	assert(cur->I1 == 0 && cur->I2 == 0 && !cur->List2);
	Jim_DecrRefCount(interp, p);
	return tclvarNext(vtabCursor);
}

__device__ static RC tclvarColumn(IVTableCursor *cur_, FuncContext *fctx, int i)
{
	tclvar_cursor *cur = (tclvar_cursor*)cur_;
	Jim_Interp *interp = ((tclvar_vtab *)cur_->IVTable)->interp;
	Jim_Obj *p1 = Jim_ListGetIndex(interp, cur->List1, cur->I1);
	Jim_Obj *p2 = Jim_ListGetIndex(interp, cur->List2, cur->I2);
	const char *z1 = Jim_String(p1);
	const char *z2 = (p2 ? Jim_String(p2) : "");
	switch (i) {
	case 0: {
		Vdbe::Result_Text(fctx, z1, -1, DESTRUCTOR_TRANSIENT);
		break; }
	case 1: {
		Vdbe::Result_Text(fctx, z2, -1, DESTRUCTOR_TRANSIENT);
		break; }
	case 2: {
		Jim_Obj *val = Jim_GetVar2(interp, z1, (*z2 ? z2 : nullptr), 0);
		Vdbe::Result_Text(fctx, Jim_String(val), -1, DESTRUCTOR_TRANSIENT);
		break; }
	}
	return RC_OK;
}

__device__ static RC tclvarRowid(IVTableCursor *cur, int64 *rowid)
{
	*rowid = 0;
	return RC_OK;
}

__device__ static bool tclvarEof(IVTableCursor *cur_)
{
	tclvar_cursor *cur = (tclvar_cursor *)cur_;
	return (cur->List2 ? false : true);
}

__device__ static RC tclvarBestIndex(IVTable *tab, IIndexInfo *idxInfo)
{
	int ii;
	for (ii = 0; ii < idxInfo->Constraints.length; ii++)
	{
		IIndexInfo::Constraint const *cons = &idxInfo->Constraints[ii];
		if (cons->Column == 0 && cons->Usable && cons->OP == INDEX_CONSTRAINT_EQ)
		{
			IIndexInfo::ConstraintUsage *usage = &idxInfo->ConstraintUsages[ii];
			usage->Omit = 0;
			usage->ArgvIndex = 1;
			return RC_OK;
		}
	}
	for (ii = 0; ii < idxInfo->Constraints.length; ii++)
	{
		IIndexInfo::Constraint const *cons = &idxInfo->Constraints[ii];
		if (cons->Column == 0 && cons->Usable && cons->OP == INDEX_CONSTRAINT_MATCH)
		{
			IIndexInfo::ConstraintUsage *usage = &idxInfo->ConstraintUsages[ii];
			usage->Omit = 1;
			usage->ArgvIndex = 1;
			return RC_OK;
		}
	}
	return RC_OK;
}

// A virtual table module that provides read-only access to a Tcl global variable namespace.
__constant__ static ITableModule _tclvarModule = {
	0,							// Version
	tclvarConnect,
	tclvarConnect,
	tclvarBestIndex,
	tclvarDisconnect, 
	tclvarDisconnect,
	tclvarOpen,					// Open - open a cursor
	tclvarClose,				// Close - close a cursor
	tclvarFilter,				// Filter - configure scan constraints
	tclvarNext,					// Next - advance a cursor
	tclvarEof,					// Eof - check for end of scan
	tclvarColumn,				// Column - read data
	tclvarRowid,				// Rowid - read data
	nullptr,					// Update
	nullptr,					// Begin
	nullptr,					// Sync
	nullptr,					// Commit
	nullptr,					// Rollback
	nullptr,					// FindMethod
	nullptr,					// Rename
};

// Decode a pointer to an sqlite3 object.
__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);

// Register the echo virtual table module.
__device__ static int register_tclvar_module(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
#ifndef OMIT_VIRTUALTABLE
	VTable::CreateModule(ctx, "tclvar", &_tclvarModule, (void *)interp, nullptr);
#endif
	return JIM_OK;
}

#endif

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_tclvar_module", register_tclvar_module, nullptr },
};
__device__ int Sqlitetesttclvar_Init(Jim_Interp *interp)
{
#ifndef OMIT_VIRTUALTABLE
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
#endif
	return JIM_OK;
}
