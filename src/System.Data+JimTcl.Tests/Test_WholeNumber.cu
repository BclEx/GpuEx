
// This file implements a virtual table that returns the whole numbers between 1 and 4294967295, inclusive.
// Example:
//     CREATE VIRTUAL TABLE nums USING wholenumber;
//     SELECT value FROM nums WHERE value<10;
//
// Results in:
//     1 2 3 4 5 6 7 8 9
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <assert.h>
#include <string.h>

#ifndef OMIT_VIRTUALTABLE

// A wholenumber cursor object
typedef struct wholenumber_cursor wholenumber_cursor;
struct wholenumber_cursor
{
	IVTableCursor base;  // Base class - must be first
	int64 Value;      // Current value
	int64 MaxValue;     // Maximum value
};

// Methods for the wholenumber module
// Note that for this virtual table, the xCreate and xConnect methods are identical.
__device__ static RC WholenumberConnect(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtable, char **err)
{
	IVTable *new_ = *vtable = (IVTable *)_alloc(sizeof(*new_));
	if (!new_) return RC_NOMEM;
	VTable::DeclareVTable(ctx, "CREATE TABLE x(value)");
	_memset(new_, 0, sizeof(*new_));
	return RC_OK;
}

// The xDisconnect and xDestroy methods are also the same
__device__ static RC WholenumberDisconnect(IVTable *vtab)
{
	_free(vtab);
	return RC_OK;
}

// Open a new wholenumber cursor.
__device__ static RC WholenumberOpen(IVTable *p, IVTableCursor **cursor)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)_alloc(sizeof(*cur));
	if (!cur) return RC_NOMEM;
	_memset(cur, 0, sizeof(*cur));
	*cursor = &cur->base;
	return RC_OK;
}

// Close a wholenumber cursor.
__device__ static RC WholenumberClose(IVTableCursor *cur)
{
	_free(cur);
	return RC_OK;
}

// Advance a cursor to its next row of output
__device__ static RC WholenumberNext(IVTableCursor *cur2)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)cur2;
	cur->Value++;
	return RC_OK;
}

// Return the value associated with a wholenumber.
__device__ static RC WholenumberColumn(IVTableCursor *cur2, FuncContext *fctx, int i)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)cur2;
	Vdbe::Result_Int64(fctx, cur->Value);
	return RC_OK;
}

// The rowid.
__device__ static RC WholenumberRowid(IVTableCursor *cur2, int64 *rowid)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)cur2;
	*rowid = cur->Value;
	return RC_OK;
}

// When the wholenumber_cursor.rLimit value is 0 or less, that is a signal that the cursor has nothing more to output.
__device__ static bool WholenumberEof(IVTableCursor *cur2)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)cur2;
	return (cur->Value > cur->MaxValue || cur->Value == 0);
}

// Called to "rewind" a cursor back to the beginning so that it starts its output over again.  Always called at least once
// prior to any wholenumberColumn, wholenumberRowid, or wholenumberEof call.
//
//    idxNum   Constraints
//    ------   ---------------------
//      0      (none)
//      1      value > $argv0
//      2      value >= $argv0
//      4      value < $argv0
//      8      value <= $argv0
//
//      5      value > $argv0 AND value < $argv1
//      6      value >= $argv0 AND value < $argv1
//      9      value > $argv0 AND value <= $argv1
//     10      value >= $argv0 AND value <= $argv1
__device__ static RC WholenumberFilter(IVTableCursor *vtabCursor, int idxNum, const char *idxStr, int argc, Mem **args)
{
	wholenumber_cursor *cur = (wholenumber_cursor *)vtabCursor;
	int64 v;
	int i = 0;
	cur->Value = 1;
	cur->MaxValue = 0xffffffff; // 4294967295
	if (idxNum & 3)
	{
		v = Vdbe::Value_Int64(args[0]) + (idxNum&1);
		if (v > cur->Value && v <= cur->MaxValue) cur->Value = v;
		i++;
	}
	if (idxNum & 12)
	{
		v = Vdbe::Value_Int64(args[i]) - ((idxNum>>2)&1);
		if (v >= cur->Value && v < cur->MaxValue) cur->MaxValue = v;
	}
	return RC_OK;
}

// Search for terms of these forms:
//
//  (1)  value > $value
//  (2)  value >= $value
//  (4)  value < $value
//  (8)  value <= $value
//
// idxNum is an ORed combination of 1 or 2 with 4 or 8.
__device__ static RC WholenumberBestIndex(IVTable *vtab, IIndexInfo *idxInfo)
{
	int idxNum = 0;
	int argvIdx = 1;
	int ltIdx = -1;
	int gtIdx = -1;
	const IIndexInfo::Constraint *constraint = idxInfo->Constraints.data;
	for (int i = 0; i < idxInfo->Constraints.length; i++, constraint++)
	{
		if (!constraint->Usable) continue;
		if ((idxNum & 3) == 0 && constraint->OP == INDEX_CONSTRAINT_GT) { idxNum |= 1; ltIdx = i; }
		if ((idxNum & 3) == 0 && constraint->OP == INDEX_CONSTRAINT_GE) { idxNum |= 2; ltIdx = i; }
		if ((idxNum & 12) == 0 && constraint->OP == INDEX_CONSTRAINT_LT) { idxNum |= 4; gtIdx = i; }
		if ((idxNum & 12) == 0 && constraint->OP == INDEX_CONSTRAINT_LE) { idxNum |= 8; gtIdx = i; }
	}
	idxInfo->IdxNum = idxNum;
	if (ltIdx >= 0)
	{
		idxInfo->ConstraintUsages[ltIdx].ArgvIndex = argvIdx++;
		idxInfo->ConstraintUsages[ltIdx].Omit = true;
	}
	if (gtIdx >= 0)
	{
		idxInfo->ConstraintUsages[gtIdx].ArgvIndex = argvIdx;
		idxInfo->ConstraintUsages[gtIdx].Omit = true;
	}
	if (idxInfo->OrderBys.length == 1 && !idxInfo->OrderBys[0].Desc)
		idxInfo->OrderByConsumed = true;
	idxInfo->EstimatedCost = (double)1;
	return RC_OK;
}

// A virtual table module that provides read-only access to a Tcl global variable namespace.
__constant__ static ITableModule _wholenumberModule =
{
	0,							// iVersion
	WholenumberConnect,
	WholenumberConnect,
	WholenumberBestIndex,
	WholenumberDisconnect, 
	WholenumberDisconnect,
	WholenumberOpen,			// xOpen - open a cursor
	WholenumberClose,			// xClose - close a cursor
	WholenumberFilter,			// xFilter - configure scan constraints
	WholenumberNext,			// xNext - advance a cursor
	WholenumberEof,				// xEof - check for end of scan
	WholenumberColumn,			// xColumn - read data
	WholenumberRowid,			// xRowid - read data
	nullptr,					// xUpdate
	nullptr,					// xBegin
	nullptr,					// xSync
	nullptr,					// xCommit
	nullptr,					// xRollback
	nullptr,					// xFindMethod
	nullptr,					// xRename
};

#endif

// Register the wholenumber virtual table
__device__ int wholenumber_register(Context *ctx)
{
	RC rc = RC_OK;
#ifndef OMIT_VIRTUALTABLE
	rc = VTable::CreateModule(ctx, "wholenumber", &_wholenumberModule, nullptr, nullptr);
#endif
	return rc;
}

#ifdef _TEST
#include <Jim.h>
// Decode a pointer to an sqlite3 object.
__device__ extern int GetDbPointer(Jim_Interp *interp, char *a, Context **ctx);

// Register the echo virtual table module.
__device__ static int register_wholenumber_module(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, (char *)args[1], &ctx)) return JIM_ERROR;
	wholenumber_register(ctx);
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_wholenumber_module", register_wholenumber_module, nullptr },
};
__device__ int Sqlitetestwholenumber_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
	return JIM_OK;
}

#endif
