// This file contains obscure tests of the C-interface required for completeness. Test code is written in C for these cases
// as there is not much point in binding to Tcl.
#include "Test.cu.h"

// c_collation_test
__device__ static int c_collation_test(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	const char *errFunction = "N/A";

	// Open a database.
	Context *ctx;
	RC rc = DataEx::Open(":memory:", &ctx);
	if (rc != RC_OK)
	{
		errFunction = "sqlite3_open";
		goto error_out;
	}

	rc = DataEx::CreateCollation(ctx, "collate", (TEXTENCODE)456, nullptr, nullptr);
	if (rc != RC_MISUSE)
	{
		DataEx::Close(ctx);
		errFunction = "sqlite3_create_collation";
		goto error_out;
	}

	DataEx::Close(ctx);
	return JIM_OK;

error_out:
	Jim_ResetResult(interp);
	Jim_AppendResult(interp, "Error testing function: ", errFunction, nullptr);
	return JIM_ERROR;
}

// c_realloc_test
__device__ static int c_realloc_test(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	const char *errFunction = "N/A";

	void *p = _alloc(5);
	if (!p)
	{
		errFunction = "sqlite3_malloc";
		goto error_out;
	}

	// Test that realloc()ing a block of memory to a negative size is the same as free()ing that memory.
	p = _realloc(p, -1);
	if (p)
	{
		errFunction = "sqlite3_realloc";
		goto error_out;
	}
	return JIM_OK;

error_out:
	Jim_ResetResult(interp);
	Jim_AppendResult(interp, "Error testing function: ", errFunction, nullptr);
	return JIM_ERROR;
}

// c_misuse_test
__device__ static int c_misuse_test(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, nullptr);
		return JIM_ERROR;
	}
	const char *errFunction = "N/A";

	// Open a database. Then close it again. We need to do this so that we have a "closed database handle" to pass to various API functions.
	Context *ctx = nullptr;
	RC rc = DataEx::Open(":memory:", &ctx);
	if (rc != RC_OK)
	{
		errFunction = "sqlite3_open";
		goto error_out;
	}
	DataEx::Close(ctx);

	rc = DataEx::ErrCode(ctx);
	if (rc != RC_MISUSE)
	{
		errFunction = "sqlite3_errcode";
		goto error_out;
	}

	Vdbe *stmt = (Vdbe *)1234;
	rc = Prepare::Prepare_(ctx, nullptr, 0, &stmt, nullptr);
	if (rc != RC_MISUSE)
	{
		errFunction = "sqlite3_prepare";
		goto error_out;
	}
	_assert(!stmt); // Verify that pStmt is zeroed even on a MISUSE error

	stmt = (Vdbe *)1234;
	rc = Prepare::Prepare_v2(ctx, nullptr, 0, &stmt, nullptr);
	if (rc != RC_MISUSE)
	{
		errFunction = "sqlite3_prepare_v2";
		goto error_out;
	}
	_assert(!stmt);

#ifndef OMIT_UTF16
	stmt = (Vdbe *)1234;
	rc = Prepare::Prepare16(ctx, nullptr, 0, &stmt, nullptr);
	if (rc != RC_MISUSE)
	{
		errFunction = "sqlite3_prepare16";
		goto error_out;
	}
	_assert(!stmt);
	stmt = (Vdbe *)1234;
	rc = Prepare::Prepare16_v2(ctx, nullptr, 0, &stmt, nullptr);
	if (rc != RC_MISUSE)
	{
		errFunction = "sqlite3_prepare16_v2";
		goto error_out;
	}
	_assert(!stmt);
#endif
	return JIM_OK;

error_out:
	Jim_ResetResult(interp);
	Jim_AppendResult(interp, "Error testing function: ", errFunction, nullptr);
	return JIM_ERROR;
}

// Register commands with the TCL interpreter.
__constant__ static struct
{
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "c_misuse_test",    c_misuse_test, nullptr },
	{ "c_realloc_test",   c_realloc_test, nullptr },
	{ "c_collation_test", c_collation_test, nullptr },
};
__device__ int Sqlitetest9_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
	return JIM_OK;
}
