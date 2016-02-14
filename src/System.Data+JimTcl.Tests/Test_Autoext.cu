// Test extension for testing the sqlite3_auto_extension() function.
#include <Jim.h>
#include <Core+Vdbe\Core+Ext.cu.h>

#ifndef OMIT_LOAD_EXTENSION
static EXTENSION_INIT1
	// The sqr() SQL function returns the square of its input value.
	__device__ static void sqrFunc(FuncContext *fctx, int argc, Mem **args)
{
	double r = Vdbe::Value_Double(args[0]);
	Vdbe::Result_Double(fctx, r*r);
}

// This is the entry point to register the extension for the sqr() function.
static int sqr_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	EXTENSION_INIT2(api);
	DataEx::CreateFunction(ctx, "sqr", 1, TEXTENCODE_ANY, 0, sqrFunc, 0, 0);
	return 0;
}

// The cube() SQL function returns the cube of its input value.
__device__ static void cubeFunc(FuncContext *fctx, int argc, Mem **args)
{
	double r = Vdbe::Value_Double(args[0]);
	Vdbe::Result_Double(fctx, r*r*r);
}

// This is the entry point to register the extension for the cube() function.
__device__ static int cube_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	EXTENSION_INIT2(api);
	DataEx::CreateFunction(ctx, "cube", 1, TEXTENCODE_ANY, 0, cubeFunc, 0, 0);
	return 0;
}

// This is a broken extension entry point
__device__ static int broken_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	EXTENSION_INIT2(api);
	char *err = _mprintf("broken autoext!");
	*errMsg = err;
	return 1;
}

// tclcmd:   sqlite3_auto_extension_sqr
//
// Register the "sqr" extension to be loaded automatically.
__device__ static int autoExtSqrObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	RC rc = DataEx::AutoExtension((void(*)())sqr_init);
	Jim_SetResultInt(interp, rc);
	return RC_OK;
}

// tclcmd:   sqlite3_auto_extension_cube
//
// Register the "cube" extension to be loaded automatically.
__device__ static int autoExtCubeObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	RC rc = DataEx::AutoExtension((void(*)())cube_init);
	Jim_SetResultInt(interp, rc);
	return RC_OK;
}

// tclcmd:   sqlite3_auto_extension_broken
//
// Register the broken extension to be loaded automatically.
__device__ static int autoExtBrokenObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	RC rc = DataEx::AutoExtension((void(*)())broken_init);
	Jim_SetResultInt(interp, rc);
	return RC_OK;
}

#endif

// tclcmd:   sqlite3_reset_auto_extension
//
// Reset all auto-extensions
__device__ static int resetAutoExtObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	DataEx::ResetAutoExtension();
	return RC_OK;
}

// This procedure registers the TCL procs defined in this file.
__device__ int Sqlitetest_autoext_Init(Jim_Interp *interp)
{
#ifndef OMIT_LOAD_EXTENSION
	Jim_CreateCommand(interp, "sqlite3_auto_extension_sqr", autoExtSqrObjCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_auto_extension_cube", autoExtCubeObjCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_auto_extension_broken", autoExtBrokenObjCmd, nullptr, nullptr);
#endif
	Jim_CreateCommand(interp, "sqlite3_reset_auto_extension", resetAutoExtObjCmd, nullptr, nullptr);
	return JIM_OK;
}
