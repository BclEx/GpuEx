// Test extension for testing the sqlite3_auto_extension() function.
#include <Tcl.h>
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
	Main::CreateFunction(ctx, "sqr", 1, TEXTENCODE_ANY, 0, sqrFunc, 0, 0);
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
	Main::CreateFunction(ctx, "cube", 1, TEXTENCODE_ANY, 0, cubeFunc, 0, 0);
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
__device__ static int autoExtSqrObjCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	RC rc = Main::AutoExtension((void(*)())sqr_init);
	Tcl_SetObjResult(interp, (int)rc);
	return RC_OK;
}

// tclcmd:   sqlite3_auto_extension_cube
//
// Register the "cube" extension to be loaded automatically.
__device__ static int autoExtCubeObjCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	RC rc = Main::AutoExtension((void(*)())cube_init);
	Tcl_SetObjResult(interp, (int)rc);
	return RC_OK;
}

// tclcmd:   sqlite3_auto_extension_broken
//
// Register the broken extension to be loaded automatically.
__device__ static int autoExtBrokenObjCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	RC rc = Main::AutoExtension((void(*)())broken_init);
	Tcl_SetObjResult(interp, (int)rc);
	return RC_OK;
}

#endif

// tclcmd:   sqlite3_reset_auto_extension
//
// Reset all auto-extensions
__device__ static int resetAutoExtObjCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	Main::ResetAutoExtension();
	return RC_OK;
}

// This procedure registers the TCL procs defined in this file.
__device__ int Sqlitetest_autoext_Init(Tcl_Interp *interp)
{
#ifndef OMIT_LOAD_EXTENSION
	Tcl_CreateCommand(interp, "sqlite3_auto_extension_sqr", autoExtSqrObjCmd, nullptr, nullptr);
	Tcl_CreateCommand(interp, "sqlite3_auto_extension_cube", autoExtCubeObjCmd, nullptr, nullptr);
	Tcl_CreateCommand(interp, "sqlite3_auto_extension_broken", autoExtBrokenObjCmd, nullptr, nullptr);
#endif
	Tcl_CreateCommand(interp, "sqlite3_reset_auto_extension", resetAutoExtObjCmd, nullptr, nullptr);
	return TCL_OK;
}
