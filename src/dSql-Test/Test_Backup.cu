// This file contains test logic for the sqlite3_backup() interface.
#include <Tcl.h>
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <assert.h>

// These functions are implemented in test1.c.
__device__ extern int getDbPointer(Tcl_Interp *, const char *, Context **);
__device__ extern const char *sqlite3TestErrorName(int);

enum BackupSubCommandEnum
{
	BACKUP_STEP, BACKUP_FINISH, BACKUP_REMAINING, BACKUP_PAGECOUNT
};
__constant__ struct BackupSubCommand {
	const char *Cmd;
	enum BackupSubCommandEnum ECmd;
	int Argc;
	const char *Args;
} _subs[] = {
	{ "step",      BACKUP_STEP      , 1, "npage" },
	{ "finish",    BACKUP_FINISH    , 0, ""      },
	{ "remaining", BACKUP_REMAINING , 0, ""      },
	{ "pagecount", BACKUP_PAGECOUNT , 0, ""      },
	{ nullptr, (BackupSubCommandEnum)0, 0, nullptr }
};

__device__ static int backupTestCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	Backup *p = (Backup *)clientData;
	int cmd;
	int rc2 = Tcl_GetIndex(interp, args[1], (const void **)_subs, sizeof(_subs[0]), "option", 0, &cmd);
	if (rc2 != TCL_OK)
		return rc2;
	if (argc != (2 + _subs[cmd].Argc))
	{
		Tcl_WrongNumArgs(interp, 2, args, _subs[cmd].Args);
		return TCL_ERROR;
	}

	RC rc;
	switch (_subs[cmd].ECmd)
	{
	case BACKUP_FINISH: {
		const char *cmdName = args[0];
		Tcl_CmdInfo cmdInfo;
		Tcl_GetCommandInfo(interp, cmdName, &cmdInfo);
		cmdInfo.deleteProc = 0;
		Tcl_SetCommandInfo(interp, cmdName, &cmdInfo);
		Tcl_DeleteCommand(interp, (char *)cmdName);
		rc = Backup::Finish(p);
		Tcl_SetResult(interp, (char *)sqlite3TestErrorName(rc), TCL_STATIC);
		break; }
	case BACKUP_STEP: {
		int pages;
		if (Tcl_GetInt(interp, args[2], &pages) != TCL_OK)
			return TCL_ERROR;
		rc = p->Step(pages);
		Tcl_SetResult(interp, (char *)sqlite3TestErrorName(rc), TCL_STATIC);
		break; }
	case BACKUP_REMAINING:
		Tcl_SetObjResult(interp, (int)p->Remaining);
		break;
	case BACKUP_PAGECOUNT:
		Tcl_SetObjResult(interp, (int)p->Pagecount);
		break;
	}
	return TCL_OK;
}

__device__ static void backupTestFinish(ClientData clientData)
{
	Backup *p = (Backup *)clientData;
	Backup::Finish(p);
}

//     sqlite3_backup CMDNAME DESTHANDLE DESTNAME SRCHANDLE SRCNAME
__device__ static int backupTestInit(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 6)
	{
		Tcl_WrongNumArgs(interp, 1, args, "CMDNAME DESTHANDLE DESTNAME SRCHANDLE SRCNAME");
		return TCL_ERROR;
	}

	const char *cmd = args[1];
	Context *destCtx;
	getDbPointer(interp, args[2], &destCtx);
	const char *destName = args[3];
	Context *srcCtx;
	getDbPointer(interp, args[4], &srcCtx);
	const char *srcName = args[5];

	Backup *p = Backup::Init(destCtx, destName, srcCtx, srcName);
	if (!p)
	{
		Tcl_AppendResult(interp, "sqlite3_backup_init() failed", nullptr);
		return TCL_ERROR;
	}

	Tcl_CreateCommand(interp, (char *)cmd, backupTestCmd, (ClientData)p, backupTestFinish);
	Tcl_SetObjResult(interp, args[1]);
	return TCL_OK;
}

__device__ int Sqlitetestbackup_Init(Tcl_Interp *interp)
{
	Tcl_CreateCommand(interp, "sqlite3_backup", backupTestInit, nullptr, nullptr);
	return TCL_OK;
}
