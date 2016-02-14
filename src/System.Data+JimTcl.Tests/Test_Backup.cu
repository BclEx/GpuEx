// This file contains test logic for the sqlite3_backup() interface.
#include <JimEx.h>
#include <Core+Vdbe\Core+Vdbe.cu.h>

// These functions are implemented in test1.c.
__device__ extern int GetDbPointer(Jim_Interp *, const char *, Context **);
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

__device__ static int backupTestCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	Backup *p = (Backup *)clientData;
	int cmd;
	int rc2 = Jim_GetEnumFromStruct(interp, args[1], (const void **)_subs, sizeof(_subs[0]), &cmd, "option", 0);
	if (rc2 != JIM_OK)
		return rc2;
	if (argc != (2 + _subs[cmd].Argc))
	{
		Jim_WrongNumArgs(interp, 2, args, _subs[cmd].Args);
		return JIM_ERROR;
	}

	RC rc;
	switch (_subs[cmd].ECmd)
	{
	case BACKUP_FINISH: {
		const char *cmdName = Jim_String(args[0]);
		Jim_CmdInfo cmdInfo;
		Jim_GetCommandInfo(interp, args[0], &cmdInfo);
		cmdInfo.deleteProc = nullptr;
		Jim_SetCommandInfo(interp, args[0], &cmdInfo);
		Jim_DeleteCommand(interp, cmdName);
		rc = Backup::Finish(p);
		Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
		break; }
	case BACKUP_STEP: {
		int pages;
		if (Jim_GetInt(interp, args[2], &pages) != JIM_OK)
			return JIM_ERROR;
		rc = p->Step(pages);
		Jim_SetResultString(interp, (char *)sqlite3TestErrorName(rc), -1);
		break; }
	case BACKUP_REMAINING:
		Jim_SetResultInt(interp, p->Remaining);
		break;
	case BACKUP_PAGECOUNT:
		Jim_SetResultInt(interp, p->Pagecount);
		break;
	}
	return JIM_OK;
}

__device__ static void backupTestFinish(ClientData clientData, Jim_Interp *interp)
{
	Backup *p = (Backup *)clientData;
	Backup::Finish(p);
}

//     sqlite3_backup CMDNAME DESTHANDLE DESTNAME SRCHANDLE SRCNAME
__device__ static int backupTestInit(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 6)
	{
		Jim_WrongNumArgs(interp, 1, args, "CMDNAME DESTHANDLE DESTNAME SRCHANDLE SRCNAME");
		return JIM_ERROR;
	}

	const char *cmd = Jim_String(args[1]);
	Context *destCtx;
	GetDbPointer(interp, Jim_String(args[2]), &destCtx);
	const char *destName = Jim_String(args[3]);
	Context *srcCtx;
	GetDbPointer(interp, Jim_String(args[4]), &srcCtx);
	const char *srcName = Jim_String(args[5]);

	Backup *p = Backup::Init(destCtx, destName, srcCtx, srcName);
	if (!p)
	{
		Jim_AppendResult(interp, "sqlite3_backup_init() failed", nullptr);
		return JIM_ERROR;
	}

	Jim_CreateCommand(interp, (char *)cmd, backupTestCmd, (ClientData)p, backupTestFinish);
	Jim_SetResult(interp, args[1]);
	return JIM_OK;
}

__device__ int Sqlitetestbackup_Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "sqlite3_backup", backupTestInit, nullptr, nullptr);
	return JIM_OK;
}
