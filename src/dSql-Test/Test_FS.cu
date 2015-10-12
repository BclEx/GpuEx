// Code for testing the virtual table interfaces.  This code is not included in the SQLite library.  It is used for automated
// testing of the SQLite library.
//
// The FS virtual table is created as follows:
//
//   CREATE VIRTUAL TABLE tbl USING fs(idx);
//
// where idx is the name of a table in the db with 2 columns.  The virtual table also has two columns - file path and file contents.
//
// The first column of table idx must be an IPK, and the second contains file paths. For example:
//
//   CREATE TABLE idx(id INTEGER PRIMARY KEY, path TEXT);
//   INSERT INTO idx VALUES(4, '/etc/passwd');
//
// Adding the row to the idx table automatically creates a row in the virtual table with rowid=4, path=/etc/passwd and a text field that 
// contains data read from file /etc/passwd on disk.
#include <Core+Vdbe\VdbeInt.cu.h>
#include <Tcl.h>

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#if OS_UNIX
#include <unistd.h>
#endif
#if OS_WIN
#include <io.h>
#endif

#ifndef OMIT_VIRTUALTABLE

// A fs virtual-table object 
struct fs_vtab
{
	IVTable base;
	Context *Ctx;
	char *Db;                   // Name of db containing zTbl
	char *Tbl;						// Name of docid->file map table
};

// A fs cursor object
struct fs_cursor
{
	IVTableCursor base;
	Vdbe *Stmt;
	char *Buf;
	int BufLength;
	int Alloc;
};

// This function is the implementation of both the xConnect and xCreate methods of the fs virtual table.
//
// The args[] array contains the following:
//
//   args[0]   -> module name  ("fs")
//   args[1]   -> database name
//   args[2]   -> table name
//   args[...] -> other module argument fields.
// Note that for this virtual table, the xCreate and xConnect methods are identical.
__device__ static RC fsConnect(Context *ctx, void *aux, int argc, const char *const *args, IVTable **vtabs, char **err)
{
	if (argc != 4)
	{
		*err = _mprintf("wrong number of arguments");
		return RC_ERROR;
	}
	const char *db = args[1];
	const char *tbl = args[3];

	int bytes = sizeof(fs_vtab) + (int)_strlen(tbl) + 1 + (int)_strlen(db) + 1;
	fs_vtab *vtab = (fs_vtab *)_allocZero(bytes);
	if (!vtab) return RC_NOMEM;

	vtab->Tbl = (char *)&vtab[1];
	vtab->Db = &vtab->Tbl[_strlen(tbl)+1];
	vtab->Ctx = ctx;
	_memcpy(vtab->Tbl, tbl, _strlen(tbl));
	_memcpy(vtab->Db, db, _strlen(db));
	*vtabs = &vtab->base;
	VTable::DeclareVTable(ctx, "CREATE TABLE xyz(path TEXT, data TEXT)");

	return RC_OK;
}

// The xDisconnect and xDestroy methods are also the same
__device__ static RC fsDisconnect(IVTable *vtab)
{
	_free(vtab);
	return RC_OK;
}

// Open a new fs cursor.
__device__ static RC fsOpen(IVTable *vTab, IVTableCursor **cursors)
{
	fs_cursor *cur = (fs_cursor *)_allocZero(sizeof(fs_cursor));
	*cursors = &cur->base;
	return RC_OK;
}

// Close a fs cursor.
__device__ static RC fsClose(IVTableCursor *cur2)
{
	fs_cursor *cur = (fs_cursor *)cur2;
	Vdbe::Finalize(cur->Stmt);
	_free(cur->Buf);
	_free(cur);
	return RC_OK;
}

__device__ static RC fsNext(IVTableCursor *cur2)
{
	fs_cursor *cur = (fs_cursor *)cur2;
	RC rc = cur->Stmt->Step();
	if (rc == RC_ROW || rc == RC_DONE) rc = RC_OK;
	return rc;
}

__device__ static RC fsFilter(IVTableCursor *cur2, int idxNum, const char *idxStr, int argc, Mem **args)
{
	fs_cursor *cur = (fs_cursor *)cur2;
	fs_vtab *p = (fs_vtab *)(cur2->IVTable);

	RC rc;
	_assert((idxNum == 0 && argc == 0) || (idxNum == 1 && argc == 1));
	if (idxNum == 1)
	{
		char *stmt = _mprintf("SELECT * FROM %Q.%Q WHERE rowid=?", p->Db, p->Tbl);
		if (!stmt) return RC_NOMEM;
		rc = Prepare::Prepare_v2(p->Ctx, stmt, -1, &cur->Stmt, 0);
		_free(stmt);
		if (rc == RC_OK)
			Vdbe::Bind_Value(cur->Stmt, 1, args[0]);
	}
	else
	{
		char *stmt = _mprintf("SELECT * FROM %Q.%Q", p->Db, p->Tbl);
		if (!stmt) return RC_NOMEM;
		rc = Prepare::Prepare_v2(p->Ctx, stmt, -1, &cur->Stmt, 0);
		_free(stmt);
	}

	if (rc == RC_OK)
		rc = fsNext(cur2); 
	return rc;
}

__device__ static RC fsColumn(IVTableCursor *cur2, FuncContext *fctx, int i)
{
	fs_cursor *cur = (fs_cursor *)cur2;

	_assert(i == 0 || i == 1);
	if (i == 0)
		Vdbe::Result_Value(fctx, Vdbe::Column_Value(cur->Stmt, 0));
	else
	{
		const char *file = (const char *)Vdbe::Column_Text(cur->Stmt, 1);
		int fd = open(file, O_RDONLY);
		if (fd < 0) return RC_IOERR;
		struct stat sbuf;
		_fstat(fd, &sbuf);

		if (sbuf.st_size >= cur->Alloc)
		{
			int newLength = sbuf.st_size*2;
			char *new_;
			if (newLength < 1024) newLength = 1024;

			char *new_ = (char *)_realloc(cur->Buf, newLength);
			if (!new_)
			{
				_close(fd);
				return RC_NOMEM;
			}
			cur->Buf = new_;
			cur->Alloc = newLength;
		}

		_read(fd, cur->Buf, sbuf.st_size);
		_close(fd);
		cur->BufLength = sbuf.st_size;
		cur->Buf[cur->BufLength] = '\0';

		Vdbe::Result_Text(fctx, cur->Buf, -1, DESTRUCTOR_TRANSIENT);
	}
	return RC_OK;
}

__device__ static RC fsRowid(IVTableCursor *cur2, int64 *rowid)
{
	fs_cursor *cur = (fs_cursor *)cur2;
	*rowid = Vdbe::Column_Int64(cur->Stmt, 0);
	return RC_OK;
}

__device__ static bool fsEof(IVTableCursor *cur2)
{
	fs_cursor *cur = (fs_cursor *)cur;
	return (Vdbe::Data_Count(cur->Stmt) == 0);
}

__device__ static RC fsBestIndex(IVTable *vtab, IIndexInfo *idxInfo)
{
	for (int ii = 0; ii < idxInfo->Constraints.length; ii++)
	{
		IIndexInfo::Constraint const *cons = &idxInfo->Constraints[ii];
		if (cons->Column < 0 && cons->Usable && cons->OP == INDEX_CONSTRAINT_EQ)
		{
			IIndexInfo::ConstraintUsage *usage = &idxInfo->ConstraintUsages[ii];
			usage->Omit = 0;
			usage->ArgvIndex = 1;
			idxInfo->IdxNum = 1;
			idxInfo->EstimatedCost = 1.0;
			break;
		}
	}
	return RC_OK;
}

// A virtual table module that provides read-only access to a Tcl global variable namespace.
__constant__ static ITableModule _fsModule = {
	0,							// iVersion
	fsConnect,
	fsConnect,
	fsBestIndex,
	fsDisconnect, 
	fsDisconnect,
	fsOpen,						// xOpen - open a cursor
	fsClose,					// xClose - close a cursor
	fsFilter,					// xFilter - configure scan constraints
	fsNext,						// xNext - advance a cursor
	fsEof,						// xEof - check for end of scan
	fsColumn,					// xColumn - read data
	fsRowid,					// xRowid - read data
	nullptr,					// xUpdate
	nullptr,					// xBegin
	nullptr,					// xSync
	nullptr,					// xCommit
	nullptr,					// xRollback
	nullptr,                    // xFindMethod
	nullptr,					// xRename
};

// Decode a pointer to an sqlite3 object.
__device__ extern int getDbPointer(Tcl_Interp *interp, const char *a, Context **ctx);

// Register the echo virtual table module.
__device__ static int register_fs_module(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2)
	{
		Tcl_WrongNumArgs(interp, 1, args, "DB");
		return TCL_ERROR;
	}
	Context *ctx;
	if( getDbPointer(interp, args[1], &ctx)) return TCL_ERROR;
#ifndef OMIT_VIRTUALTABLE
	VTable::CreateModule(ctx, "fs", &_fsModule, (void *)interp);
#endif
	return TCL_OK;
}

#endif

// Register commands with the TCL interpreter.
__constant__ static struct
{
	char *Name;
	Tcl_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_fs_module", register_fs_module, nullptr },
};

__device__ int Sqlitetestfs_Init(Tcl_Interp *interp)
{
#ifndef OMIT_VIRTUALTABLE
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Tcl_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
#endif
	return TCL_OK;
}
