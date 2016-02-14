#include "Test.cu.h"

// Interpret an SQLite error number
__device__ static char *errorName(int rc)
{
	char *name;
	switch (rc)
	{
	case RC_OK:         name = "SQLITE_OK";          break;
	case RC_ERROR:      name = "SQLITE_ERROR";       break;
	case RC_PERM:       name = "SQLITE_PERM";        break;
	case RC_ABORT:      name = "SQLITE_ABORT";       break;
	case RC_BUSY:       name = "SQLITE_BUSY";        break;
	case RC_NOMEM:      name = "SQLITE_NOMEM";       break;
	case RC_READONLY:   name = "SQLITE_READONLY";    break;
	case RC_INTERRUPT:  name = "SQLITE_INTERRUPT";   break;
	case RC_IOERR:      name = "SQLITE_IOERR";       break;
	case RC_CORRUPT:    name = "SQLITE_CORRUPT";     break;
	case RC_FULL:       name = "SQLITE_FULL";        break;
	case RC_CANTOPEN:   name = "SQLITE_CANTOPEN";    break;
	case RC_PROTOCOL:   name = "SQLITE_PROTOCOL";    break;
	case RC_EMPTY:      name = "SQLITE_EMPTY";       break;
	case RC_SCHEMA:     name = "SQLITE_SCHEMA";      break;
	case RC_CONSTRAINT: name = "SQLITE_CONSTRAINT";  break;
	case RC_MISMATCH:   name = "SQLITE_MISMATCH";    break;
	case RC_MISUSE:     name = "SQLITE_MISUSE";      break;
	case RC_NOLFS:      name = "SQLITE_NOLFS";       break;
	default:			name = "SQLITE_Unknown";     break;
	}
	return name;
}

// Page size and reserved size used for testing.
__constant__ static int test_pagesize = 1024;

// Dummy page reinitializer
__device__ static void pager_test_reiniter(IPage *notUsed)
{
	return;
}

// Usage:   pager_open FILENAME N-PAGE
//
// Open a new pager
__device__ static int pager_open(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " FILENAME N-PAGE\"", nullptr);
		return JIM_ERROR;
	}
	int pages;
	if (Jim_GetInt(interp, args[2], &pages)) return JIM_ERROR;
	Pager *pager;
	RC rc = Pager::Open(VSystem::FindVfs(nullptr), &pager, Jim_String(args[1]), 0, (IPager::PAGEROPEN)0, (VSystem::OPEN)(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_MAIN_DB), pager_test_reiniter);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	pager->SetCacheSize(pages);
	uint32 pageSize = test_pagesize;
	pager->SetPageSize(&pageSize, -1);
	char buf[100];
	__snprintf(buf, sizeof(buf), "%p", pager);
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   pager_close ID
//
// Close the given pager.
__device__ static int pager_close(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->Close();
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_rollback ID
//
// Rollback changes
__device__ static int pager_rollback(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->Rollback();
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_commit ID
//
// Commit all changes
__device__ static int pager_commit(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->CommitPhaseOne(nullptr, false);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	rc = pager->CommitPhaseTwo();
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_stmt_begin ID
//
// Start a new checkpoint.
__device__ static int pager_stmt_begin(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->OpenSavepoint(1);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_stmt_rollback ID
//
// Rollback changes to a checkpoint
__device__ static int pager_stmt_rollback(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->Savepoint(IPager::SAVEPOINT_ROLLBACK, 0);
	pager->Savepoint(IPager::SAVEPOINT_RELEASE, 0);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_stmt_commit ID
//
// Commit changes to a checkpoint
__device__ static int pager_stmt_commit(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = pager->Savepoint(IPager::SAVEPOINT_RELEASE, 0);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   pager_stats ID
//
// Return pager statistics.
__constant__ static char *_stats_names[] = {
	"ref", "page", "max", "size", "state", "err",
	"hit", "miss", "ovfl",
};
__device__ static int pager_stats(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int *a = pager->Stats;
	for (int i = 0; i < 9; i++)
	{
		char buf[100];
		Jim_AppendElement(interp, _stats_names[i]);
		__snprintf(buf, sizeof(buf), "%d", a[i]);
		Jim_AppendElement(interp, buf);
	}
	return JIM_OK;
}

// Usage:   pager_pagecount ID
//
// Return the size of the database file.
__device__ static int pager_pagecount(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	Pid pages;
	pager->Pages(&pages);
	char buf[100];
	__snprintf(buf, sizeof(buf), "%d", pages);
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   page_get ID PGNO
//
// Return a pointer to a page from the database.
__device__ static int page_get(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID PGNO\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int pgid;
	if (Jim_GetInt(interp, args[2], &pgid)) return JIM_ERROR;
	RC rc = pager->SharedLock();
	IPage *page;
	if (rc == RC_OK)
		rc = pager->Acquire(pgid, &page, false);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	char buf[100];
	__snprintf(buf, sizeof(buf), "%p", page);
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   page_lookup ID PGNO
//
// Return a pointer to a page if the page is already in cache. If not in cache, return an empty string.
__device__ static int page_lookup(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID PGNO\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int pgid;
	if (Jim_GetInt(interp, args[2], &pgid)) return JIM_ERROR;
	IPage *page = pager->Lookup(pgid);
	if (page)
	{
		char buf[100];
		__snprintf(buf, sizeof(buf), "%p", page);
		Jim_AppendResult(interp, buf, nullptr);
	}
	return JIM_OK;
}

// Usage:   pager_truncate ID PGNO
__device__ static int pager_truncate(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID PGNO\"", nullptr);
		return JIM_ERROR;
	}
	Pager *pager = (Pager *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int pgid;
	if (Jim_GetInt(interp, args[2], &pgid)) return JIM_ERROR;
	pager->TruncateImage(pgid);
	return JIM_OK;
}


// Usage:   page_unref PAGE
//
// Drop a pointer to a page.
__device__ static int page_unref(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " PAGE\"", nullptr);
		return JIM_ERROR;
	}
	IPage *page = (IPage *)sqlite3TestTextToPtr(Jim_String(args[1]));
	Pager::Unref(page);
	return JIM_OK;
}

// Usage:   page_read PAGE
//
// Return the content of a page
__device__ static int page_read(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " PAGE\"", nullptr);
		return JIM_ERROR;
	}
	IPage *page = (IPage *)sqlite3TestTextToPtr(Jim_String(args[1]));
	char buf[100];
	_memcpy(buf, Pager::GetData(page), sizeof(buf));
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   page_number PAGE
//
// Return the page number for a page.
__device__ static int page_number(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " PAGE\"", nullptr);
		return JIM_ERROR;
	}
	IPage *page = (IPage *)sqlite3TestTextToPtr(Jim_String(args[1]));
	char buf[100];
	__snprintf(buf, sizeof(buf), "%d", Pager::get_PageID(page));
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   page_write PAGE DATA
//
// Write something into a page.
__device__ static int page_write(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " PAGE DATA\"", nullptr);
		return JIM_ERROR;
	}
	IPage *page = (IPage *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = Pager::Write(page);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	char *data = (char *)Pager::GetData(page);
	_strncpy(data, Jim_String(args[2]), test_pagesize-1);
	data[test_pagesize-1] = 0;
	return JIM_OK;
}

#ifndef OMIT_DISKIO
// Usage:   fake_big_file  N  FILENAME
//
// Write a few bytes at the N megabyte point of FILENAME.  This will create a large file.  If the file was a valid SQLite database, then
// the next time the database is opened, SQLite will begin allocating new pages after N.  If N is 2096 or bigger, this will test the
// ability of SQLite to write to large files.
__device__ static int fake_big_file(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " N-MEGABYTES FILE\"", nullptr);
		return JIM_ERROR;
	}
	int n;
	if (Jim_GetInt(interp, args[1], &n)) return JIM_ERROR;
	VSystem *vfs = VSystem::FindVfs(nullptr);
	int fileLength = (int)_strlen(Jim_String(args[2]));
	char *file = (char *)_alloc(fileLength+2);
	if (!file) return JIM_ERROR;
	_memcpy(file, Jim_String(args[2]), fileLength+1);
	file[fileLength+1] = 0;
	VFile *fd = nullptr;
	RC rc = vfs->OpenAndAlloc(file, &fd, (VSystem::OPEN)(VSystem::OPEN_CREATE|VSystem::OPEN_READWRITE|VSystem::OPEN_MAIN_DB), nullptr);
	if (rc)
	{
		Jim_AppendResult(interp, "open failed: ", errorName(rc), nullptr);
		_free(file);
		return JIM_ERROR;
	}
	int64 offset = n;
	offset *= 1024*1024;
	rc = fd->Write("Hello, World!", 14, offset);
	fd->CloseAndFree();
	_free(file);
	if (rc)
	{
		Jim_AppendResult(interp, "write failed: ", errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}
#endif

// test_control_pending_byte  PENDING_BYTE
//
// Set the PENDING_BYTE using the sqlite3_test_control() interface.
__device__ static int testPendingByte(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " PENDING-BYTE\"", nullptr);
		return JIM_ERROR;
	}
	int byte;
	if (Jim_GetInt(interp, args[1], &byte)) return JIM_ERROR;
	RC rc = DataEx::TestControl(DataEx::TESTCTRL_PENDING_BYTE, byte);
	Jim_SetResultInt(interp, rc);
	return JIM_OK;
}  

// sqlite3BitvecBuiltinTest SIZE PROGRAM
//
// Invoke the SQLITE_TESTCTRL_BITVEC_TEST operator on test_control. See comments on sqlite3BitvecBuiltinTest() for additional information.
__device__ static int testBitvecBuiltinTest(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " SIZE PROGRAM\"", nullptr);
	}
	int sz;
	if (Jim_GetInt(interp, args[1], &sz)) return JIM_ERROR;
	const char *z = Jim_String(args[2]);
	int progLength = 0;
	int prog[100];
	while (progLength < 99 && *z)
	{
		while (*z && !_isdigit(*z)) { z++; }
		if (!*z) break;
		prog[progLength++] = _atoi(z);
		while (_isdigit(*z)) { z++; }
	}
	prog[progLength] = 0;
	RC rc = DataEx::TestControl(DataEx::TESTCTRL_BITVEC_TEST, sz, prog);
	Jim_SetResultInt(interp, rc);
	return JIM_OK;
}  

// Register commands with the TCL interpreter.
extern int sqlite3_io_error_persist;
extern int sqlite3_io_error_pending;
extern int sqlite3_io_error_hit;
extern int sqlite3_io_error_hardhit;
extern int sqlite3_diskfull_pending;
extern int sqlite3_diskfull;
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "pager_open",              pager_open          },
	{ "pager_close",             pager_close         },
	{ "pager_commit",            pager_commit        },
	{ "pager_rollback",          pager_rollback      },
	{ "pager_stmt_begin",        pager_stmt_begin    },
	{ "pager_stmt_commit",       pager_stmt_commit   },
	{ "pager_stmt_rollback",     pager_stmt_rollback },
	{ "pager_stats",             pager_stats         },
	{ "pager_pagecount",         pager_pagecount     },
	{ "page_get",                page_get            },
	{ "page_lookup",             page_lookup         },
	{ "page_unref",              page_unref          },
	{ "page_read",               page_read           },
	{ "page_write",              page_write          },
	{ "page_number",             page_number         },
	{ "pager_truncate",          pager_truncate      },
#ifndef OMIT_DISKIO
	{ "fake_big_file",           fake_big_file       },
#endif
	{ "sqlite3BitvecBuiltinTest",testBitvecBuiltinTest     },
	{ "sqlite3_test_control_pending_byte", testPendingByte },
};
__device__ int Sqlitetest2_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	Jim_LinkVar(interp, "sqlite_io_error_pending", (char *)&sqlite3_io_error_pending, JIM_LINK_INT);
	Jim_LinkVar(interp, "sqlite_io_error_persist", (char *)&sqlite3_io_error_persist, JIM_LINK_INT);
	Jim_LinkVar(interp, "sqlite_io_error_hit", (char *)&sqlite3_io_error_hit, JIM_LINK_INT);
	Jim_LinkVar(interp, "sqlite_io_error_hardhit", (char *)&sqlite3_io_error_hardhit, JIM_LINK_INT);
	Jim_LinkVar(interp, "sqlite_diskfull_pending", (char *)&sqlite3_diskfull_pending, JIM_LINK_INT);
	Jim_LinkVar(interp, "sqlite_diskfull", (char *)&sqlite3_diskfull, JIM_LINK_INT);
#ifndef OMIT_WSD
	Jim_LinkVar(interp, "sqlite_pending_byte", (char *)&_Core_PendingByte, JIM_LINK_INT|JIM_LINK_READ_ONLY);
#endif
	return JIM_OK;
}
