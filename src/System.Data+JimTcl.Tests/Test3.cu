#include "Test.cu.h"
#include "..\System.Data.net\Core+Btree\BtreeInt.cu.h"

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
	case RC_LOCKED:     name = "SQLITE_LOCKED";      break;
	default:            name = "SQLITE_Unknown";     break;
	}
	return name;
}

// A bogus sqlite3 connection structure for use in the btree tests.
__device__ static Context _sCtx;
__device__ static int _refsSqlite3 = 0;

// Usage:   btree_open FILENAME NCACHE
//
// Open a new database
__device__ static int btree_open(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " FILENAME NCACHE FLAGS\"", nullptr);
		return JIM_ERROR;
	}
	int cacheSize;
	if (Jim_GetInt(interp, args[2], &cacheSize)) return JIM_ERROR;
	_refsSqlite3++;
	if (_refsSqlite3 == 1)
	{
		_sCtx.Vfs = VSystem::FindVfs(nullptr);
		_sCtx.Mutex = _mutex_alloc(MUTEX_RECURSIVE);
		_mutex_enter(_sCtx.Mutex);
	}
	int n = (int)_strlen(Jim_String(args[1]));
	char *filename = (char *)_alloc(n+2);
	if (!filename) return JIM_ERROR;
	_memcpy(filename, Jim_String(args[1]), n+1);
	filename[n+1] = 0;
	Btree *bt;
	RC rc = Btree::Open(_sCtx.Vfs, filename, &_sCtx, &bt, (Btree::OPEN)0, (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_MAIN_DB));
	_free(filename);
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	bt->SetCacheSize(cacheSize);
	char buf[100];
	__snprintf(buf, sizeof(buf), "%p", bt);
	Jim_AppendResult(interp, buf, nullptr);
	return JIM_OK;
}

// Usage:   btree_close ID
//
// Close the given database.
__device__ static int btree_close(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));
	RC rc = bt->Close();
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	_refsSqlite3--;
	if (_refsSqlite3 == 0)
	{
		_mutex_leave(_sCtx.Mutex);
		_mutex_free(_sCtx.Mutex);
		_sCtx.Mutex = 0;
		_sCtx.Vfs = 0;
	}
	return JIM_OK;
}

// Usage:   btree_begin_transaction ID
//
// Start a new transaction
__device__ static int btree_begin_transaction(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));
	bt->Enter();
	RC rc = bt->BeginTrans(1);
	bt->Leave();
	if (rc != RC_OK)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Usage:   btree_pager_stats ID
//
// Returns pager statistics
__constant__ static char *_stats_names[] = {
	"ref", "page", "max", "size", "state", "err",
	"hit", "miss", "ovfl", "read", "write"
};
__device__ static int btree_pager_stats(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));

	// Normally in this file, with a b-tree handle opened using the [btree_open] command it is safe to call sqlite3BtreeEnter() directly.
	// But this function is sometimes called with a btree handle obtained from an open SQLite connection (using [btree_from_db]). In this case
	// we need to obtain the mutex for the controlling SQLite handle before it is safe to call sqlite3BtreeEnter().
	_mutex_enter(bt->Ctx->Mutex);

	bt->Enter();
	int *a = bt->get_Pager()->Stats;
	for (int i = 0; i < 11; i++)
	{
		char buf[100];
		Jim_AppendElement(interp, _stats_names[i]);
		__snprintf(buf, sizeof(buf), "%d", a[i]);
		Jim_AppendElement(interp, buf);
	}
	bt->Leave();

	// Release the mutex on the SQLite handle that controls this b-tree
	_mutex_leave(bt->Ctx->Mutex);
	return JIM_OK;
}

// Usage:   btree_cursor ID TABLENUM WRITEABLE
//
// Create a new cursor.  Return the ID for the cursor.
__device__ static int btree_cursor(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 4)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID TABLENUM WRITEABLE\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int tableId;
	bool wrFlag;
	if (Jim_GetInt(interp, args[2], &tableId)) return JIM_ERROR;
	if (Jim_GetBoolean(interp, args[3], &wrFlag)) return JIM_ERROR;
	BtCursor *cur = (BtCursor *)Jim_Alloc(Btree::CursorSize());
	_memset(cur, 0, Btree::CursorSize());
	bt->Enter();
	RC rc = RC_OK;
#ifndef OMIT_SHARED_CACHE
	rc = bt->LockTable(tableId, wrFlag);
#endif
	if (rc == RC_OK)
		rc = bt->Cursor(tableId, wrFlag, 0, cur);
	bt->Leave();
	if (rc)
	{
		Jim_Free((char *)cur);
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	char buf[30];
	__snprintf(buf, sizeof(buf), "%p", cur);
	Jim_AppendResult(interp, buf, nullptr);
	return RC_OK;
}

// Usage:   btree_close_cursor ID
//
// Close a cursor opened using btree_cursor.
__device__ static int btree_close_cursor(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	BtCursor *cur = (BtCursor *)sqlite3TestTextToPtr(Jim_String(args[1]));
	Btree *bt = cur->Btree;
	bt->Enter();
	RC rc = Btree::CloseCursor(cur);
	bt->Leave();
	Jim_Free((char *)cur);
	if (rc)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	return RC_OK;
}

// Usage:   btree_next ID
//
// Move the cursor to the next entry in the table.  Return 0 on success or 1 if the cursor was already on the last entry in the table or if
// the table is empty.
__device__ static int btree_next(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	BtCursor *cur = (BtCursor *)sqlite3TestTextToPtr(Jim_String(args[1]));
	cur->Btree->Enter();
	int res = 0;
	RC rc = Btree::Next_(cur->Next, &res);
	cur->Btree->Leave();
	if (rc)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	char buf[100];
	__snprintf(buf, sizeof(buf), "%d", res);
	Jim_AppendResult(interp, buf, nullptr);
	return RC_OK;
}

// Usage:   btree_first ID
//
// Move the cursor to the first entry in the table.  Return 0 if the cursor was left point to something and 1 if the table is empty.
__device__ static int btree_first(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	BtCursor *cur = (BtCursor *)sqlite3TestTextToPtr(Jim_String(args[1]));
	cur->Btree->Enter();
	int res = 0;
	RC rc = Btree::First(cur, &res);
	cur->Btree->Leave();
	if (rc)
	{
		Jim_AppendResult(interp, errorName(rc), nullptr);
		return JIM_ERROR;
	}
	char buf[100];
	__snprintf(buf, sizeof(buf), "%d", res);
	Jim_AppendResult(interp, buf, nullptr);
	return RC_OK;
}

// Usage:   btree_eof ID
//
// Return TRUE if the given cursor is not pointing at a valid entry.
// Return FALSE if the cursor does point to a valid entry.
__device__ static int btree_eof(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	BtCursor *cur = (BtCursor *)sqlite3TestTextToPtr(Jim_String(args[1]));
	cur->Btree->Enter();
	bool rc = Btree::Eof(cur);
	cur->Btree->Leave();
	char buf[50];
	__snprintf(buf, sizeof(buf), "%d", rc);
	Jim_AppendResult(interp, buf, nullptr);
	return RC_OK;
}

// Usage:   btree_payload_size ID
//
// Return the number of bytes of payload
__device__ static int btree_payload_size(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	BtCursor *cur = (BtCursor *)sqlite3TestTextToPtr(Jim_String(args[1]));
	cur->Btree->Enter();

	// The cursor may be in "require-seek" state. If this is the case, the call to BtreeDataSize() will fix it.
	uint64 n1;
	int n2;
	Btree::DataSize(cur, (uint32 *)&n2);
	if (cur->Pages[cur->ID]->IntKey)
		n1 = 0;
	else
		Btree::KeySize(cur, (int64 *)&n1);
	cur->Btree->Leave();
	char buf[50];
	__snprintf(buf, sizeof(buf), "%d", (int)(n1+n2));
	Jim_AppendResult(interp, buf, nullptr);
	return RC_OK;
}

// usage:   varint_test  START  MULTIPLIER  COUNT  INCREMENT
//
// This command tests the putVarint() and getVarint() routines, both for accuracy and for speed.
//
// An integer is written using putVarint() and read back with getVarint() and varified to be unchanged.  This repeats COUNT
// times.  The first integer is START*MULTIPLIER.  Each iteration increases the integer by INCREMENT.
//
// This command returns nothing if it works.  It returns an error message if something goes wrong.
__device__ static int btree_varint_test(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	int n1, n2, i, j;
	if (argc != 5)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " START MULTIPLIER COUNT INCREMENT\"", nullptr);
		return JIM_ERROR;
	}
	uint32 start, mult, count, incr;
	if (Jim_GetInt(interp, args[1], (int *)&start)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], (int *)&mult)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[3], (int *)&count)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[4], (int *)&incr)) return JIM_ERROR;
	uint64 in = start;
	in *= mult;
	uint64 out;
	unsigned char buf[100];
	for (int i = 0; i < (int)count; i++)
	{
		char err[200];
		n1 = _convert_putvarint(buf, in);
		if (n1 > 9 || n1 < 1)
		{
			_sprintf(err, "putVarint returned %d - should be between 1 and 9", n1);
			Jim_AppendResult(interp, err, nullptr);
			return JIM_ERROR;
		}
		n2 = _convert_getvarint(buf, &out);
		if (n1 != n2)
		{
			_sprintf(err, "putVarint returned %d and getVarint returned %d", n1, n2);
			Jim_AppendResult(interp, err, nullptr);
			return JIM_ERROR;
		}
		if (in != out)
		{
			_sprintf(err, "Wrote 0x%016llx and got back 0x%016llx", in, out);
			Jim_AppendResult(interp, err, nullptr);
			return JIM_ERROR;
		}
		if ((in & 0xffffffff) == in)
		{
			uint32 out32;
			n2 = _convert_getvarint32(buf, out32);
			out = out32;
			if (n1 != n2)
			{
				_sprintf(err, "putVarint returned %d and GetVarint32 returned %d", n1, n2);
				Jim_AppendResult(interp, err, nullptr);
				return JIM_ERROR;
			}
			if (in != out)
			{
				_sprintf(err, "Wrote 0x%016llx and got back 0x%016llx from GetVarint32", in, out);
				Jim_AppendResult(interp, err, nullptr);
				return JIM_ERROR;
			}
		}

		// In order to get realistic timings, run getVarint 19 more times. This is because getVarint is called about 20 times more often than putVarint.
		for (int j = 0; j < 19; j++)
			_convert_getvarint(buf, &out);
		in += incr;
	}
	return JIM_OK;
}

// usage:   btree_from_db  DB-HANDLE
//
// This command returns the btree handle for the main database associated
// with the database-handle passed as the argument. Example usage:
//
// sqlite3 db test.db
// set bt [btree_from_db db]
__device__ static int btree_from_db(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2 && argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " DB-HANDLE ?N?\"", nullptr);
		return JIM_ERROR;
	}
	Jim_CmdInfo info;
	if (Jim_GetCommandInfo(interp, Jim_String(args[1]), &info) != 1)
	{
		Jim_AppendResult(interp, "No such db-handle: \"", args[1], "\"", nullptr);
		return JIM_ERROR;
	}
	int db = (argc == 3 ? _atoi(Jim_String(args[2])) : 0);

	Context *ctx = *((Context **)info.objClientData);
	_assert(ctx);

	Btree *bt = ctx->DBs[db].Bt;
	char buf[100];
	__snprintf(buf, sizeof(buf), "%p", bt);
	Jim_SetResultString(interp, buf, -1);
	return JIM_OK;
}

// Usage:   btree_ismemdb ID
//
// Return true if the B-Tree is in-memory.
__device__ static int btree_ismemdb(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " ID\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));
	_mutex_enter(bt->Ctx->Mutex);
	bt->Enter();
	bool res = bt->get_Pager()->MemoryDB;
	bt->Leave();
	_mutex_leave(bt->Ctx->Mutex);
	Jim_SetResultBool(interp, res);
	return RC_OK;
}

// usage:   btree_set_cache_size ID NCACHE
//
// Set the size of the cache used by btree $ID.
__device__ static int btree_set_cache_size(ClientData notUsed, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " BT NCACHE\"", nullptr);
		return JIM_ERROR;
	}
	Btree *bt = (Btree *)sqlite3TestTextToPtr(Jim_String(args[1]));
	int cacheSize;
	if (Jim_GetInt(interp, args[2], &cacheSize)) return JIM_ERROR;

	_mutex_enter(bt->Ctx->Mutex);
	bt->Enter();
	bt->SetCacheSize(cacheSize);
	bt->Leave();
	_mutex_leave(bt->Ctx->Mutex);
	return JIM_OK;
}      

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "btree_open",               btree_open               },
	{ "btree_close",              btree_close              },
	{ "btree_begin_transaction",  btree_begin_transaction  },
	{ "btree_pager_stats",        btree_pager_stats        },
	{ "btree_cursor",             btree_cursor             },
	{ "btree_close_cursor",       btree_close_cursor       },
	{ "btree_next",               btree_next               },
	{ "btree_eof",                btree_eof                },
	{ "btree_payload_size",       btree_payload_size       },
	{ "btree_first",              btree_first              },
	{ "btree_varint_test",        btree_varint_test        },
	{ "btree_from_db",            btree_from_db            },
	{ "btree_ismemdb",            btree_ismemdb            },
	{ "btree_set_cache_size",     btree_set_cache_size     }
};
__device__ int Sqlitetest3_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	return JIM_OK;
}
