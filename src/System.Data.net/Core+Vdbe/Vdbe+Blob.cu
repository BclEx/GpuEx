// vdbeblob.cu
#include "VdbeInt.cu.h"

namespace Core {

#pragma region Blob
#ifndef OMIT_INCRBLOB

	struct Incrblob
	{
		int Flags;              // Copy of "flags" passed to sqlite3_blob_open()
		int Bytes;              // Size of open blob, in bytes
		uint32 Offset;			// Byte offset of blob in cursor data
		int Col;				// Table column this handle is open on
		BtCursor *Cursor;		// Cursor pointing at blob row
		Vdbe *Stmt;				// Statement holding cursor open
		Context *Ctx;           // The associated database
	};

	__device__ static RC BlobSeekToRow(Incrblob *p, int64 row, char **errOut)
	{
		char *err = nullptr; // Error message
		Vdbe *v = (Vdbe *)p->Stmt;

		// Set the value of the SQL statements only variable to integer iRow. This is done directly instead of using sqlite3_bind_int64() to avoid 
		// triggering asserts related to mutexes.
		_assert(v->Vars[0].Flags & MEM_Int);
		v->Vars[0].u.I = row;

		RC rc = v->Step();
		if (rc == RC_ROW)
		{
			uint32 type = v->Cursors[0]->Types[p->Col];
			if (type < 12)
			{
				err = _mtagprintf(p->Ctx, "cannot open value of type %s", type == 0 ? "null" : type == 7 ? "real" : "integer");
				rc = RC_ERROR;
				Vdbe::Finalize(v);
				p->Stmt = nullptr;
			}
			else
			{
				p->Offset = v->Cursors[0]->Offsets[p->Col];
				p->Bytes = Vdbe::SerialTypeLen(type);
				p->Cursor = v->Cursors[0]->Cursor;
				p->Cursor->EnterCursor();
				Btree::CacheOverflow(p->Cursor);
				p->Cursor->LeaveCursor();
			}
		}

		if (rc == RC_ROW)
			rc = RC_OK;
		else if (p->Stmt)
		{
			rc = Vdbe::Finalize(p->Stmt);
			p->Stmt = nullptr;
			if (rc == RC_OK)
			{
				err = _mtagprintf(p->Ctx, "no such rowid: %lld", row);
				rc = RC_ERROR;
			}
			else
				err = _mtagprintf(p->Ctx, "%s", Main::ErrMsg(p->Ctx));
		}

		_assert(rc != RC_OK || err == nullptr);
		_assert(rc != RC_ROW && rc != RC_DONE);
		*errOut = err;
		return rc;
	}

	__constant__ static const Vdbe::VdbeOpList _openBlob[] =
	{
		{OP_Transaction, 0, 0, 0},     // 0: Start a transaction
		{OP_VerifyCookie, 0, 0, 0},    // 1: Check the schema cookie
		{OP_TableLock, 0, 0, 0},       // 2: Acquire a read or write lock

		// One of the following two instructions is replaced by an OP_Noop.
		{OP_OpenRead, 0, 0, 0},        // 3: Open cursor 0 for reading
		{OP_OpenWrite, 0, 0, 0},       // 4: Open cursor 0 for read/write

		{OP_Variable, 1, 1, 1},        // 5: Push the rowid to the stack
		{OP_NotExists, 0, 10, 1},      // 6: Seek the cursor
		{OP_Column, 0, 0, 1},          // 7
		{OP_ResultRow, 1, 0, 0},       // 8
		{OP_Goto, 0, 5, 0},            // 9
		{OP_Close, 0, 0, 0},           // 10
		{OP_Halt, 0, 0, 0},            // 11
	};

	__device__ RC Vdbe::Blob_Open(Context *ctx, const char *dbName, const char *tableName, const char *columnName, int64 row, int flags, Blob **blobOut)
	{
		// This VDBE program seeks a btree cursor to the identified db/table/row entry. The reason for using a vdbe program instead
		// of writing code to use the b-tree layer directly is that the vdbe program will take advantage of the various transaction,
		// locking and error handling infrastructure built into the vdbe.
		//
		// After seeking the cursor, the vdbe executes an OP_ResultRow. Code external to the Vdbe then "borrows" the b-tree cursor and
		// uses it to implement the blob_read(), blob_write() and blob_bytes() functions.
		//
		// The sqlite3_blob_close() function finalizes the vdbe program, which closes the b-tree cursor and (possibly) commits the transaction.
		RC rc = RC_OK;
		flags = !!flags; // flags = (flags ? 1 : 0);
		*blobOut = nullptr;

		MutexEx::Enter(ctx->Mutex);
		Incrblob *blob = (Incrblob *)_tagalloc2(ctx, sizeof(Incrblob), true);
		if (!blob) goto blob_open_out;
		Parse *parse = (Parse *)_stackalloc(ctx, sizeof(*parse), false);
		if (!parse) goto blob_open_out;

		char *err = nullptr;
		int attempts = 0;
		do
		{
			_memset(parse, 0, sizeof(Parse));
			parse->Ctx = ctx;
			_tagfree(ctx, err);
			err = nullptr;

			Btree::EnterAll(ctx);
			Table *table = parse->LocateTable(false, tableName, dbName);
			if (table && IsVirtual(table))
			{
				table = nullptr;
				parse->ErrorMsg("cannot open virtual table: %s", tableName);
			}
#ifndef OMIT_VIEW
			if (table && table->Select)
			{
				table = nullptr;
				parse->ErrorMsg("cannot open view: %s", tableName);
			}
#endif
			if (!table)
			{
				if (parse->ErrMsg)
				{
					_tagfree(ctx, err);
					err = parse->ErrMsg;
					parse->ErrMsg = nullptr;
				}
				rc = RC_ERROR;
				Btree::LeaveAll(ctx);
				goto blob_open_out;
			}

			// Now search table for the exact column.
			int col; // Index of zColumn in row-record
			for (col = 0; col < table->Cols.length; col++)
				if (!_strcmp(table->Cols[col].Name, columnName))
					break;
			if (col == table->Cols.length)
			{
				_tagfree(ctx, err);
				err = _mtagprintf(ctx, "no such column: \"%s\"", columnName);
				rc = RC_ERROR;
				Btree::LeaveAll(ctx);
				goto blob_open_out;
			}

			// If the value is being opened for writing, check that the column is not indexed, and that it is not part of a foreign key. 
			// It is against the rules to open a column to which either of these descriptions applies for writing.
			if (flags)
			{
				const char *fault = nullptr;
#ifndef OMIT_FOREIGN_KEY
				if (ctx->Flags & BContext::FLAG_ForeignKeys)
				{
					// Check that the column is not part of an FK child key definition. It is not necessary to check if it is part of a parent key, as parent
					// key columns must be indexed. The check below will pick up this case.
					for (FKey *fkey = table->FKeys; fkey; fkey = fkey->NextFrom)
						for (int j = 0; j < fkey->Cols.length; j++)
							if (fkey->Cols[j].From == col)
								fault = "foreign key";
				}
#endif
				for (Index *index = table->Index; index; index = index->Next)
					for (int j = 0; j < index->Columns.length; j++)
						if (index->Columns[j] == col)
							fault = "indexed";
				if (fault)
				{
					_tagfree(ctx, err);
					err = _mtagprintf(ctx, "cannot open %s column for writing", fault);
					rc = RC_ERROR;
					Btree::LeaveAll(ctx);
					goto blob_open_out;
				}
			}

			blob->Stmt = Vdbe::Create(ctx);
			_assert(blob->Stmt || ctx->MallocFailed);
			if (blob->Stmt)
			{
				Vdbe *v = blob->Stmt;
				int db = Prepare::SchemaToIndex(ctx, table->Schema);
				v->AddOpList(_lengthof(_openBlob), _openBlob);
				// Configure the OP_Transaction
				v->ChangeP1(0, db);
				v->ChangeP2(0, flags);
				// Configure the OP_VerifyCookie
				v->ChangeP1(1, db);
				v->ChangeP2(1, table->Schema->SchemaCookie);
				v->ChangeP3(1, table->Schema->Generation);
				// Make sure a mutex is held on the table to be accessed
				v->UsesBtree(db); 

				// Configure the OP_TableLock instruction
#ifdef OMIT_SHARED_CACHE
				v->ChangeToNoop(2);
#else
				v->ChangeP1(2, db);
				v->ChangeP2(2, table->Id);
				v->ChangeP3(2, flags);
				v->ChangeP4(2, table->Name, Vdbe::P4T_TRANSIENT);
#endif

				// Remove either the OP_OpenWrite or OpenRead. Set the P2 parameter of the other to table->tnum.
				v->ChangeToNoop(4 - flags);
				v->ChangeP2(3 + flags, table->Id);
				v->ChangeP3(3 + flags, db);

				// Configure the number of columns. Configure the cursor to think that the table has one more column than it really
				// does. An OP_Column to retrieve this imaginary column will always return an SQL NULL. This is useful because it means
				// we can invoke OP_Column to fill in the vdbe cursors type and offset cache without causing any IO.
				v->ChangeP4(3+flags, (char *)INT_TO_PTR(table->Cols.length+1), Vdbe::P4T_INT32);
				v->ChangeP2(7, table->Cols.length);
				if (!ctx->MallocFailed)
				{
					parse->Vars.length = 1;
					parse->Mems = 1;
					parse->Tabs = 1;
					v->MakeReady(parse);
				}
			}

			blob->Flags = flags;
			blob->Col = col;
			blob->Ctx = ctx;
			Btree::LeaveAll(ctx);
			if (ctx->MallocFailed)
				goto blob_open_out;
			Vdbe::Bind_Int64(blob->Stmt, 1, row);
			rc = BlobSeekToRow(blob, row, &err);
		} while((++attempts) < 5 && rc == RC_SCHEMA);

blob_open_out:
		if (rc == RC_OK && !ctx->MallocFailed)
			*blobOut = (Blob *)blob;
		else
		{
			if (blob && blob->Stmt) Vdbe::Finalize(blob->Stmt);
			_tagfree(ctx, blob);
		}
		Main::Error(ctx, rc, (err ? "%s" : nullptr), err);
		_tagfree(ctx, err);
		_stackfree(ctx, parse);
		rc = Main::ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ RC Vdbe::Blob_Close(Blob *blob)
	{
		Incrblob *p = (Incrblob *)blob;
		if (p)
		{
			Context *ctx = p->Ctx;
			MutexEx::Enter(ctx->Mutex);
			RC rc = Finalize(p->Stmt);
			_tagfree(ctx, p);
			MutexEx::Leave(ctx->Mutex);
			return rc;
		}
		return RC_OK;
	}

	__device__ static RC BlobReadWrite(Blob *blob, void *z, uint32 n, uint32 offset, RC (*call)(BtCursor *, uint32, uint32, void *))
	{
		Incrblob *p = (Incrblob *)blob;
		if (!p) return SysEx_MISUSE_BKPT;
		Context *ctx = p->Ctx;
		MutexEx::Enter(ctx->Mutex);
		Vdbe *v = (Vdbe *)p->Stmt;

		RC rc;
		if (n < 0 || offset < 0 || (offset + n) > p->Bytes)
		{
			rc = RC_ERROR; // Request is out of range. Return a transient error.
			Main::Error(ctx, RC_ERROR, 0);
		}
		else if (!v)
			rc = RC_ABORT; // If there is no statement handle, then the blob-handle has already been invalidated. Return SQLITE_ABORT in this case.
		else
		{
			_assert(ctx == v->Ctx);
			p->Cursor->EnterCursor();
			rc = call(p->Cursor, offset + p->Offset, n, z); // Call either BtreeData() or BtreePutData(). If SQLITE_ABORT is returned, clean-up the statement handle.
			p->Cursor->LeaveCursor();
			if (rc == RC_ABORT)
			{
				Vdbe::Finalize(v);
				p->Stmt = nullptr;
			}
			else
			{
				ctx->ErrCode = rc;
				v->RC_ = rc;
			}
		}
		rc = Main::ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ RC Vdbe::Blob_Read(Blob *blob, void *z, int n, int offset)
	{
		return BlobReadWrite(blob, z, n, offset, Btree::Data);
	}

	__device__ RC Vdbe::Blob_Write(Blob *blob, const void *z, int n, int offset)
	{
		return BlobReadWrite(blob, (void *)z, n, offset, Btree::PutData);
	}

	__device__ int Vdbe::Blob_Bytes(Blob *blob)
	{
		Incrblob *p = (Incrblob *)blob;
		return (p && p->Stmt ? p->Bytes : 0);
	}

	__device__ RC Vdbe::Blob_Reopen(Blob *blob, int64 row)
	{
		Incrblob *p = (Incrblob *)blob;
		if (!p) return SysEx_MISUSE_BKPT;
		Context *ctx = p->Ctx;
		MutexEx::Enter(ctx->Mutex);

		RC rc;
		if (!p->Stmt)
			rc = RC_ABORT; // If there is no statement handle, then the blob-handle has already been invalidated. Return SQLITE_ABORT in this case.
		else
		{
			char *err;
			rc = BlobSeekToRow(p, row, &err);
			if (rc != RC_OK)
			{
				Main::Error(ctx, rc, (err ? "%s" : nullptr), err);
				_tagfree(ctx, err);
			}
			_assert(rc != RC_SCHEMA);
		}

		rc = Main::ApiExit(ctx, rc);
		_assert(rc == RC_OK || !p->Stmt);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

#endif
#pragma endregion

}