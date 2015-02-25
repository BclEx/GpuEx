using Core.IO;
using System;
using System.Diagnostics;

namespace Core
{
    public partial class Vdbe
    {
        #region IncrBlob
#if !OMIT_INCRBLOB

        class Incrblob : Blob
        {
            public int Flags;               // Copy of "flags" passed to sqlite3_blob_open()
            public int Bytes;               // Size of open blob, in bytes
            public uint Offset;              // Byte offset of blob in cursor data
            public int Col;				    // Table column this handle is open on
            public Btree.BtCursor Cursor;   // Cursor pointing at blob row
            public Vdbe Stmt;               // Statement holding cursor open
            public Context Ctx;             // The associated database
        }

        static RC BlobSeekToRow(Incrblob p, long row, out string errOut)
        {
            string err = null; // Error message
            Vdbe v = p.Stmt;

            // Set the value of the SQL statements only variable to integer iRow. This is done directly instead of using sqlite3_bind_int64() to avoid 
            // triggering asserts related to mutexes.
            Debug.Assert((v.Vars[0].Flags & MEM.Int) != 0);
            v.Vars[0].u.I = row;

            RC rc = v.Step();
            if (rc == RC.ROW)
            {
                uint type = v.Cursors[0].Types[p.Col];
                if (type < 12)
                {
                    err = C._mtagprintf(p.Ctx, "cannot open value of type %s", type == 0 ? "null" : type == 7 ? "real" : "integer");
                    rc = RC.ERROR;
                    Vdbe.Finalize(v);
                    p.Stmt = null;
                }
                else
                {
                    p.Offset = v.Cursors[0].Offsets[p.Col];
                    p.Bytes = SerialTypeLen(type);
                    p.Cursor = v.Cursors[0].Cursor;
                    p.Cursor.EnterCursor();
                    Btree.CacheOverflow(p.Cursor);
                    p.Cursor.LeaveCursor();
                }
            }

            if (rc == RC.ROW)
                rc = RC.OK;
            else if (p.Stmt != null)
            {
                rc = Vdbe.Finalize(p.Stmt);
                p.Stmt = null;
                if (rc == RC.OK)
                {
                    err = C._mtagprintf(p.Ctx, "no such rowid: %lld", row);
                    rc = RC.ERROR;
                }
                else
                    err = C._mtagprintf(p.Ctx, "%s", Main.Errmsg(p.Ctx));
            }

            Debug.Assert(rc != RC.OK || err == null);
            Debug.Assert(rc != RC.ROW && rc != RC.DONE);

            errOut = err;
            return rc;
        }

        static const Vdbe.VdbeOpList[] _openBlob =
        {
            new Vdbe.VdbeOpList(OP.Transaction, 0, 0, 0),     // 0: Start a transaction
            new Vdbe.VdbeOpList(OP.VerifyCookie, 0, 0, 0),    // 1: Check the schema cookie
            new Vdbe.VdbeOpList(OP.TableLock, 0, 0, 0),       // 2: Acquire a read or write lock

            // One of the following two instructions is replaced by an OP_Noop.
            new Vdbe.VdbeOpList(OP.OpenRead, 0, 0, 0),        // 3: Open cursor 0 for reading
            new Vdbe.VdbeOpList(OP.OpenWrite, 0, 0, 0),       // 4: Open cursor 0 for read/write

            new Vdbe.VdbeOpList(OP.Variable, 1, 1, 1),        // 5: Push the rowid to the stack
            new Vdbe.VdbeOpList(OP.NotExists, 0, 9, 1),       // 6: Seek the cursor
            new Vdbe.VdbeOpList(OP.Column, 0, 0, 1),          // 7
            new Vdbe.VdbeOpList(OP.ResultRow, 1, 0, 0),       // 8
            new Vdbe.VdbeOpList(OP.Close, 0, 0, 0),           // 9
            new Vdbe.VdbeOpList(OP.Halt, 0, 0, 0),            // 10
        };
        public RC Blob_Open(Context ctx, string dbName, string tableName, string columnName, long row, int flags, ref Blob blobOut)
        {
            // This VDBE program seeks a btree cursor to the identified db/table/row entry. The reason for using a vdbe program instead
            // of writing code to use the b-tree layer directly is that the vdbe program will take advantage of the various transaction,
            // locking and error handling infrastructure built into the vdbe.
            //
            // After seeking the cursor, the vdbe executes an OP_ResultRow. Code external to the Vdbe then "borrows" the b-tree cursor and
            // uses it to implement the blob_read(), blob_write() and blob_bytes() functions.
            //
            // The sqlite3_blob_close() function finalizes the vdbe program, which closes the b-tree cursor and (possibly) commits the transaction.
            RC rc = RC.OK;
            flags = (flags != 0 ? 1 : 0);
            blobOut = null;

            MutexEx.Enter(ctx.Mutex);
            Incrblob blob = new Incrblob();
            if (blob == null) goto blob_open_out;
            Parse parse = new Parse();
            if (parse == null) goto blob_open_out;

            string err = null;
            int attempts = 0;
            do
            {
                parse._memset();
                parse.Ctx = ctx;
                C._tagfree(ctx, ref err);
                err = null;

                Btree.EnterAll(ctx);
                Table table = sqlite3LocateTable(parse, 0, tableName, dbName);
                if (table != null && E.IsVirtual(table))
                {
                    table = null;
                    parse.ErrorMsg("cannot open virtual table: %s", tableName);
                }
#if !OMIT_VIEW
                if (table != null && table.Select != null)
                {
                    table = null;
                    parse.ErrorMsg("cannot open view: %s", tableName);
                }
#endif
                if (table == null)
                {
                    if (parse.ErrMsg != null)
                    {
                        C._tagfree(ctx, ref err);
                        err = parse.ErrMsg;
                        parse.ErrMsg = null;
                    }
                    rc = RC.ERROR;
                    Btree.LeaveAll(ctx);
                    goto blob_open_out;
                }

                // Now search table for the exact column.
                int col; // Index of zColumn in row-record
                for (col = 0; col < table.Cols.length; col++)
                    if (string.Equals(table.Cols[col].Name, columnName))
                        break;
                if (col == table.Cols.length)
                {
                    C._tagfree(ctx, ref err);
                    err = C._mtagprintf(ctx, "no such column: \"%s\"", columnName);
                    rc = RC.ERROR;
                    Btree.LeaveAll(ctx);
                    goto blob_open_out;
                }

                // If the value is being opened for writing, check that the column is not indexed, and that it is not part of a foreign key. 
                // It is against the rules to open a column to which either of these descriptions applies for writing.
                if (flags != 0)
                {
                    string fault = null;
#if !OMIT_FOREIGN_KEY
                    if ((ctx.Flags & BContext.FLAG.ForeignKeys) != 0)
                    {
                        // Check that the column is not part of an FK child key definition. It is not necessary to check if it is part of a parent key, as parent
                        // key columns must be indexed. The check below will pick up this case.
                        for (FKey fkey = table.FKeys; fkey != null; fkey = fkey.NextFrom)
                            for (int j = 0; j < fkey.Cols.length; j++)
                                if (fkey.Cols[j].From == col)
                                    fault = "foreign key";
                    }
#endif
                    for (Index index = table.Index; index != null; index = index.Next)
                        for (int j = 0; j < index.Columns.length; j++)
                            if (index.Columns[j] == col)
                                fault = "indexed";
                    if (fault != null)
                    {
                        C._tagfree(ctx, ref err);
                        err = C._mtagprintf(ctx, "cannot open %s column for writing", fault);
                        rc = RC.ERROR;
                        Btree.LeaveAll(ctx);
                        goto blob_open_out;
                    }
                }

                blob.Stmt = Vdbe.Create(ctx);
                Debug.Assert(blob.Stmt != null || ctx.MallocFailed);
                if (blob.Stmt != null)
                {
                    Vdbe v = blob.Stmt;
                    int db = sqlite3SchemaToIndex(ctx, table.Schema);
                    v.AddOpList(_openBlob.Length, _openBlob);
                    // Configure the OP_Transaction
                    v.ChangeP1(0, db);
                    v.ChangeP2(0, flags);
                    // Configure the OP_VerifyCookie
                    v.ChangeP1(1, db);
                    v.ChangeP2(1, table.Schema.SchemaCookie);
                    v.ChangeP3(1, table.Schema.Generation);
                    // Make sure a mutex is held on the table to be accessed
                    v.UsesBtree(db);

                    // Configure the OP_TableLock instruction
#if OMIT_SHARED_CACHE
				v.ChangeToNoop(2);
#else
                    v.ChangeP1(2, db);
                    v.ChangeP2(2, table.Id);
                    v.ChangeP3(2, flags);
                    v.ChangeP4(2, table.Name, Vdbe.P4T.TRANSIENT);
#endif

                    // Remove either the OP_OpenWrite or OpenRead. Set the P2 parameter of the other to table->tnum.
                    v.ChangeToNoop(4 - flags);
                    v.ChangeP2(3 + flags, table.Id);
                    v.ChangeP3(3 + flags, db);

                    // Configure the number of columns. Configure the cursor to think that the table has one more column than it really
                    // does. An OP_Column to retrieve this imaginary column will always return an SQL NULL. This is useful because it means
                    // we can invoke OP_Column to fill in the vdbe cursors type and offset cache without causing any IO.
                    v.ChangeP4(3 + flags, table.Cols.length + 1, Vdbe.P4T.INT32);
                    v.ChangeP2(7, table.Cols.length);
                    if (!ctx.MallocFailed)
                    {
                        parse.Vars.length = 1;
                        parse.Mems = 1;
                        parse.Tabs = 1;
                        v.MakeReady(parse);
                    }
                }

                blob.Flags = flags;
                blob.Col = col;
                blob.Ctx = ctx;
                Btree.LeaveAll(ctx);
                if (ctx.MallocFailed)
                    goto blob_open_out;
                Vdbe.Bind_Int64(blob.Stmt, 1, row);
                rc = BlobSeekToRow(blob, row, out err);
            } while ((++attempts) < 5 && rc == RC.SCHEMA);

        blob_open_out:
            if (rc == RC.OK && !ctx.MallocFailed)
                blobOut = (Blob)blob;
            else
            {
                if (blob != null && blob.Stmt != null) Vdbe.Finalize(blob.Stmt);
                C._tagfree(ctx, ref blob);
            }
            sqlite3Error(ctx, rc, (err != null ? "%s" : null), err);
            C._tagfree(ctx, ref err);
            C._stackfree(ctx, ref parse);
            rc = SysEx.ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public RC Blob_Close(Blob blob)
        {
            Incrblob p = (Incrblob)blob;
            if (p != null)
            {
                Context ctx = p.Ctx;
                MutexEx.Enter(ctx.Mutex);
                RC rc = Finalize(p.Stmt);
                C._tagfree(ctx, ref p);
                MutexEx.Free(ctx.Mutex);
                return rc;
            }
            return RC.OK;
        }

        static RC BlobReadWrite(Blob blob, string z, uint n, uint offset, Func<Btree.BtCursor, uint, uint, string, RC> call)
        {
            Incrblob p = (Incrblob)blob;
            if (p == null) return SysEx.MISUSE_BKPT();
            Context ctx = p.Ctx;
            MutexEx.Enter(ctx.Mutex);
            Vdbe v = (Vdbe)p.Stmt;

            RC rc;
            if (n < 0 || offset < 0 || (offset + n) > p.Bytes)
            {
                rc = RC.ERROR; // Request is out of range. Return a transient error.
                sqlite3Error(ctx, RC.ERROR, 0);
            }
            else if (v == null)
                rc = RC.ABORT; // If there is no statement handle, then the blob-handle has already been invalidated. Return SQLITE_ABORT in this case.
            else
            {
                Debug.Assert(ctx == v.Ctx);
                p.Cursor.EnterCursor();
                rc = call(p.Cursor, offset + p.Offset, n, z); // Call either BtreeData() or BtreePutData(). If SQLITE_ABORT is returned, clean-up the statement handle.
                p.Cursor.LeaveCursor();
                if (rc == RC.ABORT)
                {
                    Finalize(v);
                    p.Stmt = null;
                }
                else
                {
                    ctx.ErrCode = rc;
                    v.RC_ = rc;
                }
            }
            rc = SysEx.ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public int Blob_Read(Blob blob, object z, int n, int offset)
        {
            return BlobReadWrite(blob, z, n, offset, Btree.Data);
        }

        public int Blob_Write(Blob blob, string z, int n, int offset)
        {
            return BlobReadWrite(blob, z, n, offset, Btree.PutData);
        }

        public int Blob_Bytes(Blob blob)
        {
            Incrblob p = (Incrblob)blob;
            return (p != null ? p.Bytes : 0);
        }

#endif
        #endregion
    }
}
