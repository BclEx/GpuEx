using System;
using System.Diagnostics;
using System.IO;
using System.Text;

using Pid = System.UInt32;
#region Limits
#if MAX_ATTACHED
using yDbMask = System.Int64; 
#else
using yDbMask = System.Int32;
#endif
#if _64BITSTATS
using tRowcnt = System.UInt64; // 64-bit only if requested at compile-time
#else
using tRowcnt = System.UInt32; // 32-bit is the default
#endif
#endregion

namespace Core
{
    public partial class Parse
    {
        public void BeginParse(int explainFlag)
        {
            Explain = (byte)explainFlag;
            VarsSeen = 0;
        }

#if !OMIT_SHARED_CACHE
        public class TableLock
        {
            public int DB;             // The database containing the table to be locked
            public int Table;            // The root page of the table to be locked
            public bool IsWriteLock;    // True for write lock.  False for a read lock
            public string Name;        // Name of the table
        }

        public void TableLock(int db, int table, bool isWriteLock, string name)
        {
            Debug.Assert(db >= 0);
            Parse toplevel = E.Parse_Toplevel(this);
            TableLock tableLock;
            for (int i = 0; i < toplevel.TableLocks.length; i++)
            {
                tableLock = toplevel.TableLocks[i];
                if (tableLock.DB == db && tableLock.Table == table)
                {
                    tableLock.IsWriteLock = (tableLock.IsWriteLock || isWriteLock);
                    return;
                }
            }
            int bytes = 1 * (toplevel.TableLocks.length + 1);
            toplevel.TableLocks.data = SysEx.TagRealloc<TableLock>(toplevel.Ctx, 1, toplevel.TableLocks.data, bytes);
            if (toplevel.TableLocks.data != null)
            {
                tableLock = toplevel.TableLocks[toplevel.TableLocks.length++];
                tableLock.DB = db;
                tableLock.Table = table;
                tableLock.IsWriteLock = isWriteLock;
                tableLock.Name = name;
            }
            else
            {
                toplevel.TableLocks.length = 0;
                toplevel.Ctx.MallocFailed = true;
            }
        }

        static void CodeTableLocks(Parse parse)
        {
            Vdbe v = parse.GetVdbe();
            Debug.Assert(v != null); // sqlite3GetVdbe cannot fail: VDBE already allocated
            for (int i = 0; i < parse.TableLocks.length; i++)
            {
                TableLock p = parse.TableLocks[i];
                int p1 = p.DB;
                v.AddOp4(OP.TableLock, p1, p.Table, p.IsWriteLock, p.Name, Vdbe.P4T.STATIC);
            }
        }
#else
    static void CodeTableLocks(Parse parse) { }
#endif

        public void FinishCoding()
        {
            Context ctx = Ctx;
            if (ctx.MallocFailed || Nested != 0 || Errs != 0)
                return;

            // Begin by generating some termination code at the end of the vdbe program
            Vdbe v = GetVdbe();
            Debug.Assert(!IsMultiWrite || v.AssertMayAbort(MayAbort));
            if (v != null)
            {
                v.AddOp0(OP.Halt);

                // The cookie mask contains one bit for each database file open. (Bit 0 is for main, bit 1 is for temp, and so forth.)  Bits are
                // set for each database that is used.  Generate code to start a transaction on each used database and to verify the schema cookie
                // on each used database.
                if (CookieGoto > 0)
                {
                    int db; yDbMask mask;
                    v.JumpHere(CookieGoto - 1);
                    for (db = 0, mask = 1; db < ctx.DBs.length; mask <<= 1, db++)
                    {
                        if ((mask & CookieMask) == 0)
                            continue;
                        v.UsesBtree(db);
                        v.AddOp2(OP.Transaction, db, (mask & WriteMask) != 0);
                        if (!ctx.Init.Busy)
                        {
                            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                            v.AddOp3(OP.VerifyCookie, db, CookieValue[db], (int)ctx.DBs[db].Schema.Generation);
                        }
                    }
#if !OMIT_VIRTUALTABLE
                    {
                        for (int i = 0; i < VTableLocks.length; i++)
                        {
                            VTable vtable = VTable.GetVTable(ctx, VTableLocks[i]);
                            v.AddOp4(OP.VBegin, 0, 0, 0, vtable, Vdbe.P4T.VTAB);
                        }
                        VTableLocks.length = 0;
                    }
#endif

                    // Once all the cookies have been verified and transactions opened, obtain the required table-locks. This is a no-op unless the 
                    // shared-cache feature is enabled.
                    CodeTableLocks(this);
                    // Initialize any AUTOINCREMENT data structures required.
                    AutoincrementBegin();
                    // Finally, jump back to the beginning of the executable code.
                    v.AddOp2(OP.Goto, 0, CookieGoto);
                }
            }


            // Get the VDBE program ready for execution
            if (v != null && C._ALWAYS(Errs == 0) && !ctx.MallocFailed)
            {
#if DEBUG && !WINRT
                TextWriter trace = ((ctx.Flags & BContext.FLAG.VdbeTrace) != 0 ? Console.Out : null);
                v.Trace(trace);
#endif
                Debug.Assert(CacheLevel == 0); // Disables and re-enables match
                // A minimum of one cursor is required if autoincrement is used See ticket [a696379c1f08866]
                if (Ainc != null && Tabs == 0)
                    Tabs = 1;
                v.MakeReady(this);
                RC = RC.DONE;
                ColNamesSet = 0;
            }
            else
                RC = RC.ERROR;
            Tabs = 0;
            Mems = 0;
            Sets = 0;
            VarsSeen = 0;
            CookieMask = 0;
            CookieGoto = 0;
        }


        static object _nestedParseLock = new object();
        public void NestedParse(string format, params object[] args)
        {

            //# define SAVE_SZ  (Parse.Length - offsetof(Parse,nVar))
            //  char saveBuf[SAVE_SZ];
            if (Errs != 0)
                return;
            Debug.Assert(Nested < 10); // Nesting should only be of limited depth
            //  va_list ap;
            string sql; //  string zSql;
            lock (lock_va_list)
            {
                va_start(args, format);
                sql = sqlite3VMPrintf(ctx, format, args);
                va_end(ref args);
            }
            if (sql == null)
                return;   // A malloc must have failed
            lock (_nestedParseLock)
            {
                Nested++;
                SaveMembers();     //  memcpy(saveBuf, pParse.nVar, SAVE_SZ);
                ResetMembers();    //  memset(pParse.nVar, 0, SAVE_SZ);
                string errMsg = string.Empty;
                RunParser(sql, ref errMsg);
                Context ctx = Ctx;
                C._tagfree(ctx, ref errMsg);
                C._tagfree(ctx, ref sql);
                RestoreMembers();  //  memcpy(pParse.nVar, saveBuf, SAVE_SZ);
                Nested--;
            }
        }

        public static Table FindTable(Context ctx, string name, string dbName)
        {
            Debug.Assert(name != null);
            int nameLength = name.Length;
            // All mutexes are required for schema access.  Make sure we hold them.
            Debug.Assert(dbName != null || Btree.HoldsAllMutexes(ctx));
            Table table = null;
            for (int i = E.OMIT_TEMPDB; i < ctx.DBs.length; i++)
            {
                int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
                if (dbName != null && !string.Equals(dbName, ctx.DBs[j].Name, StringComparison.OrdinalIgnoreCase))
                    continue;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, j, null));
                table = (Table)ctx.DBs[j].Schema.TableHash.Find(name, nameLength, (Table)null);
                if (table != null)
                    break;
            }
            return table;
        }


        public Table LocateTable(bool isView, string name, string dbName)
        {
            // Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
            if (ReadSchema() != RC.OK)
                return null;

            Table table = FindTable(Ctx, name, dbName);
            if (table == null)
            {
                string msg = (isView ? "no such view" : "no such table");
                if (dbName != null)
                    ErrorMsg("%s: %s.%s", msg, dbName, name);
                else
                    ErrorMsg("%s: %s", msg, name);
                CheckSchema = 1;
            }
            return table;
        }

        public Table LocateTableItem(bool isView, SrcList.SrcListItem item)
        {
            Debug.Assert(item.Schema == null || item.Database == null);
            string dbName;
            if (item.Schema != null)
            {
                int db = SchemaToIndex(Ctx, item.Schema);
                dbName = Ctx.DBs[db].Name;
            }
            else
                dbName = item.Database;
            return LocateTable(isView, item.Name, dbName);
        }

        public static Index FindIndex(Context ctx, string name, string dbName)
        {
            // All mutexes are required for schema access.  Make sure we hold them.
            Debug.Assert(dbName != null || Btree.HoldsAllMutexes(ctx));
            Index p = null;
            int nameLength = name.Length;
            for (int i = E.OMIT_TEMPDB; i < ctx.DBs.length; i++)
            {
                int j = (i < 2 ? i ^ 1 : i);  // Search TEMP before MAIN
                Schema schema = ctx.DBs[j].Schema;
                Debug.Assert(schema != null);
                if (dbName != null && !string.Equals(dbName, ctx.DBs[j].Name, StringComparison.OrdinalIgnoreCase))
                    continue;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, j, null));
                p = schema.IndexHash.Find(name, nameLength, (Index)null);
                if (p != null)
                    break;
            }
            return p;
        }

        static void FreeIndex(Context ctx, ref Index p)
        {
#if !OMIT_ANALYZE
            Parse.DeleteIndexSamples(ctx, p);
#endif
            C._tagfree(ctx, ref p.ColAff);
            C._tagfree(ctx, ref p);
        }

        public static void UnlinkAndDeleteIndex(Context ctx, int db, string indexName)
        {
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            Hash hash = ctx.DBs[db].Schema.IndexHash;
            int indexNameLength = indexName.Length;
            Index index = hash.Insert(indexName, indexNameLength, (Index)null);
            if (C._ALWAYS(index != null))
            {
                if (index.Table.Index == index)
                    index.Table.Index = index.Next;
                else
                {
                    // Justification of ALWAYS();  The index must be on the list of indices.
                    Index p = index.Table.Index;
                    while (C._ALWAYS(p != null) && p.Next != index)
                        p = p.Next;
                    if (C._ALWAYS(p != null && p.Next == index))
                        p.Next = index.Next;
                }
                FreeIndex(ctx, ref index);
            }
            ctx.Flags |= Context.FLAG.InternChanges;
        }

        public static void CollapseDatabaseArray(Context ctx)
        {
            // If one or more of the auxiliary database files has been closed, then remove them from the auxiliary database list.  We take the
            // opportunity to do this here since we have just deleted all of the schema hash tables and therefore do not have to make any changes
            // to any of those tables.
            int i, j;
            for (i = j = 2; i < ctx.DBs.length; i++)
            {
                Context.DB db = ctx.DBs[i];
                if (db.Bt == null)
                {
                    C._tagfree(ctx, ref db.Name);
                    continue;
                }
                if (j < i)
                    ctx.DBs[j] = ctx.DBs[i];
                j++;
            }
            if (ctx.DBs.length != j)
                ctx.DBs[j] = new Context.DB();
            ctx.DBs.length = j;
            if (ctx.DBs.length <= 2 && ctx.DBs.data != ctx.DBStatics)
                Array.Copy(ctx.DBs.data, ctx.DBStatics, 2);
        }

        public static void ResetOneSchema(Context ctx, int db)
        {
            Debug.Assert(db < ctx.DBs.length);
            // Case 1:  Reset the single schema identified by iDb
            Context.DB dbobj = ctx.DBs[db];
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            Debug.Assert(dbobj.Schema != null);
            Callback.SchemaClear(dbobj.Schema);
            // If any database other than TEMP is reset, then also reset TEMP since TEMP might be holding triggers that reference tables in the other database.
            if (db != 1)
            {
                dbobj = ctx.DBs[1];
                Debug.Assert(dbobj.Schema != null);
                Callback.SchemaClear(dbobj.Schema);
            }
        }

        public static void ResetAllSchemasOfConnection(Context ctx)
        {
            // Case 2 (from here to the end): Reset all schemas for all attached databases.
            Btree.EnterAll(ctx);
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Context.DB db = ctx.DBs[i];
                if (db.Schema != null)
                    Callback.SchemaClear(db.Schema);
            }
            ctx.Flags &= ~Context.FLAG.InternChanges;
            VTable.UnlockList(ctx);
            Btree.LeaveAll(ctx);
            CollapseDatabaseArray(ctx);
        }

        public static void CommitInternalChanges(Context ctx)
        {
            ctx.Flags &= ~Context.FLAG.InternChanges;
        }

        public static void DeleteColumnNames(Context ctx, Table table)
        {
            Debug.Assert(table != null);
            for (int i = 0; i < table.Cols.length; i++)
            {
                Column col = table.Cols[i];
                if (col != null)
                {
                    C._tagfree(ctx, ref col.Name);
                    Expr.Delete(ctx, ref col.Dflt);
                    C._tagfree(ctx, ref col.Dflt);
                    C._tagfree(ctx, ref col.Type);
                    C._tagfree(ctx, ref col.Coll);
                }
                C._tagfree(ctx, ref table.Cols.data);
            }
        }

        public static void DeleteTable(Context ctx, ref Table table)
        {
            Debug.Assert(table == null || table.Refs > 0);

            // Do not delete the table until the reference count reaches zero.
            if (table == null) return;
            if ((ctx == null || ctx.BusyHandler == null) && --table.Refs > 0) return;

            // Record the number of outstanding lookaside allocations in schema Tables prior to doing any free() operations.  Since schema Tables do not use
            // lookaside, this number should not change.
#if DEBUG
            int lookaside = (ctx != null && (table.TabFlags & TF.Ephemeral) == 0 ? ctx.Lookaside.Outs : 0); // Used to verify lookaside not used for schema 
#endif

            // Delete all indices associated with this table.
            Index index, next;
            for (index = table.Index; index != null; index = next)
            {
                next = index.Next;
                Debug.Assert(index.Schema == table.Schema);
                //if(ctx == null || ctx.BytesFreed == 0){
                string name = index.Name;
#if DEBUG || COVERAGE_TEST
                Index oldIndex = index.Schema.IndexHash.Insert(name, name.Length, (Index)null);
#else
                index.Schema.IndexHash.Insert(name, name.Length, (Index)null);
#endif
                Debug.Assert(ctx == null || Btree.SchemaMutexHeld(ctx, 0, index.Schema));
                Debug.Assert(oldIndex == null || oldIndex == index);
                //}
                FreeIndex(ctx, ref index);
            }

            // Delete any foreign keys attached to this table.
            FkDelete(ctx, table);

            // Delete the Table structure itself.
            DeleteColumnNames(ctx, table);
            C._tagfree(ctx, ref table.Name);
            C._tagfree(ctx, ref table.ColAff);
            Select.Delete(ctx, ref table.Select);
#if !OMIT_CHECK
            Expr.Delete(ctx, ref table.Check);
#endif
#if !OMIT_VIRTUALTABLE
            VTable.Clear(ctx, table);
#endif
            C._tagfree(ctx, ref table);

            // Verify that no lookaside memory was used by schema tables
            Debug.Assert(lookaside != null || lookaside == ctx.Lookaside.Outs);
        }

        public static void UnlinkAndDeleteTable(Context ctx, int db, string tableName)
        {
            Debug.Assert(ctx != null);
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            Debug.Assert(tableName != null);
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            C.ASSERTCOVERAGE(tableName.Length == 0);  // Zero-length table names are allowed
            Context.DB db2 = ctx.DBs[db];
            Table table = db2.Schema.TableHash.Insert(tableName, tableName.Length, (Table)null);
            DeleteTable(ctx, ref table);
            ctx.Flags |= Context.FLAG.InternChanges;
        }

        public static string NameFromToken(Context ctx, Token name)
        {
            if (name != null && name.data != null)
            {
                string nameAsString = name.data.Substring(0, (int)name.length);
                Parse.Dequote(ref nameAsString);
                return nameAsString;
            }
            return null;
        }

        public void OpenMasterTable(int db)
        {
            Vdbe v = GetVdbe();
            TableLock(db, MASTER_ROOT, 1, SCHEMA_TABLE(db));
            v.AddOp3(OP.OpenWrite, 0, MASTER_ROOT, db);
            v.ChangeP4(-1, (int)5, Vdbe.P4T.INT32); // 5 column table
            if (Tabs == 0)
                Tabs = 1;
        }

        public static int FindDbName(Context ctx, string name)
        {
            int db = -1; // Database number
            if (name != null)
            {
                int nameLength = name.Length;
                Context.DB db2;
                for (db = (ctx.DBs.length - 1); db >= 0; db--)
                {
                    db2 = ctx.DBs[db];
                    if ((E.OMIT_TEMPDB == 0 || db != 1) && nameLength == db2.Name.Length && !string.Equals(db2.Name, name, StringComparison.OrdinalIgnoreCase))
                        break;
                }
            }
            return db;
        }

        public static int FindDb(Context ctx, Token name)
        {
            string nameAsString = NameFromToken(ctx, name);// Name we are searching for
            int i = FindDbName(ctx, nameAsString); // Database number
            C._tagfree(ctx, ref nameAsString);
            return i;
        }

        public int TwoPartName(Token name1, Token name2, ref Token unqual)
        {
            Context ctx = Ctx;
            int db; // Database holding the object
            if (C._ALWAYS(name2 != null) && name2.length > 0)
            {
                if (ctx.Init.Busy)
                {
                    ErrorMsg("corrupt database");
                    Errs++;
                    return -1;
                }
                unqual = name2;
                db = FindDb(ctx, name1);
                if (db < 0)
                {
                    ErrorMsg("unknown database %T", name1);
                    Errs++;
                    return -1;
                }
            }
            else
            {
                Debug.Assert(ctx.Init.DB == 0 || ctx.Init.Busy);
                db = ctx.Init.DB;
                unqual = name1;
            }
            return db;
        }

        public RC CheckObjectName(string name)
        {
            if (!Ctx.Init.Busy && Nested == 0 && (Ctx.Flags & Context.FLAG.WriteSchema) == 0 && name.StartsWith("sqlite_", StringComparison.OrdinalIgnoreCase))
            {
                ErrorMsg("object name reserved for internal use: %s", name);
                return RC.ERROR;
            }
            return RC.OK;
        }

        public void StartTable(Token name1, Token name2, bool isTemp, bool isView, bool isVirtual, bool noErr)
        {
            // The table or view name to create is passed to this routine via tokens pName1 and pName2. If the table name was fully qualified, for example:
            //
            // CREATE TABLE xxx.yyy (...);
            // 
            // Then pName1 is set to "xxx" and pName2 "yyy". On the other hand if the table name is not fully qualified, i.e.:
            //
            // CREATE TABLE yyy(...);
            //
            // Then pName1 is set to "yyy" and pName2 is "".
            //
            // The call below sets the unqual pointer to point at the token (pName1 or pName2) that stores the unqualified table name. The variable iDb is
            // set to the index of the database that the table or view is to be created in.
            Token unqual = new Token(); // Unqualified name of the table to create
            int db = TwoPartName(name1, name2, ref unqual); // Database number to create the table in
            if (db < 0)
                return;
            if (E.OMIT_TEMPDB == 0 && isTemp && name2.length > 0 && db != 1)
            {
                // If creating a temp table, the name may not be qualified. Unless the database name is "temp" anyway.
                ErrorMsg("temporary table name must be unqualified");
                return;
            }
            if (E.OMIT_TEMPDB == 0 && isTemp)
                db = 1;
            NameToken = unqual;
            Context ctx = Ctx;
            string name = NameFromToken(ctx, unqual); // The name of the new table
            if (name == null)
                return;
            Vdbe v;
            Table table;
            if (CheckObjectName(name) != RC.OK)
                goto begin_table_error;
            if (ctx.Init.DB == 1)
                isTemp = true;
#if !OMIT_AUTHORIZATION
            //Debug.Assert((isTemp & 1) == isTemp);
            {
                AUTH code;
                string dbName = ctx.DBs[db].Name;
                if (Auth.Check(this, AUTH.INSERT, E.SCHEMA_TABLE(isTemp), 0, dbName))
                    goto begin_table_error;
                if (isView)
                    code = (E.OMIT_TEMPDB == 0 && isTemp ? AUTH.CREATE_TEMP_VIEW : AUTH.CREATE_VIEW);
                else
                    code = (E.OMIT_TEMPDB == 0 && isTemp ? AUTH.CREATE_TEMP_TABLE : AUTH.CREATE_TABLE);
                if (!isVirtual && Auth.Check(this, code, name, null, dbName) != 0)
                    goto begin_table_error;
            }
#endif

            // Make sure the new table name does not collide with an existing index or table name in the same database.  Issue an error message if
            // it does. The exception is if the statement being parsed was passed to an sqlite3_declare_vtab() call. In that case only the column names
            // and types will be used, so there is no need to test for namespace collisions.
            if (!E.INDECLARE_VTABLE(this))
            {
                string dbName = ctx.DBs[db].Name;
                if (ReadSchema(this) != RC.OK)
                    goto begin_table_error;
                table = FindTable(ctx, name, dbName);
                if (table != null)
                {
                    if (!noErr)
                        ErrorMsg("table %T already exists", unqual);
                    else
                    {
                        Debug.Assert(!ctx.Init.Busy);
                        CodeVerifySchema(this, db);
                    }
                    goto begin_table_error;
                }
                if (FindIndex(ctx, name, dbName) != null)
                {
                    ErrorMsg("there is already an index named %s", name);
                    goto begin_table_error;
                }
            }

            table = new Table();
            if (table == null)
            {
                ctx.MallocFailed = true;
                RC = RC.NOMEM;
                Errs++;
                goto begin_table_error;
            }
            table.Name = name;
            table.PKey = -1;
            table.Schema = ctx.DBs[db].Schema;
            table.Refs = 1;
            table.RowEst = 1000000;
            Debug.Assert(NewTable == null);
            NewTable = table;

            // If this is the magic sqlite_sequence table used by autoincrement, then record a pointer to this table in the main database structure
            // so that INSERT can find the table easily.
#if !OMIT_AUTOINCREMENT
            if (Nested == 0 && name == "sqlite_sequence")
            {
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                table.Schema.SeqTable = table;
            }
#endif

            // Begin generating the code that will insert the table record into the SQLITE_MASTER table.  Note in particular that we must go ahead
            // and allocate the record number for the table entry now.  Before any PRIMARY KEY or UNIQUE keywords are parsed.  Those keywords will cause
            // indices to be created and the table record must come before the indices.  Hence, the record number for the table must be allocated now.
            if (!ctx.Init.Busy && (v = GetVdbe()) != null)
            {
                BeginWriteOperation(0, db);
#if !OMIT_VIRTUALTABLE
                if (isVirtual)
                    v.AddOp0(OP.VBegin);
#endif

                // If the file format and encoding in the database have not been set, set them now.
                int reg1 = RegRowid = ++Mems;
                int reg2 = RegRoot = ++Mems;
                int reg3 = ++Mems;
                v.AddOp3(OP.ReadCookie, db, reg3, BTREE_FILE_FORMAT);
                v.UsesBtree(db);
                int j1 = v.AddOp1(OP.If, reg3);
                int fileFormat = (ctx.Flags & Context.FLAG.LegacyFileFmt) != 0 ? 1 : MAX_FILE_FORMAT;
                v.AddOp2(OP.Integer, fileFormat, reg3);
                v.AddOp3(OP.SetCookie, db, BTREE_FILE_FORMAT, reg3);
                v.AddOp2(OP.Integer, Context.CTXENCODE(ctx), reg3);
                v.AddOp3(OP.SetCookie, db, BTREE_TEXT_ENCODING, reg3);
                v.JumpHere(j1);

                // This just creates a place-holder record in the sqlite_master table. The record created does not contain anything yet.  It will be replaced
                // by the real entry in code generated at sqlite3EndTable().
                //
                // The rowid for the new entry is left in register pParse->regRowid. The root page number of the new table is left in reg pParse->regRoot.
                // The rowid and root page number values are needed by the code that sqlite3EndTable will generate.
#if !OMIT_VIEW || !OMIT_VIRTUALTABLE
                if (isView || isVirtual)
                    v.AddOp2(OP.Integer, 0, reg2);
                else
#endif
                    v.AddOp2(OP.CreateTable, db, reg2);
                OpenMasterTable(db);
                v.AddOp2(OP.NewRowid, 0, reg1);
                v.AddOp2(OP.Null, 0, reg3);
                v.AddOp3(OP.Insert, 0, reg3, reg1);
                v.ChangeP5(OPFLAG_APPEND);
                v.AddOp0(OP.Close);
            }
            return;

        begin_table_error:
            C._tagfree(ctx, ref name);
            return;
        }

        public void AddColumn(Token name)
        {
            Table table;
            if ((table = NewTable) == null)
                return;
            Context ctx = Ctx;
#if MAX_COLUMN || !MAX_COLUMN
            if (table.Cols.length + 1 > ctx.Limits[(int)LIMIT.COLUMN])
            {
                ErrorMsg("too many columns on %s", table.Name);
                return;
            }
#endif
            string nameAsString = NameFromToken(ctx, name);
            if (nameAsString == null)
                return;
            for (int i = 0; i < table.Cols.length; i++)
            {
                if (string.Equals(nameAsString, table.Cols[i].Name, StringComparison.OrdinalIgnoreCase))
                {
                    ErrorMsg("duplicate column name: %s", nameAsString);
                    C._tagfree(ctx, ref nameAsString);
                    return;
                }
            }
            if ((table.Cols.length & 0x7) == 0)
                Array.Resize(ref table.Cols.data, table.Cols.length + 8);
            table.Cols[table.Cols.length] = new Column();
            Column col = table.aCol[table.nCol];
            col.Name = nameAsString;

            // If there is no type specified, columns have the default affinity 'NONE'. If there is a type specified, then sqlite3AddColumnType() will
            // be called next to set pCol->affinity correctly.
            col.Affinity = AFF.NONE;
            table.Cols.length++;
        }

        public void AddNotNull(byte onError)
        {
            Table table = NewTable;
            if (table == null || C._NEVER(table.Cols.length < 1))
                return;
            table.Cols[table.Cols.length - 1].NotNull = onError;
        }

        public static AFF AffinityType(string data)
        {
            data = data.ToLower();
            if (data.Contains("char") || data.Contains("clob") || data.Contains("text"))
                return AFF.TEXT;
            if (data.Contains("blob"))
                return AFF.NONE;
            if (data.Contains("doub") || data.Contains("floa") || data.Contains("real"))
                return AFF.REAL;
            if (data.Contains("int"))
                return AFF.INTEGER;
            return AFF.NUMERIC;
        }

        public void AddColumnType(Token type)
        {
            Table table = NewTable;
            if (table == null || C._NEVER(table.Cols.length < 1))
                return;
            Column col = table.Cols[table.Cols.length - 1];
            Debug.Assert(col.Type == null);
            col.Type = NameFromToken(Ctx, type);
            col.Affinity = AffinityType(col.Type);
        }

        public void AddDefaultValue(ExprSpan span)
        {
            Context ctx = Ctx;
            Table table = NewTable;
            if (table != null)
            {
                Column col = (table.Cols[table.Cols.length - 1]);
                if (!span.Expr.IsConstantOrFunction())
                    ErrorMsg("default value of column [%s] is not constant", col.Name);
                else
                {
                    // A copy of pExpr is used instead of the original, as pExpr contains tokens that point to volatile memory. The 'span' of the expression
                    // is required by pragma table_info.
                    Expr.Delete(ctx, ref col.Dflt);
                    col.Dflt = Expr.Dup(ctx, span.Expr, EXPRDUP.REDUCE);
                    C._tagfree(ctx, ref col.Dflt);
                    col.Dflt = span.Start.Substring(0, span.Start.Length - span.End.Length);
                }
            }
            Expr.Delete(ctx, ref span.Expr);
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        //public void AddPrimaryKey(int null_2, OE onError, bool autoInc, SO sortOrder) { AddPrimaryKey(null, onError, autoInc, sortOrder); }
        public void AddPrimaryKey(ExprList list, OE onError, bool autoInc, SO sortOrder)
        {
            Table table = NewTable;
            if (table == null || E.INDECLARE_VTABLE(this))
                goto primary_key_exit;
            if ((table.TabFlags & TF.HasPrimaryKey) != 0)
            {
                ErrorMsg("table \"%s\" has more than one primary key", table.Name);
                goto primary_key_exit;
            }
            table.TabFlags |= TF.HasPrimaryKey;
            int col = -1;
            if (list == null)
            {
                col = table.Cols.length - 1;
                table.Cols[col].ColFlags |= COLFLAG.PRIMKEY;
            }
            else
            {
                for (int i = 0; i < list.Exprs; i++)
                {
                    for (col = 0; col < table.Cols.length; col++)
                        if (string.Equals(list.Ids[i].Name, table.Cols[col].Name, StringComparison.OrdinalIgnoreCase))
                            break;
                    if (col < table.Cols.length)
                        table.Cols[col].ColFlags |= COLFLAG.PRIMKEY;
                }
                if (list.Exprs > 1)
                    col = -1;
            }
            string type = null;
            if (col >= 0 && col < table.Cols.length)
                type = table.Cols[col].Type;
            if (type != null && type.Equals("INTEGER", StringComparison.OrdinalIgnoreCase) && sortOrder == SO.ASC)
            {
                table.PKey = (short)col;
                table.KeyConf = (byte)onError;
                table.TabFlags |= (autoInc ? TF.Autoincrement : 0);
            }
            else if (autoInc)
            {
#if !OMIT_AUTOINCREMENT
                ErrorMsg("AUTOINCREMENT is only allowed on an INTEGER PRIMARY KEY");
#endif
            }
            else
            {
                Index index = CreateIndex(0, 0, 0, list, onError, 0, 0, sortOrder, 0);
                if (index != null)
                    index.AutoIndex = 2;
                list = null;
            }

        primary_key_exit:
            Expr.ListDelete(Ctx, ref list);
            return;
        }


        public void AddCheckConstraint(Expr checkExpr)
        {
            Context ctx = Ctx;
#if !OMIT_CHECK
            Table table = NewTable;
            if (table != null && !E.INDECLARE_VTABLE(this))
            {
                table.Check = Expr.ListAppend(table.Check, checkExpr);
                if (ConstraintName.length != 0)
                    Expr.ListSetName(table.Check, ConstraintName, 1);
            }
            else
#endif
                Expr.Delete(ctx, ref checkExpr);
        }

        public void AddCollateType(Token token)
        {
            Table table;
            if ((table = NewTable) == null)
                return;
            int col = table.Cols.length - 1;
            Context ctx = Ctx;
            string collName = NameFromToken(ctx, token); // Dequoted name of collation sequence
            if (collName == null)
                return;
            if (LocateCollSeq(collName) != null)
            {
                table.Cols[col].Coll = collName;
                // If the column is declared as "<name> PRIMARY KEY COLLATE <type>", then an index may have been created on this column before the
                // collation type was added. Correct this if it is the case.
                for (Index index = table.Index; index != null; index = index.Next)
                {
                    Debug.Assert(index.Columns.length == 1);
                    if (index.Columns[0] == col)
                        index.CollNames[0] = table.Cols[col].Coll;
                }
            }
            else
                C._tagfree(ctx, ref collName);
        }

        public CollSeq LocateCollSeq(string name)
        {
            Context ctx = Ctx;
            TEXTENCODE encode = E.CTXENCODE(ctx);
            bool initbusy = ctx.Init.Busy;
            CollSeq coll = Callback.FindCollSeq(ctx, encode, name, initbusy);
            if (!initbusy && (coll == null || coll.Cmp == null))
                coll = Callback.GetCollSeq(this, encode, coll, name);
            return coll;
        }

        public void ChangeCookie(int db)
        {
            int r1 = Expr.GetTempReg();
            Context ctx = Ctx;
            Vdbe v = V;
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            v.AddOp2(OP.Integer, ctx.DBs[db].Schema.SchemaCookie + 1, r1);
            v.AddOp3(OP.SetCookie, db, BTREE_SCHEMA_VERSION, r1);
            Expr.ReleaseTempReg(r1);
        }

        static int IdentLength(string z)
        {
            int size;
            for (size = 0; size < z.Length; size++)
                if (z[size] == (byte)'"')
                    size++;
            return size + 2;
        }

        static void IdentPut(StringBuilder z, ref int idx, string signedIdent)
        {
            string ident = signedIdent;
            int i = idx;
            int j;
            for (j = 0; j < ident.Length; j++)
                if (!char.IsLetterOrDigit(ident[j]) && ident[j] != '_')
                    break;
            bool needQuote = char.IsDigit(ident[0]) || KeywordCode(ident, j) != TK.ID;
            if (!needQuote) needQuote = (j < ident.Length && ident[j] != 0);
            else { if (i == z.Length) z.Append('\0'); z[i++] = '"'; }
            for (j = 0; j < ident.Length; j++)
            {
                if (i == z.Length) z.Append('\0'); z[i++] = ident[j];
                if (ident[j] == '"') { if (i == z.Length) z.Append('\0'); z[i++] = '"'; }
            }
            if (needQuote) { if (i == z.Length) z.Append('\0'); z[i++] = '"'; }
            idx = i;
        }

        static string[] _createTableStmt_Types = new string[]  {
            " TEXT",// AFF_TEXT
            "", // AFF_NONE
            " NUM", // AFF_NUMERIC
            " INT", // AFF_INTEGER
            " REAL"}; // AFF_REAL
        static string CreateTableStmt(Context ctx, Table table)
        {
            Column col;
            int i, n = 0;
            for (i = 0; i < table.Cols.length; i++)
            {
                col = table.Cols[i];
                n += identLength(col.Name) + 5;
            }
            n += IdentLength(table.Name);
            string sep, sep2, end;
            if (n < 50)
            {
                sep = "";
                sep2 = ",";
                end = ")";
            }
            else
            {
                sep = "\n  ";
                sep2 = ",\n  ";
                end = "\n)";
            }
            n += 35 + 6 * table.Cols.length;
            StringBuilder stmt = new StringBuilder(n);
            stmt.Append("CREATE TABLE ");
            int k = stmt.Length;
            identPut(stmt, ref k, table.zName);
            stmt.Append('(');
            for (i = 0; i < table.nCol; i++)
            {
                col = table.Cols[i];
                stmt.Append(sep);
                k = stmt.Length;
                sep = sep2;
                IdentPut(stmt, ref k, col.Name);
                Debug.Assert(col.Affinity - AFF.TEXT >= 0);
                Debug.Assert(col.Affinity - AFF.TEXT < _createTableStmt_Types.Length);
                C.ASSERTCOVERAGE(col.Affinity == AFF.TEXT);
                C.ASSERTCOVERAGE(col.Affinity == AFF.NONE);
                C.ASSERTCOVERAGE(col.Affinity == AFF.NUMERIC);
                C.ASSERTCOVERAGE(col.Affinity == AFF.INTEGER);
                C.ASSERTCOVERAGE(col.Affinity == AFF.REAL);

                string type = _createTableStmt_Types[col.Affinity - SQLITE_AFF_TEXT];
                int typeLength = type.Length;
                Debug.Assert(col.Affinity == AFF.NONE || col.Affinity == sqlite3AffinityType(type));
                stmt.Append(type);
                k += typeLength;
                Debug.Assert(k <= n);
            }
            stmt.Append(end);
            return stmt.ToString();
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        //public void EndTable(Token cons, Token end, int null_4) { EndTable(cons, end, null); }
        //public void EndTable(int null_2, int null_3, Select select) { EndTable(null, null, select); }
        public void EndTable(Token cons, Token end, Select select)
        {
            Context ctx = Ctx;
            if ((end == null && select == null) || ctx.MallocFailed)
                return;
            Table table = NewTable;
            if (table == null)
                return;
            Debug.Assert(!ctx.Init.Busy || select == null);

            int db = Prepare.SchemaToIndex(ctx, table.Schema);
#if !OMIT_CHECK
            // Resolve names in all CHECK constraint expressions.
            if (table.Check != null)
            {
                SrcList src; // Fake SrcList for pParse.pNewTable
                NameContext nc; // Name context for pParse.pNewTable
                nc = new NameContext();
                src = new SrcList();
                src.Srcs = 1;
                src.Ids = new SrcList.SrcListItem[1];
                src.Ids[0] = new SrcList.SrcListItem();
                src.Ids[0].Name = table.Name;
                src.Ids[0].Table = table;
                src.Ids[0].Cursor = -1;
                nc.Parse = this;
                nc.SrcList = src;
                nc.NCFlags = NC.IsCheck;
                ExprList list = table.Check; // List of all CHECK constraints
                for (int i = 0; i < list.Exprs; i++)
                    if (Walker.ResolveExprNames(nc, ref list.Ids[i].Expr))
                        return;
            }
#endif

            // If the db->init.busy is 1 it means we are reading the SQL off the "sqlite_master" or "sqlite_temp_master" table on the disk.
            // So do not write to the disk again.  Extract the root page number for the table from the db->init.newTnum field.  (The page number
            // should have been put there by the sqliteOpenCb routine.)
            if (ctx.Init.Busy)
                table.Id = ctx.Init.NewTid;

            // If not initializing, then create a record for the new table in the SQLITE_MASTER table of the database.
            // If this is a TEMPORARY table, write the entry into the auxiliary file instead of into the main database file.
            if (!ctx.Init.Busy)
            {
                Vdbe v = GetVdbe();
                if (C._NEVER(v == null))
                    return;
                v.AddOp1(OP.Close, 0);

                // Initialize zType for the new view or table.
                string type; // "view" or "table"
                string type2; // "VIEW" or "TABLE
                if (table.Select == null)
                {
                    // A regular table
                    type = "table";
                    type2 = "TABLE";
                }
#if !OMIT_VIEW
                else
                {
                    // A view
                    type = "view";
                    type2 = "VIEW";
                }
#endif

                // If this is a CREATE TABLE xx AS SELECT ..., execute the SELECT statement to populate the new table. The root-page number for the
                // new table is in register pParse->regRoot.
                //
                // Once the SELECT has been coded by sqlite3Select(), it is in a suitable state to query for the column names and types to be used
                // by the new table.
                //
                // A shared-cache write-lock is not required to write to the new table, as a schema-lock must have already been obtained to create it. Since
                // a schema-lock excludes all other database users, the write-lock would be redundant.
                if (select != null)
                {
                    Debug.Assert(Tabs == 1);
                    v.AddOp3(OP.OpenWrite, 1, RegRoot, ctx);
                    v.ChangeP5(1);
                    Tabs = 2;
                    SelectDest dest = new SelectDest();
                    Select.DestInit(dest, SRT.Table, 1);
                    Select.Select_(this, select, ref dest);
                    v.AddOp1(OP.Close, 1);
                    if (Errs == 0)
                    {
                        Table selectTable = sqlite3ResultSetOfSelect(parse, select);
                        if (selectTable == null)
                            return;
                        Debug.Assert(table.Cols.data == null);
                        table.Cols.length = selectTable.Cols.length;
                        table.Cols = selectTable.Cols;
                        selectTable.Cols.length = 0;
                        selectTable.Cols.data = null;
                        DeleteTable(ctx, ref selectTable);
                    }
                }

                // Compute the complete text of the CREATE statement
                int n;
                string stmt; // Text of the CREATE TABLE or CREATE VIEW statement
                if (select != null)
                    stmt = CreateTableStmt(ctx, table);
                else
                {
                    n = (int)(NameToken.data.Length - end.data.Length) + 1;
                    stmt = C._mtagprintf(ctx, "CREATE %s %.*s", type2, n, NameToken.data);
                }

                // A slot for the record has already been allocated in the SQLITE_MASTER table.  We just need to update that slot with all
                // the information we've collected.
                object[] args = new object[]{ ctx.DBs[db].Name, SCHEMA_TABLE(db),
				    type, table.Name, table.Name, RegRoot, stmt,
				    RegRowid };
                NestedParse(pParse,
                    "UPDATE %Q.%s " +
                    "SET type='%s', name=%Q, tbl_name=%Q, rootpage=#%d, sql=%Q " +
                    "WHERE rowid=#%d", args);
                C._tagfree(ctx, ref stmt);
                ChangeCookie(ctx);

#if !OMIT_AUTOINCREMENT
                // Check to see if we need to create an sqlite_sequence table for keeping track of autoincrement keys.
                if ((table.TabFlags & TF.Autoincrement) != 0)
                {
                    Context.DB dbobj = ctx.DBs[db];
                    Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                    if (dbobj.Schema.SeqTable == null)
                    {
                        object[] args2 = new[] { dbobj.Name };
                        NestedParse("CREATE TABLE %Q.sqlite_sequence(name,seq)", args2);
                    }
                }
#endif

                // Reparse everything to update our internal data structures
                v.AddParseSchemaOp(ctx, C._mtagprintf(ctx, "tbl_name='%q'", table.Name));
            }


            // Add the table to the in-memory representation of the database.
            if (ctx.Init.Busy)
            {
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                Schema schema = table.Schema;
                Table oldTable = schema.TableHash.Insert(table.Name, table.Name.Length, table);
                if (oldTable != null)
                {
                    Debug.Assert(table == oldTable); // Malloc must have failed inside HashInsert()
                    ctx.MallocFailed = true;
                    return;
                }
                NewTable = null;
                ctx.Flags |= Context.FLAG.InternChanges;
#if !OMIT_ALTERTABLE
                if (table.Select == null)
                {
                    string name = NameToken.data;
                    Debug.Assert(select == null && cons != null && end != null);
                    if (cons.data == null)
                        cons = end;
                    int nameLength = name.Length - cons.data.Length;
                    table.AddColOffset = 13 + nameLength;
                }
#endif
            }
        }

#if !OMIT_VIEW
        public void CreateView(Token begin, Token name1, Token name2, Select select, bool isTemp, bool noErr)
        {
            Context ctx = Ctx;
            if (VarsSeen > 0)
            {
                ErrorMsg("parameters are not allowed in views");
                Select.Delete(ctx, ref select);
                return;
            }
            StartTable(name1, name2, isTemp, 1, 0, noErr);
            Table table = NewTable;
            if (table == null || Errs != 0)
            {
                Select.Delete(ctx, ref select);
                return;
            }
            Token name = null;
            TwoPartName(name1, name2, ref name);
            int db = SchemaToIndex(ctx, table.Schema);
            DbFixer fix = new DbFixer();
            if (FixInit(fix, this, db, "view", name) != 0 && FixSelect(fix, select) != 0)
            {
                Select.Delete(ctx, ref select);
                return;
            }

            // Make a copy of the entire SELECT statement that defines the view. This will force all the Expr.token.z values to be dynamically
            // allocated rather than point to the input string - which means that they will persist after the current sqlite3_exec() call returns.
            table.Select = Select.Dup(ctx, select, EXPRDUP.REDUCE);
            Select.Delete(ctx, ref select);
            if (ctx.MallocFailed)
                return;
            if (ctx.Init.Busy)
                ViewGetColumnNames(table);

            // Locate the end of the CREATE VIEW statement.  Make sEnd point to the end.
            Token end = LastToken;
            if (C._ALWAYS(end.data[0] != 0) && end.data[0] != ';')
                end.data = end.data.Substring(end.length);
            end.length = 0;

            int n = (int)(begin.data.Length - end.data.Length);
            string z = begin.data;
            while (C._ALWAYS(n > 0) && char.IsWhiteSpace(z[n - 1])) { n--; }
            end.data = z.Substring(n - 1);
            end.length = 1;

            // Use sqlite3EndTable() to add the view to the SQLITE_MASTER table
            EndTable(null, end, null);
            return;
        }
#else
        //public void CreateView(Token begin, Token name1, Token name2, Select select, bool isTemp, bool noErr) { }
#endif

#if !OMIT_VIEW || !OMIT_VIRTUALTABLE

        public int ViewGetColumnNames(Table table)
        {
            Table pSelTab;

            Debug.Assert(table != null);
            Context ctx = Ctx; // Database connection for malloc errors
#if !OMIT_VIRTUALTABLE
            if (VTable.CallConnect(parse, table) != 0)
                return (int)RC.ERROR;
            if (E.IsVirtual(table))
                return 0;
#endif
#if !OMIT_VIEW
            // A positive nCol means the columns names for this view are already known.
            if (table.Cols.length > 0)
                return 0;

            // A negative nCol is a special marker meaning that we are currently trying to compute the column names.  If we enter this routine with
            // a negative nCol, it means two or more views form a loop, like this:
            //     CREATE VIEW one AS SELECT * FROM two;
            //     CREATE VIEW two AS SELECT * FROM one;
            // Actually, the error above is now caught prior to reaching this point. But the following test is still important as it does come up
            // in the following:
            //     CREATE TABLE main.ex1(a);
            //     CREATE TEMP VIEW ex1 AS SELECT a FROM ex1;
            //     SELECT * FROM temp.ex1;
            if (table.Cols.length < 0)
            {
                ErrorMsg("view %s is circularly defined", table.Name);
                return 1;
            }
            Debug.Assert(table.Cols.length >= 0);

            // If we get this far, it means we need to compute the table names. Note that the call to sqlite3ResultSetOfSelect() will expand any
            // "*" elements in the results set of the view and will assign cursors to the elements of the FROM clause.  But we do not want these changes
            // to be permanent.  So the computation is done on a copy of the SELECT statement that defines the view.
            Debug.Assert(table.Select != null);
            int errs = 0; // Number of errors encountered
            Select select = Select.Dup(ctx, table.Select, 0); // Copy of the SELECT that implements the view
            if (select != null)
            {
                bool enableLookaside = ctx.Lookaside.Enabled;
                int n = Tabs; // Temporarily holds the number of cursors assigned
                SrcListAssignCursors(select.Src);
                table.Cols.length = -1;
                ctx.Lookaside.Enabled = false;
#if !OMIT_AUTHORIZATION
                object auth = ctx.Auth;
                ctx.Auth = 0;
                Table selectTable = ResultSetOfSelect(select); // A fake table from which we get the result set
                ctx.Auth = auth;
#else
                Table selectTable = ResultSetOfSelect(select); // A fake table from which we get the result set
#endif
                ctx.Lookaside.Enabled = enableLookaside;
                Tabs = n;
                if (selectTable != null)
                {
                    Debug.Assert(table.Cols.data == null);
                    table.Cols.length = selectTable.Cols.length;
                    table.Cols = selectTable.Cols;
                    selectTable.Cols.length = 0;
                    selectTable.Cols.data = null;
                    DeleteTable(ctx, ref selectTable);
                    Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, table.Schema));
                    table.Schema.Flags |= SCHEMA.UnresetViews;
                }
                else
                {
                    table.Cols.length = 0;
                    errs++;
                }
                Select.Delete(ctx, ref select);
            }
            else
                errs++;
#endif
            return errs;
        }
#endif

#if !OMIT_VIEW
        public static void ViewResetAll(Context ctx, int db)
        {
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            if (!E.DbHasProperty(ctx, db, SCHEMA.UnresetViews))
                return;
            for (HashElem i = ctx.DBs[db].Schema.TableHash.First; i != null; i = i.Next)
            {
                Table table = (Table)i.Data;
                if (table.Select != null)
                {
                    DeleteColumnNames(ctx, table);
                    table.Cols.data = null;
                    table.Cols.length = 0;

                }
            }
            E.DbClearProperty(ctx, db, SCHEMA.UnresetViews);
        }
#else
        public static void ViewResetAll(Context ctx, int db) { }
#endif

#if !OMIT_AUTOVACUUM
        public static void RootPageMoved(Context ctx, int db, int from, int to)
        {
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            Context.DB dbobj = ctx.DBs[db];
            Hash hash = dbobj.Schema.TableHash;
            for (HashElem elem = hash.First; elem != null; elem = elem.Next)
            {
                Table table = (Table)elem.Data;
                if (table.Id == from)
                    table.Id = to;
            }
            hash = dbobj.Schema.IndexHash;
            for (HashElem elem = hash.First; elem != null; elem = elem.Next)
            {
                Index index = (Index)elem.Data;
                if (index.Id == from)
                    index.Id = to;
            }
        }
#endif

        static void DestroyRootPage(Parse parse, int table, int db)
        {
            Vdbe v = parse.GetVdbe();
            int r1 = parse.GetTempReg();
            v.AddOp3(v, OP.Destroy, table, r1, db);
            parse.MayAbort();
#if !OMIT_AUTOVACUUM
            // OP_Destroy stores an in integer r1. If this integer is non-zero, then it is the root page number of a table moved to
            // location iTable. The following code modifies the sqlite_master table to reflect this.
            //
            // The "#NNN" in the SQL is a special constant that means whatever value is in register NNN.  See grammar rules associated with the TK_REGISTER
            // token for additional information.
            object[] args = new[] { parse.Ctx.DBs[db].Name, E.SCHEMA_TABLE(db), table, r1, r1 };
            parse.NestedParse("UPDATE %Q.%s SET rootpage=%d WHERE #%d AND rootpage=#%d", args);
#endif
            parse.ReleaseTempReg(r1);
        }

        static void DestroyTable(Parse parse, Table table)
        {
#if OMIT_AUTOVACUUM
            pIdx;
            int db = SchemaToIndex(parse.Ctx, table.Schema);
            DestroyRootPage(parse, table.Id, db);
            for (Index index = table.Index; index != null; index = index.Next)
                DestroyRootPage(parse, index.Id, db);
#else
            // If the database may be auto-vacuum capable (if SQLITE_OMIT_AUTOVACUUM is not defined), then it is important to call OP_Destroy on the
            // table and index root-pages in order, starting with the numerically largest root-page number. This guarantees that none of the root-pages
            // to be destroyed is relocated by an earlier OP_Destroy. i.e. if the following were coded:
            //
            // OP_Destroy 4 0
            // ...
            // OP_Destroy 5 0
            //
            // and root page 5 happened to be the largest root-page number in the database, then root page 5 would be moved to page 4 by the 
            // "OP_Destroy 4 0" opcode. The subsequent "OP_Destroy 5 0" would hit a free-list page.
            int tableId = table.tnum;
            int destroyedId = 0;
            while (true)
            {
                int largestId = 0;
                if (destroyedId == 0 || tableId < destroyedId)
                    largestId = tableId;
                for (Index index = table.Index; index != null; index = index.Next)
                {
                    int indexId = index.Id;
                    Debug.Assert(index.Schema == table.Schema);
                    if ((destroyedId == 0 || (indexId < destroyedId)) && indexId > largestId)
                        largestId = indexId;
                }
                if (largestId == 0)
                    return;
                else
                {
                    int db = SchemaToIndex(parse.db, table.Schema);
                    Debug.Assert(db >= 0 && db < parse.Ctx.DBs.length);
                    DestroyRootPage(parse, largestId, db);
                    destroyedId = largestId;
                }
            }
#endif
        }

        public void ClearStatTables(int db, string type, string name)
        {
            string dbName = Ctx.DBs[db].Name;
            for (int i = 1; i <= 3; i++)
            {
                string tableName = "sqlite_stat" + i;
                if (FindTable(Ctx, tableName, dbName))
                {
                    object[] args = new { dbName, tableName, type, name };
                    NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", args);
                }
            }
        }

        public void CodeDropTable(Table table, int db, bool isView)
        {
            Context ctx = Ctx;
            Context.DB dbobj = ctx.DBs[db];

            Vdbe v = GetVdbe();
            Debug.Assert(v != null);
            BeginWriteOperation(1, db);

#if !OMIT_VIRTUALTABLE
            if (IsVirtual(table))
                v.AddOp0(OP.VBegin);
#endif

            // Drop all triggers associated with the table being dropped. Code is generated to remove entries from sqlite_master and/or
            // sqlite_temp_master if required.
            Trigger trigger = sqlite3TriggerList(this, table);
            while (trigger != null)
            {
                Debug.Assert(trigger.Schema == table.Schema || trigger.Schema == ctx.DBs[1].Schema);
                sqlite3DropTriggerPtr(this, trigger);
                trigger = trigger->Next;
            }

#if !OMIT_AUTOINCREMENT
            // Remove any entries of the sqlite_sequence table associated with the table being dropped. This is done before the table is dropped
            // at the btree level, in case the sqlite_sequence table needs to move as a result of the drop (can happen in auto-vacuum mode).
            if ((table.TabFlags & TF_Autoincrement) != 0)
            {
                object[] args = new[] { dbobj.Name, table.Name };
                NestedParse("DELETE FROM %Q.sqlite_sequence WHERE name=%Q", args);
            }
#endif

            // Drop all SQLITE_MASTER table and index entries that refer to the table. The program name loops through the master table and deletes
            // every row that refers to a table of the same name as the one being dropped. Triggers are handled separately because a trigger can be
            // created in the temp database that refers to a table in another database.
            object[] args2 = new[] { dbobj.Name, E.SCHEMA_TABLE(db), table.Name };
            NestedParse("DELETE FROM %Q.%s WHERE tbl_name=%Q and type!='trigger'", args2);
            if (!isView && !IsVirtual(table))
                DestroyTable(this, table);

            // Remove the table entry from SQLite's internal schema and modify the schema cookie.
            if (IsVirtual(table))
                v.AddOp4(OP.VDestroy, db, 0, 0, table.Name, 0);
            v.AddOp4(OP.DropTable, db, 0, 0, table.Name, 0);
            ChangeCookie(db);
            ViewResetAll(ctx, db);
        }

        public void DropTable(SrcList name, bool isView, bool noErr)
        {
            Context ctx = Ctx;
            if (ctx.MallocFailed)
                goto exit_drop_table;

            Debug.Assert(Errs == 0);
            Debug.Assert(name.Srcs == 1);
            if (noErr)
                ctx.SuppressErr++;
            Table table = LocateTableItem(isView, name.Ids[0]);
            if (noErr)
                ctx.suppressErr--;

            if (table == null)
            {
                if (noErr)
                    CodeVerifyNamedSchema(this, name.Ids[0].Database);
                goto exit_drop_table;
            }
            int db = SchemaToIndex(ctx, table.Schema);
            Debug.Assert(db >= 0 && db < ctx.DBs.length);

            // If pTab is a virtual table, call ViewGetColumnNames() to ensure it is initialized.
            if (IsVirtual(table) && ViewGetColumnNames(table) != 0)
                goto exit_drop_table;
#if !OMIT_AUTHORIZATION
            {
                string tableName = E.SCHEMA_TABLE(db);
                string dbName = ctx.aDb[db].zName;
                if (Auth.Check(this, AUTH.DELETE, tableName, 0, dbName))
                    goto exit_drop_table;
                AUTH code;
                string arg2;
                if (isView)
                {
                    code = (!E.OMIT_TEMPDB && db == 1 ? AUTH.DROP_TEMP_VIEW : AUTH.DROP_VIEW);
                    arg2 = 0;
                }
#if !OMIT_VIRTUALTABLE
                else if (IsVirtual(table))
                {
                    code = AUTH.DROP_VTABLE;
                    arg2 = VTable.GetVTable(ctx, table)->Mod->Name;
                }
#endif
                else
                {
                    code = (!E.OMIT_TEMPDB && db == 1 ? AUTH.DROP_TEMP_TABLE : AUTH.DROP_TABLE);
                    arg2 = 0;
                }
                if (Auth.Check(this, code, table.Name, arg2, dbName) || Auth.Check(this, AUTH.DELETE, table.Name, null, dbName))
                    goto exit_drop_table;
            }
#endif
            if (table.Name.StartsWith("sqlite_", StringComparison.OrdinalIgnoreCase))
            {
                ErrorMsg("table %s may not be dropped", table.Name);
                goto exit_drop_table;
            }

#if !OMIT_VIEW
            // Ensure DROP TABLE is not used on a view, and DROP VIEW is not used on a table.
            if (isView && table.Select == null)
            {
                ErrorMsg("use DROP TABLE to delete table %s", table.Name);
                goto exit_drop_table;
            }
            if (!isView && table.Select != null)
            {
                ErrorMsg("use DROP VIEW to delete view %s", table.Name);
                goto exit_drop_table;
            }
#endif

            // Generate code to remove the table from the master table on disk.
            Vdbe v = GetVdbe();
            if (v != null)
            {
                BeginWriteOperation(1, db);
                ClearStatTables(db, "tbl", table->Name);
                FkDropTable(name, table);
                CodeDropTable(table, db, isView);
            }

        exit_drop_table:
            SrcListDelete(ctx, ref name);
        }


        // OVERLOADS, so I don't need to rewrite parse.c
        public void CreateForeignKey(int null_2, Token to, ExprList toCol, int flags) { CreateForeignKey(null, to, toCol, flags); }
        public void CreateForeignKey(ExprList fromCol, Token to, ExprList toCol, int flags)
        {
            Context ctx = Ctx;
#if !OMIT_FOREIGN_KEY
            FKey nextTo;
            Table table = NewTable;
            int i;

            Debug.Assert(to != null);
            FKey fkey = null;
            if (table == null || E.INDECLARE_VTABLE(this))
                goto fk_end;
            int cols;
            if (fromCol == null)
            {
                int col = table.Cols.length - 1;
                if (C._NEVER(col < 0))
                    goto fk_end;
                if (toCol != null && toCol.Exprs != 1)
                {
                    ErrorMsg("foreign key on %s should reference only one column of table %T", table.Cols[col].Name, to);
                    goto fk_end;
                }
                cols = 1;
            }
            else if (toCol != null && toCol.Exprs != fromCol.Exprs)
            {
                ErrorMsg("number of columns in foreign key does not match the number of columns in the referenced table");
                goto fk_end;
            }
            else
                cols = fromCol.Exprs;
            //int bytes = sizeof(FKey) + (cols-1)*sizeof(fkey.Cols[0]) + to.length + 1;
            //if (toCol != null)
            //    for (i = 0; i < toCol.Exprs; i++)
            //        bytes += toCol->Ids[i].Name.Length + 1;
            fkey = new FKey();
            if (fkey == null)
                goto fk_end;
            fkey.From = table;
            fkey.NextFrom = table.FKey;
            fkey.Cols = new FKey.ColMap[cols];
            fkey.Cols[0] = new FKey.ColMap();
            fkey.To = to.data.Substring(0, to.length);
            Parse.Dequote(ref fkey.To);
            fkey.Cols = cols;
            if (fromCol == null)
                fkey.Cols[0].From = table.Cols.length - 1;
            else
            {
                for (i = 0; i < nCol; i++)
                {
                    if (fkey.Cols[i] == null)
                        fkey.Cols[i] = new FKey.ColMap();
                    int j;
                    for (j = 0; j < table.nCol; j++)
                    {
                        if (string.Equals(table.Cols[j].Name, fromCol.Ids[i].Name, StringComparison.OrdinalIgnoreCase))
                        {
                            fkey.Cols[i].From = j;
                            break;
                        }
                    }
                    if (j >= table.Cols.length)
                    {
                        ErrorMsg("unknown column \"%s\" in foreign key definition", fromCol.Ids[i].Name);
                        goto fk_end;
                    }
                }
            }
            if (toCol != null)
            {
                for (i = 0; i < nCol; i++)
                {
                    if (fkey.Cols[i] == null)
                        fkey.Cols[i] = new FKey.ColMap();
                    fkey.Cols[i].Col = toCol.Ids[i].Name;
                }
            }
            fkey.IsDeferred = false;
            fkey.Actions[0] = (byte)(flags & 0xff);             // ON DELETE action
            fkey.Actions[1] = (byte)((flags >> 8) & 0xff);      // ON UPDATE action

            Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, table.Schema));
            nextTo = table.Schema.FKeyHash.Insert(fkey.To, fkey.To.Length, fkey);
            if (nextTo == fkey)
            {
                ctx.MallocFailed = true;
                goto fk_end;
            }
            if (nextTo != null)
            {
                Debug.Assert(nextTo.PrevTo == null);
                fkey.NextTo = nextTo;
                nextTo.PrevTo = fkey;
            }

            // Link the foreign key to the table as the last step.
            table.FKeys = fkey;
            fkey = null;

        fk_end:
            C._tagfree(ctx, ref fkey);
#endif
            Expr.ListDelete(ctx, ref fromCol);
            Expr.ListDelete(ctx, ref toCol);
        }

        public void DeferForeignKey(bool isDeferred)
        {
#if !OMIT_FOREIGN_KEY
            Table table;
            FKey fkey;
            if ((table = NewTable) == null || (fkey = table.FKeys) == null) return;
            Debug.Assert(isDeferred == 0 || isDeferred == 1); /* EV: R-30323-21917 */
            fkey.IsDeferred = isDeferred;
#endif
        }

        public void RefillIndex(Index index, int memRootPage)
        {
            Context ctx = Ctx; // The database connection
            int db = SchemaToIndex(ctx, index.pSchema);

#if !OMIT_AUTHORIZATION
            if (Auth.Check(this, AUTH.REINDEX, index.Name, 0, ctx.DBs[db].Name))
                return;
#endif

            // Require a write-lock on the table to perform this operation
            Table table = index.Table; // The table that is indexed
            TableLock(db, table.Id, 1, table.Name);
            Vdbe v = GetVdbe(); // Generate code into this virtual machine
            if (v == null) return;
            int tid; // Root page of index
            if (memRootPage >= 0)
                tid = memRootPage;
            else
            {
                tid = index.Id;
                v.AddOp2(OP.Clear, tid, db);
            }
            int indexIdx = Tabs++; // Btree cursor used for pIndex
            KeyInfo key = IndexKeyinfo(index); // KeyInfo for index
            v.AddOp4(OP.OpenWrite, indexIdx, tid, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
            v.ChangeP5(Vdbe.OPFLAG.BULKCSR | (memRootPage >= 0 ? Vdbe.OPFLAG.P2ISREG : 0));
            sqlite3OpenTable(pParse, tableIdx, db, table, OP_OpenRead);

            // Open the sorter cursor if we are to use one.
            int sorterIdx = Tabs++; // Cursor opened by OpenSorter (if in use)
            v.AddOp4(OP.SorterOpen, sorterIdx, 0, 0, key, Vdbe.P4T.KEYINFO);

            // Open the table. Loop through all rows of the table, inserting index records into the sorter.
            int tableIdx = Tabs++; // Btree cursor used for pTab
            OpenTable(tableIdx, db, table, OP.OpenRead);
            int addr1 = v.AddOp2(OP.Rewind, tableIdx, 0); // Address of top of loop
            int regRecord = GetTempReg(); // Register holding assemblied index record

            GenerateIndexKey(index, tableIdx, regRecord, 1);
            v.AddOp2(OP.SorterInsert, sorterIdx, regRecord);
            v.AddOp2(OP.Next, tableIdx, addr1 + 1);
            v.JumpHere(addr1);
            addr1 = v.AddOp2(OP_SorterSort, sorterIdx, 0);
            int addr2; // Address to jump to for next iteration
            if (indexIdx->OnError != OE.None)
            {
                int j2 = v.CurrentAddr() + 3;
                v.AddOp2(OP.Goto, 0, j2);
                addr2 = v.CurrentAddr();
                v.AddOp3(OP.SorterCompare, sorterIdx, j2, regRecord);
                HaltConstraint(SQLITE_CONSTRAINT_UNIQUE, OE_Abort, "indexed columns are not unique", Vdbe.P4T.STATIC);
            }
            else
                addr2 = v.CurrentAddr();

            v.AddOp2(OP.SorterData, sorterIdx, regRecord);
            v.AddOp3(OP.IdxInsert, indexIdx, regRecord, 1);
            v.ChangeP5(OPFLAG_USESEEKRESULT);
            ReleaseTempReg(regRecord);
            v.AddOp2(OP.Next, tableIdx, addr1 + 1);
            v.JumpHere(addr1);
            v.AddOp1(OP.Close, tableIdx);
            v.AddOp1(OP.Close, indexIdx);
            v.AddOp1(OP.Close, sorterIdx);
        }


        // OVERLOADS, so I don't need to rewrite parse.c
        //public Index CreateIndex(int null_2, int null_3, int null_4, int null_5, OE onError, int null_7, int null_8, int sortOrder, bool ifNotExist) { return CreateIndex(null, null, null, null, onError, null, null, sortOrder, ifNotExist); }
        //public Index CreateIndex(int null_2, int null_3, int null_4, ExprList list, OE onError, int null_7, int null_8, int sortOrder, bool ifNotExist) { return CreateIndex(null, null, null, list, onError, null, null, sortOrder, ifNotExist); }
        public Index CreateIndex(Token name1, Token name2, SrcList tableName, ExprList list, OE onError, Token start, Token end, int sortOrder, bool ifNotExist)
        {
            Debug.Assert(start == null || end != null); // pEnd must be non-NULL if pStart is
            Debug.Assert(Errs == 0); // Never called with prior errors
            Index r = null; // Pointer to return
            Context ctx = Ctx;
            if (ctx.MallocFailed || E.INDECLARE_VTABLE(this) || ReadSchema(this) != RC.OK)
                goto exit_create_index;

            // Find the table that is to be indexed.  Return early if not found.
            int db; // Index of the database that is being written
            Table table = null; // Table to be indexed
            DbFixer fix = new DbFixer(); // For assigning database names to pTable
            Token nameAsToken = null; // Unqualified name of the index to create
            if (tableName != null)
            {
                // Use the two-part index name to determine the database to search for the table. 'Fix' the table name to this db
                // before looking up the table.
                Debug.Assert(name1 != null && name2 != null);
                db = TwoPartName(name1, name2, ref nameAsToken);
                if (db < 0) goto exit_create_index;
                Debug.Assert(nameAsToken != null && nameAsToken.data != null);
#if !OMIT_TEMPDB
                // If the index name was unqualified, check if the table is a temp table. If so, set the database to 1. Do not do this
                // if initialising a database schema.
                if (!ctx.Init.Busy)
                {
                    table = SrcListLookup(this, tableName);
                    if (name2.length == 0 && table != null && table.Schema == ctx.DBs[1].Schema)
                        db = 1;
                }
#endif

                // Because the parser constructs pTblName from a single identifier, sqlite3FixSrcList can never fail.
                if (Attach.FixInit(fix, this, db, "index", nameAsToken) != 0 && Attach.FixSrcList(fix, tableName) != 0)
                    Debugger.Break();
                table = LocateTableItem(false, tableName.Ids[0]);
                Debug.Assert(!ctx->MallocFailed || table == null);
                if (table == null) goto exit_create_index;
                Debug.Assert(ctx.Dbs[db].Schema == table.Schema);
            }
            else
            {
                Debug.Assert(nameAsToken == null);
                Debug.Assert(start == null);
                table = NewTable;
                if (table == null) goto exit_create_index;
                db = SchemaToIndex(ctx, table.Schema);
            }
            Context.DB dbobj = ctx.DBs[db]; // The specific table containing the indexed database

            Debug.Assert(table != null);
            Debug.Assert(Errs == 0);
            if (table.Name.StartsWith("sqlite_", StringComparison.OrdinalIgnoreCase) && !table.Name.StartsWith("altertab_", StringComparison.OrdinalIgnoreCase))
            {
                ErrorMsg("table %s may not be indexed", table.Name);
                goto exit_create_index;
            }
#if !OMIT_VIEW
            if (table.Select != null)
            {
                ErrorMsg("views may not be indexed");
                goto exit_create_index;
            }
#endif
            if (IsVirtual(table))
            {
                ErrorMsg("virtual tables may not be indexed");
                goto exit_create_index;
            }

            // Find the name of the index.  Make sure there is not already another index or table with the same name.  
            //
            // Exception:  If we are reading the names of permanent indices from the sqlite_master table (because some other process changed the schema) and
            // one of the index names collides with the name of a temporary table or index, then we will continue to process this index.
            //
            // If pName==0 it means that we are dealing with a primary key or UNIQUE constraint.  We have to invent our own name.
            string name = null; // Name of the index
            if (nameAsToken != null)
            {
                name = NameFromToken(ctx, nameAsToken);
                if (name == null) goto exit_create_index;
                Debug.Assert(nameAsToken.data != null);
                if (CheckObjectName(name) != RC.OK)
                    goto exit_create_index;
                if (!ctx.Init.Busy)
                {
                    if (FindTable(ctx, name, null) != null)
                    {
                        ErrorMsg("there is already a table named %s", name);
                        goto exit_create_index;
                    }
                }
                if (FindIndex(ctx, name, dbobj.Name) != null)
                {
                    if (ifNotExist == 0)
                        sqlite3ErrorMsg(pParse, "index %s already exists", name);
                    else
                    {
                        Debug.Assert(!ctx.Init.Busy);
                        CodeVerifySchema(db);
                    }
                    goto exit_create_index;
                }
            }
            else
            {
                int n = 0;
                Index loop;
                for (loop = table.Index, n = 1; loop != null; loop = loop.Next, n++) { }
                name = SysEx.Mprintf(ctx, "sqlite_autoindex_%s_%d", table.Name, n);
                if (name == null)
                    goto exit_create_index;
            }

            // Check for authorization to create an index.
            int i;
#if !OMIT_AUTHORIZATION
            {
                string dbName = dbobj.Name;
                if (Auth.Check(this, AUTH.INSERT, E.SCHEMA_TABLE(db), 0, dbName))
                    goto exit_create_index;
                i = (!OMIT_TEMPDB2 && db == 1 ? AUTH.CREATE_TEMP_INDEX : AUTH.REATE_INDEX);
                if (Auth.Check(this, i, name, table.Name, dbName))
                    goto exit_create_index;
            }
#endif

            // If pList==0, it means this routine was called to make a primary key out of the last column added to the table under construction.
            // So create a fake list to simulate this.
            if (list == null)
            {
                Token nullId = new Token(); // Fake token for an empty ID list
                nullId.data = table.aCol[table.Cols.length - 1].Name;
                nullId.length = nullId.data.Length;
                list = ExprList.Append(this, null, null);
                if (list == null)
                    goto exit_create_index;
                ExprList.SetName(this, list, nullId, 0);
                list.Ids[0].SortOrder = (byte)sortOrder;
            }

            Index pRet = null;            // Pointer to return
            int sortOrderMask;
            ExprList_item pListItem;      // For looping over pList
            StringBuilder extra = new StringBuilder();

            // Figure out how many bytes of space are required to store explicitly specified collation sequence names.
            int extraLength = 0;
            for (i = 0; i < list.Exprs; i++)
            {
                Expr expr = list.Ids[i].Expr;
                if (expr != null)
                {
                    CollSeq coll = Expr.CollSeq(this, expr); ;
                    if (coll != null)
                        extraLength += (1 + coll.Name.Length);
                }
            }

            // Allocate the index structure.
            int nameLength = name.Length; // Number of characters in zName
            int cols = list.Exprs;
            Index index = new Index(); // The index to be created
            if (ctx.MallocFailed)
                goto exit_create_index;
            StringBuilder extra = new StringBuilder(nameLength + 1);
            index.RowEsts = new int[cols + 1];
            index.CollNames = new string[cols + 1];
            index.Columns = new int[cols + 1];//(int )(pIndex->azColl[nCol]);
            index.SortOrders = new byte[cols + 1];
            index.Name = name;
            index.Table = table;
            index.Columns.length = list.Exprs;
            index.onError = (OE)onError;
            index.AutoIndex = (byte)(nameAsToken == null ? 1 : 0);
            index.Schema = ctx.DBs[db].Schema;
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));

            // Check to see if we should honor DESC requests on index columns
            int sortOrderMask = (dbobj.Schema.FileFormat >= 4 ? 1 : 0); // Honor/Ignore DESC - 1 to honor DESC in index.  0 to ignore.

            // Scan the names of the columns of the table to be indexed and load the column indices into the Index structure.  Report an error
            // if any column is not found.
            //
            // TODO:  Add a test to make sure that the same column is not named more than once within the same index.  Only the first instance of
            // the column will ever be used by the optimizer.  Note that using the same column more than once cannot be an error because that would 
            // break backwards compatibility - it needs to be a warning.
            int j;
            ExprList.ExprListItem listItem; // For looping over pList
            for (i = 0, listItem = list.Ids[0]; i < list.Exprs; i++, listItem = list.Ids[i])
            {
                string colName = pListItem.zName;
                Column tableCol;
                for (j = 0, tableCol = table.Cols[0]; j < table.nCol; j++, tableCol = table.Cols[j])
                    if (string.Equals(colName, tableCol.zName, StringComparison.OrdinalIgnoreCase))
                        break;
                if (j >= table.Cols.length)
                {
                    ErrorMsg("table %s has no column named %s", table.Name, colName);
                    CheckSchema = 1;
                    goto exit_create_index;
                }
                index.Columns[i] = j;
                CollSeq coll; // Collating sequence
                string collName; // Collation sequence name
                if (listItem.Expr != null && (coll = Expr.CollSeq(this, listItem.Expr)) != null)
                {
                    collName = listItem.Expr.Coll.Name;
                    int collNameLength = collName.Length;
                    Debug.Assert(extraLength >= collNameLength);
                    extra = new StringBuilder(collName.Substring(0, collNameLength));
                    collName = extra.ToString();
                    extraLength -= collNameLength;
                }
                else
                {
                    collName = table.Cols[j].Coll;
                    if (collName == null)
                        collName = "BINARY";
                }
                if (!ctx.Init.Busy && LocateCollSeq(collName) == null)
                    goto exit_create_index;
                index.CollNames[i] = collName;
                byte requestedSortOrder = (u8)((pListItem.sortOrder & sortOrderMask) != 0 ? 1 : 0);
                index.SortOrders[i] = (byte)requestedSortOrder;
            }
            DefaultRowEst(index);

            if (table == NewTable)
            {
                // This routine has been called to create an automatic index as a result of a PRIMARY KEY or UNIQUE clause on a column definition, or
                // a PRIMARY KEY or UNIQUE clause following the column definitions. i.e. one of:
                //
                // CREATE TABLE t(x PRIMARY KEY, y);
                // CREATE TABLE t(x, y, UNIQUE(x, y));
                //
                // Either way, check to see if the table already has such an index. If so, don't bother creating this one. This only applies to
                // automatically created indices. Users can do as they wish with explicit indices.
                //
                // Two UNIQUE or PRIMARY KEY constraints are considered equivalent (and thus suppressing the second one) even if they have different sort orders.
                //
                // If there are different collating sequences or if the columns of the constraint occur in different orders, then the constraints are
                // considered distinct and both result in separate indices.
                for (Index index2 = table.Index; index2 != null; index2 = index2.pNext)
                {
                    Debug.Assert(index2.onError != OE.None);
                    Debug.Assert(index2.autoIndex != 0);
                    Debug.Assert(index.onError != OE.None);
                    if (index2.Columns.length != index.Columns.length) continue;
                    int k;
                    for (k = 0; k < index2.nColumn; k++)
                    {
                        if (index2.Columns[k] != index.Columns[k]) break;
                        string z1 = index2.azColl[k];
                        string z2 = index.azColl[k];
                        if (z1 != z2 && !string.Equals(z1, z2, StringComparison.OrdinalIgnoreCase)) break;
                    }
                    if (k == index2.nColumn)
                    {
                        if (index2.onError != index.onError)
                        {
                            // This constraint creates the same index as a previous constraint specified somewhere in the CREATE TABLE statement.
                            // However the ON CONFLICT clauses are different. If both this constraint and the previous equivalent constraint have explicit
                            // ON CONFLICT clauses this is an error. Otherwise, use the explicitly specified behavior for the index.
                            if (!(index2.onError == OE_Default || index.onError == OE_Default))
                                ErrorMsg("conflicting ON CONFLICT clauses specified");
                            if (index2.OnError == OE.Default)
                                index2.OnError = index.OnError;
                        }
                        goto exit_create_index;
                    }
                }
            }

            // Link the new Index structure to its table and to the other in-memory database structures.
            if (!ctx.Init.Busy)
            {
                Debug.Assert(sqlite3SchemaMutexHeld(ctx, 0, index.pSchema));
                Index p = index.Schema.IndexHash.Insert(index.Name, index.Name.Length, index);
                if (p != null)
                {
                    Debug.Assert(p == index); // Malloc must have failed
                    ctx.MallocFailed = true;
                    goto exit_create_index;
                }
                ctx.Flags |= Context.FLAG.InternChanges;
                if (tableName != null)
                    index.Id = ctx.Init.NewTid;
            }

            // If the db->init.busy is 0 then create the index on disk.  This involves writing the index into the master table and filling in the
            // index with the current table contents.
            //
            // The db->init.busy is 0 when the user first enters a CREATE INDEX command.  db->init.busy is 1 when a database is opened and 
            // CREATE INDEX statements are read out of the master table.  In the latter case the index already exists on disk, which is why
            // we don't want to recreate it.
            //
            // If pTblName==0 it means this index is generated as a primary key or UNIQUE constraint of a CREATE TABLE statement.  Since the table
            // has just been created, it contains no data and the index initialization step can be skipped.
            else
            {
                Vdbe v = sqlite3GetVdbe(pParse);
                if (v == null)
                    goto exit_create_index;

                // Create the rootpage for the index
                BeginWriteOperation(1, db);
                int memId = ++Mems;
                v.AddOp2(OP.CreateIndex, db, memId);

                // Gather the complete text of the CREATE INDEX statement into the zStmt variable
                string stmt;
                // A named index with an explicit CREATE INDEX statement
                if (start != null)
                {
                    Debug.Assert(end != null);
                    stmt = sqlite3MPrintf(ctx, "CREATE%s INDEX %.*s", (onError == OE_None ? string.Empty : " UNIQUE"), (int)(nameAsToken.data.Length - end.data.Length) + 1, nameAsToken.data);
                }
                // An automatic index created by a PRIMARY KEY or UNIQUE constraint zStmt = sqlite3MPrintf("");
                else
                    stmt = null;

                // Add an entry in sqlite_master for this index
                object[] args = new[] { ctx.DBs[db].Name, E.SCHEMA_TABLE(db),
                    index.Name,
                    table.Name,
                    memId,
                    stmt };
                NestedParse("INSERT INTO %Q.%s VALUES('index',%Q,%Q,#%d,%Q);", args);
                C._tagfree(ctx, ref stmt);

                // Fill the index with data and reparse the schema. Code an OP_Expire to invalidate all pre-compiled statements.
                if (tableName != null)
                {
                    RefillIndex(index, memId);
                    ChangeCookie(db);
                    v.AddParseSchemaOp(v, db, SysEx.Mprintf(ctx, "name='%q' AND type='index'", index.Name));
                    v.AddOp1(OP_Expire, 0);
                }
            }

            // When adding an index to the list of indices for a table, make sure all indices labeled OE_Replace come after all those labeled
            // OE_Ignore.  This is necessary for the correct constraint check processing (in sqlite3GenerateConstraintChecks()) as part of
            // UPDATE and INSERT statements.  
            if (ctx.Init.Busy || tableName == null)
            {
                if (onError != OE.Replace || table.Index == null || table.Index.OnError == OE.Replace)
                {
                    index.Next = table.Index;
                    table.Index = index;
                }
                else
                {
                    Index otherIndex = table.Index;
                    while (otherIndex.Next != null && otherIndex.Next.OnError != OE.Replace)
                        otherIndex = otherIndex.Next;
                    index.Next = otherIndex.Next;
                    otherIndex.Next = index;
                }
                r = index;
                index = null;
            }

            // Clean up before exiting
        exit_create_index:
            if (index != null)
            {
                C._free(ctx, ref index.ColAff);
                C._free(ctx, ref index);
            }
            ExprList.Delete(ctx, ref list);
            SrcList.Delete(ctx, ref tableName);
            C._free(ctx, ref name);
            return r;
        }

        public static void DefaultRowEst(Index index)
        {
            tRowcnt[] rowEsts = index.RowEsts;
            Debug.Assert(rowEsts != null);
            rowEsts[0] = (int)index.Table.RowEst;
            if (rowEsts[0] < 10) rowEsts[0] = 10;
            tRowcnt n = 10;
            for (int i = 1; i <= index.Columns.length; i++)
            {
                rowEsts[i] = n;
                if (n > 5) n--;
            }
            if (index.OnError != OE.None)
                rowEsts[index.Columns.length] = 1;
        }

        public void DropIndex(SrcList name, bool ifExists)
        {
            Context ctx = Ctx;
            Debug.Assert(Errs == 0); // Never called with prior errors
            if (ctx.MmallocFailed)
                goto exit_drop_index;
            Debug.Assert(name.Srcs == 1);
            if (ReadSchema() != RC.OK)
                goto exit_drop_index;
            Index index = FindIndex(ctx, name.Ids[0].Name, name.Ids[0].Database);
            if (index == null)
            {
                if (!ifExists)
                    ErrorMsg("no such index: %S", name);
                else
                    CodeVerifyNamedSchema(this, name.Ids[0].Database);
                CheckSchema = 1;
                goto exit_drop_index;
            }
            if (index.AutoIndex != 0)
            {
                ErrorMsg("index associated with UNIQUE or PRIMARY KEY constraint cannot be dropped");
                goto exit_drop_index;
            }
            int db = SchemaToIndex(ctx, index.Schema);
#if !OMIT_AUTHORIZATION
            {
                Table table = index.Table;
                string dbName = ctx.DBs[db].Name;
                string tableName = E.SCHEMA_TABLE(db);
                AUTH code = (!E.OMIT_TEMPDB && db != 0 ? AUTH.DROP_TEMP_INDEX : AUTH.DROP_INDEX);
                if (Auth.Check(this, AUTH.DELETE, tableName, null, dbName) || Auth.Check(this, code, index.Name, table.Name, dbName))
                    goto exit_drop_index;
            }
#endif

            // Generate code to remove the index and from the master table
            Vdbe v = GetVdbe();
            if (v != null)
            {
                BeginWriteOperation(1, db);
                object[] args = new[] { ctx.DBs[db].Name, E.SCHEMA_TABLE(db), index.Name };
                NestedParse("DELETE FROM %Q.%s WHERE name=%Q AND type='index'", args);
                ClearStatTables(db, "idx", index.Name);
                ChangeCookie(db);
                DestroyRootPage(index.Id, db);
                v.AddOp4(OP.DropIndex, db, 0, 0, index.Name, 0);
            }

        exit_drop_index:
            sqlite3SrcListDelete(ctx, ref name);
        }

        public static T[] ArrayAllocate<T>(Context ctx, T[] array_, int entrySize, ref int entryLength, ref int index) where T : new()
        {
            int n = entryLength;
            if ((n & (n - 1)) == 0)
            {
                int newSize = (n == 0 ? 1 : 2 * n);
                Array.Resize(ref array_, newSize);
            }
            array_[entryLength] = new T();
            index = entryLength;
            ++entryLength;
            return array_;
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        //public static IdList IdListAppend(Context ctx, int null_2, Token token) { return IdListAppend(ctx, null, token); }
        public static IdList IdListAppend(Context ctx, IdList list, Token token)
        {
            int i = 0;
            if (list == null)
            {
                list = new IdList();
                if (list == null) return null;
                list.Ids.length = 0;
            }
            list.Ids = (IdList.IdListItem[])ArrayAllocate(ctx, list.Ids, -1, ref list.Ids.length, ref i);
            if (i < 0)
            {
                IdListDelete(ctx, ref list);
                return null;
            }
            list.Ids[i].Name = NameFromToken(ctx, token);
            return list;
        }

        public static void IdListDelete(Context ctx, ref IdList list)
        {
            if (list == null) return;
            for (int i = 0; i < list.Ids.length; i++)
                C._tagfree(ctx, ref list.Ids[i].Name);
            C._tagfree(ctx, ref list.Ids);
            C._tagfree(ctx, ref list);
        }

        public static int IdListIndex(IdList list, string name)
        {
            if (list == null) return -1;
            for (int i = 0; i < list.nId; i++)
                if (string.Equals(list.Ids[i].Name, name, StringComparison.OrdinalIgnoreCase))
                    return i;
            return -1;
        }

        public static SrcList SrcListEnlarge(Context ctx, SrcList src, int extra, int start)
        {
            // Sanity checking on calling parameters
            Debug.Assert(start >= 0);
            Debug.Assert(extra >= 1);
            Debug.Assert(src != null);
            Debug.Assert(start <= src.Srcs);

            // Allocate additional space if needed
            if (src.Srcs + extra > src.Allocs)
            {
                int allocs = src.Srcs + extra;
                src.Allocs = (short)allocs;
                Array.Resize(ref src.Ids, allocs);
            }

            // Move existing slots that come after the newly inserted slots out of the way */
            for (int i = src.Srcs - 1; i >= start; i--)
                src.Ids[i + extra] = src.Ids[i];
            src.Srcs += (short)extra;
            // Zero the newly allocated slots
            for (int i = start; i < start + extra; i++)
            {
                src.Ids[i] = new SrcList.SrcListItem();
                src.Ids[i].Cursor = -1;
            }
            // Return a pointer to the enlarged SrcList
            return src;
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        //public static SrcList SrcListAppend(Context ctx, int null_2, Token table, int null_4) { return SrcListAppend(ctx, null, table, null); }
        //public static SrcList SrcListAppend(Context ctx, int null_2, Token table, Token database) { return SrcListAppend(ctx, null, table, database); }
        public static SrcList SrcListAppend(Context ctx, SrcList list, Token table, Token database)
        {
            Debug.Assert(database == null || table != null); // Cannot have C without B
            if (list == null)
            {
                list = new SrcList();
                list.Allocs = 1;
                list.Ids = new SrcList.SrcListItem[1];
            }
            list = SrcListEnlarge(ctx, list, 1, list.Srcs);
            if (ctx.MallocFailed)
            {
                SrcListDelete(ctx, ref list);
                return null;
            }
            SrcList.SrcListItem item = list.Ids[list.Srcs - 1];
            if (database != null && string.IsNullOrEmpty(database.data))
                database = null;
            if (database != null)
            {
                Token tempToken = database;
                database = table;
                table = tempToken;
            }
            item.Name = NameFromToken(ctx, table);
            item.Database = NameFromToken(ctx, database);
            return list;
        }

        public void SrcListAssignCursors(SrcList list)
        {
            Debug.Assert(list != null || Ctx.MallocFailed);
            int i;
            SrcList.SrcListItem item;
            if (list != null)
                for (i = 0, item = list.Ids[0]; i < list.Srcs; i++, item = list.Ids[i])
                {
                    if (item.Cursor >= 0) break;
                    item.Cursor = Tabs++;
                    if (item.Select != null)
                        SrcListAssignCursors(parse, item.Select.Src);
                }
        }

        public static void SrcListDelete(Context ctx, ref SrcList list)
        {
            if (list == null) return;
            int i;
            SrcList.SrcListItem item;
            for (i = 0, item = list.Ids[0]; i < list.Srcs; i++, item = list.Ids[i])
            {
                C._tagfree(ctx, ref item.Database);
                C._tagfree(ctx, ref item.Name);
                C._tagfree(ctx, ref item.Alias);
                C._tagfree(ctx, ref item.Index);
                DeleteTable(ctx, ref item.Table);
                Select.Delete(ctx, ref item.Select);
                Expr.Delete(ctx, ref item.On);
                IdListDelete(ctx, ref item.Using);
            }
            C._tagfree(ctx, ref list);
        }


        // OVERLOADS, so I don't need to rewrite parse.c
        //public SrcList SrcListAppendFromTerm(SrcList list, int null_3, int null_4, Token pAlias, Select pSubquery, Expr on, IdList using_) { return SrcListAppendFromTerm(list, null, null, pAlias, pSubquery, on, using_); }
        //public SrcList SrcListAppendFromTerm(SrcList list, Token table, Token database, Token alias, int null_6, Expr on, IdList using_) { return SrcListAppendFromTerm(list, table, database, alias, null, on, using_); }
        public SrcList SrcListAppendFromTerm(SrcList list, Token table, Token database, Token alias, Select subquery, Expr on, IdList using_)
        {
            Context ctx = Ctx;
            if (list == null && (on != null || using_ != null))
            {
                ErrorMsg("a JOIN clause is required before %s", (on != null ? "ON" : "USING"));
                goto append_from_error;
            }
            list = SrcListAppend(ctx, list, table, database);
            if (list == null || C._NEVER(list.Srcs == 0))
                goto append_from_error;
            SrcList.SrcListItem item = list.Ids[list.Srcs - 1];
            Debug.Assert(alias != null);
            if (alias.length != 0)
                item.Alias = NameFromToken(ctx, alias);
            item.Select = subquery;
            item.On = on;
            item.Using = using_;
            return list;

        append_from_error:
            Debug.Assert(list == null);
            Expr.Delete(ctx, ref on);
            IdListDelete(ctx, ref using_);
            Select.Delete(ctx, ref subquery);
            return null;
        }

        public void SrcListIndexedBy(SrcList list, Token indexedBy)
        {
            Debug.Assert(indexedBy != null);
            if (list != null && C._ALWAYS(list.Srcs > 0))
            {
                SrcList.SrcListItem item = list.Ids[list.Srcs - 1];
                Debug.Assert(!item.NotIndexed && item.Index == null);
                if (indexedBy.length == 1 && indexedBy.data == null)
                    item.notIndexed = 1; // A "NOT INDEXED" clause was supplied. See parse.y construct "indexed_opt" for details.
                else
                    item.Index = NameFromToken(Ctx, indexedBy);
            }
        }

        public static void SrcListShiftJoinType(SrcList list)
        {
            if (list != null && list.Ids != null)
            {
                for (int i = list.Srcs - 1; i > 0; i--)
                    list.Ids[i].Jointype = list.Ids[i - 1].Jointype;
                list.Ids[0].Jointype = 0;
            }
        }

        public void BeginTransaction(int type)
        {
            Context ctx = Ctx;
            Debug.Assert(ctx != null);
            /*  if( db.aDb[0].pBt==0 ) return; */
            if (Auth.Check(this, AUTH.TRANSACTION, "BEGIN", null, null) != 0)
                return;
            Vdbe v = GetVdbe();
            if (v == null) return;
            if (type != TK.DEFERRED)
                for (int i = 0; i < ctx.DBs.length; i++)
                {
                    v.AddOp2(OP.Transaction, i, (type == TK.EXCLUSIVE ? 2 : 1));
                    v.UsesBtree(i);
                }
            v.AddOp2(OP.AutoCommit, 0, 0);
        }

        public void CommitTransaction()
        {
            Debug.Assert(Ctx != null);
#if !OMIT_AUTHORIZATION
            if (Auth.Check(this, AUTH.TRANSACTION, "COMMIT", null, null) != 0)
                return;
#endif
            Vdbe v = GetVdbe();
            if (v != null)
                v.AddOp2(OP.AutoCommit, 1, 0);
        }

        public void RollbackTransaction()
        {
            Debug.Assert(Ctx != null);
#if !OMIT_AUTHORIZATION
            if (Auth.Check(this, AUTH.TRANSACTION, "ROLLBACK", null, null) != 0)
                return;
#endif
            Vdbe v = GetVdbe(pParse);
            if (v != null)
                v.AddOp2(OP.AutoCommit, 1, 1);
        }

#if !OMIT_AUTHORIZATION
        const string[] _savepoint_Names = { "BEGIN", "RELEASE", "ROLLBACK" };
#endif
        public void Savepoint(IPager.SAVEPOINT op, Token name)
        {
#if !OMIT_AUTHORIZATION
            Debug.Assert(SAVEPOINT_BEGIN == 0 && SAVEPOINT_RELEASE == 1 && SAVEPOINT_ROLLBACK == 2);
#endif
            string nameAsString = NameFromToken(Ctx, name);
            if (nameAsString != null)
            {
                Vdbe v = sqlite3GetVdbe(pParse);
                if (v == null
#if !OMIT_AUTHORIZATION
 || Auth.Check(pParse, AUTH.SAVEPOINT, _savepoint_Names[(int)op], nameAsString, 0)
#endif
)
                {
                    C._tagfree(Ctx, ref nameAsString);
                    return;
                }
                v.AddOp4(OP.Savepoint, op, 0, 0, nameAsString, Vdbe.P4T.DYNAMIC);
            }
        }

        public int OpenTempDatabase()
        {
            Context ctx = Ctx;
            if (ctx.DBs[1].Bt == null && !Explain)
            {
                Btree bt = null;
                const VSystem.OPEN flags = VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE | VSystem.OPEN.EXCLUSIVE | VSystem.OPEN.DELETEONCLOSE | VSystem.OPEN.TEMP_DB;
                RC rc = BtreeOpen(ctx.Vfs, null, ctx, ref bt, (Btree.OPEN)0, flags);
                if (rc != RC.OK)
                {
                    ErrorMsg("unable to open a temporary database file for storing temporary tables");
                    RC = rc;
                    return 1;
                }
                ctx.DBs[1].Bt = bt;
                Debug.Assert(ctx.DBs[1].Schema != null);
                if (Btree.SetPageSize(bt, ctx.NextPagesize, -1, 0) == RC.NOMEM)
                {
                    ctx.MallocFailed = true;
                    return 1;
                }
            }
            return 0;
        }

        public void CodeVerifySchema(int db)
        {
            Parse toplevel = E.Parse_Toplevel(this);
#if !OMIT_TRIGGER
            if (toplevel != this)
            {
                // This branch is taken if a trigger is currently being coded. In this case, set cookieGoto to a non-zero value to show that this function
                // has been called. This is used by the sqlite3ExprCodeConstants() function.
                CookieGoto = -1;
            }
#endif
            if (toplevel.CookieGoto == 0)
            {
                Vdbe v = toplevel.GetVdbe();
                if (v == null) return; // This only happens if there was a prior error
                toplevel.CookieGoto = v.AddOp2(OP.Goto, 0, 0) + 1;
            }
            if (db >= 0)
            {
                Context ctx = toplevel.Ctx;
                Debug.Assert(db < ctx.DBs.length);
                Debug.Assert(ctx.DBs[db].Bt != null || db == 1);
                Debug.Assert(db < MAX_ATTACHED + 2);
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                yDbMask mask = ((yDbMask)1) << db;
                if ((toplevel.CookieMask & mask) == 0)
                {
                    toplevel.CookieMask |= mask;
                    toplevel.CookieValue[db] = ctx.DBs[db].Schema.SchemaCookie;
                    if (!E.OMIT_TEMPDB && db == 1)
                        toplevel.OpenTempDatabase();
                }
            }
        }

        public void CodeVerifyNamedSchema(string dbName)
        {
            Context ctx = Ctx;
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Context.DB dbObj = ctx.DBs[i];
                if (dbObj.Bt != null && (dbName == null || !string.Equals(dbName, dbObj.zName)))
                    CodeVerifySchema(i);
            }
        }

        public void BeginWriteOperation(int setStatement, int db)
        {
            Parse toplevel = E.Parse_Toplevel(this);
            CodeVerifySchema(db);
            toplevel.WriteMask |= ((yDbMask)1) << db;
            toplevel.IsMultiWrite |= setStatement;
        }

        public void MultiWrite()
        {
            Parse toplevel = E.Parse_Toplevel(this);
            toplevel.IsMultiWrite = 1;
        }

        public void MayAbort()
        {
            Parse toplevel = E.Parse_Toplevel(this);
            toplevel.MayAbort = 1;
        }

        public void HaltConstraint(RC errorCode, int onError, string p4, int p4type)
        {
            Vdbe v = GetVdbe();
            Debug.Assert((errCode & 0xff) == RC.CONSTRAINT);
            if (onError == OE.Abort)
                MayAbort();
            v.AddOp4(OP.Halt, errCode, onError, 0, p4, p4type);
        }

#if !OMIT_REINDEX
        static bool CollationMatch(string collName, Index index)
        {
            int i;
            Debug.Assert(collName != null);
            for (i = 0; i < index.nColumn; i++)
            {
                string z = index.azColl[i];
                Debug.Assert(z != null);
                if (z.Equals(collName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        static void ReindexTable(Parse parse, Table table, string collName)
        {
            for (Index index = table.Index; index != null; index = index.Next)
                if (collName == null || CollationMatch(collName, index))
                {
                    int db = SchemaToIndex(parse.Ctx, table.Schema);
                    BeginWriteOperation(0, db);
                    RefillIndex(index, -1);
                }
        }

        static void ReindexDatabases(Parse parse, string collName)
        {
            Context ctx = Ctx;    // The database connection
            Debug.Assert(Btree.HoldsAllMutexes(ctx));  // Needed for schema access
            int db; // The database index number
            Context.DB dbObj; // A single database
            for (db = 0, dbObj = ctx.DBs[0]; db < ctx.DBs.legth; db++, dbObj = ctx.DBs[db])
            {
                Debug.Assert(dbObj != null);
                for (HashElem k = dbObj.Schema.TableHash.First; k != null; k = k.Next)
                {
                    Table table = (Table)k.data; // A table in the database
                    ReindexTable(parse, table, collName);
                }
            }
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        //public void Reindex(int null_2, int null_3) { Reindex(null, null); }
        public void Reindex(Token name1, Token name2)
        {
            Context ctx = Ctx; // The database connection
            // Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
            if (ReadSchema(this) != RC.OK)
                return;
            if (name1 == null)
            {
                ReindexDatabases(this, null);
                return;
            }
            else if (C._NEVER(name2 == null) || string.IsNullOrEmpty(name2.data))
            {
                Debug.Assert(name1.data != null);
                string collName = NameFromToken(ctx, name1);
                if (collName == null) return;
                CollSeq coll = FindCollSeq(ctx, Context.CTXENCODE(ctx), collName, 0); // Collating sequence to be reindexed, or NULL
                if (coll != null)
                {
                    ReindexDatabases(this, collName);
                    C._tagfree(ctx, ref collName);
                    return;
                }
                C._tagfree(ctx, ref collName);
            }
            Token objName = new Token();  // Name of the table or index to be reindexed
            int db = TwoPartName(name1, name2, ref objName); // The database index number
            if (db < 0)
                return;
            string z = NameFromToken(ctx, objName); // Name of a table or index
            if (z == null) return;
            string dbName = ctx.DBs[db].Name; // Name of the database
            Table table = FindTable(ctx, z, dbName); // A table in the database
            if (table != null)
            {
                ReindexTable(this, table, null);
                C._tagfree(ctx, ref z);
                return;
            }
            Index index = sqlite3FindIndex(ctx, z, dbName); // An index associated with pTab
            C._tagfree(ctx, ref z);
            if (index != null)
            {
                BeginWriteOperation(0, db);
                RefillIndex(index, -1);
                return;
            }
            ErrorMsg("unable to identify the object to be reindexed");
        }
#endif

        public KeyInfo IndexKeyinfo(Index index)
        {
            int colLength = index.Columns.length;
            Context ctx = Ctx;
            KeyInfo key = new KeyInfo();
            if (key != null)
            {
                key.db = parse.db;
                key.SortOrders = new byte[colLength];
                key.Colls = new CollSeq[colLength];
                for (int i = 0; i < colLength; i++)
                {
                    string collName = index.CollNames[i];
                    Debug.Assert(collName != null);
                    key.Colls[i] = LocateCollSeq(this, collName);
                    key.SortOrders[i] = index.SortOrders[i];
                }
                key.Fields = (ushort)colLength;
            }
            if (Errs != 0)
            {
                C._tagfree(ctx, ref key);
                key = null;
            }
            return key;
        }
    }
}
