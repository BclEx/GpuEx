using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Prepare
    {
        static void CorruptSchema(InitData data, string obj, string extra)
        {
            Context ctx = data.Ctx;
            if (!ctx.MallocFailed && (ctx.Flags & Context.FLAG.RecoveryMode) == 0)
            {
                if (obj == null) obj = "?";
                C._setstring(ref data.ErrMsg, ctx, "malformed database schema (%s)", obj);
                if (extra == null)
                    data.ErrMsg = C._mtagappendf(ctx, data.ErrMsg, "%s - %s", data.ErrMsg, extra);
                data.RC = (ctx.MallocFailed ? RC.NOMEM : SysEx.CORRUPT_BKPT());
            }
        }

        public static bool InitCallback(object init, int argc, string[] argv, object notUsed1)
        {
            InitData data = (InitData)init;
            Context ctx = data.Ctx;
            int db = data.Db;

            Debug.Assert(argc == 3);
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            E.DbClearProperty(ctx, db, SCHEMA.Empty);
            if (ctx.MallocFailed)
            {
                CorruptSchema(data, argv[0], null);
                return true;
            }

            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            if (argv == null) return false; // Might happen if EMPTY_RESULT_CALLBACKS are on */
            if (argv[1] == null)
                CorruptSchema(data, argv[0], null);
            else if (!string.IsNullOrEmpty(argv[2]))
            {
                // Call the parser to process a CREATE TABLE, INDEX or VIEW. But because ctx->init.busy is set to 1, no VDBE code is generated
                // or executed.  All the parser does is build the internal data structures that describe the table, index, or view.
                Debug.Assert(ctx.Init.Busy);
                ctx.Init.DB = (byte)db;
                ctx.Init.NewTid = ConvertEx.Atoi(argv[1]);
                ctx.Init.OrphanTrigger = false;
                Vdbe stmt = null;
#if DEBUG
                int rcp = Prepare(ctx, argv[2], -1, ref stmt, null);
#else
                Prepare(ctx, argv[2], -1, ref stmt, null);
#endif
                RC rc = ctx.ErrCode;
#if DEBUG
                Debug.Assert(((int)rc & 0xFF) == (rcp & 0xFF));
#endif
                ctx.Init.DB = 0;
                if (rc != RC.OK)
                {
                    if (ctx.Init.OrphanTrigger)
                        Debug.Assert(db == 1);
                    else
                    {
                        data.RC = rc;
                        if (rc == RC.NOMEM)
                            ctx.MallocFailed = true;
                        else if (rc != RC.INTERRUPT && (RC)((int)rc & 0xFF) != RC.LOCKED)
                            CorruptSchema(data, argv[0], sqlite3_errmsg(ctx));
                    }
                }
                stmt.Finalize();
            }
            else if (argv[0] == null)
                CorruptSchema(data, null, null);
            else
            {
                // If the SQL column is blank it means this is an index that was created to be the PRIMARY KEY or to fulfill a UNIQUE
                // constraint for a CREATE TABLE.  The index should have already been created when we processed the CREATE TABLE.  All we have
                // to do here is record the root page number for that index.
                Index index = Parse.FindIndex(ctx, argv[0], ctx.DBs[db].Name);
                if (index == null)
                {
                    // This can occur if there exists an index on a TEMP table which has the same name as another index on a permanent index.  Since
                    // the permanent table is hidden by the TEMP table, we can also safely ignore the index on the permanent table.
                    // Do Nothing
                }
                else if (!ConvertEx.Atoi(argv[1], ref index.Id))
                    CorruptSchema(data, argv[0], "invalid rootpage");
            }
            return false;
        }

        // The master database table has a structure like this
        static readonly string master_schema =
        "CREATE TABLE sqlite_master(\n" +
        "  type text,\n" +
        "  name text,\n" +
        "  tbl_name text,\n" +
        "  rootpage integer,\n" +
        "  sql text\n" +
        ")";
#if !OMIT_TEMPDB
        static readonly string temp_master_schema =
        "CREATE TEMP TABLE sqlite_temp_master(\n" +
        "  type text,\n" +
        "  name text,\n" +
        "  tbl_name text,\n" +
        "  rootpage integer,\n" +
        "  sql text\n" +
        ")";
#else
        static readonly string temp_master_schema = null;
#endif

        //ctx.pDfltColl = sqlite3FindCollSeq( ctx, SQLITE_UTF8, "BINARY", 0 );
        public static RC InitOne(Context ctx, int db, ref string errMsg)
        {
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            Debug.Assert(ctx.DBs[db].Schema != null);
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            Debug.Assert(db == 1 || ctx.DBs[db].Bt.HoldsMutex());
            RC rc;
            int i;

            // zMasterSchema and zInitScript are set to point at the master schema and initialisation script appropriate for the database being
            // initialized. zMasterName is the name of the master table.
            string masterSchema = (E.OMIT_TEMPDB == 0 && db == 1 ? temp_master_schema : master_schema);
            string masterName = E.SCHEMA_TABLE(db);

            // Construct the schema tables.
            string[] args = new string[4];
            args[0] = masterName;
            args[1] = "1";
            args[2] = masterSchema;
            args[3] = null;
            InitData initData = new InitData();
            initData.Ctx = ctx;
            initData.Db = db;
            initData.RC = RC.OK;
            initData.ErrMsg = errMsg;
            InitCallback(initData, 3, args, null);
            if (initData.RC != 0)
            {
                rc = initData.RC;
                goto error_out;
            }
            Table table = Parse.FindTable(ctx, masterName, ctx.DBs[db].Name);
            if (C._ALWAYS(table != null))
                table.TabFlags |= TF.Readonly;

            // Create a cursor to hold the database open
            Context.DB dbAsObj = ctx.DBs[db];
            if (dbAsObj.Bt == null)
            {
                if (E.OMIT_TEMPDB == 0 && C._ALWAYS(db == 1))
                    E.DbSetProperty(ctx, 1, SCHEMA.SchemaLoaded);
                return RC.OK;
            }

            // If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
            // will be closed before this function returns.
            bool openedTransaction = false;
            dbAsObj.Bt.Enter();
            if (!dbAsObj.Bt.IsInReadTrans())
            {
                rc = dbAsObj.Bt.BeginTrans(0);
                if (rc != RC.OK)
                {
                    C._setstring(ref errMsg, ctx, "%s", Main.ErrStr(rc));
                    goto initone_error_out;
                }
                openedTransaction = true;
            }

            // Get the database meta information.
            // Meta values are as follows:
            //    meta[0]   Schema cookie.  Changes with each schema change.
            //    meta[1]   File format of schema layer.
            //    meta[2]   Size of the page cache.
            //    meta[3]   Largest rootpage (auto/incr_vacuum mode)
            //    meta[4]   Db text encoding. 1:UTF-8 2:UTF-16LE 3:UTF-16BE
            //    meta[5]   User version
            //    meta[6]   Incremental vacuum mode
            //    meta[7]   unused
            //    meta[8]   unused
            //    meta[9]   unused
            // Note: The #defined SQLITE_UTF* symbols in sqliteInt.h correspond to the possible values of meta[4].
            uint[] meta = new uint[5];
            for (i = 0; i < meta.Length; i++)
                dbAsObj.Bt.GetMeta((Btree.META)i + 1, ref meta[i]);
            dbAsObj.Schema.SchemaCookie = (int)meta[(int)Btree.META.SCHEMA_VERSION - 1];

            // If opening a non-empty database, check the text encoding. For the main database, set sqlite3.enc to the encoding of the main database.
            // For an attached ctx, it is an error if the encoding is not the same as sqlite3.enc.
            if (meta[(int)Btree.META.TEXT_ENCODING - 1] != 0) // text encoding
            {
                if (db == 0)
                {
#if !OMIT_UTF16
                    // If opening the main database, set ENC(db).
                    TEXTENCODE encoding = (TEXTENCODE)(meta[(int)Btree.META.TEXT_ENCODING - 1] & 3);
                    if (encoding == 0) encoding = TEXTENCODE.UTF8;
                    E.CTXENCODE(ctx, encoding);
#else
				    E.CTXENCODE(ctx, TEXTENCODE_UTF8);
#endif
                }
                else
                {
                    // If opening an attached database, the encoding much match ENC(db)
                    if ((TEXTENCODE)meta[(int)Btree.META.TEXT_ENCODING - 1] != E.CTXENCODE(ctx))
                    {
                        C._setstring(ref errMsg, ctx, "attached databases must use the same text encoding as main database");
                        rc = RC.ERROR;
                        goto initone_error_out;
                    }
                }
            }
            else
                E.DbSetProperty(ctx, db, SCHEMA.Empty);
            dbAsObj.Schema.Encode = E.CTXENCODE(ctx);

            if (dbAsObj.Schema.CacheSize == 0)
            {
                dbAsObj.Schema.CacheSize = DEFAULT_CACHE_SIZE;
                dbAsObj.Bt.SetCacheSize(dbAsObj.Schema.CacheSize);
            }

            // file_format==1    Version 3.0.0.
            // file_format==2    Version 3.1.3.  // ALTER TABLE ADD COLUMN
            // file_format==3    Version 3.1.4.  // ditto but with non-NULL defaults
            // file_format==4    Version 3.3.0.  // DESC indices.  Boolean constants
            dbAsObj.Schema.FileFormat = (byte)meta[(int)Btree.META.FILE_FORMAT - 1];
            if (dbAsObj.Schema.FileFormat == 0)
                dbAsObj.Schema.FileFormat = 1;
            if (dbAsObj.Schema.FileFormat > MAX_FILE_FORMAT)
            {
                C._setstring(ref errMsg, ctx, "unsupported file format");
                rc = RC.ERROR;
                goto initone_error_out;
            }

            // Ticket #2804:  When we open a database in the newer file format, clear the legacy_file_format pragma flag so that a VACUUM will
            // not downgrade the database and thus invalidate any descending indices that the user might have created.
            if (db == 0 && meta[(int)Btree.META.FILE_FORMAT - 1] >= 4)
                ctx.Flags &= ~Context.FLAG.LegacyFileFmt;

            // Read the schema information out of the schema tables
            Debug.Assert(ctx.Init.Busy);
            {
                string sql = C._mtagprintf(ctx, "SELECT name, rootpage, sql FROM '%q'.%s ORDER BY rowid", ctx.DBs[db].Name, masterName);
#if !OMIT_AUTHORIZATION
                {
                    Func<object, int, string, string, string, string, ARC> auth = ctx.Auth;
                    ctx.Auth = null;
#endif
                    rc = sqlite3_exec(ctx, sql, InitCallback, initData, 0);
                    //: errMsg = initData.ErrMsg;
#if !OMIT_AUTHORIZATION
                    ctx.Auth = auth;
                }
#endif
                if (rc == RC.OK) rc = initData.RC;
                C._tagfree(ctx, ref sql);
#if !OMIT_ANALYZE
                if (rc == RC.OK)
                    sqlite3AnalysisLoad(ctx, db);
#endif
            }
            if (ctx.MallocFailed)
            {
                rc = RC.NOMEM;
                Main.ResetAllSchemasOfConnection(ctx);
            }
            if (rc == RC.OK || (ctx.Flags & Context.FLAG.RecoveryMode) != 0)
            {
                // Black magic: If the SQLITE_RecoveryMode flag is set, then consider the schema loaded, even if errors occurred. In this situation the 
                // current sqlite3_prepare() operation will fail, but the following one will attempt to compile the supplied statement against whatever subset
                // of the schema was loaded before the error occurred. The primary purpose of this is to allow access to the sqlite_master table
                // even when its contents have been corrupted.
                E.DbSetProperty(ctx, db, DB.SchemaLoaded);
                rc = RC.OK;
            }

        // Jump here for an error that occurs after successfully allocating curMain and calling sqlite3BtreeEnter(). For an error that occurs
        // before that point, jump to error_out.
        initone_error_out:
            if (openedTransaction)
                dbAsObj.Bt.Commit();
            dbAsObj.Bt.Leave();

        error_out:
            if (rc == RC.NOMEM || rc == RC.IOERR_NOMEM)
                ctx.MallocFailed = true;
            return rc;
        }

        public static RC Init(Context ctx, ref string errMsg)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            RC rc = RC.OK;
            ctx.Init.Busy = true;
            for (int i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
            {
                if (E.DbHasProperty(ctx, i, SCHEMA.SchemaLoaded) || i == 1) continue;
                rc = InitOne(ctx, i, ref errMsg);
                if (rc != 0)
                    ResetOneSchema(ctx, i);
            }

            // Once all the other databases have been initialized, load the schema for the TEMP database. This is loaded last, as the TEMP database
            // schema may contain references to objects in other databases.
#if !OMIT_TEMPDB
            if (rc == RC.OK && C._ALWAYS(ctx.DBs.length > 1) && !E.DbHasProperty(ctx, 1, SCHEMA.SchemaLoaded))
            {
                rc = InitOne(ctx, 1, ref errMsg);
                if (rc != 0)
                    ResetOneSchema(ctx, 1);
            }
#endif

            bool commitInternal = !((ctx.Flags & Context.FLAG.InternChanges) != 0);
            ctx.Init.Busy = false;
            if (rc == RC.OK && commitInternal)
                CommitInternalChanges(ctx);
            return rc;
        }

        public static RC ReadSchema(Parse parse)
        {
            RC rc = RC.OK;
            Context ctx = parse.Ctx;
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            if (!ctx.Init.Busy)
                rc = Init(ctx, ref parse.ErrMsg);
            if (rc != RC.OK)
            {
                parse.RC = rc;
                parse.Errs++;
            }
            return rc;
        }

        static void SchemaIsValid(Parse parse)
        {
            Context ctx = parse.Ctx;
            Debug.Assert(parse.CheckSchema != 0);
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            for (int db = 0; db < ctx.DBs.length; db++)
            {
                bool openedTransaction = false; // True if a transaction is opened
                Btree bt = ctx.DBs[db].Bt; // Btree database to read cookie from
                if (bt == null) continue;

                // If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
                // will be closed immediately after reading the meta-value.
                if (!bt.IsInReadTrans())
                {
                    RC rc = bt.BeginTrans(0);
                    if (rc == RC.NOMEM || rc == RC.IOERR_NOMEM)
                        ctx.MallocFailed = true;
                    if (rc != RC.OK) return;
                    openedTransaction = true;
                }

                // Read the schema cookie from the database. If it does not match the value stored as part of the in-memory schema representation,
                // set Parse.rc to SQLITE_SCHEMA.
                uint cookie;
                bt.GetMeta(Btree.META.SCHEMA_VERSION, ref cookie);
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                if (cookie != ctx.DBs[db].Schema.SchemaCookie)
                {
                    ResetOneSchema(ctx, db);
                    parse.RC = RC.SCHEMA;
                }

                // Close the transaction, if one was opened.
                if (openedTransaction)
                    bt.Commit();
            }
        }

        public static int SchemaToIndex(Context ctx, Schema schema)
        {
            // If pSchema is NULL, then return -1000000. This happens when code in expr.c is trying to resolve a reference to a transient table (i.e. one
            // created by a sub-select). In this case the return value of this function should never be used.
            //
            // We return -1000000 instead of the more usual -1 simply because using -1000000 as the incorrect index into ctx->aDb[] is much 
            // more likely to cause a segfault than -1 (of course there are assert() statements too, but it never hurts to play the odds).
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            int i = -1000000;
            if (schema != null)
            {
                for (i = 0; C._ALWAYS(i < ctx.DBs.length); i++)
                    if (ctx.DBs[i].Schema == schema)
                        break;
                Debug.Assert(i >= 0 && i < ctx.DBs.length);
            }
            return i;
        }

#if !OMIT_EXPLAIN
        static readonly string[] _colName = new string[] {
            "addr", "opcode", "p1", "p2", "p3", "p4", "p5", "comment",
            "selectid", "order", "from", "detail" };
#endif
        public static RC Prepare_(Context ctx, string sql, int bytes, bool isPrepareV2, Vdbe reprepare, ref Vdbe stmtOut, ref string tailOut)
        {
            stmtOut = null;
            tailOut = null;
            string errMsg = null; // Error message
            RC rc = RC.OK;
            int i;

            // Allocate the parsing context
            Parse parse = new Parse(); // Parsing context
            if (parse == null)
            {
                rc = RC.NOMEM;
                goto end_prepare;
            }
            parse.Reprepare = reprepare;
            parse.LastToken.data = null; //: C#?
            Debug.Assert(tailOut == null);
            Debug.Assert(!ctx.MallocFailed);
            Debug.Assert(MutexEx.Held(ctx.Mutex));

            // Check to verify that it is possible to get a read lock on all database schemas.  The inability to get a read lock indicates that
            // some other database connection is holding a write-lock, which in turn means that the other connection has made uncommitted changes
            // to the schema.
            //
            // Were we to proceed and prepare the statement against the uncommitted schema changes and if those schema changes are subsequently rolled
            // back and different changes are made in their place, then when this prepared statement goes to run the schema cookie would fail to detect
            // the schema change.  Disaster would follow.
            //
            // This thread is currently holding mutexes on all Btrees (because of the sqlite3BtreeEnterAll() in sqlite3LockAndPrepare()) so it
            // is not possible for another thread to start a new schema change while this routine is running.  Hence, we do not need to hold 
            // locks on the schema, we just need to make sure nobody else is holding them.
            //
            // Note that setting READ_UNCOMMITTED overrides most lock detection, but it does *not* override schema lock detection, so this all still
            // works even if READ_UNCOMMITTED is set.
            for (i = 0; i < ctx.DBs.length; i++)
            {
                Btree bt = ctx.DBs[i].Bt;
                if (bt != null)
                {
                    Debug.Assert(bt.HoldsMutex());
                    rc = bt.SchemaLocked();
                    if (rc != 0)
                    {
                        string dbName = ctx.DBs[i].Name;
                        sqlite3Error(ctx, rc, "database schema is locked: %s", dbName);
                        C.ASSERTCOVERAGE((ctx.Flags & Context.FLAG.ReadUncommitted) != 0);
                        goto end_prepare;
                    }
                }
            }

            VTable.UnlockList(ctx);

            parse.Ctx = ctx;
            parse.QueryLoops = (double)1;
            if (bytes >= 0 && (bytes == 0 || sql[bytes - 1] != 0))
            {
                int maxLen = ctx.aLimit[SQLITE_LIMIT_SQL_LENGTH];
                C.ASSERTCOVERAGE(bytes == maxLen);
                C.ASSERTCOVERAGE(bytes == maxLen + 1);
                if (bytes > maxLen)
                {
                    sqlite3Error(ctx, RC.TOOBIG, "statement too long");
                    rc = SysEx.ApiExit(ctx, RC.TOOBIG);
                    goto end_prepare;
                }
                string sqlCopy = sql.Substring(0, bytes);
                if (sqlCopy != null)
                {
                    parse.RunParser(sqlCopy, ref errMsg);
                    C._tagfree(ctx, ref sqlCopy);
                    parse.Tail = null; //: &sql[parse->Tail - sqlCopy];
                }
                else
                    parse.Tail = null; //: &sql[bytes];
            }
            else
                parse.RunParser(sql, ref errMsg);
            Debug.Assert((int)parse.QueryLoops == 1);

            if (ctx.MallocFailed)
                parse.RC = RC.NOMEM;
            if (parse.RC == RC.DONE)
                parse.RC = RC.OK;
            if (parse.CheckSchema != 0)
                SchemaIsValid(parse);
            if (ctx.MallocFailed)
                parse.RC = RC.NOMEM;
            tailOut = (parse.Tail == null ? null : parse.Tail.ToString());
            rc = parse.RC;

            Vdbe v = parse.V;
#if !OMIT_EXPLAIN
            if (rc == RC.OK && parse.V != null && parse.Explain != 0)
            {
                int first, max;
                if (parse.Explain == 2)
                {
                    v.SetNumCols(4);
                    first = 8;
                    max = 12;
                }
                else
                {
                    v.SetNumCols(8);
                    first = 0;
                    max = 8;
                }
                for (i = first; i < max; i++)
                    v.SetColName(i - first, COLNAME_NAME, _colName[i], C.DESTRUCTOR_STATIC);
            }
#endif

            Debug.Assert(!ctx.Init.Busy || !isPrepareV2);
            if (!ctx.Init.Busy)
                Vdbe.SetSql(v, sql, (int)(sql.Length - (parse.Tail == null ? 0 : parse.Tail.Length)), isPrepareV2);
            if (v != null && (rc != RC.OK || ctx.MallocFailed))
            {
                v.Finalize();
                Debug.Assert(stmtOut == null);
            }
            else
                stmtOut = v;

            if (errMsg != null)
            {
                sqlite3Error(ctx, rc, "%s", errMsg);
                C._tagfree(ctx, ref errMsg);
            }
            else
                sqlite3Error(ctx, rc, null);

            // Delete any TriggerPrg structures allocated while parsing this statement.
            while (parse.TriggerPrg != null)
            {
                TriggerPrg t = parse.TriggerPrg;
                parse.TriggerPrg = t.Next;
                C._tagfree(ctx, ref t);
            }

        end_prepare:
            //sqlite3StackFree( db, pParse );
            rc = SysEx.ApiExit(ctx, rc);
            Debug.Assert((RC)((int)rc & ctx.ErrMask) == rc);
            return rc;
        }

        //C# Version w/o End of Parsed String
        public static RC LockAndPrepare(Context ctx, string sql, int bytes, bool isPrepareV2, Vdbe reprepare, ref Vdbe stmtOut, string dummy1) { string tailOut = null; return LockAndPrepare(ctx, sql, bytes, isPrepareV2, reprepare, ref stmtOut, ref tailOut); }
        public static RC LockAndPrepare(Context ctx, string sql, int bytes, bool isPrepareV2, Vdbe reprepare, ref Vdbe stmtOut, ref string tailOut)
        {
            if (!sqlite3SafetyCheckOk(ctx))
            {
                stmtOut = null;
                tailOut = null;
                return SysEx.MISUSE_BKPT();
            }
            MutexEx.Enter(ctx.Mutex);
            Btree.EnterAll(ctx);
            RC rc = Prepare_(ctx, sql, bytes, isPrepareV2, reprepare, ref stmtOut, ref tailOut);
            if (rc == RC.SCHEMA)
            {
                stmtOut.Finalize();
                rc = Prepare_(ctx, sql, bytes, isPrepareV2, reprepare, ref stmtOut, ref tailOut);
            }
            Btree.LeaveAll(ctx);
            MutexEx.Leave(ctx.Mutex);
            Debug.Assert(rc == RC.OK || stmtOut == null);
            return rc;
        }

        public static RC Reprepare(Vdbe p)
        {
            Context ctx = p.Ctx;
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            string sql = Vdbe.Sql(p);
            Debug.Assert(sql != null); // Reprepare only called for prepare_v2() statements
            Vdbe newVdbe = new Vdbe();
            RC rc = LockAndPrepare(ctx, sql, -1, false, p, ref newVdbe, null);
            if (rc != 0)
            {
                if (rc == RC.NOMEM)
                    ctx.MallocFailed = true;
                Debug.Assert(newVdbe == null);
                return rc;
            }
            else
                Debug.Assert(newVdbe != null);
            Vdbe.Swap((Vdbe)newVdbe, p);
            Vdbe.TransferBindings(newVdbe, (Vdbe)p);
            newVdbe.ResetStepResult();
            newVdbe.Finalize();
            return RC.OK;
        }


        //C# Overload for ignore error out
        static public RC Prepare_(Context ctx, string sql, int bytes, ref Vdbe vdbeOut, string dummy1) { string tailOut = null; return Prepare_(ctx, sql, bytes, ref vdbeOut, ref tailOut); }
        static public RC Prepare_(Context ctx, StringBuilder sql, int bytes, ref Vdbe vdbeOut, string dummy1) { string tailOut = null; return Prepare_(ctx, sql.ToString(), bytes, ref vdbeOut, ref tailOut); }
        static public RC Prepare_(Context ctx, string sql, int bytes, ref Vdbe vdbeOut, ref string tailOut)
        {
            RC rc = LockAndPrepare(ctx, sql, bytes, false, null, ref vdbeOut, ref tailOut);
            Debug.Assert(rc == RC.OK || vdbeOut == null); // VERIFY: F13021
            return rc;
        }

        public static RC Prepare_v2(Context ctx, string sql, int bytes, ref Vdbe stmtOut, string dummy1) { string tailOut = null; return Prepare_v2(ctx, sql, bytes, ref stmtOut, ref tailOut); }
        public static RC Prepare_v2(Context ctx, string sql, int bytes, ref Vdbe stmtOut, ref string tailOut)
        {
            RC rc = LockAndPrepare(ctx, sql, bytes, true, null, ref stmtOut, ref tailOut);
            Debug.Assert(rc == RC.OK || stmtOut == null); // VERIFY: F13021
            return rc;
        }

#if !OMIT_UTF16
        public static RC Prepare16(Context ctx, string sql, int bytes, bool isPrepareV2, out Vdbe stmtOut, out string tailOut)
        {
            // This function currently works by first transforming the UTF-16 encoded string to UTF-8, then invoking sqlite3_prepare(). The
            // tricky bit is figuring out the pointer to return in *pzTail.
            stmtOut = null;
            if (!sqlite3SafetyCheckOk(ctx))
                return SysEx.MISUSE_BKPT();
            MutexEx.Enter(ctx.Mutex);
            RC rc = RC.OK;
            string tail8 = null;
            string sql8 = Vdbe.Utf16to8(ctx, sql, bytes, TEXTENCODE.UTF16NATIVE);
            if (sql8 != null)
                rc = LockAndPrepare(ctx, sql8, -1, isPrepareV2, null, ref stmtOut, ref tail8);
            if (tail8 != null && tailOut != null)
            {
                // If sqlite3_prepare returns a tail pointer, we calculate the equivalent pointer into the UTF-16 string by counting the unicode
                // characters between zSql8 and zTail8, and then returning a pointer the same number of characters into the UTF-16 string.
                Debugger.Break();
                //: int charsParsed = Vdbe::Utf8CharLen(sql8, (int)(tail8 - sql8));
                //: *tailOut = (uint8 *)sql + Vdbe::Utf16ByteLen(sql, charsParsed);
            }
            C._tagfree(ctx, ref sql8);
            rc = SysEx.ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public static RC Prepare16(Context ctx, string sql, int bytes, out Vdbe stmtOut, out string tailOut)
        {
            RC rc = Prepare16(ctx, sql, bytes, false, out stmtOut, out tailOut);
            Debug.Assert(rc == RC.OK || stmtOut == null); // VERIFY: F13021
            return rc;
        }

        public static RC Prepare16_v2(Context ctx, string sql, int bytes, out Vdbe stmtOut, out string tailOut)
        {
            RC rc = Prepare16(ctx, sql, bytes, true, out stmtOut, out tailOut);
            Debug.Assert(rc == RC.OK || stmtOut == null); // VERIFY: F13021
            return rc;
        }
#endif
    }
}
