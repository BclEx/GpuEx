using System;
using System.Diagnostics;
using System.Text;
using System.Globalization;
using Core.IO;

namespace Core.Command
{
    public partial class Pragma
    {
        // moved to ConvertEx
        //public static byte GetSafetyLevel(string z, int omitFull, byte dflt);
        //public static bool GetBoolean(string z, byte dflt);

#if !(OMIT_PRAGMA)

        static IPager.LOCKINGMODE GetLockingMode(string z)
        {
            if (z != null)
            {
                if (string.Equals(z, "exclusive", StringComparison.OrdinalIgnoreCase)) return IPager.LOCKINGMODE.EXCLUSIVE;
                if (string.Equals(z, "normal", StringComparison.OrdinalIgnoreCase)) return IPager.LOCKINGMODE.NORMAL;
            }
            return IPager.LOCKINGMODE.QUERY;
        }

#if !OMIT_AUTOVACUUM
        static Btree.AUTOVACUUM GetAutoVacuum(string z)
        {
            if (string.Equals(z, "none", StringComparison.OrdinalIgnoreCase)) return Btree.AUTOVACUUM.NONE;
            if (string.Equals(z, "full", StringComparison.OrdinalIgnoreCase)) return Btree.AUTOVACUUM.FULL;
            if (string.Equals(z, "incremental", StringComparison.OrdinalIgnoreCase)) return Btree.AUTOVACUUM.INCR;
            int i = ConvertEx.Atoi(z);
            return (Btree.AUTOVACUUM)(i >= 0 && i <= 2 ? i : 0);
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static int GetTempStore(string z)
        {
            if (z[0] >= '0' && z[0] <= '2') return z[0] - '0';
            else if (string.Equals(z, "file", StringComparison.OrdinalIgnoreCase)) return 1;
            else if (string.Equals(z, "memory", StringComparison.OrdinalIgnoreCase)) return 2;
            else return 0;
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static RC InvalidateTempStorage(Parse parse)
        {
            Context ctx = parse.Ctx;
            if (ctx.DBs[1].Bt != null)
            {
                if (ctx.AutoCommit == 0 || ctx.DBs[1].Bt.IsInReadTrans())
                {
                    parse.ErrorMsg("temporary storage cannot be changed from within a transaction");
                    return RC.ERROR;
                }
                ctx.DBs[1].Bt.Close();
                ctx.DBs[1].Bt = null;
                Parse.ResetInternalSchema(ctx, -1);
            }
            return RC.OK;
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static RC ChangeTempStorage(Parse parse, string storageType)
        {
            int ts = GetTempStore(storageType);
            Context ctx = parse.Ctx;
            if (ctx.TempStore == ts) return RC.OK;
            if (InvalidateTempStorage(parse) != RC.OK)
                return RC.ERROR;
            ctx.TempStore = (byte)ts;
            return RC.OK;
        }
#endif

        static void ReturnSingleInt(Parse parse, string label, long value)
        {
            Vdbe v = parse.GetVdbe();
            int mem = ++parse.Mems;
            v.AddOp4(OP.Int64, 0, mem, 0, value, Vdbe.P4T.INT64);
            v.SetNumCols(1);
            v.SetColName(0, COLNAME_NAME, label, C.DESTRUCTOR_STATIC);
            v.AddOp2(OP.ResultRow, mem, 1);
        }

#if !OMIT_FLAG_PRAGMAS
        struct sPragmaType
        {
            public string Name; // Name of the pragma
            public Context.FLAG Mask; // Mask for the db.flags value
            public sPragmaType(string name, Context.FLAG mask)
            {
                Name = name;
                Mask = mask;
            }
        }
        static readonly sPragmaType[] _pragmas = new sPragmaType[]
        {
            new sPragmaType("full_column_names",        Context.FLAG.FullColNames),
            new sPragmaType("short_column_names",       Context.FLAG.ShortColNames),
            new sPragmaType("count_changes",            Context.FLAG.CountRows),
            new sPragmaType("empty_result_callbacks",   Context.FLAG.NullCallback),
            new sPragmaType("legacy_file_format",       Context.FLAG.LegacyFileFmt),
            new sPragmaType("fullfsync",                Context.FLAG.FullFSync),
            new sPragmaType("checkpoint_fullfsync",     Context.FLAG.CkptFullFSync),
            new sPragmaType("reverse_unordered_selects", Context.FLAG.ReverseOrder),
            #if !OMIT_AUTOMATIC_INDEX
            new sPragmaType("automatic_index",          Context.FLAG.AutoIndex),
            #endif
            #if DEBUG
            new sPragmaType("sql_trace",                Context.FLAG.SqlTrace),
            new sPragmaType("vdbe_listing",             Context.FLAG.VdbeListing),
            new sPragmaType("vdbe_trace",               Context.FLAG.VdbeTrace),
		    new sPragmaType("vdbe_addoptrace",          Context.FLAG.VdbeAddopTrace),
		    new sPragmaType("vdbe_debug",               Context.FLAG.SqlTrace|Context.FLAG.VdbeListing|Context.FLAG.VdbeTrace),
            #endif
            #if !OMIT_CHECK
            new sPragmaType("ignore_check_constraints", Context.FLAG.IgnoreChecks),
            #endif
            // The following is VERY experimental
            new sPragmaType("writable_schema",          Context.FLAG.WriteSchema|Context.FLAG.RecoveryMode),

            // TODO: Maybe it shouldn't be possible to change the ReadUncommitted flag if there are any active statements.
            new sPragmaType( "read_uncommitted",         Context.FLAG.ReadUncommitted),
            new sPragmaType( "recursive_triggers",       Context.FLAG.RecTriggers),

            // This flag may only be set if both foreign-key and trigger support are present in the build.
            #if !OMIT_FOREIGN_KEY && !OMIT_TRIGGER
            new sPragmaType("foreign_keys",             Context.FLAG.ForeignKeys),
            #endif
        };

        static bool FlagPragma(Parse parse, string left, string right)
        {
            int i;
            sPragmaType p;
            for (i = 0; i < _pragmas.Length; i++)
            {
                p = _pragmas[i];
                if (string.Equals(left, p.Name, StringComparison.OrdinalIgnoreCase))
                {
                    Context ctx = parse.Ctx;
                    Vdbe v = parse.GetVdbe();
                    Debug.Assert(v != null); // Already allocated by sqlite3Pragma()
                    if (C._ALWAYS(v))
                    {
                        if (right == null)
                            ReturnSingleInt(parse, p.Name, ((ctx.Flags & p.Mask) != 0) ? 1 : 0);
                        else
                        {
                            Context.FLAG mask = p.Mask; // Mask of bits to set or clear.
                            if (ctx.AutoCommit == 0)
                                mask &= ~(Context.FLAG.ForeignKeys); // Foreign key support may not be enabled or disabled while not in auto-commit mode.
                            if (ConvertEx.GetBoolean(right, 0))
                                ctx.Flags |= mask;
                            else
                                ctx.Flags &= ~mask;
                            // Many of the flag-pragmas modify the code generated by the SQL compiler (eg. count_changes). So add an opcode to expire all
                            // compiled SQL statements after modifying a pragma value.
                            v.AddOp2(OP.Expire, 0, 0);
                        }
                    }
                    return true;
                }
            }
            return false;
        }
#endif

#if !OMIT_FOREIGN_KEY
        static string ActionName(OE action)
        {
            switch (action)
            {
                case OE.SetNull: return "SET NULL";
                case OE.SetDflt: return "SET DEFAULT";
                case OE.Cascade: return "CASCADE";
                case OE.Restrict: return "RESTRICT";
                default: Debug.Assert(action == OE.None); return "NO ACTION";
            }
        }
#endif

        static readonly string[] _modeNames = {
"delete", "persist", "off", "truncate", "memory"
#if !OMIT_WAL
, "wal"
#endif
};
        public static string JournalModename(IPager.JOURNALMODE mode)
        {
            Debug.Assert((int)IPager.JOURNALMODE.DELETE == 0);
            Debug.Assert((int)IPager.JOURNALMODE.PERSIST == 1);
            Debug.Assert((int)IPager.JOURNALMODE.OFF == 2);
            Debug.Assert((int)IPager.JOURNALMODE.TRUNCATE == 3);
            Debug.Assert((int)IPager.JOURNALMODE.JMEMORY == 4);
            Debug.Assert((int)IPager.JOURNALMODE.WAL == 5);
            Debug.Assert(mode >= 0 && (int)mode <= _modeNames.Length);
            if ((int)mode == _modeNames.Length) return null;
            return _modeNames[(int)mode];
        }

        class EncName
        {
            public string Name;
            public TEXTENCODE Encode;
            public EncName(string name, TEXTENCODE encode) { Name = name; Encode = encode; }
        };
        static EncName[] _encnames = new EncName[]
        {
            new EncName("UTF8",     TEXTENCODE.UTF8),
            new EncName("UTF-8",    TEXTENCODE.UTF8), // Must be element [1]
            new EncName("UTF-16le (not supported)", TEXTENCODE.UTF16LE), // Must be element [2]
            new EncName("UTF-16be (not supported)", TEXTENCODE.UTF16BE), // Must be element [3]
            new EncName("UTF16le (not supported)",  TEXTENCODE.UTF16LE),
            new EncName("UTF16be (not supported)",  TEXTENCODE.UTF16BE),
            new EncName("UTF-16 (not supported)",   0), // SQLITE_UTF16NATIVE
            new EncName("UTF16",    0), // SQLITE_UTF16NATIVE
            new EncName(null, 0 )
        };

        //#if !(ENABLE_LOCKING_STYLE)
        //#if (__APPLE__)
        ////#define ENABLE_LOCKING_STYLE 1
        //#else //#define ENABLE_LOCKING_STYLE 0
        //#endif
        //#endif

        const int INTEGRITY_CHECK_ERROR_MAX = 100;

        static readonly Vdbe.VdbeOpList[] _getCacheSize = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Transaction, 0, 0,        0),                         // 0
            new Vdbe.VdbeOpList(OP.ReadCookie,  0, 1,        BTREE_DEFAULT_CACHE_SIZE),  // 1
            new Vdbe.VdbeOpList(OP.IfPos,       1, 7,        0),
            new Vdbe.VdbeOpList(OP.Integer,     0, 2,        0),
            new Vdbe.VdbeOpList(OP.Subtract,    1, 2,        1),
            new Vdbe.VdbeOpList(OP.IfPos,       1, 7,        0),
            new Vdbe.VdbeOpList(OP.Integer,     0, 1,        0),  // 6
            new Vdbe.VdbeOpList(OP.ResultRow,   1, 1,        0),
        };
        static readonly Vdbe.VdbeOpList[] _setMeta6 = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Transaction,    0,               1,        0),    // 0
            new Vdbe.VdbeOpList(OP.ReadCookie,     0,               1,        BTREE_LARGEST_ROOT_PAGE),    // 1
            new Vdbe.VdbeOpList(OP.If,             1,               0,        0),    // 2
            new Vdbe.VdbeOpList(OP.Halt,           RC.OK,       OE.Abort, 0),    // 3
            new Vdbe.VdbeOpList(OP.Integer,        0,               1,        0),    // 4
            new Vdbe.VdbeOpList(OP.SetCookie,      0,               BTREE_INCR_VACUUM, 1),    // 5
        };


        // Code that appears at the end of the integrity check.  If no error messages have been generated, output OK.  Otherwise output the error message
        static readonly Vdbe.VdbeOpList[] _endCode = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.AddImm,      1, 0,        0),    // 0
            new Vdbe.VdbeOpList(OP.IfNeg,       1, 0,        0),    // 1
            new Vdbe.VdbeOpList(OP.String8,     0, 3,        0),    // 2
            new Vdbe.VdbeOpList(OP.ResultRow,   3, 1,        0),
        };
        static readonly Vdbe.VdbeOpList[] _idxErr = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.AddImm,      1, -1,  0),
            new Vdbe.VdbeOpList(OP.String8,     0,  3,  0),    // 1
            new Vdbe.VdbeOpList(OP.Rowid,       1,  4,  0),
            new Vdbe.VdbeOpList(OP.String8,     0,  5,  0),    // 3
            new Vdbe.VdbeOpList(OP.String8,     0,  6,  0),    // 4
            new Vdbe.VdbeOpList(OP.Concat,      4,  3,  3),
            new Vdbe.VdbeOpList(OP.Concat,      5,  3,  3),
            new Vdbe.VdbeOpList(OP.Concat,      6,  3,  3),
            new Vdbe.VdbeOpList(OP.ResultRow,   3,  1,  0),
            new Vdbe.VdbeOpList(OP.IfPos,       1,  0,  0),    // 9
            new Vdbe.VdbeOpList(OP.Halt,        0,  0,  0),
        };
        static readonly Vdbe.VdbeOpList[] _cntIdx = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Integer,      0,  3,  0),
            new Vdbe.VdbeOpList(OP.Rewind,       0,  0,  0),  // 1
            new Vdbe.VdbeOpList(OP.AddImm,       3,  1,  0),
            new Vdbe.VdbeOpList(OP.Next,         0,  0,  0),  // 3
            new Vdbe.VdbeOpList(OP.Eq,           2,  0,  3),  // 4
            new Vdbe.VdbeOpList(OP.AddImm,       1, -1,  0),
            new Vdbe.VdbeOpList(OP.String8,      0,  2,  0),  // 6
            new Vdbe.VdbeOpList(OP.String8,      0,  3,  0),  // 7
            new Vdbe.VdbeOpList(OP.Concat,       3,  2,  2),
            new Vdbe.VdbeOpList(OP.ResultRow,    2,  1,  0),
        };
        // Write the specified cookie value
        static readonly Vdbe.VdbeOpList[] _setCookie = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Transaction,    0,  1,  0),    // 0
            new Vdbe.VdbeOpList(OP.Integer,        0,  1,  0),    // 1
            new Vdbe.VdbeOpList(OP.SetCookie,      0,  0,  1),    // 2
        };
        // Read the specified cookie value
        static readonly Vdbe.VdbeOpList[] _readCookie = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Transaction,     0,  0,  0),    // 0
            new Vdbe.VdbeOpList(OP.ReadCookie,      0,  1,  0),    // 1
            new Vdbe.VdbeOpList(OP.ResultRow,       1,  1,  0)
        };
        static readonly string[] _lockNames =
        {
            "unlocked", "shared", "reserved", "pending", "exclusive"
        };

        // OVERLOADS, so I don't need to rewrite parse.c
        public static void Pragma_(Parse parse, Token id1, Token id2, int null4, bool minusFlag) { Pragma_(parse, id1, id2, null, minusFlag); }
        public static void Pragma_(Parse parse, Token id1, Token id2, Token value, bool minusFlag)
        {
            Context ctx = parse.Ctx;
            Vdbe v = parse.V = Vdbe.Create(ctx);
            if (v == null) return;
            v.RunOnlyOnce();
            parse.Mems = 2;

            // Interpret the [database.] part of the pragma statement. iDb is the index of the database this pragma is being applied to in db.aDb[].
            Token id = new Token(); // Pointer to <id> token
            int db = parse.TwoPartName(id1, id2, ref id); // Database index for <database>
            if (db < 0) return;
            Context.DB dbAsObj = ctx.DBs[db]; // The specific database being pragmaed

            // If the temp database has been explicitly named as part of the pragma, make sure it is open.
            if (db == 1 && parse.OpenTempDatabase() != 0)
                return;

            string left = Parse.NameFromToken(ctx, id); // Nul-terminated UTF-8 string <id>
            if (string.IsNullOrEmpty(left)) return;
            string right = (minusFlag ? C._mtagprintf(ctx, "-%T", value) : Parse.NameFromToken(ctx, value)); // Nul-terminated UTF-8 string <value>, or NULL

            Debug.Assert(id2 != null);
            string dbName = (dbName = id2.length > 0 ? dbAsObj.Name : null); // The database name
            if (Auth.Check(parse, AUTH.PRAGMA, left, right, dbName) != ARC.OK)
                goto pragma_out;

            // Send an SQLITE_FCNTL_PRAGMA file-control to the underlying VFS connection.  If it returns SQLITE_OK, then assume that the VFS
            // handled the pragma and generate a no-op prepared statement.
            string[] fcntls = new string[4]; // Argument to FCNTL_PRAGMA
            fcntls[0] = null;
            fcntls[1] = left;
            fcntls[2] = right;
            fcntls[3] = null;
            ctx.BusyHandler.Busys = 0;
            RC rc = Main.FileControl(ctx, dbName, VFile.FCNTL.PRAGMA, ref fcntls); // return value form SQLITE_FCNTL_PRAGMA
            if (rc == RC.OK)
            {
                if (fcntls[0] != null)
                {
                    int mem = ++parse.Mems;
                    v.AddOp4(OP.String8, 0, mem, 0, fcntls[0], 0);
                    v.SetNumCols(1);
                    v.SetColName(0, COLNAME_NAME, "result", C.DESTRUCTOR_STATIC);
                    v.AddOp2(OP.ResultRow, mem, 1);
                    C._free(ref fcntls[0]);
                }
            }
            else if (rc != RC.NOTFOUND)
            {
                if (fcntls[0] != null)
                {
                    parse.ErrorMsg("%s", fcntls[0]);
                    C._free(ref fcntls[0]);
                }
                parse.Errs++;
                parse.RC = rc;
            }


#if !OMIT_PAGER_PRAGMAS && !OMIT_DEPRECATED
            //  PRAGMA [database.]default_cache_size
            //  PRAGMA [database.]default_cache_size=N
            //
            // The first form reports the current persistent setting for the page cache size.  The value returned is the maximum number of
            // pages in the page cache.  The second form sets both the current page cache size value and the persistent page cache size value
            // stored in the database file.
            //
            // Older versions of SQLite would set the default cache size to a negative number to indicate synchronous=OFF.  These days, synchronous
            // is always on by default regardless of the sign of the default cache size.  But continue to take the absolute value of the default cache
            // size of historical compatibility.
            if (string.Equals(left, "default_cache_size", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                v.UsesBtree(db);
                if (right == null)
                {
                    v.SetNumCols(1);
                    v.SetColName(0, COLNAME_NAME, "cache_size", C.DESTRUCTOR_STATIC);
                    parse.Mems += 2;
                    int addr = v.AddOpList(_getCacheSize.Length, _getCacheSize);
                    v.ChangeP1(addr, db);
                    v.ChangeP1(addr + 1, db);
                    v.ChangeP1(addr + 6, SQLITE_DEFAULT_CACHE_SIZE);
                }
                else
                {
                    int size = ConvertEx.AbsInt32(ConvertEx.Atoi(right));
                    parse.BeginWriteOperation(0, db);
                    v.AddOp2(OP.Integer, size, 1);
                    v.AddOp3(OP.SetCookie, db, BTREE_DEFAULT_CACHE_SIZE, 1);
                    Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                    dbAsObj.Schema.CacheSize = size;
                    dbAsObj.Bt.SetCacheSize(dbAsObj.Schema.CacheSize);
                }
            }
#endif

#if !OMIT_PAGER_PRAGMAS
            //  PRAGMA [database.]page_size
            //  PRAGMA [database.]page_size=N
            //
            // The first form reports the current setting for the database page size in bytes.  The second form sets the
            // database page size value.  The value can only be set if the database has not yet been created.
            else if (string.Equals(left, "page_size", StringComparison.OrdinalIgnoreCase))
            {
                Btree bt = dbAsObj.Bt;
                Debug.Assert(bt != null);
                if (right == null)
                {
                    int size = (C._ALWAYS(bt != null) ? bt.GetPageSize() : 0);
                    ReturnSingleInt(parse, "page_size", size);
                }
                else
                {
                    // Malloc may fail when setting the page-size, as there is an internal buffer that the pager module resizes using sqlite3_realloc().
                    ctx.NextPagesize = ConvertEx.Atoi(right);
                    if (bt.SetPageSize(ctx.NextPagesize, -1, false) == RC.NOMEM)
                        ctx.MallocFailed = true;
                }
            }
            //  PRAGMA [database.]secure_delete
            //  PRAGMA [database.]secure_delete=ON/OFF
            //
            // The first form reports the current setting for the secure_delete flag.  The second form changes the secure_delete
            // flag setting and reports thenew value.
            else if (string.Equals(left, "secure_delete", StringComparison.OrdinalIgnoreCase))
            {
                Btree bt = dbAsObj.Bt;
                Debug.Assert(bt != null);
                int b = (right != null ? (ConvertEx.GetBoolean(right, 0) ? 1: 0) : -1);
                if (id2.length == 0 && b >= 0)
                    for (int ii = 0; ii < ctx.DBs.length; ii++)
                        ctx.DBs[ii].Bt.SecureDelete(b);
                b = bt.SecureDelete(b);
                ReturnSingleInt(parse, "secure_delete", b);
            }
            //  PRAGMA [database.]max_page_count
            //  PRAGMA [database.]max_page_count=N
            //
            // The first form reports the current setting for the maximum number of pages in the database file.  The 
            // second form attempts to change this setting.  Both forms return the current setting.
            //
            //  PRAGMA [database.]page_count
            //
            // Return the number of pages in the specified database.
            else if (string.Equals(left, "page_count", StringComparison.OrdinalIgnoreCase) || string.Equals(left, "max_page_count", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                parse.CodeVerifySchema(db);
                int regId = ++parse.Mems;
                if (left[0] == 'p')
                    v.AddOp2(OP.Pagecount, db, regId);
                else
                    v.AddOp3(OP.MaxPgcnt, db, regId, ConvertEx.Atoi(right));
                v.AddOp2(OP.ResultRow, regId, 1);
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, left, C.DESTRUCTOR_TRANSIENT);
            }

                //  PRAGMA [database.]locking_mode
            //  PRAGMA [database.]locking_mode = (normal|exclusive)
            else if (left.Equals("locking_mode", StringComparison.OrdinalIgnoreCase))
            {
                string ret = "normal";
                IPager.LOCKINGMODE mode = GetLockingMode(right);
                if (id2.length == 0 && mode == IPager.LOCKINGMODE.QUERY)
                    mode = ctx.DefaultLockMode; // Simple "PRAGMA locking_mode;" statement. This is a query for the current default locking mode (which may be different to the locking-mode of the main database).
                else
                {
                    Pager pager;
                    if (id2.length == 0)
                    {
                        // This indicates that no database name was specified as part of the PRAGMA command. In this case the locking-mode must be
                        // set on all attached databases, as well as the main db file.
                        //
                        // Also, the sqlite3.dfltLockMode variable is set so that any subsequently attached databases also use the specified
                        // locking mode.
                        Debug.Assert(dbAsObj == ctx.DBs[0]);
                        for (int ii = 2; ii < ctx.DBs.length; ii++)
                        {
                            pager = ctx.DBs[ii].Bt.get_Pager();
                            pager.LockingMode(mode);
                        }
                        ctx.DefaultLockMode = mode;
                    }
                    pager = dbAsObj.Bt.get_Pager();
                    mode = (pager.LockingMode(mode) ? 1 : 0);
                }

                Debug.Assert(mode == IPager.LOCKINGMODE.NORMAL || mode == IPager.LOCKINGMODE.EXCLUSIVE);
                if (mode == IPager.LOCKINGMODE.EXCLUSIVE)
                    ret = "exclusive";
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "locking_mode", C.DESTRUCTOR_STATIC);
                v.AddOp4(OP.String8, 0, 1, 0, ret, 0);
                v.AddOp2(OP.ResultRow, 1, 1);
            }

                //  PRAGMA [database.]journal_mode
            //  PRAGMA [database.]journal_mode = (delete|persist|off|truncate|memory|wal|off)
            else if (string.Equals(left, "journal_mode", StringComparison.OrdinalIgnoreCase))
            {
                // Force the schema to be loaded on all databases.  This causes all database files to be opened and the journal_modes set.  This is
                // necessary because subsequent processing must know if the databases are in WAL mode.
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "journal_mode", C.DESTRUCTOR_STATIC);

                IPager.JOURNALMODE mode; // One of the PAGER_JOURNALMODE_XXX symbols
                if (right == null)
                    mode = IPager.JOURNALMODE.JQUERY; // If there is no "=MODE" part of the pragma, do a query for the current mode
                else
                {
                    string modeName;
                    int n = right.Length;
                    for (mode = 0; (modeName = JournalModename(mode)) != null; mode++)
                        if (string.Compare(right, 0, modeName, 0, n, StringComparison.OrdinalIgnoreCase) == 0) break;
                    if (modeName == null)
                        mode = IPager.JOURNALMODE.JQUERY;  // If the "=MODE" part does not match any known journal mode, then do a query
                }
                if (mode == IPager.JOURNALMODE.JQUERY && id2.length == 0) // Convert "PRAGMA journal_mode" into "PRAGMA main.journal_mode"
                {
                    db = 0;
                    id2.length = 1;
                }
                for (int ii = ctx.DBs.length - 1; ii >= 0; ii--)
                {
                    if (ctx.DBs[ii].Bt != null && (ii == db || id2.length == 0))
                    {
                        v.UsesBtree(v, ii);
                        v.AddOp3(OP.JournalMode, ii, 1, mode);
                    }
                }
                v.AddOp2(OP.ResultRow, 1, 1);
            }

                //  PRAGMA [database.]journal_size_limit
            //  PRAGMA [database.]journal_size_limit=N
            //
            // Get or set the size limit on rollback journal files.
            else if (string.Equals(left, "journal_size_limit", StringComparison.OrdinalIgnoreCase))
            {
                Pager pager = dbAsObj.Bt.get_Pager();
                long limit = -2;
                if (right == null)
                {
                    ConvertEx.Atoi64(right, ref limit, 1000000, TEXTENCODE.UTF8);
                    if (limit < -1) limit = -1;
                }
                limit = pager.SetJournalSizeLimit(limit);
                ReturnSingleInt(parse, "journal_size_limit", limit);
            }
#endif

#if !OMIT_AUTOVACUUM
            //  PRAGMA [database.]auto_vacuum
            //  PRAGMA [database.]auto_vacuum=N
            //
            // Get or set the value of the database 'auto-vacuum' parameter. The value is one of:  0 NONE 1 FULL 2 INCREMENTAL
            else if (string.Equals(left, "auto_vacuum", StringComparison.OrdinalIgnoreCase))
            {
                Btree bt = dbAsObj.Bt;
                Debug.Assert(bt != null);
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                if (right == null)
                {
                    Btree.AUTOVACUUM auto_ = (C._ALWAYS(bt) ? bt.GetAutoVacuum() : DEFAULT_AUTOVACUUM);
                    ReturnSingleInt(parse, "auto_vacuum", (int)auto_);
                }
                else
                {
                    Btree.AUTOVACUUM auto_ = GetAutoVacuum(right);
                    Debug.Assert((int)auto_ >= 0 && (int)auto_ <= 2);
                    ctx.NextAutovac = auto_;
                    if (C._ALWAYS((int)auto_ >= 0))
                    {
                        // Call SetAutoVacuum() to set initialize the internal auto and incr-vacuum flags. This is required in case this connection
                        // creates the database file. It is important that it is created as an auto-vacuum capable db.
                        rc = bt.SetAutoVacuum(auto_);
                        if (rc == RC.OK && (auto_ == Btree.AUTOVACUUM.FULL || auto_ == Btree.AUTOVACUUM.INCR))
                        {
                            // When setting the auto_vacuum mode to either "full" or  "incremental", write the value of meta[6] in the database
                            // file. Before writing to meta[6], check that meta[3] indicates that this really is an auto-vacuum capable database.
                            int addr = v.AddOpList(v, _setMeta6.Length, setMeta6);
                            v.ChangeP1(addr, db);
                            v.ChangeP1(addr + 1, db);
                            v.ChangeP2(addr + 2, addr + 4);
                            v.ChangeP1(addr + 4, auto_ - 1);
                            v.ChangeP1(addr + 5, db);
                            v.UsesBtree(db);
                        }
                    }
                }
            }

                //  PRAGMA [database.]incremental_vacuum(N)
            //
            // Do N steps of incremental vacuuming on a database.
            else if (string.Equals(left, "incremental_vacuum", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                int limit = 0;
                if (right == null || !ConvertEx.Atoi(right, ref limit) || limit <= 0)
                    limit = 0x7fffffff;
                parse.BeginWriteOperation(0, db);
                v.AddOp2(OP.Integer, limit, 1);
                int addr = v.AddOp1(OP.IncrVacuum, db);
                v.AddOp1(OP.ResultRow, 1);
                v.AddOp2(OP.AddImm, 1, -1);
                v.AddOp2(OP.IfPos, 1, addr);
                v.JumpHere(addr);
            }

#endif

#if !OMIT_PAGER_PRAGMAS
            //  PRAGMA [database.]cache_size
            //  PRAGMA [database.]cache_size=N
            //
            // The first form reports the current local setting for the page cache size. The second form sets the local
            // page cache size value.  If N is positive then that is the number of pages in the cache.  If N is negative, then the
            // number of pages is adjusted so that the cache uses -N kibibytes of memory.
            else if (string.Equals(left, "cache_size", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                if (right == null)
                    ReturnSingleInt(parse, "cache_size", dbAsObj.Schema.CacheSize);
                else
                {
                    int size = ConvertEx.Atoi(right);
                    dbAsObj.Schema.CacheSize = size;
                    dbAsObj.Bt.SetCacheSize(dbAsObj.Schema.CacheSize);
                }
            }

            //   PRAGMA temp_store
            //   PRAGMA temp_store = "default"|"memory"|"file"
            //
            // Return or set the local value of the temp_store flag.  Changing the local value does not make changes to the disk file and the default
            // value will be restored the next time the database is opened.
            //
            // Note that it is possible for the library compile-time options to override this setting
            else if (string.Equals(left, "temp_store", StringComparison.OrdinalIgnoreCase))
            {
                if (right == null)
                    ReturnSingleInt(parse, "temp_store", ctx.TempStore);
                else
                    ChangeTempStorage(parse, right);
            }

            //   PRAGMA temp_store_directory
            //   PRAGMA temp_store_directory = ""|"directory_name"
            //
            // Return or set the local value of the temp_store_directory flag.  Changing the value sets a specific directory to be used for temporary files.
            // Setting to a null string reverts to the default temporary directory search. If temporary directory is changed, then invalidateTempStorage.
            else if (string.Equals(left, "temp_store_directory", StringComparison.OrdinalIgnoreCase))
            {
                if (right == null)
                {
                    if (Main.g_temp_directory != null)
                    {
                        v.SetNumCols(1);
                        v.SetColName(0, COLNAME_NAME, "temp_store_directory", C.DESTRUCTOR_STATIC);
                        v.AddOp4(OP.String8, 0, 1, 0, Main.g_temp_directory, 0);
                        v.AddOp2(OP.ResultRow, 1, 1);
                    }
                }
                else
                {
#if !OMIT_WSD
                    if (right.Length > 0)
                    {
                        int res = 0;
                        rc = ctx.Vfs.Access(right, VSystem.ACCESS.READWRITE, ref res);
                        if (rc != RC.OK || res == 0)
                        {
                            parse.ErrorMsg("not a writable directory");
                            goto pragma_out;
                        }
                    }
                    if (TEMP_STORE == 0 || (TEMP_STORE == 1 && ctx.TempStore <= 1) || (TEMP_STORE == 2 && ctx.TempStore == 1))
                        InvalidateTempStorage(parse);
                    C._free(ref Main.g_temp_directory);
                    Main.g_temp_directory = (right.Length > 0 ? right : null);
#endif
                }
            }

#if OS_WIN
            //   PRAGMA data_store_directory
            //   PRAGMA data_store_directory = ""|"directory_name"
            //
            // Return or set the local value of the data_store_directory flag.  Changing the value sets a specific directory to be used for database files that
            // were specified with a relative pathname.  Setting to a null string reverts to the default database directory, which for database files specified with
            // a relative path will probably be based on the current directory for the process.  Database file specified with an absolute path are not impacted
            // by this setting, regardless of its value.
            else if (string.Equals(left, "data_store_directory", StringComparison.OrdinalIgnoreCase))
            {
                if (right == null)
                {
                    if (sqlite3_data_directory == null)
                    {
                        v.SetNumCols(1);
                        v.SetColName(0, COLNAME_NAME, "data_store_directory", C.DESTRUCTOR_STATIC);
                        v.AddOp4(OP.String8, 0, 1, 0, Main.g_data_directory, 0);
                        v.AddOp2(OP.ResultRow, 1, 1);
                    }
                }
                else
                {
#if !OMIT_WSD
                    if (right.Length > 0)
                    {
                        int res;
                        rc = ctx.Vfs.Access(right, VSystem.ACCESS.READWRITE, ref res);
                        if (rc != RC.OK || res == 0)
                        {
                            parse.ErrorMsg("not a writable directory");
                            goto pragma_out;
                        }
                    }
                    C._free(ref sqlite3_data_directory);
                    Main.g_data_directory = (right.Length > 0 ? right : null);
#endif
                }
            }
#endif

#if !ENABLE_LOCKING_STYLE
            //   PRAGMA [database.]lock_proxy_file
            //   PRAGMA [database.]lock_proxy_file = ":auto:"|"lock_file_path"
            //
            // Return or set the value of the lock_proxy_file flag.  Changing the value sets a specific file to be used for database access locks.
            else if (string.Equals(left, "lock_proxy_file", StringComparison.OrdinalIgnoreCase))
            {
                if (right != null)
                {
                    Pager pager = dbAsObj.Bt.get_Pager();
                    IO.VFile file = pager.get_File();
                    int proxy_file_path = 0;
                    file.FileControl(SQLITE_GET_LOCKPROXYFILE, ref proxy_file_path);
                    if (proxy_file_path != 0)
                    {
                        v.SetNumCols(1);
                        v.SetColName(0, COLNAME_NAME, "lock_proxy_file", C.DESTRUCTOR_STATIC);
                        v.AddOp4(OP.String8, 0, 1, 0, proxy_file_path, 0);
                        v.AddOp2(OP.ResultRow, 1, 1);
                    }
                }
                else
                {
                    Pager pager = dbAsObj.Bt.get_Pager();
                    IO.VFile file = pager.get_File();
                    int dummy1 = (right.Length > 0 ? right : null);
                    rc = file.FileControl(SQLITE_SET_LOCKPROXYFILE, ref dummy1);
                    if (rc != RC.OK)
                    {
                        parse.ErrorMsg("failed to set lock proxy file");
                        goto pragma_out;
                    }
                }
            }
#endif

            //   PRAGMA [database.]synchronous
            //   PRAGMA [database.]synchronous=OFF|ON|NORMAL|FULL
            //
            // Return or set the local value of the synchronous flag.  Changing the local value does not make changes to the disk file and the
            // default value will be restored the next time the database is opened.
            else if (string.Equals(left, "synchronous", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                if (right == null)
                    ReturnSingleInt(parse, "synchronous", dbAsObj.SafetyLevel - 1);
                else
                {
                    if (ctx.AutoCommit == 0)
                        parse.ErrorMsg("Safety level may not be changed inside a transaction");
                    else
                        dbAsObj.SafetyLevel = (byte)(ConvertEx.GetSafetyLevel(right, 0, 1) + 1);
                }
            }

#endif

#if !OMIT_FLAG_PRAGMAS
            // The FlagPragma() subroutine also generates any necessary code there is nothing more to do here
            else if (FlagPragma(parse, left, right) != 0) { }
#endif

#if !OMIT_SCHEMA_PRAGMAS
            //   PRAGMA table_info(<table>)
            //
            // Return a single row for each column of the named table. The columns of the returned data set are:
            //
            // cid:        Column id (numbered from left to right, starting at 0)
            // name:       Column name
            // type:       Column declaration type.
            // notnull:    True if 'NOT NULL' is part of column declaration
            // dflt_value: The default value for the column, if any.
            else if (string.Equals(left, "table_info", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                Table table = Parse.FindTable(ctx, right, dbName);
                if (table != null)
                {
                    Index pk;
                    for (pk = table.Index; pk != null && pk.AutoIndex != 2; pk = pk.Next) { }
                    v.SetNumCols(6);
                    parse.Mems = 6;
                    parse.CodeVerifySchema(db);
                    v.SetColName(0, COLNAME_NAME, "cid", C.DESTRUCTOR_STATIC);
                    v.SetColName(1, COLNAME_NAME, "name", C.DESTRUCTOR_STATIC);
                    v.SetColName(2, COLNAME_NAME, "type", C.DESTRUCTOR_STATIC);
                    v.SetColName(3, COLNAME_NAME, "notnull", C.DESTRUCTOR_STATIC);
                    v.SetColName(4, COLNAME_NAME, "dflt_value", C.DESTRUCTOR_STATIC);
                    v.SetColName(5, COLNAME_NAME, "pk", C.DESTRUCTOR_STATIC);
                    parse.ViewGetColumnNames(table);
                    int hidden = 0;
                    for (int i = 0; i < table.Cols.length; i++)
                    {
                        Column col = table.Cols[i];
                        if (E.IsHiddenColumn(col))
                        {
                            hidden++;
                            continue;
                        }
                        v.AddOp2(OP.Integer, i - hidden, 1);
                        v.AddOp4(OP.String8, 0, 2, 0, colName, 0);
                        v.AddOp4(OP.String8, 0, 3, 0, (col.Type != null ? col.Type : string.Empty), 0);
                        v.AddOp2(OP.Integer, (col.NotNull != 0 ? 1 : 0), 4);
                        if (col.DfltName != null)
                            v.AddOp4(OP.String8, 0, 5, 0, col.DfltName, 0);
                        else
                            v.AddOp2(OP.Null, 0, 5);
                        int k;
                        if ((col.ColFlags & COLFLAG.PRIMKEY) == 0)
                            k = 0;
                        else if (pk == null)
                            k = 1;
                        else
                        {
                            for (k = 1; C._ALWAYS(k <= table.Cols.length) && pk.Columns[k - 1] != i; k++) { }
                        }
                        v.AddOp2(OP.Integer, k, 6);
                        v.AddOp2(OP.ResultRow, 1, 6);
                    }
                }
            }
            else if (string.Equals(left, "index_info", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                Index index = Parse.FindIndex(ctx, right, dbName);
                if (index != null)
                {
                    Table table = index.Table;
                    v.SetNumCols(3);
                    parse.Mems = 3;
                    parse.CodeVerifySchema(db);
                    v.SetColName(0, COLNAME_NAME, "seqno", C.DESTRUCTOR_STATIC);
                    v.SetColName(1, COLNAME_NAME, "cid", C.DESTRUCTOR_STATIC);
                    v.SetColName(2, COLNAME_NAME, "name", C.DESTRUCTOR_STATIC);
                    for (int i = 0; i < index.Columns.length; i++)
                    {
                        int cid = index.Columns[i];
                        v.AddOp2(OP.Integer, i, 1);
                        v.AddOp2(OP.Integer, cid, 2);
                        Debug.Assert(table.Cols.length > cid);
                        v.AddOp4(OP.String8, 0, 3, 0, table.Cols[cid].Name, 0);
                        v.AddOp2(OP.ResultRow, 1, 3);
                    }
                }
            }
            else if (string.Equals(left, "index_list", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                Table table = Parse.FindTable(ctx, right, dbName);
                if (table != null)
                {
                    v = parse.GetVdbe();
                    Index index = table.Index;
                    if (index != null)
                    {
                        v.SetNumCols(3);
                        parse.Mems = 3;
                        parse.CodeVerifySchema(db);
                        v.SetColName(0, COLNAME_NAME, "seq", C.DESTRUCTOR_STATIC);
                        v.SetColName(1, COLNAME_NAME, "name", C.DESTRUCTOR_STATIC);
                        v.SetColName(2, COLNAME_NAME, "unique", C.DESTRUCTOR_STATIC);
                        int i = 0;
                        while (index != null)
                        {
                            v.AddOp2(OP.Integer, i, 1);
                            v.AddOp4(OP.String8, 0, 2, 0, index.Name, 0);
                            v.AddOp2(OP.Integer, (index.OnError != OE.None ? 1 : 0), 3);
                            v.AddOp2(OP.ResultRow, 1, 3);
                            ++i;
                            index = index.Next;
                        }
                    }
                }
            }
            else if (string.Equals(left, "database_list", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                v.SetNumCols(3);
                parse.Mems = 3;
                v.SetColName(0, COLNAME_NAME, "seq", C.DESTRUCTOR_STATIC);
                v.SetColName(1, COLNAME_NAME, "name", C.DESTRUCTOR_STATIC);
                v.SetColName(2, COLNAME_NAME, "file", C.DESTRUCTOR_STATIC);
                for (int i = 0; i < ctx.DBs.length; i++)
                {
                    if (ctx.DBs[i].Bt == null) continue;
                    Debug.Assert(ctx.DBs[i].Name != null);
                    v.AddOp2(OP.Integer, i, 1);
                    v.AddOp4(OP.String8, 0, 2, 0, ctx.DBs[i].Name, 0);
                    v.AddOp4(OP.String8, 0, 3, 0, ctx.DBs[i].Bt.get_Filename(), 0);
                    v.AddOp2(OP.ResultRow, 1, 3);
                }
            }
            else if (string.Equals(left, "collation_list", StringComparison.OrdinalIgnoreCase))
            {
                v.SetNumCols(2);
                parse.Mems = 2;
                v.SetColName(0, COLNAME_NAME, "seq", C.DESTRUCTOR_STATIC);
                v.SetColName(1, COLNAME_NAME, "name", C.DESTRUCTOR_STATIC);
                int i = 0;
                for (HashElem p = ctx.CollSeqs.First; p != null; p = p.Next)
                {
                    CollSeq coll = ((CollSeq[])p.Data)[0];
                    v.AddOp2(OP.Integer, i++, 1);
                    v.AddOp4(OP.String8, 0, 2, 0, coll.Name, 0);
                    v.AddOp2(OP.ResultRow, 1, 2);
                }
            }
#endif
#if !OMIT_FOREIGN_KEY
            else if (string.Equals(left, "foreign_key_list", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                Table table = Parse.FindTable(ctx, right, dbName);
                if (table != null)
                {
                    v = parse.GetVdbe();
                    FKey fk = table.FKeys;
                    if (fk != null)
                    {
                        v.SetNumCols(8);
                        parse.Mems = 8;
                        parse.VerifySchema(db);
                        v.SetColName(0, COLNAME_NAME, "id", C.DESTRUCTOR_STATIC);
                        v.SetColName(1, COLNAME_NAME, "seq", C.DESTRUCTOR_STATIC);
                        v.SetColName(2, COLNAME_NAME, "table", C.DESTRUCTOR_STATIC);
                        v.SetColName(3, COLNAME_NAME, "from", C.DESTRUCTOR_STATIC);
                        v.SetColName(4, COLNAME_NAME, "to", C.DESTRUCTOR_STATIC);
                        v.SetColName(5, COLNAME_NAME, "on_update", C.DESTRUCTOR_STATIC);
                        v.SetColName(6, COLNAME_NAME, "on_delete", C.DESTRUCTOR_STATIC);
                        v.SetColName(7, COLNAME_NAME, "match", C.DESTRUCTOR_STATIC);
                        int i = 0;
                        while (fk != null)
                        {
                            for (int j = 0; j < fk.Cols.length; j++)
                            {
                                string colName = fk.Cols[j].Col;
                                string onDelete = ActionName(fk.Actions[0]);
                                string onUpdate = ActionName(fk.Actions[1]);
                                v.AddOp2(OP.Integer, i, 1);
                                v.AddOp2(OP.Integer, j, 2);
                                v.AddOp4(OP.String8, 0, 3, 0, fk.To, 0);
                                v.AddOp4(OP.String8, 0, 4, 0, table.Cols[fk.Cols[j].From].Name, 0);
                                v.AddOp4(colName != null ? OP.String8 : OP.Null, 0, 5, 0, colName, 0);
                                v.AddOp4(OP.String8, 0, 6, 0, onUpdate, 0);
                                v.AddOp4(OP.String8, 0, 7, 0, onDelete, 0);
                                v.AddOp4(OP.String8, 0, 8, 0, "NONE", 0);
                                v.AddOp2(OP.ResultRow, 1, 8);
                            }
                            ++i;
                            fk = fk.NextFrom;
                        }
                    }
                }
            }
#if !OMIT_TRIGGER
            else if (string.Equals(left, "foreign_key_check", StringComparison.OrdinalIgnoreCase))
            {
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                int regResult = parse.Mems + 1; // 3 registers to hold a result row
                parse.Mems += 4;
                int regKey = ++parse.Mems; // Register to hold key for checking the FK
                int regRow = ++parse.Mems; // Registers to hold a row from pTab
                v = parse.GetVdbe();
                v.SetNumCols(4);
                v.SetColName(0, COLNAME_NAME, "table", C.DESTRUCTOR_STATIC);
                v.SetColName(1, COLNAME_NAME, "rowid", C.DESTRUCTOR_STATIC);
                v.SetColName(2, COLNAME_NAME, "parent", C.DESTRUCTOR_STATIC);
                v.SetColName(3, COLNAME_NAME, "fkid", C.DESTRUCTOR_STATIC);
                parse.CodeVerifySchema(db);
                HashElem k = ctx.DBs[db].Schema.TableHash.First; // Loop counter:  Next table in schema
                int i; // Loop counter:  Foreign key number for pTab
                while (k != null)
                {
                    Table table; // Child table contain "REFERENCES" keyword
                    if (right != null)
                    {
                        table = parse.LocateTable(false, right, dbName);
                        k = null;
                    }
                    else
                    {
                        table = (Table)k.Data;
                        k = k.Next;
                    }
                    if (table == null || table.FKeys == null) continue;
                    parse.TableLock(db, table.Id, false, table.Name);
                    if (table.Cols.length + regRow > parse.Mems) parse.Mems = table.Cols.length + regRow;
                    sqlite3OpenTable(parse, 0, db, table, OP.OpenRead);
                    v.AddOp4(OP.String8, 0, regResult, 0, table.Name, Vdbe.P4T.TRANSIENT);
                    Table parent; // Parent table that child points to
                    Index index; // Index in the parent table
                    int[] cols; // child to parent column mapping
                    int x; // result variable
                    FKey fk; // A foreign key constraint
                    for (i = 1, fk = table.FKeys[0]; fk != null; i++, fk = fk.NextFrom)
                    {
                        parent = parse.LocateTable(false, fk.To, dbName);
                        if (parent == null) break;
                        index = null;
                        parse.TableLock(db, parent.Id, false, parent.Name);
                        x = parse.FKLocateIndex(parent, fk, ref index, null);
                        if (x == 0)
                        {
                            if (index == null)
                                sqlite3OpenTable(parse, i, db, parent, OP.OpenRead);
                            else
                            {
                                KeyInfo key = parse.IndexKeyinfo(index);
                                v.AddOp3(OP.OpenRead, i, index.Id, db);
                                v.ChangeP4(-1, (string)key, Vdbe.P4T.KEYINFO_HANDOFF);
                            }
                        }
                        else
                        {
                            k = null;
                            break;
                        }
                    }
                    if (fk != null) break;
                    if (parse.Tabs < i) parse.Tabs = i;
                    int addrTop = v.AddOp1(OP.Rewind, 0); // Top of a loop checking foreign keys
                    for (i = 1, fk = table.FKeys[0]; fk != null; i++, fk = fk.NextFrom)
                    {
                        parent = parse.LocateTable(false, fk.To, dbName);
                        Debug.Assert(parent != null);
                        index = null;
                        cols = null;
                        x = parse.FkLocateIndex(parent, fk, ref index, ref cols);
                        Debug.Assert(x == 0);
                        int addrOk = v.MakeLabel(); // Jump here if the key is OK
                        if (index == null)
                        {
                            int keyId = fk.Cols[0].From;
                            Debug.Assert(keyId >= 0 && keyId < table.Cols.length);
                            if (keyId != table.PKey)
                            {
                                v.AddOp3(OP.Column, 0, keyId, regRow);
                                sqlite3ColumnDefault(v, table, keyId, regRow);
                                v.AddOp2(OP.IsNull, regRow, addrOk);
                                v.AddOp2(OP.MustBeInt, regRow, v.CurrentAddr() + 3);
                            }
                            else
                                v.AddOp2(OP.Rowid, 0, regRow);
                            v.AddOp3(OP.NotExists, i, 0, regRow);
                            v.AddOp2(OP.Goto, 0, addrOk);
                            v.JumpHere(v.CurrentAddr() - 2);
                        }
                        else
                        {
                            for (int j = 0; j < fk.Cols.length; j++)
                            {
                                Expr.CodeGetColumnOfTable(v, table, 0, (cols != null ? cols[j] : fk.Cols[0].From), regRow + j);
                                v.AddOp2(OP.IsNull, regRow + j, addrOk);
                            }
                            v.AddOp3(OP.MakeRecord, regRow, fk.Cols.length, regKey);
                            v.ChangeP4(-1, sqlite3IndexAffinityStr(v, index), Vdbe.P4T.TRANSIENT);
                            v.AddOp4Int(OP.Found, i, addrOk, regKey, 0);
                        }
                        v.AddOp2(OP.Rowid, 0, regResult + 1);
                        v.AddOp4(OP.String8, 0, regResult + 2, 0, fk.To, Vdbe.P4T.TRANSIENT);
                        v.AddOp2(OP.Integer, i - 1, regResult + 3);
                        v.AddOp2(OP.ResultRow, regResult, 4);
                        v.ResolveLabel(addrOk);
                        C._tagfree(ctx, ref cols);
                    }
                    v.AddOp2(OP.Next, 0, addrTop + 1);
                    v.JumpHere(addrTop);
                }
            }
#endif
#endif

#if !NDEBUG
            else if (string.Equals(left, "parser_trace", StringComparison.OrdinalIgnoreCase))
            {
                if (right != null)
                {
                    if (ConvertEx.GetBoolean(right))
                        Parser.Trace(Console.Out, "parser: ");
                    else
                        Parser.Trace(null, null);
                }
            }
#endif

            // Reinstall the LIKE and GLOB functions.  The variant of LIKE used will be case sensitive or not depending on the RHS.
            else if (left.Equals("case_sensitive_like", StringComparison.OrdinalIgnoreCase))
            {
                if (right != null)
                    Func.RegisterLikeFunctions(ctx, ConvertEx.GetBoolean(right, 0));
            }

#if !OMIT_INTEGRITY_CHECK
            // Pragma "quick_check" is an experimental reduced version of
            // integrity_check designed to detect most database corruption without most of the overhead of a full integrity-check.
            else if (string.Equals(left, "integrity_check", StringComparison.OrdinalIgnoreCase) || string.Equals(left, "quick_check", StringComparison.OrdinalIgnoreCase))
            {
                bool isQuick = (char.ToLowerInvariant(left[0]) == 'q');

                // If the PRAGMA command was of the form "PRAGMA <ctx>.integrity_check", then db is set to the index of the database identified by <ctx>.
                // In this case, the integrity of database db only is verified by the VDBE created below.
                //
                // Otherwise, if the command was simply "PRAGMA integrity_check" (or "PRAGMA quick_check"), then db is set to 0. In this case, set db
                // to -1 here, to indicate that the VDBE should verify the integrity of all attached databases.
                Debug.Assert(db >= 0);
                Debug.Assert(db == 0 || id2.data != null);
                if (id2.data == null) db = -1;

                // Initialize the VDBE program
                if (Prepare.ReadSchema(parse) != 0) goto pragma_out;
                parse.Mems = 6;
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "integrity_check", C.DESTRUCTOR_STATIC);

                // Set the maximum error count
                int maxErr = INTEGRITY_CHECK_ERROR_MAX;
                if (right != null)
                {
                    ConvertEx.Atoi(right, ref maxErr);
                    if (maxErr <= 0)
                        maxErr = INTEGRITY_CHECK_ERROR_MAX;
                }
                v.AddOp2(OP.Integer, maxErr, 1); // reg[1] holds errors left

                // Do an integrity check on each database file
                for (int i = 0; i < ctx.DBs.length; i++)
                {

                    if (E.OMIT_TEMPDB != 0 && i == 1) continue;
                    if (db >= 0 && i != db) continue;

                    parse.CodeVerifySchema(i);
                    int addr = v.AddOp1(OP.IfPos, 1); // Halt if out of errors
                    v.AddOp2(OP.Halt, 0, 0);
                    v.JumpHere(addr);

                    // Do an integrity check of the B-Tree
                    //
                    // Begin by filling registers 2, 3, ... with the root pages numbers for all tables and indices in the database.
                    int cnt = 0;
                    Debug.Assert(Btree.SchemaMutexHeld(ctx, i, null));
                    Hash tables = ctx.DBs[i].Schema.TableHash;
                    HashElem x;
                    for (x = tables.First; x != null; x = x.Next)
                    {
                        Table table = (Table)x.Data;
                        v.AddOp2(OP.Integer, table.id, 2 + cnt);
                        cnt++;
                        for (Index index = table.Index; index != null; index = index.Next)
                        {
                            v.AddOp2(OP.Integer, index.Id, 2 + cnt);
                            cnt++;
                        }
                    }

                    // Make sure sufficient number of registers have been allocated
                    if (parse.Mems < cnt + 4)
                        parse.Mems = cnt + 4;

                    // Do the b-tree integrity checks
                    v.AddOp3(OP.IntegrityCk, 2, cnt, 1);
                    v.ChangeP5((byte)i);
                    addr = v.AddOp1(OP.IsNull, 2);
                    v.AddOp4(OP.String8, 0, 3, 0, C._mtagprintf(ctx, "*** in database %s ***\n", ctx.DBs[i].Name), Vdbe.P4T.DYNAMIC);
                    v.AddOp3(OP.Move, 2, 4, 1);
                    v.AddOp3(OP.Concat, 4, 3, 2);
                    v.AddOp2(OP.ResultRow, 2, 1);
                    v.JumpHere(addr);

                    // Make sure all the indices are constructed correctly.
                    for (x = tables.First; x != null && !isQuick; x = x.Next)
                    {
                        Table table = (Table)x.Data;
                        if (table.Index == null) continue;
                        addr = v.AddOp1(OP.IfPos, 1); // Stop if out of errors
                        v.AddOp2(OP.Halt, 0, 0);
                        v.JumpHere(addr);
                        parse.OpenTableAndIndices(table, 1, OP.OpenRead);
                        v.AddOp2(OP.Integer, 0, 2); // reg(2) will count entries
                        int loopTop = v.AddOp2(OP.Rewind, 1, 0);
                        v.AddOp2(OP.AddImm, 2, 1); // increment entry count
                        int j;
                        Index index;
                        for (j = 0, index = table.Index; index != null; index = index.Next, j++)
                        {
                            int r1 = parse.GenerateIndexKey(index, 1, 3, false);
                            int jmp2 = v.AddOp4Int(OP.Found, j + 2, 0, r1, index.Columns.length + 1);
                            addr = v.AddOpList(_idxErr.Length, _idxErr);
                            v.ChangeP4(addr + 1, "rowid ", C.DESTRUCTOR_STATIC);
                            v.ChangeP4(addr + 3, " missing from index ", C.DESTRUCTOR_STATIC);
                            v.ChangeP4(addr + 4, index.Name, Vdbe.P4T.TRANSIENT);
                            v.JumpHere(addr + 9);
                            v.JumpHere(jmp2);
                        }
                        v.AddOp2(OP.Next, 1, loopTop + 1);
                        v.JumpHere(v, loopTop);
                        for (j = 0, index = table.Index; index != null; index = index.Next, j++)
                        {
                            addr = v.AddOp1(OP.IfPos, 1);
                            v.AddOp2(OP.Halt, 0, 0);
                            v.JumpHere(addr);
                            addr = v.AddOpList(_cntIdx.Length, _cntIdx);
                            v.ChangeP1(addr + 1, j + 2);
                            v.ChangeP2(addr + 1, addr + 4);
                            v.ChangeP1(addr + 3, j + 2);
                            v.ChangeP2(addr + 3, addr + 2);
                            v.JumpHere(addr + 4);
                            v.ChangeP4(addr + 6, "wrong # of entries in index ", Vdbe.P4T.STATIC);
                            v.ChangeP4(addr + 7, index.Name, Vdbe.P4T.TRANSIENT);
                        }
                    }
                }
                addr = v.AddOpList(_endCode.Length, _endCode);
                v.ChangeP2(addr, -maxErr);
                v.JumpHere(addr + 1);
                v.ChangeP4(addr + 2, "ok", Vdbe.P4T.STATIC);
            }
#endif


#if !OMIT_UTF16
            //   PRAGMA encoding
            //   PRAGMA encoding = "utf-8"|"utf-16"|"utf-16le"|"utf-16be"
            //
            // In its first form, this pragma returns the encoding of the main database. If the database is not initialized, it is initialized now.
            //
            // The second form of this pragma is a no-op if the main database file has not already been initialized. In this case it sets the default
            // encoding that will be used for the main database file if a new file is created. If an existing main database file is opened, then the
            // default text encoding for the existing database is used.
            // 
            // In all cases new databases created using the ATTACH command are created to use the same default text encoding as the main database. If
            // the main database has not been initialized and/or created when ATTACH is executed, this is done before the ATTACH operation.
            //
            // In the second form this pragma sets the text encoding to be used in new database files created using this database handle. It is only
            // useful if invoked immediately after the main database i
            else if (left.Equals("encoding", StringComparison.OrdinalIgnoreCase))
            {
                if (right == null) // "PRAGMA encoding"
                {
                    if (sqlite3ReadSchema(parse) != 0) goto pragma_out;
                    v.SetNumCols(1);
                    v.SetColName(0, COLNAME_NAME, "encoding", C.DESTRUCTOR_STATIC);
                    v.AddOp2(OP.String8, 0, 1);
                    Debug.Assert(_encodeNames[(int)TEXTENCODE.UTF8].Encode == TEXTENCODE.UTF8);
                    Debug.Assert(_encodeNames[(int)TEXTENCODE.UTF16LE].Encode == TEXTENCODE.UTF16LE);
                    Debug.Assert(_encodeNames[(int)TEXTENCODE.UTF16BE].Encode == TEXTENCODE.UTF16BE);
                    v.ChangeP4(-1, _encodeNames[E.CTXENCODE(parse.Ctx)].Name, Vdbe.P4T.STATIC);
                    v.AddOp2(OP.ResultRow, 1, 1);
                }
                else // "PRAGMA encoding = XXX"
                {
                    // Only change the value of sqlite.enc if the database handle is not initialized. If the main database exists, the new sqlite.enc value
                    // will be overwritten when the schema is next loaded. If it does not already exists, it will be created to use the new encoding value.
                    if (!(E.DbHasProperty(ctx, 0, SCHEMA.SchemaLoaded)) || E.DbHasProperty(ctx, 0, SCHEMA.Empty))
                    {
                        int encodeIdx;
                        for (encodeIdx = 0; _encnames[encodeIdx].Name != null; encodeIdx++)
                        {
                            if (string.Equals(right, _encnames[encodeIdx].Name, StringComparison.OrdinalIgnoreCase))
                            {
                                parse.Ctx.DBStatics[0].Schema.Encode = (_encnames[encodeIdx].Encode != 0 ? _encnames[encodeIdx].Encode : TEXTENCODE.UTF16NATIVE);
                                break;
                            }
                        }
                        if (_encnames[encodeIdx].Name == null)
                            parse.ErrorMsg("unsupported encoding: %s", right);
                    }
                }
            }
#endif

#if !OMIT_SCHEMA_VERSION_PRAGMAS
            //   PRAGMA [database.]schema_version
            //   PRAGMA [database.]schema_version = <integer>
            //
            //   PRAGMA [database.]user_version
            //   PRAGMA [database.]user_version = <integer>
            //
            // The pragma's schema_version and user_version are used to set or get the value of the schema-version and user-version, respectively. Both
            // the schema-version and the user-version are 32-bit signed integers stored in the database header.
            //
            // The schema-cookie is usually only manipulated internally by SQLite. It is incremented by SQLite whenever the database schema is modified (by
            // creating or dropping a table or index). The schema version is used by SQLite each time a query is executed to ensure that the internal cache
            // of the schema used when compiling the SQL query matches the schema of the database against which the compiled query is actually executed.
            // Subverting this mechanism by using "PRAGMA schema_version" to modify the schema-version is potentially dangerous and may lead to program
            // crashes or database corruption. Use with caution!
            //
            // The user-version is not used internally by SQLite. It may be used by applications for any purpose.
            else if (string.Equals(left, "schema_version", StringComparison.OrdinalIgnoreCase) || string.Equals(left, "user_version", StringComparison.OrdinalIgnoreCase) || string.Equals(left, "freelist_count", StringComparison.OrdinalIgnoreCase))
            {
                v.UsesBtree(db);
                int cookie; // Cookie index. 1 for schema-cookie, 6 for user-cookie.
                switch (left[0])
                {
                    case 'f':
                    case 'F': cookie = BTREE_FREE_PAGE_COUNT; break;
                    case 's':
                    case 'S': cookie = BTREE_SCHEMA_VERSION; break;
                    default: cookie = BTREE_USER_VERSION; break;
                }
                if (right != null && cookie != BTREE_FREE_PAGE_COUNT)
                {
                    int addr = v.AddOpList(v, _setCookie.Length, _setCookie);
                    v.ChangeP1(addr, db);
                    v.ChangeP1(addr + 1, ConvertEx.Atoi(right));
                    v.ChangeP1(addr + 2, db);
                    v.ChangeP2(addr + 2, cookie);
                }
                else
                {
                    int addr = v.AddOpList(_readCookie.Length, _readCookie);
                    v.ChangeP1(addr, db);
                    v.ChangeP1(addr + 1, db);
                    v.ChangeP3(addr + 1, cookie);
                    v.SetNumCols(1);
                    v.SetColName(0, COLNAME_NAME, left, C.DESTRUCTOR_TRANSIENT);
                }
            }
#endif

#if !OMIT_COMPILEOPTION_DIAGS
            //   PRAGMA compile_options
            //
            // Return the names of all compile-time options used in this build, one option per row.
            else if (string.Equals(left, "compile_options", StringComparison.OrdinalIgnoreCase))
            {
                v.SetNumCols(1);
                parse.Mems = 1;
                v.SetColName(0, COLNAME_NAME, "compile_option", C.DESTRUCTOR_STATIC);
                string opt;
                int i = 0;
                while ((opt = sqlite3_compileoption_get(i++)) != null)
                {
                    v.AddOp4(OP.String8, 0, 1, 0, opt, 0);
                    v.AddOp2(OP.ResultRow, 1, 1);
                }
            }
#endif

#if OMIT_WAL
            //   PRAGMA [database.]wal_checkpoint = passive|full|restart
            //
            // Checkpoint the database.
            else if (string.Equals(left, "wal_checkpoint"))
            {
                int bt = (id2.data != null ? db : MAX_ATTACHED);
                int mode = SQLITE_CHECKPOINT_PASSIVE;
                if (right != null)
                {
                    if (string.Equals(right, "full")) mode = SQLITE_CHECKPOINT_FULL;
                    else if (string.Equals(right, "restart")) mode = SQLITE_CHECKPOINT_RESTART;
                }
                if (sqlite3ReadSchema(parse)) goto pragma_out;
                v.SetNumCols(3);
                parse.Mems = 3;
                v.SetColName(0, COLNAME_NAME, "busy", C.DESTRUCTOR_STATIC);
                v.SetColName(1, COLNAME_NAME, "log", C.DESTRUCTOR_STATIC);
                v.SetColName(2, COLNAME_NAME, "checkpointed", C.DESTRUCTOR_STATIC);
                v.AddOp3(OP.Checkpoint, bt, mode, 1);
                v.AddOp2(OP.ResultRow, 1, 3);
            }

            //   PRAGMA wal_autocheckpoint
            //   PRAGMA wal_autocheckpoint = N
            //
            // Configure a database connection to automatically checkpoint a database after accumulating N frames in the log. Or query for the current value of N.
            else if (string.Equals(left, "wal_autocheckpoint"))
            {
                if (right != null)
                    sqlite3_wal_autocheckpoint(ctx, ConvertEx.Atoi(right));
                ReturnSingleInt(parse, "wal_autocheckpoint", ctx.WalCallback == sqlite3WalDefaultHook ? ctx.WalArg : 0);
            }
#endif

            //  PRAGMA shrink_memory
            //
            // This pragma attempts to free as much memory as possible from the current database connection.
            else if (string.Equals(left, "shrink_memory"))
                sqlite3_db_release_memory(ctx);

            //   PRAGMA busy_timeout
            //   PRAGMA busy_timeout = N
            //
            // Call sqlite3_busy_timeout(ctx, N).  Return the current timeout value if one is set.  If no busy handler or a different busy handler is set
            // then 0 is returned.  Setting the busy_timeout to 0 or negative disables the timeout.
            else if (string.Equals(left, "busy_timeout"))
            {
                if (right != null)
                    sqlite3_busy_timeout(ctx, ConvertEx.Atoi(right));
                ReturnSingleInt(parse, "timeout", ctx->BusyTimeout);
            }

#if DEBUG || TEST
            // Report the current state of file logs for all databases
            else if (string.Equals(left, "lock_status", StringComparison.OrdinalIgnoreCase))
            {
                v.SetNumCols(2);
                parse.Mems = 2;
                v.SetColName(0, COLNAME_NAME, "database", C.DESTRUCTOR_STATIC);
                v.SetColName(1, COLNAME_NAME, "status", C.DESTRUCTOR_STATIC);
                for (int i = 0; i < ctx.DBs.length; i++)
                {
                    if (ctx.DBs[i].Name == null) continue;
                    v.AddOp4(OP.String8, 0, 1, 0, ctx.DBs[i].Name, Vdbe.P4T.STATIC);
                    Btree bt = ctx.DBs[i].Bt;
                    string state = "unknown";
                    Pager pager;
                    long j = 0;
                    if (bt == null || (pager = bt.get_Pager()) == null) state = "closed";
                    else if (sqlite3_file_control(ctx, i != 0 ? ctx.DBs[i].Name : null, SQLITE_FCNTL_LOCKSTATE, ref j) == RC.OK) state = _lockNames[j];
                    v.AddOp4(OP.String8, 0, 2, 0, state, Vdbe.P4T.STATIC);
                    v.AddOp2(OP.ResultRow, 1, 2);
                }
            }
#endif

#if HAS_CODEC
            // needed to support key/rekey/hexrekey with pragma cmds
            else if (string.Equals(left, "key", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                sqlite3_key(ctx, right, right.Length);
            }
            else if (string.Equals(left, "rekey", StringComparison.OrdinalIgnoreCase) && right != null)
            {
                sqlite3_rekey(ctx, right, sqlite3Strlen30(right));
            }
            else if (right != null && (string.Equals(left, "hexkey", StringComparison.OrdinalIgnoreCase) || string.Equals(left, "hexrekey", StringComparison.OrdinalIgnoreCase)))
            {
                StringBuilder key = new StringBuilder(40);
                right = right.ToLower(new CultureInfo("en-us"));
                if (right.Length != 34) return; // expected '0x0102030405060708090a0b0c0d0e0f10'
                for (int i = 2; i < right.Length; i += 2)
                {
                    int h1 = right[i];
                    int h2 = right[i + 1];
                    h1 += 9 * (1 & (h1 >> 6));
                    h2 += 9 * (1 & (h2 >> 6));
                    key.Append(Convert.ToChar((h2 & 0x0f) | ((h1 & 0xf) << 4)));
                }
                if ((left[3] & 0xf) == 0xb)
                    sqlite3_key(ctx, key.ToString(), key.Length);
                else
                    sqlite3_rekey(ctx, key.ToString(), key.Length);
            }
#endif

#if HAS_CODEC || ENABLE_CEROD
            else if (string.Equals(left, "activate_extensions", StringComparison.OrdinalIgnoreCase))
            {
#if HAS_CODEC
                if (right != null && right.Length >= 4 && string.StCompareOrdinal(right, "see-", 4) == 0)
                    sqlite3_activate_see(right.Substring(4));
#endif
#if ENABLE_CEROD
                if (right != null && right.Length >= 6 && string.CompareOrdinal(right, "cerod-", 6) == 0)
                    sqlite3_activate_cerod(right.Substring(6));
#endif
            }
#endif

#if !OMIT_PAGER_PRAGMAS
            // Reset the safety level, in case the fullfsync flag or synchronous setting changed.
            else if (ctx.AutoCommit != 0)
            {
                dbAsObj.Bt.SetSafetyLevel(dbAsObj.SafetyLevel, (ctx.Flags & Context.FLAG.FullFSync) != 0, (ctx.flags & Context.FLAG.CkptFullFSync) != 0);
            }
#endif

        pragma_out:
            C._tagfree(ctx, ref left);
            C._tagfree(ctx, ref right);
        }

#endif
    }
}
