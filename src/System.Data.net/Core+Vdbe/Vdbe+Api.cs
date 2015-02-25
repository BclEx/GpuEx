using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Vdbe
    {
        #region Name1

        static bool VdbeSafety(Vdbe p)
        {
            if (p.Ctx == null)
            {
                SysEx.LOG(RC.MISUSE, "API called with finalized prepared statement");
                return true;
            }
            return false;
        }

        static bool VdbeSafetyNotNull(Vdbe p)
        {
            if (p == null)
            {
                SysEx.LOG(RC.MISUSE, "API called with NULL prepared statement");
                return true;
            }
            return VdbeSafety(p);
        }

        public static RC Finalize(Vdbe p)
        {
            if (p == null)
                return RC.OK; // IMPLEMENTATION-OF: R-57228-12904 Invoking sqlite3_finalize() on a NULL pointer is a harmless no-op.
            Context ctx = p.Ctx;
            if (VdbeSafety(p)) return SysEx.MISUSE_BKPT();
            MutexEx.Enter(ctx.Mutex);
            RC rc = p.Finalize();
            rc = SysEx.ApiExit(ctx, rc);
            Main.LeaveMutexAndCloseZombie(ctx);
            return rc;
        }

        public static RC Reset(Vdbe p)
        {
            if (p == null)
                return RC.OK;
#if THREADSAFE
            MutexEx mutex = p.Ctx.Mutex;
#endif
            MutexEx.Enter(mutex);
            RC rc = p.Reset();
            p.Rewind();
            Debug.Assert((rc & (RC)p.Ctx.ErrMask) == rc);
            rc = SysEx.ApiExit(p.Ctx, rc);
            MutexEx.Leave(mutex);
            return rc;
        }

        public RC ClearBindings(Vdbe p)
        {
#if  THREADSAFE
            MutexEx mutex = Ctx.Mutex;
#endif
            MutexEx.Enter(mutex);
            for (int i = 0; i < Vars.length; i++)
            {
                MemRelease(p.Vars[i]);
                p.Vars[i].Flags = MEM.Null;
            }
            if (p.IsPrepareV2 && p.Expmask != 0)
                p.Expired = true;
            MutexEx.Leave(mutex);
            return RC.OK;
        }

        #endregion

        #region Value
        // The following routines extract information from a Mem or Mem structure.

        public static byte[] Value_Blob(Mem p)
        {
            if ((p.Flags & (MEM.Blob | MEM.Str)) != 0)
            {
                MemExpandBlob(p);
                if (p.ZBLOB == null && p.Z != null)
                {
                    if (p.Z.Length == 0)
                        p.ZBLOB = new byte[1];
                    else
                    {
                        p.ZBLOB = new byte[p.Z.Length];
                        Debug.Assert(p.ZBLOB.Length == p.Z.Length);
                        for (int i = 0; i < p.ZBLOB.Length; i++)
                            p.ZBLOB[i] = (byte)p.Z[i];
                    }
                    p.Z = null;
                }
                p.Flags &= ~MEM.Str;
                p.Flags |= MEM.Blob;
                return (p.N != 0 ? p.ZBLOB : null);
            }
            return (Value_Text(p) == null ? null : Encoding.UTF8.GetBytes(Value_Text(p)));
        }

        public static int Value_Bytes(Mem p) { return ValueBytes(p, TEXTENCODE.UTF8); }
        public static int Balue_Bytes16(Mem p) { return ValueBytes(p, TEXTENCODE.UTF16NATIVE); }
        public static double Value_Double(Mem p) { return RealValue(p); }
        public static int Value_Int(Mem p) { return (int)IntValue(p); }
        public static long Value_Int64(Mem p) { return IntValue(p); }
        public static string Value_Text(Mem p) { return ValueText(p, TEXTENCODE.UTF8); }
#if  !OMIT_UTF16
        public static string Value_Text16(Mem p) { return ValueText(p, TEXTENCODE.UTF16NATIVE); }
        public static string Value_Text16be(Mem p) { return ValueText(p, TEXTENCODE.UTF16BE); }
        public static string Value_Text16le(Mem p) { return ValueText(p, TEXTENCODE.UTF16LE); }
#endif
        public static TYPE Value_Type(Mem p) { return p.Type; }

        #endregion

        #region Result
        // The following routines are used by user-defined functions to specify the function result.
        //
        // The setStrOrError() funtion calls sqlite3VdbeMemSetStr() to store the result as a string or blob but if the string or blob is too large, it
        // then sets the error code to SQLITE_TOOBIG

        static void SetResultStrOrError(FuncContext fctx, string z, int o, int n, TEXTENCODE encode, Action del)
        {
            if (Vdbe.MemSetStr(fctx.S, z, o, n, encode, del) == RC.TOOBIG)
                Vdbe.Result_ErrorOverflow(fctx);
        }
        public static void Result_Blob(FuncContext fctx, string z, int n, Action del)
        {
            Debug.Assert(n >= 0);
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            SetResultStrOrError(fctx, z, 0, n, (TEXTENCODE)0, del);
        }
        public static void Result_Double(FuncContext fctx, double val)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetDouble(fctx.S, val);
        }
        public static void Result_Error(FuncContext fctx, string z, int n)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            SetResultStrOrError(fctx, z, n, TEXTENCODE.UTF8, DESTRUCTOR_TRANSIENT);
            fctx.IsError = RC.ERROR;
        }
#if  !OMIT_UTF16
        void Result_Error16(FuncContext fctx, string z, int n)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            fctx.IsError = RC.ERROR;
            MemSetStr(fctx.S, z, n, TEXTENCODE.UTF16NATIVE, DESTRUCTOR_TRANSIENT);
        }
#endif
        public static void Result_Int(FuncContext fctx, int val)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetInt64(fctx.S, (long)val);
        }
        public static void Result_Int64(FuncContext ctx, long value)
        {
            Debug.Assert(MutexEx.Held(ctx.S.Ctx.Mutex));
            MemSetInt64(ctx.S, value);
        }
        public static void Result_Null(FuncContext fctx)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetNull(fctx.S);
        }
        public static void Result_Text(FuncContext fctx, string z, int o, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            SetResultStrOrError(fctx, z, o, n, TEXTENCODE.UTF8, del);
        }
        public static void Result_Text(FuncContext fctx, StringBuilder z, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            SetResultStrOrError(fctx, z.ToString(), 0, n, TEXTENCODE.UTF8, del);
        }
        public static void Result_Text(FuncContext fctx, string z, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            SetResultStrOrError(fctx, z, 0, n, TEXTENCODE.UTF8, del);
        }
#if !OMIT_UTF16
        void Result_Text16(FuncContext fctx, string z, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetStr(fctx.S, z, n, TEXTENCODE.UTF16NATIVE, del);
        }
        void Result_Text16be(FuncContext fctx, string z, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetStr(fctx.S, z, n, TEXTENCODE.UTF16BE, del);
        }
        void Result_Text16le(FuncContext fctx, string z, int n, Action del)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetStr(fctx.S, z, n, TEXTENCODE.UTF16LE, del);
        }
#endif
        public static void Result_Value(FuncContext fctx, Mem value)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemCopy(fctx.S, value);
        }
        public static void Result_ZeroBlob(FuncContext fctx, int n)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetZeroBlob(fctx.S, n);
        }
        public static void Result_ErrorCode(FuncContext fctx, RC errCode)
        {
            fctx.IsError = errCode;
            if ((fctx.S.Flags & MEM.Null) != 0)
                SetResultStrOrError(fctx, SysEx.ErrStr(errCode), -1, TEXTENCODE.UTF8, DESTRUCTOR_STATIC);
        }
        public static void Result_ErrorOverflow(FuncContext fctx)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            fctx.IsError = RC.ERROR;
            SetResultStrOrError(fctx, "string or blob too big", -1, TEXTENCODE.UTF8, DESTRUCTOR_STATIC);
        }
        public static void Result_ErrorNoMem(FuncContext fctx)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            MemSetNull(fctx.S);
            fctx.IsError = RC.NOMEM;
            fctx.S.Ctx.MallocFailed = true;
        }

        #endregion

        #region Step

        static RC DoWalCallbacks(Context ctx)
        {
            RC rc = RC.OK;
#if !OMIT_WAL
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Btree bt = ctx.DBs[i].Bt;
                if (bt != null)
                {
                    int entrys = sqlite3PagerWalCallback(bt.get_Pager());
                    if (ctx.WalCallback != null && entrys > 0 && rc == RC.OK)
                        rc = ctx.WalCallback(ctx.WalArg, ctx, ctx.DBs[i].Name, entrys);
                }
            }
#endif
            return rc;
        }

        public RC Step2()
        {
            if (Magic != VDBE_MAGIC_RUN)
            {
                // We used to require that sqlite3_reset() be called before retrying sqlite3_step() after any error or after SQLITE_DONE.  But beginning
                // with version 3.7.0, we changed this so that sqlite3_reset() would be called automatically instead of throwing the SQLITE_MISUSE error.
                // This "automatic-reset" change is not technically an incompatibility, since any application that receives an SQLITE_MISUSE is broken by
                // definition.
                //
                // Nevertheless, some published applications that were originally written for version 3.6.23 or earlier do in fact depend on SQLITE_MISUSE 
                // returns, and those were broken by the automatic-reset change.  As a work-around, the SQLITE_OMIT_AUTORESET compile-time restores the
                // legacy behavior of returning SQLITE_MISUSE for cases where the previous sqlite3_step() returned something other than a SQLITE_LOCKED
                // or SQLITE_BUSY error.
#if OMIT_AUTORESET
                if (RC == RC.BUSY || RC == RC.LOCKED)
                    Reset(this);
                else
                    return SysEx.MISUSE_BKPT();
#else
                Reset(this);
#endif
            }

            // Check that malloc() has not failed. If it has, return early.
            RC rc;
            Context ctx = Ctx;
            if (ctx.MallocFailed)
            {
                RC_ = RC.NOMEM;
                return RC.NOMEM;
            }

            if (PC <= 0 && Expired)
            {
                RC_ = RC.SCHEMA;
                rc = RC.ERROR;
                goto end_of_step;
            }
            if (PC < 0)
            {
                // If there are no other statements currently running, then reset the interrupt flag.  This prevents a call to sqlite3_interrupt
                // from interrupting a statement that has not yet started.
                if (ctx.ActiveVdbeCnt == 0)
                    ctx.u1.IsInterrupted = false;
                Debug.Assert(ctx.WriteVdbeCnt > 0 || ctx.AutoCommit == 0 || ctx.DeferredCons == 0);
#if  !OMIT_TRACE
                if (ctx.Profile != null && !ctx.Init.Busy)
                    ctx.Vfs.CurrentTimeInt64(ref StartTime);
#endif
                ctx.ActiveVdbeCnt++;
                if (!ReadOnly) ctx.WriteVdbeCnt++;
                PC = 0;
            }

#if  !OMIT_EXPLAIN
            if (_explain)
                rc = List();
            else
#endif
            {
                ctx.VdbeExecCnt++;
                rc = Exec();
                ctx.VdbeExecCnt--;
            }

#if !OMIT_TRACE
            // Invoke the profile callback if there is one
            if (rc != RC.ROW && ctx.Profile != null && !ctx.Init.Busy && Sql != null)
            {
                long now = 0;
                ctx.Vfs.CurrentTimeInt64(ref now);
                ctx.Profile(ctx.ProfileArg, Sql, (now - StartTime) * 1000000);
            }
#endif

            if (rc == RC.DONE)
            {
                Debug.Assert(RC_ == RC.OK);
                RC_ = DoWalCallbacks(ctx);
                if (RC_ != RC.OK)
                    rc = RC.ERROR;
            }

            ctx.ErrCode = rc;
            if (SysEx.ApiExit(Ctx, RC_) == RC.NOMEM)
                RC_ = RC.NOMEM;

        end_of_step:
            // At this point local variable rc holds the value that should be returned if this statement was compiled using the legacy 
            // sqlite3_prepare() interface. According to the docs, this can only be one of the values in the first assert() below. Variable p->rc 
            // contains the value that would be returned if sqlite3_finalize() were called on statement p.
            Debug.Assert(rc == RC.ROW || rc == RC.DONE || rc == RC.ERROR || rc == RC.BUSY || rc == RC.MISUSE);
            Debug.Assert(RC_ != RC.ROW && RC_ != RC.DONE);
            // If this statement was prepared using sqlite3_prepare_v2(), and an error has occurred, then return the error code in p->rc to the
            // caller. Set the error code in the database handle to the same value.
            if (IsPrepareV2 && rc != RC.ROW && rc != RC.DONE)
                rc = TransferError();
            return (rc & (RC)ctx.ErrMask);
        }

        public const int MAX_SCHEMA_RETRY = 5;

        public RC Step()
        {
            RC rc = RC.OK;      // Result from sqlite3Step()
            RC rc2 = RC.OK;     // Result from sqlite3Reprepare()
            int cnt = 0;             // Counter to prevent infinite loop of reprepares
            Context ctx = Ctx; // The database connection
            MutexEx.Enter(ctx.Mutex);
            DoingRerun = false;
            while ((rc = Step2()) == RC.SCHEMA && cnt++ < MAX_SCHEMA_RETRY && (rc2 = rc = Reprepare()) == RC.OK)
            {
                Reset(this);
                DoingRerun = true;
                Debug.Assert(!Expired);
            }
            if (rc2 != RC.OK && C._ALWAYS(IsPrepareV2) && C._ALWAYS(ctx.Err != null))
            {
                // This case occurs after failing to recompile an sql statement. The error message from the SQL compiler has already been loaded 
                // into the database handle. This block copies the error message from the database handle into the statement and sets the statement
                // program counter to 0 to ensure that when the statement is finalized or reset the parser error message is available via
                // sqlite3_errmsg() and sqlite3_errcode().
                string err = Value_Text(ctx.Err);
                C._tagfree(ctx, ref ErrMsg);
                if (!ctx.MallocFailed) { ErrMsg = err; RC_ = rc2; }
                else { ErrMsg = null; RC_ = rc = RC.NOMEM; }
            }
            rc = SysEx.ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        #endregion

        #region Name3

        public static object User_Data(FuncContext fctx)
        {
            Debug.Assert(fctx != null && fctx.Func != null);
            return fctx.Func.UserData;
        }

        public static Context Context_Ctx(FuncContext fctx)
        {
            Debug.Assert(fctx != null && fctx.Func != null);
            return fctx.S.Ctx;
        }

        public static void InvalidFunction(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            string name = fctx.Func.Name;
            string err = C._mprintf("unable to use function %s in the requested context", name);
            Result_Error(fctx, err, -1);
            C._free(ref err);
        }

        public static Mem Aggregate_Context(FuncContext fctx, int bytes)
        {
            Debug.Assert(fctx != null && fctx.Func != null && fctx.Func.Step != null);
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            Mem mem = fctx.Mem;
            C.ASSERTCOVERAGE(bytes < 0);
            if ((mem.Flags & MEM.Agg) == 0)
            {
                if (bytes <= 0)
                {
                    MemReleaseExternal(mem);
                    mem.Flags = 0;
                    mem.Z = null;
                }
                else
                {
                    MemGrow(mem, bytes, 0);
                    mem.Flags = MEM.Agg;
                    mem.u.Def = fctx.Func;
                }
            }
            return Mem.ToMem_(mem);
        }

        public static object get_Auxdata(FuncContext fctx, int arg)
        {
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            VdbeFunc vdbeFunc = fctx.VdbeFunc;
            if (vdbeFunc == null || arg >= vdbeFunc.AuxsLength || arg < 0)
                return null;
            return vdbeFunc.Auxs[arg].Aux;
        }

        public static void set_Auxdata(FuncContext fctx, int args, object aux, Action delete)
        {
            if (args < 0) goto failed;
            Debug.Assert(MutexEx.Held(fctx.S.Ctx.Mutex));
            VdbeFunc vdbeFunc = fctx.VdbeFunc;
            if (vdbeFunc == null || vdbeFunc.AuxsLength <= args)
            {
                int auxLength = (vdbeFunc != null ? vdbeFunc.AuxsLength : 0);
                int newSize = args;
                vdbeFunc = new VdbeFunc();
                if (vdbeFunc == null)
                    goto failed;
                fctx.VdbeFunc = vdbeFunc;
                vdbeFunc.Auxs[auxLength] = new VdbeFunc.AuxData();
                vdbeFunc.AuxsLength = args + 1;
                vdbeFunc.Func = fctx.Func;
            }
            VdbeFunc.AuxData auxData = vdbeFunc.Auxs[args];
            if (auxData.Aux != null && auxData.Aux is IDisposable)
                (auxData.Aux as IDisposable).Dispose();
            auxData.Aux = aux;
            return;

        failed:
            if (aux != null && aux is IDisposable)
                (aux as IDisposable).Dispose();
        }

        public static int Column_Count(Vdbe p)
        {
            return (p != null ? p.ResColumns : 0);
        }

        public static int Data_Count(Vdbe p)
        {
            return (p == null || p.ResultSet == null ? 0 : p.ResColumns);
        }

        #endregion

        #region Column
        // The following routines are used to access elements of the current row in the result set.

        // If the value passed as the second argument is out of range, return a pointer to the following static Mem object which contains the
        // value SQL NULL. Even though the Mem structure contains an element of type i64, on certain architectures (x86) with certain compiler
        // switches (-Os), gcc may align this Mem object on a 4-byte boundary instead of an 8-byte one. This all works fine, except that when
        // running with SQLITE_DEBUG defined the SQLite code sometimes assert()s that a Mem structure is located on an 8-byte boundary. To prevent
        // these assert()s from failing, when building with SQLITE_DEBUG defined using gcc, we force nullMem to be 8-byte aligned using the magical
        // __attribute__((aligned(8))) macro.
        private static Mem _nullMem = new Mem(null, "", (double)0, 0, 0, MEM.Null, TYPE.NULL, 0
#if DEBUG
, null, null  // scopyFrom, filler
#endif
);
        static Mem ColumnMem(Vdbe p, int i)
        {
            Mem r;
            if (p != null && p.ResultSet != null && i < p.ResColumns && i >= 0)
            {
                MutexEx.Enter(p.Ctx.Mutex);
                r = p.ResultSet[i];
            }
            else
            {
                if (p != null && C._ALWAYS(p.Ctx != null))
                {
                    MutexEx.Enter(p.Ctx.Mutex);
                    SysEx.Error(p.Ctx, RC.RANGE, 0);
                }
                r = _nullMem;
            }
            return r;
        }

        static void ColumnMallocFailure(Vdbe p)
        {
            // If malloc() failed during an encoding conversion within an sqlite3_column_XXX API, then set the return code of the statement to
            // RC_NOMEM. The next call to _step() (if any) will return RC_ERROR and _finalize() will return NOMEM.
            if (p != null)
            {
                p.RC_ = SysEx.ApiExit(p.Ctx, p.RC_);
                MutexEx.Leave(p.Ctx.Mutex);
            }
        }

        public static byte[] Column_Blob(Vdbe p, int i) { byte[] val = Value_Blob(ColumnMem(p, i)); ColumnMallocFailure(p); return val; } // Even though there is no encoding conversion, value_blob() might need to call malloc() to expand the result of a zeroblob() expression.
        public static int Column_Bytes(Vdbe p, int i) { int val = Value_Bytes(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static int Column_Bytes16(Vdbe p, int i) { int val = Balue_Bytes16(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static double Column_Double(Vdbe p, int i) { double val = Value_Double(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static int Column_Int(Vdbe p, int i) { int val = Value_Int(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static long Column_Int64(Vdbe p, int i) { long val = Value_Int64(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static string Column_Text(Vdbe p, int i) { string val = Value_Text(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
        public static Mem Column_Value(Vdbe p, int i)
        {
            Mem r = ColumnMem(p, i);
            if ((r.Flags & MEM.Static) != 0)
            {
                r.Flags &= ~MEM.Static;
                r.Flags |= MEM.Ephem;
            }
            ColumnMallocFailure(p);
            return r;
        }
#if !OMIT_UTF16
        public string Column_Text16(Vdbe p, int i) { string val = Value_Text16(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
#endif
        public static TYPE Column_Type(Vdbe p, int i) { TYPE type = Value_Type(ColumnMem(p, i)); ColumnMallocFailure(p); return type; }

        public static string ColumnName(Vdbe p, int n, Func<Mem, string> func, bool useType)
        {
            Context ctx = p.Ctx;
            Debug.Assert(ctx != null);
            string r = null;
            int n2 = Column_Count(p);
            if (n < n2 && n >= 0)
            {
                n += (useType ? n2 : 0);
                MutexEx.Enter(ctx.Mutex);
                Debug.Assert(ctx.MallocFailed);
                r = func(p.ColNames[n]);
                // A malloc may have failed inside of the xFunc() call. If this is the case, clear the mallocFailed flag and return NULL.
                if (ctx.MallocFailed)
                {
                    ctx.MallocFailed = false;
                    r = null;
                }
                MutexEx.Leave(ctx.Mutex);
            }
            return r;
        }

        public static string Column_Name(Vdbe p, int n) { return ColumnName(p, n, Value_Text, COLNAME_NAME); }
#if !OMIT_UTF16
        public static string Column_Name16(Vdbe p, int n) { return ColumnName(p, n, Value_Text16, COLNAME_NAME); }
#endif

#if OMIT_DECLTYPE && ENABLE_COLUMN_METADATA
#error "Must not define both OMIT_DECLTYPE and ENABLE_COLUMN_METADATA"
#endif
#if !OMIT_DECLTYPE
        public static string Column_Decltype(Vdbe p, int n) { return ColumnName(p, n, Value_Text, COLNAME_DECLTYPE); }
#if !OMIT_UTF16
        public static string Column_Decltype16(Vdbe p, int n) { return ColumnName(p, n, Value_Text16, COLNAME_DECLTYPE); }
#endif
#endif
#if ENABLE_COLUMN_METADATA
        public static string Column_DatabaseName(Vdbe p, int n) { return ColumnName(p, n, Value_Text, COLNAME_DATABASE); }
#if !OMIT_UTF16
        public static string Column_DatabaseName16(Vdbe p, int n) { return ColumnName(p, n, Value_Text16, COLNAME_DATABASE); }
#endif
        public static string Column_TableName(Vdbe p, int n) { return ColumnName(p, n, Value_Text, COLNAME_TABLE); }
#if !OMIT_UTF16
        public static string Column_TableName16(Vdbe p, int n) { return ColumnName(p, n, Value_Text16, COLNAME_TABLE); }
#endif
        public static string Column_OriginName(Vdbe p, int n) { return ColumnName(p, n, Value_Text, COLNAME_COLUMN); }
#if !OMIT_UTF16
        public static string Column_OriginName16(Vdbe p, int n) { return ColumnName(p, n, Value_Text16, COLNAME_COLUMN); }
#endif
#endif

        #endregion

        #region Bind
        // Routines used to attach values to wildcards in a compiled SQL statement.

        static RC VdbeUnbind(Vdbe p, int i)
        {
            if (VdbeSafetyNotNull(p))
                return SysEx.MISUSE_BKPT();
            MutexEx.Enter(p.Ctx.Mutex);
            if (p.Magic != VDBE_MAGIC_RUN || p.PC >= 0)
            {
                SysEx.Error(p.Ctx, RC.MISUSE, 0);
                MutexEx.Leave(p.Ctx.Mutex);
                SysEx.LOG(RC.MISUSE, "bind on a busy prepared statement: [%s]", p.Sql);
                return SysEx.MISUSE_BKPT();
            }
            if (i < 1 || i > p.Vars)
            {
                SysEx.Error(p.Ctx, RC.RANGE, 0);
                MutexEx.Leave(p.Ctx.Mutex);
                return RC.RANGE;
            }
            i--;
            Mem var = p.Vars[i];
            MemRelease(var);
            var.Flags = MEM.Null;
            SysEx.Error(p.Ctx, RC.OK, 0);

            // If the bit corresponding to this variable in Vdbe.expmask is set, then binding a new value to this variable invalidates the current query plan.
            //
            // IMPLEMENTATION-OF: R-48440-37595 If the specific value bound to host parameter in the WHERE clause might influence the choice of query plan
            // for a statement, then the statement will be automatically recompiled, as if there had been a schema change, on the first sqlite3_step() call
            // following any change to the bindings of that parameter.
            if (p.IsPrepareV2 && ((i < 32 && p.Expmask != 0 & ((uint)1 << i) != 0) || p.Expmask == 0xffffffff))
                p.Expired = true;
            return RC.OK;
        }

        static RC BindText(Vdbe p, int i, byte[] z, int n, Action<byte[]> del, TEXTENCODE encoding)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
            {
                if (z != null)
                {
                    Mem var = p.Vars[i - 1];
                    rc = MemSetBlob(var, z, n, encoding, del);
                    if (rc == RC.OK && encoding != 0)
                        rc = ChangeEncoding(var, E.CTXENCODE(p.Ctx));
                    SysEx.Error(p.Ctx, rc, 0);
                    rc = SysEx.ApiExit(p.Ctx, rc);
                }
                MutexEx.Leave(p.Ctx.Mutex);
            }
            else if (del != null)
                del(z);
            return rc;
        }

        static RC BindText(Vdbe p, int i, string z, int n, Action<string> del, TEXTENCODE encoding)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
            {
                if (z != null)
                {
                    Mem var = p.Vars[i - 1];
                    rc = MemSetStr(var, z, n, encoding, del);
                    if (rc == RC.OK && encoding != 0)
                        rc = ChangeEncoding(var, E.CTXENCODE(p.Ctx));
                    SysEx.Error(p.Ctx, rc, 0);
                    rc = SysEx.ApiExit(p.Ctx, rc);
                }
                MutexEx.Leave(p.Ctx.Mutex);
            }
            else if (del != null)
                del(z);
            return rc;
        }

        public static RC Bind_Blob(Vdbe p, int i, string z, int n, Action<string> del) { return BindText(p, i, z, n, del, (TEXTENCODE)0); }
        public static RC Bind_Double(Vdbe p, int i, double value)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
            {
                MemSetDouble(p.Vars[i - 1], value);
                MutexEx.Leave(p.Ctx.Mutex);
            }
            return rc;
        }
        public static RC Bind_Int(Vdbe p, int i, int value) { return Bind_Int64(p, i, (long)value); }
        public static RC Bind_Int64(Vdbe p, int i, long value)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
            {
                MemSetInt64(p.Vars[i - 1], value);
                MutexEx.Leave(p.Ctx.Mutex);
            }
            return rc;
        }
        public static RC Bind_Null(Vdbe p, int i)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
                MutexEx.Leave(p.Ctx.Mutex);
            return rc;
        }

        public static RC Bind_Text(Vdbe p, int i, string z, int n, Action<string> del) { return BindText(p, i, z, n, del, TEXTENCODE.UTF8); }
        public static RC Bind_Blob(Vdbe p, int i, byte[] z, int n, Action<string> del) { return BindText(p, i, z, n >= 0 ? n : z.Length, del, (TEXTENCODE)0); }
#if !OMIT_UTF16
        public static RC Bind_Text16(Vdbe p, int i, string z, int n, Action<string> del) { return BindText(p, i, z, n, del, TEXTENCODE.UTF16NATIVE); }
#endif

        public static RC Bind_Value(Vdbe p, int i, Mem value)
        {
            RC rc;
            switch (value.Type)
            {
                case TYPE.INTEGER:
                    {
                        rc = Bind_Int64(p, i, value.u.I);
                        break;
                    }
                case TYPE.FLOAT:
                    {
                        rc = Bind_Double(p, i, value.R);
                        break;
                    }
                case TYPE.BLOB:
                    {
                        if ((value.Flags & MEM.Zero) != 0)
                            rc = Bind_Zeroblob(p, i, value.u.Zeros);
                        else
                            rc = Bind_Blob(p, i, value.ZBLOB, value.N, DESTRUCTOR_TRANSIENT);
                        break;
                    }
                case TYPE.TEXT:
                    {
                        rc = BindText(p, i, value.Z, value.N, DESTRUCTOR_TRANSIENT, value.Encode);
                        break;
                    }
                default:
                    {
                        rc = Bind_Null(p, i);
                        break;
                    }
            }
            return rc;
        }

        public static RC Bind_Zeroblob(Vdbe p, int i, int n)
        {
            RC rc = VdbeUnbind(p, i);
            if (rc == RC.OK)
            {
                MemSetZeroBlob(p.Vars[i - 1], n);
                MutexEx.Leave(p.Ctx.Mutex);
            }
            return rc;
        }

        public static int Bind_ParameterCount(Vdbe p) { return (p != null ? (int)p.Vars.length : 0); }
        public static string Bind_ParameterName(Vdbe p, int i) { return (p == null || i < 1 || i > p.VarNames.length ? null : p.VarNames[i - 1]); }

        public static int ParameterIndex(Vdbe p, string name, int nameLength)
        {
            if (p == null)
                return 0;
            if (name != null)
            {
                for (int i = 0; i < p.VarNames.length; i++)
                {
                    string z = p.VarNames[i];
                    if (z != null && z == name)
                        return i + 1;
                }
            }
            return 0;
        }
        public static int Bind_ParameterIndex(Vdbe p, string name) { return ParameterIndex(p, name, name.Length); }
        public static RC TransferBindings(Vdbe from, Vdbe to)
        {
            Debug.Assert(to.Ctx == from.Ctx);
            Debug.Assert(to.Vars.length == from.Vars.length);
            MutexEx.Enter(to.Ctx.Mutex);
            for (int i = 0; i < from.Vars.length; i++)
                MemMove(to.Vars[i], from.Vars[i]);
            MutexEx.Leave(to.Ctx.Mutex);
            return RC.OK;
        }

        #endregion

        #region Stmt

        public static Context Stmt_Ctx(Vdbe p) { return (p != null ? p.Ctx : null); }
        public static bool Stmt_Readonly(Vdbe p) { return (p != null ? p.ReadOnly : true); }
        public static bool Stmt_Busy(Vdbe p) { return (p != null && p.PC > 0 && p.Magic == VDBE_MAGIC_RUN); }
        public static Vdbe Stmt_Next(Context ctx, Vdbe p)
        {
            MutexEx.Enter(ctx.Mutex);
            Vdbe next = (p == null ? ctx.Vdbes[0] : p.Next);
            MutexEx.Leave(ctx.Mutex);
            return next;
        }
        public static int Stmt_Status(Vdbe p, OP op, bool resetFlag)
        {
            int v = p.Counters[(int)op - 1];
            if (resetFlag) p.Counters[(int)op - 1] = 0;
            return v;
        }

        #endregion
    }
}
