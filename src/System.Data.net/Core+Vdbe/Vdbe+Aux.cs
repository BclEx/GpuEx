using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using FILE = System.IO.TextWriter;
using Pid = System.UInt32;
#region Limits
#if !MAX_VARIABLE_NUMBER
using yVar = System.Int16;
#else
    using yVar = System.Int32; 
#endif
#if MAX_ATTACHED
    using yDbMask = System.Int64;
#else
using yDbMask = System.Int32;
#endif
#endregion
using P4_t = Core.Vdbe.VdbeOp.P4_t;
using Core.IO;

namespace Core
{
    public partial class Vdbe
    {
        public static Vdbe Create(Context ctx)
        {
            Vdbe p = new Vdbe();
            if (p == null) return null;
            p.Ctx = ctx;
            if (ctx.Vdbes != null)
                ctx.Vdbes.Prev = p;
            p.Next = ctx.Vdbes;
            p.Prev = null;
            ctx.Vdbes = p;
            p.Magic = VDBE_MAGIC_INIT;
            return p;
        }

        public static void SetSql(Vdbe p, string z, int n, bool isPrepareV2)
        {
            if (p == null) return;
#if OMIT_TRACE && !ENABLE_SQLLOG
            if (isPrepareV2 == 0) return;
#endif
            Debug.Assert(p.Sql_ == null);
            p.Sql_ = z.Substring(0, n); //: _tagstrndup(p->Ctx, z, n);
            p.IsPrepareV2 = isPrepareV2;
        }

        public static string Sql(Vdbe stmt)
        {
            Vdbe p = (Vdbe)stmt;
            return (p != null && p.IsPrepareV2 ? p.Sql_ : null);
        }

        public static void Swap(Vdbe a, Vdbe b)
        {
            Vdbe tmp_ = new Vdbe();
            a._memcpy(tmp_);
            b._memcpy(a);
            tmp_._memcpy(b);
            Vdbe tmp = tmp = a.Next;
            a.Next = b.Next;
            b.Next = tmp;
            tmp = a.Prev;
            a.Prev = b.Prev;
            b.Prev = tmp;
            string tmpSql = a.Sql_;
            a.Sql_ = b.Sql_;
            b.Sql_ = tmpSql;
            b.IsPrepareV2 = a.IsPrepareV2;
        }

#if DEBUG
        public void set_Trace(FILE trace)
        {
            Trace = trace;
        }
#endif

        static RC GrowOps(Vdbe p)
        {
            int newLength = (p.OpsAlloc != 0 ? p.OpsAlloc * 2 : 1024 / 1);
            p.OpsAlloc = newLength;
            C._tagrealloc_or_create(p.Ctx, ref p.Ops.data, newLength);
            return (p.Ops.data != null ? RC.OK : RC.NOMEM);
        }

        public int AddOp3(OP op, int p1, int p2, int p3)
        {
            int i = Ops.length;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert((int)op > 0 && (int)op < 0xff);
            if (OpsAlloc <= i)
                if (GrowOps(this) != RC.OK)
                    return 1;
            Ops.length++;
            if (Ops[i] == null) Ops[i] = new VdbeOp();
            VdbeOp opAsObj = Ops[i];
            opAsObj.Opcode = op;
            opAsObj.P5 = 0;
            opAsObj.P1 = p1;
            opAsObj.P2 = p2;
            opAsObj.P3 = p3;
            opAsObj.P4.P = null;
            opAsObj.P4Type = Vdbe.P4T.NOTUSED;
#if DEBUG
            opAsObj.Comment = null;
            if ((Ctx.Flags & BContext.FLAG.VdbeAddopTrace) != 0)
                PrintOp(null, i, Ops[i]);
#endif
#if VDBE_PROFILE
            opAsObj.Cycles = 0;
            opAsObj.Cnt = 0;
#endif
            return i;
        }
        public int AddOp0(OP op) { return AddOp3(op, 0, 0, 0); }
        public int AddOp1(OP op, int p1) { return AddOp3(op, p1, 0, 0); }
        public int AddOp2(OP op, int p1, bool b2) { return AddOp3(op, p1, (int)(b2 ? 1 : 0), 0); }
        public int AddOp2(OP op, int p1, int p2) { return AddOp3(op, p1, p2, 0); }
        public int AddOp4(OP op, int p1, int p2, int p3, int p4, Vdbe.P4T p4t) // int
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, char p4, Vdbe.P4T p4t) // char
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4.ToString() }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, StringBuilder p4, Vdbe.P4T p4t) // StringBuilder
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4.ToString() }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, string p4, Vdbe.P4T p4t) // string
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Z = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, byte[] p4, Vdbe.P4T p4t) // byte[]
        {
            Debug.Assert(op == OP.Null || p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(p, addr, new P4_t { Z = Encoding.UTF8.GetString(p4, 0, p4.Length) }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, int[] p4, Vdbe.P4T p4t) // P4T_INTARRAY
        {
            Debug.Assert(p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Is = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, long p4, Vdbe.P4T p4t) // P4T_INT64
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I64 = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, double p4, Vdbe.P4T p4t) // DOUBLE (REAL)
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Real = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, FuncDef p4, Vdbe.P4T p4t) // FUNCDEF
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Func = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, CollSeq p4, Vdbe.P4T p4t) // CollSeq
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { Coll = p4 }, p4t);
            return addr;
        }
        public int AddOp4(OP op, int p1, int p2, int p3, KeyInfo p4, Vdbe.P4T p4t) // KeyInfo
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { KeyInfo = p4 }, p4t);
            return addr;
        }

#if !OMIT_VIRTUALTABLE
        public int AddOp4(OP op, int p1, int p2, int p3, VTable p4, Vdbe.P4T p4t) // VTable
        {
            Debug.Assert(p4 != null);
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(p, addr, new P4_t { VTable = p4 }, p4t);
            return addr;
        }
#endif

        public void AddParseSchemaOp(int db, string where_)
        {
            int addr = AddOp3(OP.ParseSchema, db, 0, 0);
            ChangeP4(addr, where_, Vdbe.P4T.DYNAMIC);
            for (int j = 0; j < Ctx.DBs.length; j++) UsesBtree(j);
        }

        public int AddOp4Int(OP op, int p1, int p2, int p3, int p4)
        {
            int addr = AddOp3(op, p1, p2, p3);
            ChangeP4(addr, new P4_t { I = p4 }, Vdbe.P4T.INT32);
            return addr;
        }

        public int MakeLabel()
        {
            int i = Labels.length++;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            C._tagrealloc_or_create(Ctx, ref Labels.data, (i * 2 + 1));
            if (Labels.data != null)
                Labels[i] = -1;
            return -1 - i;
        }

        public void ResolveLabel(int x)
        {
            int j = -1 - x;
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert(j >= 0 && j < Labels.length);
            if (Labels.data != null)
                Labels[j] = Ops.length;
        }

        public void set_RunOnlyOnce()
        {
            RunOnlyOnce = true;
        }

#if DEBUG

        public class VdbeOpIter
        {
            public Vdbe V;                    // Vdbe to iterate through the opcodes of
            public array_t<SubProgram> Subs;     // Array of subprograms
            public int Addr;                  // Address of next instruction to return
            public int SubId;                 // 0 = main program, 1 = first sub-program etc.
        }

        static VdbeOp OpIterNext(VdbeOpIter p)
        {
            VdbeOp r = null;
            if (p.SubId <= p.Subs.length)
            {
                Vdbe v = p.V;
                array_t<VdbeOp> ops = (p.SubId == 0 ? v.Ops : p.Subs[p.SubId - 1].Ops);
                Debug.Assert(p.Addr < ops.length);

                r = ops[p.Addr];
                p.Addr++;
                if (p.Addr == ops.length)
                {
                    p.SubId++;
                    p.Addr = 0;
                }
                if (r.P4Type == Vdbe.P4T.SUBPROGRAM)
                {
                    int bytes = p.Subs.length + 1;
                    int j;
                    for (j = 0; j < p.Subs.length; j++)
                        if (p.Subs[j] == r.P4.Program) break;
                    if (j == p.Subs.length)
                    {
                        C._tagrealloc_or_free2(v.Ctx, ref p.Subs.data, bytes);
                        p.Subs[p.Subs.length++] = r.P4.Program;
                    }
                }
            }
            return r;
        }

        public bool AssertMayAbort(bool mayAbort)
        {
            VdbeOpIter sIter;
            sIter = new VdbeOpIter();
            sIter.V = this;

            bool hasAbort = false;
            VdbeOp op;
            while ((op = OpIterNext(sIter)) != null)
            {
                OP opcode = (OP)op.Opcode;
                if (opcode == OP.Destroy || opcode == OP.VUpdate || opcode == OP.VRename
#if !OMIT_FOREIGN_KEY
 || (opcode == OP.FkCounter && op.P1 == 0 && op.P2 == 1)
#endif
 || ((opcode == OP.Halt || opcode == OP.HaltIfNull)
 && (op.P1 == (int)RC.CONSTRAINT && op.P2 == (int)OE.Abort)))
                {
                    hasAbort = true;
                    break;
                }
            }
            C._tagfree(Ctx, ref sIter.Subs.data);

            // Return true if hasAbort==mayAbort. Or if a malloc failure occurred. If malloc failed, then the while() loop above may not have iterated
            // through all opcodes and hasAbort may be set incorrectly. Return true for this case to prevent the assert() in the callers frame from failing.            return (hasAbort == mayAbort) ? 1 : 0;//v.db.mallocFailed !=0|| hasAbort==mayAbort );
            return (Ctx.MallocFailed || hasAbort == mayAbort);
        }
#endif

        static void ResolveP2Values(Vdbe p, ref int maxFuncArgs)
        {
            int maxArgs = maxFuncArgs;
            int[] labels = p.Labels.data;
            p.ReadOnly = true;

            Vdbe.VdbeOp op;
            int i;
            for (i = 0; i < p.Ops.length; i++)
            {
                op = p.Ops[i];
                OP opcode = op.Opcode;
                op.Opflags = E._opcodeProperty[(int)opcode];
                if (opcode == OP.Function || opcode == OP.AggStep)
                {
                    if (op.P5 > maxArgs) maxArgs = op.P5;
                }
                else if ((opcode == OP.Transaction && op.P2 != 0) || opcode == OP.Vacuum)
                {
                    p.ReadOnly = false;

                }
#if !OMIT_VIRTUALTABLE
                else if (opcode == OP.VUpdate)
                {
                    if (op.P2 > maxArgs) maxArgs = op.P2;
                }
                else if (opcode == OP.VFilter)
                {
                    Debug.Assert(p.Ops.length - i >= 3);
                    Debug.Assert(p.Ops[i - 1].Opcode == OP.Integer);
                    int n = p.Ops[i - 1].P1;
                    if (n > maxArgs) maxArgs = n;
                }
#endif
                else if (opcode == OP.Next || opcode == OP.SorterNext)
                {
                    op.P4.Advance = Btree.Next_;
                    op.P4Type = Vdbe.P4T.ADVANCE;
                }
                else if (opcode == OP.Prev)
                {
                    op.P4.Advance = Btree.Previous;
                    op.P4Type = Vdbe.P4T.ADVANCE;
                }
                if ((op.Opflags & OPFLG.JUMP) != 0 && op.P2 < 0)
                {
                    Debug.Assert(-1 - op.P2 < p.Labels.length);
                    op.P2 = labels[-1 - op.P2];
                }
            }
            C._tagfree(p.Ctx, ref p.Labels.data);
            p.Labels.data = null;
            maxFuncArgs = maxArgs;
        }

        public int CurrentAddr()
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            return Ops.length;
        }

        public VdbeOp[] TakeOpArray(ref int opsLength, ref int maxArgs)
        {
            VdbeOp[] ops = Ops.data;
            Debug.Assert(ops != null && !Ctx.MallocFailed);
            Debug.Assert(BtreeMask == 0); // Check that sqlite3VdbeUsesBtree() was not called on this VM
            ResolveP2Values(this, ref maxArgs);
            opsLength = Ops.length;
            Ops.data = null;
            return ops;
        }

        public int AddOpList(int opsLength, VdbeOpList[] ops)
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            if (Ops.length + opsLength > OpsAlloc && GrowOps(this) != 0)
                return 0;
            int addr = Ops.length;
            if (C._ALWAYS(opsLength > 0))
            {
                VdbeOpList in_;
                for (int i = 0; i < opsLength; i++)
                {
                    in_ = ops[i];
                    int p2 = in_.P2;
                    if (Ops[i + addr] == null)
                        Ops[i + addr] = new VdbeOp();
                    VdbeOp out_ = Ops[i + addr];
                    out_.Opcode = in_.Opcode;
                    out_.P1 = in_.P1;
                    out_.P2 = (p2 < 0 && ((OPFLG)E._opcodeProperty[(int)out_.Opcode] & OPFLG.JUMP) != 0 ? addr + (-1 - p2) : p2);
                    out_.P3 = in_.P3;
                    out_.P4Type = Vdbe.P4T.NOTUSED;
                    out_.P4.P = null;
                    out_.P5 = 0;
#if DEBUG
                    out_.Comment = null;
                    if ((Ctx.Flags & BContext.FLAG.VdbeAddopTrace) != 0)
                        PrintOp(null, i + addr, Ops[i + addr]);
#endif
                }
                Ops.length += opsLength;
            }
            return addr;
        }

        public void ChangeP1(int addr, int val) { if (Ops.length > addr) Ops[addr].P1 = val; }
        public void ChangeP2(int addr, int val) { if (Ops.length > addr) Ops[addr].P2 = val; }
        public void ChangeP3(int addr, int val) { if (Ops.length > addr) Ops[addr].P3 = val; }
        public void ChangeP5(byte val) { if (Ops.data != null) { Debug.Assert(Ops.length > 0); Ops[Ops.length - 1].P5 = val; } }

        public void JumpHere(int addr)
        {
            Debug.Assert(addr >= 0);
            ChangeP2(addr, Ops.length);
        }

        static void FreeEphemeralFunction(Context ctx, FuncDef def)
        {
            if (C._ALWAYS(def != null) && (def.Flags & FUNC.EPHEM) != 0)
                C._tagfree(ctx, ref def);
        }
        static void FreeP4(Context ctx, P4T p4t, object p4)
        {
            Debug.Assert(ctx != null);
            if (p4 != null)
            {
                switch (p4t)
                {
                    case P4T.REAL:
                    case P4T.INT64:
                    case P4T.DYNAMIC:
                    case P4T.KEYINFO:
                    case P4T.INTARRAY:
                    case P4T.KEYINFO_HANDOFF:
                        {
                            C._tagfree(ctx, ref p4);
                            break;
                        }
                    case P4T.MPRINTF:
                        {
                            if (ctx.BytesFreed == 0) C._free(ref p4);
                            break;
                        }
                    case P4T.VDBEFUNC:
                        {
                            VdbeFunc vdbeFunc = (VdbeFunc)p4;
                            FreeEphemeralFunction(ctx, vdbeFunc);
                            if (ctx.BytesFreed == 0) Vdbe.DeleteAuxData(vdbeFunc, 0);
                            C._tagfree(ctx, ref vdbeFunc);
                            break;
                        }
                    case P4T.FUNCDEF:
                        {
                            FreeEphemeralFunction(ctx, (FuncDef)p4);
                            break;
                        }
                    case P4T.MEM:
                        {
                            if (ctx.BytesFreed == 0)
                                Vdbe.ValueFree(ref (Mem)p4);
                            else
                            {
                                Mem p = (Mem)p4;
                                //C._tagfree(ctx, ref p.Malloc);
                                C._tagfree(ctx, ref p);
                            }
                            break;
                        }
                    case P4T.VTAB:
                        {
                            if (ctx.BytesFreed == 0) ((VTable)p4).Unlock();
                            break;
                        }
                }
            }
        }

        static void VdbeFreeOpArray(Context ctx, ref VdbeOp[] ops, int opsLength)
        {
            if (ops != null)
            {
                for (int opIdx = ops.Length; opIdx < opsLength; opIdx++)
                {
                    VdbeOp op = ops[opIdx];
                    FreeP4(ctx, op.P4Type, op.P4.P);
#if DEBUG
                    C._tagfree(ctx, ref op.Comment);
#endif
                }
            }
            C._tagfree(ctx, ref ops);
        }

        public void LinkSubProgram(SubProgram p)
        {
            p.Next = Programs;
            Programs = p;
        }

        public void ChangeToNoop(int addr)
        {
            if (Ops.data != null)
            {
                VdbeOp op = Ops[addr];
                FreeP4(Ctx, op.P4Type, op.P4.P);
                op._memset();
                op.Opcode = OP.Noop;
            }
        }

        public void ChangeP4(int addr, CollSeq coll, int n) { ChangeP4(addr, new P4_t { Coll = coll }, n); } // P4_COLLSEQ 
        public void ChangeP4(int addr, FuncDef func, int n) { ChangeP4(addr, new P4_t { Func = func }, n); } // P4_FUNCDEF
        public void ChangeP4(int addr, int i, int n) { ChangeP4(addr, new P4_t { I = i }, n); } // P4_INT32
        public void ChangeP4(int addr, KeyInfo keyInfo, int n) { ChangeP4(addr, new P4_t { KeyInfo = keyInfo }, n); } // P4T_KEYINFO
        public void ChangeP4(int addr, char c, int n) { ChangeP4(addr, new P4_t { Z = c.ToString() }, n); } // CHAR
        public void ChangeP4(int addr, Mem m, int n) { ChangeP4(addr, new P4_t { Mem = m }, n); } // MEM
        public void ChangeP4(int addr, string z, Action<object> notUsed1) { ChangeP4(addr, new P4_t { Z = z }, (int)P4T.DYNAMIC); } // STRING + Type
        public void ChangeP4(int addr, SubProgram program, int n) { ChangeP4(addr, new P4_t { Program = program }, n); } // SUBPROGRAM
        public void ChangeP4(int addr, string z, int n) { ChangeP4(addr, new P4_t { Z = (n > 0 && n <= z.Length ? z.Substring(0, n) : z) }, n); }
        public void ChangeP4(int addr, P4_t p4, int n)
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Context ctx = Ctx;
            if (Ops.data == null || ctx.MallocFailed)
            {
                if (n != (int)P4T.KEYINFO && n != (int)P4T.VTAB)
                    FreeP4(ctx, (P4T)n, p4);
                return;
            }
            Debug.Assert(Ops.length > 0);
            Debug.Assert(addr < Ops.length);
            if (addr < 0)
                addr = Ops.length - 1;
            VdbeOp op = Ops[addr];
            FreeP4(ctx, op.P4Type, op.P4.P);
            op.P4.P = null;
            if (n == (int)P4T.INT32)
            {
                // Note: this cast is safe, because the origin data point was an int that was cast to a (string ).
                op.P4.I = p4.I; //: PTR_TO_INT(p4);
                op.P4Type = P4T.INT32;
            }
            else if (n == (int)P4T.INT64)
            {
                op.P4.I64 = p4.I64;
                op.P4Type = (P4T)n;
            }
            else if (n == (int)P4T.REAL)
            {
                op.P4.Real = p4.Real;
                op.P4Type = (P4T)n;
            }
            else if (p4 == null)
            {
                op.P4.P = null;
                op.P4Type = P4T.NOTUSED;
            }
            else if (n == (int)P4T.KEYINFO)
            {
                int fields = p4.KeyInfo.Fields;
                KeyInfo keyInfo = new KeyInfo();
                op.P4.KeyInfo = keyInfo;
                if (keyInfo != null)
                {
                    keyInfo = p4.KeyInfo._memcpy();
                    op.P4Type = P4T.KEYINFO;
                }
                else
                {
                    ctx.MallocFailed = true;
                    op.P4Type = P4T.NOTUSED;
                }
            }
            else if (n == (int)P4T.KEYINFO_HANDOFF)
            {
                op.P4.KeyInfo = p4.KeyInfo;
                op.P4Type = P4T.KEYINFO;
            }
            else if (n == (int)P4T.FUNCDEF)
            {
                op.P4.Func = p4.Func;
                op.P4Type = P4T.FUNCDEF;
            }
            else if (n == (int)P4T.COLLSEQ)
            {
                op.P4.Coll = p4.Coll;
                op.P4Type = P4T.COLLSEQ;
            }
            else if (n == (int)P4T.DYNAMIC || n == (int)P4T.STATIC || n == (int)P4T.MPRINTF)
            {
                op.P4.Z = p4.Z;
                op.P4Type = P4T.DYNAMIC;
            }
            else if (n == (int)P4T.MEM)
            {
                op.P4.Mem = p4.Mem;
                op.P4Type = P4T.MEM;
            }
            else if (n == (int)P4T.INTARRAY)
            {
                op.P4.Is = p4.Is;
                op.P4Type = P4T.INTARRAY;
            }
            else if (n == (int)P4T.SUBPROGRAM)
            {
                op.P4.Program = p4.Program;
                op.P4Type = P4T.SUBPROGRAM;
            }
            else if (n == (int)P4T.VTAB)
            {
                op.P4.VTable = p4.VTable;
                op.P4Type = P4T.VTAB;
                p4.VTable.Lock();
                Debug.Assert(p4.VTable.Ctx == ctx);
            }
            else if (n < 0)
            {
                op.P4.P = p4.P;
                op.P4Type = (P4T)n;
            }
            else
            {
                //: if (n == 0) n = _strlen30(p4);
                op.P4.Z = p4.Z;
                op.P4Type = P4T.DYNAMIC;
            }
        }

#if !NDEBUG
        public void Comment(string fmt, params object[] args)
        {
            Debug.Assert(Ops.length > 0 || Ops.data == null);
            Debug.Assert(Ops.data == null || Ops[Ops.length - 1].Comment == null || Ctx.MallocFailed);
            if (Ops.length != 0)
            {
                string z = C._vmtagprintf(Ctx, fmt, args);
                Ops[Ops.length - 1].Comment = z;
            }
        }
        public void NoopComment(string fmt, params object[] args)
        {
            AddOp0(OP.Noop);
            Debug.Assert(Ops.length > 0 || Ops.data == null);
            Debug.Assert(Ops.data == null || Ops[Ops.length - 1].Comment == null || Ctx.MallocFailed);
            if (Ops.length != 0)
            {
                string z = C._vmtagprintf(Ctx, fmt, args);
                Ops[p.Ops.length - 1].Comment = z;
            }
        }
#else
        public void Comment(string fmt, params object[] args) { }
        public void NoopComment(string fmt, params object[] args) { }
#endif

        const VdbeOp _dummy = new VdbeOp();  // Ignore the MSVC warning about no initializer
        public VdbeOp GetOp(int addr)
        {
            // C89 specifies that the constant "dummy" will be initialized to all zeros, which is correct.  MSVC generates a warning, nevertheless.
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            if (addr < 0)
            {
#if OMIT_TRACE
                if (Ops.length == 0) return _dummy;
#endif
                addr = Ops.length - 1;
            }
            Debug.Assert((addr >= 0 && addr < Ops.length) || Ctx.MallocFailed);
            return (Ctx.MallocFailed ? _dummy : Ops[addr]);
        }

#if !OMIT_EXPLAIN || !NDEBUG || VDBE_PROFILE || DEBUG
        //static StringBuilder temp = new StringBuilder(100);
        static string DisplayP4(VdbeOp op, StringBuilder temp, int tempLength)
        {
            temp.Length = 0;
            Debug.Assert(tempLength >= 20);
            switch (op.P4Type)
            {
                case P4T.KEYINFO_STATIC:
                case P4T.KEYINFO:
                    {
                        KeyInfo keyInfo = op.P4.KeyInfo;
                        Debug.Assert(keyInfo.SortOrders != null);
                        C.__snprintf(temp, tempLength, "keyinfo(%d", keyInfo.Fields);
                        int i = temp.Length;
                        for (int j = 0; j < keyInfo.Fields; j++)
                        {
                            CollSeq coll = keyInfo.Colls[j];
                            if (coll != null)
                            {
                                string collName = coll.Name;
                                int collNameLength = collName.Length;
                                if (i + collNameLength > tempLength)
                                {
                                    temp.Append(",...");
                                    break;
                                }
                                temp.Append(",");
                                if (keyInfo.SortOrders != null && keyInfo.SortOrders[j] != 0)
                                    temp.Append("-");
                                temp.Append(coll.Name);
                                i += collNameLength;
                            }
                            else if (i + 4 < tempLength)
                            {
                                temp.Append(",nil");
                                i += 4;
                            }
                        }
                        temp.Append(")");
                        Debug.Assert(i < tempLength);
                        break;
                    }
                case P4T.COLLSEQ:
                    {
                        CollSeq coll = op.P4.Coll;
                        C.__snprintf(temp, tempLength, "collseq(%.20s)", (coll != null ? coll.Name : "null"));
                        break;
                    }
                case P4T.FUNCDEF:
                    {
                        FuncDef def = op.P4.Func;
                        C.__snprintf(temp, tempLength, "%s(%d)", def.Name, def.Args);
                        break;
                    }
                case P4T.INT64:
                    {
                        C.__snprintf(temp, tempLength, "%lld", op.P4.I64);
                        break;
                    }
                case P4T.INT32:
                    {
                        C.__snprintf(temp, tempLength, "%d", op.P4.I);
                        break;
                    }
                case P4T.REAL:
                    {
                        C.__snprintf(temp, tempLength, "%.16g", op.P4.Real);
                        break;
                    }
                case P4T.MEM:
                    {
                        Mem mem = op.P4.Mem;
                        Debug.Assert((mem.Flags & MEM.Null) == 0);
                        if ((mem.Flags & MEM.Str) != 0) temp.Append(mem.Z);
                        else if ((mem.Flags & MEM.Int) != 0) C.__snprintf(temp, tempLength, "%lld", mem.u.I);
                        else if ((mem.Flags & MEM.Real) != 0) C.__snprintf(temp, tempLength, "%.16g", mem.R);
                        else { Debug.Assert((mem.Flags & MEM.Blob) != 0); temp = new StringBuilder("(blob)"); }
                        break;
                    }
#if !OMIT_VIRTUALTABLE
                case P4T.VTAB:
                    {
                        IVTable vtable = op.P4.VTable.IVTable;
                        C.__snprintf(temp, tempLength, "vtab:%p:%p", vtable, vtable.IModule);
                        break;
                    }
#endif
                case P4T.INTARRAY:
                    {
                        C.__snprintf(temp, tempLength, "intarray");
                        break;
                    }
                case P4T.SUBPROGRAM:
                    {
                        C.__snprintf(temp, tempLength, "program");
                        break;
                    }
                default:
                    {
                        if (op.P4.Z != null)
                            temp.Append(op.P4.Z);
                        break;
                    }
            }
            Debug.Assert(temp != null);
            return temp.ToString();
        }
#endif

        public void UsesBtree(int i)
        {
            Debug.Assert(i >= 0 && i < Ctx.DBs.length && i < (int)sizeof(yDbMask) * 8);
            Debug.Assert(i < (int)sizeof(yDbMask) * 8);
            BtreeMask |= ((yDbMask)1) << i;
            if (i != 1 && Ctx.DBs[i].Bt.Sharable())
                LockMask |= ((yDbMask)1) << i;
        }

#if !(OMIT_SHARED_CACHE) && THREADSAFE
        public void Enter()
        {
            if (LockMask == 0) return;  // The common case
            Context ctx = Ctx;
            array_t<BContext.DB> dbs = ctx.DBs;
            int i;
            yDbMask mask;
            for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
                if (i != 1 && (mask & LockMask) != 0 && C._ALWAYS(dbs[i].Bt != null))
                    dbs[i].Bt.Enter();
        }

        public void Leave()
        {
            if (LockMask == 0) return;  // The common case
            Context ctx = Ctx;
            array_t<BContext.DB> dbs = ctx.DBs;
            int i;
            yDbMask mask;
            for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
                if (i != 1 && (mask & LockMask) != 0 && C._ALWAYS(dbs[i].Bt != null))
                    dbs[i].Bt.Leave();
        }
#endif


#if VDBE_PROFILE || DEBUG
        public static void PrintOp(FILE out_, int pc, VdbeOp op)
        {
            if (out_ == null) out_ = Console.Out;
            StringBuilder ptr = new StringBuilder(50);
            string p4 = DisplayP4(op, ptr, ptr.Length);
            StringBuilder out2_ = new StringBuilder(10);
            C.__snprintf(out2_, 999, "%4d %-13s %4d %4d %4d %-4s %.2X %s\n", pc,
            OpcodeName(op.Opcode), op.P1, op.P2, op.P3, p4, op.P5,
#if  DEBUG
 (op.Comment != null ? op.Comment : string.Empty)
#else
 string.Empty
#endif
);
            out_.Write(out2_);
        }
#endif

        static void ReleaseMemArray(Mem[] p, int n) { ReleaseMemArray(p, 0, n); }
        static void ReleaseMemArray(Mem[] p, int offset, int n)
        {
            if (p != null && p.Length > offset && p[offset] != null && n != 0)
            {
                Context ctx = p[offset].Ctx;
                bool mallocFailed = ctx.MallocFailed;
                if (ctx != null && ctx.BytesFreed != 0)
                {
                    for (int pIdx = offset; pIdx < n; pIdx++)
                        C._tagfree(ctx, ref p[pIdx].TextBuilder_);
                    return;
                }
                Mem end = p[n];
                for (int pIdx = offset; pIdx < n; pIdx++)
                {
                    var p2 = p[pIdx];
                    Debug.Assert((p[pIdx + 1]) == end || n == 1 || pIdx == p.Length - 1 || p[pIdx].Ctx == p[pIdx + 1].Ctx);

                    // This block is really an inlined version of sqlite3VdbeMemRelease() that takes advantage of the fact that the memory cell value is 
                    // being set to NULL after releasing any dynamic resources.
                    //
                    // The justification for duplicating code is that according to callgrind, this causes a certain test case to hit the CPU 4.7 
                    // percent less (x86 linux, gcc version 4.1.2, -O6) than if sqlite3MemRelease() were called from here. With -O2, this jumps
                    // to 6.6 percent. The test case is inserting 1000 rows into a table with no indexes using a single prepared INSERT statement, bind() 
                    // and reset(). Inserts are grouped into a transaction.
                    if (p2 != null)
                    {
                        if ((p2.Flags & (MEM.Agg | MEM.Dyn | MEM.Frame | MEM.RowSet)) != 0)
                            MemRelease(p2);
                        else if (p2.Mem_ != null)
                        {
                            C._tagfree(ctx, ref p2.Mem_);
                            p2.Mem_ = null;
                        }
                        else if (p2.TextBuilder_ != null)
                        {
                            C._tagfree(ctx, ref p2.TextBuilder_);
                            p2.TextBuilder_ = null;
                        }
                        p2.Flags = MEM.Invalid;
                        p2.Z = null;
                    }
                }
                ctx.MallocFailed = mallocFailed;
            }
        }

        public static void FrameDelete(VdbeFrame p)
        {
            int i;
            //Mem[] mems = VdbeFrameMem(p);
            VdbeCursor[] cursors = p.aChildCsr;// (VdbeCursor)aMem[p.nChildMem];
            for (i = 0; i < p.nChildCsr; i++)
                FreeCursor(p.v, cursors[i]);
            ReleaseMemArray(p.aChildMem, p.nChildMem);
            p = null;// sqlite3DbFree( p.v.db, p );
        }

#if !OMIT_EXPLAIN
        public RC List()
        {
            Context ctx = Ctx; // The database connection
            int i;

            Debug.Assert(HasExplain != 0);
            Debug.Assert(Magic == VDBE_MAGIC_RUN);
            Debug.Assert(RC_ == RC.OK || RC_ == RC.BUSY || RC_ == RC.NOMEM);

            // Even though this opcode does not use dynamic strings for the result, result columns may become dynamic if the user calls
            // sqlite3_column_text16(), causing a translation to UTF-16 encoding.
            if (ResultSet == null) ResultSet = new Mem[0]; //Mem* pMem = p.pResultSet = p.aMem[1];
            ReleaseMemArray(ResultSet, 8);
            ResultSet = null;

            if (RC_ == RC.NOMEM)
            {
                // This happens if a malloc() inside a call to sqlite3_column_text() or sqlite3_column_text16() failed.
                ctx.MallocFailed = true;
                return RC.ERROR;
            }

            // When the number of output rows reaches rows, that means the listing has finished and sqlite3_step() should return SQLITE_DONE.
            // rows is the sum of the number of rows in the main program, plus the sum of the number of rows in all trigger subprograms encountered
            // so far.  The rows value will increase as new trigger subprograms are encountered, but p->pc will eventually catch up to rows.
            int rows = Ops.length; // Stop when row count reaches this
            SubProgram[] subs = null; // Array of sub-vdbes
            Mem sub = null; // Memory cell hold array of subprogs
            int subsLength = 0; // Number of sub-vdbes seen so far

            if (HasExplain == 1)
            {
                // The first 8 memory cells are used for the result set.  So we will commandeer the 9th cell to use as storage for an array of pointers
                // to trigger subprograms.  The VDBE is guaranteed to have at least 9 cells.
                Debug.Assert(Mems.length > 9);
                sub = Mems[9];
                if ((sub.Flags & MEM.Blob) != 0)
                {
                    // On the first call to sqlite3_step(), sub will hold a NULL.  It is initialized to a BLOB by the P4_SUBPROGRAM processing logic below
                    subsLength = subs.Length; //: sub->N/sizeof(Vdbe *);
                    subs = Mems[9].SubProgram_; //: (SubProgram **)sub->Z
                }
                for (i = 0; i < subsLength; i++)
                    rows += subs[i].Ops.length;
            }

            int memIdx = 0;
            if (memIdx >= ResultSet.Length) Array.Resize(ref ResultSet, 8 + ResultSet.Length);
            ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
            Mem mem = ResultSet[memIdx++]; // First Mem of result set

            do
            {
                i = PC++;
            } while (i < rows && HasExplain == 2 && Ops[i].Opcode != OP.Explain);

            RC rc = RC.OK;
            if (i >= rows)
            {
                RC_ = RC.OK;
                rc = RC.DONE;
            }
            else if (ctx.u1.IsInterrupted)
            {
                RC_ = RC.INTERRUPT;
                rc = RC.ERROR;
                C._setstring(ref ErrMsg, ctx, sqlite3ErrStr(RC_));
            }
            else
            {
                string z;
                VdbeOp op;
                if (i < Ops.length)
                    op = Ops[i]; // The output line number is small enough that we are still in the main program.
                else
                {
                    // We are currently listing subprograms.  Figure out which one and pick up the appropriate opcode.
                    i -= Ops.length;
                    int j;
                    for (j = 0; i >= subs[j].Ops.length; j++)
                        i -= subs[j].Ops.length;
                    op = subs[j].Ops[i];
                }
                if (HasExplain == 1)
                {
                    mem.Flags = MEM.Int;
                    mem.Type = TYPE.INTEGER;
                    mem.u.I = i; // Program counter
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; //: mem++;

                    mem.Flags = MEM.Static | MEM.Str | MEM.Term;
                    mem.Z = OpcodeName(op.Opcode); // Opcode
                    Debug.Assert(mem.Z != null);
                    mem.N = mem.Z.Length;
                    mem.Type = TYPE.TEXT;
                    mem.Encode = TEXTENCODE.UTF8;
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; //: mem++;

                    // When an OP_Program opcode is encounter (the only opcode that has a P4_SUBPROGRAM argument), expand the size of the array of subprograms
                    // kept in p->aMem[9].z to hold the new program - assuming this subprogram has not already been seen.
                    if (op.P4Type == P4T.SUBPROGRAM)
                    {
                        //: int bytes = (subsLength+1)*sizeof(SubProgram *);
                        int j;
                        for (j = 0; j < subsLength; j++)
                            if (subs[j] == op.P4.Program)
                                break;
                        if (j == subsLength)
                        {
                            //: && MemGrow(sub, bytes, subsLength != 0) == RC.OK)
                            Array.Resize(ref subs, subsLength + 1);
                            sub.SubProgram_ = subs; //: (SubProgram)pSub.z;
                            subs[subsLength++] = op.P4.Program;
                            sub.Flags |= MEM.Blob;
                            sub.N = 0; //: subsLength*sizeof(SubProgram *);
                        }
                    }
                }

                mem.Flags = MEM.Int;
                mem.u.I = op.P1; // P1
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                mem.Flags = MEM.Int;
                mem.u.I = op.P2; // P2
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                mem.Flags = MEM.Int;
                mem.u.I = op.P3; // P3
                mem.Type = TYPE.INTEGER;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                //if (MemGrow(mem, 32, 0) != 0) // P4
                //{
                //    Debug.Assert(ctx.MallocFailed);
                //    return RC.ERROR;
                //}

                mem.Flags = MEM.Dyn | MEM.Str | MEM.Term;
                z = DisplayP4(op, mem.Z, 32);
                if (z != mem.Z)
                    MemSetStr(mem, z, -1, TEXTENCODE.UTF8, null);
                else
                {
                    Debug.Assert(mem.Z != null);
                    mem.N = mem.Z.Length;
                    mem.Encode = TEXTENCODE.UTF8;
                }
                mem.Type = TYPE.TEXT;
                if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                mem = ResultSet[memIdx++]; //: mem++;

                if (HasExplain == 1)
                {
                    //if (MemGrow(mem, 4, 0) != 0)
                    //{
                    //    Debug.Assert(ctx.MallocFailed);
                    //    return RC.ERROR;
                    //}
                    mem.Flags = MEM.Dyn | MEM.Str | MEM.Term;
                    mem.N = 2;
                    mem.Z = op.P5.ToString("x2"); //: __snprintf(mem->Z, 3, "%.2x", op->P5); // P5
                    mem.Type = TYPE.TEXT;
                    mem.Encode = TEXTENCODE.UTF8;
                    if (ResultSet[memIdx] == null) ResultSet[memIdx] = C._alloc(ResultSet[memIdx]);
                    mem = ResultSet[memIdx++]; // mem++;

#if DEBUG
                    if (op.Comment != null)
                    {
                        mem.Flags = MEM.Str | MEM.Term;
                        mem.Z = op.Comment; // Comment
                        mem.N = (mem.Z != null ? mem.Z.Length : 0);
                        mem.Encode = TEXTENCODE.UTF8;
                        mem.Type = TYPE.TEXT;
                    }
                    else
#endif
                    {
                        mem.Flags = MEM.Null;
                        mem.Type = TYPE.NULL;
                    }
                }

                ResColumns = (ushort)(8 - 4 * (HasExplain - 1));
                RC_ = RC.OK;
                rc = RC.ROW;
            }
            return rc;
        }
#endif

#if  DEBUG
        public void PrintSql()
        {
            int opsLength = Ops.length;
            if (opsLength < 1) return;
            VdbeOp op = Ops[0];
            if (op.Opcode == OP.Trace && op.P4.Z != null)
            {
                string z = op.P4.Z;
                z = z.Trim(); //: while (_isspace(*z)) z++;
                Console.Write("SQL: [{0}]\n", z);
            }
        }
#endif

#if !OMIT_TRACE && ENABLE_IOTRACE
        public void IOTraceSql()
        {
            if (!SysEx.IOTrace) return;
            int opsLength = Ops.length;
            if (opsLength < 1) return;
            VdbeOp op = Ops[0];
            if (op.Opcode == OP.Trace && op.P4.Z != null)
            {
                string z = string.Empty;
                C.__snprintf(z, 1000, "%s", op.P4.Z);
                SysEx.IOTRACE("SQL %s\n", z.Trim());
            }
        }
#endif

        //: __device__ static void *AllocSpace(void *buf, int bytes, uint8 **from, uint8 *end, int *bytesOut)
        //: {
        //:     _assert(_HASALIGNMENT8(*from));
        //:     if (buf) return buf;
        //:     bytes = _ROUND8(bytes);
        //:     if (&(*from)[bytes] <= end)
        //:     {
        //:         buf = (void *)*from;
        //:         *from += bytes;
        //:     }
        //:     else
        //:         *bytesOut += bytes;
        //:     return buf;
        //: }

        public void Rewind()
        {
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Debug.Assert(Ops.length > 0); // There should be at least one opcode.

            // Set the magic to VDBE_MAGIC_RUN sooner rather than later.
            Magic = VDBE_MAGIC_RUN;
#if DEBUG
            for (int i = 1; i < Mems.length; i++)
                Debug.Assert(Mems[i].Ctx == Ctx);
#endif
            PC = -1;
            RC_ = RC.OK;
            ErrorAction = OE.Abort;
            Magic = VDBE_MAGIC_RUN;
            Changes = 0;
            CacheCtr = 1;
            MinWriteFileFormat = 255;
            StatementID = 0;
            FkConstraints = 0;
#if VDBE_PROFILE
            for (int i = 0; i < Ops.length; i++)
            {
                Ops[i].Cnt = 0;
                Ops[i].Cycles = 0;
            }
#endif
        }

        public void MakeReady(Parse parse)
        {
            Debug.Assert(parse != null);
            Debug.Assert(Magic == VDBE_MAGIC_INIT);
            Context ctx = Ctx; // The database connection
            Debug.Assert(!ctx.MallocFailed);
            int vars = parse.Vars.length; // Number of parameters
            int mems = parse.Mems; // Number of VM memory registers
            int cursors = parse.Tabs; // Number of cursors required
            int args = parse.MaxArgs; // Number of arguments in subprograms
            int onces = parse.Onces; // Number of OP_Once instructions
            if (onces == 0) onces = 1; // Ensure at least one byte in p->aOnceFlag[]

            // For each cursor required, also allocate a memory cell. Memory cells (nMem+1-nCursor)..nMem, inclusive, will never be used by
            // the vdbe program. Instead they are used to allocate space for VdbeCursor/BtCursor structures. The blob of memory associated with 
            // cursor 0 is stored in memory cell nMem. Memory cell (nMem-1) stores the blob of memory associated with cursor 1, etc.
            //
            // See also: allocateCursor().
            mems += cursors;

            // Allocate space for memory registers, SQL variables, VDBE cursors and an array to marshal SQL function arguments in.
            //: uint8* csr = (uint8*)&Ops[Ops.length]; // Memory avaliable for allocation
            //: uint8* end = (uint8*)&Ops[OpsAlloc]; // First byte past end of zCsr[]
            ResolveP2Values(this, ref args);
            UsesStmtJournal = (parse.IsMultiWrite != 0 && parse.MayAbort);
            if (parse.Explain != 0 && mems < 10)
                mems = 10;
            //: _memset(csr, 0, end - csr);
            //: csr += (csr - (uint8*)0) & 7;
            //: _assert(_HASALIGNMENT8(csr));
            Expired = false;

            // Memory for registers, parameters, cursor, etc, is allocated in two passes.  On the first pass, we try to reuse unused space at the 
            // end of the opcode array.  If we are unable to satisfy all memory requirements by reusing the opcode array tail, then the second
            // pass will fill in the rest using a fresh allocation.  
            //
            // This two-pass approach that reuses as much memory as possible from the leftover space at the end of the opcode array can significantly
            // reduce the amount of memory held by a prepared statement.
            int n;
            //int bytes; // How much extra memory is needed
            //: do
            //: {
            //:     bytes = 0;
            //:     Mems.data = (Mem*)AllocSpace(Mems.data, mems * sizeof(Mem), &csr, end, &bytes);
            //:     Vars.data = (Mem*)AllocSpace(Vars.data, vars * sizeof(Mem), &csr, end, &bytes);
            //:     Args = (Mem**)AllocSpace(Args, args * sizeof(Mem*), &csr, end, &bytes);
            //:     VarNames.data = (char**)AllocSpace(VarNames.data, vars * sizeof(char*), &csr, end, &bytes);
            //:     Cursors.data = (VdbeCursor**)AllocSpace(Cursors.data, cursors * sizeof(VdbeCursor*), &csr, end, &bytes);
            //:     OnceFlags.data = (uint8*)AllocSpace(OnceFlags.data, onces, &csr, end, &bytes);
            //:     if (bytes)
            //:         FreeThis = _tagalloc2(ctx, bytes, true);
            //:     csr = FreeThis;
            //:     end = &csr[bytes];
            //: } while (bytes && !ctx->MallocFailed);
            Mems.data = new Mem[mems + 1]; //C#: aMem is 1 based, so allocate 1 extra cell
            Vars.data = new Mem[vars == 0 ? 1 : vars];
            Args = new Mem[args == 0 ? 1 : args];
            VarNames.data = new string[vars == 0 ? 1 : vars];            
            Cursors.data = new VdbeCursor[cursors == 0 ? 1 : cursors]; Cursors[0] = new VdbeCursor();
            OnceFlags.data = new byte[onces == 0 ? 1 : onces];

            Cursors.length = cursors;
            OnceFlags.length = onces;
            if (Vars.data != null)
            {
                Vars.length = (yVar)vars;
                for (n = 0; n < vars; n++)
                {
                    Vars[n] = C._alloc(Vars[n])
                    Vars[n].Flags = MEM.Null;
                    Vars[n].Ctx = ctx;
                }
            }
            if (VarNames.data != null)
            {
                VarNames.length = parse.Vars.length;
                for (n = 0; n < VarNames.length; n++)
                {
                    VarNames[n] = parse.Vars[n]; //: _memcpy(VarNames.data, parse->Vars.data, VarNames.length*sizeof(VarNames[0]));
                    parse.Vars[n] = string.Empty; //: _memset(parse->Vars.data, 0, parse->Vars.length*sizeof(parse->Vars[0]));
                }
            }
            if (Mems.data != null)
            {
                //: Mems.data--; // aMem[] goes from 1..nMem
                Mems.length = mems; // not from 0..nMem-1
                for (n = 0; n <= mems; n++)
                {
                    Mems[n] = C._alloc(Mems[n]);
                    Mems[n].Flags = MEM.Null;
                    Mems[n].Ctx = ctx;
                }
            }
            HasExplain = parse.Explain;
            Rewind();
        }

        public void FreeCursor(VdbeCursor cur)
        {
            if (cur == null)
                return;

            if (cur.Bt != null)
                cur.Bt.Close(); // The pCx.pCursor will be close automatically, if it exists, by the call above.
            else if (cur.Cursor != null)
                Btree.CloseCursor(cur.Cursor);
#if !OMIT_VIRTUALTABLE
            if (cur.VtabCursor != null)
            {
                IVTableCursor vtabCursor = cur.VtabCursor;
                ITableModule module = cur.IModule;
                InVtabMethod = 1;
                module.Close(vtabCursor);
                InVtabMethod = 0;
            }
#endif
        }

        public static int FrameRestore(VdbeFrame frame)
        {
            Vdbe v = frame.V;
            v.OnceFlags.data = frame.OnceFlags.data;
            v.OnceFlags.length = frame.OnceFlags.length;
            v.Ops.data = frame.Ops.data;
            v.Ops.length = frame.Ops.length;
            v.Mems.data = frame.Mems.data;
            v.Mems.length = frame.Mems.length;
            v.Cursors.data = frame.Cursors.data;
            v.Cursors.length = frame.Cursors.length;
            v.Ctx.LastRowID = frame.LastRowID;
            v.Changes = frame.Changes;
            return frame.PC;
        }

        static void CloseAllCursors(Vdbe p)
        {
            if (p.Frames.data != null)
            {
                VdbeFrame frame;
                for (frame = p.Frames.data; frame.Parent != null; frame = frame.Parent) { }
                FrameRestore(frame);
            }
            p.Frames.data = null;
            p.Frames.length = 0;
            if (p.Cursors.data != null)
            {
                int i;
                for (i = 0; i < p.Cursors.length; i++)
                {
                    VdbeCursor cur = p.Cursors[i];
                    if (cur != null)
                    {
                        p.FreeCursor(cur);
                        p.Cursors[i] = null;
                    }
                }
            }
            if (p.Mems.data != null)
                ReleaseMemArray(p.Mems.data, 1, p.Mems.length);
            while (p.DelFrames != null)
            {
                VdbeFrame del = p.DelFrames;
                p.DelFrames = del.Parent;
                FrameDelete(del);
            }
        }

        static void Cleanup(Vdbe p)
        {
            Context ctx = p.Ctx;
#if DEBUG
            // Execute assert() statements to ensure that the Vdbe.apCsr[] and Vdbe.aMem[] arrays have already been cleaned up.
            int i;
            if (p.Cursors.data != null) for (i = 0; i <= p.Cursors.length; i++) Debug.Assert(p.Cursors[i] == null);
            if (p.Mems.data != null)
                for (i = 1; i <= p.Mems.length; i++) Debug.Assert(p.Mems[i].Flags == MEM.Invalid);
#endif
            C._tagfree(ctx, ref p.ErrMsg);
            p.ErrMsg = null;
            p.ResultSet = null;
        }

        public void SetNumCols(int resColumns)
        {
            Context ctx = Ctx;
            ReleaseMemArray(ColNames, ResColumns * COLNAME_N);
            C._tagfree(ctx, ref ColNames);
            int n = resColumns * COLNAME_N;
            ResColumns = (ushort)resColumns;
            Mem colName;
            ColNames = new Mem[n]; //: (Mem *)_tagalloc2(ctx, sizeof(Mem)*n, true);
            if (ColNames == null) return;
            while (n-- > 0)
            {
                ColNames[n] = C._tagalloc(ColNames[n]);
                colName = ColNames[n];
                colName.Flags = MEM.Null;
                colName.Ctx = ctx;
            }
        }

        public RC SetColName(int idx, int var, string name, Action<object> del)
        {
            Debug.Assert(idx < ResColumns);
            Debug.Assert(var < COLNAME_N);
            if (Ctx.MallocFailed)
            {
                Debug.Assert(name == null || del != C.DESTRUCTOR_DYNAMIC);
                return RC.NOMEM;
            }
            Debug.Assert(ColNames != null);
            Mem colName = ColNames[idx + var * ResColumns];
            RC rc = MemSetStr(colName, name, -1, TEXTENCODE.UTF8, del);
            Debug.Assert(rc != 0 || name == null || (colName.Flags & MEM.Term) != 0);
            return rc;
        }

        static RC VdbeCommit(Context ctx, Vdbe p)
        {
#if !OMIT_VIRTUALTABLE
            // With this option, sqlite3VtabSync() is defined to be simply SQLITE_OK so p is not used. 
#endif

            // Before doing anything else, call the xSync() callback for any virtual module tables written in this transaction. This has to
            // be done before determining whether a master journal file is required, as an xSync() callback may add an attached database
            // to the transaction.
            RC rc = VTable.Sync(ctx, ref p.ErrMsg);

            int trans = 0; // Number of databases with an active write-transaction
            bool needXcommit = false;

            // This loop determines (a) if the commit hook should be invoked and (b) how many database files have open write transactions, not 
            // including the temp database. (b) is important because if more than one database file has an open write transaction, a master journal
            // file is required for an atomic commit.
            int i;
            for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
            {
                Btree bt = ctx.DBs[i].Bt;
                if (bt.IsInTrans())
                {
                    needXcommit = true;
                    if (i != 1) trans++;
                    bt.Enter();
                    rc = bt.get_Pager().ExclusiveLock();
                    bt.Leave();
                }
            }
            if (rc != RC.OK)
                return rc;

            // If there are any write-transactions at all, invoke the commit hook
            if (needXcommit && ctx.CommitCallback != null)
            {
                rc = ctx.CommitCallback(ctx.CommitArg);
                if (rc != 0)
                    return RC.CONSTRAINT_COMMITHOOK;
            }

            // The simple case - no more than one database file (not counting the TEMP database) has a transaction active.   There is no need for the
            // master-journal.
            //
            // If the return value of sqlite3BtreeGetFilename() is a zero length string, it means the main database is :memory: or a temp file.  In 
            // that case we do not support atomic multi-file commits, so use the simple case then too.
            if (ctx.DBs[0].Bt.get_Filename().Length == 0 || trans <= 1)
            {
                for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        rc = bt.CommitPhaseOne(null);
                }

                // Do the commit only if all databases successfully complete phase 1. If one of the BtreeCommitPhaseOne() calls fails, this indicates an
                // IO error while deleting or truncating a journal file. It is unlikely, but could happen. In this case abandon processing and return the error.
                for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        rc = bt.CommitPhaseTwo(false);
                }
                if (rc == RC.OK)
                    VTable.Commit(ctx);
            }

            // The complex case - There is a multi-file write-transaction active. This requires a master journal file to ensure the transaction is
            // committed atomicly.
#if !OMIT_DISKIO
            else
            {
                VSystem vfs = ctx.Vfs;
                string mainFileName = ctx.DBs[0].Bt.get_Filename();

                // Select a master journal file name
                string masterName = string.Empty; // File-name for the master journal
                if (masterName == null) return RC.NOMEM;
                int res = 0;
                int retryCount = 0;
                VFile master = null;
                do
                {
                    if (retryCount != 0)
                    {
                        if (retryCount > 100)
                        {
                            SysEx.LOG(RC.FULL, "MJ delete: %s", masterName);
                            vfs.Delete(masterName, false);
                            break;
                        }
                        else if (retryCount == 1)
                            SysEx.LOG(RC.FULL, "MJ collide: %s", masterName);
                    }
                    retryCount++;
                    long random = 0;
                    SysEx.PutRandom(sizeof(uint), ref random);
                    masterName = C._mtagprintf(ctx, "%s--mj%06X9%02X", mainFileName, (random >> 8) & 0xffffff, random & 0xff);
                    // The antipenultimate character of the master journal name must be "9" to avoid name collisions when using 8+3 filenames.
                    Debug.Assert(masterName[masterName.Length - 3] == '9');
                    SysEx.FileSuffix3(mainFileName, ref masterName);
                    rc = vfs.Access(masterName, VSystem.ACCESS.EXISTS, out res);
                } while (rc == RC.OK && res != 0);
                VSystem.OPEN dummy1;
                if (rc == RC.OK)
                    rc = vfs.OpenAndAlloc(masterName, out master, VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE | VSystem.OPEN.EXCLUSIVE | VSystem.OPEN.MASTER_JOURNAL, out dummy1); // Open the master journal. 
                if (rc != RC.OK)
                {
                    C._tagfree(ctx, ref masterName);
                    return rc;
                }

                // Write the name of each database file in the transaction into the new master journal file. If an error occurs at this point close
                // and delete the master journal file. All the individual journal files still have 'null' as the master journal pointer, so they will roll
                // back independently if a failure occurs.
                bool needSync = false;
                long offset = 0;
                for (i = 0; i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt.IsInTrans())
                    {
                        string fileName = bt.get_Journalname();
                        if (fileName == null)
                            continue; // Ignore TEMP and :memory: databases
                        Debug.Assert(fileName != string.Empty);
                        if (!needSync && !bt.SyncDisabled())
                            needSync = true;
                        rc = master.Write(Encoding.UTF8.GetBytes(fileName), fileName.Length, offset);
                        offset += fileName.Length;
                        if (rc != RC.OK)
                        {
                            master.CloseAndFree();
                            vfs.Delete(masterName, false);
                            C._tagfree(ctx, ref masterName);
                            return rc;
                        }
                    }
                }

                // Sync the master journal file. If the IOCAP_SEQUENTIAL device flag is set this is not required.
                if (needSync && (master.get_DeviceCharacteristics() & VFile.IOCAP.SEQUENTIAL) == 0 && (rc = master.Sync(VFile.SYNC.NORMAL)) != RC.OK)
                {
                    master.CloseAndFree();
                    vfs.Delete(masterName, false);
                    C._tagfree(ctx, ref masterName);
                    return rc;
                }

                // Sync all the db files involved in the transaction. The same call sets the master journal pointer in each individual journal. If
                // an error occurs here, do not delete the master journal file.
                //
                // If the error occurs during the first call to sqlite3BtreeCommitPhaseOne(), then there is a chance that the
                // master journal file will be orphaned. But we cannot delete it, in case the master journal file name was written into the journal
                // file before the failure occurred.
                for (i = 0; rc == RC.OK && i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        rc = bt.CommitPhaseOne(masterName);
                }
                master.CloseAndFree();
                Debug.Assert(rc != RC.BUSY);
                if (rc != RC.OK)
                {
                    C._tagfree(ctx, ref masterName);
                    return rc;
                }

                // Delete the master journal file. This commits the transaction. After doing this the directory is synced again before any individual
                // transaction files are deleted.
                rc = vfs.Delete(masterName, true);
                C._tagfree(ctx, ref masterName);
                masterName = null;
                if (rc != 0)
                    return rc;

                // All files and directories have already been synced, so the following calls to sqlite3BtreeCommitPhaseTwo() are only closing files and
                // deleting or truncating journals. If something goes wrong while this is happening we don't really care. The integrity of the
                // transaction is already guaranteed, but some stray 'cold' journals may be lying around. Returning an error code won't help matters.
                Pager.disable_simulated_io_errors();
                C._benignalloc_begin();
                for (i = 0; i < ctx.DBs.length; i++)
                {
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                        bt.CommitPhaseTwo(false);
                }
                C._benignalloc_end();
                Pager.enable_simulated_io_errors();

                VTable.Commit(ctx);
            }
#endif
            return rc;
        }

#if !NDEBUG
        static void CheckActiveVdbeCnt(Context ctx)
        {
            int cnt = 0;
            int writes = 0;
            Vdbe p = ctx.Vdbes;
            while (p != null)
            {
                if (p.Magic == VDBE_MAGIC_RUN && p.PC >= 0)
                {
                    cnt++;
                    if (!p.ReadOnly)
                        writes++;
                }
                p = p.Next;
            }
            Debug.Assert(cnt == ctx.ActiveVdbeCnt);
            Debug.Assert(writes == ctx.WriteVdbeCnt);
        }
#else
        static void CheckActiveVdbeCnt(Context ctx) { }
#endif

        public RC CloseStatement(IPager.SAVEPOINT op)
        {
            Context ctx = Ctx;
            RC rc = RC.OK;

            // If statementID is greater than zero, then this Vdbe opened a statement transaction that should be closed here. The only exception
            // is that an IO error may have occurred, causing an emergency rollback. In this case (db->nStatement==0), and there is nothing to do.
            if (ctx.Statements != 0 && StatementID != 0)
            {
                int savepoint = StatementID - 1;

                Debug.Assert(op == IPager.SAVEPOINT.ROLLBACK || op == IPager.SAVEPOINT.RELEASE);
                Debug.Assert(ctx.Statements > 0);
                Debug.Assert(StatementID == (ctx.Statements + ctx.SavepointsLength));

                for (int i = 0; i < ctx.DBs.length; i++)
                {
                    RC rc2 = RC.OK;
                    Btree bt = ctx.DBs[i].Bt;
                    if (bt != null)
                    {
                        if (op == IPager.SAVEPOINT.ROLLBACK)
                            rc2 = bt.Savepoint(IPager.SAVEPOINT.ROLLBACK, savepoint);
                        if (rc2 == RC.OK)
                            rc2 = bt.Savepoint(IPager.SAVEPOINT.RELEASE, savepoint);
                        if (rc == RC.OK)
                            rc = rc2;
                    }
                }
                ctx.Statements--;
                StatementID = 0;

                if (rc == RC.OK)
                {
                    if (op == IPager.SAVEPOINT.ROLLBACK)
                        rc = VTable.Savepoint(ctx, IPager.SAVEPOINT.ROLLBACK, savepoint);
                    if (rc == RC.OK)
                        rc = VTable.Savepoint(ctx, IPager.SAVEPOINT.RELEASE, savepoint);
                }

                // If the statement transaction is being rolled back, also restore the database handles deferred constraint counter to the value it had when 
                // the statement transaction was opened.
                if (op == IPager.SAVEPOINT.ROLLBACK)
                    ctx.DeferredCons = StmtDefCons;
            }
            return rc;
        }

#if !OMIT_FOREIGN_KEY
        public RC CheckFk(bool deferred)
        {
            Context ctx = Ctx;
            if ((deferred && ctx.DeferredCons > 0) || (!deferred && FkConstraints > 0))
            {
                RC_ = RC.CONSTRAINT;
                ErrorAction = OE.Abort;
                C._setstring(ref ErrMsg, ctx, "foreign key constraint failed");
                return RC.ERROR;
            }
            return RC.OK;
        }
#endif

        public RC Halt()
        {
            RC rc;
            Context ctx = Ctx;

            // This function contains the logic that determines if a statement or transaction will be committed or rolled back as a result of the
            // execution of this virtual machine. 
            //
            // If any of the following errors occur:
            //
            //     RC_NOMEM
            //     RC_IOERR
            //     RC_FULL
            //     RC_INTERRUPT
            //
            // Then the internal cache might have been left in an inconsistent state.  We need to rollback the statement transaction, if there is
            // one, or the complete transaction if there is no statement transaction.
            if (ctx.MallocFailed)
                RC_ = RC.NOMEM;

            if (OnceFlags.data != null) Array.Clear(OnceFlags.data, 0, OnceFlags.length);
            CloseAllCursors(this);
            if (Magic != VDBE_MAGIC_RUN)
                return RC.OK;
            CheckActiveVdbeCnt(ctx);

            // No commit or rollback needed if the program never started
            if (PC >= 0)
            {
                IPager.SAVEPOINT statementOp = 0;
                // Lock all btrees used by the statement
                Enter();

                // Check for one of the special errors
                RC mrc = (RC)((int)RC_ & 0xff); // Primary error code from p.rc
                Debug.Assert(RC_ != RC.IOERR_BLOCKED); // This error no longer exists
                bool isSpecialError = (mrc == RC.NOMEM || mrc == RC.IOERR || mrc == RC.INTERRUPT || mrc == RC.FULL); // Set to true if a 'special' error
                if (isSpecialError)
                {
                    // If the query was read-only and the error code is SQLITE_INTERRUPT, no rollback is necessary. Otherwise, at least a savepoint 
                    // transaction must be rolled back to restore the database to a consistent state.
                    //
                    // Even if the statement is read-only, it is important to perform a statement or transaction rollback operation. If the error 
                    // occurred while writing to the journal, sub-journal or database file as part of an effort to free up cache space (see function
                    // pagerStress() in pager.c), the rollback is required to restore the pager to a consistent state.
                    if (!ReadOnly || mrc != RC.INTERRUPT)
                    {
                        if ((mrc == RC.NOMEM || mrc == RC.FULL) && UsesStmtJournal)
                            statementOp = IPager.SAVEPOINT.ROLLBACK;
                        else
                        {
                            // We are forced to roll back the active transaction. Before doing so, abort any other statements this handle currently has active.
                            Main.RollbackAll(ctx);
                            Main.CloseSavepoints(ctx);
                            ctx.AutoCommit = 1;
                        }
                    }
                }

                // Check for immediate foreign key violations.
                if (RC_ == RC.OK)
                    CheckFk(false);

                // If the auto-commit flag is set and this is the only active writer VM, then we do either a commit or rollback of the current transaction. 
                //
                // Note: This block also runs if one of the special errors handled above has occurred. 
                if (!VTable.InSync(ctx) && ctx.AutoCommit != 0 && ctx.WriteVdbeCnt == (!ReadOnly ? 1 : 0))
                {
                    if (RC_ == RC.OK || (ErrorAction == OE.Fail && !isSpecialError))
                    {
                        rc = CheckFk(true);
                        if (rc != RC.OK)
                        {
                            if (C._NEVER(ReadOnly))
                            {
                                Leave();
                                return RC.ERROR;
                            }
                            rc = RC.CONSTRAINT;
                        }
                        else
                            // The auto-commit flag is true, the vdbe program was successful or hit an 'OR FAIL' constraint and there are no deferred foreign
                            // key constraints to hold up the transaction. This means a commit is required.
                            rc = VdbeCommit(ctx, this);
                        if (rc == RC.BUSY && ReadOnly)
                        {
                            Leave();
                            return RC.BUSY;
                        }
                        else if (rc != RC.OK)
                        {
                            RC_ = rc;
                            Main.RollbackAll(ctx);
                        }
                        else
                        {
                            ctx.DeferredCons = 0;
                            Main.CommitInternalChanges(ctx);
                        }
                    }
                    else
                        Main.RollbackAll(ctx);
                    ctx.Statements = 0;
                }
                else if (statementOp == 0)
                {
                    if (RC_ == RC.OK || ErrorAction == OE.Fail)
                        statementOp = IPager.SAVEPOINT.RELEASE;
                    else if (ErrorAction == OE.Abort)
                        statementOp = IPager.SAVEPOINT.ROLLBACK;
                    else
                    {
                        Main.RollbackAll(ctx);
                        Main.CloseSavepoints(ctx);
                        ctx.AutoCommit = 1;
                    }
                }

                // If eStatementOp is non-zero, then a statement transaction needs to be committed or rolled back. Call sqlite3VdbeCloseStatement() to
                // do so. If this operation returns an error, and the current statement error code is SQLITE_OK or SQLITE_CONSTRAINT, then promote the
                // current statement error code.
                if (statementOp != 0)
                {
                    rc = CloseStatement(statementOp);
                    if (rc != 0)
                    {
                        if (RC_ == RC.OK || RC_ == RC.CONSTRAINT)
                        {
                            RC_ = rc;
                            C._tagfree(ctx, ref ErrMsg);
                            ErrMsg = null;
                        }
                        Main.RollbackAll(ctx);
                        Main.CloseSavepoints(ctx);
                        ctx.AutoCommit = 1;
                    }
                }

                // If this was an INSERT, UPDATE or DELETE and no statement transaction has been rolled back, update the database connection change-counter. 
                if (ChangeCntOn)
                {
                    SetChanges(ctx, (statementOp != IPager.SAVEPOINT.ROLLBACK ? Changes : 0));
                    Changes = 0;
                }

                // Release the locks
                Leave();
            }

            // We have successfully halted and closed the VM.  Record this fact.
            if (PC >= 0)
            {
                ctx.ActiveVdbeCnt--;
                if (!ReadOnly)
                    ctx.WriteVdbeCnt--;
                Debug.Assert(ctx.ActiveVdbeCnt >= ctx.WriteVdbeCnt);
            }
            Magic = VDBE_MAGIC_HALT;
            CheckActiveVdbeCnt(ctx);
            if (ctx.MallocFailed)
                RC_ = RC.NOMEM;

            // If the auto-commit flag is set to true, then any locks that were held by connection ctx have now been released. Call sqlite3ConnectionUnlocked() 
            // to invoke any required unlock-notify callbacks.
            if (ctx.AutoCommit)
                Notify.ConnectionUnlocked(ctx);

            Debug.Assert(ctx.ActiveVdbeCnt > 0 || ctx.AutoCommit == 0 || ctx.Statements == 0);
            return (RC_ == RC.BUSY ? RC.BUSY : RC.OK);
        }

        public void ResetStepResult()
        {
            RC_ = RC.OK;
        }

        public RC TransferError()
        {
            Context ctx = Ctx;
            RC rc = RC_;
            if (ErrMsg != null)
            {
                bool mallocFailed = ctx.MallocFailed;
                C._benignalloc_begin();
                ValueSetStr(ctx.Err, -1, p.ErrMsg, TEXTENCODE.UTF8, C.DESTRUCTOR_TRANSIENT);
                C._benignalloc_end();
                ctx.MallocFailed = mallocFailed;
                ctx.ErrCode = rc;
            }
            else
                sqlite3Error(ctx, rc, null);
            return rc;
        }

#if ENABLE_SQLLOG
        static void VdbeInvokeSqllog(Vdbe p)
        {
            if (_Sqllog != null && p.RC_ == RC.OK && p.Sql_ != null && p.PC >= 0)
            {
                string expanded = p.ExpandSql(p.Sql_);
                Debug.Assert(!p.Ctx.Init.Busy);
                if (expanded != null)
                {
                    _Sqllog(_SqllogArg, p.Ctx, expanded, 1);
                    C._tagfree(p.Ctx, ref expanded);
                }
            }
        }
#else
        static void VdbeInvokeSqllog(Vdbe p) { }
#endif

        public RC Reset()
        {
            Context ctx = Ctx;

            // If the VM did not run to completion or if it encountered an error, then it might not have been halted properly.  So halt it now.
            Halt();


            // If the VDBE has be run even partially, then transfer the error code and error message from the VDBE into the main database structure.  But
            // if the VDBE has just been set to run but has not actually executed any instructions yet, leave the main database error information unchanged.
            if (PC >= 0)
            {
                VdbeInvokeSqllog(this);
                TransferError();
                C._tagfree(ctx, ref ErrMsg);
                ErrMsg = null;
                if (RunOnlyOnce) Expired = true;
            }
            else if (RC_ != 0 && Expired)
            {
                // The expired flag was set on the VDBE before the first call to sqlite3_step(). For consistency (since sqlite3_step() was
                // called), set the database error in this case as well.
                sqlite3Error(ctx, RC_, 0);
                ValueSetStr(ctx.Err, -1, ErrMsg, TEXTENCODE.UTF8, C.DESTRUCTOR_TRANSIENT);
                C._tagfree(ctx, ref ErrMsg);
                ErrMsg = null;
            }

            // Reclaim all memory used by the VDBE
            Cleanup(this);

            // Save profiling information from this VDBE run.
#if VDBE_PROFILE
            {
                FILE out_ = Console.Out; // fopen("vdbe_profile.out", "a");
                if (out_ != null)
                {
                    int i;
                    out_.Write("---- ");
                    for (i = 0; i < Ops.length; i++)
                        out_.Write("%02x", Ops[i].Opcode);
                    out_.Write(out_.NewLine);
                    for (i = 0; i < Ops.length; i++)
                    {
                        out_.Write("%6d %10lld %8lld ", Ops[i].Cnt, Ops[i].Cycles, (Ops[i].Cnt > 0 ? Ops[i].Cycles / Ops[i].Cnt : 0));
                        PrintOp(out_, i, Ops[i]);
                    }
                    out_.Close();
                }
            }
#endif
            Magic = VDBE_MAGIC_INIT;
            return (RC)((int)RC_ & ctx.ErrMask);
        }

        public RC Finalize()
        {
            RC rc = RC.OK;
            if (Magic == VDBE_MAGIC_RUN || Magic == VDBE_MAGIC_HALT)
            {
                rc = Reset();
                Debug.Assert(((int)rc & Ctx.ErrMask) == (int)rc);
            }
            Delete(this);
            return rc;
        }

        public static void DeleteAuxData(VdbeFunc func, int mask)
        {
            for (int i = 0; i < func.AuxsLength; i++)
            {
                VdbeFunc.AuxData aux = func.Auxs[i];
                if ((i > 31 || (mask & (((uint)1) << i)) == 0 && aux.Aux != null))
                {
                    if (aux.Aux != null && aux.Aux is IDisposable)
                        (aux.Aux as IDisposable).Dispose();
                    if (aux.Delete != null)
                        aux.Delete(aux.Aux);
                    aux.Aux = null;
                }
            }
        }

        public void ClearObject(Context ctx)
        {
            Debug.Assert(Ctx == null || Ctx == ctx);
            ReleaseMemArray(Vars.data, Vars.length);
            ReleaseMemArray(ColNames, ResColumns, COLNAME_N);
            SubProgram sub, next;
            for (sub = Programs; sub != null; sub = next)
            {
                next = sub.Next;
                VdbeFreeOpArray(ctx, ref sub.Ops.data, sub.Ops.length);
                C._tagfree(ctx, ref sub);
            }
            for (int i = VarNames.length - 1; i >= 0; i--) C._tagfree(ctx, ref VarNames.data[i]);
            VdbeFreeOpArray(ctx, ref Ops.data, Ops.length);
            C._tagfree(ctx, ref Labels.data);
            C._tagfree(ctx, ref ColNames);
            C._tagfree(ctx, ref Sql_);
            C._tagfree(ctx, ref FreeThis);
#if ENABLE_TREE_EXPLAIN
            C._tagfree(ctx, ref _explain);
            C._tagfree(ctx, ref _explainString);
#endif
        }

        public static void Delete(Vdbe p)
        {
            if (C._NEVER(p == null)) return;
            Context ctx = p.Ctx;
            p.ClearObject(ctx);
            if (p.Prev != null)
                p.Prev.Next = p.Next;
            else
            {
                Debug.Assert(ctx.Vdbes == p);
                ctx.Vdbes = p.Next;
            }
            if (p.Next != null)
                p.Next.Prev = p.Prev;
            p.Magic = VDBE_MAGIC_DEAD;
            p.Ctx = null;
            C._tagfree(ctx, ref p);
        }

        public static RC CursorMoveto(VdbeCursor p)
        {
            if (p.DeferredMoveto)
            {
                Debug.Assert(p.IsTable);
                int res = 0;
                RC rc = Btree.MovetoUnpacked(p.Cursor, null, p.MovetoTarget, 0, out res);
                if (rc != 0) return rc;
                p.LastRowid = p.MovetoTarget;
                if (res != 0)
                    return SysEx.CORRUPT_BKPT();
                p.RowidIsValid = true;
#if  TEST
                _search_count++;
#endif
                p.DeferredMoveto = false;
                p.CacheStatus = VdbeCursor.CACHE_STALE;
            }
            else if (C._ALWAYS(p.Cursor != null))
            {
                bool hasMoved = false;
                RC rc = Btree.CursorHasMoved(p.Cursor, out hasMoved);
                if (rc != 0) return rc;
                if (hasMoved)
                {
                    p.CacheStatus = VdbeCursor.CACHE_STALE;
                    p.NullRow = true;
                }
            }
            return RC.OK;
        }

        #region Serialize & UnpackedRecord

        public static uint SerialType(Mem mem, int fileFormat)
        {
            MEM flags = mem.Flags;
            if ((flags & MEM.Null) != 0) return 0;
            if ((flags & MEM.Int) != 0)
            {
                // Figure out whether to use 1, 2, 4, 6 or 8 bytes.
                const long MAX_6BYTE = ((((long)0x00008000) << 32) - 1);
                long i = mem.u.I;
                ulong u = (i < 0 ? (i < -MAX_6BYTE ? 6 : (ulong)-i) : (ulong)i); // MAX_6BYTE test prevents: u = -(-9223372036854775808)
                if (u <= 127) return 1;
                if (u <= 32767) return 2;
                if (u <= 8388607) return 3;
                if (u <= 2147483647) return 4;
                if (u <= MAX_6BYTE) return 5;
                return 6;
            }
            if ((flags & MEM.Real) != 0) return 7;
            Debug.Assert(mem.Ctx.MallocFailed || (flags & (MEM.Str | MEM.Blob)) != 0);
            int n = mem.N;
            if ((flags & MEM.Zero) != 0)
                n += mem.u.Zeros;
            else if ((flags & MEM.Blob) != 0)
                n = (mem.Z_ != null ? mem.Z_.Length : (mem.Z != null ? mem.Z.Length : 0));
            else
            {
                if (mem.Z != null)
                    n = Encoding.UTF8.GetByteCount(mem.N < mem.Z.Length ? mem.Z.Substring(0, mem.N) : mem.Z);
                else
                    n = mem.Z_.Length;
                mem.N = n;
            }
            Debug.Assert(n >= 0);
            return (uint)((n * 2) + 12 + ((flags & MEM.Str) != 0 ? 1 : 0));
        }

        static uint[] _serialTypeSize = new uint[] { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
        public static uint SerialTypeLen(uint serialType)
        {
            if (serialType >= 12)
                return (uint)((serialType - 12) / 2);
            return _serialTypeSize[serialType];
        }

        public static uint SerialPut(byte[] buf, uint offset, int bufLength, Mem mem, int fileFormat)
        {
            uint serialType = SerialType(mem, fileFormat);
            uint len;

            // Integer and Real
            if (serialType <= 7 && serialType > 0)
            {
                ulong v;
                if (serialType == 7)
                {
                    //: Debug.Assert(sizeof(v) == sizeof(mem.R));
#if WINDOWS_PHONE || WINDOWS_MOBILE
                    v = (ulong)BitConverter.ToInt64(BitConverter.GetBytes(mem.R), 0); //: _memcpy(&v, &mem->R, sizeof(v));
#else
                    v = (ulong)BitConverter.DoubleToInt64Bits(mem.R); //: _memcpy(&v, &mem->R, sizeof(v));
#endif
                    //: SwapMixedEndianFloat(v);
                }
                else
                    v = (ulong)mem.u.I;
                uint i;
                len = i = SerialTypeLen(serialType);
                Debug.Assert(len <= (uint)bufLength);
                while (i-- != 0)
                {
                    buf[offset + i] = (byte)(v & 0xFF);
                    v >>= 8;
                }
                return len;
            }

            // String or blob
            if (serialType >= 12)
            {
                Debug.Assert(mem.N + ((mem.Flags & MEM.Zero) != 0 ? mem.u.Zeros : 0) == (int)SerialTypeLen(serialType)); // TO DO -- PASS TESTS WITH THIS ON
                Debug.Assert(mem.N <= bufLength);
                len = (uint)mem.N;
                if (len != 0)
                    if (mem.Z_ == null && mem.Z == null) { }
                    else if (mem.Z_ != null && ((mem.Flags & MEM.Blob) != 0 || mem.Z == null))
                        Buffer.BlockCopy(mem.Z_, 0, buf, offset, (int)len); //: _memcpy(buf, mem->Z, len);
                    else
                        Buffer.BlockCopy(Encoding.UTF8.GetBytes(mem.Z), 0, buf, offset, (int)len); //: _memcpy(buf, mem->Z, len);
                if ((mem.Flags & MEM.Zero) != 0)
                {
                    len += (uint)mem.u.Zeros;
                    Debug.Assert(bufLength >= 0);
                    if (len > (uint)bufLength)
                        len = (uint)bufLength;
                    Array.Clear(buf, offset + mem.N, (int)(len - mem.N)); //: _memset(&buf[mem->N], 0, len - mem->N);
                }
                return len;
            }

            // NULL or constants 0 or 1
            return 0;
        }

        public static uint SerialGet(byte[] buf, uint offset, uint serialType, Mem mem)
        {
            switch (serialType)
            {
                case 10: // Reserved for future use
                case 11: // Reserved for future use
                case 0:
                    { // NULL
                        mem.Flags = MEM.Null;
                        mem.N = 0;
                        mem.Z = null;
                        mem.Z_ = null;
                        break;
                    }
                case 1:
                    { // 1-byte signed integer
                        mem.u.I = (sbyte)buf[offset + 0];
                        mem.Flags = MEM.Int;
                        return 1;
                    }
                case 2:
                    { // 2-byte signed integer
                        mem.u.I = (int)((((sbyte)buf[offset + 0]) << 8) | buf[offset + 1]);
                        mem.Flags = MEM.Int;
                        return 2;
                    }
                case 3:
                    { // 3-byte signed integer
                        mem.u.I = (int)((((sbyte)buf[offset + 0]) << 16) | (buf[offset + 1] << 8) | buf[offset + 2]);
                        mem.Flags = MEM.Int;
                        return 3;
                    }
                case 4:
                    { // 4-byte signed integer
                        mem.u.I = (int)(((sbyte)buf[offset + 0] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3]);
                        mem.Flags = MEM.Int;
                        return 4;
                    }
                case 5:
                    { // 6-byte signed integer
                        ulong x = (ulong)((((sbyte)buf[offset + 0]) << 8) | buf[offset + 1]);
                        uint y = (uint)((buf[offset + 2] << 24) | (buf[offset + 3] << 16) | (buf[offset + 4] << 8) | buf[offset + 5]);
                        x = (x << 32) | y;
                        mem.u.I = (long)x;
                        mem.Flags = MEM.Int;
                        return 6;
                    }
                case 6:   // 8-byte signed integer
                case 7:
                    { // IEEE floating point
#if !NDEBUG && !OMIT_FLOATING_POINT
                        // Verify that integers and floating point values use the same byte order.  Or, that if SQLITE_MIXED_ENDIAN_64BIT_FLOAT is
                        // defined that 64-bit floating point values really are mixed endian.
                        const ulong t1 = ((ulong)0x3ff00000) << 32;
                        const double r1 = 1.0;
                        ulong t2 = t1;
                        //: SwapMixedEndianFloat(t2);
                        Debug.Assert(sizeof(double) == sizeof(ulong) && C._memcmp(BitConverter.GetBytes(r1), BitConverter.GetBytes(t2), sizeof(double)) == 0);
#endif
                        ulong x = (ulong)((buf[offset + 0] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3]);
                        uint y = (uint)((buf[offset + 4] << 24) | (buf[offset + 5] << 16) | (buf[offset + 6] << 8) | buf[offset + 7]);
                        x = (x << 32) | y;
                        if (serialType == 6)
                        {
                            mem.u.I = (long)x;
                            mem.Flags = MEM.Int;
                        }
                        else
                        {
                            Debug.Assert(sizeof(long) == 8 && sizeof(double) == 8);
                            //: SwapMixedEndianFloat(x);
#if WINDOWS_PHONE || WINDOWS_MOBILE
                            mem.R = BitConverter.ToDouble(BitConverter.GetBytes((long)x), 0); //: _memcpy(&mem->R, &x, sizeof(x));
#else
                            mem.R = BitConverter.Int64BitsToDouble((long)x); //: _memcpy(&mem->R, &x, sizeof(x));
#endif
                            mem.Flags = (double.IsNaN(mem.R) ? MEM.Null : MEM.Real);
                        }
                        return 8;
                    }
                case 8:    // Integer 0
                case 9:
                    {  // Integer 1
                        mem.u.I = serialType - 8;
                        mem.Flags = MEM.Int;
                        return 0;
                    }
                default:
                    {
                        uint len = (serialType - 12) / 2;
                        mem.N = (int)len;
                        mem.Del = null;
                        if ((serialType & 0x01) != 0)
                        {
                            mem.Flags = MEM.Str | MEM.Ephem;
                            if (len <= buf.Length - offset)
                            {
                                mem.Z = Encoding.UTF8.GetString(buf, offset, (int)len); //: mem->Z = (char *)buf;
                                mem.N = mem.Z.Length;
                            }
                            else
                            {
                                mem.Z = string.Empty; // Corrupted Data
                                mem.N = 0;
                            }
                            mem.Z_ = null;
                        }
                        else
                        {
                            mem.Flags = MEM.Blob | MEM.Ephem;
                            mem.Z_ = new byte[len];
                            //buf.CopyTo(mem.Z_, 0);
                            Buffer.BlockCopy(buf, offset, mem.Z_, 0, (len <= buf.Length - offset ? (int)len : buf.Length - offset - 1));
                            mem.Z = null;
                        }
                        return len;
                    }
            }
            return 0;
        }

        public static UnpackedRecord AllocUnpackedRecord(KeyInfo keyInfo)
        {
            var p = new UnpackedRecord();
            p.Mems = new Mem[p.Fields + 1];
            p.KeyInfo = keyInfo;
            p.Fields = (ushort)(keyInfo.Fields + 1);
            return p;
        }

        public static void RecordUnpack(KeyInfo keyInfo, int keyLength, byte[] key, UnpackedRecord p)
        {
            byte[] keys = key;
            Mem mem;
            p.Flags = 0;
            //: Debug.Assert(C._HASALIGNMENT8(mem));
            int szHdr;
            uint idx = (uint)ConvertEx.GetVarint32(keys, 0, out szHdr);
            int d = szHdr;
            ushort u = 0; // Unsigned loop counter
            while (idx < szHdr && u < p.Fields && d <= keyLength)
            {
                p.Mems[u] = mem = C._alloc(p.Mems[u]);
                uint serialType;
                idx += (uint)ConvertEx.GetVarint32(keys, idx, out serialType);
                mem.Encode = keyInfo.Encode;
                mem.Ctx = (Context)keyInfo.Ctx;
                //mem->Flags = 0; // sqlite3VdbeSerialGet() will set this for us
                //: mem.Malloc = null;
                d += (int)SerialGet(keys, d, serialType, mem);
                u++;
            }
            Debug.Assert(u <= keyInfo.Fields + 1);
            p.Fields = (ushort)u;
        }

        static Mem _mem1 = new Mem();
        // ALTERNATE FORM for C#
        //public static int RecordCompare(int key1Length, byte[] key1, UnpackedRecord key2) { return sqlite3VdbeRecordCompare(key1Length, key1, 0, key2); }
        public static int RecordCompare(int key1Length, byte[] key1, uint offset, UnpackedRecord key2)
        {
            int i = 0;
            int rc = 0;
            byte[] key1s = new byte[key1.Length - offset];

            KeyInfo keyInfo = key2.pKeyInfo;
            _mem1.Encode = keyInfo.Encode;
            _mem1.Ctx = (Context)keyInfo.Ctx;
            // _mem1.flags = 0; // Will be initialized by sqlite3VdbeSerialGet()
            // ASSERTONLY(mem1.Malloc = 0;) // Only needed by Debug.Assert() statements

            // Compilers may complain that mem1.u.i is potentially uninitialized. We could initialize it, as shown here, to silence those complaints.
            // But in fact, mem1.u.i will never actually be used uninitialized, and doing the unnecessary initialization has a measurable negative performance
            // impact, since this routine is a very high runner.  And so, we choose to ignore the compiler warnings and leave this variable uninitialized.
            //  _mem1.u.i = 0;  // not needed, here to silence compiler warning
            uint szHdr1; // Number of bytes in header
            uint idx1 = (uint)(ConvertEx.GetVarint32(key1, offset, out szHdr1)); // Offset into aKey[] of next header element
            int d1 = (int)szHdr1; // Offset into aKey[] of next data element
            int fields = keyInfo.Fields;
            Debug.Assert(keyInfo.SortOrders != null);
            while (idx1 < szHdr1 && i < key2.Fields)
            {
                uint serialType1;

                // Read the serial types for the next element in each key.
                idx1 += (uint)(ConvertEx.GetVarint32(key1s, (uint)(offset + idx1), out serialType1));
                if (d1 <= 0 || d1 >= key1Length && SerialTypeLen(serialType1) > 0) break;

                // Extract the values to be compared.
                d1 += (int)SerialGet(key1s, offset + d1, serialType1, _mem1);

                // Do the comparison
                rc = MemCompare(_mem1, key2.Mems[i], (i < fields ? keyInfo.Colls[i] : null));
                if (rc != 0)
                {
                    //: Debug.Assert(_mem1.Malloc == null); // See comment below

                    // Invert the result if we are using DESC sort order.
                    if (i < fields && keyInfo.SortOrders != null && keyInfo.SortOrders[i] != 0)
                        rc = -rc;

                    // If the PREFIX_SEARCH flag is set and all fields except the final rowid field were equal, then clear the PREFIX_SEARCH flag and set 
                    // pPKey2->rowid to the value of the rowid field in (pKey1, nKey1). This is used by the OP_IsUnique opcode.
                    if ((key2.Flags & UNPACKED_PREFIX_SEARCH) != 0 && i == (key2.Fields - 1))
                    {
                        Debug.Assert(idx1 == szHdr1 && rc != 0);
                        Debug.Assert((_mem1.flags & MEM_Int) != 0);
                        key2.Flags &= ~UNPACKED_PREFIX_SEARCH;
                        key2.Rowid = _mem1.u.I;
                    }
                    return rc;
                }
                i++;
            }

            // No memory allocation is ever used on mem1.  Prove this using the following assert().  If the assert() fails, it indicates a
            // memory leak and a need to call sqlite3VdbeMemRelease(&mem1).
            //Debug.Assert(mem1.Malloc == null);

            // rc==0 here means that one of the keys ran out of fields and all the fields up to that point were equal. If the UNPACKED_INCRKEY
            // flag is set, then break the tie by treating key2 as larger. If the UPACKED_PREFIX_MATCH flag is set, then keys with common prefixes
            // are considered to be equal.  Otherwise, the longer key is the larger.  As it happens, the pPKey2 will always be the longer
            // if there is a difference.
            Debug.Assert(rc == 0);
            if ((key2.Flags & UNPACKED_INCRKEY) != 0) rc = -1;
            else if ((key2.flags & UNPACKED_PREFIX_MATCH) != 0) { } // Leave rc==0 
            else if (idx1 < szHdr1) rc = 1;
            return rc;
        }

        #endregion

        #region Index Entry

        public static int IdxRowid(Context ctx, BtCursor cur, ref int64 rowid)
        {
            // Get the size of the index entry.  Only indices entries of less than 2GiB are support - anything large must be database corruption.
            // Any corruption is detected in sqlite3BtreeParseCellPtr(), though, so this code can safely assume that nCellKey is 32-bits  
            Debug.Assert(Btree.CursorIsValid(cur));
            long cellKeyLength = 0;
            RC rc = Btree.KeySize(cur, ref cellKeyLength);
            Debug.Assert(rc == RC.OK); // pCur is always valid so KeySize cannot fail
            Debug.Assert(((uint)cellKeyLength & MAX_U32) == (ulong)cellKeyLength);

            // Read in the complete content of the index entry
            Mem m = C._alloc(m);
            //: memset(&m, 0, sizeof(m));
            rc = MemFromBtree(cur, 0, (int)cellKeyLength, true, m);
            if (rc != 0)
                return rc;

            // The index entry must begin with a header size
            uint szHdr = 0; // Size of the header
            ConvertEx.GetVarint32(m.Z_, 0, out szHdr);
            C.ASSERTCOVERAGE(szHdr == 3);
            C.ASSERTCOVERAGE(szHdr == m.N);
            if (unlikely(szHdr < 3 || (int)szHdr > m.n))
                goto idx_rowid_corruption;

            // The last field of the index should be an integer - the ROWID. Verify that the last entry really is an integer.
            uint typeRowid = 0; // Serial type of the rowid
            ConvertEx.GetVarint32(m.Z_, szHdr - 1, out typeRowid);
            C.ASSERTCOVERAGE(typeRowid == 1);
            C.ASSERTCOVERAGE(typeRowid == 2);
            C.ASSERTCOVERAGE(typeRowid == 3);
            C.ASSERTCOVERAGE(typeRowid == 4);
            C.ASSERTCOVERAGE(typeRowid == 5);
            C.ASSERTCOVERAGE(typeRowid == 6);
            C.ASSERTCOVERAGE(typeRowid == 8);
            C.ASSERTCOVERAGE(typeRowid == 9);
            if (unlikely(typeRowid < 1 || typeRowid > 9 || typeRowid == 7))
                goto idx_rowid_corruption;
            uint lenRowid = (uint)SerialTypeLen(typeRowid); // Size of the rowid
            C.ASSERTCOVERAGE((uint)m.N == szHdr + lenRowid);
            if (unlikely((uint)m.N < szHdr + lenRowid))
                goto idx_rowid_corruption;

            // Fetch the integer off the end of the index record
            Mem v = C._alloc(v);
            SerialGet(m.Z_, (int)(m.N - lenRowid), typeRowid, v);
            rowid = v.u.I;
            MemRelease(m);
            return RC.OK;

        // Jump here if database corruption is detected after m has been allocated.  Free the m object and return SQLITE_CORRUPT.
        idx_rowid_corruption:
            //: ASSERTCOVERAGE(m.Malloc != nullptr);
            MemRelease(m);
            return SysEx.CORRUPT_BKPT();
        }

        public static int IdxKeyCompare(VdbeCursor c, UnpackedRecord unpacked, ref int r)
        {
            Btree.BtCursor cur = c.Cursor;
            Mem m = null;
            Debug.Assert(Btree.CursorIsValid(cur));
            long cellKeyLength = 0;
            RC rc = Btree.KeySize(cur, ref cellKeyLength);
            Debug.Assert(rc == RC.OK); // pCur is always valid so KeySize cannot fail
            // nCellKey will always be between 0 and 0xffffffff because of the say that btreeParseCellPtr() and sqlite3GetVarint32() are implemented
            if (cellKeyLength <= 0 || cellKeyLength > 0x7fffffff)
            {
                r = 0;
                return SysEx.CORRUPT_BKPT();
            }
            m = C._alloc(m); //: _memset(&m, 0, sizeof(m));
            rc = MemFromBtree(cur, 0, (int)cellKeyLength, true, m);
            if (rc != 0)
                return rc;
            Debug.Assert((unpacked.Flags & UNPACKED_IGNORE_ROWID) != 0);
            r = RecordCompare(m.N, m.Z_, unpacked);
            MemRelease(m);
            return RC.OK;
        }

        #endregion

        public static void SetChanges(Context ctx, int changes)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            ctx.Changes = changes;
            ctx.TotalChanges += changes;
        }

        public void CountChanges()
        {
            ChangeCntOn = true;
        }

        public static void ExpirePreparedStatements(Context ctx)
        {
            for (Vdbe p = ctx.Vdbes; p != null; p = p.Next)
                p.Expired = true;
        }

        public Context get_Ctx()
        {
            return Ctx;
        }

        public Mem GetValue(int var, AFF aff)
        {
            Debug.Assert(var > 0);
            Mem mem = Vars[var - 1];
            if ((mem.Flags & MEM.Null) == 0)
            {
                Mem r = ValueNew(Ctx);
                if (r != null)
                {
                    MemCopy(r, mem);
                    ValueApplyAffinity(r, aff, TEXTENCODE_UTF8);
                    MemStoreType(r);
                }
                return r;
            }
            return null;
        }

        public void SetVarmask(int var)
        {
            Debug.Assert(var > 0);
            if (var > 32)
                Expmask = 0xffffffff;
            else
                Expmask |= ((uint)1 << (var - 1));
        }
    }
}
