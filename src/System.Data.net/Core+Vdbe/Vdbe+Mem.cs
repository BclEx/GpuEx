using Core.IO;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace Core
{
    public partial class Vdbe
    {
        public static RC ChangeEncoding(Mem mem, TEXTENCODE newEncode)
        {
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            Debug.Assert(newEncode == TEXTENCODE.UTF8 || newEncode == TEXTENCODE.UTF16LE || newEncode == TEXTENCODE.UTF16BE);
            if ((mem.Flags & MEM.Str) == 0 || mem.Encode == newEncode)
            {
                if (mem.Z == null && mem.Z_ != null)
                    mem.Z = Encoding.UTF8.GetString(mem.Z_, 0, mem.Z_.Length);
                return RC.OK;
            }
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
#if OMIT_UTF16
            return RC.ERROR;
#else
            // MemTranslate() may return SQLITE_OK or SQLITE_NOMEM. If NOMEM is returned, then the encoding of the value may not have changed.
            RC rc = MemTranslate(mem, newEncode);
            Debug.Assert(rc == RC.OK || rc == RC.NOMEM);
            Debug.Assert(rc == RC.OK || mem.Encode != newEncode);
            Debug.Assert(rc == RC.NOMEM || mem.Encode == newEncode);
            return rc;
#endif
        }

        public static RC MemGrow(Mem mem, int newSize, bool preserve)
        {
            //Debug.Assert((mem.Malloc != null && mem.Malloc == mem.Z ? 1 : 0) +
            //    ((mem.Flags & MEM.Dyn) && mem.Del ? 1 : 0) +
            //    ((mem.Flags & MEM.Ephem) ? 1 : 0) +
            //    ((mem.Flags & MEM.Static) ? 1 : 0) <= 1);
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);

            // If the preserve flag is set to true, then the memory cell must already contain a valid string or blob value.
            Debug.Assert(!preserve || (mem.Flags & (MEM.Blob | MEM.Str)) != 0);

            if (newSize < 32) newSize = 32;
            //: if (_tagallocsize(mem->Ctx, mem->Malloc) < newSize)
            if (preserve) //: && mem->Z == mem->Malloc)
            {
                if (mem.Z == null)
                    mem.Z = null;
                else
                    if (newSize < mem.Z.Length)
                        mem.Z = mem.Z.Substring(0, newSize);
                preserve = false;
            }
            else
            {
                //: _tagfree(mem->Ctx, mem->Malloc);
                mem.Z = null; //: mem->Malloc = (char*)_tagalloc(mem->Ctx, newSize);
            }

            //: if (mem.Z && preserve && mem.Malloc && mem.Z != mem->Malloc)
            //:     _memcpy(mem.Malloc, mem.Z, mem.N);
            if ((mem.Flags & MEM.Dyn) != 0 && mem.Del != null)
            {
                Debug.Assert(mem.Del != C.DESTRUCTOR_DYNAMIC);
                mem.Del(mem.Z);
            }

            //: mem.Z = mem->Malloc;
            mem.Flags = (MEM)(mem.Z == null ? MEM.Null : mem.Flags & ~(MEM.Ephem | MEM.Static));
            mem.Del = null;
            return (mem.Z != null ? RC.OK : RC.NOMEM);
        }

        public static RC MemMakeWriteable(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            E.ExpandBlob(mem);
            MEM f = mem.Flags;
            if ((f & (MEM.Str | MEM.Blob)) != 0) //: mem->Z != mem->Malloc)
            {
                if (MemGrow(mem, mem.N + 2, true) != 0)
                    return RC.NOMEM;
                //: mem.Z[mem.N] = 0;
                //: mem.Z[mem.N + 1] = 0;
                mem.Flags |= MEM.Term;
#if DEBUG
                mem.ScopyFrom = null;
#endif
            }
            return RC.OK;
        }

#if !OMIT_INCRBLOB
        public static RC MemExpandBlob(Mem mem)
        {
            if ((mem.Flags & MEM.Zero) != 0)
            {
                Debug.Assert((mem.Flags & MEM.Blob) != 0);
                Debug.Assert((mem.Flags & MEM.RowSet) == 0);
                Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
                // Set nByte to the number of bytes required to store the expanded blob.
                int bytes = mem.N + mem.u.Zeros;
                if (bytes <= 0)
                    bytes = 1;
                if (MemGrow(mem, bytes, true) != 0)
                    return RC.NOMEM;
                //: _memset(&mem->Z[mem->N], 0, mem->u.Zeros);
                mem.Z_ = Encoding.UTF8.GetBytes(mem.Z);
                mem.Z = null;
                mem.N += (int)mem.u.Zeros;
                mem.u.I = 0;
                mem.Flags &= ~(MEM.Zero | MEM.Static | MEM.Ephem | MEM.Term);
                mem.Flags |= MEM.Dyn;
            }
            return RC.OK;
        }
#endif

        public static RC MemNulTerminate(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            if ((mem.Flags & MEM.Term) != 0 || (mem.Flags & MEM.Str) == 0)
                return RC.OK; // Nothing to do
            if (MemGrow(mem, mem.N + 2, true) != 0)
                return RC.NOMEM;
            //mem.Z[mem.N] = 0;
            //mem.Z[mem.N + 1] = 0;
            if (mem.Z != null && mem.N < mem.Z.Length)
                mem.Z = mem.Z.Substring(0, mem.N);
            mem.Flags |= MEM.Term;
            return RC.OK;
        }

        public static RC MemStringify(Mem mem, TEXTENCODE encode)
        {
            MEM f = mem.Flags;
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((f & MEM.Zero) == 0);
            Debug.Assert((f & (MEM.Str | MEM.Blob)) == 0);
            Debug.Assert((f & (MEM.Int | MEM.Real)) != 0);
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            //: Debug.Assert(C._HASALIGNMENT8(mem));

            const int bytes = 32;
            if (MemGrow(mem, bytes, false) != 0)
                return RC.NOMEM;

            // For a Real or Integer, use sqlite3_mprintf() to produce the UTF-8 string representation of the value. Then, if the required encoding
            // is UTF-16le or UTF-16be do a translation.
            // FIX ME: It would be better if sqlite3_snprintf() could do UTF-16.
            if ((f & MEM.Int) != 0)
                mem.Z = mem.u.I.ToString(); //: __snprintf(mem->Z, bytes, "%lld", mem->u.I);
            else
            {
                Debug.Assert((f & MEM.Real) != 0);
                if (double.IsNegativeInfinity(mem.R)) mem.Z = "-Inf";
                else if (double.IsInfinity(mem.R)) mem.Z = "Inf";
                else if (double.IsPositiveInfinity(mem.R)) mem.Z = "+Inf";
                else if (mem.R.ToString(CultureInfo.InvariantCulture).Contains(".")) mem.Z = mem.R.ToString(CultureInfo.InvariantCulture).ToLower(); //: __snprintf(mem->Z, bytes, "%!.15g", mem->R);
                else mem.Z = mem.R.ToString(CultureInfo.InvariantCulture) + ".0";
            }
            mem.N = mem.Z.Length;
            mem.Encode = TEXTENCODE.UTF8;
            mem.Flags |= MEM.Str | MEM.Term;
            ChangeEncoding(mem, encode);
            return RC.OK;
        }

        public static RC MemFinalize(Mem mem, FuncDef func)
        {
            RC rc = RC.OK;
            if (C._ALWAYS(func != null && func.Finalize != null))
            {
                Debug.Assert((mem.Flags & MEM.Null) != 0 || func == mem.u.Def);
                Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
                //memset(&ctx, 0, sizeof(ctx));
                FuncContext ctx = new FuncContext();
                ctx.S.Flags = MEM.Null;
                ctx.S.Ctx = mem.Ctx;
                ctx.Mem = mem;
                ctx.Func = func;
                func.Finalize(ctx); // IMP: R-24505-23230
                Debug.Assert((mem.Flags & MEM.Dyn) == 0 && mem.Del == null);
                C._tagfree(mem.Ctx, ref mem.Z_); //: mem->Malloc);
                ctx.S._memcpy(ref mem);
                rc = ctx.IsError;
            }
            return rc;
        }

        public static void MemReleaseExternal(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            if ((mem.Flags & MEM.Agg) != 0)
            {
                MemFinalize(mem, mem.u.Def);
                Debug.Assert((mem.Flags & MEM.Agg) == 0);
                MemRelease(mem);
            }
            else if ((mem.Flags & MEM.Dyn) != 0 && mem.Del != null)
            {
                Debug.Assert((mem.Flags & MEM.RowSet) == 0);
                mem.Del(mem.Z);
                mem.Del = null;
            }
            else if ((mem.Flags & MEM.RowSet) != 0) RowSet_Clear(mem.u.RowSet);
            else if ((mem.Flags & MEM.Frame) != 0) MemSetNull(mem);
            mem.N = 0;
            mem.Z = null;
            mem.Z_ = null;
        }

        public static void MemRelease(Mem mem)
        {
            E.VdbeMemRelease(mem);
            C._tagfree(mem.Ctx, ref mem.Z_); //: mem->Malloc);
            mem.Z = null;
            mem.Del = null;
        }

        static long DoubleToInt64(double r)
        {
#if OMIT_FLOATING_POINT
            return r; // When floating-point is omitted, double and int64 are the same thing
#else
            if (r < (double)long.MinValue) return long.MinValue;
            // minInt is correct here - not maxInt.  It turns out that assigning a very large positive number to an integer results in a very large
            // negative integer.  This makes no sense, but it is what x86 hardware does so for compatibility we will do the same in software.
            else if (r > (double)long.MaxValue) return long.MinValue;
            else return (long)r;
#endif
        }

        public static long IntValue(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            // assert( EIGHT_BYTE_ALIGNMENT(pMem) );
            MEM flags = mem.Flags;
            if ((flags & MEM.Int) != 0) return mem.u.I;
            else if ((flags & MEM.Real) != 0) return DoubleToInt64(mem.R);
            else if ((flags & (MEM.Str)) != 0)
            {
                Debug.Assert(mem.Z != null || mem.N == 0);
                C.ASSERTCOVERAGE(mem.Z == null);
                long value;
                ConvertEx.Atoi64(mem.Z, out value, mem.N, mem.Encode);
                return value;
            }
            else if ((flags & (MEM.Blob)) != 0)
            {
                Debug.Assert(mem.Z_ != null || mem.N == 0);
                C.ASSERTCOVERAGE(mem.Z_ == null);
                long value;
                ConvertEx.Atoi64(Encoding.UTF8.GetString(mem.Z_, 0, mem.N), out value, mem.N, mem.Encode);
                return value;
            }
            return 0;
        }

        public static double RealValue(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            //Debug.Assert(C._HASALIGNMENT8(mem));
            if ((mem.Flags & MEM.Real) != 0) return mem.R;
            else if ((mem.Flags & MEM.Int) != 0) return (double)mem.u.I;
            else if ((mem.Flags & (MEM.Str)) != 0)
            {
                double val = (double)0;
                ConvertEx.Atof(mem.Z, ref val, mem.N, mem.Encode);
                return val;
            }
            else if ((mem.Flags & (MEM.Blob)) != 0)
            {
                double val = (double)0;
                Debug.Assert(mem.Z_ != null || mem.N == 0);
                ConvertEx.Atof(Encoding.UTF8.GetString(mem.Z_, 0, mem.N), ref val, mem.N, mem.Encode);
                return val;
            }
            return (double)0;
        }

        public static void IntegerAffinity(Mem mem)
        {
            Debug.Assert((mem.Flags & MEM.Real) != 0);
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            //Debug.Assert(C._HASALIGNMENT8(mem));
            mem.u.I = DoubleToInt64(mem.R);
            // Only mark the value as an integer if
            //    (1) the round-trip conversion real->int->real is a no-op, and
            //    (2) The integer is neither the largest nor the smallest possible integer (ticket #3922)
            // The second and third terms in the following conditional enforces the second condition under the assumption that addition overflow causes
            // values to wrap around.  On x86 hardware, the third term is always true and could be omitted.  But we leave it in because other
            // architectures might behave differently.
            if (mem.R == (double)mem.u.I && mem.u.I > long.MinValue && C._ALWAYS(mem.u.I < long.MaxValue))
                mem.Flags |= MEM.Int;
        }

        public static RC MemIntegerify(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            //Debug.Assert(C._HASALIGNMENT8(mem));
            mem.u.I = IntValue(mem);
            E.MemSetTypeFlag(mem, MEM.Int);
            return RC.OK;
        }

        public static RC MemRealify(Mem mem)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            //Debug.Assert(C._HASALIGNMENT8(mem));
            mem.R = RealValue(mem);
            E.MemSetTypeFlag(mem, MEM.Real);
            return RC.OK;
        }

        public static RC MemNumerify(Mem mem)
        {
            if ((mem.Flags & (MEM.Int | MEM.Real | MEM.Null)) == 0)
            {
                Debug.Assert((mem.Flags & (MEM.Blob | MEM.Str)) != 0);
                Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
                if ((mem.Flags & MEM.Blob) != 0 && mem.Z == null)
                {
                    if (ConvertEx.Atoi64(Encoding.UTF8.GetString(mem.Z_, 0, mem.Z_.Length), out mem.u.I, mem.N, mem.Encode) == 0)
                        E.MemSetTypeFlag(mem, MEM.Int);
                    else
                    {
                        mem.R = RealValue(mem);
                        E.MemSetTypeFlag(mem, MEM.Real);
                        IntegerAffinity(mem);
                    }
                }
                else if (ConvertEx.Atoi64(mem.Z, out mem.u.I, mem.N, mem.Encode) == 0)
                    E.MemSetTypeFlag(mem, MEM.Int);
                else
                {
                    mem.R = RealValue(mem);
                    E.MemSetTypeFlag(mem, MEM.Real);
                    IntegerAffinity(mem);
                }
            }
            Debug.Assert((mem.Flags & (MEM.Int | MEM.Real | MEM.Null)) != 0);
            mem.Flags &= ~(MEM.Str | MEM.Blob);
            return RC.OK;
        }

#if !OMIT_FLOATING_POINT
        public static void MemSetNull(Mem mem)
        {
            if ((mem.Flags & MEM.Frame) != 0)
            {
                VdbeFrame frame = mem.u.Frame;
                frame.Parent = frame.V.DelFrames;
                frame.V.DelFrames = frame;
            }
            if ((mem.Flags & MEM.RowSet) != 0)
                RowSet_Clear(mem.u.RowSet);
            E.MemSetTypeFlag(mem, MEM.Null);
            mem.Type = TYPE.NULL;
            C._free(ref mem.Z_);
            mem.Z = null;
        }
#endif

        public static void MemSetZeroBlob(Mem mem, int n)
        {
            MemRelease(mem);
            mem.Flags = MEM.Blob | MEM.Zero;
            mem.Type = TYPE.BLOB;
            mem.N = 0;
            if (n < 0) n = 0;
            mem.u.Zeros = n;
            mem.Encode = TEXTENCODE.UTF8;
#if OMIT_INCRBLOB
            MemGrow(mem, n, 0);
            mem.N = n;
            mem.Z = null; //:_memset(mem->Z, 0, n);
            mem.Z_ = C._alloc(n);
#endif
        }

        public static void MemSetInt64(Mem mem, long val)
        {
            MemRelease(mem);
            mem.u.I = val;
            mem.Flags = MEM.Int;
            mem.Type = TYPE.INTEGER;
        }

#if !OMIT_FLOATING_POINT
        public static void MemSetDouble(Mem mem, double val)
        {
            if (double.IsNaN(val))
                MemSetNull(mem);
            else
            {
                MemRelease(mem);
                mem.R = val;
                mem.Flags = MEM.Real;
                mem.Type = TYPE.FLOAT;
            }
        }
#endif

        public static void MemSetRowSet(Mem mem)
        {
            Context ctx = mem.Ctx;
            Debug.Assert(ctx != null);
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            MemRelease(mem);
            //: mem.Malloc = C._alloc(ctx, 64);
            if (ctx.MallocFailed)
                mem.Flags = MEM.Null;
            else
            {
                //: Debug.Assert(mem.Malloc);
                mem.u.RowSet = new RowSet(ctx, 5); //: RowSet_Init(ctx, mem->Malloc, _tagallocsize(ctx, mem->Malloc));
                Debug.Assert(mem.u.RowSet != null);
                mem.Flags = MEM.RowSet;
            }
        }

        public static bool MemTooBig(Mem mem)
        {
            //Debug.Assert(p.Ctx != null);
            if ((mem.Flags & (MEM.Str | MEM.Blob)) != 0)
            {
                int n = mem.N;
                if ((mem.Flags & MEM.Zero) != 0)
                    n += mem.u.Zeros;
                return n > mem.Ctx.Limits[(int)LIMIT.LENGTH];
            }
            return false;
        }

#if DEBUG
        static void MemAboutToChange(Vdbe p, Mem mem)
        {
            Mem x;
            int i;
            for (i = 1; i <= p.Mems.length; i++)
            {
                x = p.Mems[i];
                if (x.ScopyFrom == mem)
                {
                    x.Flags |= MEM.Invalid;
                    x.ScopyFrom = null;
                }
            }
            mem.ScopyFrom = null;
        }
#endif

        //: #define MEMCELLSIZE (size_t)(&(((Mem *)0)->Malloc)) // Size of struct Mem not including the Mem.zMalloc member.
        public static void MemShallowCopy(Mem to, Mem from, MEM srcType)
        {
            Debug.Assert((from.Flags & MEM.RowSet) == 0);
            MemReleaseExternal(to);
            from._memcpy(ref to);
            to.Del = null;
            if ((from.Flags & MEM.Static) != 0)
            {
                to.Flags &= ~(MEM.Dyn | MEM.Static | MEM.Ephem);
                Debug.Assert(srcType == MEM.Ephem || srcType == MEM.Static);
                to.Flags |= srcType;
            }
        }

        public static RC MemCopy(Mem to, Mem from)
        {
            Debug.Assert((from.Flags & MEM.RowSet) == 0);
            E.VdbeMemRelease(to);
            from._memcpy(ref to);
            to.Flags &= ~MEM.Dyn;
            RC rc = RC.OK;
            if ((to.Flags & (MEM.Str | MEM.Blob)) != 0)
            {
                if ((from.Flags & MEM.Static) == 0)
                {
                    to.Flags |= MEM.Ephem;
                    rc = MemMakeWriteable(to);
                }
            }
            return rc;
        }

        public static void MemMove(Mem to, Mem from)
        {
            Debug.Assert(from.Ctx == null || MutexEx.Held(from.Ctx.Mutex));
            Debug.Assert(to.Ctx == null || MutexEx.Held(to.Ctx.Mutex));
            Debug.Assert(from.Ctx == null || to.Ctx == null || from.Ctx == to.Ctx);
            MemRelease(to);
            from._memcpy(ref to);
            from.Flags = MEM.Null;
            from.Del = null;
            from.Z = null;
            from.Z_ = null;
        }

        public static RC MemSetStr(Mem mem, byte[] z, int n, TEXTENCODE encode, Action<object> del) { return MemSetStr(mem, z, 0, (n >= 0 ? n : z.Length), encode, del); }
        public static RC MemSetStr(Mem mem, byte[] z, int offset, int n, TEXTENCODE encode, Action<object> del)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            // If z is a NULL pointer, set pMem to contain an SQL NULL.
            if (z == null || z.Length < offset)
            {
                MemSetNull(mem);
                return RC.OK;
            }

            int limit = (mem.Ctx != null ? mem.Ctx.Limits[(int)LIMIT.LENGTH] : CORE_MAX_LENGTH); // Maximum allowed string or blob size
            MEM flags = (encode == 0 ? MEM.Blob : MEM.Str); // New value for pMem->flags
            int bytes = n; // New value for pMem->n
            if (bytes < 0)
            {
                Debug.Assert(encode != 0);
                if (encode == TEXTENCODE.UTF8)
                    for (bytes = 0; bytes <= limit && bytes < z.Length - offset && z[offset + bytes] != 0; bytes++) { }
                else
                    for (bytes = 0; bytes <= limit && z[bytes + offset] != 0 || z[offset + bytes + 1] != 0; bytes += 2) { }
            }

            // The following block sets the new values of Mem.z and Mem.xDel. It also sets a flag in local variable "flags" to indicate the memory
            // management (one of MEM_Dyn or MEM_Static).
            Debug.Assert(encode == 0);
            {
                mem.Z = null;
                mem.Z_ = C._alloc(n);
                Buffer.BlockCopy(z, offset, mem.Z_, 0, n);
            }
            mem.N = bytes;
            mem.Flags = MEM.Blob | MEM.Term;
            mem.Encode = (encode == 0 ? TEXTENCODE.UTF8 : encode);
            mem.Type = (encode == 0 ? TYPE.BLOB : TYPE.TEXT);

#if !OMIT_UTF16
            if (mem.Encode != TEXTENCODE.UTF8 && MemHandleBom(mem) != 0) return RC.NOMEM;
#endif
            return (bytes > limit ? RC.TOOBIG : RC.OK);
        }

        static RC MemSetStr(Mem mem, string z, int n, TEXTENCODE encode, Action<object> del) { return MemSetStr(mem, z, 0, n, encode, del); }
        public static RC MemSetStr(Mem mem, string z, int offset, int n, TEXTENCODE encode, Action<object> del)
        {
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            // If z is a NULL pointer, set pMem to contain an SQL NULL.
            if (z == null || z.Length < offset)
            {
                MemSetNull(mem);
                return RC.OK;
            }

            int limit = (mem.Ctx != null ? mem.Ctx.Limits[(int)LIMIT.LENGTH] : CORE_MAX_LENGTH); // Maximum allowed string or blob size
            MEM flags = (encode == 0 ? MEM.Blob : MEM.Str); // New value for pMem->flags
            int bytes = n; // New value for pMem->n
            if (bytes < 0)
            {
                Debug.Assert(encode != 0);
                if (encode == TEXTENCODE.UTF8)
                    for (bytes = 0; bytes <= limit && bytes < z.Length - offset && z[offset + bytes] != 0; bytes++) { }
                else
                    for (bytes = 0; bytes <= limit && z[bytes + offset] != 0 || z[offset + bytes + 1] != 0; bytes += 2) { }
                flags |= MEM.Term;
            }

            // The following block sets the new values of Mem.z and Mem.xDel. It also sets a flag in local variable "flags" to indicate the memory
            // management (one of MEM_Dyn or MEM_Static).
            if (del == C.DESTRUCTOR_TRANSIENT)
            {
                int allocs = bytes;
                if ((flags & MEM.Term) != 0) allocs += (encode == TEXTENCODE.UTF8 ? 1 : 2);
                if (bytes > limit) return RC.TOOBIG;
                if (MemGrow(mem, (int)allocs, false) != 0) return RC.NOMEM;
                //if (allocs < z.Length) mem.Z = new byte[allocs]; Buffer.BlockCopy(z, 0, mem.Z, 0, allocs); }
                //else
                if (encode == 0)
                {
                    mem.Z = null;
                    mem.Z_ = C._alloc(n);
                    for (int i = 0; i < n && i < z.Length - offset; i++)
                        mem.Z_[i] = (byte)z[offset + i];
                }
                else
                {
                    mem.Z = (n > 0 && z.Length - offset > n ? z.Substring(offset, n) : z.Substring(offset));
                    C._free(ref mem.Z_);
                }
            }
            else if (del == C.DESTRUCTOR_DYNAMIC)
            {
                MemRelease(mem);
                //: mem->Malloc = mem->Z = (char *)z;
                if (encode == 0)
                {
                    mem.Z = null;
                    if (mem.Z_ != null)
                        C._free(ref mem.Z_);
                    mem.Z_ = Encoding.UTF8.GetBytes(offset == 0 ? z : z.Length + offset < n ? z.Substring(offset, n) : z.Substring(offset));
                }
                else
                {
                    mem.Z = (n > 0 && z.Length - offset > n ? z.Substring(offset, n) : z.Substring(offset));
                    mem.Z_ = null;
                }
                mem.Del = null;
            }
            else
            {
                MemRelease(mem);
                if (encode == 0) //: mem->Z = (char *)z;
                {
                    mem.Z = null;
                    if (mem.Z_ != null)
                        C._free(ref mem.Z_);
                    mem.Z_ = Encoding.UTF8.GetBytes(offset == 0 ? z : z.Length + offset < n ? z.Substring(offset, n) : z.Substring(offset));
                }
                else
                {
                    mem.Z = (n > 0 && z.Length - offset > n ? z.Substring(offset, n) : z.Substring(offset));
                    C._free(ref mem.Z_);
                }
                mem.Del = del;
                flags |= (del == C.DESTRUCTOR_STATIC ? MEM.Static : MEM.Dyn);
            }
            mem.N = bytes;
            mem.Flags = MEM.Blob | MEM.Term;
            mem.Encode = (encode == 0 ? TEXTENCODE.UTF8 : encode);
            mem.Type = (encode == 0 ? TYPE.BLOB : TYPE.TEXT);

#if !OMIT_UTF16
            if (mem.Encode != TEXTENCODE.UTF8 && MemHandleBom(mem) != 0) return RC.NOMEM;
#endif
            return (bytes > limit ? RC.TOOBIG : RC.OK);
        }

        public static int MemCompare(Mem mem1, Mem mem2, CollSeq coll)
        {
            MEM f1 = mem1.Flags;
            MEM f2 = mem2.Flags;
            MEM cf = f1 | f2;
            Debug.Assert((cf & MEM.RowSet) == 0);

            // If one value is NULL, it is less than the other. If both values are NULL, return 0.
            if ((cf & MEM.Null) != 0)
                return (f2 & MEM.Null) - (f1 & MEM.Null);

            // If one value is a number and the other is not, the number is less. If both are numbers, compare as reals if one is a real, or as integers if both values are integers.
            if ((cf & (MEM.Int | MEM.Real)) != 0)
            {
                if ((f1 & (MEM.Int | MEM.Real)) == 0) return 1;
                if ((f2 & (MEM.Int | MEM.Real)) == 0) return -1;
                if ((f1 & f2 & MEM.Int) == 0)
                {
                    double r1 = ((f1 & MEM.Real) == 0 ? (double)mem1.u.I : mem1.R);
                    double r2 = ((f2 & MEM.Real) == 0 ? (double)mem2.u.I : mem2.R);
                    if (r1 < r2) return -1;
                    if (r1 > r2) return 1;
                    return 0;
                }
                Debug.Assert((f1 & MEM.Int) != 0);
                Debug.Assert((f2 & MEM.Int) != 0);
                if (mem1.u.I < mem2.u.I) return -1;
                if (mem1.u.I > mem2.u.I) return 1;
                return 0;
            }

            // If one value is a string and the other is a blob, the string is less. If both are strings, compare using the collating functions.
            int r;
            if ((cf & MEM.Str) != 0)
            {
                if ((f1 & MEM.Str) == 0) return 1;
                if ((f2 & MEM.Str) == 0) return -1;

                Debug.Assert(mem1.Encode == mem2.Encode);
                Debug.Assert(mem1.Encode == TEXTENCODE.UTF8 || mem1.Encode == TEXTENCODE.UTF16LE || mem1.Encode == TEXTENCODE.UTF16BE);
                // The collation sequence must be defined at this point, even if the user deletes the collation sequence after the vdbe program is
                // compiled (this was not always the case).
                Debug.Assert(coll == null || coll.Cmp != null);
                if (coll != null)
                {
                    if (mem1.Encode == coll.Encode)
                        return coll.Cmp(coll.User, mem1.N, mem1.Z, mem2.N, mem2.Z); // The strings are already in the correct encoding.  Call the comparison function directly
                    else
                    {
                        Mem c1 = C._alloc(c1); //: _memset(&c1, 0, sizeof(c1));
                        Mem c2 = C._alloc(c2); //: _memset(&c2, 0, sizeof(c2));
                        MemShallowCopy(c1, mem1, MEM.Ephem);
                        MemShallowCopy(c2, mem2, MEM.Ephem);
                        string v1 = ValueText(c1, coll.Encode);
                        int n1 = (v1 == null ? 0 : c1.N);
                        string v2 = ValueText(c2, coll.Encode);
                        int n2 = (v2 == null ? 0 : c2.N);
                        r = coll.Cmp(coll.User, n1, v1, n2, v2);
                        MemRelease(c1);
                        MemRelease(c2);
                        return r;
                    }
                }
                // If a NULL pointer was passed as the collate function, fall through to the blob case and use memcmp().
            }
            // Both values must be blobs.  Compare using memcmp().
            if ((mem1.Flags & MEM.Blob) != 0)
                if (mem1.Z_ != null) r = C._memcmp(mem1.Z_, mem2.Z_, (mem1.N > mem2.N ? mem2.N : mem1.N));
                else r = C._memcmp(mem1.Z, mem2.Z_, (mem1.N > mem2.N ? mem2.N : mem1.N));
            else r = C._memcmp(mem1.Z, mem2.Z, (mem1.N > mem2.N ? mem2.N : mem1.N));
            if (r == 0)
                r = mem1.N - mem2.N;
            return r;
        }

        public static RC MemFromBtree(Btree.BtCursor cur, int offset, int amount, bool key, Mem mem)
        {
            RC rc = RC.OK;

            Debug.Assert(Btree.CursorIsValid(cur));

            // Note: the calls to BtreeKeyFetch() and DataFetch() below assert() that both the BtShared and database handle mutexes are held.
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            int available = 0; // Number of bytes available on the local btree page
            uint dummy1;
            byte[] data = (byte[])(key ? Btree.KeyFetch(cur, ref available, out dummy1) : Btree.DataFetch(cur, ref available, out dummy1)); // Data from the btree layer
            Debug.Assert(data != null);

            if (offset + amount <= available && (mem.Flags & MEM.Dyn) == 0)
            {
                MemRelease(mem);
                mem.Z_ = C._alloc(amount);
                Buffer.BlockCopy(data, offset, mem.Z_, 0, amount); //: mem->Z = &data[offset];
                mem.Flags = MEM.Blob | MEM.Ephem;
            }
            else if ((rc = MemGrow(mem, amount + 2, false)) == RC.OK)
            {
                mem.Flags = MEM.Blob | MEM.Dyn | MEM.Term;
                mem.Encode = 0;
                mem.Type = TYPE.BLOB;
                mem.Z = null;
                mem.Z_ = C._alloc(amount);
                rc = (key ? Btree.Key(cur, (uint)offset, (uint)amount, mem.Z_) : Btree.Data(cur, (uint)offset, (uint)amount, mem.Z_));
                //: mem->Z[amount] = 0;
                //: mem->Z[amount + 1] = 0;
                if (rc != RC.OK)
                    MemRelease(mem);
            }
            mem.N = amount;
            return rc;
        }

        #region Value

        public static string ValueText(Mem mem, TEXTENCODE encode)
        {
            if (mem == null) return null;
            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((encode & (TEXTENCODE)3) == (encode & ~TEXTENCODE.UTF16_ALIGNED));
            Debug.Assert((mem.Flags & MEM.RowSet) == 0);
            if ((mem.Flags & MEM.Null) != 0) return null;
            Debug.Assert(((int)MEM.Blob >> 3) == (int)MEM.Str);
            mem.Flags |= (MEM)((int)(mem.Flags & MEM.Blob) >> 3);
            E.ExpandBlob(mem);
            if ((mem.Flags & MEM.Str) != 0)
            {
                ChangeEncoding(mem, encode & ~TEXTENCODE.UTF16_ALIGNED);
                if ((encode & TEXTENCODE.UTF16_ALIGNED) != 0 && (1 & (mem.Z[0])) == 1)
                {
                    Debug.Assert((mem.Flags & (MEM.Ephem | MEM.Static)) != 0);
                    if (MemMakeWriteable(mem) != RC.OK) return null;
                }
                MemNulTerminate(mem); // IMP: R-31275-44060
            }
            else
            {
                Debug.Assert((mem.Flags & MEM.Blob) == 0);
                MemStringify(mem, encode);
                Debug.Assert((1 & (mem.Z[0])) == 0);
            }
            Debug.Assert(mem.Encode == (encode & ~TEXTENCODE.UTF16_ALIGNED) || mem.Ctx == null || mem.Ctx.MallocFailed);
            return (mem.Encode == (encode & ~TEXTENCODE.UTF16_ALIGNED) ? mem.Z : null);
        }

        public static Mem ValueNew(Context ctx)
        {
            Mem p = null;
            p = C._tagalloc(ctx, p);
            if (p != null)
            {
                p.Flags = MEM.Null;
                p.Type = TYPE.NULL;
                p.Ctx = ctx;
            }
            return p;
        }

        public static RC ValueFromExpr(Context ctx, Expr expr, TEXTENCODE encode, AFF affinity, ref Mem value)
        {
            if (expr == null)
            {
                value = null;
                return RC.OK;
            }
            TK op = expr.OP;

            // op can only be TK_REGISTER if we have compiled with SQLITE_ENABLE_STAT3. The ifdef here is to enable us to achieve 100% branch test coverage even when SQLITE_ENABLE_STAT3 is omitted.
#if ENABLE_STAT3
            if (op == TK.REGISTER) op = expr.OP2;
#else
            if (C._NEVER(op == TK.REGISTER)) op = expr.OP2;
#endif

            // Handle negative integers in a single step.  This is needed in the case when the value is -9223372036854775808.
            int negInt = 1;
            string neg = string.Empty;
            if (op == TK.UMINUS && (expr.Left.OP == TK.INTEGER || expr.Left.OP == TK.FLOAT))
            {
                expr = expr.Left;
                op = expr.OP;
                negInt = -1;
                neg = "-";
            }

            Mem mem = null;
            string memAsString = null;
            if (op == TK.STRING || op == TK.FLOAT || op == TK.INTEGER)
            {
                mem = ValueNew(ctx);
                if (mem == null) goto no_mem;
                if (E.ExprHasProperty(expr, EP.IntValue))
                    MemSetInt64(mem, (long)expr.u.I * negInt);
                else
                {
                    memAsString = C._mtagprintf(ctx, "%s%s", neg, expr.u.Token);
                    if (memAsString == null) goto no_mem;
                    ValueSetStr(mem, -1, memAsString, TEXTENCODE.UTF8, C.DESTRUCTOR_DYNAMIC);
                    if (op == TK.FLOAT) mem.Type = TYPE.FLOAT;
                }
                if ((op == TK.INTEGER || op == TK.FLOAT) && affinity == AFF.NONE)
                    ValueApplyAffinity(mem, AFF.NUMERIC, TEXTENCODE.UTF8);
                else
                    ValueApplyAffinity(mem, affinity, TEXTENCODE.UTF8);
                if ((mem.Flags & (MEM.Int | MEM.Real)) != 0) mem.Flags &= ~MEM.Str;
                if (encode != TEXTENCODE.UTF8)
                    ChangeEncoding(mem, encode);
            }
            else if (op == TK.UMINUS)
            {
                // This branch happens for multiple negative signs.  Ex: -(-5)
                if (ValueFromExpr(ctx, expr.Left, encode, affinity, ref mem) == RC.OK)
                {
                    MemNumerify(mem);
                    if (mem.u.I == long.MinValue)
                    {
                        mem.Flags &= MEM.Int;
                        mem.Flags |= MEM.Real;
                        mem.R = (double)int.MaxValue;
                    }
                    else
                        mem.u.I = -mem.u.I;
                    mem.R = -mem.R;
                    ValueApplyAffinity(mem, affinity, encode);
                }
            }
            else if (op == TK.NULL)
            {
                mem = ValueNew(ctx);
                if (mem == null) goto no_mem;
            }
#if !OMIT_BLOB_LITERAL
            else if (op == TK.BLOB)
            {
                Debug.Assert(expr.u.Token[0] == 'x' || expr.u.Token[0] == 'X');
                Debug.Assert(expr.u.Token[1] == '\'');
                mem = ValueNew(ctx);
                if (mem == null) goto no_mem;
                memAsString = expr.u.Token.Substring(2);
                int memAsStringLength = memAsString.Length - 1;
                Debug.Assert(memAsString[memAsStringLength] == '\'');
                byte[] blob = C._taghextoblob(ctx, memAsString, memAsStringLength);
                MemSetStr(mem, Encoding.UTF8.GetString(blob, 0, blob.Length), memAsStringLength / 2, 0, C.DESTRUCTOR_DYNAMIC);
            }
#endif
            if (mem != null)
                MemStoreType(mem);
            value = mem;
            return RC.OK;

        no_mem:
            ctx.MallocFailed = true;
            C._tagfree(ctx, ref memAsString);
            ValueFree(ref mem);
            value = null;
            return RC.NOMEM;
        }

        public static void ValueSetStr(Mem mem, int n, string z, TEXTENCODE encode, Action<object> del)
        {
            if (mem != null) MemSetStr(mem, z, n, encode, del);
        }

        public static void ValueFree(ref Mem mem)
        {
            if (mem == null) return;
            MemRelease(mem);
            C._tagfree(mem.Ctx, ref mem);
        }

        static int ValueBytes(Mem mem, TEXTENCODE encode)
        {
            if ((mem.Flags & MEM.Blob) != 0 || ValueText(mem, encode) != null)
                return ((mem.Flags & MEM.Zero) != 0 ? mem.N + mem.u.Zeros : (mem.Z == null ? mem.Z_.Length : mem.N));
            return 0;
        }

        #endregion
    }
}