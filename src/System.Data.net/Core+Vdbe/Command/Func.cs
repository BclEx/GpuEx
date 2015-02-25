// func.c
using System;
using System.Diagnostics;
using System.Text;

namespace Core.Command
{
    public static class Func
    {
        public static CollSeq GetFuncCollSeq(FuncContext fctx)
        {
            return fctx.Coll;
        }

        public static void SkipAccumulatorLoad(FuncContext fctx)
        {
            fctx.SkipFlag = true;
        }

        static void MinMaxFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc > 1);
            int mask = (Vdbe.User_Data(fctx) == null ? 0 : -1); // 0 for min() or 0xffffffff for max()
            CollSeq coll = GetFuncCollSeq(fctx);
            Debug.Assert(coll != null);
            Debug.Assert(mask == -1 || mask == 0);
            C.ASSERTCOVERAGE(mask == 0);
            int best = 0;
            if (Vdbe.Value_Type(argv[0]) == TYPE.NULL) return;
            for (int i = 1; i < argc; i++)
            {
                if (Vdbe.Value_Type(argv[i]) == TYPE.NULL) return;
                if ((sqlite3MemCompare(argv[best], argv[i], coll) ^ mask) >= 0)
                {
                    C.ASSERTCOVERAGE(mask == 0);
                    best = i;
                }
            }
            Vdbe.Result_Value(fctx, argv[best]);
        }

        static void TypeofFunc(FuncContext fctx, int notUsed1, Mem[] argv)
        {
            string z;
            switch (Vdbe.Value_Type(argv[0]))
            {
                case TYPE.INTEGER: z = "integer"; break;
                case TYPE.TEXT: z = "text"; break;
                case TYPE.FLOAT: z = "real"; break;
                case TYPE.BLOB: z = "blob"; break;
                default: z = "null"; break;
            }
            Vdbe.Result_Text(fctx, z, -1, C.DESTRUCTOR_STATIC);
        }


        static void LengthFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            int len;
            Debug.Assert(argc == 1);
            switch (Vdbe.Value_Type(argv[0]))
            {
                case TYPE.BLOB:
                case TYPE.INTEGER:
                case TYPE.FLOAT:
                    {
                        Vdbe.Result_Int(fctx, Vdbe.Value_Bytes(argv[0]));
                        break;
                    }
                case TYPE.TEXT:
                    {
                        byte[] z = Vdbe.Value_Blob(argv[0]);
                        if (z == null) return;
                        len = 0;
                        int zIdx = 0;
                        while (zIdx < z.Length && z[zIdx] != '\0')
                        {
                            len++;
                            C._strskiputf8(z, ref zIdx);
                        }
                        Vdbe.Result_Int(fctx, len);
                        break;
                    }
                default:
                    {
                        Vdbe.Result_Null(fctx);
                        break;
                    }
            }
        }

        static void AbsFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            switch (Vdbe.Value_Type(argv[0]))
            {
                case TYPE.INTEGER:
                    {
                        long ival = Vdbe.Value_Int64(argv[0]);
                        if (ival < 0)
                        {
                            if ((ival << 1) == 0)
                            {
                                // IMP: R-35460-15084 If X is the integer -9223372036854775807 then abs(X) throws an integer overflow error since there is no
                                // equivalent positive 64-bit two complement value.
                                Vdbe.Result_Error(fctx, "integer overflow", -1);
                                return;
                            }
                            ival = -ival;
                        }
                        Vdbe.Result_Int64(fctx, ival);
                        break;
                    }
                case TYPE.NULL:
                    {
                        // IMP: R-37434-19929 Abs(X) returns NULL if X is NULL.
                        Vdbe.Result_Null(fctx);
                        break;
                    }
                default:
                    {
                        // Because sqlite3_value_double() returns 0.0 if the argument is not something that can be converted into a number, we have:
                        // IMP: R-57326-31541 Abs(X) return 0.0 if X is a string or blob that cannot be converted to a numeric value. 
                        double rval = Vdbe.Value_Double(argv[0]);
                        if (rval < 0) rval = -rval;
                        Vdbe.Result_Double(fctx, rval);
                        break;
                    }
            }
        }

        static void InstrFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            TYPE typeHaystack = Vdbe.Value_Type(argv[0]);
            TYPE typeNeedle = Vdbe.Value_Type(argv[1]);
            if (typeHaystack == TYPE.NULL || typeNeedle == TYPE.NULL) return;
            int haystackLength = Vdbe.Value_Bytes(argv[0]);
            int needleLength = Vdbe.Value_Bytes(argv[1]);
            byte[] haystack;
            byte[] needle;
            bool isText;
            if (typeHaystack == TYPE.BLOB && typeNeedle == TYPE.BLOB)
            {
                haystack = Vdbe.Value_Blob(argv[0]);
                needle = Vdbe.Value_Blob(argv[1]);
                isText = false;
            }
            else
            {
                haystack = Vdbe.Value_Text(argv[0]);
                needle = Vdbe.Value_Text(argv[1]);
                isText = true;
            }
            int n = 1;
            while (needleLength <= haystackLength && C._memcmp(haystack, needle, needleLength) != 0)
            {
                n++;
                do
                {
                    haystackLength--;
                    haystack++;
                } while (isText && (haystack[0] & 0xc0) == 0x80);
            }
            if (needleLength > haystackLength) n = 0;
            Vdbe.Result_Int(fctx, n);
        }

        static void SubstrFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 3 || argc == 2);
            if (Vdbe.Value_Type(argv[1]) == TYPE.NULL || (argc == 3 && Vdbe.Value_Type(argv[2]) == TYPE.NULL))
                return;
            TYPE p0type = Vdbe.Value_Type(argv[0]);
            int p1 = Vdbe.Value_Int(argv[1]);
            int len;
            string z = string.Empty;
            string z2;
            byte[] zAsBlob_ = null;
            if (p0type == TYPE.BLOB)
            {
                len = Vdbe.Value_Bytes(argv[0]);
                zAsBlob_ = argv[0].ZBLOB;
                if (zAsBlob_ == null) return;
                //Debug.Assert(len == zAsBlob_.Length);
            }
            else
            {
                z = Vdbe.Value_Text(argv[0]);
                if (z == null) return;
                len = 0;
                if (p1 < 0)
                    for (z2 = z; z2 != string.Empty; len++)
                        C._strskiputf8(ref z2);
            }
            long p2;
            bool negP2 = false;
            if (argc == 3)
            {
                p2 = Vdbe.Value_Int(argv[2]);
                if (p2 < 0)
                {
                    p2 = -p2;
                    negP2 = true;
                }
            }
            else
                p2 = (Vdbe.Context_Ctx(fctx)).Limits[(int)LIMIT.LENGTH];
            if (p1 < 0)
            {
                p1 += len;
                if (p1 < 0)
                {
                    p2 += p1;
                    if (p2 < 0) p2 = 0;
                    p1 = 0;
                }
            }
            else if (p1 > 0)
                p1--;
            else if (p2 > 0)
                p2--;
            if (negP2)
            {
                p1 -= (int)p2;
                if (p1 < 0)
                {
                    p2 += p1;
                    p1 = 0;
                }
            }
            Debug.Assert(p1 >= 0 && p2 >= 0);
            if (p0type != TYPE.BLOB)
            {
                while (z != null && p1 != 0)
                {
                    C._strskiputf8(ref z);
                    p1--;
                }
                for (z2 = z; z2 != null && p2 != 0; p2--)
                    C._strskiputf8(ref z2);
                Vdbe.Result_Text(fctx, z, p1, p2 <= z.Length - p1 ? p2 : z.Length - p1, C.DESTRUCTOR_TRANSIENT);
            }
            else
            {
                if (p1 + p2 > len)
                {
                    p2 = len - p1;
                    if (p2 < 0) p2 = 0;
                }
                StringBuilder sb = new StringBuilder(zAsBlob_.Length);
                if (zAsBlob_.Length == 0 || p1 > zAsBlob_.Length)
                    sb.Length = 0;
                else
                    for (int i = p1; i < p1 + p2; i++)
                        sb.Append((char)zAsBlob_[i]);
                Vdbe.Result_Blob(fctx, sb.ToString(), (int)p2, C.DESTRUCTOR_TRANSIENT);
            }
        }

#if !OMIT_FLOATING_POINT
        static void RoundFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1 || argc == 2);
            int n = 0;
            if (argc == 2)
            {
                if (Vdbe.Value_Type(argv[1]) == TYPE.NULL) return;
                n = Vdbe.Value_Int(argv[1]);
                if (n > 30) n = 30;
                if (n < 0) n = 0;
            }
            if (Vdbe.Value_Type(argv[0]) == TYPE.NULL)
                return;
            double r = Vdbe.Value_Double(argv[0]);
            // If Y==0 and X will fit in a 64-bit int, handle the rounding directly, otherwise use printf.
            if (n == 0 && r >= 0 && r < long.MaxValue - 1)
                r = (double)((long)(r + 0.5));
            else if (n == 0 && r < 0 && (-r) < long.MaxValue - 1)
                r = -(double)((long)((-r) + 0.5));
            else
            {
                string buf = C._mprintf("%.*f", n, r);
                if (buf == null)
                {
                    Vdbe.Result_ErrorNoMem(fctx);
                    return;
                }
                ConvertEx.Atof(buf, ref r, buf.Length, TEXTENCODE.UTF8);
                C._free(ref buf);
            }
            Vdbe.Result_Double(fctx, r);
        }
#endif

        static T[] ContextMalloc<T>(FuncContext fctx, long bytes)
            where T : struct
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            Debug.Assert(bytes > 0);
            C.ASSERTCOVERAGE(bytes == ctx.Limits[(int)LIMIT.LENGTH]);
            C.ASSERTCOVERAGE(bytes == ctx.Limits[(int)LIMIT.LENGTH] + 1);
            T[] z;
            if (bytes > ctx.Limits[(int)LIMIT.LENGTH])
            {
                Vdbe.Result_ErrorOverflow(fctx);
                z = null;
            }
            else
            {
                z = new T[(int)bytes];
                if (z == null)
                    Vdbe.Result_ErrorNoMem(fctx);
            }
            return z;
        }

        static void UpperFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z2 = Vdbe.Value_Text(argv[0]);
            int n = Vdbe.Value_Bytes(argv[0]);
            // Verify that the call to _bytes() does not invalidate the _text() pointer
            Debug.Assert(z2 == Vdbe.Value_Text(argv[0]));
            if (z2 != null)
            {
                string z1 = (z2.Length == 0 ? string.Empty : z2.Substring(0, n).ToUpperInvariant()); //: Many
                Vdbe.Result_Text(fctx, z1, -1, null); //: _free
            }
        }

        static void LowerFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z2 = Vdbe.Value_Text(argv[0]);
            int n = Vdbe.Value_Bytes(argv[0]);
            // Verify that the call to _bytes() does not invalidate the _text() pointer
            Debug.Assert(z2 == Vdbe.Value_Text(argv[0]));
            if (z2 != null)
            {
                string z1 = (z2.Length == 0 ? string.Empty : z2.Substring(0, n).ToLowerInvariant());
                Vdbe.Result_Text(fctx, z1, -1, null); //: _free
            }
        }

        //: #define ifnullFunc versionFunc // Substitute function - never called

        static void RandomFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            long r = 0;
            SysEx.PutRandom(sizeof(long), ref r);
            if (r < 0)
            {
                // We need to prevent a random number of 0x8000000000000000 (or -9223372036854775808) since when you do abs() of that
                // number of you get the same value back again.  To do this in a way that is testable, mask the sign bit off of negative
                // values, resulting in a positive value.  Then take the 2s complement of that positive value.  The end result can
                // therefore be no less than -9223372036854775807.
                r = -(r ^ (((long)1) << 63));
            }
            Vdbe.Result_Int64(fctx, r);
        }

        static void RandomBlob(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            int n = Vdbe.Value_Int(argv[0]);
            if (n < 1)
                n = 1;
            char[] p = ContextMalloc<char>(fctx, n);
            if (p != null)
            {
                long p_ = 0;
                for (int i = 0; i < n; i++)
                {
                    SysEx.PutRandom(sizeof(char), ref p_);
                    p[i] = (char)(p_ & 0x7F);
                }
                Vdbe.Result_Blob(fctx, new string(p), n, null); //: _free
            }
        }

        static void LastInsertRowid(FuncContext fctx, int notUsed1, Mem[] notUsed2
        )
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            // IMP: R-51513-12026 The last_insert_rowid() SQL function is a wrapper around the sqlite3_last_insert_rowid() C/C++ interface function.
            Vdbe.Result_Int64(fctx, Vdbe.Last_InsertRowid(ctx));
        }

        static void Changes(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            Vdbe.Result_Int(fctx, sqlite3_changes(ctx));
        }

        static void TotalChanges(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            // IMP: R-52756-41993 This function is a wrapper around the sqlite3_total_changes() C/C++ interface.
            Vdbe.Result_Int(fctx, sqlite3_total_changes(ctx));
        }

        struct CompareInfo
        {
            public char MatchAll;
            public char MatchOne;
            public char MatchSet;
            public bool NoCase;
        };

        static CompareInfo _globInfo = new CompareInfo { MatchAll = '*', MatchOne = '?', MatchSet = '[', NoCase = false };
        static CompareInfo _likeInfoNorm = new CompareInfo { MatchAll = '%', MatchOne = '_', MatchSet = '\0', NoCase = true }; // The correct SQL-92 behavior is for the LIKE operator to ignore case.  Thus  'a' LIKE 'A' would be true.
        static CompareInfo _likeInfoAlt = new CompareInfo { MatchAll = '%', MatchOne = '_', MatchSet = '\0', NoCase = false }; // If SQLITE_CASE_SENSITIVE_LIKE is defined, then the LIKE operator is case sensitive causing 'a' LIKE 'A' to be false

        static bool PatternCompare(string pattern, string string_, CompareInfo info, uint escape)
        {
            uint c, c2;
            int invert;
            int seen;
            int matchOne = (int)info.MatchOne;
            int matchAll = (int)info.MatchAll;
            int matchSet = (int)info.MatchSet;
            bool noCase = info.NoCase;
            bool prevEscape = false; // True if the previous character was 'escape'

            string inPattern = pattern; // Entered Pattern

            while ((c = SysEx.Utf8Read(pattern, ref pattern)) != 0)
            {
                if (c == matchAll && !prevEscape)
                {
                    while ((c = SysEx.Utf8Read(pattern, ref pattern)) == matchAll || c == matchOne)
                        if (c == matchOne && SysEx.Utf8Read(string_, ref string_) == 0)
                            return false;
                    if (c == 0)
                        return true;
                    else if (c == escape)
                    {
                        c = SysEx.Utf8Read(pattern, ref pattern);
                        if (c == 0)
                            return false;
                    }
                    else if (c == matchSet)
                    {
                        Debug.Assert(escape == 0); // This is GLOB, not LIKE
                        Debug.Assert(matchSet < 0x80); // '[' is a single-byte character
                        int stringIdx = 0;
                        while (stringIdx < string_.Length && !PatternCompare(inPattern.Substring(inPattern.Length - pattern.Length - 1), string_.Substring(stringIdx), info, escape))
                            C._strskiputf8(string_, ref stringIdx);
                        return (stringIdx < string_.Length);
                    }
                    while ((c2 = SysEx.Utf8Read(string_, ref string_)) != 0)
                    {
                        if (noCase)
                        {
                            c2 = (uint)char.ToLowerInvariant((char)c2);
                            c = (uint)char.ToLowerInvariant((char)c);
                            while (c2 != 0 && c2 != c)
                            {
                                c2 = SysEx.Utf8Read(string_, ref string_);
                                c2 = (uint)char.ToLowerInvariant((char)c2);
                            }
                        }
                        else
                        {
                            while (c2 != 0 && c2 != c)
                                c2 = SysEx.Utf8Read(string_, ref string_);
                        }
                        if (c2 == 0) return false;
                        if (PatternCompare(pattern, string_, info, escape)) return true;
                    }
                    return false;
                }
                else if (c == matchOne && !prevEscape)
                {
                    if (SysEx.Utf8Read(string_, ref string_) == 0)
                        return false;
                }
                else if (c == matchSet)
                {
                    uint prior_c = 0;
                    Debug.Assert(escape == 0); // This only occurs for GLOB, not LIKE
                    seen = 0;
                    invert = 0;
                    c = SysEx.Utf8Read(string_, ref string_);
                    if (c == 0) return false;
                    c2 = SysEx.Utf8Read(pattern, ref pattern);
                    if (c2 == '^')
                    {
                        invert = 1;
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    if (c2 == ']')
                    {
                        if (c == ']') seen = 1;
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    while (c2 != 0 && c2 != ']')
                    {
                        if (c2 == '-' && pattern[0] != ']' && pattern[0] != 0 && prior_c > 0)
                        {
                            c2 = SysEx.Utf8Read(pattern, ref pattern);
                            if (c >= prior_c && c <= c2) seen = 1;
                            prior_c = 0;
                        }
                        else
                        {
                            if (c == c2)
                                seen = 1;
                            prior_c = c2;
                        }
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    if (c2 == 0 || (seen ^ invert) == 0)
                        return false;
                }
                else if (escape == c && !prevEscape)
                    prevEscape = true;
                else
                {
                    c2 = SysEx.Utf8Read(string_, ref string_);
                    if (noCase)
                    {
                        c = (uint)char.ToLowerInvariant((char)c);
                        c2 = (uint)char.ToLowerInvariant((char)c2);
                    }
                    if (c != c2)
                        return false;
                    prevEscape = false;
                }
            }
            return (string_.Length == 0);
        }


#if TEST
        static int _likeCount = 0;
#endif

        static void LikeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            uint escape = 0;
            Context ctx = Vdbe.Context_Ctx(fctx);

            string zB = Vdbe.Value_Text(argv[0]);
            string zA = Vdbe.Value_Text(argv[1]);

            // Limit the length of the LIKE or GLOB pattern to avoid problems of deep recursion and N*N behavior in patternCompare().
            int pats = Vdbe.Value_Bytes(argv[0]);
            C.ASSERTCOVERAGE(pats == ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH]);
            C.ASSERTCOVERAGE(pats == ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH] + 1);
            if (pats > ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH])
            {
                Vdbe.Rsult_Error(fctx, "LIKE or GLOB pattern too complex", -1);
                return;
            }
            Debug.Assert(zB == Vdbe.Value_Text(argv[0])); // Encoding did not change

            if (argc == 3)
            {
                // The escape character string must consist of a single UTF-8 character. Otherwise, return an error.
                string zEscape = Vdbe.Value_Text(argv[2]);
                if (zEscape == null) return;
                if (SysEx.Utf8CharLen(zEscape, -1) != 1)
                {
                    Vdbe.Result_Error(fctx, "ESCAPE expression must be a single character", -1);
                    return;
                }
                escape = SysEx.Utf8Read(zEscape, ref zEscape);
            }
            if (zA != null && zB != null)
            {
                CompareInfo info = (CompareInfo)Vdbe.User_Data(fctx);
#if TEST
                _likeCount++;
#endif
                Vdbe.Result_Int(fctx, PatternCompare(zB, zA, info, escape) ? 1 : 0);
            }
        }

        static void NullifFunc(FuncContext fctx, int ntUsed1, Mem[] argv)
        {
            CollSeq coll = GetFuncCollSeq(fctx);
            if (sqlite3MemCompare(argv[0], argv[1], coll) != 0)
                Vdbe.Result_Value(fctx, argv[0]);
        }

        static void VersionFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            // IMP: R-48699-48617 This function is an SQL wrapper around the sqlite3_libversion() C-interface.
            Vdbe.Result_Text(fctx, sqlite3_libversion(), -1, C.DESTRUCTOR_STATIC);
        }

        static void SourceidFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            // IMP: R-24470-31136 This function is an SQL wrapper around the sqlite3_sourceid() C interface.
            Vdbe.Result_Text(fctx, sqlite3_sourceid(), -1, C.DESTRUCTOR_STATIC);
        }

        static void ErrlogFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            SysEx.LOG(Vdbe.Value_Int(argv[0]), "%s", Vdbe.Value_Text(argv[1]));
        }

#if !OMIT_COMPILEOPTION_DIAGS
        static void CompileoptionusedFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            // IMP: R-39564-36305 The sqlite_compileoption_used() SQL function is a wrapper around the sqlite3_compileoption_used() C/C++ function.
            string optName;
            if ((optName = Vdbe.Value_Text(argv[0])) != null)
                Vdbe.Rsult_Int(fctx, CompileTimeOptionUsed(optName));
        }
        static void CompileoptiongetFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            // IMP: R-04922-24076 The sqlite_compileoption_get() SQL function is a wrapper around the sqlite3_compileoption_get() C/C++ function.
            int n = Vdbe.Value_Int(argv[0]);
            Vdbe.Result_Text(fctx, CompileTimeGet(n), -1, C.DESTRUCTOR_STATIC);
        }
#endif

        static char[] _hexdigits = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

        // [EXPERIMENTAL]
        static void QuoteFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            switch (Vdbe.Value_Type(argv[0]))
            {
                case TYPE.FLOAT:
                    {
                        StringBuilder b = new StringBuilder(50);
                        b.AppendFormat("%!.15g", r1);
                        double r1 = Vdbe.Value_Double(argv[0]);
                        double r2;
                        ConvertEx.Atof(b.ToString(), ref r2, 20, TEXTENCODE.UTF8);
                        b.Length = 0;
                        if (r1 != r2)
                            b.AppendFormat("%!.20e", r1);
                        Vdbe.Result_Text(fctx, b.ToString(), -1, C.DESTRUCTOR_TRANSIENT);
                        break;
                    }
                case TYPE.INTEGER:
                    {
                        Vdbe.Result_Value(fctx, argv[0]);
                        break;
                    }
                case TYPE.BLOB:
                    {
                        byte[] blob = Vdbe.Value_Blob(argv[0]);
                        int blobLength = Vdbe.Value_Bytes(argv[0]);
                        Debug.Assert(blob.Length == Vdbe.Value_Blob(argv[0]).Length); // No encoding change
                        StringBuilder z = new StringBuilder(2 * blobLength + 4); //: ContextMalloc(fctx, (2*(int64)blobLength)+4);
                        if (z != null)
                        {
                            for (int i = 0; i < blobLength; i++)
                            {
                                z.Append(_hexdigits[(blob[i] >> 4) & 0x0F]);
                                z.Append(_hexdigits[(blob[i]) & 0x0F]);
                            }
                            z.Append[(blobLength * 2) + 2] = '\'';
                            z.Append[(blobLength * 2) + 3] = '\0';
                            z.Append[0] = 'X';
                            z.Append[1] = '\'';
                            Vdbe.Result_Text(fctx, z, -1, C.DESTRUCTOR_TRANSIENT);
                            C._free(ref z);
                        }
                        break;
                    }
                case TYPE.TEXT:
                    {

                        string zarg = Vdbe.Value_Text(argv[0]);
                        if (zarg == null || zarg.Length == 0) return;
                        int i, j;
                        ulong n;
                        for (i = 0, n = 0; i < zarg.Length; i++) { if (zarg[i] == '\'') n++; }
                        StringBuilder z = new StringBuilder(i + (int)n + 3);
                        if (z != null)
                        {
                            z.Append('\'');
                            for (i = 0, j = 1; i < zarg.Length && zarg[i] != 0; i++)
                            {
                                z.Append((char)zarg[i]);
                                j++;
                                if (zarg[i] == '\'')
                                {
                                    z.Append('\'');
                                    j++;
                                }
                            }
                            z.Append('\'');
                            j++;
                            //: z[j] = 0;
                            Vdbe.Result_Text(fctx, z, j, null); //: C._free
                        }
                        break;
                    }
                default:
                    {
                        Debug.Assert(Vdbe.Value_Type(argv[0]) == TYPE.NULL);
                        Vdbe.Result_Text(fctx, "NULL", 4, C.DESTRUCTOR_STATIC);
                        break;
                    }
            }
        }

        static void UnicodeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z = Vdbe.Value_Text(argv[0]);
            if (z != null && z[0] != 0) Vdbe.Result_Int(fctx, SysEx.Utf8Read(z, ref z));
        }

        //static void CharFunc(FuncContext fctx, int argc, Mem[] argv)
        //{
        //    string z, z2;
        //    z = z2 = sqlite3_malloc(argc * 4);
        //    if (z == null)
        //    {
        //        sqlite3_result_error_nomem(fctx);
        //        return;
        //    }
        //    for (int i = 0; i < argc; i++)
        //    {
        //        long x = sqlite3_value_int64(argv[i]);
        //        if (x < 0 || x > 0x10ffff) x = 0xfffd;
        //        ulong c = (ulong)(x & 0x1fffff);
        //        if (c < 0x00080)
        //            *z2++ = (uint8)(c & 0xFF);
        //        else if (c < 0x00800)
        //        {
        //            *z2++ = 0xC0 + (uint8)((c >> 6) & 0x1F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //        else if (c < 0x10000)
        //        {
        //            *z2++ = 0xE0 + (uint8)((c >> 12) & 0x0F);
        //            *z2++ = 0x80 + (uint8)((c >> 6) & 0x3F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //        else
        //        {
        //            *z2++ = 0xF0 + (uint8)((c >> 18) & 0x07);
        //            *z2++ = 0x80 + (uint8)((c >> 12) & 0x3F);
        //            *z2++ = 0x80 + (uint8)((c >> 6) & 0x3F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //    }
        //    sqlite3_result_text(fctx, (char*)z, (int)(z2 - z), null); // _free
        //}

        static void HexFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            byte[] blob = Vdbe.Value_Blob(argv[0]);
            int n = Vdbe.Value_Bytes(argv[0]);
            Debug.Assert(n == (blob == null ? 0 : blob.Length)); // No encoding change
            StringBuilder zHex = new StringBuilder(n * 2 + 1); //: ContextMalloc(fctx, ((int64)n)*2 + 1);
            if (zHex != null)
            {
                for (int i = 0; i < n; i++)
                {
                    byte c = blob[i];
                    zHex.Append(_hexdigits[(c >> 4) & 0xf]);
                    zHex.Append(_hexdigits[c & 0xf]);
                }
                //: *z = 0;
                Vdbe.Result_Text(fctx, zHex, n * 2, null); //: C._free
            }
        }

        static void ZeroblobFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            Debug.Assert(argc == 1);
            long n = Vdbe.Value_Int64(argv[0]);
            C.ASSERTCOVERAGE(n == ctx.Limits[(int)LIMIT.LENGTH]);
            C.ASSERTCOVERAGE(n == ctx.Limits[(int)LIMIT.LENGTH] + 1);
            if (n > ctx.Limits[(int)LIMIT.LENGTH])
                Vdbe.Result_ErrorOverflow(fctx);
            else
                Vdbe.Result_ZeroBlob(fctx, (int)n); // IMP: R-00293-64994
        }

        static void ReplaceFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            //int loopLimit;    // Last zStr[] that might match zPattern[]
            Debug.Assert(argc == 3);
            string string_ = Vdbe.Value_Text(argv[0]); // The input string A
            if (string_ == null) return;
            int stringLength = Vdbe.Value_Bytes(argv[0]); // Size of zStr
            Debug.Assert(string_ == Vdbe.Value_Text(argv[0]));  /* No encoding change */
            string pattern = Vdbe.Value_Text(argv[1]); // The pattern string B
            if (pattern == null)
            {
                Debug.Assert(Vdbe.Value_Type(argv[1]) == TYPE.NULL || Vdbe.Context_Ctx(fctx).MallocFailed);
                return;
            }
            if (pattern == string.Empty)
            {
                Debug.Assert(Vdbe.Value_Type(argv[1]) != TYPE.NULL);
                Vdbe.Result_Value(fctx, argv[0]);
                return;
            }
            int patternLength = Vdbe.Value_Bytes(argv[1]); // Size of zPattern
            Debug.Assert(pattern == Vdbe.Value_Text(argv[1]));  /* No encoding change */
            string replacement = Vdbe.Value_Text(argv[2]); // The replacement string C
            if (replacement == null) return;
            int replacementLength = Vdbe.Value_Bytes(argv[2]); // Size of zRep
            Debug.Assert(replacement == Vdbe.Value_Text(argv[2]));
            long outLength = stringLength + 1; // Maximum size of zOut
            Debug.Assert(outLength < MAX_LENGTH);
            string out_ = null; // The output
            if (outLength <= Vdbe.Context_Ctx(fctx).Limits[(int)LIMIT.LENGTH])
                try { out_ = string_.Replace(pattern, replacement); j = out_.Length; }
                catch { j = 0; }
            if (j == 0 || j > Vdbe.Context_Ctx(fctx).Limits[(int)LIMIT.LENGTH])
                Vdbe.Result_ErrorOverflow(fctx);
            else
                Vdbe.Result_Text(fctx, out_, j, null); //: _free
        }

        static int[] _trimOneLength = new int[] { 1 };
        static byte[] _trimOne = new byte[] { (byte)' ' };
        static void TrimFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string charSet; // Set of characters to trim
            int[] charsLength = null; // Length of each character in zCharSet
            byte[][] chars = null; // Individual characters in zCharSet
            int charSetLength = 0; // Number of characters in zCharSet
            int flags; // 1: trimleft  2: trimright  3: trim

            if (Vdbe.Value_Type(argv[0]) == TYPE.NULL)
                return;
            string in_ = Vdbe.Value_Text(argv[0]); // Input string
            if (in_ == null) return;
            int inLength = Vdbe.Value_Bytes(argv[0]); // Number of bytes in input
            //? Debug.Assert(in_ == Vdbe.Value_Text(argv[0]));
            if (argc == 1)
            {
                charSetLength = 1;
                charsLength = _trimOneLength;
                chars = new byte[1][];
                chars[0] = _trimOne;
                charSet = null;
            }
            else if ((charSet = Vdbe.Value_Text(argv[1])) == null)
                return;
            else
            {
                byte[] zbytes = null;
                if ((zbytes = Vdbe.Value_Blob(argv[1])) != null)
                {
                    int iz = 0;
                    for (charSetLength = 0; iz < zbytes.Length; charSetLength++)
                        C._strskiputf8(zbytes, ref iz);
                    if (charSetLength > 0)
                    {
                        chars = new byte[charSetLength][];
                        if (chars == null)
                            return;
                        charsLength = new int[charSetLength];

                        int iz0 = 0;
                        int iz1 = 0;
                        for (int ii = 0; ii < charSetLength; ii++)
                        {
                            C._strskiputf8(zbytes, ref iz1);
                            charsLength[ii] = iz1 - iz0;
                            chars[ii] = new byte[charsLength[ii]];
                            Buffer.BlockCopy(zbytes, iz0, chars[ii], 0, chars[ii].Length);
                            iz0 = iz1;
                        }
                    }
                }
            }
            int i;
            int inIdx_ = 0; // C# string pointer
            byte[] zblob = Vdbe.Value_Blob(argv[0]);
            if (charSetLength > 0)
            {
                flags = (int)Vdbe.User_Data(fctx);
                if ((flags & 1) != 0)
                {
                    while (inLength > 0)
                    {
                        int len = 0;
                        for (i = 0; i < charSetLength; i++)
                        {
                            len = charsLength[i];
                            if (len <= inLength && C._memcmp(zblob, inIdx_, chars[i], len) == 0) break;
                        }
                        if (i >= charSetLength) break;
                        inIdx_ += len;
                        inLength -= len;
                    }
                }
                if ((flags & 2) != 0)
                {
                    while (inLength > 0)
                    {
                        int len = 0;
                        for (i = 0; i < charSetLength; i++)
                        {
                            len = charsLength[i];
                            if (len <= inLength && C._memcmp(zblob, inIdx_ + inLength - len, chars[i], len) == 0) break;
                        }
                        if (i >= charSetLength) break;
                        inLength -= len;
                    }
                }
                if (charSet != null)
                    C._free(ref chars);
            }
            StringBuilder sb = new StringBuilder(inLength);
            for (i = 0; i < inLength; i++)
                sb.Append((char)zblob[inIdx_ + i]);
            Vdbe.Result_Text(fctx, sb, inLength, C.DESTRUCTOR_TRANSIENT);
        }

#if SOUNDEX
        static const byte[] _soundexCode = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
};

        static void SoundexFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            string z = Vdbe.Value_Text(argv[0]);
            if (z == null) z = string.Empty;
            int i;
            for (i = 0; z[i] && !char.IsLetter(z[i]); i++) { }
            byte[] r = new byte[8];
            if (z[i])
            {
                byte prevcode = _soundexCode[z[i] & 0x7f];
                r[0] = char.ToUpperInvariant(z[i]);
                int j;
                for (j = 1; j < 4 && z[i]; i++)
                {
                    byte code = _soundexCode[z[i] & 0x7f];
                    if (code > 0)
                    {
                        if (code != prevcode)
                        {
                            prevcode = code;
                            r[j++] = code + '0';
                        }
                    }
                    else
                        prevcode = 0;
                }
                while (j < 4)
                    r[j++] = '0';
                r[j] = 0;
                Vdbe.Result_Text(fctx, r, 4, C.DESTRUCTOR_TRANSIENT);
            }
            else
                Vdbe.Result_Text(fctx, "?000", 4, C.DESTRUCTOR_STATIC); // IMP: R-64894-50321 The string "?000" is returned if the argument is NULL or contains no ASCII alphabetic characters.
        }
#endif

#if !OMIT_LOAD_EXTENSION
        static void LoadExtFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string file = Vdbe.Value_Text(argv[0]);
            Context ctx = Vdbe.Context_Ctx(fctx);
            string errMsg = string.Empty;
            string proc = (argc == 2 ? Vdbe.Value_Text(argv[1]) : string.Empty);
            if (file != null && Vdbe.LoadExtension(ctx, file, proc, ref errMsg) != 0)
            {
                Vdbe.Result_Error(fctx, errMsg, -1);
                C._tagfree(ctx, ref errMsg);
            }
        }
#endif

        public class SumCtx
        {
            public double RSum;     // Floating point sum
            public long ISum;       // Integer sum
            public long Count;      // Number of elements summed
            public bool Overflow;   // True if integer overflow seen
            public bool Approx;     // True if non-integer value was input to the sum
            public Mem _M;

            public static SumCtx ToCtx(Mem ctx)
            {
                var p = Mem.ToSumCtx_(ctx);
                p._M = ctx;
                p.ISum = (ctx.z == null ? 0 : Convert.ToInt64(ctx.z));
                return p;
            }
        };

        static void SumStep(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            SumCtx p = SumCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 1));
            TYPE type = Vdbe.Value_NumericType(argv[0]);
            if (p != null && type != TYPE.NULL)
            {
                p.Count++;
                if (type == TYPE.INTEGER)
                {
                    long v = Vdbe.Value_Int64(argv[0]);
                    p.RSum += v;
                    if (!(p.Approx | p.Overflow) && sqlite3AddInt64(ref p.ISum, v) != 0)
                        p.Overflow = true;
                }
                else
                {
                    p.RSum += Vdbe.Value_Double(argv[0]);
                    p.Approx = true;
                }
            }
        }

        static void SumFinalize(FuncContext fctx)
        {
            SumCtx p = SumCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 0));
            if (p != null && p.Count > 0)
            {
                if (p.Overflow)
                    Vdbe.Result_Error(fctx, "integer overflow", -1);
                else if (p.Approx)
                    Vdbe.Result_Double(fctx, p.RSum);
                else
                    Vdbe.Result_Int64(fctx, p.ISum);
                p.Count = 0; // Reset for C#
            }
        }

        static void AvgFinalize(FuncContext fctx)
        {
            SumCtx p = SumCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 0));
            if (p != null && p.Count > 0)
                Vdbe.Result_Double(fctx, p.RSum / (double)p.Count);
        }

        static void TotalFinalize(FuncContext fctx)
        {
            SumCtx p = SumCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 0));
            Vdbe.Result_Double(fctx, p != null ? p.RSum : (double)0); // (double)0 In case of OMIT_FLOATING_POINT...
        }

        public class CountCtx
        {
            long _n;
            internal Mem _M;

            public long N
            {
                get { return _n; }
                set
                {
                    _n = value;
                    if (_M != null) _M.z = _n.ToString();
                }
            }

            public static CountCtx ToCtx(Mem ctx)
            {
                if (ctx == null) return null;
                return new CountCtx
                {
                    _M = ctx,
                    _n = (_M == null || _M.z == null ? 0 : Convert.ToInt64(_M.z)),
                };
            }
        }

        static void CountStep(FuncContext fctx, int argc, Mem[] argv)
        {
            CountCtx p = CountCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 1));
            if ((argc == 0 || Vdbe.Value_Type(argv[0]) != TYPE.NULL) && p._M != null)
                p.N++;
        }

        static void CountFinalize(FuncContext fctx)
        {
            CountCtx p = CountCtx.ToCtx(Vdbe.Aggregate_Context(fctx, 0));
            Vdbe.Result_Int64(fctx, p != null ? p.N : 0);
        }

        static void MinMaxStep(FuncContext fctx, int notUsed1, Mem[] argv)
        {
            Mem arg = (Mem)argv[0];
            Mem best = (Mem)Vdbe.Aggregate_Context(fctx, 1);
            if (best == null) return;
            if (Vdbe.Value_Type(argv[0]) == TYPE.NULL)
            {
                if (best.Flags != 0) sqlite3SkipAccumulatorLoad(fctx);
            }
            else if (best.Flags != 0)
            {
                CollSeq coll = GetFuncCollSeq(fctx);
                // This step function is used for both the min() and max() aggregates, the only difference between the two being that the sense of the
                // comparison is inverted. For the max() aggregate, the sqlite3_user_data() function returns (void *)-1. For min() it
                // returns (void *)db, where db is the sqlite3* database pointer. Therefore the next statement sets variable 'max' to 1 for the max()
                // aggregate, or 0 for min().
                bool max = ((int)Vdbe.User_Data(fctx) != 0);
                int cmp = sqlite3MemCompare(best, arg, coll);
                if ((max && cmp < 0) || (!max && cmp > 0))
                    sqlite3VdbeMemCopy(best, arg);
                else
                    sqlite3SkipAccumulatorLoad(fctx);
            }
            else
                sqlite3VdbeMemCopy(best, arg);
        }

        static void MinMaxFinalize(FuncContext fctx)
        {
            Mem r = (Mem)Vdbe.Aggregate_Context(fctx, 0);
            if (r != null)
            {
                if (r.Flags != 0)
                    Vdbe.Result_Value(fctx, r);
                Vdbe.MemRelease(r);
            }
        }

        static void GroupConcatStep(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1 || argc == 2);
            if (Vdbe.Value_Type(argv[0]) == TYPE.NULL) return;
            TextBuilder b = Mem.ToTextBuilder_(Vdbe.Aggregate_Context(fctx, 1), 100);
            if (b != null)
            {
                Context ctx = Vdbe.Context_Ctx(fctx);
                bool firstTerm = (b.Tag == null);
                b.Tag = ctx;
                b.MaxSize = ctx.Limits[(int)LIMIT.LENGTH];
                if (!firstTerm)
                {
                    string zSep;
                    int nSep;
                    if (argc == 2)
                    {
                        zSep = Vdbe.Value_Text(argv[1]);
                        nSep = Vdbe.Value_Bytes(argv[1]);
                    }
                    else
                    {
                        zSep = ",";
                        nSep = 1;
                    }
                    b.Append(zSep, nSep);
                }
                string zVal = Vdbe.Value_Text(argv[0]);
                int nVal = Vdbe.Value_Bytes(argv[0]);
                b.Append(zVal, nVal);
            }
        }

        static void GroupConcatFinalize(FuncContext fctx)
        {
            TextBuilder b = Mem.ToTextBuilder_(Vdbe.Aggregate_Context(fctx, 0), 100);
            if (b != null)
            {
                if (b.Overflowed)
                    Vdbe.Result_ErrorOverflow(fctx);
                else if (b.AllocFailed)
                    Vdbe.Result_ErrorNoMem(fctx);
                else
                    Vdbe.Result_Text(fctx, b.ToString(), -1, null);
            }
        }

        public static void RegisterBuiltinFunctions(Context ctx)
        {
            RC rc = sqlite3_overload_function(ctx, "MATCH", 2);
            Debug.Assert(rc == RC.NOMEM || rc == RC.OK);
            if (rc == RC.NOMEM)
                ctx.MallocFailed = true;
        }

        static void SetLikeOptFlag(Context ctx, string name, FUNC flagVal)
        {
            FuncDef def = Callback.FindFunction(ctx, name, name.Length, 2, TEXTENCODE.UTF8, 0);
            if (C._ALWAYS(def != null))
                def.Flags = flagVal;
        }

        public static void RegisterLikeFunctions(Context ctx, bool caseSensitive)
        {
            CompareInfo info = (caseSensitive ? _likeInfoAlt : _likeInfoNorm);
            sqlite3CreateFunc(ctx, "like", 2, TEXTENCODE.UTF8, info, LikeFunc, null, null, null);
            sqlite3CreateFunc(ctx, "like", 3, TEXTENCODE.UTF8, info, LikeFunc, null, null, null);
            sqlite3CreateFunc(ctx, "glob", 2, TEXTENCODE.UTF8, _globInfo, LikeFunc, null, null, null);
            SetLikeOptFlag(ctx, "glob", FUNC.LIKE | FUNC.CASE);
            SetLikeOptFlag(ctx, "like", caseSensitive ? (FUNC.LIKE | FUNC.CASE) : FUNC.LIKE);
        }

        public static bool IsLikeFunction(Context ctx, Expr expr, ref bool isNoCase, char[] wc)
        {
            if (expr.OP != TK.FUNCTION || expr.x.List == null || expr.x.List.Exprs != 2)
                return false;
            Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect));
            FuncDef def = Callback.FindFunction(ctx, expr.u.Token, expr.u.Token.Length, 2, TEXTENCODE.UTF8, 0);
            if (C._NEVER(def == null) || (def.Flags & FUNC.LIKE) == 0)
                return false;
            // The memcpy() statement assumes that the wildcard characters are the first three statements in the compareInfo structure.  The asserts() that follow verify that assumption
            //: memcpy(wc, def.UserData, 3);
            wc[0] = ((CompareInfo)def.UserData).MatchAll;
            wc[1] = ((CompareInfo)def.UserData).MatchOne;
            wc[2] = ((CompareInfo)def.UserData).MatchSet;
            //: Debug.Assert((char *)&_likeInfoAlt == (char *)&_likeInfoAlt.MatchAll);
            //: Debug.Assert(&((char *)&_likeInfoAlt)[1] == (char *)&_likeInfoAlt.MatchOne);
            //: Debug.Assert(&((char *)&_likeInfoAlt)[2] == (char *)&_likeInfoAlt.MatchSet);
            isNoCase = ((def.Flags & FUNC.CASE) == 0);
            return true;
        }

        // The following array holds FuncDef structures for all of the functions defined in this file.
        //
        // The array cannot be constant since changes are made to the FuncDef.pHash elements at start-time.  The elements of this array are read-only after initialization is complete.
        FuncDef[] _builtinFunc =  {
FUNCTION("ltrim",              1, 1, 0, TrimFunc         ),
FUNCTION("ltrim",              2, 1, 0, TrimFunc         ),
FUNCTION("rtrim",              1, 2, 0, TrimFunc         ),
FUNCTION("rtrim",              2, 2, 0, TrimFunc         ),
FUNCTION("trim",               1, 3, 0, TrimFunc         ),
FUNCTION("trim",               2, 3, 0, TrimFunc         ),
FUNCTION("min",               -1, 0, 1, MinMaxFunc       ),
FUNCTION("min",                0, 0, 1, null                ),
AGGREGATE("min",               1, 0, 1, MinMaxStep,      MinMaxFinalize),
FUNCTION("max",               -1, 1, 1, MinMaxFunc       ),
FUNCTION("max",                0, 1, 1, null                ),
AGGREGATE("max",               1, 1, 1, MinMaxStep,      MinMaxFinalize),
FUNCTION2("typeof",            1, 0, 0, TypeofFunc       ),
FUNCTION2("length",            1, 0, 0, LengthFunc       ),
FUNCTION("substr",             2, 0, 0, SubstrFunc       ),
FUNCTION("substr",             3, 0, 0, SubstrFunc       ),
FUNCTION("abs",                1, 0, 0, AbsFunc          ),
#if !OMIT_FLOATING_POINT
FUNCTION("round",              1, 0, 0, RoundFunc        ),
FUNCTION("round",              2, 0, 0, RoundFunc        ),
#endif
FUNCTION("upper",              1, 0, 0, UpperFunc        ),
FUNCTION("lower",              1, 0, 0, LowerFunc        ),
FUNCTION("coalesce",           1, 0, 0, null             ),
FUNCTION("coalesce",           0, 0, 0, null             ),
/*  FUNCTION(coalesce,          -1, 0, 0, ifnullFunc       ), */
// use versionFunc here just for a dummy placeholder
new FuncDef(-1,SQLITE_UTF8,SQLITE_FUNC_COALESCE,null,null,versionFunc,null,null,"coalesce",null,null), 
FUNCTION("hex",                1, 0, 0, hexFunc          ),
/*  FUNCTION(ifnull,             2, 0, 0, ifnullFunc       ), */
// use versionFunc here just for a dummy placeholder
new FuncDef(2,SQLITE_UTF8,SQLITE_FUNC_COALESCE,null,null,versionFunc,null,null,"ifnull",null,null),
FUNCTION("random",             0, 0, 0, randomFunc       ),
FUNCTION("randomblob",         1, 0, 0, randomBlob       ),
FUNCTION("nullif",             2, 0, 1, nullifFunc       ),
FUNCTION("sqlite_version",     0, 0, 0, versionFunc      ),
FUNCTION("sqlite_source_id",   0, 0, 0, sourceidFunc     ),
FUNCTION("sqlite_log",         2, 0, 0, errlogFunc       ),
#if !OMIT_COMPILEOPTION_DIAGS
FUNCTION("sqlite_compileoption_used",1, 0, 0, compileoptionusedFunc  ),
FUNCTION("sqlite_compileoption_get", 1, 0, 0, compileoptiongetFunc  ),
#endif
FUNCTION("quote",              1, 0, 0, quoteFunc        ),
FUNCTION("last_insert_rowid",  0, 0, 0, last_insert_rowid),
FUNCTION("changes",            0, 0, 0, changes          ),
FUNCTION("total_changes",      0, 0, 0, total_changes    ),
FUNCTION("replace",            3, 0, 0, replaceFunc      ),
FUNCTION("zeroblob",           1, 0, 0, zeroblobFunc     ),
#if SOUNDEX
FUNCTION("soundex",            1, 0, 0, soundexFunc      ),
#endif
#if !OMIT_LOAD_EXTENSION
FUNCTION("load_extension",     1, 0, 0, loadExt          ),
FUNCTION("load_extension",     2, 0, 0, loadExt          ),
#endif
AGGREGATE("sum",               1, 0, 0, sumStep,         sumFinalize    ),
AGGREGATE("total",             1, 0, 0, sumStep,         totalFinalize    ),
AGGREGATE("avg",               1, 0, 0, sumStep,         avgFinalize    ),
/*AGGREGATE("count",             0, 0, 0, countStep,       countFinalize  ), */
/* AGGREGATE(count,             0, 0, 0, countStep,       countFinalize  ), */
new FuncDef( 0,SQLITE_UTF8,SQLITE_FUNC_COUNT,null,null,null,countStep,countFinalize,"count",null,null),
AGGREGATE("count",             1, 0, 0, countStep,       countFinalize  ),
AGGREGATE("group_concat",      1, 0, 0, groupConcatStep, groupConcatFinalize),
AGGREGATE("group_concat",      2, 0, 0, groupConcatStep, groupConcatFinalize),

LIKEFUNC("glob", 2, _globInfo, FUNC.LIKE|FUNC.CASE),
#if CASE_SENSITIVE_LIKE
LIKEFUNC("like", 2, _likeInfoAlt, FUNC_LIKE|FUNC.CASE),
LIKEFUNC("like", 3, _likeInfoAlt, FUNC_LIKE|FUNC.CASE),
#else
LIKEFUNC("like", 2, _likeInfoNorm, FUNC.LIKE),
LIKEFUNC("like", 3, _likeInfoNorm, FUNC.LIKE),
#endif
                                  };

        public static void RegisterGlobalFunctions()
        {
            FuncDefHash hash = sqlite3GlobalFunctions;
            FuncDef func = _builtinFunc;
            for (int i = 0; i < _lengthof(_builtinFunc); i++)
                hash.Insert(func[i]);
            Date.RegisterDateTimeFunctions();
#if !OMIT_ALTERTABLE
            Alter.Functions();
#endif
        }
    }
}

//public struct sFuncs
//{
//    public string zName;
//    public sbyte nArg;
//    public u8 argType;           /* 1: 0, 2: 1, 3: 2,...  N:  N-1. */
//    public u8 eTextRep;          /* 1: UTF-16.  0: UTF-8 */
//    public u8 needCollSeq;
//    public dxFunc xFunc; //(FuncContext*,int,sqlite3_value **);

//    // Constructor
//    public sFuncs(string zName, sbyte nArg, u8 argType, u8 eTextRep, u8 needCollSeq, dxFunc xFunc)
//    {
//        this.zName = zName;
//        this.nArg = nArg;
//        this.argType = argType;
//        this.eTextRep = eTextRep;
//        this.needCollSeq = needCollSeq;
//        this.xFunc = xFunc;
//    }
//};

//public struct sAggs
//{
//    public string zName;
//    public sbyte nArg;
//    public u8 argType;
//    public u8 needCollSeq;
//    public dxStep xStep; //(FuncContext*,int,sqlite3_value**);
//    public dxFinal xFinalize; //(FuncContext*);
//    // Constructor
//    public sAggs(string zName, sbyte nArg, u8 argType, u8 needCollSeq, dxStep xStep, dxFinal xFinalize)
//    {
//        this.zName = zName;
//        this.nArg = nArg;
//        this.argType = argType;
//        this.needCollSeq = needCollSeq;
//        this.xStep = xStep;
//        this.xFinalize = xFinalize;
//    }
//}