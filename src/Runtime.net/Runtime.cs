using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public struct array_t<T>
    {
        public int length;
        public T[] data;
        public array_t(T[] a) { data = a; length = 0; }
        public array_t(T[] a, int b) { data = a; length = b; }
        public static implicit operator T[](array_t<T> p) { return p.data; }
        public T this[int i] { get { return data[i]; } set { data[i] = value; } }
    };
    public struct array_t2<TLength, T> where TLength : struct
    {
        public TLength length;
        public T[] data;
        public array_t2(T[] a) { data = a; length = default(TLength); }
        public array_t2(T[] a, TLength b) { data = a; length = b; }
        public static implicit operator T[](array_t2<TLength, T> p) { return p.data; }
        public T this[int i] { get { return data[i]; } set { data[i] = value; } }
    };
    public struct array_t3<TLength, T> where TLength : struct
    {
        public TLength length;
        public T[] data;
        public array_t3(T[] a) { data = a; length = default(TLength); }
        public array_t3(T[] a, TLength b) { data = a; length = b; }
        public static implicit operator T[](array_t3<TLength, T> p) { return p.data; }
        public T this[int i] { get { return data[i]; } set { data[i] = value; } }
    };

    [Flags]
    public enum MEMTYPE : byte
    {
        HEAP = 0x01,         // General heap allocations
        LOOKASIDE = 0x02,    // Might have been lookaside memory
        SCRATCH = 0x04,      // Scratch allocations
        PCACHE = 0x08,       // Page cache allocations
        DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
    }

    //////////////////////
    // PRINT
    #region PRINT
    public partial class TextBuilder
    {
        public object Tag;			// Optional database for lookaside.  Can be NULL
        public string Base;		// A base allocation.  Not from malloc.
        public StringBuilder Text; // The string collected so far
        public int Size { get { return Text.Length; } } // Amount of space allocated in zText
        public int MaxSize;		// Maximum allowed string length
        public bool AllocFailed;  // Becomes true if any memory allocation fails
        public bool Overflowed { get { return MaxSize > 0 && Text.Length > MaxSize; } } // Becomes true if string size exceeds limits

        public void AppendSpace(int length)
        {
            Text.AppendFormat("{0," + length + "}", "");
        }

        public void AppendFormat(string fmt, params object[] args)
        {
            C.vxprintf(this, false, fmt, args);
        }

        public void Append(string z, int length)
        {
            Debug.Assert(z != null || length == 0);
            if (Overflowed)
            {
                C.ASSERTCOVERAGE(Overflowed);
                return;
            }
            if (length < 0)
                length = z.Length;
            if (length == 0 || C._NEVER(z == null))
                return;
            Text.Append(z.Substring(0, length <= z.Length ? length : z.Length));
        }

        public new string ToString()
        {
            return Text.ToString();
        }

        public void Reset()
        {
            Text.Length = 0;
        }

        public static void Init(TextBuilder b, int capacity, int maxSize)
        {
            b.Text.Length = 0;
            if (b.Text.Capacity < capacity)
                b.Text.Capacity = capacity;
            b.Tag = null;
            b.MaxSize = maxSize;
        }

        public TextBuilder() { Text = new StringBuilder(); }
        public TextBuilder(int n) { Text = new StringBuilder(n); }
    }
    #endregion

    public static partial class C
    {
        public const double BIG_DOUBLE = 1e99;

        #region ASSERT
        public static bool _NEVER(bool x) { return x; }
        public static void ASSERTCOVERAGE(bool p) { }
        public static T _ALWAYS<T>(T x) { if (!x.Equals(default(T))) Debug.Assert(false); return x; }
        #endregion

        //////////////////////
        // STDARGS
        #region STDARGS

        static long __arg(ref int idx, object[] args, long notUsed)
        {
            if (args[idx] is long)
                return Convert.ToInt64(args[idx++]);
            return (long)(args[idx++].GetHashCode());
        }
        static int __arg(ref int idx, object[] args, int notUsed)
        {
            if (Convert.ToInt64(args[idx]) > 0 && Convert.ToUInt32(args[idx]) > int.MaxValue)
                return (int)(Convert.ToUInt32(args[idx++]) - int.MaxValue - 1);
            return (int)Convert.ToInt32(args[idx++]);
        }
        static double __arg(ref int idx, object[] args, double notUsed)
        {
            return Convert.ToDouble(args[idx++]);
        }
        static string __arg(ref int idx, object[] args, string notUsed)
        {
            if (args.Length < idx - 1 || args[idx] == null) { idx++; return "NULL"; }
            if (args[idx] is byte[])
            {
                if (Encoding.UTF8.GetString((byte[])args[idx], 0, ((byte[])args[idx]).Length) == "\0")
                {
                    idx++;
                    return string.Empty;
                }
                return Encoding.UTF8.GetString((byte[])args[idx], 0, ((byte[])args[idx++]).Length);
            }
            if (args[idx] is int) { idx++; return null; }
            if (args[idx] is StringBuilder) return (string)args[idx++].ToString();
            if (args[idx] is char) return ((char)args[idx++]).ToString();
            return (string)args[idx++];
        }

        #endregion

        //////////////////////
        // UTF
        #region UTF

        static byte[] _utf8Trans1 = new byte[] 
    {
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x00, 0x00,
};

        public static uint _utf8read(string z, ref string zNext)
        {
            // Same as READ_UTF8() above but without the zTerm parameter. For this routine, we assume the UTF8 string is always zero-terminated.
            if (string.IsNullOrEmpty(z)) return 0;
            int zIdx = 0;
            uint c = z[zIdx++];
            if (c >= 0xc0)
            {
                //c = _utf8Trans1[c - 0xc0];
                while (zIdx != z.Length && (z[zIdx] & 0xc0) == 0x80)
                    c = (uint)((c << 6) + (0x3f & z[zIdx++]));
                if (c < 0x80 || (c & 0xFFFFF800) == 0xD800 || (c & 0xFFFFFFFE) == 0xFFFE) c = 0xFFFD;
            }
            zNext = z.Substring(zIdx);
            return c;
        }

        public static int _utf8charlength(string z, int bytes)
        {
            if (z.Length == 0) return 0;
            int zLength = z.Length;
            int zTerm = (bytes >= 0 && bytes <= zLength ? bytes : zLength);
            if (zTerm == zLength)
                return zLength - (z[zTerm - 1] == 0 ? 1 : 0);
            else
                return bytes;
        }

#if DEBUG
        public static int _utf8to8(byte[] z)
        {
            try
            {
                string z2 = Encoding.UTF8.GetString(z, 0, z.Length);
                byte[] zOut = Encoding.UTF8.GetBytes(z2);
                {
                    Array.Copy(zOut, 0, z, 0, z.Length);
                    return z.Length;
                }
            }
            catch (EncoderFallbackException) { return 0; }
        }
#endif

#if !OMIT_UTF16
        public static int _utf16bytelength(byte[] z, int chars)
        {
            string z2 = Encoding.UTF32.GetString(z, 0, z.Length);
            return Encoding.UTF32.GetBytes(z2).Length;
        }

        //#if defined(TEST)
        //	__device__ void _runtime_utfselftest()
        //	{
        //		unsigned int i, t;
        //		unsigned char buf[20];
        //		unsigned char *z;
        //		int n;
        //		unsigned int c;
        //		for (i = 0; i < 0x00110000; i++)
        //		{
        //			z = buf;
        //			WRITE_UTF8(z, i);
        //			n = (int)(z - buf);
        //			_assert(n > 0 && n <= 4);
        //			z[0] = 0;
        //			z = buf;
        //			c = Utf8Read((const uint8 **)&z);
        //			t = i;
        //			if (i >= 0xD800 && i <= 0xDFFF) t = 0xFFFD;
        //			if ((i&0xFFFFFFFE) == 0xFFFE) t = 0xFFFD;
        //			_assert(c == t);
        //			_assert((z - buf) == n);
        //		}
        //		for (i = 0; i < 0x00110000; i++)
        //		{
        //			if (i >= 0xD800 && i < 0xE000) continue;
        //			z = buf;
        //			WRITE_UTF16LE(z, i);
        //			n = (int)(z - buf);
        //			_assert(n > 0 && n <= 4);
        //			z[0] = 0;
        //			z = buf;
        //			READ_UTF16LE(z, 1, c);
        //			_assert(c == i);
        //			_assert((z - buf) == n);
        //		}
        //		for (i = 0; i < 0x00110000; i++)
        //		{
        //			if (i >= 0xD800 && i < 0xE000) continue;
        //			z = buf;
        //			WRITE_UTF16BE(z, i);
        //			n = (int)(z-buf);
        //			_assert(n > 0 && n <= 4);
        //			z[0] = 0;
        //			z = buf;
        //			READ_UTF16BE(z, 1, c);
        //			_assert(c == i);
        //			_assert((z - buf) == n);
        //		}
        //	}
        //#endif

#endif

        #endregion

        //////////////////////
        // FUNC
        #region FUNC

        public static void _strskiputf8(string z, ref int idx)
        {
            idx++;
            if (idx < z.Length && z[idx - 1] >= 0xC0)
                while (idx < z.Length && (z[idx] & 0xC0) == 0x80)
                    idx++;
        }
        public static void _strskiputf8(byte[] z, ref int idx)
        {
            idx++;
            if (idx < z.Length && z[idx - 1] >= 0xC0)
                while (idx < z.Length && (z[idx] & 0xC0) == 0x80)
                    idx++;
        }

        public static int _memcmp(byte[] a, byte[] b, int limit)
        {
            if (a.Length < limit) return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit) return +1;
            for (int i = 0; i < limit; i++)
                if (a[i] != b[i]) return (a[i] < b[i] ? -1 : 1);
            return 0;
        }
        public static int _memcmp(string a, byte[] b, int limit)
        {
            if (a.Length < limit) return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit) return +1;
            char[] cA = a.ToCharArray();
            for (int i = 0; i < limit; i++)
                if (cA[i] != b[i]) return (cA[i] < b[i] ? -1 : 1);
            return 0;
        }
        public static int _memcmp(byte[] a, int aOffset, byte[] b, int limit)
        {
            if (a.Length < aOffset + limit) return (a.Length - aOffset < b.Length ? -1 : +1);
            if (b.Length < limit) return +1;
            for (int i = 0; i < limit; i++)
                if (a[i + aOffset] != b[i]) return (a[i + aOffset] < b[i] ? -1 : 1);
            return 0;
        }
        public static int _memcmp(byte[] a, int aOffset, byte[] b, int bOffset, int limit)
        {
            if (a.Length < aOffset + limit) return (a.Length - aOffset < b.Length - bOffset ? -1 : +1);
            if (b.Length < bOffset + limit) return +1;
            for (int i = 0; i < limit; i++)
                if (a[i + aOffset] != b[i + bOffset])
                    return (a[i + aOffset] < b[i + bOffset] ? -1 : 1);
            return 0;
        }
        public static int _memcmp(byte[] a, int aOffset, string b, int limit)
        {
            if (a.Length < aOffset + limit) return (a.Length - aOffset < b.Length ? -1 : +1);
            if (b.Length < limit) return +1;
            for (int i = 0; i < limit; i++)
                if (a[i + aOffset] != b[i]) return (a[i + aOffset] < b[i] ? -1 : 1);
            return 0;
        }
        public static int _memcmp(string a, string b, int limit)
        {
            if (a.Length < limit) return (a.Length < b.Length ? -1 : +1);
            if (b.Length < limit) return +1;
            int rc;
            if ((rc = string.Compare(a, 0, b, 0, limit, StringComparison.Ordinal)) == 0) return 0;
            return (rc < 0 ? -1 : +1);
        }

        public static char _hextobyte(char h)
        {
            Debug.Assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
            return (char)((h + 9 * (1 & (h >> 6))) & 0xf);
        }
#if !OMIT_BLOB_LITERAL
        public static byte[] _taghextoblob(object tag, string z, int size)
        {
            StringBuilder b = new StringBuilder(size / 2 + 1);
            size--;
            if (b != null)
            {
                int i;
                for (i = 0; i < size; i += 2)
                    b.Append(Convert.ToChar((C._hextobyte(z[i]) << 4) | C._hextobyte(z[i + 1])));
                //z_[i / 2] = '\0'; ;
            }
            return Encoding.UTF8.GetBytes(b.ToString());
        }
#endif

        #endregion

        //////////////////////
        // MEMORY ALLOCATION
        #region MEMORY ALLOCATION

        public static int _ROUND8(int x) { return (x + 7) & ~7; }
        public static int _ROUNDDOWN8(int x) { return x & ~7; }
#if BYTEALIGNED4
        public static bool _HASALIGNMENT8(int x) { return true; }
#else
        public static bool _HASALIGNMENT8(int x) { return true; }
#endif

#if MEMDEBUG
        //public static void _memdbg_settype<T>(T X, MEMTYPE Y);
        //public static bool _memdbg_hastype<T>(T X, MEMTYPE Y);
        //public static bool _memdbg_nottype<T>(T X, MEMTYPE Y);
#else
        public static void _memdbg_settype<T>(T X, MEMTYPE Y) { }
        public static bool _memdbg_hastype<T>(T X, MEMTYPE Y) { return true; }
        public static bool _memdbg_nottype<T>(T X, MEMTYPE Y) { return true; }
#endif

        public static byte[] _alloc(int size) { return new byte[size]; }
        public static byte[] _alloc2(int size, bool clear) { return new byte[size]; }
        public static T[] _alloc<T>(byte s, int size) where T : struct { return new T[size / s]; }
        public static T[] _alloc2<T>(byte s, int size, bool clear) where T : struct { return new T[size / s]; }
        public static byte[] _tagalloc(object tag, int size) { return new byte[size]; }
        public static byte[] _tagalloc2(object tag, int size, bool clear) { return new byte[size]; }
        public static T[] _tagalloc<T>(object tag, int size) where T : struct { return new T[size]; }
        public static T[] _tagalloc<T>(object tag, byte s, int size) where T : struct { return new T[size / s]; }
        public static T[] _tagalloc2<T>(object tag, byte s, int size, bool clear) where T : struct { return new T[size / s]; }
        public static int _allocsize(byte[] p)
        {
            Debug.Assert(_memdbg_hastype(p, MEMTYPE.HEAP));
            Debug.Assert(_memdbg_nottype(p, MEMTYPE.DB));
            return p.Length;
        }
        public static int _tagallocsize(object tag, byte[] p)
        {
            Debug.Assert(_memdbg_hastype(p, MEMTYPE.HEAP));
            Debug.Assert(_memdbg_nottype(p, MEMTYPE.DB));
            return p.Length;
        }
        public static void _free<T>(ref T p) where T : class { }
        public static void _tagfree<T>(object tag, ref T p) where T : class { p = null; }
        static byte[][] _scratch; // Scratch memory
        public static byte[] _stackalloc(object tag, int size) { return new byte[size]; }
        public static byte[][] _stackalloc(object tag, byte[][] cell, int n)
        {
            cell = _scratch;
            if (cell == null)
                cell = new byte[n < 200 ? 200 : n][];
            else if (cell.Length < n)
                Array.Resize(ref cell, n);
            _scratch = null;
            return cell;
        }
        public static void _stackfree<T>(object tag, ref T p) where T : class
        {
            byte[][] cell = (p as byte[][]);
            if (cell != null)
            {
                if (_scratch == null || _scratch.Length < cell.Length)
                {
                    Debug.Assert(_memdbg_hastype(cell, MEMTYPE.SCRATCH));
                    Debug.Assert(_memdbg_nottype(cell, ~MEMTYPE.SCRATCH));
                    _memdbg_settype(cell, MEMTYPE.HEAP);
                    _scratch = cell;
                }
                // larger Scratch 2 already in use, let the C# GC handle
                cell = null;
            }
            p = null;
        }
        public static T[] _realloc<T>(int s, T[] p, int bytes)
        {
            var newT = new T[bytes / s];
            Array.Copy(p, newT, Math.Min(p.Length, newT.Length));
            return newT;
        }
        public static T[] _tagrealloc<T>(object tag, int s, T[] p, int bytes)
        {
            var newT = new T[bytes / s];
            Array.Copy(p, newT, Math.Min(p.Length, newT.Length));
            return newT;
        }
        public static bool _heapnearlyfull() { return false; }
        public static T[] _tagrealloc_or_free<T>(object tag, ref T[] old, int newSize)
        {
            T[] p = _tagrealloc(tag, 0, old, newSize);
            if (p == null) _tagfree(tag, ref old);
            return p;
        }
        public static void _tagrealloc_or_free2<T>(object tag, ref T[] data, int newSize)
        {
            Array.Resize(ref data, newSize);
        }
        public static void _tagrealloc_or_create<T>(object tag, ref T[] data, int newSize) where T : new()
        {
            if (data == null)
                data = new T[newSize];
            else
                Array.Resize(ref data, newSize);
        }

        public const Action<object> DESTRUCTOR_TRANSIENT = null;
        public const Action<object> DESTRUCTOR_STATIC = null;
        public const Action<object> DESTRUCTOR_DYNAMIC = null;

        #endregion

        //////////////////////
        // PRINT
        #region PRINT
# if SMALL_STACK
        const int BUFSIZE = 50;
# else
        const int BUFSIZE = 350; // Size of the output buffer
#endif
        enum TYPE : byte
        {
            RADIX = 1,          // Integer types.  %d, %x, %o, and so forth
            FLOAT = 2,          // Floating point.  %f
            EXP = 3,            // Exponentional notation. %e and %E
            GENERIC = 4,        // Floating or exponential, depending on exponent. %g
            SIZE = 5,           // Return number of characters processed so far. %n
            STRING = 6,         // Strings. %s
            DYNSTRING = 7,      // Dynamically allocated strings. %z
            PERCENT = 8,        // Percent symbol. %%
            CHARX = 9,          // Characters. %c
            // The rest are extensions, not normally found in printf()
            SQLESCAPE = 10,     // Strings with '\'' doubled.  %q
            SQLESCAPE2 = 11,    // Strings with '\'' doubled and enclosed in '', NULL pointers replaced by SQL NULL.  %Q
            TOKEN = 12,         // a pointer to a Token structure
            SRCLIST = 13,       // a pointer to a SrcList
            POINTER = 14,       // The %p conversion
            SQLESCAPE3 = 15,    // %w . Strings with '\"' doubled
            ORDINAL = 16,       // %r . 1st, 2nd, 3rd, 4th, etc.  English only
            //
            INVALID = 0,        // Any unrecognized conversion type 
        }

        enum FLAG : byte
        {
            SIGNED = 1, // True if the value to convert is signed
            INTERN = 2, // True if for internal use only
            STRING = 4, // Allow infinity precision
        }

        class Info // Information about each format field
        {
            public char Fmttype; // The format field code letter
            public byte Base; // The _base for radix conversion
            public FLAG Flags; // One or more of FLAG_ constants below
            public TYPE Type; // Conversion paradigm
            public byte Charset; // Offset into aDigits[] of the digits string
            public byte Prefix; // Offset into aPrefix[] of the prefix string

            public Info(char fmttype, byte base_, FLAG flags, TYPE type, byte charset, byte prefix)
            {
                Fmttype = fmttype;
                Base = base_;
                Flags = flags;
                Type = type;
                Charset = charset;
                Prefix = prefix;
            }
        }

        static string _digits = "0123456789ABCDEF0123456789abcdef";
        static string _prefix = "-x0\000X0";
        static Info[] _info = new Info[] {
            new Info(  'd', 10, (FLAG)1, TYPE.RADIX,      0,  0 ),
            new Info(   's',  0, (FLAG)4, TYPE.STRING,     0,  0 ),
            new Info(   'g',  0, (FLAG)1, TYPE.GENERIC,    30, 0 ),
            new Info(   'z',  0, (FLAG)4, TYPE.DYNSTRING,  0,  0 ),
            new Info(   'q',  0, (FLAG)4, TYPE.SQLESCAPE,  0,  0 ),
            new Info(   'Q',  0, (FLAG)4, TYPE.SQLESCAPE2, 0,  0 ),
            new Info(   'w',  0, (FLAG)4, TYPE.SQLESCAPE3, 0,  0 ),
            new Info(   'c',  0, (FLAG)0, TYPE.CHARX,      0,  0 ),
            new Info(   'o',  8, (FLAG)0, TYPE.RADIX,      0,  2 ),
            new Info(   'u', 10, (FLAG)0, TYPE.RADIX,      0,  0 ),
            new Info(   'x', 16, (FLAG)0, TYPE.RADIX,      16, 1 ),
            new Info(   'X', 16, (FLAG)0, TYPE.RADIX,      0,  4 ),
            #if !OMIT_FLOATING_POINT
            new Info(   'f',  0, (FLAG)1, TYPE.FLOAT,      0,  0 ),
            new Info(   'e',  0, (FLAG)1, TYPE.EXP,        30, 0 ),
            new Info(   'E',  0, (FLAG)1, TYPE.EXP,        14, 0 ),
            new Info(   'G',  0, (FLAG)1, TYPE.GENERIC,    14, 0 ),
            #endif
            new Info(   'i', 10, (FLAG)1, TYPE.RADIX,      0,  0 ),
            new Info(   'n',  0, (FLAG)0, TYPE.SIZE,       0,  0 ),
            new Info(   '%',  0, (FLAG)0, TYPE.PERCENT,    0,  0 ),
            new Info(   'p', 16, (FLAG)0, TYPE.POINTER,    0,  1 ),
            // All the rest have the FLAG_INTERN bit set and are thus for internal use only
            new Info(   'T',  0, (FLAG)2, TYPE.TOKEN,      0,  0 ),
            new Info(   'S',  0, (FLAG)2, TYPE.SRCLIST,    0,  0 ),
            new Info(   'r', 10, (FLAG)3, TYPE.ORDINAL,    0,  0 ),
        };

#if !OMIT_FLOATING_POINT
        static char GetDigit(ref double val, ref int cnt)
        {
            if (cnt++ >= 16) return '\0';
            int digit = (int)val;
            double d = digit;
            val = (val - d) * 10.0;
            return (char)digit;
        }
#endif

        static readonly char[] _ord = "thstndrd".ToCharArray();
        internal static void vxprintf(TextBuilder b, bool useExtended, string fmt, object[] args)
        {
            int argsIdx = 0;
            char[] buf = new char[BUFSIZE]; // Conversion buffer
            int bufpt = 0; // Pointer to the conversion buffer
            fmt += '\0'; int fmt_ = 0; // Work around string pointer
            int c; // Next character in the format string
            bool flag_leftjustify = false;  // True if "-" flag is present
            int width = 0; // Width of the current field
            int length = 0; // Length of the field
            for (; fmt_ <= fmt.Length && (c = fmt[fmt_]) != 0; ++fmt_)
            {
                if (c != '%')
                {
                    bufpt = fmt_;
                    int amt = 1;
                    while (fmt_ < fmt.Length && (c = (fmt[++fmt_])) != '%' && c != 0) amt++;
                    b.Append(fmt.Substring(bufpt, amt), amt);
                    if (c == 0) break;
                }
                if (fmt_ < fmt.Length && (c = (fmt[++fmt_])) == 0)
                {
                    b.Append("%", 1);
                    break;
                }
                // Find out what flags are present
                flag_leftjustify = false; // True if "-" flag is present
                bool flag_plussign = false; // True if "+" flag is present
                bool flag_blanksign = false; // True if " " flag is present
                bool flag_alternateform = false; // True if "#" flag is present
                bool flag_altform2 = false; // True if "!" flag is present
                bool flag_zeropad = false; // True if field width constant starts with zero
                bool done = false; // Loop termination flag
                do
                {
                    switch (c)
                    {
                        case '-': flag_leftjustify = true; break;
                        case '+': flag_plussign = true; break;
                        case ' ': flag_blanksign = true; break;
                        case '#': flag_alternateform = true; break;
                        case '!': flag_altform2 = true; break;
                        case '0': flag_zeropad = true; break;
                        default: done = true; break;
                    }
                } while (!done && fmt_ < fmt.Length - 1 && (c = (fmt[++fmt_])) != 0);
                // Get the field width
                width = 0; // Width of the current field
                if (c == '*')
                {
                    width = __arg(ref argsIdx, args, (int)0);
                    if (width < 0)
                    {
                        flag_leftjustify = true;
                        width = -width;
                    }
                    c = fmt[++fmt_];
                }
                else
                {
                    while (c >= '0' && c <= '9')
                    {
                        width = width * 10 + c - '0';
                        c = fmt[++fmt_];
                    }
                }
                if (width > BUFSIZE - 10) width = BUFSIZE - 12;
                // Get the precision
                int precision; // Precision of the current field
                if (c == '.')
                {
                    precision = 0;
                    c = fmt[++fmt_];
                    if (c == '*')
                    {
                        precision = __arg(ref argsIdx, args, (int)0);
                        if (precision < 0) precision = -precision;
                        c = fmt[++fmt_];
                    }
                    else
                    {
                        while (c >= '0' && c <= '9')
                        {
                            precision = precision * 10 + c - '0';
                            c = fmt[++fmt_];
                        }
                    }
                }
                else
                    precision = -1;
                // Get the conversion type modifier
                bool flag_long; // True if "l" flag is present
                bool flag_longlong; // True if the "ll" flag is present
                if (c == 'l')
                {
                    flag_long = true;
                    c = fmt[++fmt_];
                    if (c == 'l')
                    {
                        flag_longlong = true;
                        c = fmt[++fmt_];
                    }
                    else
                        flag_longlong = false;
                }
                else
                    flag_long = flag_longlong = false;
                // Fetch the info entry for the field
                Info info = _info[0]; // Pointer to the appropriate info structure
                TYPE type = TYPE.INVALID; // Conversion paradigm
                int i;
                for (i = 0; i < _info.Length; i++)
                {
                    if (c == _info[i].Fmttype)
                    {
                        info = _info[i];
                        if (useExtended || (info.Flags & FLAG.INTERN) == 0) type = info.Type;
                        else return;
                        break;
                    }
                }

                char prefix; // Prefix character.  "+" or "-" or " " or '\0'. */
                long longvalue;
                double realvalue; // Value for real types
#if !OMIT_FLOATING_POINT
                int exp, e2; // exponent of real numbers
                int nsd; // Number of significant digits returned
                double rounder; // Used for rounding floating point values
                bool flag_dp; // True if decimal point should be shown
                bool flag_rtz; // True if trailing zeros should be removed

#endif

                // At this point, variables are initialized as follows:
                //   flag_alternateform          TRUE if a '#' is present.
                //   flag_altform2               TRUE if a '!' is present.
                //   flag_plussign               TRUE if a '+' is present.
                //   flag_leftjustify            TRUE if a '-' is present or if the field width was negative.
                //   flag_zeropad                TRUE if the width began with 0.
                //   flag_long                   TRUE if the letter 'l' (ell) prefixed the conversion character.
                //   flag_longlong               TRUE if the letter 'll' (ell ell) prefixed the conversion character.
                //   flag_blanksign              TRUE if a ' ' is present.
                //   width                       The specified field width.  This is always non-negative.  Zero is the default.
                //   precision                   The specified precision.  The default is -1.
                //   type                        The class of the conversion.
                //   info                        Pointer to the appropriate info struct.
                char[] extra = null; // Malloced memory used by some conversion
                char[] out_; // Rendering buffer
                int outLength; // Size of the rendering buffer
                switch (type)
                {
                    case TYPE.POINTER:
                        flag_longlong = true;
                        flag_long = false;
                        // Fall through into the next case
                        goto case TYPE.RADIX;
                    case TYPE.ORDINAL:
                    case TYPE.RADIX:
                        if ((info.Flags & FLAG.SIGNED) != 0)
                        {
                            long v;
                            if (flag_longlong) v = __arg(ref argsIdx, args, (long)0);
                            else if (flag_long) v = __arg(ref argsIdx, args, (long)0);
                            else v = __arg(ref argsIdx, args, (int)0);
                            if (v < 0)
                            {
                                longvalue = (v == long.MinValue ? ((long)((ulong)1) << 63) : -v);
                                prefix = '-';
                            }
                            else
                            {
                                longvalue = v;
                                if (flag_plussign) prefix = '+';
                                else if (flag_blanksign) prefix = ' ';
                                else prefix = '\0';
                            }
                        }
                        else
                        {
                            if (flag_longlong) longvalue = __arg(ref argsIdx, args, (long)0);
                            else if (flag_long) longvalue = __arg(ref argsIdx, args, (long)0);
                            else longvalue = __arg(ref argsIdx, args, (long)0);
                            prefix = '\0';
                        }
                        if (longvalue == 0) flag_alternateform = false;
                        if (flag_zeropad && precision < width - (prefix != '\0' ? 1 : 0))
                            precision = width - ((prefix != '\0') ? 1 : 0);
                        if (precision < BUFSIZE - 10)
                        {
                            outLength = BUFSIZE;
                            out_ = buf;
                        }
                        else
                        {
                            outLength = precision + 10;
                            out_ = extra = new char[outLength];
                            if (out_ == null)
                            {
                                b.AllocFailed = true;
                                return;
                            }
                        }
                        //bufpt = buf.Length;
                        char[] bufOrd_ = null;
                        if (type == TYPE.ORDINAL)
                        {
                            int x = (int)(longvalue % 10);
                            if (x >= 4 || (longvalue / 10) % 10 == 1) x = 0;
                            bufOrd_ = new char[2];
                            bufOrd_[0] = _ord[x * 2];
                            bufOrd_[1] = _ord[x * 2 + 1];
                        }
                        {
                            //: register const char *cset = &_digits[info->Charset]; // Use registers for speed
                            //: register int base = info->Base;
                            //: do // Convert to ascii
                            //: {                                           
                            //:     *(--bufpt) = cset[longvalue % base];
                            //:     longvalue = longvalue / base;
                            //: } while(longvalue > 0);
                            char[] buf_;
                            switch (info.Base)
                            {
                                case 16: buf_ = longvalue.ToString("x").ToCharArray(); break;
                                case 8: buf_ = Convert.ToString((long)longvalue, 8).ToCharArray(); break;
                                default: buf_ = (flag_zeropad ? longvalue.ToString(new string('0', width - (prefix != '\0' ? 1 : 0))) : longvalue.ToString()).ToCharArray(); break;
                            }
                            bufpt = buf.Length - buf_.Length - (bufOrd_ == null ? 0 : 2);
                            Array.Copy(buf_, 0, buf, bufpt, buf_.Length);
                            if (bufOrd_ != null)
                            {
                                buf[buf.Length - 1] = bufOrd_[1];
                                buf[buf.Length - 2] = bufOrd_[0];
                            }
                        }
                        length = buf.Length - bufpt; //: (int)(&out_[outLength-1]-bufpt);
                        for (i = precision - length; i > 0; i--) buf[(--bufpt)] = '0'; // Zero pad
                        if (prefix != '\0') buf[--bufpt] = prefix; // Add sign
                        if (flag_alternateform && info.Prefix != 0)  // Add "0" or "0x"
                        {
                            char x;
                            int pre = info.Prefix;
                            for (; (x = _prefix[pre]) != 0; pre++) buf[--bufpt] = x;
                        }
                        length = buf.Length - bufpt; //: (int)(&out_[outLength-1]-bufpt);
                        break;
                    case TYPE.FLOAT:
                    case TYPE.EXP:
                    case TYPE.GENERIC:
                        realvalue = __arg(ref argsIdx, args, (double)0);
#if OMIT_FLOATING_POINT
                        length = 0;
#else
                        if (precision < 0) precision = 6; // Set default precision
                        if (realvalue < 0.0)
                        {
                            realvalue = -realvalue;
                            prefix = '-';
                        }
                        else
                        {
                            if (flag_plussign) prefix = '+';
                            else if (flag_blanksign) prefix = ' ';
                            else prefix = '\0';
                        }
                        if (type == TYPE.GENERIC && precision > 0) precision--;
#if false
                        // Rounding works like BSD when the constant 0.4999 is used.  Wierd!
                        for (i = precision, rounder = 0.4999; i > 0; i--, rounder *= 0.1) { }
#else
                        // It makes more sense to use 0.5
                        for (i = precision, rounder = 0.5; i > 0; i--, rounder *= 0.1) { }
#endif
                        if (type == TYPE.FLOAT) realvalue += rounder;
                        // Normalize realvalue to within 10.0 > realvalue >= 1.0
                        exp = 0;
#if WINDOWS_MOBILE
                        //+ Tryparse doesn't exist on Windows Moble and what will Tryparsing a double do?
                        if (double.IsNaN(realvalue))
#else
                        double d;
                        if (double.IsNaN(realvalue) || !(double.TryParse(Convert.ToString(realvalue), out d)))
#endif
                        {
                            buf[0] = 'N'; buf[1] = 'a'; buf[2] = 'N';
                            length = 3;
                            break;
                        }
                        if (realvalue > 0.0)
                        {
                            while (realvalue >= 1e32 && exp <= 350) { realvalue *= 1e-32; exp += 32; }
                            while (realvalue >= 1e8 && exp <= 350) { realvalue *= 1e-8; exp += 8; }
                            while (realvalue >= 10.0 && exp <= 350) { realvalue *= 0.1; exp++; }
                            while (realvalue < 1e-8) { realvalue *= 1e8; exp -= 8; }
                            while (realvalue < 1.0) { realvalue *= 10.0; exp--; }
                            if (exp > 350)
                            {
                                if (prefix == '-') { buf[0] = '-'; buf[1] = 'I'; buf[2] = 'n'; buf[3] = 'f'; bufpt = 4; }
                                else if (prefix == '+') { buf[0] = '+'; buf[1] = 'I'; buf[2] = 'n'; buf[3] = 'f'; bufpt = 4; }
                                else { buf[0] = 'I'; buf[1] = 'n'; buf[2] = 'f'; bufpt = 3; }
                                length = bufpt;
                                break;
                            }
                        }
                        bufpt = 0;
                        // If the field type is etGENERIC, then convert to either etEXP or etFLOAT, as appropriate.
                        if (type != TYPE.FLOAT)
                        {
                            realvalue += rounder;
                            if (realvalue >= 10.0) { realvalue *= 0.1; exp++; }
                        }
                        if (type == TYPE.GENERIC)
                        {
                            flag_rtz = !flag_alternateform;
                            if (exp < -4 || exp > precision) type = TYPE.EXP;
                            else { precision = precision - exp; type = TYPE.FLOAT; }
                        }
                        else
                            flag_rtz = flag_altform2;
                        e2 = (type == TYPE.EXP ? 0 : exp);
                        if (e2 + precision + width > BUFSIZE - 15)
                        {
                            buf = extra = new char[e2 + precision + width + 15];
                            if (buf == null)
                            {
                                b.AllocFailed = true;
                                return;
                            }
                        }
                        out_ = buf;
                        nsd = 16 + (flag_altform2 ? 10 : 0);
                        flag_dp = (precision > 0) | flag_alternateform | flag_altform2;
                        // The sign in front of the number
                        if (prefix != '\0') buf[bufpt++] = prefix;
                        // Digits prior to the decimal point
                        if (e2 < 0) buf[bufpt++] = '0';
                        else for (; e2 >= 0; e2--) buf[bufpt++] = (char)(GetDigit(ref realvalue, ref nsd) + '0');
                        // The decimal point
                        if (flag_dp) buf[bufpt++] = '.';
                        // "0" digits after the decimal point but before the first significant digit of the number
                        for (e2++; e2 < 0; precision--, e2++) { Debug.Assert(precision > 0); buf[bufpt++] = '0'; }
                        // Significant digits after the decimal point
                        while ((precision--) > 0) buf[bufpt++] = (char)(GetDigit(ref realvalue, ref nsd) + '0');
                        // Remove trailing zeros and the "." if no digits follow the "."
                        if (flag_rtz && flag_dp)
                        {
                            while (buf[bufpt - 1] == '0') buf[--bufpt] = '\0';
                            Debug.Assert(bufpt > 0);
                            if (buf[bufpt - 1] == '.')
                            {
                                if (flag_altform2) buf[(bufpt++)] = '0';
                                else buf[(--bufpt)] = '0';
                            }
                        }
                        // Add the "eNNN" suffix
                        if (type == TYPE.EXP)
                        {
                            buf[bufpt++] = _digits[info.Charset];
                            if (exp < 0) { buf[bufpt++] = '-'; exp = -exp; }
                            else buf[bufpt++] = '+';
                            if (exp >= 100) { buf[bufpt++] = (char)(exp / 100 + '0'); exp %= 100; } // 100's digit
                            buf[bufpt++] = (char)(exp / 10 + '0'); // 10's digit
                            buf[bufpt++] = (char)(exp % 10 + '0'); // 1's digit
                        }
                        //: bufpt = 0;

                        // The converted number is in buf[] and zero terminated. Output it. Note that the number is in the usual order, not reversed as with integer conversions.
                        length = bufpt; //: (int)(bufpt-buf);
                        bufpt = 0;

                        // Special case:  Add leading zeros if the flag_zeropad flag is set and we are not left justified
                        if (flag_zeropad && !flag_leftjustify && length < width)
                        {
                            int pad = width - length;
                            for (i = width; i >= pad; i--) buf[bufpt + i] = buf[bufpt + i - pad];
                            i = (prefix != '\0' ? 1 : 0);
                            while (pad-- != 0) buf[(bufpt++) + i] = '0';
                            length = width;
                            bufpt = 0;
                        }
#endif
                        break;
                    case TYPE.SIZE:
                        args[0] = b.Size; //: *(__arg(args,int*)) = b->Size;
                        length = width = 0;
                        break;
                    case TYPE.PERCENT:
                        buf[0] = '%';
                        bufpt = 0;
                        length = 1;
                        break;
                    case TYPE.CHARX:
                        c = __arg(ref argsIdx, args, (char)0);
                        buf[0] = (char)c;
                        if (precision >= 0)
                        {
                            for (i = 1; i < precision; i++) buf[i] = (char)c;
                            length = precision;
                        }
                        else length = 1;
                        bufpt = 0;
                        break;
                    case TYPE.STRING:
                    case TYPE.DYNSTRING:
                        bufpt = 0;
                        string bufStr = __arg(ref argsIdx, args, (string)null);
                        if (bufStr.Length > buf.Length) buf = new char[bufStr.Length];
                        bufStr.ToCharArray().CopyTo(buf, 0);
                        bufpt = bufStr.Length;
                        if (bufpt == 0) buf[0] = '\0';
                        else if (type == TYPE.DYNSTRING) extra = buf;
                        if (precision >= 0) for (length = 0; length < precision && length < bufStr.Length && buf[length] != 0; length++) { }
                        else length = bufpt;
                        bufpt = 0;
                        break;
                    case TYPE.SQLESCAPE:
                    case TYPE.SQLESCAPE2:
                    case TYPE.SQLESCAPE3:
                        {
                            char q = (type == TYPE.SQLESCAPE3 ? '"' : '\''); // Quote character
                            string escarg = __arg(ref argsIdx, args, (string)null) + '\0';
                            bool isnull = (escarg == string.Empty || escarg == "NULL\0");
                            if (isnull) escarg = (type == TYPE.SQLESCAPE2 ? "NULL\0" : "(NULL)\0");
                            int k = precision;
                            int j, n;
                            char ch;
                            for (i = n = 0; k != 0 && (ch = escarg[i]) != 0; i++, k--)
                                if (ch == q) n++;
                            bool needQuote = (!isnull && type == TYPE.SQLESCAPE2);
                            n += i + 1 + (needQuote ? 2 : 0);
                            if (n > BUFSIZE)
                            {
                                buf = extra = new char[n];
                                if (buf == null)
                                {
                                    b.AllocFailed = true;
                                    return;
                                }
                                bufpt = 0;
                            }
                            else
                                bufpt = 0;
                            j = 0;
                            if (needQuote) buf[bufpt + j++] = q;
                            k = i;
                            for (i = 0; i < k; i++)
                            {
                                buf[bufpt + j++] = ch = escarg[i];
                                if (ch == q) buf[bufpt + j++] = ch;
                            }
                            if (needQuote) buf[bufpt + j++] = q;
                            buf[bufpt + j] = '\0';
                            length = j;
                            // The precision in %q and %Q means how many input characters to consume, not the length of the output...
                            // if (precision>=0 && precision<length) length = precision;
                            break;
                        }
                    //case TYPE.TOKEN:
                    //    {
                    //        Token token = (args[argsIdx] is string ? new Token { z = __arg(ref argsIdx, args, (string)null); n = token.z.Length; } : __arg(args, (Token)null));
                    //        if (token != null) b.Append(token.z.ToString(), (int)token.n);
                    //        length = width = 0;
                    //        break;
                    //    }
                    //case TYPE.SRCLIST:
                    //    {
                    //        SrcList src = __arg(ref argsIdx, args, (SrcList)null);
                    //        int k = __arg(ref argsIdx, args, (int)0);
                    //        SrcList.SrcListItem item = src.a[k];
                    //        Debug.Assert(k >= 0 && k < src.nSrc);
                    //        if (item.DatabaseName != null)
                    //        {
                    //            b.Append(item.DatabaseName, -1);
                    //            b.Append(".", 1);
                    //        }
                    //        b.Append(item.zName, -1);
                    //        length = width = 0;
                    //        break;
                    //    }
                    default:
                        {
                            Debug.Assert(type == TYPE.INVALID);
                            return;
                        }
                }
                // The text of the conversion is pointed to by "bufpt" and is "length" characters long.  The field width is "width".  Do the output.
                if (!flag_leftjustify)
                {
                    int nspace = width - length;
                    if (nspace > 0) b.AppendSpace(nspace);
                }
                if (length > 0) b.Append(new string(buf, bufpt, length), length);
                if (flag_leftjustify)
                {
                    int nspace = width - length;
                    if (nspace > 0) b.AppendSpace(nspace);
                }
                extra = null;
            }
        }

        static TextBuilder _b = new TextBuilder(BUFSIZE);

        public static string _vmtagprintf(object tag, string fmt, params object[] args)
        {
            if (fmt == null) return null;
            if (args.Length == 0) return fmt;
            //string z;
            Debug.Assert(tag != null);
            TextBuilder.Init(_b, BUFSIZE, 0); //? tag.Limits[SQLITE_LIMIT_LENGTH]);
            _b.Tag = tag;
            _b.Text.Length = 0;
            vxprintf(_b, true, fmt, args);
            //? if (!_b.MallocFailed) tag.MallocFailed = true;
            return _b.ToString();
        }


        public static string _vmprintf(string fmt, params object[] args)
        {
            //: if (!RuntimeInitialize()) return null;
            //: TextBuilder b = new TextBuilder(BUFSIZE);
            TextBuilder.Init(_b, BUFSIZE, BUFSIZE);
            //: b.AllocType = 2;
            vxprintf(_b, false, fmt, args);
            return _b.ToString();
        }

        static public void __vsnprintf(StringBuilder buf, int bufLen, string fmt, params object[] args)
        {
            if (bufLen <= 0) return;
            //: TextBuilder b = new TextBuilder(BUFSIZE);
            TextBuilder.Init(_b, bufLen, 0);
            //: b.AllocType = 0;
            vxprintf(_b, false, fmt, args);
            buf.Length = 0;
            if (bufLen > 1 && bufLen <= _b.Text.Length)
                _b.Text.Length = bufLen - 1;
            buf.Append(_b.ToString());
            return;
        }

        //public static string __vsnprintf(ref string buf, int bufLen, string fmt, params object[] args)
        //{
        //    if (bufLen <= 0) return buf;
        //    TextBuilder b = new TextBuilder(BUFSIZE);
        //    TextBuilder.Init(b, bufLen, 0);
        //    //: b.AllocType = 0;
        //    vxprintf(b, false, fmt, args);
        //    string z = b.ToString();
        //    return (buf = z);
        //}

        #endregion

        //////////////////////
        // SNPRINTF
        #region SNPRINTF
        public static void __snprintf(StringBuilder buf, int bufLen, string fmt, params object[] args)
        {
            buf.EnsureCapacity(BUFSIZE);
            __vsnprintf(buf, bufLen, fmt, args);
            return;
        }
        #endregion

        //////////////////////
        // MPRINTF
        #region MPRINTF
        public static string _mprintf(string fmt, params object[] args)
        {
            //: if (!RuntimeInitialize()) return null;
            string z = _vmprintf(fmt, args);
            return z;
        }

        public static string _mtagprintf(object tag, string fmt, params object[] args)
        {
            string z = _vmtagprintf(tag, fmt, args);
            return z;
        }

        public static string _mtagappendf(object tag, string src, string fmt, params object[] args)
        {
            string z = _vmtagprintf(tag, fmt, args);
            _tagfree(tag, ref src);
            return z;
        }

        public static string _mtagassignf(object tag, ref string src, string fmt, params object[] args)
        {
            string z = _vmtagprintf(tag, fmt, args);
            _tagfree(tag, ref src);
            return z;
        }

        public static void _setstring(ref string z, object tag, string fmt, params object[] args)
        {
            string z2 = _vmtagprintf(tag, fmt, args);
            _tagfree(tag, ref z);
            z = z2;
        }

        #endregion

        #region Unfiled
        //        static void renderLogMsg(int iErrCode, string zFormat, params object[] ap)
        //        {
        //            //StrAccum acc;                          /* String accumulator */
        //            //char zMsg[SQLITE_PRINT_BUF_SIZE*3];    /* Complete log message */
        //            sqlite3StrAccumInit(_b, null, SQLITE_PRINT_BUF_SIZE * 3, 0);
        //            //acc.useMalloc = 0;
        //            sqlite3VXPrintf(_b, 0, zFormat, ap);
        //            SysEx_GlobalStatics.xLog(SysEx_GlobalStatics.pLogArg, iErrCode,
        //            sqlite3StrAccumFinish(_b));
        //        }

        //        static void sqlite3_log(int iErrCode, string zFormat, params va_list[] ap)
        //        {
        //            if (SysEx_GlobalStatics.xLog != null)
        //            {
        //                //va_list ap;                             /* Vararg list */
        //                lock (lock_va_list)
        //                {
        //                    va_start(ap, zFormat);
        //                    renderLogMsg(iErrCode, zFormat, ap);
        //                    va_end(ref ap);
        //                }
        //            }
        //        }

        //#if DEBUG || TRACE
        //        static void _dprintf(string zFormat, params va_list[] ap)
        //        {
        //            //va_list ap;
        //            lock (lock_va_list)
        //            {
        //                //StrAccum acc = new StrAccum( SQLITE_PRINT_BUF_SIZE );
        //                sqlite3StrAccumInit(_b, null, SQLITE_PRINT_BUF_SIZE, 0);
        //                //acc.useMalloc = 0;
        //                va_start(ap, zFormat);
        //                sqlite3VXPrintf(_b, 0, zFormat, ap);
        //                va_end(ref ap);
        //            }
        //            Console.Write(sqlite3StrAccumFinish(_b));
        //            //fflush(stdout);
        //        }
        //#endif
        #endregion

        public static bool _isxdigit(char p)
        {
            throw new NotImplementedException();
        }
    }
}