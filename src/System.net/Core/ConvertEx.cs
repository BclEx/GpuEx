using System.Diagnostics;
namespace Core
{
    public enum TEXTENCODE : byte
    {
        UTF8 = 1,
        UTF16LE = 2,
        UTF16BE = 3,
        UTF16 = 4, // Use native byte order
        ANY = 5, // sqlite3_create_function only
        UTF16_ALIGNED = 8, // sqlite3_create_collation only
        //
        UTF16NATIVE = UTF16LE,
    }

    public static class ConvertEx
    {
        #region Varint

        // The variable-length integer encoding is as follows:
        //
        // KEY:
        //         A = 0xxxxxxx    7 bits of data and one flag bit
        //         B = 1xxxxxxx    7 bits of data and one flag bit
        //         C = xxxxxxxx    8 bits of data
        //  7 bits - A
        // 14 bits - BA
        // 21 bits - BBA
        // 28 bits - BBBA
        // 35 bits - BBBBA
        // 42 bits - BBBBBA
        // 49 bits - BBBBBBA
        // 56 bits - BBBBBBBA
        // 64 bits - BBBBBBBBC

        public static byte PutVarint(byte[] p, int v) { return PutVarint(p, 0, (ulong)v); }
        public static byte PutVarint(byte[] p, uint offset, int v) { return PutVarint(p, offset, (ulong)v); }
        public static byte PutVarint(byte[] p, ulong v) { return PutVarint(p, 0, (ulong)v); }
        public static byte PutVarint(byte[] p, uint offset, ulong v)
        {
            int i, j; byte n;
            if ((v & (((ulong)0xff000000) << 32)) != 0)
            {
                p[offset + 8] = (byte)v;
                v >>= 8;
                for (i = 7; i >= 0; i--)
                {
                    p[offset + i] = (byte)((v & 0x7f) | 0x80);
                    v >>= 7;
                }
                return 9;
            }
            n = 0;
            var b = new byte[10];
            do
            {
                b[n++] = (byte)((v & 0x7f) | 0x80);
                v >>= 7;
            } while (v != 0);
            b[0] &= 0x7f;
            Debug.Assert(n <= 9);
            for (i = 0, j = n - 1; j >= 0; j--, i++)
                p[offset + i] = b[j];
            return n;
        }

        public static byte PutVarint32(byte[] p, int v)
        {
            if ((v & ~0x7f) == 0) { p[0] = (byte)v; return 1; }
            if ((v & ~0x3fff) == 0) { p[0] = (byte)((v >> 7) | 0x80); p[1] = (byte)(v & 0x7f); return 2; }
            return PutVarint(p, 0, v);
        }
        public static byte PutVarint32(byte[] p, uint offset, int v)
        {
            if ((v & ~0x7f) == 0) { p[offset] = (byte)v; return 1; }
            if ((v & ~0x3fff) == 0) { p[offset] = (byte)((v >> 7) | 0x80); p[offset + 1] = (byte)(v & 0x7f); return 2; }
            return PutVarint(p, offset, v);
        }

        // Bitmasks used by sqlite3GetVarint().  These precomputed constants are defined here rather than simply putting the constant expressions
        // inline in order to work around bugs in the RVT compiler.
        //
        // SLOT_2_0     A mask for  (0x7f<<14) | 0x7f
        // SLOT_4_2_0   A mask for  (0x7f<<28) | SLOT_2_0
        private const uint SLOT_0_2_0 = 0x001fc07f;
        private const uint SLOT_4_2_0 = 0xf01fc07f;
        private const uint MAX_U32 = (uint)((((ulong)1) << 32) - 1);

        public static byte GetVarint(byte[] p, out int v) { v = p[0]; if (v < 0x80) return 1; ulong uv; var r = _getVarint(p, 0, out uv); v = (int)uv; return r; }
        public static byte GetVarint(byte[] p, out uint v) { v = p[0]; if (v < 0x80) return 1; ulong uv; var r = _getVarint(p, 0, out uv); v = (uint)uv; return r; }
        public static byte GetVarint(byte[] p, uint offset, out int v) { v = p[offset]; if (v < 0x80) return 1; ulong uv; var r = _getVarint(p, offset, out uv); v = (int)uv; return r; }
        public static byte GetVarint(byte[] p, uint offset, out uint v) { v = p[offset]; if (v < 0x80) return 1; ulong uv; var r = _getVarint(p, offset, out uv); v = (uint)uv; return r; }
        public static byte GetVarint(byte[] p, uint offset, out long v) { v = p[offset]; if (v < 0x80) return 1; ulong uv; var r = _getVarint(p, offset, out uv); v = (long)uv; return r; }
        public static byte GetVarint(byte[] p, uint offset, out ulong v) { v = p[offset]; if (v < 0x80) return 1; var r = _getVarint(p, offset, out v); return r; }
        private static byte _getVarint(byte[] p, uint offset, out ulong v)
        {
            uint a, b, s;
            a = p[offset + 0];
            // a: p0 (unmasked) 
            if ((a & 0x80) == 0)
            {
                v = a;
                return 1;
            }
            b = p[offset + 1];
            // b: p1 (unmasked)
            if (0 == (b & 0x80))
            {
                a &= 0x7f;
                a = a << 7;
                a |= b;
                v = a;
                return 2;
            }
            // Verify that constants are precomputed correctly
            Debug.Assert(SLOT_0_2_0 == ((0x7f << 14) | 0x7f));
            Debug.Assert(SLOT_4_2_0 == ((0xfU << 28) | (0x7f << 14) | 0x7f));
            a = a << 14;
            a |= p[offset + 2];
            // a: p0<<14 | p2 (unmasked)
            if (0 == (a & 0x80))
            {
                a &= SLOT_0_2_0;
                b &= 0x7f;
                b = b << 7;
                a |= b;
                v = a;
                return 3;
            }
            // CSE1 from below
            a &= SLOT_0_2_0;
            b = b << 14;
            b |= p[offset + 3];
            // b: p1<<14 | p3 (unmasked)
            if (0 == (b & 0x80))
            {
                b &= SLOT_0_2_0;
                // moved CSE1 up
                // a &= (0x7f<<14)|0x7f;
                a = a << 7;
                a |= b;
                v = a;
                return 4;
            }
            // a: p0<<14 | p2 (masked)
            // b: p1<<14 | p3 (unmasked)
            // 1:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            // moved CSE1 up
            // a &= (0x7f<<14)|0x7f;
            b &= SLOT_0_2_0;
            s = a;
            // s: p0<<14 | p2 (masked)
            a = a << 14;
            a |= p[offset + 4];
            // a: p0<<28 | p2<<14 | p4 (unmasked)
            if (0 == (a & 0x80))
            {
                b = b << 7;
                a |= b;
                s = s >> 18;
                v = ((ulong)s) << 32 | a;
                return 5;
            }
            // 2:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            s = s << 7;
            s |= b;
            // s: p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            b = b << 14;
            b |= p[offset + 5];
            // b: p1<<28 | p3<<14 | p5 (unmasked) 
            if (0 == (b & 0x80))
            {
                a &= SLOT_0_2_0;
                a = a << 7;
                a |= b;
                s = s >> 18;
                v = ((ulong)s) << 32 | a;
                return 6;
            }
            a = a << 14;
            a |= p[offset + 6];
            // a: p2<<28 | p4<<14 | p6 (unmasked)
            if (0 == (a & 0x80))
            {
                a &= SLOT_4_2_0;
                b &= SLOT_0_2_0;
                b = b << 7;
                a |= b;
                s = s >> 11;
                v = ((ulong)s) << 32 | a;
                return 7;
            }
            // CSE2 from below
            a &= SLOT_0_2_0;
            //p++;
            b = b << 14;
            b |= p[offset + 7];
            // b: p3<<28 | p5<<14 | p7 (unmasked)
            if (0 == (b & 0x80))
            {
                b &= SLOT_4_2_0;
                // moved CSE2 up
                a = a << 7;
                a |= b;
                s = s >> 4;
                v = ((ulong)s) << 32 | a;
                return 8;
            }
            a = a << 15;
            a |= p[offset + 8];
            // a: p4<<29 | p6<<15 | p8 (unmasked) 
            // moved CSE2 up
            b &= SLOT_0_2_0;
            b = b << 8;
            a |= b;
            s = s << 4;
            b = p[offset + 4];
            b &= 0x7f;
            b = b >> 3;
            s |= b;
            v = ((ulong)s) << 32 | a;
            return 9;
        }

        public static byte GetVarint32(byte[] p, out int v) { v = p[0]; if (v < 0x80) return 1; uint uv; var r = _getVarint32(p, 0, out uv); v = (int)uv; return r; }
        public static byte GetVarint32(byte[] p, out uint v) { v = p[0]; if (v < 0x80) return 1; return _getVarint32(p, 0, out v); }
        public static byte GetVarint32(byte[] p, uint offset, out int v) { v = p[offset]; if (v < 0x80) return 1; uint uv; var r = _getVarint32(p, offset, out uv); v = (int)uv; return r; }
        public static byte GetVarint32(byte[] p, uint offset, out uint v) { v = p[offset]; if (v < 0x80) return 1; return _getVarint32(p, offset, out v); }
        private static byte _getVarint32(byte[] p, uint offset, out uint v)
        {
            uint a, b;
            // The 1-byte case.  Overwhelmingly the most common.  Handled inline  by the getVarin32() macro
            a = p[offset + 0];
            // a: p0 (unmasked)
            // The 2-byte case
            b = (offset + 1 < p.Length ? p[offset + 1] : (uint)0);
            // b: p1 (unmasked)
            if (0 == (b & 0x80))
            {
                // Values between 128 and 16383
                a &= 0x7f;
                a = a << 7;
                v = a | b;
                return 2;
            }
            // The 3-byte case
            a = a << 14;
            a |= (offset + 2 < p.Length ? p[offset + 2] : (uint)0);
            // a: p0<<14 | p2 (unmasked)
            if (0 == (a & 0x80))
            {
                // Values between 16384 and 2097151
                a &= (0x7f << 14) | (0x7f);
                b &= 0x7f;
                b = b << 7;
                v = a | b;
                return 3;
            }
            // A 32-bit varint is used to store size information in btrees. Objects are rarely larger than 2MiB limit of a 3-byte varint.
            // A 3-byte varint is sufficient, for example, to record the size of a 1048569-byte BLOB or string.
            // We only unroll the first 1-, 2-, and 3- byte cases.  The very rare larger cases can be handled by the slower 64-bit varint routine.
            {
                ulong ulong_v = 0;
                byte n = _getVarint(p, offset, out ulong_v);
                Debug.Assert(n > 3 && n <= 9);
                v = ((ulong_v & MAX_U32) != ulong_v ? 0xffffffff : (uint)ulong_v);
                return n;
            }
        }
        public static byte GetVarint32(string p, uint offset, out int v)
        {
            v = p[(int)offset]; if (v < 0x80) return 1;
            var a = new byte[4];
            a[0] = (byte)p[(int)offset + 0];
            a[1] = (byte)p[(int)offset + 1];
            a[2] = (byte)p[(int)offset + 2];
            a[3] = (byte)p[(int)offset + 3];
            uint uv; var r = _getVarint32(a, 0, out uv); v = (int)uv; return r;
        }
        public static byte GetVarint32(string p, uint offset, out uint v)
        {
            v = p[(int)offset]; if (v < 0x80) return 1;
            var a = new byte[4];
            a[0] = (byte)p[(int)offset + 0];
            a[1] = (byte)p[(int)offset + 1];
            a[2] = (byte)p[(int)offset + 2];
            a[3] = (byte)p[(int)offset + 3];
            return _getVarint32(a, 0, out v);
        }

        public static byte GetVarintLength(ulong v)
        {
            byte i = 0;
            do { i++; v >>= 7; }
            while (v != 0 && C._ALWAYS(i < 9));
            return i;
        }

        #endregion

        #region Get/Put

        public static uint Get4(byte[] p) { return (uint)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]); }
        public static uint Get4(byte[] p, int offset) { return (offset + 3 > p.Length) ? 0 : (uint)((p[0 + offset] << 24) | (p[1 + offset] << 16) | (p[2 + offset] << 8) | p[3 + offset]); }
        public static uint Get4(byte[] p, uint offset) { return (offset + 3 > p.Length) ? 0 : (uint)((p[0 + offset] << 24) | (p[1 + offset] << 16) | (p[2 + offset] << 8) | p[3 + offset]); }

        public static void Put4(byte[] p, int v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, int offset, int v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, int offset, uint v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint offset, int v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint offset, uint v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, long v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, ulong v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, uint offset, long v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, uint offset, ulong v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        //public static void Put4(string p, int offset, int v)
        //{
        //    var a = new byte[4];
        //    a[0] = (byte)p[offset + 0];
        //    a[1] = (byte)p[offset + 1];
        //    a[2] = (byte)p[offset + 2];
        //    a[3] = (byte)p[offset + 3];
        //    Put4(a, 0, v);
        //}
        //public static void Put4(string p, int offset, uint v)
        //{
        //    var a = new byte[4];
        //    a[0] = (byte)p[offset + 0];
        //    a[1] = (byte)p[offset + 1];
        //    a[2] = (byte)p[offset + 2];
        //    a[3] = (byte)p[offset + 3];
        //    Put4(a, 0, v);
        //}

        public static ushort Get2(byte[] p) { return (ushort)(p[0] << 8 | p[1]); }
        public static ushort Get2(byte[] p, int offset) { return (ushort)(p[offset + 0] << 8 | p[offset + 1]); }
        public static ushort Get2(byte[] p, uint offset) { return (ushort)(p[offset + 0] << 8 | p[offset + 1]); }
        public static ushort Get2nz(byte[] p, int offset) { return (ushort)(((((int)Get2(p, offset)) - 1) & 0xffff) + 1); }
        public static ushort Get2nz(byte[] p, uint offset) { return (ushort)(((((int)Get2(p, offset)) - 1) & 0xffff) + 1); }

        public static void Put2(byte[] p, int v)
        {
            p[0] = (byte)(v >> 8);
            p[1] = (byte)v;
        }
        public static void Put2(byte[] p, uint v)
        {
            p[0] = (byte)(v >> 8);
            p[1] = (byte)v;
        }
        public static void Put2(byte[] p, int offset, int v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, int offset, uint v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, uint offset, int v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, uint offset, uint v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, int offset, ushort v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, int offset, short v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, uint offset, ushort v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }
        public static void Put2(byte[] p, uint offset, short v)
        {
            p[offset + 0] = (byte)(v >> 8);
            p[offset + 1] = (byte)v;
        }

        #endregion

        #region AtoX

        public static bool Atof(string z, ref double out_, int length, TEXTENCODE encode)
        {
#if !OMIT_FLOATING_POINT
            out_ = 0.0; // Default return value, in case of an error
            if (string.IsNullOrEmpty(z))
                return false;

            // getsize
            int zIdx = 0;
            int incr = (encode == TEXTENCODE.UTF8 ? 1 : 2);
            if (encode == TEXTENCODE.UTF16BE) zIdx++;

            // skip leading spaces            
            while (zIdx < length && char.IsWhiteSpace(z[zIdx])) zIdx++;
            if (zIdx >= length) return false;

            // get sign of significand
            int sign = 1; // sign of significand
            if (z[zIdx] == '-') { sign = -1; zIdx += incr; }
            else if (z[zIdx] == '+') zIdx += incr;

            // sign * significand * (10 ^ (esign * exponent))
            long s = 0;      // significand
            int d = 0;      // adjust exponent for shifting decimal point
            int esign = 1;  // sign of exponent
            int e = 0;      // exponent
            bool nonNum = true;  // True exponent is either not used or is well-formed
            int digits = 0;

            // skip leading zeroes
            while (zIdx < z.Length && z[zIdx] == '0') { zIdx += incr; digits++; }

            // copy max significant digits to significand
            while (zIdx < length && char.IsDigit(z[zIdx]) && s < ((long.MaxValue - 9) / 10)) { s = s * 10 + (z[zIdx] - '0'); zIdx += incr; digits++; }
            while (zIdx < length && char.IsDigit(z[zIdx])) { zIdx += incr; digits++; d++; }
            if (zIdx >= length) goto do_atof_calc;

            // if decimal point is present
            if (z[zIdx] == '.')
            {
                zIdx += incr;
                // copy digits from after decimal to significand (decrease exponent by d to shift decimal right)
                while (zIdx < length && char.IsDigit(z[zIdx]) && s < ((long.MaxValue - 9) / 10)) { s = s * 10 + (z[zIdx] - '0'); zIdx += incr; digits++; d--; }
                while (zIdx < length && char.IsDigit(z[zIdx])) { zIdx += incr; digits++; } // skip non-significant digits
            }
            if (zIdx >= length) goto do_atof_calc;

            // if exponent is present
            if (z[zIdx] == 'e' || z[zIdx] == 'E')
            {
                zIdx += incr;
                nonNum = false;
                if (zIdx >= length) goto do_atof_calc;
                // get sign of exponent
                if (z[zIdx] == '-') { esign = -1; zIdx += incr; }
                else if (z[zIdx] == '+') zIdx += incr;
                // copy digits to exponent
                while (zIdx < length && char.IsDigit(z[zIdx])) { e = e * 10 + (z[zIdx] - '0'); zIdx += incr; nonNum = true; }
            }

            // skip trailing spaces
            if (digits != 0 && nonNum) while (zIdx < length && char.IsWhiteSpace(z[zIdx])) zIdx += incr;

        do_atof_calc:

            // adjust exponent by d, and update sign
            e = (e * esign) + d;
            if (e < 0) { esign = -1; e *= -1; }
            else esign = 1;

            // if !significand 
            double result = 0.0;
            if (s == 0)
                result = (sign < 0 && digits != 0 ? -0.0 : 0.0); // In the IEEE 754 standard, zero is signed. Add the sign if we've seen at least one digit
            else
            {
                // attempt to reduce exponent
                if (esign > 0) while (s < (long.MaxValue / 10) && e > 0) { e--; s *= 10; }
                else while ((s % 10) == 0 && e > 0) { e--; s /= 10; }

                // adjust the sign of significand
                s = (sign < 0 ? -s : s);

                // if exponent, scale significand as appropriate and store in result.
                if (e != 0)
                {
                    double scale = 1.0;
                    // attempt to handle extremely small/large numbers better
                    if (e > 307 && e < 342)
                    {
                        while ((e % 308) != 0) { scale *= 1.0e+1; e -= 1; }
                        if (esign < 0) { result = s / scale; result /= 1.0e+308; }
                        else { result = s * scale; result *= 1.0e+308; }
                    }
                    else if (e >= 342)
                        result = (esign < 0 ? 0.0 * s : 1e308 * 1e308 * s); // Infinity
                    else
                    {
                        // 1.0e+22 is the largest power of 10 than can be represented exactly.
                        while ((e % 22) != 0) { scale *= 1.0e+1; e -= 1; }
                        while (e > 0) { scale *= 1.0e+22; e -= 22; }
                        result = (esign < 0 ? s / scale : s * scale);
                    }
                }
                else
                    result = (double)s;
            }


            out_ = result; // store the result
            return (zIdx >= length && digits > 0 && nonNum); // return true if number and no extra non-whitespace chracters after
#else
            return !Atoi64(z, out_, length, encode);
#endif
        }

        static int Compare2pow63(string z, int incr)
        {
            string pow63 = "922337203685477580"; // 012345678901234567
            int c = 0;
            for (int i = 0; c == 0 && i < 18; i++)
                c = (z[i * incr] - pow63[i]) * 10;
            if (c == 0)
            {
                c = z[18 * incr] - '8';
                C.ASSERTCOVERAGE(c == -1);
                C.ASSERTCOVERAGE(c == 0);
                C.ASSERTCOVERAGE(c == +1);
            }
            return c;
        }

        public static int Atoi64(string z, out long out_, int length, TEXTENCODE encode)
        {
            if (z == null)
            {
                out_ = 0;
                return 1;
            }

            // get size
            int zIdx = 0;//  string zStart;
            int incr = (encode == TEXTENCODE.UTF8 ? 1 : 2);
            if (encode == TEXTENCODE.UTF16BE) zIdx++;

            // skip leading spaces            
            while (zIdx < length && char.IsWhiteSpace(z[zIdx])) zIdx += incr;

            // get sign of significand
            int neg = 0; // assume positive
            if (zIdx < length)
            {
                if (z[zIdx] == '-') { neg = 1; zIdx += incr; }
                else if (z[zIdx] == '+') zIdx += incr;
            }

            if (length > z.Length) length = z.Length;

            // skip leading zeros
            while (zIdx < length - 1 && z[zIdx] == '0') zIdx += incr;

            // Skip leading zeros.
            ulong u = 0;
            int c = 0;
            int i; for (i = zIdx; i < length && (c = z[i]) >= '0' && c <= '9'; i += incr) u = u * 10 + (ulong)(c - '0');

            if (u > long.MaxValue) out_ = long.MinValue;
            else out_ = (neg != 0 ? -(long)u : (long)u);

            C.ASSERTCOVERAGE(i - zIdx == 18);
            C.ASSERTCOVERAGE(i - zIdx == 19);
            C.ASSERTCOVERAGE(i - zIdx == 20);
            if ((c != 0 && i < length) || i == zIdx || i - zIdx > 19 * incr) return 1; // zNum is empty or contains non-numeric text or is longer than 19 digits (thus guaranteeing that it is too large)
            else if (i - zIdx < 19 * incr) { Debug.Assert(u <= long.MaxValue); return 0; } // Less than 19 digits, so we know that it fits in 64 bits
            else
            {
                c = Compare2pow63(z.Substring(zIdx), incr); // zNum is a 19-digit numbers.  Compare it against 9223372036854775808.
                if (c < 0) { Debug.Assert(u <= long.MaxValue); return 0; } // zNum is less than 9223372036854775808 so it fits
                else if (c > 0) return 1; // zNum is greater than 9223372036854775808 so it overflows 
                else { Debug.Assert(u - 1 == long.MaxValue); Debug.Assert(out_ == long.MinValue); return neg != 0 ? 0 : 2; } // zNum is exactly 9223372036854775808.  Fits if negative.  The special case 2 overflow if positive
            }
        }

        public static bool Atoi(string z, ref int out_) { return Atoi(z, 0, ref out_); }
        public static bool Atoi(string z, int zIdx, ref int out_)
        {
            int neg = 0;
            if (z[zIdx] == '-') { neg = 1; zIdx++; }
            else if (z[zIdx] == '+') zIdx++;
            while (zIdx < z.Length && z[zIdx] == '0') zIdx++;
            long v = 0;
            int i, c;
            for (i = 0; i < 11 && i + zIdx < z.Length && (c = z[zIdx + i] - '0') >= 0 && c <= 9; i++) { v = v * 10 + c; }
            // The longest decimal representation of a 32 bit integer is 10 digits:
            //             1234567890
            //     2^31 -> 2147483648
            C.ASSERTCOVERAGE(i == 10);
            if (i > 10) return false;
            C.ASSERTCOVERAGE(v - neg == 2147483647);
            if (v - neg > 2147483647) return false;
            out_ = (int)(neg != 0 ? -v : v);
            return true;
        }

        public static int Atoi(string z)
        {
            int x = 0;
            if (!string.IsNullOrEmpty(z))
                Atoi(z, ref x);
            return x;
        }

        #endregion

        #region From: Pragma_c
        
        // 123456789 123456789
        static readonly string _safetyLevelText = "onoffalseyestruefull";
        static readonly int[] _safetyLevelOffset = new int[] { 0, 1, 2, 4, 9, 12, 16 };
        static readonly int[] _safetyLevelLength = new int[] { 2, 2, 3, 5, 3, 4, 4 };
        static readonly byte[] _safetyLevelValue = new byte[] { 1, 0, 0, 0, 1, 1, 2 };

        public static byte GetSafetyLevel(string z, int omitFull, byte dflt)
        {
            if (char.IsDigit(z[0]))
                return (byte)ConvertEx.Atoi(z);
            int n = z.Length;
            for (int i = 0; i < _safetyLevelLength.Length - omitFull; i++)
                if (_safetyLevelLength[i] == n && string.CompareOrdinal(_safetyLevelText.Substring(_safetyLevelOffset[i]), 0, z, 0, n) == 0)
                    return _safetyLevelValue[i];
            return dflt;
        }

        public static bool GetBoolean(string z, byte dflt)
        {
            return (GetSafetyLevel(z, 1, dflt) != 0);
        }

        #endregion
    }
}