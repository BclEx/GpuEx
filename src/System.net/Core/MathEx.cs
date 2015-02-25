using System;
using System.Diagnostics;
namespace Core
{
    public class MathEx
    {
        public static bool Add(ref long aRef, long b)
        {
            long a = aRef;
            C.ASSERTCOVERAGE(a == 0);
            C.ASSERTCOVERAGE(a == 1);
            C.ASSERTCOVERAGE(b == -1);
            C.ASSERTCOVERAGE(b == 0);
            if (b >= 0)
            {
                C.ASSERTCOVERAGE(a > 0 && long.MaxValue - a == b);
                C.ASSERTCOVERAGE(a > 0 && long.MaxValue - a == b - 1);
                if (a > 0 && long.MaxValue - a < b) return true;
                aRef += b;
            }
            else
            {
                C.ASSERTCOVERAGE(a < 0 && -(a + long.MaxValue) == b + 1);
                C.ASSERTCOVERAGE(a < 0 && -(a + long.MaxValue) == b + 2);
                if (a < 0 && -(a + long.MaxValue) > b + 1) return true;
                aRef += b;
            }
            return false;
        }
        public static bool Sub(ref long aRef, long b)
        {
            C.ASSERTCOVERAGE(b == long.MinValue + 1);
            if (b == long.MinValue)
            {
                long a = aRef;
                C.ASSERTCOVERAGE(a == -1);
                C.ASSERTCOVERAGE(a == 0);
                if (a >= 0) return true;
                aRef -= b;
                return false;
            }
            return Add(ref aRef, -b);
        }

        const long TWOPOWER32 = (((long)1) << 32);
        const long TWOPOWER31 = (((long)1) << 31);
        public static bool Mul(ref long aRef, long b)
        {
            long a = aRef;
            long a1 = a / TWOPOWER32;
            long a0 = a % TWOPOWER32;
            long b1 = b / TWOPOWER32;
            long b0 = b % TWOPOWER32;
            if (a1 * b1 != 0) return true;
            Debug.Assert(a1 * b0 == 0 || a0 * b1 == 0);
            long r = a1 * b0 + a0 * b1;
            C.ASSERTCOVERAGE(r == (-TWOPOWER31) - 1);
            C.ASSERTCOVERAGE(r == (-TWOPOWER31));
            C.ASSERTCOVERAGE(r == TWOPOWER31);
            C.ASSERTCOVERAGE(r == TWOPOWER31 - 1);
            if (r < (-TWOPOWER31) || r >= TWOPOWER31) return true;
            r *= TWOPOWER32;
            if (Add(ref r, a0 * b0)) return true;
            aRef = r;
            return false;
        }

        public static int Abs(int x)
        {
            if (x >= 0) return x;
            if (x == (int)0x8000000) return 0x7fffffff;
            return -x;
        }
    }
}