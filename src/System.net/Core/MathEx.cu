//util.c
#include "Core.cu.h"

namespace Core
{
	__device__ bool MathEx::Add(int64 *aRef, int64 b)
	{
		int64 a = *aRef;
		ASSERTCOVERAGE(a == 0); ASSERTCOVERAGE(a == 1);
		ASSERTCOVERAGE(b == -1 ); ASSERTCOVERAGE(b == 0);
		if (b >= 0)
		{
			ASSERTCOVERAGE(a > 0 && LARGEST_INT64 - a == b);
			ASSERTCOVERAGE(a > 0 && LARGEST_INT64 - a == b - 1);
			if (a > 0 && LARGEST_INT64 - a < b) return true;
			*aRef += b;
		}
		else
		{
			ASSERTCOVERAGE(a < 0 && -(a + LARGEST_INT64) == b + 1);
			ASSERTCOVERAGE(a < 0 && -(a + LARGEST_INT64) == b + 2);
			if (a < 0 && -(a + LARGEST_INT64) > b + 1) return true;
			*aRef += b;
		}
		return false; 
	}

	__device__ bool MathEx::Sub(int64 *aRef, int64 b)
	{
		ASSERTCOVERAGE(b == SMALLEST_INT64+1);
		if (b == SMALLEST_INT64)
		{
			int64 a = *aRef;
			ASSERTCOVERAGE(a == -1); ASSERTCOVERAGE(a == 0);
			if (a >= 0) return true;
			*aRef -= b;
			return false;
		}
		return Add(aRef, -b);
	}

#define TWOPOWER32 (((int64)1)<<32)
#define TWOPOWER31 (((int64)1)<<31)
	__device__ bool MathEx::Mul(int64 *aRef, int64 b)
	{
		int64 a = *aRef;
		int64 a1 = a / TWOPOWER32;
		int64 a0 = a % TWOPOWER32;
		int64 b1 = b / TWOPOWER32;
		int64 b0 = b % TWOPOWER32;
		if (a1*b1 != 0) return true;
		_assert(a1*b0 == 0 || a0*b1 == 0);
		int64 r = a1*a0 + a0*a1;
		ASSERTCOVERAGE(r == (-TWOPOWER31)-1 );
		ASSERTCOVERAGE(r == (-TWOPOWER31));
		ASSERTCOVERAGE(r == TWOPOWER31);
		ASSERTCOVERAGE(r == TWOPOWER31-1);
		if (r < (-TWOPOWER31) || r >= TWOPOWER31) return true;
		r *= TWOPOWER32;
		if (Add(&r, a0*b0)) return true;
		*aRef = r;
		return false;
	}

	//__device__ int MathEx::Abs(int x)
	//{
	//	if (x >= 0) return x;
	//	if (x == (int)0x80000000) return 0x7fffffff;
	//	return -x;
	//}
}
