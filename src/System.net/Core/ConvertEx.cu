// util.c
#include "Core.cu.h"

namespace Core
{

#pragma region Varint

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

#define SLOT_2_0     0x001fc07f
#define SLOT_4_2_0   0xf01fc07f

	__device__ int ConvertEx::PutVarint(unsigned char *p, uint64 v)
	{
		int i, j, n;
		if (v & (((uint64)0xff000000) << 32))
		{
			p[8] = (uint8)v;
			v >>= 8;
			for (i = 7; i >= 0; i--)
			{
				p[i] = (uint8)((v & 0x7f) | 0x80);
				v >>= 7;
			}
			return 9;
		}    
		n = 0;
		uint8 b[10];
		do
		{
			b[n++] = (uint8)((v & 0x7f) | 0x80);
			v >>= 7;
		} while (v != 0);
		b[0] &= 0x7f;
		_assert(n <= 9);
		for (i = 0, j = n - 1; j >= 0; j--, i++)
			p[i] = b[j];
		return n;
	}

	__device__ int ConvertEx::PutVarint32(unsigned char *p, uint32 v)
	{
		if ((v & ~0x3fff) == 0)
		{
			p[0] = (uint8)((v>>7) | 0x80);
			p[1] = (uint8)(v & 0x7f);
			return 2;
		}
		return PutVarint(p, v);
	}

	__device__ uint8 ConvertEx::GetVarint(const unsigned char *p, uint64 *v)
	{
		uint32 a, b, s;
		a = *p;
		// a: p0 (unmasked)
		if (!(a & 0x80))
		{
			*v = a;
			return 1;
		}
		p++;
		b = *p;
		// b: p1 (unmasked)
		if (!(b & 0x80))
		{
			a &= 0x7f;
			a = a << 7;
			a |= b;
			*v = a;
			return 2;
		}
		// Verify that constants are precomputed correctly
		_assert(SLOT_2_0 == ((0x7f << 14) | 0x7f));
		_assert(SLOT_4_2_0 == ((0xfU << 28) | (0x7f << 14) | 0x7f));
		p++;
		a = a << 14;
		a |= *p;
		// a: p0<<14 | p2 (unmasked)
		if (!(a & 0x80))
		{
			a &= SLOT_2_0;
			b &= 0x7f;
			b = b << 7;
			a |= b;
			*v = a;
			return 3;
		}
		// CSE1 from below
		a &= SLOT_2_0;
		p++;
		b = b << 14;
		b |= *p;
		// b: p1<<14 | p3 (unmasked)
		if (!(b & 0x80))
		{
			b &= SLOT_2_0;
			// moved CSE1 up
			// a &= (0x7f<<14)|(0x7f);
			a = a << 7;
			a |= b;
			*v = a;
			return 4;
		}
		// a: p0<<14 | p2 (masked)
		// b: p1<<14 | p3 (unmasked)
		// 1:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
		// moved CSE1 up
		// a &= (0x7f<<14)|(0x7f);
		b &= SLOT_2_0;
		s = a;
		// s: p0<<14 | p2 (masked)
		p++;
		a = a << 14;
		a |= *p;
		// a: p0<<28 | p2<<14 | p4 (unmasked)
		if (!(a & 0x80))
		{
			// we can skip these cause they were (effectively) done above in calc'ing s
			// a &= (0x7f<<28)|(0x7f<<14)|0x7f;
			// b &= (0x7f<<14)|0x7f;
			b = b << 7;
			a |= b;
			s = s >> 18;
			*v = ((uint64)s) << 32 | a;
			return 5;
		}
		// 2:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
		s = s << 7;
		s |= b;
		// s: p0<<21 | p1<<14 | p2<<7 | p3 (masked)
		p++;
		b = b << 14;
		b |= *p;
		/* b: p1<<28 | p3<<14 | p5 (unmasked) */
		if (!(b & 0x80))
		{
			// we can skip this cause it was (effectively) done above in calc'ing s
			// b &= (0x7f<<28)|(0x7f<<14)|0x7f;
			a &= SLOT_2_0;
			a = a << 7;
			a |= b;
			s = s >> 18;
			*v = ((uint64)s) << 32 | a;
			return 6;
		}
		p++;
		a = a << 14;
		a |= *p;
		// a: p2<<28 | p4<<14 | p6 (unmasked)
		if (!(a & 0x80))
		{
			a &= SLOT_4_2_0;
			b &= SLOT_2_0;
			b = b << 7;
			a |= b;
			s = s>>11;
			*v = ((uint64)s) << 32 | a;
			return 7;
		}
		// CSE2 from below
		a &= SLOT_2_0;
		p++;
		b = b << 14;
		b |= *p;
		// b: p3<<28 | p5<<14 | p7 (unmasked)
		if (!(b & 0x80))
		{
			b &= SLOT_4_2_0;
			// moved CSE2 up
			// a &= (0x7f<<14)|0x7f;
			a = a << 7;
			a |= b;
			s = s >> 4;
			*v = ((uint64)s) << 32 | a;
			return 8;
		}
		p++;
		a = a << 15;
		a |= *p;
		// a: p4<<29 | p6<<15 | p8 (unmasked)
		// moved CSE2 up
		// a &= (0x7f<<29)|(0x7f<<15)|(0xff);
		b &= SLOT_2_0;
		b = b << 8;
		a |= b;
		s = s << 4;
		b = p[-4];
		b &= 0x7f;
		b = b >> 3;
		s |= b;
		*v = ((uint64)s) << 32 | a;
		return 9;
	}

	__device__ uint8 ConvertEx::GetVarint32(const unsigned char *p, uint32 *v)
	{
		uint32 a, b;
		// The 1-byte case.  Overwhelmingly the most common.  Handled inline by the getVarin32() macro
		a = *p;
		// a: p0 (unmasked)
		// The 2-byte case
		p++;
		b = *p;
		// b: p1 (unmasked)
		if (!(b & 0x80))
		{
			// Values between 128 and 16383
			a &= 0x7f;
			a = a << 7;
			*v = a | b;
			return 2;
		}
		// The 3-byte case
		p++;
		a = a << 14;
		a |= *p;
		// a: p0<<14 | p2 (unmasked)
		if (!(a & 0x80))
		{
			// Values between 16384 and 2097151
			a &= (0x7f << 14) | 0x7f;
			b &= 0x7f;
			b = b << 7;
			*v = a | b;
			return 3;
		}

		// A 32-bit varint is used to store size information in btrees. Objects are rarely larger than 2MiB limit of a 3-byte varint.
		// A 3-byte varint is sufficient, for example, to record the size of a 1048569-byte BLOB or string.
		// We only unroll the first 1-, 2-, and 3- byte cases.  The very rare larger cases can be handled by the slower 64-bit varint routine.
#if 1
		{
			p -= 2;
			uint64 v64;
			uint8 n = GetVarint(p, &v64);
			_assert(n > 3 && n <= 9);
			*v = ((v64 & MAX_TYPE(uint32)) != v64 ? 0xffffffff : (uint32)v64);
			return n;
		}

#else
		// For following code (kept for historical record only) shows an unrolling for the 3- and 4-byte varint cases.  This code is
		// slightly faster, but it is also larger and much harder to test.
		p++;
		b = b << 14;
		b |= *p;
		// b: p1<<14 | p3 (unmasked)
		if (!(b & 0x80))
		{
			// Values between 2097152 and 268435455
			b &= (0x7f << 14) | 0x7f;
			a &= (0x7f << 14) | 0x7f;
			a = a << 7;
			*v = a | b;
			return 4;
		}
		p++;
		a = a << 14;
		a |= *p;
		// a: p0<<28 | p2<<14 | p4 (unmasked)
		if (!(a & 0x80))
		{
			// Values  between 268435456 and 34359738367
			a &= SLOT_4_2_0;
			b &= SLOT_4_2_0;
			b = b << 7;
			*v = a | b;
			return 5;
		}
		// We can only reach this point when reading a corrupt database file.  In that case we are not in any hurry.  Use the (relatively
		// slow) general-purpose sqlite3GetVarint() routine to extract the value.
		{
			p -= 4;
			uint64 v64;
			uint8 n = GetVarint(p, &v64);
			assert(n > 5 && n <= 9);
			*v = (uint32)v64;
			return n;
		}
#endif
	}

	__device__ int ConvertEx::GetVarintLength(uint64 v)
	{
		int i = 0;
		do { i++; v >>= 7; }
		while (v != 0 && _ALWAYS(i < 9));
		return i;
	}

#pragma endregion

#pragma region Get/Put
	// USE C#
#pragma endregion

#pragma region AtoX

	__device__ bool ConvertEx::Atof(const char *z, double *out, int length, TEXTENCODE encode)
	{
#ifndef OMIT_FLOATING_POINT
		_assert(encode == TEXTENCODE_UTF8 || encode == TEXTENCODE_UTF16LE || encode == TEXTENCODE_UTF16BE);
		*out = 0.0; // Default return value, in case of an error
		const char *end = z + length;

		// get size
		int incr;
		bool nonNum = false;
		if (encode == TEXTENCODE_UTF8)
			incr = 1;
		else
		{
			_assert(TEXTENCODE_UTF16LE == 2 && TEXTENCODE_UTF16BE == 3);
			incr = 2;
			int i; for (i = 3 - encode; i < length && z[i] == 0; i += 2) { }
			nonNum = (i < length);
			end = z + i + encode - 3;
			z += (encode & 1);
		}

		// skip leading spaces
		while (z < end && _isspace(*z)) z += incr;
		if (z >= end) return false;

		// get sign of significand
		int sign = 1; // sign of significand
		if (*z == '-') { sign = -1; z += incr; }
		else if (*z == '+') z += incr;

		// sign * significand * (10 ^ (esign * exponent))
		int digits = 0; 
		bool eValid = true;  // True exponent is either not used or is well-formed
		int64 s = 0;   // significand
		int esign = 1; // sign of exponent
		int e = 0; // exponent
		int d = 0; // adjust exponent for shifting decimal point

		// skip leading zeroes
		while (z < end && z[0] == '0') z += incr, digits++;

		// copy max significant digits to significand
		while (z < end && _isdigit(*z) && s < ((LARGEST_INT64 - 9) / 10)) { s = s * 10 + (*z - '0'); z += incr, digits++; }
		while (z < end && _isdigit(*z)) z += incr, digits++, d++; // skip non-significant significand digits (increase exponent by d to shift decimal left)
		if (z >= end) goto do_atof_calc;

		// if decimal point is present
		if (*z == '.')
		{
			z += incr;
			// copy digits from after decimal to significand (decrease exponent by d to shift decimal right)
			while (z < end && _isdigit(*z) && s < ((LARGEST_INT64 - 9) / 10)) { s = s * 10 + (*z - '0'); z += incr, digits++, d--; }
			while (z < end && _isdigit(*z)) z += incr, digits++; // skip non-significant digits
		}
		if (z >= end) goto do_atof_calc;

		// if exponent is present
		if (*z == 'e' || *z == 'E')
		{
			z += incr;
			eValid = false;
			if (z >= end) goto do_atof_calc;
			// get sign of exponent
			if (*z == '-') { esign = -1; z += incr; }
			else if (*z == '+') z += incr;
			// copy digits to exponent
			while (z < end && _isdigit(*z)) { e = (e < 10000 ? e * 10 + (*z - '0') : 10000); z += incr; eValid = true; }
		}

		// skip trailing spaces
		if (digits && eValid) while (z < end && _isspace(*z)) z += incr;

do_atof_calc:
		// adjust exponent by d, and update sign
		e = (e * esign) + d;
		if (e < 0) { esign = -1; e *= -1; }
		else esign = 1;

		// if !significand
		double result;
		if (!s)
			result = (sign < 0 && digits ? -0.0 : 0.0); // In the IEEE 754 standard, zero is signed. Add the sign if we've seen at least one digit
		else
		{
			// attempt to reduce exponent
			if (esign > 0) while (s < (LARGEST_INT64 / 10) && e > 0) e--, s *= 10;
			else while (!(s % 10) && e > 0) e--, s /= 10;

			// adjust the sign of significand
			s = (sign < 0 ? -s : s);

			// if exponent, scale significand as appropriate and store in result.
			if (e)
			{
#if __CUDACC__
				double scale = 1.0;
#else
				long double scale = 1.0;
#endif
				// attempt to handle extremely small/large numbers better
				if (e > 307 && e < 342)
				{
					while (e % 308) { scale *= 1.0e+1; e -= 1; }
					if (esign < 0) { result = s / scale; result /= 1.0e+308; }
					else { result = s * scale; result *= 1.0e+308; }
				}
				else if (e >= 342)
					result = (esign < 0 ? 0.0 * s : 1e308 * 1e308 * s); // Infinity
				else
				{
					// 1.0e+22 is the largest power of 10 than can be represented exactly. */
					while (e % 22) { scale *= 1.0e+1; e -= 1; }
					while (e > 0) { scale *= 1.0e+22; e -= 22; }
					result = (esign < 0 ? s / scale : s * scale);
				}
			}
			else
				result = (double)s;
		}

		*out = result; // store the result
		return (z > end && digits > 0 && eValid && !nonNum); // return true if number and no extra non-whitespace chracters after
#else
		return !Atoi64(z, rResult, length, enc);
#endif
	}

	__device__ static int Compare2pow63(const char *z, int incr)
	{
		const char *pow63 = "922337203685477580"; // 012345678901234567
		int c = 0;
		for (int i = 0; c == 0 && i < 18; i++)
			c = (z[i * incr] - pow63[i]) * 10;
		if (c == 0)
		{
			c = z[18 * incr] - '8';
			ASSERTCOVERAGE(c == -1);
			ASSERTCOVERAGE(c == 0);
			ASSERTCOVERAGE(c == +1);
		}
		return c;
	}

	__device__ int ConvertEx::Atoi64(const char *z, int64 *out, int length, TEXTENCODE encode)
	{
		_assert(encode == TEXTENCODE_UTF8 || encode == TEXTENCODE_UTF16LE || encode == TEXTENCODE_UTF16BE);
		//	*out = 0.0; // Default return value, in case of an error
		const char *start;
		const char *end = z + length;

		// get size
		int incr;
		bool nonNum = false;
		if (encode == TEXTENCODE_UTF8)
			incr = 1;
		else
		{
			_assert(TEXTENCODE_UTF16LE == 2 && TEXTENCODE_UTF16BE == 3);
			incr = 2;
			int i; for (i = 3 - encode; i < length && z[i] == 0; i += 2) { }
			nonNum = (i < length);
			end = z + i + encode - 3;
			z += (encode & 1);
		}

		// skip leading spaces
		while (z < end && _isspace(*z)) z += incr;

		// get sign of significand
		int neg = 0; // assume positive
		if (z < end)
		{
			if (*z == '-') { neg = 1; z += incr; }
			else if (*z == '+') z += incr;
		}
		start = z;

		// skip leading zeros
		while (z < end && z[0] == '0') z += incr;

		uint64 u = 0;
		int c = 0;
		int i; for (i = 0; &z[i] < end && (c = z[i]) >= '0' && c <= '9'; i += incr) u = u * 10 + c - '0';
		if (u > LARGEST_INT64) *out = SMALLEST_INT64;
		else *out = (neg ?  -(int64)u : (int64)u);

		ASSERTCOVERAGE(i == 18);
		ASSERTCOVERAGE(i == 19);
		ASSERTCOVERAGE(i == 20);
		if ((c != 0 && &z[i] < end) || (i == 0 && start == z) || i > 19 * incr || nonNum) return 1; // z is empty or contains non-numeric text or is longer than 19 digits (thus guaranteeing that it is too large)
		else if (i < 19 * incr) { _assert(u <= LARGEST_INT64); return 0; } // Less than 19 digits, so we know that it fits in 64 bits
		else // zNum is a 19-digit numbers.  Compare it against 9223372036854775808.
		{
			c = Compare2pow63(z, incr);
			if (c < 0) { _assert(u <= LARGEST_INT64); return 0; } // zNum is less than 9223372036854775808 so it fits
			else if (c > 0) return 1; // zNum is greater than 9223372036854775808 so it overflows
			else { _assert(u-1 == LARGEST_INT64); _assert(*out == SMALLEST_INT64); return neg ? 0 : 2; } //(neg ? 0 : 2); } // z is exactly 9223372036854775808.  Fits if negative.  The special case 2 overflow if positive
		}
	}

	__device__ bool ConvertEx::Atoi(const char *z, int *out)
	{
		int neg = 0;
		if (z[0] == '-') { neg = 1; z++; }
		else if (z[0] == '+') z++;
		while (z[0] == '0') z++;
		int64 v = 0;
		int i, c;
		for (i = 0; i < 11 && (c = z[i] - '0') >= 0 && c <= 9; i++) { v = v*10 + c; }
		// The longest decimal representation of a 32 bit integer is 10 digits:
		//             1234567890
		//     2^31 -> 2147483648
		ASSERTCOVERAGE(i == 10);
		if (i > 10) return false;
		ASSERTCOVERAGE(v-neg == 2147483647);
		if (v - neg > 2147483647) return false;
		*out = (int)(neg ? -v : v);
		return true;
	}

#pragma endregion

#pragma region From: Pragma_c

	// 123456789 123456789
	__constant__ static const char _safetyLevelText[] = "onoffalseyestruefull";
	__constant__ static const uint8 _safetyLevelOffset[] = {0, 1, 2, 4, 9, 12, 16};
	__constant__ static const uint8 _safetyLevelLength[] = {2, 2, 3, 5, 3, 4, 4};
	__constant__ static const uint8 _safetyLevelValue[] =  {1, 0, 0, 0, 1, 1, 2};

	__device__ uint8 ConvertEx::GetSafetyLevel(const char *z, int omitFull, uint8 dflt)
	{
		if (_isdigit(*z))
			return (uint8)Atoi(z);
		int n = _strlen30(z);
		for (int i = 0; i < _lengthof(_safetyLevelLength) - omitFull; i++)
			if (_safetyLevelLength[i] == n && !_strncmp(&_safetyLevelText[_safetyLevelOffset[i]],z,n))
				return _safetyLevelValue[i];
		return dflt;
	}

	__device__ bool ConvertEx::GetBoolean(const char *z, uint8 dflt)
	{
		return (GetSafetyLevel(z, 1, dflt) != 0);
	}

#pragma endregion
}
