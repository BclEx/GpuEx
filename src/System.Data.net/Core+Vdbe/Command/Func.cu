// func.c
#include "..\VdbeInt.cu.h"
#include <stdlib.h>

namespace Core {
#ifndef OMIT_COMPILEOPTION_DIAGS
	__device__ extern bool CompileTimeOptionUsed(const char *optName);
	__device__ extern const char *CompileTimeGet(int id);
#endif
}

namespace Core { namespace Command
{
	__device__ CollSeq *Func::GetFuncCollSeq(FuncContext *fctx)
	{
		return fctx->Coll;
	}

	__device__ void Func::SkipAccumulatorLoad(FuncContext *fctx)
	{
		fctx->SkipFlag = true;
	}

	__device__ static void MinMaxFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc > 1);
		int mask = (Vdbe::User_Data(fctx) == nullptr ? 0 : -1); // 0 for min() or 0xffffffff for max()
		CollSeq *coll = Func::GetFuncCollSeq(fctx);
		_assert(coll);
		_assert(mask == -1 || mask == 0);
		int best = 0;
		if (Vdbe::Value_Type(argv[0]) == TYPE_NULL) return;
		for (int i = 1; i < argc; i++)
		{
			if (Vdbe::Value_Type(argv[i]) == TYPE_NULL) return;
			if ((Vdbe::MemCompare(argv[best], argv[i], coll)^mask) >= 0)
			{
				ASSERTCOVERAGE(mask == 0);
				best = i;
			}
		}
		Vdbe::Result_Value(fctx, argv[best]);
	}

	__device__ static void TypeOfFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		const char *z;
		switch (Vdbe::Value_Type(argv[0]))
		{
		case TYPE_INTEGER: z = "integer"; break;
		case TYPE_TEXT: z = "text"; break;
		case TYPE_FLOAT: z = "real"; break;
		case TYPE_BLOB: z = "blob"; break;
		default: z = "null"; break;
		}
		Vdbe::Result_Text(fctx, z, -1, DESTRUCTOR_STATIC);
	}

	__device__ static void LengthFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		int len;
		_assert(argc == 1);
		switch (Vdbe::Value_Type(argv[0]))
		{
		case TYPE_BLOB:
		case TYPE_INTEGER:
		case TYPE_FLOAT: {
			Vdbe::Result_Int(fctx, Vdbe::Value_Bytes(argv[0]));
			break; }
		case TYPE_TEXT: {
			const unsigned char *z = Vdbe::Value_Text(argv[0]);
			if (!z) return;
			len = 0;
			while (*z)
			{
				len++;
				_strskiputf8(z);
			}
			Vdbe::Result_Int(fctx, len);
			break; }
		default: {
			Vdbe::Result_Null(fctx);
			break; }
		}
	}

	__device__ static void AbsFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		switch (Vdbe::Value_Type(argv[0]))
		{
		case TYPE_INTEGER: {
			int64 ival = Vdbe::Value_Int64(argv[0]);
			if (ival < 0)
			{
				if ((ival<<1) == 0)
				{
					// IMP: R-35460-15084 If X is the integer -9223372036854775807 then abs(X) throws an integer overflow error since there is no
					// equivalent positive 64-bit two complement value.
					Vdbe::Result_Error(fctx, "integer overflow", -1);
					return;
				}
				ival = -ival;
			} 
			Vdbe::Result_Int64(fctx, ival);
			break; }
		case TYPE_NULL: {
			// IMP: R-37434-19929 Abs(X) returns NULL if X is NULL.
			Vdbe::Result_Null(fctx);
			break; }
		default: {
			// Because sqlite3_value_double() returns 0.0 if the argument is not something that can be converted into a number, we have:
			// IMP: R-57326-31541 Abs(X) return 0.0 if X is a string or blob that cannot be converted to a numeric value. 
			double rval = Vdbe::Value_Double(argv[0]);
			if (rval < 0) rval = -rval;
			Vdbe::Result_Double(fctx, rval);
			break; }
		}
	}

	__device__ static void InstrFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		TYPE typeHaystack = Vdbe::Value_Type(argv[0]);
		TYPE typeNeedle = Vdbe::Value_Type(argv[1]);
		if (typeHaystack == TYPE_NULL || typeNeedle == TYPE_NULL) return;
		int haystackLength = Vdbe::Value_Bytes(argv[0]);
		int needleLength = Vdbe::Value_Bytes(argv[1]);
		const unsigned char *haystack;
		const unsigned char *needle;
		bool isText;
		if (typeHaystack == TYPE_BLOB && typeNeedle == TYPE_BLOB)
		{
			haystack = (const unsigned char *)Vdbe::Value_Blob(argv[0]);
			needle = (const unsigned char *)Vdbe::Value_Blob(argv[1]);
			isText = false;
		}
		else
		{
			haystack = Vdbe::Value_Text(argv[0]);
			needle = Vdbe::Value_Text(argv[1]);
			isText = true;
		}
		int n = 1;
		while (needleLength <= haystackLength && _memcmp(haystack, needle, needleLength) != 0)
		{
			n++;
			do
			{
				haystackLength--;
				haystack++;
			} while (isText && (haystack[0] & 0xc0) == 0x80);
		}
		if (needleLength > haystackLength) n = 0;
		Vdbe::Result_Int(fctx, n);
	}

	__device__ static void SubstrFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 3 || argc == 2);
		if (Vdbe::Value_Type(argv[1]) == TYPE_NULL || (argc == 3 && Vdbe::Value_Type(argv[2]) == TYPE_NULL))
			return;
		TYPE p0type = Vdbe::Value_Type(argv[0]);
		int64 p1 = Vdbe::Value_Int(argv[1]);
		int len;
		const unsigned char *z;
		const unsigned char *z2;
		if (p0type == TYPE_BLOB)
		{
			len = Vdbe::Value_Bytes(argv[0]);
			z = (const unsigned char *)Vdbe::Value_Blob(argv[0]);
			if (!z) return;
			//_assert(len == Vdbe::Value_Bytes(argv[0]));
		}
		else
		{
			z = Vdbe::Value_Text(argv[0]);
			if (!z) return;
			len = 0;
			if (p1 < 0)
				for (z2 = z; *z2; len++)
					_strskiputf8(z2);
		}
		int64 p2;
		bool negP2 = false;
		if (argc == 3)
		{
			p2 = Vdbe::Value_Int(argv[2]);
			if (p2 < 0)
			{
				p2 = -p2;
				negP2 = true;
			}
		}
		else
			p2 = Vdbe::Context_Ctx(fctx)->Limits[LIMIT_LENGTH];
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
			p1 -= p2;
			if (p1 < 0)
			{
				p2 += p1;
				p1 = 0;
			}
		}
		_assert(p1 >= 0 && p2 >= 0);
		if (p0type != TYPE_BLOB)
		{
			while (*z && p1)
			{
				_strskiputf8(z);
				p1--;
			}
			for (z2 = z; *z2 && p2; p2--)
				_strskiputf8(z2);
			Vdbe::Result_Text(fctx, (char*)z, (int)(z2-z), DESTRUCTOR_TRANSIENT);
		}
		else
		{
			if (p1 + p2 > len)
			{
				p2 = len-p1;
				if (p2 < 0) p2 = 0;
			}
			Vdbe::Result_Blob(fctx, (char*)&z[p1], (int)p2, DESTRUCTOR_TRANSIENT);
		}
	}

#ifndef OMIT_FLOATING_POINT
	__device__ static void RoundFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1 || argc == 2);
		int n = 0;
		if (argc == 2)
		{
			if (Vdbe::Value_Type(argv[1]) == TYPE_NULL) return;
			n = Vdbe::Value_Int(argv[1]);
			if (n > 30) n = 30;
			if (n < 0) n = 0;
		}
		if (Vdbe::Value_Type(argv[0]) == TYPE_NULL) return;
		double r = Vdbe::Value_Double(argv[0]);
		// If Y==0 and X will fit in a 64-bit int, handle the rounding directly, otherwise use printf.
		if (n == 0 && r >= 0 && r < LARGEST_INT64-1)
			r = (double)((int64)(r+0.5));
		else if (n == 0 && r < 0 && (-r) < LARGEST_INT64-1)
			r = -(double)((int64)((-r)+0.5));
		else
		{
			char *buf = _mprintf("%.*f",n,r);
			if (!buf)
			{
				Vdbe::Result_ErrorNoMem(fctx);
				return;
			}
			ConvertEx::Atof(buf, &r, _strlen30(buf), TEXTENCODE_UTF8);
			_free(buf);
		}
		Vdbe::Result_Double(fctx, r);
	}
#endif

	__device__ static void *ContextMalloc(FuncContext *fctx, int64 bytes)
	{

		Context *ctx = Vdbe::Context_Ctx(fctx);
		_assert(bytes > 0);
		ASSERTCOVERAGE(bytes == ctx->Limits[LIMIT_LENGTH]);
		ASSERTCOVERAGE(bytes == ctx->Limits[LIMIT_LENGTH]+1);
		char *z;
		if (bytes > ctx->Limits[LIMIT_LENGTH])
		{
			Vdbe::Result_ErrorOverflow(fctx);
			z = nullptr;
		}
		else
		{
			z = (char *)_alloc((int)bytes);
			if (!z)
				Vdbe::Result_ErrorNoMem(fctx);
		}
		return z;
	}

	__device__ static void UpperFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const char *z2 = (char *)Vdbe::Value_Text(argv[0]);
		int n = Vdbe::Value_Bytes(argv[0]);
		// Verify that the call to _bytes() does not invalidate the _text() pointer
		_assert(z2 == (char *)Vdbe::Value_Text(argv[0]));
		if (z2)
		{
			char *z1 = (char *)ContextMalloc(fctx, ((int64)n)+1);
			if (z1)
			{
				for (int i = 0; i < n; i++)
					z1[i] = _toupper(z2[i]);
				Vdbe::Result_Text(fctx, z1, n, _free);
			}
		}
	}

	__device__ static void LowerFunc(FuncContext *fctx, int argc, Mem **argv){
		const char *z2 = (char *)Vdbe::Value_Text(argv[0]);
		int n = Vdbe::Value_Bytes(argv[0]);
		// Verify that the call to _bytes() does not invalidate the _text() pointer
		_assert(z2 == (char *)Vdbe::Value_Text(argv[0]));
		if (z2)
		{
			char *z1 = (char *)ContextMalloc(fctx, ((int64)n)+1);
			if (z1)
			{
				for (int i = 0; i < n; i++)
					z1[i] = _tolower(z2[i]);
				Vdbe::Result_Text(fctx, z1, n, _free);
			}
		}
	}

#define IfNullFunc VersionFunc // Substitute function - never called

	__device__ static void RandomFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		int64 r;
		SysEx::PutRandom(sizeof(r), &r);
		if (r < 0)
		{
			// We need to prevent a random number of 0x8000000000000000 (or -9223372036854775808) since when you do abs() of that
			// number of you get the same value back again.  To do this in a way that is testable, mask the sign bit off of negative
			// values, resulting in a positive value.  Then take the 2s complement of that positive value.  The end result can
			// therefore be no less than -9223372036854775807.
			r = -(r & LARGEST_INT64);
		}
		Vdbe::Result_Int64(fctx, r);
	}

	__device__ static void RandomBlob(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		int n = Vdbe::Value_Int(argv[0]);
		if (n < 1)
			n = 1;
		unsigned char *p = (unsigned char *)ContextMalloc(fctx, n);
		if (p)
		{
			SysEx::PutRandom(n, p);
			Vdbe::Result_Blob(fctx, (char *)p, n, _free);
		}
	}

	__device__ static void LastInsertRowid(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		// IMP: R-51513-12026 The last_insert_rowid() SQL function is a wrapper around the sqlite3_last_insert_rowid() C/C++ interface function.
		Vdbe::Result_Int64(fctx, Main::CtxLastInsertRowid(ctx));
	}

	__device__ static void Changes(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		Vdbe::Result_Int(fctx, Main::CtxChanges(ctx));
	}

	__device__ static void TotalChanges(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		// IMP: R-52756-41993 This function is a wrapper around the sqlite3_total_changes() C/C++ interface.
		Vdbe::Result_Int(fctx, Main::CtxTotalChanges(ctx));
	}

	struct CompareInfo
	{
		uint8 MatchAll;
		uint8 MatchOne;
		uint8 MatchSet;
		bool NoCase;
	};

	__device__ static const CompareInfo _globInfo = { '*', '?', '[', false };
	__device__ static const CompareInfo _likeInfoNorm = { '%', '_',   0, true }; // The correct SQL-92 behavior is for the LIKE operator to ignore case.  Thus  'a' LIKE 'A' would be true.
	__device__ static const CompareInfo _likeInfoAlt = { '%', '_',   0, false }; // If SQLITE_CASE_SENSITIVE_LIKE is defined, then the LIKE operator is case sensitive causing 'a' LIKE 'A' to be false

	__device__ static bool PatternCompare(const uint8 *pattern, const uint8 *string, const CompareInfo *info, uint32 escape)
	{
		uint32 c, c2;
		int invert;
		int seen;
		uint8 matchOne = info->MatchOne;
		uint8 matchAll = info->MatchAll;
		uint8 matchSet = info->MatchSet;
		bool noCase = info->NoCase; 
		bool prevEscape = false; // True if the previous character was 'escape'

		while ((c = _utf8read(&pattern)) != 0)
		{
			if (c == matchAll && !prevEscape)
			{
				while ((c = _utf8read(&pattern)) == matchAll || c == matchOne)
					if (c == matchOne && _utf8read(&string) == 0)
						return false;
				if (c == 0)
					return true;
				else if (c == escape)
				{
					c = _utf8read(&pattern);
					if (c == 0)
						return false;
				}
				else if (c == matchSet)
				{
					_assert(escape == 0); // This is GLOB, not LIKE
					_assert(matchSet < 0x80); // '[' is a single-byte character
					while (*string && !PatternCompare(&pattern[-1], string, info, escape))
						_strskiputf8(string);
					return (*string != 0);
				}
				while ((c2 = _utf8read(&string)) != 0)
				{
					if (noCase)
					{
						c2 = _tolower(c2);
						c = _tolower(c);
						while (c2 != 0 && c2 != c)
						{
							c2 = _utf8read(&string);
							c2 = _tolower(c2);
						}
					}
					else
					{
						while (c2 != 0 && c2 != c)
							c2 = _utf8read(&string);
					}
					if (c2 == 0) return false;
					if (PatternCompare(pattern, string, info, escape)) return true;
				}
				return false;
			}
			else if (c == matchOne && !prevEscape)
			{
				if (_utf8read(&string) == 0)
					return false;
			}
			else if (c == matchSet)
			{
				uint32 prior_c = 0;
				_assert(escape == 0); // This only occurs for GLOB, not LIKE
				seen = 0;
				invert = 0;
				c = _utf8read(&string);
				if (c == 0) return false;
				c2 = _utf8read(&pattern);
				if (c2 == '^')
				{
					invert = 1;
					c2 = _utf8read(&pattern);
				}
				if (c2 == ']')
				{
					if (c == ']') seen = 1;
					c2 = _utf8read(&pattern);
				}
				while (c2 && c2 != ']')
				{
					if (c2 == '-' && pattern[0] != ']' && pattern[0] != 0 && prior_c > 0)
					{
						c2 = _utf8read(&pattern);
						if (c >= prior_c && c <= c2) seen = 1;
						prior_c = 0;
					}
					else
					{
						if (c == c2)
							seen = 1;
						prior_c = c2;
					}
					c2 = _utf8read(&pattern);
				}
				if (c2 == 0 || (seen ^ invert) == 0)
					return false;
			}
			else if (escape == c && !prevEscape)
				prevEscape = true;
			else
			{
				c2 = _utf8read(&string);
				if (noCase)
				{
					c = _tolower(c);
					c2 = _tolower(c2);
				}
				if (c != c2)
					return false;
				prevEscape = false;
			}
		}
		return (*string == 0);
	}

#ifdef TEST
	__device__ static int _likeCount = 0;
#endif
	__device__ static void LikeFunc(FuncContext *fctx,  int argc,  Mem **argv)
	{
		uint32 escape = 0;
		Context *ctx = Vdbe::Context_Ctx(fctx);

		const unsigned char *zB = Vdbe::Value_Text(argv[0]);
		const unsigned char *zA = Vdbe::Value_Text(argv[1]);

		// Limit the length of the LIKE or GLOB pattern to avoid problems of deep recursion and N*N behavior in patternCompare().
		int pats = Vdbe::Value_Bytes(argv[0]);
		ASSERTCOVERAGE(pats == ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH]);
		ASSERTCOVERAGE(pats == ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH]+1);
		if (pats > ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH])
		{
			Vdbe::Result_Error(fctx, "LIKE or GLOB pattern too complex", -1);
			return;
		}
		_assert(zB == Vdbe::Value_Text(argv[0])); // Encoding did not change

		if (argc == 3)
		{
			// The escape character string must consist of a single UTF-8 character. Otherwise, return an error.
			const unsigned char *zEscape = Vdbe::Value_Text(argv[2]);
			if (!zEscape) return;
			if (_utf8charlength((char *)zEscape, -1) != 1)
			{
				Vdbe::Result_Error(fctx, "ESCAPE expression must be a single character", -1);
				return;
			}
			escape = _utf8read(&zEscape);
		}
		if (zA && zB)
		{
			CompareInfo *info = (CompareInfo *)Vdbe::User_Data(fctx);
#ifdef TEST
			_likeCount++;
#endif
			Vdbe::Result_Int(fctx, PatternCompare(zB, zA, info, escape));
		}
	}

	__device__ static void NullIfFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		CollSeq *coll = Func::GetFuncCollSeq(fctx);
		if (Vdbe::MemCompare(argv[0], argv[1], coll) != 0)
			Vdbe::Result_Value(fctx, argv[0]);
	}

	__device__ static void VersionFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		// IMP: R-48699-48617 This function is an SQL wrapper around the sqlite3_libversion() C-interface.
		Vdbe::Result_Text(fctx, CORE_VERSION, -1, DESTRUCTOR_STATIC);
	}

	__device__ static void SourceIdFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		// IMP: R-24470-31136 This function is an SQL wrapper around the sqlite3_sourceid() C interface.
		Vdbe::Result_Text(fctx, CORE_SOURCE_ID, -1, DESTRUCTOR_STATIC);
	}

	__device__ static void ErrlogFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		SysEx_LOG((RC)Vdbe::Value_Int(argv[0]), "%s", Vdbe::Value_Text(argv[1]));
	}


#ifndef OMIT_COMPILEOPTION_DIAGS
	__device__ static void CompileoptionusedFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		// IMP: R-39564-36305 The sqlite_compileoption_used() SQL function is a wrapper around the sqlite3_compileoption_used() C/C++ function.
		const char *optName;
		if ((optName = (const char *)Vdbe::Value_Text(argv[0])) != nullptr)
			Vdbe::Result_Int(fctx, CompileTimeOptionUsed(optName));
	}


	__device__ static void CompileoptiongetFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		// IMP: R-04922-24076 The sqlite_compileoption_get() SQL function is a wrapper around the sqlite3_compileoption_get() C/C++ function.
		int n = Vdbe::Value_Int(argv[0]);
		Vdbe::Result_Text(fctx, CompileTimeGet(n), -1, DESTRUCTOR_STATIC);
	}
#endif

	// [EXPERIMENTAL]
	__device__ static const char _hexdigits[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'  };
	__device__ static void QuoteFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		switch (Vdbe::Value_Type(argv[0]))
		{
		case TYPE_FLOAT: {
			double r1 = Vdbe::Value_Double(argv[0]);
			char b[50];
			__snprintf(b, sizeof(b), "%!.15g", r1);
			double r2;
			ConvertEx::Atof(b, &r2, 20, TEXTENCODE_UTF8);
			if (r1 != r2)
				__snprintf(b, sizeof(b), "%!.20e", r1);
			Vdbe::Result_Text(fctx, b, -1, DESTRUCTOR_TRANSIENT);
			break; }

		case TYPE_INTEGER: {
			Vdbe::Result_Value(fctx, argv[0]);
			break; }

		case TYPE_BLOB: {
			char *z = 0;
			char const *blob = (char const *)Vdbe::Value_Blob(argv[0]);
			int blobLength = Vdbe::Value_Bytes(argv[0]);
			_assert(blob == Vdbe::Value_Blob(argv[0])); // No encoding change
			z = (char *)ContextMalloc(fctx, (2*(int64)blobLength)+4); 
			if (z)
			{
				for (int i = 0; i < blobLength; i++)
				{
					z[(i*2)+2] = _hexdigits[(blob[i]>>4)&0x0F];
					z[(i*2)+3] = _hexdigits[(blob[i])&0x0F];
				}
				z[(blobLength*2)+2] = '\'';
				z[(blobLength*2)+3] = '\0';
				z[0] = 'X';
				z[1] = '\'';
				Vdbe::Result_Text(fctx, z, -1, DESTRUCTOR_TRANSIENT);
				_free(z);
			}
			break; }

		case TYPE_TEXT: {
			const unsigned char *zarg = Vdbe::Value_Text(argv[0]);
			if (zarg == nullptr) return;
			int i, j;
			uint64 n;
			for (i = 0, n = 0; zarg[i]; i++) { if (zarg[i] == '\'') n++; }
			char *z = (char *)ContextMalloc(fctx, ((int64)i)+((int64)n)+3);
			if (z)
			{
				z[0] = '\'';
				for (i = 0, j = 1; zarg[i]; i++)
				{
					z[j++] = zarg[i];
					if (zarg[i] == '\'')
						z[j++] = '\'';
				}
				z[j++] = '\'';
				z[j] = 0;
				Vdbe::Result_Text(fctx, z, j, _free);
			}
			break; }

		default: {
			_assert(Vdbe::Value_Type(argv[0]) == TYPE_NULL);
			Vdbe::Result_Text(fctx, "NULL", 4, DESTRUCTOR_STATIC);
			break; }
		}
	}

	__device__ static void UnicodeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const unsigned char *z = Vdbe::Value_Text(argv[0]);
		if (z && z[0]) Vdbe::Result_Int(fctx, _utf8read(&z));
	}

	__device__ static void CharFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		unsigned char *z, *z2;
		z = z2 = (unsigned char *)_alloc(argc*4);
		if (!z)
		{
			Vdbe::Result_ErrorNoMem(fctx);
			return;
		}
		for (int i = 0; i < argc; i++)
		{
			int64 x = Vdbe::Value_Int64(argv[i]);
			if (x < 0 || x > 0x10ffff) x = 0xfffd;
			unsigned c = (unsigned)(x & 0x1fffff);
			if (c < 0x00080)
				*z2++ = (uint8)(c & 0xFF);
			else if (c < 0x00800)
			{
				*z2++ = 0xC0 + (uint8)((c>>6) & 0x1F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
			else if (c < 0x10000)
			{
				*z2++ = 0xE0 + (uint8)((c>>12)&0x0F);
				*z2++ = 0x80 + (uint8)((c>>6) & 0x3F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
			else
			{
				*z2++ = 0xF0 + (uint8)((c>>18) & 0x07);
				*z2++ = 0x80 + (uint8)((c>>12) & 0x3F);
				*z2++ = 0x80 + (uint8)((c>>6) & 0x3F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
		}
		Vdbe::Result_Text(fctx, (char *)z, (int)(z2 - z), _free);
	}

	__device__ static void HexFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		const unsigned char *blob = (const unsigned char *)Vdbe::Value_Blob(argv[0]);
		int n = Vdbe::Value_Bytes(argv[0]);
		_assert(blob == Vdbe::Value_Blob(argv[0])); // No encoding change
		char *zHex, *z;
		zHex = z = (char *)ContextMalloc(fctx, ((int64)n)*2 + 1);
		if (zHex)
		{
			for (int i = 0; i < n; i++, blob++)
			{
				unsigned char c = *blob;
				*(z++) = _hexdigits[(c>>4)&0xf];
				*(z++) = _hexdigits[c&0xf];
			}
			*z = 0;
			Vdbe::Result_Text(fctx, zHex, n*2, _free);
		}
	}

	__device__ static void ZeroBlobFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		Context *ctx = Vdbe::Context_Ctx(fctx);
		int64 n = Vdbe::Value_Int64(argv[0]);
		ASSERTCOVERAGE(n == ctx->Limits[LIMIT_LENGTH]);
		ASSERTCOVERAGE(n == ctx->Limits[LIMIT_LENGTH]+1);
		if (n > ctx->Limits[LIMIT_LENGTH])
			Vdbe::Result_ErrorOverflow(fctx);
		else
			Vdbe::Result_ZeroBlob(fctx, (int)n); // IMP: R-00293-64994
	}

	__device__ static void ReplaceFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 3);
		const unsigned char *string = Vdbe::Value_Text(argv[0]); // The input string A
		if (!string) return;
		int stringLength = Vdbe::Value_Bytes(argv[0]); // Size of string
		_assert(string == Vdbe::Value_Text(argv[0])); // No encoding change
		const unsigned char *pattern = Vdbe::Value_Text(argv[1]);
		if (!pattern)
		{
			_assert(Vdbe::Value_Type(argv[1]) == TYPE_NULL || Vdbe::Context_Ctx(fctx)->MallocFailed);
			return;
		}
		if (pattern[0] == 0)
		{
			_assert(Vdbe::Value_Type(argv[1]) != TYPE_NULL);
			Vdbe::Result_Value(fctx, argv[0]);
			return;
		}
		int patternLength = Vdbe::Value_Bytes(argv[1]); // Size of pattern
		_assert(pattern == Vdbe::Value_Text(argv[1])); // No encoding change
		const unsigned char *replacement = Vdbe::Value_Text(argv[2]); // The replacement string C
		if (!replacement) return;
		int replacementLength = Vdbe::Value_Bytes(argv[2]); // Size of replacement
		_assert(replacement == Vdbe::Value_Text(argv[2]));
		int64 outLength = stringLength + 1; // Maximum size of out
		_assert(outLength < CORE_MAX_LENGTH);
		unsigned char *out = (unsigned char *)ContextMalloc(fctx, (int64)outLength); // The output
		if (!out)
			return;
		int loopLimit = stringLength - patternLength; // Last string[] that might match pattern[]
		int i, j;
		for (i = j = 0; i <= loopLimit; i++)
		{
			if (string[i] != pattern[0] || _memcmp(&string[i], pattern, patternLength))
				out[j++] = string[i];
			else
			{
				Context *ctx = Vdbe::Context_Ctx(fctx);
				outLength += replacementLength - patternLength;
				ASSERTCOVERAGE(outLength-1 == ctx->Limits[LIMIT_LENGTH]);
				ASSERTCOVERAGE(outLength-2 == ctx->Limits[LIMIT_LENGTH]);
				if (outLength-1 > ctx->Limits[LIMIT_LENGTH])
				{
					Vdbe::Result_ErrorOverflow(fctx);
					_free(out);
					return;
				}
				uint8 *oldOut = out;
				out = (unsigned char *)_realloc(out, (int)outLength);
				if (!out)
				{
					Vdbe::Result_ErrorNoMem(fctx);
					_free(oldOut);
					return;
				}
				_memcpy(&out[j], replacement, replacementLength);
				j += replacementLength;
				i += patternLength-1;
			}
		}
		_assert(j+stringLength-i+1 == outLength);
		_memcpy(&out[j], &string[i], stringLength-i);
		j += stringLength - i;
		_assert(j <= outLength);
		out[j] = 0;
		Vdbe::Result_Text(fctx, (char *)out, j, _free);
	}

	__device__ static const unsigned char _trimOneLength[] = { 1 };
	__device__ static unsigned char * const _trimOne[] = { (uint8 *)" " };
	__device__ static void TrimFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const unsigned char *charSet; // Set of characters to trim
		int charSetLength; // Number of characters in charSet
		unsigned char *charsLength = nullptr; // Length of each character in charSet
		unsigned char **chars = nullptr; // Individual characters in charSet
		int flags; // 1: trimleft  2: trimright  3: trim

		if (Vdbe::Value_Type(argv[0]) == TYPE_NULL)
			return;
		const unsigned char *in = Vdbe::Value_Text(argv[0]); // Input string
		if (!in) return;
		int inLength = Vdbe::Value_Bytes(argv[0]); // Number of bytes in input
		//? _assert(in == Vdbe::Value_Text(argv[0]));
		if (argc == 1)
		{
			charSetLength = 1;
			charsLength = (uint8 *)_trimOneLength;
			chars = (unsigned char **)_trimOne;
			charSet = nullptr;
		}
		else if ((charSet = Vdbe::Value_Text(argv[1])) == nullptr)
			return;
		else
		{
			const unsigned char *z;
			for (z = charSet, charSetLength = 0; *z; charSetLength++)
				_strskiputf8(z);
			if (charSetLength > 0)
			{
				chars = (unsigned char **)ContextMalloc(fctx, ((int64)charSetLength)*(sizeof(char *)+1));
				if (!chars)
					return;
				charsLength = (unsigned char *)&chars[charSetLength];
				for (z = charSet, charSetLength = 0; *z; charSetLength++)
				{
					chars[charSetLength] = (unsigned char *)z;
					_strskiputf8(z);
					charsLength[charSetLength] = (uint8)(z - chars[charSetLength]);
				}
			}
		}
		int i;
		if (charSetLength > 0)
		{
			flags = PTR_TO_INT(Vdbe::User_Data(fctx));
			if (flags & 1)
			{
				while (inLength > 0)
				{
					int len = 0;
					for (i = 0; i < charSetLength; i++)
					{
						len = charsLength[i];
						if (len <= inLength && !_memcmp(in, chars[i], len)) break;
					}
					if (i >= charSetLength) break;
					in += len;
					inLength -= len;
				}
			}
			if (flags & 2)
			{
				while (inLength > 0)
				{
					int len = 0;
					for (i = 0; i < charSetLength; i++)
					{
						len = charsLength[i];
						if (len <= inLength && !_memcmp(&in[inLength-len], chars[i], len)) break;
					}
					if (i >= charSetLength) break;
					inLength -= len;
				}
			}
			if (charSet)
				_free(chars);
		}
		Vdbe::Result_Text(fctx, (char *)in, inLength, DESTRUCTOR_TRANSIENT);
	}

#ifdef SOUNDEX
	__device__ static const unsigned char _soundexCode[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
		1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
		1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
	};

	__device__ static void SoundexFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		const uint8 *z = (uint8 *)Vdbe::Value_Text(argv[0]);
		if (!z) z = (uint8 *)"";
		int i;
		for (i = 0; z[i] && !_isalpha(z[i]); i++) { }
		char r[8];
		if (z[i])
		{
			uint8 prevcode = _soundexCode[z[i]&0x7f];
			r[0] = _toupper(z[i]);
			int j;
			for (j = 1; j < 4 && z[i]; i++)
			{
				uint8 code = _soundexCode[z[i]&0x7f];
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
			Vdbe::Result_Text(fctx, r, 4, DESTRUCTOR_TRANSIENT);
		}
		else
			Vdbe::Result_Text(fctx, "?000", 4, DESTRUCTOR_STATIC); // IMP: R-64894-50321 The string "?000" is returned if the argument is NULL or contains no ASCII alphabetic characters. */
	}
#endif

#ifndef OMIT_LOAD_EXTENSION
	__device__ static void LoadExtFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const char *file = (const char *)Vdbe::Value_Text(argv[0]);
		Context *ctx = Vdbe::Context_Ctx(fctx);
		char *errMsg = nullptr;
		const char *proc = (argc == 2 ? (const char *)Vdbe::Value_Text(argv[1]) : nullptr);
		if (file && Main::LoadExtension(ctx, file, proc, &errMsg))
		{
			Vdbe::Result_Error(fctx, errMsg, -1);
			_free(errMsg);
		}
	}
#endif

	struct SumCtx
	{
		double RSum;	// Floating point sum
		int64 ISum;		// Integer sum
		int64 Count;	// Number of elements summed
		bool Overflow;	// True if integer overflow seen
		bool Approx;	// True if non-integer value was input to the sum
	};

	__device__ static void SumStep(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		SumCtx *p = (SumCtx *)Vdbe::Aggregate_Context(fctx, sizeof(*p));
		TYPE type = Vdbe::ValueNumericType(argv[0]);
		if (p && type != TYPE_NULL)
		{
			p->Count++;
			if (type == TYPE_INTEGER)
			{
				int64 v = Vdbe::Value_Int64(argv[0]);
				p->RSum += v;
				if (!(p->Approx | p->Overflow) && MathEx::Add(&p->ISum, v))
					p->Overflow = true;
			}
			else
			{
				p->RSum += Vdbe::Value_Double(argv[0]);
				p->Approx = true;
			}
		}
	}

	__device__ static void SumFinalize(FuncContext *fctx)
	{
		SumCtx *p = (SumCtx *)Vdbe::Aggregate_Context(fctx, 0);
		if (p && p->Count > 0)
		{
			if (p->Overflow)
				Vdbe::Result_Error(fctx, "integer overflow", -1);
			else if (p->Approx)
				Vdbe::Result_Double(fctx, p->RSum);
			else
				Vdbe::Result_Int64(fctx, p->ISum);
		}
	}

	__device__ static void AvgFinalize(FuncContext *fctx)
	{
		SumCtx *p = (SumCtx *)Vdbe::Aggregate_Context(fctx, 0);
		if (p && p->Count > 0)
			Vdbe::Result_Double(fctx, p->RSum/(double)p->Count);
	}

	__device__ static void TotalFinalize(FuncContext *fctx)
	{
		SumCtx *p = (SumCtx *)Vdbe::Aggregate_Context(fctx, 0);
		Vdbe::Result_Double(fctx, p ? p->RSum : (double)0); // (double)0 In case of OMIT_FLOATING_POINT...
	}

	struct CountCtx
	{
		int64 N;
	};

	__device__ static void CountStep(FuncContext *fctx, int argc, Mem **argv)
	{
		CountCtx *p = (CountCtx *)Vdbe::Aggregate_Context(fctx, sizeof(*p));
		if ((argc == 0 || Vdbe::Value_Type(argv[0]) != TYPE_NULL) && p)
			p->N++;
	}

	__device__ static void CountFinalize(FuncContext *fctx)
	{
		CountCtx *p = (CountCtx *)Vdbe::Aggregate_Context(fctx, 0);
		Vdbe::Result_Int64(fctx, p ? p->N : 0);
	}

	__device__ static void MinMaxStep(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		Mem *arg = (Mem *)argv[0];
		Mem *best = (Mem *)Vdbe::Aggregate_Context(fctx, sizeof(*best));
		if (!best) return;
		if (Vdbe::Value_Type(argv[0]) == TYPE_NULL)
		{
			if (best->Flags) Func::SkipAccumulatorLoad(fctx);
		}
		else if (best->Flags)
		{
			CollSeq *coll = Func::GetFuncCollSeq(fctx);
			// This step function is used for both the min() and max() aggregates, the only difference between the two being that the sense of the
			// comparison is inverted. For the max() aggregate, the sqlite3_user_data() function returns (void *)-1. For min() it
			// returns (void *)db, where db is the sqlite3* database pointer. Therefore the next statement sets variable 'max' to 1 for the max()
			// aggregate, or 0 for min().
			bool max = (Vdbe::User_Data(fctx) != 0);
			int cmp = Vdbe::MemCompare(best, arg, coll);
			if ((max && cmp < 0) || (!max && cmp > 0))
				Vdbe::MemCopy(best, arg);
			else
				Func::SkipAccumulatorLoad(fctx);
		}
		else
			Vdbe::MemCopy(best, arg);
	}

	__device__ static void MinMaxFinalize(FuncContext *fctx)
	{
		Mem *r = (Mem *)Vdbe::Aggregate_Context(fctx, 0);
		if (r)
		{
			if (r->Flags)
				Vdbe::Result_Value(fctx, r);
			Vdbe::MemRelease(r);
		}
	}

	__device__ static void GroupConcatStep(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1 || argc == 2);
		if (Vdbe::Value_Type(argv[0]) == TYPE_NULL) return;
		TextBuilder *b = (TextBuilder *)Vdbe::Aggregate_Context(fctx, sizeof(*b));
		if (b)
		{
			Context *ctx = Vdbe::Context_Ctx(fctx);
			bool firstTerm = (b->AllocType == 0);
			b->AllocType = 2;
			b->MaxSize = ctx->Limits[LIMIT_LENGTH];
			if (!firstTerm)
			{
				const char *zSep;
				int nSep;
				if (argc == 2)
				{
					zSep = (char *)Vdbe::Value_Text(argv[1]);
					nSep = Vdbe::Value_Bytes(argv[1]);
				}
				else
				{
					zSep = ",";
					nSep = 1;
				}
				b->Append(zSep, nSep);
			}
			const char *zVal = (char *)Vdbe::Value_Text(argv[0]);
			int nVal = Vdbe::Value_Bytes(argv[0]);
			b->Append(zVal, nVal);
		}
	}

	__device__ static void GroupConcatFinalize(FuncContext *fctx)
	{
		TextBuilder *b = (TextBuilder *)Vdbe::Aggregate_Context(fctx, 0);
		if (b)
		{
			if (b->Overflowed)
				Vdbe::Result_ErrorOverflow(fctx);
			else if (b->AllocFailed)
				Vdbe::Result_ErrorNoMem(fctx);
			else   
				Vdbe::Result_Text(fctx, b->ToString(), -1,  _free);
		}
	}

	__device__ void Func::RegisterBuiltinFunctions(Context *ctx)
	{
		RC rc = Main::OverloadFunction(ctx, "MATCH", 2);
		_assert(rc == RC_NOMEM || rc == RC_OK);
		if (rc == RC_NOMEM)
			ctx->MallocFailed = true;
	}

	__device__ static void SetLikeOptFlag(Context *ctx, const char *name, FUNC flagVal)
	{
		FuncDef *def = Callback::FindFunction(ctx, name, _strlen30(name), 2, TEXTENCODE_UTF8, 0);
		if (_ALWAYS(def))
			def->Flags = flagVal;
	}

	__device__ void Func::RegisterLikeFunctions(Context *ctx, bool caseSensitive)
	{
		CompareInfo *info = (caseSensitive ? (CompareInfo *)&_likeInfoAlt : (CompareInfo *)&_likeInfoNorm);
		Main::CreateFunc(ctx, "like", 2, TEXTENCODE_UTF8, info, LikeFunc, 0, 0, 0);
		Main::CreateFunc(ctx, "like", 3, TEXTENCODE_UTF8, info, LikeFunc, 0, 0, 0);
		Main::CreateFunc(ctx, "glob", 2, TEXTENCODE_UTF8, (CompareInfo *)&_globInfo, LikeFunc, 0, 0, 0);
		SetLikeOptFlag(ctx, "glob", FUNC_LIKE | FUNC_CASE);
		SetLikeOptFlag(ctx, "like", caseSensitive ? (FUNC_LIKE | FUNC_CASE) : FUNC_LIKE);
	}

	__device__ bool Func::IsLikeFunction(Context *ctx, Expr *expr, bool *isNocase, char *wc)
	{
		if (expr->OP != TK_FUNCTION  || !expr->x.List || expr->x.List->Exprs != 2)
			return false;
		_assert(!ExprHasProperty(expr, EP_xIsSelect));
		FuncDef *def = Callback::FindFunction(ctx, expr->u.Token, _strlen30(expr->u.Token), 2, TEXTENCODE_UTF8, 0);
		if (_NEVER(def == nullptr) || (def->Flags & FUNC_LIKE) == 0)
			return false;
		// The memcpy() statement assumes that the wildcard characters are the first three statements in the compareInfo structure.  The asserts() that follow verify that assumption
		_memcpy(wc, (char *)def->UserData, 3);
		_assert((char *)&_likeInfoAlt == (char*)&_likeInfoAlt.MatchAll);
		_assert(&((char *)&_likeInfoAlt)[1] == (char*)&_likeInfoAlt.MatchOne);
		_assert(&((char *)&_likeInfoAlt)[2] == (char*)&_likeInfoAlt.MatchSet);
		*isNocase = ((def->Flags & FUNC_CASE) == 0);
		return true;
	}

	// The following array holds FuncDef structures for all of the functions defined in this file.
	//
	// The array cannot be constant since changes are made to the FuncDef.pHash elements at start-time.  The elements of this array are read-only after initialization is complete.
#if __CUDACC__
	__constant__ static _WSD FuncDef g_builtinFuncs[56];
	static _WSD FuncDef h_builtinFuncs[56] =
#else
	static _WSD FuncDef g_builtinFuncs[56] =
#endif
	{
		FUNCTION(ltrim,              1, 1, 0, TrimFunc         ),
		FUNCTION(ltrim,              2, 1, 0, TrimFunc         ),
		FUNCTION(rtrim,              1, 2, 0, TrimFunc         ),
		FUNCTION(rtrim,              2, 2, 0, TrimFunc         ),
		FUNCTION(trim,               1, 3, 0, TrimFunc         ),
		FUNCTION(trim,               2, 3, 0, TrimFunc         ),
		FUNCTION(min,               -1, 0, 1, MinMaxFunc       ),
		FUNCTION(min,                0, 0, 1, nullptr          ),
		AGGREGATE(min,               1, 0, 1, MinMaxStep,      MinMaxFinalize),
		FUNCTION(max,               -1, 1, 1, MinMaxFunc       ),
		FUNCTION(max,                0, 1, 1, nullptr          ),
		AGGREGATE(max,               1, 1, 1, MinMaxStep,      MinMaxFinalize),
		FUNCTION2(typeof,            1, 0, 0, TypeOfFunc,  FUNC_TYPEOF),
		FUNCTION2(length,            1, 0, 0, LengthFunc,  FUNC_LENGTH),
		FUNCTION(instr,              2, 0, 0, InstrFunc        ),
		FUNCTION(substr,             2, 0, 0, SubstrFunc       ),
		FUNCTION(substr,             3, 0, 0, SubstrFunc       ),
		FUNCTION(unicode,            1, 0, 0, UnicodeFunc      ),
		FUNCTION(char,              -1, 0, 0, CharFunc         ),
		FUNCTION(abs,                1, 0, 0, AbsFunc          ),
#ifndef FLOATING_POINT
		FUNCTION(round,              1, 0, 0, RoundFunc        ),
		FUNCTION(round,              2, 0, 0, RoundFunc        ),
#endif
		FUNCTION(upper,              1, 0, 0, UpperFunc        ),
		FUNCTION(lower,              1, 0, 0, LowerFunc        ),
		FUNCTION(coalesce,           1, 0, 0, nullptr          ),
		FUNCTION(coalesce,           0, 0, 0, nullptr          ),
		FUNCTION2(coalesce,         -1, 0, 0, IfNullFunc,  FUNC_COALESCE),
		FUNCTION(hex,                1, 0, 0, HexFunc          ),
		FUNCTION2(ifnull,            2, 0, 0, IfNullFunc,  FUNC_COALESCE),
		FUNCTION(random,             0, 0, 0, RandomFunc       ),
		FUNCTION(randomblob,         1, 0, 0, RandomBlob       ),
		FUNCTION(nullif,             2, 0, 1, NullIfFunc       ),
		FUNCTION(sqlite_version,     0, 0, 0, VersionFunc      ),
		FUNCTION(sqlite_source_id,   0, 0, 0, SourceIdFunc     ),
		FUNCTION(sqlite_log,         2, 0, 0, ErrlogFunc       ),
#ifndef OMIT_COMPILEOPTION_DIAGS
		FUNCTION(sqlite_compileoption_used,1, 0, 0, CompileoptionusedFunc),
		FUNCTION(sqlite_compileoption_get, 1, 0, 0, CompileoptiongetFunc),
#endif
		FUNCTION(quote,              1, 0, 0, QuoteFunc        ),
		FUNCTION(last_insert_rowid,  0, 0, 0, LastInsertRowid  ),
		FUNCTION(changes,            0, 0, 0, Changes          ),
		FUNCTION(total_changes,      0, 0, 0, TotalChanges     ),
		FUNCTION(replace,            3, 0, 0, ReplaceFunc      ),
		FUNCTION(zeroblob,           1, 0, 0, ZeroBlobFunc     ),
#ifdef SOUNDEX
		FUNCTION(soundex,            1, 0, 0, SoundexFunc      ),
#endif
#ifndef OMIT_LOAD_EXTENSION
		FUNCTION(load_extension,     1, 0, 0, LoadExtFunc      ),
		FUNCTION(load_extension,     2, 0, 0, LoadExtFunc      ),
#endif
		AGGREGATE(sum,               1, 0, 0, SumStep,         SumFinalize),
		AGGREGATE(total,             1, 0, 0, SumStep,         TotalFinalize),
		AGGREGATE(avg,               1, 0, 0, SumStep,         AvgFinalize),
		//AGGREGATE(count,             0, 0, 0, CountStep,       CountFinalize),
		{0, TEXTENCODE_UTF8, FUNC_COUNT, 0, 0, 0, CountStep, CountFinalize, "count", 0, 0}, 
		AGGREGATE(count,             1, 0, 0, CountStep,       CountFinalize),
		AGGREGATE(group_concat,      1, 0, 0, GroupConcatStep, GroupConcatFinalize),
		AGGREGATE(group_concat,      2, 0, 0, GroupConcatStep, GroupConcatFinalize),
		LIKEFUNC(glob, 2, &_globInfo, FUNC_LIKE|FUNC_CASE),
#ifdef CASE_SENSITIVE_LIKE
		LIKEFUNC(like, 2, &_likeInfoAlt, FUNC_LIKE|FUNC_CASE),
		LIKEFUNC(like, 3, &_likeInfoAlt, FUNC_LIKE|FUNC_CASE),
#else
		LIKEFUNC(like, 2, &_likeInfoNorm, FUNC_LIKE),
		LIKEFUNC(like, 3, &_likeInfoNorm, FUNC_LIKE),
#endif
	};

	__device__ void Func::RegisterGlobalFunctions()
		{
			FuncDefHash *hash = &Main_GlobalFunctions;
			FuncDef *funcs = (FuncDef *)&_GLOBAL(FuncDef, g_builtinFuncs);
			for (int i = 0; i < _lengthof(g_builtinFuncs); i++)
				Callback::FuncDefInsert(hash, &funcs[i]);
			Date_::RegisterDateTimeFunctions();
#ifndef OMIT_ALTERTABLE
			Alter::Functions();
#endif
		}

}}