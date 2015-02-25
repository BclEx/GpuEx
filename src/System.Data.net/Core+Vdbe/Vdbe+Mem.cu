// vdbemem.cu
#include "VdbeInt.cu.h"

namespace Core {

#pragma region Memory

	__device__ RC Vdbe::ChangeEncoding(Mem *mem, TEXTENCODE newEncode)
	{
		_assert((mem->Flags & MEM_RowSet) == 0);
		_assert(newEncode == TEXTENCODE_UTF8 || newEncode == TEXTENCODE_UTF16LE || newEncode == TEXTENCODE_UTF16BE);
		if (!(mem->Flags & MEM_Str) || mem->Encode == newEncode)
			return RC_OK;
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex) );
#ifdef OMIT_UTF16
		return RC_ERROR;
#else
		// MemTranslate() may return SQLITE_OK or SQLITE_NOMEM. If NOMEM is returned, then the encoding of the value may not have changed.
		RC rc = MemTranslate(mem, newEncode);
		_assert(rc == RC_OK || rc == RC_NOMEM);
		_assert(rc == RC_OK || mem->Encode != newEncode);
		_assert(rc == RC_NOMEM || mem->Encode == newEncode);
		return rc;
#endif
	}

	__device__ RC Vdbe::MemGrow(Mem *mem, size_t newSize, bool preserve)
	{
		_assert((mem->Malloc && mem->Malloc == mem->Z ? 1 : 0) +
			((mem->Flags & MEM_Dyn) && mem->Del ? 1 : 0) + 
			((mem->Flags & MEM_Ephem) ? 1 : 0) + 
			((mem->Flags & MEM_Static) ? 1 : 0) <= 1);
		_assert((mem->Flags & MEM_RowSet) == 0);

		// If the preserve flag is set to true, then the memory cell must already contain a valid string or blob value.
		_assert(!preserve || mem->Flags & (MEM_Blob | MEM_Str));

		if (newSize < 32) newSize = 32;
		if (_tagallocsize(mem->Ctx, mem->Malloc) < newSize)
			if (preserve && mem->Z == mem->Malloc)
			{
				mem->Z = mem->Malloc = (char *)_tagrealloc_or_free(mem->Ctx, mem->Z, newSize);
				preserve = false;
			}
			else
			{
				_tagfree(mem->Ctx, mem->Malloc);
				mem->Malloc = (char *)_tagalloc(mem->Ctx, newSize);
			}

			if (mem->Z && preserve && mem->Malloc && mem->Z != mem->Malloc)
				_memcpy(mem->Malloc, mem->Z, mem->N);
			if (mem->Flags & MEM_Dyn && mem->Del)
			{
				_assert(mem->Del != DESTRUCTOR_DYNAMIC);
				mem->Del((void *)(mem->Z));
			}

			mem->Z = mem->Malloc;
			mem->Flags = (MEM)(!mem->Z ? MEM_Null : mem->Flags & ~(MEM_Ephem | MEM_Static));
			mem->Del = nullptr;
			return (mem->Z ? RC_OK : RC_NOMEM);
	}

	__device__ RC Vdbe::MemMakeWriteable(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert((mem->Flags & MEM_RowSet) == 0);
		ExpandBlob(mem);
		MEM f = mem->Flags;
		if ((f & (MEM_Str | MEM_Blob)) && mem->Z != mem->Malloc)
		{
			if (MemGrow(mem, mem->N + 2, 1))
				return RC_NOMEM;
			mem->Z[mem->N] = 0;
			mem->Z[mem->N + 1] = 0;
			mem->Flags |= MEM_Term;
#ifdef _DEBUG
			mem->ScopyFrom = nullptr;
#endif
		}
		return RC_OK;
	}

#ifndef OMIT_INCRBLOB
	__device__ RC Vdbe::MemExpandBlob(Mem *mem)
	{
		if (mem->Flags & MEM_Zero)
		{
			_assert(mem->Flags & MEM_Blob);
			_assert((mem->Flags & MEM_RowSet) == 0);
			_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
			// Set nByte to the number of bytes required to store the expanded blob.
			int bytes = mem->N + mem->u.Zeros;
			if (bytes <= 0)
				bytes = 1;
			if (MemGrow(mem, bytes, true))
				return RC_NOMEM;
			_memset(&mem->Z[mem->N], 0, mem->u.Zeros);
			mem->N += mem->u.Zeros;
			mem->Flags &= ~(MEM_Zero | MEM_Term);
		}
		return RC_OK;
	}
#endif

	__device__ RC Vdbe::MemNulTerminate(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		if ((mem->Flags & MEM_Term) != 0 || (mem->Flags & MEM_Str) == 0)
			return RC_OK; // Nothing to do
		if (MemGrow(mem, mem->N + 2, true))
			return RC_NOMEM;
		mem->Z[mem->N] = 0;
		mem->Z[mem->N + 1] = 0;
		mem->Flags |= MEM_Term;
		return RC_OK;
	}

	__device__ RC Vdbe::MemStringify(Mem *mem, TEXTENCODE encode)
	{
		MEM f = mem->Flags;
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert(!(f & MEM_Zero));
		_assert(!(f & (MEM_Str | MEM_Blob)));
		_assert(f & (MEM_Int | MEM_Real));
		_assert((mem->Flags & MEM_RowSet) == 0);
		_assert(_HASALIGNMENT8(mem));

		const int bytes = 32;
		if (MemGrow(mem, bytes, false))
			return RC_NOMEM;

		// For a Real or Integer, use sqlite3_mprintf() to produce the UTF-8 string representation of the value. Then, if the required encoding
		// is UTF-16le or UTF-16be do a translation.
		// FIX ME: It would be better if sqlite3_snprintf() could do UTF-16.
		if (f & MEM_Int)
			__snprintf(mem->Z, bytes, "%lld", mem->u.I);
		else
		{
			_assert(f & MEM_Real);
			__snprintf(mem->Z, bytes, "%!.15g", mem->R);
		}
		mem->N = _strlen30(mem->Z);
		mem->Encode = TEXTENCODE_UTF8;
		mem->Flags |= MEM_Str|MEM_Term;
		ChangeEncoding(mem, encode);
		return RC_OK;
	}

	__device__ RC Vdbe::MemFinalize(Mem *mem, FuncDef *func)
	{
		RC rc = RC_OK;
		if (_ALWAYS(func && func->Finalize))
		{
			_assert((mem->Flags & MEM_Null) != 0 || func == mem->u.Def);
			_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
			FuncContext ctx;
			_memset(&ctx, 0, sizeof(ctx));
			ctx.S.Flags = MEM_Null;
			ctx.S.Ctx = mem->Ctx;
			ctx.Mem = mem;
			ctx.Func = func;
			func->Finalize(&ctx); // IMP: R-24505-23230
			_assert((mem->Flags & MEM_Dyn) == 0 && !mem->Del);
			_tagfree(mem->Ctx, mem->Malloc);
			_memcpy(mem, &ctx.S, sizeof(ctx.S));
			rc = ctx.IsError;
		}
		return rc;
	}

	__device__ void Vdbe::MemReleaseExternal(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		if (mem->Flags & MEM_Agg)
		{
			MemFinalize(mem, mem->u.Def);
			_assert((mem->Flags & MEM_Agg) == 0);
			MemRelease(mem);
		}
		else if (mem->Flags & MEM_Dyn && mem->Del)
		{
			_assert((mem->Flags & MEM_RowSet) == 0);
			_assert(mem->Del != DESTRUCTOR_DYNAMIC);
			mem->Del((void *)mem->Z);
			mem->Del = nullptr;
		}
		else if (mem->Flags & MEM_RowSet) RowSet_Clear(mem->u.RowSet);
		else if (mem->Flags & MEM_Frame) MemSetNull(mem);
	}

	__device__ void Vdbe::MemRelease(Mem *mem)
	{
		VdbeMemRelease(mem);
		_tagfree(mem->Ctx, mem->Malloc);
		mem->Z = mem->Malloc = nullptr;
		mem->Del = nullptr;
	}

	__device__ static int64 DoubleToInt64(double r)
	{
#ifdef OMIT_FLOATING_POINT
		return r; // When floating-point is omitted, double and int64 are the same thing
#else
		if (r < (double)SMALLEST_INT64) return SMALLEST_INT64;
		// minInt is correct here - not maxInt.  It turns out that assigning a very large positive number to an integer results in a very large
		// negative integer.  This makes no sense, but it is what x86 hardware does so for compatibility we will do the same in software.
		else if (r > (double)LARGEST_INT64) return SMALLEST_INT64;
		else return (int64)r;
#endif
	}

	__device__ int64 Vdbe::IntValue(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert(_HASALIGNMENT8(mem));
		MEM flags = mem->Flags;
		if (flags & MEM_Int) return mem->u.I;
		else if (flags & MEM_Real) return DoubleToInt64(mem->R);
		else if (flags & (MEM_Str | MEM_Blob))
		{
			_assert(mem->Z || mem->Z == 0);
			ASSERTCOVERAGE(mem->Z == 0);
			int64 value = 0;
			ConvertEx::Atoi64(mem->Z, &value, mem->N, mem->Encode);
			return value;
		}
		return 0;
	}

	__device__ double Vdbe::RealValue(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert(_HASALIGNMENT8(mem));
		if (mem->Flags & MEM_Real) return mem->R;
		else if (mem->Flags & MEM_Int) return (double)mem->u.I;
		else if (mem->Flags & (MEM_Str | MEM_Blob)) { double val = (double)0; ConvertEx::Atof(mem->Z, &val, mem->N, mem->Encode); return val; } // (double)0 In case of SQLITE_OMIT_FLOATING_POINT...
		return (double)0; // (double)0 In case of SQLITE_OMIT_FLOATING_POINT...
	}

	void Vdbe::IntegerAffinity(Mem *mem)
	{
		_assert(mem->Flags & MEM_Real);
		_assert((mem->Flags & MEM_RowSet) == 0);
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert(_HASALIGNMENT8(mem));
		mem->u.I = DoubleToInt64(mem->R);
		// Only mark the value as an integer if
		//    (1) the round-trip conversion real->int->real is a no-op, and
		//    (2) The integer is neither the largest nor the smallest possible integer (ticket #3922)
		// The second and third terms in the following conditional enforces the second condition under the assumption that addition overflow causes
		// values to wrap around.  On x86 hardware, the third term is always true and could be omitted.  But we leave it in because other
		// architectures might behave differently.
		if (mem->R == (double)mem->u.I
			&& mem->u.I > SMALLEST_INT64
#if defined(__i486__) || defined(__x86_64__)
			&& _ALWAYS(mem->u.I < LARGEST_INT64)
#else
			&& mem->u.I < LARGEST_INT64
#endif
			)
			mem->Flags |= MEM_Int;
	}

	__device__ RC Vdbe::MemIntegerify(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert((mem->Flags & MEM_RowSet) == 0);
		_assert(_HASALIGNMENT8(mem));
		mem->u.I = IntValue(mem);
		MemSetTypeFlag(mem, MEM_Int);
		return RC_OK;
	}

	__device__ RC Vdbe::MemRealify(Mem *mem)
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert(_HASALIGNMENT8(mem));
		mem->R = RealValue(mem);
		MemSetTypeFlag(mem, MEM_Real);
		return RC_OK;
	}

	__device__ RC Vdbe::MemNumerify(Mem *mem)
	{
		if ((mem->Flags & (MEM_Int | MEM_Real | MEM_Null)) == 0)
		{
			_assert((mem->Flags & (MEM_Blob | MEM_Str)) != 0);
			_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
			if (ConvertEx::Atoi64(mem->Z, &mem->u.I, mem->N, mem->Encode) == 0)
				MemSetTypeFlag(mem, MEM_Int);
			else
			{
				mem->R = RealValue(mem);
				MemSetTypeFlag(mem, MEM_Real);
				IntegerAffinity(mem);
			}
		}
		_assert((mem->Flags & (MEM_Int | MEM_Real | MEM_Null)) != 0);
		mem->Flags &= ~(MEM_Str | MEM_Blob);
		return RC_OK;
	}

	__device__ void Vdbe::MemSetNull(Mem *mem)
	{
		if (mem->Flags & MEM_Frame)
		{
			VdbeFrame *frame = mem->u.Frame;
			frame->Parent = frame->V->DelFrames;
			frame->V->DelFrames = frame;
		}
		if (mem->Flags & MEM_RowSet)
			RowSet_Clear(mem->u.RowSet);
		MemSetTypeFlag(mem, MEM_Null);
		mem->Type = TYPE_NULL;
	}

	__device__ void Vdbe::MemSetZeroBlob(Mem *mem, int n)
	{
		MemRelease(mem);
		mem->Flags = (MEM)(MEM_Blob | MEM_Zero);
		mem->Type = TYPE_BLOB;
		mem->N = 0;
		if (n < 0) n = 0;
		mem->u.Zeros = n;
		mem->Encode = TEXTENCODE_UTF8;
#ifdef OMIT_INCRBLOB
		MemGrow(mem, n, 0);
		if (mem->Z)
		{
			mem->N = n;
			_memset(mem->Z, 0, n);
		}
#endif
	}

	__device__ void Vdbe::MemSetInt64(Mem *mem, int64 value)
	{
		MemRelease(mem);
		mem->u.I = value;
		mem->Flags = MEM_Int;
		mem->Type = TYPE_INTEGER;
	}

#ifndef OMIT_FLOATING_POINT
	__device__ void Vdbe::MemSetDouble(Mem *mem, double value)
	{
		if (_isnan(value))
			MemSetNull(mem);
		else
		{
			MemRelease(mem);
			mem->R = value;
			mem->Flags = MEM_Real;
			mem->Type = TYPE_FLOAT;
		}
	}
#endif

	__device__ void Vdbe::MemSetRowSet(Mem *mem)
	{
		Context *ctx = mem->Ctx;
		_assert(ctx);
		_assert((mem->Flags & MEM_RowSet) == 0);
		MemRelease(mem);
		mem->Malloc = (char *)_tagalloc(ctx, 64);
		if (ctx->MallocFailed)
			mem->Flags = MEM_Null;
		else
		{
			_assert(mem->Malloc);
			mem->u.RowSet = RowSet_Init(ctx, mem->Malloc, _tagallocsize(ctx, mem->Malloc));
			_assert(mem->u.RowSet != nullptr);
			mem->Flags = MEM_RowSet;
		}
	}

	__device__ bool Vdbe::MemTooBig(Mem *mem)
	{
		_assert(mem->Ctx);
		if (mem->Flags & (MEM_Str | MEM_Blob))
		{
			int n = mem->N;
			if (mem->Flags & MEM_Zero)
				n += mem->u.Zeros;
			return (n > mem->Ctx->Limits[LIMIT_LENGTH]);
		}
		return false; 
	}

#ifdef _DEBUG
	__device__ void Vdbe::MemAboutToChange(Vdbe *p, Mem *mem)
	{
		Mem *x;
		int i;
		for (i = 1, x = &p->Mems.data[1]; i <= p->Mems.length; i++, x++)
		{
			if (x->ScopyFrom == mem)
			{
				x->Flags |= MEM_Invalid;
				x->ScopyFrom = nullptr;
			}
		}
		mem->ScopyFrom = nullptr;
	}
#endif

#define MEMCELLSIZE (size_t)(&(((Mem *)0)->Malloc)) // Size of struct Mem not including the Mem.zMalloc member.
	__device__ void Vdbe::MemShallowCopy(Mem *to, const Mem *from, uint16 srcType)
	{
		_assert((from->Flags & MEM_RowSet) == 0);
		VdbeMemRelease(to);
		_memcpy(to, from, MEMCELLSIZE);
		to->Del = nullptr;
		if ((from->Flags & MEM_Static) == 0)
		{
			to->Flags &= ~(MEM_Dyn | MEM_Static | MEM_Ephem);
			_assert(srcType == MEM_Ephem || srcType == MEM_Static);
			to->Flags |= srcType;
		}
	}

	__device__ RC Vdbe::MemCopy(Mem *to, const Mem *from)
	{
		_assert((from->Flags & MEM_RowSet) == 0);
		VdbeMemRelease(to);
		_memcpy(to, from, MEMCELLSIZE);
		to->Flags &= ~MEM_Dyn;
		RC rc = RC_OK;
		if (to->Flags & (MEM_Str | MEM_Blob))
		{
			if ((from->Flags & MEM_Static) == 0)
			{
				to->Flags |= MEM_Ephem;
				rc = MemMakeWriteable(to);
			}
		}
		return rc;
	}

	__device__ void Vdbe::MemMove(Mem *to, Mem *from)
	{
		_assert(!from->Ctx || MutexEx::Held(from->Ctx->Mutex));
		_assert(!to->Ctx || MutexEx::Held(to->Ctx->Mutex));
		_assert(!from->Ctx || !to->Ctx || from->Ctx == to->Ctx);
		MemRelease(to);
		_memcpy(to, from, sizeof(Mem));
		from->Flags = MEM_Null;
		from->Del = nullptr;
		from->Malloc = nullptr;
	}

	__device__ RC Vdbe::MemSetStr(Mem *mem, const char *z, int n, TEXTENCODE encode, void (*del)(void *))
	{
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert((mem->Flags & MEM_RowSet) == 0);
		// If z is a NULL pointer, set pMem to contain an SQL NULL.
		if (!z)
		{
			MemSetNull(mem);
			return RC_OK;
		}
		int  limit = (mem->Ctx ? mem->Ctx->Limits[LIMIT_LENGTH] : CORE_MAX_LENGTH); // Maximum allowed string or blob size
		MEM flags = (encode == 0 ? MEM_Blob : MEM_Str); // New value for pMem->flags
		int bytes = n; // New value for pMem->n
		if (bytes < 0)
		{
			_assert(encode != 0);
			if (encode == TEXTENCODE_UTF8)
				for (bytes = 0; bytes <= limit && z[bytes]; bytes++) { }
			else
				for (bytes = 0; bytes <= limit && (z[bytes] | z[bytes + 1]); bytes += 2) { }
				flags |= MEM_Term;
		}

		// The following block sets the new values of Mem.z and Mem.xDel. It also sets a flag in local variable "flags" to indicate the memory
		// management (one of MEM_Dyn or MEM_Static).
		if (del == DESTRUCTOR_TRANSIENT)
		{
			int alloc = bytes;
			if (flags & MEM_Term) alloc += (encode == TEXTENCODE_UTF8 ? 1 : 2);
			if (bytes > limit) return RC_TOOBIG;
			if (MemGrow(mem, alloc, false)) return RC_NOMEM;
			_memcpy(mem->Z, z, alloc);
		}
		else if (del == DESTRUCTOR_DYNAMIC)
		{
			MemRelease(mem);
			mem->Malloc = mem->Z = (char *)z;
			mem->Del = nullptr;
		}
		else
		{
			MemRelease(mem);
			mem->Z = (char *)z;
			mem->Del = del;
			flags |= (del == (void *)DESTRUCTOR_STATIC ? MEM_Static : MEM_Dyn);
		}
		mem->N = bytes;
		mem->Flags = flags;
		mem->Encode = (encode == 0 ? TEXTENCODE_UTF8 : encode);
		mem->Type = (encode == 0 ? TYPE_BLOB : TYPE_TEXT);

#ifndef OMIT_UTF16
		if (mem->Encode != TEXTENCODE_UTF8 && MemHandleBom(mem)) return RC_NOMEM;
#endif
		return (bytes > limit ? RC_TOOBIG : RC_OK);
	}

	__device__ int Vdbe::MemCompare(const Mem *mem1, const Mem *mem2, const CollSeq *coll)
	{
		MEM f1 = mem1->Flags;
		MEM f2 = mem2->Flags;
		MEM cf = (MEM)(f1 | f2);
		_assert((cf & MEM_RowSet) == 0);

		// If one value is NULL, it is less than the other. If both values are NULL, return 0.
		if (cf & MEM_Null)
			return (f2 & MEM_Null) - (f1 &MEM_Null);

		// If one value is a number and the other is not, the number is less. If both are numbers, compare as reals if one is a real, or as integers if both values are integers.
		if (cf & (MEM_Int | MEM_Real))
		{
			if (!(f1 & (MEM_Int | MEM_Real))) return 1;
			if (!(f2 & (MEM_Int | MEM_Real))) return -1;
			if ((f1 & f2 & MEM_Int) == 0)
			{
				double r1 = ((f1 & MEM_Real) == 0 ? (double)mem1->u.I : mem1->R);
				double r2 = ((f2 & MEM_Real) == 0 ? (double)mem2->u.I : mem2->R);
				if (r1 < r2) return -1;
				if (r1 > r2) return 1;
				return 0;
			}
			_assert(f1 & MEM_Int);
			_assert(f2 & MEM_Int);
			if (mem1->u.I < mem2->u.I) return -1;
			if (mem1->u.I > mem2->u.I) return 1;
			return 0;
		}

		// If one value is a string and the other is a blob, the string is less. If both are strings, compare using the collating functions.
		int r;
		if (cf & MEM_Str)
		{
			if ((f1 & MEM_Str) == 0) return 1;
			if ((f2 & MEM_Str) == 0) return -1;
			_assert(mem1->Encode == mem2->Encode);
			_assert(mem1->Encode == TEXTENCODE_UTF8 || mem1->Encode == TEXTENCODE_UTF16LE || mem1->Encode == TEXTENCODE_UTF16BE);
			// The collation sequence must be defined at this point, even if the user deletes the collation sequence after the vdbe program is
			// compiled (this was not always the case).
			_assert(!coll || coll->Cmp);
			if (coll)
			{
				if (mem1->Encode == coll->Encode)
					return coll->Cmp(coll->User, mem1->N, mem1->Z, mem2->N, mem2->Z); // The strings are already in the correct encoding.  Call the comparison function directly
				Mem c1; _memset(&c1, 0, sizeof(c1));
				Mem c2; _memset(&c2, 0, sizeof(c2));
				Vdbe::MemShallowCopy(&c1, mem1, MEM_Ephem);
				Vdbe::MemShallowCopy(&c2, mem2, MEM_Ephem);
				const void *v1 = Vdbe::ValueText(&c1, coll->Encode);
				int n1 = (!v1 ? 0 : c1.N);
				const void *v2 = Vdbe::ValueText(&c2, coll->Encode);
				int n2 = (!v2 ? 0 : c2.N);
				r = coll->Cmp(coll->User, n1, v1, n2, v2);
				Vdbe::MemRelease(&c1);
				Vdbe::MemRelease(&c2);
				return r;
			}
			// If a NULL pointer was passed as the collate function, fall through to the blob case and use memcmp().
		}
		// Both values must be blobs.  Compare using memcmp().
		r = _memcmp(mem1->Z, mem2->Z, (mem1->N > mem2->N ? mem2->N : mem1->N));
		if (r == 0)
			r = mem1->N - mem2->N;
		return r;
	}

	__device__ RC Vdbe::MemFromBtree(BtCursor *cur, int offset, int amount, bool key, Mem *mem)
	{
		_assert(Btree::CursorIsValid(cur));

		// Note: the calls to BtreeKeyFetch() and DataFetch() below assert() that both the BtShared and database handle mutexes are held.
		_assert((mem->Flags & MEM_RowSet) == 0);
		int available = 0; // Number of bytes available on the local btree page
		char *data = (char *)(key ? Btree::KeyFetch(cur, &available) : Btree::DataFetch(cur, &available)); // Data from the btree layer
		_assert(data);

		RC rc = RC_OK;
		if (offset + amount <= available && (mem->Flags & MEM_Dyn) == 0)
		{
			MemRelease(mem);
			mem->Z = &data[offset];
			mem->Flags = (MEM)(MEM_Blob | MEM_Ephem);
		}
		else if ((rc = MemGrow(mem, amount + 2, false)) == RC_OK)
		{
			mem->Flags = (MEM)(MEM_Blob | MEM_Dyn | MEM_Term);
			mem->Encode = (TEXTENCODE)0;
			mem->Type = TYPE_BLOB;
			rc = (key ? Btree::Key(cur, offset, amount, mem->Z) : Btree::Data(cur, offset, amount, mem->Z));
			mem->Z[amount] = 0;
			mem->Z[amount + 1] = 0;
			if (rc != RC_OK)
				MemRelease(mem);
		}
		mem->N = amount;
		return rc;
	}

#pragma region Value

	__device__ const void *Vdbe::ValueText(Mem *mem, TEXTENCODE encode)
	{
		if (!mem) return nullptr;
		_assert(!mem->Ctx || MutexEx::Held(mem->Ctx->Mutex));
		_assert((encode & 3) == (encode & ~TEXTENCODE_UTF16_ALIGNED));
		_assert((mem->Flags & MEM_RowSet) == 0);
		if (mem->Flags & MEM_Null) return nullptr;
		_assert((MEM_Blob >> 3) == MEM_Str);
		mem->Flags |= (mem->Flags & MEM_Blob) >> 3;
		ExpandBlob(mem);
		if (mem->Flags & MEM_Str)
		{
			ChangeEncoding(mem, (TEXTENCODE)(encode & ~TEXTENCODE_UTF16_ALIGNED));
			if ((encode & TEXTENCODE_UTF16_ALIGNED) != 0 && (1 & PTR_TO_INT(mem->Z)) == 1)
			{
				_assert((mem->Flags & (MEM_Ephem | MEM_Static)) != 0);
				if (MemMakeWriteable(mem) != RC_OK) return nullptr;
			}
			MemNulTerminate(mem); // IMP: R-31275-44060
		}
		else
		{
			_assert((mem->Flags & MEM_Blob) == 0);
			MemStringify(mem, encode);
			_assert((1 & PTR_TO_INT(mem->Z)) == 0);
		}
		_assert(mem->Encode == (encode & ~TEXTENCODE_UTF16_ALIGNED) || !mem->Ctx || mem->Ctx->MallocFailed);
		return (mem->Encode == (encode & ~TEXTENCODE_UTF16_ALIGNED) ? mem->Z : nullptr);
	}


	__device__ Mem *Vdbe::ValueNew(Context *ctx)
	{
		Mem *p = (Mem *)_tagalloc2(ctx, sizeof(*p), true);
		if (p)
		{
			p->Flags = MEM_Null;
			p->Type = TYPE_NULL;
			p->Ctx = ctx;
		}
		return p;
	}

	__device__ RC Vdbe::ValueFromExpr(Context *ctx, Expr *expr, TEXTENCODE encode, AFF affinity, Mem **value)
	{
		if (!expr)
		{
			*value = nullptr;
			return RC_OK;
		}
		TK op = expr->OP;
		// op can only be TK_REGISTER if we have compiled with SQLITE_ENABLE_STAT3. The ifdef here is to enable us to achieve 100% branch test coverage even when SQLITE_ENABLE_STAT3 is omitted.
#ifdef ENABLE_STAT3
		if (op == TK_REGISTER) op = expr->OP2;
#else
		if (_NEVER(op == TK_REGISTER)) op = expr->OP2;
#endif

		// Handle negative integers in a single step.  This is needed in the case when the value is -9223372036854775808.
		int negInt = 1;
		const char *neg = "";
		if (op == TK_UMINUS && (expr->Left->OP == TK_INTEGER || expr->Left->OP == TK_FLOAT))
		{
			expr = expr->Left;
			op = expr->OP;
			negInt = -1;
			neg = "-";
		}

		Mem *mem = nullptr;
		char *memAsString = nullptr;
		if (op == TK_STRING || op == TK_FLOAT || op == TK_INTEGER)
		{
			mem = ValueNew(ctx);
			if (!mem) goto no_mem;
			if (ExprHasProperty(expr, EP_IntValue))
				MemSetInt64(mem, (int64)expr->u.I * negInt);
			else
			{
				memAsString = _mtagprintf(ctx, "%s%s", neg, expr->u.Token);
				if (!memAsString) goto no_mem;
				ValueSetStr(mem, -1, memAsString, TEXTENCODE_UTF8, DESTRUCTOR_DYNAMIC);
				if (op == TK_FLOAT) mem->Type = TYPE_FLOAT;
			}
			if ((op == TK_INTEGER || op == TK_FLOAT) && affinity == AFF_NONE)
				ValueApplyAffinity(mem, AFF_NUMERIC, TEXTENCODE_UTF8);
			else
				ValueApplyAffinity(mem, affinity, TEXTENCODE_UTF8);
			if (mem->Flags & (MEM_Int | MEM_Real)) mem->Flags &= ~MEM_Str;
			if (encode != TEXTENCODE_UTF8)
				ChangeEncoding(mem, encode);
		}
		else if (op == TK_UMINUS)
		{
			// This branch happens for multiple negative signs.  Ex: -(-5)
			if (ValueFromExpr(ctx, expr->Left, encode, affinity, &mem) == RC_OK)
			{
				Vdbe::MemNumerify(mem);
				if (mem->u.I == SMALLEST_INT64)
				{
					mem->Flags &= MEM_Int;
					mem->Flags |= MEM_Real;
					mem->R = (double)LARGEST_INT64;
				}
				else
					mem->u.I = -mem->u.I;
				mem->R = -mem->R;
				ValueApplyAffinity(mem, affinity, encode);
			}
		}
		else if (op == TK_NULL)
		{
			mem = ValueNew(ctx);
			if (!mem) goto no_mem;
		}
#ifndef OMIT_BLOB_LITERAL
		else if (op == TK_BLOB)
		{
			_assert(expr->u.Token[0] == 'x' || expr->u.Token[0] == 'X');
			_assert(expr->u.Token[1] == '\'');
			mem = ValueNew(ctx);
			if (!mem) goto no_mem;
			memAsString = &expr->u.Token[2];
			int memAsStringLength = _strlen30(memAsString) - 1;
			_assert(memAsString[memAsStringLength] == '\'');
			MemSetStr(mem, (const char *)_taghextoblob(ctx, memAsString, memAsStringLength), memAsStringLength / 2, (TEXTENCODE)0, DESTRUCTOR_DYNAMIC);
		}
#endif
		if (mem)
			MemStoreType(mem);
		*value = mem;
		return RC_OK;

no_mem:
		ctx->MallocFailed = true;
		_tagfree(ctx, memAsString);
		ValueFree(mem);
		*value = nullptr;
		return RC_NOMEM;
	}

	__device__ void Vdbe::ValueSetStr(Mem *mem, int n, const void *z, TEXTENCODE encode, void (*del)(void *))
	{
		if (mem) MemSetStr(mem, (const char *)z, n, encode, del);
	}

	__device__ void Vdbe::ValueFree(Mem *mem)
	{
		if (!mem) return;
		MemRelease(mem);
		_tagfree(mem->Ctx, mem);
	}

	__device__ int Vdbe::ValueBytes(Mem *mem, TEXTENCODE encode)
	{
		if ((mem->Flags & MEM_Blob) != 0 || ValueText(mem, encode))
			return (mem->Flags & MEM_Zero ? mem->N + mem->u.Zeros : mem->N);
		return 0;
	}

#pragma endregion

#pragma endregion
}