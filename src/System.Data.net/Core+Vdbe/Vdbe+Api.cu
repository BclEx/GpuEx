// vdbeapi.cu
#include "VdbeInt.cu.h"

#pragma region Name1

__device__ static bool VdbeSafety(Vdbe *p)
{
	if (p->Ctx == nullptr)
	{
		SysEx_LOG(RC_MISUSE, "API called with finalized prepared statement");
		return true;
	}
	return false;
}

__device__ static bool VdbeSafetyNotNull(Vdbe *p)
{
	if (p == nullptr)
	{
		SysEx_LOG(RC_MISUSE, "API called with NULL prepared statement");
		return true;
	}
	return VdbeSafety(p);
}

__device__ RC Vdbe::Finalize(Vdbe *p)
{
	if (p == nullptr)
		return RC_OK; // IMPLEMENTATION-OF: R-57228-12904 Invoking sqlite3_finalize() on a NULL pointer is a harmless no-op.
	Context *ctx = p->Ctx;
	if (VdbeSafety(p)) return SysEx_MISUSE_BKPT;
	MutexEx::Enter(ctx->Mutex);
	RC rc = p->Finalize();
	rc = Main::ApiExit(ctx, rc);
	Main::LeaveMutexAndCloseZombie(ctx);
	return rc;
}

__device__ RC Vdbe::Reset(Vdbe *p)
{
	if (p == nullptr)
		return RC_OK;
#if THREADSAFE
	MutexEx mutex = p->Ctx->Mutex;
#endif
	MutexEx::Enter(mutex);
	RC rc = p->Reset();
	p->Rewind();
	_assert((rc & p->Ctx->ErrMask) == rc);
	rc = Main::ApiExit(p->Ctx, rc);
	MutexEx::Leave(mutex);
	return rc;
}

__device__ RC Vdbe::ClearBindings()
{
#if THREADSAFE
	MutexEx mutex = Ctx->Mutex;
#endif
	MutexEx::Enter(mutex);
	for (int i = 0; i < Vars.length; i++)
	{
		Vdbe::MemRelease(&Vars[i]);
		Vars[i].Flags = MEM_Null;
	}
	if (IsPrepareV2 && Expmask)
		Expired = true;
	MutexEx::Leave(mutex);
	return RC_OK;
}

#pragma endregion

#pragma region Value
// The following routines extract information from a Mem or Mem structure.

__device__ const void *Vdbe::Value_Blob(Mem *p)
{
	if (p->Flags & (MEM_Blob|MEM_Str))
	{
		MemExpandBlob(p);
		p->Flags &= ~MEM_Str;
		p->Flags |= MEM_Blob;
		return (p->N ? p->Z : nullptr);
	}
	return Value_Text(p);
}
__device__ int Vdbe::Value_Bytes(Mem *p) { return ValueBytes(p, TEXTENCODE_UTF8); }
__device__ int Vdbe::Value_Bytes16(Mem *p) { return ValueBytes(p, TEXTENCODE_UTF16NATIVE); }
__device__ double Vdbe::Value_Double(Mem *p) { return RealValue(p); }
__device__ int Vdbe::Value_Int(Mem *p) { return (int)IntValue(p); }
__device__ int64 Vdbe::Value_Int64(Mem *p) { return IntValue(p); }
__device__ const unsigned char *Vdbe::Value_Text(Mem *p) { return (const unsigned char *)ValueText(p, TEXTENCODE_UTF8); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Value_Text16(Mem *p) { return ValueText(p, TEXTENCODE_UTF16NATIVE); }
__device__ const void *Vdbe::Value_Text16be(Mem *p) { return ValueText(p, TEXTENCODE_UTF16BE); }
__device__ const void *Vdbe::Value_Text16le(Mem *p) { return ValueText(p, TEXTENCODE_UTF16LE); }
#endif
__device__ TYPE Vdbe::Value_Type(Mem *p) { return p->Type; }

#pragma endregion

#pragma region Result
// The following routines are used by user-defined functions to specify the function result.
//
// The setStrOrError() funtion calls sqlite3VdbeMemSetStr() to store the result as a string or blob but if the string or blob is too large, it
// then sets the error code to SQLITE_TOOBIG

__device__ inline static void SetResultStrOrError(FuncContext *fctx, const char *z, int n, TEXTENCODE encode, void (*del)(void *))
{
	if (Vdbe::MemSetStr(&fctx->S, z, n, encode, del) == RC_TOOBIG)
		Vdbe::Result_ErrorOverflow(fctx);
}
__device__ void Vdbe::Result_Blob(FuncContext *fctx, const void *z, int n, void (*del)(void *))
{
	_assert(n >= 0);
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	SetResultStrOrError(fctx, (const char *)z, n, (TEXTENCODE)0, del);
}
__device__ void Vdbe::Result_Double(FuncContext *fctx, double value)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetDouble(&fctx->S, value);
}
__device__ void Vdbe::Result_Error(FuncContext *fctx, const char *z, int n)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	fctx->IsError = RC_ERROR;
	MemSetStr(&fctx->S, z, n, TEXTENCODE_UTF8, DESTRUCTOR_TRANSIENT);
}
#ifndef OMIT_UTF16
__device__ void Vdbe::Result_Error16(FuncContext *fctx, const void *z, int n)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	fctx->IsError = RC_ERROR;
	MemSetStr(&fctx->S, (const char *)z, n, TEXTENCODE_UTF16NATIVE, DESTRUCTOR_TRANSIENT);
}
#endif
__device__ void Vdbe::Result_Int(FuncContext *fctx, int value)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetInt64(&fctx->S, (int64)value);
}
__device__ void Vdbe::Result_Int64(FuncContext *fctx, int64 value)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetInt64(&fctx->S, value);
}
__device__ void Vdbe::Result_Null(FuncContext *fctx)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetNull(&fctx->S);
}
__device__ void Vdbe::Result_Text(FuncContext *fctx, const char *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	SetResultStrOrError(fctx, z, n, TEXTENCODE_UTF8, del);
}
#ifndef OMIT_UTF16
__device__ void Vdbe::Result_Text16(FuncContext *fctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	SetResultStrOrError(fctx, (const char *)z, n, TEXTENCODE_UTF16NATIVE, del);
}
__device__ void Vdbe::Result_Text16be(FuncContext *fctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	SetResultStrOrError(fctx, (const char *)z, n, TEXTENCODE_UTF16BE, del);
}
__device__ void Vdbe::Result_Text16le(FuncContext *fctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	SetResultStrOrError(fctx, (const char *)z, n, TEXTENCODE_UTF16LE, del);
}
#endif
__device__ void Vdbe::Result_Value(FuncContext *fctx, Mem *value)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemCopy(&fctx->S, value);
}
__device__ void Vdbe::Result_ZeroBlob(FuncContext *fctx, int n)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetZeroBlob(&fctx->S, n);
}
__device__ void Vdbe::Result_ErrorCode(FuncContext *fctx, RC errCode)
{
	fctx->IsError = errCode;
	if (fctx->S.Flags & MEM_Null)
		Vdbe::MemSetStr(&fctx->S, Main::ErrStr(errCode), -1, TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
}
__device__ void Vdbe::Result_ErrorOverflow(FuncContext *fctx)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	fctx->IsError = RC_TOOBIG;
	MemSetStr(&fctx->S, "string or blob too big", -1, TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
}
__device__ void Vdbe::Result_ErrorNoMem(FuncContext *fctx)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	MemSetNull(&fctx->S);
	fctx->IsError = RC_NOMEM;
	fctx->S.Ctx->MallocFailed = true;
}

#pragma endregion

#pragma region Step

__device__ static RC DoWalCallbacks(Context *ctx)
{
	RC rc = RC_OK;
#ifndef OMIT_WAL
	for (int i = 0; i < ctx->DBs.length; i++)
	{
		Btree *bt = ctx->DBs[i].Bt;
		if (bt)
		{
			int entrys = sqlite3PagerWalCallback(bt->get_Pager());
			if (ctx->WalCallback && entrys > 0 && rc == RC_OK)
				rc = ctx->WalCallback(ctx->WalArg, ctx, ctx->DBs[i].Name, entrys);
		}
	}
#endif
	return rc;
}

__device__ RC Vdbe::Step2()
{
	if (Magic != VDBE_MAGIC_RUN)
	{
		// We used to require that sqlite3_reset() be called before retrying sqlite3_step() after any error or after SQLITE_DONE.  But beginning
		// with version 3.7.0, we changed this so that sqlite3_reset() would be called automatically instead of throwing the SQLITE_MISUSE error.
		// This "automatic-reset" change is not technically an incompatibility, since any application that receives an SQLITE_MISUSE is broken by
		// definition.
		//
		// Nevertheless, some published applications that were originally written for version 3.6.23 or earlier do in fact depend on SQLITE_MISUSE 
		// returns, and those were broken by the automatic-reset change.  As a work-around, the SQLITE_OMIT_AUTORESET compile-time restores the
		// legacy behavior of returning SQLITE_MISUSE for cases where the previous sqlite3_step() returned something other than a SQLITE_LOCKED
		// or SQLITE_BUSY error.
#ifdef OMIT_AUTORESET
		if (RC == RC_BUSY || RC == RC_LOCKED)
			Reset(this);
		else
			return RC_MISUSE_BKPT;
#else
		Reset(this);
#endif
	}

	// Check that malloc() has not failed. If it has, return early.
	RC rc;
	Context *ctx = Ctx;
	if (ctx->MallocFailed)
	{
		RC_ = RC_NOMEM;
		return RC_NOMEM;
	}
	if (PC <= 0 && Expired)
	{
		RC_ = RC_SCHEMA;
		rc = RC_ERROR;
		goto end_of_step;
	}
	if (PC < 0)
	{
		// If there are no other statements currently running, then reset the interrupt flag.  This prevents a call to sqlite3_interrupt
		// from interrupting a statement that has not yet started.
		if (ctx->ActiveVdbeCnt == 0)
			ctx->u1.IsInterrupted = false;
		_assert(ctx->WriteVdbeCnt > 0 || !ctx->AutoCommit || !ctx->DeferredCons);
#ifndef OMIT_TRACE
		if (ctx->Profile && !ctx->Init.Busy)
			ctx->Vfs->CurrentTimeInt64(&StartTime);
#endif
		ctx->ActiveVdbeCnt++;
		if (!ReadOnly) ctx->WriteVdbeCnt++;
		PC = 0;
	}

#ifndef OMIT_EXPLAIN
	if (_explain)
		rc = List();
	else
#endif
	{
		ctx->VdbeExecCnt++;
		rc = Exec();
		ctx->VdbeExecCnt--;
	}

#ifndef OMIT_TRACE
	// Invoke the profile callback if there is one
	if (rc != RC_ROW && ctx->Profile && !ctx->Init.Busy && Sql_)
	{
		int64 now;
		ctx->Vfs->CurrentTimeInt64(&now);
		ctx->Profile(ctx->ProfileArg, Sql_, (now - StartTime)*1000000);
	}
#endif

	if (rc == RC_DONE)
	{
		_assert(RC_ == RC_OK);
		RC_ = DoWalCallbacks(ctx);
		if (RC_ != RC_OK)
			rc = RC_ERROR;
	}

	ctx->ErrCode = rc;
	if (Main::ApiExit(Ctx, RC_) == RC_NOMEM)
		RC_ = RC_NOMEM;

end_of_step:
	// At this point local variable rc holds the value that should be returned if this statement was compiled using the legacy 
	// sqlite3_prepare() interface. According to the docs, this can only be one of the values in the first assert() below. Variable p->rc 
	// contains the value that would be returned if sqlite3_finalize() were called on statement p.
	_assert(rc == RC_ROW || rc == RC_DONE || rc == RC_ERROR || rc == RC_BUSY || rc == RC_MISUSE);
	_assert(RC_ != RC_ROW && RC_ != RC_DONE);
	// If this statement was prepared using sqlite3_prepare_v2(), and an error has occurred, then return the error code in p->rc to the
	// caller. Set the error code in the database handle to the same value.
	if (IsPrepareV2 && rc != RC_ROW && rc != RC_DONE)
		rc = TransferError();
	return (RC)(rc & ctx->ErrMask);
}

#ifndef MAX_SCHEMA_RETRY
#define MAX_SCHEMA_RETRY 5
#endif

__device__ RC Vdbe::Step()
{
	RC rc = RC_OK; // Result from sqlite3Step()
	RC rc2 = RC_OK; // Result from sqlite3Reprepare()
	int cnt = 0; // Counter to prevent infinite loop of reprepares
	Context *ctx = Ctx; // The database connection
	MutexEx::Enter(ctx->Mutex);
	DoingRerun = false;
	while ((rc = Step2()) == RC_SCHEMA && cnt++ < MAX_SCHEMA_RETRY && (rc2 = rc = Prepare::Reprepare(this)) == RC_OK)
	{
		Reset(this);
		DoingRerun = true;
		_assert(!Expired);
	}
	if (rc2 != RC_OK && _ALWAYS(IsPrepareV2) && _ALWAYS(ctx->Err))
	{
		// This case occurs after failing to recompile an sql statement. The error message from the SQL compiler has already been loaded 
		// into the database handle. This block copies the error message from the database handle into the statement and sets the statement
		// program counter to 0 to ensure that when the statement is finalized or reset the parser error message is available via
		// sqlite3_errmsg() and sqlite3_errcode().
		const char *err = (const char *)Value_Text(ctx->Err); 
		_tagfree(ctx, ErrMsg);
		if (!ctx->MallocFailed) { ErrMsg = _tagstrdup(ctx, err); RC_ = rc2; }
		else { ErrMsg = nullptr; RC_ = rc = RC_NOMEM; }
	}
	rc = Main::ApiExit(ctx, rc);
	MutexEx::Leave(ctx->Mutex);
	return rc;
}

#pragma endregion

#pragma region Name3

__device__ void *Vdbe::User_Data(FuncContext *fctx)
{
	_assert(fctx && fctx->Func);
	return fctx->Func->UserData;
}

__device__ Context *Vdbe::Context_Ctx(FuncContext *fctx)
{
	_assert(fctx && fctx->Func);
	return fctx->S.Ctx;
}

__device__ void Vdbe::InvalidFunction(FuncContext *fctx, int notUsed1, Mem **notUsed2)
{
	const char *name = fctx->Func->Name;
	char *err = _mprintf("unable to use function %s in the requested context", name);
	Result_Error(fctx, err, -1);
	_free(err);
}

__device__ void *Vdbe::Aggregate_Context(FuncContext *fctx, int bytes)
{
	_assert(fctx && fctx->Func && fctx->Func->Step);
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	Mem *mem = fctx->Mem;
	ASSERTCOVERAGE(bytes < 0);
	if ((mem->Flags & MEM_Agg) == 0)
	{
		if (bytes <= 0)
		{
			MemReleaseExternal(mem);
			mem->Flags = MEM_Null;
			mem->Z = nullptr;
		}
		else
		{
			MemGrow(mem, bytes, 0);
			mem->Flags = MEM_Agg;
			mem->u.Def = fctx->Func;
			if (mem->Z)
				_memset(mem->Z, 0, bytes);
		}
	}
	return (void *)mem->Z;
}

__device__ void *Vdbe::get_Auxdata(FuncContext *fctx, int arg)
{
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	VdbeFunc *vdbeFunc = fctx->VdbeFunc;
	if (!vdbeFunc || arg >= vdbeFunc->AuxsLength || arg < 0)
		return nullptr;
	return vdbeFunc->Auxs[arg].Aux;
}

__device__ void Vdbe::set_Auxdata(FuncContext *fctx, int args, void *aux, void (*delete_)(void*))
{
	if (args < 0) goto failed;
	_assert(MutexEx::Held(fctx->S.Ctx->Mutex));
	VdbeFunc *vdbeFunc = fctx->VdbeFunc;
	if (!vdbeFunc || vdbeFunc->AuxsLength <= args)
	{
		int auxLength = (vdbeFunc ? vdbeFunc->AuxsLength : 0);
		int newSize = sizeof(VdbeFunc) + sizeof(VdbeFunc::AuxData)*args;
		vdbeFunc = (VdbeFunc *)_tagrealloc(fctx->S.Ctx, vdbeFunc, newSize);
		if (!vdbeFunc)
			goto failed;
		fctx->VdbeFunc = vdbeFunc;
		_memset(&vdbeFunc->Auxs[auxLength], 0, sizeof(VdbeFunc::AuxData)*(args+1-auxLength));
		vdbeFunc->AuxsLength = args+1;
		vdbeFunc->Func = fctx->Func;
	}
	VdbeFunc::AuxData *auxData = &vdbeFunc->Auxs[args];
	if (auxData->Aux && auxData->Delete)
		auxData->Delete(auxData->Aux);
	auxData->Aux = aux;
	auxData->Delete = delete_;
	return;
failed:
	if (delete_)
		delete_(aux);
}

__device__ int Vdbe::Column_Count(Vdbe *p)
{
	return (p ? p->ResColumns : 0);
}

__device__ int Vdbe::Data_Count(Vdbe *p)
{
	return (!p || !p->ResultSet ? 0 : p->ResColumns);
}

#pragma endregion

#pragma region Column
// The following routines are used to access elements of the current row in the result set.

// If the value passed as the second argument is out of range, return a pointer to the following static Mem object which contains the
// value SQL NULL. Even though the Mem structure contains an element of type i64, on certain architectures (x86) with certain compiler
// switches (-Os), gcc may align this Mem object on a 4-byte boundary instead of an 8-byte one. This all works fine, except that when
// running with SQLITE_DEBUG defined the SQLite code sometimes assert()s that a Mem structure is located on an 8-byte boundary. To prevent
// these assert()s from failing, when building with SQLITE_DEBUG defined using gcc, we force nullMem to be 8-byte aligned using the magical
// __attribute__((aligned(8))) macro.
__device__ static const Mem _nullMem 
#if defined(_DEBUG) && defined(__GNUC__)
	__attribute__((aligned(8))) 
#endif
	= {0, "", (double)0, {0}, 0, MEM_Null, TYPE_NULL, (TEXTENCODE)0,
#ifdef _DEBUG
	0, 0,  // scopyFrom, filler
#endif
	0, 0 };

__device__ static Mem *ColumnMem(Vdbe *p, int i)
{
	Mem *r;
	if (p && p->ResultSet != 0 && i < p->ResColumns && i >= 0)
	{
		MutexEx::Enter(p->Ctx->Mutex);
		r = &p->ResultSet[i];
	}
	else
	{
		if (p && _ALWAYS(p->Ctx))
		{
			MutexEx::Enter(p->Ctx->Mutex);
			Main::Error(p->Ctx, RC_RANGE, 0);
		}
		r = (Mem *)&_nullMem;
	}
	return r;
}

__device__ static void ColumnMallocFailure(Vdbe *p)
{
	// If malloc() failed during an encoding conversion within an sqlite3_column_XXX API, then set the return code of the statement to
	// RC_NOMEM. The next call to _step() (if any) will return RC_ERROR and _finalize() will return NOMEM.
	if (p)
	{
		p->RC_ = Main::ApiExit(p->Ctx, p->RC_);
		MutexEx::Leave(p->Ctx->Mutex);
	}
}

__device__ const void *Vdbe::Column_Blob(Vdbe *p, int i) { const void *val = Value_Blob(ColumnMem(p, i)); ColumnMallocFailure(p); return val; } // Even though there is no encoding conversion, value_blob() might need to call malloc() to expand the result of a zeroblob() expression. 
__device__ int Vdbe::Column_Bytes(Vdbe *p, int i) { int val = Value_Bytes(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ int Vdbe::Column_Bytes16(Vdbe *p, int i) { int val = Value_Bytes16(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ double Vdbe::Column_Double(Vdbe *p, int i) { double val = Value_Double(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ int Vdbe::Column_Int(Vdbe *p, int i) { int val = Value_Int(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ int64 Vdbe::Column_Int64(Vdbe *p, int i) { int64 val = Value_Int64(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ const unsigned char *Vdbe::Column_Text(Vdbe *p, int i) { const unsigned char *val = Value_Text(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
__device__ Mem *Vdbe::Column_Value(Vdbe *p, int i)
{
	Mem *val = ColumnMem(p, i);
	if (val->Flags & MEM_Static)
	{
		val->Flags &= ~MEM_Static;
		val->Flags |= MEM_Ephem;
	}
	ColumnMallocFailure(p);
	return val;
}
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_Text16(Vdbe *p, int i) { const void *val = Value_Text16(ColumnMem(p, i)); ColumnMallocFailure(p); return val; }
#endif
__device__ TYPE Vdbe::Column_Type(Vdbe *p, int i) { TYPE type = Value_Type(ColumnMem(p, i)); ColumnMallocFailure(p); return type; }

__device__ static const void *ColumnName(Vdbe *p, int n, const void *(*func)(Mem *), bool useType)
{
	Context *ctx = p->Ctx;
	_assert(ctx != nullptr);
	const void *r = nullptr;
	int n2 = Vdbe::Column_Count(p);
	if (n < n2 && n >= 0)
	{
		n += useType*n2;
		MutexEx::Enter(ctx->Mutex);
		_assert(!ctx->MallocFailed);
		r = func(&p->ColNames[n]);
		// A malloc may have failed inside of the xFunc() call. If this is the case, clear the mallocFailed flag and return NULL.
		if (ctx->MallocFailed)
		{
			ctx->MallocFailed = false;
			r = nullptr;
		}
		MutexEx::Leave(ctx->Mutex);
	}
	return r;
}

__device__ const char *Vdbe::Column_Name(Vdbe *p, int n) { return (const char *)ColumnName(p, n, (const void *(*)(Mem *))Value_Text, COLNAME_NAME); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_Name16(Vdbe *p, int n) { return ColumnName(p, n, (const void *(*)(Mem *))Value_Text16, COLNAME_NAME); }
#endif

#if defined(OMIT_DECLTYPE) && defined(ENABLE_COLUMN_METADATA)
#error "Must not define both SQLITE_OMIT_DECLTYPE and SQLITE_ENABLE_COLUMN_METADATA"
#endif
#ifndef OMIT_DECLTYPE
__device__ const char *Vdbe::Column_Decltype(Vdbe *p, int n) { return (const char *)ColumnName(p, n, (const void *(*)(Mem *))Value_Text, COLNAME_DECLTYPE); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_Decltype16(Vdbe *p, int n) { return ColumnName(p, n, (const void *(*)(Mem *))Value_Text16, COLNAME_DECLTYPE); }
#endif
#endif

#ifdef ENABLE_COLUMN_METADATA
__device__ const char *Vdbe::Column_DatabaseName(Vdbe *p, int n) { return (const char *)ColumnName(p, n, (const void *(*)(Mem *))Value_Text, COLNAME_DATABASE); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_DatabaseName16(Vdbe *p, int n) { return ColumnName(p, n, (const void *(*)(Mem *))Value_Text16, COLNAME_DATABASE); }
#endif

__device__ const char *Vdbe::Column_TableName(Vdbe *p, int n) { return (const char *)ColumnName(p, n, (const void *(*)(Mem *))Value_Text, COLNAME_TABLE); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_TableName16(Vdbe *p, int n) { return ColumnName(p, n, (const void *(*)(Mem *))Value_Text16, COLNAME_TABLE); }
#endif

__device__ const char *Vdbe::Column_OriginName(Vdbe *p, int n) { return (const char *)ColumnName(p, n, (const void *(*)(Mem *))Value_Text, COLNAME_COLUMN); }
#ifndef OMIT_UTF16
__device__ const void *Vdbe::Column_OriginName16(Vdbe *p, int n) { return ColumnName(p, n, (const void *(*)(Mem *)Value_Text16, COLNAME_COLUMN); }
#endif
#endif

#pragma endregion

#pragma region Bind

static RC VdbeUnbind(Vdbe *p, int i)
{
	Mem *var;
	if (VdbeSafetyNotNull(p))
		return SysEx_MISUSE_BKPT;
	MutexEx::Enter(p->Ctx->Mutex);
	if (p->Magic != VDBE_MAGIC_RUN || p->PC >= 0)
	{
		Main::Error(p->Ctx, RC_MISUSE, 0);
		MutexEx::Leave(p->Ctx->Mutex);
		SysEx_LOG(RC_MISUSE, "bind on a busy prepared statement: [%s]", p->Sql);
		return SysEx_MISUSE_BKPT;
	}
	if (i < 1 || i > p->Vars.length)
	{
		Main::Error(p->Ctx, RC_RANGE, 0);
		MutexEx::Leave(p->Ctx->Mutex);
		return RC_RANGE;
	}
	i--;
	var = &p->Vars[i];
	Vdbe::MemRelease(var);
	var->Flags = MEM_Null;
	Main::Error(p->Ctx, RC_OK, 0);

	// If the bit corresponding to this variable in Vdbe.expmask is set, then binding a new value to this variable invalidates the current query plan.
	//
	// IMPLEMENTATION-OF: R-48440-37595 If the specific value bound to host parameter in the WHERE clause might influence the choice of query plan
	// for a statement, then the statement will be automatically recompiled, as if there had been a schema change, on the first sqlite3_step() call
	// following any change to the bindings of that parameter.
	if (p->IsPrepareV2 && ((i < 32 && p->Expmask & ((uint32)1 << i)) || p->Expmask == 0xffffffff))
		p->Expired = true;
	return RC_OK;
}

__device__ static RC BindText(Vdbe *p, int i, const void *z, int n, void (*del)(void *), TEXTENCODE encoding)
{
	RC rc = VdbeUnbind(p, i);
	if (rc == RC_OK)
	{
		if (z)
		{
			Mem *var = &p->Vars[i-1];
			rc = Vdbe::MemSetStr(var, (const char *)z, n, encoding, del);
			if (rc == RC_OK && encoding != 0)
				rc = Vdbe::ChangeEncoding(var, CTXENCODE(p->Ctx));
			Main::Error(p->Ctx, rc, 0);
			Main::ApiExit(p->Ctx, rc);
		}
		MutexEx::Leave(p->Ctx->Mutex);
	}
	else if (del != DESTRUCTOR_STATIC && del != DESTRUCTOR_TRANSIENT)
		del((void *)z);
	return rc;
}
__device__ RC Vdbe::Bind_Blob(Vdbe *p, int i, const void *z, int n, void (*del)(void *)) { return BindText(p, i, z, n, del, (TEXTENCODE)0); }
__device__ RC Vdbe::Bind_Double(Vdbe *p, int i, double value)
{
	RC rc = VdbeUnbind(p, i);
	if (rc == RC_OK)
	{
		MemSetDouble(&p->Vars[i-1], value);
		MutexEx::Leave(p->Ctx->Mutex);
	}
	return rc;
}
__device__ RC Vdbe::Bind_Int(Vdbe *p, int i, int value) { return Bind_Int64(p, i, (int64)value); }
__device__ RC Vdbe::Bind_Int64(Vdbe *p, int i, int64 value)
{
	RC rc = VdbeUnbind(p, i);
	if (rc == RC_OK)
	{
		MemSetInt64(&p->Vars[i-1], value);
		MutexEx::Leave(p->Ctx->Mutex);
	}
	return rc;
}
__device__ RC Vdbe::Bind_Null(Vdbe *p, int i)
{
	RC rc = VdbeUnbind(p, i);
	if (rc == RC_OK)
		MutexEx::Leave(p->Ctx->Mutex);
	return rc;
}
__device__ RC Vdbe::Bind_Text(Vdbe *p, int i, const char *z, int n, void (*del)(void *)) { return BindText(p, i, z, n, del, TEXTENCODE_UTF8); }
#ifndef OMIT_UTF16
__device__ RC Vdbe::Bind_Text16(Vdbe *p, int i, const void *z, int n, void (*del)(void *)) { return BindText(p, i, z, n, del, TEXTENCODE_UTF16NATIVE); }
#endif
__device__ RC Bind_Value(Vdbe *p, int i, const Mem *value)
{
	RC rc;
	switch (value->Type)
	{
	case TYPE_INTEGER: {
		rc = Vdbe::Bind_Int64(p, i, value->u.I);
		break; }
	case TYPE_FLOAT: {
		rc = Vdbe::Bind_Double(p, i, value->R);
		break; }
	case TYPE_BLOB: {
		if (value->Flags & MEM_Zero)
			rc = Vdbe::Bind_Zeroblob(p, i, value->u.Zeros);
		else
			rc = Vdbe::Bind_Blob(p, i, value->Z, value->N, DESTRUCTOR_TRANSIENT);
		break; }
	case TYPE_TEXT: {
		rc = BindText(p, i, value->Z, value->N, DESTRUCTOR_TRANSIENT, value->Encode);
		break; }
	default: {
		rc = Vdbe::Bind_Null(p, i);
		break; }
	}
	return rc;
}
__device__ RC Vdbe::Bind_Zeroblob(Vdbe *p, int i, int n)
{
	RC rc = VdbeUnbind(p, i);
	if (rc == RC_OK)
	{
		MemSetZeroBlob(&p->Vars[i-1], n);
		MutexEx::Leave(p->Ctx->Mutex);
	}
	return rc;
}

__device__ int Vdbe::Bind_ParameterCount(Vdbe *p) { return (p ? p->Vars.length : 0); }
__device__ const char *Vdbe::Bind_ParameterName(Vdbe *p, int i) { return (p == 0 || i < 1 || i > p->VarNames.length ? 0: p->VarNames[i-1]); }
__device__ int Vdbe::ParameterIndex(Vdbe *p, const char *name, int nameLength)
{
	if (!p)
		return 0;
	if (name)
	{
		for (int i = 0; i < p->VarNames.length; i++)
		{
			const char *z = p->VarNames[i];
			if (z && !_strncmp(z, name, nameLength) && z[nameLength] == 0)
				return i+1;
		}
	}
	return 0;
}
__device__ int Vdbe::Bind_ParameterIndex(Vdbe *p, const char *name) { return ParameterIndex(p, name, _strlen30(name)); }

__device__ RC Vdbe::TransferBindings(Vdbe *from, Vdbe *to)
{
	_assert(to->Ctx == from->Ctx);
	_assert(to->Vars.length == from->Vars.length);
	MutexEx::Enter(to->Ctx->Mutex);
	for (int i = 0; i < from->Vars.length; i++)
		MemMove(&to->Vars[i], &from->Vars[i]);
	MutexEx::Leave(to->Ctx->Mutex);
	return RC_OK;
}

#pragma endregion

#pragma region Stmt

__device__ Context *Vdbe::Stmt_Ctx(Vdbe *p) { return (p ? p->Ctx : nullptr); }
__device__ bool Vdbe::Stmt_Readonly(Vdbe *p) { return (p ? p->ReadOnly : true); }
__device__ bool Vdbe::Stmt_Busy(Vdbe *p) { return (p && p->PC > 0 && p->Magic == VDBE_MAGIC_RUN); }
__device__ Vdbe *Vdbe::Stmt_Next(Context *ctx, Vdbe *p)
{
	MutexEx::Enter(ctx->Mutex);
	Vdbe *next = (!p ? ctx->Vdbes : p->Next);
	MutexEx::Leave(ctx->Mutex);
	return next;
}
__device__ int Vdbe::Stmt_Status(Vdbe *p, OP op, bool resetFlag)
{
	int v = p->Counters[op-1];
	if (resetFlag) p->Counters[op-1] = 0;
	return v;
}

#pragma endregion
