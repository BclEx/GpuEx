#include "vdbeInt.cu.h"

#pragma region Preamble
namespace Core {

	// Properties of opcodes.  The OPFLG_INITIALIZER macro is created by mkopcodeh.awk during compilation.  Data is obtained
	// from the comments following the "case OP_xxxx:" statements in the vdbe.c file.  
	const unsigned char _opcodeProperty[] = OPFLG_INITIALIZER;

#ifdef TEST
	__device__ extern int g_search_count;
#endif

}
#pragma endregion

#pragma region Aux

__device__ Vdbe *Vdbe::Create(Context *ctx)
{
	Vdbe *p = (Vdbe *)_tagalloc2(ctx, sizeof(Vdbe), true);
	if (!p) return nullptr;
	p->Ctx = ctx;
	if (ctx->Vdbes)
		ctx->Vdbes->Prev = p;
	p->Next = ctx->Vdbes;
	p->Prev = nullptr;
	ctx->Vdbes = p;
	p->Magic = VDBE_MAGIC_INIT;
	return p;
}

__device__ void Vdbe::SetSql(Vdbe *p, const char *z, int n, bool isPrepareV2)
{
	if (!p) return;
#if defined(OMIT_TRACE) && !defined(ENABLE_SQLLOG)
	if (!isPrepareV2) return;
#endif
	_assert(!p->Sql_);
	p->Sql_ = _tagstrndup(p->Ctx, z, n);
	p->IsPrepareV2 = (bool)isPrepareV2;
}

__device__ const char *Vdbe::Sql(Vdbe *stmt)
{
	Vdbe *p = (Vdbe *)stmt;
	return (p && p->IsPrepareV2 ? p->Sql_ : nullptr);
}

__device__ void Vdbe::Swap(Vdbe *a, Vdbe *b)
{
	Vdbe tmp_ = *a;
	*a = *b;
	*b = tmp_;
	Vdbe *tmp = a->Next;
	a->Next = b->Next;
	b->Next = tmp;
	tmp = a->Prev;
	a->Prev = b->Prev;
	b->Prev = tmp;
	char *tmpSql = a->Sql_;
	a->Sql_ = b->Sql_;
	b->Sql_ = tmpSql;
	b->IsPrepareV2 = a->IsPrepareV2;
}

#ifdef _DEBUG
__device__ void Vdbe::set_Trace(FILE *trace)
{
	Trace = trace;
}
#endif

__device__ static RC GrowOps(Vdbe *p)
{
	int newLength = (p->OpsAlloc ? p->OpsAlloc*2 : (int)(1024/sizeof(Vdbe::VdbeOp)));
	Vdbe::VdbeOp *newData = (Vdbe::VdbeOp *)_tagrealloc(p->Ctx, p->Ops.data, newLength * sizeof(Vdbe::VdbeOp));
	if (newData)
	{
		p->OpsAlloc = _tagallocsize(p->Ctx, newData)/sizeof(Vdbe::VdbeOp);
		p->Ops.data = newData;
	}
	return (newData ? RC_OK : RC_NOMEM);
}

__device__ int Vdbe::AddOp3(OP op, int p1, int p2, int p3)
{
	int i = Ops.length;
	_assert(Magic == VDBE_MAGIC_INIT);
	_assert(op > 0 && op < 0xff);
	if (OpsAlloc <= i)
		if (GrowOps(this))
			return 1;
	Ops.length++;
	Vdbe::VdbeOp *opAsObj = &Ops[i];
	opAsObj->Opcode = op;
	opAsObj->P5 = 0;
	opAsObj->P1 = p1;
	opAsObj->P2 = p2;
	opAsObj->P3 = p3;
	opAsObj->P4.P = nullptr;
	opAsObj->P4Type = Vdbe::P4T_NOTUSED;
#ifdef _DEBUG
	opAsObj->Comment = nullptr;
	if (Ctx->Flags & BContext::FLAG_VdbeAddopTrace)
		PrintOp(nullptr, i, &Ops[i]);
#endif
#ifdef VDBE_PROFILE
	opAsObj->Cycles = 0;
	opAsObj->Cnt = 0;
#endif
	return i;
}
__device__ int Vdbe::AddOp0(OP op) { return AddOp3(op, 0, 0, 0); }
__device__ int Vdbe::AddOp1(OP op, int p1) { return AddOp3(op, p1, 0, 0); }
__device__ int Vdbe::AddOp2(OP op, int p1, int p2) { return AddOp3(op, p1, p2, 0); }
__device__ int Vdbe::AddOp4(OP op, int p1, int p2, int p3, const char *p4, int p4t)
{
	int addr = AddOp3(op, p1, p2, p3);
	ChangeP4(addr, p4, p4t);
	return addr;
}

__device__ void Vdbe::AddParseSchemaOp(int db, char *where_)
{
	int addr = AddOp3(OP_ParseSchema, db, 0, 0);
	ChangeP4(addr, where_, Vdbe::P4T_DYNAMIC);
	for (int j = 0; j < Ctx->DBs.length; j++) UsesBtree(j);
}

__device__ int Vdbe::AddOp4Int(OP op, int p1, int p2, int p3, int p4)
{
	int addr = AddOp3(op, p1, p2, p3);
	ChangeP4(addr, (const char *)INT_TO_PTR(p4), Vdbe::P4T_INT32);
	return addr;
}

__device__ int Vdbe::MakeLabel()
{
	int i = Labels.length++;
	_assert(Magic == VDBE_MAGIC_INIT);
	if ((i & (i-1)) == 0) //? always
		Labels.data = (int *)_tagrealloc_or_free(Ctx, Labels.data, (i*2+1)*sizeof(Labels[0]));
	if (Labels.data)
		Labels[i] = -1;
	return -1 - i;
}

__device__ void Vdbe::ResolveLabel(int x)
{
	int j = -1-x;
	_assert(Magic == VDBE_MAGIC_INIT);
	_assert(j >= 0 && j < Labels.length);
	if (Labels.data)
		Labels[j] = Ops.length;
}

__device__ void Vdbe::set_RunOnlyOnce()
{
	RunOnlyOnce = true;
}

#ifdef _DEBUG

struct VdbeOpIter
{
	Vdbe *V;					// Vdbe to iterate through the opcodes of
	array_t<Vdbe::SubProgram*> Subs;	// Array of subprograms
	int Addr;					// Address of next instruction to return
	int SubId;					// 0 = main program, 1 = first sub-program etc.
};

__device__ static Vdbe::VdbeOp *OpIterNext(VdbeOpIter *p)
{
	Vdbe::VdbeOp *r = nullptr;
	if (p->SubId <= p->Subs.length)
	{
		Vdbe *v = p->V;
		array_t<Vdbe::VdbeOp> ops = (p->SubId == 0 ? v->Ops : p->Subs[p->SubId-1]->Ops);
		_assert(p->Addr < ops.length);

		r = &ops[p->Addr];
		p->Addr++;
		if (p->Addr == ops.length)
		{
			p->SubId++;
			p->Addr = 0;
		}
		if (r->P4Type == Vdbe::P4T_SUBPROGRAM)
		{
			int j;
			for (j = 0; j < p->Subs.length; j++)
				if (p->Subs[j] == r->P4.Program) break;
			if (j == p->Subs.length)
			{
				int bytes = (p->Subs.length+1)*sizeof(Vdbe::SubProgram *);
				p->Subs.data = (Vdbe::SubProgram **)_tagrealloc_or_free(v->Ctx, p->Subs.data, bytes);
				if (!p->Subs.data)
					r = nullptr;
				else
					p->Subs[p->Subs.length++] = r->P4.Program;
			}
		}
	}
	return r;
}

__device__ bool Vdbe::AssertMayAbort(bool mayAbort)
{
	VdbeOpIter sIter;
	_memset(&sIter, 0, sizeof(sIter));
	sIter.V = this;

	bool hasAbort = false;
	Vdbe::VdbeOp *op;
	while ((op = OpIterNext(&sIter)) != nullptr)
	{
		OP opcode = op->Opcode;
		if (opcode == OP_Destroy || opcode == OP_VUpdate || opcode == OP_VRename 
#ifndef OMIT_FOREIGN_KEY
			|| (opcode == OP_FkCounter && op->P1 == 0 && op->P2 == 1) 
#endif
			|| ((opcode == OP_Halt || opcode == OP_HaltIfNull) 
			&& ((op->P1&0xff) == RC_CONSTRAINT && op->P2 == OE_Abort)))
		{
			hasAbort = true;
			break;
		}
	}
	_tagfree(Ctx, sIter.Subs.data);

	// Return true if hasAbort==mayAbort. Or if a malloc failure occurred. If malloc failed, then the while() loop above may not have iterated
	// through all opcodes and hasAbort may be set incorrectly. Return true for this case to prevent the assert() in the callers frame from failing.
	return (Ctx->MallocFailed || hasAbort == mayAbort);
}
#endif

__device__ static void ResolveP2Values(Vdbe *p, int *maxFuncArgs)
{
	int maxArgs = *maxFuncArgs;
	int *labels = p->Labels.data;
	p->ReadOnly = true;

	Vdbe::VdbeOp *op;
	int i;
	for (op = p->Ops, i = p->Ops.length-1; i >= 0; i--, op++)
	{
		OP opcode = op->Opcode;
		op->Opflags = _opcodeProperty[opcode];
		if (opcode == OP_Function || opcode == OP_AggStep)
		{
			if (op->P5 > maxArgs) maxArgs = op->P5;
		}
		else if ((opcode == OP_Transaction && op->P2 != 0) || opcode == OP_Vacuum)
		{
			p->ReadOnly = false;
		}
#ifndef OMIT_VIRTUALTABLE
		else if (opcode == OP_VUpdate)
		{
			if (op->P2 > maxArgs) maxArgs = op->P2;
		}
		else if (opcode == OP_VFilter)
		{
			_assert(p->Ops.length - i >= 3);
			_assert(op[-1].Opcode == OP_Integer);
			int n = op[-1].P1;
			if (n > maxArgs) maxArgs = n;
		}
#endif
		else if (opcode == OP_Next || opcode == OP_SorterNext)
		{
			op->P4.Advance = Btree::Next_;
			op->P4Type = Vdbe::P4T_ADVANCE;
		}
		else if (opcode == OP_Prev)
		{
			op->P4.Advance = Btree::Previous;
			op->P4Type = Vdbe::P4T_ADVANCE;
		}
		if ((op->Opflags & OPFLG_JUMP) != 0 && op->P2 < 0)
		{
			_assert(-1 - op->P2 < p->Labels.length);
			op->P2 = labels[-1 - op->P2];
		}
	}
	_tagfree(p->Ctx, p->Labels.data);
	p->Labels.data = nullptr;
	*maxFuncArgs = maxArgs;
}

__device__ int Vdbe::CurrentAddr()
{
	_assert(Magic == VDBE_MAGIC_INIT);
	return Ops.length;
}

__device__ Vdbe::VdbeOp *Vdbe::TakeOpArray(int *opsLength, int *maxArgs)
{
	Vdbe::VdbeOp *ops = Ops.data;
	_assert(ops && !Ctx->MallocFailed);
	_assert(BtreeMask == 0); // Check that sqlite3VdbeUsesBtree() was not called on this VM
	ResolveP2Values(this, maxArgs);
	*opsLength = Ops.length;
	Ops.data = nullptr;
	return ops;
}

__device__ int Vdbe::AddOpList(int opsLength, VdbeOpList const *ops)
{
	_assert(Magic == VDBE_MAGIC_INIT);
	if (Ops.length + opsLength > OpsAlloc && GrowOps(this))
		return 0;
	int addr = Ops.length;
	if (_ALWAYS(opsLength > 0))
	{
		VdbeOpList const *in_ = ops;
		for (int i = 0; i < opsLength; i++, in_++)
		{
			int p2 = in_->P2;
			VdbeOp *out_ = &Ops[i+addr];
			out_->Opcode = in_->Opcode;
			out_->P1 = in_->P1;
			out_->P2 = (p2 < 0 && (_opcodeProperty[out_->Opcode] & OPFLG_JUMP) != 0 ? addr + ADDR(p2) : p2);
			out_->P3 = in_->P3;
			out_->P4Type = Vdbe::P4T_NOTUSED;
			out_->P4.P = nullptr;
			out_->P5 = 0;
#ifdef _DEBUG
			out_->Comment = nullptr;
			if (Ctx->Flags & BContext::FLAG_VdbeAddopTrace)
				PrintOp(nullptr, i+addr, &Ops[i+addr]);
#endif
		}
		Ops.length += opsLength;
	}
	return addr;
}

__device__ void Vdbe::ChangeP1(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P1 = val; }
__device__ void Vdbe::ChangeP2(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P2 = val; }
__device__ void Vdbe::ChangeP3(uint32 addr, int val) { if ((uint32)Ops.length > addr) Ops[addr].P3 = val; }
__device__ void Vdbe::ChangeP5(uint8 val) { if (Ops.data) { _assert(Ops.length > 0); Ops[Ops.length-1].P5 = val; } }

__device__ void Vdbe::JumpHere(int addr)
{
	_assert(addr >= 0 || Ctx->MallocFailed);
	if (addr >= 0) ChangeP2(addr, Ops.length);
}

__device__ static void FreeEphemeralFunction(Context *ctx, FuncDef *def)
{
	if (_ALWAYS(def) && (def->Flags & FUNC_EPHEM) != 0)
		_tagfree(ctx, def);
}
__device__ static void VdbeFreeOpArray(Context *ctx, Vdbe::VdbeOp *ops, int opsLength);
__device__ static void FreeP4(Context *ctx, Vdbe::P4T p4t, void *p4)
{
	_assert(ctx);
	if (p4)
	{
		switch (p4t)
		{
		case Vdbe::P4T_REAL:
		case Vdbe::P4T_INT64:
		case Vdbe::P4T_DYNAMIC:
		case Vdbe::P4T_KEYINFO:
		case Vdbe::P4T_INTARRAY:
		case Vdbe::P4T_KEYINFO_HANDOFF: {
			_tagfree(ctx, p4);
			break; }
		case Vdbe::P4T_MPRINTF: {
			if (ctx->BytesFreed == 0) _free(p4);
			break; }
		case Vdbe::P4T_VDBEFUNC: {
			VdbeFunc *vdbeFunc = (VdbeFunc *)p4;
			FreeEphemeralFunction(ctx, (FuncDef *)vdbeFunc);
			if (ctx->BytesFreed == 0) Vdbe::DeleteAuxData(vdbeFunc, 0);
			_tagfree(ctx, vdbeFunc);
			break; }
		case Vdbe::P4T_FUNCDEF: {
			FreeEphemeralFunction(ctx, (FuncDef *)p4);
			break; }
		case Vdbe::P4T_MEM: {
			if (ctx->BytesFreed == 0)
				Vdbe::ValueFree((Mem *)p4);
			else
			{
				Mem *p = (Mem *)p4;
				_tagfree(ctx, p->Malloc);
				_tagfree(ctx, p);
			}
			break; }
		case Vdbe::P4T_VTAB : {
			if (ctx->BytesFreed == 0) ((VTable *)p4)->Unlock();
			break; }
		}
	}
}

__device__ static void VdbeFreeOpArray(Context *ctx, Vdbe::VdbeOp *ops, int opsLength)
{
	if (ops)
	{
		for (Vdbe::VdbeOp *op = ops; op < &ops[opsLength]; op++)
		{
			FreeP4(ctx, op->P4Type, op->P4.P);
#ifdef _DEBUG
			_tagfree(ctx, op->Comment);
#endif     
		}
	}
	_tagfree(ctx, ops);
}

__device__ void Vdbe::LinkSubProgram(SubProgram *p)
{
	p->Next = Programs;
	Programs = p;
}

__device__ void Vdbe::ChangeToNoop(int addr)
{
	if (Ops.data)
	{
		Vdbe::VdbeOp *op = &Ops[addr];
		FreeP4(Ctx, op->P4Type, op->P4.P);
		_memset(op, 0, sizeof(op[0]));
		op->Opcode = OP_Noop;
	}
}

__device__ void Vdbe::ChangeP4(int addr, const char *p4, int n)
{
	_assert(Magic == VDBE_MAGIC_INIT);
	Context *ctx = Ctx;
	if (!Ops.data || ctx->MallocFailed)
	{
		if (n != P4T_KEYINFO && n != P4T_VTAB)
			FreeP4(ctx, (P4T)n, (void *)*(char **)&p4);
		return;
	}
	_assert(Ops.length > 0);
	_assert(addr < Ops.length);
	if (addr < 0)
		addr = Ops.length - 1;
	Vdbe::VdbeOp *op = &Ops[addr];
	_assert(op->P4Type == P4T_NOTUSED || op->P4Type == P4T_INT32);
	FreeP4(ctx, op->P4Type, op->P4.P);
	op->P4.P = nullptr;
	if ((P4T)n == P4T_INT32)
	{
		// Note: this cast is safe, because the origin data point was an int that was cast to a (const char *).
		op->P4.I = PTR_TO_INT(p4);
		op->P4Type = P4T_INT32;
	}
	else if (!p4)
	{
		op->P4.P = nullptr;
		op->P4Type = P4T_NOTUSED;
	}
	else if ((P4T)n == P4T_KEYINFO)
	{
		int fields = ((KeyInfo *)p4)->Fields;
		KeyInfo *keyInfo;
		int bytes = sizeof(*keyInfo) + (fields-1)*sizeof(keyInfo->Colls[0]) + fields;
		keyInfo = (KeyInfo *)_tagalloc(nullptr, bytes);
		op->P4.KeyInfo = keyInfo;
		if (keyInfo)
		{
			_memcpy((char *)keyInfo, p4, bytes - fields);
			SO *sortOrders = keyInfo->SortOrders;
			_assert(sortOrders != nullptr);
			keyInfo->SortOrders = (SO *)&keyInfo->Colls[fields];
			_memcpy(keyInfo->SortOrders, sortOrders, fields);
			op->P4Type = P4T_KEYINFO;
		}
		else
		{
			ctx->MallocFailed = true;
			op->P4Type = P4T_NOTUSED;
		}
	}
	else if ((P4T)n == P4T_KEYINFO_HANDOFF)
	{
		op->P4.P = (void *)p4;
		op->P4Type = P4T_KEYINFO;
	}
	else if ((P4T)n == P4T_VTAB)
	{
		op->P4.P = (void *)p4;
		op->P4Type = P4T_VTAB;
		(((VTable *)p4))->Lock();
		_assert(((VTable *)p4)->Ctx == ctx);
	}
	else if (n < 0)
	{
		op->P4.P = (void *)p4;
		op->P4Type = (P4T)n;
	}
	else
	{
		if (n == 0) n = _strlen30(p4);
		op->P4.Z = _tagstrndup(ctx, p4, n);
		op->P4Type = P4T_DYNAMIC;
	}
}

#ifndef NDEBUG
__device__ void Vdbe::Comment(const char *format, va_list *args)
{
	_assert(Ops.length > 0 || !Ops.data);
	_assert(!Ops.data || !Ops[Ops.length-1].Comment || Ctx->MallocFailed);
	if (Ops.length)
	{
		_assert(Ops.data);
		_tagfree(Ctx, Ops[Ops.length-1].Comment);
		Ops[Ops.length-1].Comment = _vmtagprintf(Ctx, format, args, nullptr);
	}
}
__device__ void Vdbe::NoopComment(const char *format, va_list *args)
{
	AddOp0(OP_Noop);
	_assert(Ops.length > 0 || !Ops.data);
	_assert(!Ops.data || !Ops[Ops.length-1].Comment || Ctx->MallocFailed);
	if (Ops.length)
	{
		_assert(Ops.data);
		_tagfree(Ctx, Ops[Ops.length-1].Comment);
		Ops[Ops.length-1].Comment = _vmtagprintf(Ctx, format, args, nullptr);
	}
}
#endif

__constant__ static Vdbe::VdbeOp _dummy1;  // Ignore the MSVC warning about no initializer
__device__ Vdbe::VdbeOp *Vdbe::GetOp(int addr)
{
	// C89 specifies that the constant "dummy" will be initialized to all zeros, which is correct.  MSVC generates a warning, nevertheless.
	_assert(Magic == VDBE_MAGIC_INIT);
	if (addr < 0)
	{
#ifdef OMIT_TRACE
		if (Ops.length == 0) return (VdbeOp *)&_dummy;
#endif
		addr = Ops.length - 1;
	}
	_assert((addr >= 0 && addr < Ops.length) || Ctx->MallocFailed);
	return (Ctx->MallocFailed ? (VdbeOp *)&_dummy1 : &Ops[addr]);
}

#if !defined(OMIT_EXPLAIN) || !defined(NDEBUG) || defined(VDBE_PROFILE) || defined(_DEBUG)
__device__ static char *DisplayP4(Vdbe::VdbeOp *op, char *temp, int tempLength)
{
	char *p4 = temp;
	_assert(tempLength >= 20);
	switch (op->P4Type)
	{
	case Vdbe::P4T_KEYINFO_STATIC:
	case Vdbe::P4T_KEYINFO: {
		KeyInfo *keyInfo = op->P4.KeyInfo;
		_assert(keyInfo->SortOrders);
		__snprintf(temp, tempLength, "keyinfo(%d", keyInfo->Fields);
		int i = _strlen30(temp);
		for (int j = 0; j < keyInfo->Fields; j++)
		{
			CollSeq *coll = keyInfo->Colls[j];
			const char *collName = (coll ? coll->Name : "nil");
			int collNameLength = _strlen30(collName);
			if (i+collNameLength > tempLength-6)
			{
				_memcpy(&temp[i], ",...", 4);
				break;
			}
			temp[i++] = ',';
			if (keyInfo->SortOrders[j])
				temp[i++] = '-';
			_memcpy(&temp[i], collName, collNameLength+1);
			i += collNameLength;
		}
		temp[i++] = ')';
		temp[i] = 0;
		_assert(i < tempLength);
		break; }
	case Vdbe::P4T_COLLSEQ: {
		CollSeq *coll = op->P4.Coll;
		__snprintf(temp, tempLength, "collseq(%.20s)", coll->Name);
		break; }
	case Vdbe::P4T_FUNCDEF: {
		FuncDef *def = op->P4.Func;
		__snprintf(temp, tempLength, "%s(%d)", def->Name, def->Args);
		break; }
	case Vdbe::P4T_INT64: {
		__snprintf(temp, tempLength, "%lld", *op->P4.I64);
		break; }
	case Vdbe::P4T_INT32: {
		__snprintf(temp, tempLength, "%d", op->P4.I);
		break; }
	case Vdbe::P4T_REAL: {
		__snprintf(temp, tempLength, "%.16g", *op->P4.Real);
		break; }
	case Vdbe::P4T_MEM: {
		Mem *mem = op->P4.Mem;
		if (mem->Flags & MEM_Str) p4 = mem->Z;
		else if (mem->Flags & MEM_Int) __snprintf(temp, tempLength, "%lld", mem->u.I);
		else if (mem->Flags & MEM_Real) __snprintf(temp, tempLength, "%.16g", mem->R);
		else if (mem->Flags & MEM_Null) __snprintf(temp, tempLength, "NULL");
		else { _assert(mem->Flags & MEM_Blob); p4 = "(blob)"; }
		break; }

#ifndef OMIT_VIRTUALTABLE
	case Vdbe::P4T_VTAB: {
		IVTable *vtable = op->P4.VTable->IVTable;
		__snprintf(temp, tempLength, "vtab:%p:%p", vtable, vtable->IModule);
		break; }
#endif
	case Vdbe::P4T_INTARRAY: {
		__snprintf(temp, tempLength, "intarray");
		break; }
	case Vdbe::P4T_SUBPROGRAM: {
		__snprintf(temp, tempLength, "program");
		break; }
	case Vdbe::P4T_ADVANCE: {
		temp[0] = 0;
		break; }
	default: {
		p4 = op->P4.Z;
		if (!p4)
		{
			p4 = temp;
			temp[0] = 0;
		} }
	}
	_assert(p4);
	return p4;
}
#endif

__device__ void Vdbe::UsesBtree(int i)
{
	_assert(i >= 0 && i < Ctx->DBs.length && i < (int)sizeof(yDbMask)*8);
	_assert(i < (int)sizeof(BtreeMask)*8);
	BtreeMask |= ((yDbMask)1)<<i;
	if (i != 1 && Ctx->DBs[i].Bt->Sharable())
		LockMask |= ((yDbMask)1)<<i;
}

#if !defined(OMIT_SHARED_CACHE) && THREADSAFE>0
__device__ void Vdbe::Enter()
{
	if (LockMask == 0) return;  // The common case
	Context *ctx = Ctx;
	array_t<BContext::DB> dbs = ctx->DBs;
	int i;
	yDbMask mask;
	for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
		if (i != 1 && (mask & LockMask) != 0 && _ALWAYS(dbs[i].Bt != nullptr))
			dbs[i].Bt->Enter();
}
__device__ void Vdbe::Leave()
{
	if (LockMask == 0) return;  // The common case
	Context *ctx = Ctx;
	array_t<BContext::DB> dbs = ctx->DBs;
	int i;
	yDbMask mask;
	for (i = 0, mask = 1; i < dbs.length; i++, mask += mask)
		if (i != 1 && (mask & LockMask) != 0 && _ALWAYS(dbs[i].Bt != nullptr))
			dbs[i].Bt->Leave();
}
#endif

#if defined(VDBE_PROFILE) || defined(_DEBUG)
__device__ void Vdbe::PrintOp(FILE *out_, int pc, Vdbe::VdbeOp *op)
{
	if (!out_) out_ = stdout;
	char ptr[50];
	char *p4 = DisplayP4(op, ptr, sizeof(ptr));
	fprintf(out_, "%4d %-13s %4d %4d %4d %-4s %.2X %s\n", pc,
		OpcodeName(op->Opcode), op->P1, op->P2, op->P3, p4, op->P5,
#ifdef _DEBUG
		(op->Comment ? op->Comment : "")
#else
		""
#endif
		);
	fflush(out_);
}
#endif

__device__ static void ReleaseMemArray(Mem *p, int n)
{
	if (p && n)
	{
		Context *ctx = p->Ctx;
		Mem *end;
		bool mallocFailed = ctx->MallocFailed;
		if (ctx->BytesFreed)
		{
			for (end = &p[n]; p < end; p++)
				_tagfree(ctx, p->Malloc);
			return;
		}
		for (end = &p[n]; p < end; p++)
		{
			_assert((&p[1]) == end || p[0].Ctx == p[1].Ctx);

			// This block is really an inlined version of sqlite3VdbeMemRelease() that takes advantage of the fact that the memory cell value is 
			// being set to NULL after releasing any dynamic resources.
			//
			// The justification for duplicating code is that according to callgrind, this causes a certain test case to hit the CPU 4.7 
			// percent less (x86 linux, gcc version 4.1.2, -O6) than if sqlite3MemRelease() were called from here. With -O2, this jumps
			// to 6.6 percent. The test case is inserting 1000 rows into a table with no indexes using a single prepared INSERT statement, bind() 
			// and reset(). Inserts are grouped into a transaction.
			if (p->Flags&(MEM_Agg|MEM_Dyn|MEM_Frame|MEM_RowSet))
				Vdbe::MemRelease(p);
			else if (p->Malloc)
			{
				_tagfree(ctx, p->Malloc);
				p->Malloc = nullptr;
			}
			p->Flags = MEM_Invalid;
		}
		ctx->MallocFailed = mallocFailed;
	}
}

__device__ void Vdbe::FrameDelete(VdbeFrame *p)
{
	Mem *mems = VdbeFrameMem(p);
	VdbeCursor **cursors = (VdbeCursor **)&mems[p->ChildMems];
	for (int i = 0; i < p->ChildCursors; i++)
		p->V->FreeCursor(cursors[i]);
	ReleaseMemArray(mems, p->ChildMems);
	_tagfree(p->V->Ctx, p);
}

#ifndef OMIT_EXPLAIN
__device__ RC Vdbe::List()
{
	Context *ctx = Ctx; // The database connection
	int i;

	_assert(HasExplain != 0);
	_assert(Magic == VDBE_MAGIC_RUN );
	_assert(RC_ == RC_OK || RC_ == RC_BUSY || RC_ == RC_NOMEM);

	// Even though this opcode does not use dynamic strings for the result, result columns may become dynamic if the user calls
	// sqlite3_column_text16(), causing a translation to UTF-16 encoding.
	Mem *mem = &Mems[1]; // First Mem of result set
	ReleaseMemArray(mem, 8);
	ResultSet = nullptr;

	if (RC_ == RC_NOMEM)
	{
		// This happens if a malloc() inside a call to sqlite3_column_text() or sqlite3_column_text16() failed.
		ctx->MallocFailed = true;
		return RC_ERROR;
	}

	// When the number of output rows reaches rows, that means the listing has finished and sqlite3_step() should return SQLITE_DONE.
	// rows is the sum of the number of rows in the main program, plus the sum of the number of rows in all trigger subprograms encountered
	// so far.  The rows value will increase as new trigger subprograms are encountered, but p->pc will eventually catch up to rows.
	int rows = Ops.length; // Stop when row count reaches this
	Mem *sub = nullptr; // Memory cell hold array of subprogs
	SubProgram **subs = nullptr; // Array of sub-vdbes
	int subsLength = 0; // Number of sub-vdbes seen so far
	if (HasExplain == 1)
	{
		// The first 8 memory cells are used for the result set.  So we will commandeer the 9th cell to use as storage for an array of pointers
		// to trigger subprograms.  The VDBE is guaranteed to have at least 9 cells.
		_assert(Mems.length > 9);
		sub = &Mems[9];
		if (sub->Flags & MEM_Blob)
		{
			// On the first call to sqlite3_step(), sub will hold a NULL.  It is initialized to a BLOB by the P4_SUBPROGRAM processing logic below
			subsLength = sub->N/sizeof(Vdbe *);
			subs = (SubProgram **)sub->Z;
		}
		for (i = 0; i < subsLength; i++)
			rows += subs[i]->Ops.length;
	}

	do
	{
		i = PC++;
	} while (i < rows && HasExplain == 2 && Ops[i].Opcode != OP_Explain);

	RC rc = RC_OK;
	if (i >= rows)
	{
		RC_ = RC_OK;
		rc = RC_DONE;
	}
	else if (ctx->u1.IsInterrupted)
	{
		RC_ = RC_INTERRUPT;
		rc = RC_ERROR;
		_setstring(&ErrMsg, ctx, "%s", Main::ErrStr(RC_));
	}
	else
	{
		char *z;
		VdbeOp *op;
		if (i < Ops.length)
			op = &Ops[i]; // The output line number is small enough that we are still in the main program.
		else
		{
			// We are currently listing subprograms.  Figure out which one and pick up the appropriate opcode.
			i -= Ops.length;
			int j;
			for (j = 0; i >= subs[j]->Ops.length; j++)
				i -= subs[j]->Ops.length;
			op = &subs[j]->Ops[i];
		}
		if (HasExplain == 1)
		{
			mem->Flags = MEM_Int;
			mem->Type = TYPE_INTEGER;
			mem->u.I = i; // Program counter
			mem++;

			mem->Flags = MEM_Static|MEM_Str|MEM_Term;
			mem->Z = (char *)OpcodeName(op->Opcode); // Opcode
			_assert(mem->Z != nullptr);
			mem->N = _strlen30(mem->Z);
			mem->Type = TYPE_TEXT;
			mem->Encode = TEXTENCODE_UTF8;
			mem++;

			// When an OP_Program opcode is encounter (the only opcode that has a P4_SUBPROGRAM argument), expand the size of the array of subprograms
			// kept in p->aMem[9].z to hold the new program - assuming this subprogram has not already been seen.
			if (op->P4Type == P4T_SUBPROGRAM)
			{
				int bytes = (subsLength+1)*sizeof(SubProgram *);
				int j;
				for (j = 0; j < subsLength; j++)
					if (subs[j] == op->P4.Program) break;
				if (j == subsLength && MemGrow(sub, bytes, subsLength!=0) == RC_OK)
				{
					subs = (SubProgram **)sub->Z;
					subs[subsLength++] = op->P4.Program;
					sub->Flags |= MEM_Blob;
					sub->N = subsLength*sizeof(SubProgram *);
				}
			}
		}

		mem->Flags = MEM_Int;
		mem->u.I = op->P1; // P1
		mem->Type = TYPE_INTEGER;
		mem++;

		mem->Flags = MEM_Int;
		mem->u.I = op->P2; // P2
		mem->Type = TYPE_INTEGER;
		mem++;

		mem->Flags = MEM_Int;
		mem->u.I = op->P3; // P3
		mem->Type = TYPE_INTEGER;
		mem++;

		if (MemGrow(mem, 32, 0)) // P4
		{            
			_assert(ctx->MallocFailed);
			return RC_ERROR;
		}
		mem->Flags = MEM_Dyn|MEM_Str|MEM_Term;
		z = DisplayP4(op, mem->Z, 32);
		if (z != mem->Z)
			MemSetStr(mem, z, -1, TEXTENCODE_UTF8, 0);
		else
		{
			_assert(mem->Z != nullptr);
			mem->N = _strlen30(mem->Z);
			mem->Encode = TEXTENCODE_UTF8;
		}
		mem->Type = TYPE_TEXT;
		mem++;

		if (HasExplain == 1)
		{
			if (MemGrow(mem, 4, 0))
			{
				_assert(ctx->MallocFailed);
				return RC_ERROR;
			}
			mem->Flags = MEM_Dyn|MEM_Str|MEM_Term;
			mem->N = 2;
			__snprintf(mem->Z, 3, "%.2x", op->P5); // P5
			mem->Type = TYPE_TEXT;
			mem->Encode = TEXTENCODE_UTF8;
			mem++;

#ifdef _DEBUG
			if (op->Comment)
			{
				mem->Flags = MEM_Str|MEM_Term;
				mem->Z = op->Comment; // Comment
				mem->N = _strlen30(mem->Z);
				mem->Encode = TEXTENCODE_UTF8;
				mem->Type = TYPE_TEXT;
			}
			else
#endif
			{
				mem->Flags = MEM_Null; 
				mem->Type = TYPE_NULL;
			}
		}

		ResColumns = 8 - 4*(HasExplain - 1);
		ResultSet = &Mems[1];
		RC_ = RC_OK;
		rc = RC_ROW;
	}
	return rc;
}
#endif

#ifdef _DEBUG
__device__ void Vdbe::PrintSql()
{
	int opsLength = Ops.length;
	if (opsLength < 1) return;
	VdbeOp *op = &Ops[0];
	if (op->Opcode == OP_Trace && op->P4.Z != nullptr)
	{
		const char *z = op->P4.Z;
		while (_isspace(*z)) z++;
		_printf("SQL: [%s]\n", z);
	}
}
#endif

#if !defined(OMIT_TRACE) && defined(ENABLE_IOTRACE)
__device__ void Vdbe::IOTraceSql()
{
	if (!IOTrace) return;
	int opsLength = Ops.length;
	if (opsLength < 1) return;
	VdbeOp *op = &Ops[0];
	if (op->Opcode == OP_Trace && op->P4.Z != nullptr)
	{
		char z[1000];
		__snprintf(z, sizeof(z), "%s", op->P4.Z);
		int i, j;
		for (i = 0; _isspace(z[i]); i++) { }
		for (j = 0; z[i]; i++)
		{
			if (_isspace(z[i]))
			{
				if (z[i-1] != ' ') z[j++] = ' ';
			}
			else z[j++] = z[i];
		}
		z[j] = 0;
		SysEx_IOTRACE("SQL %s\n", z);
	}
}
#endif

__device__ static void *AllocSpace(void *buf, int bytes, uint8 **from, uint8 *end, int *bytesOut)
{
	_assert(_HASALIGNMENT8(*from));
	if (buf) return buf;
	bytes = _ROUND8(bytes);
	if (&(*from)[bytes] <= end)
	{
		buf = (void *)*from;
		*from += bytes;
	}
	else
		*bytesOut += bytes;
	return buf;
}

__device__ void Vdbe::Rewind()
{
	_assert(Magic == VDBE_MAGIC_INIT);
	_assert(Ops.length > 0); // There should be at least one opcode.
	// Set the magic to VDBE_MAGIC_RUN sooner rather than later.
	Magic = VDBE_MAGIC_RUN;
#ifdef _DEBUG
	for (int i = 1; i < Mems.length; i++)
		_assert(Mems[i].Ctx == Ctx);
#endif
	PC = -1;
	RC_ = RC_OK;
	ErrorAction = OE_Abort;
	Magic = VDBE_MAGIC_RUN;
	Changes = 0;
	CacheCtr = 1;
	MinWriteFileFormat = 255;
	StatementID = 0;
	FkConstraints = 0;
#ifdef VDBE_PROFILE
	for (int i = 0; i < Ops.length; i++)
	{
		Ops[i].Cnt = 0;
		Ops[i].Cycles = 0;
	}
#endif
}

__device__ void Vdbe::MakeReady(Parse *parse)
{         
	_assert(Ops.length > 0);
	_assert(parse != nullptr);
	_assert(Magic == VDBE_MAGIC_INIT);
	Context *ctx = Ctx; // The database connection
	_assert(!ctx->MallocFailed);
	int vars = parse->Vars.length; // Number of parameters
	int mems = parse->Mems; // Number of VM memory registers
	int cursors = parse->Tabs; // Number of cursors required
	int args = parse->MaxArgs; // Number of arguments in subprograms
	int onces = parse->Onces; // Number of OP_Once instructions
	if (onces == 0) onces = 1; // Ensure at least one byte in p->aOnceFlag[]

	// For each cursor required, also allocate a memory cell. Memory cells (nMem+1-nCursor)..nMem, inclusive, will never be used by
	// the vdbe program. Instead they are used to allocate space for VdbeCursor/BtCursor structures. The blob of memory associated with 
	// cursor 0 is stored in memory cell nMem. Memory cell (nMem-1) stores the blob of memory associated with cursor 1, etc.
	//
	// See also: allocateCursor().
	mems += cursors;

	// Allocate space for memory registers, SQL variables, VDBE cursors and an array to marshal SQL function arguments in.
	uint8 *csr = (uint8 *)&Ops[Ops.length]; // Memory avaliable for allocation
	uint8 *end = (uint8 *)&Ops[OpsAlloc]; // First byte past end of zCsr[]
	ResolveP2Values(this, &args);
	UsesStmtJournal = (uint8)(parse->IsMultiWrite && parse->_MayAbort);
	if (parse->Explain && mems < 10)
		mems = 10;
	_memset(csr, 0, end-csr);
	csr += (csr - (uint8 *)0)&7;
	_assert(_HASALIGNMENT8(csr));
	Expired = false;

	// Memory for registers, parameters, cursor, etc, is allocated in two passes.  On the first pass, we try to reuse unused space at the 
	// end of the opcode array.  If we are unable to satisfy all memory requirements by reusing the opcode array tail, then the second
	// pass will fill in the rest using a fresh allocation.  
	//
	// This two-pass approach that reuses as much memory as possible from the leftover space at the end of the opcode array can significantly
	// reduce the amount of memory held by a prepared statement.
	int n;
	int bytes; // How much extra memory is needed
	do
	{
		bytes = 0;
		Mems.data = (Mem *)AllocSpace(Mems.data, mems*sizeof(Mem), &csr, end, &bytes);
		Vars.data = (Mem *)AllocSpace(Vars.data, vars*sizeof(Mem), &csr, end, &bytes);
		Args = (Mem **)AllocSpace(Args, args*sizeof(Mem *), &csr, end, &bytes);
		VarNames.data = (char **)AllocSpace(VarNames.data, vars*sizeof(char *), &csr, end, &bytes);
		Cursors.data = (VdbeCursor **)AllocSpace(Cursors.data, cursors*sizeof(VdbeCursor *), &csr, end, &bytes);
		OnceFlags.data = (uint8 *)AllocSpace(OnceFlags.data, onces, &csr, end, &bytes);
		if (bytes)
			FreeThis = _tagalloc2(ctx, bytes, true);
		csr = (uint8 *)FreeThis;
		end = &csr[bytes];
	} while (bytes && !ctx->MallocFailed);

	Cursors.length = cursors;
	OnceFlags.length = onces;
	if (Vars.data)
	{
		Vars.length = (yVars)vars;
		for (n = 0; n < vars; n++)
		{
			Vars[n].Flags = MEM_Null;
			Vars[n].Ctx = ctx;
		}
	}
	if (VarNames.data)
	{
		VarNames.length = parse->Vars.length;
		_memcpy(VarNames.data, parse->Vars.data, VarNames.length*sizeof(VarNames[0]));
		_memset(parse->Vars.data, 0, parse->Vars.length*sizeof(parse->Vars[0]));
	}
	if (Mems.data)
	{
		Mems.data--; // aMem[] goes from 1..nMem
		Mems.length = mems; // not from 0..nMem-1
		for (n = 1; n <= mems; n++)
		{
			Mems[n].Flags = MEM_Invalid;
			Mems[n].Ctx = ctx;
		}
	}
	HasExplain = parse->Explain;
	Rewind();
}

__device__ void Vdbe::FreeCursor(VdbeCursor *cur)
{
	if (!cur)
		return;
	SorterClose(Ctx, cur);
	if (cur->Bt)
		cur->Bt->Close(); // The pCx->pCursor will be close automatically, if it exists, by the call above.
	else if (cur->Cursor)
		Btree::CloseCursor(cur->Cursor);
#ifndef OMIT_VIRTUALTABLE
	if (cur->VtabCursor)
	{
		IVTableCursor *vtabCursor = cur->VtabCursor;
		const ITableModule *module = cur->IModule;
		InVtabMethod = 1;
		module->Close(vtabCursor);
		InVtabMethod = 0;
	}
#endif
}

__device__ int Vdbe::FrameRestore(VdbeFrame *frame)
{
	Vdbe *v = frame->V;
	v->OnceFlags.data = frame->OnceFlags.data;
	v->OnceFlags.length = frame->OnceFlags.length;
	v->Ops.data = frame->Ops.data;
	v->Ops.length = frame->Ops.length;
	v->Mems.data = frame->Mems.data;
	v->Mems.length = frame->Mems.length;
	v->Cursors.data = frame->Cursors.data;
	v->Cursors.length = frame->Cursors.length;
	v->Ctx->LastRowID = frame->LastRowID;
	v->Changes = frame->Changes;
	return frame->PC;
}

__device__ static void CloseAllCursors(Vdbe *p)
{
	if (p->Frames)
	{
		VdbeFrame *frame;
		for (frame = p->Frames; frame->Parent; frame = frame->Parent);
		Vdbe::FrameRestore(frame);
	}
	p->Frames = nullptr;
	p->FramesLength = 0;
	if (p->Cursors.data)
	{
		for (int i = 0; i < p->Cursors.length; i++)
		{
			VdbeCursor *cur = p->Cursors[i];
			if (cur)
			{
				p->FreeCursor(cur);
				p->Cursors[i] = nullptr;
			}
		}
	}
	if (p->Mems.data)
		ReleaseMemArray(&p->Mems[1], p->Mems.length);
	while (p->DelFrames)
	{
		VdbeFrame *del = p->DelFrames;
		p->DelFrames = del->Parent;
		Vdbe::FrameDelete(del);
	}
}

__device__ static void Cleanup(Vdbe *p)
{
	Context *ctx = p->Ctx;

#ifdef _DEBUG
	// Execute assert() statements to ensure that the Vdbe.apCsr[] and Vdbe.aMem[] arrays have already been cleaned up.
	int i;
	if (p->Cursors.data) for (i = 0; i < p->Cursors.length; i++) _assert(!p->Cursors[i]);
	if (p->Mems.data)
		for (i = 1; i <= p->Mems.length; i++) _assert(p->Mems[i].Flags == MEM_Invalid);
#endif
	_tagfree(ctx, p->ErrMsg);
	p->ErrMsg = nullptr;
	p->ResultSet = nullptr;
}

__device__ void Vdbe::SetNumCols(int resColumns)
{
	Context *ctx = Ctx;
	ReleaseMemArray(ColNames, ResColumns*COLNAME_N);
	_tagfree(ctx, ColNames);
	int n = resColumns*COLNAME_N;
	ResColumns = (uint16)resColumns;
	Mem *colName; 
	ColNames = colName = (Mem *)_tagalloc2(ctx, sizeof(Mem)*n, true);
	if (!ColNames) return;
	while (n-- > 0)
	{
		colName->Flags = MEM_Null;
		colName->Ctx = ctx;
		colName++;
	}
}

__device__ RC Vdbe::SetColName(int idx, int var, const char *name, void (*del)(void*))
{
	_assert(idx < ResColumns);
	_assert(var < COLNAME_N);
	if (Ctx->MallocFailed)
	{
		_assert(!name || del != DESTRUCTOR_DYNAMIC);
		return RC_NOMEM;
	}
	_assert(ColNames);
	Mem *colName = &(ColNames[idx+var*ResColumns]);
	RC rc = MemSetStr(colName, name, -1, TEXTENCODE_UTF8, del);
	_assert(rc != 0 || !name || (colName->Flags&MEM_Term) != 0);
	return rc;
}

__device__ static RC VdbeCommit(Context *ctx, Vdbe *p)
{
#ifdef OMIT_VIRTUALTABLE
	// With this option, sqlite3VtabSync() is defined to be simply SQLITE_OK so p is not used. 
#endif

	// Before doing anything else, call the xSync() callback for any virtual module tables written in this transaction. This has to
	// be done before determining whether a master journal file is required, as an xSync() callback may add an attached database
	// to the transaction.
	RC rc = VTable::Sync(ctx, &p->ErrMsg);

	int trans = 0; // Number of databases with an active write-transaction
	bool needXcommit = false;

	// This loop determines (a) if the commit hook should be invoked and (b) how many database files have open write transactions, not 
	// including the temp database. (b) is important because if more than one database file has an open write transaction, a master journal
	// file is required for an atomic commit.
	int i;
	for (i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
	{
		Btree *bt = ctx->DBs[i].Bt;
		if (bt->IsInTrans())
		{
			needXcommit = true;
			if (i != 1) trans++;
			bt->Enter();
			rc = bt->get_Pager()->ExclusiveLock();
			bt->Leave();
		}
	}
	if (rc != RC_OK)
		return rc;

	// If there are any write-transactions at all, invoke the commit hook
	if (needXcommit && ctx->CommitCallback)
	{
		rc = ctx->CommitCallback(ctx->CommitArg);
		if (rc)
			return RC_CONSTRAINT_COMMITHOOK;
	}

	// The simple case - no more than one database file (not counting the TEMP database) has a transaction active.   There is no need for the
	// master-journal.
	//
	// If the return value of sqlite3BtreeGetFilename() is a zero length string, it means the main database is :memory: or a temp file.  In 
	// that case we do not support atomic multi-file commits, so use the simple case then too.
	if (_strlen30(ctx->DBs[0].Bt->get_Filename()) == 0 || trans <= 1)
	{
		for (i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
				rc = bt->CommitPhaseOne(nullptr);
		}

		// Do the commit only if all databases successfully complete phase 1. If one of the BtreeCommitPhaseOne() calls fails, this indicates an
		// IO error while deleting or truncating a journal file. It is unlikely, but could happen. In this case abandon processing and return the error.
		for (i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
				rc = bt->CommitPhaseTwo(false);
		}
		if (rc == RC_OK)
			VTable::Commit(ctx);
	}

	// The complex case - There is a multi-file write-transaction active. This requires a master journal file to ensure the transaction is
	// committed atomicly.
#ifndef OMIT_DISKIO
	else
	{
		VSystem *vfs = ctx->Vfs;
		char const *mainFileName = ctx->DBs[0].Bt->get_Filename();

		// Select a master journal file name
		int mainFileNameLength = _strlen30(mainFileName);
		char *masterName = _mtagprintf(ctx, "%s-mjXXXXXX9XXz", mainFileName); // File-name for the master journal
		if (!masterName) return RC_NOMEM;
		int res;
		int retryCount = 0;
		VFile *master = nullptr;
		do
		{
			if (retryCount)
			{
				if (retryCount > 100)
				{
					SysEx_LOG(RC_FULL, "MJ delete: %s", masterName);
					vfs->Delete(masterName, false);
					break;
				}
				else if (retryCount == 1)
					SysEx_LOG(RC_FULL, "MJ collide: %s", masterName);
			}
			retryCount++;
			uint32 random;
			SysEx::PutRandom(sizeof(random), &random);
			__snprintf(&masterName[mainFileNameLength], 13, "-mj%06X9%02X", (random>>8)&0xffffff, random&0xff);
			// The antipenultimate character of the master journal name must be "9" to avoid name collisions when using 8+3 filenames.
			_assert(masterName[_strlen30(masterName)-3] == '9');
			VSystem::FileSuffix3(mainFileName, masterName);
			rc = vfs->Access(masterName, VSystem::ACCESS_EXISTS, &res);
		} while (rc == RC_OK && res);
		if (rc == RC_OK)
			rc = vfs->OpenAndAlloc(masterName, &master, VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_EXCLUSIVE|VSystem::OPEN_MASTER_JOURNAL, nullptr); // Open the master journal. 
		if (rc != RC_OK)
		{
			_tagfree(ctx, masterName);
			return rc;
		}

		// Write the name of each database file in the transaction into the new master journal file. If an error occurs at this point close
		// and delete the master journal file. All the individual journal files still have 'null' as the master journal pointer, so they will roll
		// back independently if a failure occurs.
		bool needSync = false;
		int64 offset = 0;
		for (i = 0; i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt->IsInTrans())
			{
				char const *fileName = bt->get_Journalname();
				if (!fileName)
					continue;  // Ignore TEMP and :memory: databases
				_assert(fileName[0] != 0);
				if (!needSync && !bt->SyncDisabled())
					needSync = true;
				rc = master->Write(fileName, _strlen30(fileName)+1, offset);
				offset += _strlen30(fileName)+1;
				if (rc != RC_OK)
				{
					master->CloseAndFree();
					vfs->Delete(masterName, false);
					_tagfree(ctx, masterName);
					return rc;
				}
			}
		}

		// Sync the master journal file. If the IOCAP_SEQUENTIAL device flag is set this is not required.
		if (needSync && (master->get_DeviceCharacteristics() & VFile::IOCAP_SEQUENTIAL) == 0 && (rc = master->Sync(VFile::SYNC_NORMAL)) != RC_OK)
		{
			master->CloseAndFree();
			vfs->Delete(masterName, false);
			_tagfree(ctx, masterName);
			return rc;
		}

		// Sync all the db files involved in the transaction. The same call sets the master journal pointer in each individual journal. If
		// an error occurs here, do not delete the master journal file.
		//
		// If the error occurs during the first call to sqlite3BtreeCommitPhaseOne(), then there is a chance that the
		// master journal file will be orphaned. But we cannot delete it, in case the master journal file name was written into the journal
		// file before the failure occurred.
		for (i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
				rc = bt->CommitPhaseOne(masterName);
		}
		master->CloseAndFree();
		_assert(rc != RC_BUSY);
		if (rc != RC_OK)
		{
			_tagfree(ctx, masterName);
			return rc;
		}

		// Delete the master journal file. This commits the transaction. After doing this the directory is synced again before any individual
		// transaction files are deleted.
		rc = vfs->Delete(masterName, true);
		_tagfree(ctx, masterName);
		masterName = nullptr;
		if (rc)
			return rc;

		// All files and directories have already been synced, so the following calls to sqlite3BtreeCommitPhaseTwo() are only closing files and
		// deleting or truncating journals. If something goes wrong while this is happening we don't really care. The integrity of the
		// transaction is already guaranteed, but some stray 'cold' journals may be lying around. Returning an error code won't help matters.
		disable_simulated_io_errors();
		_benignalloc_begin();
		for (i = 0; i < ctx->DBs.length; i++)
		{
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
				bt->CommitPhaseTwo(true);
		}
		_benignalloc_end();
		enable_simulated_io_errors();

		VTable::Commit(ctx);
	}
#endif
	return rc;
}

#ifndef NDEBUG
__device__ static void CheckActiveVdbeCnt(Context *ctx)
{
	int cnt = 0;
	int writes = 0;
	Vdbe *p = ctx->Vdbes;
	while (p)
	{
		if (p->Magic == VDBE_MAGIC_RUN && p->PC >= 0)
		{
			cnt++;
			if (!p->ReadOnly) writes++;
		}
		p = p->Next;
	}
	_assert(cnt == ctx->ActiveVdbeCnt);
	_assert(writes == ctx->WriteVdbeCnt);
}
#else
#define CheckActiveVdbeCnt(x)
#endif

__device__ RC Vdbe::CloseStatement(IPager::SAVEPOINT op)
{
	Context *const ctx = Ctx;
	RC rc = RC_OK;

	// If statementID is greater than zero, then this Vdbe opened a statement transaction that should be closed here. The only exception
	// is that an IO error may have occurred, causing an emergency rollback. In this case (db->nStatement==0), and there is nothing to do.
	if (ctx->Statements && StatementID)
	{
		const int savepoint = StatementID-1;

		_assert(op == IPager::SAVEPOINT_ROLLBACK || op == IPager::SAVEPOINT_RELEASE);
		_assert(ctx->Statements > 0);
		_assert(StatementID == (ctx->Statements + ctx->SavepointsLength));

		for (int i = 0; i < ctx->DBs.length; i++)
		{
			RC rc2 = RC_OK;
			Btree *bt = ctx->DBs[i].Bt;
			if (bt)
			{
				if (op == IPager::SAVEPOINT_ROLLBACK)
					rc2 = bt->Savepoint(IPager::SAVEPOINT_ROLLBACK, savepoint);
				if (rc2 == RC_OK)
					rc2 = bt->Savepoint(IPager::SAVEPOINT_RELEASE, savepoint);
				if (rc == RC_OK)
					rc = rc2;
			}
		}
		ctx->Statements--;
		StatementID = 0;

		if (rc == RC_OK)
		{
			if (op == IPager::SAVEPOINT_ROLLBACK)
				rc = VTable::Savepoint(ctx, IPager::SAVEPOINT_ROLLBACK, savepoint);
			if (rc == RC_OK)
				rc = VTable::Savepoint(ctx, IPager::SAVEPOINT_RELEASE, savepoint);
		}

		// If the statement transaction is being rolled back, also restore the database handles deferred constraint counter to the value it had when 
		// the statement transaction was opened.
		if (op == IPager::SAVEPOINT_ROLLBACK)
			ctx->DeferredCons = StmtDefCons;
	}
	return rc;
}

#ifndef OMIT_FOREIGN_KEY
__device__ RC Vdbe::CheckFk(bool deferred)
{
	Context *ctx = Ctx;
	if ((deferred && ctx->DeferredCons > 0) || (!deferred && FkConstraints > 0))
	{
		RC_ = RC_CONSTRAINT_FOREIGNKEY;
		ErrorAction = OE_Abort;
		_setstring(&ErrMsg, ctx, "foreign key constraint failed");
		return RC_ERROR;
	}
	return RC_OK;
}
#endif

__device__ RC Vdbe::Halt()
{
	RC rc;
	Context *ctx = Ctx;

	// This function contains the logic that determines if a statement or transaction will be committed or rolled back as a result of the
	// execution of this virtual machine. 
	//
	// If any of the following errors occur:
	//
	//     RC_NOMEM
	//     RC_IOERR
	//     RC_FULL
	//     RC_INTERRUPT
	//
	// Then the internal cache might have been left in an inconsistent state.  We need to rollback the statement transaction, if there is
	// one, or the complete transaction if there is no statement transaction.
	if (ctx->MallocFailed)
		RC_ = RC_NOMEM;

	if (OnceFlags.data) _memset(OnceFlags.data, 0, OnceFlags.length);
	CloseAllCursors(this);
	if (Magic != VDBE_MAGIC_RUN)
		return RC_OK;
	CheckActiveVdbeCnt(ctx);

	// No commit or rollback needed if the program never started
	if (PC >= 0)
	{
		IPager::SAVEPOINT statementOp = (IPager::SAVEPOINT)0;
		// Lock all btrees used by the statement
		Enter();

		// Check for one of the special errors
		RC mrc = (RC)(RC_ & 0xff); // Primary error code from p->rc
		_assert(RC_ != RC_IOERR_BLOCKED); // This error no longer exists
		bool isSpecialError = (mrc == RC_NOMEM || mrc == RC_IOERR || mrc == RC_INTERRUPT || mrc == RC_FULL); // Set to true if a 'special' error
		if (isSpecialError)
		{
			// If the query was read-only and the error code is SQLITE_INTERRUPT, no rollback is necessary. Otherwise, at least a savepoint 
			// transaction must be rolled back to restore the database to a consistent state.
			//
			// Even if the statement is read-only, it is important to perform a statement or transaction rollback operation. If the error 
			// occurred while writing to the journal, sub-journal or database file as part of an effort to free up cache space (see function
			// pagerStress() in pager.c), the rollback is required to restore the pager to a consistent state.
			if (!ReadOnly || mrc != RC_INTERRUPT)
			{
				if ((mrc == RC_NOMEM || mrc == RC_FULL) && UsesStmtJournal)
					statementOp = IPager::SAVEPOINT_ROLLBACK;
				else
				{
					// We are forced to roll back the active transaction. Before doing so, abort any other statements this handle currently has active.
					Main::RollbackAll(ctx, RC_ABORT_ROLLBACK);
					Main::CloseSavepoints(ctx);
					ctx->AutoCommit = 1;
				}
			}
		}

		// Check for immediate foreign key violations.
		if (RC_ == RC_OK)
			CheckFk(false);

		// If the auto-commit flag is set and this is the only active writer VM, then we do either a commit or rollback of the current transaction. 
		//
		// Note: This block also runs if one of the special errors handled above has occurred. 
		if (!VTable::InSync(ctx) && ctx->AutoCommit && ctx->WriteVdbeCnt == (!ReadOnly))
		{
			if (RC_ == RC_OK || (ErrorAction == OE_Fail && !isSpecialError))
			{
				rc = CheckFk(true);
				if (rc != RC_OK)
				{
					if (_NEVER(ReadOnly))
					{
						Leave();
						return RC_ERROR;
					}
					rc = RC_CONSTRAINT_FOREIGNKEY;
				}
				else
					// The auto-commit flag is true, the vdbe program was successful or hit an 'OR FAIL' constraint and there are no deferred foreign
					// key constraints to hold up the transaction. This means a commit is required.
					rc = VdbeCommit(ctx, this);
				if (rc == RC_BUSY && ReadOnly)
				{
					Leave();
					return RC_BUSY;
				}
				else if (rc != RC_OK)
				{
					RC_ = rc;
					Main::RollbackAll(ctx, RC_OK);
				}
				else
				{
					ctx->DeferredCons = 0;
					Parse::CommitInternalChanges(ctx);
				}
			}
			else
				Main::RollbackAll(ctx, RC_OK);
			ctx->Statements = 0;
		}
		else if (statementOp == 0)
		{
			if (RC_ == RC_OK || ErrorAction == OE_Fail)
				statementOp = IPager::SAVEPOINT_RELEASE;
			else if (ErrorAction == OE_Abort)
				statementOp = IPager::SAVEPOINT_ROLLBACK;
			else
			{
				Main::RollbackAll(ctx, RC_ABORT_ROLLBACK);
				Main::CloseSavepoints(ctx);
				ctx->AutoCommit = 1;
			}
		}

		// If eStatementOp is non-zero, then a statement transaction needs to be committed or rolled back. Call sqlite3VdbeCloseStatement() to
		// do so. If this operation returns an error, and the current statement error code is SQLITE_OK or SQLITE_CONSTRAINT, then promote the
		// current statement error code.
		if (statementOp)
		{
			rc = CloseStatement(statementOp);
			if (rc)
			{
				if (RC_ == RC_OK || (RC_&0xff) == RC_CONSTRAINT)
				{
					RC_ = rc;
					_tagfree(ctx, ErrMsg);
					ErrMsg = nullptr;
				}
				Main::RollbackAll(ctx, RC_ABORT_ROLLBACK);
				Main::CloseSavepoints(ctx);
				ctx->AutoCommit = 1;
			}
		}

		// If this was an INSERT, UPDATE or DELETE and no statement transaction has been rolled back, update the database connection change-counter. 
		if (ChangeCntOn)
		{
			SetChanges(ctx, (statementOp != IPager::SAVEPOINT_ROLLBACK ? Changes : 0));
			Changes = 0;
		}

		// Release the locks
		Leave();
	}

	// We have successfully halted and closed the VM.  Record this fact.
	if (PC >= 0)
	{
		ctx->ActiveVdbeCnt--;
		if (!ReadOnly)
			ctx->WriteVdbeCnt--;
		_assert(ctx->ActiveVdbeCnt >= ctx->WriteVdbeCnt);
	}
	Magic = VDBE_MAGIC_HALT;
	CheckActiveVdbeCnt(ctx);
	if (ctx->MallocFailed)
		RC_ = RC_NOMEM;

	// If the auto-commit flag is set to true, then any locks that were held by connection ctx have now been released. Call sqlite3ConnectionUnlocked() 
	// to invoke any required unlock-notify callbacks.
	if (ctx->AutoCommit)
		ctx->ConnectionUnlocked();

	_assert(ctx->ActiveVdbeCnt > 0 || ctx->AutoCommit == 0 || ctx->Statements == 0);
	return (RC_ == RC_BUSY ? RC_BUSY : RC_OK);
}

__device__ void Vdbe::ResetStepResult()
{
	RC_ = RC_OK;
}

__device__ RC Vdbe::TransferError()
{
	Context *ctx = Ctx;
	RC rc = RC_;
	if (ErrMsg)
	{
		bool mallocFailed = ctx->MallocFailed;
		_benignalloc_begin();
		Vdbe::ValueSetStr(ctx->Err, -1, ErrMsg, TEXTENCODE_UTF8, DESTRUCTOR_TRANSIENT);
		_benignalloc_end();
		ctx->MallocFailed = mallocFailed;
		ctx->ErrCode = rc;
	}
	else
		Main::Error(ctx, rc, nullptr);
	return rc;
}

#ifdef ENABLE_SQLLOG
static void VdbeInvokeSqllog(Vdbe *p)
{
	if (_Sqllog && p->RC_ == RC_OK && p->Sql_ && p->PC >= 0)
	{
		char *expanded = p->ExpandSql(p->Sql_);
		_assert(!p->Ctx->Init.Busy);
		if (expanded)
		{
			_Sqllog(_SqllogArg, p->Ctx, expanded, 1);
			_tagfree(p->Ctx, expanded);
		}
	}
}
#else
#define VdbeInvokeSqllog(x)
#endif

__device__ RC Vdbe::Reset()
{
	Context *ctx = Ctx;

	// If the VM did not run to completion or if it encountered an error, then it might not have been halted properly.  So halt it now.
	Halt();

	// If the VDBE has be run even partially, then transfer the error code and error message from the VDBE into the main database structure.  But
	// if the VDBE has just been set to run but has not actually executed any instructions yet, leave the main database error information unchanged.
	if (PC >= 0)
	{
		VdbeInvokeSqllog(this);
		TransferError();
		_tagfree(ctx, ErrMsg);
		ErrMsg = nullptr;
		if (RunOnlyOnce) Expired = true;
	}
	else if (RC_ && Expired)
	{
		// The expired flag was set on the VDBE before the first call to sqlite3_step(). For consistency (since sqlite3_step() was
		// called), set the database error in this case as well.
		Main::Error(ctx, RC_, nullptr);
		Vdbe::ValueSetStr(ctx->Err, -1, ErrMsg, TEXTENCODE_UTF8, DESTRUCTOR_TRANSIENT);
		_tagfree(ctx, ErrMsg);
		ErrMsg = nullptr;
	}

	// Reclaim all memory used by the VDBE
	Cleanup(this);

	// Save profiling information from this VDBE run.
#ifdef VDBE_PROFILE
	{
		FILE *out_ = fopen("x64\\vdbe_profile.out", "a");
		if (out_)
		{
			int i;
			fprintf(out_, "---- ");
			for (i = 0; i < Ops.length; i++)
				fprintf(out_, "%02x", Ops[i].Opcode);
			fprintf(out_, "\n");
			for (i = 0; i < Ops.length; i++)
			{
				fprintf(out_, "%6d %10lld %8lld ", Ops[i].Cnt, Ops[i].Cycles, (Ops[i].Cnt > 0 ? Ops[i].Cycles/Ops[i].Cnt : 0));
				PrintOp(out_, i, &Ops[i]);
			}
			fclose(out_);
		}
	}
#endif
	Magic = VDBE_MAGIC_INIT;
	return (RC)(RC_ & ctx->ErrMask);
}

__device__ RC Vdbe::Finalize()
{
	RC rc = RC_OK;
	if (Magic == VDBE_MAGIC_RUN || Magic == VDBE_MAGIC_HALT)
	{
		rc = Reset();
		_assert((rc & Ctx->ErrMask) == rc);
	}
	Delete(this);
	return rc;
}

__device__ void Vdbe::DeleteAuxData(VdbeFunc *func, int mask)
{
	for (int i = 0; i < func->AuxsLength; i++)
	{
		VdbeFunc::AuxData *aux = &func->Auxs[i];
		if ((i > 31 || !(mask&(((uint32)1)<<i))) && aux->Aux)
		{
			if (aux->Delete)
				aux->Delete(aux->Aux);
			aux->Aux = nullptr;
		}
	}
}

__device__ void Vdbe::ClearObject(Context *ctx)
{
	_assert(Ctx == nullptr || Ctx == ctx);
	ReleaseMemArray(Vars.data, Vars.length);
	ReleaseMemArray(ColNames, ResColumns*COLNAME_N);
	SubProgram *sub, *next;
	for (sub = Programs; sub; sub = next)
	{
		next = sub->Next;
		VdbeFreeOpArray(ctx, sub->Ops.data, sub->Ops.length);
		_tagfree(ctx, sub);
	}
	for (int i = VarNames.length-1; i >= 0; i--) _tagfree(ctx, VarNames[i]);
	VdbeFreeOpArray(ctx, Ops.data, Ops.length);
	_tagfree(ctx, Labels.data);
	_tagfree(ctx, ColNames);
	_tagfree(ctx, Sql_);
	_tagfree(ctx, FreeThis);
#if defined(ENABLE_TREE_EXPLAIN)
	_tagfree(ctx, _explain);
	_tagfree(ctx, _explainString);
#endif
}

__device__ void Vdbe::Delete(Vdbe *p)
{
	if (_NEVER(p == nullptr)) return;
	Context *ctx = p->Ctx;
	_assert(MutexEx::Held(ctx->Mutex));
	p->ClearObject(ctx);
	if (p->Prev)
		p->Prev->Next = p->Next;
	else
	{
		_assert(ctx->Vdbes == p);
		ctx->Vdbes = p->Next;
	}
	if (p->Next)
		p->Next->Prev = p->Prev;
	p->Magic = VDBE_MAGIC_DEAD;
	p->Ctx = nullptr;
	_tagfree(ctx, p);
}

__device__ RC Vdbe::CursorMoveto(VdbeCursor *p)
{
	if (p->DeferredMoveto)
	{
		_assert(p->IsTable);
		int res;
		RC rc = Btree::MovetoUnpacked(p->Cursor, 0, p->MovetoTarget, 0, &res);
		if (rc) return rc;
		p->LastRowid = p->MovetoTarget;
		if (res != 0) return SysEx_CORRUPT_BKPT;
		p->RowidIsValid = true;
#ifdef TEST
		g_search_count++;
#endif
		p->DeferredMoveto = false;
		p->CacheStatus = CACHE_STALE;
	}
	else if (_ALWAYS(p->Cursor))
	{
		bool hasMoved;
		RC rc = Btree::CursorHasMoved(p->Cursor, &hasMoved);
		if (rc) return rc;
		if (hasMoved)
		{
			p->CacheStatus = CACHE_STALE;
			p->NullRow = true;
		}
	}
	return RC_OK;
}

#pragma region Serialize & UnpackedRecord

__device__ uint32 Vdbe::SerialType(Mem *mem, int fileFormat)
{
	MEM flags = mem->Flags;
	if (flags & MEM_Null) return 0;
	if (flags & MEM_Int)
	{
		// Figure out whether to use 1, 2, 4, 6 or 8 bytes.
#define MAX_6BYTE ((((int64)0x00008000)<<32)-1)
		int64 i = mem->u.I;
		uint64 u = (i < 0 ? (i < -MAX_6BYTE ? 6 : -i) : i); // MAX_6BYTE test prevents: u = -(-9223372036854775808)
		if (u <= 127) return ((i&1) == i && fileFormat >= 4 ? 8+(uint32)u : 1);
		if (u <= 32767) return 2;
		if (u <= 8388607) return 3;
		if (u <= 2147483647) return 4;
		if (u <= MAX_6BYTE) return 5;
		return 6;
	}
	if (flags & MEM_Real) return 7;
	_assert(mem->Ctx->MallocFailed || flags & (MEM_Str|MEM_Blob));
	int n = mem->N;
	if (flags & MEM_Zero)
		n += mem->u.Zeros;
	_assert(n >= 0);
	return ((n*2) + 12 + ((flags & MEM_Str) != 0));
}

__constant__ static const uint8 _serialTypeSize[] = { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
__device__ uint32 Vdbe::SerialTypeLen(uint32 serialType)
{
	if (serialType >= 12)
		return (serialType-12) / 2;
	return _serialTypeSize[serialType];
}

#ifdef MIXED_ENDIAN_64BIT_FLOAT
__device__ static uint64 FloatSwap(uint64 in_)
{
	union
	{
		uint64 R;
		uint32 I[2];
	} u;
	u.R = in_;
	uint32 t = u.I[0];
	u.I[0] = u.I[1];
	u.I[1] = t;
	return u.R;
}
#define SwapMixedEndianFloat(X) X = FloatSwap(X)
#else
#define SwapMixedEndianFloat(X)
#endif

__device__ uint32 Vdbe::SerialPut(uint8 *buf, int bufLength, Mem *mem, int fileFormat)
{
	uint32 serialType = SerialType(mem, fileFormat);
	uint32 len;

	// Integer and Real
	if (serialType <= 7 && serialType > 0)
	{
		uint64 v;
		if (serialType == 7)
		{
			_assert(sizeof(v) == sizeof(mem->R));
			_memcpy((uint8 *)&v, (uint8 *)&mem->R, sizeof(v));
			SwapMixedEndianFloat(v);
		}
		else
			v = mem->u.I;
		uint32 i;
		len = i = SerialTypeLen(serialType);
		_assert(len <= (uint32)bufLength);
		while (i--)
		{
			buf[i] = (uint8)(v & 0xFF);
			v >>= 8;
		}
		return len;
	}

	// String or blob
	if (serialType >= 12)
	{
		_assert(mem->N + ((mem->Flags & MEM_Zero)?mem->u.Zeros:0) == (int)SerialTypeLen(serialType));
		_assert(mem->N <= bufLength);
		len = mem->N;
		_memcpy(buf, (uint8 *)mem->Z, len);
		if (mem->Flags & MEM_Zero)
		{
			len += mem->u.Zeros;
			assert(bufLength >= 0);
			if (len > (uint32)bufLength)
				len = (uint32)bufLength;
			_memset(&buf[mem->N], 0, len - mem->N);
		}
		return len;
	}

	// NULL or constants 0 or 1
	return 0;
}

__device__ uint32 Vdbe::SerialGet(const unsigned char *buf, uint32 serialType, Mem *mem)
{
	switch (serialType)
	{
	case 10: // Reserved for future use
	case 11: // Reserved for future use
	case 0: { // NULL
		mem->Flags = MEM_Null;
		break; }
	case 1: { // 1-byte signed integer
		mem->u.I = (signed char)buf[0];
		mem->Flags = MEM_Int;
		return 1; }
	case 2: { // 2-byte signed integer
		mem->u.I = (((signed char)buf[0])<<8) | buf[1];
		mem->Flags = MEM_Int;
		return 2; }
	case 3: { // 3-byte signed integer
		mem->u.I = (((signed char)buf[0])<<16) | (buf[1]<<8) | buf[2];
		mem->Flags = MEM_Int;
		return 3; }
	case 4: { // 4-byte signed integer
		mem->u.I = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
		mem->Flags = MEM_Int;
		return 4; }
	case 5: { // 6-byte signed integer
		uint64 x = (((signed char)buf[0])<<8) | buf[1];
		uint32 y = (buf[2]<<24) | (buf[3]<<16) | (buf[4]<<8) | buf[5];
		x = (x<<32) | y;
		mem->u.I = *(int64*)&x;
		mem->Flags = MEM_Int;
		return 6; }
	case 6: // 8-byte signed integer
	case 7: { // IEEE floating point
#if !defined(NDEBUG) && !defined(OMIT_FLOATING_POINT)
		// Verify that integers and floating point values use the same byte order.  Or, that if SQLITE_MIXED_ENDIAN_64BIT_FLOAT is
		// defined that 64-bit floating point values really are mixed endian.
		static const uint64 t1 = ((uint64)0x3ff00000)<<32;
		static const double r1 = 1.0;
		uint64 t2 = t1;
		SwapMixedEndianFloat(t2);
		_assert(sizeof(r1) == sizeof(t2) && _memcmp(&r1, &t2, sizeof(r1)) == 0);
#endif
		uint64 x = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
		uint32 y = (buf[4]<<24) | (buf[5]<<16) | (buf[6]<<8) | buf[7];
		x = (x<<32) | y;
		if (serialType == 6)
		{
			mem->u.I = *(int64*)&x;
			mem->Flags = MEM_Int;
		}
		else
		{
			_assert(sizeof(x) == 8 && sizeof(mem->R) == 8);
			SwapMixedEndianFloat(x);
			_memcpy((uint8 *)&mem->R, (uint8 *)&x, sizeof(x));
			mem->Flags = (_isnan(mem->R) ? MEM_Null : MEM_Real);
		}
		return 8; }
	case 8: // Integer 0
	case 9: { // Integer 1
		mem->u.I = serialType-8;
		mem->Flags = MEM_Int;
		return 0; }
	default: {
		uint32 len = (serialType-12)/2;
		mem->Z = (char *)buf;
		mem->N = len;
		mem->Del = nullptr;
		if (serialType&0x01)
			mem->Flags = MEM_Str | MEM_Ephem;
		else
			mem->Flags = MEM_Blob | MEM_Ephem;
		return len; }
	}
	return 0;
}

__device__ UnpackedRecord *Vdbe::AllocUnpackedRecord(KeyInfo *keyInfo, char *space, int spaceLength, char **freeOut)
{
	// We want to shift the pointer pSpace up such that it is 8-byte aligned. Thus, we need to calculate a value, nOff, between 0 and 7, to shift 
	// it by.  If pSpace is already 8-byte aligned, nOff should be zero.
	int offset = (8 - (PTR_TO_INT(space) & 7)) & 7; // Increment pSpace by nOff to align it
	int bytes = _ROUND8(sizeof(UnpackedRecord)) + sizeof(Mem)*(keyInfo->Fields+1); // Number of bytes required for *p
	UnpackedRecord *p; // Unpacked record to return
	if (bytes > spaceLength+offset)
	{
		p = (UnpackedRecord *)_tagalloc(keyInfo->Ctx, bytes);
		*freeOut = (char *)p;
		if (!p) return nullptr;
	}
	else
	{
		p = (UnpackedRecord *)&space[offset];
		*freeOut = nullptr;
	}
	p->Mems = (Mem *)&((char *)p)[_ROUND8(sizeof(UnpackedRecord))];
	_assert(keyInfo->SortOrders != nullptr);
	p->KeyInfo = keyInfo;
	p->Fields = keyInfo->Fields + 1;
	return p;
}

__device__ void Vdbe::RecordUnpack(KeyInfo *keyInfo, int keyLength, const void *key, UnpackedRecord *p)
{
	const unsigned char *keys = (const unsigned char *)key;
	Mem *mem = p->Mems;
	p->Flags = (UNPACKED)0;
	_assert(_HASALIGNMENT8(mem));
	uint32 szHdr;
	uint32 idx = ConvertEx_GetVarint32(keys, szHdr); // Offset in keys[] to read from
	int d = szHdr;
	uint16 u = 0; // Unsigned loop counter
	while (idx < szHdr && u < p->Fields && d <= keyLength)
	{
		uint32 serialType;
		idx += ConvertEx_GetVarint32(&keys[idx], serialType);
		mem->Encode = keyInfo->Encode;
		mem->Ctx = (Context *)keyInfo->Ctx;
		//mem->Flags = 0; // sqlite3VdbeSerialGet() will set this for us
		mem->Malloc = nullptr;
		d += SerialGet(&keys[d], serialType, mem);
		mem++;
		u++;
	}
	_assert(u <= keyInfo->Fields + 1 );
	p->Fields = u;
}

__device__ int Vdbe::RecordCompare(int key1Length, const void *key1, UnpackedRecord *key2)
{
	int i = 0;
	int rc = 0;
	const unsigned char *key1s = (const unsigned char *)key1;

	KeyInfo *keyInfo = key2->KeyInfo;
	Mem mem1;
	mem1.Encode = keyInfo->Encode;
	mem1.Ctx = (Context *)keyInfo->Ctx;
	// mem1.flags = 0; // Will be initialized by sqlite3VdbeSerialGet()
	{
		ASSERTONLY(mem1.Malloc = nullptr;) // Only needed by assert() statements
	}

	// Compilers may complain that mem1.u.i is potentially uninitialized. We could initialize it, as shown here, to silence those complaints.
	// But in fact, mem1.u.i will never actually be used uninitialized, and doing the unnecessary initialization has a measurable negative performance
	// impact, since this routine is a very high runner.  And so, we choose to ignore the compiler warnings and leave this variable uninitialized.
	//  mem1.u.i = 0;  // not needed, here to silence compiler warning
	uint32 szHdr1; // Number of bytes in header
	uint32 idx1 = ConvertEx_GetVarint32(key1s, szHdr1); // Offset into keys[] of next header element
	int d1 = szHdr1; // Offset into keys[] of next data element
	int fields = keyInfo->Fields;
	_assert(keyInfo->SortOrders != nullptr);
	while (idx1 < szHdr1 && i < key2->Fields)
	{
		uint32 serialType1;

		// Read the serial types for the next element in each key.
		idx1 += ConvertEx_GetVarint32(key1s+idx1, serialType1);
		if (d1 >= key1Length && SerialTypeLen(serialType1) > 0) break;

		// Extract the values to be compared.
		d1 += SerialGet(&key1s[d1], serialType1, &mem1);

		// Do the comparison
		rc = MemCompare(&mem1, &key2->Mems[i], (i < fields ? keyInfo->Colls[i] : nullptr));
		if (rc != 0)
		{
			_assert(mem1.Malloc == nullptr); // See comment below

			// Invert the result if we are using DESC sort order.
			if (i < fields && keyInfo->SortOrders[i])
				rc = -rc;

			// If the PREFIX_SEARCH flag is set and all fields except the final rowid field were equal, then clear the PREFIX_SEARCH flag and set 
			// pPKey2->rowid to the value of the rowid field in (pKey1, nKey1). This is used by the OP_IsUnique opcode.
			if ((key2->Flags & UNPACKED_PREFIX_SEARCH) && i == (key2->Fields - 1))
			{
				_assert(idx1 == szHdr1 && rc);
				_assert(mem1.Flags & MEM_Int);
				key2->Flags &= ~UNPACKED_PREFIX_SEARCH;
				key2->Rowid = mem1.u.I;
			}
			return rc;
		}
		i++;
	}

	// No memory allocation is ever used on mem1.  Prove this using the following assert().  If the assert() fails, it indicates a
	// memory leak and a need to call sqlite3VdbeMemRelease(&mem1).
	_assert(mem1.Malloc == nullptr);

	// rc==0 here means that one of the keys ran out of fields and all the fields up to that point were equal. If the UNPACKED_INCRKEY
	// flag is set, then break the tie by treating key2 as larger. If the UPACKED_PREFIX_MATCH flag is set, then keys with common prefixes
	// are considered to be equal.  Otherwise, the longer key is the larger.  As it happens, the pPKey2 will always be the longer
	// if there is a difference.
	_assert(rc == 0);
	if (key2->Flags & UNPACKED_INCRKEY) rc = -1;
	else if (key2->Flags & UNPACKED_PREFIX_MATCH) { } // Leave rc==0 
	else if (idx1 < szHdr1) rc = 1;
	return rc;
}

#pragma endregion

#pragma region Index Entry

__device__ RC Vdbe::IdxRowid(Context *ctx, BtCursor *cur, int64 *rowid)
{
	// Get the size of the index entry.  Only indices entries of less than 2GiB are support - anything large must be database corruption.
	// Any corruption is detected in sqlite3BtreeParseCellPtr(), though, so this code can safely assume that nCellKey is 32-bits  
	_assert(Btree::CursorIsValid(cur));
	int64 cellKeyLength = 0;
	RC rc; ASSERTONLY(rc =)Btree::KeySize(cur, &cellKeyLength);
	_assert(rc == RC_OK); // pCur is always valid so KeySize cannot fail
	_assert((cellKeyLength & MAX_UTYPE(uint32)) == (uint64)cellKeyLength);

	// Read in the complete content of the index entry
	Mem m;
	_memset(&m, 0, sizeof(m));
	rc = MemFromBtree(cur, 0, (int)cellKeyLength, true, &m);
	if (rc)
		return rc;

	// The index entry must begin with a header size
	uint32 szHdr; // Size of the header
	ConvertEx_GetVarint32((uint8 *)m.Z, szHdr);
	ASSERTCOVERAGE(szHdr == 3);
	ASSERTCOVERAGE(szHdr == m.N);
	if (unlikely(szHdr<3 || (int)szHdr>m.N))
		goto idx_rowid_corruption;

	// The last field of the index should be an integer - the ROWID. Verify that the last entry really is an integer.
	uint32 typeRowid; // Serial type of the rowid
	ConvertEx_GetVarint32((uint8 *)&m.Z[szHdr-1], typeRowid);
	ASSERTCOVERAGE(typeRowid == 1);
	ASSERTCOVERAGE(typeRowid == 2);
	ASSERTCOVERAGE(typeRowid == 3);
	ASSERTCOVERAGE(typeRowid == 4);
	ASSERTCOVERAGE(typeRowid == 5);
	ASSERTCOVERAGE(typeRowid == 6);
	ASSERTCOVERAGE(typeRowid == 8);
	ASSERTCOVERAGE(typeRowid == 9);
	if (unlikely(typeRowid < 1 || typeRowid > 9 || typeRowid == 7))
		goto idx_rowid_corruption;
	uint32 lenRowid = SerialTypeLen(typeRowid); // Size of the rowid
	ASSERTCOVERAGE((uint32)m.N == szHdr+lenRowid);
	if (unlikely((uint32)m.N < szHdr+lenRowid))
		goto idx_rowid_corruption;

	// Fetch the integer off the end of the index record
	Mem v;
	SerialGet((uint8 *)&m.Z[m.N - lenRowid], typeRowid, &v);
	*rowid = v.u.I;
	MemRelease(&m);
	return RC_OK;

	// Jump here if database corruption is detected after m has been allocated.  Free the m object and return SQLITE_CORRUPT.
idx_rowid_corruption:
	ASSERTCOVERAGE(m.Malloc != nullptr);
	MemRelease(&m);
	return SysEx_CORRUPT_BKPT;
}

__device__ RC Vdbe::IdxKeyCompare(VdbeCursor *c, UnpackedRecord *unpacked, int *r)
{
	BtCursor *cur = c->Cursor;
	Mem m;
	_assert(Btree::CursorIsValid(cur));
	int64 cellKeyLength = 0;
	RC rc; ASSERTONLY(rc =)Btree::KeySize(cur, &cellKeyLength);
	_assert(rc == RC_OK); // pCur is always valid so KeySize cannot fail
	// nCellKey will always be between 0 and 0xffffffff because of the say that btreeParseCellPtr() and sqlite3GetVarint32() are implemented
	if (cellKeyLength <= 0 || cellKeyLength > 0x7fffffff)
	{
		*r = 0;
		return SysEx_CORRUPT_BKPT;
	}
	_memset(&m, 0, sizeof(m));
	rc = MemFromBtree(cur, 0, (int)cellKeyLength, true, &m);
	if (rc)
		return rc;
	_assert(unpacked->Flags & UNPACKED_PREFIX_MATCH);
	*r = RecordCompare(m.N, m.Z, unpacked);
	MemRelease(&m);
	return RC_OK;
}

#pragma endregion


__device__ void Vdbe::SetChanges(Context *ctx, int changes)
{
	_assert(MutexEx::Held(ctx->Mutex));
	ctx->Changes = changes;
	ctx->TotalChanges += changes;
}

__device__ void Vdbe::CountChanges()
{
	ChangeCntOn = 1;
}

__device__ void Vdbe::ExpirePreparedStatements(Context *ctx)
{
	for (Vdbe *p = ctx->Vdbes; p; p = p->Next)
		p->Expired = true;
}

__device__ Context *Vdbe::get_Ctx()
{
	return Ctx;
}

__device__ Mem *Vdbe::GetValue(int var, AFF aff)
{
	_assert(var > 0);
	Mem *mem = &Vars[var-1];
	if ((mem->Flags & MEM_Null) == 0)
	{
		Mem *r = ValueNew(Ctx);
		if (r)
		{
			MemCopy(r, mem);
			Vdbe::ValueApplyAffinity(r, aff, TEXTENCODE_UTF8);
			MemStoreType(r);
		}
		return r;
	}
	return nullptr;
}

__device__ void Vdbe::SetVarmask(int var)
{
	_assert(var > 0);
	if (var > 32)
		Expmask = 0xffffffff;
	else
		Expmask |= ((uint32)1 << (var-1));
}

#pragma endregion