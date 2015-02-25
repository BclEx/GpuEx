#include "Core+Vdbe.cu.h"
#include <new.h>

namespace Core
{
	__device__ static void CallCollNeeded(Context *ctx, TEXTENCODE encode, const char *name)
	{
		_assert(!ctx->CollNeeded || !ctx->CollNeeded16);
		if (ctx->CollNeeded)
		{
			char *external = _tagstrdup(ctx, name);
			if (!external) return;
			ctx->CollNeeded(ctx->CollNeededArg, ctx, encode, external);
			_tagfree(ctx, external);
		}
#ifndef OMIT_UTF16
		if (ctx->CollNeeded16)
		{
			Mem *tmp = Vdbe::ValueNew(ctx);
			Vdbe::ValueSetStr(tmp, -1, name, TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
			char const *external = (char const *)Vdbe::ValueText(tmp, TEXTENCODE_UTF16NATIVE);
			if (external)
				ctx->CollNeeded16(ctx->CollNeededArg, ctx, CTXENCODE(ctx), external);
			Vdbe::ValueFree(tmp);
		}
#endif
	}

	__device__ static const TEXTENCODE _textEncodes[] = { TEXTENCODE_UTF16BE, TEXTENCODE_UTF16LE, TEXTENCODE_UTF8 };
	__device__ static RC SynthCollSeq(Context *ctx, CollSeq *coll)
	{
		char *z = coll->Name;
		for (int i = 0; i < 3; i++)
		{
			CollSeq *coll2 = Callback::FindCollSeq(ctx, _textEncodes[i], z, false);
			if (coll2->Cmp)
			{
				_memcpy(coll, coll2, sizeof(CollSeq));
				coll->Del = nullptr; // Do not copy the destructor
				return RC_OK;
			}
		}
		return RC_ERROR;
	}

	__device__ CollSeq *Callback::GetCollSeq(Parse *parse, TEXTENCODE encode, CollSeq *coll, const char *name)
	{
		Context *ctx = parse->Ctx;
		CollSeq *p = coll;
		if (!p)
			p = FindCollSeq(ctx, encode, name, false);
		if (!p || !p->Cmp)
		{
			// No collation sequence of this type for this encoding is registered. Call the collation factory to see if it can supply us with one.
			CallCollNeeded(ctx, encode, name);
			p = FindCollSeq(ctx, encode, name, false);
		}
		if (p && !p->Cmp && SynthCollSeq(ctx, p))
			p = nullptr;
		_assert(!p || p->Cmp);
		if (!p)
			parse->ErrorMsg("no such collation sequence: %s", name);
		return p;
	}

	__device__ RC Callback::CheckCollSeq(Parse *parse, CollSeq *coll)
	{
		if (coll)
		{
			CollSeq *p = GetCollSeq(parse, CTXENCODE(parse->Ctx), coll, coll->Name);
			if (!p)
				return RC_ERROR;
			_assert(p == coll);
		}
		return RC_OK;
	}

	__device__ static CollSeq *FindCollSeqEntry(Context *ctx, const char *name, bool create)
	{
		int nameLength = _strlen30(name);
		CollSeq *coll = (CollSeq *)ctx->CollSeqs.Find(name, nameLength);
		if (!coll && create)
		{
			coll = (CollSeq *)_tagalloc2(ctx, 3*sizeof(*coll) + nameLength + 1, true);
			if (coll)
			{
				coll[0].Name = (char *)&coll[3];
				coll[0].Encode = TEXTENCODE_UTF8;
				coll[1].Name = (char *)&coll[3];
				coll[1].Encode = TEXTENCODE_UTF16LE;
				coll[2].Name = (char *)&coll[3];
				coll[2].Encode = TEXTENCODE_UTF16BE;
				_memcpy(coll[0].Name, name, nameLength);
				coll[0].Name[nameLength] = 0;
				CollSeq *del = (CollSeq *)ctx->CollSeqs.Insert(coll[0].Name, nameLength, coll);
				// If a malloc() failure occurred in sqlite3HashInsert(), it will return the pColl pointer to be deleted (because it wasn't added to the hash table).
				_assert(!del || del == coll);
				if (del)
				{
					ctx->MallocFailed = true;
					_tagfree(ctx, del);
					coll = nullptr;
				}
			}
		}
		return coll;
	}

	__device__ CollSeq *Callback::FindCollSeq(Context *ctx, TEXTENCODE encode, const char *name, bool create)
	{
		CollSeq *colls = (name ? FindCollSeqEntry(ctx, name, create) : ctx->DefaultColl);
		_assert(TEXTENCODE_UTF8 == 1 && TEXTENCODE_UTF16LE == 2 && TEXTENCODE_UTF16BE == 3);
		_assert(encode >= TEXTENCODE_UTF8 && encode <= TEXTENCODE_UTF16BE);
		return (colls != nullptr ? &colls[encode-1] : nullptr);
	}

#define FUNC_PERFECT_MATCH 6  // The score for a perfect match
	__device__ static int MatchQuality(FuncDef *p, int args, TEXTENCODE encode)
	{
		// nArg of -2 is a special case
		if (args == -2) return (!p->Func && !p->Step ? 0 : FUNC_PERFECT_MATCH);
		// Wrong number of arguments means "no match"
		if (p->Args != args && p->Args >= 0) return 0;
		// Give a better score to a function with a specific number of arguments than to function that accepts any number of arguments.
		int match = (p->Args == args ? 4 : 1);
		// Bonus points if the text encoding matches
		if (encode == p->PrefEncode)
			match += 2; // Exact encoding match
		else if ((encode & p->PrefEncode & 2) != 0)
			match += 1; // Both are UTF16, but with different byte orders
		return match;
	}

	__device__ static FuncDef *FunctionSearch(FuncDefHash *hash, int h, const char *func, int funcLength)
	{
		for (FuncDef *p = hash->data[h]; p; p = p->Hash)
			if (!_strncmp(p->Name, func, funcLength) && p->Name[funcLength] == 0)
				return p;
		return nullptr;
	}

	__device__ void Callback::FuncDefInsert(FuncDefHash *hash, FuncDef *def)
	{
		int nameLength = _strlen30(def->Name);
		int h = (_tolower(def->Name[0]) + nameLength) % _lengthof(hash->data); // Hash value
		FuncDef *other = FunctionSearch(hash, h, def->Name, nameLength);
		if (other)
		{
			_assert(other != def && other->Next != def);
			def->Next = other->Next;
			other->Next = def;
		}
		else
		{
			def->Next = nullptr;
			def->Hash = hash->data[h];
			hash->data[h] = def;
		}
	}

	__device__ FuncDef *Callback::FindFunction(Context *ctx, const char *name, int nameLength, int args, TEXTENCODE encode, bool createFlag)
	{
		_assert(args >= -2);
		_assert(args >= -1 || !createFlag);
		_assert(encode == TEXTENCODE_UTF8 || encode == TEXTENCODE_UTF16LE || encode == TEXTENCODE_UTF16BE);
		int h = (_tolower(name[0]) + nameLength) % _lengthof(ctx->Funcs.data); // Hash value

		// First search for a match amongst the application-defined functions.
		FuncDef *best = nullptr; // Best match found so far
		int bestScore = 0; // Score of best match
		FuncDef *p = FunctionSearch(&ctx->Funcs, h, name, nameLength);
		while (p)
		{
			int score = MatchQuality(p, args, encode);
			if (score > bestScore)
			{
				best = p;
				bestScore = score;
			}
			p = p->Next;
		}

		// If no match is found, search the built-in functions.
		//
		// If the SQLITE_PreferBuiltin flag is set, then search the built-in functions even if a prior app-defined function was found.  And give
		// priority to built-in functions.
		//
		// Except, if createFlag is true, that means that we are trying to install a new function.  Whatever FuncDef structure is returned it will
		// have fields overwritten with new information appropriate for the new function.  But the FuncDefs for built-in functions are read-only.
		// So we must not search for built-ins when creating a new function.
		if (!createFlag && (!best || (ctx->Flags & Context::FLAG_PreferBuiltin) != 0))
		{
			FuncDefHash *hash = &Main_GlobalFunctions;
			bestScore = 0;
			p = FunctionSearch(hash, h, name, nameLength);
			while (p)
			{
				int score = MatchQuality(p, args, encode);
				if (score > bestScore)
				{
					best = p;
					bestScore = score;
				}
				p = p->Next;
			}
		}

		// If the createFlag parameter is true and the search did not reveal an exact match for the name, number of arguments and encoding, then add a
		// new entry to the hash table and return it.
		if (createFlag && bestScore < FUNC_PERFECT_MATCH && (best = (FuncDef *)_tagalloc2(ctx, sizeof(*best)+nameLength+1, true)) != nullptr)
		{
			best->Name = (char *)&best[1];
			best->Args = (uint16)args;
			best->PrefEncode = encode;
			_memcpy(best->Name, name, nameLength);
			best->Name[nameLength] = 0;
			FuncDefInsert(&ctx->Funcs, best);
		}

		return (best && (best->Step || best->Func || createFlag) ? best : nullptr);
	}

	__device__ void Callback::SchemaClear(void *p)
	{
		Schema *schema = (Schema *)p;
		Hash temp1 = schema->TableHash;
		Hash temp2 = schema->TriggerHash;
		new (&schema->TriggerHash) Hash();
		schema->IndexHash.Clear();
		HashElem *elem;
		for (elem = temp2.First; elem; elem = elem->Next)
			Trigger::DeleteTrigger(nullptr, (Trigger *)elem->Data);
		temp2.Clear();
		new (&schema->TableHash) Hash();
		for (elem = temp1.First; elem; elem = elem->Next)
			Parse::DeleteTable(nullptr, (Table *)elem->Data);
		temp1.Clear();
		schema->FKeyHash.Clear();
		schema->SeqTable = nullptr;
		if (schema->Flags & SCHEMA_SchemaLoaded)
		{
			schema->Generation++;
			schema->Flags &= ~SCHEMA_SchemaLoaded;
		}
	}

	__device__ Schema *Callback::SchemaGet(Context *ctx, Btree *bt)
	{
		Schema *p = (bt ? bt->Schema(sizeof(Schema), SchemaClear) : (Schema *)_tagalloc2(nullptr, sizeof(Schema), true));
		if (!p)
			ctx->MallocFailed = true;
		else if (!p->FileFormat)
		{
			p->TableHash.Init();
			p->IndexHash.Init();
			p->TriggerHash.Init();
			p->FKeyHash.Init();
			p->Encode = TEXTENCODE_UTF8;
		}
		return p;
	}
}