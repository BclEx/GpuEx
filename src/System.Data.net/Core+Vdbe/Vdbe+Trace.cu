// vdbetrace.c
#include "VdbeInt.cu.h"

namespace Core {

#pragma region Trace
#ifndef OMIT_TRACE

	__device__ static int FindNextHostParameter(const char *sql, int *tokens)
	{
		int total = 0;
		*tokens = 0;
		while (sql[0])
		{
			int tokenType;
			int n = Parse::GetToken((uint8 *)sql, &tokenType);
			_assert(n > 0 && tokenType != TK_ILLEGAL);
			if (tokenType == TK_VARIABLE)
			{
				*tokens = n;
				break;
			}
			total += n;
			sql += n;
		}
		return total;
	}

	__device__ char *Vdbe::ExpandSql(const char *rawSql)
	{
		Context *ctx = Ctx; // The database connection
		char bBase[100]; // Initial working space
		TextBuilder b; // Accumulate the output here
		TextBuilder::Init(&b, bBase, sizeof(bBase), ctx->Limits[LIMIT_LENGTH]);
		b.Tag = ctx;
		int nextIndex = 1; // Index of next ? host parameter
		int idx = 0; // Index of a host parameter
		if (ctx->VdbeExecCnt > 1)
			while (*rawSql)
			{
				const char *start = rawSql;
				while (*(rawSql++) != '\n' && *rawSql);
				b.Append("-- ", 3);
				b.Append(start, (int)(rawSql - start));
			}
		else
			while (rawSql[0])
			{
				int tokenLength; // Length of the parameter token
				int n = FindNextHostParameter(rawSql, &tokenLength); // Length of a token prefix
				_assert(n > 0);
				b.Append(rawSql, n);
				rawSql += n;
				_assert(rawSql[0] || !tokenLength);
				if (!tokenLength) break;
				if (rawSql[0] == '?')
				{
					if (tokenLength > 1)
					{
						_assert(_isdigit(rawSql[1]));
						ConvertEx::Atoi(&rawSql[1], &idx);
					}
					else
						idx = nextIndex;
				}
				else
				{
					_assert(rawSql[0] == ':' || rawSql[0] == '$' || rawSql[0] == '@');
					ASSERTCOVERAGE(rawSql[0] == ':');
					ASSERTCOVERAGE(rawSql[0] == '$');
					ASSERTCOVERAGE(rawSql[0] == '@');
					idx = ParameterIndex(this, rawSql, tokenLength);
					_assert(idx > 0);
				}
				rawSql += tokenLength;
				nextIndex = idx + 1;
				_assert(idx > 0 && idx <= Vars.length);
				Mem *var = &Vars[idx - 1]; // Value of a host parameter
				if (var->Flags & MEM_Null) b.Append("NULL", 4);
				else if (var->Flags & MEM_Int) b.AppendFormat("%lld", var->u.I);
				else if (var->Flags & MEM_Real) b.AppendFormat("%!.15g", var->R);
				else if (var->Flags & MEM_Str)
				{
#ifndef OMIT_UTF16
					TEXTENCODE encode = CTXENCODE(ctx);
					if (encode != TEXTENCODE_UTF8)
					{
						Mem utf8;
						_memset(&utf8, 0, sizeof(utf8));
						utf8.Ctx = ctx;
						MemSetStr(&utf8, var->Z, var->N, encode, DESTRUCTOR_STATIC);
						ChangeEncoding(&utf8, TEXTENCODE_UTF8);
						b.AppendFormat("'%.*q'", utf8.N, utf8.Z);
						MemRelease(&utf8);
					}
					else
#endif
						b.AppendFormat("'%.*q'", var->N, var->Z);
				}
				else if (var->Flags & MEM_Zero) b.AppendFormat("zeroblob(%d)", var->u.Zeros);
				else
				{
					_assert(var->Flags & MEM_Blob);
					b.Append("x'", 2);
					for (int i = 0; i < var->N; i++)
						b.AppendFormat("%02x", var->Z[i] & 0xff);
					b.Append("'", 1);
				}
			}
			return b.ToString();
	}

#endif
#pragma endregion

#pragma region Explain
#if defined(ENABLE_TREE_EXPLAIN)

	__device__ void Vdbe::ExplainBegin(Vdbe *vdbe)
	{
		if (vdbe)
		{
			_benignalloc_begin();
			Explain *p = (Explain *)_alloc2(sizeof(Explain), true);
			if (p)
			{
				p->Vdbe_ = vdbe;
				_free(vdbe->_explain);
				vdbe->_explain = p;
				TextBuilder::Init(&p->Str, p->ZBase, sizeof(p->ZBase), CORE_MAX_LENGTH);
				p->Str.AllocType = 2;
			}
			else
				_benignalloc_end();
		}
	}

	__device__ inline static int EndsWithNL(Explain *p)
	{
		return (p && p->Str.Text && p->Str.Size && p->Str.Text[p->Str.Size-1]=='\n');
	}

	__device__ void Vdbe::ExplainPrintf(Vdbe *vdbe, const char *format, va_list *args)
	{
		Explain *p;
		if (vdbe && (p = vdbe->_explain) != nullptr)
		{
			if (p->Indents && EndsWithNL(p))
			{
				int n = p->IndentLength;
				if (n > _lengthof(p->Indents)) n = _lengthof(p->Indents);
				p->Str.AppendSpace(p->Indents[n-1]);
			}   
			p->Str.AppendFormat(true, format, args);
		}
	}

	__device__ void Vdbe::ExplainNL(Vdbe *vdbe)
	{
		Explain *p;
		if (vdbe && (p = vdbe->_explain) != nullptr && !EndsWithNL(p))
			p->Str.Append("\n", 1);
	}

	__device__ void Vdbe::ExplainPush(Vdbe *vdbe)
	{
		Explain *p;
		if (vdbe && (p = vdbe->_explain) != nullptr)
		{
			if (p->Str.Text && p->IndentLength < _lengthof(p->Indents))
			{
				const char *z = p->Str.Text;
				int i = p->Str.Size-1;
				int x;
				while (i >= 0 && z[i] != '\n' ) { i--; }
				x = (p->Str.Size - 1) - i;
				if( p->IndentLength && x < p->Indents[p->IndentLength-1])
					x = p->Indents[p->IndentLength-1];
				p->Indents[p->IndentLength] = x;
			}
			p->IndentLength++;
		}
	}

	__device__ void Vdbe::ExplainPop(Vdbe *p)
	{
		if (p && p->_explain) p->_explain->IndentLength--;
	}

	__device__ void Vdbe::ExplainFinish(Vdbe *p)
	{
		if (p && p->_explain)
		{
			_free(p->_explainString);
			ExplainNL(p);
			p->_explainString = p->_explain->Str.ToString();
			_free(p->_explain); p->_explain = nullptr;
			_benignalloc_end();
		}
	}

	__device__ const char *Vdbe::Explanation(Vdbe *p)
	{
		return (p && p->_explainString ? p->_explainString : nullptr);
	}

#endif
#pragma endregion
}
