#include "Core+Vdbe.cu.h"

namespace Core
{
	__constant__ const Token g_intTokens[] = {
		{ "0", 1 },
		{ "1", 1 }
	};

	__device__ AFF Expr::Affinity()
	{
		Expr *expr = SkipCollate();
		uint8 op = OP;
		if (op == TK_SELECT)
		{
			_assert(expr->Flags&EP_xIsSelect);
			return expr->x.Select->EList->Ids[0].Expr->Affinity();
		}
#ifndef OMIT_CAST
		if (op == TK_CAST)
		{
			_assert(!ExprHasProperty(expr, EP_IntValue));
			return Parse::AffinityType(expr->u.Token);
		}
#endif
		if ((op == TK_AGG_COLUMN || op == TK_COLUMN || op == TK_REGISTER) && expr->Table)
		{
			// op == TK_REGISTER && expr->Table happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
			int j = expr->ColumnIdx;
			if (j < 0)
				return AFF_INTEGER;
			_assert(expr->Table && j < expr->Table->Cols.length);
			return expr->Table->Cols[j].Affinity;
		}
		return expr->Aff;
	}

	__device__ Expr *Expr::AddCollateToken(Parse *parse, Expr *expr, Token *collName)
	{
		if (collName->length > 0)
		{
			Expr *newExpr = Expr::Alloc(parse->Ctx, TK_COLLATE, collName, 1);
			if (newExpr)
			{
				newExpr->Left = expr;
				newExpr->Flags |= EP_Collate;
				expr = newExpr;
			}
		}
		return expr;
	}

	__device__ Expr *Expr::AddCollateString(Parse *parse, Expr *expr, const char *z)
	{
		_assert(z);
		Token s;
		s.data = z;
		s.length = _strlen30(z);
		return AddCollateToken(parse, expr, &s);
	}

	__device__ Expr *Expr::SkipCollate()
	{
		Expr *expr = this;
		while (expr && (expr->OP == TK_COLLATE || expr->OP == TK_AS))
			expr = expr->Left;
		return expr;
	}

	__device__ CollSeq *Expr::CollSeq(Parse *parse)
	{
		Context *ctx = parse->Ctx;
		Core::CollSeq *coll = nullptr;
		Expr *p = this;
		while (p)
		{
			int op = p->OP;
			if (op == TK_CAST || op == TK_UPLUS)
			{
				p = p->Left;
				continue;
			}
			_assert(op != TK_REGISTER || p->OP2 != TK_COLLATE);
			if (op == TK_COLLATE)
			{
				if (ctx->Init.Busy) // Do not report errors when parsing while the schema 
					coll = Callback::FindCollSeq(ctx, CTXENCODE(ctx), p->u.Token, false);
				else
					coll = Callback::GetCollSeq(parse, CTXENCODE(ctx), nullptr, p->u.Token);
				break;
			}
			if (p->Table && (op == TK_AGG_COLUMN || op == TK_COLUMN || op == TK_REGISTER || op == TK_TRIGGER))
			{
				// op==TK_REGISTER && p->pTab!=0 happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
				int j = p->ColumnIdx;
				if (j >= 0)
				{
					const char *nameColl = p->Table->Cols[j].Coll;
					coll = Callback::FindCollSeq(ctx, CTXENCODE(ctx), nameColl, false);
				}
				break;
			}
			if (p->Flags & EP_Collate)
				p = (_ALWAYS(p->Left) && (p->Left->Flags & EP_Collate) != 0 ? p->Left : p->Right);
			else
				break;
		}
		if (Callback::CheckCollSeq(parse, coll))
			coll = nullptr;
		return coll;
	}

	__device__ AFF Expr::CompareAffinity(AFF aff2)
	{
		AFF aff1 = Affinity();
		if (aff1 && aff2) // Both sides of the comparison are columns. If one has numeric affinity, use that. Otherwise use no affinity.
			return (IsNumericAffinity(aff1) || IsNumericAffinity(aff2) ? AFF_NUMERIC : AFF_NONE);
		else if (!aff1 && !aff2) // Neither side of the comparison is a column.  Compare the results directly.
			return AFF_NONE;
		else // One side is a column, the other is not. Use the columns affinity.
		{
			_assert(aff1 == 0 || aff2 == 0);
			return (AFF)(aff1 + aff2);
		}
	}

	__device__ static AFF ComparisonAffinity(Expr *expr)
	{
		_assert(expr->OP == TK_EQ || expr->OP == TK_IN || expr->OP == TK_LT ||
			expr->OP == TK_GT || expr->OP == TK_GE || expr->OP == TK_LE ||
			expr->OP == TK_NE || expr->OP == TK_IS || expr->OP == TK_ISNOT);
		_assert(expr->Left);
		AFF aff = expr->Left->Affinity();
		if (expr->Right)
			aff = expr->Right->CompareAffinity(aff);
		else if (ExprHasProperty(expr, EP_xIsSelect))
			aff = expr->x.Select->EList->Ids[0].Expr->CompareAffinity(aff);
		else if (!aff)
			aff = AFF_NONE;
		return aff;
	}

	__device__ bool Expr::ValidIndexAffinity(AFF indexAff)
	{
		AFF aff = ComparisonAffinity(this);
		switch (aff)
		{
		case AFF_NONE:
			return true;
		case AFF_TEXT:
			return (indexAff == AFF_TEXT);
		default:
			return IsNumericAffinity(indexAff);
		}
	}

	__device__ static AFF BinaryCompareP5(Expr *expr1, Expr *expr2, AFF jumpIfNull)
	{
		AFF aff = expr2->Affinity();
		aff = (AFF)(expr1->CompareAffinity(aff) | jumpIfNull);
		return aff;
	}

	__device__ CollSeq *Expr::BinaryCompareCollSeq(Parse *parse, Expr *left, Expr *right)
	{
		_assert(left);
		Core::CollSeq *coll;
		if (left->Flags & EP_Collate)
			coll = left->CollSeq(parse);
		else if (right && (right->Flags & EP_Collate))
			coll = right->CollSeq(parse);
		else{
			coll = left->CollSeq(parse);
			if (!coll)
				coll = right->CollSeq(parse);
		}
		return coll;
	}

	__device__ static int CodeCompare(Parse *parse, Expr *left, Expr *right, int opcode, int in1, int in2, int dest, AFF jumpIfNull)
	{
		CollSeq *p4 = Expr::BinaryCompareCollSeq(parse, left, right);
		AFF p5 = BinaryCompareP5(left, right, jumpIfNull);
		Vdbe *v = parse->V;
		int addr = v->AddOp4(opcode, in2, dest, in1, (const char *)p4, Vdbe::P4T_COLLSEQ);
		v->ChangeP5((uint8)p5);
		return addr;
	}

#if MAX_EXPR_DEPTH > 0
	__device__ RC Expr::CheckHeight(Parse *parse, int height)
	{
		RC rc = RC_OK;
		int maxHeight = parse->Ctx->Limits[LIMIT_EXPR_DEPTH];
		if (height > maxHeight)
		{
			parse->ErrorMsg("Expression tree is too large (maximum depth %d)", maxHeight);
			rc = RC_ERROR;
		}
		return rc;
	}

	__device__ static void HeightOfExpr(Expr *p, int *height)
	{
		if (p)
			if (p->Height > *height)
				*height = p->Height;
	}

	__device__ static void HeightOfExprList(ExprList *p, int *height)
	{
		if (p)
			for (int i = 0; i < p->Exprs; i++)
				HeightOfExpr(p->Ids[i].Expr, height);
	}

	__device__ static void HeightOfSelect(Select *p, int *height)
	{
		if (p)
		{
			HeightOfExpr(p->Where, height);
			HeightOfExpr(p->Having, height);
			HeightOfExpr(p->Limit, height);
			HeightOfExpr(p->Offset, height);
			HeightOfExprList(p->EList, height);
			HeightOfExprList(p->GroupBy, height);
			HeightOfExprList(p->OrderBy, height);
			HeightOfSelect(p->Prior, height);
		}
	}

	__device__ static void ExprSetHeight(Expr *p)
	{
		int height = 0;
		HeightOfExpr(p->Left, &height);
		HeightOfExpr(p->Right, &height);
		if (ExprHasProperty(p, EP_xIsSelect))
			HeightOfSelect(p->x.Select, &height);
		else
			HeightOfExprList(p->x.List, &height);
		p->Height = height + 1;
	}

	__device__ void Expr::SetHeight(Parse *parse)
	{
		ExprSetHeight(this);
		CheckHeight(parse, this->Height);
	}

	__device__ int Expr::SelectExprHeight(Select *select)
	{
		int height = 0;
		HeightOfSelect(select, &height);
		return height;
	}
#else
	//#define ExprSetHeight(y)
#endif

	__device__ Expr *Expr::Alloc(Context *ctx, int op, const Token *token, bool dequote)
	{
		uint32 extraSize = 0;
		int value = 0;
		if (token)
		{
			if (op != TK_INTEGER || !token->data || !ConvertEx::Atoi(token->data, &value))
			{
				extraSize = token->length + 1;
				_assert(value >= 0);
			}
		}
		Expr *newExpr = (Expr *)_tagalloc2(ctx, sizeof(Expr)+extraSize, true);
		if (newExpr)
		{
			newExpr->OP = (uint8)op;
			newExpr->Agg = -1;
			if (token)
			{
				if (extraSize == 0)
				{
					newExpr->Flags |= EP_IntValue;
					newExpr->u.I = value;
				}
				else
				{
					int c;
					newExpr->u.Token = (char *)&newExpr[1];
					_assert(token->data || token->length == 0);
					if (token->length)
						_memcpy(newExpr->u.Token, token->data, token->length);
					newExpr->u.Token[token->length] = 0;
					if (dequote && extraSize >= 3 && ((c = token->data[0]) == '\'' || c == '"' || c == '[' || c == '`'))
					{
						Parse::Dequote(newExpr->u.Token);
						if (c == '"')
							newExpr->Flags |= EP_DblQuoted;
					}
				}
			}
#if MAX_EXPR_DEPTH > 0
			newExpr->Height = 1;
#endif  
		}
		return newExpr;
	}

	__device__ Expr *Expr::Expr_(Context *ctx, int op, const char *token)
	{
		Token x;
		x.data = (char *)token;
		x.length = (token ? _strlen30(token) : 0);
		return Alloc(ctx, op, &x, false);
	}

	__device__ void Expr::AttachSubtrees(Context *ctx, Expr *root, Expr *left, Expr *right)
	{
		if (!root)
		{
			_assert(ctx->MallocFailed);
			Delete(ctx, left);
			Delete(ctx, right);
		}
		else
		{
			if (right)
			{
				root->Right = right;
				root->Flags |= EP_Collate & right->Flags;
			}
			if (left)
			{
				root->Left = left;
				root->Flags |= EP_Collate & left->Flags;
			}
			ExprSetHeight(root);
		}
	}

	__device__ Expr *Expr::PExpr_(Parse *parse, int op, Expr *left, Expr *right, const Token *token)
	{
		Context *ctx = parse->Ctx;
		Expr *p;
		if (op == TK_AND && left && right) // Take advantage of short-circuit false optimization for AND
			p = And(ctx, left, right);
		else
		{
			p = Alloc(ctx, op, token, true);
			AttachSubtrees(ctx, p, left, right);
		}
		if (p)
			CheckHeight(parse, p->Height);
		return p;
	}

	__device__ static bool ExprAlwaysFalse(Expr *p)
	{
		if (ExprHasProperty(p, EP_FromJoin)) return false;
		int v = 0;
		if (!p->IsInteger(&v)) return false;
		return !v;
	}

	__device__ Expr *Expr::And(Context *ctx, Expr *left, Expr *right)
	{
		if (!left)
			return right;
		else if (!right)
			return left;
		else if (ExprAlwaysFalse(left) || ExprAlwaysFalse(right))
		{
			Delete(ctx, left);
			Delete(ctx, right);
			return Alloc(ctx, TK_INTEGER, &g_intTokens[0], false);
		}
		else
		{
			Expr *newExpr = Alloc(ctx, TK_AND, nullptr, false);
			AttachSubtrees(ctx, newExpr, left, right);
			return newExpr;
		}
	}

	__device__ Expr *Expr::Function(Parse *parse, ExprList *list, Token *token)
	{
		_assert(token);
		Context *ctx = parse->Ctx;
		Expr *newExpr = Alloc(ctx, TK_FUNCTION, token, true);
		if (!newExpr)
		{
			ListDelete(ctx, list); // Avoid memory leak when malloc fails
			return nullptr;
		}
		newExpr->x.List = list;
		_assert(!ExprHasProperty(newExpr, EP_xIsSelect));
		ExprSetHeight(newExpr);
		return newExpr;
	}

	__device__ void Expr::AssignVarNumber(Parse *parse, Expr *expr)
	{
		if (!expr)
			return;
		_assert(!ExprHasAnyProperty(expr, EP_IntValue|EP_Reduced|EP_TokenOnly));
		const char *z = expr->u.Token;
		_assert(z && z[0] != 0);
		Context *ctx = parse->Ctx;
		if (z[1] == 0) 
		{
			_assert(z[0] == '?');
			// Wildcard of the form "?".  Assign the next variable number
			expr->ColumnIdx = (yVars)(++parse->VarsSeen);
		}
		else
		{
			yVars x = 0;
			uint32 length = _strlen30(z);
			if (z[0] == '?') 
			{
				// Wildcard of the form "?nnn".  Convert "nnn" to an integer and use it as the variable number
				int64 i;
				bool ok = !ConvertEx::Atoi64(&z[1], &i, length-1, TEXTENCODE_UTF8);
				expr->ColumnIdx = x = (yVars)i;
				ASSERTCOVERAGE(i == 0);
				ASSERTCOVERAGE(i == 1);
				ASSERTCOVERAGE(i == ctx->Limits[LIMIT_VARIABLE_NUMBER]-1);
				ASSERTCOVERAGE(i == ctx->Limits[LIMIT_VARIABLE_NUMBER]);
				if (!ok || i < 1 || i > ctx->Limits[LIMIT_VARIABLE_NUMBER])
				{
					parse->ErrorMsg("variable number must be between ?1 and ?%d", ctx->Limits[LIMIT_VARIABLE_NUMBER]);
					x = 0;
				}
				if (i > parse->VarsSeen)
					parse->VarsSeen = (int)i;
			}
			else
			{
				// Wildcards like ":aaa", "$aaa" or "@aaa".  Reuse the same variable number as the prior appearance of the same name, or if the name
				// has never appeared before, reuse the same variable number
				yVars i;
				for (i = 0; i < parse->Vars.length; i++)
				{
					if (parse->Vars[i] && !_strcmp(parse->Vars[i], z))
					{
						expr->ColumnIdx = x = (yVars)i + 1;
						break;
					}
				}
				if (x == 0)
					expr->ColumnIdx = x = (yVars)(++parse->VarsSeen);
			}
			if (x > 0)
			{
				if (x > parse->Vars.length)
				{
					char **a = (char **)_tagrealloc(ctx, parse->Vars.data, x * sizeof(a[0]));
					if (!a) return;  // Error reported through db->mallocFailed
					parse->Vars.data = a;
					_memset(&a[parse->Vars.length], 0, (x - parse->Vars.length)*sizeof(a[0]));
					parse->Vars.length = x;
				}
				if (z[0] != '?' || parse->Vars[x-1] == nullptr)
				{
					_tagfree(ctx, parse->Vars[x-1]);
					parse->Vars[x-1] = _tagstrndup(ctx, z, length);
				}
			}
		} 
		if (!parse->Errs && parse->VarsSeen > ctx->Limits[LIMIT_VARIABLE_NUMBER])
			parse->ErrorMsg("too many SQL variables");
	}

	__device__ void Expr::Delete(Context *ctx, Expr *expr)
	{
		if (!expr) return;
		// Sanity check: Assert that the IntValue is non-negative if it exists
		_assert(!ExprHasProperty(expr, EP_IntValue) || expr->u.I >= 0);
		if (!ExprHasAnyProperty(expr, EP_TokenOnly))
		{
			Delete(ctx, expr->Left);
			Delete(ctx, expr->Right);
			if (!ExprHasProperty(expr, EP_Reduced) && (expr->Flags2 & EP2_MallocedToken))
				_tagfree(ctx, expr->u.Token);
			if (ExprHasProperty(expr, EP_xIsSelect))
				Select::Delete(ctx, expr->x.Select);
			else
				ListDelete(ctx, expr->x.List);
		}
		if (!ExprHasProperty(expr, EP_Static))
			_tagfree(ctx, expr);
	}

#pragma region Clone

	__device__ static int ExprStructSize(Expr *expr)
	{
		if (ExprHasProperty(expr, EP_TokenOnly)) return EXPR_TOKENONLYSIZE;
		if (ExprHasProperty(expr, EP_Reduced)) return EXPR_REDUCEDSIZE;
		return EXPR_FULLSIZE;
	}

	__device__ static int DupedExprStructSize(Expr *expr, int flags)
	{
		_assert(flags == EXPRDUP_REDUCE || flags == 0); // Only one flag value allowed
		int size;
		if (!(flags & EXPRDUP_REDUCE))
			size = EXPR_FULLSIZE;
		else
		{
			_assert(!ExprHasAnyProperty(expr, EP_TokenOnly|EP_Reduced));
			_assert(!ExprHasProperty(expr, EP_FromJoin)); 
			_assert(!(expr->Flags2 & EP2_MallocedToken));
			_assert(!(expr->Flags2 & EP2_Irreducible));
			size = (expr->Left || expr->Right || expr->x.List ? EXPR_REDUCEDSIZE | EP_Reduced : EXPR_TOKENONLYSIZE | EP_TokenOnly);
		}
		return size;
	}

	__device__ static int DupedExprNodeSize(Expr *expr, int flags)
	{
		int bytes = DupedExprStructSize(expr, flags) & 0xfff;
		if (!ExprHasProperty(expr, EP_IntValue) && expr->u.Token)
			bytes += _strlen30(expr->u.Token) + 1;
		return _ROUND8(bytes);
	}

	__device__ static int DupedExprSize(Expr *expr, int flags)
	{
		int bytes = 0;
		if (expr)
		{
			bytes = DupedExprNodeSize(expr, flags);
			if (flags & EXPRDUP_REDUCE)
				bytes += DupedExprSize(expr->Left, flags) + DupedExprSize(expr->Right, flags);
		}
		return bytes;
	}

	__device__ static Expr *ExprDup(Context *ctx, Expr *expr, int flags, uint8 **buffer)
	{
		Expr *newExpr = nullptr; // Value to return
		if (expr)
		{
			const bool isReduced = (flags & EXPRDUP_REDUCE);
			_assert(!buffer || isReduced);

			// Figure out where to write the new Expr structure.
			uint8 *alloc;
			EP staticFlag;
			if (buffer)
			{
				alloc = *buffer;
				staticFlag = EP_Static;
			}
			else
			{
				alloc = (uint8 *)_tagalloc(ctx, DupedExprSize(expr, flags));
				staticFlag = (EP)0;
			}
			newExpr = (Expr *)alloc;
			if (newExpr)
			{
				// Set nNewSize to the size allocated for the structure pointed to by pNew. This is either EXPR_FULLSIZE, EXPR_REDUCEDSIZE or
				// EXPR_TOKENONLYSIZE. nToken is set to the number of bytes consumed * by the copy of the p->u.zToken string (if any).
				const unsigned structSize = DupedExprStructSize(expr, flags);
				const int newSize = structSize & 0xfff;
				int tokenLength = (!ExprHasProperty(expr, EP_IntValue) && expr->u.Token ? _strlen30(expr->u.Token) + 1 : 0);
				if (isReduced)
				{
					_assert(!ExprHasProperty(expr, EP_Reduced));
					_memcpy(alloc, (uint8 *)expr, newSize);
				}
				else
				{
					int size = ExprStructSize(expr);
					_memcpy(alloc, (uint8 *)expr, size);
					_memset(&alloc[size], 0, EXPR_FULLSIZE-size);
				}

				// Set the EP_Reduced, EP_TokenOnly, and EP_Static flags appropriately.
				newExpr->Flags &= ~(EP_Reduced|EP_TokenOnly|EP_Static);
				newExpr->Flags |= (EP)(structSize & (EP_Reduced|EP_TokenOnly));
				newExpr->Flags |= staticFlag;
				// Copy the p->u.zToken string, if any.
				if (tokenLength)
				{
					char *token = newExpr->u.Token = (char *)&alloc[newSize];
					_memcpy(token, expr->u.Token, tokenLength);
				}
				if (!((expr->Flags | newExpr->Flags) & EP_TokenOnly))
				{
					// Fill in the pNew->x.pSelect or pNew->x.pList member.
					if (ExprHasProperty(expr, EP_xIsSelect))
						newExpr->x.Select = Expr::SelectDup(ctx, expr->x.Select, isReduced);
					else
						newExpr->x.List = Expr::ListDup(ctx, expr->x.List, isReduced);
				}
				// Fill in pNew->pLeft and pNew->pRight.
				if (ExprHasAnyProperty(newExpr, EP_Reduced|EP_TokenOnly))
				{
					alloc += DupedExprNodeSize(expr, flags);
					if (ExprHasProperty(newExpr, EP_Reduced))
					{
						newExpr->Left = ExprDup(ctx, expr->Left, EXPRDUP_REDUCE, &alloc);
						newExpr->Right = ExprDup(ctx, expr->Right, EXPRDUP_REDUCE, &alloc);
					}
					if (buffer)
						*buffer = alloc;
				}
				else
				{
					newExpr->Flags2 = (EP2)0;
					if (!ExprHasAnyProperty(expr, EP_TokenOnly))
					{
						newExpr->Left = ExprDup(ctx, expr->Left, 0, nullptr);
						newExpr->Right = ExprDup(ctx, expr->Right, 0, nullptr);
					}
				}
			}
		}
		return newExpr;
	}

	__device__ Expr *Expr::Dup(Context *ctx, Expr *expr, int flags) { return ExprDup(ctx, expr, flags, nullptr); }
	__device__ ExprList *Expr::ListDup(Context *ctx, ExprList *list, int flags)
	{
		if (!list) return nullptr;
		ExprList *newList = (ExprList *)_tagalloc(ctx, sizeof(*newList));
		if (!newList) return nullptr;
		int i;
		newList->ECursor = 0;
		newList->Exprs = i = list->Exprs;
		if ((flags & EXPRDUP_REDUCE) == 0)
		{
			for (i = 1; i < list->Exprs; i += i) { }
		}
		ExprList::ExprListItem *item;
		newList->Ids = item = (ExprList::ExprListItem *)_tagalloc(ctx, i * sizeof(list->Ids[0]));
		if (!item)
		{
			_tagfree(ctx, newList);
			return nullptr;
		} 
		ExprList::ExprListItem *oldItem = list->Ids;
		for (i = 0; i < list->Exprs; i++, item++, oldItem++)
		{
			Expr *oldExpr = oldItem->Expr;
			item->Expr = Dup(ctx, oldExpr, flags);
			item->Name = _tagstrdup(ctx, oldItem->Name);
			item->Span = _tagstrdup(ctx, oldItem->Span);
			item->SortOrder = oldItem->SortOrder;
			item->Done = false;
			item->OrderByCol = oldItem->OrderByCol;
			item->Alias = oldItem->Alias;
		}
		return newList;
	}

#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER) || !defined(OMIT_SUBQUERY)
	__device__ SrcList *Expr::SrcListDup(Context *ctx, SrcList *list, int flags)
	{
		if (!list)
			return 0;
		int bytes = sizeof(*list) + (list->Srcs > 0 ? sizeof(list->Ids[0]) * (list->Srcs-1) : 0);
		SrcList *newList = (SrcList *)_tagalloc(ctx, bytes);
		if (!newList)
			return nullptr;
		newList->Srcs = newList->Allocs = list->Srcs;
		for (int i = 0; i < list->Srcs; i++)
		{
			SrcList::SrcListItem *newItem = &newList->Ids[i];
			SrcList::SrcListItem *oldItem = &list->Ids[i];
			newItem->Schema = oldItem->Schema;
			newItem->Database = _tagstrdup(ctx, oldItem->Database);
			newItem->Name = _tagstrdup(ctx, oldItem->Name);
			newItem->Alias = _tagstrdup(ctx, oldItem->Alias);
			newItem->Jointype = oldItem->Jointype;
			newItem->Cursor = oldItem->Cursor;
			newItem->AddrFillSub = oldItem->AddrFillSub;
			newItem->RegReturn = oldItem->RegReturn;
			newItem->IsCorrelated = oldItem->IsCorrelated;
			newItem->ViaCoroutine = oldItem->ViaCoroutine;
			newItem->IndexName = _tagstrdup(ctx, oldItem->IndexName);
			newItem->NotIndexed = oldItem->NotIndexed;
			newItem->Index = oldItem->Index;
			Core::Table *table = newItem->Table = oldItem->Table;
			if (table)
				table->Refs++;
			newItem->Select = SelectDup(ctx, oldItem->Select, flags);
			newItem->On = Dup(ctx, oldItem->On, flags);
			newItem->Using = IdListDup(ctx, oldItem->Using);
			newItem->ColUsed = oldItem->ColUsed;
		}
		return newList;
	}
	__device__ IdList *Expr::IdListDup(Context *ctx, IdList *list)
	{
		if (!list)
			return nullptr;
		IdList *newList = (IdList *)_tagalloc(ctx, sizeof(*newList));
		if (!newList)
			return nullptr;
		newList->Ids.length = list->Ids.length;
		newList->Ids.data = (IdList::IdListItem *)_tagalloc(ctx, list->Ids.length * sizeof(list->Ids[0]));
		if (!newList->Ids.data)
		{
			_tagfree(ctx, newList);
			return nullptr;
		}
		// Note that because the size of the allocation for p->a[] is not necessarily a power of two, sqlite3IdListAppend() may not be called
		// on the duplicate created by this function.
		for (int i = 0; i < list->Ids.length; i++)
		{
			IdList::IdListItem *newItem = &newList->Ids[i];
			IdList::IdListItem *oldItem = &list->Ids[i];
			newItem->Name = _tagstrdup(ctx, oldItem->Name);
			newItem->Idx = oldItem->Idx;
		}
		return newList;
	}
	__device__ Select *Expr::SelectDup(Context *ctx, Select *select, int flags)
	{
		if (!select)
			return nullptr;
		Select *newSelect = (Select *)_tagalloc(ctx, sizeof(*select));
		if (!newSelect)
			return nullptr;
		newSelect->EList = ListDup(ctx, select->EList, flags);
		newSelect->Src = SrcListDup(ctx, select->Src, flags);
		newSelect->Where = Dup(ctx, select->Where, flags);
		newSelect->GroupBy = ListDup(ctx, select->GroupBy, flags);
		newSelect->Having = Dup(ctx, select->Having, flags);
		newSelect->OrderBy = ListDup(ctx, select->OrderBy, flags);
		newSelect->OP = select->OP;
		Select *prior;
		newSelect->Prior = prior = SelectDup(ctx, select->Prior, flags);
		if (prior)
			prior->Next = newSelect;
		newSelect->Next = nullptr;
		newSelect->Limit = ExprDup(ctx, select->Limit, flags, nullptr);
		newSelect->Offset = ExprDup(ctx, select->Offset, flags, nullptr);
		newSelect->LimitId = 0;
		newSelect->OffsetId = 0;
		newSelect->SelFlags = (SF)(select->SelFlags & ~SF_UsesEphemeral);
		newSelect->Rightmost = nullptr;
		newSelect->AddrOpenEphms[0] = (::OP)-1;
		newSelect->AddrOpenEphms[1] = (::OP)-1;
		newSelect->AddrOpenEphms[2] = (::OP)-1;
		return newSelect;
	}
#else
	__device__ Select *Expr::SelectDup(Context *ctx, Select *select, int flags) { _assert(select); return nullptr; }
#endif

#pragma endregion

	__device__ ExprList *Expr::ListAppend(Parse *parse, ExprList *list, Expr *expr)
	{
		Context *ctx = parse->Ctx;
		if (!list)
		{
			list = (ExprList *)_tagalloc2(ctx, sizeof(ExprList), true);
			if (!list)
				goto no_mem;
			list->Ids = (ExprList::ExprListItem *)_tagalloc(ctx, sizeof(list->Ids[0]));
			if (!list->Ids)
				goto no_mem;
		}
		else if ((list->Exprs & (list->Exprs-1)) == 0)
		{
			_assert(list->Exprs > 0);
			ExprList::ExprListItem *ids = (ExprList::ExprListItem *)_tagrealloc(ctx, list->Ids, list->Exprs*2*sizeof(list->Ids[0]));
			if (!ids)
				goto no_mem;
			list->Ids = ids;
		}
		_assert(list->Ids);
		ExprList::ExprListItem *item = &list->Ids[list->Exprs++];
		_memset(item, 0, sizeof(*item));
		item->Expr = expr;
		return list;

no_mem:     
		// Avoid leaking memory if malloc has failed.
		Delete(ctx, expr);
		ListDelete(ctx, list);
		return nullptr;
	}

	__device__ void Expr::ListSetName(Parse *parse, ExprList *list, Token *name, bool dequote)
	{
		Context *ctx = parse->Ctx;
		_assert(list || ctx->MallocFailed);
		if (list)
		{
			_assert(list->Exprs > 0);
			ExprList::ExprListItem *item = &list->Ids[list->Exprs-1];
			_assert(!item->Name);
			item->Name = _tagstrndup(ctx, name->data, name->length);
			if (dequote && item->Name)
				Parse::Dequote(item->Name);
		}
	}

	__device__ void Expr::ListSetSpan(Parse *parse, ExprList *list, ExprSpan *span)
	{
		Context *ctx = parse->Ctx;
		_assert(list || ctx->MallocFailed);
		if (list)
		{
			ExprList::ExprListItem *item = &list->Ids[list->Exprs-1];
			_assert(list->Exprs > 0);
			_assert(ctx->MallocFailed || item->Expr == span->Expr);
			_tagfree(ctx, item->Span);
			item->Span = _tagstrndup(ctx, (char *)span->Start, (int)(span->End - span->Start));
		}
	}

	__device__ void Expr::ListCheckLength(Parse *parse, ExprList *list, const char *object)
	{
		int max = parse->Ctx->Limits[LIMIT_COLUMN];
		ASSERTCOVERAGE(list && list->Exprs == max);
		ASSERTCOVERAGE(list && list->Exprs == max+1);
		if (list && list->Exprs > max)
			parse->ErrorMsg("too many columns in %s", object);
	}

	__device__ void Expr::ListDelete(Context *ctx, ExprList *list)
	{
		if (!list)
			return;
		_assert(list->Ids || list->Exprs == 0);
		int i;
		ExprList::ExprListItem *item;
		for (item = list->Ids, i = 0; i < list->Exprs; i++, item++)
		{
			Delete(ctx, item->Expr);
			_tagfree(ctx, item->Name);
			_tagfree(ctx, item->Span);
		}
		_tagfree(ctx, list->Ids);
		_tagfree(ctx, list);
	}

#pragma region Walker - Expression Tree Walker

	__device__ static WRC ExprNodeIsConstant(Walker *walker, Expr *expr)
	{
		// If pWalker->u.i is 3 then any term of the expression that comes from the ON or USING clauses of a join disqualifies the expression
		// from being considered constant. */
		if (walker->u.I == 3 && ExprHasAnyProperty(expr, EP_FromJoin))
		{
			walker->u.I = 0;
			return WRC_Abort;
		}
		switch (expr->OP)
		{
		case TK_FUNCTION:
			// Consider functions to be constant if all their arguments are constant and pWalker->u.i==2
			if (walker->u.I == 2)
				return WRC_Continue;
			// Fall through
		case TK_ID:
		case TK_COLUMN:
		case TK_AGG_FUNCTION:
		case TK_AGG_COLUMN:
			ASSERTCOVERAGE(expr->OP == TK_ID);
			ASSERTCOVERAGE(expr->OP == TK_COLUMN);
			ASSERTCOVERAGE(expr->OP == TK_AGG_FUNCTION);
			ASSERTCOVERAGE(expr->OP == TK_AGG_COLUMN);
			walker->u.I = 0;
			return WRC_Abort;
		default:
			ASSERTCOVERAGE(expr->OP == TK_SELECT); // selectNodeIsConstant will disallow
			ASSERTCOVERAGE(expr->OP == TK_EXISTS); // selectNodeIsConstant will disallow
			return WRC_Continue;
		}
	}

	__device__ static WRC SelectNodeIsConstant(Walker *walker, Select *notUsed)
	{
		walker->u.I = 0;
		return WRC_Abort;
	}

	__device__ static bool ExprIsConst(Expr *expr, int initFlag)
	{
		Walker w;
		w.u.I = initFlag;
		w.ExprCallback = ExprNodeIsConstant;
		w.SelectCallback = SelectNodeIsConstant;
		w.WalkExpr(expr);
		return w.u.I != 0;
	}

	__device__ bool Expr::IsConstant() { return ExprIsConst(this, 1); }
	__device__ bool Expr::IsConstantNotJoin() { return ExprIsConst(this, 3); }
	__device__ bool Expr::IsConstantOrFunction() { return ExprIsConst(this, 2); }
	__device__ bool Expr::IsInteger(int *value)
	{
		// If an expression is an integer literal that fits in a signed 32-bit integer, then the EP_IntValue flag will have already been set
		int rc = 0;
		_assert(OP != TK_INTEGER || (Flags & EP_IntValue) != 0 || !ConvertEx::Atoi(u.Token, &rc));
		if (Flags & EP_IntValue)
		{
			*value = u.I;
			return true;
		}
		switch (OP)
		{
		case TK_UPLUS: return Left->IsInteger(value);
		case TK_UMINUS:
			int v;
			if (Left->IsInteger(&v))
			{
				*value = -v;
				return true;
			}
			return false;
		default: return false;
		}
	}

#pragma endregion

	__device__ bool Expr::CanBeNull()
	{
		Expr *expr = this;
		while (expr->OP == TK_UPLUS || expr->OP == TK_UMINUS) expr = expr->Left;
		uint8 op = expr->OP;
		if (op == TK_REGISTER)
			op = expr->OP2;
		switch (op)
		{
		case TK_INTEGER:
		case TK_STRING:
		case TK_FLOAT:
		case TK_BLOB: return false;
		default: return true;
		}
	}

	__device__ void Expr::CodeIsNullJump(Vdbe *v, const Expr *expr, int reg, int dest)
	{
		if (((Expr *)expr)->CanBeNull())
			v->AddOp2(OP_IsNull, reg, dest);
	}

	__device__ bool Expr::NeedsNoAffinityChange(AFF aff)
	{
		if (aff == AFF_NONE)
			return true;
		Expr *expr = this;
		while (expr->OP == TK_UPLUS || expr->OP == TK_UMINUS) expr = expr->Left;
		uint8 op = expr->OP;
		if (op == TK_REGISTER)
			op = expr->OP2;
		switch (op)
		{
		case TK_INTEGER: return aff == AFF_INTEGER || aff == AFF_NUMERIC;
		case TK_FLOAT: return aff == AFF_REAL || aff == AFF_NUMERIC;
		case TK_STRING: return aff == AFF_TEXT;
		case TK_BLOB: return true;
		case TK_COLUMN:
			_assert(expr->TableId >= 0); // p cannot be part of a CHECK constraint
			return expr->ColumnIdx < 0 && (aff == AFF_INTEGER || aff == AFF_NUMERIC);
		default: return false;
		}
	}

	__device__ bool Expr::IsRowid(const char *z)
	{
		return (!_strcmp(z, "_ROWID_") || !_strcmp(z, "ROWID") || !_strcmp(z, "OID"));
	}

#pragma region SUBQUERY

	__device__ int Expr::CodeOnce(Parse *parse)
	{
		Vdbe *v = parse->GetVdbe(); // Virtual machine being coded
		return v->AddOp1(OP_Once, parse->Onces++);
	}

#ifndef OMIT_SUBQUERY
	__device__ static bool IsCandidateForInOpt(Select *select)
	{
		if (select) return false;					// right-hand side of IN is SELECT
		if (select->Prior) return false;			// Not a compound SELECT
		if (select->SelFlags & (SF_Distinct|SF_Aggregate))
		{
			ASSERTCOVERAGE((select->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Distinct);
			ASSERTCOVERAGE((select->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Aggregate);
			return false; // No DISTINCT keyword and no aggregate functions
		}
		_assert(!select->GroupBy);					// Has no GROUP BY clause
		if (select->Limit) return false;			// Has no LIMIT clause
		_assert(select->Offset == false);           // No LIMIT means no OFFSET
		if (select->Where) return 0;				// Has no WHERE clause
		SrcList *src = select->Src;
		_assert(src);
		if (src->Srcs != 1) return false;			// Single term in FROM clause
		if (src->Ids[0].Select) return false;		// FROM is not a subquery or view
		Table *table = src->Ids[0].Table;
		if (_NEVER(table == nullptr)) return false;
		_assert(!table->Select);					// FROM clause is not a view
		if (IsVirtual(table)) return false;			// FROM clause not a virtual table
		ExprList *list = select->EList;
		if (list->Exprs != 1) return false;			// One column in the result set
		if (list->Ids[0].Expr->OP != TK_COLUMN) return false; // Result is a column
		return true;
	}

	__device__ IN_INDEX Expr::FindInIndex(Parse *parse, Expr *expr, int *notFound)
	{
		_assert(expr->OP == TK_IN);
		IN_INDEX type = (IN_INDEX)0; // Type of RHS table. IN_INDEX_*
		int tableIdx = parse->Tabs++; // Cursor of the RHS table
		bool mustBeUnique = (notFound == nullptr); // True if RHS must be unique
		Vdbe *v = parse->GetVdbe(); // Virtual machine being coded

		// Check to see if an existing table or index can be used to satisfy the query.  This is preferable to generating a new ephemeral table.
		Select *select = (ExprHasProperty(expr, EP_xIsSelect) ? expr->x.Select : nullptr); // SELECT to the right of IN operator
		if (_ALWAYS(parse->Errs == 0) && IsCandidateForInOpt(select))
		{
			_assert(select); // Because of isCandidateForInOpt(p)
			_assert(select->EList); // Because of isCandidateForInOpt(p)
			_assert(select->EList->Ids[0].Expr); // Because of isCandidateForInOpt(p)
			_assert(select->Src); // Because of isCandidateForInOpt(p)

			Context *ctx = parse->Ctx; // Database connection
			Core::Table *table = select->Src->Ids[0].Table; // Table <table>.
			Expr *expr2 = select->EList->Ids[0].Expr; // Expression <column>
			int col = expr2->ColumnIdx; // Index of column <column>

			// Code an OP_VerifyCookie and OP_TableLock for <table>.
			int db = Prepare::SchemaToIndex(ctx, table->Schema); // Database idx for pTab
			parse->CodeVerifySchema(db);
			parse->TableLock(db, table->Id, 0, table->Name);

			// This function is only called from two places. In both cases the vdbe has already been allocated. So assume sqlite3GetVdbe() is always successful here.
			_assert(v);
			if (col < 0)
			{
				int addr = CodeOnce(parse);
				Insert::OpenTable(parse, tableIdx, db, table, OP_OpenRead);
				type = IN_INDEX_ROWID;
				v->JumpHere(addr);
			}
			else
			{
				// The collation sequence used by the comparison. If an index is to be used in place of a temp-table, it must be ordered according
				// to this collation sequence.
				Core::CollSeq *req = BinaryCompareCollSeq(parse, expr->Left, expr2);
				// Check that the affinity that will be used to perform the comparison is the same as the affinity of the column. If
				// it is not, it is not possible to use any index.
				bool validAffinity = expr->ValidIndexAffinity(table->Cols[col].Affinity);
				for (Index *index = table->Index; index && type == 0 && validAffinity; index = index->Next)
					if (index->Columns[0] == col && Callback::FindCollSeq(ctx, CTXENCODE(ctx), index->CollNames[0], false) == req && (!mustBeUnique || (index->Columns.length == 1 && index->OnError != OE_None)))
					{
						KeyInfo *key = parse->IndexKeyinfo(index);
						int addr = CodeOnce(parse);
						v->AddOp4(OP_OpenRead, tableIdx, index->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
						v->Comment("%s", index->Name);
						_assert(IN_INDEX_INDEX_DESC == IN_INDEX_INDEX_ASC+1);
						type = (IN_INDEX)(IN_INDEX_INDEX_ASC + index->SortOrders[0]);
						v->JumpHere(addr);
						if (notFound && !table->Cols[col].NotNull)
						{
							*notFound = ++parse->Mems;
							v->AddOp2(OP_Null, 0, *notFound);
						}
					}
			}
		}

		if (type == 0)
		{
			// Could not found an existing table or index to use as the RHS b-tree. We will have to generate an ephemeral table to do the job.
			double savedQueryLoops = parse->QueryLoops;
			int mayHaveNull = 0;
			type = IN_INDEX_EPH;
			if (notFound)
			{
				*notFound = mayHaveNull = ++parse->Mems;
				v->AddOp2(OP_Null, 0, *notFound);
			}
			else
			{
				ASSERTCOVERAGE(parse->QueryLoops > (double)1);
				parse->QueryLoops = (double)1;
				if (expr->Left->ColumnIdx < 0 && !ExprHasAnyProperty(expr, EP_xIsSelect))
					type = IN_INDEX_ROWID;
			}
			CodeSubselect(parse, expr, mayHaveNull, type == IN_INDEX_ROWID);
			parse->QueryLoops = savedQueryLoops;
		}
		else
			expr->TableId = tableIdx;
		return type;
	}

	__device__ static SO _CodeSubselect_SortOrder = (SO)0; // Fake aSortOrder for keyInfo
	__device__ int Expr::CodeSubselect(Parse *parse, Expr *expr, int mayHaveNull, bool isRowid)
	{
		int reg = 0; // Register storing resulting
		Vdbe *v = parse->GetVdbe();
		if (_NEVER(!v))
			return 0;
		CachePush(parse);
		// This code must be run in its entirety every time it is encountered if any of the following is true:
		//    *  The right-hand side is a correlated subquery
		//    *  The right-hand side is an expression list containing variables
		//    *  We are inside a trigger
		// If all of the above are false, then we can run this code just once save the results, and reuse the same result on subsequent invocations.
		int testAddr = (!ExprHasAnyProperty(expr, EP_VarSelect) ? CodeOnce(parse) : -1); // One-time test address
#ifndef OMIT_EXPLAIN
		if (parse->Explain == 2)
		{
			char *msg = _mtagprintf(parse->Ctx, "EXECUTE %s%s SUBQUERY %d", (testAddr >= 0 ? "" : "CORRELATED "), (expr->OP == TK_IN ? "LIST" : "SCALAR"), parse->NextSelectId);
			v->AddOp4(OP_Explain, parse->SelectId, 0, 0, msg, Vdbe::P4T_DYNAMIC);
		}
#endif
		switch (expr->OP)
		{
		case TK_IN: {
			Expr *left = expr->Left; // the LHS of the IN operator
			if (mayHaveNull)
				v->AddOp2(OP_Null, 0, mayHaveNull);
			AFF affinity = left->Affinity(); // Affinity of the LHS of the IN

			// Whether this is an 'x IN(SELECT...)' or an 'x IN(<exprlist>)' expression it is handled the same way.  An ephemeral table is 
			// filled with single-field index keys representing the results from the SELECT or the <exprlist>.
			//
			// If the 'x' expression is a column value, or the SELECT... statement returns a column value, then the affinity of that
			// column is used to build the index keys. If both 'x' and the SELECT... statement are columns, then numeric affinity is used
			// if either column has NUMERIC or INTEGER affinity. If neither 'x' nor the SELECT... statement are columns, then numeric affinity is used.
			expr->TableId = parse->Tabs++;
			int addr = v->AddOp2(OP_OpenEphemeral, expr->TableId, !isRowid); // Address of OP_OpenEphemeral instruction
			if (!mayHaveNull) v->ChangeP5(Btree::OPEN_UNORDERED);
			KeyInfo keyInfo; // Keyinfo for the generated table
			_memset(&keyInfo, 0, sizeof(keyInfo));
			keyInfo.Fields = 1;
			keyInfo.SortOrders = &_CodeSubselect_SortOrder;

			if (ExprHasProperty(expr, EP_xIsSelect))
			{
				// Case 1:     expr IN (SELECT ...)
				// Generate code to write the results of the select into the temporary table allocated and opened above.
				_assert(!isRowid);
				SelectDest dest;
				Select::DestInit(&dest, SRT_Set, expr->TableId);
				dest.AffSdst = affinity;
				_assert((expr->TableId&0x0000FFFF) == expr->TableId);
				expr->x.Select->LimitId = 0;
				if (Select::Select_(parse, expr->x.Select, &dest))
					return 0;
				ExprList *list = expr->x.Select->EList;
				if (_ALWAYS(list && list->Exprs > 0))
					keyInfo.Colls[0] = BinaryCompareCollSeq(parse, expr->Left, list->Ids[0].Expr);
			}
			else if (_ALWAYS(expr->x.List))
			{
				// Case 2:     expr IN (exprlist)
				// For each expression, build an index key from the evaluation and store it in the temporary table. If <expr> is a column, then use
				// that columns affinity when building index keys. If <expr> is not a column, use numeric affinity.
				ExprList *list = expr->x.List;
				if (!affinity)
					affinity = AFF_NONE;
				keyInfo.Colls[0] = expr->Left->CollSeq(parse);
				keyInfo.SortOrders = &_CodeSubselect_SortOrder;

				// Loop through each expression in <exprlist>.
				int r1 = GetTempReg(parse);
				int r2 = GetTempReg(parse);
				v->AddOp2(OP_Null, 0, r2);
				int i;
				ExprList::ExprListItem *item;
				for (i = list->Exprs, item = list->Ids; i > 0; i--, item++)
				{
					Expr *e2 = item->Expr;
					// If the expression is not constant then we will need to disable the test that was generated above that makes sure
					// this code only executes once.  Because for a non-constant expression we need to rerun this code each time.
					if (testAddr >= 0 && !e2->IsConstant())
					{
						v->ChangeToNoop(testAddr);
						testAddr = -1;
					}
					// Evaluate the expression and insert it into the temp table
					int valToIns;
					if (isRowid && e2->IsInteger(&valToIns))
						v->AddOp3(OP_InsertInt, expr->TableId, r2, valToIns);
					else
					{
						int r3 = CodeTarget(parse, e2, r1);
						if (isRowid)
						{
							v->AddOp2(OP_MustBeInt, r3, v->CurrentAddr()+2);
							v->AddOp3(OP_Insert, expr->TableId, r2, r3);
						}
						else
						{
							v->AddOp4(OP_MakeRecord, r3, 1, r2, (char *)&affinity, 1);
							CacheAffinityChange(parse, r3, 1);
							v->AddOp2(OP_IdxInsert, expr->TableId, r2);
						}
					}
				}
				ReleaseTempReg(parse, r1);
				ReleaseTempReg(parse, r2);
			}
			if (!isRowid)
				v->ChangeP4(addr, (const char *)&keyInfo, Vdbe::P4T_KEYINFO);
			break; }

		case TK_EXISTS:
		case TK_SELECT:
		default: {
			ASSERTCOVERAGE(expr->OP == TK_EXISTS);
			ASSERTCOVERAGE(expr->OP == TK_SELECT);
			_assert(expr->OP == TK_EXISTS || expr->OP == TK_SELECT);
			_assert(ExprHasProperty(expr, EP_xIsSelect));

			// If this has to be a scalar SELECT.  Generate code to put the value of this select in a memory cell and record the number
			// of the memory cell in iColumn.  If this is an EXISTS, write an integer 0 (not exists) or 1 (exists) into a memory cell
			// and record that memory cell in iColumn.
			Select *sel = expr->x.Select; // SELECT statement to encode
			SelectDest dest; // How to deal with SELECt result
			Select::DestInit(&dest, (SRT)0, ++parse->Mems);
			if (expr->OP == TK_SELECT)
			{
				dest.Dest = SRT_Mem;
				v->AddOp2(OP_Null, 0, dest.SDParmId);
				v->Comment("Init subquery result");
			}
			else
			{
				dest.Dest = SRT_Exists;
				v->AddOp2(OP_Integer, 0, dest.SDParmId);
				v->Comment("Init EXISTS result");
			}
			Expr::Delete(parse->Ctx, sel->Limit);
			sel->Limit = PExpr_(parse, TK_INTEGER, 0, 0, &g_intTokens[1]);
			sel->LimitId = 0;
			if (Select::Select_(parse, sel, &dest))
				return 0;
			reg = dest.SDParmId;
			ExprSetIrreducible(expr);
			break; }
		}
		if (testAddr >= 0)
			v->JumpHere(testAddr);
		CachePop(parse, 1);
		return reg;
	}

	__device__ void Expr::CodeIN(Parse *parse, Expr *expr, int destIfFalse, int destIfNull)
	{
		// Compute the RHS. After this step, the table with cursor expr->TableId will contains the values that make up the RHS.
		Vdbe *v = parse->V; // Statement under construction
		_assert(v); // OOM detected prior to this routine
		v->NoopComment("begin IN expr");
		int rhsHasNull = 0; // Register that is true if RHS contains NULL values
		int type = FindInIndex(parse, expr, &rhsHasNull); // Type of the RHS

		// Figure out the affinity to use to create a key from the results of the expression. affinityStr stores a static string suitable for P4 of OP_MakeRecord.
		AFF affinity = ComparisonAffinity(expr); // Comparison affinity to use

		// Code the LHS, the <expr> from "<expr> IN (...)".
		CachePush(parse);
		int r1 = GetTempReg(parse); // Temporary use register
		Code(parse, expr->Left, r1);

		// If the LHS is NULL, then the result is either false or NULL depending on whether the RHS is empty or not, respectively.
		if (destIfNull == destIfFalse)
		{
			// Shortcut for the common case where the false and NULL outcomes are the same.
			v->AddOp2(OP_IsNull, r1, destIfNull);
		}
		else
		{
			int addr1 = v->AddOp1(OP_NotNull, r1);
			v->AddOp2(OP_Rewind, expr->TableId, destIfFalse);
			v->AddOp2(OP_Goto, 0, destIfNull);
			v->JumpHere(addr1);
		}

		if (type == IN_INDEX_ROWID)
		{
			// In this case, the RHS is the ROWID of table b-tree
			v->AddOp2(OP_MustBeInt, r1, destIfFalse);
			v->AddOp3(OP_NotExists, expr->TableId, destIfFalse, r1);
		}
		else
		{
			// In this case, the RHS is an index b-tree.
			v->AddOp4(OP_Affinity, r1, 1, 0, (char *)&affinity, 1);

			// If the set membership test fails, then the result of the  "x IN (...)" expression must be either 0 or NULL. If the set
			// contains no NULL values, then the result is 0. If the set contains one or more NULL values, then the result of the
			// expression is also NULL.
			if (rhsHasNull == 0 || destIfFalse == destIfNull)
			{
				// This branch runs if it is known at compile time that the RHS cannot contain NULL values. This happens as the result
				// of a "NOT NULL" constraint in the database schema.
				//
				// Also run this branch if NULL is equivalent to FALSE for this particular IN operator.
				v->AddOp4Int(OP_NotFound, expr->TableId, destIfFalse, r1, 1);
			}
			else
			{
				// In this branch, the RHS of the IN might contain a NULL and the presence of a NULL on the RHS makes a difference in the outcome.

				// First check to see if the LHS is contained in the RHS. If so, then the presence of NULLs in the RHS does not matter, so jump
				// over all of the code that follows.
				int j1 = v->AddOp4Int(OP_Found, expr->TableId, 0, r1, 1);

				// Here we begin generating code that runs if the LHS is not contained within the RHS.  Generate additional code that
				// tests the RHS for NULLs.  If the RHS contains a NULL then jump to destIfNull.  If there are no NULLs in the RHS then
				// jump to destIfFalse.
				int j2 = v->AddOp1(OP_NotNull, rhsHasNull);
				int j3 = v->AddOp4Int(OP_Found, expr->TableId, 0, rhsHasNull, 1);
				v->AddOp2(OP_Integer, -1, rhsHasNull);
				v->JumpHere(j3);
				v->AddOp2(OP_AddImm, rhsHasNull, 1);
				v->JumpHere(j2);

				// Jump to the appropriate target depending on whether or not the RHS contains a NULL
				v->AddOp2(OP_If, rhsHasNull, destIfNull);
				v->AddOp2(OP_Goto, 0, destIfFalse);

				// The OP_Found at the top of this branch jumps here when true, causing the overall IN expression evaluation to fall through.
				v->JumpHere(j1);
			}
		}
		ReleaseTempReg(parse, r1);
		CachePop(parse, 1);
		v->Comment("end IN expr");
	}

#endif
#pragma endregion

	__device__ static char *Dup8bytes(Vdbe *v, const char *in)
	{
		char *out = (char *)_tagalloc(v->Ctx, 8);
		if (out)
			_memcpy(out, in, 8);
		return out;
	}

#ifndef OMIT_FLOATING_POINT
	//
	// Generate an instruction that will put the floating point value described by z[0..n-1] into register iMem.
	//
	// The z[] string will probably not be zero-terminated.  But the z[n] character is guaranteed to be something that does not look
	// like the continuation of the number.
	__device__ static void CodeReal(Vdbe *v, const char *z, bool negateFlag, int mem)
	{
		if (_ALWAYS(z))
		{
			double value;
			ConvertEx::Atof(z, &value, _strlen30(z), TEXTENCODE_UTF8);
			_assert(!_isnan(value)); // The new AtoF never returns NaN
			if (negateFlag) value = -value;
			char *value2 = Dup8bytes(v, (char *)&value);
			v->AddOp4(OP_Real, 0, mem, 0, value2, Vdbe::P4T_REAL);
		}
	}
#endif

	__device__ static void CodeInteger(Parse *parse, Expr *expr, bool negateFlag, int mem)
	{
		Vdbe *v = parse->V;
		if (expr->Flags & EP_IntValue)
		{
			int i = expr->u.I;
			_assert(i >= 0);
			if (negateFlag) i = -i;
			v->AddOp2(OP_Integer, i, mem);
		}
		else
		{
			const char *z = expr->u.Token;
			_assert(z);
			int64 value;
			int c = ConvertEx::Atoi64(z, &value, _strlen30(z), TEXTENCODE_UTF8);
			if (c == 0 || (c == 2 && negateFlag))
			{
				if (negateFlag) { value = (c == 2 ? SMALLEST_INT64 : -value); }
				char *value2 = Dup8bytes(v, (char *)&value);
				v->AddOp4(OP_Int64, 0, mem, 0, value2, Vdbe::P4T_INT64);
			}
			else
			{
#ifdef OMIT_FLOATING_POINT
				parse->ErrorMsg("oversized integer: %s%s", (negateFlag ? "-" : ""), z);
#else
				CodeReal(v, z, negateFlag, mem);
#endif
			}
		}
	}

#pragma region Column Cache

	__device__ static void CacheEntryClear(Parse *parse, Parse::ColCache *p)
	{
		if (p->TempReg)
		{
			if (parse->TempReg.length < _lengthof(parse->TempReg.data))
				parse->TempReg[parse->TempReg.length++] = p->Reg;
			p->TempReg = 0;
		}
	}

	__device__ void Expr::CacheStore(Parse *parse, int table, int column, int reg)
	{
		_assert(reg > 0);  // Register numbers are always positive
		_assert(column >= -1 && column < 32768);  // Finite column numbers
		// The SQLITE_ColumnCache flag disables the column cache.  This is used for testing only - to verify that SQLite always gets the same answer
		// with and without the column cache.
		if (CtxOptimizationDisabled(parse->Ctx, OPTFLAG_ColumnCache))
			return;
		// First replace any existing entry.
		// Actually, the way the column cache is currently used, we are guaranteed that the object will never already be in cache.  Verify this guarantee.
		int i;
		Parse::ColCache *p;
#ifndef NDEBUG
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
			_assert(p->Reg == 0 || p->Table != table || p->Column != column);
#endif
		// Find an empty slot and replace it
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg == 0)
			{
				p->Level = parse->CacheLevel;
				p->Table = table;
				p->Column = column;
				p->Reg = reg;
				p->TempReg = 0;
				p->Lru = parse->CacheCnt++;
				return;
			}
		}
		// Replace the last recently used
		int minLru = 0x7fffffff;
		int idxLru = -1;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Lru < minLru)
			{
				idxLru = i;
				minLru = p->Lru;
			}
		}
		if (_ALWAYS(idxLru >= 0))
		{
			p = &parse->ColCaches[idxLru];
			p->Level = parse->CacheLevel;
			p->Table = table;
			p->Column = column;
			p->Reg = reg;
			p->TempReg = 0;
			p->Lru = parse->CacheCnt++;
			return;
		}
	}

	__device__ void Expr::CacheRemove(Parse *parse, int reg, int regs)
	{
		int last = reg + regs - 1;
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= reg && r <= last)
			{
				CacheEntryClear(parse, p);
				p->Reg = 0;
			}
		}
	}

	__device__ void Expr::CachePush(Parse *parse)
	{
		parse->CacheLevel++;
	}

	__device__ void Expr::CachePop(Parse *parse, int n)
	{
		_assert(n > 0);
		_assert(parse->CacheLevel >= n);
		parse->CacheLevel -= n;
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
			if (p->Reg && p->Level > parse->CacheLevel)
			{
				CacheEntryClear(parse, p);
				p->Reg = 0;
			}
	}

	__device__ void Expr::CachePinRegister(Parse *parse, int reg)
	{
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
			if (p->Reg == reg)
				p->TempReg = 0;
	}

	__device__ void Expr::CodeGetColumnOfTable(Vdbe *v, Core::Table *table, int tableId, int column, int regOut)
	{
		if (column < 0 || column == table->PKey)
			v->AddOp2(OP_Rowid, tableId, regOut);
		else
		{
			int op = (IsVirtual(table) ? OP_VColumn : OP_Column);
			v->AddOp3(op, tableId, column, regOut);
		}
		if (column >= 0)
			Update::ColumnDefault(v, table, column, regOut);
	}

	__device__ int Expr::CodeGetColumn(Parse *parse, Core::Table *table, int column, int tableId, int reg, uint8 p5)
	{
		Vdbe *v = parse->V;
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			if (p->Reg > 0 && p->Table == tableId && p->Column == column)
			{
				p->Lru = parse->CacheCnt++;
				CachePinRegister(parse, p->Reg);
				return p->Reg;
			}
		}  
		_assert(v);
		CodeGetColumnOfTable(v, table, tableId, column, reg);
		if (p5)
			v->ChangeP5(p5);
		else
			CacheStore(parse, tableId, column, reg);
		return reg;
	}

	__device__ void Expr::CacheClear(Parse *parse)
	{
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
			if (p->Reg)
			{
				CacheEntryClear(parse, p);
				p->Reg = 0;
			}
	}

	__device__ void Expr::CacheAffinityChange(Parse *parse, int start, int count)
	{ 
		CacheRemove(parse, start, count);
	}

	__device__ void Expr::CodeMove(Parse *parse, int from, int to, int regs)
	{
		_assert(from >= to + regs || from + regs <= to);
		parse->V->AddOp3(OP_Move, from, to, regs - 1);
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= from && r < from + regs)
				p->Reg += to - from;
		}
	}

#pragma endregion

#if defined(_DEBUG) || defined(COVERAGE_TEST)
	__device__ static int UsedAsColumnCache(Parse *parse, int from, int to)
	{
		int i;
		Parse::ColCache *p;
		for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
		{
			int r = p->Reg;
			if (r >= from && r <= to)
				return true;
		}
		return false;
	}
#endif

	__device__ int Expr::CodeTarget(Parse *parse, Expr *expr, int target)
	{
		_assert(target > 0 && target <= parse->Mems);
		Context *ctx = parse->Ctx; // The database connection
		Vdbe *v = parse->V; // The VM under construction
		if (!v)
		{
			_assert(ctx->MallocFailed);
			return 0;
		}

		int inReg = target;       // Results stored in register inReg
		int regFree1 = 0;         // If non-zero free this temporary register
		int regFree2 = 0;         // If non-zero free this temporary register
		int r1, r2, r3, r4;       // Various register numbers

		int op = (!expr ? TK_NULL : expr->OP); // The opcode being coded
		switch (op)
		{
		case TK_AGG_COLUMN: {
			Core::AggInfo *aggInfo = expr->AggInfo;
			AggInfo::AggInfoColumn *col = &aggInfo->Columns[expr->Agg];
			if (!aggInfo->DirectMode)
			{
				_assert(col->Mem > 0);
				inReg = col->Mem;
				break;
			}
			else if (aggInfo->UseSortingIdx)
			{
				v->AddOp3(OP_Column, aggInfo->SortingIdxPTab, col->SorterColumn, target);
				break;
			} }
							// Otherwise, fall thru into the TK_COLUMN case
		case TK_COLUMN: {
			if (expr->TableId < 0)
			{
				// This only happens when coding check constraints
				_assert(parse->CkBase > 0);
				inReg = expr->ColumnIdx + parse->CkBase;
			}
			else
				inReg = CodeGetColumn(parse, expr->Table, expr->ColumnIdx, expr->TableId, target, expr->OP2);
			break; }

		case TK_INTEGER: {
			CodeInteger(parse, expr, false, target);
			break; }

#ifndef OMIT_FLOATING_POINT
		case TK_FLOAT: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			CodeReal(v, expr->u.Token, false, target);
			break; }
#endif

		case TK_STRING: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			v->AddOp4(OP_String8, 0, target, 0, expr->u.Token, 0);
			break; }

		case TK_NULL: {
			v->AddOp2(OP_Null, 0, target);
			break; }

#ifndef OMIT_BLOB_LITERAL
		case TK_BLOB: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			_assert(expr->u.Token[0] == 'x' || expr->u.Token[0] == 'X');
			_assert(expr->u.Token[1] == '\'' );
			const char *z = &expr->u.Token[2];
			int n = _strlen30(z) - 1;
			_assert(z[n] == '\'');
			char *blob = (char *)_taghextoblob(ctx, z, n);
			v->AddOp4(OP_Blob, n/2, target, 0, blob, Vdbe::P4T_DYNAMIC);
			break; }
#endif

		case TK_VARIABLE: {
			_assert(!ExprHasProperty(expr, EP_IntValue));
			_assert(expr->u.Token != 0);
			_assert(expr->u.Token[0] != 0);
			v->AddOp2(OP_Variable, expr->ColumnIdx, target);
			if (expr->u.Token[1] != 0)
			{
				_assert(expr->u.Token[0] == '?' || !_strcmp(expr->u.Token, parse->Vars[expr->ColumnIdx-1]));
				v->ChangeP4(-1, parse->Vars[expr->ColumnIdx-1], Vdbe::P4T_STATIC);
			}
			break; }

		case TK_REGISTER: {
			inReg = expr->TableId;
			break; }

		case TK_AS: {
			inReg = CodeTarget(parse, expr->Left, target);
			break; }

#ifndef OMIT_CAST
		case TK_CAST: {
			// Expressions of the form:   CAST(pLeft AS token)
			inReg = CodeTarget(parse, expr->Left, target);
			_assert(!ExprHasProperty(expr, EP_IntValue));
			AFF aff = Parse::AffinityType(expr->u.Token);
			int toOP = (int)aff - AFF_TEXT + OP_ToText;
			_assert(toOP == OP_ToText	|| aff != AFF_TEXT);
			_assert(toOP == OP_ToBlob    || aff != AFF_NONE);
			_assert(toOP == OP_ToNumeric || aff != AFF_NUMERIC);
			_assert(toOP == OP_ToInt     || aff != AFF_INTEGER);
			_assert(toOP == OP_ToReal    || aff != AFF_REAL);
			ASSERTCOVERAGE(toOP == OP_ToText);
			ASSERTCOVERAGE(toOP == OP_ToBlob);
			ASSERTCOVERAGE(toOP == OP_ToNumeric);
			ASSERTCOVERAGE(toOP == OP_ToInt);
			ASSERTCOVERAGE(toOP == OP_ToReal);
			if (inReg != target)
			{
				v->AddOp2(OP_SCopy, inReg, target);
				inReg = target;
			}
			v->AddOp1(toOP, inReg);
			ASSERTCOVERAGE(UsedAsColumnCache(parse, inReg, inReg));
			CacheAffinityChange(parse, inReg, 1);
			break; }
#endif

		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			_assert(TK_LT == OP_Lt);
			_assert(TK_LE == OP_Le);
			_assert(TK_GT == OP_Gt);
			_assert(TK_GE == OP_Ge);
			_assert(TK_EQ == OP_Eq);
			_assert(TK_NE == OP_Ne);
			ASSERTCOVERAGE(op == TK_LT);
			ASSERTCOVERAGE(op == TK_LE);
			ASSERTCOVERAGE(op == TK_GT);
			ASSERTCOVERAGE(op == TK_GE);
			ASSERTCOVERAGE(op == TK_EQ);
			ASSERTCOVERAGE(op == TK_NE);
			r1 = CodeTemp(parse, expr->Left, &regFree1);
			r2 = CodeTemp(parse, expr->Right, &regFree2);
			CodeCompare(parse, expr->Left, expr->Right, op, r1, r2, inReg, AFF_BIT_STOREP2);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_IS:
		case TK_ISNOT: {
			ASSERTCOVERAGE(op == TK_IS);
			ASSERTCOVERAGE(op == TK_ISNOT);
			r1 = CodeTemp(parse, expr->Left, &regFree1);
			r2 = CodeTemp(parse, expr->Right, &regFree2);
			op = (op == TK_IS ? TK_EQ : TK_NE);
			CodeCompare(parse, expr->Left, expr->Right, op, r1, r2, inReg, AFF_BIT_STOREP2 | AFF_BIT_NULLEQ);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_AND:
		case TK_OR:
		case TK_PLUS:
		case TK_STAR:
		case TK_MINUS:
		case TK_REM:
		case TK_BITAND:
		case TK_BITOR:
		case TK_SLASH:
		case TK_LSHIFT:
		case TK_RSHIFT: 
		case TK_CONCAT: {
			_assert(TK_AND == OP_And);
			_assert(TK_OR == OP_Or);
			_assert(TK_PLUS == OP_Add);
			_assert(TK_MINUS == OP_Subtract);
			_assert(TK_REM == OP_Remainder);
			_assert(TK_BITAND == OP_BitAnd);
			_assert(TK_BITOR == OP_BitOr);
			_assert(TK_SLASH == OP_Divide);
			_assert(TK_LSHIFT == OP_ShiftLeft);
			_assert(TK_RSHIFT == OP_ShiftRight);
			_assert(TK_CONCAT == OP_Concat);
			ASSERTCOVERAGE(op == TK_AND);
			ASSERTCOVERAGE(op == TK_OR);
			ASSERTCOVERAGE(op == TK_PLUS);
			ASSERTCOVERAGE(op == TK_MINUS);
			ASSERTCOVERAGE(op == TK_REM);
			ASSERTCOVERAGE(op == TK_BITAND);
			ASSERTCOVERAGE(op == TK_BITOR);
			ASSERTCOVERAGE(op == TK_SLASH);
			ASSERTCOVERAGE(op == TK_LSHIFT);
			ASSERTCOVERAGE(op == TK_RSHIFT);
			ASSERTCOVERAGE(op == TK_CONCAT);
			r1 = CodeTemp(parse, expr->Left, &regFree1);
			r2 = CodeTemp(parse, expr->Right, &regFree2);
			v->AddOp3(op, r2, r1, target);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_UMINUS: {
			Expr *left = expr->Left;
			_assert(left);
			if (left->OP == TK_INTEGER)
			{
				CodeInteger(parse, left, 1, target);
#ifndef OMIT_FLOATING_POINT
			}
			else if (left->OP == TK_FLOAT)
			{
				_assert(!ExprHasProperty(expr, EP_IntValue));
				CodeReal(v, left->u.Token, 1, target);
#endif
			}
			else
			{
				regFree1 = r1 = GetTempReg(parse);
				v->AddOp2(OP_Integer, 0, r1);
				r2 = CodeTemp(parse, expr->Left, &regFree2);
				v->AddOp3(OP_Subtract, r2, r1, target);
				ASSERTCOVERAGE(regFree2 == 0);
			}
			inReg = target;
			break; }

		case TK_BITNOT:
		case TK_NOT: {
			_assert(TK_BITNOT == OP_BitNot);
			_assert(TK_NOT == OP_Not);
			ASSERTCOVERAGE(op == TK_BITNOT);
			ASSERTCOVERAGE(op == TK_NOT);
			r1 = CodeTemp(parse, expr->Left, &regFree1);
			ASSERTCOVERAGE(regFree1 == 0);
			inReg = target;
			v->AddOp2(op, r1, inReg);
			break; }

		case TK_ISNULL:
		case TK_NOTNULL: {
			int addr;
			_assert(TK_ISNULL == OP_IsNull);
			_assert(TK_NOTNULL == OP_NotNull);
			ASSERTCOVERAGE(op == TK_ISNULL);
			ASSERTCOVERAGE(op == TK_NOTNULL);
			v->AddOp2(OP_Integer, 1, target);
			r1 = CodeTemp(parse, expr->Left, &regFree1);
			ASSERTCOVERAGE(regFree1 == 0);
			addr = v->AddOp1(op, r1);
			v->AddOp2(OP_AddImm, target, -1);
			v->JumpHere(addr);
			break; }

		case TK_AGG_FUNCTION: {
			Core::AggInfo *info = expr->AggInfo;
			if (!info)
			{
				_assert(!ExprHasProperty(expr, EP_IntValue));
				parse->ErrorMsg("misuse of aggregate: %s()", expr->u.Token);
			}
			else
				inReg = info->Funcs[expr->Agg].Mem;
			break; }

		case TK_CONST_FUNC:
		case TK_FUNCTION: {
			_assert(!ExprHasProperty(expr, EP_xIsSelect));
			ASSERTCOVERAGE(op == TK_CONST_FUNC);
			ASSERTCOVERAGE(op == TK_FUNCTION);
			ExprList *farg = (ExprHasAnyProperty(expr, EP_TokenOnly) ? nullptr : expr->x.List); // List of function arguments
			int fargLength = (farg ? farg->Exprs : 0); // Number of function arguments */
			_assert(!ExprHasProperty(expr, EP_IntValue));
			const char *id = expr->u.Token; // The function name
			int idLength = _strlen30(id); // Length of the function name in bytes
			TEXTENCODE encode = CTXENCODE(ctx); // The text encoding used by this database
			FuncDef *def = Callback::FindFunction(ctx, id, idLength, fargLength, encode, false); // The function definition object
			if (!def)
			{
				parse->ErrorMsg("unknown function: %.*s()", idLength, id);
				break;
			}

			// Attempt a direct implementation of the built-in COALESCE() and IFNULL() functions.  This avoids unnecessary evalation of
			// arguments past the first non-NULL argument.
			int i;
			if (def->Flags & FUNC_COALESCE)
			{
				int endCoalesce = v->MakeLabel();
				_assert(fargLength >= 2);
				Code(parse, farg->Ids[0].Expr, target);
				for (i = 1; i < fargLength; i++)
				{
					v->AddOp2(OP_NotNull, target, endCoalesce);
					CacheRemove(parse, target, 1);
					CachePush(parse);
					Code(parse, farg->Ids[i].Expr, target);
					CachePop(parse, 1);
				}
				v->ResolveLabel(endCoalesce);
				break;
			}

			if (farg)
			{
				r1 = GetTempRange(parse, fargLength);
				// For length() and typeof() functions with a column argument, set the P5 parameter to the OP_Column opcode to OPFLAG_LENGTHARG
				// or OPFLAG_TYPEOFARG respectively, to avoid unnecessary data loading.
				if ((def->Flags & (FUNC_LENGTH | FUNC_TYPEOF)) != 0)
				{
					_assert(fargLength == 1);
					_assert(farg->Ids[0].Expr);
					uint8 exprOP = farg->Ids[0].Expr->OP;
					if (exprOP == TK_COLUMN || exprOP == TK_AGG_COLUMN)
					{
						_assert(FUNC_LENGTH == Vdbe::OPFLAG_LENGTHARG);
						_assert(FUNC_TYPEOF == Vdbe::OPFLAG_TYPEOFARG);
						ASSERTCOVERAGE(def->Flags == FUNC_LENGTH);
						farg->Ids[0].Expr->OP2 = def->Flags;
					}
				}
				CachePush(parse); // Ticket 2ea2425d34be
				CodeExprList(parse, farg, r1, 1);
				CachePop(parse, 1); // Ticket 2ea2425d34be
			}
			else
				r1 = 0;
#ifndef OMIT_VIRTUALTABLE
			// Possibly overload the function if the first argument is a virtual table column.
			//
			// For infix functions (LIKE, GLOB, REGEXP, and MATCH) use the second argument, not the first, as the argument to test to
			// see if it is a column in a virtual table.  This is done because the left operand of infix functions (the operand we want to
			// control overloading) ends up as the second argument to the function.  The expression "A glob B" is equivalent to 
			// "glob(B,A).  We want to use the A in "A glob B" to test for function overloading.  But we use the B term in "glob(B,A)".
			if (fargLength >= 2 && (expr->Flags & EP_InfixFunc) ? 1 : 0)
				def = VTable::OverloadFunction(ctx, def, fargLength, farg->Ids[1].Expr);
			else if (fargLength > 0)
				def = VTable::OverloadFunction(ctx, def, fargLength, farg->Ids[0].Expr);
#endif

			int constMask = 0; // Mask of function arguments that are constant
			Core::CollSeq *coll = nullptr; // A collating sequence
			for (i = 0; i < fargLength; i++)
			{
				if (i < 32 && farg->Ids[i].Expr->IsConstant())
					constMask |= (1<<i);
				if ((def->Flags & FUNC_NEEDCOLL) != 0 && !coll)
					coll = farg->Ids[i].Expr->CollSeq(parse);
			}
			if (def->Flags & FUNC_NEEDCOLL)
			{
				if (!coll)
					coll = ctx->DefaultColl; 
				v->AddOp4(OP_CollSeq, 0, 0, 0, (char *)coll, Vdbe::P4T_COLLSEQ);
			}
			v->AddOp4(OP_Function, constMask, r1, target, (char *)def, Vdbe::P4T_FUNCDEF);
			v->ChangeP5((uint8)fargLength);
			if (fargLength)
				ReleaseTempRange(parse, r1, fargLength);
			break; }

#ifndef OMIT_SUBQUERY
		case TK_EXISTS:
		case TK_SELECT: {
			ASSERTCOVERAGE(op == TK_EXISTS);
			ASSERTCOVERAGE(op == TK_SELECT);
			inReg = CodeSubselect(parse, expr, 0, false);
			break; }

		case TK_IN: {
			int destIfFalse = v->MakeLabel();
			int destIfNull = v->MakeLabel();
			v->AddOp2(OP_Null, 0, target);
			CodeIN(parse, expr, destIfFalse, destIfNull);
			v->AddOp2(OP_Integer, 1, target);
			v->ResolveLabel(destIfFalse);
			v->AddOp2(OP_AddImm, target, 0);
			v->ResolveLabel(destIfNull);
			break; }
#endif


					//
					//    x BETWEEN y AND z
					//
					// This is equivalent to
					//
					//    x>=y AND x<=z
					//
					// X is stored in pExpr->pLeft.
					// Y is stored in pExpr->pList->a[0].pExpr.
					// Z is stored in pExpr->pList->a[1].pExpr.
		case TK_BETWEEN: {
			Expr *left = expr->Left;
			ExprList::ExprListItem *item = expr->x.List->Ids;
			Expr *right = item->Expr;
			r1 = CodeTemp(parse, left, &regFree1);
			r2 = CodeTemp(parse, right, &regFree2);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			r3 = GetTempReg(parse);
			r4 = GetTempReg(parse);
			CodeCompare(parse, left, right, OP_Ge, r1, r2, r3, AFF_BIT_STOREP2);
			item++;
			right = item->Expr;
			ReleaseTempReg(parse, regFree2);
			r2 = CodeTemp(parse, right, &regFree2);
			ASSERTCOVERAGE(regFree2 == 0);
			CodeCompare(parse, left, right, OP_Le, r1, r2, r4, AFF_BIT_STOREP2);
			v->AddOp3(OP_And, r3, r4, target);
			ReleaseTempReg(parse, r3);
			ReleaseTempReg(parse, r4);
			break; }

		case TK_COLLATE: 
		case TK_UPLUS: {
			inReg = CodeTarget(parse, expr->Left, target);
			break; }

		case TK_TRIGGER: {
			// If the opcode is TK_TRIGGER, then the expression is a reference to a column in the new.* or old.* pseudo-tables available to
			// trigger programs. In this case Expr.iTable is set to 1 for the new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
			// is set to the column of the pseudo-table to read, or to -1 to read the rowid field.
			//
			// The expression is implemented using an OP_Param opcode. The p1 parameter is set to 0 for an old.rowid reference, or to (i+1)
			// to reference another column of the old.* pseudo-table, where i is the index of the column. For a new.rowid reference, p1 is
			// set to (n+1), where n is the number of columns in each pseudo-table. For a reference to any other column in the new.* pseudo-table, p1
			// is set to (n+2+i), where n and i are as defined previously. For example, if the table on which triggers are being fired is
			// declared as:
			//
			//   CREATE TABLE t1(a, b);
			//
			// Then p1 is interpreted as follows:
			//
			//   p1==0   ->    old.rowid     p1==3   ->    new.rowid
			//   p1==1   ->    old.a         p1==4   ->    new.a
			//   p1==2   ->    old.b         p1==5   ->    new.b       
			Core::Table *table = expr->Table;
			int p1 = expr->TableId * (table->Cols.length + 1) + 1 + expr->ColumnIdx;
			_assert(expr->TableId == 0 || expr->TableId == 1);
			_assert(expr->ColumnIdx >= -1 && expr->ColumnIdx < table->Cols.length);
			_assert(table->PKey < 0 || expr->ColumnIdx != table->PKey);
			_assert(p1 >= 0 && p1 < (table->Cols.length*2+2));
			v->AddOp2(OP_Param, p1, target);
			v->Comment("%s.%s -> $%d", (expr->TableId ? "new" : "old"), (expr->ColumnIdx < 0 ? "rowid" : expr->Table->Cols[expr->ColumnIdx].Name), target);

#ifndef OMIT_FLOATING_POINT
			// If the column has REAL affinity, it may currently be stored as an integer. Use OP_RealAffinity to make sure it is really real.
			if (expr->ColumnIdx >= 0 && table->Cols[expr->ColumnIdx].Affinity == AFF_REAL)
				v->AddOp1(OP_RealAffinity, target);
#endif
			break; }


						 // Form A:
						 //   CASE x WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 //
						 // Form B:
						 //   CASE WHEN e1 THEN r1 WHEN e2 THEN r2 ... WHEN eN THEN rN ELSE y END
						 //
						 // Form A is can be transformed into the equivalent form B as follows:
						 //   CASE WHEN x=e1 THEN r1 WHEN x=e2 THEN r2 ...
						 //        WHEN x=eN THEN rN ELSE y END
						 //
						 // X (if it exists) is in pExpr->pLeft.
						 // Y is in pExpr->pRight.  The Y is also optional.  If there is no
						 // ELSE clause and no other term matches, then the result of the
						 // exprssion is NULL.
						 // Ei is in pExpr->pList->a[i*2] and Ri is pExpr->pList->a[i*2+1].
						 //
						 // The result of the expression is the Ri for the first matching Ei, or if there is no matching Ei, the ELSE term Y, or if there is
						 // no ELSE term, NULL.
		default: _assert(op == TK_CASE); {

			ASSERTONLY(int cacheLevel = parse->CacheLevel);
			_assert(!ExprHasProperty(expr, EP_xIsSelect) && expr->x.List);
			_assert((expr->x.List->Exprs % 2) == 0);
			_assert(expr->x.List->Exprs > 0);
			ExprList *list = expr->x.List; // List of WHEN terms
			ExprList::ExprListItem *elems = list->Ids; // Array of WHEN terms
			int exprs = list->Exprs; // 2x number of WHEN terms
			int endLabel = v->MakeLabel(); // GOTO label for end of CASE stmt
			Expr *x; // The X expression
			Expr opCompare; // The X==Ei expression
			Expr *test = nullptr; // X==Ei (form A) or just Ei (form B)
			if ((x = expr->Left) != nullptr)
			{
				Expr cacheX = *x; // Cached expression X
				ASSERTCOVERAGE(x->OP == TK_COLUMN);
				ASSERTCOVERAGE(x->OP == TK_REGISTER);
				cacheX.TableId = CodeTemp(parse, x, &regFree1);
				ASSERTCOVERAGE(regFree1 == 0);
				cacheX.OP = TK_REGISTER;
				opCompare.OP = TK_EQ;
				opCompare.Left = &cacheX;
				test = &opCompare;
				// Ticket b351d95f9cd5ef17e9d9dbae18f5ca8611190001: The value in regFree1 might get SCopy-ed into the file result.
				// So make sure that the regFree1 register is not reused for other purposes and possibly overwritten.
				regFree1 = 0;
			}
			for (int i = 0; i < exprs; i += 2)
			{
				CachePush(parse);
				if (x)
				{
					_assert(test);
					opCompare.Right = elems[i].Expr;
				}
				else
					test = elems[i].Expr;
				int nextCase = v->MakeLabel(); // GOTO label for next WHEN clause
				ASSERTCOVERAGE(test->OP == TK_COLUMN);
				test->IfFalse(parse, nextCase, AFF_BIT_JUMPIFNULL);
				ASSERTCOVERAGE(elems[i+1].Expr->OP == TK_COLUMN);
				ASSERTCOVERAGE(elems[i+1].Expr->OP == TK_REGISTER);
				Code(parse, elems[i+1].Expr, target);
				v->AddOp2(OP_Goto, 0, endLabel);
				CachePop(parse, 1);
				v->ResolveLabel(nextCase);
			}
			if (expr->Right)
			{
				CachePush(parse);
				Code(parse, expr->Right, target);
				CachePop(parse, 1);
			}
			else
				v->AddOp2(OP_Null, 0, target);
			_assert(ctx->MallocFailed || parse->Errs > 0 || parse->CacheLevel == cacheLevel);
			v->ResolveLabel(endLabel);
			break; }

#ifndef OMIT_TRIGGER
		case TK_RAISE: {
			_assert(expr->Aff == OE_Rollback  || expr->Aff == OE_Abort || expr->Aff == OE_Fail || expr->Aff == OE_Ignore);
			if (!parse->TriggerTab)
			{
				parse->ErrorMsg("RAISE() may only be used within a trigger-program");
				return 0;
			}
			if (expr->Aff == OE_Abort)
				parse->MayAbort();
			_assert(!ExprHasProperty(expr, EP_IntValue));
			if (expr->Aff == OE_Ignore)
				v->AddOp4(OP_Halt, RC_OK, OE_Ignore, 0, expr->u.Token, 0);
			else
				parse->HaltConstraint(RC_CONSTRAINT_TRIGGER, expr->Aff, expr->u.Token, 0);
			break; }
#endif

		}
		ReleaseTempReg(parse, regFree1);
		ReleaseTempReg(parse, regFree2);
		return inReg;
	}

	__device__ int Expr::CodeTemp(Parse *parse, Expr *expr, int *reg)
	{
		int r1 = GetTempReg(parse);
		int r2 = CodeTarget(parse, expr, r1);
		if (r2 == r1)
			*reg = r1;
		else
		{
			ReleaseTempReg(parse, r1);
			*reg = 0;
		}
		return r2;
	}

	__device__ int Expr::Code(Parse *parse, Expr *expr, int target)
	{
		_assert(target > 0 && target <= parse->Mems);
		if (expr && expr->OP == TK_REGISTER)
			parse->V->AddOp2(OP_Copy, expr->TableId, target);
		else
		{
			int inReg = CodeTarget(parse, expr, target);
			_assert(parse->V || parse->Ctx->MallocFailed);
			if (inReg != target && parse->V)
				parse->V->AddOp2(OP_SCopy, inReg, target);
		}
		return target;
	}

	__device__ int Expr::CodeAndCache(Parse *parse, Expr *expr, int target)
	{
		Vdbe *v = parse->V;
		int inReg = Code(parse, expr, target);
		_assert(target > 0);
		// This routine is called for terms to INSERT or UPDATE.  And the only other place where expressions can be converted into TK_REGISTER is
		// in WHERE clause processing.  So as currently implemented, there is no way for a TK_REGISTER to exist here.  But it seems prudent to
		// keep the ALWAYS() in case the conditions above change with future modifications or enhancements. */
		if (_ALWAYS(expr->OP != TK_REGISTER))
		{  
			int mem = ++parse->Mems;
			v->AddOp2(OP_Copy, inReg, mem);
			expr->TableId = mem;
			expr->OP2 = expr->OP;
			expr->OP = TK_REGISTER;
		}
		return inReg;
	}

#pragma region Explain
#ifdef ENABLE_TREE_EXPLAIN
	__device__ void Expr::ExplainExpr(Vdbe *v, Expr *expr)
	{
		const char *binOp = nullptr;   // Binary operator
		const char *uniOp = nullptr;   // Unary operator
		TK op = (!expr ? TK_NULL : expr->OP); // The opcode being coded
		switch (op)
		{
		case TK_AGG_COLUMN: {
			Vdbe::ExplainPrintf(v, "AGG{%d:%d}", expr->TableId, expr->ColumnIdx);
			break; }

		case TK_COLUMN: {
			if (expr->TableId < 0) // This only happens when coding check constraints
				Vdbe::ExplainPrintf(v, "COLUMN(%d)", expr->ColumnIdx);
			else
				Vdbe::ExplainPrintf(v, "{%d:%d}", expr->Table, expr->ColumnIdx);
			break; }

		case TK_INTEGER: {
			if (expr->Flags & EP_IntValue)
				Vdbe::ExplainPrintf(v, "%d", expr->u.I);
			else
				Vdbe::ExplainPrintf(v, "%s", expr->u.Token);
			break; }

#ifndef OMIT_FLOATING_POINT
		case TK_FLOAT: {
			Vdbe::ExplainPrintf(v, "%s", expr->u.Token);
			break; }
#endif
		case TK_STRING: {
			Vdbe::ExplainPrintf(v, "%Q", expr->u.Token);
			break; }

		case TK_NULL: {
			Vdbe::ExplainPrintf(v, "NULL");
			break; }

#ifndef OMIT_BLOB_LITERAL
		case TK_BLOB: {
			Vdbe::ExplainPrintf(v, "%s", expr->u.Token);
			break; }
#endif
		case TK_VARIABLE: {
			Vdbe::ExplainPrintf(v, "VARIABLE(%s,%d)", expr->u.Token, expr->ColumnIdx);
			break; }

		case TK_REGISTER: {
			Vdbe::ExplainPrintf(v, "REGISTER(%d)", expr->TableId);
			break; }

		case TK_AS: {
			ExplainExpr(v, expr->Left);
			break; }

#ifndef OMIT_CAST
		case TK_CAST: {
			// Expressions of the form:   CAST(pLeft AS token)
			const char *aff = "unk";
			switch (Parse::AffinityType(expr->u.Token))
			{
			case AFF_TEXT: aff = "TEXT"; break;
			case AFF_NONE: aff = "NONE"; break;
			case AFF_NUMERIC: aff = "NUMERIC"; break;
			case AFF_INTEGER: aff = "INTEGER"; break;
			case AFF_REAL: aff = "REAL"; break;
			}
			Vdbe::ExplainPrintf(v, "CAST-%s(", aff);
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v, ")");
			break; }
#endif

		case TK_LT: binOp = "LT"; break;
		case TK_LE: binOp = "LE"; break;
		case TK_GT: binOp = "GT"; break;
		case TK_GE: binOp = "GE"; break;
		case TK_NE: binOp = "NE"; break;
		case TK_EQ: binOp = "EQ"; break;
		case TK_IS: binOp = "IS"; break;
		case TK_ISNOT: binOp = "ISNOT"; break;
		case TK_AND: binOp = "AND"; break;
		case TK_OR: binOp = "OR"; break;
		case TK_PLUS: binOp = "ADD"; break;
		case TK_STAR: binOp = "MUL"; break;
		case TK_MINUS: binOp = "SUB"; break;
		case TK_REM: binOp = "REM"; break;
		case TK_BITAND: binOp = "BITAND"; break;
		case TK_BITOR: binOp = "BITOR"; break;
		case TK_SLASH: binOp = "DIV"; break;
		case TK_LSHIFT: binOp = "LSHIFT"; break;
		case TK_RSHIFT: binOp = "RSHIFT"; break;
		case TK_CONCAT: binOp = "CONCAT"; break;

		case TK_UMINUS: uniOp = "UMINUS"; break;
		case TK_UPLUS: uniOp = "UPLUS"; break;
		case TK_BITNOT: uniOp = "BITNOT"; break;
		case TK_NOT: uniOp = "NOT"; break;
		case TK_ISNULL: uniOp = "ISNULL"; break;
		case TK_NOTNULL: uniOp = "NOTNULL"; break;

		case TK_COLLATE: {
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v,".COLLATE(%s)", expr->u.Token);
			break; }

		case TK_AGG_FUNCTION:
		case TK_CONST_FUNC:
		case TK_FUNCTION: {
			ExprList *farg = (ExprHasAnyProperty(expr, EP_TokenOnly) ? nullptr : expr->x.List); // List of function arguments
			if (op == TK_AGG_FUNCTION)
				Vdbe::ExplainPrintf(v, "AGG_FUNCTION%d:%s(", expr->OP2, expr->u.Token);
			else
				Vdbe::ExplainPrintf(v, "FUNCTION:%s(", expr->u.Token);
			if (farg)
				ExplainExprList(v, farg);
			Vdbe::ExplainPrintf(v, ")");
			break; }

#ifndef OMIT_SUBQUERY
		case TK_EXISTS: {
			Vdbe::ExplainPrintf(v, "EXISTS(");
			Select::ExplainSelect(v, expr->x.Select);
			Vdbe::ExplainPrintf(v,")");
			break; }

		case TK_SELECT: {
			Vdbe::ExplainPrintf(v, "(");
			Select::ExplainSelect(v, expr->x.Select);
			Vdbe::ExplainPrintf(v, ")");
			break; }

		case TK_IN: {
			Vdbe::ExplainPrintf(v, "IN(");
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v, ",");
			if (ExprHasProperty(expr, EP_xIsSelect))
				Select::ExplainSelect(v, expr->x.Select);
			else
				ExplainExprList(v, expr->x.List);
			Vdbe::ExplainPrintf(v, ")");
			break; }
#endif

					//    x BETWEEN y AND z
					// This is equivalent to
					//    x>=y AND x<=z
					// X is stored in pExpr->pLeft.
					// Y is stored in pExpr->pList->a[0].pExpr.
					// Z is stored in pExpr->pList->a[1].pExpr.
		case TK_BETWEEN: {
			Expr *x = expr->Left;
			Expr *y = expr->x.List->Ids[0].Expr;
			Expr *z = expr->x.List->Ids[1].Expr;
			Vdbe::ExplainPrintf(v, "BETWEEN(");
			ExplainExpr(v, x);
			Vdbe::ExplainPrintf(v, ",");
			ExplainExpr(v, y);
			Vdbe::ExplainPrintf(v, ",");
			ExplainExpr(v, z);
			Vdbe::ExplainPrintf(v, ")");
			break; }

		case TK_TRIGGER: {
			// If the opcode is TK_TRIGGER, then the expression is a reference to a column in the new.* or old.* pseudo-tables available to
			// trigger programs. In this case Expr.iTable is set to 1 for the new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
			// is set to the column of the pseudo-table to read, or to -1 to read the rowid field.
			Vdbe::ExplainPrintf(v, "%s(%d)", (expr->TableId ? "NEW" : "OLD"), expr->ColumnIdx);
			break; }

		case TK_CASE: {
			Vdbe::ExplainPrintf(v, "CASE(");
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v, ",");
			ExplainExprList(v, expr->x.List);
			break; }

#ifndef OMIT_TRIGGER
		case TK_RAISE: {
			const char *type = "unk";
			switch (expr->Aff)
			{
			case OE_Rollback:   type = "rollback";  break;
			case OE_Abort:      type = "abort";     break;
			case OE_Fail:       type = "fail";      break;
			case OE_Ignore:     type = "ignore";    break;
			}
			Vdbe::ExplainPrintf(v, "RAISE-%s(%s)", type, expr->u.Token);
			break; }
#endif
		}

		if (binOp)
		{
			Vdbe::ExplainPrintf(v,"%s(", binOp);
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v,",");
			ExplainExpr(v, expr->Right);
			Vdbe::ExplainPrintf(v,")");
		}
		else if (uniOp)
		{
			Vdbe::ExplainPrintf(v,"%s(", uniOp);
			ExplainExpr(v, expr->Left);
			Vdbe::ExplainPrintf(v,")");
		}
	}

	__device__ void Expr::ExplainExprList(Vdbe *v, ExprList *list)
	{
		if (!list || list->Exprs == 0)
		{
			Vdbe::ExplainPrintf(v, "(empty-list)");
			return;
		}
		else if (list->Exprs == 1)
			ExplainExpr(v, list->Ids[0].Expr);
		else
		{
			Vdbe::ExplainPush(v);
			for (int i = 0; i < list->Exprs; i++)
			{
				Vdbe::ExplainPrintf(v, "item[%d] = ", i);
				Vdbe::ExplainPush(v);
				ExplainExpr(v, list->Ids[i].Expr);
				Vdbe::ExplainPop(v);
				if (list->Ids[i].Name)
					Vdbe::ExplainPrintf(v, " AS %s", list->Ids[i].Name);
				if (list->Ids[i].SpanIsTab)
					Vdbe::ExplainPrintf(v, " (%s)", list->Ids[i].Span);
				if (i < list->Exprs-1)
					Vdbe::ExplainNL(v);
			}
			Vdbe::ExplainPop(v);
		}
	}
#endif
#pragma endregion

	__device__ static bool IsAppropriateForFactoring(Expr *expr)
	{
		if (!expr->IsConstantNotJoin())
			return false; // Only constant expressions are appropriate for factoring
		if ((expr->Flags & EP_FixedDest) == 0)
			return true; // Any constant without a fixed destination is appropriate
		while (expr->OP == TK_UPLUS) expr = expr->Left;
		switch (expr->OP)
		{
#ifndef OMIT_BLOB_LITERAL
		case TK_BLOB:
#endif
		case TK_VARIABLE:
		case TK_INTEGER:
		case TK_FLOAT:
		case TK_NULL:
		case TK_STRING: {
			ASSERTCOVERAGE(expr->OP == TK_BLOB);
			ASSERTCOVERAGE(expr->OP == TK_VARIABLE);
			ASSERTCOVERAGE(expr->OP == TK_INTEGER);
			ASSERTCOVERAGE(expr->OP == TK_FLOAT);
			ASSERTCOVERAGE(expr->OP == TK_NULL);
			ASSERTCOVERAGE(expr->OP == TK_STRING);
			// Single-instruction constants with a fixed destination are better done in-line.  If we factor them, they will just end
			// up generating an OP_SCopy to move the value to the destination register.
			return false; }

		case TK_UMINUS: {
			if (expr->Left->OP == TK_FLOAT || expr->Left->OP == TK_INTEGER)
				return false;
			break; }
		}
		return true;
	}

	__device__ static WRC EvalConstExpr(Walker *walker, Expr *expr)
	{
		Parse *parse = walker->Parse;
		switch (expr->OP)
		{
		case TK_IN:
		case TK_REGISTER: {
			return WRC_Prune; }
		case TK_COLLATE: {
			return WRC_Continue; }
		case TK_FUNCTION:
		case TK_AGG_FUNCTION:
		case TK_CONST_FUNC: {
			// The arguments to a function have a fixed destination. Mark them this way to avoid generated unneeded OP_SCopy
			// instructions. 
			ExprList *list = expr->x.List;
			_assert(!ExprHasProperty(expr, EP_xIsSelect));
			if (list)
			{
				int i;
				ExprList::ExprListItem *item;
				for (i = list->Exprs, item = list->Ids; i > 0; i--, item++)
					if (_ALWAYS(item->Expr))
						item->Expr->Flags |= EP_FixedDest;
			}
			break; }
		}
		if (IsAppropriateForFactoring(expr))
		{
			int r1 = ++parse->Mems;
			int r2 = Expr::CodeTarget(parse, expr, r1);
			// If r2!=r1, it means that register r1 is never used.  That is harmless but suboptimal, so we want to know about the situation to fix it.
			// Hence the following assert:
			_assert(r2 == r1);
			expr->OP2 = expr->OP;
			expr->OP = TK_REGISTER;
			expr->TableId = r2;
			return WRC_Prune;
		}
		return WRC_Continue;
	}

	__device__ void Expr::CodeConstants(Parse *parse, Expr *expr)
	{
		if (parse->CookieGoto || CtxOptimizationDisabled(parse->Ctx, OPTFLAG_FactorOutConst))
			return;
		Walker w;
		w.ExprCallback = EvalConstExpr;
		w.SelectCallback = nullptr;
		w.Parse = parse;
		w.WalkExpr(expr);
	}

	__device__ int Expr::CodeExprList(Parse *parse, ExprList *list, int target, bool doHardCopy)
	{
		_assert(list);
		_assert(target > 0);
		_assert(parse->V); // Never gets this far otherwise
		int n = list->Exprs;
		int i;
		ExprList::ExprListItem *item;
		for (item = list->Ids, i = 0; i < n; i++, item++)
		{
			Expr *expr = item->Expr;
			int inReg = CodeTarget(parse, expr, target+i);
			if (inReg != target+i)
				parse->V->AddOp2((doHardCopy ? OP_Copy : OP_SCopy), inReg, target+i);
		}
		return n;
	}

	__device__ static void ExprCodeBetween(Parse *parse, Expr *expr, int dest, bool jumpIfTrue, AFF jumpIfNull)
	{
		_assert(!ExprHasProperty(expr, EP_xIsSelect));
		Expr exprX = *expr->Left; // The  x  subexpression
		Expr compLeft; // The  x>=y  term
		Expr compRight;   // The  x<=z  term
		Expr exprAnd; // The AND operator in  x>=y AND x<=z
		exprAnd.OP = TK_AND;
		exprAnd.Left = &compLeft;
		exprAnd.Right = &compRight;
		compLeft.OP = TK_GE;
		compLeft.Left = &exprX;
		compLeft.Right = expr->x.List->Ids[0].Expr;
		compRight.OP = TK_LE;
		compRight.Left = &exprX;
		compRight.Right = expr->x.List->Ids[1].Expr;
		int regFree1 = 0; // Temporary use register
		exprX.TableId = Expr::CodeTemp(parse, &exprX, &regFree1);
		exprX.OP = TK_REGISTER;
		if (jumpIfTrue)
			exprAnd.IfTrue(parse, dest, jumpIfNull);
		else
			exprAnd.IfFalse(parse, dest, jumpIfNull);
		Expr::ReleaseTempReg(parse, regFree1);

		// Ensure adequate test coverage
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 != 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 == 0);
		ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 != 0);
	}

	__device__ void Expr::IfTrue(Parse *parse, int dest, AFF jumpIfNull)
	{
		_assert(jumpIfNull == AFF_BIT_JUMPIFNULL || jumpIfNull == 0);
		Vdbe *v = parse->V;
		if (_NEVER(!v)) return;  // Existence of VDBE checked by caller

		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		int op = OP;
		switch (op)
		{
		case TK_AND: {
			int d2 = v->MakeLabel();
			ASSERTCOVERAGE(jumpIfNull == 0);
			CachePush(parse);
			Left->IfFalse(parse, d2, (AFF)(jumpIfNull ^ AFF_BIT_JUMPIFNULL));
			Right->IfTrue(parse, dest, jumpIfNull);
			v->ResolveLabel(d2);
			CachePop(parse, 1);
			break; }

		case TK_OR: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			Left->IfTrue(parse, dest, jumpIfNull);
			Right->IfTrue(parse, dest, jumpIfNull);
			break; }

		case TK_NOT: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			Left->IfFalse(parse, dest, jumpIfNull);
			break; }

		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			_assert(TK_LT == OP_Lt);
			_assert(TK_LE == OP_Le);
			_assert(TK_GT == OP_Gt);
			_assert(TK_GE == OP_Ge);
			_assert(TK_EQ == OP_Eq);
			_assert(TK_NE == OP_Ne);
			ASSERTCOVERAGE(op == TK_LT);
			ASSERTCOVERAGE(op == TK_LE);
			ASSERTCOVERAGE(op == TK_GT);
			ASSERTCOVERAGE(op == TK_GE);
			ASSERTCOVERAGE(op == TK_EQ);
			ASSERTCOVERAGE(op == TK_NE);
			ASSERTCOVERAGE(jumpIfNull == 0);
			r1 = CodeTemp(parse, Left, &regFree1);
			r2 = CodeTemp(parse, Right, &regFree2);
			CodeCompare(parse, Left, Right, op, r1, r2, dest, jumpIfNull);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_IS:
		case TK_ISNOT: {
			ASSERTCOVERAGE(op == TK_IS);
			ASSERTCOVERAGE(op == TK_ISNOT);
			r1 = CodeTemp(parse, Left, &regFree1);
			r2 = CodeTemp(parse, Right, &regFree2);
			op = (op == TK_IS ? TK_EQ : TK_NE);
			CodeCompare(parse, Left, Right, op, r1, r2, dest, AFF_BIT_NULLEQ);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_ISNULL:
		case TK_NOTNULL: {
			_assert(TK_ISNULL == OP_IsNull);
			_assert(TK_NOTNULL == OP_NotNull);
			ASSERTCOVERAGE(op == TK_ISNULL);
			ASSERTCOVERAGE(op == TK_NOTNULL);
			r1 = CodeTemp(parse, Left, &regFree1);
			v->AddOp2(op, r1, dest);
			ASSERTCOVERAGE(regFree1 == 0);
			break; }

		case TK_BETWEEN: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprCodeBetween(parse, this, dest, 1, jumpIfNull);
			break; }

#ifndef OMIT_SUBQUERY
		case TK_IN: {
			int destIfFalse = v->MakeLabel();
			int destIfNull = (jumpIfNull ? dest : destIfFalse);
			CodeIN(parse, this, destIfFalse, destIfNull);
			v->AddOp2(OP_Goto, 0, dest);
			v->ResolveLabel(destIfFalse);
			break; }
#endif
		default: {
			r1 = CodeTemp(parse, this, &regFree1);
			v->AddOp3(OP_If, r1, dest, jumpIfNull != 0);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(jumpIfNull == 0);
			break; }
		}
		ReleaseTempReg(parse, regFree1);
		ReleaseTempReg(parse, regFree2);  
	}

	__device__ void Expr::IfFalse(Parse *parse, int dest, AFF jumpIfNull)
	{
		Vdbe *v = parse->V;
		_assert(jumpIfNull == AFF_BIT_JUMPIFNULL || jumpIfNull == 0);
		if (_NEVER(v == nullptr)) return; // Existence of VDBE checked by caller

		int regFree1 = 0;
		int regFree2 = 0;
		int r1, r2;

		// The value of pExpr->op and op are related as follows:
		//
		//       pExpr->op            op
		//       ---------          ----------
		//       TK_ISNULL          OP_NotNull
		//       TK_NOTNULL         OP_IsNull
		//       TK_NE              OP_Eq
		//       TK_EQ              OP_Ne
		//       TK_GT              OP_Le
		//       TK_LE              OP_Gt
		//       TK_GE              OP_Lt
		//       TK_LT              OP_Ge
		//
		// For other values of pExpr->op, op is undefined and unused. The value of TK_ and OP_ constants are arranged such that we
		// can compute the mapping above using the following expression. Assert()s verify that the computation is correct.
		int op = ((OP+(TK_ISNULL&1))^1) - (TK_ISNULL&1);

		// Verify correct alignment of TK_ and OP_ constants
		_assert(OP != TK_ISNULL || op == OP_NotNull);
		_assert(OP != TK_NOTNULL || op == OP_IsNull);
		_assert(OP != TK_NE || op == OP_Eq);
		_assert(OP != TK_EQ || op == OP_Ne);
		_assert(OP != TK_LT || op == OP_Ge);
		_assert(OP != TK_LE || op == OP_Gt);
		_assert(OP != TK_GT || op == OP_Le);
		_assert(OP != TK_GE || op == OP_Lt);

		switch (OP)
		{
		case TK_AND: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			Left->IfFalse(parse, dest, jumpIfNull);
			Right->IfFalse(parse, dest, jumpIfNull);
			break; }

		case TK_OR: {
			int d2 = v->MakeLabel();
			ASSERTCOVERAGE(jumpIfNull == 0);
			CachePush(parse);
			Left->IfTrue(parse, d2, (AFF)(jumpIfNull ^ AFF_BIT_JUMPIFNULL));
			Right->IfFalse(parse, dest, jumpIfNull);
			v->ResolveLabel(d2);
			CachePop(parse, 1);
			break; }

		case TK_NOT: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			Left->IfTrue(parse, dest, jumpIfNull);
			break; }

		case TK_LT:
		case TK_LE:
		case TK_GT:
		case TK_GE:
		case TK_NE:
		case TK_EQ: {
			ASSERTCOVERAGE(op == TK_LT);
			ASSERTCOVERAGE(op == TK_LE);
			ASSERTCOVERAGE(op == TK_GT);
			ASSERTCOVERAGE(op == TK_GE);
			ASSERTCOVERAGE(op == TK_EQ);
			ASSERTCOVERAGE(op == TK_NE);
			ASSERTCOVERAGE(jumpIfNull == 0);
			r1 = CodeTemp(parse, Left, &regFree1);
			r2 = CodeTemp(parse, Right, &regFree2);
			CodeCompare(parse, Left, Right, op, r1, r2, dest, jumpIfNull);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_IS:
		case TK_ISNOT: {
			ASSERTCOVERAGE(OP == TK_IS);
			ASSERTCOVERAGE(OP == TK_ISNOT);
			r1 = CodeTemp(parse, Left, &regFree1);
			r2 = CodeTemp(parse, Right, &regFree2);
			op = (OP == TK_IS ? TK_NE : TK_EQ);
			CodeCompare(parse, Left, Right, op, r1, r2, dest, AFF_BIT_NULLEQ);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(regFree2 == 0);
			break; }

		case TK_ISNULL:
		case TK_NOTNULL: {
			ASSERTCOVERAGE(op == TK_ISNULL);
			ASSERTCOVERAGE(op == TK_NOTNULL);
			r1 = CodeTemp(parse, Left, &regFree1);
			v->AddOp2(op, r1, dest);
			ASSERTCOVERAGE(regFree1 == 0);
			break; }

		case TK_BETWEEN: {
			ASSERTCOVERAGE(jumpIfNull == 0);
			ExprCodeBetween(parse, this, dest, 0, jumpIfNull);
			break; }

#ifndef OMIT_SUBQUERY
		case TK_IN: {
			if (jumpIfNull)
				CodeIN(parse, this, dest, dest);
			else
			{
				int destIfNull = v->MakeLabel();
				CodeIN(parse, this, dest, destIfNull);
				v->ResolveLabel(destIfNull);
			}
			break; }
#endif
		default: {
			r1 = CodeTemp(parse, this, &regFree1);
			v->AddOp3(OP_IfNot, r1, dest, jumpIfNull!=0);
			ASSERTCOVERAGE(regFree1 == 0);
			ASSERTCOVERAGE(jumpIfNull == 0);
			break; }
		}
		ReleaseTempReg(parse, regFree1);
		ReleaseTempReg(parse, regFree2);
	}

	__device__ int Expr::Compare(Expr *a, Expr *b)
	{
		if (!a || !b)
			return (b == a ? 0 : 2);
		_assert(!ExprHasAnyProperty(a, EP_TokenOnly|EP_Reduced));
		_assert(!ExprHasAnyProperty(b, EP_TokenOnly|EP_Reduced));
		if (ExprHasProperty(a, EP_xIsSelect) || ExprHasProperty(b, EP_xIsSelect))
			return 2;
		if ((a->Flags & EP_Distinct) != (b->Flags & EP_Distinct)) return 2;
		if (a->OP != b->OP)
		{
			if (a->OP == TK_COLLATE && Compare(a->Left, b) < 2) return 1;
			if (b->OP == TK_COLLATE && Compare(a, b->Left) < 2) return 1;
			return 2;
		}
		if (Compare(a->Left, b->Left)) return 2;
		if (Compare(a->Right, b->Right)) return 2;
		if (ListCompare(a->x.List, b->x.List)) return 2;
		if (a->TableId != b->TableId || a->ColumnIdx != b->ColumnIdx) return 2;
		if (ExprHasProperty(a, EP_IntValue))
		{
			if (!ExprHasProperty(b, EP_IntValue) || a->u.I != b->u.I) return 2;
		}
		else if (a->OP != TK_COLUMN && _ALWAYS(a->OP != TK_AGG_COLUMN) && a->u.Token)
		{
			if (ExprHasProperty(b, EP_IntValue) || _NEVER(b->u.Token == 0)) return 2;
			if (_strcmp(a->u.Token, b->u.Token))
				return (a->OP == TK_COLLATE ? 1 : 2);
		}
		return 0;
	}

	__device__ int Expr::ListCompare(ExprList *a, ExprList *b)
	{
		if (!a && !b) return 0;
		if (!a || !b) return 1;
		if (a->Exprs != b->Exprs) return 1;
		for (int i = 0; i < a->Exprs; i++)
		{
			Expr *exprA = a->Ids[i].Expr;
			Expr *exprB = b->Ids[i].Expr;
			if (a->Ids[i].SortOrder != b->Ids[i].SortOrder) return 1;
			if (Compare(exprA, exprB)) return 1;
		}
		return 0;
	}

	struct SrcCount
	{
		SrcList *Src; // One particular FROM clause in a nested query
		int This; // Number of references to columns in pSrcList
		int Other; // Number of references to columns in other FROM clauses
	};

	__device__ static WRC ExprSrcCount(Walker *walker, Expr *expr)
	{
		// The NEVER() on the second term is because sqlite3FunctionUsesThisSrc() is always called before sqlite3ExprAnalyzeAggregates() and so the
		// TK_COLUMNs have not yet been converted into TK_AGG_COLUMN.  If sqlite3FunctionUsesThisSrc() is used differently in the future, the
		// NEVER() will need to be removed.
		if (expr->OP == TK_COLUMN || _NEVER(expr->OP == TK_AGG_COLUMN))
		{
			int i;
			SrcCount *p = walker->u.SrcCount;
			SrcList *src = p->Src;
			for (i = 0; i < src->Srcs; i++)
				if (expr->TableId == src->Ids[i].Cursor) break;
			if (i < src->Srcs)
				p->This++;
			else
				p->Other++;
		}
		return WRC_Continue;
	}

	__device__ bool Expr::FunctionUsesThisSrc(SrcList *srcList)
	{
		_assert(OP == TK_AGG_FUNCTION);
		Walker w;
		_memset(&w, 0, sizeof(w));
		w.ExprCallback = ExprSrcCount;
		SrcCount cnt;
		w.u.SrcCount = &cnt;
		cnt.Src = srcList;
		cnt.This = 0;
		cnt.Other = 0;
		w.WalkExprList(x.List);
		return (cnt.This > 0 || cnt.Other == 0);
	}

	__device__ static int AddAggInfoColumn(Context *ctx, AggInfo *info)
	{
		int i;
		info->Columns.data = (AggInfo::AggInfoColumn *)Parse::ArrayAllocate(ctx, info->Columns.data, sizeof(info->Columns[0]), &info->Columns.length, &i);
		return i;
	}    

	__device__ static int AddAggInfoFunc(Context *ctx, AggInfo *info)
	{
		int i;
		info->Funcs.data = (AggInfo::AggInfoFunc *)Parse::ArrayAllocate(ctx, info->Funcs.data, sizeof(info->Funcs[0]), &info->Funcs.length, &i);
		return i;
	}    

	__device__ static WRC AnalyzeAggregate(Walker *walker, Expr *expr)
	{
		NameContext *nc = walker->u.NC;
		Parse *parse = nc->Parse;
		SrcList *srcList = nc->SrcList;
		AggInfo *aggInfo = nc->AggInfo;
		Context *ctx = parse->Ctx;
		switch (expr->OP)
		{
		case TK_AGG_COLUMN:
		case TK_COLUMN: {
			ASSERTCOVERAGE(expr->OP == TK_AGG_COLUMN);
			ASSERTCOVERAGE(expr->OP == TK_COLUMN);
			// Check to see if the column is in one of the tables in the FROM clause of the aggregate query
			if (_ALWAYS(srcList != nullptr))
			{
				int i;
				SrcList::SrcListItem *item;
				for (i = 0, item = srcList->Ids; i < srcList->Srcs; i++, item++)
				{
					_assert(!ExprHasAnyProperty(expr, EP_TokenOnly|EP_Reduced));
					if (expr->TableId == item->Cursor)
					{
						// If we reach this point, it means that pExpr refers to a table that is in the FROM clause of the aggregate query.  
						//
						// Make an entry for the column in pAggInfo->aCol[] if there is not an entry there already.
						int k;
						AggInfo::AggInfoColumn *col;
						for (k = 0, col = aggInfo->Columns.data; k < aggInfo->Columns.length; k++, col++)
							if (col->TableID == expr->TableId && col->Column == expr->ColumnIdx)
								break;
						if ((k >= aggInfo->Columns.length) && (k = AddAggInfoColumn(ctx, aggInfo)) >= 0)
						{
							col = &aggInfo->Columns[k];
							col->Table = expr->Table;
							col->TableID = expr->TableId;
							col->Column = expr->ColumnIdx;
							col->Mem = ++parse->Mems;
							col->SorterColumn = -1;
							col->Expr = expr;
							if (aggInfo->GroupBy)
							{
								ExprList *gb = aggInfo->GroupBy;
								ExprList::ExprListItem *term = gb->Ids;
								int n = gb->Exprs;
								for (int j = 0; j < n; j++, term++)
								{
									Expr *e = term->Expr;
									if (e->OP == TK_COLUMN && e->TableId == expr->TableId && e->ColumnIdx == expr->ColumnIdx)
									{
										col->SorterColumn = j;
										break;
									}
								}
							}
							if (col->SorterColumn < 0)
								col->SorterColumn = aggInfo->SortingColumns++;
						}
						// There is now an entry for pExpr in pAggInfo->aCol[] (either because it was there before or because we just created it).
						// Convert the pExpr to be a TK_AGG_COLUMN referring to that pAggInfo->aCol[] entry.
						ExprSetIrreducible(expr);
						expr->AggInfo = aggInfo;
						expr->OP = TK_AGG_COLUMN;
						expr->Agg = (int16)k;
						break;
					}
				}
			}
			return WRC_Prune; }

		case TK_AGG_FUNCTION: {
			if ((nc->NCFlags & NC_InAggFunc) == 0 && walker->WalkerDepth == expr->OP2)
			{
				// Check to see if pExpr is a duplicate of another aggregate function that is already in the pAggInfo structure
				int i;
				AggInfo::AggInfoFunc *item = aggInfo->Funcs;
				for (i = 0; i < aggInfo->Funcs.length; i++, item++)
					if (!Expr::Compare(item->Expr, expr))
						break;
				if (i >= aggInfo->Funcs.length)
				{
					// pExpr is original.  Make a new entry in pAggInfo->aFunc[]
					TEXTENCODE encode = CTXENCODE(ctx);
					i = AddAggInfoFunc(ctx, aggInfo);
					if (i >= 0)
					{
						_assert(!ExprHasProperty(expr, EP_xIsSelect));
						item = &aggInfo->Funcs[i];
						item->Expr = expr;
						item->Mem = ++parse->Mems;
						_assert(!ExprHasProperty(expr, EP_IntValue));
						item->Func = Callback::FindFunction(ctx, expr->u.Token, _strlen30(expr->u.Token), (expr->x.List ? expr->x.List->Exprs : 0), encode, false);
						item->Distinct = (expr->Flags & EP_Distinct ? parse->Tabs++ : -1);
					}
				}
				// Make pExpr point to the appropriate pAggInfo->aFunc[] entry
				_assert(!ExprHasAnyProperty(expr, EP_TokenOnly|EP_Reduced));
				ExprSetIrreducible(expr);
				expr->Agg = (int16)i;
				expr->AggInfo = aggInfo;
				return WRC_Prune;
			}
			return WRC_Continue; }
		}
		return WRC_Continue;
	}

	__device__ static WRC AnalyzeAggregatesInSelect(Walker *walker, Select *select)
	{
		return WRC_Continue;
	}

	__device__ void Expr::AnalyzeAggregates(NameContext *nc, Expr *expr)
	{
		Walker w;
		_memset(&w, 0, sizeof(w));
		w.ExprCallback = AnalyzeAggregate;
		w.SelectCallback = AnalyzeAggregatesInSelect;
		w.u.NC = nc;
		_assert(nc->SrcList != nullptr);
		w.WalkExpr(expr);
	}

	__device__ void Expr::AnalyzeAggList(NameContext *nc, ExprList *list)
	{
		int i;
		ExprList::ExprListItem *item;
		if (list)
			for (i = 0, item = list->Ids; i < list->Exprs; i++, item++)
				AnalyzeAggregates(nc, item->Expr);
	}

#pragma region Registers

	__device__ int Expr::GetTempReg(Parse *parse)
	{
		if (parse->TempReg.length == 0)
			return ++parse->Mems;
		return parse->TempReg[--parse->TempReg.length];
	}

	__device__ void Expr::ReleaseTempReg(Parse *parse, int reg)
	{
		if (reg && parse->TempReg.length < _lengthof(parse->TempReg.data))
		{
			int i;
			Parse::ColCache *p;
			for (i = 0, p = parse->ColCaches; i < N_COLCACHE; i++, p++)
			{
				if (p->Reg == reg)
				{
					p->TempReg = 1;
					return;
				}
			}
			parse->TempReg[parse->TempReg.length++] = reg;
		}
	}

	__device__ int Expr::GetTempRange(Parse *parse, int regs)
	{
		int i = parse->RangeRegIdx;
		int n = parse->RangeRegs;
		if (regs <= n)
		{
			_assert(!UsedAsColumnCache(parse, i, i+n-1));
			parse->RangeRegIdx += regs;
			parse->RangeRegs -= regs;
		}
		else
		{
			i = parse->Mems + 1;
			parse->Mems += regs;
		}
		return i;
	}

	__device__ void Expr::ReleaseTempRange(Parse *parse, int reg, int regs)
	{
		CacheRemove(parse, reg, regs);
		if (regs > parse->RangeRegs)
		{
			parse->RangeRegs = regs;
			parse->RangeRegIdx = reg;
		}
	}

	__device__ void Expr::ClearTempRegCache(Parse *parse)
	{
		parse->TempReg.length = 0;
		parse->RangeRegs = 0;
	}

#pragma endregion

}