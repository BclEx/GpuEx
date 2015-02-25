#include "VdbeInt.cu.h"

namespace Core
{
#if _DEBUG
	__device__ bool WhereTrace = false;
#define WHERETRACE(X, ...) if (WhereTrace) { _fprintf(nullptr, X, __VA_ARGS__); }
#else
#define WHERETRACE(X, ...)
#endif

	struct WhereClause;
	struct WhereMaskSet;
	struct WhereOrInfo;
	struct WhereAndInfo;
	struct WhereCost;
	struct WhereTerm;

	enum WO : uint16
	{
		WO_IN = 0x001,
		WO_EQ = 0x002,
		WO_LT = (WO_EQ<<(TK_LT-TK_EQ)),
		WO_LE = (WO_EQ<<(TK_LE-TK_EQ)),
		WO_GT = (WO_EQ<<(TK_GT-TK_EQ)),
		WO_GE = (WO_EQ<<(TK_GE-TK_EQ)),
		WO_MATCH = 0x040,
		WO_ISNULL = 0x080,
		WO_OR = 0x100,			// Two or more OR-connected terms
		WO_AND = 0x200,			// Two or more AND-connected terms
		WO_EQUIV = 0x400,       // Of the form A==B, both columns
		WO_NOOP = 0x800,		// This term does not restrict search space
		WO_ALL = 0xfff,			// Mask of all possible WO_* values
		WO_SINGLE = 0x0ff,      // Mask of all non-compound WO_* values
	};
	__device__ inline void operator|=(WO &a, int b) { a = (WO)(a | b); }
	__device__ inline WO operator|(WO a, WO b) { return (WO)((int)a | (int)b); }
	__device__ inline WO operator&(WO a, WO b) { return (WO)((int)a & (int)b); }

	enum TERM : uint8
	{
		TERM_DYNAMIC = 0x01,   // Need to call sqlite3ExprDelete(db, pExpr)
		TERM_VIRTUAL = 0x02,   // Added by the optimizer.  Do not code
		TERM_CODED = 0x04,   // This term is already coded
		TERM_COPIED = 0x08,   // Has a child
		TERM_ORINFO = 0x10,   // Need to free the WhereTerm.u.pOrInfo object
		TERM_ANDINFO = 0x20,   // Need to free the WhereTerm.u.pAndInfo obj
		TERM_OR_OK = 0x40,   // Used during OR-clause processing
#ifdef ENABLE_STAT3
		TERM_VNULL = 0x80,   // Manufactured x>NULL or x<=NULL term
#else
		TERM_VNULL = 0x00,   // Disabled if not using stat3
#endif
	};
	__device__ inline void operator|=(TERM &a, int b) { a = (TERM)(a | b); }
	__device__ inline void operator&=(TERM &a, int b) { a = (TERM)(a & b); }
	__device__ inline TERM operator|(TERM a, TERM b) { return (TERM)((int)a | (int)b); }

	struct WhereTerm
	{
		Expr *Expr;				// Pointer to the subexpression that is this term
		int Parent;				// Disable pWC->a[iParent] when this term disabled
		int LeftCursor;			// Cursor number of X in "X <op> <expr>"
		union
		{
			int LeftColumn;         // Column number of X in "X <op> <expr>"
			WhereOrInfo *OrInfo;	// Extra information if (eOperator & WO_OR)!=0
			WhereAndInfo *AndInfo;	// Extra information if (eOperator& WO_AND)!=0
		} u;
		WO EOperator;			// A WO_xx value describing <op>
		TERM WtFlags;           // TERM_xxx bit flags.  See below
		uint8 Childs;           // Number of children that must disable us
		WhereClause *WC;		// The clause this term is part of
		Bitmask PrereqRight;    // Bitmask of tables used by pExpr->pRight
		Bitmask PrereqAll;      // Bitmask of tables referenced by pExpr
	};

	struct WhereClause
	{
		Parse *Parse;				// The parser context
		WhereMaskSet *MaskSet;		// Mapping of table cursor numbers to bitmasks
		WhereClause *Outer;			// Outer conjunction
		uint8 OP;					// Split operator.  TK_AND or TK_OR
		uint16 WctrlFlags;			// Might include WHERE_AND_ONLY
		int Terms;					// Number of terms
		array_t<WhereTerm> Slots;	// Each a[] describes a term of the WHERE cluase 
#if defined(SMALL_STACK)
		WhereTerm Statics[1];		// Initial static space for a[]
#else
		WhereTerm Statics[8];		// Initial static space for a[]
#endif
	};

	struct WhereOrInfo
	{
		WhereClause WC;          // Decomposition into subterms
		Bitmask Indexable;       // Bitmask of all indexable tables in the clause
	};

	struct WhereAndInfo
	{
		WhereClause WC;          // The subexpression broken out
	};

	struct WhereMaskSet
	{
		int n;                        // Number of assigned cursor values
		int ix[BMS];                  // Cursor assigned to each bit
	};

	struct WhereCost
	{
		WherePlan Plan;		// The lookup strategy
		double Cost;		// Overall cost of pursuing this search strategy
		Bitmask Used;		// Bitmask of cursors used by this plan
	};

	const uint32 WHERE_ROWID_EQ = 0x00001000;		// rowid=EXPR or rowid IN (...)
	const uint32 WHERE_ROWID_RANGE = 0x00002000;		// rowid<EXPR and/or rowid>EXPR
	const uint32 WHERE_COLUMN_EQ = 0x00010000;		// x=EXPR or x IN (...) or x IS NULL
	const uint32 WHERE_COLUMN_RANGE = 0x00020000;	// x<EXPR and/or x>EXPR
	const uint32 WHERE_COLUMN_IN = 0x00040000;		// x IN (...)
	const uint32 WHERE_COLUMN_NULL = 0x00080000;		// x IS NULL
	const uint32 WHERE_INDEXED = 0x000f0000;			// Anything that uses an index
	const uint32 WHERE_NOT_FULLSCAN = 0x100f3000;	// Does not do a full table scan
	const uint32 WHERE_IN_ABLE = 0x080f1000;			// Able to support an IN operator
	const uint32 WHERE_TOP_LIMIT = 0x00100000;		// x<EXPR or x<=EXPR constraint
	const uint32 WHERE_BTM_LIMIT = 0x00200000;		// x>EXPR or x>=EXPR constraint
	const uint32 WHERE_BOTH_LIMIT = 0x00300000;		// Both x>EXPR and x<EXPR
	const uint32 WHERE_IDX_ONLY = 0x00400000;		// Use index only - omit table
	const uint32 WHERE_ORDERED = 0x00800000;			// Output will appear in correct order
	const uint32 WHERE_REVERSE = 0x01000000;			// Scan in reverse order */
	const uint32 WHERE_UNIQUE = 0x02000000;			// Selects no more than one row
	const uint32 WHERE_ALL_UNIQUE = 0x04000000;		// This and all prior have one row
	const uint32 WHERE_OB_UNIQUE = 0x00004000;		// Values in ORDER BY columns are 
	// different for every output row
	const uint32 WHERE_VIRTUALTABLE = 0x08000000;	// Use virtual-table processing
	const uint32 WHERE_MULTI_OR = 0x10000000;		// OR using multiple indices
	const uint32 WHERE_TEMP_INDEX = 0x20000000;		// Uses an ephemeral index
	const uint32 WHERE_DISTINCT_ = 0x40000000;		// Correct order for DISTINCT
	const uint32 WHERE_COVER_SCAN = 0x80000000;		// Full scan of a covering index

	struct WhereBestIdx
	{
		Parse *Parse;					// Parser context
		WhereClause *WC;				// The WHERE clause
		SrcList::SrcListItem *Src;      // The FROM clause term to search
		Bitmask NotReady;               // Mask of cursors not available
		Bitmask NotValid;               // Cursors not available for any purpose
		ExprList *OrderBy;				// The ORDER BY clause
		ExprList *Distinct;				// The select-list if query is DISTINCT
		IIndexInfo **IdxInfo;	// Index information passed to xBestIndex
		int i;							// Which loop is being coded
		int n;							// # of loops
		WhereLevel *Levels;             // Info about outer loops
		WhereCost Cost;					// Lowest cost query plan
	};

	__device__ static bool CompareCost(const WhereCost *probe, const WhereCost *baseline)
	{
		if (probe->Cost < baseline->Cost) return true;
		if (probe->Cost > baseline->Cost) return false;
		if (probe->Plan.OBSats > baseline->Plan.OBSats) return true;
		if (probe->Plan.Rows < baseline->Plan.Rows) return true;
		return false;
	}

	__device__ static void WhereClauseInit(WhereClause *wc, Parse *parse, WhereMaskSet *maskSet, uint16 wctrlFlags)
	{
		wc->Parse = parse;
		wc->MaskSet = maskSet;
		wc->Outer = nullptr;
		wc->Terms = 0;
		wc->Slots.length = _lengthof(wc->Statics);
		wc->Slots.data = wc->Statics;
		wc->WctrlFlags = wctrlFlags;
	}

	__device__ static void WhereClauseClear(WhereClause *);
	__device__ static void WhereOrInfoDelete(Context *ctx, WhereOrInfo *p)
	{
		WhereClauseClear(&p->WC);
		_tagfree(ctx, p);
	}
	__device__ static void WhereAndInfoDelete(Context *ctx, WhereAndInfo *p)
	{
		WhereClauseClear(&p->WC);
		_tagfree(ctx, p);
	}

	__device__ static void WhereClauseClear(WhereClause *wc)
	{
		Context *ctx = wc->Parse->Ctx;
		int i;
		WhereTerm *a;
		for (i = wc->Terms-1, a = wc->Slots.data; i >= 0; i--, a++)
		{
			if (a->WtFlags & TERM_DYNAMIC)
				Expr::Delete(ctx, a->Expr);
			if (a->WtFlags & TERM_ORINFO)
				WhereOrInfoDelete(ctx, a->u.OrInfo);
			else if (a->WtFlags & TERM_ANDINFO)
				WhereAndInfoDelete(ctx, a->u.AndInfo);
		}
		if (wc->Slots.data != wc->Statics)
			_tagfree(ctx, wc->Slots.data);
	}

	__device__ static int WhereClauseInsert(WhereClause *wc, Expr *p, TERM wtFlags)
	{
		ASSERTCOVERAGE(wtFlags & TERM_VIRTUAL);  // EV: R-00211-15100
		if (wc->Terms >= wc->Slots.length)
		{
			WhereTerm *old = wc->Slots.data;
			Context *ctx = wc->Parse->Ctx;
			wc->Slots.data = (WhereTerm *)_tagalloc(ctx, sizeof(wc->Slots[0])*wc->Slots.length*2);
			if (!wc->Slots.data)
			{
				if (wtFlags & TERM_DYNAMIC)
					Expr::Delete(ctx, p);
				wc->Slots.data = old;
				return 0;
			}
			_memcpy(wc->Slots.data, old, sizeof(wc->Slots[0])*wc->Terms);
			if (old != wc->Statics)
				_tagfree(ctx, old);
			wc->Slots.length = (int)_tagallocsize(ctx, wc->Slots.data)/sizeof(wc->Slots[0]);
		}
		int idx;
		WhereTerm *term = &wc->Slots[idx = wc->Terms++];
		term->Expr = p->SkipCollate();
		term->WtFlags = wtFlags;
		term->WC = wc;
		term->Parent = -1;
		return idx;
	}

	__device__ static void WhereSplit(WhereClause *wc, Expr *expr, uint8 op)
	{
		wc->OP = op;
		if (!expr) return;
		if (expr->OP != op)
			WhereClauseInsert(wc, expr, (TERM)0);
		else
		{
			WhereSplit(wc, expr->Left, op);
			WhereSplit(wc, expr->Right, op);
		}
	}

#define InitMaskSet(P)  _memset(P, 0, sizeof(*P))

	__device__ static Bitmask GetMask(WhereMaskSet *maskSet, int cursor)
	{
		_assert(maskSet->n <= (int)sizeof(Bitmask)*8);
		for (int i = 0; i < maskSet->n; i++)
			if (maskSet->ix[i] == cursor)
				return ((Bitmask)1) << i;
		return 0;
	}

	__device__ static void CreateMask(WhereMaskSet *maskSet, int cursor)
	{
		_assert(maskSet->n < _lengthof(maskSet->ix));
		maskSet->ix[maskSet->n++] = cursor;
	}

	__device__ static Bitmask ExprListTableUsage(WhereMaskSet *, ExprList *);
	__device__ static Bitmask ExprSelectTableUsage(WhereMaskSet *, Select *);
	__device__ static Bitmask ExprTableUsage(WhereMaskSet *maskSet, Expr *p)
	{
		Bitmask mask = 0;
		if (!p) return 0;
		if (p->OP == TK_COLUMN)
		{
			mask = GetMask(maskSet, p->TableId);
			return mask;
		}
		mask = ExprTableUsage(maskSet, p->Right);
		mask |= ExprTableUsage(maskSet, p->Left);
		if (ExprHasProperty(p, EP_xIsSelect))
			mask |= ExprSelectTableUsage(maskSet, p->x.Select);
		else
			mask |= ExprListTableUsage(maskSet, p->x.List);
		return mask;
	}
	__device__ static Bitmask ExprListTableUsage(WhereMaskSet *maskSet, ExprList *list)
	{
		Bitmask mask = 0;
		if (list)
			for(int i = 0; i < list->Exprs; i++)
				mask |= ExprTableUsage(maskSet, list->Ids[i].Expr);
		return mask;
	}
	__device__ static Bitmask ExprSelectTableUsage(WhereMaskSet *maskSet, Select *s)
	{
		Bitmask mask = 0;
		while (s)
		{
			mask |= ExprListTableUsage(maskSet, s->EList);
			mask |= ExprListTableUsage(maskSet, s->GroupBy);
			mask |= ExprListTableUsage(maskSet, s->OrderBy);
			mask |= ExprTableUsage(maskSet, s->Where);
			mask |= ExprTableUsage(maskSet, s->Having);
			SrcList *src = s->Src;
			if (_ALWAYS(src != nullptr))
			{
				for (int i = 0; i < src->Srcs; i++)
				{
					mask |= ExprSelectTableUsage(maskSet, src->Ids[i].Select);
					mask |= ExprTableUsage(maskSet, src->Ids[i].On);
				}
			}
			s = s->Prior;
		}
		return mask;
	}

	__device__ static bool AllowedOp(int op)
	{
		_assert(TK_GT > TK_EQ && TK_GT < TK_GE);
		_assert(TK_LT > TK_EQ && TK_LT < TK_GE);
		_assert(TK_LE > TK_EQ && TK_LE < TK_GE);
		_assert(TK_GE == TK_EQ+4);
		return (op == TK_IN || (op >= TK_EQ && op <= TK_GE) || op == TK_ISNULL);
	}

#define SWAP(TYPE,A,B) { TYPE t = A; A = B; B = t; }

	__device__ static void ExprCommute(Parse *parse, Expr *expr)
	{
		uint16 expRight = (expr->Right->Flags & EP_Collate);
		uint16 expLeft = (expr->Left->Flags & EP_Collate);
		_assert(AllowedOp(expr->OP) && expr->OP != TK_IN);
		if (expRight == expLeft)
		{
			// Either X and Y both have COLLATE operator or neither do
			if (expRight)
				expr->Right->Flags &= ~EP_Collate; // Both X and Y have COLLATE operators.  Make sure X is always used by clearing the EP_Collate flag from Y.
			else if (expr->Left->CollSeq(parse))
				expr->Left->Flags |= EP_Collate; // Neither X nor Y have COLLATE operators, but X has a non-default collating sequence.  So add the EP_Collate marker on X to cause it to be searched first.
		}
		SWAP(Expr *, expr->Right, expr->Left);
		if (expr->OP >= TK_GT)
		{
			_assert(TK_LT == TK_GT+2);
			_assert(TK_GE == TK_LE+2);
			_assert(TK_GT > TK_EQ);
			_assert(TK_GT < TK_LE);
			_assert(expr->OP >= TK_GT && expr->OP <= TK_GE);
			expr->OP = ((expr->OP - TK_GT)^2) + TK_GT;
		}
	}

	__device__ static WO OperatorMask(int op)
	{
		_assert(AllowedOp(op));
		WO c;
		if (op == TK_IN)
			c = WO_IN;
		else if (op == TK_ISNULL)
			c = WO_ISNULL;
		else
		{
			_assert((WO_EQ << (op - TK_EQ)) < 0x7fff);
			c = (WO)(WO_EQ << (op - TK_EQ));
		}
		_assert(op != TK_ISNULL || c == WO_ISNULL);
		_assert(op != TK_IN || c == WO_IN);
		_assert(op != TK_EQ || c == WO_EQ);
		_assert(op != TK_LT || c == WO_LT);
		_assert(op != TK_LE || c == WO_LE);
		_assert(op != TK_GT || c == WO_GT);
		_assert(op != TK_GE || c == WO_GE);
		return c;
	}

	static WhereTerm *FindTerm(WhereClause *wc, int cursor, int column, Bitmask notReady, WO op, Index *idx)
	{
		WhereTerm *result = nullptr; // The answer to return
		int origColumn = column;  // Original value of iColumn
		int equivsLength = 2; // Number of entires in aEquiv[]
		int equivId = 2; // Number of entries of aEquiv[] processed so far
		int equivs[22]; // iCur,iColumn and up to 10 other equivalents

		_assert(cursor >= 0);
		equivs[0] = cursor;
		equivs[1] = column;
		WhereClause *origWC = wc; // Original pWC value
		WhereTerm *term; // Term being examined as possible result
		int j, k;
		for (;;)
		{
			for (wc = origWC; wc; wc = wc->Outer)
			{
				for (term = wc->Slots.data, k = wc->Terms; k; k--, term++)
					if (term->LeftCursor == cursor && term->u.LeftColumn == column)
					{
						if ((term->PrereqRight & notReady) == 0 && (term->EOperator & op & WO_ALL) != 0)
						{
							if (origColumn >= 0 && idx && (term->EOperator & WO_ISNULL) == 0)
							{
								Expr *x = term->Expr;
								Parse *parse = wc->Parse;
								AFF idxaff = idx->Table->Cols[origColumn].Affinity;
								if (!x->ValidIndexAffinity(idxaff))
									continue;
								// Figure out the collation sequence required from an index for it to be useful for optimising expression pX. Store this
								// value in variable pColl.
								_assert(x->Left);
								CollSeq *coll = Expr::BinaryCompareCollSeq(parse, x->Left, x->Right);
								if (!coll) coll = parse->Ctx->DefaultColl;
								for (j = 0; idx->Columns[j] != origColumn; j++)
									if (_NEVER(j >= idx->Columns.length)) return nullptr;
								if (_strcmp(coll->Name, idx->CollNames[j]))
									continue;
							}
							if (term->PrereqRight == 0)
							{
								result = term;
								goto findTerm_success;
							}
							else if (!result)
								result = term;
						}
						if ((term->EOperator & WO_EQUIV) != 0 && equivsLength < _lengthof(equivs))
						{
							Expr *x = term->Expr->Right->SkipCollate();
							_assert(x->OP == TK_COLUMN);
							for (j = 0; j < equivsLength; j+=2)
								if (equivs[j] == x->TableId && equivs[j+1] == x->ColumnIdx) break;
							if (j == equivsLength)
							{
								equivs[j] = x->TableId;
								equivs[j+1] = x->ColumnIdx;
								equivsLength += 2;
							}
						}
					}
			}
			if (equivId >= equivsLength) break;
			cursor = equivs[equivId++];
			column = equivs[equivId++];
		}
findTerm_success:
		return result;
	}

	__device__ static void ExprAnalyze(SrcList *, WhereClause *, int); // PROTOTYPE
	__device__ static void ExprAnalyzeAll(SrcList *tabList, WhereClause *wc)
	{
		for (int i = wc->Terms-1; i >= 0; i--)
			ExprAnalyze(tabList, wc, i);
	}

#ifndef OMIT_LIKE_OPTIMIZATION
	__device__ static int IsLikeOrGlob(Parse *parse, Expr *expr, Expr **prefix, bool *isComplete, bool *noCase)
	{
		int cnt;                   // Number of non-wildcard prefix characters
		char wc[3];                // Wildcard characters
		Context *ctx = parse->Ctx;  // Database connection
		Mem *val = nullptr;

		if (!Command::Func::IsLikeFunction(ctx, expr, noCase, wc))
			return 0;
		ExprList *list = expr->x.List; // List of operands to the LIKE operator
		Expr *left = list->Ids[1].Expr; // Right and left size of LIKE operator
		if (left->OP != TK_COLUMN || left->Affinity() != AFF_TEXT || IsVirtual(left->Table))
			return 0; // IMP: R-02065-49465 The left-hand side of the LIKE or GLOB operator must be the name of an indexed column with TEXT affinity.
		_assert(left->ColumnIdx != -1); // Because IPK never has AFF_TEXT

		Expr *right = list->Ids[0].Expr; // Right and left size of LIKE operator
		int op = right->OP; // Opcode of pRight
		if (op == TK_REGISTER)
			op = right->OP2;
		const char *z = nullptr; // String on RHS of LIKE operator
		if (op == TK_VARIABLE)
		{
			Vdbe *reprepare = parse->Reprepare;
			int column = right->ColumnIdx;
			val = reprepare->GetValue(column, AFF_NONE);
			if (val && Vdbe::Value_Type(val) == TYPE_TEXT)
				z = (char *)Vdbe::Value_Type(val);
			parse->V->SetVarmask(column);
			_assert(right->OP == TK_VARIABLE || right->OP == TK_REGISTER);
		}
		else if (op == TK_STRING)
			z = right->u.Token;
		if (z)
		{
			cnt = 0;
			int c; // One character in z[]
			while ((c = z[cnt]) != 0 && c != wc[0] && c != wc[1] && c != wc[2]) cnt++;
			if (cnt != 0 && (uint8)z[cnt-1] != 255)
			{
				*isComplete = (c == wc[0] && z[cnt+1] == 0);
				Expr *prefix2 = Expr::Expr_(ctx, TK_STRING, z);
				if (prefix2) prefix2->u.Token[cnt] = 0;
				*prefix = prefix2;
				if (op == TK_VARIABLE)
				{
					Vdbe *v = parse->V;
					v->SetVarmask(right->ColumnIdx);
					if (*isComplete && right->u.Token[1])
					{
						// If the rhs of the LIKE expression is a variable, and the current value of the variable means there is no need to invoke the LIKE
						// function, then no OP_Variable will be added to the program. This causes problems for the sqlite3_bind_parameter_name()
						// API. To workaround them, add a dummy OP_Variable here.
						int r1 = Expr::GetTempReg(parse);
						Expr::CodeTarget(parse, right, r1);
						v->ChangeP3(v->CurrentAddr()-1, 0);
						Expr::ReleaseTempReg(parse, r1);
					}
				}
			}
			else
				z = nullptr;
		}
		Vdbe::ValueFree(val);
		return (z != nullptr);
	}
#endif


#ifndef OMIT_VIRTUALTABLE
	__device__ static bool IsMatchOfColumn(Expr *expr)
	{
		if (expr->OP != TK_FUNCTION || _strcmp(expr->u.Token, "match"))
			return false;
		ExprList *list = expr->x.List;
		if (list->Exprs != 2 || list->Ids[1].Expr->OP != TK_COLUMN)
			return false;
		return true;
	}
#endif

	__device__ static void TransferJoinMarkings(Expr *derived, Expr *base_)
	{
		derived->Flags |= base_->Flags & EP_FromJoin;
		derived->RightJoinTable = base_->RightJoinTable;
	}

#if !defined(OMIT_OR_OPTIMIZATION) && !defined(OMIT_SUBQUERY)
	__device__ static void ExprAnalyzeOrTerm(SrcList *src, WhereClause *wc, int idxTerm)
	{
		Parse *parse = wc->Parse; // Parser context
		Context *ctx = parse->Ctx; // Database connection
		WhereTerm *term = &wc->Slots[idxTerm]; // The term to be analyzed
		Expr *expr = term->Expr; // The expression of the term
		WhereMaskSet *maskSet = wc->MaskSet; // Table use masks

		// Break the OR clause into its separate subterms.  The subterms are stored in a WhereClause structure containing within the WhereOrInfo
		// object that is attached to the original OR clause term.
		_assert((term->WtFlags & (TERM_DYNAMIC|TERM_ORINFO|TERM_ANDINFO)) == 0);
		_assert(expr->OP == TK_OR);
		WhereOrInfo *orInfo; // Additional information associated with pTerm
		term->u.OrInfo = orInfo = (WhereOrInfo *)_tagalloc2(ctx, sizeof(*orInfo), true);
		if (!orInfo) return;
		term->WtFlags |= TERM_ORINFO;
		WhereClause *orWc = &orInfo->WC; // Breakup of pTerm into subterms
		WhereClauseInit(orWc, wc->Parse, maskSet, wc->WctrlFlags);
		WhereSplit(orWc, expr, TK_OR);
		ExprAnalyzeAll(src, orWc);
		if (ctx->MallocFailed) return;
		_assert(orWc->Terms >= 2);

		// Compute the set of tables that might satisfy cases 1 or 2.
		int i;
		WhereTerm *orTerm; // A Sub-term within the pOrWc
		Bitmask indexable = ~(Bitmask)0; // Tables that are indexable, satisfying case 2
		Bitmask chngToIN = ~(Bitmask)0; // Tables that might satisfy case 1
		for (i = orWc->Terms-1, orTerm = orWc->Slots; i >= 0 && indexable; i--, orTerm++)
		{
			if ((orTerm->EOperator & WO_SINGLE) == 0)
			{
				_assert((orTerm->WtFlags & (TERM_ANDINFO | TERM_ORINFO)) == 0);
				chngToIN = 0;
				WhereAndInfo *andInfo = (WhereAndInfo *)_tagalloc(ctx, sizeof(*andInfo));
				if (andInfo)
				{
					WhereTerm *andTerm;
					int j;
					Bitmask b = 0;
					orTerm->u.AndInfo = andInfo;
					orTerm->WtFlags |= TERM_ANDINFO;
					orTerm->EOperator = WO_AND;
					WhereClause *andWC = &andInfo->WC;
					WhereClauseInit(andWC, wc->Parse, maskSet, wc->WctrlFlags);
					WhereSplit(andWC, orTerm->Expr, TK_AND);
					ExprAnalyzeAll(src, andWC);
					andWC->Outer = wc;
					ASSERTCOVERAGE(ctx->MallocFailed);
					if (!ctx->MallocFailed)
					{
						for (j = 0, andTerm = andWC->Slots; j < andWC->Terms; j++, andTerm++)
						{
							_assert(andTerm->Expr);
							if (AllowedOp(andTerm->Expr->OP))
								b |= GetMask(maskSet, andTerm->LeftCursor);
						}
					}
					indexable &= b;
				}
			}
			else if (orTerm->WtFlags & TERM_COPIED) { } // Skip this term for now.  We revisit it when we process the corresponding TERM_VIRTUAL term
			else
			{
				Bitmask b = GetMask(maskSet, orTerm->LeftCursor);
				if (orTerm->WtFlags & TERM_VIRTUAL)
				{
					WhereTerm *other = &orWc->Slots[term->Parent];
					b |= GetMask(maskSet, other->LeftCursor);
				}
				indexable &= b;
				if ((orTerm->EOperator & WO_EQ) == 0)
					chngToIN = 0;
				else
					chngToIN &= b;
			}
		}

		// Record the set of tables that satisfy case 2.  The set might be empty.
		orInfo->Indexable = indexable;
		term->EOperator = (WO)(indexable == 0 ? 0 : WO_OR);

		// chngToIN holds a set of tables that *might* satisfy case 1.  But we have to do some additional checking to see if case 1 really
		// is satisfied.
		//
		// chngToIN will hold either 0, 1, or 2 bits.  The 0-bit case means that there is no possibility of transforming the OR clause into an
		// IN operator because one or more terms in the OR clause contain something other than == on a column in the single table.  The 1-bit
		// case means that every term of the OR clause is of the form "table.column=expr" for some single table.  The one bit that is set
		// will correspond to the common table.  We still need to check to make sure the same column is used on all terms.  The 2-bit case is when
		// the all terms are of the form "table1.column=table2.column".  It might be possible to form an IN operator with either table1.column
		// or table2.column as the LHS if either is common to every term of the OR clause.
		//
		// Note that terms of the form "table.column1=table.column2" (the same table on both sizes of the ==) cannot be optimized.
		if (chngToIN)
		{
			bool okToChngToIN = false; // True if the conversion to IN is valid
			int columnId = -1; // Column index on lhs of IN operator
			int cursorId = -1; // Table cursor common to all terms
			int j = 0;

			// Search for a table and column that appears on one side or the other of the == operator in every subterm.  That table and column
			// will be recorded in iCursor and iColumn.  There might not be any such table and column.  Set okToChngToIN if an appropriate table
			// and column is found but leave okToChngToIN false if not found.
			for (j = 0; j < 2 && !okToChngToIN; j++)
			{
				for (i = orWc->Terms-1, orTerm = orWc->Slots; i >= 0; i--, orTerm++)
				{
					_assert(orTerm->EOperator & WO_EQ);
					orTerm->WtFlags &= ~TERM_OR_OK;
					if (orTerm->LeftCursor == cursorId)
					{
						// This is the 2-bit case and we are on the second iteration and current term is from the first iteration.  So skip this term.
						_assert(j == 1);
						continue;
					}
					if ((chngToIN & GetMask(maskSet, orTerm->LeftCursor)) == 0)
					{
						// This term must be of the form t1.a==t2.b where t2 is in the chngToIN set but t1 is not.  This term will be either preceeded
						// or follwed by an inverted copy (t2.b==t1.a).  Skip this term and use its inversion.
						ASSERTCOVERAGE(orTerm->WtFlags & TERM_COPIED);
						ASSERTCOVERAGE(orTerm->WtFlags & TERM_VIRTUAL);
						_assert(orTerm->WtFlags & (TERM_COPIED|TERM_VIRTUAL));
						continue;
					}
					columnId = orTerm->u.LeftColumn;
					cursorId = orTerm->LeftCursor;
					break;
				}
				if (i < 0)
				{
					// No candidate table+column was found.  This can only occur on the second iteration
					_assert(j == 1);
					_assert(_ispoweroftwo(chngToIN));
					_assert(chngToIN == GetMask(maskSet, cursorId));
					break;
				}
				ASSERTCOVERAGE(j == 1);

				// We have found a candidate table and column.  Check to see if that table and column is common to every term in the OR clause
				okToChngToIN = true;
				for (; i >= 0 && okToChngToIN; i--, orTerm++)
				{
					_assert(orTerm->EOperator & WO_EQ);
					if(orTerm->LeftCursor != cursorId)
						orTerm->WtFlags &= ~TERM_OR_OK;
					else if (orTerm->u.LeftColumn != columnId)
						okToChngToIN = false;
					else
					{
						// If the right-hand side is also a column, then the affinities of both right and left sides must be such that no type
						// conversions are required on the right.  (Ticket #2249)
						AFF affRight = orTerm->Expr->Right->Affinity();
						AFF affLeft = orTerm->Expr->Left->Affinity();
						if (affRight != 0 && affRight != affLeft)
							okToChngToIN = false;
						else
							orTerm->WtFlags |= TERM_OR_OK;
					}
				}
			}

			// At this point, okToChngToIN is true if original pTerm satisfies
			// case 1.  In that case, construct a new virtual term that is pTerm converted into an IN operator. EV: R-00211-15100
			if (okToChngToIN)
			{
				Expr *dup; // A transient duplicate expression
				ExprList *list = nullptr; // The RHS of the IN operator
				Expr *left = nullptr; // The LHS of the IN operator
				for (i = orWc->Terms-1, orTerm = orWc->Slots; i >= 0; i--, orTerm++)
				{
					if ((orTerm->WtFlags & TERM_OR_OK) == 0) continue;
					_assert(orTerm->EOperator & WO_EQ);
					_assert(orTerm->LeftCursor == cursorId);
					_assert(orTerm->u.LeftColumn == columnId);
					dup = Expr::Dup(ctx, orTerm->Expr->Right, 0);
					list = Expr::ListAppend(wc->Parse, list, dup);
					left = orTerm->Expr->Left;
				}
				_assert(left);
				dup = Expr::Dup(ctx, left, 0);
				Expr *newExpr = Expr::PExpr_(parse, TK_IN, dup, 0, 0); // The complete IN operator
				if (newExpr)
				{
					TransferJoinMarkings(newExpr, expr);
					_assert(!ExprHasProperty(newExpr, EP_xIsSelect));
					newExpr->x.List = list;
					int idxNew = WhereClauseInsert(wc, newExpr, TERM_VIRTUAL|TERM_DYNAMIC);
					ASSERTCOVERAGE(idxNew == 0);
					ExprAnalyze(src, wc, idxNew);
					term = &wc->Slots[idxTerm];
					wc->Slots[idxNew].Parent = idxTerm;
					term->Childs = 1;
				}
				else
					Expr::ListDelete(ctx, list);
				term->EOperator = WO_NOOP; // case 1 trumps case 2
			}
		}
	}
#endif

	__device__ static void ExprAnalyze(SrcList *src, WhereClause *wc, int idxTerm)
	{
		Parse *parse = wc->Parse; // Parsing context
		Context *ctx = parse->Ctx; // Database connection

		if (ctx->MallocFailed) return;
		WhereTerm *term = &wc->Slots[idxTerm]; // The term to be analyzed
		WhereMaskSet *maskSet = wc->MaskSet; // Set of table index masks
		Expr *expr = term->Expr; // The expression to be analyzed
		_assert(expr->OP != TK_AS && expr->OP != TK_COLLATE );
		Bitmask prereqLeft = ExprTableUsage(maskSet, expr->Left); // Prerequesites of the pExpr->pLeft
		int op = expr->OP; // Top-level operator.  pExpr->op
		if (op == TK_IN)
		{
			_assert(expr->Right == nullptr);
			term->PrereqRight = (ExprHasProperty(expr, EP_xIsSelect) ? ExprSelectTableUsage(maskSet, expr->x.Select) : ExprListTableUsage(maskSet, expr->x.List));
		}
		else if (op == TK_ISNULL)
			term->PrereqRight = 0;
		else
			term->PrereqRight = ExprTableUsage(maskSet, expr->Right);
		Bitmask prereqAll = ExprTableUsage(maskSet, expr); // Prerequesites of pExpr
		Bitmask extraRight = 0; // Extra dependencies on LEFT JOIN
		if (ExprHasProperty(expr, EP_FromJoin))
		{
			Bitmask x = GetMask(maskSet, expr->RightJoinTable);
			prereqAll |= x;
			extraRight = x-1; // ON clause terms may not be used with an index on left table of a LEFT JOIN.  Ticket #3015
		}
		term->PrereqAll = prereqAll;
		term->LeftCursor = -1;
		term->Parent = -1;
		term->EOperator = (WO)0;
		if (AllowedOp(op))
		{
			Expr *left = expr->Left->SkipCollate();
			Expr *right = expr->Right->SkipCollate();
			WO opMask = ((term->PrereqRight & prereqLeft) == 0 ? WO_ALL : WO_EQUIV);
			if (left->OP == TK_COLUMN)
			{
				term->LeftCursor = left->TableId;
				term->u.LeftColumn = left->ColumnIdx;
				term->EOperator = OperatorMask(op) & opMask;
			}
			if (right && right->OP == TK_COLUMN)
			{
				WhereTerm *newTerm;
				Expr *dup;
				WO extraOp = (WO)0; // Extra bits for newTerm->eOperator
				if (term->LeftCursor >= 0)
				{
					dup = Expr::Dup(ctx, expr, 0);
					if (ctx->MallocFailed)
					{
						Expr::Delete(ctx, dup);
						return;
					}
					int idxNew = WhereClauseInsert(wc, dup, TERM_VIRTUAL|TERM_DYNAMIC);
					if (idxNew == 0) return;
					newTerm = &wc->Slots[idxNew];
					newTerm->Parent = idxTerm;
					term = &wc->Slots[idxTerm];
					term->Childs = 1;
					term->WtFlags |= TERM_COPIED;
					if (expr->OP == TK_EQ && !ExprHasProperty(expr, EP_FromJoin) && CtxOptimizationEnabled(ctx, OPTFLAG_Transitive))
					{
						term->EOperator |= WO_EQUIV;
						extraOp = WO_EQUIV;
					}
				}
				else
				{
					dup = expr;
					newTerm = term;

				}
				ExprCommute(parse, dup);
				left = dup->Left->SkipCollate();
				newTerm->LeftCursor = left->TableId;
				newTerm->u.LeftColumn = left->ColumnIdx;
				ASSERTCOVERAGE((prereqLeft | extraRight) != prereqLeft);
				newTerm->PrereqRight = prereqLeft | extraRight;
				newTerm->PrereqAll = prereqAll;
				newTerm->EOperator = (OperatorMask(dup->OP) | extraOp) & opMask;
			}
		}
#ifndef OMIT_BETWEEN_OPTIMIZATION
		// If a term is the BETWEEN operator, create two new virtual terms that define the range that the BETWEEN implements.  For example:
		//      a BETWEEN b AND c
		//
		// is converted into:
		//      (a BETWEEN b AND c) AND (a>=b) AND (a<=c)
		//
		// The two new terms are added onto the end of the WhereClause object. The new terms are "dynamic" and are children of the original BETWEEN
		// term.  That means that if the BETWEEN term is coded, the children are skipped.  Or, if the children are satisfied by an index, the original
		// BETWEEN term is skipped.
		else if (expr->OP == TK_BETWEEN && wc->OP ==TK_AND)
		{
			ExprList *list = expr->x.List;
			static const uint8 ops[] = { TK_GE, TK_LE };
			_assert(list);
			_assert(list->Exprs == 2);
			for (int i = 0; i < 2; i++)
			{
				Expr *newExpr = Expr::PExpr_(parse, ops[i], Expr::Dup(ctx, expr->Left, 0), Expr::Dup(ctx, list->Ids[i].Expr, 0), nullptr);
				int idxNew = WhereClauseInsert(wc, newExpr, TERM_VIRTUAL|TERM_DYNAMIC);
				ASSERTCOVERAGE(idxNew == 0);
				ExprAnalyze(src, wc, idxNew);
				term = &wc->Slots[idxTerm];
				wc->Slots[idxNew].Parent = idxTerm;
			}
			term->Childs = 2;
		}
#endif
#if !defined(OMIT_OR_OPTIMIZATION) && !defined(OMIT_SUBQUERY)
		// Analyze a term that is composed of two or more subterms connected by an OR operator.
		else if (expr->OP ==TK_OR)
		{
			_assert(wc->OP == TK_AND);
			ExprAnalyzeOrTerm(src, wc, idxTerm);
			term = &wc->Slots[idxTerm];
		}
#endif
#ifndef OMIT_LIKE_OPTIMIZATION
		// Add constraints to reduce the search space on a LIKE or GLOB operator.
		//
		// A like pattern of the form "x LIKE 'abc%'" is changed into constraints
		//          x>='abc' AND x<'abd' AND x LIKE 'abc%'
		// The last character of the prefix "abc" is incremented to form the termination condition "abd".
		Expr *str1 = nullptr; // RHS of LIKE/GLOB operator
		bool isComplete = false; // RHS of LIKE/GLOB ends with wildcard
		bool noCase = false; // LIKE/GLOB distinguishes case
		if (wc->OP == TK_AND && IsLikeOrGlob(parse, expr, &str1, &isComplete, &noCase))
		{
			Expr *left = expr->x.List->Ids[1].Expr; // LHS of LIKE/GLOB operator
			Expr *str2 = Expr::Dup(ctx, str1, 0); // Copy of pStr1 - RHS of LIKE/GLOB operator
			if (!ctx->MallocFailed)
			{
				uint8 *cRef = (uint8 *)&str2->u.Token[_strlen30(str2->u.Token)-1];
				uint8 c = *cRef; // Last character before the first wildcard
				if (noCase)
				{
					// The point is to increment the last character before the first wildcard.  But if we increment '@', that will push it into the
					// alphabetic range where case conversions will mess up the inequality.  To avoid this, make sure to also run the full
					// LIKE on all candidate expressions by clearing the isComplete flag
					if (c == 'A'-1) isComplete = false; // EV: R-64339-08207
					c = _tolower(c);
				}
				*cRef = c + 1;
			}
			Token sCollSeqName; // Name of collating sequence
			sCollSeqName.data = (noCase ? "NOCASE" : "BINARY");
			sCollSeqName.length = 6;
			Expr *newExpr1 = Expr::Dup(ctx, left, 0);
			newExpr1 = Expr::PExpr_(parse, TK_GE, Expr::AddCollateToken(parse, newExpr1, &sCollSeqName), str1, 0);
			int idxNew1 = WhereClauseInsert(wc, newExpr1, TERM_VIRTUAL|TERM_DYNAMIC);
			ASSERTCOVERAGE(idxNew1 == 0);
			ExprAnalyze(src, wc, idxNew1);
			Expr *newExpr2 = Expr::Dup(ctx, left, 0);
			newExpr2 = Expr::PExpr_(parse, TK_LT, Expr::AddCollateToken(parse, newExpr2, &sCollSeqName), str2, 0);
			int idxNew2 = WhereClauseInsert(wc, newExpr2, TERM_VIRTUAL|TERM_DYNAMIC);
			ASSERTCOVERAGE(idxNew2 == 0);
			ExprAnalyze(src, wc, idxNew2);
			term = &wc->Slots[idxTerm];
			if (isComplete)
			{
				wc->Slots[idxNew1].Parent = idxTerm;
				wc->Slots[idxNew2].Parent = idxTerm;
				term->Childs = 2;
			}
		}
#endif
#ifndef OMIT_VIRTUALTABLE
		// Add a WO_MATCH auxiliary term to the constraint set if the current expression is of the form:  column MATCH expr.
		// This information is used by the xBestIndex methods of virtual tables.  The native query optimizer does not attempt
		// to do anything with MATCH functions.
		if (IsMatchOfColumn(expr))
		{
			Expr *right = expr->x.List->Ids[0].Expr;
			Expr *left = expr->x.List->Ids[1].Expr;
			Bitmask prereqExpr = ExprTableUsage(maskSet, right);
			Bitmask prereqColumn = ExprTableUsage(maskSet, left);
			if ((prereqExpr & prereqColumn) == 0)
			{
				Expr *newExpr = Expr::PExpr_(parse, TK_MATCH, nullptr, Expr::Dup(ctx, right, 0), 0);
				int idxNew = WhereClauseInsert(wc, newExpr, TERM_VIRTUAL|TERM_DYNAMIC);
				ASSERTCOVERAGE(idxNew == 0);
				WhereTerm *newTerm = &wc->Slots[idxNew];
				newTerm->PrereqRight = prereqExpr;
				newTerm->LeftCursor = left->TableId;
				newTerm->u.LeftColumn = left->ColumnIdx;
				newTerm->EOperator = WO_MATCH;
				newTerm->Parent = idxTerm;
				term = &wc->Slots[idxTerm];
				term->Childs = 1;
				term->WtFlags |= TERM_COPIED;
				newTerm->PrereqAll = term->PrereqAll;
			}
		}
#endif
#ifdef ENABLE_STAT3
		// When sqlite_stat3 histogram data is available an operator of the form "x IS NOT NULL" can sometimes be evaluated more efficiently
		// as "x>NULL" if x is not an INTEGER PRIMARY KEY.  So construct a virtual term of that form.
		//
		// Note that the virtual term must be tagged with TERM_VNULL.  This TERM_VNULL tag will suppress the not-null check at the beginning
		// of the loop.  Without the TERM_VNULL flag, the not-null check at the start of the loop will prevent any results from being returned.
		if (expr->OP == TK_NOTNULL && expr->Left->OP == TK_COLUMN && expr->Left->ColumnIdx >= 0)
		{
			Expr *left = expr->Left;
			Expr *newExpr = Expr::PExpr_(parse, TK_GT, Expr::Dup(ctx, left, 0), Expr::PExpr_(parse, TK_NULL, 0, 0, 0), 0);
			int idxNew = WhereClauseInsert(wc, newExpr, (TERM)(TERM_VIRTUAL|TERM_DYNAMIC|TERM_VNULL));
			if (idxNew != 0)
			{
				WhereTerm *newTerm = &wc->Slots[idxNew];
				newTerm->PrereqRight = 0;
				newTerm->LeftCursor = left->TableId;
				newTerm->u.LeftColumn = left->ColumnIdx;
				newTerm->EOperator = WO_GT;
				newTerm->Parent = idxTerm;
				term = &wc->Slots[idxTerm];
				term->Childs = 1;
				term->WtFlags |= TERM_COPIED;
				newTerm->PrereqAll = term->PrereqAll;
			}
		}
#endif

		// Prevent ON clause terms of a LEFT JOIN from being used to drive an index for tables to the left of the join.
		term->PrereqRight |= extraRight;
	}

	__device__ static int FindIndexCol(Parse *parse, ExprList *list, int baseId, Index *index, int column)
	{
		const char *collName = index->CollNames[column];
		for (int i = 0; i < list->Exprs; i++)
		{
			Expr *expr = list->Ids[i].Expr->SkipCollate();
			if (expr->OP == TK_COLUMN && expr->ColumnIdx == index->Columns[column] && expr->TableId == baseId)
			{
				CollSeq *coll = list->Ids[i].Expr->CollSeq(parse);
				if (_ALWAYS(coll) && !_strcmp(coll->Name, collName))
					return i;
			}
		}
		return -1;
	}

	__device__ static bool IsDistinctIndex(Parse *parse, WhereClause *wc, Index *index, int baseId, ExprList *distinct, int eqCols)
	{
		_assert(distinct);
		if (!index->Name || distinct->Exprs >= BMS) return false;
		ASSERTCOVERAGE(distinct->Exprs == BMS-1);

		// Loop through all the expressions in the distinct list. If any of them are not simple column references, return early. Otherwise, test if the
		// WHERE clause contains a "col=X" clause. If it does, the expression can be ignored. If it does not, and the column does not belong to the
		// same table as index pIdx, return early. Finally, if there is no matching "col=X" expression and the column is on the same table as pIdx,
		// set the corresponding bit in variable mask.
		int i; 
		Bitmask mask = 0; // Mask of unaccounted for pDistinct exprs
		for (i = 0; i < distinct->Exprs; i++)
		{
			Expr *expr = distinct->Ids[i].Expr->SkipCollate();
			if (expr->OP != TK_COLUMN) return false;
			WhereTerm *term = FindTerm(wc, expr->TableId, expr->ColumnIdx, ~(Bitmask)0, WO_EQ, 0);
			if (term)
			{
				Expr *x = term->Expr;
				CollSeq *p1 = Expr::BinaryCompareCollSeq(parse, x->Left, x->Right);
				CollSeq *p2 = expr->CollSeq(parse);
				if (p1 == p2) continue;
			}
			if (expr->TableId != baseId) return false;
			mask |= (((Bitmask)1) << i);
		}
		for (i = eqCols; mask && i < index->Columns.length; i++)
		{
			int exprId = FindIndexCol(parse, distinct, baseId, index, i);
			if (exprId < 0) break;
			mask &= ~(((Bitmask)1) << exprId);
		}
		return (mask == 0);
	}

	__device__ static bool IsDistinctRedundant(Parse *parse, SrcList *list, WhereClause *wc, ExprList *distinct)
	{
		// If there is more than one table or sub-select in the FROM clause of this query, then it will not be possible to show that the DISTINCT 
		// clause is redundant.
		if (list->Srcs != 1) return false;
		int baseId = list->Ids[0].Cursor;
		Table *table = list->Ids[0].Table;

		// If any of the expressions is an IPK column on table iBase, then return true. Note: The (p->iTable==iBase) part of this test may be false if the
		// current SELECT is a correlated sub-query.
		int i;
		for (i = 0; i < distinct->Exprs; i++)
		{
			Expr *expr = distinct->Ids[i].Expr->SkipCollate();
			if (expr->OP == TK_COLUMN && expr->TableId == baseId && expr->ColumnIdx < 0) return true;
		}

		// Loop through all indices on the table, checking each to see if it makes the DISTINCT qualifier redundant. It does so if:
		//   1. The index is itself UNIQUE, and
		//   2. All of the columns in the index are either part of the pDistinct list, or else the WHERE clause contains a term of the form "col=X",
		//      where X is a constant value. The collation sequences of the comparison and select-list expressions must match those of the index.
		//   3. All of those index columns for which the WHERE clause does not contain a "col=X" term are subject to a NOT NULL constraint.
		for (Index *index = table->Index; index; index = index->Next)
		{
			if (index->OnError == OE_None) continue;
			for (i = 0; i < index->Columns.length; i++)
			{
				int column = index->Columns[i];
				if (!FindTerm(wc, baseId, column, ~(Bitmask)0, WO_EQ, index))
				{
					int indexColumn = FindIndexCol(parse, distinct, baseId, index, i);
					if (indexColumn < 0 || table->Cols[index->Columns[i]].NotNull == 0)
						break;
				}
			}
			if (i == index->Columns.length) // This index implies that the DISTINCT qualifier is redundant.
				return true;
		}
		return false;
	}

	__device__ static double EstLog(double n)
	{
		double logN = 1;
		double x = 10;
		while (n > x)
		{
			logN += 1;
			x *= 10;
		}
		return logN;
	}

#if !defined(OMIT_VIRTUALTABLE) && defined(_DEBUG)
	__device__ static void TRACE_IDX_INPUTS(IIndexInfo *p)
	{
		if (!WhereTrace) return;
		for (int i = 0; i < p->Constraints.length; i++)
			_fprintf(nullptr, "  constraint[%d]: col=%d termid=%d op=%d usabled=%d\n",
			i,
			p->Constraints[i].Column,
			p->Constraints[i].TermOffset,
			p->Constraints[i].OP,
			p->Constraints[i].Usable);
		for (int i = 0; i < p->OrderBys.length; i++)
			_fprintf(nullptr, "  orderby[%d]: col=%d desc=%d\n",
			i,
			p->OrderBys[i].Column,
			p->OrderBys[i].Desc);
	}
	__device__ static void TRACE_IDX_OUTPUTS(IIndexInfo *p)
	{
		if (!WhereTrace) return;
		for (int i = 0; i < p->Constraints.length; i++)
			_fprintf(nullptr, "  usage[%d]: argvIdx=%d omit=%d\n",
			i,
			p->ConstraintUsages[i].ArgvIndex,
			p->ConstraintUsages[i].Omit);
		_fprintf(nullptr, "  idxNum=%d\n", p->IdxNum);
		_fprintf(nullptr, "  idxStr=%s\n", p->IdxStr);
		_fprintf(nullptr, "  orderByConsumed=%d\n", p->OrderByConsumed);
		_fprintf(nullptr, "  estimatedCost=%g\n", p->EstimatedCost);
	}
#else
#define TRACE_IDX_INPUTS(A)
#define TRACE_IDX_OUTPUTS(A)
#endif

	__device__ static void BestIndex(WhereBestIdx *); // PROTOTYPE

	__device__ static void BestOrClauseIndex(WhereBestIdx *p)
	{
#ifndef OMIT_OR_OPTIMIZATION
		WhereClause *wc = p->WC; // The WHERE clause
		SrcList::SrcListItem *src = p->Src; // The FROM clause term to search
		const int cursor = src->Cursor; // The cursor of the table
		const Bitmask maskSrc = GetMask(wc->MaskSet, cursor); // Bitmask for pSrc

		// The OR-clause optimization is disallowed if the INDEXED BY or NOT INDEXED clauses are used or if the WHERE_AND_ONLY bit is set.
		if (src->NotIndexed || src->Index)
			return;
		if (wc->WctrlFlags & WHERE_AND_ONLY)
			return;

		// Search the WHERE clause terms for a usable WO_OR term.
		WhereTerm *const wcEnd = &wc->Slots[wc->Terms]; // End of pWC->a[]
		for (WhereTerm *term = wc->Slots; term < wcEnd; term++) // A single term of the WHERE clause
		{
			if ((term->EOperator & WO_OR) != 0 && ((term->PrereqAll & ~maskSrc) & p->NotReady) == 0 && (term->u.OrInfo->Indexable & maskSrc) != 0)
			{
				unsigned int flags = WHERE_MULTI_OR;
				double total = 0;
				double rows = 0;
				Bitmask used = 0;

				WhereBestIdx sBOI;
				sBOI = *p;
				sBOI.OrderBy = nullptr;
				sBOI.Distinct = nullptr;
				sBOI.IdxInfo = nullptr;

				WhereClause *const orWC = &term->u.OrInfo->WC;
				WhereTerm *const orWCEnd = &orWC->Slots[orWC->Terms];
				for (WhereTerm *orTerm = orWC->Slots; orTerm < orWCEnd; orTerm++)
				{
					WHERETRACE("... Multi-index OR testing for term %d of %d....\n", (orTerm - orWC->Slots), (term - wc->Slots));
					if ((orTerm->EOperator & WO_AND) != 0)
					{
						sBOI.WC = &orTerm->u.AndInfo->WC;
						BestIndex(&sBOI);
					}
					else if (orTerm->LeftCursor == cursor)
					{
						WhereClause tempWC;
						tempWC.Parse = wc->Parse;
						tempWC.MaskSet = wc->MaskSet;
						tempWC.Outer = wc;
						tempWC.OP = TK_AND;
						tempWC.Slots = orTerm;
						tempWC.WctrlFlags = 0;
						tempWC.Terms = 1;
						sBOI.WC = &tempWC;
						BestIndex(&sBOI);
					}
					else
						continue;
					total += sBOI.Cost.Cost;
					rows += sBOI.Cost.Plan.Rows;
					used |= sBOI.Cost.Used;
					if (total >= p->Cost.Cost) break;
				}

				// If there is an ORDER BY clause, increase the scan cost to account for the cost of the sort.
				if (p->OrderBy != nullptr)
				{
					WHERETRACE("... sorting increases OR cost %.9g to %.9g\n", total, total + rows * EstLog(rows));
					total += rows * EstLog(rows);
				}

				// If the cost of scanning using this OR term for optimization is less than the current cost stored in pCost, replace the contents of pCost.
				WHERETRACE("... multi-index OR cost=%.9g nrow=%.9g\n", total, rows);
				if (total < p->Cost.Cost)
				{
					p->Cost.Cost = total;
					p->Cost.Used = used;
					p->Cost.Plan.Rows = rows;
					p->Cost.Plan.OBSats = (p->i ? p->Levels[p->i-1].Plan.OBSats : 0);
					p->Cost.Plan.WsFlags = flags;
					p->Cost.Plan.u.Term = term;
				}
			}
		}
#endif
	}

#ifndef OMIT_AUTOMATIC_INDEX
	__device__ static bool TermCanDriveIndex(WhereTerm *term, SrcList::SrcListItem *src, Bitmask notReady)
	{
		if (term->LeftCursor != src->Cursor) return false;
		if ((term->EOperator & WO_EQ) == 0) return false;
		if ((term->PrereqRight & notReady)!=0 ) return false;
		AFF aff = src->Table->Cols[term->u.LeftColumn].Affinity;
		if (!term->Expr->ValidIndexAffinity(aff)) return false;
		return true;
	}
#endif

#ifndef OMIT_AUTOMATIC_INDEX
	__device__ static void BestAutomaticIndex(WhereBestIdx *p)
	{
		Parse *parse = p->Parse; // The parsing context
		WhereClause *wc = p->WC; // The WHERE clause
		SrcList::SrcListItem *src = p->Src; // The FROM clause term to search

		if (parse->QueryLoops <= (double)1) return; // There is no point in building an automatic index for a single scan
		if ((parse->Ctx->Flags & Context::FLAG_AutoIndex) == 0) return; // Automatic indices are disabled at run-time
		if ((p->Cost.Plan.WsFlags & WHERE_NOT_FULLSCAN) != 0 && (p->Cost.Plan.WsFlags & WHERE_COVER_SCAN) == 0) return; // We already have some kind of index in use for this query.
		if (src->ViaCoroutine) return; // Cannot index a co-routine
		if (src->NotIndexed) return; // The NOT INDEXED clause appears in the SQL.
		if (src->IsCorrelated) return; // The source is a correlated sub-query. No point in indexing it.

		_assert(parse->QueryLoops >= (double)1);
		Table *table = src->Table; // Table tht might be indexed
		double tableRows = table->RowEst; // Rows in the input table
		double logN = EstLog(tableRows); // log(nTableRow)
		double costTempIdx = 2*logN*(tableRows / parse->QueryLoops + 1); // per-query cost of the transient index
		if (costTempIdx >= p->Cost.Cost) return; // The cost of creating the transient table would be greater than doing the full table scan

		// Search for any equality comparison term
		WhereTerm *wcEnd = &wc->Slots[wc->Terms]; // End of pWC->a[]
		for (WhereTerm *term = wc->Slots; term < wcEnd; term++) // A single term of the WHERE clause
			if (TermCanDriveIndex(term, src, p->NotReady))
			{
				WHERETRACE("auto-index reduces cost from %.1f to %.1f\n", p->Cost.Cost, costTempIdx);
				p->Cost.Cost = costTempIdx;
				p->Cost.Plan.Rows = logN + 1;
				p->Cost.Plan.WsFlags = WHERE_TEMP_INDEX;
				p->Cost.Used = term->PrereqRight;
				break;
			}
	}
#else
#define BestAutomaticIndex(A) // no-op
#endif

#ifndef OMIT_AUTOMATIC_INDEX
	__device__ static void ConstructAutomaticIndex(Parse *parse, WhereClause *wc, SrcList::SrcListItem *src, Bitmask notReady, WhereLevel *level)
	{
		// Generate code to skip over the creation and initialization of the transient index on 2nd and subsequent iterations of the loop.
		Vdbe *v = parse->V; // Prepared statement under construction
		_assert(v);
		int addrInit = Expr::CodeOnce(parse); // Address of the initialization bypass jump

		// Count the number of columns that will be added to the index and used to match WHERE clause constraints
		int columns = 0; // Number of columns in the constructed index
		Table *table = src->Table; // The table being indexed
		WhereTerm *wcEnd = &wc->Slots[wc->Terms]; // End of pWC->a[]
		Bitmask idxCols = 0; // Bitmap of columns used for indexing
		WhereTerm *term; // A single term of the WHERE clause
		for (term = wc->Slots; term < wcEnd; term++)
		{
			if (TermCanDriveIndex(term, src, notReady))
			{
				int column = term->u.LeftColumn;
				Bitmask mask = (column >= BMS ? ((Bitmask)1)<<(BMS-1) : ((Bitmask)1)<<column);
				ASSERTCOVERAGE(column == BMS);
				ASSERTCOVERAGE(column == BMS-1);
				if ((idxCols & mask) == 0)
				{
					columns++;
					idxCols |= mask;
				}
			}
		}
		_assert(columns > 0);
		level->Plan.Eqs = columns;

		// Count the number of additional columns needed to create a covering index.  A "covering index" is an index that contains all
		// columns that are needed by the query.  With a covering index, the original table never needs to be accessed.  Automatic indices must
		// be a covering index because the index will not be updated if the original table changes and the index and table cannot both be used
		// if they go out of sync.
		Bitmask extraCols = src->ColUsed & (~idxCols | (((Bitmask)1)<<(BMS-1))); // Bitmap of additional columns
		int maxBitCol = (table->Cols.length >= BMS-1 ? BMS-1 : table->Cols.length); // Maximum column in pSrc->colUsed
		ASSERTCOVERAGE(table->Cols.length == BMS-1);
		ASSERTCOVERAGE(table->Cols.length == BMS-2);
		int i;
		for (i = 0; i < maxBitCol; i++)
			if (extraCols & (((Bitmask)1) << i)) columns++;
		if (src->ColUsed & (((Bitmask)1)<<(BMS-1)))
			columns += table->Cols.length - BMS + 1;
		level->Plan.WsFlags |= WHERE_COLUMN_EQ | WHERE_IDX_ONLY | WO_EQ;

		// Construct the Index object to describe this index
		int bytes = sizeof(Index); // Byte of memory needed for pIdx
		bytes += columns*sizeof(int); // Index.aiColumn
		bytes += columns*sizeof(char*); // Index.azColl
		bytes += columns; // Index.aSortOrder
		Index *index = (Index *)_tagalloc2(parse->Ctx, bytes, true); // Object describing the transient index
		if (!index) return;
		level->Plan.u.Index = index;
		index->CollNames = (char **)&index[1];
		index->Columns.data = (int *)&index->CollNames[columns];
		index->SortOrders = (SO *)&index->Columns[columns];
		index->Name = "auto-index";
		index->Columns.length = columns;
		index->Table = table;
		int n = 0; // Column counter
		idxCols = 0;
		for (term = wc->Slots; term < wcEnd; term++)
		{
			if (TermCanDriveIndex(term, src, notReady))
			{
				int column = term->u.LeftColumn;
				Bitmask mask = (column >= BMS ? ((Bitmask)1)<<(BMS-1) : ((Bitmask)1)<<column);
				if ((idxCols & mask) == 0)
				{
					Expr *x = term->Expr;
					idxCols |= mask;
					index->Columns[n] = term->u.LeftColumn;
					CollSeq *coll = Expr::BinaryCompareCollSeq(parse, x->Left, x->Right); // Collating sequence to on a column
					index->CollNames[n] = (_ALWAYS(coll) ? coll->Name : "BINARY");
					n++;
				}
			}
		}
		_assert(level->Plan.Eqs == (uint32)n);

		// Add additional columns needed to make the automatic index into a covering index
		for (i = 0; i < maxBitCol; i++)
		{
			if (extraCols & (((Bitmask)1)<<i))
			{
				index->Columns[n] = i;
				index->CollNames[n] = "BINARY";
				n++;
			}
		}
		if (src->ColUsed & (((Bitmask)1)<<(BMS-1)))
		{
			for (i = BMS-1; i < table->Cols.length; i++)
			{
				index->Columns[n] = i;
				index->CollNames[n] = "BINARY";
				n++;
			}
		}
		_assert(n == columns);

		// Create the automatic index
		KeyInfo *keyinfo = parse->IndexKeyinfo(index); // Key information for the index
		_assert(level->IdxCur >= 0);
		v->AddOp4(OP_OpenAutoindex, level->IdxCur, columns+1, 0, (char *)keyinfo, Vdbe::P4T_KEYINFO_HANDOFF);
		v->Comment("for %s", table->Name);

		// Fill the automatic index with content
		int addrTop = v->AddOp1(OP_Rewind, level->TabCur); // Top of the index fill loop
		int regRecord = Expr::GetTempReg(parse); // Register holding an index record
		Delete::GenerateIndexKey(parse, index, level->TabCur, regRecord, 1);
		v->AddOp2(OP_IdxInsert, level->IdxCur, regRecord);
		v->ChangeP5(Vdbe::OPFLAG_USESEEKRESULT);
		v->AddOp2(OP_Next, level->TabCur, addrTop+1);
		v->ChangeP5(Vdbe::STMTSTATUS_AUTOINDEX);
		v->JumpHere(addrTop);
		Expr::ReleaseTempReg(parse, regRecord);

		// Jump here when skipping the initialization
		v->JumpHere(addrInit);
	}
#endif

#ifndef OMIT_VIRTUALTABLE
	__device__ static IIndexInfo *AllocateIndexInfo(WhereBestIdx *p)
	{
		Parse *parse = p->Parse; 
		WhereClause *wc = p->WC;
		SrcList::SrcListItem *src = p->Src;
		ExprList *orderBy = p->OrderBy;
		WHERETRACE("Recomputing index info for %s...\n", src->Table->Name);
		// The direct assignment in the previous line is possible only because the WO_ and SQLITE_INDEX_CONSTRAINT_ codes are identical. The
		// following asserts verify this fact.
		_assert(WO_EQ == INDEX_CONSTRAINT_EQ);
		_assert(WO_LT == INDEX_CONSTRAINT_LT);
		_assert(WO_LE == INDEX_CONSTRAINT_LE);
		_assert(WO_GT == INDEX_CONSTRAINT_GT);
		_assert(WO_GE == INDEX_CONSTRAINT_GE);
		_assert(WO_MATCH == INDEX_CONSTRAINT_MATCH);

		// Count the number of possible WHERE clause constraints referring to this virtual table
		int i;
		int terms;
		WhereTerm *term;
		for (i = terms = 0, term = wc->Slots; i < wc->Terms; i++, term++)
		{
			if (term->LeftCursor != src->Cursor) continue;
			_assert(_ispoweroftwo(term->EOperator & ~WO_EQUIV));
			ASSERTCOVERAGE(term->EOperator & WO_IN);
			ASSERTCOVERAGE(term->EOperator & WO_ISNULL);
			if (term->EOperator & WO_ISNULL) continue;
			if (term->WtFlags & TERM_VNULL) continue;
			terms++;
		}

		// If the ORDER BY clause contains only columns in the current virtual table then allocate space for the aOrderBy part of
		// the sqlite3_index_info structure.
		int orderBys = 0;
		if (orderBy)
		{
			int n = orderBy->Exprs;
			for (i = 0; i < n; i++)
			{
				Expr *expr = orderBy->Ids[i].Expr;
				if (expr->OP != TK_COLUMN || expr->TableId != src->Cursor) break;
			}
			if (i == n)
				orderBys = n;
		}

		// Allocate the sqlite3_index_info structure
		IIndexInfo *idxInfo = (IIndexInfo *)_tagalloc2(parse->Ctx, sizeof(IIndexInfo) + (sizeof(IIndexInfo::Constraint) + sizeof(IIndexInfo::ConstraintUsage))*terms + sizeof(IIndexInfo::Orderby)*orderBys, true);
		if (!idxInfo)
		{
			parse->ErrorMsg("out of memory");
			return nullptr; // (double)0 In case of SQLITE_OMIT_FLOATING_POINT...
		}

		// Initialize the structure.  The sqlite3_index_info structure contains many fields that are declared "const" to prevent xBestIndex from
		// changing them.  We have to do some funky casting in order to initialize those fields.
		IIndexInfo::Constraint *idxCons = (IIndexInfo::Constraint *)&idxInfo[1];
		IIndexInfo::Orderby *idxOrderBy = (IIndexInfo::Orderby *)&idxCons[terms];
		IIndexInfo::ConstraintUsage *usage = (IIndexInfo::ConstraintUsage *)&idxOrderBy[orderBys];
		idxInfo->Constraints.length = terms;
		idxInfo->OrderBys.length = orderBys;
		idxInfo->Constraints.data = idxCons;
		idxInfo->OrderBys.data = idxOrderBy;
		idxInfo->ConstraintUsages.data = usage;
		int j;
		for (i = j = 0, term = wc->Slots; i < wc->Terms; i++, term++)
		{
			if (term->LeftCursor != src->Cursor) continue;
			_assert(_ispoweroftwo(term->EOperator & ~WO_EQUIV));
			ASSERTCOVERAGE(term->EOperator & WO_IN);
			ASSERTCOVERAGE(term->EOperator & WO_ISNULL);
			if (term->EOperator & WO_ISNULL) continue;
			if (term->WtFlags & TERM_VNULL) continue;
			idxCons[j].Column = term->u.LeftColumn;
			idxCons[j].TermOffset = i;
			WO op = (WO)term->EOperator & WO_ALL;
			if (op == WO_IN) op = WO_EQ;
			// The direct Debug.Assignment in the previous line is possible only because the WO_ and SQLITE_INDEX_CONSTRAINT_ codes are identical.
			idxCons[j].OP = (INDEX_CONSTRAINT)op;
			_assert(op & (WO_IN|WO_EQ|WO_LT|WO_LE|WO_GT|WO_GE|WO_MATCH));
			j++;
		}
		for (i = 0; i < orderBys; i++)
		{
			Expr *expr = orderBy->Ids[i].Expr;
			idxOrderBy[i].Column = expr->ColumnIdx;
			idxOrderBy[i].Desc = (orderBy->Ids[i].SortOrder != 0);
		}
		return idxInfo;
	}

	__device__ static int VTableBestIndex(Parse *parse, Table *table, IIndexInfo *p)
	{
		IVTable *vtable = VTable::GetVTable(parse->Ctx, table)->IVTable;
		WHERETRACE("xBestIndex for %s\n", table->Name);
		TRACE_IDX_INPUTS(p);
		RC rc = vtable->IModule->BestIndex(vtable, p);
		TRACE_IDX_OUTPUTS(p);
		if (rc != RC_OK)
		{
			if (rc == RC_NOMEM) parse->Ctx->MallocFailed = true;
			else if (!vtable->ErrMsg) parse->ErrorMsg("%s", Main::ErrStr(rc));
			else parse->ErrorMsg("%s", vtable->ErrMsg);
		}
		_free(vtable->ErrMsg);
		vtable->ErrMsg = nullptr;
		for (int i = 0; i < p->Constraints.length; i++)
			if (!p->Constraints[i].Usable && p->ConstraintUsages[i].ArgvIndex > 0)
				parse->ErrorMsg("table %s: xBestIndex returned an invalid plan", table->Name);
		return parse->Errs;
	}

	__device__ static void BestVirtualIndex(WhereBestIdx *p)
	{
		Parse *parse = p->Parse; // The parsing context
		WhereClause *wc = p->WC; // The WHERE clause
		SrcList::SrcListItem *src = p->Src; // The FROM clause term to search
		Table *table = src->Table;

		// Make sure wsFlags is initialized to some sane value. Otherwise, if the malloc in allocateIndexInfo() fails and this function returns leaving
		// wsFlags in an uninitialized state, the caller may behave unpredictably.
		_memset(&p->Cost, 0, sizeof(p->Cost));
		p->Cost.Plan.WsFlags = WHERE_VIRTUALTABLE;

		// If the sqlite3_index_info structure has not been previously allocated and initialized, then allocate and initialize it now.
		IIndexInfo *idxInfo = p->IdxInfo[0];
		if (!idxInfo)
			*p->IdxInfo = idxInfo = AllocateIndexInfo(p);
		// At this point, the sqlite3_index_info structure that pIdxInfo points to will have been initialized, either during the current invocation or
		// during some prior invocation.  Now we just have to customize the details of pIdxInfo for the current invocation and pass it to xBestIndex.
		if (!idxInfo) return;

		// The module name must be defined. Also, by this point there must be a pointer to an sqlite3_vtab structure. Otherwise
		// sqlite3ViewGetColumnNames() would have picked up the error. 
		_assert(table->ModuleArgs.data && table->ModuleArgs[0]);
		_assert(VTable::GetVTable(parse->Ctx, table));

		// Try once or twice.  On the first attempt, allow IN optimizations. If an IN optimization is accepted by the virtual table xBestIndex
		// method, but the  pInfo->aConstrainUsage.omit flag is not set, then the query will not work because it might allow duplicate rows in
		// output.  In that case, run the xBestIndex method a second time without the IN constraints.  Usually this loop only runs once.
		// The loop will exit using a "break" statement.
		SO sortOrder;
		int orderBys;
		int allowIN; // Allow IN optimizations
		for (allowIN = 1; allowIN != 0; allowIN--)
		{
			_assert(allowIN == 0 || allowIN == 1);
			// Set the aConstraint[].usable fields and initialize all output variables to zero.
			//
			// aConstraint[].usable is true for constraints where the right-hand side contains only references to tables to the left of the current
			// table.  In other words, if the constraint is of the form:
			//           column = expr
			// and we are evaluating a join, then the constraint on column is only valid if all tables referenced in expr occur to the left
			// of the table containing column.
			//
			// The aConstraints[] array contains entries for all constraints on the current table.  That way we only have to compute it once
			// even though we might try to pick the best index multiple times. For each attempt at picking an index, the order of tables in the
			// join might be different so we have to recompute the usable flag each time.
			int i, j;
			IIndexInfo::Constraint *idxCons = idxInfo->Constraints;
			IIndexInfo::ConstraintUsage *usage = idxInfo->ConstraintUsages;
			for (i = 0; i < idxInfo->Constraints.length; i++, idxCons++)
			{
				j = idxCons->TermOffset;
				WhereTerm *term = &wc->Slots[j];
				idxCons->Usable = ((term->PrereqRight & p->NotReady) == 0 && (allowIN || (term->EOperator & WO_IN) == 0));
			}
			_memset(usage, 0, sizeof(usage[0])*idxInfo->Constraints.length);
			if (idxInfo->NeedToFreeIdxStr)
				_free(idxInfo->IdxStr);
			idxInfo->IdxStr = nullptr;
			idxInfo->IdxNum = 0;
			idxInfo->NeedToFreeIdxStr = false;
			idxInfo->OrderByConsumed = false;
			// ((double)2) In case of SQLITE_OMIT_FLOATING_POINT...
			idxInfo->EstimatedCost = BIG_DOUBLE / ((double)2);
			orderBys = idxInfo->OrderBys.length;
			if (!p->OrderBy)
				idxInfo->OrderBys.length = 0;
			if (VTableBestIndex(parse, table, idxInfo))
				return;

			sortOrder = SO_ASC; // Sort order for IN clauses
			idxCons = idxInfo->Constraints;
			for (i = 0; i < idxInfo->Constraints.length; i++, idxCons++)
			{
				if (usage[i].ArgvIndex > 0)
				{
					j = idxCons->TermOffset;
					WhereTerm *term = &wc->Slots[j];
					p->Cost.Used |= term->PrereqRight;
					if ((term->EOperator & WO_IN) != 0)
					{
						// Do not attempt to use an IN constraint if the virtual table says that the equivalent EQ constraint cannot be safely omitted.
						// If we do attempt to use such a constraint, some rows might be repeated in the output.
						if (usage[i].Omit == 0) break;
						for (int k = 0; k < idxInfo->OrderBys.length; k++)
							if (idxInfo->OrderBys[k].Column == idxCons->Column)
							{
								sortOrder = (idxInfo->OrderBys[k].Desc ? SO_DESC : SO_ASC);
								break;
							}
					}
				}
			}
			if (i >= idxInfo->Constraints.length) break;
		}

		// If there is an ORDER BY clause, and the selected virtual table index does not satisfy it, increase the cost of the scan accordingly. This
		// matches the processing for non-virtual tables in bestBtreeIndex().
		double cost = idxInfo->EstimatedCost;
		if (p->OrderBy && !idxInfo->OrderByConsumed)
			cost += EstLog(cost)*cost;
		// The cost is not allowed to be larger than SQLITE_BIG_DBL (the inital value of lowestCost in this loop. If it is, then the (cost<lowestCost) test below will never be true.
		// Use "(double)2" instead of "2.0" in case OMIT_FLOATING_POINT is defined.
		p->Cost.Cost = ((BIG_DOUBLE/((double)2)) < cost ? (BIG_DOUBLE/((double)2)) : cost);
		p->Cost.Plan.u.VTableIndex = idxInfo;
		if (idxInfo->OrderByConsumed)
		{
			_assert(sortOrder == 0 || sortOrder == 1);
			p->Cost.Plan.WsFlags |= WHERE_ORDERED + sortOrder*WHERE_REVERSE;
			p->Cost.Plan.OBSats = orderBys;
		}
		else
			p->Cost.Plan.OBSats = (p->i ? p->Levels[p->i-1].Plan.OBSats : 0);
		p->Cost.Plan.Eqs = 0;
		idxInfo->OrderBys.length = orderBys;

		// Try to find a more efficient access pattern by using multiple indexes to optimize an OR expression within the WHERE clause. 
		BestOrClauseIndex(p);
	}
#endif

#ifdef ENABLE_STAT3
	__device__ static RC WhereKeyStats(Parse *parse, Index *index, Mem *val, bool roundUp, tRowcnt *stats)
	{
		_assert(index->Samples.length > 0);
		if (!val) return RC_ERROR;
		tRowcnt n = index->RowEsts[0];
		IndexSample *samples = index->Samples;
		TYPE type = Vdbe::Value_Type(val);

		int i;
		int64 v;
		double r;
		bool isEq = false;
		if (type == TYPE_INTEGER)
		{
			v = Vdbe::Value_Int64(val);
			r = (double)v;
			for (i = 0; i < index->Samples.length; i++)
			{
				if (samples[i].Type == TYPE_NULL) continue;
				if (samples[i].Type >= TYPE_TEXT) break;
				if (samples[i].Type == TYPE_INTEGER)
				{
					if (samples[i].u.I >= v)
					{
						isEq = (samples[i].u.I == v);
						break;
					}
				}
				else
				{
					_assert(samples[i].Type == TYPE_FLOAT);
					if (samples[i].u.R >= r)
					{
						isEq = (samples[i].u.R == r);
						break;
					}
				}
			}
		}
		else if (type == TYPE_FLOAT)
		{
			r = Vdbe::Value_Double(val);
			double rS;
			for (i = 0; i < index->Samples.length; i++)
			{
				if (samples[i].Type == TYPE_NULL) continue;
				if (samples[i].Type >= TYPE_TEXT) break;
				if (samples[i].Type == TYPE_FLOAT)
					rS = samples[i].u.R;
				else
					rS = (double)samples[i].u.I;
				if (rS >= r)
				{
					isEq = rS==r;
					break;
				}
			}
		}
		else if (type == TYPE_NULL)
		{
			i = 0;
			if (samples[0].Type == TYPE_NULL) isEq = true;
		}
		else
		{
			_assert(type == TYPE_TEXT || type == TYPE_BLOB);
			for (i = 0; i < index->Samples.length; i++)
				if (samples[i].Type == TYPE_TEXT || samples[i].Type == TYPE_BLOB)
					break;
			if (i < index->Samples.length)
			{      
				Context *ctx = parse->Ctx;
				CollSeq *coll;
				const uint8 *z;
				if (type == TYPE_BLOB)
				{
					z = (const uint8 *)Vdbe::Value_Blob(val);
					coll = ctx->DefaultColl;
					_assert(coll->Encode == TEXTENCODE_UTF8);
				}
				else
				{
					coll = Callback::GetCollSeq(parse, TEXTENCODE_UTF8, 0, *index->CollNames);
					if (!coll)
						return RC_ERROR;
					z = (const uint8 *)Vdbe::ValueText(val, coll->Encode);
					if (!z)
						return RC_NOMEM;
					_assert(z && coll && coll->Cmp);
				}
				n = Vdbe::ValueBytes(val, coll->Encode);

				for (; i < index->Samples.length; i++)
				{
					int c;
					TYPE sampleType = samples[i].Type;
					if (sampleType < type) continue;
					if (sampleType != type) break;
#ifndef OMIT_UTF16
					if (coll->Encode != TEXTENCODE_UTF8)
					{
						int sampleBytes;
						char *sampleZ = Vdbe::Utf8to16(ctx, coll->Encode, samples[i].u.Z, samples[i].Bytes, &sampleBytes);
						if (!sampleZ)
						{
							_assert(ctx->MallocFailed);
							return RC_NOMEM;
						}
						c = coll->Cmp(coll->User, sampleBytes, sampleZ, n, z);
						_tagfree(ctx, sampleZ);
					}
					else
#endif
					{
						c = coll->Cmp(coll->User, samples[i].Bytes, samples[i].u.Z, n, z);
					}
					if (c >= 0)
					{
						if (c == 0) isEq = true;
						break;
					}
				}
			}
		}

		// At this point, aSample[i] is the first sample that is greater than or equal to pVal.  Or if i==pIdx->nSample, then all samples are less
		// than pVal.  If aSample[i]==pVal, then isEq==1.
		if (isEq)
		{
			_assert(i < index->Samples.length);
			stats[0] = samples[i].Lts;
			stats[1] = samples[i].Eqs;
		}
		else
		{
			tRowcnt lowerId, upperId;
			if (i == 0)
			{
				lowerId = 0;
				upperId = samples[0].Lts;
			}
			else
			{
				upperId = (i >= index->Samples.length ? n : samples[i].Lts);
				lowerId = samples[i-1].Eqs + samples[i-1].Lts;
			}
			stats[1] = index->AvgEq;
			tRowcnt gapId = (lowerId >= upperId ? 0 : upperId - lowerId);
			gapId = (roundUp ? (gapId*2)/3 : gapId/3);
			stats[0] = lowerId + gapId;
		}
		return RC_OK;
	}

	static RC ValueFromExpr(Parse *parse, Expr *expr, AFF aff, Mem **val)
	{
		if (expr->OP == TK_VARIABLE || (expr->OP == TK_REGISTER && expr->OP2 == TK_VARIABLE))
		{
			int varId = expr->ColumnIdx;
			parse->V->SetVarmask(varId);
			*val = parse->Reprepare->GetValue(varId, aff);
			return RC_OK;
		}
		return Vdbe::ValueFromExpr(parse->Ctx, expr, TEXTENCODE_UTF8, aff, val);
	}
#endif

	__device__ static RC WhereRangeScanEst(Parse *parse, Index *p, int eqs, WhereTerm *lower, WhereTerm *upper, double *rangeDiv)
	{
		RC rc = RC_OK;
#ifdef ENABLE_STAT3
		if (eqs == 0 && p->Samples.length)
		{
			Mem *rangeVal;
			tRowcnt lowerId = 0;
			tRowcnt upperId = p->RowEsts[0];
			tRowcnt a[2];
			AFF aff = p->Table->Cols[p->Columns[0]].Affinity;
			if (lower)
			{
				Expr *expr = lower->Expr->Right;
				rc = ValueFromExpr(parse, expr, aff, &rangeVal);
				_assert((lower->EOperator & (WO_GT | WO_GE)) != 0);
				if (rc == RC_OK && WhereKeyStats(parse, p, rangeVal, false, a) == RC_OK)
				{
					lowerId = a[0];
					if ((lower->EOperator & WO_GT) != 0) lowerId += a[1];
				}
				Vdbe::ValueFree(rangeVal);
			}
			if (rc == RC_OK && upper)
			{
				Expr *expr = upper->Expr->Right;
				rc = ValueFromExpr(parse, expr, aff, &rangeVal);
				_assert((upper->EOperator & (WO_LT | WO_LE)) != 0);
				if (rc == RC_OK && WhereKeyStats(parse, p, rangeVal, true, a) == RC_OK)
				{
					upperId = a[0];
					if ((upper->EOperator & WO_LE) != 0) upperId += a[1];
				}
				Vdbe::ValueFree(rangeVal);
			}
			if (rc == RC_OK)
			{
				*rangeDiv = (upperId <= lowerId ? (double)p->RowEsts[0] : (double)p->RowEsts[0]/(double)(upperId - lowerId));
				WHERETRACE("range scan regions: %u..%u  div=%g\n", (uint32)lowerId, (uint32)upperId, *rangeDiv);
				return RC_OK;
			}
		}
#endif
		_assert(lower || upper);
		*rangeDiv = (double)1;
		if (lower && (lower->WtFlags & TERM_VNULL) == 0) *rangeDiv *= (double)4;
		if (upper) *rangeDiv *= (double)4;
		return rc;
	}

#ifdef ENABLE_STAT3
	__device__ static RC WhereEqualScanEst(Parse *parse, Index *p, Expr *expr, double *rows)
	{
		Mem *rhs = nullptr; // VALUE on right-hand side of pTerm
		RC rc; // Subfunction return code
		tRowcnt a[2]; // Statistics
		_assert(p->Samples);
		_assert(p->Samples.length > 0);
		AFF aff = p->Table->Cols[p->Columns[0]].Affinity; // Column affinity
		if (expr)
		{
			rc = ValueFromExpr(parse, expr, aff, &rhs);
			if (rc) goto cancel;
		}
		else
			rhs = Vdbe::ValueNew(parse->Ctx);
		if (!rhs) return RC_NOTFOUND;
		rc = WhereKeyStats(parse, p, rhs, 0, a);
		if (rc == RC_OK)
		{
			WHERETRACE("equality scan regions: %d\n", (int)a[1]);
			*rows = a[1];
		}

cancel:
		Vdbe::ValueFree(rhs);
		return rc;
	}

	__device__ static RC WhereInScanEst(Parse *parse, Index *p, ExprList *list, double *rows)
	{
		RC rc = RC_OK; // Subfunction return code
		double ests; // Number of rows for a single term
		double rowEsts = (double)0; // New estimate of the number of rows
		int i;
		_assert(p->Samples);
		for (i = 0; rc == RC_OK && i < list->Exprs; i++)
		{
			ests = p->RowEsts[0];
			rc = WhereEqualScanEst(parse, p, list->Ids[i].Expr, &ests);
			rowEsts += ests;
		}
		if (rc == RC_OK)
		{
			if (rowEsts > p->RowEsts[0]) rowEsts = p->RowEsts[0];
			*rows = rowEsts;
			WHERETRACE("IN row estimate: est=%g\n", rowEsts);
		}
		return rc;
	}

#endif

	__device__  static int IsOrderedColumn(WhereBestIdx *p, int table, int column)
	{
		WhereLevel *level = &p->Levels[p->i-1];
		Index *index;
		SO sortOrder;
		int i, j;
		for (i = p->i-1; i >= 0; i--, level--)
		{
			if (level->TabCur != table) continue;
			if ((level->Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
				return 1;
			_assert((level->Plan.WsFlags & WHERE_ORDERED) != 0);
			if ((index = level->Plan.u.Index) != nullptr)
			{
				if (column < 0)
				{
					sortOrder = SO_ASC;
					ASSERTCOVERAGE((level->Plan.WsFlags & WHERE_REVERSE) != 0);
				}
				else
				{
					int n = index->Columns.length;
					for (j = 0; j < n; j++)
						if (column == index->Columns[j]) break;
					if (j >= n) return 0;
					sortOrder = index->SortOrders[j];
					ASSERTCOVERAGE((level->Plan.WsFlags & WHERE_REVERSE) != 0);
				}
			}
			else
			{
				if (column != -1) return 0;
				sortOrder = SO_ASC;
				ASSERTCOVERAGE((level->Plan.WsFlags & WHERE_REVERSE) != 0);
			}
			if ((level->Plan.WsFlags & WHERE_REVERSE) != 0)
			{
				_assert(sortOrder == SO_ASC || sortOrder == SO_DESC);
				ASSERTCOVERAGE(sortOrder == SO_DESC);
				sortOrder = (SO)(1 - (int)sortOrder);
			}
			return sortOrder + 2;
		}
		return 0;
	}

	__device__ static int IsSortingIndex(WhereBestIdx *p, Index *index, int baseId, int *rev, bool *unique)
	{
		int i; // Number of pIdx terms used
		int sortOrder = 2; // 0: forward.  1: backward.  2: unknown

		Parse *parse = p->Parse; // Parser context
		Context *ctx = parse->Ctx; // Database connection
		int priorSats; // ORDER BY terms satisfied by outer loops
		bool outerObUnique; // Outer loops generate different values in every row for the ORDER BY columns
		if (p->i == 0)
		{
			priorSats = 0;
			outerObUnique = true;
		}
		else
		{
			uint32 wsFlags = p->Levels[p->i-1].Plan.WsFlags;
			priorSats = p->Levels[p->i-1].Plan.OBSats;
			if ((wsFlags & WHERE_ORDERED) == 0) // This loop cannot be ordered unless the next outer loop is also ordered
				return priorSats;
			if (CtxOptimizationDisabled(ctx, OPTFLAG_OrderByIdxJoin)) // Only look at the outer-most loop if the OrderByIdxJoin optimization is disabled
				return priorSats;
			ASSERTCOVERAGE(wsFlags & WHERE_OB_UNIQUE);
			ASSERTCOVERAGE(wsFlags & WHERE_ALL_UNIQUE);
			outerObUnique = ((wsFlags & (WHERE_OB_UNIQUE|WHERE_ALL_UNIQUE)) != 0);
		}
		ExprList *orderBy = p->OrderBy; // The ORDER BY clause
		_assert(orderBy);
		if (index->Unordered) // Hash indices (indicated by the "unordered" tag on sqlite_stat1) cannot be used for sorting
			return priorSats;
		int terms = orderBy->Exprs; // Number of ORDER BY terms
		bool uniqueNotNull = (index->OnError != OE_None); // pIdx is UNIQUE with all terms are NOT NULL
		_assert(terms > 0);

		// Argument pIdx must either point to a 'real' named index structure, or an index structure allocated on the stack by bestBtreeIndex() to
		// represent the rowid index that is part of every table.
		_assert(index->Name || (index->Columns.length == 1 && index->Columns[0] == -1));

		// Match terms of the ORDER BY clause against columns of the index.
		//
		// Note that indices have pIdx->nColumn regular columns plus one additional column containing the rowid.  The rowid column
		// of the index is also allowed to match against the ORDER BY clause.
		int j = priorSats; // Number of ORDER BY terms satisfied
		ExprList::ExprListItem *obItem;// A term of the ORDER BY clause
		Table *table = index->Table; // Table that owns index pIdx
		bool seenRowid = false; // True if an ORDER BY rowid term is seen
		for (i = 0, obItem = &orderBy->Ids[j]; j < terms && i <= index->Columns.length; i++)
		{
			// If the next term of the ORDER BY clause refers to anything other than a column in the "base" table, then this index will not be of any
			// further use in handling the ORDER BY.
			Expr *obExpr = obItem->Expr->SkipCollate(); // The expression of the ORDER BY pOBItem
			if (obExpr->OP != TK_COLUMN || obExpr->TableId != baseId)
				break;

			// Find column number and collating sequence for the next entry in the index
			int column; // The i-th column of the index.  -1 for rowid
			SO sortOrder2; // 1 for DESC, 0 for ASC on the i-th index term
			const char *collName; // Name of collating sequence for i-th index term
			if (index->Name &&  i < index->Columns.length)
			{
				column = index->Columns[i];
				if (column == index->Table->PKey)
					column = -1;
				sortOrder2 = index->SortOrders[i];
				collName = index->CollNames[i];
				_assert(collName != nullptr);
			}
			else
			{
				column = -1;
				sortOrder2 = (SO)0;
				collName = nullptr;
			}

			// Check to see if the column number and collating sequence of the index match the column number and collating sequence of the ORDER BY
			// clause entry.  Set isMatch to 1 if they both match.
			bool isMatch; // ORDER BY term matches the index term
			if (obExpr->ColumnIdx == column)
			{
				if (collName)
				{
					CollSeq *coll = obItem->Expr->CollSeq(parse); // The collating sequence of pOBExpr
					if (!coll) coll = ctx->DefaultColl;
					isMatch = !_strcmp(coll->Name, collName);
				}
				else isMatch = true;
			}
			else isMatch = false;

			// termSortOrder is 0 or 1 for whether or not the access loop should run forward or backwards (respectively) in order to satisfy this 
			// term of the ORDER BY clause.
			_assert(obItem->SortOrder==0 || obItem->SortOrder == 1);
			_assert(sortOrder2 == 0 || sortOrder2 == 1);
			SO termSortOrder = (SO)(sortOrder2 ^ obItem->SortOrder); // Sort order for this term

			// If X is the column in the index and ORDER BY clause, check to see if there are any X= or X IS NULL constraints in the WHERE clause.
			int isEq; // Subject to an == or IS NULL constraint
			WhereTerm *constraint = FindTerm(p->WC, baseId, column, p->NotReady, WO_EQ|WO_ISNULL|WO_IN, index); // A constraint in the WHERE clause
			if (!constraint) isEq = 0;
			else if ((constraint->EOperator & WO_IN) != 0) isEq = 0;
			else if ((constraint->EOperator & WO_ISNULL) != 0) { uniqueNotNull = false; isEq = 1; } // "X IS NULL" means X has only a single value
			else if (constraint->PrereqRight == 0) isEq = 1;  // Constraint "X=constant" means X has only a single value
			else
			{
				Expr *right = constraint->Expr->Right;
				if (right->OP == TK_COLUMN)
				{
					WHERETRACE("       .. isOrderedColumn(tab=%d,col=%d)", right->TableId, right->ColumnIdx);
					isEq = IsOrderedColumn(p, right->TableId, right->ColumnIdx);
					WHERETRACE(" -> isEq=%d\n", isEq);
					// If the constraint is of the form X=Y where Y is an ordered value in an outer loop, then make sure the sort order of Y matches the
					// sort order required for X.
					if (isMatch && isEq >= 2 && isEq != obItem->SortOrder + 2)
					{
						ASSERTCOVERAGE(isEq == 2);
						ASSERTCOVERAGE(isEq == 3);
						break;
					}
				}
				else isEq = 0; // "X=expr" places no ordering constraints on X
			}
			if (!isMatch)
			{
				if (isEq == 0) break;
				else continue;
			}
			else if (isEq != 1)
			{
				if (sortOrder == 2) sortOrder = termSortOrder;
				else if (termSortOrder != sortOrder) break;
			}
			j++;
			obItem++;
			if (column < 0)
			{
				seenRowid = true;
				break;
			}
			else if (table->Cols[column].NotNull == 0 && isEq != 1)
			{
				ASSERTCOVERAGE(isEq == 0);
				ASSERTCOVERAGE(isEq == 2);
				ASSERTCOVERAGE(isEq == 3);
				uniqueNotNull = false;
			}
		}
		if (seenRowid)
			uniqueNotNull = true;
		else if (!uniqueNotNull || i < index->Columns.length)
			uniqueNotNull = false;

		// If we have not found at least one ORDER BY term that matches the index, then show no progress.
		if (obItem == &orderBy->Ids[priorSats]) return priorSats;

		// Either the outer queries must generate rows where there are no two rows with the same values in all ORDER BY columns, or else this
		// loop must generate just a single row of output.  Example:  Suppose the outer loops generate A=1 and A=1, and this loop generates B=3
		// and B=4.  Then without the following test, ORDER BY A,B would generate the wrong order output: 1,3 1,4 1,3 1,4
		if (!outerObUnique && !uniqueNotNull) return priorSats;
		*unique = uniqueNotNull;

		// Return the necessary scan order back to the caller
		*rev = sortOrder & 1;

		// If there was an "ORDER BY rowid" term that matched, or it is only possible for a single row from this table to match, then skip over
		// any additional ORDER BY terms dealing with this table.
		if (uniqueNotNull)
		{
			// Advance j over additional ORDER BY terms associated with base
			WhereMaskSet *ms = p->WC->MaskSet;
			Bitmask m = ~GetMask(ms, baseId);
			while (j < terms && (ExprTableUsage(ms, orderBy->Ids[j].Expr) & m) == 0) j++;
		}
		return j;
	}

	__constant__ bool _UseCis = true;
	__device__ static void BestBtreeIndex(WhereBestIdx *p)
	{
		Parse *parse = p->Parse;  // The parsing context
		WhereClause *wc = p->WC;  // The WHERE clause
		SrcList::SrcListItem *src = p->Src; // The FROM clause term to search
		int cursor = src->Cursor;   // The cursor of the table to be accessed

		// Initialize the cost to a worst-case value
		_memset(&p->Cost, 0, sizeof(p->Cost));
		p->Cost.Cost = BIG_DOUBLE;

		// If the pSrc table is the right table of a LEFT JOIN then we may not use an index to satisfy IS NULL constraints on that table.  This is
		// because columns might end up being NULL if the table does not match - a circumstance which the index cannot help us discover.  Ticket #2177.
		Index *probe; // An index we are evaluating
		Index *index; // Copy of pProbe, or zero for IPK index
		int wsFlagMask; // Allowed flags in p->cost.plan.wsFlag
		WO eqTermMask; // Current mask of valid equality operators
		WO idxEqTermMask = (src->Jointype & JT_LEFT ? WO_EQ|WO_IN : WO_EQ|WO_IN|WO_ISNULL); // Index mask of valid equality operators
		if (src->Index)
		{
			// An INDEXED BY clause specifies a particular index to use
			index = probe = src->Index;
			wsFlagMask = ~(WHERE_ROWID_EQ|WHERE_ROWID_RANGE);
			eqTermMask = idxEqTermMask;
		}
		else
		{
			Index sPk; // A fake index object for the primary key
			tRowcnt rowEstPks[2]; // The aiRowEst[] value for the sPk index
			int columnPks = -1; // The aColumn[] value for the sPk index
			// There is no INDEXED BY clause.  Create a fake Index object in local variable sPk to represent the rowid primary key index.  Make this
			// fake index the first in a chain of Index objects with all of the real indices to follow */
			_memset(&sPk, 0, sizeof(Index));
			sPk.Columns.length = 1;
			sPk.Columns.data = &columnPks;
			sPk.RowEsts = rowEstPks;
			sPk.OnError = OE_Replace;
			sPk.Table = src->Table;
			rowEstPks[0] = src->Table->RowEst;
			rowEstPks[1] = 1;
			Index *first = src->Table->Index; // First of real indices on the table
			if (!src->NotIndexed)
				sPk.Next = first; // The real indices of the table are only considered if the NOT INDEXED qualifier is omitted from the FROM clause
			probe = &sPk;
			wsFlagMask = ~(WHERE_COLUMN_IN|WHERE_COLUMN_EQ|WHERE_COLUMN_NULL|WHERE_COLUMN_RANGE);
			eqTermMask = WO_EQ|WO_IN;
			index = nullptr;
		}

		int orderBys = (p->OrderBy ? p->OrderBy->Exprs : 0); // Number of ORDER BY terms
		int priorSats; // ORDER BY terms satisfied by outer loops
		bool sortInit; // Initializer for bSort in inner loop
		bool distInit; // Initializer for bDist in inner loop
		if (p->i)
		{
			priorSats = p->Levels[p->i-1].Plan.OBSats;
			sortInit = (priorSats < orderBys);
			distInit = false;
		}
		else
		{
			priorSats = 0;
			sortInit = (orderBys > 0);
			distInit = (p->Distinct != nullptr);
		}

		// Loop over all indices looking for the best one to use
		for (; probe; index = probe= probe->Next)
		{
			const tRowcnt *const rowEsts = probe->RowEsts;
			WhereCost pc; // Cost of using pProbe
			double log10N = (double)1; // base-10 logarithm of nRow (inexact)

			// The following variables are populated based on the properties of index being evaluated. They are then used to determine the expected
			// cost and number of rows returned.
			//
			//  pc.plan.nEq: 
			//    Number of equality terms that can be implemented using the index. In other words, the number of initial fields in the index that
			//    are used in == or IN or NOT NULL constraints of the WHERE clause.
			//
			//  nInMul:  
			//    The "in-multiplier". This is an estimate of how many seek operations SQLite must perform on the index in question. For example, if the 
			//    WHERE clause is:
			//      WHERE a IN (1, 2, 3) AND b IN (4, 5, 6)
			//
			//    SQLite must perform 9 lookups on an index on (a, b), so nInMul is set to 9. Given the same schema and either of the following WHERE 
			//    clauses:
			//      WHERE a =  1
			//      WHERE a >= 2
			//
			//    nInMul is set to 1.
			//
			//    If there exists a WHERE term of the form "x IN (SELECT ...)", then the sub-select is assumed to return 25 rows for the purposes of 
			//    determining nInMul.
			//
			//  bInEst:  
			//    Set to true if there was at least one "x IN (SELECT ...)" term used in determining the value of nInMul.  Note that the RHS of the
			//    IN operator must be a SELECT, not a value list, for this variable to be true.
			//
			//  rangeDiv:
			//    An estimate of a divisor by which to reduce the search space due to inequality constraints.  In the absence of sqlite_stat3 ANALYZE
			//    data, a single inequality reduces the search space to 1/4rd its original size (rangeDiv==4).  Two inequalities reduce the search
			//    space to 1/16th of its original size (rangeDiv==16).
			//
			//  bSort:   
			//    Boolean. True if there is an ORDER BY clause that will require an external sort (i.e. scanning the index being evaluated will not 
			//    correctly order records).
			//
			//  bDist:
			//    Boolean. True if there is a DISTINCT clause that will require an external btree.
			//
			//  bLookup: 
			//    Boolean. True if a table lookup is required for each index entry visited.  In other words, true if this is not a covering index.
			//    This is always false for the rowid primary key index of a table. For other indexes, it is true unless all the columns of the table
			//    used by the SELECT statement are present in the index (such an index is sometimes described as a covering index).
			//    For example, given the index on (a, b), the second of the following two queries requires table b-tree lookups in order to find the value
			//    of column c, but the first does not because columns a and b are both available in the index.
			//             SELECT a, b    FROM tbl WHERE a = 1;
			//             SELECT a, b, c FROM tbl WHERE a = 1;
			bool inEst = false;				// True if "x IN (SELECT...)" seen
			int inMul = 1;					// Number of distinct equalities to lookup
			double rangeDiv = (double)1;	// Estimated reduction in search space
			int bounds = 0;					// Number of range constraints seen
			bool sort = sortInit;			// True if external sort required
			bool dist = distInit;			// True if index cannot help with DISTINCT
			bool lookup = false;			// True if not a covering index
#ifdef ENABLE_STAT3
			WhereTerm *firstTerm = nullptr; // First term matching the index
#endif

			WHERETRACE("   %s(%s):\n", src->Table->Name, (index ? index->Name : "ipk"));
			_memset(&pc, 0, sizeof(pc));
			pc.Plan.OBSats = priorSats;

			// Determine the values of pc.plan.nEq and nInMul
			WhereTerm *term; // A single term of the WHERE clause
			for (pc.Plan.Eqs = 0; pc.Plan.Eqs < probe->Columns.length; pc.Plan.Eqs++)
			{
				int j = probe->Columns[pc.Plan.Eqs];
				term = FindTerm(wc, cursor, j, p->NotReady, eqTermMask, index);
				if (!term) break;
				pc.Plan.WsFlags |= (WHERE_COLUMN_EQ|WHERE_ROWID_EQ);
				ASSERTCOVERAGE(term->WC != wc);
				if (term->EOperator & WO_IN)
				{
					Expr *expr = term->Expr;
					pc.Plan.WsFlags |= WHERE_COLUMN_IN;
					if (ExprHasProperty(expr, EP_xIsSelect))
					{
						// "x IN (SELECT ...)":  Assume the SELECT returns 25 rows
						inMul *= 25;
						inEst = true;
					}
					else if (_ALWAYS(expr->x.List && expr->x.List->Exprs))
						inMul *= expr->x.List->Exprs; // "x IN (value, value, ...)"
				}
				else if (term->EOperator & WO_ISNULL)
					pc.Plan.WsFlags |= WHERE_COLUMN_NULL;
#ifdef ENABLE_STAT3
				if (pc.Plan.Eqs == 0 && probe->Samples.data) firstTerm = term;
#endif
				pc.Used |= term->PrereqRight;
			}

			// If the index being considered is UNIQUE, and there is an equality constraint for all columns in the index, then this search will find
			// at most a single row. In this case set the WHERE_UNIQUE flag to indicate this to the caller.
			//
			// Otherwise, if the search may find more than one row, test to see if there is a range constraint on indexed column (pc.plan.nEq+1) that
			// can be optimized using the index.
			if (pc.Plan.Eqs == probe->Columns.length && probe->OnError != OE_None)
			{
				ASSERTCOVERAGE(pc.Plan.WsFlags & WHERE_COLUMN_IN);
				ASSERTCOVERAGE(pc.Plan.WsFlags & WHERE_COLUMN_NULL);
				if ((pc.Plan.WsFlags & (WHERE_COLUMN_IN|WHERE_COLUMN_NULL)) == 0)
				{
					pc.Plan.WsFlags |= WHERE_UNIQUE;
					if (p->i == 0 || (p->Levels[p->i-1].Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
						pc.Plan.WsFlags |= WHERE_ALL_UNIQUE;
				}
			}
			else if (!probe->Unordered)
			{
				int j = (pc.Plan.Eqs == probe->Columns.length ? -1 : probe->Columns[pc.Plan.Eqs]);
				if (FindTerm(wc, cursor, j, p->NotReady, WO_LT|WO_LE|WO_GT|WO_GE, index))
				{
					WhereTerm *top = FindTerm(wc, cursor, j, p->NotReady, WO_LT|WO_LE, index);
					WhereTerm *btm = FindTerm(wc, cursor, j, p->NotReady, WO_GT|WO_GE, index);
					WhereRangeScanEst(parse, probe, pc.Plan.Eqs, btm, top, &rangeDiv);
					if (top)
					{
						bounds = 1;
						pc.Plan.WsFlags |= WHERE_TOP_LIMIT;
						pc.Used |= top->PrereqRight;
						ASSERTCOVERAGE(top->WC != wc);
					}
					if (btm)
					{
						bounds++;
						pc.Plan.WsFlags |= WHERE_BTM_LIMIT;
						pc.Used |= btm->PrereqRight;
						ASSERTCOVERAGE(btm->WC != wc);
					}
					pc.Plan.WsFlags |= (WHERE_COLUMN_RANGE|WHERE_ROWID_RANGE);
				}
			}

			// If there is an ORDER BY clause and the index being considered will naturally scan rows in the required order, set the appropriate flags
			// in pc.plan.wsFlags. Otherwise, if there is an ORDER BY clause but the index will scan rows in a different order, set the bSort
			// variable.
			if (sort && (src->Jointype & JT_LEFT) == 0)
			{
				int rev = 2;
				bool obUnique = false;
				WHERETRACE("      --> before isSortIndex: nPriorSat=%d\n", priorSats);
				pc.Plan.OBSats = IsSortingIndex(p, probe, cursor, &rev, &obUnique);
				WHERETRACE("      --> after  isSortIndex: bRev=%d bObU=%d nOBSat=%d\n", rev, obUnique, pc.Plan.OBSats);
				if (priorSats < pc.Plan.OBSats || (pc.Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
				{
					pc.Plan.WsFlags |= WHERE_ORDERED;
					if (obUnique) pc.Plan.WsFlags |= WHERE_OB_UNIQUE;
				}
				if (orderBys == pc.Plan.OBSats)
				{
					sort = false;
					pc.Plan.WsFlags |= WHERE_ROWID_RANGE|WHERE_COLUMN_RANGE;
				}
				if (rev & 1) pc.Plan.WsFlags |= WHERE_REVERSE;
			}

			// If there is a DISTINCT qualifier and this index will scan rows in order of the DISTINCT expressions, clear bDist and set the appropriate
			// flags in pc.plan.wsFlags.
			if (dist && IsDistinctIndex(parse, wc, probe, cursor, p->Distinct, pc.Plan.Eqs) && (pc.Plan.WsFlags & WHERE_COLUMN_IN) == 0)
			{
				dist = false;
				pc.Plan.WsFlags |= WHERE_ROWID_RANGE|WHERE_COLUMN_RANGE|WHERE_DISTINCT_;
			}

			// If currently calculating the cost of using an index (not the IPK index), determine if all required column data may be obtained without 
			// using the main table (i.e. if the index is a covering index for this query). If it is, set the WHERE_IDX_ONLY flag in
			// pc.plan.wsFlags. Otherwise, set the bLookup variable to true.
			if (index)
			{
				Bitmask m = src->ColUsed;
				for (int j = 0; j < index->Columns.length; j++)
				{
					int x = index->Columns[j];
					if (x < BMS-1)
						m &= ~(((Bitmask)1)<<x);
				}
				if (m == 0)
					pc.Plan.WsFlags |= WHERE_IDX_ONLY;
				else
					lookup = true;
			}

			// Estimate the number of rows of output.  For an "x IN (SELECT...)" constraint, do not let the estimate exceed half the rows in the table.
			pc.Plan.Rows = (double)(rowEsts[pc.Plan.Eqs] * inMul);
			if (inEst && pc.Plan.Rows * 2 > rowEsts[0])
			{
				pc.Plan.Rows = rowEsts[0]/2;
				inMul = (int)(pc.Plan.Rows / rowEsts[pc.Plan.Eqs]);
			}

#ifdef ENABLE_STAT3
			// If the constraint is of the form x=VALUE or x IN (E1,E2,...) and we do not think that values of x are unique and if histogram
			// data is available for column x, then it might be possible to get a better estimate on the number of rows based on
			// VALUE and how common that value is according to the histogram.
			if (pc.Plan.Rows > (double)1 && pc.Plan.Eqs == 1 && firstTerm && rowEsts[1] > 1)
			{
				_assert((firstTerm->EOperator & (WO_EQ|WO_ISNULL|WO_IN)) != 0);
				if (firstTerm->EOperator & (WO_EQ|WO_ISNULL))
				{
					ASSERTCOVERAGE(firstTerm->EOperator & WO_EQ);
					ASSERTCOVERAGE(firstTerm->EOperator & WO_EQUIV);
					ASSERTCOVERAGE(firstTerm->EOperator & WO_ISNULL);
					WhereEqualScanEst(parse, probe, firstTerm->Expr->Right, &pc.Plan.Rows);
				}
				else if (!inEst)
				{
					_assert(firstTerm->EOperator & WO_IN);
					WhereInScanEst(parse, probe, firstTerm->Expr->x.List, &pc.Plan.Rows);
				}
			}
#endif

			// Adjust the number of output rows and downward to reflect rows that are excluded by range constraints.
			pc.Plan.Rows = pc.Plan.Rows / rangeDiv;
			if (pc.Plan.Rows < 1) pc.Plan.Rows = 1;

			// Experiments run on real SQLite databases show that the time needed to do a binary search to locate a row in a table or index is roughly
			// log10(N) times the time to move from one row to the next row within a table or index.  The actual times can vary, with the size of
			// records being an important factor.  Both moves and searches are slower with larger records, presumably because fewer records fit
			// on one page and hence more pages have to be fetched.
			//
			// The ANALYZE command and the sqlite_stat1 and sqlite_stat3 tables do not give us data on the relative sizes of table and index records.
			// So this computation assumes table records are about twice as big as index records
			if ((pc.Plan.WsFlags &~(WHERE_REVERSE|WHERE_ORDERED|WHERE_OB_UNIQUE)) == WHERE_IDX_ONLY &&
				(wc->WctrlFlags & WHERE_ONEPASS_DESIRED) == 0 && _UseCis && CtxOptimizationEnabled(parse->Ctx, OPTFLAG_CoverIdxScan))
			{
				// This index is not useful for indexing, but it is a covering index. A full-scan of the index might be a little faster than a full-scan
				// of the table, so give this case a cost slightly less than a table scan.
				pc.Cost = rowEsts[0]*3 + probe->Columns.length;
				pc.Plan.WsFlags |= WHERE_COVER_SCAN|WHERE_COLUMN_RANGE;
			}
			else if ((pc.Plan.WsFlags & WHERE_NOT_FULLSCAN) == 0)
			{
				// The cost of a full table scan is a number of move operations equal to the number of rows in the table.
				//
				// We add an additional 4x penalty to full table scans.  This causes the cost function to err on the side of choosing an index over
				// choosing a full scan.  This 4x full-scan penalty is an arguable decision and one which we expect to revisit in the future.  But
				// it seems to be working well enough at the moment.
				pc.Cost = rowEsts[0]*4;
				pc.Plan.WsFlags &= ~WHERE_IDX_ONLY;
				if (index)
				{
					pc.Plan.WsFlags &= ~WHERE_ORDERED;
					pc.Plan.OBSats = priorSats;
				}
			}
			else
			{
				log10N = EstLog(rowEsts[0]);
				pc.Cost = pc.Plan.Rows;
				if (index)
				{
					if (lookup)
					{
						// For an index lookup followed by a table lookup:
						//    nInMul index searches to find the start of each index range
						//  + nRow steps through the index
						//  + nRow table searches to lookup the table entry using the rowid
						pc.Cost += (inMul + pc.Plan.Rows)*log10N;
					}
					else
					{
						// For a covering index:
						//     nInMul index searches to find the initial entry 
						//   + nRow steps through the index
						pc.Cost += inMul*log10N;
					}
				}
				else
				{
					// For a rowid primary key lookup:
					//    nInMult table searches to find the initial entry for each range
					//  + nRow steps through the table
					pc.Cost += inMul*log10N;
				}
			}

			// Add in the estimated cost of sorting the result.  Actual experimental measurements of sorting performance in SQLite show that sorting time
			// adds C*N*log10(N) to the cost, where N is the number of rows to be sorted and C is a factor between 1.95 and 4.3.  We will split the
			// difference and select C of 3.0.
			if (sort)
			{
				double m = EstLog(pc.Plan.Rows*(orderBys - pc.Plan.OBSats)/orderBys);
				m *= (double)(pc.Plan.OBSats ? 2 : 3);
				pc.Cost += pc.Plan.Rows*m;
			}
			if (dist)
				pc.Cost += pc.Plan.Rows * EstLog(pc.Plan.Rows)*3;

			//// Cost of using this index has now been computed ////

			// If there are additional constraints on this table that cannot be used with the current index, but which might lower the number
			// of output rows, adjust the nRow value accordingly.  This only matters if the current index is the least costly, so do not bother
			// with this step if we already know this index will not be chosen. Also, never reduce the output row count below 2 using this step.
			//
			// It is critical that the notValid mask be used here instead of the notReady mask.  When computing an "optimal" index, the notReady
			// mask will only have one bit set - the bit for the current table. The notValid mask, on the other hand, always has all bits set for
			// tables that are not in outer loops.  If notReady is used here instead of notValid, then a optimal index that depends on inner joins loops
			// might be selected even when there exists an optimal index that has no such dependency.
			if (pc.Plan.Rows > 2 && pc.Cost <= p->Cost.Cost)
			{
				int k;
				int skipEqs = pc.Plan.Eqs; // Number of == constraints to skip
				int skipRanges = bounds; // Number of < constraints to skip
				Bitmask thisTab = GetMask(wc->MaskSet, cursor); // Bitmap for pSrc
				for (term = wc->Slots, k = wc->Terms; pc.Plan.Rows > 2 && k; k--, term++)
				{
					if (term->WtFlags & TERM_VIRTUAL) continue;
					if ((term->PrereqAll & p->NotValid) != thisTab) continue;
					if (term->EOperator & (WO_EQ|WO_IN|WO_ISNULL))
					{
						if (skipEqs) skipEqs--; // Ignore the first pc.plan.nEq equality matches since the index has already accounted for these
						else pc.Plan.Rows /= 10; // Assume each additional equality match reduces the result set size by a factor of 10
					}
					else if (term->EOperator & (WO_LT|WO_LE|WO_GT|WO_GE))
					{
						if (skipRanges) skipRanges--; // Ignore the first nSkipRange range constraints since the index has already accounted for these
						// Assume each additional range constraint reduces the result set size by a factor of 3.  Indexed range constraints reduce
						// the search space by a larger factor: 4.  We make indexed range more selective intentionally because of the subjective 
						// observation that indexed range constraints really are more selective in practice, on average.
						else pc.Plan.Rows /= 3;
					}
					else if ((term->EOperator & WO_NOOP) == 0)
						pc.Plan.Rows /= 2; // Any other expression lowers the output row count by half
				}
				if (pc.Plan.Rows < 2) pc.Plan.Rows = 2;
			}

			WHERETRACE(
				"      nEq=%d nInMul=%d rangeDiv=%d bSort=%d bLookup=%d wsFlags=0x%08x\n"
				"      notReady=0x%llx log10N=%.1f nRow=%.1f cost=%.1f\n"
				"      used=0x%llx nOBSat=%d\n",
				pc.Plan.Eqs, inMul, (int)rangeDiv, sort, lookup, pc.Plan.WsFlags,
				p->NotReady, log10N, pc.Plan.Rows, pc.Cost, pc.Used,
				pc.Plan.OBSats);

			// If this index is the best we have seen so far, then record this index and its cost in the p->cost structure.
			if ((!index || pc.Plan.WsFlags) && CompareCost(&pc, &p->Cost))
			{
				p->Cost = pc;
				p->Cost.Plan.WsFlags &= wsFlagMask;
				p->Cost.Plan.u.Index = index;
			}

			// If there was an INDEXED BY clause, then only that one index is considered.
			if (src->Index) break;

			// Reset masks for the next index in the loop
			wsFlagMask = ~(WHERE_ROWID_EQ|WHERE_ROWID_RANGE);
			eqTermMask = idxEqTermMask;
		}

		// If there is no ORDER BY clause and the SQLITE_ReverseOrder flag is set, then reverse the order that the index will be scanned
		// in. This is used for application testing, to help find cases where application behavior depends on the (undefined) order that
		// SQLite outputs rows in in the absence of an ORDER BY clause.
		if (!p->OrderBy && (parse->Ctx->Flags & Context::FLAG_ReverseOrder))
			p->Cost.Plan.WsFlags |= WHERE_REVERSE;

		_assert(p->OrderBy || (p->Cost.Plan.WsFlags & WHERE_ORDERED) == 0);
		_assert(!p->Cost.Plan.u.Index || (p->Cost.Plan.WsFlags & WHERE_ROWID_EQ) == 0);
		_assert(!src->Index || !p->Cost.Plan.u.Index || p->Cost.Plan.u.Index == src->Index);

		WHERETRACE("   best index is %s cost=%.1f\n", (p->Cost.Plan.u.Index ? p->Cost.Plan.u.Index->Name : "ipk"), p->Cost.Cost);

		BestOrClauseIndex(p);
		BestAutomaticIndex(p);
		p->Cost.Plan.WsFlags |= eqTermMask;
	}

	__device__ static void BestIndex(WhereBestIdx *p)
	{
#ifndef OMIT_VIRTUALTABLE
		if (IsVirtual(p->Src->Table))
		{
			IIndexInfo *indexInfo = nullptr;
			p->IdxInfo = &indexInfo;
			BestVirtualIndex(p);
			_assert(indexInfo || p->Parse->Ctx->MallocFailed);
			if (indexInfo && indexInfo->NeedToFreeIdxStr)
				_free(indexInfo->IdxStr);
			_tagfree(p->Parse->Ctx, indexInfo);
		}
		else
#endif
		{
			BestBtreeIndex(p);
		}
	}

	__device__ static void DisableTerm(WhereLevel *level, WhereTerm *term)
	{
		if (term && (term->WtFlags & TERM_CODED) == 0 && (level->LeftJoin == 0 || ExprHasProperty(term->Expr, EP_FromJoin)))
		{
			term->WtFlags |= TERM_CODED;
			if (term->Parent >= 0)
			{
				WhereTerm *other = &term->WC->Slots[term->Parent];
				if (--other->Childs == 0)
					DisableTerm(level, other);
			}
		}
	}

	__device__ static void CodeApplyAffinity(Parse *parse, int baseId, int n, char *affs)
	{
		Vdbe *v = parse->V;
		if (!affs)
		{
			_assert(parse->Ctx->MallocFailed);
			return;
		}
		_assert(v != nullptr);
		// Adjust base and n to skip over SQLITE_AFF_NONE entries at the beginning and end of the affinity string.
		while (n > 0 && affs[0] == AFF_NONE)
		{
			n--;
			baseId++;
			affs++;
		}
		while (n > 1 && affs[n-1] == AFF_NONE) n--;
		// Code the OP_Affinity opcode if there is anything left to do.
		if (n > 0)
		{
			v->AddOp2(OP_Affinity, baseId, n);
			v->ChangeP4(-1, (const char *)affs, n);
			Expr::CacheAffinityChange(parse, baseId, n);
		}
	}

	__device__ static int CodeEqualityTerm(Parse *parse, WhereTerm *term, WhereLevel *level, int eq, int targetId)
	{
		Expr *x = term->Expr;
		Vdbe *v = parse->V;
		int regId; // Register holding results
		_assert(targetId > 0);
		if (x->OP == TK_EQ)
			regId = Expr::CodeTarget(parse, x->Right, targetId);
		else if (x->OP == TK_ISNULL)
		{
			regId = targetId;
			v->AddOp2(OP_Null, 0, regId);
#ifndef OMIT_SUBQUERY
		}
		else
		{
			bool rev = ((level->Plan.WsFlags & WHERE_REVERSE) != 0);
			if ((level->Plan.WsFlags & WHERE_INDEXED) != 0 && level->Plan.u.Index->SortOrders[eq])
			{
				ASSERTCOVERAGE(eq == 0);
				ASSERTCOVERAGE(eq == level->Plan.u.Index->Columns.length-1);
				ASSERTCOVERAGE(eq > 0 && eq+1 < level->Plan.u.Index->Columns.length);
				ASSERTCOVERAGE(rev);
				rev = !rev;
			}
			_assert(x->OP == TK_IN);
			regId = targetId;
			IN_INDEX type = Expr::FindInIndex(parse, x, nullptr);
			if (type == IN_INDEX_INDEX_DESC)
			{
				ASSERTCOVERAGE(rev);
				rev = !rev;
			}
			int tableId = x->TableId;
			v->AddOp2(rev ? OP_Last : OP_Rewind, tableId, 0);
			_assert(level->Plan.WsFlags & WHERE_IN_ABLE);
			if (level->u.in.InLoopsLength == 0)
				level->AddrNxt = v->MakeLabel();
			level->u.in.InLoopsLength++;
			level->u.in.InLoops = (WhereLevel::InLoop *)_tagrealloc_or_free(parse->Ctx, level->u.in.InLoops, sizeof(level->u.in.InLoops[0])*level->u.in.InLoopsLength);
			WhereLevel::InLoop *in = level->u.in.InLoops;
			if (in)
			{
				in += level->u.in.InLoopsLength - 1;
				in->Cur = tableId;
				if (type == IN_INDEX_ROWID)
					in->AddrInTop = v->AddOp2(OP_Rowid, tableId, regId);
				else
					in->AddrInTop = v->AddOp3(OP_Column, tableId, 0, regId);
				in->EndLoopOp = (rev ? OP_Prev : OP_Next);
				v->AddOp1(OP_IsNull, regId);
			}
			else
				level->u.in.InLoopsLength = 0;
#endif
		}
		DisableTerm(level, term);
		return regId;
	}

	__device__ static int CodeAllEqualityTerms(Parse *parse, WhereLevel *level, WhereClause *wc, Bitmask notReady, int extraRegs, char **affsOut)
	{
		// This module is only called on query plans that use an index.
		_assert(level->Plan.WsFlags & WHERE_INDEXED);
		Index *index = level->Plan.u.Index; // The index being used for this loop

		// Figure out how many memory cells we will need then allocate them.
		int baseId = parse->Mems + 1; // Base register
		int regs = level->Plan.Eqs + extraRegs; // Number of registers to allocate
		parse->Mems += regs;

		Vdbe *v = parse->V; // The vm under construction
		char *affs = _tagstrdup(parse->Ctx, Insert::IndexAffinityStr(v, index)); // Affinity string to return
		if (!affs)
			parse->Ctx->MallocFailed = true;

		// Evaluate the equality constraints
		int eqs = level->Plan.Eqs; // The number of == or IN constraints to code
		_assert(index->Columns.length >= eqs);
		int cur = level->TabCur; // The cursor of the table
		for (int j = 0; j < eqs; j++)
		{
			int k = index->Columns[j];
			WhereTerm *term = FindTerm(wc, cur, k, notReady, (WO)level->Plan.WsFlags, index); // A single constraint term
			if (!term) break;
			// The following true for indices with redundant columns. Ex: CREATE INDEX i1 ON t1(a,b,a); SELECT * FROM t1 WHERE a=0 AND b=0;
			ASSERTCOVERAGE((term->WtFlags & TERM_CODED) != 0);
			ASSERTCOVERAGE(term->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
			int r1 = CodeEqualityTerm(parse, term, level, j, baseId+j);
			if (r1 != baseId+j)
				if (regs == 1)
				{
					Expr::ReleaseTempReg(parse, baseId);
					baseId = r1;
				}
				else
					v->AddOp2(OP_SCopy, r1, baseId+j);
			ASSERTCOVERAGE(term->EOperator & WO_ISNULL);
			ASSERTCOVERAGE(term->EOperator & WO_IN);
			if ((term->EOperator & (WO_ISNULL|WO_IN)) == 0)
			{
				Expr *right = term->Expr->Right;
				Expr::CodeIsNullJump(v, right, baseId+j, level->AddrBrk);
				if (affs)
					if (right->CompareAffinity((AFF)affs[j]) == AFF_NONE || right->NeedsNoAffinityChange((AFF)affs[j]))
						affs[j] = AFF_NONE;
			}
		}
		*affsOut = affs;
		return baseId;
	}

#ifndef OMIT_EXPLAIN

	__device__ static void ExplainAppendTerm(TextBuilder *b, int term, const char *columnName, const char *opName)
	{
		if (term) b->Append(" AND ", 5);
		b->Append(columnName, -1);
		b->Append(opName, 1);
		b->Append("?", 1);
	}

	__device__ static char *ExplainIndexRange(Context *ctx, WhereLevel *level, Table *table)
	{
		WherePlan *plan = &level->Plan;
		Index *index = plan->u.Index;
		int eqs = plan->Eqs;
		Column *cols = table->Cols;
		int *columns = index->Columns;
		if (eqs == 0 && (plan->WsFlags & (WHERE_BTM_LIMIT|WHERE_TOP_LIMIT)) == 0)
			return nullptr;
		TextBuilder b;
		TextBuilder::Init(&b, 0, 0, CORE_MAX_LENGTH);
		b.Tag = ctx;
		b.Append(" (", 2);
		int i;
		for (i = 0; i < eqs; i++)
			ExplainAppendTerm(&b, i, cols[columns[i]].Name, "=");
		int j = i;
		if (plan->WsFlags & WHERE_BTM_LIMIT)
			ExplainAppendTerm(&b, i++, (j == index->Columns.length ? "rowid" : cols[columns[j]].Name), ">");
		if (plan->WsFlags & WHERE_TOP_LIMIT)
			ExplainAppendTerm(&b, i, (j == index->Columns.length ? "rowid" : cols[columns[j]].Name), "<");
		b.Append(")", 1);
		return b.ToString();
	}

	__device__ static void ExplainOneScan(Parse *parse, SrcList *list,  WhereLevel *level, int levelId, int fromId, uint16 wctrlFlags)
	{
		if (parse->Explain == 2)
		{
			uint32 flags = level->Plan.WsFlags;
			SrcList::SrcListItem *item = &list->Ids[level->From];
			Vdbe *v = parse->V; // VM being constructed
			Context *ctx = parse->Ctx; // Database handle

			if ((flags&WHERE_MULTI_OR) || (wctrlFlags&WHERE_ONETABLE_ONLY)) return;
			bool isSearch = (level->Plan.Eqs > 0 || (flags & (WHERE_BTM_LIMIT|WHERE_TOP_LIMIT)) != 0 || (wctrlFlags&(WHERE_ORDERBY_MIN|WHERE_ORDERBY_MAX))); // True for a SEARCH. False for SCAN.

			char *b = _mtagprintf(ctx, "%s", isSearch?"SEARCH":"SCAN"); // Text to add to EQP output
			if (item->Select)
				b = _mtagappendf(ctx, b, "%s SUBQUERY %d", b, item->SelectId);
			else
				b = _mtagappendf(ctx, b, "%s TABLE %s", b, item->Name);
			if (item->Alias)
				b = _mtagappendf(ctx, b, "%s AS %s", b, item->Alias);
			if ((flags & WHERE_INDEXED) != 0)
			{
				char *where = ExplainIndexRange(ctx, level, item->Table);
				b = _mtagappendf(ctx, b, "%s USING %s%sINDEX%s%s%s", b,
					((flags & WHERE_TEMP_INDEX) ? "AUTOMATIC " : ""),
					((flags & WHERE_IDX_ONLY) ? "COVERING " : ""),
					((flags & WHERE_TEMP_INDEX) ? "" : " "),
					((flags & WHERE_TEMP_INDEX) ? "" : level->Plan.u.Index->Name),
				where);
				_tagfree(ctx, where);
			}
			else if (flags & (WHERE_ROWID_EQ|WHERE_ROWID_RANGE))
			{
				b = _mtagappendf(ctx, b, "%s USING INTEGER PRIMARY KEY", b);
				if (flags&WHERE_ROWID_EQ) b = _mtagappendf(ctx, b, "%s (rowid=?)", b);
				else if ((flags&WHERE_BOTH_LIMIT) == WHERE_BOTH_LIMIT) b = _mtagappendf(ctx, b, "%s (rowid>? AND rowid<?)", b);
				else if (flags&WHERE_BTM_LIMIT) b = _mtagappendf(ctx, b, "%s (rowid>?)", b);
				else if (flags&WHERE_TOP_LIMIT) b = _mtagappendf(ctx, b, "%s (rowid<?)", b);
			}
#ifndef OMIT_VIRTUALTABLE
			else if ((flags & WHERE_VIRTUALTABLE) != 0)
			{
				IIndexInfo *vtableIndex = level->Plan.u.VTableIndex;
				b = _mtagappendf(ctx, b, "%s VIRTUAL TABLE INDEX %d:%s", b, vtableIndex->IdxNum, vtableIndex->IdxStr);
			}
#endif
			int64 rows; // Expected number of rows visited by scan
			if (wctrlFlags&(WHERE_ORDERBY_MIN|WHERE_ORDERBY_MAX))
			{
				ASSERTCOVERAGE(wctrlFlags & WHERE_ORDERBY_MIN);
				rows = 1;
			}
			else
				rows = (int64)level->Plan.Rows;
			b = _mtagappendf(ctx, b, "%s (~%lld rows)", b, rows);
			int selectId = parse->SelectId; // Select id (left-most output column)
			v->AddOp4(OP_Explain, selectId, levelId, fromId, b, Vdbe::P4T_DYNAMIC);
		}
	}
#else
#define ExplainOneScan(u, v, w, x, y, z)
#endif

	static Bitmask CodeOneLoopStart(WhereInfo *winfo, int levelId, uint16 wctrlFlags, Bitmask notReady)
	{
		int j, k;
		int addrNxt; // Where to jump to continue with the next IN case
		WhereTerm *term; // A WHERE clause term
		int rowidRegId = 0; // Rowid is stored in this register, if not zero
		int releaseRegId = 0; // Temp register to free before returning

		Parse *parse = winfo->Parse; // Parsing context
		Vdbe *v = parse->V; // The prepared stmt under constructions
		WhereClause *wc = winfo->WC; // Decomposition of the entire WHERE clause 
		WhereLevel *level = &winfo->Data[levelId]; // The where level to be coded
		SrcList::SrcListItem *item = &winfo->TabList->Ids[level->From]; // FROM clause term being coded
		int cur = item->Cursor; // The VDBE cursor for the table
		bool rev = ((level->Plan.WsFlags & WHERE_REVERSE) != 0); // True if we need to scan in reverse order
		bool omitTable = ((level->Plan.WsFlags & WHERE_IDX_ONLY) != 0 && (wctrlFlags & WHERE_FORCE_TABLE) == 0); // True if we use the index only

		// Create labels for the "break" and "continue" instructions for the current loop.  Jump to addrBrk to break out of a loop.
		// Jump to cont to go immediately to the next iteration of the loop.
		//
		// When there is an IN operator, we also have a "addrNxt" label that means to continue with the next IN value combination.  When
		// there are no IN operators in the constraints, the "addrNxt" label is the same as "addrBrk".
		int addrBrk = level->AddrBrk = level->AddrNxt = v->MakeLabel(); // Jump here to break out of the loop
		int addrCont = level->AddrCont = v->MakeLabel(); // Jump here to continue with next cycle

		// If this is the right table of a LEFT OUTER JOIN, allocate and initialize a memory cell that records if this table matches any
		// row of the left table of the join.
		if (level->From > 0 && (item[0].Jointype & JT_LEFT) != 0)
		{
			level->LeftJoin = ++parse->Mems;
			v->AddOp2(OP_Integer, 0, level->LeftJoin);
			v->Comment("init LEFT JOIN no-match flag");
		}

		// Special case of a FROM clause subquery implemented as a co-routine
		if (item->ViaCoroutine)
		{
			int regYield = item->RegReturn;
			v->AddOp2(OP_Integer, item->AddrFillSub-1, regYield);
			level->P2 = v->AddOp1(OP_Yield, regYield);
			v->Comment("next row of co-routine %s", item->Table->Name);
			v->AddOp2(OP_If, regYield+1, addrBrk);
			level->OP = OP_Goto;
		}
		else
		{
#ifndef OMIT_VIRTUALTABLE
			if ((level->Plan.WsFlags & WHERE_VIRTUALTABLE) != 0)
			{
				// Case 0:  The table is a virtual-table.  Use the VFilter and VNext to access the data.
				IIndexInfo *vtableIndex = level->Plan.u.VTableIndex;
				int constraintsLength = vtableIndex->Constraints.length;
				IIndexInfo::ConstraintUsage *usages = vtableIndex->ConstraintUsages;
				const IIndexInfo::Constraint *constraints = vtableIndex->Constraints;

				Expr::CachePush(parse);
				int regId = Expr::GetTempRange(parse, constraintsLength+2); // P3 Value for OP_VFilter
				int addrNotFound = level->AddrBrk;
				for (j = 1; j <= constraintsLength; j++)
				{
					for (k = 0; k < constraintsLength; k++)
					{
						if (usages[k].ArgvIndex == j)
						{
							int targetId = regId+j+1;
							term = &wc->Slots[constraints[k].TermOffset];
							if (term->EOperator & WO_IN)
							{
								CodeEqualityTerm(parse, term, level, k, targetId);
								addrNotFound = level->AddrNxt;
							}
							else
								Expr::Code(parse, term->Expr->Right, targetId);
							break;
						}
					}
					if (k == constraintsLength) break;
				}
				v->AddOp2(OP_Integer, vtableIndex->IdxNum, regId);
				v->AddOp2(OP_Integer, j-1, regId+1);
				v->AddOp4(OP_VFilter, cur, addrNotFound, regId, vtableIndex->IdxStr, vtableIndex->NeedToFreeIdxStr ? Vdbe::P4T_MPRINTF : Vdbe::P4T_STATIC);
				vtableIndex->NeedToFreeIdxStr = false;
				for (j = 0; j < constraintsLength; j++)
				{
					if (usages[j].Omit)
					{
						int termId = constraints[j].TermOffset;
						DisableTerm(level, &wc->Slots[termId]);
					}
				}
				level->OP = OP_VNext;
				level->P1 = cur;
				level->P2 = v->CurrentAddr();
				Expr::ReleaseTempRange(parse, regId, constraintsLength+2);
				Expr::CachePop(parse, 1);
			}
			else
#endif
				if (level->Plan.WsFlags & WHERE_ROWID_EQ)
				{
					// Case 1:  We can directly reference a single row using an equality comparison against the ROWID field.  Or
					// we reference multiple rows using a "rowid IN (...)" construct.
					releaseRegId = Expr::GetTempReg(parse);
					term = FindTerm(wc, cur, -1, notReady, WO_EQ|WO_IN, 0);
					_assert(term != nullptr);
					_assert(term->Expr != nullptr);
					_assert(!omitTable);
					ASSERTCOVERAGE(term->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
					rowidRegId = CodeEqualityTerm(parse, term, level, 0, releaseRegId);
					addrNxt = level->AddrNxt;
					v->AddOp2(OP_MustBeInt, rowidRegId, addrNxt);
					v->AddOp3(OP_NotExists, cur, addrNxt, rowidRegId);
					Expr::CacheAffinityChange(parse, rowidRegId, 1);
					Expr::CacheStore(parse, cur, -1, rowidRegId);
					v->Comment("pk");
					level->OP = OP_Noop;
				}
				else if (level->Plan.WsFlags & WHERE_ROWID_RANGE)
				{
					// Case 2:  We have an inequality comparison against the ROWID field.
					int testOp = OP_Noop;
					int memEndValue = 0;
					_assert(!omitTable);
					WhereTerm *start = FindTerm(wc, cur, -1, notReady, WO_GT|WO_GE, 0);
					WhereTerm *end = FindTerm(wc, cur, -1, notReady, WO_LT|WO_LE, 0);
					if (rev)
					{
						term = start;
						start = end;
						end = term;
					}
					if (start)
					{
						// The following constant maps TK_xx codes into corresponding seek opcodes.  It depends on a particular ordering of TK_xx
						const uint8 _moveOps[] = {
							OP_SeekGt, // TK_GT
							OP_SeekLe, // TK_LE
							OP_SeekLt, // TK_LT
							OP_SeekGe  // TK_GE
						};
						_assert(TK_LE == TK_GT+1); // Make sure the ordering..
						_assert(TK_LT == TK_GT+2); //  ... of the TK_xx values...
						_assert(TK_GE == TK_GT+3); //  ... is correcct.

						ASSERTCOVERAGE(start->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
						Expr *x = start->Expr; // The expression that defines the start bound
						_assert(x != nullptr); 
						_assert(start->LeftCursor == cur);
						int tempId; // Registers for holding the start boundary
						int r1 = Expr::CodeTemp(parse, x->Right, &tempId); // Registers for holding the start boundary
						v->AddOp3(_moveOps[x->OP-TK_GT], cur, addrBrk, r1);
						v->Comment("pk");
						Expr::CacheAffinityChange(parse, r1, 1);
						Expr::ReleaseTempReg(parse, tempId);
						DisableTerm(level, start);
					}
					else
						v->AddOp2(rev ? OP_Last : OP_Rewind, cur, addrBrk);
					if (end)
					{
						Expr *x = end->Expr;
						_assert(x != nullptr);
						_assert(end->LeftCursor == cur);
						ASSERTCOVERAGE(end->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
						memEndValue = ++parse->Mems;
						Expr::Code(parse, x->Right, memEndValue);
						if (x->OP == TK_LT || x->OP == TK_GT)
							testOp = (rev ? OP_Le : OP_Ge);
						else
							testOp = (rev ? OP_Lt : OP_Gt);
						DisableTerm(level, end);
					}
					int startId = v->CurrentAddr();
					level->OP = (rev ? OP_Prev : OP_Next);
					level->P1 = cur;
					level->P2 = startId;
					if (!start && !end)
						level->P5 = Vdbe::STMTSTATUS_FULLSCAN_STEP;
					else
						_assert(level->P5 == 0);
					if (testOp != OP_Noop)
					{
						rowidRegId = releaseRegId = Expr::GetTempReg(parse);
						v->AddOp2(OP_Rowid, cur, rowidRegId);
						Expr::CacheStore(parse, cur, -1, rowidRegId);
						v->AddOp3(testOp, memEndValue, addrBrk, rowidRegId);
						v->ChangeP5(AFF_NUMERIC | AFF_BIT_JUMPIFNULL);
					}
				}
				else if (level->Plan.WsFlags & (WHERE_COLUMN_RANGE|WHERE_COLUMN_EQ))
				{
					// Case 3: A scan using an index.
					//
					//         The WHERE clause may contain zero or more equality terms ("==" or "IN" operators) that refer to the N
					//         left-most columns of the index. It may also contain inequality constraints (>, <, >= or <=) on the indexed
					//         column that immediately follows the N equalities. Only the right-most column can be an inequality - the rest must
					//         use the "==" and "IN" operators. For example, if the index is on (x,y,z), then the following clauses are all optimized:
					//
					//            x=5
					//            x=5 AND y=10
					//            x=5 AND y<10
					//            x=5 AND y>5 AND y<10
					//            x=5 AND y=5 AND z<=10
					//
					//         The z<10 term of the following cannot be used, only the x=5 term:
					//            x=5 AND z<10
					//
					//         N may be zero if there are inequality constraints. If there are no inequality constraints, then N is at least one.
					//
					//         This case is also used when there are no WHERE clause constraints but an index is selected anyway, in order
					//         to force the output order to conform to an ORDER BY.
					static const uint8 _startOps[] = {
						0,
						0,
						OP_Rewind,           // 2: (!startConstraints && startEq &&  !bRev)
						OP_Last,             // 3: (!startConstraints && startEq &&   bRev)
						OP_SeekGt,           // 4: (startConstraints  && !startEq && !bRev)
						OP_SeekLt,           // 5: (startConstraints  && !startEq &&  bRev)
						OP_SeekGe,           // 6: (startConstraints  &&  startEq && !bRev)
						OP_SeekLe            // 7: (startConstraints  &&  startEq &&  bRev)
					};
					static const uint8 _endOps[] = {
						OP_Noop,             // 0: (!end_constraints)
						OP_IdxGE,            // 1: (end_constraints && !bRev)
						OP_IdxLT             // 2: (end_constraints && bRev)
					};
					int eqs = level->Plan.Eqs; // Number of == or IN terms
					Index *index = level->Plan.u.Index; // The index we will be using
					int idxCur = level->IdxCur; // The VDBE cursor for the index
					k = (eqs == index->Columns.length ? -1 : index->Columns[eqs]);

					// If this loop satisfies a sort order (pOrderBy) request that was passed to this function to implement a "SELECT min(x) ..." 
					// query, then the caller will only allow the loop to run for a single iteration. This means that the first row returned
					// should not have a NULL value stored in 'x'. If column 'x' is the first one after the eqs equality constraints in the index,
					// this requires some special handling.
					bool isMinQuery = false; // If this is an optimized SELECT min(x)..
					int extraRegs = 0; // Number of extra registers needed
					if ((wctrlFlags&WHERE_ORDERBY_MIN) != 0 && (level->Plan.WsFlags&WHERE_ORDERED) && index->Columns.length > eqs)
					{
						// _assert(orderBy->Exprs == 1);
						// _assert(orderBy->Ids[0].Expr->Column == index->Columns[eqs]);
						isMinQuery = true;
						extraRegs = 1;
					}

					// Find any inequality constraint terms for the start and end of the range. 
					WhereTerm *rangeEnd = nullptr; // Inequality constraint at range end
					if (level->Plan.WsFlags & WHERE_TOP_LIMIT)
					{
						rangeEnd = FindTerm(wc, cur, k, notReady, (WO_LT|WO_LE), index);
						extraRegs = 1;
					}
					WhereTerm *rangeStart = nullptr; // Inequality constraint at range start
					if (level->Plan.WsFlags & WHERE_BTM_LIMIT)
					{
						rangeStart = FindTerm(wc, cur, k, notReady, (WO_GT|WO_GE), index);
						extraRegs = 1;
					}

					// Generate code to evaluate all constraint terms using == or IN and store the values of those terms in an array of registers starting at baseId.
					char *startAffs; // Affinity for start of range constraint
					int baseId = CodeAllEqualityTerms(parse, level, wc, notReady, extraRegs, &startAffs); // Base register holding constraint values
					char *endAffs = _tagstrdup(parse->Ctx, startAffs); // Affinity for end of range constraint
					addrNxt = level->AddrNxt;

					// If we are doing a reverse order scan on an ascending index, or  a forward order scan on a descending index, interchange the start and end terms (rangeStart and rangeEnd).
					if ((eqs < index->Columns.length && rev == (index->SortOrders[eqs] == SO_ASC)) || (rev && index->Columns.length == eqs))
						SWAP(WhereTerm *, rangeEnd, rangeStart);

					ASSERTCOVERAGE(rangeStart && rangeStart->EOperator & WO_LE);
					ASSERTCOVERAGE(rangeStart && rangeStart->EOperator & WO_GE);
					ASSERTCOVERAGE(rangeEnd && rangeEnd->EOperator & WO_LE);
					ASSERTCOVERAGE(rangeEnd && rangeEnd->EOperator & WO_GE);
					bool startEq = (!rangeStart || rangeStart->EOperator & (WO_LE|WO_GE)); // True if range start uses ==, >= or <=
					bool endEq = (!rangeEnd || rangeEnd->EOperator & (WO_LE|WO_GE)); // True if range end uses ==, >= or <=
					bool startConstraints = (rangeStart || eqs > 0); // Start of range is constrained

					// Seek the index cursor to the start of the range.
					int constraints = eqs; // Number of constraint terms
					if (rangeStart)
					{
						Expr *right = rangeStart->Expr->Right;
						Expr::Code(parse, right, baseId+eqs);
						if ((rangeStart->WtFlags & TERM_VNULL) == 0)
							Expr::CodeIsNullJump(v, right, baseId+eqs, addrNxt);
						if (startAffs)
							if (right->CompareAffinity((AFF)startAffs[eqs]) == AFF_NONE || right->NeedsNoAffinityChange((AFF)startAffs[eqs]))
								startAffs[eqs] = AFF_NONE; // Since the comparison is to be performed with no conversions applied to the operands, set the affinity to apply to pRight to AFF_NONE.
						constraints++;
						ASSERTCOVERAGE(rangeStart->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
					}
					else if (isMinQuery)
					{
						v->AddOp2(OP_Null, 0, baseId+eqs);
						constraints++;
						startEq = 0;
						startConstraints = 1;
					}
					CodeApplyAffinity(parse, baseId, constraints, startAffs);
					int op = _startOps[(startConstraints<<2) + (startEq<<1) + rev]; // Instruction opcode
					_assert(op != 0);
					ASSERTCOVERAGE(op == OP_Rewind);
					ASSERTCOVERAGE(op == OP_Last);
					ASSERTCOVERAGE(op == OP_SeekGt);
					ASSERTCOVERAGE(op == OP_SeekGe);
					ASSERTCOVERAGE(op == OP_SeekLe);
					ASSERTCOVERAGE(op == OP_SeekLt);
					v->AddOp4Int(op, idxCur, addrNxt, baseId, constraints);

					// Load the value for the inequality constraint at the end of the range (if any).
					constraints = eqs;
					if (rangeEnd)
					{
						Expr *right = rangeEnd->Expr->Right;
						Expr::CacheRemove(parse, baseId+eqs, 1);
						Expr::Code(parse, right, baseId+eqs);
						if ((rangeEnd->WtFlags & TERM_VNULL) == 0)
							Expr::CodeIsNullJump(v, right, baseId+eqs, addrNxt);
						if (endAffs)
							if (right->CompareAffinity((AFF)endAffs[eqs]) == AFF_NONE || right->NeedsNoAffinityChange((AFF)endAffs[eqs]))
								endAffs[eqs] = AFF_NONE; // Since the comparison is to be performed with no conversions applied to the operands, set the affinity to apply to pRight to AFF_NONE.
						CodeApplyAffinity(parse, baseId, eqs+1, endAffs);
						constraints++;
						ASSERTCOVERAGE(rangeEnd->WtFlags & TERM_VIRTUAL); // EV: R-30575-11662
					}
					_tagfree(parse->Ctx, startAffs);
					_tagfree(parse->Ctx, endAffs);

					// Top of the loop body
					level->P2 = v->CurrentAddr();

					// Check if the index cursor is past the end of the range.
					op = _endOps[(rangeEnd || eqs) * (1 + rev)];
					ASSERTCOVERAGE(op == OP_Noop);
					ASSERTCOVERAGE(op == OP_IdxGE);
					ASSERTCOVERAGE(op == OP_IdxLT);
					if (op != OP_Noop)
					{
						v->AddOp4Int(op, idxCur, addrNxt, baseId, constraints);
						v->ChangeP5(endEq != rev ? 1 : 0);
					}

					// If there are inequality constraints, check that the value of the table column that the inequality contrains is not NULL.
					// If it is, jump to the next iteration of the loop.
					int r1 = Expr::GetTempReg(parse); // Temp register
					ASSERTCOVERAGE(level->Plan.WsFlags & WHERE_BTM_LIMIT);
					ASSERTCOVERAGE(level->Plan.WsFlags & WHERE_TOP_LIMIT);
					if ((level->Plan.WsFlags & (WHERE_BTM_LIMIT|WHERE_TOP_LIMIT)) != 0)
					{
						v->AddOp3(OP_Column, idxCur, eqs, r1);
						v->AddOp2(OP_IsNull, r1, addrCont);
					}
					Expr::ReleaseTempReg(parse, r1);

					// Seek the table cursor, if required
					DisableTerm(level, rangeStart);
					DisableTerm(level, rangeEnd);
					if (!omitTable)
					{
						rowidRegId = releaseRegId = Expr::GetTempReg(parse);
						v->AddOp2(OP_IdxRowid, idxCur, rowidRegId);
						Expr::CacheStore(parse, cur, -1, rowidRegId);
						v->AddOp2(OP_Seek, cur, rowidRegId); // Deferred seek
					}

					// Record the instruction used to terminate the loop. Disable WHERE clause terms made redundant by the index range scan.
					if (level->Plan.WsFlags & WHERE_UNIQUE) level->OP = OP_Noop;
					else if (rev) level->OP = OP_Prev;
					else level->OP = OP_Next;
					level->P1 = idxCur;
					if (level->Plan.WsFlags & WHERE_COVER_SCAN)
						level->P5 = Vdbe::STMTSTATUS_FULLSCAN_STEP;
					else
						_assert(level->P5 == 0);
				}
				else
#ifndef OMIT_OR_OPTIMIZATION
					if (level->Plan.WsFlags & WHERE_MULTI_OR)
					{
						// Case 4:  Two or more separately indexed terms connected by OR
						//
						// Example:
						//
						//   CREATE TABLE t1(a,b,c,d);
						//   CREATE INDEX i1 ON t1(a);
						//   CREATE INDEX i2 ON t1(b);
						//   CREATE INDEX i3 ON t1(c);
						//
						//   SELECT * FROM t1 WHERE a=5 OR b=7 OR (c=11 AND d=13)
						//
						// In the example, there are three indexed terms connected by OR. The top of the loop looks like this:
						//
						//          Null       1                # Zero the rowset in reg 1
						//
						// Then, for each indexed term, the following. The arguments to RowSetTest are such that the rowid of the current row is inserted
						// into the RowSet. If it is already present, control skips the Gosub opcode and jumps straight to the code generated by WhereEnd().
						//
						//        sqlite3WhereBegin(<term>)
						//          RowSetTest                  # Insert rowid into rowset
						//          Gosub      2 A
						//        sqlite3WhereEnd()
						//
						// Following the above, code to terminate the loop. Label A, the target of the Gosub above, jumps to the instruction right after the Goto.
						//
						//          Null       1                # Zero the rowset in reg 1
						//          Goto       B                # The loop is finished.
						//
						//       A: <loop body>                 # Return data, whatever.
						//
						//          Return     2                # Jump back to the Gosub
						//
						//       B: <after the loop>
						int regReturn = ++parse->Mems; // Register used with OP_Gosub
						int regRowset = 0; // Register for RowSet object
						int regRowid = 0;  // Register holding rowid

						term = level->Plan.u.Term;
						_assert(term != nullptr);
						_assert(term->EOperator & WO_OR);
						_assert((term->WtFlags & TERM_ORINFO) != 0);
						WhereClause *orWC = &term->u.OrInfo->WC; // The OR-clause broken out into subterms
						level->OP = OP_Return;
						level->P1 = regReturn;

						// Set up a new SrcList in orTab containing the table being scanned by this loop in the a[0] slot and all notReady tables in a[1..] slots.
						// This becomes the SrcList in the recursive call to sqlite3WhereBegin().
						SrcList *orTab; // Shortened table list or OR-clause generation
						if (winfo->Levels > 1)
						{
							int notReadys = winfo->Levels - levelId - 1; // The number of notReady tables
							orTab = (SrcList *)_stackalloc(parse->Ctx, sizeof(*orTab) + notReadys*sizeof(orTab->Ids[0]), false);
							if (!orTab) return notReady;
							orTab->Allocs = (int16)(notReadys + 1);
							orTab->Srcs = orTab->Allocs;
							_memcpy(orTab->Ids, item, sizeof(*item));
							SrcList::SrcListItem *origSrc = winfo->TabList->Ids; // Original list of tables
							for (k = 1; k <= notReadys; k++)
								_memcpy(&orTab->Ids[k], &origSrc[level[k].From], sizeof(orTab->Ids[k]));
						}
						else
							orTab = winfo->TabList;

						// Initialize the rowset register to contain NULL. An SQL NULL is equivalent to an empty rowset.
						//
						// Also initialize regReturn to contain the address of the instruction immediately following the OP_Return at the bottom of the loop. This
						// is required in a few obscure LEFT JOIN cases where control jumps over the top of the loop into the body of it. In this case the 
						// correct response for the end-of-loop code (the OP_Return) is to fall through to the next instruction, just as an OP_Next does if
						// called on an uninitialized cursor.
						if ((wctrlFlags & WHERE_DUPLICATES_OK) == 0)
						{
							regRowset = ++parse->Mems;
							regRowid = ++parse->Mems;
							v->AddOp2(OP_Null, 0, regRowset);
						}
						int retInit = v->AddOp2(OP_Integer, 0, regReturn); // Address of regReturn init

						// If the original WHERE clause is z of the form:  (x1 OR x2 OR ...) AND y Then for every term xN, evaluate as the subexpression: xN AND z
						// That way, terms in y that are factored into the disjunction will be picked up by the recursive calls to sqlite3WhereBegin() below.
						//
						// Actually, each subexpression is converted to "xN AND w" where w is the "interesting" terms of z - terms that did not originate in the
						// ON or USING clause of a LEFT JOIN, and terms that are usable as indices.
						Expr *andExpr = nullptr;  // An ".. AND (...)" expression
						if (wc->Terms > 1)
						{
							for (int termId = 0; termId < wc->Terms; termId++)
							{
								Expr *expr = wc->Slots[termId].Expr;
								if (ExprHasProperty(expr, EP_FromJoin)) continue;
								if (wc->Slots[termId].WtFlags & (TERM_VIRTUAL|TERM_ORINFO)) continue;
								if ((wc->Slots[termId].EOperator & WO_ALL) == 0) continue;
								expr = Expr::Dup(parse->Ctx, expr, 0);
								andExpr = Expr::And(parse->Ctx, andExpr, expr);
							}
							if (andExpr)
								andExpr = Expr::PExpr_(parse, TK_AND, 0, andExpr, 0);
						}

						Index *cov = nullptr; // Potential covering index (or NULL)
						int covCur = parse->Tabs++; // Cursor used for index scans (if any)
						int loopBodyId = v->MakeLabel(); // Start of loop body
						bool untestedTerms = false; // Some terms not completely tested
						for (int ii = 0; ii < orWC->Terms; ii++)
						{
							WhereTerm *orTerm = &orWC->Slots[ii];
							if (orTerm->LeftCursor == cur || (orTerm->EOperator & WO_AND) != 0)
							{
								Expr *orExpr = orTerm->Expr;
								if (andExpr)
								{
									andExpr->Left = orExpr;
									orExpr = andExpr;
								}
								// Loop through table entries that match term orTerm.
								WhereInfo *subWInfo = WhereInfo::Begin(parse, orTab, orExpr, 0, 0, WHERE_OMIT_OPEN_CLOSE | WHERE_AND_ONLY | WHERE_FORCE_TABLE | WHERE_ONETABLE_ONLY, covCur); // Info for single OR-term scan
								_assert(subWInfo || parse->Errs || parse->Ctx->MallocFailed);
								if (subWInfo)
								{
									ExplainOneScan(parse, orTab, &subWInfo->Data[0], levelId, level->From, 0);
									if ((wctrlFlags & WHERE_DUPLICATES_OK) == 0)
									{
										int setId = (ii == orWC->Terms-1 ? -1 : ii);
										int r = Expr::CodeGetColumn(parse, item->Table, -1, cur, regRowid, 0);
										v->AddOp4Int(OP_RowSetTest, regRowset, v->CurrentAddr()+2, r, setId);
									}
									v->AddOp2(OP_Gosub, regReturn, loopBodyId);

									// The subWInfo->untestedTerms flag means that this OR term contained one or more AND term from a notReady table.  The
									// terms from the notReady table could not be tested and will need to be tested later.
									if (subWInfo->UntestedTerms) untestedTerms = true;

									// If all of the OR-connected terms are optimized using the same index, and the index is opened using the same cursor number
									// by each call to sqlite3WhereBegin() made by this loop, it may be possible to use that index as a covering index.
									//
									// If the call to sqlite3WhereBegin() above resulted in a scan that uses an index, and this is either the first OR-connected term
									// processed or the index is the same as that used by all previous terms, set cov to the candidate covering index. Otherwise, set 
									// cov to NULL to indicate that no candidate covering index will be available.
									WhereLevel *lvl = &subWInfo->Data[0];
									if ((lvl->Plan.WsFlags & WHERE_INDEXED) != 0 && (lvl->Plan.WsFlags & WHERE_TEMP_INDEX) == 0 && (ii == 0 || lvl->Plan.u.Index == cov))
									{
										_assert(lvl->IdxCur == covCur);
										cov = lvl->Plan.u.Index;
									}
									else
										cov = nullptr;

									// Finish the loop through table entries that match term orTerm.
									WhereInfo::End(subWInfo);
								}
							}
						}
						level->u.Covidx = cov;
						if (cov) level->IdxCur = covCur;
						if (andExpr)
						{
							andExpr->Left = nullptr;
							Expr::Delete(parse->Ctx, andExpr);
						}
						v->ChangeP1(retInit, v->CurrentAddr());
						v->AddOp2(OP_Goto, 0, level->AddrBrk);
						v->ResolveLabel(loopBodyId);

						if (winfo->Levels > 1) _stackfree(parse->Ctx, orTab);
						if (!untestedTerms) DisableTerm(level, term);
					}
					else
#endif
					{
						// Case 5:  There is no usable index.  We must do a complete scan of the entire table.
						static const OP _steps[] = { OP_Next, OP_Prev };
						static const OP _starts[] = { OP_Rewind, OP_Last };
						_assert(rev == 0 || rev == 1);
						_assert(omitTable == 0);
						level->OP = _steps[rev];
						level->P1 = cur;
						level->P2 = 1 + v->AddOp2(_starts[rev], cur, addrBrk);
						level->P5 = Vdbe::STMTSTATUS_FULLSCAN_STEP;
					}
		}
		notReady &= ~GetMask(wc->MaskSet, cur);

		// Insert code to test every subexpression that can be completely computed using the current set of tables.
		//
		// IMPLEMENTATION-OF: R-49525-50935 Terms that cannot be satisfied through the use of indices become tests that are evaluated against each row of the relevant input tables.
		for (term = wc->Slots, j = wc->Terms; j > 0; j--, term++)
		{
			ASSERTCOVERAGE(term->WtFlags & TERM_VIRTUAL); // IMP: R-30575-11662
			ASSERTCOVERAGE(term->WtFlags & TERM_CODED);
			if( term->WtFlags & (TERM_VIRTUAL|TERM_CODED)) continue;
			if ((term->PrereqAll & notReady) != 0)
			{
				ASSERTCOVERAGE(winfo->UntestedTerms == 0 && (winfo->WctrlFlags & WHERE_ONETABLE_ONLY) != 0);
				winfo->UntestedTerms = 1;
				continue;
			}
			Expr *e = term->Expr;
			_assert(e != nullptr);
			if (level->LeftJoin && !ExprHasProperty(e, EP_FromJoin))
				continue;
			e->IfFalse(parse, addrCont, AFF_BIT_JUMPIFNULL);
			term->WtFlags |= TERM_CODED;
		}

		// For a LEFT OUTER JOIN, generate code that will record the fact that at least one row of the right table has matched the left table.  
		if (level->LeftJoin)
		{
			level->AddrFirst = v->CurrentAddr();
			v->AddOp2(OP_Integer, 1, level->LeftJoin);
			v->Comment("record LEFT JOIN hit");
			Expr::CacheClear(parse);
			for (term = wc->Slots, j = 0; j < wc->Terms; j++, term++)
			{
				ASSERTCOVERAGE(term->WtFlags & TERM_VIRTUAL); // IMP: R-30575-11662
				ASSERTCOVERAGE(term->WtFlags & TERM_CODED);
				if (term->WtFlags & (TERM_VIRTUAL|TERM_CODED)) continue;
				if ((term->PrereqAll & notReady) != 0)
				{
					_assert(winfo->UntestedTerms);
					continue;
				}
				_assert(term->Expr);
				term->Expr->IfFalse(parse, addrCont, AFF_BIT_JUMPIFNULL);
				term->WtFlags |= TERM_CODED;
			}
		}
		Expr::ReleaseTempReg(parse, releaseRegId);
		return notReady;
	}

	__device__ static void WhereInfoFree(Context *ctx, WhereInfo *winfo)
	{
		if (_ALWAYS(winfo))
		{
			for (int i = 0; i < winfo->Levels; i++)
			{
				IIndexInfo *info = winfo->Data[i].IndexInfo;
				if (info)
				{
					// _assert(!info->NeedToFreeIdxStr || ctx->MallocFailed);
					if (info->NeedToFreeIdxStr) _free(info->IdxStr);
					_tagfree(ctx, info);
				}
				if (winfo->Data[i].Plan.WsFlags & WHERE_TEMP_INDEX)
				{
					Index *index = winfo->Data[i].Plan.u.Index;
					if (index)
					{
						_tagfree(ctx, index->ColAff);
						_tagfree(ctx, index);
					}
				}
			}
			WhereClauseClear(winfo->WC);
			_tagfree(ctx, winfo);
		}
	}

#if defined(TEST)
	char _queryPlan[BMS*2*40]; // Text of the join
	static int _queryPlanIdx = 0; // Next free slow in _query_plan[]
#endif

	__device__ WhereInfo *WhereInfo::Begin(::Parse *parse, SrcList *tabList, Expr *where,  ExprList *orderBy, ExprList *distinct, WHERE wctrlFlags, int idxCur)
	{
		Vdbe *v = parse->V; // The virtual database engine
		Bitmask notReady; // Cursors that are not yet positioned
		int ii;

		// Variable initialization
		WhereBestIdx sWBI; // Best index search context
		_memset(&sWBI, 0, sizeof(sWBI));
		sWBI.Parse = parse;

		// The number of tables in the FROM clause is limited by the number of bits in a Bitmask 
		ASSERTCOVERAGE(tabList->Srcs == BMS);
		if (tabList->Srcs > BMS)
		{
			parse->ErrorMsg("at most %d tables in a join", BMS);
			return nullptr;
		}

		// This function normally generates a nested loop for all tables in tabList.  But if the WHERE_ONETABLE_ONLY flag is set, then we should
		// only generate code for the first table in tabList and assume that any cursors associated with subsequent tables are uninitialized.
		int tabListLength = ((wctrlFlags & WHERE_ONETABLE_ONLY) ? 1 : tabList->Srcs); // Number of elements in tabList

		// Allocate and initialize the WhereInfo structure that will become the return value. A single allocation is used to store the WhereInfo
		// struct, the contents of WhereInfo.a[], the WhereClause structure and the WhereMaskSet structure. Since WhereClause contains an 8-byte
		// field (type Bitmask) it must be aligned on an 8-byte boundary on some architectures. Hence the ROUND8() below.
		Context *ctx = parse->Ctx; // Database connection
		int bytesWInfo = _ROUND8(sizeof(WhereInfo)+(tabListLength-1)*sizeof(WhereLevel)); // Num. bytes allocated for WhereInfo struct
		WhereInfo *winfo = (WhereInfo *)_tagalloc2(ctx, bytesWInfo +  sizeof(WhereClause) + sizeof(WhereMaskSet), true); // Will become the return value of this function
		if (ctx->MallocFailed)
		{
			_tagfree(ctx, winfo);
			winfo = nullptr;
			goto whereBeginError;
		}
		winfo->Levels = tabListLength;
		winfo->Parse = parse;
		winfo->TabList = tabList;
		winfo->BreakId = v->MakeLabel();
		winfo->WC = sWBI.WC = (WhereClause *)&((uint8 *)winfo)[bytesWInfo];
		winfo->WctrlFlags = wctrlFlags;
		winfo->SavedNQueryLoop = parse->QueryLoops;
		WhereMaskSet *maskSet = (WhereMaskSet*)&sWBI.WC[1]; // The expression mask set
		InitMaskSet(maskSet);
		sWBI.Levels = winfo->Data;

		// Disable the DISTINCT optimization if SQLITE_DistinctOpt is set via sqlite3_test_ctrl(SQLITE_TESTCTRL_OPTIMIZATIONS,...)
		if (CtxOptimizationDisabled(ctx, OPTFLAG_DistinctOpt)) distinct = nullptr;

		// Split the WHERE clause into separate subexpressions where each subexpression is separated by an AND operator.
		WhereClauseInit(sWBI.WC, parse, maskSet, wctrlFlags);
		Expr::CodeConstants(parse, where);
		WhereSplit(sWBI.WC, where, TK_AND); // IMP: R-15842-53296

		// Special case: a WHERE clause that is constant.  Evaluate the expression and either jump over all of the code or fall thru.
		if (where && (tabListLength == 0 || where->IsConstantNotJoin()))
		{
			where->IfFalse(parse, winfo->BreakId, AFF_BIT_JUMPIFNULL);
			where = nullptr;
		}

		// Assign a bit from the bitmask to every term in the FROM clause.
		//
		// When assigning bitmask values to FROM clause cursors, it must be the case that if X is the bitmask for the N-th FROM clause term then
		// the bitmask for all FROM clause terms to the left of the N-th term is (X-1).   An expression from the ON clause of a LEFT JOIN can use
		// its Expr.iRightJoinTable value to find the bitmask of the right table of the join.  Subtracting one from the right table bitmask gives a
		// bitmask for all tables to the left of the join.  Knowing the bitmask for all tables to the left of a left join is important.  Ticket #3015.
		//
		// Note that bitmasks are created for all tabList->nSrc tables in tabList, not just the first tabListLength tables.  tabListLength is normally
		// equal to tabList->nSrc but might be shortened to 1 if the WHERE_ONETABLE_ONLY flag is set.
		for (ii = 0; ii < tabList->Srcs; ii++)
			CreateMask(maskSet, tabList->Ids[ii].Cursor);
#ifdef _DEBUG
		{
			Bitmask toTheLeft = 0;
			for (ii = 0; ii < tabList->Srcs; ii++)
			{
				Bitmask m = GetMask(maskSet, tabList->Ids[ii].Cursor);
				_assert((m-1) == toTheLeft);
				toTheLeft |= m;
			}
		}
#endif

		// Analyze all of the subexpressions.  Note that exprAnalyze() might add new virtual terms onto the end of the WHERE clause.  We do not
		// want to analyze these virtual terms, so start analyzing at the end and work forward so that the added virtual terms are never processed.
		ExprAnalyzeAll(tabList, sWBI.WC);
		if (ctx->MallocFailed)
			goto whereBeginError;

		// Check if the DISTINCT qualifier, if there is one, is redundant. If it is, then set pDistinct to NULL and WhereInfo.eDistinct to
		// WHERE_DISTINCT_UNIQUE to tell the caller to ignore the DISTINCT.
		if (distinct && IsDistinctRedundant(parse, tabList, sWBI.WC, distinct))
		{
			distinct = nullptr;
			winfo->EDistinct = WHERE_DISTINCT_UNIQUE;
		}

		// Chose the best index to use for each table in the FROM clause.
		// This loop fills in the following fields:
		//   winfo->a[].pIdx      The index to use for this level of the loop.
		//   winfo->a[].wsFlags   WHERE_xxx flags associated with pIdx
		//   winfo->a[].nEq       The number of == and IN constraints
		//   winfo->a[].From     Which term of the FROM clause is being coded
		//   winfo->a[].iTabCur   The VDBE cursor for the database table
		//   winfo->a[].iIdxCur   The VDBE cursor for the index
		//   winfo->a[].pTerm     When wsFlags==WO_OR, the OR-clause term
		// This loop also figures out the nesting order of tables in the FROM clause.
		sWBI.NotValid = ~(Bitmask)0;
		sWBI.OrderBy = orderBy;
		sWBI.n = tabListLength;
		sWBI.Distinct = distinct;
		int andFlags = ~0; // AND-ed combination of all pWC->a[].wtFlags
		WHERETRACE("*** Optimizer Start ***\n");
		int fromId; // First unused FROM clause element
		WhereLevel *level; // A single level in pWInfo->a[]
		for (sWBI.i = fromId = 0, level = winfo->Data; sWBI.i < tabListLength; sWBI.i++, level++)
		{
			int j; // For looping over FROM tables
			int bestJ = -1; // The value of j

			WhereCost bestPlan; // Most efficient plan seen so far
			_memset(&bestPlan, 0, sizeof(bestPlan));
			bestPlan.Cost = BIG_DOUBLE;
			WHERETRACE("*** Begin search for loop %d ***\n", sWBI.i);

			// Loop through the remaining entries in the FROM clause to find the next nested loop. The loop tests all FROM clause entries
			// either once or twice. 
			//
			// The first test is always performed if there are two or more entries remaining and never performed if there is only one FROM clause entry
			// to choose from.  The first test looks for an "optimal" scan.  In this context an optimal scan is one that uses the same strategy
			// for the given FROM clause entry as would be selected if the entry were used as the innermost nested loop.  In other words, a table
			// is chosen such that the cost of running that table cannot be reduced by waiting for other tables to run first.  This "optimal" test works
			// by first assuming that the FROM clause is on the inner loop and finding its query plan, then checking to see if that query plan uses any
			// other FROM clause terms that are sWBI.notValid.  If no notValid terms are used then the "optimal" query plan works.
			//
			// Note that the WhereCost.nRow parameter for an optimal scan might not be as small as it would be if the table really were the innermost
			// join.  The nRow value can be reduced by WHERE clause constraints that do not use indices.  But this nRow reduction only happens if the
			// table really is the innermost join.  
			//
			// The second loop iteration is only performed if no optimal scan strategies were found by the first iteration. This second iteration
			// is used to search for the lowest cost scan overall.
			//
			// Without the optimal scan step (the first iteration) a suboptimal plan might be chosen for queries like this:
			//   CREATE TABLE t1(a, b); 
			//   CREATE TABLE t2(c, d);
			//   SELECT * FROM t2, t1 WHERE t2.rowid = t1.a;
			//
			// The best strategy is to iterate through table t1 first. However it is not possible to determine this with a simple greedy algorithm.
			// Since the cost of a linear scan through table t2 is the same as the cost of a linear scan through table t1, a simple greedy 
			// algorithm may choose to use t2 for the outer loop, which is a much costlier approach.
			int unconstrained = 0; // Number tables without INDEXED BY
			Bitmask notIndexed = 0; // Mask of tables that cannot use an index

			// The optimal scan check only occurs if there are two or more tables available to be reordered
			Bitmask m; // Bitmask value for j or bestJ
			int ckOptimal; // Do the optimal scan check
			if (fromId == tabListLength-1)
				ckOptimal = 0; // Common case of just one table in the FROM clause
			else
			{
				ckOptimal = -1;
				for (j = fromId, sWBI.Src = &tabList->Ids[j]; j < tabListLength; j++, sWBI.Src++)
				{
					m = GetMask(maskSet, sWBI.Src->Cursor);
					if ((m & sWBI.NotValid) == 0)
					{
						if (j == fromId) fromId++;
						continue;
					}
					if (j > fromId && (sWBI.Src->Jointype & (JT_LEFT|JT_CROSS)) != 0) break;
					if (++ckOptimal) break;
					if ((sWBI.Src->Jointype & JT_LEFT) != 0) break;
				}
			}
			_assert(ckOptimal == 0 || ckOptimal == 1);
			for (int isOptimal = ckOptimal; isOptimal >= 0 && bestJ < 0; isOptimal--)
			{
				for (j = fromId, sWBI.Src = &tabList->Ids[j]; j < tabListLength; j++, sWBI.Src++)
				{
					// This break and one like it in the ckOptimal computation loop above prevent table reordering across LEFT and CROSS JOINs.
					// The LEFT JOIN case is necessary for correctness.  The prohibition against reordering across a CROSS JOIN is an SQLite feature that
					// allows the developer to control table reordering
					if (j > fromId && (sWBI.Src->Jointype & (JT_LEFT|JT_CROSS)) != 0)
						break;
					m = GetMask(maskSet, sWBI.Src->Cursor);
					if ((m & sWBI.NotValid) == 0)
					{
						_assert(j > fromId);
						continue;
					}
					sWBI.NotReady = (isOptimal ? m : sWBI.NotValid);
					if (!sWBI.Src->Index) unconstrained++;

					WHERETRACE("   === trying table %d (%s) with isOptimal=%d ===\n", j, sWBI.Src->Table->Name, isOptimal);
					_assert(sWBI.Src->Table);
#ifndef OMIT_VIRTUALTABLE
					if (IsVirtual(sWBI.Src->Table))
					{
						sWBI.IdxInfo = &winfo->Data[j].IndexInfo;
						BestVirtualIndex(&sWBI);
					}
					else 
#endif
						BestBtreeIndex(&sWBI);
					_assert(isOptimal || (sWBI.Cost.Used & sWBI.NotValid) == 0);

					// If an INDEXED BY clause is present, then the plan must use that index if it uses any index at all
					_assert(sWBI.Src->Index == nullptr || (sWBI.Cost.Plan.WsFlags & WHERE_NOT_FULLSCAN) == 0 || sWBI.Cost.Plan.u.Index == sWBI.Src->Index);

					if (isOptimal && (sWBI.Cost.Plan.WsFlags & WHERE_NOT_FULLSCAN) == 0)
						notIndexed |= m;
					if (isOptimal)
						winfo->Data[j].OptCost = sWBI.Cost.Cost;
					else if (ckOptimal)
					{
						// If two or more tables have nearly the same outer loop cost, but very different inner loop (optimal) cost, we want to choose
						// for the outer loop that table which benefits the least from being in the inner loop.  The following code scales the 
						// outer loop cost estimate to accomplish that.
						WHERETRACE("   scaling cost from %.1f to %.1f\n", sWBI.Cost.Cost, sWBI.Cost.Cost / winfo->Data[j].OptCost);
						sWBI.Cost.Cost /= winfo->Data[j].OptCost;
					}

					// Conditions under which this table becomes the best so far:
					//   (1) The table must not depend on other tables that have not yet run.  (In other words, it must not depend on tables in inner loops.)
					//   (2) (This rule was removed on 2012-11-09.  The scaling of the cost using the optimal scan cost made this rule obsolete.)
					//   (3) All tables have an INDEXED BY clause or this table lacks an INDEXED BY clause or this table uses the specific
					//       index specified by its INDEXED BY clause.  This rule ensures that a best-so-far is always selected even if an impossible
					//       combination of INDEXED BY clauses are given.  The error will be detected and relayed back to the application later.
					//       The NEVER() comes about because rule (2) above prevents An indexable full-table-scan from reaching rule (3).
					//   (4) The plan cost must be lower than prior plans, where "cost" is defined by the compareCost() function above. 
					if ((sWBI.Cost.Used & sWBI.NotValid) == 0 && // (1)
						(unconstrained == 0 || !sWBI.Src->Index || _NEVER((sWBI.Cost.Plan.WsFlags & WHERE_NOT_FULLSCAN) != 0)) && // (3)
						(bestJ < 0 || CompareCost(&sWBI.Cost, &bestPlan))) // (4)
					{
						WHERETRACE("   === table %d (%s) is best so far\n       cost=%.1f, nRow=%.1f, nOBSat=%d, wsFlags=%08x\n",
							j, sWBI.Src->Table->Name,
							sWBI.Cost.Cost, sWBI.Cost.Plan.Rows,
							sWBI.Cost.Plan.OBSats, sWBI.Cost.Plan.WsFlags);
						bestPlan = sWBI.Cost;
						bestJ = j;
					}

					// In a join like "w JOIN x LEFT JOIN y JOIN z"  make sure that table y (and not table z) is always the next inner loop inside of table x.
					if ((sWBI.Src->Jointype & JT_LEFT) != 0) break;
				}
			}
			_assert(bestJ >= 0);
			_assert(sWBI.NotValid & GetMask(maskSet, tabList->Ids[bestJ].Cursor));
			_assert(bestJ==fromId || (tabList->Ids[fromId].Jointype & JT_LEFT) == 0);
			ASSERTCOVERAGE(bestJ > fromId && (tabList->Ids[fromId].Jointype & JT_CROSS) != 0);
			ASSERTCOVERAGE(bestJ > fromId && bestJ < tabListLength-1 && (tabList->Ids[bestJ+1].Jointype & JT_LEFT) != 0);
			WHERETRACE("*** Optimizer selects table %d (%s) for loop %d with:\n    cost=%.1f, nRow=%.1f, nOBSat=%d, wsFlags=0x%08x\n",
				bestJ, tabList->Ids[bestJ].Table->Name,
				level - winfo->Data, bestPlan.Cost, bestPlan.Plan.Rows,
				bestPlan.Plan.OBSats, bestPlan.Plan.WsFlags);
			if ((bestPlan.Plan.WsFlags & WHERE_DISTINCT_) != 0)
			{
				_assert(winfo->EDistinct == 0);
				winfo->EDistinct = WHERE_DISTINCT_ORDERED;
			}
			andFlags &= bestPlan.Plan.WsFlags;
			level->Plan = bestPlan.Plan;
			level->TabCur = tabList->Ids[bestJ].Cursor;
			ASSERTCOVERAGE(bestPlan.Plan.WsFlags & WHERE_INDEXED);
			ASSERTCOVERAGE(bestPlan.Plan.WsFlags & WHERE_TEMP_INDEX);
			if (bestPlan.Plan.WsFlags & (WHERE_INDEXED|WHERE_TEMP_INDEX))
				level->IdxCur = ((wctrlFlags & WHERE_ONETABLE_ONLY) && (bestPlan.Plan.WsFlags & WHERE_TEMP_INDEX) == 0 ? idxCur : parse->Tabs++);
			else
				level->IdxCur = -1;
			sWBI.NotValid &= ~GetMask(maskSet, tabList->Ids[bestJ].Cursor);
			level->From = (uint8)bestJ;
			if (bestPlan.Plan.Rows >= (double)1)
				parse->QueryLoops *= bestPlan.Plan.Rows;

			// Check that if the table scanned by this loop iteration had an INDEXED BY clause attached to it, that the named index is being
			// used for the scan. If not, then query compilation has failed. Return an error.
			Index *index = tabList->Ids[bestJ].Index; // Index for FROM table at pTabItem
			if (index)
				if( (bestPlan.Plan.WsFlags & WHERE_INDEXED) == 0)
				{
					parse->ErrorMsg("cannot use index: %s", index->Name);
					goto whereBeginError;
				}
				else // If an INDEXED BY clause is used, the bestIndex() function is guaranteed to find the index specified in the INDEXED BY clause if it find an index at all.
					_assert(bestPlan.Plan.u.Index == index);
		}
		WHERETRACE("*** Optimizer Finished ***\n");
		if (parse->Errs || ctx->MallocFailed)
			goto whereBeginError;
		if (tabListLength)
		{
			level--;
			winfo->OBSats = level->Plan.OBSats;
		}
		else
			winfo->OBSats = 0;

		// If the total query only selects a single row, then the ORDER BY clause is irrelevant.
		if ((andFlags & WHERE_UNIQUE) != 0 && orderBy)
		{
			_assert(tabListLength == 0 || (level->Plan.WsFlags & WHERE_ALL_UNIQUE) != 0);
			winfo->OBSats = orderBy->Exprs;
		}

		// If the caller is an UPDATE or DELETE statement that is requesting to use a one-pass algorithm, determine if this is appropriate.
		// The one-pass algorithm only works if the WHERE clause constraints the statement to update a single row.
		_assert((wctrlFlags & WHERE_ONEPASS_DESIRED) == 0 || winfo->Levels == 1);
		if ((wctrlFlags & WHERE_ONEPASS_DESIRED) != 0 && (andFlags & WHERE_UNIQUE) != 0)
		{
			winfo->OkOnePass = true;
			winfo->Data[0].Plan.WsFlags &= ~WHERE_IDX_ONLY;
		}

		// Open all tables in the tabList and any indices selected for searching those tables.
		parse->CodeVerifySchema(-1); // Insert the cookie verifier Goto
		notReady = ~(Bitmask)0;
		winfo->RowOuts = (double)1;
		for (ii = 0, level = winfo->Data; ii < tabListLength; ii++, level++)
		{
			SrcList::SrcListItem *tabItem = &tabList->Ids[level->From];
			Table *table = tabItem->Table; // Table to open
			winfo->RowOuts *= level->Plan.Rows;
			int db = Prepare::SchemaToIndex(ctx, table->Schema); // Index of database containing table/index
			if ((table->TabFlags & TF_Ephemeral) != 0 || table->Select) { } // Do nothing
			else
#ifndef OMIT_VIRTUALTABLE
				if ((level->Plan.WsFlags & WHERE_VIRTUALTABLE) != 0)
				{
					const char *vtable = (const char *)VTable::GetVTable(ctx, table);
					int cur = tabItem->Cursor;
					v->AddOp4(OP_VOpen, cur, 0, 0, vtable, Vdbe::P4T_VTAB);
				}
				else if (IsVirtual(table)) { } // noop
				else
#endif
					if ((level->Plan.WsFlags & WHERE_IDX_ONLY) == 0 && (wctrlFlags & WHERE_OMIT_OPEN_CLOSE) == 0)
					{
						int op = (winfo->OkOnePass ? OP_OpenWrite : OP_OpenRead);
						Insert::OpenTable(parse, tabItem->Cursor, db, table, op);
						ASSERTCOVERAGE(table->Cols.length == BMS-1);
						ASSERTCOVERAGE(table->Cols.length == BMS);
						if (!winfo->OkOnePass && table->Cols.length < BMS)
						{
							Bitmask b = tabItem->ColUsed;
							int n = 0;
							for (; b; b = b >> 1, n++) { }
							v->ChangeP4(v->CurrentAddr()-1, (char *)INT_TO_PTR(n), Vdbe::P4T_INT32);
							_assert(n <= table->Cols.length);
						}
					}
					else
						parse->TableLock(db, table->Id, 0, table->Name);
#ifndef OMIT_AUTOMATIC_INDEX
				if ((level->Plan.WsFlags & WHERE_TEMP_INDEX) != 0)
					ConstructAutomaticIndex(parse, sWBI.WC, tabItem, notReady, level);
				else
#endif
					if ((level->Plan.WsFlags & WHERE_INDEXED) != 0)
					{
						Index *index = level->Plan.u.Index;
						KeyInfo *keyInfo = parse->IndexKeyinfo(index);
						int indexCur = level->IdxCur;
						_assert(index->Schema == table->Schema);
						_assert(indexCur >= 0);
						v->AddOp4(OP_OpenRead, indexCur, index->Id, db, (char *)keyInfo, Vdbe::P4T_KEYINFO_HANDOFF);
						v->Comment("%s", index->Name);
					}
					parse->CodeVerifySchema(db);
					notReady &= ~GetMask(sWBI.WC->MaskSet, tabItem->Cursor);
		}
		winfo->TopId = v->CurrentAddr();
		if (ctx->MallocFailed) goto whereBeginError;

		// Generate the code to do the search.  Each iteration of the for loop below generates code for a single nested loop of the VM program.
		notReady = ~(Bitmask)0;
		for (ii = 0; ii < tabListLength; ii++)
		{
			level = &winfo->Data[ii];
			ExplainOneScan(parse, tabList, level, ii, level->From, wctrlFlags);
			notReady = CodeOneLoopStart(winfo, ii, wctrlFlags, notReady);
			winfo->ContinueId = level->AddrCont;
		}

		// For testing and debugging use only
#ifdef TEST
		// Record in the query plan information about the current table and the index used to access it (if any).  If the table itself
		// is not used, its name is just '{}'.  If no index is used the index is listed as "{}".  If the primary key is used the index name is '*'.
		for (ii = 0; ii < tabListLength; ii++)
		{
			level = &winfo->Data[ii];
			int w = level->Plan.WsFlags;
			SrcList::SrcListItem *tabItem = &tabList->Ids[level->From];
			char *z = tabItem->Alias;
			if (!z) z = tabItem->Table->Name;
			int n = _strlen30(z);
			if (n + _queryPlanIdx < sizeof(_queryPlan)-10)
			{
				if ((w & WHERE_IDX_ONLY) != 0 && (w & WHERE_COVER_SCAN) == 0)
				{
					_memcpy(&_queryPlan[_queryPlanIdx], "{}", 2);
					_queryPlanIdx += 2;
				}
				else
				{
					_memcpy(&_queryPlan[_queryPlanIdx], z, n);
					_queryPlanIdx += n;
				}
				_queryPlan[_queryPlanIdx++] = ' ';
			}
			ASSERTCOVERAGE(w & WHERE_ROWID_EQ);
			ASSERTCOVERAGE(w & WHERE_ROWID_RANGE);
			if (w & (WHERE_ROWID_EQ|WHERE_ROWID_RANGE))
			{
				_memcpy(&_queryPlan[_queryPlanIdx], "* ", 2);
				_queryPlanIdx += 2;
			}
			else if ((w & WHERE_INDEXED) != 0 && (w & WHERE_COVER_SCAN) == 0)
			{
				n = _strlen30(level->Plan.u.Index->Name);
				if (n + _queryPlanIdx < sizeof(_queryPlan)-2)
				{
					_memcpy(&_queryPlan[_queryPlanIdx], level->Plan.u.Index->Name, n);
					_queryPlanIdx += n;
					_queryPlan[_queryPlanIdx++] = ' ';
				}
			}
			else
			{
				_memcpy(&_queryPlan[_queryPlanIdx], "{} ", 3);
				_queryPlanIdx += 3;
			}
		}
		while (_queryPlanIdx>0 && _queryPlan[_queryPlanIdx-1] == ' ')
			_queryPlan[--_queryPlanIdx] = 0;
		_queryPlan[_queryPlanIdx] = 0;
		_queryPlanIdx = 0;
#endif

		// Record the continuation address in the WhereInfo structure.  Then clean up and return.
		return winfo;

		// Jump here if malloc fails
whereBeginError:
		if (winfo)
		{
			parse->QueryLoops = winfo->SavedNQueryLoop;
			WhereInfoFree(ctx, winfo);
		}
		return nullptr;
	}

	__device__ void WhereInfo::End(WhereInfo *winfo)
	{
		::Parse *parse = winfo->Parse;
		Vdbe *v = parse->V;
		int i;
		WhereLevel *level;
		SrcList *tabList = winfo->TabList;
		Context *ctx = parse->Ctx;

		// Generate loop termination code.
		Expr::CacheClear(parse);
		for (i = winfo->Levels-1; i >= 0; i--)
		{
			level = &winfo->Data[i];
			v->ResolveLabel(level->AddrCont);
			if (level->OP != OP_Noop)
			{
				v->AddOp2(level->OP, level->P1, level->P2);
				v->ChangeP5(level->P5);
			}
			if ((level->Plan.WsFlags & WHERE_IN_ABLE) && level->u.in.InLoopsLength > 0)
			{
				v->ResolveLabel(level->AddrNxt);
				int j;
				WhereLevel::InLoop *in_;
				for (j = level->u.in.InLoopsLength, in_ = &level->u.in.InLoops[j-1]; j > 0; j--, in_--)
				{
					v->JumpHere(in_->AddrInTop+1);
					v->AddOp2(in_->EndLoopOp, in_->Cur, in_->AddrInTop);
					v->JumpHere(in_->AddrInTop-1);
				}
				_tagfree(ctx, level->u.in.InLoops);
			}
			v->ResolveLabel(level->AddrBrk);
			if (level->LeftJoin)
			{
				int addr = v->AddOp1(OP_IfPos, level->LeftJoin);
				_assert((level->Plan.WsFlags & WHERE_IDX_ONLY) == 0 || (level->Plan.WsFlags & WHERE_INDEXED) != 0);
				if ((level->Plan.WsFlags & WHERE_IDX_ONLY) == 0)
					v->AddOp1(OP_NullRow, tabList->Ids[i].Cursor);
				if (level->IdxCur >= 0)
					v->AddOp1(OP_NullRow, level->IdxCur);
				if (level->OP == OP_Return)
					v->AddOp2(OP_Gosub, level->P1, level->AddrFirst);
				else
					v->AddOp2(OP_Goto, 0, level->AddrFirst);
				v->JumpHere(addr);
			}
		}

		// The "break" point is here, just past the end of the outer loop. Set it.
		v->ResolveLabel(winfo->BreakId);

		// Close all of the cursors that were opened by sqlite3WhereBegin.
		_assert(winfo->Levels == 1 || winfo->Levels == tabList->Srcs);
		for (i = 0, level = winfo->Data; i < winfo->Levels; i++, level++)
		{
			SrcList::SrcListItem *tabItem = &tabList->Ids[level->From];
			Table *table = tabItem->Table;
			_assert(table != nullptr);
			if ((table->TabFlags & TF_Ephemeral) == 0 && !table->Select && (winfo->WctrlFlags & WHERE_OMIT_OPEN_CLOSE) == 0)
			{
				uint ws = level->Plan.WsFlags;
				if (!winfo->OkOnePass && (ws & WHERE_IDX_ONLY) == 0)
					v->AddOp1(OP_Close, tabItem->Cursor);
				if ((ws & WHERE_INDEXED) != 0 && (ws & WHERE_TEMP_INDEX) == 0)
					v->AddOp1(OP_Close, level->IdxCur);
			}

			// If this scan uses an index, make code substitutions to read data from the index in preference to the table. Sometimes, this means
			// the table need never be read from. This is a performance boost, as the vdbe level waits until the table is read before actually
			// seeking the table cursor to the record corresponding to the current position in the index.
			// 
			// Calls to the code generator in between sqlite3WhereBegin and sqlite3WhereEnd will have created code that references the table
			// directly.  This loop scans all that code looking for opcodes that reference the table and converts them into opcodes that
			// reference the index.
			Index *idx = nullptr;
			if (level->Plan.WsFlags & WHERE_INDEXED)
				idx = level->Plan.u.Index;
			else if (level->Plan.WsFlags & WHERE_MULTI_OR)
				idx = level->u.Covidx;
			if (idx && !ctx->MallocFailed)
			{
				int last = v->CurrentAddr();
				int k;
				Vdbe::VdbeOp *op;
				for (k = winfo->TopId, op = v->GetOp(winfo->TopId); k < last; k++, op++)
				{
					if (op->P1 != level->TabCur) continue;
					if (op->Opcode == OP_Column)
					{
						int j;
						for (j = 0; j < idx->Columns.length; j++)
						{
							if (op->P2 == idx->Columns[j])
							{
								op->P2 = j;
								op->P1 = level->IdxCur;
								break;
							}
						}
						_assert((level->Plan.WsFlags & WHERE_IDX_ONLY) == 0 || j < idx->Columns.length);
					}
					else if (op->Opcode == OP_Rowid)
					{
						op->P1 = level->IdxCur;
						op->Opcode = OP_IdxRowid;
					}
				}
			}
		}

		// Final cleanup
		parse->QueryLoops = winfo->SavedNQueryLoop;
		WhereInfoFree(ctx, winfo);
		return;
	}
}