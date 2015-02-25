// select.c
#include "..\VdbeInt.cu.h"

namespace Core
{
	__device__ static void ClearSelect(Context *ctx, Select *p)
	{
		Expr::ListDelete(ctx, p->EList);
		Parse::SrcListDelete(ctx, p->Src);
		Expr::Delete(ctx, p->Where);
		Expr::ListDelete(ctx, p->GroupBy);
		Expr::Delete(ctx, p->Having);
		Expr::ListDelete(ctx, p->OrderBy);
		Select::Delete(ctx, p->Prior);
		Expr::Delete(ctx, p->Limit);
		Expr::Delete(ctx, p->Offset);
	}

	__device__ void Select::DestInit(SelectDest *dest, SRT dest2, int parmId)
	{
		dest->Dest = dest2;
		dest->SDParmId = parmId;
		dest->AffSdst = (AFF)0;
		dest->SdstId = 0;
		dest->Sdsts = 0;
	}

	__device__ Select *Select::New(Parse *parse, ExprList *list, SrcList *src, Expr *where_, ExprList *groupBy, Expr *having, ExprList *orderBy, SF selFlags, Expr *limit, Expr *offset)
	{
		Context *ctx = parse->Ctx;
		Select *newSelect = (Select *)_tagalloc2(ctx, sizeof(*newSelect), true);
		_assert(ctx->MallocFailed || !offset || limit); // OFFSET implies LIMIT
		Select standin;
		if (!newSelect)
		{
			_assert(ctx->MallocFailed);
			newSelect = &standin;
			_memset(newSelect, 0, sizeof(*newSelect));
		}
		if (!list)
			list = Expr::ListAppend(parse, nullptr, Expr::Expr_(ctx, TK_ALL, nullptr));
		newSelect->EList = list;
		if (!src) src = (SrcList *)_tagalloc2(ctx, sizeof(*src), true);
		newSelect->Src = src;
		newSelect->Where = where_;
		newSelect->GroupBy = groupBy;
		newSelect->Having = having;
		newSelect->OrderBy = orderBy;
		newSelect->SelFlags = selFlags;
		newSelect->OP = TK_SELECT;
		newSelect->Limit = limit;
		newSelect->Offset = offset;
		_assert(offset == nullptr || limit != nullptr);
		newSelect->AddrOpenEphms[0] = (::OP)-1;
		newSelect->AddrOpenEphms[1] = (::OP)-1;
		newSelect->AddrOpenEphms[2] = (::OP)-1;
		if (ctx->MallocFailed)
		{
			ClearSelect(ctx, newSelect);
			if (newSelect != &standin) _tagfree(ctx, newSelect);
			newSelect = nullptr;
		}
		else
			_assert(newSelect->Src != nullptr || parse->Errs > 0);
		_assert(newSelect != &standin);
		return newSelect;
	}

	__device__ void Select::Delete(Context *ctx, Select *p)
	{
		if (p)
		{
			ClearSelect(ctx, p);
			_tagfree(ctx, p);
		}
	}

	/// 0123456789 123456789 123456789 123
	__constant__ static const char _keyTexts[] = "naturaleftouterightfullinnercross";
	struct keywords_t
	{
		uint8 I;        // Beginning of keyword text in zKeyText[]
		uint8 Chars;    // Length of the keyword in characters
		JT Code;     // Join type mask
	};
#if __CUDACC__
	__constant__ static keywords_t _keywords[7];
	static const keywords_t h_keywords[7] =
#else
	static const keywords_t _keywords[7] =
#endif
	{
		{ 0,  7, JT_NATURAL                },	// natural
		{ 6,  4, JT_LEFT|JT_OUTER          },	// left
		{ 10, 5, JT_OUTER                  },	// outer
		{ 14, 5, JT_RIGHT|JT_OUTER         },	// right
		{ 19, 4, JT_LEFT|JT_RIGHT|JT_OUTER },	// full
		{ 23, 5, JT_INNER                  },	// inner
		{ 28, 5, JT_INNER|JT_CROSS         },	// cross
	};

	__device__ JT Select::JoinType(Parse *parse, Token *a, Token *b, Token *c)
	{
		Token *alls[3];
		alls[0] = a;
		alls[1] = b;
		alls[2] = b;
		JT jointype = (JT)0;
		for (int i = 0; i < 3 && alls[i]; i++)
		{
			Token *p = alls[i];
			int j;
			for (j = 0; j < _lengthof(_keywords); j++)
			{
				if (p->length == _keywords[j].Chars && !_strncmp((char *)p->data, &_keyTexts[_keywords[j].I], p->length))
				{
					jointype |= _keywords[j].Code;
					break;
				}
			}
			ASSERTCOVERAGE(j == 0 || j == 1 || j == 2 || j == 3 || j == 4 || j == 5 || j == 6);
			if (j >= _lengthof(_keywords))
			{
				jointype |= JT_ERROR;
				break;
			}
		}
		if ((jointype & (JT_INNER | JT_OUTER)) == (JT_INNER | JT_OUTER) || (jointype & JT_ERROR) != 0)
		{
			const char *sp = " ";
			_assert(b);
			if (!c) { sp++; }
			parse->ErrorMsg("unknown or unsupported join type: %T %T%s%T", a, b, sp, c);
			jointype = JT_INNER;
		}
		else if ((jointype & JT_OUTER) != 0 && (jointype & (JT_LEFT | JT_RIGHT)) != JT_LEFT)
		{
			parse->ErrorMsg("RIGHT and FULL OUTER JOINs are not currently supported");
			jointype = JT_INNER;
		}
		return jointype;
	}

	__device__ static int ColumnIndex(Table *table, const char *colName)
	{
		for (int i = 0; i < table->Cols.length; i++)
			if (!_strcmp(table->Cols[i].Name, colName)) return i;
		return -1;
	}

	__device__ static bool TableAndColumnIndex(SrcList *src, int n, const char *colName, int *tableOut, int *colIdOut)
	{
		_assert((!tableOut) == (!colIdOut)); // Both or neither are NULL
		for (int i = 0; i < n; i++)
		{
			int colId = ColumnIndex(src->Ids[i].Table, colName); // Index of column matching zCol
			if (colId >= 0)
			{
				if (tableOut)
				{
					*tableOut = i;
					*colIdOut = colId;
				}
				return true;
			}
		}
		return false;
	}

	__device__ static void AddWhereTerm(Parse *parse, SrcList *src, int leftId, int colLeftId, int rightId, int colRightId, bool isOuterJoin, Expr **where_)
	{
		_assert(leftId < rightId);
		_assert(src->Srcs > rightId);
		_assert(src->Ids[leftId].Table);
		_assert(src->Ids[rightId].Table);

		Context *ctx = parse->Ctx;
		Expr *e1 = Walker::CreateColumnExpr(ctx, src, leftId, colLeftId);
		Expr *e2 = Walker::CreateColumnExpr(ctx, src, rightId, colRightId);

		Expr *eq = Expr::PExpr_(parse, TK_EQ, e1, e2, nullptr);
		if (eq && isOuterJoin)
		{
			ExprSetProperty(eq, EP_FromJoin);
			_assert(!ExprHasAnyProperty(eq, EP_TokenOnly|EP_Reduced));
			ExprSetIrreducible(eq);
			eq->RightJoinTable = (int16)e2->TableId;
		}
		*where_ = Expr::And(ctx, *where_, eq);
	}

	__device__ static void SetJoinExpr(Expr *p, int table)
	{
		while (p)
		{
			ExprSetProperty(p, EP_FromJoin);
			_assert(!ExprHasAnyProperty(p, EP_TokenOnly|EP_Reduced));
			ExprSetIrreducible(p);
			p->RightJoinTable = (int16)table;
			SetJoinExpr(p->Left, table);
			p = p->Right;
		} 
	}

	__device__ bool Select::ProcessJoin(Parse *parse)
	{
		SrcList *src = Src; // All tables in the FROM clause
		SrcList::SrcListItem *left = &src->Ids[0]; // Left table being joined
		SrcList::SrcListItem *right = &left[1]; // Right table being joined
		int j;
		for (int i =0 ; i < src->Srcs-1; i++, right++, left++)
		{
			Table *leftTable = left->Table;
			Table *rightTable = right->Table;

			if (_NEVER(!leftTable || !rightTable)) continue;
			bool isOuter = ((right->Jointype & JT_OUTER) != 0);

			// When the NATURAL keyword is present, add WHERE clause terms for every column that the two tables have in common.
			if (right->Jointype & JT_NATURAL)
			{
				if (right->On || right->Using)
				{
					parse->ErrorMsg("a NATURAL join may not have an ON or USING clause");
					return true;
				}
				for (j = 0; j < rightTable->Cols.length; j++)
				{
					char *name = rightTable->Cols[j].Name; // Name of column in the right table
					int leftId; // Matching left table
					int leftColId; // Matching column in the left table
					if (TableAndColumnIndex(src, i+1, name, &leftId, &leftColId))
						AddWhereTerm(parse, src, leftId, leftColId, i+1, j, isOuter, &Where);
				}
			}

			// Disallow both ON and USING clauses in the same join
			if (right->On && right->Using)
			{
				parse->ErrorMsg("cannot have both ON and USING clauses in the same join");
				return true;
			}

			// Add the ON clause to the end of the WHERE clause, connected by
			if (right->On)
			{
				if (isOuter) SetJoinExpr(right->On, right->Cursor);
				Where = Expr::And(parse->Ctx, Where, right->On);
				right->On = nullptr;
			}

			// Create extra terms on the WHERE clause for each column named in the USING clause.  Example: If the two tables to be joined are 
			// A and B and the USING clause names X, Y, and Z, then add this to the WHERE clause:    A.X=B.X AND A.Y=B.Y AND A.Z=B.Z
			// Report an error if any column mentioned in the USING clause is not contained in both tables to be joined.
			if (right->Using)
			{
				IdList *list = right->Using;
				for (j = 0; j < list->Ids.length; j++)
				{
					char *name = list->Ids[j].Name; // Name of the term in the USING clause
					int leftId; // Table on the left with matching column name
					int leftColId; // Column number of matching column on the left
					int rightColId = ColumnIndex(rightTable, name); // Column number of matching column on the right
					if (rightColId < 0 || !TableAndColumnIndex(src, i+1, name, &leftId, &leftColId))
					{
						parse->ErrorMsg("cannot join using column %s - column not present in both tables", name);
						return true;
					}
					AddWhereTerm(parse, src, leftId, leftColId, i+1, rightColId, isOuter, &Where);
				}
			}
		}
		return false;
	}

	__device__ static void PushOntoSorter(Parse *parse, ExprList *orderBy, Select *select, int regData)
	{
		Vdbe *v = parse->V;
		int exprs = orderBy->Exprs;
		int regBase = Expr::GetTempRange(parse, exprs+2);
		int regRecord = Expr::GetTempReg(parse);
		Expr::CacheClear(parse);
		Expr::CodeExprList(parse, orderBy, regBase, false);
		v->AddOp2(OP_Sequence, orderBy->ECursor, regBase+exprs);
		Expr::CodeMove(parse, regData, regBase+exprs+1, 1);
		v->AddOp3(OP_MakeRecord, regBase, exprs + 2, regRecord);
		::OP op =  (select->SelFlags & SF_UseSorter ? OP_SorterInsert : OP_IdxInsert);
		v->AddOp2(op, orderBy->ECursor, regRecord);
		Expr::ReleaseTempReg(parse, regRecord);
		Expr::ReleaseTempRange(parse, regBase, exprs+2);
		if (select->LimitId)
		{
			int limitId = (select->OffsetId ? select->OffsetId+1 : select->LimitId);
			int addr1 = v->AddOp1(OP_IfZero, limitId);
			v->AddOp2(OP_AddImm, limitId, -1);
			int addr2 = v->AddOp0(OP_Goto);
			v->JumpHere(addr1);
			v->AddOp1(OP_Last, orderBy->ECursor);
			v->AddOp1(OP_Delete, orderBy->ECursor);
			v->JumpHere(addr2);
		}
	}

	__device__ static void CodeOffset(Vdbe *v, Select *p, int continueId) 
	{
		if (p->OffsetId && continueId != 0)
		{
			v->AddOp2(OP_AddImm, p->OffsetId, -1);
			int addr = v->AddOp1(OP_IfNeg, p->OffsetId);
			v->AddOp2(OP_Goto, 0, continueId);
			v->Comment("skip OFFSET records");
			v->JumpHere(addr);
		}
	}

	__device__ static void CodeDistinct(Parse *parse, int table, int addrRepeatId, int n, int memId)
	{
		Vdbe *v = parse->V;
		int r1 = Expr::GetTempReg(parse);
		v->AddOp4Int(OP_Found, table, addrRepeatId, memId, n);
		v->AddOp3(OP_MakeRecord, memId, n, r1);
		v->AddOp2(OP_IdxInsert, table, r1);
		Expr::ReleaseTempReg(parse, r1);
	}

#ifndef OMIT_SUBQUERY
	__device__ static bool CheckForMultiColumnSelectError(Parse *parse, SelectDest *dest, int exprs)
	{
		SRT dest2 = dest->Dest;
		if (exprs > 1 && (dest2 == SRT_Mem || dest2 == SRT_Set))
		{
			parse->ErrorMsg("only a single result allowed for a SELECT that is part of an expression");
			return true;
		}
		return false;
	}
#endif

	struct DistinctCtx
	{
		bool IsTnct;				// True if the DISTINCT keyword is present
		WHERE_DISTINCT TnctType;	// One of the WHERE_DISTINCT_* operators
		int TableTnct;				// Ephemeral table used for DISTINCT processing
		int AddrTnct;				// Address of OP_OpenEphemeral opcode for tabTnct
	};

	__device__ static void SelectInnerLoop(Parse *parse, Select *p, ExprList *list, int srcTable, int columns, ExprList *orderBy, DistinctCtx *distinct, SelectDest *dest, int continueId, int breakId)
	{
		Vdbe *v = parse->V;

		_assert(v);
		if (_NEVER(!v)) return;
		_assert(list);
		WHERE_DISTINCT hasDistinct = (distinct ? distinct->TnctType : WHERE_DISTINCT_NOOP); // True if the DISTINCT keyword is present
		if (!orderBy && !hasDistinct)
			CodeOffset(v, p, continueId);

		// Pull the requested columns.
		int resultCols = (columns > 0 ? columns : list->Exprs); // Number of result columns
		if (dest->Sdsts == 0)
		{
			dest->SdstId = parse->Mems+1;
			dest->Sdsts = resultCols;
			parse->Mems += resultCols;
		}
		else
			_assert(dest->Sdsts == resultCols);
		int regResult = dest->SdstId; // Start of memory holding result set
		SRT dest2 = dest->Dest; // How to dispose of results
		int i;
		if (columns > 0)
			for (i = 0; i < columns; i++)
				v->AddOp3(OP_Column, srcTable, i, regResult+i);
		else if (dest2 != SRT_Exists)
		{
			// If the destination is an EXISTS(...) expression, the actual values returned by the SELECT are not required.
			Expr::CacheClear(parse);
			Expr::CodeExprList(parse, list, regResult, dest2 == SRT_Output);
		}
		columns = resultCols;

		// If the DISTINCT keyword was present on the SELECT statement and this row has been seen before, then do not make this row part of the result.
		if (hasDistinct)
		{
			_assert(list);
			_assert(list->Exprs == columns);
			switch (distinct->TnctType)
			{
			case WHERE_DISTINCT_ORDERED: {
				// Allocate space for the previous row
				int regPrev = parse->Mems+1; // Previous row content
				parse->Mems += columns;

				// Change the OP_OpenEphemeral coded earlier to an OP_Null sets the MEM_Cleared bit on the first register of the
				// previous value.  This will cause the OP_Ne below to always fail on the first iteration of the loop even if the first
				// row is all NULLs.
				v->ChangeToNoop(distinct->AddrTnct);
				Vdbe::VdbeOp *op = v->GetOp(distinct->AddrTnct); // No longer required OpenEphemeral instr.
				op->Opcode = OP_Null;
				op->P1 = 1;
				op->P2 = regPrev;

				int jumpId = v->CurrentAddr() + columns; // Jump destination
				for (i = 0; i < columns; i++)
				{
					CollSeq *coll = list->Ids[i].Expr->CollSeq(parse);
					if (i < columns-1)
						v->AddOp3(OP_Ne, regResult+i, jumpId, regPrev+i);
					else
						v->AddOp3(OP_Eq, regResult+i, continueId, regPrev+i);
					v->ChangeP4(-1, (const char *)coll, Vdbe::P4T_COLLSEQ);
					v->ChangeP5(AFF_BIT_NULLEQ);
				}
				_assert(v->CurrentAddr() == jumpId);
				v->AddOp3(OP_Copy, regResult, regPrev, columns-1);
				break; }
			case WHERE_DISTINCT_UNIQUE: {
				v->ChangeToNoop(distinct->AddrTnct);
				break; }
			default: {
				_assert(distinct->TnctType == WHERE_DISTINCT_UNORDERED);
				CodeDistinct(parse, distinct->TableTnct, continueId, columns, regResult);
				break; }
			}
			if (!orderBy)
				CodeOffset(v, p, continueId);
		}

		int parmId = dest->SDParmId; // First argument to disposal method
		switch (dest2)
		{
#ifndef OMIT_COMPOUND_SELECT
		case SRT_Union: {
			// In this mode, write each query result to the key of the temporary table iParm.
			int r1 = Expr::GetTempReg(parse);
			v->AddOp3(OP_MakeRecord, regResult, columns, r1);
			v->AddOp2(OP_IdxInsert, parmId, r1);
			Expr::ReleaseTempReg(parse, r1);
			break; }
		case SRT_Except: {
			// Construct a record from the query result, but instead of saving that record, use it as a key to delete elements from
			// the temporary table iParm.
			v->AddOp3(OP_IdxDelete, parmId, regResult, columns);
			break; }
#endif
		case SRT_Table:
		case SRT_EphemTab: {
			// Store the result as data using a unique key.
			int r1 = Expr::GetTempReg(parse);
			ASSERTCOVERAGE(dest2 == SRT_Table);
			ASSERTCOVERAGE(dest2 == SRT_EphemTab);
			v->AddOp3(OP_MakeRecord, regResult, columns, r1);
			if (orderBy)
				PushOntoSorter(parse, orderBy, p, r1);
			else
			{
				int r2 = Expr::GetTempReg(parse);
				v->AddOp2(OP_NewRowid, parmId, r2);
				v->AddOp3(OP_Insert, parmId, r1, r2);
				v->ChangeP5(Vdbe::OPFLAG_APPEND);
				Expr::ReleaseTempReg(parse, r2);
			}
			Expr::ReleaseTempReg(parse, r1);
			break; }
#ifndef OMIT_SUBQUERY
		case SRT_Set: {
			// If we are creating a set for an "expr IN (SELECT ...)" construct, then there should be a single item on the stack.  Write this
			// item into the set table with bogus data.
			_assert(columns == 1);
			dest->AffSdst = list->Ids[0].Expr->CompareAffinity(dest->AffSdst);
			// At first glance you would think we could optimize out the ORDER BY in this case since the order of entries in the set
			// does not matter.  But there might be a LIMIT clause, in which case the order does matter
			if (orderBy)
				PushOntoSorter(parse, orderBy, p, regResult);
			else
			{
				int r1 = Expr::GetTempReg(parse);
				v->AddOp4(OP_MakeRecord, regResult, 1, r1, (const char *)&dest->AffSdst, 1);
				Expr::CacheAffinityChange(parse, regResult, 1);
				v->AddOp2(OP_IdxInsert, parmId, r1);
				Expr::ReleaseTempReg(parse, r1);
			}
			break; }
		case SRT_Exists: {
			// If any row exist in the result set, record that fact and abort.
			v->AddOp2(OP_Integer, 1, parmId);
			// The LIMIT clause will terminate the loop for us
			break; }
		case SRT_Mem: {
			// If this is a scalar select that is part of an expression, then store the results in the appropriate memory cell and break out
			// of the scan loop.
			_assert(columns == 1);
			if (orderBy)
				PushOntoSorter(parse, orderBy, p, regResult);
			else
				Expr::CodeMove(parse, regResult, parmId, 1);
			// The LIMIT clause will jump out of the loop for us
			break; }
#endif
		case SRT_Coroutine:
		case SRT_Output: {
			// Send the data to the callback function or to a subroutine.  In the case of a subroutine, the subroutine itself is responsible for
			// popping the data from the stack.
			ASSERTCOVERAGE(dest2 == SRT_Coroutine);
			ASSERTCOVERAGE(dest2 == SRT_Output);
			if (orderBy)
			{
				int r1 = Expr::GetTempReg(parse);
				v->AddOp3(OP_MakeRecord, regResult, columns, r1);
				PushOntoSorter(parse, orderBy, p, r1);
				Expr::ReleaseTempReg(parse, r1);
			}
			else if (dest2 == SRT_Coroutine)
				v->AddOp1(OP_Yield, dest->SDParmId);
			else
			{
				v->AddOp2(OP_ResultRow, regResult, columns);
				Expr::CacheAffinityChange(parse, regResult, columns);
			}
			break; }
#if !defined(OMIT_TRIGGER)
		default: {
			// Discard the results.  This is used for SELECT statements inside the body of a TRIGGER.  The purpose of such selects is to call
			// user-defined functions that have side effects.  We do not care about the actual results of the select.
			_assert(dest2 == SRT_Discard);
			break; }
#endif
		}

		// Jump to the end of the loop if the LIMIT is reached.  Except, if there is a sorter, in which case the sorter has already limited the output for us.
		if (!orderBy && p->LimitId)
			v->AddOp3(OP_IfZero, p->LimitId, breakId, -1);
	}

	__device__ static KeyInfo *KeyInfoFromExprList(Parse *parse, ExprList *list)
	{
		Context *ctx = parse->Ctx;
		int exprs = list->Exprs;
		KeyInfo *info = (KeyInfo *)_tagalloc2(ctx, sizeof(*info) + exprs*(sizeof(CollSeq*)+1), true);
		if (info)
		{
			info->SortOrders = (SO *)&info->Colls[exprs];
			info->Fields = (uint16)exprs;
			info->Encode = CTXENCODE(ctx);
			info->Ctx = ctx;
			int i;
			ExprList::ExprListItem *item;
			for (i = 0, item = list->Ids; i < exprs; i++, item++)
			{
				CollSeq *coll = item->Expr->CollSeq(parse);
				if (!coll)
					coll = ctx->DefaultColl;
				info->Colls[i] = coll;
				info->SortOrders[i] = item->SortOrder;
			}
		}
		return info;
	}

#pragma region Explain

#ifndef OMIT_COMPOUND_SELECT
	__device__ static const char *SelectOpName(TK id)
	{
		switch (id)
		{
		case TK_ALL: return "UNION ALL";
		case TK_INTERSECT: return "INTERSECT";
		case TK_EXCEPT: return "EXCEPT";
		default: return "UNION";
		}
	}
#endif

#ifndef OMIT_EXPLAIN
	__device__ static void ExplainTempTable(Parse *parse, const char *usage_)
	{
		if (parse->Explain == 2)
		{
			Vdbe *v = parse->V;
			char *msg = _mtagprintf(parse->Ctx, "USE TEMP B-TREE FOR %s", usage_);
			v->AddOp4(OP_Explain, parse->SelectId, 0, 0, msg, Vdbe::P4T_DYNAMIC);
		}
	}
#define ExplainSetInteger(a, b) a = b
#else
#define ExplainTempTable(y, z)
#define ExplainSetInteger(y, z)
#endif

#if !defined(OMIT_EXPLAIN) && !defined(OMIT_COMPOUND_SELECT)
	__device__ static void ExplainComposite(Parse *parse, TK op, int sub1Id, int sub2Id, bool useTmp)
	{
		_assert(op == TK_UNION || op == TK_EXCEPT || op == TK_INTERSECT || op == TK_ALL);
		if (parse->Explain == 2)
		{
			Vdbe *v = parse->V;
			char *msg = _mtagprintf(parse->Ctx, "COMPOUND SUBQUERIES %d AND %d %s(%s)", sub1Id, sub2Id, (useTmp?"USING TEMP B-TREE ":""), SelectOpName(op));
			v->AddOp4(OP_Explain, parse->SelectId, 0, 0, msg, Vdbe::P4T_DYNAMIC);
		}
	}
#else
#define ExplainComposite(v, w, x, y, z)
#endif

#pragma endregion

	__device__ static void GenerateSortTail(Parse *parse, Select *p, Vdbe *v, int columns, SelectDest *dest)
	{
		int addrBreak = v->MakeLabel(); // Jump here to exit loop
		int addrContinue = v->MakeLabel(); // Jump here for next cycle
		ExprList *orderBy = p->OrderBy;
		SRT dest2 = dest->Dest;
		int parmId = dest->SDParmId;

		int tabId = orderBy->ECursor;
		int regRow = Expr::GetTempReg(parse);
		int pseudoTab = 0;
		int regRowid;
		if (dest2 == SRT_Output || dest2 == SRT_Coroutine)
		{
			pseudoTab = parse->Tabs++;
			v->AddOp3(OP_OpenPseudo, pseudoTab, regRow, columns);
			regRowid = 0;
		}
		else
			regRowid = Expr::GetTempReg(parse);
		int addr;
		if (p->SelFlags & SF_UseSorter)
		{
			int regSortOut = ++parse->Mems;
			int ptab2 = parse->Tabs++;
			v->AddOp3(OP_OpenPseudo, ptab2, regSortOut, orderBy->Exprs+2);
			addr = 1 + v->AddOp2(OP_SorterSort, tabId, addrBreak);
			CodeOffset(v, p, addrContinue);
			v->AddOp2(OP_SorterData, tabId, regSortOut);
			v->AddOp3(OP_Column, ptab2, orderBy->Exprs+1, regRow);
			v->ChangeP5(Vdbe::OPFLAG_CLEARCACHE);
		}
		else
		{
			addr = 1 + v->AddOp2(OP_Sort, tabId, addrBreak);
			CodeOffset(v, p, addrContinue);
			v->AddOp3(OP_Column, tabId, orderBy->Exprs+1, regRow);
		}
		switch (dest2)
		{
		case SRT_Table:
		case SRT_EphemTab: {
			ASSERTCOVERAGE(dest2 == SRT_Table);
			ASSERTCOVERAGE(dest2 == SRT_EphemTab);
			v->AddOp2(OP_NewRowid, parmId, regRowid);
			v->AddOp3(OP_Insert, parmId, regRow, regRowid);
			v->ChangeP5(Vdbe::OPFLAG_APPEND);
			break; }
#ifndef OMIT_SUBQUERY
		case SRT_Set: {
			_assert(columns == 1);
			v->AddOp4(OP_MakeRecord, regRow, 1, regRowid, (const char *)&dest->AffSdst, 1);
			Expr::CacheAffinityChange(parse, regRow, 1);
			v->AddOp2(OP_IdxInsert, parmId, regRowid);
			break; }
		case SRT_Mem: {
			_assert(columns == 1);
			Expr::CodeMove(parse, regRow, parmId, 1);
			// The LIMIT clause will terminate the loop for us
			break; }
#endif
		default: {
			_assert(dest2 == SRT_Output || dest2 == SRT_Coroutine); 
			ASSERTCOVERAGE(dest2 == SRT_Output);
			ASSERTCOVERAGE(dest2 == SRT_Coroutine);
			for (int i = 0; i < columns; i++)
			{
				_assert(regRow != dest->SdstId+i);
				v->AddOp3(OP_Column, pseudoTab, i, dest->SdstId+i);
				if (i == 0)
					v->ChangeP5(Vdbe::OPFLAG_CLEARCACHE);
			}
			if (dest2 == SRT_Output)
			{
				v->AddOp2(OP_ResultRow, dest->SdstId, columns);
				Expr::CacheAffinityChange(parse, dest->SdstId, columns);
			}
			else
				v->AddOp1(OP_Yield, dest->SDParmId);
			break; }
		}
		Expr::ReleaseTempReg(parse, regRow);
		Expr::ReleaseTempReg(parse, regRowid);

		// The bottom of the loop
		v->ResolveLabel(addrContinue);
		if (p->SelFlags & SF_UseSorter)
			v->AddOp2(OP_SorterNext, tabId, addr);
		else
			v->AddOp2(OP_Next, tabId, addr);
		v->ResolveLabel(addrBreak);
		if (dest2 == SRT_Output || dest2 == SRT_Coroutine)
			v->AddOp2(OP_Close, pseudoTab, 0);
	}

	__device__ static const char *ColumnType(NameContext *nc, Expr *expr, const char **originDbNameOut, const char **originTableNameOut, const char **originColumnNameOut)
	{
		char const *typeName = nullptr;
		char const *originDbName = nullptr;
		char const *originTableName = nullptr;
		char const *originColumnName = nullptr;
		int j;
		if (_NEVER(!expr) || !nc->SrcList) return nullptr;

		switch (expr->OP)
		{
		case TK_AGG_COLUMN:
		case TK_COLUMN: {
			// The expression is a column. Locate the table the column is being extracted from in NameContext.pSrcList. This table may be real
			// database table or a subquery.
			Table *table = nullptr; // Table structure column is extracted from
			Select *s = nullptr; // Select the column is extracted from
			int colId = expr->ColumnIdx; // Index of column in pTab
			ASSERTCOVERAGE(expr->OP == TK_AGG_COLUMN);
			ASSERTCOVERAGE(expr->OP == TK_COLUMN);
			while (nc && !table)
			{
				SrcList *tabList = nc->SrcList;
				for (j = 0; j < tabList->Srcs && tabList->Ids[j].Cursor != expr->TableId; j++);
				if (j < tabList->Srcs)
				{
					table = tabList->Ids[j].Table;
					s = tabList->Ids[j].Select;
				}
				else
					nc = nc->Next;
			}
			if (!table)
			{
				// At one time, code such as "SELECT new.x" within a trigger would cause this condition to run.  Since then, we have restructured how
				// trigger code is generated and so this condition is no longer possible. However, it can still be true for statements like
				// the following:
				//
				//   CREATE TABLE t1(col INTEGER);
				//   SELECT (SELECT t1.col) FROM FROM t1;
				//
				// when columnType() is called on the expression "t1.col" in the sub-select. In this case, set the column type to NULL, even
				// though it should really be "INTEGER".
				//
				// This is not a problem, as the column type of "t1.col" is never used. When columnType() is called on the expression 
				// "(SELECT t1.col)", the correct type is returned (see the TK_SELECT branch below.
				break;
			}
			_assert(table && expr->Table == table);
			if (s)
			{
				// The "table" is actually a sub-select or a view in the FROM clause of the SELECT statement. Return the declaration type and origin
				// data for the result-set column of the sub-select.
				if (colId >= 0 && _ALWAYS(colId < s->EList->Exprs))
				{
					// If colId is less than zero, then the expression requests the rowid of the sub-select or view. This expression is legal (see 
					// test case misc2.2.2) - it always evaluates to NULL.
					NameContext sNC;
					Expr *p = s->EList->Ids[colId].Expr;
					sNC.SrcList = s->Src;
					sNC.Next = nc;
					sNC.Parse = nc->Parse;
					typeName = ColumnType(&sNC, p, &originDbName, &originTableName, &originColumnName); 
				}
			}
			else if (_ALWAYS(table->Schema))
			{
				// A real table
				_assert(!s);
				if (colId < 0) colId = table->PKey;
				_assert(colId == -1 || (colId >= 0 && colId < table->Cols.length));
				if (colId < 0)
				{
					typeName = "INTEGER";
					originColumnName = "rowid";
				}
				else
				{
					typeName = table->Cols[colId].Type;
					originColumnName = table->Cols[colId].Name;
				}
				originTableName = table->Name;
				if (nc->Parse)
				{
					Context *ctx = nc->Parse->Ctx;
					int db = Prepare::SchemaToIndex(ctx, table->Schema);
					originDbName = ctx->DBs[db].Name;
				}
			}
			break; }
#ifndef OMIT_SUBQUERY
		case TK_SELECT: {
			// The expression is a sub-select. Return the declaration type and origin info for the single column in the result set of the SELECT statement.
			NameContext sNC;
			Select *s = expr->x.Select;
			Expr *p = s->EList->Ids[0].Expr;
			_assert(ExprHasProperty(expr, EP_xIsSelect));
			sNC.SrcList = s->Src;
			sNC.Next = nc;
			sNC.Parse = nc->Parse;
			typeName = ColumnType(&sNC, p, &originDbName, &originTableName, &originColumnName); 
			break; }
#endif
		}

		if (originDbNameOut)
		{
			_assert(originTableNameOut && originColumnNameOut);
			*originDbNameOut = originDbName;
			*originTableNameOut = originTableName;
			*originColumnNameOut = originColumnName;
		}
		return typeName;
	}

	__device__ static void GenerateColumnTypes(Parse *parse, SrcList *tabList, ExprList *list)
	{
#ifndef OMIT_DECLTYPE
		Vdbe *v = parse->V;
		NameContext sNC;
		sNC.SrcList = tabList;
		sNC.Parse = parse;
		for (int i = 0; i < list->Exprs; i++)
		{
			Expr *p = list->Ids[i].Expr;
			const char *typeName;
#ifdef ENABLE_COLUMN_METADATA
			const char *origDbName = nullptr;
			const char *origTableName = nullptr;
			const char *origColumnName = nullptr;
			typeName = ColumnType(&sNC, p, &origDbName, &origTableName, &origColumnName);

			// The vdbe must make its own copy of the column-type and other column specific strings, in case the schema is reset before this
			// virtual machine is deleted.
			v->SetColName(i, COLNAME_DATABASE, origDbName, DESTRUCTOR_TRANSIENT);
			v->SetColName(i, COLNAME_TABLE, origTableName, DESTRUCTOR_TRANSIENT);
			v->SetColName(i, COLNAME_COLUMN, origColumnName, DESTRUCTOR_TRANSIENT);
#else
			typeName = ColumnType(&sNC, p, nullptr, nullptr, nullptr);
#endif
			v->SetColName(i, COLNAME_DECLTYPE, typeName, DESTRUCTOR_TRANSIENT);
		}
#endif
	}

	__device__ static void GenerateColumnNames(Parse *parse, SrcList *tabList, ExprList *list)
	{
#ifndef OMIT_EXPLAIN
		// If this is an EXPLAIN, skip this step
		if (parse->Explain)
			return;
#endif
		Vdbe *v = parse->V;
		Context *ctx = parse->Ctx;
		if (parse->ColNamesSet || _NEVER(!v) || ctx->MallocFailed) return;
		parse->ColNamesSet = 1;
		bool fullNames = ((ctx->Flags & Context::FLAG_FullColNames) != 0);
		bool shortNames = ((ctx->Flags & Context::FLAG_ShortColNames) != 0);
		v->SetNumCols(list->Exprs);
		for (int i = 0; i < list->Exprs; i++)
		{
			Expr *p = list->Ids[i].Expr;
			if (_NEVER(!p)) continue;
			if (list->Ids[i].Name)
			{
				char *name = list->Ids[i].Name;
				v->SetColName(i, COLNAME_NAME, name, DESTRUCTOR_TRANSIENT);
			}
			else if ((p->OP == TK_COLUMN || p->OP == TK_AGG_COLUMN) && tabList)
			{
				int colId = p->ColumnIdx;
				int j;
				for (j = 0; _ALWAYS(j < tabList->Srcs); j++)
					if (tabList->Ids[j].Cursor == p->TableId) break;
				_assert(j < tabList->Srcs);
				Table *table = tabList->Ids[j].Table;
				if (colId < 0) colId = table->PKey;
				_assert(colId == -1 || (colId >= 0 && colId < table->Cols.length));
				char *colName = (colId < 0 ? "rowid" : table->Cols[colId].Name);
				if (!shortNames && !fullNames)
					v->SetColName(i, COLNAME_NAME, _tagstrdup(ctx, list->Ids[i].Span), DESTRUCTOR_DYNAMIC);
				else if (fullNames)
				{
					char *name = _mtagprintf(ctx, "%s.%s", table->Name, colName);
					v->SetColName(i, COLNAME_NAME, name, DESTRUCTOR_DYNAMIC);
				}
				else
					v->SetColName(i, COLNAME_NAME, colName, DESTRUCTOR_TRANSIENT);
			}
			else
				v->SetColName(i, COLNAME_NAME, _tagstrdup(ctx, list->Ids[i].Span), DESTRUCTOR_DYNAMIC);
		}
		GenerateColumnTypes(parse, tabList, list);
	}

	__device__ static RC SelectColumnsFromExprList(Parse *parse, ExprList *list, int16 *colsLengthOut, Column **colsOut)
	{
		Context *ctx = parse->Ctx; // Database connection
		int j;

		int colsLength; // Number of columns in the result set
		Column *cols; // For looping over result columns
		if (list)
		{
			colsLength = list->Exprs;
			cols = (Column *)_tagalloc2(ctx, sizeof(cols[0])*colsLength, true);
			ASSERTCOVERAGE(cols == nullptr);
		}
		else
		{
			colsLength = 0;
			cols = nullptr;
		}
		*colsLengthOut = colsLength;
		*colsOut = cols;

		int i;
		Column *col; // For looping over result columns
		for (i = 0, col = cols; i < colsLength; i++, col++)
		{
			// Get an appropriate name for the column
			Expr *p = list->Ids[i].Expr->SkipCollate(); // Expression for a single result column
			char *name; // Column name
			if ((name = list->Ids[i].Name) != nullptr)
				name = _tagstrdup(ctx, name); // If the column contains an "AS <name>" phrase, use <name> as the name 
			else
			{
				Expr *colExpr = p; // The expression that is the result column name
				while (colExpr->OP == TK_DOT)
				{
					colExpr = colExpr->Right;
					_assert(colExpr != nullptr);
				}
				if (colExpr->OP == TK_COLUMN && _ALWAYS(colExpr->Table != nullptr))
				{
					// For columns use the column name name
					int colId = colExpr->ColumnIdx;
					Table *table = colExpr->Table; // Table associated with this expression
					if (colId < 0) colId = table->PKey;
					name = _mtagprintf(ctx, "%s", (col >= 0 ? table->Cols[colId].Name : "rowid"));
				}
				else if (colExpr->OP == TK_ID)
				{
					_assert(!ExprHasProperty(colExpr, EP_IntValue));
					name = _mtagprintf(ctx, "%s", colExpr->u.Token);
				}
				else
					name = _mtagprintf(ctx, "%s", list->Ids[i].Span); // Use the original text of the column expression as its name
			}
			if (ctx->MallocFailed)
			{
				_tagfree(ctx, name);
				break;
			}

			// Make sure the column name is unique.  If the name is not unique, append a integer to the name so that it becomes unique.
			int nameLength = _strlen30(name); // Size of name in zName[]
			int cnt; // Index added to make the name unique
			for (j = cnt = 0; j < i; j++)
			{
				if (!_strcmp(cols[j].Name, name))
				{
					int k;
					for (k = nameLength-1; k > 1 && _isdigit(name[k]); k--) { }
					if (name[k] == ':') nameLength = k;
					name[nameLength] = 0;
					char *newName = _mtagprintf(ctx, "%s:%d", name, ++cnt);
					_tagfree(ctx, name);
					name = newName;
					j = -1;
					if (!name) break;
				}
			}
			col->Name = name;
		}
		if (ctx->MallocFailed)
		{
			for (j = 0; j < i; j++)
				_tagfree(ctx, cols[j].Name);
			_tagfree(ctx, cols);
			*colsOut = nullptr;
			*colsLengthOut = 0;
			return RC_NOMEM;
		}
		return RC_OK;
	}

	__device__ static void SelectAddColumnTypeAndCollation(Parse *parse, int colsLength, Column *cols, Select *select)
	{
		Context *ctx = parse->Ctx;
		_assert(select!=0);
		_assert((select->SelFlags & SF_Resolved) != 0);
		_assert(colsLength == select->EList->Exprs || ctx->MallocFailed);
		if (ctx->MallocFailed) return;
		NameContext sNC;
		_memset(&sNC, 0, sizeof(sNC));
		sNC.SrcList = select->Src;
		ExprList::ExprListItem *ids = select->EList->Ids;
		int i;
		Column *col;
		for (i = 0, col = cols; i < colsLength; i++, col++)
		{
			Expr *p = ids[i].Expr;
			col->Type = _tagstrdup(ctx, ColumnType(&sNC, p, nullptr, nullptr, nullptr));
			col->Affinity = p->Affinity();
			if (col->Affinity == 0) col->Affinity = AFF_NONE;
			CollSeq *coll = p->CollSeq(parse);
			if (coll)
				col->Coll = _tagstrdup(ctx, coll->Name);
		}
	}

	__device__ Table *Select::ResultSetOfSelect(Parse *parse)
	{
		Context *ctx = parse->Ctx;
		Context::FLAG savedFlags = ctx->Flags;
		ctx->Flags &= ~Context::FLAG_FullColNames;
		ctx->Flags |= Context::FLAG_ShortColNames;
		Select *select = this;
		select->Prep(parse, nullptr);
		if (parse->Errs) return nullptr;
		while (select->Prior) select = select->Prior;
		ctx->Flags = savedFlags;
		Table *table = (Table *)_tagalloc2(ctx, sizeof(Table), true);
		if (!table)
			return nullptr;
		// The sqlite3ResultSetOfSelect() is only used n contexts where lookaside is disabled
		_assert(!ctx->Lookaside.Enabled);
		table->Refs = 1;
		table->Name = nullptr;
		table->RowEst = 1000000;
		SelectColumnsFromExprList(parse, select->EList, &table->Cols.length, &table->Cols.data);
		SelectAddColumnTypeAndCollation(parse, table->Cols.length, table->Cols.data, select);
		table->PKey = -1;
		if (ctx->MallocFailed)
		{
			Parse::DeleteTable(ctx, table);
			return nullptr;
		}
		return table;
	}

	__device__ Vdbe *Parse::GetVdbe()
	{
		Vdbe *v = V;
		if (!v)
		{
			v = V = Vdbe::Create(Ctx);
#ifndef OMIT_TRACE
			if (v)
				v->AddOp0(OP_Trace);
#endif
		}
		return v;
	}

	__device__ static void ComputeLimitRegisters(Parse *parse, Select *p, int breakId)
	{
		int limitId = 0;
		int offsetId;
		if (p->LimitId) return;

		// "LIMIT -1" always shows all rows.  There is some contraversy about what the correct behavior should be.
		// The current implementation interprets "LIMIT 0" to mean no rows.
		Expr::CacheClear(parse);
		_assert(!p->Offset || p->Limit);
		if (p->Limit)
		{
			p->LimitId = limitId = ++parse->Mems;
			Vdbe *v = parse->GetVdbe();
			if (_NEVER(!v)) return; // VDBE should have already been allocated
			int n;
			if (p->Limit->IsInteger(&n))
			{
				v->AddOp2(OP_Integer, n, limitId);
				v->Comment("LIMIT counter");
				if (n == 0)
					v->AddOp2(OP_Goto, 0, breakId);
				else if (p->SelectRows > (double)n)
					p->SelectRows = (double)n;
			}
			else
			{
				Expr::Code(parse, p->Limit, limitId);
				v->AddOp1(OP_MustBeInt, limitId);
				v->Comment("LIMIT counter");
				v->AddOp2(OP_IfZero, limitId, breakId);
			}
			if (p->Offset)
			{
				p->OffsetId = offsetId = ++parse->Mems;
				parse->Mems++; // Allocate an extra register for limit+offset
				Expr::Code(parse, p->Offset, offsetId);
				v->AddOp1(OP_MustBeInt, offsetId);
				v->Comment("OFFSET counter");
				int addr1 = v->AddOp1(OP_IfPos, offsetId);
				v->AddOp2(OP_Integer, 0, offsetId);
				v->JumpHere(addr1);
				v->AddOp3(OP_Add, limitId, offsetId, offsetId+1);
				v->Comment("LIMIT+OFFSET");
				addr1 = v->AddOp1(OP_IfPos, limitId);
				v->AddOp2(OP_Integer, -1, offsetId+1);
				v->JumpHere(addr1);
			}
		}
	}

	//MultiSelect

#ifndef OMIT_COMPOUND_SELECT
	__device__ static CollSeq *MultiSelectCollSeq(Parse *parse, Select *p, int colId)
	{
		CollSeq *r = (p->Prior ? MultiSelectCollSeq(parse, p->Prior, colId) : nullptr);
		_assert(colId >= 0);
		if (!r && colId < p->EList->Exprs)
			r = p->EList->Ids[colId].Expr->CollSeq(parse);
		return r;
	}

	__device__ static RC MultiSelectOrderBy(Parse *parse, Select *p, SelectDest *destOut); // Forward reference
	__device__ static RC MultiSelect(Parse *parse, Select *p, SelectDest *dest)
	{
		RC rc = RC_OK; // Success code from a subroutine
#ifndef OMIT_EXPLAIN
		int sub1Id; // EQP id of left-hand query
		int sub2Id; // EQP id of right-hand query
#endif

		// Make sure there is no ORDER BY or LIMIT clause on prior SELECTs.  Only the last (right-most) SELECT in the series may have an ORDER BY or LIMIT.
		_assert(p && p->Prior); // Calling function guarantees this much 
		Context *ctx = parse->Ctx; // Database connection
		Select *prior = p->Prior; // Another SELECT immediately to our left
		_assert(prior->Rightmost != prior);
		_assert(prior->Rightmost == p->Rightmost);
		SelectDest dest2 = *dest; // Alternative data destination
		if (prior->OrderBy)
		{
			parse->ErrorMsg("ORDER BY clause should come after %s not before", SelectOpName(p->OP));
			rc = RC_ERROR;
			goto multi_select_end;
		}
		if (prior->Limit)
		{
			parse->ErrorMsg("LIMIT clause should come after %s not before", SelectOpName(p->OP));
			rc = RC_ERROR;
			goto multi_select_end;
		}

		Vdbe *v = parse->GetVdbe(); // Generate code to this VDBE
		_assert(v); // The VDBE already created by calling function

		// Create the destination temporary table if necessary
		if (dest2.Dest == SRT_EphemTab)
		{
			_assert(p->EList);
			v->AddOp2(OP_OpenEphemeral, dest2.SDParmId, p->EList->Exprs);
			v->ChangeP5(Btree::OPEN_UNORDERED);
			dest2.Dest = SRT_Table;
		}

		// Make sure all SELECTs in the statement have the same number of elements in their result sets.
		_assert(p->EList && prior->EList);
		if (p->EList->Exprs != prior->EList->Exprs)
		{
			if (p->SelFlags & SF_Values)
				parse->ErrorMsg("all VALUES must have the same number of terms");
			else
				parse->ErrorMsg("SELECTs to the left and right of %s do not have the same number of result columns", SelectOpName(p->OP));
			rc = RC_ERROR;
			goto multi_select_end;
		}

		// Compound SELECTs that have an ORDER BY clause are handled separately.
		if (p->OrderBy)
			return MultiSelectOrderBy(parse, p, dest);

		// Generate code for the left and right SELECT statements.
		Select *delete_ = nullptr;  // Chain of simple selects to delete
		switch (p->OP)
		{
		case TK_ALL: {
			int addr = 0;
			int limits;
			_assert(!prior->Limit);
			prior->LimitId = p->LimitId;
			prior->OffsetId = p->OffsetId;
			prior->Limit = p->Limit;
			prior->Offset = p->Offset;
			ExplainSetInteger(sub1Id, parse->NextSelectId);
			rc = Select::Select_(parse, prior, &dest2);
			p->Limit = nullptr;
			p->Offset = nullptr;
			if (rc)
				goto multi_select_end;
			p->Prior = nullptr;
			p->LimitId = prior->LimitId;
			p->OffsetId = prior->OffsetId;
			if (p->LimitId)
			{
				addr = v->AddOp1(OP_IfZero, p->LimitId);
				v->Comment("Jump ahead if LIMIT reached");
			}
			ExplainSetInteger(sub2Id, parse->NextSelectId);
			rc = Select::Select_(parse, p, &dest2);
			ASSERTCOVERAGE(rc != RC_OK);
			delete_ = p->Prior;
			p->Prior = prior;
			p->SelectRows += prior->SelectRows;
			if (prior->Limit && prior->Limit->IsInteger(&limits) && p->SelectRows > (double)limits)
				p->SelectRows = (double)limits;
			if (addr)
				v->JumpHere(addr);
			break; }
		case TK_EXCEPT:
		case TK_UNION: {
			ASSERTCOVERAGE(p->OP == TK_EXCEPT);
			ASSERTCOVERAGE(p->OP == TK_UNION);
			SRT priorOp = SRT_Union; // The SRT_ operation to apply to prior selects
			int unionTab; // Cursor number of the temporary table holding result
			if (dest2.Dest == priorOp && _ALWAYS(!p->Limit && !p->Offset))
			{
				// We can reuse a temporary table generated by a SELECT to our right.
				_assert(p->Rightmost != p); // Can only happen for leftward elements of a 3-way or more compound
				_assert(!p->Limit); // Not allowed on leftward elements
				_assert(!p->Offset); // Not allowed on leftward elements
				unionTab = dest2.SDParmId;
			}
			else
			{
				// We will need to create our own temporary table to hold the intermediate results.
				unionTab = parse->Tabs++;
				_assert(!p->OrderBy);
				int addr = v->AddOp2(OP_OpenEphemeral, unionTab, 0);
				_assert(p->AddrOpenEphms[0] == (::OP)-1);
				p->AddrOpenEphms[0] = addr;
				p->Rightmost->SelFlags |= SF_UsesEphemeral;
				_assert(p->EList);
			}

			// Code the SELECT statements to our left
			_assert(!prior->OrderBy);
			SelectDest uniondest;
			Select::DestInit(&uniondest, priorOp, unionTab);
			ExplainSetInteger(sub1Id, parse->NextSelectId);
			rc = Select::Select_(parse, prior, &uniondest);
			if (rc)
				goto multi_select_end;

			// Code the current SELECT statement
			SRT op = (SRT)0; // One of the SRT_ operations to apply to self
			if (p->OP == TK_EXCEPT)
				op = SRT_Except;
			else
			{
				_assert(p->OP == TK_UNION);
				op = SRT_Union;
			}
			p->Prior = nullptr;
			Expr *limit = p->Limit; // Saved values of p->nLimit and p->nOffset
			p->Limit = nullptr;
			Expr *offset = p->Offset; // Saved values of p->nLimit and p->nOffset
			p->Offset = nullptr;
			uniondest.Dest = op;
			ExplainSetInteger(sub2Id, parse->NextSelectId);
			rc = Select::Select_(parse, p, &uniondest);
			ASSERTCOVERAGE(rc != RC_OK);
			// Query flattening in sqlite3Select() might refill p->pOrderBy. Be sure to delete p->pOrderBy, therefore, to avoid a memory leak.
			Expr::ListDelete(ctx, p->OrderBy);
			delete_ = p->Prior;
			p->Prior = prior;
			p->OrderBy = nullptr;
			if (p->OP == TK_UNION ) p->SelectRows += prior->SelectRows;
			Expr::Delete(ctx, p->Limit);
			p->Limit = limit;
			p->Offset = offset;
			p->LimitId = 0;
			p->OffsetId = 0;

			// Convert the data in the temporary table into whatever form it is that we currently need.
			_assert(unionTab == dest2.SDParmId || dest2.Dest != priorOp);
			if (dest2.Dest != priorOp)
			{
				_assert(p->EList);
				if (dest2.Dest == SRT_Output)
				{
					Select *first = p;
					while (first->Prior) first = first->Prior;
					GenerateColumnNames(parse, nullptr, first->EList);
				}
				int breakId = v->MakeLabel();
				int continueId = v->MakeLabel();
				ComputeLimitRegisters(parse, p, breakId);
				v->AddOp2(OP_Rewind, unionTab, breakId);
				int startId = v->CurrentAddr();
				SelectInnerLoop(parse, p, p->EList, unionTab, p->EList->Exprs, nullptr, nullptr, &dest2, continueId, breakId);
				v->ResolveLabel(continueId);
				v->AddOp2(OP_Next, unionTab, startId);
				v->ResolveLabel(breakId);
				v->AddOp2(OP_Close, unionTab, 0);
			}
			break; }
		default: {
			_assert(p->OP == TK_INTERSECT); 

			// INTERSECT is different from the others since it requires two temporary tables.  Hence it has its own case.  Begin
			// by allocating the tables we will need.
			int tab1 = parse->Tabs++;
			int tab2 = parse->Tabs++;
			_assert(!p->OrderBy);

			int addr = v->AddOp2(OP_OpenEphemeral, tab1, 0);
			_assert(p->AddrOpenEphms[0] == (::OP)-1);
			p->AddrOpenEphms[0] = addr;
			p->Rightmost->SelFlags |= SF_UsesEphemeral;
			_assert(p->EList);

			// Code the SELECTs to our left into temporary table "tab1".
			SelectDest intersectdest;
			Select::DestInit(&intersectdest, SRT_Union, tab1);
			ExplainSetInteger(sub1Id, parse->NextSelectId);
			rc = Select::Select_(parse, prior, &intersectdest);
			if (rc)
				goto multi_select_end;

			// Code the current SELECT into temporary table "tab2"
			addr = v->AddOp2(OP_OpenEphemeral, tab2, 0);
			_assert(p->AddrOpenEphms[1] == (::OP)-1);
			p->AddrOpenEphms[1] = addr;
			p->Prior = nullptr;
			Expr *limit = p->Limit;
			p->Limit = nullptr;
			Expr *offset = p->Offset;
			p->Offset = nullptr;
			intersectdest.SDParmId = tab2;
			ExplainSetInteger(sub2Id, parse->NextSelectId);
			rc = Select::Select_(parse, p, &intersectdest);
			ASSERTCOVERAGE(rc != RC_OK);
			delete_ = p->Prior;
			p->Prior = prior;
			if (p->SelectRows > prior->SelectRows) p->SelectRows = prior->SelectRows;
			Expr::Delete(ctx, p->Limit);
			p->Limit = limit;
			p->Offset = offset;

			// Generate code to take the intersection of the two temporary tables.
			_assert(p->EList);
			if (dest2.Dest == SRT_Output)
			{
				Select *first = p;
				while (first->Prior) first = first->Prior;
				GenerateColumnNames(parse, 0, first->EList);
			}
			int breakId = v->MakeLabel();
			int continueId = v->MakeLabel();
			ComputeLimitRegisters(parse, p, breakId);
			v->AddOp2(OP_Rewind, tab1, breakId);
			int r1 = Expr::GetTempReg(parse);
			int startId = v->AddOp2(OP_RowKey, tab1, r1);
			v->AddOp4Int(OP_NotFound, tab2, continueId, r1, 0);
			Expr::ReleaseTempReg(parse, r1);
			SelectInnerLoop(parse, p, p->EList, tab1, p->EList->Exprs, nullptr, nullptr, &dest2, continueId, breakId);
			v->ResolveLabel(continueId);
			v->AddOp2(OP_Next, tab1, startId);
			v->ResolveLabel(breakId);
			v->AddOp2(OP_Close, tab2, 0);
			v->AddOp2(OP_Close, tab1, 0);
			break; }
		}

		ExplainComposite(parse, p->OP, sub1Id, sub2Id, p->OP != TK_ALL);

		// Compute collating sequences used by temporary tables needed to implement the compound select.
		// Attach the KeyInfo structure to all temporary tables.
		//
		// This section is run by the right-most SELECT statement only.
		// SELECT statements to the left always skip this part.  The right-most
		// SELECT might also skip this part if it has no ORDER BY clause and no temp tables are required.
		if (p->SelFlags & SF_UsesEphemeral)
		{
			_assert(p->Rightmost == p);
			int cols = p->EList->Exprs; // Number of columns in result set
			KeyInfo *keyInfo = (KeyInfo *)_tagalloc2(ctx, sizeof(*keyInfo)+cols*(sizeof(CollSeq*) + 1), true); // Collating sequence for the result set
			if (!keyInfo)
			{
				rc = RC_NOMEM;
				goto multi_select_end;
			}

			keyInfo->Encode = CTXENCODE(ctx);
			keyInfo->Fields = (uint16)cols;

			int i;
			CollSeq **coll; // For looping through keyInfo->aColl[]
			for (i = 0, coll = keyInfo->Colls; i < cols; i++, coll++)
			{
				*coll = MultiSelectCollSeq(parse, p, i);
				if (*coll == nullptr)
					*coll = ctx->DefaultColl;
			}
			keyInfo->SortOrders = (SO *)coll;

			for (Select *loop = p; loop; loop = loop->Prior)
			{
				for (i = 0; i < 2; i++)
				{
					int addr = loop->AddrOpenEphms[i];
					if (addr < 0)
					{
						// If [0] is unused then [1] is also unused.  So we can always safely abort as soon as the first unused slot is found
						_assert(loop->AddrOpenEphms[1] == (::OP)-1);
						break;
					}
					v->ChangeP2(addr, cols);
					v->ChangeP4(addr, (char *)keyInfo, Vdbe::P4T_KEYINFO);
					loop->AddrOpenEphms[i] = (::OP)-1;
				}
			}
			_tagfree(ctx, keyInfo);
		}

multi_select_end:
		dest->SdstId = dest2.SdstId;
		dest->Sdsts = dest2.Sdsts;
		Select::Delete(ctx, delete_);
		return rc;
	}
#endif

	__device__ static int GenerateOutputSubroutine(Parse *parse, Select *p, SelectDest *in_, SelectDest *dest, int regReturn, int regPrev, KeyInfo *keyInfo, Vdbe::P4T p4type, int breakId)
	{
		Vdbe *v = parse->V;
		int addr = v->CurrentAddr();
		int continueId = v->MakeLabel();

		// Suppress duplicates for UNION, EXCEPT, and INTERSECT 
		if (regPrev)
		{
			int j1 = v->AddOp1(OP_IfNot, regPrev);
			int j2 = v->AddOp4(OP_Compare, in_->SdstId, regPrev+1, in_->Sdsts, (char *)keyInfo, p4type);
			v->AddOp3(OP_Jump, j2+2, continueId, j2+2);
			v->JumpHere(j1);
			v->AddOp3(OP_Copy, in_->SdstId, regPrev+1, in_->Sdsts-1);
			v->AddOp2(OP_Integer, 1, regPrev);
		}
		if (parse->Ctx->MallocFailed) return 0;

		// Suppress the first OFFSET entries if there is an OFFSET clause
		CodeOffset(v, p, continueId);

		switch (dest->Dest)
		{
		case SRT_Table:
		case SRT_EphemTab: {
			// Store the result as data using a unique key.
			int r1 = Expr::GetTempReg(parse);
			int r2 = Expr::GetTempReg(parse);
			ASSERTCOVERAGE(dest->Dest == SRT_Table);
			ASSERTCOVERAGE(dest->Dest == SRT_EphemTab);
			v->AddOp3(OP_MakeRecord, in_->SdstId, in_->Sdsts, r1);
			v->AddOp2(OP_NewRowid, dest->SDParmId, r2);
			v->AddOp3(OP_Insert, dest->SDParmId, r1, r2);
			v->ChangeP5(Vdbe::OPFLAG_APPEND);
			Expr::ReleaseTempReg(parse, r2);
			Expr::ReleaseTempReg(parse, r1);
			break; }

#ifndef OMIT_SUBQUERY
		case SRT_Set: {
			// If we are creating a set for an "expr IN (SELECT ...)" construct, then there should be a single item on the stack.  Write this
			// item into the set table with bogus data.
			_assert(in_->Sdsts == 1);
			dest->AffSdst = p->EList->Ids[0].Expr->CompareAffinity(dest->AffSdst);
			int r1 = Expr::GetTempReg(parse);
			v->AddOp4(OP_MakeRecord, in_->SdstId, 1, r1, (const char *)&dest->AffSdst, 1);
			Expr::CacheAffinityChange(parse, in_->SdstId, 1);
			v->AddOp2(OP_IdxInsert, dest->SDParmId, r1);
			Expr::ReleaseTempReg(parse, r1);
			break; }
#if 0 // Never occurs on an ORDER BY query
		case SRT_Exists: {
			// If any row exist in the result set, record that fact and abort.
			v->AddOp2(OP_Integer, 1, dest->SDParmId);
			// The LIMIT clause will terminate the loop for us
			break; }
#endif
		case SRT_Mem: {
			// If this is a scalar select that is part of an expression, then store the results in the appropriate memory cell and break out
			// of the scan loop.
			_assert(in_->Sdsts == 1);
			Expr::CodeMove(parse, in_->SdstId, dest->SDParmId, 1);
			// The LIMIT clause will jump out of the loop for us
			break; }
#endif
		case SRT_Coroutine: {
			// The results are stored in a sequence of registers starting at pDest->iSdst.  Then the co-routine yields.
			if (dest->SdstId == 0)
			{
				dest->SdstId = Expr::GetTempRange(parse, in_->Sdsts);
				dest->Sdsts = in_->Sdsts;
			}
			Expr::CodeMove(parse, in_->SdstId, dest->SdstId, dest->Sdsts);
			v->AddOp1(OP_Yield, dest->SDParmId);
			break; }
		default: {
			// If none of the above, then the result destination must be SRT_Output.  This routine is never called with any other
			// destination other than the ones handled above or SRT_Output.
			//
			// For SRT_Output, results are stored in a sequence of registers.  Then the OP_ResultRow opcode is used to cause sqlite3_step() to
			// return the next row of result.
			_assert(dest->Dest == SRT_Output);
			v->AddOp2(OP_ResultRow, in_->SdstId, in_->Sdsts);
			Expr::CacheAffinityChange(parse, in_->SdstId, in_->Sdsts);
			break; }
		}

		// Jump to the end of the loop if the LIMIT is reached.
		if (p->LimitId)
			v->AddOp3(OP_IfZero, p->LimitId, breakId, -1);

		// Generate the subroutine return
		v->ResolveLabel(continueId);
		v->AddOp1(OP_Return, regReturn);
		return addr;
	}

#ifndef OMIT_COMPOUND_SELECT
	__device__ static RC MultiSelectOrderBy(Parse *parse, Select *p, SelectDest *dest)
	{
		int i, j;
#ifndef OMIT_EXPLAIN
		int sub1Id; // EQP id of left-hand query
		int sub2Id; // EQP id of right-hand query
#endif

		_assert(p->OrderBy);
		Context *ctx = parse->Ctx; // Database connection
		Vdbe *v = parse->V; // Generate code to this VDBE
		_assert(v != 0); // Already thrown the error if VDBE alloc failed
		int labelEnd = v->MakeLabel(); // Label for the end of the overall SELECT stmt
		int labelCmpr = v->MakeLabel(); // Label for the start of the merge algorithm

		// Patch up the ORDER BY clause
		TK op = p->OP;  // One of TK_ALL, TK_UNION, TK_EXCEPT, TK_INTERSECT
		Select *prior = p->Prior; // Another SELECT immediately to our left
		_assert(!prior->OrderBy);
		ExprList *orderBy = p->OrderBy; // The ORDER BY clause
		_assert(orderBy);
		int orderBys = orderBy->Exprs; // Number of terms in the ORDER BY clause

		// For operators other than UNION ALL we have to make sure that the ORDER BY clause covers every term of the result set.  Add
		// terms to the ORDER BY clause as necessary.
		if (op != TK_ALL)
		{
			for (i = 1; !ctx->MallocFailed && i <= p->EList->Exprs; i++)
			{
				ExprList::ExprListItem *item;
				for (j = 0, item = orderBy->Ids; j < orderBys; j++, item++)
				{
					_assert(item->OrderByCol > 0);
					if (item->OrderByCol == i) break;
				}
				if (j == orderBys)
				{
					Expr *newExpr = Expr::Expr_(ctx, TK_INTEGER, nullptr);
					if (!newExpr) return RC_NOMEM;
					newExpr->Flags |= EP_IntValue;
					newExpr->u.I = i;
					orderBy = Expr::ListAppend(parse, orderBy, newExpr);
					if (orderBy) orderBy->Ids[orderBys++].OrderByCol = (uint16)i;
				}
			}
		}

		// Compute the comparison permutation and keyinfo that is used with the permutation used to determine if the next
		// row of results comes from selectA or selectB.  Also add explicit collations to the ORDER BY clause terms so that when the subqueries
		// to the right and the left are evaluated, they use the correct collation.
		KeyInfo *keyMerge; // Comparison information for merging rows
		int *permutes = (int *)_tagalloc(ctx, sizeof(int)*orderBys); // Mapping from ORDER BY terms to result set columns
		if (permutes)
		{
			ExprList::ExprListItem *item;
			for (i = 0, item = orderBy->Ids; i < orderBys; i++, item++)
			{
				_assert(item->OrderByCol > 0 && item->OrderByCol <= p->EList->Exprs);
				permutes[i] = item->OrderByCol - 1;
			}
			keyMerge = (KeyInfo *)_tagalloc(ctx, sizeof(*keyMerge)+orderBys*(sizeof(CollSeq*)+1));
			if (keyMerge)
			{
				keyMerge->SortOrders = (SO *)&keyMerge->Colls[orderBys];
				keyMerge->Fields = (uint16)orderBys;
				keyMerge->Encode = CTXENCODE(ctx);
				for (i = 0; i < orderBys; i++)
				{
					CollSeq *coll;
					Expr *term = orderBy->Ids[i].Expr;
					if (term->Flags & EP_Collate)
						coll = term->CollSeq(parse);
					else
					{
						coll = MultiSelectCollSeq(parse, p, permutes[i]);
						if (!coll) coll = ctx->DefaultColl;
						orderBy->Ids[i].Expr = Expr::AddCollateString(parse, term, coll->Name);
					}
					keyMerge->Colls[i] = coll;
					keyMerge->SortOrders[i] = orderBy->Ids[i].SortOrder;
				}
			}
		}
		else
			keyMerge = nullptr;

		// Reattach the ORDER BY clause to the query.
		p->OrderBy = orderBy;
		prior->OrderBy = Expr::ListDup(ctx, orderBy, 0);

		// Allocate a range of temporary registers and the KeyInfo needed for the logic that removes duplicate result rows when the
		// operator is UNION, EXCEPT, or INTERSECT (but not UNION ALL).
		KeyInfo *keyDup = nullptr; // Comparison information for duplicate removal
		//:_assert(!keyDup); // "Managed" code needs this.  Ticket #3382.
		int regPrev; // A range of registers to hold previous output
		if (op == TK_ALL)
			regPrev = 0;
		else
		{
			int exprs = p->EList->Exprs;
			_assert(orderBys >= exprs || ctx->MallocFailed);
			regPrev = parse->Mems+1;
			parse->Mems += exprs+1;
			v->AddOp2(OP_Integer, 0, regPrev);
			keyDup = (KeyInfo *)_tagalloc2(ctx, sizeof(*keyDup) + exprs*(sizeof(CollSeq*)+1), true);
			if (keyDup)
			{
				keyDup->SortOrders = (SO *)&keyDup->Colls[exprs];
				keyDup->Fields = (uint16)exprs;
				keyDup->Encode = CTXENCODE(ctx);
				for (i = 0; i < exprs; i++)
				{
					keyDup->Colls[i] = MultiSelectCollSeq(parse, p, i);
					keyDup->SortOrders[i] = (SO)0;
				}
			}
		}

		// Separate the left and the right query from one another
		p->Prior = nullptr;
		Walker::ResolveOrderGroupBy(parse, p, p->OrderBy, "ORDER");
		if (!prior->Prior)
			Walker::ResolveOrderGroupBy(parse, prior, prior->OrderBy, "ORDER");

		// Compute the limit registers
		int regLimitA; // Limit register for select-A
		int regLimitB; // Limit register for select-A
		ComputeLimitRegisters(parse, p, labelEnd);
		if (p->LimitId && op == TK_ALL)
		{
			regLimitA = ++parse->Mems;
			regLimitB = ++parse->Mems;
			v->AddOp2(OP_Copy, (p->OffsetId ? p->OffsetId+1 : p->LimitId), regLimitA);
			v->AddOp2(OP_Copy, regLimitA, regLimitB);
		}
		else
			regLimitA = regLimitB = 0;
		Expr::Delete(ctx, p->Limit);
		p->Limit = nullptr;
		Expr::Delete(ctx, p->Offset);
		p->Offset = nullptr;

		int regAddrA = ++parse->Mems; // Address register for select-A coroutine
		int regEofA = ++parse->Mems; // Flag to indicate when select-A is complete
		int regAddrB = ++parse->Mems; // Address register for select-B coroutine
		int regEofB = ++parse->Mems; // Flag to indicate when select-B is complete
		int regOutA = ++parse->Mems; // Address register for the output-A subroutine
		int regOutB = ++parse->Mems; // Address register for the output-B subroutine
		SelectDest destA; // Destination for coroutine A
		SelectDest destB; // Destination for coroutine B
		Select::DestInit(&destA, SRT_Coroutine, regAddrA);
		Select::DestInit(&destB, SRT_Coroutine, regAddrB);

		// Jump past the various subroutines and coroutines to the main merge loop
		int j1 = v->AddOp0(OP_Goto); // Jump instructions that get retargetted
		int addrSelectA = v->CurrentAddr(); // Address of the select-A coroutine

		// Generate a coroutine to evaluate the SELECT statement to the left of the compound operator - the "A" select.
		v->NoopComment("Begin coroutine for left SELECT");
		prior->LimitId = regLimitA;
		ExplainSetInteger(sub1Id, parse->NextSelectId);
		Select::Select_(parse, prior, &destA);
		v->AddOp2(OP_Integer, 1, regEofA);
		v->AddOp1(OP_Yield, regAddrA);
		v->NoopComment("End coroutine for left SELECT");

		// Generate a coroutine to evaluate the SELECT statement on the right - the "B" select
		int addrSelectB = v->CurrentAddr(); // Address of the select-B coroutine
		v->NoopComment("Begin coroutine for right SELECT");
		int savedLimit = p->LimitId; // Saved value of p->iLimit
		int savedOffset = p->OffsetId; // Saved value of p->iOffset
		p->LimitId = regLimitB;
		p->OffsetId = 0;  
		ExplainSetInteger(sub2Id, parse->NextSelectId);
		Select::Select_(parse, p, &destB);
		p->LimitId = savedLimit;
		p->OffsetId = savedOffset;
		v->AddOp2(OP_Integer, 1, regEofB);
		v->AddOp1(OP_Yield, regAddrB);
		v->NoopComment("End coroutine for right SELECT");

		// Generate a subroutine that outputs the current row of the A select as the next output row of the compound select.
		v->NoopComment("Output routine for A");
		int addrOutA = GenerateOutputSubroutine(parse, p, &destA, dest, regOutA, regPrev, keyDup, Vdbe::P4T_KEYINFO_HANDOFF, labelEnd); // Address of the output-A subroutine
		int addrOutB = 0; // Address of the output-B subroutine

		// Generate a subroutine that outputs the current row of the B select as the next output row of the compound select.
		if (op == TK_ALL || op == TK_UNION)
		{
			v->NoopComment("Output routine for B");
			addrOutB = GenerateOutputSubroutine(parse, p, &destB, dest, regOutB, regPrev, keyDup, Vdbe::P4T_KEYINFO_STATIC, labelEnd);
		}

		// Generate a subroutine to run when the results from select A are exhausted and only data in select B remains.
		int addrEofA; // Address of the select-A-exhausted subroutine
		v->NoopComment("eof-A subroutine");
		if (op == TK_EXCEPT || op == TK_INTERSECT)
			addrEofA = v->AddOp2(OP_Goto, 0, labelEnd);
		else
		{  
			addrEofA = v->AddOp2(OP_If, regEofB, labelEnd);
			v->AddOp2(OP_Gosub, regOutB, addrOutB);
			v->AddOp1(OP_Yield, regAddrB);
			v->AddOp2(OP_Goto, 0, addrEofA);
			p->SelectRows += prior->SelectRows;
		}

		// Generate a subroutine to run when the results from select B are exhausted and only data in select A remains.
		int addrEofB; // Address of the select-B-exhausted subroutine
		if (op == TK_INTERSECT)
		{
			addrEofB = addrEofA;
			if (p->SelectRows > prior->SelectRows) p->SelectRows = prior->SelectRows;
		}
		else
		{  
			v->NoopComment("eof-B subroutine");
			addrEofB = v->AddOp2(OP_If, regEofA, labelEnd);
			v->AddOp2(OP_Gosub, regOutA, addrOutA);
			v->AddOp1(OP_Yield, regAddrA);
			v->AddOp2(OP_Goto, 0, addrEofB);
		}

		// Generate code to handle the case of A<B
		int addrAltB; // Address of the A<B subroutine
		v->NoopComment("A-lt-B subroutine");
		addrAltB = v->AddOp2(OP_Gosub, regOutA, addrOutA);
		v->AddOp1(OP_Yield, regAddrA);
		v->AddOp2(OP_If, regEofA, addrEofA);
		v->AddOp2(OP_Goto, 0, labelCmpr);

		// Generate code to handle the case of A==B
		int addrAeqB; // Address of the A==B subroutine
		if (op == TK_ALL)
		{
			addrAeqB = addrAltB;
		}
		else if (op == TK_INTERSECT)
		{
			addrAeqB = addrAltB;
			addrAltB++;
		}
		else
		{
			v->NoopComment("A-eq-B subroutine");
			addrAeqB = v->AddOp1(OP_Yield, regAddrA);
			v->AddOp2(OP_If, regEofA, addrEofA);
			v->AddOp2(OP_Goto, 0, labelCmpr);
		}

		// Generate code to handle the case of A>B
		int addrAgtB; // Address of the A>B subroutine
		v->NoopComment("A-gt-B subroutine");
		addrAgtB = v->CurrentAddr();
		if (op == TK_ALL || op == TK_UNION)
			v->AddOp2(OP_Gosub, regOutB, addrOutB);
		v->AddOp1(OP_Yield, regAddrB);
		v->AddOp2(OP_If, regEofB, addrEofB);
		v->AddOp2(OP_Goto, 0, labelCmpr);

		// This code runs once to initialize everything.
		v->JumpHere(j1);
		v->AddOp2(OP_Integer, 0, regEofA);
		v->AddOp2(OP_Integer, 0, regEofB);
		v->AddOp2(OP_Gosub, regAddrA, addrSelectA);
		v->AddOp2(OP_Gosub, regAddrB, addrSelectB);
		v->AddOp2(OP_If, regEofA, addrEofA);
		v->AddOp2(OP_If, regEofB, addrEofB);

		// Implement the main merge loop
		v->ResolveLabel(labelCmpr);
		v->AddOp4(OP_Permutation, 0, 0, 0, (char *)permutes, Vdbe::P4T_INTARRAY);
		v->AddOp4(OP_Compare, destA.SdstId, destB.SdstId, orderBys, (char *)keyMerge, Vdbe::P4T_KEYINFO_HANDOFF);
		v->ChangeP5(Vdbe::OPFLAG_PERMUTE);
		v->AddOp3(OP_Jump, addrAltB, addrAeqB, addrAgtB);

		// Jump to the this point in order to terminate the query.
		v->ResolveLabel(labelEnd);

		// Set the number of output columns
		if (dest->Dest == SRT_Output)
		{
			Select *first = prior;
			while (first->Prior) first = first->Prior;
			GenerateColumnNames(parse, nullptr, first->EList);
		}

		// Reassembly the compound query so that it will be freed correctly by the calling function
		if (p->Prior)
			Select::Delete(ctx, p->Prior);
		p->Prior = prior;

		//TBD:  Insert subroutine calls to close cursors on incomplete subqueries
		ExplainComposite(parse, p->OP, sub1Id, sub2Id, false);
		return RC_OK;
	}
#endif

#if !defined(OMIT_SUBQUERY) || !defined(OMIT_VIEW)
	__device__ static void SubstExprList(Context *, ExprList *, int, ExprList *); // Forward Declarations
	__device__ static void SubstSelect(Context *, Select *, int, ExprList *); // Forward Declarations
	__device__ static Expr *SubstExpr(Context *ctx, Expr *expr, int tableId, ExprList *list)
	{
		if (!expr) return nullptr;
		if (expr->OP == TK_COLUMN && expr->TableId == tableId)
		{
			if (expr->ColumnIdx < 0)
				expr->OP = TK_NULL;
			else
			{
				_assert(list && expr->ColumnIdx < list->Exprs);
				_assert(!expr->Left && !expr->Right);
				Expr *newExpr = Expr::Dup(ctx, list->Ids[expr->ColumnIdx].Expr, 0);
				Expr::Delete(ctx, expr);
				expr = newExpr;
			}
		}
		else
		{
			expr->Left = SubstExpr(ctx, expr->Left, tableId, list);
			expr->Right = SubstExpr(ctx, expr->Right, tableId, list);
			if (ExprHasProperty(expr, EP_xIsSelect))
				SubstSelect(ctx, expr->x.Select, tableId, list);
			else
				SubstExprList(ctx, expr->x.List, tableId, list);
		}
		return expr;
	}

	__device__ static void SubstExprList(Context *ctx, ExprList *p, int tableId, ExprList *list)
	{
		if (!p) return;
		for (int i = 0; i < p->Exprs; i++)
			p->Ids[i].Expr = SubstExpr(ctx, p->Ids[i].Expr, tableId, list);
	}

	__device__ static void SubstSelect(Context *ctx, Select *p, int tableId, ExprList *list)
	{
		if (!p) return;
		SubstExprList(ctx, p->EList, tableId, list);
		SubstExprList(ctx, p->GroupBy, tableId, list);
		SubstExprList(ctx, p->OrderBy, tableId, list);
		p->Having = SubstExpr(ctx, p->Having, tableId, list);
		p->Where = SubstExpr(ctx, p->Where, tableId, list);
		SubstSelect(ctx, p->Prior, tableId, list);
		SrcList *src = p->Src;
		_assert(src); // Even for (SELECT 1) we have: pSrc!=0 but pSrc->nSrc==0 
		int i;
		SrcList::SrcListItem *item;
		if (_ALWAYS(src))
			for (i = src->Srcs, item = src->Ids; i > 0; i--, item++)
				SubstSelect(ctx, item->Select, tableId, list);
	}

	__device__ static bool FlattenSubquery(Parse *parse, Select *p, int fromId, bool isAgg, bool subqueryIsAgg)
	{
		Context *ctx = parse->Ctx;

		// Check to see if flattening is permitted.  Return 0 if not.
		_assert(p);
		_assert(!p->Prior); // Unable to flatten compound queries
		if (CtxOptimizationDisabled(ctx, OPTFLAG_QueryFlattener)) return 0;
		SrcList *src = p->Src; // The FROM clause of the outer query
		_assert(src && fromId >= 0 && fromId < src->Srcs);
		SrcList::SrcListItem *subitem = &src->Ids[fromId]; // The subquery
		int parentId = subitem->Cursor; // VDBE cursor number of the pSub result set temp table
		Select *sub = subitem->Select; // The inner query or "subquery"
		_assert(sub);
		if (isAgg && subqueryIsAgg) return false;											// Restriction (1)
		if (subqueryIsAgg && src->Srcs > 1) return false;									// Restriction (2)
		SrcList *subSrc = sub->Src; // The FROM clause of the subquery
		_assert(subSrc);
		// Prior to version 3.1.2, when LIMIT and OFFSET had to be simple constants, not arbitrary expresssions, we allowed some combining of LIMIT and OFFSET
		// because they could be computed at compile-time.  But when LIMIT and OFFSET became arbitrary expressions, we were forced to add restrictions (13) and (14).
		if (sub->Limit && p->Limit) return false;											// Restriction (13)
		if (sub->Offset) return false;														// Restriction (14)
		if (p->Rightmost && sub->Limit) return false;										// Restriction (15)
		if (subSrc->Srcs == 0) return false;												// Restriction (7)
		if (sub->SelFlags & SF_Distinct) return false;										// Restriction (5)
		if (sub->Limit && (src->Srcs > 1 || isAgg)) return false;							// Restrictions (8)(9)
		if ((p->SelFlags & SF_Distinct) != 0 && subqueryIsAgg) return false;				// Restriction (6)
		if (p->OrderBy && sub->OrderBy) return false;                                       // Restriction (11)
		if (isAgg && sub->OrderBy) return false;											// Restriction (16)
		if (sub->Limit && p->Where) return false;											// Restriction (19)
		if (sub->Limit && (p->SelFlags & SF_Distinct) != 0) return false;					// Restriction (21)

		// OBSOLETE COMMENT 1:
		// Restriction 3:  If the subquery is a join, make sure the subquery is not used as the right operand of an outer join.  Examples of why this is not allowed:
		//
		//         t1 LEFT OUTER JOIN (t2 JOIN t3)
		//
		// If we flatten the above, we would get
		//
		//         (t1 LEFT OUTER JOIN t2) JOIN t3
		//
		// which is not at all the same thing.
		//
		// OBSOLETE COMMENT 2:
		// Restriction 12:  If the subquery is the right operand of a left outer join, make sure the subquery has no WHERE clause.
		// An examples of why this is not allowed:
		//
		//         t1 LEFT OUTER JOIN (SELECT * FROM t2 WHERE t2.x>0)
		//
		// If we flatten the above, we would get
		//
		//         (t1 LEFT OUTER JOIN t2) WHERE t2.x>0
		//
		// But the t2.x>0 test will always fail on a NULL row of t2, which effectively converts the OUTER JOIN into an INNER JOIN.
		//
		// THIS OVERRIDES OBSOLETE COMMENTS 1 AND 2 ABOVE:
		// Ticket #3300 shows that flattening the right term of a LEFT JOIN is fraught with danger.  Best to avoid the whole thing.  If the subquery is the right term of a LEFT JOIN, then do not flatten.
		if ((subitem->Jointype & JT_OUTER) != 0) return false;

		// Restriction 17: If the sub-query is a compound SELECT, then it must use only the UNION ALL operator. And none of the simple select queries
		// that make up the compound SELECT are allowed to be aggregate or distinct queries.
		Select *sub1; // Pointer to the rightmost select in sub-query
		if (sub->Prior)
		{
			if (sub->OrderBy) return false;														// Restriction 20
			if (isAgg || (p->SelFlags & SF_Distinct) != 0 || src->Srcs != 1) return false;
			for (sub1 = sub; sub1; sub1 = sub1->Prior)
			{
				ASSERTCOVERAGE((sub1->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Distinct);
				ASSERTCOVERAGE((sub1->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Aggregate);
				_assert(sub->Src);
				if ((sub1->SelFlags & (SF_Distinct|SF_Aggregate)) != 0 ||
					(sub1->Prior && sub1->OP != TK_ALL) ||
					sub1->Src->Srcs < 1 ||
					sub->EList->Exprs != sub1->EList->Exprs) return false;
				ASSERTCOVERAGE(sub1->Src->Srcs > 1);
			}
			if (p->OrderBy)																		// Restriction 18.
				for (int ii = 0; ii < p->OrderBy->Exprs; ii++)
					if (p->OrderBy->Ids[ii].OrderByCol == 0) return false;
		}

		////// If we reach this point, flattening is permitted. //////

		// Authorize the subquery
		int i;
		const char *savedAuthContext = parse->AuthContext;
		parse->AuthContext = subitem->Name;
		ASSERTONLY(i = )Auth::Check(parse, AUTH_SELECT, nullptr, nullptr, nullptr);
		ASSERTCOVERAGE(i == ARC_DENY);
		parse->AuthContext = savedAuthContext;

		// If the sub-query is a compound SELECT statement, then (by restrictions 17 and 18 above) it must be a UNION ALL and the parent query must be of the form:
		//
		//     SELECT <expr-list> FROM (<sub-query>) <where-clause> 
		//
		// followed by any ORDER BY, LIMIT and/or OFFSET clauses. This block creates N-1 copies of the parent query without any ORDER BY, LIMIT or 
		// OFFSET clauses and joins them to the left-hand-side of the original using UNION ALL operators. In this case N is the number of simple
		// select statements in the compound sub-query.
		//
		// Example:
		//
		//     SELECT a+1 FROM (
		//        SELECT x FROM tab
		//        UNION ALL
		//        SELECT y FROM tab
		//        UNION ALL
		//        SELECT abs(z*2) FROM tab2
		//     ) WHERE a!=5 ORDER BY 1
		//
		// Transformed into:
		//
		//     SELECT x+1 FROM tab WHERE x+1!=5
		//     UNION ALL
		//     SELECT y+1 FROM tab WHERE y+1!=5
		//     UNION ALL
		//     SELECT abs(z*2)+1 FROM tab2 WHERE abs(z*2)+1!=5
		//     ORDER BY 1
		//
		// We call this the "compound-subquery flattening".
		for (sub = sub->Prior; sub; sub = sub->Prior)
		{
			ExprList *orderBy = p->OrderBy;
			Expr *limit = p->Limit;
			Expr *offset = p->Offset;
			Select *prior = p->Prior;
			p->OrderBy = nullptr;
			p->Src = nullptr;
			p->Prior = nullptr;
			p->Limit = nullptr;
			p->Offset = nullptr;
			Select *newSelect = Expr::SelectDup(ctx, p, 0);
			p->Offset = offset;
			p->Limit = limit;
			p->OrderBy = orderBy;
			p->Src = src;
			p->OP = TK_ALL;
			p->Rightmost = 0;
			if (!newSelect)
				newSelect = prior;
			else
			{
				newSelect->Prior = prior;
				newSelect->Rightmost = nullptr;
			}
			p->Prior = newSelect;
			if (ctx->MallocFailed) return true;
		}

		// Begin flattening the iFrom-th entry of the FROM clause in the outer query.
		sub = sub1 = subitem->Select;

		// Delete the transient table structure associated with the subquery
		_tagfree(ctx, subitem->Database);
		_tagfree(ctx, subitem->Name);
		_tagfree(ctx, subitem->Alias);
		subitem->Database = nullptr;
		subitem->Name = nullptr;
		subitem->Alias = nullptr;
		subitem->Select = nullptr;

		// Defer deleting the Table object associated with the subquery until code generation is
		// complete, since there may still exist Expr.pTab entries that refer to the subquery even after flattening.  Ticket #3346.
		//
		// pSubitem->pTab is always non-NULL by test restrictions and tests above.
		if (_ALWAYS(subitem->Table))
		{
			Table *tabToDel = subitem->Table;
			if (tabToDel->Refs == 1)
			{
				Parse *toplevel = Parse_Toplevel(parse);
				tabToDel->NextZombie = toplevel->ZombieTab;
				toplevel->ZombieTab = tabToDel;
			}
			else
				tabToDel->Refs--;
			subitem->Table = nullptr;
		}

		// The following loop runs once for each term in a compound-subquery flattening (as described above).  If we are doing a different kind
		// of flattening - a flattening other than a compound-subquery flattening - then this loop only runs once.
		//
		// This loop moves all of the FROM elements of the subquery into the the FROM clause of the outer query.  Before doing this, remember
		// the cursor number for the original outer query FROM element in iParent.  The iParent cursor will never be used.  Subsequent code
		// will scan expressions looking for iParent references and replace those references with expressions that resolve to the subquery FROM
		// elements we are now copying in.
		Select *parent;
		for (parent = p; parent; parent = parent->Prior, sub = sub->Prior)
		{
			JT jointype = (JT)0;
			subSrc = sub->Src; // FROM clause of subquery
			int subSrcs = subSrc->Srcs; // Number of terms in subquery FROM clause
			src = parent->Src; // FROM clause of the outer query
			if (src)
			{
				_assert(parent == p); // First time through the loop
				jointype = subitem->Jointype;
			}
			else
			{
				_assert(parent != p); // 2nd and subsequent times through the loop
				src = parent->Src = Parse::SrcListAppend(ctx, 0, 0, 0);
				if (!src)
				{
					_assert(ctx->MallocFailed);
					break;
				}
			}

			// The subquery uses a single slot of the FROM clause of the outer query.  If the subquery has more than one element in its FROM clause,
			// then expand the outer query to make space for it to hold all elements of the subquery.
			//
			// Example:
			//
			//    SELECT * FROM tabA, (SELECT * FROM sub1, sub2), tabB;
			//
			// The outer query has 3 slots in its FROM clause.  One slot of the outer query (the middle slot) is used by the subquery.  The next
			// block of code will expand the out query to 4 slots.  The middle slot is expanded to two slots in order to make space for the
			// two elements in the FROM clause of the subquery.
			if (subSrcs > 1)
			{
				parent->Src = src = Parse::SrcListEnlarge(ctx, src, subSrcs-1 ,fromId+1);
				if (ctx->MallocFailed)
					break;
			}

			// Transfer the FROM clause terms from the subquery into the outer query.
			for (i = 0; i < subSrcs; i++)
			{
				Parse::IdListDelete(ctx, src->Ids[i+fromId].Using);
				src->Ids[i+fromId] = subSrc->Ids[i];
				_memset(&subSrc->Ids[i], 0, sizeof(subSrc->Ids[i]));
			}
			src->Ids[fromId].Jointype = jointype;

			// Now begin substituting subquery result set expressions for references to the iParent in the outer query.
			// 
			// Example:
			//
			//   SELECT a+5, b*10 FROM (SELECT x*3 AS a, y+10 AS b FROM t1) WHERE a>b;
			//   \                     \_____________ subquery __________/          /
			//    \_____________________ outer query ______________________________/
			//
			// We look at every expression in the outer query and every place we see "a" we substitute "x*3" and every place we see "b" we substitute "y+10".
			ExprList *list = parent->EList; // The result set of the outer query
			for (i = 0; i < list->Exprs; i++)
			{
				if (!list->Ids[i].Name)
				{
					char *name = _tagstrdup(ctx, list->Ids[i].Span);
					Parse::Dequote(name);
					list->Ids[i].Name = name;
				}
			}
			SubstExprList(ctx, parent->EList, parentId, sub->EList);
			if (isAgg)
			{
				SubstExprList(ctx, parent->GroupBy, parentId, sub->EList);
				parent->Having = SubstExpr(ctx, parent->Having, parentId, sub->EList);
			}
			if (sub->OrderBy)
			{
				_assert(!parent->OrderBy);
				parent->OrderBy = sub->OrderBy;
				sub->OrderBy = nullptr;
			}
			else if (parent->OrderBy)
				SubstExprList(ctx, parent->OrderBy, parentId, sub->EList);
			Expr *where_ = (sub->Where ? Expr::Dup(ctx, sub->Where, 0) : nullptr); // The WHERE clause
			if (subqueryIsAgg)
			{
				_assert(!parent->Having);
				parent->Having = parent->Where;
				parent->Where = where_;
				parent->Having = SubstExpr(ctx, parent->Having, parentId, sub->EList);
				parent->Having = Expr::And(ctx, parent->Having, Expr::Dup(ctx, sub->Having, 0));
				_assert(!parent->GroupBy);
				parent->GroupBy = Expr::ListDup(ctx, sub->GroupBy, 0);
			}
			else
			{
				parent->Where = SubstExpr(ctx, parent->Where, parentId, sub->EList);
				parent->Where = Expr::And(ctx, parent->Where, where_);
			}

			// The flattened query is distinct if either the inner or the outer query is distinct. 
			parent->SelFlags |= (sub->SelFlags & SF_Distinct);

			// SELECT ... FROM (SELECT ... LIMIT a OFFSET b) LIMIT x OFFSET y;
			//
			// One is tempted to try to add a and b to combine the limits.  But this does not work if either limit is negative.
			if (sub->Limit)
			{
				parent->Limit = sub->Limit;
				sub->Limit = nullptr;
			}
		}

		// Finially, delete what is left of the subquery and return success.
		Select::Delete(ctx, sub1);
		return true;
	}
#endif

	__device__ static WHERE MinMaxQuery(AggInfo *aggInfo, ExprList **minMaxOut)
	{
		WHERE r = WHERE_ORDERBY_NORMAL;
		*minMaxOut = nullptr;
		if (aggInfo->Funcs.length == 1)
		{
			Expr *expr = aggInfo->Funcs[0].Expr; // Aggregate function
			ExprList *list = expr->x.List; // Arguments to agg function
			_assert(expr->OP == TK_AGG_FUNCTION);
			if (list && list->Exprs == 1 && list->Ids[0].Expr->OP == TK_AGG_COLUMN)
			{
				const char *funcName = expr->u.Token;
				if (!_strcmp(funcName, "min"))
				{
					r = WHERE_ORDERBY_MIN;
					*minMaxOut = list;
				}
				else if (!_strcmp(funcName, "max"))
				{
					r = WHERE_ORDERBY_MAX;
					*minMaxOut = list;
				}
			}
		}
		_assert(*minMaxOut == nullptr || (*minMaxOut)->Exprs == 1);
		return r;
	}

	__device__ static Table *IsSimpleCount(Select *p, AggInfo *aggInfo)
	{
		_assert(!p->GroupBy);
		if (p->Where || p->EList->Exprs != 1 || p->Src->Srcs != 1 || p->Src->Ids[0].Select)
			return nullptr;
		Table *table = p->Src->Ids[0].Table;
		Expr *expr = p->EList->Ids[0].Expr;
		_assert(table && !table->Select && expr);
		if (IsVirtual(table)) return nullptr;
		if (expr->OP != TK_AGG_FUNCTION) return nullptr;
		if (_NEVER(aggInfo->Funcs.length == 0)) return nullptr;
		if ((aggInfo->Funcs[0].Func->Flags & FUNC_COUNT) == 0) return nullptr;
		if (expr->Flags & EP_Distinct) return nullptr;
		return table;
	}

	__device__ RC Select::IndexedByLookup(Parse *parse, SrcList::SrcListItem *from)
	{
		if (from->Table && from->IndexName)
		{
			Table *table = from->Table;
			char *indexName = from->IndexName;
			Index *index;
			for (index = table->Index; index && _strcmp(index->Name, indexName); index = index->Next) ;
			if (!index)
			{
				parse->ErrorMsg("no such index: %s", indexName, 0);
				parse->CheckSchema = 1;
				return RC_ERROR;
			}
			from->Index = index;
		}
		return RC_OK;
	}

	__device__ static WRC SelectExpander(Walker *walker, Select *p)
	{
		Parse *parse = walker->Parse;
		Context *ctx = parse->Ctx;
		int j, k;

		SF selFlags = p->SelFlags;
		p->SelFlags |= SF_Expanded;
		if (ctx->MallocFailed)
			return WRC_Abort;
		if (_NEVER(!p->Src) || (selFlags & SF_Expanded) != 0)
			return WRC_Prune;
		SrcList *tabList = p->Src;
		ExprList *list = p->EList;

		// Make sure cursor numbers have been assigned to all entries in the FROM clause of the SELECT statement.
		parse->SrcListAssignCursors(tabList);

		// Look up every table named in the FROM clause of the select.  If an entry of the FROM clause is a subquery instead of a table or view,
		// then create a transient table structure to describe the subquery.
		int i;
		SrcList::SrcListItem *from;
		for (i = 0, from = tabList->Ids; i < tabList->Srcs; i++, from++)
		{
			Table *table;
			if (from->Table)
			{
				// This statement has already been prepared.  There is no need to go further.
				_assert(i == 0);
				return WRC_Prune;
			}
			if (!from->Name)
			{
#ifndef OMIT_SUBQUERY
				Select *sel = from->Select;
				// A sub-query in the FROM clause of a SELECT
				_assert(sel);
				_assert(!from->Table);
				walker->WalkSelect(sel);
				from->Table = table = (Table *)_tagalloc2(ctx, sizeof(Table), true);
				if (!table) return WRC_Abort;
				table->Refs = 1;
				table->Name = _mtagprintf(ctx, "sqlite_subquery_%p_", (void *)table);
				while (sel->Prior) { sel = sel->Prior; }
				SelectColumnsFromExprList(parse, sel->EList, &table->Cols.length, &table->Cols.data);
				table->PKey = -1;
				table->RowEst = 1000000;
				table->TabFlags |= TF_Ephemeral;
#endif
			}
			else
			{
				// An ordinary table or view name in the FROM clause
				_assert(!from->Table);
				from->Table = table = parse->LocateTableItem(false, from);
				if (!table) return WRC_Abort;
				if (table->Refs == 0xffff)
				{
					parse->ErrorMsg("too many references to \"%s\": max 65535", table->Name);
					from->Table = nullptr;
					return WRC_Abort;
				}
				table->Refs++;
#if !defined(OMIT_VIEW) || !defined (OMIT_VIRTUALTABLE)
				if (table->Select || IsVirtual(table))
				{
					// We reach here if the named table is a really a view
					if (parse->ViewGetColumnNames(table)) return WRC_Abort;
					_assert(!from->Select);
					from->Select = Expr::SelectDup(ctx, table->Select, 0);
					walker->WalkSelect(from->Select);
				}
#endif
			}

			// Locate the index named by the INDEXED BY clause, if any.
			if (Select::IndexedByLookup(parse, from))
				return WRC_Abort;
		}

		// Process NATURAL keywords, and ON and USING clauses of joins.
		if (ctx->MallocFailed || p->ProcessJoin(parse))
			return WRC_Abort;

		// For every "*" that occurs in the column list, insert the names of all columns in all tables.  And for every TABLE.* insert the names
		// of all columns in TABLE.  The parser inserted a special expression with the TK_ALL operator for each "*" that it found in the column list.
		// The following code just has to locate the TK_ALL expressions and expand each one to the list of all columns in all tables.
		//
		// The first loop just checks to see if there are any "*" operators that need expanding.
		Expr *e, *right, *expr;
		for (k = 0; k < list->Exprs; k++)
		{
			e = list->Ids[k].Expr;
			if (e->OP == TK_ALL) break;
			_assert(e->OP != TK_DOT || e->Right != nullptr);
			_assert(e->OP != TK_DOT || (e->Left != nullptr && e->Left->OP == TK_ID));
			if (e->OP == TK_DOT && e->Right->OP == TK_ALL) break;
		}
		if (k < list->Exprs)
		{
			// If we get here it means the result set contains one or more "*" operators that need to be expanded.  Loop through each expression
			// in the result set and expand them one by one.
			ExprList::ExprListItem *a = list->Ids;
			ExprList *newList = nullptr;
			Context::FLAG flags = ctx->Flags;
			bool longNames = ((flags & Context::FLAG_FullColNames)!=0 && (flags & Context::FLAG_ShortColNames) == 0);

			// When processing FROM-clause subqueries, it is always the case that full_column_names=OFF and short_column_names=ON. The sqlite3ResultSetOfSelect() routine makes it so.
			_assert((p->SelFlags & SF_NestedFrom) == 0 || ((flags & Context::FLAG_FullColNames) == 0 && (flags & Context::FLAG_ShortColNames) != 0));

			for (k = 0; k < list->Exprs; k++)
			{
				e = a[k].Expr;
				right = e->Right;
				_assert(e->OP != TK_DOT || right != nullptr);
				if (e->OP != TK_ALL && (e->OP != TK_DOT || right->OP != TK_ALL))
				{
					// This particular expression does not need to be expanded.
					newList = Expr::ListAppend(parse, newList, a[k].Expr);
					if (newList)
					{
						newList->Ids[newList->Exprs-1].Name = a[k].Name;
						newList->Ids[newList->Exprs-1].Span = a[k].Span;
						a[k].Name = nullptr;
						a[k].Span = nullptr;
					}
					a[k].Expr = nullptr;
				}
				else
				{
					// This expression is a "*" or a "TABLE.*" and needs to be expanded.
					bool tableSeen = false; // Set to 1 when TABLE matches
					char *tokenName = nullptr; // text of name of TABLE
					if (e->OP == TK_DOT)
					{
						_assert(e->Left != nullptr);
						_assert(!ExprHasProperty(e->Left, EP_IntValue));
						tokenName = e->Left->u.Token;
					}
					for (i = 0, from = tabList->Ids; i < tabList->Srcs; i++, from++)
					{
						Table *table = from->Table;
						Select *sub = from->Select;
						char *tableName = from->Alias;
						const char *schemaName = nullptr;
						int db;
						if (!tableName)
							tableName = table->Name;
						if (ctx->MallocFailed) break;
						if (!sub || (sub->SelFlags & SF_NestedFrom) == 0)
						{
							sub = nullptr;
							if (tokenName && _strcmp(tokenName, tableName))
								continue;
							db = Prepare::SchemaToIndex(ctx, table->Schema);
							schemaName = (db >= 0 ? ctx->DBs[db].Name : "*");
						}
						for (j = 0; j < table->Cols.length; j++)
						{
							char *name = table->Cols[j].Name;
							char *colname; // The computed column name
							char *toFree; // Malloced string that needs to be freed
							Token sColname; // Computed column name as a token

							_assert(name);
							if (tokenName && sub && !Walker::MatchSpanName(sub->EList->Ids[j].Span, 0, tokenName, 0))
								continue;

							// If a column is marked as 'hidden' (currently only possible for virtual tables), do not include it in the expanded result-set list.
							if (IsHiddenColumn(&table->Cols[j]))
							{
								_assert(IsVirtual(table));
								continue;
							}
							tableSeen = true;

							if (i > 0 && !tokenName)
							{
								if ((from->Jointype & JT_NATURAL) != 0 && TableAndColumnIndex(tabList, i, name, nullptr, nullptr))
									continue; // In a NATURAL join, omit the join columns from the table to the right of the join
								if (Parse::IdListIndex(from->Using, name) >= 0)
									continue; // In a join with a USING clause, omit columns in the using clause from the table on the right.
							}
							right = Expr::Expr_(ctx, TK_ID, name);
							colname = name;
							toFree = nullptr;
							if (longNames || tabList->Srcs > 1)
							{
								Expr *left = Expr::Expr_(ctx, TK_ID, tableName);
								expr = Expr::PExpr_(parse, TK_DOT, left, right, 0);
								if (schemaName)
								{
									left = Expr::Expr_(ctx, TK_ID, schemaName);
									expr = Expr::PExpr_(parse, TK_DOT, left, expr, 0);
								}
								if (longNames)
								{
									colname = _mtagprintf(ctx, "%s.%s", tableName, name);
									toFree = colname;
								}
							}
							else
								expr = right;
							newList = Expr::ListAppend(parse, newList, expr);
							sColname.data = colname;
							sColname.length = _strlen30(colname);
							Expr::ListSetName(parse, newList, &sColname, 0);
							if (newList && (p->SelFlags & SF_NestedFrom) != 0)
							{
								ExprList::ExprListItem *x = &newList->Ids[newList->Exprs - 1];
								if (sub)
								{
									x->Span = _tagstrdup(ctx, sub->EList->Ids[j].Span);
									ASSERTCOVERAGE(x->Span == nullptr);
								}
								else
								{
									x->Span = _mtagprintf(ctx, "%s.%s.%s", schemaName, tableName, colname);
									ASSERTCOVERAGE(x->Span == nullptr);
								}
								x->SpanIsTab = 1;
							}
							_tagfree(ctx, toFree);
						}
					}
					if (!tableSeen)
					{
						if (tokenName)
							parse->ErrorMsg("no such table: %s", tokenName);
						else
							parse->ErrorMsg("no tables specified");
					}
				}
			}
			Expr::ListDelete(ctx, list);
			p->EList = newList;
		}
#if MAX_COLUMN
		if (p->EList && p->EList->Exprs > ctx->Limits[LIMIT_COLUMN])
			parse->ErrorMsg("too many columns in result set");
#endif
		return WRC_Continue;
	}

	__device__ static WRC ExprWalkNoop(Walker *notUsed1, Expr *notUsed2)
	{
		return WRC_Continue;
	}

	__device__ void Select::Expand(Parse *parse)
	{
		Walker w;
		w.SelectCallback = SelectExpander;
		w.ExprCallback = ExprWalkNoop;
		w.Parse = parse;
		w.WalkSelect(this);
	}

#ifndef OMIT_SUBQUERY
	__device__ static WRC SelectAddSubqueryTypeInfo(Walker *walker, Select *p)
	{
		_assert(p->SelFlags & SF_Resolved);
		if ((p->SelFlags & SF_HasTypeInfo) == 0)
		{
			p->SelFlags |= SF_HasTypeInfo;
			Parse *parse = walker->Parse;
			SrcList *tabList = p->Src;
			int i;
			SrcList::SrcListItem *from;
			for (i = 0, from = tabList->Ids; i < tabList->Srcs; i++, from++)
			{
				Table *table = from->Table;
				if (_ALWAYS(table != 0) && (table->TabFlags & TF_Ephemeral) != 0)
				{
					// A sub-query in the FROM clause of a SELECT
					Select *sel = from->Select;
					_assert(sel);
					while (sel->Prior) sel = sel->Prior;
					SelectAddColumnTypeAndCollation(parse, table->Cols.length, table->Cols.data, sel);
				}
			}
		}
		return WRC_Continue;
	}
#endif

	__device__ void Select::AddTypeInfo(Parse *parse)
	{
#ifndef OMIT_SUBQUERY
		Walker w;
		w.SelectCallback = SelectAddSubqueryTypeInfo;
		w.ExprCallback = ExprWalkNoop;
		w.Parse = parse;
		w.WalkSelect(this);
#endif
	}

	__device__ void Select::Prep(Parse *parse, NameContext *outerNC)
	{
		Context *ctx = parse->Ctx;
		if (ctx->MallocFailed) return;
		if (SelFlags & SF_HasTypeInfo) return;
		Expand(parse);
		if (parse->Errs || ctx->MallocFailed) return;
		Walker::ResolveSelectNames(parse, this, outerNC);
		if (parse->Errs || ctx->MallocFailed) return;
		AddTypeInfo(parse);
	}

	__device__ static void ResetAccumulator(Parse *parse, AggInfo *aggInfo)
	{
		if (aggInfo->Funcs.length + aggInfo->Columns.length == 0)
			return;
		Vdbe *v = parse->V;
		int i;
		for (i = 0; i < aggInfo->Columns.length; i++)
			v->AddOp2(OP_Null, 0, aggInfo->Columns[i].Mem);
		AggInfo::AggInfoFunc *func;
		for (func = aggInfo->Funcs.data, i = 0; i < aggInfo->Funcs.length; i++, func++)
		{
			v->AddOp2(OP_Null, 0, func->Mem);
			if (func->Distinct >= 0)
			{
				Expr *e = func->Expr;
				_assert(!ExprHasProperty(e, EP_xIsSelect));
				if (e->x.List == 0 || e->x.List->Exprs != 1)
				{
					parse->ErrorMsg("DISTINCT aggregates must have exactly one argument");
					func->Distinct = -1;
				}
				else
				{
					KeyInfo *keyInfo = KeyInfoFromExprList(parse, e->x.List);
					v->AddOp4(OP_OpenEphemeral, func->Distinct, 0, 0, (char *)keyInfo, Vdbe::P4T_KEYINFO_HANDOFF);
				}
			}
		}
	}

	__device__ static void FinalizeAggFunctions(Parse *parse, AggInfo *aggInfo)
	{
		Vdbe *v = parse->V;
		int i;
		AggInfo::AggInfoFunc *f;
		for (i = 0, f = aggInfo->Funcs.data; i < aggInfo->Funcs.length; i++, f++)
		{
			ExprList *list = f->Expr->x.List;
			_assert(!ExprHasProperty(f->Expr, EP_xIsSelect));
			v->AddOp4(OP_AggFinal, f->Mem, (list ? list->Exprs : 0), 0, (char *)f->Func, Vdbe::P4T_FUNCDEF);
		}
	}

	__device__ static void UpdateAccumulator(Parse *parse, AggInfo *aggInfo)
	{
		Vdbe *v = parse->V;
		int regHit = 0;

		aggInfo->DirectMode = 1;
		Expr::CacheClear(parse);
		int i;
		AggInfo::AggInfoFunc *f;
		for (i = 0, f = aggInfo->Funcs.data; i < aggInfo->Funcs.length; i++, f++)
		{
			ExprList *list = f->Expr->x.List;
			_assert(!ExprHasProperty(f->Expr, EP_xIsSelect));
			int args;
			int regAgg;
			if (list)
			{
				args = list->Exprs;
				regAgg = Expr::GetTempRange(parse, args);
				Expr::CodeExprList(parse, list, regAgg, 1);
			}
			else
			{
				args = 0;
				regAgg = 0;
			}
			int addrNext = 0;
			if (f->Distinct >= 0)
			{
				addrNext = v->MakeLabel();
				_assert(args == 1);
				CodeDistinct(parse, f->Distinct, addrNext, 1, regAgg);
			}
			if (f->Func->Flags & FUNC_NEEDCOLL)
			{
				CollSeq *coll = nullptr;
				int j;
				ExprList::ExprListItem *item;
				_assert(list); // pList!=0 if pF->pFunc has NEEDCOLL
				for (j = 0, item = list->Ids; !coll && j < args; j++, item++)
					coll = item->Expr->CollSeq(parse);
				if (!coll)
					coll = parse->Ctx->DefaultColl;
				if (regHit == 0 && aggInfo->Accumulators) regHit = ++parse->Mems;
				v->AddOp4(OP_CollSeq, regHit, 0, 0, (char *)coll, Vdbe::P4T_COLLSEQ);
			}
			v->AddOp4(OP_AggStep, 0, regAgg, f->Mem, (char *)f->Func, Vdbe::P4T_FUNCDEF);
			v->ChangeP5((uint8)args);
			Expr::CacheAffinityChange(parse, regAgg, args);
			Expr::ReleaseTempRange(parse, regAgg, args);
			if (addrNext)
			{
				v->ResolveLabel(addrNext);
				Expr::CacheClear(parse);
			}
		}

		// Before populating the accumulator registers, clear the column cache. Otherwise, if any of the required column values are already present 
		// in registers, sqlite3ExprCode() may use OP_SCopy to copy the value to pC->iMem. But by the time the value is used, the original register
		// may have been used, invalidating the underlying buffer holding the text or blob value. See ticket [883034dcb5].
		//
		// Another solution would be to change the OP_SCopy used to copy cached values to an OP_Copy.
		int addrHitTest = 0;
		if (regHit)
			addrHitTest = v->AddOp1(OP_If, regHit);
		Expr::CacheClear(parse);
		AggInfo::AggInfoColumn *c;
		for (i = 0, c = aggInfo->Columns.data; i < aggInfo->Accumulators; i++, c++)
			Expr::Code(parse, c->Expr, c->Mem);
		aggInfo->DirectMode = 0;
		Expr::CacheClear(parse);
		if (addrHitTest)
			v->JumpHere(addrHitTest);
	}

#ifndef OMIT_EXPLAIN
	__device__ static void ExplainSimpleCount(Parse *parse, Table *table, Index *index)
	{
		if (parse->Explain == 2)
		{
			char *eqp = _mtagprintf(parse->Ctx, "SCAN TABLE %s %s%s(~%d rows)",
				table->Name, 
				(index ? "USING COVERING INDEX " : ""),
				(index ? index->Name : ""),
				table->RowEst);
			parse->V->AddOp4(OP_Explain, parse->SelectId, 0, 0, eqp, Vdbe::P4T_DYNAMIC);
		}
	}
#else
#define ExplainSimpleCount(a, b, c)
#endif

	__device__ RC Select::Select_(Parse *parse, Select *p, SelectDest *dest)
	{
#ifndef OMIT_EXPLAIN
		int restoreSelectId = parse->SelectId;
		parse->SelectId = parse->NextSelectId++;
#endif

		Context *ctx = parse->Ctx; // The database connection
		if (p == 0 || ctx->MallocFailed || parse->Errs)
			return RC_ERROR;
		if (Auth::Check(parse, AUTH_SELECT, nullptr, nullptr, nullptr)) return RC_ERROR;
		AggInfo sAggInfo; // Information used by aggregate queries
		_memset(&sAggInfo, 0, sizeof(sAggInfo));

		if (IgnorableOrderby(dest))
		{
			_assert(dest->Dest == SRT_Exists || dest->Dest == SRT_Union || dest->Dest == SRT_Except || dest->Dest == SRT_Discard);
			// If ORDER BY makes no difference in the output then neither does DISTINCT so it can be removed too.
			Expr::ListDelete(ctx, p->OrderBy);
			p->OrderBy = nullptr;
			p->SelFlags &= ~SF_Distinct;
		}
		p->Prep(parse, nullptr);
		ExprList *orderBy = p->OrderBy; // The ORDER BY clause.  May be NULL
		SrcList *tabList = p->Src; // List of tables to select from
		ExprList *list = p->EList; // List of columns to extract.

		RC rc = RC_ERROR; // Value to return from this function
		if (parse->Errs || ctx->MallocFailed)
			goto select_end;
		bool isAgg = ((p->SelFlags & SF_Aggregate) != 0); // True for select lists like "count(*)"
		_assert(list);

		// Begin generating code.
		Vdbe *v = parse->GetVdbe(); // The virtual machine under construction
		if (!v) goto select_end;

#ifndef OMIT_SUBQUERY
		// If writing to memory or generating a set only a single column may be output.
		if (CheckForMultiColumnSelectError(parse, dest, list->Exprs))
			goto select_end;
#endif

		int i, j;
#if !defined(OMIT_SUBQUERY) || !defined(OMIT_VIEW)
		// Generate code for all sub-queries in the FROM clause
		for (i = 0; !p->Prior && i < tabList->Srcs; i++)
		{
			SrcList::SrcListItem *item = &tabList->Ids[i];
			SelectDest sDest;
			Select *sub = item->Select;
			if (!sub) continue;

			// Sometimes the code for a subquery will be generated more than once, if the subquery is part of the WHERE clause in a LEFT JOIN,
			// for example.  In that case, do not regenerate the code to manifest a view or the co-routine to implement a view.  The first instance
			// is sufficient, though the subroutine to manifest the view does need to be invoked again.
			if (item->AddrFillSub)
			{
				if (!item->ViaCoroutine)
					v->AddOp2(OP_Gosub, item->RegReturn, item->AddrFillSub);
				continue;
			}

			// Increment Parse.nHeight by the height of the largest expression tree refered to by this, the parent select. The child select
			// may contain expression trees of at most (SQLITE_MAX_EXPR_DEPTH-Parse.nHeight) height. This is a bit
			// more conservative than necessary, but much easier than enforcing an exact limit.
			parse->Height += Expr::SelectExprHeight(p);

			bool isAggSub = ((sub->SelFlags & SF_Aggregate) != 0);
			if (FlattenSubquery(parse, p, i, isAgg, isAggSub))
			{
				// This subquery can be absorbed into its parent.
				if (isAggSub)
				{
					isAgg = true;
					p->SelFlags |= SF_Aggregate;
				}
				i = -1;
			}
			else if (tabList->Srcs == 1 && (p->SelFlags & SF_Materialize) == 0 && CtxOptimizationEnabled(ctx, OPTFLAG_SubqCoroutine))
			{
				// Implement a co-routine that will return a single row of the result set on each invocation.
				item->RegReturn = ++parse->Mems;
				int addrEof = ++parse->Mems;
				// Before coding the OP_Goto to jump to the start of the main routine, ensure that the jump to the verify-schema routine has already
				// been coded. Otherwise, the verify-schema would likely be coded as part of the co-routine. If the main routine then accessed the 
				// database before invoking the co-routine for the first time (for example to initialize a LIMIT register from a sub-select), it would 
				// be doing so without having verified the schema version and obtained the required db locks. See ticket d6b36be38.
				parse->CodeVerifySchema(-1);
				v->AddOp0(OP_Goto);
				int addrTop = v->AddOp1(OP_OpenPseudo, item->Cursor);
				v->ChangeP5(1);
				v->Comment("coroutine for %s", item->Table->Name);
				item->AddrFillSub = addrTop;
				v->AddOp2(OP_Integer, 0, addrEof);
				v->ChangeP5(1);
				DestInit(&sDest, SRT_Coroutine, item->RegReturn);
				ExplainSetInteger(item->SelectId, (uint8)parse->NextSelectId);
				Select_(parse, sub, &sDest);
				item->Table->RowEst = (unsigned)sub->SelectRows;
				item->ViaCoroutine = 1;
				v->ChangeP2(addrTop, sDest.SdstId);
				v->ChangeP3(addrTop, sDest.Sdsts);
				v->AddOp2(OP_Integer, 1, addrEof);
				v->AddOp1(OP_Yield, item->RegReturn);
				v->Comment("end %s", item->Table->Name);
				v->JumpHere(addrTop-1);
				Expr::ClearTempRegCache(parse);
			}
			else
			{
				// Generate a subroutine that will fill an ephemeral table with the content of this subquery.  pItem->addrFillSub will point
				// to the address of the generated subroutine.  pItem->regReturn is a register allocated to hold the subroutine return address
				_assert(item->AddrFillSub == 0);
				item->RegReturn = ++parse->Mems;
				int topAddr = v->AddOp2(OP_Integer, 0, item->RegReturn);
				item->AddrFillSub = topAddr+1;
				v->NoopComment("materialize %s", item->Table->Name);
				int onceAddr = 0;
				if (item->IsCorrelated == 0)
					onceAddr = Expr::CodeOnce(parse); // If the subquery is no correlated and if we are not inside of a trigger, then we only need to compute the value of the subquery once.
				DestInit(&sDest, SRT_EphemTab, item->Cursor);
				ExplainSetInteger(item->SelectId, (uint8)parse->NextSelectId);
				Select_(parse, sub, &sDest);
				item->Table->RowEst = (unsigned)sub->SelectRows;
				if (onceAddr) v->JumpHere(onceAddr);
				int retAddr = v->AddOp1(OP_Return, item->RegReturn);
				v->Comment("end %s", item->Table->Name);
				v->ChangeP1(topAddr, retAddr);
				Expr::ClearTempRegCache(parse);
			}
			if (/*parse->Errs ||*/ ctx->MallocFailed)
				goto select_end;
			parse->Height -= Expr::SelectExprHeight(p);
			tabList = p->Src;
			if (!IgnorableOrderby(dest))
				orderBy = p->OrderBy;
		}
		list = p->EList;
#endif
		Expr *where_ = p->Where; // The WHERE clause.  May be NULL
		ExprList *groupBy = p->GroupBy; // The GROUP BY clause.  May be NULL
		Expr *having = p->Having; // The HAVING clause.  May be NULL
		DistinctCtx sDistinct; // Info on how to code the DISTINCT keyword
		sDistinct.IsTnct = ((p->SelFlags & SF_Distinct) != 0);

#ifndef OMIT_COMPOUND_SELECT
		// If there is are a sequence of queries, do the earlier ones first.
		if (p->Prior)
		{
			if (!p->Rightmost)
			{
				int cnt = 0;
				Select *right = nullptr;
				for (Select *loop = p; loop; loop = loop->Prior, cnt++)
				{
					loop->Rightmost = p;
					loop->Next = right;
					right = loop;
				}
				int maxSelect = ctx->Limits[LIMIT_COMPOUND_SELECT];
				if (maxSelect && cnt > maxSelect)
				{
					parse->ErrorMsg("too many terms in compound SELECT");
					goto select_end;
				}
			}
			rc = MultiSelect(parse, p, dest);
			ExplainSetInteger(parse->SelectId, restoreSelectId);
			return rc;
		}
#endif

		// If there is both a GROUP BY and an ORDER BY clause and they are identical, then disable the ORDER BY clause since the GROUP BY
		// will cause elements to come out in the correct order.  This is an optimization - the correct answer should result regardless.
		// Use the SQLITE_GroupByOrder flag with SQLITE_TESTCTRL_OPTIMIZER to disable this optimization for testing purposes.
		if (!Expr::ListCompare(p->GroupBy, orderBy) && CtxOptimizationEnabled(ctx, OPTFLAG_GroupByOrder))
			orderBy = nullptr;

		// If the query is DISTINCT with an ORDER BY but is not an aggregate, and if the select-list is the same as the ORDER BY list, then this query
		// can be rewritten as a GROUP BY. In other words, this:
		//
		//     SELECT DISTINCT xyz FROM ... ORDER BY xyz
		//
		// is transformed to:
		//
		//     SELECT xyz FROM ... GROUP BY xyz
		//
		// The second form is preferred as a single index (or temp-table) may be used for both the ORDER BY and DISTINCT processing. As originally 
		// written the query must use a temp-table for at least one of the ORDER BY and DISTINCT, and an index or separate temp-table for the other.
		if ((p->SelFlags & (SF_Distinct|SF_Aggregate)) == SF_Distinct && !Expr::ListCompare(orderBy, p->EList))
		{
			p->SelFlags &= ~SF_Distinct;
			p->GroupBy = Expr::ListDup(ctx, p->EList, 0);
			groupBy = p->GroupBy;
			orderBy = nullptr;
			// Notice that even thought SF_Distinct has been cleared from p->selFlags, the sDistinct.isTnct is still set.  Hence, isTnct represents the
			// original setting of the SF_Distinct flag, not the current setting
			_assert(sDistinct.IsTnct);
		}

		// If there is an ORDER BY clause, then this sorting index might end up being unused if the data can be 
		// extracted in pre-sorted order.  If that is the case, then the OP_OpenEphemeral instruction will be changed to an OP_Noop once
		// we figure out that the sorting index is not needed.  The addrSortIndex variable is used to facilitate that change.
		int addrSortIndex; // Address of an OP_OpenEphemeral instruction
		if (orderBy)
		{
			KeyInfo *keyInfo = KeyInfoFromExprList(parse, orderBy);
			orderBy->ECursor = parse->Tabs++;
			p->AddrOpenEphms[2] = addrSortIndex = v->AddOp4(OP_OpenEphemeral, orderBy->ECursor, orderBy->Exprs+2, 0, (char *)keyInfo, Vdbe::P4T_KEYINFO_HANDOFF);
		}
		else
			addrSortIndex = -1;

		// If the output is destined for a temporary table, open that table.
		if (dest->Dest == SRT_EphemTab)
			v->AddOp2(OP_OpenEphemeral, dest->SDParmId, list->Exprs);

		// Set the limiter.
		int endId = v->MakeLabel(); // Address of the end of the query
		p->SelectRows = (double)LARGEST_INT64;
		ComputeLimitRegisters(parse, p, endId);
		if (p->LimitId == 0 && addrSortIndex >= 0)
		{
			v->GetOp(addrSortIndex)->Opcode = OP_SorterOpen;
			p->SelFlags |= SF_UseSorter;
		}

		// Open a virtual index to use for the distinct set.
		if (p->SelFlags & SF_Distinct)
		{
			sDistinct.TableTnct = parse->Tabs++;
			sDistinct.AddrTnct = v->AddOp4(OP_OpenEphemeral, sDistinct.TableTnct, 0, 0, (char *)KeyInfoFromExprList(parse, p->EList), Vdbe::P4T_KEYINFO_HANDOFF);
			v->ChangeP5(Btree::OPEN_UNORDERED);
			sDistinct.TnctType = WHERE_DISTINCT_UNORDERED;
		}
		else
			sDistinct.TnctType = WHERE_DISTINCT_NOOP;

		WhereInfo *winfo; // Return from sqlite3WhereBegin()
		if (!isAgg && !groupBy)
		{
			// No aggregate functions and no GROUP BY clause
			ExprList *dist = (sDistinct.IsTnct ? p->EList : nullptr);

			// Begin the database scan.
			winfo = WhereInfo::Begin(parse, tabList, where_, orderBy, dist, (WHERE)0, 0);
			if (!winfo) goto select_end;
			if (winfo->RowOuts < p->SelectRows) p->SelectRows = winfo->RowOuts;
			if (winfo->EDistinct) sDistinct.TnctType = winfo->EDistinct;
			if (orderBy && winfo->OBSats == orderBy->Exprs) orderBy = nullptr;

			// If sorting index that was created by a prior OP_OpenEphemeral instruction ended up not being needed, then change the OP_OpenEphemeral into an OP_Noop.
			if (addrSortIndex >= 0 && !orderBy)
			{
				v->ChangeToNoop(addrSortIndex);
				p->AddrOpenEphms[2] = (::OP)-1;
			}

			// Use the standard inner loop.
			SelectInnerLoop(parse, p, list, 0, 0, orderBy, &sDistinct, dest, winfo->ContinueId, winfo->BreakId);

			// End the database scan loop.
			WhereInfo::End(winfo);
		}
		else
		{
			// This case when there exist aggregate functions or a GROUP BY clause or both
			// Remove any and all aliases between the result set and the GROUP BY clause.
			if (groupBy)
			{
				int k;
				ExprList::ExprListItem *item; // For looping over expression in a list
				for (k = p->EList->Exprs, item = p->EList->Ids; k > 0; k--, item++)
					item->Alias = 0;
				for (k = groupBy->Exprs, item = groupBy->Ids; k > 0; k--, item++)
					item->Alias = 0;
				if (p->SelectRows > (double)100) p->SelectRows = (double)100;
			}
			else
				p->SelectRows = (double)1;

			// Create a label to jump to when we want to abort the query
			int addrEnd = v->MakeLabel(); // End of processing for this SELECT

			// Convert TK_COLUMN nodes into TK_AGG_COLUMN and make entries in sAggInfo for all TK_AGG_FUNCTION nodes in expressions of the SELECT statement.
			NameContext sNC; // Name context for processing aggregate information
			_memset(&sNC, 0, sizeof(sNC));
			sNC.Parse = parse;
			sNC.SrcList = tabList;
			sNC.AggInfo = &sAggInfo;
			sAggInfo.SortingColumns = (groupBy ? groupBy->Exprs+1 : 0);
			sAggInfo.GroupBy = groupBy;
			Expr::AnalyzeAggList(&sNC, list);
			Expr::AnalyzeAggList(&sNC, orderBy);
			if (having)
				Expr::AnalyzeAggregates(&sNC, having);
			sAggInfo.Accumulators = sAggInfo.Columns.length;
			for (i = 0; i < sAggInfo.Funcs.length; i++)
			{
				_assert(!ExprHasProperty(sAggInfo.Funcs[i].Expr, EP_xIsSelect));
				sNC.NCFlags |= NC_InAggFunc;
				Expr::AnalyzeAggList(&sNC, sAggInfo.Funcs[i].Expr->x.List);
				sNC.NCFlags &= ~NC_InAggFunc;
			}
			if (ctx->MallocFailed) goto select_end;

			// Processing for aggregates with GROUP BY is very different and much more complex than aggregates without a GROUP BY.
			int amemId; // First Mem address for storing current GROUP BY
			int bmemId; // First Mem address for previous GROUP BY
			int useFlagId; // Mem address holding flag indicating that at least one row of the input to the aggregator has been processed
			int abortFlagId; // Mem address which causes query abort if positive
			bool groupBySort; // Rows come from source in GROUP BY order
			int sortPTab = 0; // Pseudotable used to decode sorting results
			int sortOut = 0; // Output register from the sorter
			if (groupBy)
			{
				// If there is a GROUP BY clause we might need a sorting index to implement it.  Allocate that sorting index now.  If it turns out
				// that we do not need it after all, the OP_SorterOpen instruction will be converted into a Noop.
				sAggInfo.SortingIdx = parse->Tabs++;
				KeyInfo *keyInfo = KeyInfoFromExprList(parse, groupBy); // Keying information for the group by clause
				int addrSortingIdx = v->AddOp4(OP_SorterOpen, sAggInfo.SortingIdx, sAggInfo.SortingColumns, 0, (char *)keyInfo, Vdbe::P4T_KEYINFO_HANDOFF); // The OP_OpenEphemeral for the sorting index

				// Initialize memory locations used by GROUP BY aggregate processing
				useFlagId = ++parse->Mems;
				abortFlagId = ++parse->Mems;
				int regOutputRow = ++parse->Mems; // Return address register for output subroutine
				int addrOutputRow = v->MakeLabel(); // Start of subroutine that outputs a result row
				int regReset = ++parse->Mems; // Return address register for reset subroutine
				int addrReset = v->MakeLabel(); // Subroutine for resetting the accumulator
				amemId = parse->Mems + 1;
				parse->Mems += groupBy->Exprs;
				bmemId = parse->Mems + 1;
				parse->Mems += groupBy->Exprs;
				v->AddOp2(OP_Integer, 0, abortFlagId);
				v->Comment("clear abort flag");
				v->AddOp2(OP_Integer, 0, useFlagId);
				v->Comment("indicate accumulator empty");
				v->AddOp3(OP_Null, 0, amemId, amemId + groupBy->Exprs - 1);

				// Begin a loop that will extract all source rows in GROUP BY order. This might involve two separate loops with an OP_Sort in between, or
				// it might be a single loop that uses an index to extract information in the right order to begin with.
				v->AddOp2(OP_Gosub, regReset, addrReset);
				winfo = WhereInfo::Begin(parse, tabList, where_, groupBy, nullptr, (WHERE)0, 0);
				if (!winfo) goto select_end;
				if (winfo->OBSats == groupBy->Exprs)
					groupBySort = false; // The optimizer is able to deliver rows in group by order so we do not have to sort.  The OP_OpenEphemeral table will be cancelled later because we still need to use the keyInfo
				else
				{
					// Rows are coming out in undetermined order.  We have to push each row into a sorting index, terminate the first loop,
					// then loop over the sorting index in order to get the output in sorted order
					ExplainTempTable(parse, (sDistinct.IsTnct && (p->SelFlags & SF_Distinct) == 0 ? "DISTINCT" : "GROUP BY"));

					groupBySort = true;
					int groupBys = groupBy->Exprs;
					int cols = groupBys + 1;
					j = groupBys+1;
					for (i = 0; i < sAggInfo.Columns.length; i++)
					{
						if (sAggInfo.Columns[i].SorterColumn >= j)
						{
							cols++;
							j++;
						}
					}
					int regBase = Expr::GetTempRange(parse, cols);
					Expr::CacheClear(parse);
					Expr::CodeExprList(parse, groupBy, regBase, 0);
					v->AddOp2(OP_Sequence, sAggInfo.SortingIdx, regBase + groupBys);
					j = groupBys + 1;
					for (i = 0; i < sAggInfo.Columns.length; i++)
					{
						AggInfo::AggInfoColumn *col = &sAggInfo.Columns[i];
						if (col->SorterColumn >= j)
						{
							int r1 = j + regBase;
							int r2 = Expr::CodeGetColumn(parse, col->Table, col->Column, col->TableID, r1, 0);
							if (r1 != r2)
								v->AddOp2(OP_SCopy, r2, r1);
							j++;
						}
					}
					int regRecord = Expr::GetTempReg(parse);
					v->AddOp3(OP_MakeRecord, regBase, cols, regRecord);
					v->AddOp2(OP_SorterInsert, sAggInfo.SortingIdx, regRecord);
					Expr::ReleaseTempReg(parse, regRecord);
					Expr::ReleaseTempRange(parse, regBase, cols);
					WhereInfo::End(winfo);
					sAggInfo.SortingIdxPTab = sortPTab = parse->Tabs++;
					sortOut = Expr::GetTempReg(parse);
					v->AddOp3(OP_OpenPseudo, sortPTab, sortOut, cols);
					v->AddOp2(OP_SorterSort, sAggInfo.SortingIdx, addrEnd);
					v->Comment("GROUP BY sort");
					sAggInfo.UseSortingIdx = 1;
					Expr::CacheClear(parse);
				}

				// Evaluate the current GROUP BY terms and store in b0, b1, b2... (b0 is memory location bmemId+0, b1 is bmemId+1, and so forth)
				// Then compare the current GROUP BY terms against the GROUP BY terms from the previous row currently stored in a0, a1, a2...
				int addrTopOfLoop = v->CurrentAddr(); // Top of the input loop
				Expr::CacheClear(parse);
				if (groupBySort)
					v->AddOp2(OP_SorterData, sAggInfo.SortingIdx, sortOut);
				for (j = 0; j < groupBy->Exprs; j++)
				{
					if (groupBySort)
					{
						v->AddOp3(OP_Column, sortPTab, j, bmemId+j);
						if (j == 0) v->ChangeP5(Vdbe::OPFLAG_CLEARCACHE);
					}
					else
					{
						sAggInfo.DirectMode = 1;
						Expr::Code(parse, groupBy->Ids[j].Expr, bmemId+j);
					}
				}
				v->AddOp4(OP_Compare, amemId, bmemId, groupBy->Exprs, (char *)keyInfo, Vdbe::P4T_KEYINFO);
				int j1 = v->CurrentAddr(); // A-vs-B comparision jump
				v->AddOp3(OP_Jump, j1+1, 0, j1+1);

				// Generate code that runs whenever the GROUP BY changes. Changes in the GROUP BY are detected by the previous code
				// block.  If there were no changes, this block is skipped.
				//
				// This code copies current group by terms in b0,b1,b2,... over to a0,a1,a2.  It then calls the output subroutine
				// and resets the aggregate accumulator registers in preparation for the next GROUP BY batch.
				Expr::CodeMove(parse, bmemId, amemId, groupBy->Exprs);
				v->AddOp2(OP_Gosub, regOutputRow, addrOutputRow);
				v->Comment("output one row");
				v->AddOp2(OP_IfPos, abortFlagId, addrEnd);
				v->Comment("check abort flag");
				v->AddOp2(OP_Gosub, regReset, addrReset);
				v->Comment("reset accumulator");

				// Update the aggregate accumulators based on the content of the current row
				v->JumpHere(j1);
				UpdateAccumulator(parse, &sAggInfo);
				v->AddOp2(OP_Integer, 1, useFlagId);
				v->Comment("indicate data in accumulator");

				// End of the loop
				if (groupBySort)
					v->AddOp2(OP_SorterNext, sAggInfo.SortingIdx, addrTopOfLoop);
				else
				{
					WhereInfo::End(winfo);
					v->ChangeToNoop(addrSortingIdx);
				}

				// Output the final row of result
				v->AddOp2(OP_Gosub, regOutputRow, addrOutputRow);
				v->Comment("output final row");

				// Jump over the subroutines
				v->AddOp2(OP_Goto, 0, addrEnd);

				// Generate a subroutine that outputs a single row of the result set.  This subroutine first looks at the iUseFlag.  If iUseFlag
				// is less than or equal to zero, the subroutine is a no-op.  If the processing calls for the query to abort, this subroutine
				// increments the iAbortFlag memory location before returning in order to signal the caller to abort.
				int addrSetAbort = v->CurrentAddr(); // Set the abort flag and return
				v->AddOp2(OP_Integer, 1, abortFlagId);
				v->Comment("set abort flag");
				v->AddOp1(OP_Return, regOutputRow);
				v->ResolveLabel(addrOutputRow);
				addrOutputRow = v->CurrentAddr();
				v->AddOp2(OP_IfPos, useFlagId, addrOutputRow+2);
				v->Comment("Groupby result generator entry point");
				v->AddOp1(OP_Return, regOutputRow);
				FinalizeAggFunctions(parse, &sAggInfo);
				if (having != nullptr) having->IfFalse(parse, addrOutputRow+1, AFF_BIT_JUMPIFNULL);
				SelectInnerLoop(parse, p, p->EList, 0, 0, orderBy, &sDistinct, dest, addrOutputRow+1, addrSetAbort);
				v->AddOp1(OP_Return, regOutputRow);
				v->Comment("end groupby result generator");

				// Generate a subroutine that will reset the group-by accumulator
				v->ResolveLabel(addrReset);
				ResetAccumulator(parse, &sAggInfo);
				v->AddOp1(OP_Return, regReset);

			}
			// endif pGroupBy.  Begin aggregate queries without GROUP BY: 
			else
			{
				ExprList *del = nullptr;
#ifndef OMIT_BTREECOUNT
				Table *table;
				if ((table = IsSimpleCount(p, &sAggInfo)) != nullptr)
				{
					// If isSimpleCount() returns a pointer to a Table structure, then the SQL statement is of the form:
					//
					//   SELECT count(*) FROM <tbl>
					//
					// where the Table structure returned represents table <tbl>.
					//
					// This statement is so common that it is optimized specially. The OP_Count instruction is executed either on the intkey table that
					// contains the data for table <tbl> or on one of its indexes. It is better to execute the op on an index, as indexes are almost
					// always spread across less pages than their corresponding tables.
					const int db = Prepare::SchemaToIndex(ctx, table->Schema);
					const int csr = parse->Tabs++; // Cursor to scan b-tree

					parse->CodeVerifySchema(db);
					parse->TableLock(db, table->Id, false, table->Name);

					// Search for the index that has the least amount of columns. If there is such an index, and it has less columns than the table
					// does, then we can assume that it consumes less space on disk and will therefore be cheaper to scan to determine the query result.
					// In this case set iRoot to the root page number of the index b-tree and keyInfo to the KeyInfo structure required to navigate the index.
					//
					// (2011-04-15) Do not do a full scan of an unordered index.
					//
					// In practice the KeyInfo structure will not be used. It is only passed to keep OP_OpenRead happy.
					int rootId = table->Id; // Root page of scanned b-tree
					Index *bestIndex = nullptr; // Best index found so far
					for (Index *index = table->Index; index; index = index->Next)
						if (index->Unordered == 0 && (!bestIndex || index->Columns.length < bestIndex->Columns.length))
							bestIndex = index;
					KeyInfo *keyInfo = nullptr; // Keyinfo for scanned index
					if (bestIndex && bestIndex->Columns.length < table->Cols.length)
					{
						rootId = bestIndex->Id;
						keyInfo = parse->IndexKeyinfo(bestIndex);
					}

					// Open a read-only cursor, execute the OP_Count, close the cursor.
					v->AddOp3(OP_OpenRead, csr, rootId, db);
					if (keyInfo)
						v->ChangeP4(-1, (char *)keyInfo, Vdbe::P4T_KEYINFO_HANDOFF);
					v->AddOp2(OP_Count, csr, sAggInfo.Funcs[0].Mem);
					v->AddOp1(OP_Close, csr);
					ExplainSimpleCount(parse, table, bestIndex);
				}
				else
#endif
				{
					// Check if the query is of one of the following forms:
					//
					//   SELECT min(x) FROM ...
					//   SELECT max(x) FROM ...
					//
					// If it is, then ask the code in where.c to attempt to sort results as if there was an "ORDER ON x" or "ORDER ON x DESC" clause. 
					// If where.c is able to produce results sorted in this order, then add vdbe code to break out of the processing loop after the 
					// first iteration (since the first iteration of the loop is guaranteed to operate on the row with the minimum or maximum 
					// value of x, the only row required).
					//
					// A special flag must be passed to sqlite3WhereBegin() to slightly modify behavior as follows:
					//
					//   + If the query is a "SELECT min(x)", then the loop coded by where.c should not iterate over any values with a NULL value for x.
					//
					//   + The optimizer code in where.c (the thing that decides which index or indices to use) should place a different priority on 
					//     satisfying the 'ORDER BY' clause than it does in other cases. Refer to code and comments in where.c for details.
					ExprList *minMax = nullptr;
					WHERE flag = WHERE_ORDERBY_NORMAL;
					_assert(!p->GroupBy);
					_assert(flag == 0);
					if (!p->Having)
						flag = MinMaxQuery(&sAggInfo, &minMax);
					_assert(flag == 0 || (minMax && minMax->Exprs == 1));

					if (flag)
					{
						minMax = Expr::ListDup(ctx, minMax, 0);
						del = minMax;
						if (minMax && !ctx->MallocFailed)
						{
							minMax->Ids[0].SortOrder = (flag != WHERE_ORDERBY_MIN ? SO_DESC : SO_ASC);
							minMax->Ids[0].Expr->OP = TK_COLUMN;
						}
					}

					// This case runs if the aggregate has no GROUP BY clause. The processing is much simpler since there is only a single row of output.
					ResetAccumulator(parse, &sAggInfo);
					winfo = WhereInfo::Begin(parse, tabList, where_, minMax, 0, flag, 0);
					if (!winfo)
					{
						Expr::ListDelete(ctx, del);
						goto select_end;
					}
					UpdateAccumulator(parse, &sAggInfo);
					_assert(!minMax || minMax->Exprs == 1);
					if (winfo->OBSats > 0)
					{
						v->AddOp2(OP_Goto, 0, winfo->BreakId);
						v->Comment("%s() by index", (flag == WHERE_ORDERBY_MIN ? "min" : "max"));
					}
					WhereInfo::End(winfo);
					FinalizeAggFunctions(parse, &sAggInfo);
				}

				orderBy = nullptr;
				if (having != nullptr) having->IfFalse(parse, addrEnd, AFF_BIT_JUMPIFNULL);
				SelectInnerLoop(parse, p, p->EList, 0, 0, nullptr, nullptr, dest, addrEnd, addrEnd);
				Expr::ListDelete(ctx, del);
			}
			v->ResolveLabel(addrEnd);
		}

		if (sDistinct.TnctType == WHERE_DISTINCT_UNORDERED)
			ExplainTempTable(parse, "DISTINCT");

		// If there is an ORDER BY clause, then we need to sort the results and send them to the callback one by one.
		if (orderBy)
		{
			ExplainTempTable(parse, "ORDER BY");
			GenerateSortTail(parse, p, v, list->Exprs, dest);
		}

		// Jump here to skip this query
		v->ResolveLabel(endId);

		// The SELECT was successfully coded. Set the return code to 0 to indicate no errors.
		rc = (RC)0;

		// Control jumps to here if an error is encountered above, or upon successful coding of the SELECT.
select_end:
		ExplainSetInteger(parse->SelectId, restoreSelectId);

		// Identify column names if results of the SELECT are to be output.
		if (rc == RC_OK && dest->Dest == SRT_Output)
			GenerateColumnNames(parse, tabList, list);

		_tagfree(ctx, sAggInfo.Columns.data);
		_tagfree(ctx, sAggInfo.Funcs.data);
		return rc;
	}

#if defined(ENABLE_TREE_EXPLAIN)
	__device__ static void ExplainOneSelect(Vdbe *v, Select *p)
	{
		Vdbe::ExplainPrintf(v, "SELECT ");
		if (p->SelFlags & (SF_Distinct|SF_Aggregate))
		{
			if (p->SelFlags & SF_Distinct)
				Vdbe::ExplainPrintf(v, "DISTINCT ");
			if (p->SelFlags & SF_Aggregate)
				Vdbe::ExplainPrintf(v, "agg_flag ");
			Vdbe::ExplainNL(v);
			Vdbe::ExplainPrintf(v, "   ");
		}
		Expr::ExplainExprList(v, p->EList);
		Vdbe::ExplainNL(v);
		if (p->Src && p->Src->Srcs)
		{
			int i;
			Vdbe::ExplainPrintf(v, "FROM ");
			Vdbe::ExplainPush(v);
			for (i = 0; i < p->Src->Srcs; i++)
			{
				SrcList::SrcListItem *item = &p->Src->Ids[i];
				Vdbe::ExplainPrintf(v, "{%d,*} = ", item->Cursor);
				if (item->Select)
				{
					Select::ExplainSelect(v, item->Select);
					if (item->Table)
						Vdbe::ExplainPrintf(v, " (tabname=%s)", item->Table->Name);
				}
				else if (item->Name)
					Vdbe::ExplainPrintf(v, "%s", item->Name);
				if (item->Alias)
					Vdbe::ExplainPrintf(v, " (AS %s)", item->Alias);
				if (item->Jointype & JT_LEFT)
					Vdbe::ExplainPrintf(v, " LEFT-JOIN");
				Vdbe::ExplainNL(v);
			}
			Vdbe::ExplainPop(v);
		}
		if (p->Where)
		{
			Vdbe::ExplainPrintf(v, "WHERE ");
			Expr::ExplainExpr(v, p->Where);
			Vdbe::ExplainNL(v);
		}
		if (p->GroupBy)
		{
			Vdbe::ExplainPrintf(v, "GROUPBY ");
			Expr::ExplainExprList(v, p->GroupBy);
			Vdbe::ExplainNL(v);
		}
		if (p->Having)
		{
			Vdbe::ExplainPrintf(v, "HAVING ");
			Expr::ExplainExpr(v, p->Having);
			Vdbe::ExplainNL(v);
		}
		if (p->OrderBy)
		{
			Vdbe::ExplainPrintf(v, "ORDERBY ");
			Expr::ExplainExprList(v, p->OrderBy);
			Vdbe::ExplainNL(v);
		}
		if (p->Limit)
		{
			Vdbe::ExplainPrintf(v, "LIMIT ");
			Expr::ExplainExpr(v, p->Limit);
			Vdbe::ExplainNL(v);
		}
		if (p->Offset)
		{
			Vdbe::ExplainPrintf(v, "OFFSET ");
			Expr::ExplainExpr(v, p->Offset);
			Vdbe::ExplainNL(v);
		}
	}

	__device__ void Select::ExplainSelect(Vdbe *v, Select *p)
	{
		if (!p)
		{
			Vdbe::ExplainPrintf(v, "(null-select)");
			return;
		}
		while (p->Prior) { p->Prior->Next = p; p = p->Prior; }
		Vdbe::ExplainPush(v);
		while (p)
		{
			ExplainOneSelect(v, p);
			p = p->Next;
			if (!p) break;
			Vdbe::ExplainNL(v);
			Vdbe::ExplainPrintf(v, "%s\n", SelectOpName(p->OP));
		}
		Vdbe::ExplainPrintf(v, "END");
		Vdbe::ExplainPop(v);
	}
#endif
}