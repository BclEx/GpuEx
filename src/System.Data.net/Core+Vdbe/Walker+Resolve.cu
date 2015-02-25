// resolve.c
#include "Core+Vdbe.cu.h"
//#include <stdlib.h>
//#include <string.h>

namespace Core
{
	__device__ static WRC IncrAggDepth(Walker *walker, Expr *expr)
	{
		if (expr->OP == TK_AGG_FUNCTION) expr->OP2 += walker->u.I;
		return WRC_Continue;
	}
	
	__device__ static void IncrAggFunctionDepth(Expr *expr, int n)
	{
		if (n > 0)
		{
			Walker w;
			_memset(&w, 0, sizeof(w));
			w.ExprCallback = IncrAggDepth;
			w.u.I = n;
			w.WalkExpr(expr);
		}
	}

	__device__ static void ResolveAlias(Parse *parse, ExprList *list, int colId, Expr *expr, const char *type, int subqueries)
	{
		_assert(colId >= 0 && colId < list->Exprs);
		Expr *orig = list->Ids[colId].Expr; // The iCol-th column of the result set
		_assert(orig != nullptr);
		_assert(orig->Flags & EP_Resolved);
		Context *ctx = parse->Ctx; // The database connection
		Expr *dup = Expr::Dup(ctx, orig, 0); // Copy of pOrig
		if (!dup) return;
		if (orig->OP != TK_COLUMN && type[0] != 'G')
		{
			IncrAggFunctionDepth(dup, subqueries);
			dup = Expr::PExpr_(parse, TK_AS, dup, 0, 0);
			if (!dup) return;
			if (list->Ids[colId].Alias == 0)
				list->Ids[colId].Alias = (uint16)(++parse->Alias.length);
			dup->TableId = list->Ids[colId].Alias;
		}
		if (expr->OP == TK_COLLATE)
			dup = Expr::AddCollateString(parse, dup, expr->u.Token);

		// Before calling sqlite3ExprDelete(), set the EP_Static flag. This prevents ExprDelete() from deleting the Expr structure itself,
		// allowing it to be repopulated by the memcpy() on the following line. The pExpr->u.zToken might point into memory that will be freed by the
		// sqlite3DbFree(db, pDup) on the last line of this block, so be sure to make a copy of the token before doing the sqlite3DbFree().
		ExprSetProperty(expr, EP_Static);
		Expr::Delete(ctx, expr);
		_memcpy(expr, dup, sizeof(*expr));
		if (!ExprHasProperty(expr, EP_IntValue) && expr->u.Token != nullptr)
		{
			_assert((expr->Flags & (EP_Reduced|EP_TokenOnly)) == 0);
			expr->u.Token = _tagstrdup(ctx, expr->u.Token);
			expr->Flags2 |= EP2_MallocedToken;
		}
		_tagfree(ctx, dup);
	}

	__device__ static bool NameInUsingClause(IdList *using_, const char *colName)
	{
		if (using_)
			for (int k = 0; k < using_->Ids.length; k++)
				if (!_strcmp(using_->Ids[k].Name, colName)) return true;
		return false;
	}

	__device__ bool Walker::MatchSpanName(const char *span, const char *colName, const char *table, const char *dbName)
	{
		int n;
		for (n = 0; _ALWAYS(span[n]) && span[n] != '.'; n++) { }
		if (dbName && (_strncmp(span, dbName, n) || dbName[n] != 0))
			return false;
		span += n + 1;
		for (n = 0; _ALWAYS(span[n]) && span[n] != '.'; n++) { }
		if (table && (_strncmp(span, table, n) || table[n] != 0))
			return false;
		span += n + 1;
		if (colName && _strcmp(span, colName))
			return false;
		return true;
	}

	__device__ static WRC LookupName(Parse *parse, const char *dbName, const char *tableName, const char *colName, NameContext *nc, Expr *expr)
	{
		int cnt = 0; // Number of matching column names
		int cntTab = 0; // Number of matching table names
		int subquerys = 0; // How many levels of subquery
		Context *ctx = parse->Ctx; // The database connection
		SrcList::SrcListItem *item; // Use for looping over pSrcList items
		SrcList::SrcListItem *match = nullptr; // The matching pSrcList item
		NameContext *topNC = nc;        // First namecontext in the list
		Schema *schema = nullptr;              // Schema of the expression
		bool isTrigger = false;
		int i, j;

		_assert(nc); // the name context cannot be NULL.
		_assert(colName); // The Z in X.Y.Z cannot be NULL
		_assert(!ExprHasAnyProperty(expr, EP_TokenOnly|EP_Reduced));

		// Initialize the node to no-match
		expr->TableId = -1;
		expr->Table = nullptr;
		ExprSetIrreducible(expr);

		// Translate the schema name in zDb into a pointer to the corresponding schema.  If not found, pSchema will remain NULL and nothing will match
		// resulting in an appropriate error message toward the end of this routine
		if (dbName)
		{
			for (i = 0; i < ctx->DBs.length; i++)
			{
				_assert(ctx->DBs[i].Name);
				if (!_strcmp(ctx->DBs[i].Name, dbName))
				{
					schema = ctx->DBs[i].Schema;
					break;
				}
			}
		}

		// Start at the inner-most context and move outward until a match is found
		while (nc && cnt == 0)
		{
			ExprList *list;
			SrcList *srcList = nc->SrcList;
			if (srcList)
			{
				for (i = 0, item = srcList->Ids; i < srcList->Srcs; i++, item++)
				{
					Table *table = item->Table;
					_assert(table != nullptr && table->Name != nullptr);
					_assert(table->Cols.length > 0);
					if (item->Select && (item->Select->SelFlags & SF_NestedFrom) != 0)
					{
						bool hit = false;
						list = item->Select->EList;
						for (j = 0; j < list->Exprs; j++)
						{
							if (Walker::MatchSpanName(list->Ids[j].Span, colName, tableName, dbName))
							{
								cnt++;
								cntTab = 2;
								match = item;
								expr->ColumnIdx = j;
								hit = true;
							}
						}
						if (hit || table == nullptr) continue;
					}
					if (dbName && table->Schema != schema)
						continue;
					if (tableName)
					{
						const char *tableName2 = (item->Alias ? item->Alias : table->Name);
						_assert(tableName2 != nullptr);
						if (_strcmp(tableName2, tableName))
							continue;
					}
					if (cntTab++ == 0)
						match = item;	
					Column *col;
					for (j = 0, col = table->Cols; j < table->Cols.length; j++, col++)
					{
						if (!_strcmp(col->Name, colName))
						{
							// If there has been exactly one prior match and this match is for the right-hand table of a NATURAL JOIN or is in a 
							// USING clause, then skip this match.
							if (cnt == 1)
							{
								if (item->Jointype & JT_NATURAL) continue;
								if (NameInUsingClause(item->Using, colName)) continue;
							}
							cnt++;
							match = item;
							// Substitute the rowid (column -1) for the INTEGER PRIMARY KEY
							expr->ColumnIdx = (j == table->PKey ? -1 : (int16)j);
							break;
						}
					}
				}
				if (match)
				{
					expr->TableId = match->Cursor;
					expr->Table = match->Table;
					schema = expr->Table->Schema;
				}
			}

#ifndef OMIT_TRIGGER
			// If we have not already resolved the name, then maybe it is a new.* or old.* trigger argument reference
			if (dbName == nullptr && tableName != nullptr && cnt == 0 && parse->TriggerTab != nullptr)
			{
				TK op = parse->TriggerOP;
				Table *table = nullptr;
				_assert(op == TK_DELETE || op == TK_UPDATE || op == TK_INSERT);
				if (op != TK_DELETE && !_strcmp("new", tableName))
				{
					expr->TableId = 1;
					table = parse->TriggerTab;
				}
				else if (op != TK_INSERT && !_strcmp("old", tableName))
				{
					expr->TableId = 0;
					table = parse->TriggerTab;
				}
				if (table)
				{ 
					int colId;
					schema = table->Schema;
					cntTab++;
					for (colId = 0; colId < table->Cols.length; colId++)
					{
						Column *col = &table->Cols[colId];
						if (!_strcmp(col->Name, colName))
						{
							if (colId == table->PKey)
								colId = -1;
							break;
						}
					}
					if (colId >= table->Cols.length && Expr::IsRowid(colName))
						colId = -1; // IMP: R-44911-55124
					if (colId < table->Cols.length)
					{
						cnt++;
						if (colId < 0)
							expr->Aff = AFF_INTEGER;
						else if (expr->TableId == 0)
						{
							ASSERTCOVERAGE(colId == 31);
							ASSERTCOVERAGE(colId == 32);
							parse->Oldmask |= (colId >= 32 ? 0xffffffff : (((uint32)1) << colId));
						}
						else
						{
							ASSERTCOVERAGE(colId == 31);
							ASSERTCOVERAGE(colId == 32);
							parse->Newmask |= (colId >= 32 ? 0xffffffff : (((uint32)1) << colId));
						}
						expr->ColumnIdx = (int16)colId;
						expr->Table = table;
						isTrigger = true;
					}
				}
			}
#endif

			// Perhaps the name is a reference to the ROWID
			if (cnt == 0 && cntTab == 1 && Expr::IsRowid(colName))
			{
				cnt = 1;
				expr->ColumnIdx = -1; // IMP: R-44911-55124
				expr->Aff = AFF_INTEGER;
			}

			// If the input is of the form Z (not Y.Z or X.Y.Z) then the name Z might refer to an result-set alias.  This happens, for example, when
			// we are resolving names in the WHERE clause of the following command:
			//
			//     SELECT a+b AS x FROM table WHERE x<10;
			//
			// In cases like this, replace pExpr with a copy of the expression that forms the result set entry ("a+b" in the example) and return immediately.
			// Note that the expression in the result set should have already been resolved by the time the WHERE clause is resolved.
			if (cnt == 0 && (list = nc->EList) != nullptr && tableName == nullptr)
			{
				for (j = 0; j < list->Exprs; j++)
				{
					char *asName = list->Ids[j].Name;
					if (asName != nullptr && !_strcmp(asName, colName))
					{
						_assert(expr->Left == nullptr && expr->Right == nullptr);
						_assert(expr->x.List == nullptr);
						_assert(expr->x.Select == nullptr);
						Expr *orig = list->Ids[j].Expr;
						if ((nc->NCFlags & NC_AllowAgg) == 0 && ExprHasProperty(orig, EP_Agg))
						{
							parse->ErrorMsg("misuse of aliased aggregate %s", asName);
							return WRC_Abort;
						}
						ResolveAlias(parse, list, j, expr, "", subquerys);
						cnt = 1;
						match = nullptr;
						_assert(tableName == nullptr && dbName == nullptr);
						goto lookupname_end;
					}
				} 
			}

			// Advance to the next name context.  The loop will exit when either we have a match (cnt>0) or when we run out of name contexts.
			if (cnt == 0)
			{
				nc = nc->Next;
				subquerys++;
			}
		}

		// If X and Y are NULL (in other words if only the column name Z is supplied) and the value of Z is enclosed in double-quotes, then
		// Z is a string literal if it doesn't match any column names.  In that case, we need to return right away and not make any changes to
		// pExpr.
		//
		// Because no reference was made to outer contexts, the pNC->nRef fields are not changed in any context.
		if (cnt == 0 && tableName == nullptr && ExprHasProperty(expr, EP_DblQuoted))
		{
			expr->OP = TK_STRING;
			expr->Table = nullptr;
			return WRC_Prune;
		}

		// cnt==0 means there was not match.  cnt>1 means there were two or more matches.  Either way, we have an error.
		if (cnt != 1)
		{
			const char *err = (cnt == 0 ? "no such column" : "ambiguous column name");
			if (dbName)
				parse->ErrorMsg("%s: %s.%s.%s", err, dbName, tableName, colName);
			else if (tableName)
				parse->ErrorMsg("%s: %s.%s", err, tableName, colName);
			else
				parse->ErrorMsg("%s: %s", err, colName);
			parse->CheckSchema = 1;
			topNC->Errs++;
		}

		// If a column from a table in pSrcList is referenced, then record this fact in the pSrcList.a[].colUsed bitmask.  Column 0 causes
		// bit 0 to be set.  Column 1 sets bit 1.  And so forth.  If the column number is greater than the number of bits in the bitmask
		// then set the high-order bit of the bitmask.
		if (expr->ColumnIdx >= 0 && match != nullptr)
		{
			int n = expr->ColumnIdx;
			ASSERTCOVERAGE(n == BMS-1);
			if (n >= BMS)
				n = BMS-1;
			_assert(match->Cursor == expr->TableId);
			match->ColUsed |= ((Bitmask)1) << n;
		}

		// Clean up and return
		Expr::Delete(ctx, expr->Left);
		expr->Left = nullptr;
		Expr::Delete(ctx, expr->Right);
		expr->Right = nullptr;
		expr->OP = (isTrigger ? TK_TRIGGER : TK_COLUMN);
lookupname_end:
		if (cnt == 1)
		{
			_assert(nc != nullptr);
			Auth::Read(parse, expr, schema, nc->SrcList);
			// Increment the nRef value on all name contexts from TopNC up to the point where the name matched.
			for (;;)
			{
				_assert(topNC != nullptr);
				topNC->Refs++;
				if (topNC == nc) break;
				topNC = topNC->Next;
			}
			return WRC_Prune;
		}
		return WRC_Abort;
	}

	__device__ Expr *Walker::CreateColumnExpr(Context *ctx, SrcList *src, int srcId, int colId)
	{
		Expr *p = Expr::Alloc(ctx, TK_COLUMN, nullptr, false);
		if (p)
		{
			SrcList::SrcListItem *item = &src->Ids[srcId];
			p->Table = item->Table;
			p->TableId = item->Cursor;
			if (p->Table->PKey == colId)
				p->ColumnIdx = -1;
			else
			{
				p->ColumnIdx = (yVars)colId;
				ASSERTCOVERAGE(colId == BMS);
				ASSERTCOVERAGE(colId == BMS-1);
				item->ColUsed |= ((Bitmask)1)<<(colId >= BMS ? BMS-1 : colId);
			}
			ExprSetProperty(p, EP_Resolved);
		}
		return p;
	}

	__device__ static WRC ResolveExprStep(Walker *walker, Expr *expr)
	{
		NameContext *nc = walker->u.NC;
		_assert(nc != nullptr);
		Parse *parse = nc->Parse;
		Context *ctx = parse->Ctx;
		_assert(parse == walker->Parse);

		if (ExprHasAnyProperty(expr, EP_Resolved)) return WRC_Prune;
		ExprSetProperty(expr, EP_Resolved);
#ifndef NDEBUG
		if (nc->SrcList && nc->SrcList->Allocs > 0)
		{
			SrcList *srcList = nc->SrcList;
			for (int i = 0; i < nc->SrcList->Srcs; i++)
				_assert(srcList->Ids[i].Cursor >= 0 && srcList->Ids[i].Cursor < parse->Tabs);
		}
#endif
		switch (expr->OP)
		{

#if defined(ENABLE_UPDATE_DELETE_LIMIT) && !defined(OMIT_SUBQUERY)
			// The special operator TK_ROW means use the rowid for the first column in the FROM clause.  This is used by the LIMIT and ORDER BY
			// clause processing on UPDATE and DELETE statements.
		case TK_ROW: {
			SrcList *srcList = nc->SrcList;
			_assert(srcList && srcList->Srcs == 1);
			SrcList::SrcListItem *item = srcList->Ids; 
			expr->OP = TK_COLUMN;
			expr->Table = item->Table;
			expr->TableId = item->Cursor;
			expr->ColumnIdx = -1;
			expr->Affinity = AFF_INTEGER;
			break; }
#endif

		case TK_ID: { // A lone identifier is the name of a column.
			return LookupName(parse, nullptr, nullptr, expr->u.Token, nc, expr); }

		case TK_DOT: { // A table name and column name: ID.ID Or a database, table and column: ID.ID.ID
			const char *columnName;
			const char *tableName;
			const char *dbName;
			// if (srcList == nullptr) break;
			Expr *right = expr->Right;
			if (right->OP == TK_ID)
			{
				dbName = nullptr;
				tableName = expr->Left->u.Token;
				columnName = right->u.Token;
			}
			else
			{
				_assert(right->OP == TK_DOT);
				dbName = expr->Left->u.Token;
				tableName = right->Left->u.Token;
				columnName = right->Right->u.Token;
			}
			return LookupName(parse, dbName, tableName, columnName, nc, expr); }

		case TK_CONST_FUNC:
		case TK_FUNCTION: { // Resolve function names
			ExprList *list = expr->x.List; // The argument list
			int n = (list ? list->Exprs : 0); // Number of arguments
			bool noSuchFunc = false; // True if no such function exists
			bool wrongNumArgs = false; // True if wrong number of arguments
			bool isAgg = false; // True if is an aggregate function
			TEXTENCODE encode = CTXENCODE(ctx); // The database encoding

			ASSERTCOVERAGE(expr->OP == TK_CONST_FUNC);
			_assert(!ExprHasProperty(expr, EP_xIsSelect));
			const char *id = expr->u.Token; // The function name.
			int idLength = _strlen30(id); // Number of characters in function name
			FuncDef *def = Callback::FindFunction(ctx, id, idLength, n, encode, false); // Information about the function
			if (!def)
			{
				def = Callback::FindFunction(ctx, id, idLength, -2, encode, false);
				if (!def)
					noSuchFunc = true;
				else
					wrongNumArgs = true;
			}
			else
				isAgg = (def->Func == nullptr);
#ifndef OMIT_AUTHORIZATION
			if (def)
			{
				ARC auth = Auth::Check(parse, AUTH_FUNCTION, nullptr, def->Name, nullptr); // Authorization to use the function
				if (auth != ARC_OK)
				{
					if (auth == ARC_DENY)
					{
						parse->ErrorMsg("not authorized to use function: %s", def->Name);
						nc->Errs++;
					}
					expr->OP = TK_NULL;
					return WRC_Prune;
				}
			}
#endif
			if (isAgg && (nc->NCFlags & NC_AllowAgg) == 0)
			{
				parse->ErrorMsg("misuse of aggregate function %.*s()", idLength, id);
				nc->Errs++;
				isAgg = false;
			}
			else if (noSuchFunc && !ctx->Init.Busy)
			{
				parse->ErrorMsg("no such function: %.*s", idLength, id);
				nc->Errs++;
			}
			else if (wrongNumArgs)
			{
				parse->ErrorMsg("wrong number of arguments to function %.*s()", idLength, id);
				nc->Errs++;
			}
			if (isAgg) nc->NCFlags &= ~NC_AllowAgg;
			walker->WalkExprList(list);
			if (isAgg)
			{
				NameContext *nc2 = nc;
				expr->OP = TK_AGG_FUNCTION;
				expr->OP2 = 0;
				while (nc2 && !expr->FunctionUsesThisSrc(nc2->SrcList))
				{
					expr->OP2++;
					nc2 = nc2->Next;
				}
				if (nc2) nc2->NCFlags |= NC_HasAgg;
				nc->NCFlags |= NC_AllowAgg;
			}
			// FIX ME:  Compute pExpr->affinity based on the expected return type of the function 
			return WRC_Prune; }

#ifndef OMIT_SUBQUERY
		case TK_SELECT:
		case TK_EXISTS: ASSERTCOVERAGE(expr->OP == TK_EXISTS);
#endif
		case TK_IN: {
			ASSERTCOVERAGE(expr->OP == TK_IN);
			if (ExprHasProperty(expr, EP_xIsSelect))
			{
				int refs = nc->Refs;
#ifndef OMIT_CHECK
				if ((nc->NCFlags & NC_IsCheck) != 0)
					parse->ErrorMsg("subqueries prohibited in CHECK constraints");
#endif
				walker->WalkSelect(expr->x.Select);
				_assert(nc->Refs >= refs);
				if (nc->Refs != refs)
					ExprSetProperty(expr, EP_VarSelect);
			}
			break; }

#ifndef OMIT_CHECK
		case TK_VARIABLE: {
			if ((nc->NCFlags & NC_IsCheck) != 0)
				parse->ErrorMsg("parameters prohibited in CHECK constraints");
			break; }
#endif
		}
		return (parse->Errs || ctx->MallocFailed ? WRC_Abort : WRC_Continue);
	}

	__device__ static int ResolveAsName(Parse *parse, ExprList *list, Expr *expr)
	{
		if (expr->OP == TK_ID)
		{
			char *colName = expr->u.Token;
			for (int i = 0; i < list->Exprs; i++)
			{
				char *asName = list->Ids[i].Name;
				if (asName != nullptr && !_strcmp(asName, colName))
					return i+1;
			}
		}
		return 0;
	}

	__device__ static int ResolveOrderByTermToExprList(Parse *parse, Select *select, Expr *expr)
	{
		int i;
		_assert(!expr->IsInteger(&i));
		ExprList *list = select->EList; // The columns of the result set
		// Resolve all names in the ORDER BY term expression
		NameContext nc;
		_memset(&nc, 0, sizeof(nc));
		nc.Parse = parse;
		nc.SrcList = select->Src;
		nc.EList = list;
		nc.NCFlags = NC_AllowAgg;
		nc.Errs = 0;
		Context *ctx = parse->Ctx; // Database connection
		uint8 savedSuppErr = ctx->SuppressErr; // Saved value of db->suppressErr
		ctx->SuppressErr = 1;
		bool r = Walker::ResolveExprNames(&nc, expr); // Return code from subprocedures
		ctx->SuppressErr = savedSuppErr;
		if (r) return 0;
		// Try to match the ORDER BY expression against an expression in the result set.  Return an 1-based index of the matching result-set entry.
		for (i = 0; i < list->Exprs; i++)
			if (Expr::Compare(list->Ids[i].Expr, expr) < 2)
				return i+1;
		// If no match, return 0.
		return 0;
	}

	__device__ static void ResolveOutOfRangeError(Parse *parse, const char *typeName, int i, int max)
	{
		parse->ErrorMsg("%r %s BY term out of range - should be between 1 and %d", i, typeName, max);
	}

	__device__ static int ResolveCompoundOrderBy(Parse *parse, Select *select)
	{
		ExprList *orderBy = select->OrderBy;
		if (!orderBy) return 0;
		Context *ctx = parse->Ctx;
#if MAX_COLUMN
		if (orderBy->Exprs > ctx->Limits[LIMIT_COLUMN])
		{
			parse->ErrorMsg("too many terms in ORDER BY clause");
			return 1;
		}
#endif
		int i;
		for (i = 0; i < orderBy->Exprs; i++)
			orderBy->Ids[i].Done = false;
		select->Next = nullptr;
		while (select->Prior)
		{
			select->Prior->Next = select;
			select = select->Prior;
		}
		bool moreToDo = true;
		while (select && moreToDo)
		{
			moreToDo = false;
			ExprList *list = select->EList;
			_assert(list != nullptr);
			ExprList::ExprListItem *item;
			for (i = 0, item = orderBy->Ids; i < orderBy->Exprs; i++, item++)
			{
				if (item->Done) continue;
				Expr *expr = item->Expr->SkipCollate();
				int colId = -1;
				if (expr->IsInteger(&colId))
				{
					if (colId <= 0 || colId > list->Exprs)
					{
						ResolveOutOfRangeError(parse, "ORDER", i+1, list->Exprs);
						return 1;
					}
				}
				else
				{
					colId = ResolveAsName(parse, list, expr);
					if (colId == 0)
					{
						Expr *dupExpr = Expr::Dup(ctx, expr, 0);
						if (!ctx->MallocFailed)
						{
							_assert(dupExpr);
							colId = ResolveOrderByTermToExprList(parse, select, dupExpr);
						}
						Expr::Delete(ctx, dupExpr);
					}
				}
				if (colId > 0)
				{
					// Convert the ORDER BY term into an integer column number iCol, taking care to preserve the COLLATE clause if it exists
					Expr *newExpr = Expr::Expr_(ctx, TK_INTEGER, nullptr);
					if (!newExpr) return 1;
					newExpr->Flags |= EP_IntValue;
					newExpr->u.I = colId;
					if (item->Expr == expr)
						item->Expr = newExpr;
					else
					{
						_assert(item->Expr->OP == TK_COLLATE);
						_assert(item->Expr->Left == expr);
						item->Expr->Left = newExpr;
					}
					Expr::Delete(ctx, expr);
					item->OrderByCol = (uint16)colId;
					item->Done = true;
				}
				else
					moreToDo = true;
			}
			select = select->Next;
		}
		for (i = 0; i < orderBy->Exprs; i++)
		{
			if (!orderBy->Ids[i].Done)
			{
				parse->ErrorMsg("%r ORDER BY term does not match any column in the result set", i+1);
				return 1;
			}
		}
		return 0;
	}

	__device__ bool Walker::ResolveOrderGroupBy(Core::Parse *parse, Select *select, ExprList *orderBy, const char *type)
	{
		Context *ctx = parse->Ctx;
		if (!orderBy || ctx->MallocFailed) return false;
#if MAX_COLUMN
		if (orderBy->Exprs > ctx->Limits[LIMIT_COLUMN])
		{
			parse->ErrorMsg("too many terms in %s BY clause", type);
			return true;
		}
#endif
		ExprList *list = select->EList;
		_assert(list != nullptr); // sqlite3SelectNew() guarantees this
		int i;
		ExprList::ExprListItem *item;
		for (i = 0, item = orderBy->Ids; i < orderBy->Exprs; i++, item++)
		{
			if (item->OrderByCol)
			{
				if (item->OrderByCol > list->Exprs)
				{
					ResolveOutOfRangeError(parse, type, i+1, list->Exprs);
					return true;
				}
				ResolveAlias(parse, list, item->OrderByCol-1, item->Expr, type, 0);
			}
		}
		return false;
	}

	__device__ static bool ResolveOrderGroupBy(NameContext *nc, Select *select, ExprList *orderBy, const char *type)
	{
		if (!orderBy) return false;
		int result = select->EList->Exprs; // Number of terms in the result set
		Parse *parse = nc->Parse; // Parsing context
		int i;
		ExprList::ExprListItem *item; // A term of the ORDER BY clause
		for (i = 0, item = orderBy->Ids; i < orderBy->Exprs; i++, item++)
		{
			Expr *expr = item->Expr;
			int colId = ResolveAsName(parse, select->EList, expr); // Column number
			if (colId > 0)
			{
				// If an AS-name match is found, mark this ORDER BY column as being a copy of the iCol-th result-set column.  The subsequent call to
				// sqlite3ResolveOrderGroupBy() will convert the expression to a copy of the iCol-th result-set expression.
				item->OrderByCol = (uint16)colId;
				continue;
			}
			if (expr->SkipCollate()->IsInteger(&colId))
			{
				// The ORDER BY term is an integer constant.  Again, set the column number so that sqlite3ResolveOrderGroupBy() will convert the
				// order-by term to a copy of the result-set expression
				if (colId < 1 || colId > 0xffff)
				{
					ResolveOutOfRangeError(parse, type, i+1, result);
					return true;
				}
				item->OrderByCol = (uint16)colId;
				continue;
			}

			// Otherwise, treat the ORDER BY term as an ordinary expression
			item->OrderByCol = 0;
			if (Walker::ResolveExprNames(nc, expr))
				return true;
			for (int j = 0; j < select->EList->Exprs; j++)
				if (!Expr::Compare(expr, select->EList->Ids[j].Expr))
					item->OrderByCol = j+1;
		}
		return Walker::ResolveOrderGroupBy(parse, select, orderBy, type);
	}

	__device__ static WRC ResolveSelectStep(Walker *walker, Select *p)
	{
		_assert(p != nullptr);
		if (p->SelFlags & SF_Resolved)
			return WRC_Prune;
		NameContext *outerNC = walker->u.NC; // Context that contains this SELECT
		Parse *parse = walker->Parse; // Parsing context
		Context *ctx = parse->Ctx; // Database connection

		// Normally sqlite3SelectExpand() will be called first and will have already expanded this SELECT.  However, if this is a subquery within
		// an expression, sqlite3ResolveExprNames() will be called without a prior call to sqlite3SelectExpand().  When that happens, let
		// sqlite3SelectPrep() do all of the processing for this SELECT. sqlite3SelectPrep() will invoke both sqlite3SelectExpand() and
		// this routine in the correct order.
		if ((p->SelFlags & SF_Expanded) == 0)
		{
			p->Prep(parse, outerNC);
			return (parse->Errs || ctx->MallocFailed ? WRC_Abort : WRC_Prune);
		}

		bool isCompound = (p->Prior != nullptr); // True if p is a compound select
		int compounds = 0; // Number of compound terms processed so far
		Select *leftmost = p; // Left-most of SELECT of a compound
		int i;
		NameContext nc;// Name context of this SELECT
		while (p)
		{
			_assert((p->SelFlags & SF_Expanded) != 0);
			_assert((p->SelFlags & SF_Resolved) == 0);
			p->SelFlags |= SF_Resolved;

			// Resolve the expressions in the LIMIT and OFFSET clauses. These are not allowed to refer to any names, so pass an empty NameContext.
			_memset(&nc, 0, sizeof(nc));
			nc.Parse = parse;
			if (Walker::ResolveExprNames(&nc, p->Limit) || Walker::ResolveExprNames(&nc, p->Offset))
				return WRC_Abort;

			// Recursively resolve names in all subqueries
			SrcList::SrcListItem *item;
			for (i = 0; i < p->Src->Srcs; i++)
			{
				item = &p->Src->Ids[i];
				if (item->Select)
				{
					NameContext *nc2; // Used to iterate name contexts
					int refs = 0; // Refcount for pOuterNC and outer contexts
					const char *savedContext = parse->AuthContext;

					// Count the total number of references to pOuterNC and all of its parent contexts. After resolving references to expressions in
					// pItem->pSelect, check if this value has changed. If so, then SELECT statement pItem->pSelect must be correlated. Set the
					// pItem->isCorrelated flag if this is the case.
					for (nc2 = outerNC; nc2; nc2 = nc2->Next) refs += nc2->Refs;

					if (item->Name) parse->AuthContext = item->Name;
					Walker::ResolveSelectNames(parse, item->Select, outerNC);
					parse->AuthContext = savedContext;
					if (parse->Errs || ctx->MallocFailed) return WRC_Abort;

					for (nc2 = outerNC; nc2; nc2 = nc2->Next) refs -= nc2->Refs;
					_assert(!item->IsCorrelated && refs <= 0);
					item->IsCorrelated = (refs != 0);
				}
			}

			// Set up the local name-context to pass to sqlite3ResolveExprNames() to resolve the result-set expression list.
			nc.NCFlags = NC_AllowAgg;
			nc.SrcList = p->Src;
			nc.Next = outerNC;

			// Resolve names in the result set.
			ExprList *list = p->EList; // Result set expression list
			_assert(list != nullptr);
			for (i = 0; i < list->Exprs; i++)
			{
				Expr *expr = list->Ids[i].Expr;
				if (Walker::ResolveExprNames(&nc, expr))
					return WRC_Abort;
			}

			// If there are no aggregate functions in the result-set, and no GROUP BY expression, do not allow aggregates in any of the other expressions.
			_assert((p->SelFlags & SF_Aggregate) == 0);
			ExprList *groupBy = p->GroupBy; // The GROUP BY clause
			if (groupBy || (nc.NCFlags & NC_HasAgg) != 0)
				p->SelFlags |= SF_Aggregate;
			else
				nc.NCFlags &= ~NC_AllowAgg;

			// If a HAVING clause is present, then there must be a GROUP BY clause.
			if (p->Having && !groupBy)
			{
				parse->ErrorMsg("a GROUP BY clause is required before HAVING");
				return WRC_Abort;
			}

			// Add the expression list to the name-context before parsing the other expressions in the SELECT statement. This is so that
			// expressions in the WHERE clause (etc.) can refer to expressions by aliases in the result set.
			//
			// Minor point: If this is the case, then the expression will be re-evaluated for each reference to it.
			nc.EList = p->EList;
			if (Walker::ResolveExprNames(&nc, p->Where) || Walker::ResolveExprNames(&nc, p->Having))
				return WRC_Abort;

			// The ORDER BY and GROUP BY clauses may not refer to terms in outer queries 
			nc.Next = nullptr;
			nc.NCFlags |= NC_AllowAgg;

			// Process the ORDER BY clause for singleton SELECT statements. The ORDER BY clause for compounds SELECT statements is handled
			// below, after all of the result-sets for all of the elements of the compound have been resolved.
			if (!isCompound && ResolveOrderGroupBy(&nc, p, p->OrderBy, "ORDER"))
				return WRC_Abort;
			if (ctx->MallocFailed)
				return WRC_Abort;

			// Resolve the GROUP BY clause.  At the same time, make sure the GROUP BY clause does not contain aggregate functions.
			if (groupBy)
			{
				if (ResolveOrderGroupBy(&nc, p, groupBy, "GROUP") || ctx->MallocFailed)
					return WRC_Abort;
				ExprList::ExprListItem *item;
				for (i = 0, item = groupBy->Ids; i < groupBy->Exprs; i++, item++)
					if (ExprHasProperty(item->Expr, EP_Agg))
					{
						parse->ErrorMsg("aggregate functions are not allowed in the GROUP BY clause");
						return WRC_Abort;
					}
			}

			// Advance to the next term of the compound
			p = p->Prior;
			compounds++;
		}

		// Resolve the ORDER BY on a compound SELECT after all terms of the compound have been resolved.
		return (isCompound && ResolveCompoundOrderBy(parse, leftmost) ? WRC_Abort : WRC_Prune);
	}

	__device__ bool Walker::ResolveExprNames(NameContext *nc, Expr *expr)
	{
		if (!expr) return false;
#if MAX_EXPR_DEPTH > 0
		{
			Core::Parse *parse = nc->Parse;
			if (Expr::CheckHeight(parse, expr->Height + nc->Parse->Height))
				return true;
			parse->Height += expr->Height;
		}
#endif
		bool savedHasAgg = ((nc->NCFlags & NC_HasAgg) != 0);
		nc->NCFlags &= ~NC_HasAgg;
		Walker w;
		w.ExprCallback = ResolveExprStep;
		w.SelectCallback = ResolveSelectStep;
		w.Parse = nc->Parse;
		w.u.NC = nc;
		w.WalkExpr(expr);
#if MAX_EXPR_DEPTH>0
		nc->Parse->Height -= expr->Height;
#endif
		if (nc->Errs > 0 || w.Parse->Errs > 0)
			ExprSetProperty(expr, EP_Error);
		if (nc->NCFlags & NC_HasAgg)
			ExprSetProperty(expr, EP_Agg);
		else if (savedHasAgg)
			nc->NCFlags |= NC_HasAgg;
		return ExprHasProperty(expr, EP_Error);
	}

	__device__ void Walker::ResolveSelectNames(Core::Parse *parse, Select *p, NameContext *outerNC)
	{
		_assert(p != nullptr);
		Walker w;
		w.ExprCallback = ResolveExprStep;
		w.SelectCallback = ResolveSelectStep;
		w.Parse = parse;
		w.u.NC = outerNC;
		w.WalkSelect(p);
	}
}