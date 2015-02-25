using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;
#if !MAX_VARIABLE_NUMBER
using yVars = System.Int16;
#else
using yVars = System.Int32; 
#endif

namespace Core
{
    public partial class Walker
    {
        static WRC IncrAggDepth(Walker walker, Expr expr)
        {
            if (expr.OP == TK.AGG_FUNCTION) expr.OP2 += (byte)walker.u.I;
            return WRC.Continue;
        }

        static void IncrAggFunctionDepth(Expr expr, int n)
        {
            if (n > 0)
            {
                Walker w = new Walker();
                w.ExprCallback = IncrAggDepth;
                w.u.I = n;
                w.WalkExpr(expr);
            }
        }

        static void ResolveAlias(Parse parse, ExprList list, int colId, Expr expr, string type, int subqueries)
        {
            Debug.Assert(colId >= 0 && colId < list.Exprs);
            Expr orig = list.Ids[colId].Expr; // The iCol-th column of the result set
            Debug.Assert(orig != null);
            Debug.Assert((orig.Flags & EP.Resolved) != 0);
            Context ctx = parse.Ctx; // The database connection
            Expr dup = Expr.Dup(ctx, orig, 0); // Copy of pOrig
            if (orig.OP != TK.COLUMN && (type.Length == 0 || type[0] != 'G'))
            {
                IncrAggFunctionDepth(dup, subqueries);
                dup = Expr.PExpr_(parse, TK.AS, dup, null, null);
                if (dup == null) return;
                if (list.Ids[colId].Alias == 0)
                    list.Ids[colId].Alias = (ushort)(++parse.Alias.length);
                dup.TableId = list.Ids[colId].Alias;
            }
            if (expr.OP == TK.COLLATE)
                dup = Expr.AddCollateString(parse, dup, expr.u.Token);

            // Before calling sqlite3ExprDelete(), set the EP_Static flag. This prevents ExprDelete() from deleting the Expr structure itself,
            // allowing it to be repopulated by the memcpy() on the following line.
            E.ExprSetProperty(expr, EP.Static);
            Expr.Delete(ctx, ref expr);
            expr.memcpy(dup);
            if (!E.ExprHasProperty(expr, EP.IntValue) && expr.u.Token != null)
            {
                Debug.Assert((dup.Flags & (EP.Reduced | EP.TokenOnly)) == 0);
                dup.u.Token = expr.u.Token;
                dup.Flags2 |= EP2.MallocedToken;
            }
            C._tagfree(ctx, ref dup);
        }

        static bool NameInUsingClause(IdList using_, string colName)
        {
            if (using_ != null)
                for (int k = 0; k < using_.Ids.length; k++)
                    if (string.Compare(using_.Ids[k].Name, colName, StringComparison.OrdinalIgnoreCase) == 0) return true;
            return false;
        }

        public static bool MatchSpanName(string span, string colName, string table, string dbName)
        {
            int n;
            for (n = 0; C._ALWAYS(span[n] != null) && span[n] != '.'; n++) { }
            if (dbName != null && (string.Compare(span, 0, dbName, 0, n, StringComparison.OrdinalIgnoreCase) != 0) || dbName[n] != 0)
                return false;
            span += n + 1;
            for (n = 0; C._ALWAYS(span[n] != null) && span[n] != '.'; n++) { }
            if (table != null && (string.Compare(span, 0, table, 0, n, StringComparison.OrdinalIgnoreCase) != 0 || table[n] != 0))
                return false;
            span += n + 1;
            if (colName != null && string.Compare(span, colName) != 0)
                return false;
            return true;
        }

        static WRC LookupName(Parse parse, string dbName, string tableName, string colName, NameContext nc, Expr expr)
        {
            int cnt = 0; // Number of matching column names
            int cntTab = 0; // Number of matching table names
            int subquerys = 0; // How many levels of subquery
            Context ctx = parse.Ctx; // The database connection
            SrcList.SrcListItem item; // Use for looping over pSrcList items
            SrcList.SrcListItem match = null; // The matching pSrcList item
            NameContext topNC = nc; // First namecontext in the list
            Schema schema = null;  // Schema of the expression
            bool isTrigger = false;
            int i, j;

            Debug.Assert(nc != null); // the name context cannot be NULL.
            Debug.Assert(colName != null);    // The Z in X.Y.Z cannot be NULL
            Debug.Assert(!E.ExprHasAnyProperty(expr, EP.TokenOnly | EP.Reduced));

            // Initialize the node to no-match
            expr.TableId = -1;
            expr.Table = null;
            E.ExprSetIrreducible(expr);

            // Translate the schema name in zDb into a pointer to the corresponding schema.  If not found, pSchema will remain NULL and nothing will match
            // resulting in an appropriate error message toward the end of this routine
            if (dbName != null)
            {
                for (i = 0; i < ctx.DBs.length; i++)
                {
                    Debug.Assert(ctx.DBs[i].Name != null);
                    if (string.Compare(ctx.DBs[i].Name, dbName) == 0)
                    {
                        schema = ctx.DBs[i].Schema;
                        break;
                    }
                }
            }

            // Start at the inner-most context and move outward until a match is found
            while (nc != null && cnt == 0)
            {
                ExprList list;
                SrcList srcList = nc.SrcList;
                if (srcList != null)
                {
                    for (i = 0; i < srcList.Srcs; i++)
                    {
                        item = srcList.Ids[i];
                        Table table = item.Table;
                        Debug.Assert(table != null && table.Name != null);
                        Debug.Assert(table.Cols.length > 0);
                        if (item.Select != null && (item.Select.SelFlags & SF.NestedFrom) != 0)
                        {
                            bool hit = false;
                            list = item.Select.EList;
                            for (j = 0; j < list.Exprs; j++)
                            {
                                if (Walker.MatchSpanName(list.Ids[j].Span, colName, tableName, dbName))
                                {
                                    cnt++;
                                    cntTab = 2;
                                    match = item;
                                    expr.ColumnIdx = j;
                                    hit = true;
                                }
                            }
                            if (hit || table == null) continue;
                        }
                        if (dbName != null && table.Schema != schema)
                            continue;
                        if (tableName != null)
                        {
                            string tableName2 = (item.Alias != null ? item.Alias : table.Name);
                            Debug.Assert(tableName2 != null);
                            if (!string.Equals(tableName2, tableName, StringComparison.OrdinalIgnoreCase))
                                continue;
                        }
                        if (cntTab++ == 0)
                            match = item;
                        Column col;
                        for (j = 0; j < table.Cols.length; j++)
                        {
                            col = table.Cols[j];
                            if (string.Equals(col.Name, colName, StringComparison.InvariantCultureIgnoreCase))
                            {
                                // If there has been exactly one prior match and this match is for the right-hand table of a NATURAL JOIN or is in a 
                                // USING clause, then skip this match.
                                if (cnt == 1)
                                {
                                    if ((item.Jointype & JT.NATURAL) != 0) continue;
                                    if (NameInUsingClause(item.Using, colName)) continue;
                                }
                                cnt++;
                                match = item;
                                // Substitute the rowid (column -1) for the INTEGER PRIMARY KEY
                                expr.ColumnIdx = (j == table.PKey ? -1 : (short)j);
                                break;
                            }
                        }
                    }
                    if (match != null)
                    {
                        expr.TableId = match.Cursor;
                        expr.Table = match.Table;
                        schema = expr.Table.Schema;
                    }
                }

#if !OMIT_TRIGGER
                // If we have not already resolved the name, then maybe it is a new.* or old.* trigger argument reference
                if (dbName == null && tableName != null && cnt == 0 && parse.TriggerTab != null)
                {
                    TK op = parse.TriggerOp;
                    Table table = null;
                    Debug.Assert(op == TK.DELETE || op == TK.UPDATE || op == TK.INSERT);
                    if (op != TK.DELETE && string.Equals("new", tableName, StringComparison.InvariantCultureIgnoreCase))
                    {
                        expr.TableId = 1;
                        table = parse.TriggerTab;
                    }
                    else if (op != TK.INSERT && string.Equals("old", tableName, StringComparison.InvariantCultureIgnoreCase))
                    {
                        expr.TableId = 0;
                        table = parse.TriggerTab;
                    }
                    if (table != null)
                    {
                        int colId;
                        schema = table.Schema;
                        cntTab++;
                        for (colId = 0; colId < table.Cols.length; colId++)
                        {
                            Column col = table.Cols[colId];
                            if (string.Equals(col.Name, colName, StringComparison.InvariantCultureIgnoreCase))
                            {
                                if (colId == table.PKey)
                                    colId = -1;
                                break;
                            }
                        }
                        if (colId >= table.Cols.length && Expr.IsRowid(colName))
                            colId = -1; // IMP: R-44911-55124
                        if (colId < table.Cols.length)
                        {
                            cnt++;
                            if (colId < 0)
                                expr.Aff = AFF.INTEGER;
                            else if (expr.TableId == 0)
                            {
                                C.ASSERTCOVERAGE(colId == 31);
                                C.ASSERTCOVERAGE(colId == 32);
                                parse.Oldmask |= (colId >= 32 ? 0xffffffff : (((uint)1) << colId));
                            }
                            else
                            {
                                C.ASSERTCOVERAGE(colId == 31);
                                C.ASSERTCOVERAGE(colId == 32);
                                parse.Newmask |= (colId >= 32 ? 0xffffffff : (((uint)1) << colId));
                            }
                            expr.ColumnIdx = (short)colId;
                            expr.Table = table;
                            isTrigger = true;
                        }
                    }
                }
#endif

                // Perhaps the name is a reference to the ROWID
                if (cnt == 0 && cntTab == 1 && Expr.IsRowid(colName))
                {
                    cnt = 1;
                    expr.ColumnIdx = -1; // IMP: R-44911-55124
                    expr.Aff = AFF.INTEGER;
                }

                // If the input is of the form Z (not Y.Z or X.Y.Z) then the name Z might refer to an result-set alias.  This happens, for example, when
                // we are resolving names in the WHERE clause of the following command:
                //
                //     SELECT a+b AS x FROM table WHERE x<10;
                //
                // In cases like this, replace pExpr with a copy of the expression that forms the result set entry ("a+b" in the example) and return immediately.
                // Note that the expression in the result set should have already been resolved by the time the WHERE clause is resolved.
                if (cnt == 0 && (list = nc.EList) != null && tableName == null)
                {
                    for (j = 0; j < list.Exprs; j++)
                    {
                        string asName = list.Ids[j].Name;
                        if (asName != null && string.Equals(asName, colName, StringComparison.InvariantCultureIgnoreCase))
                        {
                            Debug.Assert(expr.Left == null && expr.Right == null);
                            Debug.Assert(expr.x.List == null);
                            Debug.Assert(expr.x.Select == null);
                            Expr orig = list.Ids[j].Expr;
                            if ((nc.NCFlags & NC.AllowAgg) == 0 && E.ExprHasProperty(orig, EP.Agg))
                            {
                                parse.ErrorMsg("misuse of aliased aggregate %s", asName);
                                return WRC.Abort;
                            }
                            ResolveAlias(parse, list, j, expr, "", subquerys);
                            cnt = 1;
                            match = null;
                            Debug.Assert(tableName == null && dbName == null);
                            goto lookupname_end;
                        }
                    }
                }

                // Advance to the next name context.  The loop will exit when either we have a match (cnt>0) or when we run out of name contexts.
                if (cnt == 0)
                {
                    nc = nc.Next;
                    subquerys++;
                }
            }

            // If X and Y are NULL (in other words if only the column name Z is supplied) and the value of Z is enclosed in double-quotes, then
            // Z is a string literal if it doesn't match any column names.  In that case, we need to return right away and not make any changes to
            // pExpr.
            //
            // Because no reference was made to outer contexts, the pNC->nRef fields are not changed in any context.
            if (cnt == 0 && tableName == null && E.ExprHasProperty(expr, EP.DblQuoted))
            {
                expr.OP = TK.STRING;
                expr.Table = null;
                return WRC.Prune;
            }

            // cnt==0 means there was not match.  cnt>1 means there were two or more matches.  Either way, we have an error.
            if (cnt != 1)
            {
                string err = (cnt == 0 ? "no such column" : "ambiguous column name");
                if (dbName != null)
                    parse.ErrorMsg("%s: %s.%s.%s", err, dbName, tableName, colName);
                else if (tableName != null)
                    parse.ErrorMsg("%s: %s.%s", err, tableName, colName);
                else
                    parse.ErrorMsg("%s: %s", err, colName);
                parse.CheckSchema = 1;
                topNC.Errs++;
            }

            // If a column from a table in pSrcList is referenced, then record this fact in the pSrcList.a[].colUsed bitmask.  Column 0 causes
            // bit 0 to be set.  Column 1 sets bit 1.  And so forth.  If the column number is greater than the number of bits in the bitmask
            // then set the high-order bit of the bitmask.
            if (expr.ColumnIdx >= 0 && match != null)
            {
                int n = expr.ColumnIdx;
                C.ASSERTCOVERAGE(n == BMS - 1);
                if (n >= BMS)
                    n = BMS - 1;
                Debug.Assert(match.Cursor == expr.TableId);
                match.ColUsed |= ((Bitmask)1) << n;
            }

            // Clean up and return
            Expr.Delete(ctx, ref expr.Left);
            expr.Left = null;
            Expr.Delete(ctx, ref expr.Right);
            expr.Right = null;
            expr.OP = (isTrigger ? TK.TRIGGER : TK.COLUMN);
        lookupname_end:
            if (cnt == 1)
            {
                Debug.Assert(nc != null);
                Auth.Read(parse, expr, schema, nc.SrcList);
                // Increment the nRef value on all name contexts from TopNC up to the point where the name matched.
                for (; ; )
                {
                    Debug.Assert(topNC != null);
                    topNC.Refs++;
                    if (topNC == nc) break;
                    topNC = topNC.Next;
                }
                return WRC.Prune;
            }
            return WRC.Abort;
        }

        public static Expr CreateColumnExpr(Context ctx, SrcList src, int srcId, int colId)
        {
            Expr p = Expr.Alloc(ctx, TK.COLUMN, null, false);
            if (p != null)
            {
                SrcList.SrcListItem item = src.Ids[srcId];
                p.Table = item.Table;
                p.TableId = item.Cursor;
                if (p.Table.PKey == colId)
                    p.ColumnIdx = -1;
                else
                {
                    p.ColumnIdx = (yVars)colId;
                    C.ASSERTCOVERAGE(colId == BMS);
                    C.ASSERTCOVERAGE(colId == BMS - 1);
                    item.ColUsed |= ((Bitmask)1) << (colId >= BMS ? BMS - 1 : colId);
                }
                E.ExprSetProperty(p, EP.Resolved);
            }
            return p;
        }

        static WRC ResolveExprStep(Walker walker, Expr expr)
        {
            NameContext nc = walker.u.NC;
            Debug.Assert(nc != null);
            Parse parse = nc.Parse;
            Debug.Assert(parse == walker.Parse);

            if (E.ExprHasAnyProperty(expr, EP.Resolved)) return WRC.Prune;
            E.ExprSetProperty(expr, EP.Resolved);
#if !NDEBUG
            if (nc.SrcList != null && nc.SrcList.Allocs > 0)
            {
                SrcList srcList = nc.SrcList;
                for (int i = 0; i < nc.SrcList.Srcs; i++)
                    Debug.Assert(srcList.Ids[i].Cursor >= 0 && srcList.Ids[i].Cursor < parse.Tabs);
            }
#endif
            switch (expr.OP)
            {

#if ENABLE_UPDATE_DELETE_LIMIT && !OMIT_SUBQUERY
                // The special operator TK_ROW means use the rowid for the first column in the FROM clause.  This is used by the LIMIT and ORDER BY
                // clause processing on UPDATE and DELETE statements.
                case TK.ROW:
                    {
                        SrcList srcList = nc.SrcList;
                        Debug.Assert(srcList != null && srcList.Srcs == 1);
                        SrcList.SrcListItem item = srcList.Ids[0];
                        expr.OP = TK.COLUMN;
                        expr.Table = item.Table;
                        expr.TableId = item.Cursor;
                        expr.ColumnIdx = -1;
                        expr.Aff = AFF.INTEGER;
                        break;
                    }
#endif

                case TK.ID:  // A lone identifier is the name of a column.
                    {
                        return LookupName(parse, null, null, expr.u.Token, nc, expr);
                    }

                case TK.DOT: // A table name and column name: ID.ID Or a database, table and column: ID.ID.ID
                    {
                        string columnName;
                        string tableName;
                        string dbName;
                        // if (srcList == nullptr) break;
                        Expr right = expr.Right;
                        if (right.OP == TK.ID)
                        {
                            dbName = null;
                            tableName = expr.Left.u.Token;
                            columnName = right.u.Token;
                        }
                        else
                        {
                            Debug.Assert(right.OP == TK.DOT);
                            dbName = expr.Left.u.Token;
                            tableName = right.Left.u.Token;
                            columnName = right.Right.u.Token;
                        }
                        return LookupName(parse, dbName, tableName, columnName, nc, expr);
                    }

                case TK.CONST_FUNC:
                case TK.FUNCTION: // Resolve function names
                    {
                        ExprList list = expr.x.List; // The argument list
                        int n = (list != null ? list.Exprs : 0); // Number of arguments
                        bool noSuchFunc = false; // True if no such function exists
                        bool wrongNumArgs = false; // True if wrong number of arguments
                        bool isAgg = false; // True if is an aggregate function
                        TEXTENCODE encode = E.CTXENCODE(parse.Ctx); // The database encoding

                        C.ASSERTCOVERAGE(expr.OP == TK.CONST_FUNC);
                        Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect));
                        string id = expr.u.Token; // The function name.
                        int idLength = id.Length; // Number of characters in function name
                        FuncDef def = Callback.FindFunction(parse.Ctx, id, idLength, n, encode, false); // Information about the function
                        if (def == null)
                        {
                            def = Callback.FindFunction(parse.Ctx, id, idLength, -2, encode, false);
                            if (def == null)
                                noSuchFunc = true;
                            else
                                wrongNumArgs = true;
                        }
                        else
                            isAgg = (def.Func == null);
#if !OMIT_AUTHORIZATION
                        if (def != null)
                        {
                            ARC auth = Auth.Check(parse, AUTH.FUNCTION, null, def.Name, null); // Authorization to use the function
                            if (auth != ARC.OK)
                            {
                                if (auth == ARC.DENY)
                                {
                                    parse.ErrorMsg("not authorized to use function: %s", def.Name);
                                    nc.Errs++;
                                }
                                expr.OP = TK.NULL;
                                return WRC.Prune;
                            }
                        }
#endif
                        if (isAgg && (nc.NCFlags & NC.AllowAgg) == 0)
                        {
                            parse.ErrorMsg("misuse of aggregate function %.*s()", idLength, id);
                            nc.Errs++;
                            isAgg = false;
                        }
                        else if (noSuchFunc && !ctx.Init.Busy)
                        {
                            parse.ErrorMsg("no such function: %.*s", idLength, id);
                            nc.Errs++;
                        }
                        else if (wrongNumArgs)
                        {
                            parse.ErrorMsg("wrong number of arguments to function %.*s()", idLength, id);
                            nc.Errs++;
                        }
                        if (isAgg) nc.NCFlags &= ~NC.AllowAgg;
                        walker.WalkExprList(list);
                        if (isAgg)
                        {
                            NameContext nc2 = nc;
                            expr.OP = TK.AGG_FUNCTION;
                            expr.OP2 = 0;
                            while (nc2 != null && !expr.FunctionUsesThisSrc(nc2.SrcList))
                            {
                                expr.OP2++;
                                nc2 = nc2.Next;
                            }
                            if (nc2 != null) nc2.NCFlags |= NC.HasAgg;
                            nc.NCFlags |= NC.AllowAgg;
                        }
                        // FIX ME:  Compute pExpr->affinity based on the expected return type of the function 
                        return WRC.Prune;
                    }
#if !OMIT_SUBQUERY
                case TK.SELECT:
                case TK.EXISTS:
                    {
                        C.ASSERTCOVERAGE(expr.OP == TK.EXISTS);
                        goto case TK.IN;
                    }
#endif
                case TK.IN:
                    {
                        C.ASSERTCOVERAGE(expr.OP == TK.IN);
                        if (E.ExprHasProperty(expr, EP.xIsSelect))
                        {
                            int refs = nc.Refs;
#if !OMIT_CHECK
                            if ((nc.NCFlags & NC.IsCheck) != 0)
                                parse.ErrorMsg("subqueries prohibited in CHECK constraints");
#endif
                            walker.WalkSelect(expr.x.Select);
                            Debug.Assert(nc.Refs >= refs);
                            if (refs != nc.Refs)
                                E.ExprSetProperty(expr, EP.VarSelect);
                        }
                        break;
                    }
#if !OMIT_CHECK
                case TK.VARIABLE:
                    {
                        if ((nc.NCFlags & NC.IsCheck) != 0)
                            parse.ErrorMsg("parameters prohibited in CHECK constraints");

                        break;
                    }
#endif
            }
            return (parse.Errs != 0 || parse.Ctx.MallocFailed ? WRC.Abort : WRC.Continue);
        }

        static int ResolveAsName(Parse parse, ExprList list, Expr expr)
        {
            if (expr.OP == TK.ID)
            {
                string colName = expr.u.Token;
                for (int i = 0; i < list.Exprs; i++)
                {
                    string asName = list.Ids[i].Name;
                    if (asName != null && string.Equals(asName, colName, StringComparison.InvariantCultureIgnoreCase))
                        return i + 1;
                }
            }
            return 0;
        }

        static int ResolveOrderByTermToExprList(Parse parse, Select select, Expr expr)
        {
            int i = 0;
            Debug.Assert(!expr.IsInteger(ref i));
            ExprList list = select.EList; // The columns of the result set
            // Resolve all names in the ORDER BY term expression
            NameContext nc = new NameContext(); // Name context for resolving pE
            nc.Parse = parse;
            nc.SrcList = select.Src;
            nc.EList = list;
            nc.NCFlags = NC.AllowAgg;
            nc.Errs = 0;
            Context ctx = parse.Ctx; // Database connection
            byte savedSuppErr = ctx.SuppressErr; // Saved value of db->suppressErr
            ctx.SuppressErr = 1;
            bool r = Walker.ResolveExprNames(nc, ref expr);
            ctx.SuppressErr = savedSuppErr;
            if (r) return 0;
            // Try to match the ORDER BY expression against an expression in the result set.  Return an 1-based index of the matching result-set entry.
            for (i = 0; i < list.Exprs; i++)
                if (Expr.Compare(list.Ids[i].Expr, expr) < 2)
                    return i + 1;
            // If no match, return 0.
            return 0;
        }

        static void ResolveOutOfRangeError(Parse parse, string typeName, int i, int max)
        {
            parse.ErrorMsg("%r %s BY term out of range - should be between 1 and %d", i, typeName, max);
        }

        static int ResolveCompoundOrderBy(Parse parse, Select select)
        {
            ExprList orderBy = select.OrderBy;
            if (orderBy == null) return 0;
            Context ctx = parse.Ctx;
#if true || MAX_COLUMN
            if (orderBy.Exprs > ctx.Limits[(int)LIMIT.COLUMN])
            {
                parse.ErrorMsg("too many terms in ORDER BY clause");
                return 1;
            }
#endif
            int i;
            for (i = 0; i < orderBy.Exprs; i++)
                orderBy.Ids[i].Done = false;
            select.Next = null;
            while (select.Prior != null)
            {
                select.Prior.Next = select;
                select = select.Prior;
            }
            bool moreToDo = true;
            while (select != null && moreToDo)
            {
                moreToDo = false;
                ExprList list = select.EList;
                Debug.Assert(list != null);
                ExprList.ExprListItem item;
                for (i = 0; i < orderBy.Exprs; i++)
                {
                    item = orderBy.Ids[i];

                    Expr pDup;
                    if (item.Done) continue;
                    Expr expr = item.Expr;
                    int colId = -1;
                    if (expr.IsInteger(ref colId))
                    {
                        if (colId <= 0 || colId > list.Exprs)
                        {
                            ResolveOutOfRangeError(parse, "ORDER", i + 1, list.Exprs);
                            return 1;
                        }
                    }
                    else
                    {
                        colId = ResolveAsName(parse, list, expr);
                        if (colId == 0)
                        {
                            Expr dupExpr = Expr.Dup(ctx, expr, 0);
                            if (!ctx.MallocFailed)
                            {
                                Debug.Assert(dupExpr != null);
                                colId = ResolveOrderByTermToExprList(parse, select, dupExpr);
                            }
                            Expr.Delete(ctx, ref dupExpr);
                        }
                    }
                    if (colId > 0)
                    {
                        // Convert the ORDER BY term into an integer column number iCol, taking care to preserve the COLLATE clause if it exists
                        Expr newExpr = Expr.Expr_(ctx, TK.INTEGER, null);
                        if (newExpr == null) return 1;
                        newExpr.Flags |= EP.IntValue;
                        newExpr.u.I = colId;
                        if (item.Expr == expr)
                            item.Expr = newExpr;
                        else
                        {
                            Debug.Assert(item.Expr.OP == TK.COLLATE);
                            Debug.Assert(item.Expr.Left == expr);
                            item.Expr.Left = newExpr;
                        }
                        Expr.Delete(ctx, ref expr);
                        item.OrderByCol = (ushort)colId;
                        item.Done = true;

                    }
                    else
                        moreToDo = true;
                }
                select = select.Next;
            }
            for (i = 0; i < orderBy.Exprs; i++)
            {
                if (!orderBy.Ids[i].Done)
                {
                    parse.ErrorMsg("%r ORDER BY term does not match any column in the result set", i + 1);
                    return 1;
                }
            }
            return 0;
        }

        public static bool ResolveOrderGroupBy(Parse parse, Select select, ExprList orderBy, string type)
        {
            Context ctx = parse.Ctx;
            if (orderBy == null || parse.Ctx.MallocFailed) return 0;
#if !MAX_COLUMN
            if (orderBy.Exprs > ctx.Limits[(int)LIMIT.COLUMN])
            {
                parse.ErrorMsg("too many terms in %s BY clause", type);
                return true;
            }
#endif
            ExprList list = select.EList;
            Debug.Assert(list != null); // sqlite3SelectNew() guarantees this
            int i;
            ExprList.ExprListItem item;
            for (i = 0; i < orderBy.Exprs; i++)
            {
                item = orderBy.Ids[i];
                if (item.OrderByCol != null)
                {
                    if (item.OrderByCol > list.Exprs)
                    {
                        ResolveOutOfRangeError(parse, type, i + 1, list.Exprs);
                        return true;
                    }
                    ResolveAlias(parse, list, item.OrderByCol - 1, item.Expr, type);
                }
            }
            return false;
        }

        static bool ResolveOrderGroupBy(NameContext nc, Select select, ExprList orderBy, string type)
        {
            if (orderBy == null) return false;
            int result = select.EList.Exprs; // Number of terms in the result set
            Parse parse = nc.Parse; // Parsing context
            int i;
            ExprList.ExprListItem item; // A term of the ORDER BY clause
            for (i = 0; i < orderBy.Exprs; i++)
            {
                item = orderBy.Ids[i];
                Expr expr = item.Expr;
                int colId = ResolveAsName(parse, select.EList, expr); // Column number
                if (colId > 0)
                {
                    // If an AS-name match is found, mark this ORDER BY column as being a copy of the iCol-th result-set column.  The subsequent call to
                    // sqlite3ResolveOrderGroupBy() will convert the expression to a copy of the iCol-th result-set expression.
                    item.OrderByCol = (ushort)colId;
                    continue;
                }
                if (expr.SkipCollate().IsInteger(ref colId))
                {
                    // The ORDER BY term is an integer constant.  Again, set the column number so that sqlite3ResolveOrderGroupBy() will convert the
                    // order-by term to a copy of the result-set expression
                    if (colId < 1 || colId > 0xffff)
                    {
                        ResolveOutOfRangeError(parse, type, i + 1, result);
                        return true;
                    }
                    item.OrderByCol = (ushort)colId;
                    continue;
                }

                // Otherwise, treat the ORDER BY term as an ordinary expression
                item.OrderByCol = 0;
                if (Walker.ResolveExprNames(nc, ref expr))
                    return true;
                for (int j = 0; j < select.EList.Exprs; j++)
                    if (Expr.Compare(expr, select.EList.Ids[j].Expr) == 0)
                        item.OrderByCol = (ushort)(j + 1);
            }
            return Walker.ResolveOrderGroupBy(parse, select, orderBy, type);
        }

        static WRC ResolveSelectStep(Walker walker, Select p)
        {
            Debug.Assert(p != null);
            if ((p.SelFlags & SF.Resolved) != 0)
                return WRC.Prune;
            NameContext outerNC = walker.u.NC; // Context that contains this SELECT
            Parse parse = walker.Parse; // Parsing context
            Context ctx = parse.Ctx; // Database connection

            // Normally sqlite3SelectExpand() will be called first and will have already expanded this SELECT.  However, if this is a subquery within
            // an expression, sqlite3ResolveExprNames() will be called without a prior call to sqlite3SelectExpand().  When that happens, let
            // sqlite3SelectPrep() do all of the processing for this SELECT. sqlite3SelectPrep() will invoke both sqlite3SelectExpand() and
            // this routine in the correct order.
            if ((p.SelFlags & SF.Expanded) == 0)
            {
                p.Prep(parse, outerNC);
                return (parse.Errs != 0 || ctx.MallocFailed ? WRC.Abort : WRC.Prune);
            }

            bool isCompound = (p.Prior != null); // True if p is a compound select
            int compounds = 0; // Number of compound terms processed so far
            Select leftmost = p; // Left-most of SELECT of a compound
            int i;
            NameContext nc; // Name context of this SELECT
            while (p != null)
            {
                Debug.Assert((p.SelFlags & SF.Expanded) != 0);
                Debug.Assert((p.SelFlags & SF.Resolved) == 0);
                p.SelFlags |= SF.Resolved;

                // Resolve the expressions in the LIMIT and OFFSET clauses. These are not allowed to refer to any names, so pass an empty NameContext.
                nc = new NameContext(); //: _memset(&nc, 0, sizeof(nc));
                nc.Parse = parse;
                if (Walker.ResolveExprNames(nc, ref p.Limit) || Walker.ResolveExprNames(nc, ref p.Offset))
                    return WRC.Abort;

                // Recursively resolve names in all subqueries
                SrcList.SrcListItem item;
                for (i = 0; i < p.Src.Srcs; i++)
                {
                    item = p.Src.Ids[i];
                    if (item.Select != null)
                    {
                        NameContext nc2; // Used to iterate name contexts
                        int refs = 0; // Refcount for pOuterNC and outer contexts
                        string savedContext = parse.AuthContext;

                        // Count the total number of references to pOuterNC and all of its parent contexts. After resolving references to expressions in
                        // pItem->pSelect, check if this value has changed. If so, then SELECT statement pItem->pSelect must be correlated. Set the
                        // pItem->isCorrelated flag if this is the case.
                        for (nc2 = outerNC; nc2 != null; nc2 = nc2.Next) refs += nc2.Refs;

                        if (item.Name != null) parse.AuthContext = item.Name;
                        Walker.ResolveSelectNames(parse, item.Select, outerNC);
                        parse.AuthContext = savedContext;
                        if (parse.Errs != 0 || ctx.MallocFailed) return WRC.Abort;

                        for (nc2 = outerNC; nc2 != null; nc2 = nc2.Next) refs -= nc2.Refs;
                        Debug.Assert(!item.IsCorrelated && refs <= 0);
                        item.IsCorrelated = (refs != 0);
                    }
                }

                // Set up the local name-context to pass to sqlite3ResolveExprNames() to resolve the result-set expression list.
                nc.NCFlags = NC.AllowAgg;
                nc.SrcList = p.Src;
                nc.Next = outerNC;

                // Resolve names in the result set.
                ExprList list = p.EList; // Result set expression list
                Debug.Assert(list != null);
                for (i = 0; i < list.Exprs; i++)
                {
                    Expr expr = list.Ids[i].Expr;
                    if (Walker.ResolveExprNames(nc, ref expr))
                        return WRC.Abort;
                }

                // If there are no aggregate functions in the result-set, and no GROUP BY expression, do not allow aggregates in any of the other expressions.
                Debug.Assert((p.SelFlags & SF.Aggregate) == 0);
                ExprList groupBy = p.GroupBy; // The GROUP BY clause
                if (groupBy != null || (nc.NCFlags & NC.HasAgg) != 0)
                    p.SelFlags |= SF.Aggregate;
                else
                    nc.NCFlags &= ~NC.AllowAgg;

                // If a HAVING clause is present, then there must be a GROUP BY clause.
                if (p.Having != null && groupBy == null)
                {
                    parse.ErrorMsg("a GROUP BY clause is required before HAVING");
                    return WRC.Abort;
                }

                // Add the expression list to the name-context before parsing the other expressions in the SELECT statement. This is so that
                // expressions in the WHERE clause (etc.) can refer to expressions by aliases in the result set.
                //
                // Minor point: If this is the case, then the expression will be re-evaluated for each reference to it.
                nc.EList = p.EList;
                if (Walker.ResolveExprNames(nc, ref p.Where) || Walker.ResolveExprNames(nc, ref p.Having))
                    return WRC.Abort;

                // The ORDER BY and GROUP BY clauses may not refer to terms in outer queries 
                nc.Next = null;
                nc.NCFlags |= NC.AllowAgg;

                // Process the ORDER BY clause for singleton SELECT statements. The ORDER BY clause for compounds SELECT statements is handled
                // below, after all of the result-sets for all of the elements of the compound have been resolved.
                if (!isCompound && Walker.ResolveOrderGroupBy(nc, p, p.OrderBy, "ORDER"))
                    return WRC.Abort;
                if (ctx.MallocFailed)
                    return WRC.Abort;

                // Resolve the GROUP BY clause.  At the same time, make sure the GROUP BY clause does not contain aggregate functions.
                if (groupBy != null)
                {
                    if (Walker.ResolveOrderGroupBy(nc, p, groupBy, "GROUP") || ctx.MallocFailed)
                        return WRC.Abort;
                    ExprList.ExprListItem item2;
                    for (i = 0; i < groupBy.Exprs; i++)
                    {
                        item2 = groupBy.Ids[i];
                        if (E.ExprHasProperty(item2.Expr, EP.Agg))
                        {
                            parse.ErrorMsg("aggregate functions are not allowed in the GROUP BY clause");
                            return WRC.Abort;
                        }
                    }
                }

                // Advance to the next term of the compound
                p = p.Prior;
                compounds++;
            }

            // Resolve the ORDER BY on a compound SELECT after all terms of the compound have been resolved.
            return (isCompound && ResolveCompoundOrderBy(parse, leftmost) != 0 ? WRC.Abort : WRC.Prune);
        }

        public static bool ResolveExprNames(NameContext nc, ref Expr expr)
        {
            if (expr == null) return false;
#if MAX_EXPR_DEPTH
            {
                Parse parse = nc.Parse;
                if (Expr.CheckHeight(parse, expr.Height + nc.Parse.Height) != 0)
                    return true;
                parse.Height += expr.Height;
            }
#endif
            bool savedHasAgg = ((nc.NCFlags & NC.HasAgg) != 0);
            nc.NCFlags &= ~NC.HasAgg;
            Walker w = new Walker();
            w.ExprCallback = ResolveExprStep;
            w.SelectCallback = ResolveSelectStep;
            w.Parse = nc.Parse;
            w.u.NC = nc;
            w.WalkExpr(expr);
#if MAX_EXPR_DEPTH
            nc.Parse.Height -= expr.Height;
#endif
            if (nc.Errs > 0 || w.Parse.Errs > 0)
                E.ExprSetProperty(expr, EP.Error);
            if ((nc.NCFlags & NC.HasAgg) != 0)
                E.ExprSetProperty(expr, EP.Agg);
            else if (savedHasAgg)
                nc.NCFlags |= NC.HasAgg;
            return E.ExprHasProperty(expr, EP.Error);
        }

        public static void ResolveSelectNames(Parse parse, Select p, NameContext outerNC)
        {
            Debug.Assert(p != null);
            Walker w = new Walker();
            w.ExprCallback = ResolveExprStep;
            w.SelectCallback = ResolveSelectStep;
            w.Parse = parse;
            w.u.NC = outerNC;
            w.WalkSelect(p);
        }
    }
}
