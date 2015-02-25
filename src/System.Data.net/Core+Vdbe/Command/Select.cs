#define MAX_EXPR_DEPTH
using System;
using System.Diagnostics;
using System.Text;
using Pid = System.UInt32;

namespace Core
{
    public partial class Select
    {
        static void ClearSelect(Context ctx, Select p)
        {
            Expr.ListDelete(ctx, ref p.EList);
            Parse.SrcListDelete(ctx, ref p.Src);
            Expr.Delete(ctx, ref p.Where);
            Expr.ListDelete(ctx, ref p.GroupBy);
            Expr.Delete(ctx, ref p.Having);
            Expr.ListDelete(ctx, ref p.OrderBy);
            Select.Delete(ctx, ref p.Prior);
            Expr.Delete(ctx, ref p.Limit);
            Expr.Delete(ctx, ref p.Offset);
        }

        static void DestInit(SelectDest dest, SRT dest2, int parmId)
        {
            dest.Dest = dest2;
            dest.SDParmId = parmId;
            dest.AffSdst = '\0';
            dest.SdstId = 0;
            dest.Sdsts = 0;
        }

        public static Select New(Parse parse, int dummy1, SrcList src, int dummy2, int dummy3, int dummy4, int dummy5, SF selFlags, int dummy6, int dummy7) { return New(parse, null, src, null, null, null, null, selFlags, null, null); }
        public static Select New(Parse parse, ExprList list, SrcList src, Expr where_, ExprList groupBy, Expr having, ExprList orderBy, SF selFlags, Expr limit, Expr offset)
        {
            Context ctx = parse.Ctx;
            Select newSelect = new Select();
            Debug.Assert(ctx.MallocFailed || offset == null || limit != null); // OFFSET implies LIMIT
            // Select standin;
            if (newSelect == null)
            {
                Debug.Assert(ctx.MallocFailed);
                //newSelect = standin;
                //_memset(newSelect, 0, sizeof(newSelect));
            }
            if (list == null)
                list = Expr.ListAppend(parse, null, Expr.Expr_(ctx, TK.ALL, null));
            newSelect.EList = list;
            if (src == null) src = new SrcList();
            newSelect.Src = src;
            newSelect.Where = where_;
            newSelect.GroupBy = groupBy;
            newSelect.Having = having;
            newSelect.OrderBy = orderBy;
            newSelect.SelFlags = selFlags;
            newSelect.OP = TK.SELECT;
            newSelect.Limit = limit;
            newSelect.Offset = offset;
            Debug.Assert(offset == null || limit != null);
            newSelect.AddrOpenEphms[0] = (OP) - 1;
            newSelect.AddrOpenEphms[1] = (OP) - 1;
            newSelect.AddrOpenEphms[2] = (OP) - 1;
            if (ctx.MallocFailed)
            {
                ClearSelect(ctx, newSelect);
                //if (newSelect != standin) C._tagfree(ctx, ref newSelect);
                newSelect = null;
            }
            else
                Debug.Assert(newSelect.Src != null || parse.Errs > 0);
            //Debug.Assert(newSelect != standin);
            return newSelect;
        }

        public static void Delete(Context ctx, ref Select p)
        {
            if (p != null)
            {
                ClearSelect(ctx, p);
                C._tagfree(ctx, ref p);
            }
        }

        class Keyword
        {
            public byte I;        // Beginning of keyword text in zKeyText[]
            public byte Chars;    // Length of the keyword in characters
            public JT Code;     // Join type mask

            public Keyword(byte i, byte chars, JT code)
            {
                I = i;
                Chars = chars;
                Code = code;
            }
        }

        //   0123456789 123456789 123456789 123 */
        static readonly string _keyTexts = "naturaleftouterightfullinnercross";
        static readonly Keyword[] _keywords = new Keyword[]
        {
            /* natural */ new Keyword( 0,  7, JT.NATURAL                ),
            /* left    */ new Keyword( 6,  4, JT.LEFT|JT.OUTER          ),
            /* outer   */ new Keyword( 10, 5, JT.OUTER                  ),
            /* right   */ new Keyword( 14, 5, JT.RIGHT|JT.OUTER         ),
            /* full    */ new Keyword( 19, 4, JT.LEFT|JT.RIGHT|JT.OUTER ),
            /* inner   */ new Keyword( 23, 5, JT.INNER                  ),
            /* cross   */ new Keyword( 28, 5, JT.INNER|JT.CROSS         ),
        };


        public static JT JoinType(Parse parse, Token a, int null_3, int null_4) { return JoinType(parse, a, null, null); }
        public static JT JoinType(Parse parse, Token a, Token b, int null_4) { return JoinType(parse, a, b, null); }
        public static JT JoinType(Parse parse, Token a, Token b, Token c)
        {
            Token[] alls = new Token[3];
            alls[0] = a;
            alls[1] = b;
            alls[2] = c;
            JT jointype = 0;
            for (int i = 0; i < 3 && alls[i] != null; i++)
            {
                Token p = alls[i];
                int j;
                for (j = 0; j < _keywords.Length; j++)
                {
                    if (p.length == _keywords[j].Chars && p.data.StartsWith(_keyTexts.Substring(_keywords[j].I, _keywords[j].Chars), StringComparison.OrdinalIgnoreCase))
                    {
                        jointype |= _keywords[j].Code;
                        break;
                    }
                }
                C.ASSERTCOVERAGE(j == 0 || j == 1 || j == 2 || j == 3 || j == 4 || j == 5 || j == 6);
                if (j >= _keywords.Length)
                {
                    jointype |= JT.ERROR;
                    break;
                }
            }
            if ((jointype & (JT.INNER | JT.OUTER)) == (JT.INNER | JT.OUTER) || (jointype & JT.ERROR) != 0)
            {
                string sp = " ";
                Debug.Assert(b != null);
                if (c == null) sp = "";
                parse.ErrorMsg("unknown or unsupported join type: %T %T%s%T", a, b, sp, c);
                jointype = JT.INNER;
            }
            else if ((jointype & JT.OUTER) != 0 && (jointype & (JT.LEFT | JT.RIGHT)) != JT.LEFT)
            {
                parse.ErrorMsg("RIGHT and FULL OUTER JOINs are not currently supported");
                jointype = JT.INNER;
            }
            return jointype;
        }

        static int ColumnIndex(Table table, string colName)
        {
            for (int i = 0; i < table.Cols.length; i++)
                if (string.Equals(table.Cols[i].Name, colName, StringComparison.OrdinalIgnoreCase)) return i;
            return -1;
        }

        static bool TableAndColumnIndex(SrcList src, int n, string colName, ref int tableOut, ref int colIdOut)
        {
            for (int i = 0; i < n; i++)
            {
                int colId = ColumnIndex(src.Ids[i].Table, colName); // Index of column matching zCol
                if (colId >= 0)
                {
                    tableOut = i;
                    colIdOut = colId;
                    return true;
                }
            }
            return false;
        }

        static void AddWhereTerm(Parse parse, SrcList src, int leftId, int colLeftId, int rightId, int colRightId, bool isOuterJoin, ref Expr where_)
        {
            Context ctx = parse.Ctx;
            Debug.Assert(leftId < rightId);
            Debug.Assert(src.Srcs > rightId);
            Debug.Assert(src.Ids[leftId].Table != null);
            Debug.Assert(src.Ids[rightId].Table != null);

            Expr e1 = Walker.CreateColumnExpr(ctx, src, leftId, colLeftId);
            Expr e2 = Walker.CreateColumnExpr(ctx, src, rightId, colRightId);

            Expr eq = Expr.PExpr_(parse, TK.EQ, e1, e2, null);
            if (eq != null && isOuterJoin)
            {
                E.ExprSetProperty(eq, EP.FromJoin);
                Debug.Assert(!E.ExprHasAnyProperty(eq, EP.TokenOnly | EP.Reduced));
                E.ExprSetIrreducible(eq);
                eq.RightJoinTable = (short)e2.TableId;
            }
            where_ = Expr.And(ctx, where_, eq);
        }

        static void SetJoinExpr(Expr p, int table)
        {
            while (p != null)
            {
                E.ExprSetProperty(p, EP.FromJoin);
                Debug.Assert(!E.ExprHasAnyProperty(p, EP.TokenOnly | EP.Reduced));
                E.ExprSetIrreducible(p);
                p.RightJoinTable = (short)table;
                SetJoinExpr(p.Left, table);
                p = p.Right;
            }
        }

        public bool ProcessJoin(Parse parse)
        {
            SrcList src = Src; // All tables in the FROM clause
            SrcList.SrcListItem left; // Left table being joined
            SrcList.SrcListItem right; // Right table being joined
            int j;
            for (int i = 0; i < src.Srcs - 1; i++)
            {
                left = src.Ids[i];
                right = src.Ids[i + 1];
                Table leftTable = left.Table;
                Table rightTable = right.Table;

                if (C._NEVER(leftTable == null || rightTable == null)) continue;
                bool isOuter = ((right.Jointype & JT.OUTER) != 0);

                // When the NATURAL keyword is present, add WHERE clause terms for every column that the two tables have in common.
                if ((right.Jointype & JT.NATURAL) != 0)
                {
                    if (right.On != null || right.Using != null)
                    {
                        parse.ErrorMsg("a NATURAL join may not have an ON or USING clause");
                        return true;
                    }
                    for (j = 0; j < rightTable.Cols.length; j++)
                    {
                        string name = rightTable.Cols[j].Name; // Name of column in the right table
                        int leftId = 0; // Matching left table
                        int leftColId = 0;  // Matching column in the left table
                        if (TableAndColumnIndex(src, i + 1, name, ref leftId, ref leftColId))
                            AddWhereTerm(parse, src, leftId, leftColId, i + 1, j, isOuter, ref Where);
                    }
                }

                // Disallow both ON and USING clauses in the same join
                if (right.On != null && right.Using != null)
                {
                    parse.ErrorMsg("cannot have both ON and USING clauses in the same join");
                    return true;
                }

                // Add the ON clause to the end of the WHERE clause, connected by
                if (right.On != null)
                {
                    if (isOuter) SetJoinExpr(right.On, right.Cursor);
                    Where = Expr.And(parse.Ctx, Where, right.On);
                    right.On = null;
                }

                // Create extra terms on the WHERE clause for each column named in the USING clause.  Example: If the two tables to be joined are 
                // A and B and the USING clause names X, Y, and Z, then add this to the WHERE clause:    A.X=B.X AND A.Y=B.Y AND A.Z=B.Z
                // Report an error if any column mentioned in the USING clause is not contained in both tables to be joined.
                if (right.Using != null)
                {
                    IdList list = right.Using;
                    for (j = 0; j < list.Ids.length; j++)
                    {
                        string name = list.Ids[j].Name; // Name of the term in the USING clause
                        int leftId = 0; // Table on the left with matching column name
                        int leftColId = 0; // Column number of matching column on the left
                        int rightColId = ColumnIndex(rightTable, name); // Column number of matching column on the right
                        if (rightColId < 0 || TableAndColumnIndex(src, i + 1, name, ref leftId, ref leftColId))
                        {
                            parse.ErrorMsg("cannot join using column %s - column not present in both tables", name);
                            return true;
                        }
                        AddWhereTerm(parse, src, leftId, leftColId, i + 1, rightColId, isOuter, ref Where);
                    }
                }
            }
            return true;
        }

        static void PushOntoSorter(Parse parse, ExprList orderBy, Select select, int regData)
        {
            Vdbe v = parse.V;
            int exprs = orderBy.Exprs;
            int regBase = Expr.GetTempRange(parse, exprs + 2);
            int regRecord = Expr.GetTempReg(parse);
            Expr.CacheClear(parse);
            Expr.CodeExprList(parse, orderBy, regBase, false);
            v.AddOp2(OP.Sequence, orderBy.ECursor, regBase + exprs);
            Expr.CodeMove(parse, regData, regBase + exprs + 1, 1);
            v.AddOp3(OP.MakeRecord, regBase, exprs + 2, regRecord);
            v.AddOp2(OP.IdxInsert, orderBy.ECursor, regRecord);
            Expr.ReleaseTempReg(parse, regRecord);
            Expr.ReleaseTempRange(parse, regBase, exprs + 2);
            if (select.LimitId != 0)
            {
                int limitId = (select.OffsetId != 0 ? select.OffsetId + 1 : select.LimitId);
                int addr1 = v.AddOp1(OP.IfZero, limitId);
                v.AddOp2(OP.AddImm, limitId, -1);
                int addr2 = v.AddOp0(OP.Goto);
                v.JumpHere(addr1);
                v.AddOp1(OP.Last, orderBy.ECursor);
                v.AddOp1(OP.Delete, orderBy.ECursor);
                v.JumpHere(addr2);
            }
        }

        static void CodeOffset(Vdbe v, Select p, int continueId)
        {
            if (p.OffsetId != 0 && continueId != 0)
            {
                v.AddOp2(OP.AddImm, p.OffsetId, -1);
                int addr = v.AddOp1(OP.IfNeg, p.OffsetId);
                v.AddOp2(OP.Goto, 0, continueId);
                v.Comment("skip OFFSET records");
                v.JumpHere(addr);
            }
        }

        static void CodeDistinct(Parse parse, int table, int addrRepeatId, int n, int memId)
        {
            Vdbe v = parse.V;
            int r1 = Expr.GetTempReg(parse);
            v.AddOp4Int(OP.Found, table, addrRepeatId, memId, n);
            v.AddOp3(OP.MakeRecord, memId, n, r1);
            v.AddOp2(OP.IdxInsert, table, r1);
            Expr.ReleaseTempReg(parse, r1);
        }

#if !OMIT_SUBQUERY
        static bool CheckForMultiColumnSelectError(Parse parse, SelectDest dest, int exprs)
        {
            SRT dest2 = dest.Dest;
            if (exprs > 1 && (dest2 == SRT.Mem || dest2 == SRT.Set))
            {
                parse.ErrorMsg("only a single result allowed for a SELECT that is part of an expression");
                return true;
            }
            return false;
        }
#endif

        class DistinctCtx
        {
            public bool IsTnct;				// True if the DISTINCT keyword is present
            public WHERE_DISTINCT TnctType;	// One of the WHERE_DISTINCT_* operators
            public int TableTnct;			// Ephemeral table used for DISTINCT processing
            public int AddrTnct;	// Address of OP_OpenEphemeral opcode for tabTnct
        }

        static void SelectInnerLoop(Parse parse, Select p, ExprList list, int srcTable, int columns, ExprList orderBy, DistinctCtx distinct, SelectDest dest, int continueId, int breakId)
        {
            Vdbe v = parse.V;

            Debug.Assert(v != null);
            if (C._NEVER(v == null)) return;
            Debug.Assert(list != null);
            WHERE_DISTINCT hasDistinct = (distinct != null ? distinct.TnctType : WHERE_DISTINCT.NOOP); // True if the DISTINCT keyword is present
            if (orderBy == null && hasDistinct == (WHERE_DISTINCT)0)
                CodeOffset(v, p, continueId);

            // Pull the requested columns.
            int resultCols = (columns > 0 ? columns : list.Exprs); // Number of result columns
            if (dest.Sdsts == 0)
            {
                dest.SdstId = parse.Mems + 1;
                dest.Sdsts = resultCols;
                parse.Mems += resultCols;
            }
            else
                Debug.Assert(dest.Sdsts == resultCols);
            int regResult = dest.SdstId; // Start of memory holding result set
            SRT dest2 = dest.Dest; // How to dispose of results
            int i;
            if (columns > 0)
                for (i = 0; i < columns; i++)
                    v.AddOp3(OP.Column, srcTable, i, regResult + i);
            else if (dest2 != SRT.Exists)
            {
                // If the destination is an EXISTS(...) expression, the actual values returned by the SELECT are not required.
                Expr.CacheClear(parse);
                Expr.CodeExprList(parse, list, regResult, dest2 == SRT.Output);
            }
            columns = resultCols;

            // If the DISTINCT keyword was present on the SELECT statement and this row has been seen before, then do not make this row part of the result.
            if (hasDistinct != 0)
            {
                Debug.Assert(list != null);
                Debug.Assert(list.Exprs == columns);
                switch (distinct.TnctType)
                {
                    case WHERE_DISTINCT.ORDERED:
                        {
                            // Allocate space for the previous row
                            int regPrev = parse.Mems + 1; // Previous row content
                            parse.Mems += columns;

                            // Change the OP_OpenEphemeral coded earlier to an OP_Null sets the MEM_Cleared bit on the first register of the
                            // previous value.  This will cause the OP_Ne below to always fail on the first iteration of the loop even if the first
                            // row is all NULLs.
                            v.ChangeToNoop(distinct.AddrTnct);
                            Vdbe.VdbeOp op = v.GetOp(distinct.AddrTnct); // No longer required OpenEphemeral instr.
                            op.Opcode = OP.Null;
                            op.P1 = 1;
                            op.P2 = regPrev;

                            int jumpId = v.CurrentAddr() + columns; // Jump destination
                            for (i = 0; i < columns; i++)
                            {
                                CollSeq coll = list.Ids[i].Expr.CollSeq(parse);
                                if (i < columns - 1)
                                    v.AddOp3(OP.Ne, regResult + i, jumpId, regPrev + i);
                                else
                                    v.AddOp3(OP.Eq, regResult + i, continueId, regPrev + i);
                                v.ChangeP4(-1, coll, Vdbe.P4T.COLLSEQ);
                                v.ChangeP5(AFF.BIT_NULLEQ);
                            }
                            Debug.Assert(v.CurrentAddr() == jumpId);
                            v.AddOp3(OP.Copy, regResult, regPrev, columns - 1);
                            break;
                        }
                    case WHERE_DISTINCT.UNIQUE:
                        {
                            v.ChangeToNoop(distinct.AddrTnct);
                            break;
                        }
                    default:
                        {
                            Debug.Assert(distinct.TnctType == WHERE_DISTINCT.UNORDERED);
                            CodeDistinct(parse, distinct.TableTnct, continueId, columns, regResult);
                            break;
                        }
                }
                if (orderBy != null)
                    CodeOffset(v, p, continueId);
            }

            int paramId = dest.SDParmId; // First argument to disposal method
            switch (dest2)
            {
#if !OMIT_COMPOUND_SELECT
                case SRT.Union:
                    {
                        // In this mode, write each query result to the key of the temporary table iParm.
                        int r1 = Expr.GetTempReg(parse);
                        v.AddOp3(OP.MakeRecord, regResult, columns, r1);
                        v.AddOp2(OP.IdxInsert, paramId, r1);
                        Expr.ReleaseTempReg(parse, r1);
                        break;
                    }
                case SRT.Except:
                    {
                        // Construct a record from the query result, but instead of saving that record, use it as a key to delete elements from
                        // the temporary table iParm.
                        v.AddOp3(OP.IdxDelete, paramId, regResult, columns);
                        break;
                    }
#endif
                case SRT.Table:
                case SRT.EphemTab:
                    {
                        // Store the result as data using a unique key.
                        int r1 = Expr.GetTempReg(parse);
                        C.ASSERTCOVERAGE(dest2 == SRT.Table);
                        C.ASSERTCOVERAGE(dest2 == SRT.EphemTab);
                        v.AddOp3(OP.MakeRecord, regResult, columns, r1);
                        if (orderBy != null)
                            PushOntoSorter(parse, orderBy, p, r1);
                        else
                        {
                            int r2 = Expr.GetTempReg(parse);
                            v.AddOp2(OP.NewRowid, paramId, r2);
                            v.AddOp3(OP.Insert, paramId, r1, r2);
                            v.ChangeP5(Vdbe.OPFLAG.APPEND);
                            Expr.ReleaseTempReg(parse, r2);
                        }
                        Expr.ReleaseTempReg(parse, r1);
                        break;
                    }
#if !OMIT_SUBQUERY
                case SRT.Set:
                    {
                        // If we are creating a set for an "expr IN (SELECT ...)" construct, then there should be a single item on the stack.  Write this
                        // item into the set table with bogus data.
                        Debug.Assert(columns == 1);
                        dest.AffSdst = list.Ids[0].Expr.CompareAffinity(dest.AffSdst);
                        // At first glance you would think we could optimize out the ORDER BY in this case since the order of entries in the set
                        // does not matter.  But there might be a LIMIT clause, in which case the order does matter
                        if (orderBy != null)
                            PushOntoSorter(parse, orderBy, p, regResult);
                        else
                        {
                            int r1 = Expr.GetTempReg(parse);
                            v.AddOp4(OP.MakeRecord, regResult, 1, r1, dest.AffSdst, 1);
                            Expr.CacheAffinityChange(parse, regResult, 1);
                            v.AddOp2(OP.IdxInsert, paramId, r1);
                            Expr.ReleaseTempReg(parse, r1);
                        }
                        break;
                    }
                case SRT.Exists:
                    {
                        // If any row exist in the result set, record that fact and abort.
                        v.AddOp2(OP.Integer, 1, paramId);
                        // The LIMIT clause will terminate the loop for us
                        break;
                    }
                case SRT.Mem:
                    {
                        // If this is a scalar select that is part of an expression, then store the results in the appropriate memory cell and break out
                        // of the scan loop.
                        Debug.Assert(columns == 1);
                        if (orderBy != null)
                            PushOntoSorter(parse, orderBy, p, regResult);
                        else
                            Expr.CodeMove(parse, regResult, paramId, 1);
                        // The LIMIT clause will jump out of the loop for us
                        break;
                    }
#endif
                case SRT.Coroutine:
                case SRT.Output:
                    {
                        // Send the data to the callback function or to a subroutine.  In the case of a subroutine, the subroutine itself is responsible for
                        // popping the data from the stack.
                        C.ASSERTCOVERAGE(dest2 == SRT.Coroutine);
                        C.ASSERTCOVERAGE(dest2 == SRT.Output);
                        if (orderBy != null)
                        {
                            int r1 = Expr.GetTempReg(parse);
                            v.AddOp3(OP.MakeRecord, regResult, columns, r1);
                            PushOntoSorter(parse, orderBy, p, r1);
                            Expr.ReleaseTempReg(parse, r1);
                        }
                        else if (dest2 == SRT.Coroutine)
                            v.AddOp1(OP.Yield, dest.SDParmId);
                        else
                        {
                            v.AddOp2(OP.ResultRow, regResult, columns);
                            Expr.CacheAffinityChange(parse, regResult, columns);
                        }
                        break;
                    }
#if !OMIT_TRIGGER
                default:
                    {
                        // Discard the results.  This is used for SELECT statements inside the body of a TRIGGER.  The purpose of such selects is to call
                        // user-defined functions that have side effects.  We do not care about the actual results of the select.
                        Debug.Assert(dest2 == SRT.Discard);
                        break;
                    }
#endif
            }

            // Jump to the end of the loop if the LIMIT is reached.  Except, if there is a sorter, in which case the sorter has already limited the output for us.
            if (orderBy == null && p.LimitId != 0)
                v.AddOp3(OP.IfZero, p.LimitId, breakId, -1);
        }

        static KeyInfo KeyInfoFromExprList(Parse parse, ExprList list)
        {
            Context ctx = parse.Ctx;
            int exprs = list.Exprs;
            KeyInfo info = new KeyInfo();
            if (info != null)
            {
                info.SortOrders = new SO[exprs];
                info.Colls = new CollSeq[exprs];
                info.Fields = (ushort)exprs;
                info.Encode = E.CTXENCODE(ctx);
                info.Ctx = ctx;
                int i;
                ExprList.ExprListItem item;
                for (i = 0; i < exprs; i++)
                {
                    item = list.Ids[i];
                    CollSeq coll = item.Expr.CollSeq(parse);
                    if (coll == null)
                        coll = ctx.DefaultColl;
                    info.Colls[i] = coll;
                    info.SortOrders[i] = item.SortOrder;
                }
            }
            return info;
        }

        #region Explain

#if !OMIT_COMPOUND_SELECT
        static string SelectOpName(TK id)
        {
            switch (id)
            {
                case TK.ALL: return "UNION ALL";
                case TK.INTERSECT: return "INTERSECT";
                case TK.EXCEPT: return "EXCEPT";
                default: return "UNION";
            }
        }
#endif

#if !OMIT_EXPLAIN
        static void ExplainTempTable(Parse parse, string usage_)
        {
            if (parse.Explain == 2)
            {
                Vdbe v = parse.V;
                string msg = C._mtagprintf(parse.Ctx, "USE TEMP B-TREE FOR %s", usage_);
                v.AddOp4(OP.Explain, parse.SelectId, 0, 0, msg, Vdbe.P4T.DYNAMIC);
            }
        }

        static void ExplainSetInteger(ref int a, int b) { a = b; }
        static void ExplainSetInteger(ref byte a, int b) { a = (byte)b; }
#else
        static void ExplainTempTable(ref int a, int b) { a = b; }
        static void ExplainSetInteger(ref int a, int b) { a = b; }
#endif

#if !OMIT_EXPLAIN && !OMIT_COMPOUND_SELECT
        static void ExplainComposite(Parse parse, TK op, int sub1Id, int sub2Id, bool useTmp)
        {
            Debug.Assert(op == TK.UNION || op == TK.EXCEPT || op == TK.INTERSECT || op == TK.ALL);
            if (parse.Explain == 2)
            {
                Vdbe v = parse.V;
                string msg = C._mtagprintf(parse.Ctx, "COMPOUND SUBQUERIES %d AND %d %s(%s)", sub1Id, sub2Id, (useTmp ? "USING TEMP B-TREE " : ""), SelectOpName(op));
                v.AddOp4(OP.Explain, parse.SelectId, 0, 0, msg, Vdbe.P4T.DYNAMIC);
            }
        }
#else
        static void ExplainComposite(Parse v, int w, int x, int y, bool z) { }
#endif

        #endregion

        static void GenerateSortTail(Parse parse, Select p, Vdbe v, int columns, SelectDest dest)
        {
            int addrBreak = v.MakeLabel(); // Jump here to exit loop
            int addrContinue = v.MakeLabel(); // Jump here for next cycle
            ExprList orderBy = p.OrderBy;
            SRT dest2 = dest.Dest;
            int parmId = dest.SDParmId;

            int tabId = orderBy.ECursor;
            int regRow = Expr.GetTempReg(parse);
            int pseudoTab = 0;
            int regRowid;
            if (dest2 == SRT.Output || dest2 == SRT.Coroutine)
            {
                pseudoTab = parse.Tabs++;
                v.AddOp3(OP.OpenPseudo, pseudoTab, regRow, columns);
                regRowid = 0;
            }
            else
                regRowid = Expr.GetTempReg(parse);
            int addr;
            if ((p.SelFlags & SF.UseSorter) != 0)
            {
                int regSortOut = ++parse.Mems;
                int ptab2 = parse.Tabs++;
                v.AddOp3(OP.OpenPseudo, ptab2, regSortOut, orderBy.Exprs + 2);
                addr = 1 + v.AddOp2(OP.SorterSort, tabId, addrBreak);
                CodeOffset(v, p, addrContinue);
                v.AddOp2(OP.SorterData, tabId, regSortOut);
                v.AddOp3(OP.Column, ptab2, orderBy.Exprs + 1, regRow);
                v.ChangeP5(Vdbe.OPFLAG.CLEARCACHE);
            }
            else
            {
                addr = 1 + v.AddOp2(OP.Sort, tabId, addrBreak);
                CodeOffset(v, p, addrContinue);
                v.AddOp3(OP.Column, tabId, orderBy.Exprs + 1, regRow);
            }
            switch (dest2)
            {
                case SRT.Table:
                case SRT.EphemTab:
                    {
                        C.ASSERTCOVERAGE(dest2 == SRT.Table);
                        C.ASSERTCOVERAGE(dest2 == SRT.EphemTab);
                        v.AddOp2(OP.NewRowid, parmId, regRowid);
                        v.AddOp3(OP.Insert, parmId, regRow, regRowid);
                        v.ChangeP5(Vdbe.OPFLAG.APPEND);
                        break;
                    }
#if !OMIT_SUBQUERY
                case SRT.Set:
                    {
                        Debug.Assert(columns == 1);
                        v.AddOp4(OP.MakeRecord, regRow, 1, regRowid, dest->AffSdst, 1);
                        Expr.CacheAffinityChange(parse, regRow, 1);
                        v.AddOp2(OP.IdxInsert, parmId, regRowid);
                        break;
                    }
                case SRT.Mem:
                    {
                        Debug.Assert(columns == 1);
                        Expr.CodeMove(parse, regRow, parmId, 1);
                        // The LIMIT clause will terminate the loop for us
                        break;
                    }
#endif
                default:
                    {
                        Debug.Assert(dest2 == SRT.Output || dest2 == SRT.Coroutine);
                        C.ASSERTCOVERAGE(dest2 == SRT.Output);
                        C.ASSERTCOVERAGE(dest2 == SRT.Coroutine);
                        for (int i = 0; i < columns; i++)
                        {
                            Debug.Assert(regRow != dest.SdstId + i);
                            v.AddOp3(OP.Column, pseudoTab, i, dest.SdstId + i);
                            if (i == 0)
                                v.ChangeP5(Vdbe.OPFLAG.CLEARCACHE);
                        }
                        if (dest2 == SRT.Output)
                        {
                            v.AddOp2(OP.ResultRow, dest.SdstId, columns);
                            Expr.CacheAffinityChange(parse, dest.SdstId, columns);
                        }
                        else
                            v.AddOp1(OP.Yield, dest.SDParmId);
                        break;
                    }
            }
            Expr.ReleaseTempReg(parse, regRow);
            Expr.ReleaseTempReg(parse, regRowid);

            // The bottom of the loop
            v.ResolveLabel(addrContinue);
            v.AddOp2(OP.Next, tabId, addr);
            v.ResolveLabel(addrBreak);
            if (dest2 == SRT.Output || dest2 == SRT.Coroutine)
                v.AddOp2(OP.Close, pseudoTab, 0);
        }

        static string ColumnType(NameContext nc, Expr expr, ref string originDbNameOut, ref string originTableNameOut, ref string originColumnNameOut)
        {
            string typeName = null;
            string originDbName = null;
            string originTableName = null;
            string originColumnName = null;
            int j;
            if (C._NEVER(expr == null) || nc.SrcList == null) return null;

            switch (expr.OP)
            {
                case TK.AGG_COLUMN:
                case TK.COLUMN:
                    {
                        // The expression is a column. Locate the table the column is being extracted from in NameContext.pSrcList. This table may be real
                        // database table or a subquery.
                        Table table = null; // Table structure column is extracted from
                        Select s = null; // Select the column is extracted from
                        int colId = expr.ColumnIdx; // Index of column in pTab
                        C.ASSERTCOVERAGE(expr.OP == TK.AGG_COLUMN);
                        C.ASSERTCOVERAGE(expr.OP == TK.COLUMN);
                        while (nc != null && table == null)
                        {
                            SrcList tabList = nc.SrcList;
                            for (j = 0; j < tabList.Srcs && tabList.Ids[j].Cursor != expr.TableId; j++) ;
                            if (j < tabList.Srcs)
                            {
                                table = tabList.Ids[j].Table;
                                s = tabList.Ids[j].Select;
                            }
                            else
                                nc = nc.Next;
                        }

                        if (table == null)
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

                        Debug.Assert(table != null && expr.Table == table);
                        if (s != null)
                        {
                            // The "table" is actually a sub-select or a view in the FROM clause of the SELECT statement. Return the declaration type and origin
                            // data for the result-set column of the sub-select.
                            if (colId >= 0 && C._ALWAYS(colId < s.EList.Exprs))
                            {
                                // If colId is less than zero, then the expression requests the rowid of the sub-select or view. This expression is legal (see 
                                // test case misc2.2.2) - it always evaluates to NULL.
                                NameContext sNC = new NameContext();
                                Expr p = s.EList.Ids[colId].Expr;
                                sNC.SrcList = s.Src;
                                sNC.Next = nc;
                                sNC.Parse = nc.Parse;
                                typeName = ColumnType(sNC, p, ref originDbName, ref originTableName, ref originColumnName);
                            }
                        }
                        else if (C._ALWAYS(table.Schema))
                        {
                            // A real table
                            Debug.Assert(s == null);
                            if (colId < 0) colId = table.PKey;
                            Debug.Assert(colId == -1 || (colId >= 0 && colId < table.Cols.length));
                            if (colId < 0)
                            {
                                typeName = "INTEGER";
                                originColumnName = "rowid";
                            }
                            else
                            {
                                typeName = table.Cols[colId].Type;
                                originColumnName = table.Cols[colId].Name;
                            }
                            originTableName = table.Name;
                            if (nc.Parse != null)
                            {
                                Context ctx = nc.Parse.Ctx;
                                int db = Prepare.SchemaToIndex(ctx, table.Schema);
                                originDbName = ctx.DBs[db].Name;
                            }
                        }
                        break;
                    }
#if !OMIT_SUBQUERY
                case TK.SELECT:
                    {
                        // The expression is a sub-select. Return the declaration type and origin info for the single column in the result set of the SELECT statement.
                        NameContext sNC = new NameContext();
                        Select s = expr.x.Select;
                        Expr p = s.EList.Ids[0].Expr;
                        Debug.Assert(E.ExprHasProperty(expr, EP.xIsSelect));
                        sNC.SrcList = s.Src;
                        sNC.Next = nc;
                        sNC.Parse = nc.Parse;
                        typeName = ColumnType(sNC, p, ref originDbName, ref originTableName, ref originColumnName);
                        break;
                    }
#endif
            }

            originDbNameOut = originDbName;
            originTableNameOut = originTableName;
            originColumnNameOut = originColumnName;
            return typeName;
        }

        static void GenerateColumnTypes(Parse parse, SrcList tabList, ExprList list)
        {
#if !OMIT_DECLTYPE
            Vdbe v = parse.V;
            NameContext sNC = new NameContext();
            sNC.SrcList = tabList;
            sNC.Parse = parse;
            for (int i = 0; i < list.Exprs; i++)
            {
                Expr p = list.Ids[i].Expr;
                string typeName;
#if ENABLE_COLUMN_METADATA
                string origDbName = null;
                string origTableName = null;
                string origColumnName = null;
                typeName = ColumnType(sNC, p, ref origDbName, ref origTableName, ref origColumnName);

                // The vdbe must make its own copy of the column-type and other column specific strings, in case the schema is reset before this
                // virtual machine is deleted.
                v.SetColName(i, COLNAME_DATABASE, origDbName, C.DESTRUCTOR_TRANSIENT);
                v.SetColName(i, COLNAME_TABLE, origTableName, C.DESTRUCTOR_TRANSIENT);
                v.SetColName(i, COLNAME_COLUMN, origColumnName, C.DESTRUCTOR_TRANSIENT);
#else
                string dummy1 = null;
                typeName = ColumnType(sNC, p, ref dummy1, ref dummy1, ref dummy1);
#endif
                v.SetColName(i, COLNAME_DECLTYPE, typeName, C.DESTRUCTOR_TRANSIENT);
            }
#endif
        }

        static void GenerateColumnNames(Parse parse, SrcList tabList, ExprList list)
        {
#if !OMIT_EXPLAIN
            // If this is an EXPLAIN, skip this step
            if (parse.Explain != 0)
                return;
#endif
            Vdbe v = parse.V;
            Context ctx = parse.Ctx;
            if (parse.ColNamesSet != 0 || C._NEVER(v == null) || ctx.MallocFailed) return;
            parse.ColNamesSet = 1;
            bool fullNames = ((ctx.Flags & Context.FLAG.FullColNames) != 0);
            bool shortNames = ((ctx.Flags & Context.FLAG.ShortColNames) != 0);
            v.SetNumCols(list.Exprs);
            for (int i = 0; i < list.Exprs; i++)
            {
                Expr p = list.Ids[i].Expr;
                if (C._NEVER(p == null)) continue;
                if (list.Ids[i].Name != null)
                {
                    string name = list.Ids[i].Name;
                    v.SetColName(i, COLNAME_NAME, name, C.DESTRUCTOR_TRANSIENT);
                }
                else if ((p.OP == TK.COLUMN || p.OP == TK.AGG_COLUMN) && tabList != null)
                {
                    int colId = p.ColumnIdx;
                    int j;
                    for (j = 0; C._ALWAYS(j < tabList.Srcs); j++)
                        if (tabList.Ids[j].Cursor == p.TableId) break;
                    Debug.Assert(j < tabList.Srcs);
                    Table table = tabList.Ids[j].Table;
                    if (colId < 0) colId = table.PKey;
                    Debug.Assert(colId == -1 || (colId >= 0 && colId < table.Cols.length));
                    string colName = (colId < 0 ? "rowid" : table.Cols[colId].Name);
                    if (!shortNames && !fullNames)
                        v.SetColName(i, COLNAME_NAME, list.Ids[i].Span, C.DESTRUCTOR_DYNAMIC);
                    else if (fullNames)
                    {
                        string name = C._mtagprintf(ctx, "%s.%s", table.Name, colName);
                        v.SetColName(i, COLNAME_NAME, name, C.DESTRUCTOR_DYNAMIC);
                    }
                    else
                        v.SetColName(i, COLNAME_NAME, colName, C.DESTRUCTOR_TRANSIENT);
                }
                else
                    v.SetColName(i, COLNAME_NAME, list.Ids[i].Span, C.DESTRUCTOR_TRANSIENT);
            }
            GenerateColumnTypes(parse, tabList, list);
        }

        static RC SelectColumnsFromExprList(Parse parse, ExprList list, ref short colsLengthOut, ref Column[] colsOut)
        {
            Context ctx = parse.Ctx; // Database connection
            int j;

            int colsLength; // Number of columns in the result set
            Column[] cols; // For looping over result columns
            if (list != null)
            {
                colsLength = list.Exprs;
                cols = new Column[colsLength];
                C.ASSERTCOVERAGE(cols == null);
            }
            else
            {
                colsLength = 0;
                cols = null;
            }
            colsLengthOut = (short)colsLength;
            colsOut = cols;

            int i;
            Column col; // For looping over result columns
            for (i = 0; i < colsLength; i++)//, pCol++)
            {
                if (cols[i] == null) cols[i] = new Column();
                col = cols[i];
                // Get an appropriate name for the column
                Expr p = list.Ids[i].Expr; // Expression for a single result column
                string name; // Column name
                if (list.Ids[i].Name != null && (name = list.Ids[i].Name) != null) { }
                //: name = _tagstrdup(ctx, name); // If the column contains an "AS <name>" phrase, use <name> as the name 
                else
                {
                    Expr colExpr = p; // The expression that is the result column name

                    while (colExpr.OP == TK.DOT)
                    {
                        colExpr = colExpr.Right;
                        Debug.Assert(colExpr != null);
                    }
                    if (colExpr.OP == TK.COLUMN && C._ALWAYS(colExpr.Table != null))
                    {
                        // For columns use the column name name
                        int colId = colExpr.ColumnIdx;
                        Table table = colExpr.Table; // Table associated with this expression
                        if (colId < 0) colId = table.PKey;
                        name = C._mtagprintf(ctx, "%s", (colId >= 0 ? table.Cols[colId].Name : "rowid"));
                    }
                    else if (colExpr.OP == TK.ID)
                    {
                        Debug.Assert(!E.ExprHasProperty(colExpr, EP.IntValue));
                        name = C._mtagprintf(ctx, "%s", colExpr.u.Token);
                    }
                    else
                        name = C._mtagprintf(ctx, "%s", list.Ids[i].Span); // Use the original text of the column expression as its name
                }
                if (ctx.MallocFailed)
                {
                    C._tagfree(ctx, ref name);
                    break;
                }

                // Make sure the column name is unique.  If the name is not unique, append a integer to the name so that it becomes unique.
                int nameLength = name.Length; // Size of name in zName[]
                int cnt; // Index added to make the name unique
                for (j = cnt = 0; j < i; j++)
                {
                    if (string.Equals(cols[j].Name, name, StringComparison.OrdinalIgnoreCase))
                    {
                        name = name.Substring(0, nameLength);
                        string newName = C._mtagprintf(ctx, "%s:%d", name, ++cnt);
                        C._tagfree(ctx, ref name);
                        name = newName;
                        j = -1;
                        if (name == null) break;
                    }
                }
                col.Name = name;
            }
            if (ctx.MallocFailed)
            {
                for (j = 0; j < i; j++)
                    C._tagfree(ctx, ref cols[j].Name);
                C._tagfree(ctx, ref cols);
                colsOut = null;
                colsLengthOut = 0;
                return RC.NOMEM;
            }
            return RC.OK;
        }

        static void SelectAddColumnTypeAndCollation(Parse parse, int colsLength, Column[] cols, Select select)
        {
            Context ctx = parse.Ctx;
            Debug.Assert(select != null);
            Debug.Assert((select.SelFlags & SF.Resolved) != 0);
            Debug.Assert(colsLength == select.EList.Exprs || ctx.MallocFailed);
            if (ctx.MallocFailed) return;
            NameContext sNC = new NameContext();
            sNC.SrcList = select.Src;
            ExprList.ExprListItem[] ids = select.EList.Ids;
            int i;
            Column col;
            for (i = 0; i < colsLength; i++)
            {
                col = cols[i];
                Expr p = ids[i].Expr;
                string dummy1 = null;
                col.Type = ColumnType(sNC, p, ref dummy1, ref dummy1, ref dummy1);
                col.Affinity = p.Affinity();
                if (col.Affinity == 0) col.Affinity = AFF.NONE;
                CollSeq coll = p.CollSeq(parse);
                if (coll != null)
                    col.Coll = coll.Name;
            }
        }

        public Table ResultSetOfSelect(Parse parse)
        {
            Context ctx = parse.Ctx;
            Context.FLAG savedFlags = ctx.Flags;
            ctx.Flags &= ~Context.FLAG.FullColNames;
            ctx.Flags |= Context.FLAG.ShortColNames;
            Select select = this;
            select.Prep(parse, null);
            if (parse.Errs != 0) return null;
            while (select.Prior != null) select = select.Prior;
            ctx.Flags = savedFlags;
            Table table = new Table();
            if (table == null)
                return null;
            // The sqlite3ResultSetOfSelect() is only used n contexts where lookaside is disabled
            Debug.Assert(!ctx.Lookaside.Enabled);
            table.Refs = 1;
            table.Name = null;
            table.RowEst = 1000000;
            SelectColumnsFromExprList(parse, select.EList, ref table.Cols.length, ref table.Cols.data);
            SelectAddColumnTypeAndCollation(parse, table.Cols.length, table.Cols.data, select);
            table.PKey = -1;
            if (ctx.MallocFailed)
            {
                Parse.DeleteTable(ctx, ref table);
                return null;
            }
            return table;
        }
    }

    partial class Parse
    {
        public Vdbe GetVdbe()
        {
            Vdbe v = V;
            if (v == null)
            {
                v = V = Vdbe.Create(Ctx);
#if !OMIT_TRACE
                if (v != null)
                    v.AddOp0(OP.Trace);
#endif
            }
            return v;
        }
    }

    partial class Select
    {
        static void ComputeLimitRegisters(Parse parse, Select p, int breakId)
        {
            int limitId = 0;
            int offsetId;
            if (p.LimitId != 0) return;

            // "LIMIT -1" always shows all rows.  There is some contraversy about what the correct behavior should be.
            // The current implementation interprets "LIMIT 0" to mean no rows.
            Expr.CacheClear(parse);
            Debug.Assert(p.Offset == null || p.Limit != null);
            if (p.Limit != null)
            {
                p.LimitId = limitId = ++parse.Mems;
                Vdbe v = parse.GetVdbe();
                if (C._NEVER(v == null)) return;  // VDBE should have already been allocated
                int n = 0;
                if (p.Limit.IsInteger(ref n))
                {
                    v.AddOp2(OP.Integer, n, limitId);
                    v.Comment("LIMIT counter");
                    if (n == 0)
                        v.AddOp2(OP.Goto, 0, breakId);
                    else if (p.SelectRows > (double)n)
                        p.SelectRows = (double)n;
                }
                else
                {
                    Expr.Code(parse, p.Limit, limitId);
                    v.AddOp1(OP.MustBeInt, limitId);
                    v.Comment("LIMIT counter");
                    v.AddOp2(OP.IfZero, limitId, breakId);
                }
                if (p.Offset != null)
                {
                    p.OffsetId = offsetId = ++parse.Mems;
                    parse.Mems++; // Allocate an extra register for limit+offset
                    Expr.Code(parse, p.Offset, offsetId);
                    v.AddOp1(OP.MustBeInt, offsetId);
                    v.Comment("OFFSET counter");
                    int addr1 = v.AddOp1(OP.IfPos, offsetId);
                    v.AddOp2(OP.Integer, 0, offsetId);
                    v.JumpHere(addr1);
                    v.AddOp3(OP.Add, limitId, offsetId, offsetId + 1);
                    v.Comment("LIMIT+OFFSET");
                    addr1 = v.AddOp1(OP.IfPos, limitId);
                    v.AddOp2(OP.Integer, -1, offsetId + 1);
                    v.JumpHere(addr1);
                }
            }
        }

#if !OMIT_COMPOUND_SELECT
        static CollSeq MultiSelectCollSeq(Parse parse, Select p, int colId)
        {
            CollSeq r = (p.Prior != null ? MultiSelectCollSeq(parse, p.Prior, colId) : null);
            Debug.Assert(colId >= 0);
            if (r == null && colId < p.EList.Exprs)
                r = p.EList.Ids[colId].Expr.CollSeq(parse);
            return r;
        }

        static int MultiSelect(Parse parse, Select p, SelectDest dest)
        {
            RC rc = RC.OK;       // Success code from a subroutine
#if !OMIT_EXPLAIN
            int sub1Id = 0; // EQP id of left-hand query
            int sub2Id = 0; // EQP id of right-hand query
#endif

            // Make sure there is no ORDER BY or LIMIT clause on prior SELECTs.  Only the last (right-most) SELECT in the series may have an ORDER BY or LIMIT.
            Debug.Assert(p != null && p.Prior != null); // Calling function guarantees this much
            Context ctx = parse.Ctx; // Database connection
            Select prior = p.Prior; // Another SELECT immediately to our left
            Debug.Assert(prior.Rightmost != prior);
            Debug.Assert(prior.Rightmost == p.Rightmost);
            SelectDest dest2 = dest; // Alternative data destination
            if (prior.OrderBy != null)
            {
                parse.ErrorMsg("ORDER BY clause should come after %s not before", SelectOpName(p.OP));
                rc = 1;
                goto multi_select_end;
            }
            if (prior.Limit != null)
            {
                parse.ErrorMsg("LIMIT clause should come after %s not before", SelectOpName(p.OP));
                rc = 1;
                goto multi_select_end;
            }

            Vdbe v = parse.GetVdbe(); // Generate code to this VDBE
            Debug.Assert(v != null); // The VDBE already created by calling function

            // Create the destination temporary table if necessary
            if (dest2.Dest == SRT.EphemTab)
            {
                Debug.Assert(p.EList != null);
                v.AddOp2(OP.OpenEphemeral, dest2.SDParmId, p.EList.Exprs);
                v.ChangeP5(BTREE_UNORDERED);
                dest2.Dest = SRT.Table;
            }

            // Make sure all SELECTs in the statement have the same number of elements in their result sets.
            Debug.Assert(p.EList != null && prior.EList != null);
            if (p.EList.Exprs != prior.EList.Exprs)
            {
                parse.ErrorMsg("SELECTs to the left and right of %s do not have the same number of result columns", SelectOpName(p.OP));
                rc = 1;
                goto multi_select_end;
            }

            // Compound SELECTs that have an ORDER BY clause are handled separately.
            if (p.OrderBy != null)
                return MultiSelectOrderBy(parse, p, dest);

            // Generate code for the left and right SELECT statements.
            Select delete_ = null; // Chain of simple selects to delete
            switch (p.OP)
            {
                case TK.ALL:
                    {
                        int addr = 0;
                        int limits = 0;
                        Debug.Assert(prior.Limit == null);
                        prior.LimitId = p.LimitId;
                        prior.OffsetId = p.OffsetId;
                        prior.Limit = p.Limit;
                        prior.Offset = p.Offset;
                        ExplainSetInteger(ref sub1Id, parse.NextSelectId);
                        rc = Select.Select_(parse, prior, ref dest2);
                        p.Limit = null;
                        p.Offset = null;
                        if (rc != 0)
                            goto multi_select_end;
                        p.Prior = null;
                        p.LimitId = prior.LimitId;
                        p.OffsetId = prior.OffsetId;
                        if (p.LimitId != 0)
                        {
                            addr = v.AddOp1(OP.IfZero, p.LimitId);
                            v.Comment("Jump ahead if LIMIT reached");
                        }
                        ExplainSetInteger(ref sub2Id, parse.NextSelectId);
                        rc = Select.Select_(parse, p, ref dest2);
                        C.ASSERTCOVERAGE(rc != RC.OK);
                        delete_ = p.Prior;
                        p.Prior = prior;
                        p.SelectRows += prior.SelectRows;
                        if (prior.Limit != null && prior.Limit.IsInteger(ref limits) && p.SelectRows > (double)limits)
                            p.SelectRows = (double)limits;
                        if (addr != 0)
                            v.JumpHere(addr);
                        break;
                    }
                case TK.EXCEPT:
                case TK.UNION:
                    {
                        C.ASSERTCOVERAGE(p.OP == TK.EXCEPT);
                        C.ASSERTCOVERAGE(p.OP == TK.UNION);
                        SRT priorOp = SRT.Union; // The SRT_ operation to apply to prior selects
                        int unionTab; // VdbeCursor number of the temporary table holding result
                        if (dest2.Dest == priorOp && C._ALWAYS(p.Limit == null && p.Offset == null))
                        {
                            // We can reuse a temporary table generated by a SELECT to our right.
                            Debug.Assert(p.Rightmost != p); // Can only happen for leftward elements of a 3-way or more compound */
                            Debug.Assert(p.Limit == null); // Not allowed on leftward elements
                            Debug.Assert(p.Offset == null); // Not allowed on leftward elements
                            unionTab = dest2.SDParmId;
                        }
                        else
                        {
                            // We will need to create our own temporary table to hold the intermediate results.
                            unionTab = parse.Tabs++;
                            Debug.Assert(p.OrderBy == null);
                            int addr = v.AddOp2(OP.OpenEphemeral, unionTab, 0);
                            Debug.Assert(p.AddrOpenEphms[0] == -1);
                            p.AddrOpenEphms[0] = addr;
                            p.Rightmost.SelFlags |= SF.UsesEphemeral;
                            Debug.Assert(p.EList != null);
                        }

                        // Code the SELECT statements to our left
                        Debug.Assert(prior.OrderBy == null);
                        SelectDest uniondest = new SelectDest();
                        DestInit(uniondest, priorOp, unionTab);
                        ExplainSetInteger(ref sub1Id, parse.NextSelectId);
                        rc = Select.Select_(parse, prior, ref uniondest);
                        if (rc != 0)
                            goto multi_select_end;

                        // Code the current SELECT statement
                        SRT op = 0; // One of the SRT_ operations to apply to self
                        if (p.OP == TK.EXCEPT)
                            op = SRT.Except;
                        else
                        {
                            Debug.Assert(p.OP == TK.UNION);
                            op = SRT.Union;
                        }
                        p.Prior = null;
                        Expr limit = p.Limit; // Saved values of p.nLimit and p.nOffset
                        p.Limit = null;
                        Expr offset = p.Offset; // Saved values of p.nLimit and p.nOffset
                        p.Offset = null;
                        uniondest.Dest = op;
                        ExplainSetInteger(ref sub2Id, parse.NextSelectId);
                        rc = Select.Select_(parse, p, ref uniondest);
                        C.ASSERTCOVERAGE(rc != RC.OK);
                        // Query flattening in sqlite3Select() might refill p->pOrderBy. Be sure to delete p->pOrderBy, therefore, to avoid a memory leak.
                        Expr.ListDelete(ctx, ref p.OrderBy);
                        delete_ = p.Prior;
                        p.Prior = prior;
                        p.OrderBy = null;
                        if (p.OP == TK.UNION)
                            p.SelectRows += prior.SelectRows;
                        Expr.Delete(ctx, ref p.Limit);
                        p.Limit = limit;
                        p.Offset = offset;
                        p.LimitId = 0;
                        p.OffsetId = 0;

                        // Convert the data in the temporary table into whatever form it is that we currently need.
                        Debug.Assert(unionTab == dest2.SDParmId || dest2.Dest != priorOp);
                        if (dest2.Dest != priorOp)
                        {
                            Debug.Assert(p.EList != null);
                            if (dest2.Dest == SRT.Output)
                            {
                                Select first = p;
                                while (first.Prior != null) first = first.Prior;
                                GenerateColumnNames(parse, null, first.EList);
                            }
                            int breakId = v.MakeLabel();
                            int continueId = v.MakeLabel();
                            ComputeLimitRegisters(parse, p, breakId);
                            v.AddOp2(OP.Rewind, unionTab, breakId);
                            int startId = v.CurrentAddr();
                            SelectInnerLoop(parse, p, p.EList, unionTab, p.EList.Exprs, null, null, dest2, continueId, breakId);
                            v.ResolveLabel(continueId);
                            v.AddOp2(OP.Next, unionTab, startId);
                            v.ResolveLabel(breakId);
                            v.AddOp2(OP.Close, unionTab, 0);
                        }
                        break;
                    }
                default:
                    {
                        Debug.Assert(p.OP == TK.INTERSECT);

                        // INTERSECT is different from the others since it requires two temporary tables.  Hence it has its own case.  Begin
                        // by allocating the tables we will need.
                        int tab1 = parse.Tabs++;
                        int tab2 = parse.Tabs++;
                        Debug.Assert(p.OrderBy == null);

                        int addr = v.AddOp2(OP.OpenEphemeral, tab1, 0);
                        Debug.Assert(p.AddrOpenEphms[0] == -1);
                        p.AddrOpenEphms[0] = addr;
                        p.Rightmost.SelFlags |= SF.UsesEphemeral;
                        Debug.Assert(p.EList != null);

                        // Code the SELECTs to our left into temporary table "tab1".
                        SelectDest intersectdest = new SelectDest();
                        Select.DestInit(intersectdest, SRT_Union, tab1);
                        ExplainSetInteger(ref sub1Id, parse.NextSelectId);
                        rc = Select.Select_(parse, prior, ref intersectdest);
                        if (rc != 0)
                            goto multi_select_end;

                        // Code the current SELECT into temporary table "tab2"
                        addr = v.AddOp2(OP.OpenEphemeral, tab2, 0);
                        Debug.Assert(p.AddrOpenEphms[1] == -1);
                        p.AddrOpenEphms[1] = addr;
                        p.Prior = null;
                        Expr limit = p.Limit;
                        p.Limit = null;
                        Expr offset = p.Offset;
                        p.Offset = null;
                        intersectdest.SDParmId = tab2;
                        ExplainSetInteger(ref sub2Id, parse.NextSelectId);
                        rc = Select.Select_(parse, p, ref intersectdest);
                        C.ASSERTCOVERAGE(rc != RC.OK);
                        p.Prior = prior;
                        if (p.SelectRows > prior.SelectRows) p.SelectRows = prior.SelectRows;
                        Expr.Delete(ctx, ref p.Limit);
                        p.Limit = limit;
                        p.Offset = offset;

                        // Generate code to take the intersection of the two temporary tables.
                        Debug.Assert(p.EList != null);
                        if (dest2.Dest == SRT.Output)
                        {
                            Select first = p;
                            while (first.pPrior != null) first = first.Prior;
                            GenerateColumnNames(parse, null, first.EList);
                        }
                        int breakId = sqlite3VdbeMakeLabel(v);
                        int continueId = sqlite3VdbeMakeLabel(v);
                        ComputeLimitRegisters(parse, p, breakId);
                        v.AddOp2(OP.Rewind, tab1, breakId);
                        int r1 = Expr.GetTempReg(parse);
                        int startId = v.AddOp2(OP.RowKey, tab1, r1);
                        v.AddOp4Int(OP.NotFound, tab2, continueId, r1, 0);
                        Expr.ReleaseTempReg(parse, r1);
                        SelectInnerLoop(parse, p, p.EList, tab1, p.EList.Exprs, null, null, dest2, continueId, breakId);
                        v.ResolveLabel(continueId);
                        v.AddOp2(OP.Next, tab1, startId);
                        v.ResolveLabel(breakId);
                        v.AddOp2(OP.Close, tab2, 0);
                        v.AddOp2(OP.Close, tab1, 0);
                        break;
                    }
            }

            ExplainComposite(parse, p.op, sub1Id, sub2Id, p.op != TK_ALL);

            // Compute collating sequences used by temporary tables needed to implement the compound select.
            // Attach the KeyInfo structure to all temporary tables.
            //
            // This section is run by the right-most SELECT statement only.
            // SELECT statements to the left always skip this part.  The right-most
            // SELECT might also skip this part if it has no ORDER BY clause and no temp tables are required.
            if ((p.SelFlags & SF.UsesEphemeral) != 0)
            {
                Debug.Assert(p.Rightmost == p);
                int cols = p.EList.Exprs; // Number of columns in result set
                KeyInfo keyInfo = new KeyInfo(); keyInfo.Colls = new CollSeq[cols]; // Collating sequence for the result set
                if (keyInfo == null)
                {
                    rc = RC.NOMEM;
                    goto multi_select_end;
                }

                keyInfo.Encode = E.CTXENCODE(ctx);
                keyInfo.Fields = (ushort)cols;

                int i;
                CollSeq coll; // For looping through pKeyInfo.aColl[]
                for (i = 0; i < cols; i++)
                {
                    coll = MultiSelectCollSeq(parse, p, i);
                    if (coll == null)
                        coll = ctx.DefaultColl;
                    keyInfo.Colls[i] = coll;
                }
                //: keyInfo->SortOrders = (SO*)coll;

                for (Select loop = p; loop != null; loop = loop.Prior)
                {
                    for (i = 0; i < 2; i++)
                    {
                        int addr = loop.AddrOpenEphms[i];
                        if (addr < 0)
                        {
                            // If [0] is unused then [1] is also unused.  So we can always safely abort as soon as the first unused slot is found
                            Debug.Assert(loop.AddrOpenEphms[1] < 0);
                            break;
                        }
                        v.ChangeP2(addr, cols);
                        v.ChangeP4(addr, keyInfo, Vdbe.P4T.KEYINFO);
                        loop.AddrOpenEphms[i] = -1;
                    }
                }
                C._tagfree(ctx, ref keyInfo);
            }

        multi_select_end:
            dest.SdstId = dest2.SdstId;
            dest.Sdsts = dest2.Sdsts;
            Select.Delete(ctx, ref delete_);
            return rc;
        }
#endif

        static RC GenerateOutputSubroutine(Parse parse, Select p, SelectDest in_, SelectDest dest, int regReturn, int regPrev, KeyInfo keyInfo, int p4type, int breakId)
        {
            Vdbe v = parse.V;
            int addr = v.CurrentAddr();
            int continueId = v.MakeLabel();

            // Suppress duplicates for UNION, EXCEPT, and INTERSECT
            if (regPrev != 0)
            {
                int j1 = v.AddOp1(OP.IfNot, regPrev);
                int j2 = v.AddOp4(OP.Compare, in_.SdstId, regPrev + 1, in_.Sdsts, keyInfo, p4type);
                v.AddOp3(OP.Jump, j2 + 2, continueId, j2 + 2);
                v.JumpHere(j1);
                v->AddOp3(OP.Copy, in_.SdstId, regPrev + 1, in_.Sdsts - 1);
                v.AddOp2(OP.Integer, 1, regPrev);
            }
            if (parse.Ctx.MallocFailed) return 0;

            // Suppress the the first OFFSET entries if there is an OFFSET clause
            CodeOffset(v, p, continueId);

            switch (dest.Dest)
            {
                case SRT.Table:
                case SRT.EphemTab:
                    {
                        // Store the result as data using a unique key.
                        int r1 = Expr.GetTempReg(parse);
                        int r2 = Expr.GetTempReg(parse);
                        C.ASSERTCOVERAGE(dest.Dest == SRT.Table);
                        C.ASSERTCOVERAGE(dest.Dest == SRT.EphemTab);
                        v.AddOp3(OP.MakeRecord, in_.SdstId, in_.Sdsts, r1);
                        v.AddOp2(OP.NewRowid, dest.SDParmId, r2);
                        v.AddOp3(OP.Insert, dest.SDParmId, r1, r2);
                        v.ChangeP5(OPFLAG_APPEND);
                        Expr.ReleaseTempReg(parse, r2);
                        Expr.ReleaseTempReg(parse, r1);
                        break;
                    }

#if !OMIT_SUBQUERY
                case SRT.Set:
                    {
                        // If we are creating a set for an "expr IN (SELECT ...)" construct, then there should be a single item on the stack.  Write this
                        // item into the set table with bogus data.
                        Debug.Assert(in_.Sdsts == 1);
                        p.AffSdst = p.EList.Ids[0].Expr.CompareAffinity(dest.AffSdst);
                        int r1 = Expr.GetTempReg(parse);
                        v.AddOp4(OP.MakeRecord, in_.SdstId, 1, r1, p.AffSdst, 1);
                        Expr.CacheAffinityChange(parse, in_.SdstId, 1);
                        v.AddOp2(OP.IdxInsert, dest.SDParmId, r1);
                        Expr.ReleaseTempReg(parse, r1);
                        break;
                    }
#if false // Never occurs on an ORDER BY query
                // If any row exist in the result set, record that fact and abort.
                case SRT.Exists:
                    {
                        v.AddOp2(OP.Integer, 1, dest.SDParmId);
                        // The LIMIT clause will terminate the loop for us
                        break;
                    }
#endif
                case SRT.Mem:
                    {
                        // If this is a scalar select that is part of an expression, then store the results in the appropriate memory cell and break out
                        // of the scan loop.
                        Debug.Assert(in_.Sdsts == 1);
                        Expr.CodeMove(parse, in_.SdstId, dest.SDParmId, 1);
                        // The LIMIT clause will jump out of the loop for us
                        break;
                    }
#endif
                case SRT.Coroutine:
                    {
                        // The results are stored in a sequence of registers starting at pDest->iSdst.  Then the co-routine yields.
                        if (dest.SdstId == 0)
                        {
                            dest.SdstId = Expr.GetTempRange(parse, in_.Sdsts);
                            dest.Sdsts = in_.Sdsts;
                        }
                        Expr.CodeMove(parse, in_.SdstId, dest.SdstId, dest.Sdsts);
                        v.AddOp1(OP.Yield, dest.SDParmId);
                        break;
                    }
                default:
                    {
                        // If none of the above, then the result destination must be SRT_Output.  This routine is never called with any other
                        // destination other than the ones handled above or SRT_Output.
                        //
                        // For SRT_Output, results are stored in a sequence of registers.  Then the OP_ResultRow opcode is used to cause sqlite3_step() to
                        // return the next row of result.
                        Debug.Assert(dest.Dest == SRT.Output);
                        v.AddOp2(OP.ResultRow, in_.SdstId, in_.Sdsts);
                        Expr.CacheAffinityChange(parse, in_.SdstId, in_.Sdsts);
                        break;
                    }
            }

            // Jump to the end of the loop if the LIMIT is reached.
            if (p.LimitId != 0)
                v.AddOp3(OP.IfZero, p.LimitId, breakId, -1);

            // Generate the subroutine return
            v.ResolveLabel(continueId);
            v.AddOp1(OP.Return, regReturn);
            return addr;
        }

#if !OMIT_COMPOUND_SELECT
        static RC MultiSelectOrderBy(Parse parse, Select p, SelectDest dest)
        {
            int i, j;
#if !OMIT_EXPLAIN
            int sub1Id = 0; // EQP id of left-hand query
            int sub2Id = 0; // EQP id of right-hand query
#endif

            Debug.Assert(p.OrderBy != null);
            Context ctx = parse.Ctx; // Database connection
            Vdbe v = parse.V; // Generate code to this VDBE
            Debug.Assert(v != null); // Already thrown the error if VDBE alloc failed
            int labelEnd = v.MakeLabel(); // Label for the end of the overall SELECT stmt
            int labelCmpr = v.MakeLabel(); // Label for the start of the merge algorithm

            // Patch up the ORDER BY clause
            TK op = p.OP; // One of TK_ALL, TK_UNION, TK_EXCEPT, TK_INTERSECT
            Select prior = p.Prior; // Another SELECT immediately to our left
            Debug.Assert(prior.OrderBy == null);
            ExprList orderBy = p.OrderBy; // The ORDER BY clause
            Debug.Assert(orderBy != null);
            int orderBys = orderBy.Exprs; // Number of terms in the ORDER BY clause

            // For operators other than UNION ALL we have to make sure that the ORDER BY clause covers every term of the result set.  Add
            // terms to the ORDER BY clause as necessary.
            if (op != TK.ALL)
            {
                for (i = 1; !ctx->MallocFailed && i <= p.EList.Exprs; i++)
                {
                    ExprList.ExprListItem item;
                    for (j = 0; j < orderBys; j++)
                    {
                        item = orderBy.Ids[j];
                        Debug.Assert(item.OrderByCol > 0);
                        if (item.OrderByCol == i) break;
                    }
                    if (j == orderBys)
                    {
                        Expr newExpr = Expr.Expr_(ctx, TK.INTEGER, null);
                        if (newExpr == null) return SQLITE_NOMEM;
                        newExpr.Flags |= EP.IntValue;
                        newExpr.u.I = i;
                        orderBy = Expr.ListAppend(parse, orderBy, newExpr);
                        orderBy.Ids[orderBys++].OrderByCol = (ushort)i;
                    }
                }
            }

            // Compute the comparison permutation and keyinfo that is used with the permutation used to determine if the next
            // row of results comes from selectA or selectB.  Also add explicit collations to the ORDER BY clause terms so that when the subqueries
            // to the right and the left are evaluated, they use the correct collation.
            KeyInfo keyMerge; // Comparison information for merging rows
            int[] permutes = new int[orderBys]; // Mapping from ORDER BY terms to result set columns
            if (permutes != null)
            {
                ExprList.ExprListItem item;
                for (i = 0; i < orderBys; i++)
                {
                    item = orderBy.Ids[i];
                    Debug.Assert(item.OrderByCol > 0 && item.OrderByCol <= p.EList.Exprs);
                    permutes[i] = item.OrderByCol - 1;
                }
                keyMerge = new KeyInfo();
                if (keyMerge != null)
                {
                    keyMerge.Colls = new CollSeq[orderBys];
                    keyMerge.SortOrders = new byte[orderBys];
                    keyMerge.Fields = (ushort)orderBys;
                    keyMerge.Encode = E.CTXENCODE(ctx);
                    for (i = 0; i < orderBys; i++)
                    {
                        CollSeq coll;
                        Expr term = orderBy.Ids[i].Expr;
                        if ((term.Flags & EP.ExpCollate) != 0)
                            coll = term.CollSeq(parse);
                        else
                        {
                            coll = MultiSelectCollSeq(parse, p, permutes[i]);
                            if (coll == null) coll = ctx.DefaultColl;
                            orderBy.Ids[i].Expr = term.AddCollateString(parse, coll.Name);
                        }
                        keyMerge.Colls[i] = coll;
                        keyMerge.SortOrders[i] = orderBy.Ids[i].SortOrder;
                    }
                }
            }
            else
                keyMerge = null;

            // Reattach the ORDER BY clause to the query.
            p.OrderBy = orderBy;
            prior.OrderBy = Expr.ListDup(ctx, orderBy, 0);

            // Allocate a range of temporary registers and the KeyInfo needed for the logic that removes duplicate result rows when the
            // operator is UNION, EXCEPT, or INTERSECT (but not UNION ALL).
            KeyInfo keyDup = null; // Comparison information for duplicate removal
            //:Debug.Assert(keyDup == null); // "Managed" code needs this.  Ticket #3382.
            int regPrev; // A range of registers to hold previous output
            if (op == TK.ALL)
                regPrev = 0;
            else
            {
                int exprs = p.EList.Exprs;
                Debug.Assert(orderBys >= exprs || ctx.MallocFailed);
                regPrev = parse->Mems + 1;
                parse.Mems += exprs + 1;
                v.AddOp2(OP.Integer, 0, regPrev);
                keyDup = new KeyInfo();
                if (keyDup != null)
                {
                    keyDup.Colls = new CollSeq[exprs];
                    keyDup.SortOrders = new SO[exprs];
                    keyDup.Fields = (ushort)exprs;
                    keyDup.Encode = E.CTXENCODE(ctx);
                    for (i = 0; i < exprs; i++)
                    {
                        keyDup.Colls[i] = MultiSelectCollSeq(parse, p, i);
                        keyDup.SortOrders[i] = 0;
                    }
                }
            }

            // Separate the left and the right query from one another
            p.Prior = null;
            sqlite3ResolveOrderGroupBy(parse, p, p.OrderBy, "ORDER");
            if (prior.pPrior == null)
                sqlite3ResolveOrderGroupBy(parse, prior, prior.OrderBy, "ORDER");

            // Compute the limit registers
            int regLimitA; // Limit register for select-A
            int regLimitB; // Limit register for select-A
            ComputeLimitRegisters(parse, p, labelEnd);
            if (p.LimitId != 0 && op == TK.ALL)
            {
                regLimitA = ++parse.Mems;
                regLimitB = ++parse.Mems;
                v.AddOp2(OP.Copy, (p.OffsetId != 0 ? p.OffsetId + 1 : p.LimitId), regLimitA);
                v.AddOp2(OP.Copy, regLimitA, regLimitB);
            }
            else
                regLimitA = regLimitB = 0;
            Expr.Delete(ctx, ref p.Limit);
            p.Limit = null;
            Expr.Delete(ctx, ref p.Offset);
            p.Offset = null;

            int regAddrA = ++parse.Mems; // Address register for select-A coroutine
            int regEofA = ++parse.Mems; // Flag to indicate when select-A is complete
            int regAddrB = ++parse.Mems; // Address register for select-B coroutine
            int regEofB = ++parse.Mems; // Flag to indicate when select-B is complete
            int regOutA = ++parse.Mems; // Address register for the output-A subroutine
            int regOutB = ++parse.Mems; // Address register for the output-B subroutine
            SelectDest destA = new SelectDest(); // Destination for coroutine A
            SelectDest destB = new SelectDest(); // Destination for coroutine B
            DestInit(destA, SRT.Coroutine, regAddrA);
            DestInit(destB, SRT.Coroutine, regAddrB);

            // Jump past the various subroutines and coroutines to the main merge loop
            int j1 = v.AddOp0(OP.Goto); // Jump instructions that get retargetted
            int addrSelectA = v.CurrentAddr(); // Address of the select-A coroutine

            // Generate a coroutine to evaluate the SELECT statement to the left of the compound operator - the "A" select.
            v.NoopComment("Begin coroutine for left SELECT");
            prior.LimitId = regLimitA;
            ExplainSetInteger(ref sub1Id, parse.NextSelectId);
            Select.Select_(parse, prior, ref destA);
            v.AddOp2(OP.Integer, 1, regEofA);
            v.AddOp1(OP.Yield, regAddrA);
            v.NoopComment("End coroutine for left SELECT");

            // Generate a coroutine to evaluate the SELECT statement on the right - the "B" select
            int addrSelectB = v.CurrentAddr(); // Address of the select-B coroutine
            v.NoopComment("Begin coroutine for right SELECT");
            int savedLimit = p.LimitId; // Saved value of p.iLimit
            int savedOffset = p.OffsetId; // Saved value of p.iOffset
            p.LimitId = regLimitB;
            p.OffsetId = 0;
            ExplainSetInteger(ref sub2Id, parse.NextSelectId);
            Select.Select_(parse, p, ref destB);
            p.LimitId = savedLimit;
            p.OffsetId = savedOffset;
            v.AddOp2(OP.Integer, 1, regEofB);
            v.AddOp1(OP.Yield, regAddrB);
            v.NoopComment("End coroutine for right SELECT");

            // Generate a subroutine that outputs the current row of the A select as the next output row of the compound select.
            v.NoopComment("Output routine for A");
            int addrOutA = GenerateOutputSubroutine(parse, p, destA, dest, regOutA, regPrev, keyDup, Vdbe.P4T.KEYINFO_HANDOFF, labelEnd); // Address of the output-A subroutine
            int addrOutB = 0; // Address of the output-B subroutine

            // Generate a subroutine that outputs the current row of the B select as the next output row of the compound select.
            if (op == TK.ALL || op == TK.UNION)
            {
                v.NoopComment("Output routine for B");
                addrOutB = GenerateOutputSubroutine(parse, p, destB, dest, regOutB, regPrev, keyDup, Vdbe.P4T.KEYINFO_STATIC, labelEnd);
            }

            // Generate a subroutine to run when the results from select A are exhausted and only data in select B remains.
            int addrEofA; // Address of the select-A-exhausted subroutine
            v.NoopComment("eof-A subroutine");
            if (op == TK.EXCEPT || op == TK.INTERSECT)
                addrEofA = v.AddOp2(OP.Goto, 0, labelEnd);
            else
            {
                addrEofA = v.AddOp2(OP.If, regEofB, labelEnd);
                v.AddOp2(OP.Gosub, regOutB, addrOutB);
                v.AddOp1(OP.Yield, regAddrB);
                v.AddOp2(OP.Goto, 0, addrEofA);
                p.SelectRows += prior.SelectRows;
            }

            // Generate a subroutine to run when the results from select B are exhausted and only data in select A remains.
            int addrEofB; // Address of the select-B-exhausted subroutine
            if (op == TK_INTERSECT)
            {
                addrEofB = addrEofA;
                if (p.SelectRows > prior.SelectRows) p.SelectRows = prior.SelectRows;
            }
            else
            {
                v.NoopComment("eof-B subroutine");
                addrEofB = v.AddOp2(OP.If, regEofA, labelEnd);
                v.AddOp2(OP.Gosub, regOutA, addrOutA);
                v.AddOp1(OP.Yield, regAddrA);
                v.AddOp2(OP.Goto, 0, addrEofB);
            }

            // Generate code to handle the case of A<B
            int addrAltB; // Address of the A<B subroutine
            v.NoopComment("A-lt-B subroutine");
            addrAltB = v.AddOp2(OP.Gosub, regOutA, addrOutA);
            v.AddOp1(OP.Yield, regAddrA);
            v.AddOp2(OP.If, regEofA, addrEofA);
            v.AddOp2(OP.Goto, 0, labelCmpr);

            // Generate code to handle the case of A==B
            int addrAeqB; // Address of the A==B subroutine
            if (op == TK.ALL)
                addrAeqB = addrAltB;
            else if (op == TK.INTERSECT)
            {
                addrAeqB = addrAltB;
                addrAltB++;
            }
            else
            {
                v.NoopComment("A-eq-B subroutine");
                addrAeqB = v.AddOp1(OP.Yield, regAddrA);
                v.AddOp2(OP.If, regEofA, addrEofA);
                v.AddOp2(OP.Goto, 0, labelCmpr);
            }

            // Generate code to handle the case of A>B
            int addrAgtB; // Address of the A>B subroutine
            v.NoopComment("A-gt-B subroutine");
            addrAgtB = v.CurrentAddr();
            if (op == TK.ALL || op == TK.UNION)
                v.AddOp2(OP.Gosub, regOutB, addrOutB);
            v.AddOp1(OP.Yield, regAddrB);
            v.AddOp2(OP.If, regEofB, addrEofB);
            v.AddOp2(OP.Goto, 0, labelCmpr);

            // This code runs once to initialize everything.
            v.JumpHere(j1);
            v.AddOp2(OP.Integer, 0, regEofA);
            v.AddOp2(OP.Integer, 0, regEofB);
            v.AddOp2(OP.Gosub, regAddrA, addrSelectA);
            v.AddOp2(OP.Gosub, regAddrB, addrSelectB);
            v.AddOp2(OP.If, regEofA, addrEofA);
            v.AddOp2(OP.If, regEofB, addrEofB);

            // Implement the main merge loop
            v.ResolveLabel(labelCmpr);
            v.AddOp4(OP.Permutation, 0, 0, 0, permutes, Vdbe.P4T.INTARRAY);
            v.AddOp4(OP.Compare, destA.SdstId, destB.SdstId, orderBys, keyMerge, Vdbe.P4T.KEYINFO_HANDOFF);
            v.ChangeP5(OPFLAG_PERMUTE);
            v.AddOp3(OP.Jump, addrAltB, addrAeqB, addrAgtB);

            // Jump to the this point in order to terminate the query.
            v.ResolveLabel(labelEnd);

            // Set the number of output columns
            if (dest.Dest == SRT.Output)
            {
                Select first = prior;
                while (first.Prior != null) first = first.Prior;
                GenerateColumnNames(parse, null, first.EList);
            }

            // Reassembly the compound query so that it will be freed correctly by the calling function
            if (p.Prior != null)
                Select.Delete(ctx, ref p.Prior);
            p.Prior = prior;

            //TBD:  Insert subroutine calls to close cursors on incomplete subqueries
            ExplainComposite(parse, p.OP, sub1Id, sub2Id, false);
            return RC.OK;
        }
#endif

#if !OMIT_SUBQUERY || !OMIT_VIEW
        static Expr SubstExpr(Context ctx, Expr expr, int tableId, ExprList list)
        {
            if (expr == null) return null;
            if (expr.op == TK.COLUMN && expr.TableId == tableId)
            {
                if (expr.ColumnIdx < 0)
                    expr.OP = TK_NULL;
                else
                {
                    Debug.Assert(list != null && expr.ColumnIdx < list.Exprs);
                    Debug.Assert(expr.Left == null && expr.Right == null);
                    Expr newExpr = Expr.Dup(ctx, list.Ids[expr.ColumnIdx].Expr, 0);
                    Expr.Delete(ctx, ref expr);
                    expr = newExpr;
                }
            }
            else
            {
                expr.Left = SubstExpr(ctx, expr.pLeft, tableId, list);
                expr.Right = SubstExpr(ctx, expr.pRight, tableId, list);
                if (E.ExprHasProperty(expr, EP.xIsSelect))
                    SubstSelect(ctx, expr.x.Select, tableId, list);
                else
                    SubstExprList(ctx, expr.x.List, tableId, list);
            }
            return expr;
        }

        static void SubstExprList(Context ctx, ExprList p, int tableId, ExprList list)
        {
            if (p == null) return;
            for (int i = 0; i < p.nExpr; i++)
                p.Ids[i].Expr = SubstExpr(ctx, p.Ids[i].Expr, tableId, list);
        }

        static void SubstSelect(Context ctx, Select p, int tableId, ExprList list)
        {
            if (p == null) return;
            SubstExprList(ctx, p.EList, tableId, list);
            SubstExprList(ctx, p.GroupBy, tableId, list);
            SubstExprList(ctx, p.OrderBy, tableId, list);
            p.Having = SubstExpr(ctx, p.Having, tableId, list);
            p.Where = SubstExpr(ctx, p.Where, tableId, list);
            SubstSelect(ctx, p.Prior, tableId, list);
            SrcList src = p.Src;
            Debug.Assert(src != null); // Even for (SELECT 1) we have: pSrc!=0 but pSrc->nSrc==0
            int i;
            SrcList.SrcListItem item;
            if (C._ALWAYS(src))
                for (i = src.Srcs; i > 0; i--)
                {
                    item = src.Ids[src.Srcs - i];
                    SubstSelect(ctx, item.Select, tableId, list);
                }
        }

        static bool FlattenSubquery(Parse parse, Select p, int fromId, bool isAgg, bool subqueryIsAgg)
        {
            Context ctx = parse.Ctx;

            // Check to see if flattening is permitted.  Return 0 if not.
            Debug.Assert(p != null);
            Debug.Assert(p.Prior == null); // Unable to flatten compound queries
            if (E.CtxOptimizationDisabled(ctx, OPTFLAG.QueryFlattener)) return false;
            SrcList src = p.Src; // The FROM clause of the outer query
            Debug.Assert(src != null && fromId >= 0 && fromId < src.Srcs);
            SrcList.SrcListItem subitem = src.Ids[fromId]; // The subquery
            int parentId = subitem.Cursor; // VDBE cursor number of the pSub result set temp table
            Select sub = subitem.Select; // The inner query or "subquery"
            Debug.Assert(sub != null);
            if (isAgg && subqueryIsAgg) return false;                                                       // Restriction (1)
            if (subqueryIsAgg && src.Srcs > 1) return false;                                                // Restriction (2)
            SrcList subSrc = sub.Src; // The FROM clause of the subquery
            Debug.Assert(subSrc != null);
            // Prior to version 3.1.2, when LIMIT and OFFSET had to be simple constants, not arbitrary expresssions, we allowed some combining of LIMIT and OFFSET
            // because they could be computed at compile-time.  But when LIMIT and OFFSET became arbitrary expressions, we were forced to add restrictions (13) and (14).
            if (sub.Limit != null && p.Limit != null) return false;                                         // Restriction (13)
            if (sub.Offset != null) return false;                                                           // Restriction (14)
            if (p.Rightmost != null && sub.Limit != null) return false;                                     // Restriction (15)
            if (subSrc.Srcs == 0) return false;                                                             // Restriction (7)
            if ((sub.SelFlags & SF.Distinct) != 0) return false;                                            // Restriction (5)
            if (sub.Limit != null && (src.Srcs > 1 || isAgg)) return false;                                 // Restrictions (8)(9)
            if ((p.SelFlags & SF.Distinct) != 0 && subqueryIsAgg) return false;                             // Restriction (6)
            if (p.OrderBy != null && sub.OrderBy != null) return false;                                     // Restriction (11)
            if (isAgg && sub.OrderBy != null) return false;                                                 // Restriction (16)
            if (sub.Limit != null && p.Where != null) return false;                                         // Restriction (19)
            if (sub.Limit != null && (p.SelFlags & SF.Distinct) != 0) return false;                         // Restriction (21)

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
            if ((subitem.Hointype & JT.OUTER) != 0) return false;

            // Restriction 17: If the sub-query is a compound SELECT, then it must use only the UNION ALL operator. And none of the simple select queries
            // that make up the compound SELECT are allowed to be aggregate or distinct queries.
            Select sub1; // Pointer to the rightmost select in sub-query
            if (sub.Prior != null)
            {
                if (sub.OrderBy != null) return false;                                                     // Restriction 20
                if (isAgg || (p.SelFlags & SF.Distinct) != 0 || src.Srcs != 1) return false;
                for (sub1 = sub; sub1 != null; sub1 = sub1.Prior)
                {
                    C.ASSERTCOVERAGE((sub1.SelFlags & (SF.Distinct | SF.Aggregate)) == SF.Distinct);
                    C.ASSERTCOVERAGE((sub1.SelFlags & (SF.Distinct | SF.Aggregate)) == SF.Aggregate);
                    Debug.Assert(sub.Src != null);
                    if ((sub1.SelFlags & (SF.Distinct | SF.Aggregate)) != 0 ||
                    (sub1.Prior != null && sub1.op != TK_ALL) ||
                    sub1.Src.Srcs < 1 ||
                    sub.EList.Exprs != sub1.EList.Exprs) return false;
                    C.ASSERTCOVERAGE(sub1.Src.Srcs > 1);
                }
                if (p.OrderBy != null)                                                                      // Restriction 18.
                    for (int ii = 0; ii < p.OrderBy.Exprs; ii++)
                        if (p.OrderBys.Ids[ii].OrderByCol == 0) return false;
            }

            ////// If we reach this point, flattening is permitted. //////

            // Authorize the subquery
            string savedAuthContext = parse.AuthContext;
            parse.AuthContext = subitem.Name;
            ARC i = Auth.Check(parse, AUTH.SELECT, null, null, null);
            C.ASSERTCOVERAGE(i == ARC.DENY);
            parse.AuthContext = savedAuthContext;

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
            for (sub = sub.Prior; sub != null; sub = sub.Prior)
            {
                ExprList orderBy = p.OrderBy;
                Expr limit = p.Limit;
                Select prior = p.Prior;
                p.OrderBy = null;
                p.Src = null;
                p.Prior = null;
                p.Limit = null;
                Select newSelect = sqlite3SelectDup(ctx, p, 0);
                p.Limit = limit;
                p.OrderBy = orderBy;
                p.Src = src;
                p.OP = TK.ALL;
                p.Rightmost = null;
                if (newSelect == null)
                    newSelect = prior;
                else
                {
                    newSelect.Prior = prior;
                    newSelect.Rightmost = null;
                }
                p.Prior = newSelect;
                if (ctx.MallocFailed) return true;
            }

            // Begin flattening the iFrom-th entry of the FROM clause in the outer query.
            sub = sub1 = subitem.Select;

            // Delete the transient table structure associated with the subquery
            C._tagfree(ctx, ref subitem.Database);
            C._tagfree(ctx, ref subitem.Name);
            C._tagfree(ctx, ref subitem.Alias);
            subitem.Database = null;
            subitem.Name = null;
            subitem.Alias = null;
            subitem.Select = null;

            // Defer deleting the Table object associated with the subquery until code generation is
            // complete, since there may still exist Expr.pTab entries that refer to the subquery even after flattening.  Ticket #3346.
            //
            // pSubitem->pTab is always non-NULL by test restrictions and tests above.
            if (C._ALWAYS(subitem.Table != null))
            {
                Table tabToDel = subitem.Table;
                if (tabToDel.Refs == 1)
                {
                    Parse toplevel = E.Parse_Toplevel(parse);
                    tabToDel.NextZombie = toplevel.ZombieTab;
                    toplevel.ZombieTab = tabToDel;
                }
                else
                    tabToDel.Refs--;
                subitem.Table = null;
            }

            // The following loop runs once for each term in a compound-subquery flattening (as described above).  If we are doing a different kind
            // of flattening - a flattening other than a compound-subquery flattening - then this loop only runs once.
            //
            // This loop moves all of the FROM elements of the subquery into the the FROM clause of the outer query.  Before doing this, remember
            // the cursor number for the original outer query FROM element in iParent.  The iParent cursor will never be used.  Subsequent code
            // will scan expressions looking for iParent references and replace those references with expressions that resolve to the subquery FROM
            // elements we are now copying in.
            Select parent;
            for (parent = p; parent != null; parent = parent.Prior, sub = sub.Prior)
            {
                JT jointype = 0;
                subSrc = sub.pSrc; // FROM clause of subquery
                int subSrcs = subSrc.Srcs; // Number of terms in subquery FROM clause
                src = parent.Src; // FROM clause of the outer query
                if (src != null)
                {
                    Debug.Assert(parent == p); // First time through the loop
                    jointype = subitem.Jointype;
                }
                else
                {
                    Debug.Assert(parent != p); // 2nd and subsequent times through the loop
                    src = parent.Src = Parse.SrcListAppend(ctx, null, null, null);
                    if (src == null)
                    {
                        Debug.Assert(ctx.MallocFailed);
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
                    parent.Src = src = Parse.SrcListEnlarge(ctx, src, subSrcs - 1, fromId + 1);
                    if (ctx.MallocFailed)
                        break;
                }

                // Transfer the FROM clause terms from the subquery into the outer query.
                int i;
                for (i = 0; i < subSrcs; i++)
                {
                    Parse.IdListDelete(ctx, ref src.Ids[i + fromId].Using);
                    src.Ids[i + fromId] = subSrc.Ids[i];
                    subSrc.Ids[i] = new SrcList.SrcListItem();
                }
                subitem = src.Ids[fromId]; // Reset for C#
                src.Ids[fromId].Jointype = jointype;

                // Now begin substituting subquery result set expressions for references to the iParent in the outer query.
                // 
                // Example:
                //
                //   SELECT a+5, b*10 FROM (SELECT x*3 AS a, y+10 AS b FROM t1) WHERE a>b;
                //   \                     \_____________ subquery __________/          /
                //    \_____________________ outer query ______________________________/
                //
                // We look at every expression in the outer query and every place we see "a" we substitute "x*3" and every place we see "b" we substitute "y+10".
                ExprList list = parent.EList; // The result set of the outer query
                for (i = 0; i < list.Exprs; i++)
                {
                    if (list.Ids[i].Name == null)
                    {
                        string name = list.Ids[i].Span;
                        Parse.Dequote(ref name);
                        list.Ids[i].Name = name;
                    }
                }
                SubstExprList(ctx, parent.EList, parentId, sub.EList);
                if (isAgg)
                {
                    SubstExprList(ctx, parent.GroupBy, parentId, sub.EList);
                    parent.Having = SubstExpr(ctx, parent.Having, parentId, sub.EList);
                }
                if (sub.OrderBy != null)
                {
                    Debug.Assert(parent.OrderBy == null);
                    parent.OrderBy = sub.OrderBy;
                    sub.OrderBy = null;
                }
                else if (parent.OrderBy != null)
                    SubstExprList(ctx, parent.OrderBy, parentId, sub.EList);
                Expr where_ = (sub.pWhere != null ? sqlite3ExprDup(ctx, sub.pWhere, 0) : null); // The WHERE clause
                if (subqueryIsAgg)
                {
                    Debug.Assert(parent.Having == null);
                    parent.Having = parent.Where;
                    parent.Where = where_;
                    parent.Having = SubstExpr(ctx, parent.Having, parentId, sub.EList);
                    parent.Having = Expr.And(ctx, parent.Having, Expr.Dup(ctx, sub.Having, 0));
                    Debug.Assert(parent.GroupBy == null);
                    parent.GroupBy = Expr.ListDup(ctx, sub.GroupBy, 0);
                }
                else
                {
                    parent.Where = SubstExpr(ctx, parent.Where, parentId, sub.EList);
                    parent.Where = Expr.And(ctx, parent.Where, where_);
                }

                // The flattened query is distinct if either the inner or the outer query is distinct.
                parent.SelFlags |= (sub.SelFlags & SF.Distinct);

                // SELECT ... FROM (SELECT ... LIMIT a OFFSET b) LIMIT x OFFSET y;
                //
                // One is tempted to try to add a and b to combine the limits.  But this does not work if either limit is negative.
                if (sub.Limit != null)
                {
                    parent.Limit = sub.Limit;
                    sub.Limit = null;
                }
            }

            // Finially, delete what is left of the subquery and return success.
            Select.Delete(ctx, ref sub1);
            return true;
        }
#endif

        static WHERE MinMaxQuery(AggInfo aggInfo, out ExprList minMaxOut)
        {
            WHERE r = WHERE.ORDERBY_NORMAL;
            minMaxOut = null;
            if (aggInfo.Funcs.length == 1)
            {
                Expr expr = aggInfo.Funcs[0].Expr; // Aggregate function
                ExprList list = expr.x.List; // Arguments to agg function
                Debug.Assert(expr.OP == TK.AGG_FUNCTION);
                if (list != null && list.Exprs == 1 && list.Ids[0].Expr.OP == TK.AGG_COLUMN)
                {
                    string funcName = expr.u.Token;
                    if (string.Equals(funcName, "min", StringComparison.OrdinalIgnoreCase))
                    {
                        r = WHERE_ORDERBY_MIN;
                        minMaxOut = list;
                    }
                    else if (string.Equals(funcName, "max", StringComparison.OrdinalIgnoreCase))
                    {
                        r = WHERE_ORDERBY_MAX;
                        minMaxOut = list;
                    }
                }
            }
            Debug.Assert(minMaxOut == null || minMax.Exprs == 1);
            return r;
        }

        static Table IsSimpleCount(Select p, AggInfo aggInfo)
        {
            Debug.Assert(p.GroupBy == null);
            if (p.Where != null || p.EList.Exprs != 1 || p.Src.Srcs != 1 || p.Src.Ids[0].Select != null)
                return null;
            Table table = p.Src.Ids[0].Table;
            Expr expr = p.EList.Ids[0].Expr;
            Debug.Assert(table != null && table.Select == null && expr != null);
            if (E.IsVirtual(table)) return null;
            if (expr.OP != TK.AGG_FUNCTION) return null;
            if (C._NEVER(aggInfo.Funcs.length == 0)) return null;
            if ((aggInfo.Funcs[0].Func.Flags & FUNC.COUNT) == 0) return null;
            if ((expr.Flags & EP.Distinct) != 0) return null;
            return table;
        }

        public static RC IndexedByLookup(Parse parse, SrcList.SrcListItem from)
        {
            if (from.Table != null && from.IndexName != null && from.IndexName.Length != 0)
            {
                Table table = from.Table;
                string indexName = from.IndexName;
                Index index;
                for (index = table.Index; index != null && !string.Equals(index.Name, indexName, StringComparison.OrdinalIgnoreCase); index = index.Next) ;
                if (index == null)
                {
                    parse.ErrorMsg("no such index: %s", indexName);
                    parse.CheckSchema = 1;
                    return RC.ERROR;
                }
                from.Index = index;
            }
            return RC.OK;
        }

        static WRC SelectExpander(Walker walker, Select p)
        {
            Parse parse = walker.pParse;
            Context ctx = parse.Ctx;
            int j, k;

            SF selFlags = p.SelFlags;
            p.SelFlags |= SF.Expanded;
            if (ctx.MallocFailed)
                return WRC.Abort;
            if (C._NEVER(p.Src == null) || (p.SelFlags & SF.Expanded) != 0)
                return WRC.Prune;
            SrcList tabList = p.Src;
            ExprList list = p.EList;

            // Make sure cursor numbers have been assigned to all entries in the FROM clause of the SELECT statement.
            parse.SrcListAssignCursors(tabList);

            // Look up every table named in the FROM clause of the select.  If an entry of the FROM clause is a subquery instead of a table or view,
            // then create a transient table structure to describe the subquery.
            int i;
            SrcList.SrcListItem from;
            for (i = 0; i < tabList.Srcs; i++)
            {
                from = tabList.Ids[i];
                Table table;
                if (from.Table != null)
                {
                    // This statement has already been prepared.  There is no need to go further.
                    Debug.Assert(i == 0);
                    return WRC.Prune;
                }
                if (from.Name == null)
                {
#if !OMIT_SUBQUERY
                    Select sel = from.Select;
                    // A sub-query in the FROM clause of a SELECT
                    Debug.Assert(sel != null);
                    Debug.Assert(from.Table == null);
                    walker.WalkSelect(sel);
                    from.Table = table = new Table();
                    if (table == null) return WRC.Abort;
                    table.Refs = 1;
                    table.Name = C._mtagprintf(ctx, "sqlite_subquery_%p_", table);
                    while (sel.Prior != null) { sel = sel.pPrior; }
                    SelectColumnsFromExprList(parse, sel.EList, ref table.Cols.length, ref table.Cols.data);
                    table.PKey = -1;
                    table.RowEst = 1000000;
                    table.TabFlags |= TF.Ephemeral;
#endif
                }
                else
                {
                    // An ordinary table or view name in the FROM clause
                    Debug.Assert(from.Table == null);
                    from.Table = table = parse.LocateTableItem(false, from);
                    if (table == null) return WRC.Abort;
                    table.Refs++;
#if !OMIT_VIEW || !OMIT_VIRTUALTABLE
                    if (table.Select != null || E.IsVirtual(table))
                    {
                        // We reach here if the named table is a really a view 
                        if (parse.ViewGetColumnNames(table) != 0) return WRC.Abort;
                        Debug.Assert(from.Select == null);
                        from.Select = Expr.SelectDup(ctx, table.Select, 0);
                        walker.WalkSelect(from.Select);
                    }
#endif
                }

                // Locate the index named by the INDEXED BY clause, if any.
                if (Select.IndexedByLookup(parse, from) != 0)
                    return WRC.Abort;
            }

            // Process NATURAL keywords, and ON and USING clauses of joins.

            if (ctx.MallocFailed || p.ProcessJoin(parse))
                return WRC.dAbort;

            // For every "*" that occurs in the column list, insert the names of all columns in all tables.  And for every TABLE.* insert the names
            // of all columns in TABLE.  The parser inserted a special expression with the TK_ALL operator for each "*" that it found in the column list.
            // The following code just has to locate the TK_ALL expressions and expand each one to the list of all columns in all tables.
            //
            // The first loop just checks to see if there are any "*" operators that need expanding.
            Expr e, right, expr;
            for (k = 0; k < list.Exprs; k++)
            {
                e = list.Ids[k].Expr;
                if (e.OP == TK.ALL) break;
                Debug.Assert(e.OP != TK.DOT || e.Right != null);
                Debug.Assert(e.OP != TK.DOT || (e.Left != null && e.Left.op == TK.ID));
                if (e.OP == TK.DOT && e.Right.OP == TK.ALL) break;
            }
            if (k < list.Exprs)
            {
                // If we get here it means the result set contains one or more "*" operators that need to be expanded.  Loop through each expression
                // in the result set and expand them one by one.
                ExprList.ExprListItem[] a = list.Ids;
                ExprList newList = null;
                Context.FLAG flags = ctx.Flags;
                bool longNames = ((flags & Context.FLAG.FullColNames) != 0 && (flags & Context.FLAG.ShortColNames) == 0);

                // When processing FROM-clause subqueries, it is always the case that full_column_names=OFF and short_column_names=ON. The sqlite3ResultSetOfSelect() routine makes it so.
                Debug.Assert((p.SelFlags & SF.NestedFrom) == 0 || ((flags & Context.FLAG.FullColNames) == 0 && (flags & Context.FLAG.ShortColNames) != 0));

                for (k = 0; k < list.Exprs; k++)
                {
                    e = a[k].Expr;
                    right = e.Right;
                    Debug.Assert(e.OP != TK.DOT || right != null);
                    if (e.OP != TK.ALL && (e.OP != TK.DOT || right.OP != TK.ALL))
                    {
                        // This particular expression does not need to be expanded.
                        newList = Expr.ListAppend(parse, newList, a[k].Expr);
                        if (newList != null)
                        {
                            newList.Ids[newList.Exprs - 1].Name = a[k].Name;
                            newList.Ids[newList.Exprs - 1].Span = a[k].Span;
                            a[k].Name = null;
                            a[k].Span = null;
                        }
                        a[k].Expr = null;
                    }
                    else
                    {
                        // This expression is a "*" or a "TABLE.*" and needs to be expanded.
                        bool tableSeen = false; // Set to 1 when TABLE matches
                        string tokenName = null; // text of name of TABLE
                        if (e.OP == TK.DOT)
                        {
                            Debug.Assert(e.Left != null);
                            Debug.Assert(!E.ExprHasProperty(e.Left, EP.IntValue));
                            tokenName = e.Left.u.Token;
                        }
                        for (i = 0; i < tabList.Srcs; i++)
                        {
                            Table table = from.Table;
                            Select sub = from.Select;
                            string tableName = from.Alias;
                            string schemaName = null;
                            int db;
                            if (tableName == null)
                                tableName = table.Name;
                            if (ctx.MallocFailed) break;
                            if (sub == null || (sub.SelFlags & SF.NestedFrom) == 0)
                            {
                                sub = null;
                                if (tokenName != null && !string.Equals(tokenName, tableName, StringComparison.OrdinalIgnoreCase))
                                    continue;
                                db = Prepare.SchemaToIndex(ctx, table.Schema);
                                schemaName = (db >= 0 ? ctx.DBs[db].Name : "*");
                            }
                            for (j = 0; j < table.Cols.length; j++)
                            {
                                string name = table.Cols[j].Name;
                                string colname; // The computed column name
                                string toFree; // Malloced string that needs to be freed
                                Token sColname = new Token(); // Computed column name as a token

                                Debug.Assert(name != null);
                                if (tokenName != null && sub != null && !Walker.MatchSpanName(sub.EList.Ids[j].Span, 0, tokenName, 0))
                                    continue;

                                // If a column is marked as 'hidden' (currently only possible for virtual tables), do not include it in the expanded result-set list.
                                if (E.IsHiddenColumn(table.Cols[j]))
                                {
                                    Debug.Assert(E.IsVirtual(table));
                                    continue;
                                }
                                tableSeen = true;

                                if (i > 0 && tokenName == null)
                                {
                                    int dummy1 = 0;
                                    if ((from.Jointype & JT_NATURAL) != 0 && TableAndColumnIndex(tabList, i, name, ref dummy1, ref dummy1) != 0)
                                        continue; // In a NATURAL join, omit the join columns from the table to the right of the join
                                    if (Parse.IdListIndex(from.Using, name) >= 0)
                                        continue; // In a join with a USING clause, omit columns in the using clause from the table on the right.
                                }
                                right = Expr.Expr_(ctx, TK.ID, name);
                                colname = name;
                                toFree = null;
                                if (longNames || tabList.nSrc > 1)
                                {
                                    Expr left = Expr.Expr_(ctx, TK.ID, tableName);
                                    pExpr = Expr.PExpr(parse, TK.DOT, left, right, 0);
                                    if (schemaName)
                                    {
                                        left = Expr.Expr_(ctx, TK.ID, schemaName);
                                        expr = Expr.PExpr_(parse, TK.DOT, left, expr, 0);
                                    }
                                    if (longNames)
                                    {
                                        colname = _mtagprintf(ctx, "%s.%s", tableName, name);
                                        toFree = colname;
                                    }
                                }
                                else
                                    expr = right;
                                newList = Expr.ListAppend(parse, newList, expr);
                                sColname.data = colname;
                                sColname.length = colname.Length;
                                Expr.ListSetName(parse, newList, sColname, 0);
                                C._tagfree(ctx, ref toFree);
                            }
                        }
                        if (!tableSeen)
                        {
                            if (tokenName != null)
                                sqlite3ErrorMsg(parse, "no such table: %s", tokenName);
                            else
                                sqlite3ErrorMsg(parse, "no tables specified");
                        }
                    }
                }
                Expr.ListDelete(ctx, ref list);
                p.EList = newList;
            }
            //#if MAX_COLUMN
            if (p.EList != null && p.EList.Exprs > ctx.Limits[(int)LIMIT.COLUMN])
                parse.ErrorMsg("too many columns in result set");
            //#endif
            return WRC.Continue;
        }

        static WRC ExprWalkNoop(Walker notUsed1, ref Expr notUsed2)
        {
            return WRC.Continue;
        }

        public static void Expand(Parse parse)
        {
            Walker w = new Walker();
            w.SelectCallback = SelectExpander;
            w.ExprCallback = ExprWalkNoop;
            w.Parse = parse;
            w.WalkSelect(this);
        }

#if !OMIT_SUBQUERY
        static WRC SelectAddSubqueryTypeInfo(Walker walker, Select p)
        {
            Debug.Assert((p.SelFlags & SF.Resolved) != 0);
            if ((p.SelFlags & SF.HasTypeInfo) == 0)
            {
                p.SelFlags |= SF.HasTypeInfo;
                Parse parse = walker.pParse;
                SrcList tabList = p.pSrc;
                int i;
                SrcList.SrcListItem from;
                for (i = 0; i < tabList.Srcs; i++)
                {
                    from = tabList.Ids[i];
                    Table table = from.Table;
                    if (C._ALWAYS(table != null) && (table.TabFlags & TF.Ephemeral) != 0)
                    {
                        // A sub-query in the FROM clause of a SELECT
                        Select sel = from.Select;
                        Debug.Assert(sel != null);
                        while (sel.Prior != null) sel = sel.Prior;
                        SelectAddColumnTypeAndCollation(parse, table.Cols.length, table.Cols.data, sel);
                    }
                }
            }
            return WRC.Continue;
        }
#endif

        public static void AddTypeInfo(Parse parse)
        {
#if !OMIT_SUBQUERY
            Walker w = new Walker();
            w.SelectCallback = SelectAddSubqueryTypeInfo;
            w.ExprCallback = ExprWalkNoop;
            w.Parse = parse;
            w.WalkSelect(this);
#endif
        }

        public static void Prep(Parse parse, NameContext outerNC)
        {
            Context ctx = parse.Ctx;
            if (ctx.MallocFailed) return;
            if ((SelFlags & SF.HasTypeInfo) != 0) return;
            Expand(parse, p);
            if (parse.Errs != 0 || ctx.MallocFailed) return;
            sqlite3ResolveSelectNames(parse, p, outerNC);
            if (parse.Errs != 0 || ctx.MallocFailed != 0) return;
            AddTypeInfo(parse);
        }

        static void ResetAccumulator(Parse parse, AggInfo aggInfo)
        {
            if (aggInfo.Funcs.length + aggInfo.Columns.length == 0)
                return;
            Vdbe v = parse.V;
            int i;
            for (i = 0; i < aggInfo.Columns.length; i++)
                v.AddOp2(OP.Null, 0, aggInfo.Cols[i].Mem);
            AggInfo.AggInfoFunc func;
            for (i = 0; i < aggInfo.Funcs.length; i++)
            {
                func = aggInfo.Funcs[i];
                v.AddOp2(OP.Null, 0, func.Mem);
                if (func.Distinct >= 0)
                {
                    Expr e = func.Expr;
                    Debug.Assert(!E.ExprHasProperty(e, EP.xIsSelect));
                    if (e.x.List == null || e.x.List.Exprs != 1)
                    {
                        parse.ErrorMsg("DISTINCT aggregates must have exactly one argument");
                        func.Distinct = -1;
                    }
                    else
                    {
                        KeyInfo keyInfo = KeyInfoFromExprList(parse, e.x.List);
                        v.AddOp4(OP.OpenEphemeral, func.Distinct, 0, 0, keyInfo, Vdbe.P4T.KEYINFO_HANDOFF);
                    }
                }
            }
        }

        static void FinalizeAggFunctions(Parse parse, AggInfo aggInfo)
        {
            Vdbe v = parse.V;
            int i;
            AggInfo.AggInfoFunc f;
            for (i = 0; i < aggInfo.Funcs.length; i++)
            {
                f = aggInfo.Funcs[i];
                ExprList list = f.Expr.x.List;
                Debug.Assert(!E.ExprHasProperty(f.Expr, EP.xIsSelect));
                v.AddOp4(OP.AggFinal, f.Mem, (list != null ? list.Exprs : 0), 0, f.Func, Vdbe.P4T.FUNCDEF);
            }
        }

        static void UpdateAccumulator(Parse parse, AggInfo aggInfo)
        {
            Vdbe v = parse.V;
            int regHit = 0;

            aggInfo.DirectMode = 1;
            Expr.CacheClear(parse);
            int i;
            AggInfo.AggInfoFunc f;

            for (i = 0; i < aggInfo.Funcs.length; i++)
            {
                f = aggInfo.Funcs[i];
                ExprList list = f.Expr.x.List;
                Debug.Assert(!E.ExprHasProperty(f.Expr, EP.xIsSelect));
                int args;
                int regAgg;
                if (list != null)
                {
                    args = list.Exprs;
                    regAgg = Expr.GetTempRange(parse, args);
                    Expr.CodeExprList(parse, list, regAgg, true);
                }
                else
                {
                    args = 0;
                    regAgg = 0;
                }
                int addrNext = 0;
                if (f.Distinct >= 0)
                {
                    addrNext = v.MakeLabel();
                    Debug.Assert(args == 1);
                    CodeDistinct(parse, f.Distinct, addrNext, 1, regAgg);
                }
                if ((f.Func.Flags & FUNC.NEEDCOLL) != 0)
                {
                    CollSeq coll = null;
                    int j;
                    ExprList.ExprListItem item;
                    Debug.Assert(list != null); // pList!=0 if pF->pFunc has NEEDCOLL
                    for (j = 0; coll == null && j < args; j++)
                    {
                        item = list.Ids[j];
                        coll = item.Expr.CollSeq(parse);
                    }
                    if (coll == null)
                        coll = parse.Ctx.DefaultColl;
                    if (regHit == 0 && aggInfo.Accumulators) regHit = ++parse.Mems;
                    v.AddOp4(OP.CollSeq, 0, 0, 0, coll, Vdbe.P4T.COLLSEQ);
                }
                v.AddOp4(OP.AggStep, 0, regAgg, f.Mem, f.Func, Vdbe.P4T.FUNCDEF);
                v.ChangeP5((byte)args);
                Expr.CacheAffinityChange(parse, regAgg, args);
                Expr.ReleaseTempRange(parse, regAgg, args);
                if (addrNext != 0)
                {
                    v.ResolveLabel(addrNext);
                    Expr.CacheClear(parse);
                }
            }

            // Before populating the accumulator registers, clear the column cache. Otherwise, if any of the required column values are already present 
            // in registers, sqlite3ExprCode() may use OP_SCopy to copy the value to pC->iMem. But by the time the value is used, the original register
            // may have been used, invalidating the underlying buffer holding the text or blob value. See ticket [883034dcb5].
            //
            // Another solution would be to change the OP_SCopy used to copy cached values to an OP_Copy.
            int addrHitTest = 0;
            if (regHit)
                addrHitTest = v.AddOp1(OP.If, regHit);
            Expr.CacheClear(parse);
            AggInfo.AggInfoColumn c;
            for (i = 0; i < aggInfo.Accumulators; i++)
            {
                c = aggInfo.Columns[i];
                Expr.Code(parse, c.Expr, c.Mem);
            }
            aggInfo.DirectMode = 0;
            Expr.CacheClear(parse);
            if (addrHitTest != 0)
                v.JumpHere(addrHitTest);
        }

#if !OMIT_EXPLAIN
        static void ExplainSimpleCount(Parse parse, Table table, Index index)
        {
            if (parse.Explain == 2)
            {
                string eqp = C._mtagprintf(parse.Ctx, "SCAN TABLE %s %s%s(~%d rows)",
                    table.Name,
                    (index != null ? "USING COVERING INDEX " : ""),
                    (index != null ? index.Name : ""),
                    table.RowEst);
                parse.V.AddOp4(OP.Explain, parse.SelectId, 0, 0, eqp, Vdbe.P4T.DYNAMIC);
            }
        }
#else
        static void ExplainSimpleCount(Parse a, Table b, Index c) { }
#endif

        readonly static SelectDest _dummySD = null;
        readonly static bool _dummyB = false;

        public static RC Select_(Parse parse, Select p, ref SelectDest dest)
        {
#if !OMIT_EXPLAIN
            int restoreSelectId = parse.SelectId;
            parse.SelectId = parse.NextSelectId++;
#endif

            Context ctx = parse.Ctx; // The database connection 
            if (p == null || ctx.MallocFailed || parse.Errs != 0)
                return 1;
            if (Auth.Check(parse, AUTH.SELECT, null, null, null)) return 1;
            AggInfo sAggInfo = new AggInfo(); // Information used by aggregate queries

            if (IgnorableOrderby(dest))
            {
                Debug.Assert(dest.Dest == SRT.Exists || dest.Dest == SRT.Union || dest.Dest == SRT.Except || dest.Dest == SRT.Discard);
                // If ORDER BY makes no difference in the output then neither does DISTINCT so it can be removed too.
                Expr.ListDelete(ctx, ref p.OrderBy);
                p.OrderBy = null;
                p.SelFlags &= ~SF.Distinct;
            }
            p.Prep(parse, null);
            ExprList orderBy = p.OrderBy; // The ORDER BY clause.  May be NULL
            SrcList tabList = p.Src; // List of tables to select from
            ExprList list = p.EList; // List of columns to extract.

            if (parse.Errs != 0 || ctx.MallocFailed)
                goto select_end;
            bool isAgg = ((p.SelFlags & SF.Aggregate) != 0); // True for select lists like "count()"
            Debug.Assert(list != null);

            // Begin generating code.
            Vdbe v = parse.GetVdbe(); // The virtual machine under construction
            if (v == null)
                goto select_end;

#if !OMIT_SUBQUERY
            // If writing to memory or generating a set only a single column may be output.
            if (CheckForMultiColumnSelectError(parse, dest, list.Exprs))
                goto select_end;
#endif

            int i, j;
#if !OMIT_SUBQUERY || !OMIT_VIEW
            // Generate code for all sub-queries in the FROM clause
            for (i = 0; p.pPrior == null && i < tabList.nSrc; i++)
            {
                SrcList.SrcListItem item = tabList.Ids[i];
                SelectDest sDest = new SelectDest();
                Select sub = item.Select;
                if (sub == null) continue;

                // Sometimes the code for a subquery will be generated more than once, if the subquery is part of the WHERE clause in a LEFT JOIN,
                // for example.  In that case, do not regenerate the code to manifest a view or the co-routine to implement a view.  The first instance
                // is sufficient, though the subroutine to manifest the view does need to be invoked again.
                if (item.AddrFillSub != 0)
                {
                    if (!item.ViaCoroutine)
                        v.AddOp2(OP.Gosub, item.RegReturn, item.AddrFillSub);
                    continue;
                }

                // Increment Parse.nHeight by the height of the largest expression tree refered to by this, the parent select. The child select
                // may contain expression trees of at most (SQLITE_MAX_EXPR_DEPTH-Parse.nHeight) height. This is a bit
                // more conservative than necessary, but much easier than enforcing an exact limit.
                parse.Height += Expr.SelectExprHeight(p);

                /* Check to see if the subquery can be absorbed into the parent. */
                bool isAggSub = ((sub.SelFlags & SF.Aggregate) != 0);
                if (FlattenSubquery(parse, p, i, isAgg, isAggSub))
                {
                    if (isAggSub)
                    {
                        isAgg = true;
                        p.SelFlags |= SF.Aggregate;
                    }
                    i = -1;
                }
                else if (tabList.Srcs == 1 && (p.SelFlags & SF.Materialize) == 0 && E.CtxOptimizationEnabled(ctx, OPTFLAG.SubqCoroutine))
                {
                    // Implement a co-routine that will return a single row of the result set on each invocation.
                    item.RegReturn = ++parse.Mems;
                    int addrEof = ++parse.Mems;
                    // Before coding the OP_Goto to jump to the start of the main routine, ensure that the jump to the verify-schema routine has already
                    // been coded. Otherwise, the verify-schema would likely be coded as part of the co-routine. If the main routine then accessed the 
                    // database before invoking the co-routine for the first time (for example to initialize a LIMIT register from a sub-select), it would 
                    // be doing so without having verified the schema version and obtained the required db locks. See ticket d6b36be38.
                    parse.CodeVerifySchema(-1);
                    v.AddOp0(OP_Goto);
                    int addrTop = v.AddOp1(OP.OpenPseudo, item.Cursor);
                    v.ChangeP5(1);
                    v.Comment("coroutine for %s", item.Table.Name);
                    item.AddrFillSub = addrTop;
                    v.AddOp2(OP.Integer, 0, addrEof);
                    v.ChangeP5(1);
                    DestInit(sDest, SRT.Coroutine, item.RegReturn);
                    ExplainSetInteger(ref item.SelectId, (byte)parse.NextSelectId);
                    Select_(parse, sub, sDest);
                    item.Table.RowEst = (Pid)sub.SelectRows;
                    item.ViaCoroutine = 1;
                    v.ChangeP2(addrTop, sDest.SdstId);
                    v.ChangeP3(addrTop, sDest.Sdsts);
                    v.AddOp2(OP_Integer, 1, addrEof);
                    v.AddOp1(OP_Yield, item.RegReturn);
                    v.Comment("end %s", item.Table.Name);
                    v.JumpHere(addrTop - 1);
                    Expr.ClearTempRegCache(parse);
                }
                else
                {
                    // Generate a subroutine that will fill an ephemeral table with the content of this subquery.  pItem->addrFillSub will point
                    // to the address of the generated subroutine.  pItem->regReturn is a register allocated to hold the subroutine return address
                    Debug.Assert(item->AddrFillSub == 0);
                    item.RegReturn = ++parse.Mems;
                    int topAddr = v.AddOp2(OP.Integer, 0, item.RegReturn);
                    item.AddrFillSub = topAddr + 1;
                    v.NoopComment("materialize %s", item.Table.Name);
                    int onceAddr = 0;
                    if (item.IsCorrelated == 0)
                        onceAddr = Expr.CodeOnce(parse); // If the subquery is no correlated and if we are not inside of a trigger, then we only need to compute the value of the subquery once.
                    DestInit(sDest, SRT.EphemTab, item.Cursor);
                    ExplainSetInteger(ref item.SelectId, (byte)parse.NextSelectId);
                    Select_(parse, sub, ref sDest);
                    item.Table.RowEst = (uint)sub.SelectRows;
                    if (onceAddr != 0) v.JumpHere(onceAddr);
                    int retAddr = v.AddOp1(OP.Return, item.RegReturn);
                    v.Comment("end %s", item.Table.Name);
                    v.ChangeP1(topAddr, retAddr);
                    Expr.ClearTempRegCache(parse);
                }
                if (/*parse->Errs ||*/ ctx->MallocFailed)
                    goto select_end;
                parse.Height -= Expr.SelectExprHeight(p);
                tabList = p.Src;
                if (!IgnorableOrderby(dest))
                    orderBy = p.OrderBy;
            }
            list = p.pEList;
#endif
            Expr where_ = p.Where; // The WHERE clause.  May be NULL
            ExprList groupBy = p.GroupBy; // The GROUP BY clause.  May be NULL
            Expr having = p.Having; // The HAVING clause.  May be NULL
            DistinctCtx sDistinct = new DistinctCtx(); // Info on how to code the DISTINCT keyword
            sDistinct.IsTnct = ((p.SelFlags & SF.Distinct) != 0);

            RC rc = 1; // Value to return from this function
#if !OMIT_COMPOUND_SELECT
            // If there is are a sequence of queries, do the earlier ones first.
            if (p.Prior != null)
            {
                if (p.Rightmost == null)
                {
                    int cnt = 0;
                    Select right = null;
                    for (Select loop = p; loop != null; loop = loop.Prior, cnt++)
                    {
                        loop.Rightmost = p;
                        loop.Next = right;
                        right = loop;
                    }
                    int maxSelect = ctx.Limits[(int)LIMIT.COMPOUND_SELECT];
                    if (maxSelect != 0 && cnt > maxSelect)
                    {
                        parse.ErrorMsg("too many terms in compound SELECT");
                        goto select_end;
                    }
                }
                rc = MultiSelect(parse, p, dest);
                ExplainSetInteger(ref parse.SelectId, restoreSelectId);
                return rc;
            }
#endif

            // If there is both a GROUP BY and an ORDER BY clause and they are identical, then disable the ORDER BY clause since the GROUP BY
            // will cause elements to come out in the correct order.  This is an optimization - the correct answer should result regardless.
            // Use the SQLITE_GroupByOrder flag with SQLITE_TESTCTRL_OPTIMIZER to disable this optimization for testing purposes.
            if (Expr.ListCompare(p->GroupBy, orderBy) == 0 && E.CtxOptimizationEnabled(ctx, OPTFLAG.GroupByOrder))
                orderBy = null;

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
            if ((p.SelFlags & (SF.Distinct | SF.Aggregate)) == SF.Distinct && Expr.ListCompare(orderBy, p.EList) == 0)
            {
                p.SelFlags &= ~SF.Distinct;
                p.GroupBy = Expr.ListDup(ctx, p.EList, 0);
                groupBy = p.GroupBy;
                orderBy = null;
                // Notice that even thought SF_Distinct has been cleared from p->selFlags, the sDistinct.isTnct is still set.  Hence, isTnct represents the
                // original setting of the SF_Distinct flag, not the current setting
                Debug.Assert(sDistinct.IsTnct);
            }

            // If there is an ORDER BY clause, then this sorting index might end up being unused if the data can be 
            // extracted in pre-sorted order.  If that is the case, then the OP_OpenEphemeral instruction will be changed to an OP_Noop once
            // we figure out that the sorting index is not needed.  The addrSortIndex variable is used to facilitate that change.
            int addrSortIndex; // Address of an OP_OpenEphemeral instruction
            if (orderBy != null)
            {
                KeyInfo keyInfo = KeyInfoFromExprList(parse, orderBy);
                orderBy.ECursor = parse.Tabs++;
                p.AddrOpenEphms[2] = addrSortIndex = v.AddOp4(OP.OpenEphemeral, orderBy.ECursor, orderBy.Exprs + 2, 0, keyInfo, Vdbe.P4T.KEYINFO_HANDOFF);
            }
            else
                addrSortIndex = -1;

            // If the output is destined for a temporary table, open that table.
            if (dest.Dest == SRT.EphemTab)
                v.AddOp2(OP.OpenEphemeral, dest.SDParmId, list.Exprs);

            // Set the limiter.
            int endId = v.MakeLabel(); // Address of the end of the query
            p.SelectRows = (double)LARGEST_INT64;
            ComputeLimitRegisters(parse, p, endId);

            // Open a virtual index to use for the distinct set.
            if ((p.SelFlags & SF_Distinct) != 0)
            {
                sDistinct.TableTnct = parse.Tabs++;
                sDistinct.AddrTnct = v.AddOp4(OP.OpenEphemeral, sDistinct.TableTnct, 0, 0, KeyInfoFromExprList(parse, p.EList), Vdbe.P4T.KEYINFO_HANDOFF);
                v.ChangeP5(BTREE_UNORDERED);
                sDistinct.TnctType = WHERE_DISTINCT.UNORDERED;
            }
            else
                sDistinct.TnctType = WHERE_DISTINCT.NOOP;

            WhereInfo winfo; // Return from sqlite3WhereBegin()
            if (!isAgg && groupBy == null)
            {
                // No aggregate functions and no GROUP BY clause
                ExprList dist = (sDistinct.IsTnct ? p.EList : null);

                // Begin the database scan.
                winfo = Where.Begin(parse, tabList, where_, ref orderBy, dist, 0, 0);
                if (winfo == null) goto select_end;
                if (winfo.RowOuts < p.SelectRows) p.SelectRows = winfo.RowOuts;
                if (winfo.EDistinct != 0) sDistinct.TnctType = winfo.EDistinct;
                if (orderBy != null && winfo.OBSats == orderBy.Exprs) orderBy = null;

                // If sorting index that was created by a prior OP_OpenEphemeral instruction ended up not being needed, then change the OP_OpenEphemeral into an OP_Noop.
                if (addrSortIndex >= 0 && orderBy == null)
                {
                    v.ChangeToNoop(addrSortIndex);
                    p.AddrOpenEphms[2] = -1;
                }

                // Use the standard inner loop.
                SelectInnerLoop(parse, p, list, 0, 0, orderBy, sDistinct, dest, winfo.ContinueId, winfo.BreakId);

                // End the database scan loop.
                Where.End(winfo);
            }
            else
            {
                // This case when there exist aggregate functions or a GROUP BY clause or both
                // Remove any and all aliases between the result set and the GROUP BY clause.
                if (groupBy != null)
                {
                    int k;
                    ExprList.ExprListItem item; // For looping over expression in a list
                    for (k = p.EList.Exprs; k > 0; k--)
                    {
                        item = p.EList.Ids[p.EList.Exprs - k];
                        item.Alias = 0;
                    }
                    for (k = groupBy.Exprs; k > 0; k--)
                    {
                        item = groupBy.Ids[groupBy.Exprs - k];
                        item.Alias = 0;
                    }
                    if (p.SelectRows > (double)100) p.SelectRows = (double)100;
                }
                else
                    p.SelectRows = (double)1;

                // Create a label to jump to when we want to abort the query
                int addrEnd = v.MakeLabel(); // End of processing for this SELECT

                // Convert TK_COLUMN nodes into TK_AGG_COLUMN and make entries in sAggInfo for all TK_AGG_FUNCTION nodes in expressions of the SELECT statement.
                NameContext sNC = new NameContext(); // Name context for processing aggregate information
                sNC.Parse = parse;
                sNC.SrcList = tabList;
                sNC.AggInfo = sAggInfo;
                sAggInfo.SortingColumns = (groupBy != null ? groupBy.Exprs + 1 : 0);
                sAggInfo.GroupBy = groupBy;
                Expr.AnalyzeAggList(sNC, list);
                Expr.AnalyzeAggList(sNC, orderBy);
                if (having != null)
                    Expr.AnalyzeAggregates(sNC, ref having);
                sAggInfo.Accumulators = sAggInfo.Columns.length;
                for (i = 0; i < sAggInfo.Funcs.length; i++)
                {
                    Debug.Assert(!E.ExprHasProperty(sAggInfo.Funcs[i].Expr, EP.xIsSelect));
                    sNC.NCFlags |= NC.InAggFunc;
                    Expr.AnalyzeAggList(sNC, sAggInfo.Funcs[i].Expr.x.List);
                    sNC.NCFlags &= ~NC.InAggFunc;
                }
                if (ctx.MallocFailed) goto select_end;

                // Processing for aggregates with GROUP BY is very different and much more complex than aggregates without a GROUP BY.
                int amemId; // First Mem address for storing current GROUP BY
                int bmemId; // First Mem address for previous GROUP BY
                int useFlagId; // Mem address holding flag indicating that at least one row of the input to the aggregator has been processed
                int abortFlagId; // Mem address which causes query abort if positive
                bool groupBySort; // Rows come from source in GROUP BY order
                int sortPTab = 0; // Pseudotable used to decode sorting results
                int sortOut = 0; // Output register from the sorter
                if (groupBy != null)
                {
                    // If there is a GROUP BY clause we might need a sorting index to implement it.  Allocate that sorting index now.  If it turns out
                    // that we do not need it after all, the OP_SorterOpen instruction will be converted into a Noop.
                    sAggInfo.SortingIdx = parse.Tabs++;
                    KeyInfo keyInfo = KeyInfoFromExprList(parse, groupBy); // Keying information for the group by clause
                    int addrSortingIdx = v.AddOp4(OP.OpenEphemeral, sAggInfo.SortingIdx, sAggInfo.SortingColumns, 0, keyInfo, Vdbe.P4T.KEYINFO_HANDOFF); // The OP_OpenEphemeral for the sorting index

                    // Initialize memory locations used by GROUP BY aggregate processing
                    useFlagId = ++parse.Mems;
                    abortFlagId = ++parse.Mems;
                    int regOutputRow = ++parse.Mems; // Return address register for output subroutine
                    int addrOutputRow = v.MakeLabel(); // Start of subroutine that outputs a result row
                    int regReset = ++parse.Mems; // Return address register for reset subroutine
                    int addrReset = v.MakeLabel(); // Subroutine for resetting the accumulator
                    amemId = parse.Mems + 1;
                    parse.Mems += groupBy.Exprs;
                    bmemId = parse.Mems + 1;
                    parse.nMem += groupBy.Expr;
                    v.AddOp2(OP.Integer, 0, abortFlagId);
                    v.Comment("clear abort flag");
                    v.AddOp2(OP.Integer, 0, useFlagId);
                    v.Comment("indicate accumulator empty");
                    v.AddOp3(OP.Null, 0, amemId, amemId + groupBy.Exprs - 1);

                    // Begin a loop that will extract all source rows in GROUP BY order. This might involve two separate loops with an OP_Sort in between, or
                    // it might be a single loop that uses an index to extract information in the right order to begin with.
                    v.AddOp2(OP.Gosub, regReset, addrReset);
                    winfo = Where.Begin(parse, tabList, where_, ref groupBy, 0, 0);
                    if (winfo == null) goto select_end;
                    if (winfo.OBSats == groupBy.Exprs)
                        groupBySort = false; // The optimizer is able to deliver rows in group by order so we do not have to sort.  The OP_OpenEphemeral table will be cancelled later because we still need to use the keyInfo
                    else
                    {
                        // Rows are coming out in undetermined order.  We have to push each row into a sorting index, terminate the first loop,
                        // then loop over the sorting index in order to get the output in sorted order
                        ExplainTempTable(parse, (sDistinct.IsTnct && (p.SelFlags & SF_Distinct) == 0 ? "DISTINCT" : "GROUP BY"));

                        groupBySort = true;
                        int groupBys = groupBy.nExpr;
                        int cols = groupBys + 1;
                        j = groupBys + 1;
                        for (i = 0; i < sAggInfo.Columns.length; i++)
                        {
                            if (sAggInfo.Columns[i].SorterColumn >= j)
                            {
                                cols++;
                                j++;
                            }
                        }
                        int regBase = Expr.GetTempRange(parse, cols);
                        Expr.CacheClear(parse);
                        Expr.CodeExprList(parse, groupBy, regBase, false);
                        v.AddOp2(OP.Sequence, sAggInfo.SortingIdx, regBase + groupBys);
                        j = groupBys + 1;
                        for (i = 0; i < sAggInfo.Columns.length; i++)
                        {
                            AggInfo.AggInfoColumn col = sAggInfo.Columns[i];
                            if (col.SorterColumn >= j)
                            {
                                int r1 = j + regBase;
                                int r2 = Expr.CodeGetColumn(parse, col.Table, col.Column, col.TableID, r1, 0);
                                if (r1 != r2)
                                    v.AddOp2(OP.SCopy, r2, r1);
                                j++;
                            }
                        }
                        int regRecord = Expr.GetTempReg(parse);
                        v.AddOp3(OP.MakeRecord, regBase, cols, regRecord);
                        v.AddOp2(OP.IdxInsert, sAggInfo.sortingIdx, regRecord);
                        Expr.ReleaseTempReg(parse, regRecord);
                        Expr.ReleaseTempRange(parse, regBase, cols);
                        Where.End(winfo);
                        v.AddOp2(OP.Sort, sAggInfo.SortingIdx, addrEnd);
                        v.Comment("GROUP BY sort");
                        sAggInfo.UseSortingIdx = 1;
                        Expr.CacheClear(parse);
                    }

                    // Evaluate the current GROUP BY terms and store in b0, b1, b2... (b0 is memory location bmemId+0, b1 is bmemId+1, and so forth)
                    // Then compare the current GROUP BY terms against the GROUP BY terms from the previous row currently stored in a0, a1, a2...
                    int addrTopOfLoop = v.CurrentAddr(); // Top of the input loop
                    Expr.CacheClear(parse);
                    for (j = 0; j < groupBy.Exprs; j++)
                    {
                        if (groupBySort)
                        {
                            v.AddOp3(OP.Column, sortPTab, j, bmemId + j);
                            if (j == 0) v.ChangeP5(OPFLAG_CLEARCACHE);
                        }
                        else
                        {
                            sAggInfo.DirectMode = 1;
                            Expr.Code(parse, groupBy.Ids[j].Expr, bmemId + j);
                        }
                    }
                    v.AddOp4(OP.Compare, amemId, bmemId, groupBy.Exprs, keyInfo, Vdbe.P4T.KEYINFO);
                    int j1 = v.CurrentAddr(); // A-vs-B comparision jump
                    v.AddOp3(OP.Jump, j1 + 1, 0, j1 + 1);

                    // Generate code that runs whenever the GROUP BY changes. Changes in the GROUP BY are detected by the previous code
                    // block.  If there were no changes, this block is skipped.
                    //
                    // This code copies current group by terms in b0,b1,b2,... over to a0,a1,a2.  It then calls the output subroutine
                    // and resets the aggregate accumulator registers in preparation for the next GROUP BY batch.
                    Expr.CodeMove(parse, bmemId, amemId, groupBy.nExpr);
                    v.AddOp2(OP.Gosub, regOutputRow, addrOutputRow);
                    v.Comment("output one row");
                    v.AddOp2(OP.IfPos, abortFlagId, addrEnd);
                    v.Comment("check abort flag");
                    v.AddOp2(OP.Gosub, regReset, addrReset);
                    v.Comment("reset accumulator");

                    // Update the aggregate accumulators based on the content of the current row
                    v.JumpHere(j1);
                    UpdateAccumulator(parse, sAggInfo);
                    v.AddOp2(OP.Integer, 1, useFlagId);
                    v.Comment("indicate data in accumulator");

                    // End of the loop
                    if (groupBySort)
                        v.AddOp2(OP.Next, sAggInfo.SortingIdx, addrTopOfLoop);
                    else
                    {
                        Where.End(winfo);
                        v.ChangeToNoop(addrSortingIdx, 1);
                    }

                    // Output the final row of result
                    v.AddOp2(OP.Gosub, regOutputRow, addrOutputRow);
                    v.Comment("output final row");

                    // Jump over the subroutines
                    v.AddOp2(OP.Goto, 0, addrEnd);

                    // Generate a subroutine that outputs a single row of the result set.  This subroutine first looks at the iUseFlag.  If iUseFlag
                    // is less than or equal to zero, the subroutine is a no-op.  If the processing calls for the query to abort, this subroutine
                    // increments the iAbortFlag memory location before returning in order to signal the caller to abort.
                    int addrSetAbort = v->CurrentAddr(); // Set the abort flag and return
                    v.AddOp2(OP.Integer, 1, abortFlagId);
                    v.Comment("set abort flag");
                    v.AddOp1(OP.Return, regOutputRow);
                    v.ResolveLabel(addrOutputRow);
                    addrOutputRow = v.CurrentAddr();
                    v.AddOp2(OP_IfPos, useFlagId, addrOutputRow + 2);
                    v.Comment("Groupby result generator entry point");
                    v.AddOp1(OP.Return, regOutputRow);
                    FinalizeAggFunctions(parse, sAggInfo);
                    if (having != null) having.IfFalse(parse, addrOutputRow + 1, AFF.BIT_JUMPIFNULL);
                    SelectInnerLoop(parse, p, p.pEList, 0, 0, orderBy, distinct, dest, addrOutputRow + 1, addrSetAbort);
                    v.AddOp1(OP.Return, regOutputRow);
                    v.Comment("end groupby result generator");

                    // Generate a subroutine that will reset the group-by accumulator
                    v.ResolveLabel(addrReset);
                    ResetAccumulator(parse, sAggInfo);
                    v.AddOp1(OP.Return, regReset);
                }
                // endif pGroupBy.  Begin aggregate queries without GROUP BY: 
                else
                {
                    ExprList del = null;
#if !OMIT_BTREECOUNT
                    Table table;
                    if ((table = IsSimpleCount(p, sAggInfo)) != null)
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
                        int db = Prepare.SchemaToIndex(ctx, table.Schema);
                        int csr = parse.Tabs++; // Cursor to scan b-tree

                        parse.CodeVerifySchema(db);
                        parse.TableLock(db, table.Id, false, table.Name);

                        // Search for the index that has the least amount of columns. If there is such an index, and it has less columns than the table
                        // does, then we can assume that it consumes less space on disk and will therefore be cheaper to scan to determine the query result.
                        // In this case set iRoot to the root page number of the index b-tree and pKeyInfo to the KeyInfo structure required to navigate the index.
                        //
                        // (2011-04-15) Do not do a full scan of an unordered index.
                        //
                        // In practice the KeyInfo structure will not be used. It is only passed to keep OP_OpenRead happy.
                        int rootId = table.Id; // Root page of scanned b-tree
                        Index bestIndex = null; // Best index found so far
                        for (Index index = table.Index; index != null; index = index.Next)
                            if (index.Unordered == 0 && (bestIndex == null || index.Columns.length < bestIndex.Columns.length))
                                bestIndex = index;
                        KeyInfo keyInfo = null; // Keyinfo for scanned index
                        if (bestIndex != null && bestIndex.Columns.length < table.Cols.length)
                        {
                            rootId = bestIndex.Id;
                            keyInfo = parse.IndexKeyinfo(bestIndex);
                        }

                        // Open a read-only cursor, execute the OP_Count, close the cursor.
                        v.AddOp3(OP.OpenRead, csr, rootId, db);
                        if (keyInfo != null)
                            v.ChangeP4(-1, keyInfo, Vdbe.P4T.KEYINFO_HANDOFF);
                        v.AddOp2(OP.Count, csr, sAggInfo.Funcs[0].Mem);
                        v.AddOp1(OP.Close, csr);
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
                        ExprList minMax = null;
                        WHERE flag = WHERE.ORDERBY_NORMAL;
                        Debug.Assert(p.GroupBy == null);
                        Debug.Assert(flag == 0);
                        if (p.Having == null)
                            flag = MinMaxQuery(sAggInfo, out minMax);
                        Debug.Assert(flag == 0 || (minMax != null && minMax.Exprs == 1));

                        if (flag != 0)
                        {
                            minMax = Expr.ListDup(ctx, minMax, 0);
                            del = minMax;
                            if (minMax != null && !ctx.MallocFailed)
                            {
                                minMax.Ids[0].SortOrder = (flag != WHERE.ORDERBY_MIN ? SO.DESC : SO.ASC);
                                minMax.Ids[0].Expr.OP = TK.COLUMN;
                            }
                        }

                        // This case runs if the aggregate has no GROUP BY clause. The processing is much simpler since there is only a single row of output.
                        ResetAccumulator(parse, sAggInfo);
                        winfo = Where.Begin(parse, tabList, where_, ref minMax, 0, flag, 0);
                        if (winfo == null)
                        {
                            Expr.ListDelete(ctx, ref del);
                            goto select_end;
                        }
                        UpdateAccumulator(parse, sAggInfo);
                        Debug.Assert(minMax == null || minMax.Exprs == 1);
                        if (winfo->OBSats > 0)
                        {
                            v.AddOp2(OP.Goto, 0, winfo.BreakId);
                            v.Comment("%s() by index", (flag == WHERE.ORDERBY_MIN ? "min" : "max"));
                        }
                        Where.End(winfo);
                        FinalizeAggFunctions(parse, sAggInfo);
                    }

                    orderBy = null;
                    if (having != null) having.IfFalse(parse, addrEnd, AFF.BIT_JUMPIFNULL);
                    SelectInnerLoop(parse, p, p.EList, 0, 0, null, null, dest, addrEnd, addrEnd);
                    Expr.ListDelete(ctx, ref del);
                }
                v.ResolveLabel(addrEnd);
            }

            if (sDistinct.TnctType == WHERE_DISTINCT.UNORDERED)
                ExplainTempTable(parse, "DISTINCT");

            // If there is an ORDER BY clause, then we need to sort the results and send them to the callback one by one.
            if (orderBy != null)
            {
                ExplainTempTable(parse, "ORDER BY");
                GenerateSortTail(parse, p, v, list.Exprs, dest);
            }

            // Jump here to skip this query
            v.ResolveLabel(endId);

            // The SELECT was successfully coded. Set the return code to 0 to indicate no errors.
            rc = 0;

        // Control jumps to here if an error is encountered above, or upon successful coding of the SELECT.
        select_end:
            ExplainSetInteger(ref parse.SelectId, restoreSelectId);

            // Identify column names if results of the SELECT are to be output.
            if (rc == RC.OK && dest.Dest == SRT.Output)
                GenerateColumnNames(parse, tabList, list);

            C._tagfree(ctx, ref sAggInfo.Columns.data);
            C._tagfree(ctx, ref sAggInfo.Funcs.data);
            return rc;
        }

#if ENABLE_TREE_EXPLAIN
        static void ExplainOneSelect(Vdbe v, Select p)
        {
            Vdbe.ExplainPrintf(v, "SELECT ");
            if (p.SelFlags & (SF.Distinct | SF.Aggregate))
            {
                if (p.SelFlags & SF.Distinct)
                    Vdbe.ExplainPrintf(v, "DISTINCT ");
                if (p.SelFlags & SF.Aggregate)
                    Vdbe.ExplainPrintf(v, "agg_flag ");
                Vdbe.ExplainNL(v);
                Vdbe.ExplainPrintf(v, "   ");
            }
            Expr.ExplainExprList(v, p.EList);
            Vdbe.ExplainNL(v);
            if (p.Src != null && p.Src.Srcs)
            {
                int i;
                Vdbe.ExplainPrintf(v, "FROM ");
                Vdbe.ExplainPush(v);
                for (i = 0; i < p.Src.Srcs; i++)
                {
                    SrcList.SrcListItem item = p.Src.Ids[i];
                    Vdbe.ExplainPrintf(v, "{%d,*} = ", item.Cursor);
                    if (item.Select)
                    {
                        Select.ExplainSelect(v, item.Select);
                        if (item.Table)
                            Vdbe.ExplainPrintf(v, " (tabname=%s)", item.Table.Name);
                    }
                    else if (item.Name)
                        Vdbe.ExplainPrintf(v, "%s", item.Name);
                    if (item.Alias)
                        Vdbe.ExplainPrintf(v, " (AS %s)", item.Alias);
                    if (item.Jointype & JT_LEFT)
                        Vdbe.ExplainPrintf(v, " LEFT-JOIN");
                    Vdbe.ExplainNL(v);
                }
                Vdbe.ExplainPop(v);
            }
            if (p.Where != null)
            {
                Vdbe.ExplainPrintf(v, "WHERE ");
                Expr.ExplainExpr(v, p.Where);
                Vdbe.ExplainNL(v);
            }
            if (p.GroupBy != null)
            {
                Vdbe.ExplainPrintf(v, "GROUPBY ");
                Expr.ExplainExprList(v, p.GroupBy);
                Vdbe.ExplainNL(v);
            }
            if (p.Having != null)
            {
                Vdbe.ExplainPrintf(v, "HAVING ");
                Expr.ExplainExpr(v, p.Having);
                Vdbe.ExplainNL(v);
            }
            if (p.OrderBy != null)
            {
                Vdbe.ExplainPrintf(v, "ORDERBY ");
                Expr.ExplainExprList(v, p.OrderBy);
                Vdbe.ExplainNL(v);
            }
            if (p.Limit != null)
            {
                Vdbe.ExplainPrintf(v, "LIMIT ");
                Expr.ExplainExpr(v, p.Limit);
                Vdbe.ExplainNL(v);
            }
            if (p.Offset != null)
            {
                Vdbe.ExplainPrintf(v, "OFFSET ");
                Expr.ExplainExpr(v, p.Offset);
                Vdbe.ExplainNL(v);
            }
        }

        public static void ExplainSelect(Vdbe v, Select p)
        {
            if (p == null)
            {
                Vdbe.ExplainPrintf(v, "(null-select)");
                return;
            }
            while (p.Prior != null) { p.Prior.Next = p; p = p.Prior; }
            Vdbe.ExplainPush(v);
            while (p != null)
            {
                ExplainOneSelect(v, p);
                p = p.Next;
                if (p == null) break;
                Vdbe.ExplainNL(v);
                Vdbe.ExplainPrintf(v, "%s\n", SelectOpName(p.OP));
            }
            Vdbe.ExplainPrintf(v, "END");
            Vdbe.ExplainPop(v);
        }
#endif
    }
}
