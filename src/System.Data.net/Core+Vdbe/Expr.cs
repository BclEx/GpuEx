using System;
using System.Diagnostics;
using System.Text;
using Pid = System.UInt32;
#if !MAX_VARIABLE_NUMBER
using yVars = System.Int16;
#else
using yVars = System.Int32; 
#endif

namespace Core
{
    public partial class Expr
    {
        public AFF Affinity()
        {
            Expr expr = SkipCollate();
            TK op = OP;
            if (op == TK.SELECT)
            {
                Debug.Assert((expr.Flags & EP.xIsSelect) != 0);
                return expr.x.Select.EList.Ids[0].Expr.Affinity();
            }
#if !OMIT_CAST
            if (op == TK.CAST)
            {
                Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                return Parse.AffinityType(expr.u.Token);
            }
#endif
            if ((op == TK.AGG_COLUMN || op == TK.COLUMN || op == TK.REGISTER) && expr.Table != null)
            {
                // op == TK_REGISTER && expr->Table happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
                int j = expr.ColumnIdx;
                if (j < 0)
                    return AFF.INTEGER;
                Debug.Assert(expr.Table != null && j < expr.Table.Cols.length);
                return expr.Table.Cols[j].Affinity;
            }
            return expr.Aff;
        }

        public Expr AddCollateToken(Parse parse, Token collName)
        {
            Expr expr = this;
            if (collName.length > 0)
            {
                Expr newExpr = Alloc(parse.Ctx, TK.COLLATE, collName, true);
                if (newExpr != null)
                {
                    newExpr.Left = expr;
                    newExpr.Flags |= EP.Collate;
                    expr = newExpr;
                }
            }
            return expr;
        }

        public Expr AddCollateString(Parse parse, string z)
        {
            Debug.Assert(z != null);
            Token s;
            s.data = z;
            s.length = (uint)z.Length;
            return AddCollateToken(parse, s);
        }

        public Expr SkipCollate()
        {
            Expr expr = this;
            while (expr != null && (expr.OP == TK.COLLATE || expr.OP == TK.AS))
                expr = expr.Left;
            return expr;
        }

        public CollSeq CollSeq(Parse parse)
        {
            Context ctx = parse.Ctx;
            CollSeq coll = null;
            Expr p = this;
            while (p != null)
            {
                TK op = p.OP;
                if (op == TK.CAST || op == TK.UPLUS)
                {
                    p = p.Left;
                    continue;
                }
                Debug.Assert(op != TK.REGISTER || p.OP2 != TK.COLLATE);
                if (op == TK.COLLATE)
                {
                    if (ctx.Init.Busy) // Do not report errors when parsing while the schema 
                        coll = Callback.FindCollSeq(ctx, E.CTXENCODE(ctx), p.u.Token, false);
                    else
                        coll = Callback.GetCollSeq(parse, E.CTXENCODE(ctx), null, p.u.Token);
                    break;
                }
                if (p.Table != null && (op == TK.AGG_COLUMN || op == TK.COLUMN || op == TK.REGISTER || op == TK.TRIGGER))
                {
                    // op==TK_REGISTER && p->pTab!=0 happens when pExpr was originally a TK_COLUMN but was previously evaluated and cached in a register
                    int j = p.ColumnIdx;
                    if (j >= 0)
                    {
                        string nameColl = p.Table.Cols[j].Coll;
                        coll = Callback.FindCollSeq(ctx, E.CTXENCODE(ctx), nameColl, false);
                    }
                    break;
                }
                if ((p.Flags & EP.Collate) != 0)
                    p = (C._ALWAYS(p.Left != null) && (p.Left.Flags & EP.Collate) != 0 ? p.Left : p.Right);
                else
                    break;
            }
            if (Callback.CheckCollSeq(parse, coll) != RC.OK)
                coll = null;
            return coll;
        }

        public AFF CompareAffinity(AFF aff2)
        {
            AFF aff1 = Affinity();
            if (aff1 != 0 && aff2 != 0) // Both sides of the comparison are columns. If one has numeric affinity, use that. Otherwise use no affinity.
                return (E.IsNumericAffinity(aff1) || E.IsNumericAffinity(aff2) ? AFF.NUMERIC : AFF.NONE);
            else if (aff1 == 0 && aff2 == 0) // Neither side of the comparison is a column.  Compare the results directly.
                return AFF.NONE;
            else // One side is a column, the other is not. Use the columns affinity.
            {
                Debug.Assert(aff1 == 0 || aff2 == 0);
                return (aff1 != 0 ? aff1 : aff2);
            }
        }

        static AFF ComparisonAffinity(Expr expr)
        {
            Debug.Assert(expr.OP == TK.EQ || expr.OP == TK.IN || expr.OP == TK.LT ||
                expr.OP == TK.GT || expr.OP == TK.GE || expr.OP == TK.LE ||
                expr.OP == TK.NE || expr.OP == TK.IS || expr.OP == TK.ISNOT);
            Debug.Assert(expr.Left != null);
            AFF aff = expr.Left.Affinity();
            if (expr.Right != null)
                aff = expr.Right.CompareAffinity(aff);
            else if (E.ExprHasProperty(expr, EP.xIsSelect))
                aff = expr.x.Select.EList.Ids[0].Expr.CompareAffinity(aff);
            else if (aff == 0)
                aff = AFF.NONE;
            return aff;
        }

        public bool ValidIndexAffinity(AFF indexAff)
        {
            AFF aff = ComparisonAffinity(this);
            switch (aff)
            {
                case AFF.NONE:
                    return true;
                case AFF.TEXT:
                    return (indexAff == AFF.TEXT);
                default:
                    return E.IsNumericAffinity(indexAff);
            }
        }

        static AFF BinaryCompareP5(Expr expr1, Expr expr2, AFF jumpIfNull)
        {
            AFF aff = expr2.Affinity();
            aff = (AFF)(expr1.CompareAffinity(aff) | jumpIfNull);
            return aff;
        }

        public static CollSeq BinaryCompareCollSeq(Parse parse, Expr left, Expr right)
        {
            CollSeq coll;
            Debug.Assert(left != null);
            if ((left.Flags & EP.Collate) != 0)
                coll = left.CollSeq(parse);
            else if (right != null && (right.Flags & EP.Collate) != 0)
                coll = right.CollSeq(parse);
            else
            {
                coll = left.CollSeq(parse);
                if (coll == null)
                    coll = right.CollSeq(parse);
            }
            return coll;
        }

        static int CodeCompare(Parse parse, Expr left, Expr right, int opcode, int in1, int in2, int dest, AFF jumpIfNull)
        {
            CollSeq p4 = Expr.BinaryCompareCollSeq(parse, left, right);
            AFF p5 = BinaryCompareP5(left, right, jumpIfNull);
            Vdbe v = parse.V;
            int addr = v.AddOp4(opcode, in2, dest, in1, p4, Vdbe.P4T.COLLSEQ);
            v.ChangeP5((byte)p5);
            return addr;
        }

#if MAX_EXPR_DEPTH

        public static RC CheckHeight(Parse parse, int height)
        {
            RC rc = RC.OK;
            int maxHeight = parse.Ctx.Limits[(int)LIMIT.EXPR_DEPTH];
            if (height > maxHeight)
            {
                parse.ErrorMsg("Expression tree is too large (maximum depth %d)", maxHeight);
                rc = RC.ERROR;
            }
            return rc;
        }

        static void HeightOfExpr(Expr p, ref int height)
        {
            if (p != null)
                if (p.Height > height)
                    height = p.Height;
        }

        static void HeightOfExprList(ExprList p, ref int height)
        {
            if (p != null)
                for (int i = 0; i < p.Exprs; i++)
                    HeightOfExpr(p.Ids[i].Expr, ref height);
        }

        static void HeightOfSelect(Select p, ref int height)
        {
            if (p != null)
            {
                HeightOfExpr(p.Where, ref height);
                HeightOfExpr(p.Having, ref height);
                HeightOfExpr(p.Limit, ref height);
                HeightOfExpr(p.Offset, ref height);
                HeightOfExprList(p.EList, ref height);
                HeightOfExprList(p.GroupBy, ref height);
                HeightOfExprList(p.OrderBy, ref height);
                HeightOfSelect(p.Prior, ref height);
            }
        }

        static void ExprSetHeight(Expr p)
        {
            int height = 0;
            HeightOfExpr(p.Left, ref height);
            HeightOfExpr(p.Right, ref height);
            if (E.ExprHasProperty(p, EP.xIsSelect))
                HeightOfSelect(p.x.Select, ref height);
            else
                HeightOfExprList(p.x.List, ref height);
            p.Height = height + 1;
        }

        public void SetHeight(Parse parse)
        {
            ExprSetHeight(this);
            CheckHeight(parse, this.Height);
        }

        public static int SelectExprHeight(Select p)
        {
            int height = 0;
            HeightOfSelect(p, ref height);
            return height;
        }
#endif

        public static Expr Alloc(Context ctx, TK op, Token token, bool dequote)
        {
            uint extraSize = 0;
            int value = 0;
            if (token != null)
            {
                if (op != TK.INTEGER || token.data == null || !ConvertEx.Atoi(token.data, ref value))
                {
                    extraSize = token.length + 1;
                    Debug.Assert(value >= 0);
                }
            }
            Expr newExpr = new Expr();
            if (newExpr != null)
            {
                newExpr.OP = op;
                newExpr.Agg = -1;
                if (token != null)
                {
                    if (extraSize == 0)
                    {
                        newExpr.Flags |= EP.IntValue;
                        newExpr.u.I = value;
                    }
                    else
                    {
                        //: newExpr.u.Token = (char *)&newExpr[1];
                        Debug.Assert(token.data != null && token.length == 0);
                        if (token.length > 0)
                            newExpr.u.Token = token.data.Substring(0, (int)token.length);
                        else if (token.length == 0 && string.IsNullOrEmpty(token.data))
                            newExpr.u.Token = string.Empty;
                        int c;
                        if (dequote && extraSize >= 3 && ((c = token.data[0]) == '\'' || c == '"' || c == '[' || c == '`'))
                        {
                            Parse.Dequote(ref newExpr.u.Token);
                            if (c == '"')
                                newExpr.Flags |= EP.DblQuoted;
                        }
                    }
                }
#if MAX_EXPR_DEPTH
                newExpr.Height = 1;
#endif
            }
            return newExpr;
        }

        public static Expr Expr_(Context ctx, TK op, string token)
        {
            Token x = new Token();
            x.data = token;
            x.length = (uint)(!string.IsNullOrEmpty(token) ? token.Length : 0);
            return Alloc(ctx, op, x, false);
        }

        public static void AttachSubtrees(Context ctx, Expr root, Expr left, Expr right)
        {
            if (root == null)
            {
                Debug.Assert(!ctx.MallocFailed);
                Delete(ctx, ref left);
                Delete(ctx, ref right);
            }
            else
            {
                if (right != null)
                {
                    root.Right = right;
                    root.Flags |= EP.Collate & right.Flags;
                }
                if (left != null)
                {
                    root.Left = left;
                    root.Flags |= EP.Collate & left.Flags;
                }
                ExprSetHeight(root);
            }
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        public static Expr PExpr_(Parse parse, TK op, int null3, int null4, int null5) { return PExpr_(parse, op, null, null, null); }
        public static Expr PExpr_(Parse parse, TK op, int null3, int null4, Token token) { return PExpr_(parse, op, null, null, token); }
        public static Expr PExpr_(Parse parse, TK op, Expr left, int null4, int null5) { return PExpr_(parse, op, left, null, null); }
        public static Expr PExpr_(Parse parse, TK op, Expr left, int null4, Token token) { return PExpr_(parse, op, left, null, token); }
        public static Expr PExpr_(Parse parse, TK op, Expr left, Expr right, int null5) { return PExpr_(parse, op, left, right, null); }
        public static Expr PExpr_(Parse parse, TK op, Expr left, Expr right, Token token)
        {
            Context ctx = parse.Ctx;
            Expr p;
            if (op == TK.AND && left != null && right != null) // Take advantage of short-circuit false optimization for AND
                p = And(ctx, left, right);
            else
            {
                p = Alloc(ctx, op, token, true);
                AttachSubtrees(ctx, p, left, right);
            }
            if (p != null)
                CheckHeight(parse, p.Height);
            return p;
        }

        static bool ExprAlwaysFalse(Expr p)
        {
            if (E.ExprHasProperty(p, EP.FromJoin)) return false;
            int v = 0;
            if (!p.IsInteger(ref v)) return false;
            return (v == 0);
        }

        public static Expr And(Context ctx, Expr left, Expr right)
        {
            if (left == null)
                return right;
            else if (right == null)
                return left;
            else if (ExprAlwaysFalse(left) || ExprAlwaysFalse(right))
            {
                Delete(ctx, left);
                Delete(ctx, right);
                return Alloc(ctx, TK.INTEGER, &_IntTokens[0], false);
            }
            else
            {
                Expr newExpr = Alloc(ctx, TK.AND, null, false);
                AttachSubtrees(ctx, newExpr, left, right);
                return newExpr;
            }
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        public static Expr Function(Parse parse, int null2, Token token) { return Function(parse, null, token); }
        public static Expr Function(Parse parse, ExprList list, int null3) { return Function(parse, list, null); }
        public static Expr Function(Parse parse, ExprList list, Token token)
        {
            Debug.Assert(token != null);
            Context ctx = parse.Ctx;
            Expr newExpr = Alloc(ctx, TK.FUNCTION, token, true);
            if (newExpr == null)
            {
                ListDelete(ctx, ref list); // Avoid memory leak when malloc fails
                return null;
            }
            newExpr.x.List = list;
            Debug.Assert(!E.ExprHasProperty(newExpr, EP.xIsSelect));
            ExprSetHeight(newExpr);
            return newExpr;
        }

        public static void AssignVarNumber(Parse parse, Expr expr)
        {
            if (expr == null)
                return;
            Debug.Assert(!E.ExprHasAnyProperty(expr, EP.IntValue | EP.Reduced | EP.TokenOnly));
            string z = expr.u.Token;
            Debug.Assert(z != null && z.Length != 0);
            Context ctx = parse.Ctx;
            if (z.Length == 1)
            {
                Debug.Assert(z[0] == '?');
                // Wildcard of the form "?".  Assign the next variable number
                expr.ColumnIdx = (yVars)(++parse.VarsSeen);
            }
            else
            {
                yVars x = 0;
                int length = z.Length;
                if (z[0] == '?')
                {
                    // Wildcard of the form "?nnn".  Convert "nnn" to an integer and use it as the variable number
                    long i = 0;
                    bool ok = !ConvertEx.Atoi64(z.Substring(1), out i, length - 1, TEXTENCODE.UTF8);
                    expr.ColumnIdx = x = (yVars)i;
                    C.ASSERTCOVERAGE(i == 0);
                    C.ASSERTCOVERAGE(i == 1);
                    C.ASSERTCOVERAGE(i == ctx.Limits[(int)LIMIT.VARIABLE_NUMBER] - 1);
                    C.ASSERTCOVERAGE(i == ctx.Limits[(int)LIMIT.VARIABLE_NUMBER]);
                    if (!ok || i < 1 || i > ctx.Limits[(int)LIMIT.VARIABLE_NUMBER])
                    {
                        parse.ErrorMsg("variable number must be between ?1 and ?%d", ctx.Limits[(int)LIMIT.VARIABLE_NUMBER]);
                        x = 0;
                    }
                    if (i > parse.VarsSeen)
                        parse.VarsSeen = (int)i;
                }
                else
                {
                    // Wildcards like ":aaa", "$aaa" or "@aaa".  Reuse the same variable number as the prior appearance of the same name, or if the name
                    // has never appeared before, reuse the same variable number
                    yVars i;
                    for (i = 0; i < parse.Vars.length; i++)
                    {
                        if (parse.Vars[i] != null && string.Equals(z, parse.Vars[i], StringComparison.OrdinalIgnoreCase))
                        {
                            expr.ColumnIdx = x = (yVars)(i + 1);
                            break;
                        }
                    }
                    if (x == 0)
                        expr.ColumnIdx = x = (yVars)(++parse.VarsSeen);
                }
                if (x > 0)
                {
                    if (x > parse.Vars.length)
                    {
                        Array.Resize(ref parse.Vars.data, x);
                        parse.Vars.length = x;
                    }
                    if (z[0] != '?' || parse.Vars[x - 1] == null)
                    {
                        C._tagfree(ctx, ref parse.Vars.data[x - 1]);
                        parse.Vars[x - 1] = z.Substring(0, length);
                    }
                }
            }
            if (parse.Errs == 0 && parse.VarsSeen > ctx.Limits[(int)LIMIT.VARIABLE_NUMBER])
                parse.ErrorMsg("too many SQL variables");
        }

        public static void Delete(Context ctx, ref Expr expr)
        {
            if (expr == null) return;
            // Sanity check: Assert that the IntValue is non-negative if it exists
            Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue) || expr.u.I >= 0);
            if (!E.ExprHasAnyProperty(expr, EP.TokenOnly))
            {
                Delete(ctx, ref expr.Left);
                Delete(ctx, ref expr.Right);
                if (!E.ExprHasProperty(expr, EP.Reduced) && (expr.Flags2 & EP2.MallocedToken) != 0)
                    C._tagfree(ctx, ref expr.u.Token);
                if (E.ExprHasProperty(expr, EP.xIsSelect))
                    Select.Delete(ctx, ref expr.x.Select);
                else
                    ListDelete(ctx, ref expr.x.List);
            }
            if (!E.ExprHasProperty(expr, EP.Static))
                C._tagfree(ctx, ref expr);
        }

        #region Clone

        static int ExprStructSize(Expr expr)
        {
            if (E.ExprHasProperty(expr, EP.TokenOnly)) return E.EXPR_TOKENONLYSIZE;
            if (E.ExprHasProperty(expr, EP.Reduced)) return E.EXPR_REDUCEDSIZE;
            return E.EXPR_FULLSIZE;
        }

        static int DupedExprStructSize(Expr expr, int flags)
        {
            Debug.Assert(flags == E.EXPRDUP_REDUCE || flags == 0); // Only one flag value allowed
            int size;
            if ((flags & E.EXPRDUP_REDUCE) == 0)
                size = E.EXPR_FULLSIZE;
            else
            {
                Debug.Assert(!E.ExprHasAnyProperty(expr, EP.TokenOnly | EP.Reduced));
                Debug.Assert(!E.ExprHasProperty(expr, EP.FromJoin));
                Debug.Assert((expr.Flags2 & EP2.MallocedToken) == 0);
                Debug.Assert((expr.Flags2 & EP2.Irreducible) == 0);
                size = (expr.Left != null || expr.Right != null || expr.x.List != null ? E.EXPR_REDUCEDSIZE | (int)EP.Reduced : E.EXPR_TOKENONLYSIZE | (int)EP.TokenOnly);
            }
            return size;
        }

        static int DupedExprNodeSize(Expr expr, int flags)
        {
            int bytes = DupedExprStructSize(expr, flags) & 0xfff;
            if (!E.ExprHasProperty(expr, EP.IntValue) && expr.u.Token != null)
                bytes += expr.u.Token.Length + 1;
            return SysEx.ROUND8(bytes);
        }

        static int DupedExprSize(Expr expr, int flags)
        {
            int bytes = 0;
            if (expr != null)
            {
                bytes = DupedExprNodeSize(expr, flags);
                if ((flags & E.EXPRDUP_REDUCE) != 0)
                    bytes += DupedExprSize(expr.Left, flags) + DupedExprSize(expr.Right, flags);
            }
            return bytes;
        }

        static Expr ExprDup(Context ctx, Expr expr, int flags, ref Expr buffer)
        {
            Expr newExpr = null; // Value to return
            if (expr != null)
            {
                bool isReduced = ((flags & E.EXPRDUP_REDUCE) != 0);
                uint staticFlag = 0;
                Debug.Assert(buffer == null || isReduced);
                {
                    int structSize = DupedExprStructSize(expr, flags);
                    if (isReduced)
                    {
                        Debug.Assert(!E.ExprHasProperty(expr, EP.Reduced));
                        newExpr = expr.memcpy(E.EXPR_TOKENONLYSIZE);
                    }
                    else
                        newExpr = expr.memcpy();

                    // Set the EP_Reduced, EP_TokenOnly, and EP_Static flags appropriately.
                    newExpr.Flags &= (ushort)(~(EP.Reduced | EP.TokenOnly | EP.Static));
                    newExpr.Flags |= (ushort)(structSize & (EP.Reduced | EP.TokenOnly));
                    newExpr.Flags |= (ushort)staticFlag;
                    if (((expr.Flags | newExpr.Flags) & EP.TokenOnly) == 0)
                    {
                        // Fill in the pNew.x.pSelect or pNew.x.pList member.
                        if (E.ExprHasProperty(expr, EP.xIsSelect))
                            newExpr.x.Select = SelectDup(ctx, expr.x.Select, (isReduced ? 1 : 0));
                        else
                            newExpr.x.List = ListDup(ctx, expr.x.List, (isReduced ? 1 : 0));
                    }
                    // Fill in pNew->pLeft and pNew->pRight.
                    if (E.ExprHasAnyProperty(newExpr, EP.Reduced | EP.TokenOnly))
                    {
                        if (E.ExprHasProperty(newExpr, EP.Reduced))
                        {
                            newExpr.Left = ExprDup(ctx, expr.Left, E.EXPRDUP_REDUCE, ref buffer);
                            newExpr.Right = ExprDup(ctx, expr.Right, E.EXPRDUP_REDUCE, ref buffer);
                        }
                    }
                    else
                    {
                        newExpr.Flags2 = 0;
                        if (!E.ExprHasAnyProperty(expr, EP.TokenOnly))
                        {
                            Expr dummy = null;
                            newExpr.Left = ExprDup(ctx, expr.Left, 0, ref dummy);
                            newExpr.Right = ExprDup(ctx, expr.Right, 0, ref dummy);
                        }
                    }
                }
            }
            return newExpr;
        }

        public static Expr Dup(Context ctx, Expr expr, int flags)
        {
            Expr dummy = null;
            return ExprDup(ctx, expr, flags, ref dummy);
        }

        public static ExprList ListDup(Context ctx, ExprList list, int flags)
        {
            if (list == null) return null;
            ExprList newList = new ExprList();
            if (newList == null) return null;
            int i;
            newList.ECursor = 0;
            newList.Exprs = i = list.Exprs;
            ExprList.ExprListItem item;
            newList.Ids = new ExprList.ExprListItem[i];
            if (newList.Ids == null)
            {
                C._tagfree(ctx, ref newList.Ids);
                return null;
            }
            ExprList.ExprListItem oldItem;
            for (i = 0, oldItem = list.Ids[0]; i < list.Exprs; i++, oldItem = list.Ids[i])
            {
                newList.Ids[i] = item = new ExprList.ExprListItem();
                Expr oldExpr = oldItem.Expr;
                item.Expr = Dup(ctx, oldExpr, flags);
                item.Name = oldItem.Name;
                item.Span = oldItem.Span;
                item.SortOrder = oldItem.SortOrder;
                item.Done = false;
                item.OrderByCol = oldItem.OrderByCol;
                item.Alias = oldItem.Alias;
            }
            return newList;
        }

#if !OMIT_VIEW || !OMIT_TRIGGER || !OMIT_SUBQUERY
        static SrcList SrcListDup(Context ctx, SrcList list, int flags)
        {
            if (list == null)
                return null;
            SrcList newList = new SrcList
            {
                Ids = new SrcList.SrcListItem[list.Srcs],
            };
            if (newList == null)
                return null;
            newList.Srcs = newList.Allocs = list.Srcs;
            for (int i = 0; i < list.Srcs; i++)
            {
                SrcList.SrcListItem newItem = newList.Ids[i];
                SrcList.SrcListItem oldItem = list.Ids[i];
                newList.Ids[i] = new SrcList.SrcListItem();
                newItem.Schema = oldItem.Schema;
                newItem.Database = oldItem.Database;
                newItem.Name = oldItem.Name;
                newItem.Alias = oldItem.Alias;
                newItem.Jointype = oldItem.Jointype;
                newItem.Cursor = oldItem.Cursor;
                newItem.AddrFillSub = oldItem.AddrFillSub;
                newItem.RegReturn = oldItem.RegReturn;
                newItem.IsCorrelated = oldItem.IsCorrelated;
                newItem.ViaCoroutine = oldItem.ViaCoroutine;
                newItem.IndexName = oldItem.IndexName;
                newItem.NotIndexed = oldItem.NotIndexed;
                newItem.Index = oldItem.Index;
                Table table = newItem.Table = oldItem.Table;
                if (table != null)
                    table.Refs++;
                newItem.Select = SelectDup(ctx, oldItem.Select, flags);
                newItem.On = Dup(ctx, oldItem.On, flags);
                newItem.Using = IdListDup(ctx, oldItem.Using);
                newItem.ColUsed = oldItem.ColUsed;
            }
            return newList;
        }

        public static IdList IdListDup(Context ctx, IdList list)
        {
            if (list == null)
                return null;
            IdList newList = new IdList();
            if (newList == null)
                return null;
            newList.Ids.length = list.Ids.length;
            newList.Ids.data = new IdList.IdListItem[list.Ids.length];
            if (newList.Ids.data == null)
            {
                C._tagfree(ctx, ref newList);
                return null;
            }
            // Note that because the size of the allocation for p->a[] is not necessarily a power of two, sqlite3IdListAppend() may not be called
            // on the duplicate created by this function.
            for (int i = 0; i < list.Ids.length; i++)
            {
                IdList.IdListItem newItem = newList.Ids[i];
                IdList.IdListItem oldItem = list.Ids[i];
                newList.Ids[i] = new IdList.IdListItem();
                newItem.Name = oldItem.Name;
                newItem.Idx = oldItem.Idx;
            }
            return newList;
        }

        public static Select SelectDup(Context ctx, Select select, int flags)
        {
            if (select == null)
                return null;
            Select newSelect = new Select();
            if (newSelect == null)
                return null;
            newSelect.EList = ListDup(ctx, select.EList, flags);
            newSelect.Src = SrcListDup(ctx, select.Src, flags);
            newSelect.Where = Dup(ctx, select.Where, flags);
            newSelect.GroupBy = ListDup(ctx, select.GroupBy, flags);
            newSelect.Having = Dup(ctx, select.Having, flags);
            newSelect.OrderBy = ListDup(ctx, select.OrderBy, flags);
            newSelect.OP = select.OP;
            Select prior;
            newSelect.Prior = prior = SelectDup(ctx, select.Prior, flags);
            if (prior != null)
                prior.Next = newSelect;
            newSelect.Limit = Dup(ctx, select.Limit, flags);
            newSelect.Offset = Dup(ctx, select.Offset, flags);
            newSelect.LimitId = 0;
            newSelect.OffsetId = 0;
            newSelect.SelFlags = select.SelFlags & ~SF.UsesEphemeral;
            newSelect.Rightmost = null;
            newSelect.AddrOpenEphm[0] = -1;
            newSelect.AddrOpenEphm[1] = -1;
            newSelect.AddrOpenEphm[2] = -1;
            return newSelect;
        }
#else
        public Select SelectDup(Context ctx, Select select, int flags) { Debug.Assert(select == null); return null; }
#endif

        #endregion

        // OVERLOADS, so I don't need to rewrite parse.c
        public static ExprList ListAppend(Parse parse, int null2, Expr expr) { return ListAppend(parse, null, expr); }
        public static ExprList ListAppend(Parse parse, ExprList list, Expr expr)
        {
            Context ctx = parse.Ctx;
            if (list == null)
            {
                list = new ExprList();
                if (list == null)
                    goto no_mem;
                list.Ids = new ExprList.ExprListItem[1];
                if (list.Ids == null)
                    goto no_mem;
            }
            else if ((list.Exprs & (list.Exprs - 1)) == 0)
            {
                Debug.Assert(list.Exprs > 0);
                Array.Resize(ref list.Ids, list.Exprs * 2);
                if (list.Ids == null)
                    goto no_mem;
            }
            Debug.Assert(list.Ids != null);
            ExprList.ExprListItem item = list.Ids[list.Exprs++] = new ExprList.ExprListItem();
            list.Exprs = list.Ids.Length;
            item.Expr = expr;
            return list;

        no_mem:
            // Avoid leaking memory if malloc has failed.
            Delete(ctx, ref expr);
            ListDelete(ctx, ref list);
            return null;
        }

        public static void ListSetName(Parse parse, ExprList list, Token name, bool dequote)
        {
            Debug.Assert(list != null || parse.Ctx.MallocFailed);
            if (list != null)
            {
                Debug.Assert(list.Exprs > 0);
                ExprList.ExprListItem item = list.Ids[list.Exprs - 1];
                Debug.Assert(item.Name == null);
                item.Name = name.data.Substring(0, (int)name.length);
                if (dequote && !string.IsNullOrEmpty(item.Name))
                    Parse.Dequote(ref item.Name);
            }
        }

        public static void ListSetSpan(Parse parse, ExprList list, ExprSpan span)
        {
            Context ctx = parse.Ctx;
            Debug.Assert(list != null || ctx.MallocFailed);
            if (list != null)
            {
                ExprList.ExprListItem item = list.Ids[list.Exprs - 1];
                Debug.Assert(list.Exprs > 0);
                Debug.Assert(ctx.MallocFailed || item.Expr == span.Expr);
                C._tagfree(ctx, ref item.Span);
                item.Span = span.Start.Substring(0, span.Start.Length <= span.End.Length ? span.Start.Length : span.Start.Length - span.End.Length);
            }
        }

        public static void ListCheckLength(Parse parse, ExprList list, string object_)
        {
            int max = parse.Ctx.Limits[(int)LIMIT.COLUMN];
            C.ASSERTCOVERAGE(list != null && list.Exprs == max);
            C.ASSERTCOVERAGE(list != null && list.Exprs == max + 1);
            if (list != null && list.Exprs > max)
                parse.ErrorMsg("too many columns in %s", object_);
        }

        public static void ListDelete(Context ctx, ref ExprList list)
        {
            if (list == null)
                return;
            Debug.Assert(list.Ids != null || list.Exprs == 0);
            int i;
            ExprList.ExprListItem item = list.Ids[0];
            for (i = 0; i < list.Exprs; i++)
            {
                item = list.Ids[i];
                Delete(ctx, ref item.Expr);
                C._tagfree(ctx, ref item.Name);
                C._tagfree(ctx, ref item.Span);
            }
            C._tagfree(ctx, ref list.Ids);
            C._tagfree(ctx, ref list);
        }

        #region Walker - Expression Tree Walker

        static WRC ExprNodeIsConstant(Walker walker, Expr expr)
        {
            // If pWalker->u.i is 3 then any term of the expression that comes from the ON or USING clauses of a join disqualifies the expression
            // from being considered constant. */
            if (walker.u.I == 3 && E.ExprHasAnyProperty(expr, EP.FromJoin))
            {
                walker.u.I = 0;
                return WRC.Abort;
            }
            switch (expr.OP)
            {
                case TK.FUNCTION:
                    // Consider functions to be constant if all their arguments are constant and pWalker->u.i==2
                    if (walker.u.I == 2)
                        return WRC.Continue;
                    goto case TK.ID; // Fall through

                case TK.ID:
                case TK.COLUMN:
                case TK.AGG_FUNCTION:
                case TK.AGG_COLUMN:
                    C.ASSERTCOVERAGE(expr.OP == TK.ID);
                    C.ASSERTCOVERAGE(expr.OP == TK.COLUMN);
                    C.ASSERTCOVERAGE(expr.OP == TK.AGG_FUNCTION);
                    C.ASSERTCOVERAGE(expr.OP == TK.AGG_COLUMN);
                    walker.u.I = 0;
                    return WRC.Abort;
                default:
                    C.ASSERTCOVERAGE(expr.OP == TK.SELECT); // selectNodeIsConstant will disallow
                    C.ASSERTCOVERAGE(expr.OP == TK.EXISTS); // selectNodeIsConstant will disallow
                    return WRC.Continue;
            }
        }

        static WRC SelectNodeIsConstant(Walker walker, Select notUsed)
        {
            walker.u.I = 0;
            return WRC.Abort;
        }

        static bool ExprIsConst(Expr expr, int initFlag)
        {
            Walker w = new Walker();
            w.u.I = initFlag;
            w.ExprCallback = ExprNodeIsConstant;
            w.SelectCallback = SelectNodeIsConstant;
            w.WalkExpr(expr);
            return (w.u.I != 0);
        }

        public bool IsConstant() { return ExprIsConst(this, 1); }
        public bool IsConstantNotJoin() { return ExprIsConst(this, 3); }
        public bool IsConstantOrFunction() { return ExprIsConst(this, 2); }
        public bool IsInteger(ref int value)
        {
            // If an expression is an integer literal that fits in a signed 32-bit integer, then the EP_IntValue flag will have already been set
            bool rc = false;
            Debug.Assert(p.OP != TK.INTEGER || (p.Flags & EP.IntValue) != 0 || !ConvertEx.Atoi(p.u.Token, ref rc));
            if ((p.Flags & EP.IntValue) != 0)
            {
                value = (int)p.u.I;
                return true;
            }
            switch (p.OP)
            {
                case TK.UPLUS: return p.Left.IsInteger(ref value);
                case TK.UMINUS:
                    int v = 0;
                    if (p.Left.IsInteger(ref v))
                    {
                        value = -v;
                        rc = true;
                    }
                    return false;
                default: return false;
            }
        }

        #endregion

        public bool CanBeNull()
        {
            Expr expr = this;
            while (expr.OP == TK.UPLUS || expr.OP == TK.UMINUS) expr = expr.Left;
            TK op = expr.OP;
            if (op == TK.REGISTER)
                op = expr.OP2;
            switch (op)
            {
                case TK.INTEGER:
                case TK.STRING:
                case TK.FLOAT:
                case TK.BLOB: return false;
                default: return true;
            }
        }

        public static void CodeIsNullJump(Vdbe v, Expr expr, int reg, int dest)
        {
            if (expr.CanBeNull())
                v.AddOp2(OP.IsNull, reg, dest);
        }

        public bool NeedsNoAffinityChange(AFF aff)
        {
            if (aff == AFF.NONE)
                return true;
            Expr expr = this;
            while (expr.OP == TK.UPLUS || expr.OP == TK.UMINUS) expr = expr.Left;
            TK op = expr.OP;
            if (op == TK.REGISTER)
                op = expr.OP2;
            switch (op)
            {
                case TK.INTEGER: return aff == AFF.INTEGER || aff == AFF.NUMERIC;
                case TK.FLOAT: return aff == AFF.REAL || aff == AFF.NUMERIC;
                case TK.STRING: return aff == AFF.TEXT;
                case TK.BLOB: return true;
                case TK.COLUMN:
                    {
                        Debug.Assert(expr.TableId >= 0); // p cannot be part of a CHECK constraint
                        return expr.ColumnIdx < 0 && (aff == AFF.INTEGER || aff == AFF.NUMERIC);
                    }
                default: return false;
            }
        }

        public static bool IsRowid(string z)
        {
            return (z.Equals("_ROWID_", StringComparison.OrdinalIgnoreCase) || z.Equals("ROWID", StringComparison.OrdinalIgnoreCase) || z.Equals("OID", StringComparison.OrdinalIgnoreCase));
        }

        #region SUBQUERY

        public static int CodeOnce(Parse parse)
        {
            Vdbe v = parse.GetVdbe(); // Virtual machine being coded
            return v.AddOp1(OP.Once, parse.Onces++);
        }

#if !OMIT_SUBQUERY
        static bool IsCandidateForInOpt(Select select)
        {
            if (select == null) return false;               // right-hand side of IN is SELECT
            if (select.Prior != null) return false;         // Not a compound SELECT
            if ((select.SelFlags & (SF.Distinct | SF.Aggregate)) != 0)
            {
                C.ASSERTCOVERAGE((select.SelFlags & (SF.Distinct | SF.Aggregate)) == SF.Distinct);
                C.ASSERTCOVERAGE((select.SelFlags & (SF.Distinct | SF.Aggregate)) == SF.Aggregate);
                return false; // No DISTINCT keyword and no aggregate functions
            }
            Debug.Assert(select.GroupBy == null);           // Has no GROUP BY clause
            if (select.Limit != null) return false;         // Has no LIMIT clause
            Debug.Assert(select.Offset == null);            // No LIMIT means no OFFSET
            if (select.Where != null) return false;         // Has no WHERE clause
            SrcList src = select.Src;
            Debug.Assert(src != null);
            if (src.Srcs != 1) return false;                // Single term in FROM clause
            if (src.Ids[0].Select != null) return false;    // FROM is not a subquery or view
            Table table = src.Ids[0].Table;
            if (C._NEVER(table == null)) return false;
            Debug.Assert(table.Select == null);             // FROM clause is not a view
            if (E.IsVirtual(table)) return false;           // FROM clause not a virtual table
            ExprList list = select.EList;
            if (list.Exprs != 1) return false;              // One column in the result set
            if (list.Ids[0].Expr.OP != TK.COLUMN) return false; // Result is a column
            return true;
        }

        public static IN_INDEX FindInIndex(Parse parse, Expr expr, ref int notFound)
        {
            Debug.Assert(expr.OP == TK.IN);
            IN_INDEX type = 0; // Type of RHS table. IN_INDEX_*
            int tableIdx = parse.Tabs++; // Cursor of the RHS table
            bool mustBeUnique = true; // True if RHS must be unique
            Vdbe v = parse.GetVdbe();      // Virtual machine being coded

            // Check to see if an existing table or index can be used to satisfy the query.  This is preferable to generating a new ephemeral table.
            Select select = (E.ExprHasProperty(expr, EP.xIsSelect) ? expr.x.Select : null); // SELECT to the right of IN operator
            if (C._ALWAYS(parse.Errs == 0) && IsCandidateForInOpt(select))
            {
                Debug.Assert(select != null); // Because of isCandidateForInOpt(p)
                Debug.Assert(select.EList != null); // Because of isCandidateForInOpt(p)
                Debug.Assert(select.EList.Ids[0].Expr != null); // Because of isCandidateForInOpt(p)
                Debug.Assert(select.Src != null); // Because of isCandidateForInOpt(p)

                Context ctx = parse.Ctx; // Database connection
                Table table = select.Src.Ids[0].Table; // Table <table>.
                Expr expr2 = select.EList.Ids[0].Expr; // Expression <column>
                int col = expr2.ColumnIdx; // Index of column <column>

                // Code an OP_VerifyCookie and OP_TableLock for <table>.
                int db = SchemaToIndex(ctx, table.Schema); // Database idx for pTab
                CodeVerifySchema(parse, db);
                parse.TableLock(db, table.Id, 0, table.Name);

                // This function is only called from two places. In both cases the vdbe has already been allocated. So assume sqlite3GetVdbe() is always
                // successful here.
                Debug.Assert(v != null);
                if (col < 0)
                {
                    int addr = CodeOnce(parse);
                    Insert.OpenTable(parse, tableIdx, db, table, OP.OpenRead);
                    type = IN_INDEX.ROWID;
                    v.JumpHere(addr);
                }
                else
                {
                    // The collation sequence used by the comparison. If an index is to be used in place of a temp-table, it must be ordered according
                    // to this collation sequence.
                    CollSeq req = BinaryCompareCollSeq(parse, expr.Left, expr2);
                    // Check that the affinity that will be used to perform the comparison is the same as the affinity of the column. If
                    // it is not, it is not possible to use any index.
                    bool validAffinity = expr.ValidIndexAffinity(table.Cols[col].Affinity);
                    for (Index index = table.Index; index != null && type == 0 && validAffinity; index = index.Next)
                        if (index.Columns[0] == col && Callback.FindCollSeq(ctx, E.CTXENCODE(ctx), index.CollNames[0], false) == req && (!mustBeUnique || (index.Columns.length == 1 && index.OnError != OE.None)))
                        {
                            KeyInfo key = parse.IndexKeyinfo(index);
                            int addr = CodeOnce(parse);
                            v.AddOp4(OP.OpenRead, tableIdx, index.Id, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
                            E.VdbeComment(v, "%s", index.Name);
                            type = (IN_INDEX)IN_INDEX.INDEX_ASC + index.SortOrders[0];
                            v.JumpHere(addr);
                            if (notFound != -1 && table.Cols[col].NotNull == 0) // Klude to show prNotFound not available
                            {
                                notFound = ++parse.Mems;
                                v.AddOp2(OP.Null, 0, notFound);
                            }
                        }
                }
            }

            if (type == 0)
            {
                // Could not found an existing table or index to use as the RHS b-tree. We will have to generate an ephemeral table to do the job.
                double savedNQueryLoop = parse.QueryLoops;
                int mayHaveNull = 0;
                type = IN_INDEX.EPH;
                if (notFound != -1) // Klude to show prNotFound not available
                {
                    notFound = mayHaveNull = ++parse.Mems;
                    v.AddOp2(OP.Null, 0, notFound);
                }
                else
                {
                    C.ASSERTCOVERAGE(parse.QueryLoops > (double)1);
                    parse.QueryLoops = (double)1;
                    if (expr.Left.ColumnIdx < 0 && !E.ExprHasAnyProperty(expr, EP.xIsSelect))
                        type = IN_INDEX.ROWID;
                }
                CodeSubselect(parse, expr, mayHaveNull, type == IN_INDEX.ROWID);
                parse.QueryLoops = savedNQueryLoop;
            }
            else
                expr.TableId = tableIdx;
            return type;
        }

        static byte[] _CodeSubselect_SortOrder = new byte[] { 0 }; // Fake aSortOrder for keyInfo
        public static int CodeSubselect(Parse parse, Expr expr, int mayHaveNull, bool isRowid)
        {
            int reg = 0; // Register storing resulting
            Vdbe v = parse.GetVdbe();
            if (C._NEVER(v == null))
                return 0;
            CachePush(parse);

            // This code must be run in its entirety every time it is encountered if any of the following is true:
            //    *  The right-hand side is a correlated subquery
            //    *  The right-hand side is an expression list containing variables
            //    *  We are inside a trigger
            // If all of the above are false, then we can run this code just once save the results, and reuse the same result on subsequent invocations.
            int testAddr = (!E.ExprHasAnyProperty(expr, EP.VarSelect) ? CodeOnce(parse) : -1); // One-time test address
#if !OMIT_EXPLAIN
            if (parse.Explain == 2)
            {
                string msg = SysEx.Mprintf(parse.Ctx, "EXECUTE %s%s SUBQUERY %d", (testAddr >= 0 ? string.Empty : "CORRELATED "), (expr.OP == TK.IN ? "LIST" : "SCALAR"), parse.NextSelectId);
                v.AddOp4(OP.Explain, parse.SelectId, 0, 0, msg, Vdbe.P4T.DYNAMIC);
            }
#endif
            switch (expr.OP)
            {
                case TK.IN:
                    {
                        Expr left = expr.Left; // the LHS of the IN operator
                        if (mayHaveNull != 0)
                            v.AddOp2(OP.Null, 0, mayHaveNull);
                        AFF affinity = left.Affinity();  // Affinity of the LHS of the IN

                        // Whether this is an 'x IN(SELECT...)' or an 'x IN(<exprlist>)' expression it is handled the same way.  An ephemeral table is 
                        // filled with single-field index keys representing the results from the SELECT or the <exprlist>.
                        //
                        // If the 'x' expression is a column value, or the SELECT... statement returns a column value, then the affinity of that
                        // column is used to build the index keys. If both 'x' and the SELECT... statement are columns, then numeric affinity is used
                        // if either column has NUMERIC or INTEGER affinity. If neither 'x' nor the SELECT... statement are columns, then numeric affinity is used.
                        expr.TableId = parse.Tabs++;
                        int addr = v.AddOp2(OP.OpenEphemeral, (int)expr.TableId, !isRowid); // Address of OP_OpenEphemeral instruction
                        if (mayHaveNull == 0) v.ChangeP5(BTREE_UNORDERED);
                        KeyInfo keyInfo = new KeyInfo(); // Keyinfo for the generated table
                        keyInfo.Fields = 1;
                        keyInfo.SortOrders = _CodeSubselect_SortOrder;

                        if (E.ExprHasProperty(expr, EP.xIsSelect))
                        {
                            // Case 1:     expr IN (SELECT ...)
                            // Generate code to write the results of the select into the temporary table allocated and opened above.
                            Debug.Assert(!isRowid);
                            SelectDest dest = new SelectDest();
                            SelectDestInit(dest, SRT.Set, expr.TableId);
                            dest.AffSdst = affinity;
                            Debug.Assert((expr.TableId & 0x0000FFFF) == expr.TableId);
                            expr.x.Select.LimitId = 0;
                            if (Select(parse, expr.x.Select, ref dest) != 0)
                                return 0;
                            ExprList list = expr.x.Select.EList;
                            if (C._ALWAYS(list != null && list.Exprs > 0))
                                keyInfo.Colls[0] = BinaryCompareCollSeq(parse, expr.Left, list.Ids[0].Expr);
                        }
                        else if (C._ALWAYS(expr.x.List != null))
                        {
                            // Case 2:     expr IN (exprlist)
                            // For each expression, build an index key from the evaluation and store it in the temporary table. If <expr> is a column, then use
                            // that columns affinity when building index keys. If <expr> is not a column, use numeric affinity.
                            ExprList list = expr.x.List;
                            if (affinity == 0)
                                affinity = AFF.NONE;
                            keyInfo.Colls[0] = expr.Left.CollSeq(parse);
                            keyInfo.SortOrders = _CodeSubselect_SortOrder;

                            // Loop through each expression in <exprlist>.
                            int r1 = GetTempReg(parse);
                            int r2 = GetTempReg(parse);
                            sqlite3VdbeAddOp2(v, OP_Null, 0, r2);
                            int i;
                            ExprList.ExprListItem item;
                            for (i = 0, item = list.Ids[0]; i < list.Exprs; i++, item = list.Ids[i])
                            {
                                Expr e2 = item.Expr;
                                // If the expression is not constant then we will need to disable the test that was generated above that makes sure
                                // this code only executes once.  Because for a non-constant expression we need to rerun this code each time.
                                if (testAddr >= 0 && !e2.IsConstant())
                                {
                                    v.ChangeToNoop(testAddr - 1, 2);
                                    testAddr = -1;
                                }
                                // Evaluate the expression and insert it into the temp table
                                int valToIns = 0;
                                if (isRowid && e2.IsInteger(ref valToIns))
                                    v.AddOp3(OP.InsertInt, expr.TableId, r2, valToIns);
                                else
                                {
                                    int r3 = ExprCodeTarget(parse, e2, r1);
                                    if (isRowid)
                                    {
                                        v.AddOp2(OP.MustBeInt, r3, v.CurrentAddr() + 2);
                                        v.AddOp3(OP.Insert, expr.TableId, r2, r3);
                                    }
                                    else
                                    {
                                        v.AddOp4(OP.MakeRecord, r3, 1, r2, affinity, 1);
                                        ExprCacheAffinityChange(parse, r3, 1);
                                        v.AddOp2(OP.IdxInsert, expr.Tableidx, r2);
                                    }
                                }
                            }
                            ReleaseTempReg(parse, r1);
                            ReleaseTempReg(parse, r2);
                        }
                        if (!isRowid)
                            v.ChangeP4(addr, keyInfo, Vdbe.P4T.KEYINFO);
                        break;
                    }

                case TK.EXISTS:
                case TK.SELECT:
                default:
                    {
                        C.ASSERTCOVERAGE(expr.OP == TK.EXISTS);
                        C.ASSERTCOVERAGE(expr.OP == TK.SELECT);
                        Debug.Assert(expr.OP == TK.EXISTS || expr.OP == TK.SELECT);
                        Debug.Assert(E.ExprHasProperty(expr, EP.xIsSelect));

                        // If this has to be a scalar SELECT.  Generate code to put the value of this select in a memory cell and record the number
                        // of the memory cell in iColumn.  If this is an EXISTS, write an integer 0 (not exists) or 1 (exists) into a memory cell
                        // and record that memory cell in iColumn.

                        Select sel = expr.x.Select; // SELECT statement to encode
                        SelectDest dest = new SelectDest(); // How to deal with SELECt result
                        SelectDestInit(dest, 0, ++parse.Mems);
                        if (expr.OP == TK.SELECT)
                        {
                            dest.Dest = SRT.Mem;
                            v.AddOp2(OP.Null, 0, dest.SDParmId);
                            E.VdbeComment(v, "Init subquery result");
                        }
                        else
                        {
                            dest.Dest = SRT.Exists;
                            v.AddOp2(OP.Integer, 0, dest.SDParmId);
                            E.VdbeComment(v, "Init EXISTS result");
                        }
                        Delete(parse.Ctx, ref sel.Limit);
                        sel.Limit = PExpr_(parse, TK.INTEGER, null, null, _IntTokens[1]);
                        sel.LimitId = 0;
                        if (Select(parse, sel, ref dest) != 0)
                            return 0;
                        reg = dest.SDParmId;
                        E.ExprSetIrreducible(expr);
                        break;
                    }
            }
            if (testAddr >= 0)
                v.JumpHere(testAddr);
            ExprCachePop(parse, 1);
            return rReg;
        }

        public static void CodeIN(Parse parse, Expr expr, int destIfFalse, int destIfNull)
        {
            // Compute the RHS. After this step, the table with cursor expr->TableId will contains the values that make up the RHS.
            Vdbe v = parse.Vdbe; // Statement under construction
            Debug.Assert(v != null); // OOM detected prior to this routine
            VdbeNoopComment(v, "begin IN expr");
            int rhsHasNull = 0; // Register that is true if RHS contains NULL values
            int type = FindInIndex(parse, expr, ref rhsHasNull); // Type of the RHS

            // Figure out the affinity to use to create a key from the results of the expression. affinityStr stores a static string suitable for P4 of OP_MakeRecord.
            AFF affinity = ComparisonAffinity(expr); // Comparison affinity to use

            // Code the LHS, the <expr> from "<expr> IN (...)".
            CachePush(parse);
            int r1 = GetTempReg(parse); // Temporary use register
            ExprCode(parse, expr.pLeft, r1);

            // If the LHS is NULL, then the result is either false or NULL depending on whether the RHS is empty or not, respectively.
            if (destIfNull == destIfFalse)
            {
                // Shortcut for the common case where the false and NULL outcomes are the same.
                v.AddOp2(OP.IsNull, r1, destIfNull);
            }
            else
            {
                int addr1 = v.AddOp1(OP.NotNull, r1);
                v.AddOp2(OP.Rewind, expr.TableId, destIfFalse);
                v.AddOp2(OP.Goto, 0, destIfNull);
                v.JumpHere(addr1);
            }

            if (type == IN_INDEX.ROWID)
            {
                // In this case, the RHS is the ROWID of table b-tree
                v.AddOp2(v, OP_MustBeInt, r1, destIfFalse);
                v.AddOp3(v, OP_NotExists, expr.iTable, destIfFalse, r1);
            }
            else
            {
                // In this case, the RHS is an index b-tree.
                v.AddOp4(OP.Affinity, r1, 1, 0, affinity, 1);

                // If the set membership test fails, then the result of the  "x IN (...)" expression must be either 0 or NULL. If the set
                // contains no NULL values, then the result is 0. If the set contains one or more NULL values, then the result of the
                // expression is also NULL.
                if (rhsHasNull == 0 || destIfFalse == destIfNull)
                {
                    // This branch runs if it is known at compile time that the RHS cannot contain NULL values. This happens as the result
                    // of a "NOT NULL" constraint in the database schema.
                    //
                    // Also run this branch if NULL is equivalent to FALSE for this particular IN operator.
                    v.AddOp4Int(OP.NotFound, expr.TableId, destIfFalse, r1, 1);
                }
                else
                {
                    // In this branch, the RHS of the IN might contain a NULL and the presence of a NULL on the RHS makes a difference in the outcome.

                    // First check to see if the LHS is contained in the RHS. If so, then the presence of NULLs in the RHS does not matter, so jump
                    // over all of the code that follows.
                    int j1 = sqlite3VdbeAddOp4Int(v, OP_Found, expr.iTable, 0, r1, 1);

                    // Here we begin generating code that runs if the LHS is not contained within the RHS.  Generate additional code that
                    // tests the RHS for NULLs.  If the RHS contains a NULL then jump to destIfNull.  If there are no NULLs in the RHS then
                    // jump to destIfFalse.
                    int j2 = v.AddOp1(OP.NotNull, rhsHasNull);
                    int j3 = v.AddOp4Int(OP.Found, expr.TableId, 0, rhsHasNull, 1);
                    v.AddOp2(OP.Integer, -1, rhsHasNull);
                    v.JumpHere(j3);
                    v.AddOp2(OP.AddImm, rhsHasNull, 1);
                    v.JumpHere(j2);

                    // Jump to the appropriate target depending on whether or not the RHS contains a NULL
                    v.AddOp2(OP.If, rhsHasNull, destIfNull);
                    v.AddOp2(OP.Goto, 0, destIfFalse);

                    // The OP_Found at the top of this branch jumps here when true, causing the overall IN expression evaluation to fall through.
                    v.JumpHere(j1);
                }
            }
            ReleaseTempReg(parse, r1);
            ExprCachePop(parse, 1);
            E.VdbeComment(v, "end IN expr");
        }

#endif
        #endregion


        //static char *Dup8bytes(Vdbe v, string in) { } // SKIPED

#if !FLOATING_POINT

        static void CodeReal(Vdbe v, string z, bool negateFlag, int mem)
        {
            if (C._ALWAYS(!string.IsNullOrEmpty(z)))
            {
                double value = 0;
                //string zV;
                ConvertEx.Atof(z, ref value, z.length, TEXTENCODE.UTF8);
                Debug.Assert(!IsNaN(value)); // The new AtoF never returns NaN
                if (negateFlag) value = -value;
                v.AddOp4(OP.Real, 0, mem, 0, value, Vdbe.P4T.REAL);
            }
        }
#endif

        static void CodeInteger(Parse parse, Expr expr, bool negateFlag, int mem)
        {
            Vdbe v = parse.V;
            if ((expr.Flags & EP.IntValue) != 0)
            {
                int i = expr.u.I;
                Debug.Assert(i >= 0);
                if (negateFlag) i = -i;
                v.AddOp2(OP.Integer, i, mem);
            }
            else
            {
                string z = expr.u.Token;
                Debug.Assert(z != null);
                long value;
                int c = ConvertEx.Atoi64(z, out value, z.Length, TEXTENCODE.UTF8);
                if (c == 0 || (c == 2 && negateFlag))
                {
                    if (negateFlag)
                        value = (c == 2 ? SMALLEST_INT64 : -value);
                    v.AddOp4(OP.Int64, 0, mem, 0, value, P4_INT64);
                }
                else
                {
#if OMIT_FLOATING_POINT
                    parse.ErrorMsg("oversized integer: %s%s", (negateFlag ? "-" : string.Empty), z);
#else
                    CodeReal(v, z, negateFlag, mem);
#endif
                }
            }
        }

        #region Column Cache

        static void CacheEntryClear(Parse parse, yColCache p)
        {
            if (p.TempReg != 0)
            {
                if (parse.TempReg.length < parse.TempReg.data.Length)
                    parse.TempReg[parse.TempReg.length++] = p.Reg;
                p.TempReg = 0;
            }
        }

        public static void CacheStore(Parse parse, int table, int column, int reg)
        {
            Debug.Assert(reg > 0);  // Register numbers are always positive
            Debug.Assert(column >= -1 && column < 32768);  // Finite column numbers
            // The SQLITE_ColumnCache flag disables the column cache.  This is used for testing only - to verify that SQLite always gets the same answer
            // with and without the column cache.
            if (E.CtxOptimizationDisabled(parse.Ctx, OPTFLAG.ColumnCache))
                return;

            // First replace any existing entry.
            // Actually, the way the column cache is currently used, we are guaranteed that the object will never already be in cache.  Verify this guarantee.
            int i;
            Parse.ColCache p;
#if !NDEBUG
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
                Debug.Assert(p.Reg == 0 || p.Table != table || p.Column != column);
#endif
            // Find an empty slot and replace it
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                if (p.Reg == 0)
                {
                    p.Level = parse.CacheLevel;
                    p.Table = table;
                    p.Column = column;
                    p.Reg = reg;
                    p.TempReg = 0;
                    p.Lru = parse.CacheCnt++;
                    return;
                }
            }

            // Replace the last recently used
            int minLru = 0x7fffffff;
            int idxLru = -1;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                if (p.Lru < minLru)
                {
                    idxLru = i;
                    minLru = p.lru;
                }
            }
            if (C._ALWAYS(idxLru >= 0))
            {
                p = parse.ColCaches[idxLru];
                p.Level = parse.CacheLevel;
                p.Table = table;
                p.Column = column;
                p.Reg = reg;
                p.TempReg = 0;
                p.Lru = parse.CacheCnt++;
                return;
            }
        }

        public static void CacheRemove(Parse parse, int reg, int regs)
        {
            int last = reg + regs - 1;
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                int r = p.Reg;
                if (r >= reg && r <= last)
                {
                    CacheEntryClear(parse, p);
                    p.Reg = 0;
                }
            }
        }

        public static void CachePush(Parse parse)
        {
            parse.CacheLevel++;
        }

        public static void CachePop(Parse parse, int n)
        {
            Debug.Assert(n > 0);
            Debug.Assert(parse.CacheLevel >= n);
            parse.CacheLevel -= n;
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
                if (p.Reg != 0 && p.Level > parse.CacheLevel)
                {
                    CacheEntryClear(parse, p);
                    p.Reg = 0;
                }
        }

        static void CachePinRegister(Parse parse, int reg)
        {
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
                if (p.Reg == reg)
                    p.TempReg = 0;
        }

        public static void CodeGetColumnOfTable(Vdbe v, Table table, int tableId, int column, int regOut)
        {
            if (column < 0 || column == table.PKey)
                v.AddOp2(OP.Rowid, tableId, regOut);
            else
            {
                int op = (E.IsVirtual(table) ? OP.VColumn : OP.Column);
                v.AddOp3(op, tableId, column, regOut);
            }
            if (column >= 0)
                Update.ColumnDefault(v, table, column, regOut);
        }

        public static int CodeGetColumn(Parse parse, Table table, int column, int tableId, int reg, byte p5)
        {
            Vdbe v = parse.V;
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                if (p.Reg > 0 && p.Table == tableId && p.Column == column)
                {
                    p.Lru = parse.CacheCnt++;
                    CachePinRegister(parse, p.Reg);
                    return p.Reg;
                }
            }
            Debug.Assert(v != null);
            CodeGetColumnOfTable(v, table, tableId, column, reg);
            if (p5 != 0)
                v.ChangeP5(p5);
            else
                CacheStore(parse, tableId, column, reg);
            return reg;
        }

        public static void CacheClear(Parse parse)
        {
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
                if (p.Reg != 0)
                {
                    CacheEntryClear(parse, p);
                    p.Reg = 0;
                }
        }

        public static void CacheAffinityChange(Parse parse, int start, int count)
        {
            CacheRemove(parse, start, count);
        }

        public static void CodeMove(Parse parse, int from, int to, int regs)
        {
            Debug.Assert(from >= to + regs || from + regs <= to);
            parse.V.AddOp3(OP.Move, from, to, regs);
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                int r = p.Reg;
                if (r >= from && r < from + regs)
                    p.Reg += to - from;
            }
        }

        #endregion


#if DEBUG || COVERAGE_TEST
        static bool UsedAsColumnCache(Parse parse, int from, int to)
        {
            int i;
            Parse.ColCache p;
            for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
            {
                int r = p.Reg;
                if (r >= from && r <= to)
                    return true;
            }
            return false;
        }
#endif

        public static int CodeTarget(Parse parse, Expr expr, int target)
        {
            Debug.Assert(target > 0 && target <= parse.Mems);
            Context ctx = parse.Ctx; // The database connection
            Vdbe v = parse.V; // The VM under construction
            if (v == null)
            {
                Debug.Assert(ctx.MallocFailed);
                return 0;
            }

            int inReg = target;       // Results stored in register inReg
            int regFree1 = 0;         // If non-zero free this temporary register
            int regFree2 = 0;         // If non-zero free this temporary register
            int r1, r2, r3, r4;       // Various register numbers

            TK op = (expr == null ? TK.NULL : expr.OP); // The opcode being coded
            switch (op)
            {
                case TK.AGG_COLUMN:
                    {
                        AggInfo aggInfo = expr.AggInfo;
                        AggInfo.AggInfoColumn col = aggInfo.Cols[expr.Agg];
                        if (aggInfo.DirectMode == 0)
                        {
                            Debug.Assert(col.Mem > 0);
                            inReg = col.Mem;
                            break;
                        }
                        else if (aggInfo.UseSortingIdx != 0)
                        {
                            v.AddOp3(OP.Column, aggInfo.SortingIdxPTab, col.SorterColumn, target);
                            break;
                        }
                    }
                    goto case TK_COLUMN;
                // Otherwise, fall thru into the TK_COLUMN case
                case TK.COLUMN:
                    {
                        if (expr.TableId < 0)
                        {
                            // This only happens when coding check constraints
                            Debug.Assert(parse.CkBase > 0);
                            inReg = expr.ColumnIdx + parse.CkBase;
                        }
                        else
                            inReg = CodeGetColumn(parse, expr.Table, expr.ColumnIdx, expr.TableId, target, (byte)expr->OP2);
                        break;
                    }

                case TK.INTEGER:
                    {
                        CodeInteger(parse, expr, false, target);
                        break;
                    }

#if !OMIT_FLOATING_POINT
                case TK.FLOAT:
                    {
                        Debug.Assert(!ExprHasProperty(expr, EP.IntValue));
                        CodeReal(v, expr.u.Token, false, target);
                        break;
                    }
#endif

                case TK.STRING:
                    {
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        v.AddOp4(OP.String8, 0, target, 0, expr.u.Token, 0);
                        break;
                    }

                case TK.NULL:
                    {
                        v.AddOp2(OP.Null, 0, target);
                        break;
                    }

#if !OMIT_BLOB_LITERAL
                case TK.BLOB:
                    {
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        Debug.Assert(expr.u.Token[0] == 'x' || expr.u.Token[0] == 'X');
                        Debug.Assert(expr.u.Token[1] == '\'');
                        string z = expr.u.Token.Substring(2);
                        int n = z.Length - 1;
                        Debug.Assert(z[n] == '\'');
                        byte[] blob = ConvertEx.HexToBlob(v->Db, z, n);
                        v.AddOp4(OP.Blob, n / 2, target, 0, blob, Vdbe.P4T.DYNAMIC);
                        break;
                    }
#endif

                case TK.VARIABLE:
                    {
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        Debug.Assert(expr.u.Token != null);
                        Debug.Assert(expr.u.Token.Length != 0);
                        v.AddOp2(OP.Variable, expr.ColumnIdx, target);
                        if (expr.u.Token.Length > 1)
                        {
                            Debug.Assert(expr.u.Token[0] == '?' || string.Equals(expr.u.Token, parse.Vars[expr.ColumnIdx - 1], StringComparison.OrdinalIgnoreCase));
                            v.ChangeP4(-1, parse.Vars[expr.ColumnIdx - 1], Vdbe.P4T.STATIC);
                        }
                        break;
                    }

                case TK.REGISTER:
                    {
                        inReg = expr.TableId;
                        break;
                    }

                case TK.AS:
                    {
                        inReg = CodeTarget(parse, expr.Left, target);
                        break;
                    }

#if !OMIT_CAST
                case TK.CAST:
                    {
                        // Expressions of the form:   CAST(pLeft AS token)
                        inReg = CodeTarget(parse, expr.Left, target);
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        AFF aff = AffinityType(expr.u.Token);
                        int toOP = (int)aff - AFF.TEXT + OP.ToText;
                        Debug.Assert(toOP == OP.ToText || aff != AFF.TEXT);
                        Debug.Assert(toOP == OP.ToBlob || aff != AFF.NONE);
                        Debug.Assert(toOP == OP.ToNumeric || aff != AFF.NUMERIC);
                        Debug.Assert(toOP == OP.ToInt || aff != AFF.INTEGER);
                        Debug.Assert(toOP == OP.ToReal || aff != AFF.REAL);
                        C.ASSERTCOVERAGE(toOP == OP.ToText);
                        C.ASSERTCOVERAGE(toOP == OP.ToBlob);
                        C.ASSERTCOVERAGE(toOP == OP.ToNumeric);
                        C.ASSERTCOVERAGE(toOP == OP.ToInt);
                        C.ASSERTCOVERAGE(toOP == OP.ToReal);
                        if (inReg != target)
                        {
                            v.AddOp2(OP.SCopy, inReg, target);
                            inReg = target;
                        }
                        v.AddOp1(v, toOP, inReg);
                        C.ASSERTCOVERAGE(UsedAsColumnCache(parse, inReg, inReg));
                        CacheAffinityChange(parse, inReg, 1);
                        break;
                    }
#endif

                case TK.LT:
                case TK.LE:
                case TK.GT:
                case TK.GE:
                case TK.NE:
                case TK.EQ:
                    {
                        Debug.Assert(TK.LT == OP.Lt);
                        Debug.Assert(TK.LE == OP.Le);
                        Debug.Assert(TK.GT == OP.Gt);
                        Debug.Assert(TK.GE == OP.Ge);
                        Debug.Assert(TK.EQ == OP.Eq);
                        Debug.Assert(TK.NE == OP.Ne);
                        C.ASSERTCOVERAGE(op == TK.LT);
                        C.ASSERTCOVERAGE(op == TK.LE);
                        C.ASSERTCOVERAGE(op == TK.GT);
                        C.ASSERTCOVERAGE(op == TK.GE);
                        C.ASSERTCOVERAGE(op == TK.EQ);
                        C.ASSERTCOVERAGE(op == TK.NE);
                        r1 = CodeTemp(parse, expr.Left, ref regFree1);
                        r2 = CodeTemp(parse, expr.Right, ref regFree2);
                        CodeCompare(parse, expr.Left, expr.Right, op, r1, r2, inReg, SQLITE_STOREP2);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }

                case TK.IS:
                case TK.ISNOT:
                    {
                        C.ASSERTCOVERAGE(op == TK_IS);
                        C.ASSERTCOVERAGE(op == TK_ISNOT);
                        r1 = CodeTemp(parse, expr.Left, ref regFree1);
                        r2 = CodeTemp(parse, expr.Right, ref regFree2);
                        op = (op == TK_IS ? TK_EQ : TK_NE);
                        CodeCompare(parse, expr.Left, expr.Right, op, r1, r2, inReg, SQLITE_STOREP2 | SQLITE_NULLEQ);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }

                case TK.AND:
                case TK.OR:
                case TK.PLUS:
                case TK.STAR:
                case TK.MINUS:
                case TK.REM:
                case TK.BITAND:
                case TK.BITOR:
                case TK.SLASH:
                case TK.LSHIFT:
                case TK.RSHIFT:
                case TK.CONCAT:
                    {
                        Debug.Assert(TK.AND == OP.And);
                        Debug.Assert(TK.OR == OP.Or);
                        Debug.Assert(TK.PLUS == OP.Add);
                        Debug.Assert(TK.MINUS == OP.Subtract);
                        Debug.Assert(TK.REM == OP.Remainder);
                        Debug.Assert(TK.BITAND == OP.BitAnd);
                        Debug.Assert(TK.BITOR == OP.BitOr);
                        Debug.Assert(TK.SLASH == OP.Divide);
                        Debug.Assert(TK.LSHIFT == OP.ShiftLeft);
                        Debug.Assert(TK.RSHIFT == OP.ShiftRight);
                        Debug.Assert(TK.CONCAT == OP.Concat);
                        C.ASSERTCOVERAGE(op == TK.AND);
                        C.ASSERTCOVERAGE(op == TK.OR);
                        C.ASSERTCOVERAGE(op == TK.PLUS);
                        C.ASSERTCOVERAGE(op == TK.MINUS);
                        C.ASSERTCOVERAGE(op == TK.REM);
                        C.ASSERTCOVERAGE(op == TK.BITAND);
                        C.ASSERTCOVERAGE(op == TK.BITOR);
                        C.ASSERTCOVERAGE(op == TK.SLASH);
                        C.ASSERTCOVERAGE(op == TK.LSHIFT);
                        C.ASSERTCOVERAGE(op == TK.RSHIFT);
                        C.ASSERTCOVERAGE(op == TK.CONCAT);
                        r1 = CodeTemp(parse, expr.Left, ref regFree1);
                        r2 = CodeTemp(parse, expr.Right, ref regFree2);
                        v.AddOp3(op, r2, r1, target);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }

                case TK.UMINUS:
                    {
                        Expr left = expr.Left;
                        Debug.Assert(left != null);
                        if (left.OP == TK.INTEGER)
                        {
                            CodeInteger(parse, left, true, target);
#if !OMIT_FLOATING_POINT
                        }
                        else if (left.OP == TK.FLOAT)
                        {
                            Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                            CodeReal(v, left.u.Token, true, target);
#endif
                        }
                        else
                        {
                            regFree1 = r1 = GetTempReg(parse);
                            v.AddOp2(OP.Integer, 0, r1);
                            r2 = CodeTemp(parse, expr.Left, ref regFree2);
                            v.AddOp3(OP.Subtract, r2, r1, target);
                            C.ASSERTCOVERAGE(regFree2 == 0);
                        }
                        inReg = target;
                        break;
                    }

                case TK.BITNOT:
                case TK.NOT:
                    {
                        Debug.Assert(TK.BITNOT == OP.BitNot);
                        Debug.Assert(TK.NOT == OP.Not);
                        C.ASSERTCOVERAGE(op == TK.BITNOT);
                        C.ASSERTCOVERAGE(op == TK.NOT);
                        r1 = CodeTemp(parse, expr.Left, ref regFree1);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        inReg = target;
                        v.AddOp2(op, r1, inReg);
                        break;
                    }

                case TK.ISNULL:
                case TK.NOTNULL:
                    {
                        Debug.Assert(TK.ISNULL == OP.IsNull);
                        Debug.Assert(TK.NOTNULL == OP.NotNull);
                        C.ASSERTCOVERAGE(op == TK.ISNULL);
                        C.ASSERTCOVERAGE(op == TK.NOTNULL);
                        v.AddOp2(OP.Integer, 1, target);
                        r1 = CodeTemp(parse, expr.Left, ref regFree1);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        int addr = v.AddOp1(op, r1);
                        v.AddOp2(OP.AddImm, target, -1);
                        v.JumpHere(addr);
                        break;
                    }

                case TK.AGG_FUNCTION:
                    {
                        AggInfo info = expr.AggInfo;
                        if (info == null)
                        {
                            Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                            parse.ErrorMsg("misuse of aggregate: %s()", expr.u.Token);
                        }
                        else
                            inReg = info.Funcs[expr.Agg].Mem;
                        break;
                    }

                case TK.CONST_FUNC:
                case TK.FUNCTION:
                    {
                        Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect));
                        C.ASSERTCOVERAGE(op == TK.CONST_FUNC);
                        C.ASSERTCOVERAGE(op == TK.FUNCTION);
                        ExprList farg = (E.ExprHasAnyProperty(expr, EP.TokenOnly) ? farg = null : expr.x.List); // List of function arguments
                        int fargLength = (farg != null ? farg.Exprs : 0); // Number of function arguments
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        string id = expr.u.Token; // The function name
                        int idLength = id.Length; // Length of the function name in bytes
                        TEXTENCODE encode = E.CTXENCODE(ctx); // The text encoding used by this database
                        FuncDef def = FindFunction(ctx, id, idLength, fargLength, encode, false); // The function definition object
                        if (def == null)
                        {
                            parse.ErrorMsg("unknown function: %.*s()", idLength, id);
                            break;
                        }

                        // Attempt a direct implementation of the built-in COALESCE() and IFNULL() functions.  This avoids unnecessary evalation of
                        // arguments past the first non-NULL argument.
                        int i;
                        if ((def.Flags & FUNC.COALESCE) != 0)
                        {
                            int endCoalesce = v.MakeLabel();
                            Debug.Assert(fargLength >= 2);
                            Code(parse, farg.Ids[0].Expr, target);
                            for (i = 1; i < fargLength; i++)
                            {
                                v.AddOp2(OP.NotNull, target, endCoalesce);
                                CacheRemove(parse, target, 1);
                                CachePush(parse);
                                Code(parse, farg.Ids[i].Expr, target);
                                CachePop(parse, 1);
                            }
                            v.ResolveLabel(endCoalesce);
                            break;
                        }

                        if (farg != null)
                        {
                            r1 = GetTempRange(parse, fargLength);
                            CachePush(parse);
                            CodeExprList(parse, farg, r1, true);
                            CachePop(parse, 1);
                        }
                        else
                            r1 = 0;
#if !OMIT_VIRTUALTABLE
                        // Possibly overload the function if the first argument is a virtual table column.
                        //
                        // For infix functions (LIKE, GLOB, REGEXP, and MATCH) use the second argument, not the first, as the argument to test to
                        // see if it is a column in a virtual table.  This is done because the left operand of infix functions (the operand we want to
                        // control overloading) ends up as the second argument to the function.  The expression "A glob B" is equivalent to 
                        // "glob(B,A).  We want to use the A in "A glob B" to test for function overloading.  But we use the B term in "glob(B,A)".
                        if (fargLength >= 2 && (expr.Flags & EP.InfixFunc) != 0)
                            def = VTable.OverloadFunction(ctx, def, fargLength, farg.Ids[1].Expr);
                        else if (fargLength > 0)
                            def = VTable.OverloadFunction(ctx, def, fargLength, farg.Ids[0].Expr);
#endif

                        int constMask = 0; // Mask of function arguments that are constant
                        CollSeq coll = null; // A collating sequence
                        for (i = 0; i < fargLength; i++)
                        {
                            if (i < 32 && farg.Ids[i].Expr.IsConstant())
                                constMask |= (1 << i);
                            if ((def.Flags & FUNC.NEEDCOLL) != 0 && coll == null)
                                coll = farg.Ids[i].Expr.CollSeq(parse);
                        }
                        if ((def.Flags & FUNC.NEEDCOLL) != 0)
                        {
                            if (coll == null)
                                coll = ctx.DefaultColl;
                            v.AddOp4(OP.CollSeq, 0, 0, 0, coll, Vdbe.P4T.COLLSEQ);
                        }
                        v.AddOp4(OP.Function, constMask, r1, target, def, Vdbe.P4T.FUNCDEF);
                        v.ChangeP5((byte)fargLength);
                        if (fargLength != 0)
                            ReleaseTempRange(parse, r1, fargLength);
                        break;
                    }

#if !OMIT_SUBQUERY
                case TK.EXISTS:
                case TK.SELECT:
                    {
                        C.ASSERTCOVERAGE(op == TK.EXISTS);
                        C.ASSERTCOVERAGE(op == TK.SELECT);
                        inReg = CodeSubselect(parse, expr, 0, false);
                        break;
                    }

                case TK.IN:
                    {
                        int destIfFalse = v.MakeLabel();
                        int destIfNull = v.MakeLabel();
                        v.AddOp2(OP.Null, 0, target);
                        CodeIN(parse, expr, destIfFalse, destIfNull);
                        v.AddOp2(OP.Integer, 1, target);
                        v.ResolveLabel(destIfFalse);
                        v.AddOp2(OP.AddImm, target, 0);
                        v.ResolveLabel(destIfNull);
                        break;
                    }
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
                case TK.BETWEEN:
                    {
                        Expr left = expr.Left;
                        ExprList.ExprListItem item = expr.x.List.Ids[0];
                        Expr right = item.Expr;
                        r1 = CodeTemp(parse, left, ref regFree1);
                        r2 = CodeTemp(parse, right, ref regFree2);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        r3 = GetTempReg(parse);
                        r4 = GetTempReg(parse);
                        CodeCompare(parse, left, right, OP.Ge, r1, r2, r3, SQLITE_STOREP2);
                        item = expr.x.List.a[1]; //: item++;
                        right = item.Expr;
                        ReleaseTempReg(parse, regFree2);
                        r2 = CodeTemp(parse, right, ref regFree2);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        CodeCompare(parse, left, right, OP.Le, r1, r2, r4, SQLITE_STOREP2);
                        v.AddOp3(OP.And, r3, r4, target);
                        ReleaseTempReg(parse, r3);
                        ReleaseTempReg(parse, r4);
                        break;
                    }

                case TK.COLLATE:
                case TK.UPLUS:
                    {
                        inReg = CodeTarget(parse, expr.Left, target);
                        break;
                    }

                case TK.TRIGGER:
                    {
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
                        Table table = expr.Table;
                        int p1 = expr.TableId * (table.Cols.length + 1) + 1 + expr.ColumnIdx;
                        Debug.Assert(expr.TableId == 0 || expr.TableId == 1);
                        Debug.Assert(expr.ColumnIdx >= -1 && expr.ColumnIdx < table.Cols.length);
                        Debug.Assert(table.PKey < 0 || expr.ColumnIdx != table.PKey);
                        Debug.Assert(p1 >= 0 && p1 < (table.Cols.length * 2 + 2)); //? Is this suppose to be different

                        v.AddOp2(OP.Param, p1, target);
                        VdbeComment(v, "%s.%s -> $%d", (expr.TableId != 0 ? "new" : "old"), (expr.ColumnIdx < 0 ? "rowid" : expr.Table.Cols[expr.ColumnIdx].Name), target);

#if !OMIT_FLOATING_POINT
                        // If the column has REAL affinity, it may currently be stored as an integer. Use OP_RealAffinity to make sure it is really real.
                        if (expr.ColumnIdx >= 0 && table.Cols[expr.ColumnIdx].Affinity == AFF.REAL)
                            v.AddOp1(OP.RealAffinity, target);
#endif
                        break;
                    }

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
                default:
                    Debug.Assert(op == TK_CASE);
                    {
#if !NDEBUG
                        int cacheLevel = parse.CacheLevel;
#endif
                        Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect) && expr.x.List != null);
                        Debug.Assert((expr.x.List.Exprs % 2) == 0);
                        Debug.Assert(expr.x.List.Exprs > 0);
                        ExprList list = expr.x.List; // List of WHEN terms
                        ExprList.ExprListItem elems = list.Ids; // Array of WHEN terms
                        int exprs = list.Exprs; // 2x number of WHEN terms
                        int endLabel = v.MakeLabel(); // GOTO label for end of CASE stmt
                        Expr x; // The X expression
                        Expr opCompare = new Expr(); // The X==Ei expression
                        Expr test = null; // X==Ei (form A) or just Ei (form B)
                        if ((x = expr.Left) != null)
                        {
                            Expr cacheX = x; // Cached expression X
                            C.ASSERTCOVERAGE(x.OP == TK.COLUMN);
                            C.ASSERTCOVERAGE(x.OP == TK.REGISTER);
                            cacheX.Table = CodeTemp(parse, x, ref regFree1);
                            C.ASSERTCOVERAGE(regFree1 == 0);
                            cacheX.OP = TK.REGISTER;
                            opCompare.OP = TK.EQ;
                            opCompare.Left = cacheX;
                            test = opCompare;
                            // Ticket b351d95f9cd5ef17e9d9dbae18f5ca8611190001: The value in regFree1 might get SCopy-ed into the file result.
                            // So make sure that the regFree1 register is not reused for other purposes and possibly overwritten.
                            regFree1 = 0;
                        }
                        for (int i = 0; i < exprs; i += 2)
                        {
                            CachePush(parse);
                            if (x != null)
                            {
                                Debug.Assert(test != null);
                                opCompare.Right = elems[i].Expr;
                            }
                            else
                                test = elems[i].Expr;
                            int nextCase = v.MakeLabel(); // GOTO label for next WHEN clause
                            C.ASSERTCOVERAGE(test.OP == TK.COLUMN);
                            IfFalse(parse, test, nextCase, SQLITE_JUMPIFNULL);
                            C.ASSERTCOVERAGE(elems[i + 1].Expr.OP == TK.COLUMN);
                            C.ASSERTCOVERAGE(elems[i + 1].Expr.OP == TK.REGISTER);
                            Code(parse, elems[i + 1].Expr, target);
                            v.AddOp2(OP.Goto, 0, endLabel);
                            CachePop(parse, 1);
                            v.ResolveLabel(nextCase);
                        }
                        if (expr.Right != null)
                        {
                            CachePush(parse);
                            Code(parse, expr.Right, target);
                            CachePop(parse, 1);
                        }
                        else
                            v.AddOp2(OP.Null, 0, target);
                        Debug.Assert(ctx.MallocFailed || parse.Errs > 0 || parse.CacheLevel == cacheLevel);
                        sqlite3VdbeResolveLabel(v, endLabel);
                        break;
                    }

#if !OMIT_TRIGGER
                case TK_RAISE:
                    {
                        Debug.Assert(expr.Affinity == OE.Rollback || expr.Affinity == OE.Abort || expr.Affinity == OE.Fail || expr.Affinity == OE.Ignore);
                        if (parse.TriggerTab == null)
                        {
                            parse.ErrorMsg("RAISE() may only be used within a trigger-program");
                            return 0;
                        }
                        if (expr.Affinity == OE.Abort)
                            parse.MayAbort();
                        Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                        if (expr.Affinity == OE.Ignore)
                            v.AddOp4(OP.Halt, RC.OK, OE.Ignore, 0, expr.u.Token, 0);
                        else
                            HaltConstraint(parse, CONSTRAINT_TRIGGER, expr.Affinity, expr.u.Token, 0);
                        break;
                    }
#endif
            }
            ReleaseTempReg(parse, regFree1);
            ReleaseTempReg(parse, regFree2);
            return inReg;
        }

        public static int CodeTemp(Parse parse, Expr expr, ref int reg)
        {
            int r1 = GetTempReg(parse);
            int r2 = CodeTarget(parse, expr, r1);
            if (r2 == r1)
                reg = r1;
            else
            {
                ReleaseTempReg(parse, r1);
                reg = 0;
            }
            return r2;
        }

        public static int Code(Parse parse, Expr expr, int target)
        {
            Debug.Assert(target > 0 && target <= parse.Mems);
            if (expr != null && expr.OP == TK.REGISTER)
                parse.V.AddOp2(OP.Copy, expr.TableId, target);
            else
            {
                int inReg = CodeTarget(parse, expr, target);
                Debug.Assert(parse.V != null || parse.Ctx.MallocFailed);
                if (inReg != target && parse.V != null)
                    parse.V.AddOp2(OP.SCopy, inReg, target);
            }
            return target;
        }

        public static int CodeAndCache(Parse parse, Expr expr, int target)
        {
            Vdbe v = parse.V;
            int inReg = Code(parse, expr, target);
            Debug.Assert(target > 0);
            // This routine is called for terms to INSERT or UPDATE.  And the only other place where expressions can be converted into TK_REGISTER is
            // in WHERE clause processing.  So as currently implemented, there is no way for a TK_REGISTER to exist here.  But it seems prudent to
            // keep the ALWAYS() in case the conditions above change with future modifications or enhancements. */
            if (C._ALWAYS(expr.OP != TK.REGISTER))
            {
                int mem = ++parse.Mems;
                v.AddOp2(OP.Copy, inReg, mem);
                expr.Table = mem;
                expr.OP2 = expr.OP;
                expr.OP = TK.REGISTER;
            }
            return inReg;
        }

        #region Explain
#if ENABLE_TREE_EXPLAIN
        public static void ExplainExpr(Vdbe v, Expr expr)
        {
            string binOp = null;   // Binary operator
            string uniOp = null;   // Unary operator
            TK op = (expr == null ? TK.NULL : expr.OP); // The opcode being coded
            switch (op)
            {
                case TK.AGG_COLUMN:
                    {
                        Vdbe.ExplainPrintf(v, "AGG{%d:%d}", expr.TableId, expr.ColumnIdx);
                        break;
                    }

                case TK.COLUMN:
                    {
                        if (expr.TableId < 0) // This only happens when coding check constraints
                            Vdbe.ExplainPrintf(v, "COLUMN(%d)", expr.ColumnIdx);
                        else
                            Vdbe.ExplainPrintf(v, "{%d:%d}", expr.Table, expr.ColumnIdx);
                        break;
                    }

                case TK.INTEGER:
                    {
                        if ((expr.Flags & EP.IntValue) != 0)
                            Vdbe.ExplainPrintf(v, "%d", expr.u.I);
                        else
                            Vdbe.ExplainPrintf(v, "%s", expr.u.Token);
                        break;
                    }

#if !OMIT_FLOATING_POINT
                case TK.FLOAT:
                    {
                        Vdbe.ExplainPrintf(v, "%s", expr.u.Token);
                        break;
                    }
#endif
                case TK.STRING:
                    {
                        Vdbe.ExplainPrintf(v, "%Q", expr.u.Token);
                        break;
                    }

                case TK.NULL:
                    {
                        Vdbe.ExplainPrintf(v, "NULL");
                        break;
                    }

#if !OMIT_BLOB_LITERAL
                case TK.BLOB:
                    {
                        Vdbe.ExplainPrintf(v, "%s", expr.u.Token);
                        break;
                    }
#endif
                case TK.VARIABLE:
                    {
                        Vdbe.ExplainPrintf(v, "VARIABLE(%s,%d)", expr.u.Token, expr.ColumnIdx);
                        break;
                    }

                case TK.REGISTER:
                    {
                        Vdbe.ExplainPrintf(v, "REGISTER(%d)", expr.TableId);
                        break;
                    }

                case TK.AS:
                    {
                        ExplainExpr(v, expr.Left);
                        break;
                    }

#if !OMIT_CAST
                case TK.CAST:
                    {
                        // Expressions of the form:   CAST(pLeft AS token)
                        string aff = "unk";
                        switch (Parse.AffinityType(expr.u.Token))
                        {
                            case AFF.TEXT: aff = "TEXT"; break;
                            case AFF.NONE: aff = "NONE"; break;
                            case AFF.NUMERIC: aff = "NUMERIC"; break;
                            case AFF.INTEGER: aff = "INTEGER"; break;
                            case AFF.REAL: aff = "REAL"; break;
                        }
                        Vdbe.ExplainPrintf(v, "CAST-%s(", aff);
                        ExplainExpr(v, expr.Left);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }
#endif

                case TK.LT: binOp = "LT"; break;
                case TK.LE: binOp = "LE"; break;
                case TK.GT: binOp = "GT"; break;
                case TK.GE: binOp = "GE"; break;
                case TK.NE: binOp = "NE"; break;
                case TK.EQ: binOp = "EQ"; break;
                case TK.IS: binOp = "IS"; break;
                case TK.ISNOT: binOp = "ISNOT"; break;
                case TK.AND: binOp = "AND"; break;
                case TK.OR: binOp = "OR"; break;
                case TK.PLUS: binOp = "ADD"; break;
                case TK.STAR: binOp = "MUL"; break;
                case TK.MINUS: binOp = "SUB"; break;
                case TK.REM: binOp = "REM"; break;
                case TK.BITAND: binOp = "BITAND"; break;
                case TK.BITOR: binOp = "BITOR"; break;
                case TK.SLASH: binOp = "DIV"; break;
                case TK.LSHIFT: binOp = "LSHIFT"; break;
                case TK.RSHIFT: binOp = "RSHIFT"; break;
                case TK.CONCAT: binOp = "CONCAT"; break;

                case TK.UMINUS: uniOp = "UMINUS"; break;
                case TK.UPLUS: uniOp = "UPLUS"; break;
                case TK.BITNOT: uniOp = "BITNOT"; break;
                case TK.NOT: uniOp = "NOT"; break;
                case TK.ISNULL: uniOp = "ISNULL"; break;
                case TK.NOTNULL: uniOp = "NOTNULL"; break;

                case TK.COLLATE:
                    {
                        ExplainExpr(v, expr.Left);
                        Vdbe.ExplainPrintf(v, ".COLLATE(%s)", expr.u.Token);
                        break;
                    }

                case TK.AGG_FUNCTION:
                case TK.CONST_FUNC:
                case TK.FUNCTION:
                    {
                        ExprList farg = (E.ExprHasAnyProperty(expr, EP.TokenOnly) ? null : expr.x.List); // List of function arguments
                        if (op == TK.AGG_FUNCTION)
                            Vdbe.ExplainPrintf(v, "AGG_FUNCTION%d:%s(", expr.OP2, expr.u.Token);
                        else
                            Vdbe.ExplainPrintf(v, "FUNCTION:%s(", expr.u.Token);
                        if (farg)
                            ExplainExprList(v, farg);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }

#if !OMIT_SUBQUERY
                case TK.EXISTS:
                    {
                        Vdbe.ExplainPrintf(v, "EXISTS(");
                        Select.ExplainSelect(v, expr.x.Select);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }

                case TK.SELECT:
                    {
                        Vdbe.ExplainPrintf(v, "(");
                        Select.ExplainSelect(v, expr.x.Select);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }

                case TK.IN:
                    {
                        Vdbe.ExplainPrintf(v, "IN(");
                        ExplainExpr(v, expr.Left);
                        Vdbe.ExplainPrintf(v, ",");
                        if (E.ExprHasProperty(expr, EP.xIsSelect))
                            Select.ExplainSelect(v, expr.x.Select);
                        else
                            ExplainExprList(v, expr.x.List);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }
#endif

                //    x BETWEEN y AND z
                // This is equivalent to
                //    x>=y AND x<=z
                // X is stored in pExpr->pLeft.
                // Y is stored in pExpr->pList->a[0].pExpr.
                // Z is stored in pExpr->pList->a[1].pExpr.
                case TK.BETWEEN:
                    {
                        Expr x = expr.Left;
                        Expr y = expr.x.List.Ids[0].Expr;
                        Expr z = expr.x.List.Ids[1].Expr;
                        Vdbe.ExplainPrintf(v, "BETWEEN(");
                        ExplainExpr(v, x);
                        Vdbe.ExplainPrintf(v, ",");
                        ExplainExpr(v, u);
                        Vdbe.ExplainPrintf(v, ",");
                        ExplainExpr(v, z);
                        Vdbe.ExplainPrintf(v, ")");
                        break;
                    }

                case TK.TRIGGER:
                    {
                        // If the opcode is TK_TRIGGER, then the expression is a reference to a column in the new.* or old.* pseudo-tables available to
                        // trigger programs. In this case Expr.iTable is set to 1 for the new.* pseudo-table, or 0 for the old.* pseudo-table. Expr.iColumn
                        // is set to the column of the pseudo-table to read, or to -1 to read the rowid field.
                        Vdbe.ExplainPrintf(v, "%s(%d)", (expr.TableId ? "NEW" : "OLD"), expr.ColumnIdx);
                        break;
                    }

                case TK.CASE:
                    {
                        Vdbe.ExplainPrintf(v, "CASE(");
                        ExplainExpr(v, expr.Left);
                        Vdbe.ExplainPrintf(v, ",");
                        ExplainExprList(v, expr.x.List);
                        break;
                    }

#if !OMIT_TRIGGER
                case TK.RAISE:
                    {
                        string type = "unk";
                        switch (expr.Affinity)
                        {
                            case OE.Rollback: type = "rollback"; break;
                            case OE.Abort: type = "abort"; break;
                            case OE.Fail: type = "fail"; break;
                            case OE.Ignore: type = "ignore"; break;
                        }
                        Vdbe.ExplainPrintf(v, "RAISE-%s(%s)", type, expr.u.Token);
                        break;
                    }
#endif
            }

            if (binOp != null)
            {
                Vdbe.ExplainPrintf(v, "%s(", binOp);
                ExplainExpr(v, expr.Left);
                Vdbe.ExplainPrintf(v, ",");
                ExplainExpr(v, expr.Right);
                Vdbe.ExplainPrintf(v, ")");
            }
            else if (uniOp != null)
            {
                Vdbe.ExplainPrintf(v, "%s(", uniOp);
                ExplainExpr(v, expr.Left);
                Vdbe.ExplainPrintf(v, ")");
            }
        }

        public static void ExplainExprList(Vdbe v, ExprList list)
        {
            if (list == null || list.Exprs == 0)
            {
                Vdbe.ExplainPrintf(v, "(empty-list)");
                return;
            }
            else if (list.Exprs == 1)
                ExplainExpr(v, list.Ids[0].Expr);
            else
            {
                Vdbe.ExplainPush(v);
                for (int i = 0; i < list.Exprs; i++)
                {
                    Vdbe.ExplainPrintf(v, "item[%d] = ", i);
                    Vdbe.ExplainPush(v);
                    ExplainExpr(v, list.Ids[i].Expr);
                    Vdbe.ExplainPop(v);
                    if (list.Ids[i].Name)
                        Vdbe.ExplainPrintf(v, " AS %s", list.Ids[i].Name);
                    if (list.Ids[i].SpanIsTab)
                        Vdbe.ExplainPrintf(v, " (%s)", list.Ids[i].Span);
                    if (i < list.Exprs - 1)
                        Vdbe.ExplainNL(v);
                }
                Vdbe.ExplainPop(v);
            }
        }
#endif
        #endregion

        static bool IsAppropriateForFactoring(Expr expr)
        {
            if (!expr.IsConstantNotJoin())
                return false; // Only constant expressions are appropriate for factoring
            if ((expr.Flags & EP.FixedDest) == 0)
                return true;  // Any constant without a fixed destination is appropriate
            while (expr.OP == TK.UPLUS) expr = expr.Left;
            switch (expr.OP)
            {
#if !OMIT_BLOB_LITERAL
                case TK.BLOB:
#endif
                case TK.VARIABLE:
                case TK.INTEGER:
                case TK.FLOAT:
                case TK.NULL:
                case TK.STRING:
                    {
                        C.ASSERTCOVERAGE(expr.OP == TK.BLOB);
                        C.ASSERTCOVERAGE(expr.OP == TK.VARIABLE);
                        C.ASSERTCOVERAGE(expr.OP == TK.INTEGER);
                        C.ASSERTCOVERAGE(expr.OP == TK.FLOAT);
                        C.ASSERTCOVERAGE(expr.OP == TK.NULL);
                        C.ASSERTCOVERAGE(expr.OP == TK.STRING);
                        // Single-instruction constants with a fixed destination are better done in-line.  If we factor them, they will just end
                        // up generating an OP_SCopy to move the value to the destination register.
                        return false;
                    }

                case TK.UMINUS:
                    {
                        if (expr.Left.OP == TK.FLOAT || expr.Left.OP == TK.INTEGER)
                            return false;
                        break;
                    }
            }
            return true;
        }

        static WRC EvalConstExpr(Walker walker, ref Expr expr)
        {
            Parse parse = walker.Parse;
            switch (expr.OP)
            {
                case TK.IN:
                case TK.REGISTER:
                    {
                        return WRC.Prune;
                    }
                case TK.COLLATE:
                    {
                        return WRC.Continue;
                    }
                case TK.FUNCTION:
                case TK.AGG_FUNCTION:
                case TK.CONST_FUNC:
                    {
                        // The arguments to a function have a fixed destination. Mark them this way to avoid generated unneeded OP_SCopy
                        // instructions. 
                        ExprList list = expr.x.pList;
                        Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect));
                        if (list != null)
                        {
                            int i = list.Exprs;
                            ExprList.ExprListItem item; ;
                            for (i = list.Exprs, item = list.Ids[0]; i > 0; i--, item = list.Ids[list.Exprs - i])
                                if (C._ALWAYS(item.Expr != null))
                                    item.Expr.Flags |= EP.FixedDest;
                        }
                        break;
                    }
            }
            if (IsAppropriateForFactoring(expr))
            {
                int r1 = ++parse.Mems;
                int r2 = CodeTarget(parse, expr, r1);
                // If r2!=r1, it means that register r1 is never used.  That is harmless but suboptimal, so we want to know about the situation to fix it.
                // Hence the following assert:
                Debug.Assert(r2 == r1);
                expr.OP2 = expr.OP;
                expr.OP = TK.REGISTER;
                expr.TableId = r2;
                return WRC.Prune;
            }
            return WRC.Continue;
        }

        public static void CodeConstants(Parse parse, Expr expr)
        {
            if (parse.CookieGoto != 0 || E.CtxOptimizationDisabled(parse.Ctx, OPTFLAG.FactorOutConst))
                return;
            Walker w = new Walker();
            w.ExprCallback = EvalConstExpr;
            w.SelectCallback = null;
            w.Parse = parse;
            WalkExpr(w, expr);
        }

        public static int CodeExprList(Parse parse, ExprList list, int target, bool doHardCopy)
        {
            Debug.Assert(list != null);
            Debug.Assert(target > 0);
            Debug.Assert(parse.V != null); // Never gets this far otherwise
            int n = list.Exprs;
            int i;
            ExprList.ExprListItem item;
            for (item = list.Ids[0], i = 0; i < n; i++, item = list.Ids[i])
            {
                Expr expr = item.Expr;
                int inReg = CodeTarget(parse, expr, target + i);
                if (inReg != target + i)
                    parse.V.AddOp2((doHardCopy ? OP.Copy : OP.SCopy), inReg, target + i);
            }
            return n;
        }

        static void ExprCodeBetween(Parse parse, Expr expr, int dest, bool jumpIfTrue, int jumpIfNull)
        {
            Debug.Assert(!ExprHasProperty(expr, EP_xIsSelect));
            Expr exprX = expr.Left.memcpy(); // The  x  subexpression
            Expr compLeft = new Expr();  // The  x>=y  term
            Expr compRight = new Expr(); /// The  x<=z  term
            Expr exprAnd = new Expr(); // The AND operator in  x>=y AND x<=z
            exprAnd.OP = TK.AND;
            exprAnd.Left = compLeft;
            exprAnd.Right = compRight;
            compLeft.OP = TK.GE;
            compLeft.Left = exprX;
            compLeft.Right = expr.x.List.Ids[0].Expr;
            compRight.OP = TK.LE;
            compRight.Left = exprX;
            compRight.Right = expr.x.List.Ids[1].Expr;
            int regFree1 = 0; // Temporary use register
            exprX.TableId = CodeTemp(parse, exprX, ref regFree1);
            exprX.OP = TK.REGISTER;
            if (jumpIfTrue)
                IfTrue(parse, exprAnd, dest, jumpIfNull);
            else
                IfFalse(parse, exprAnd, dest, jumpIfNull);
            ReleaseTempReg(parse, regFree1);

            // Ensure adequate test coverage
            C.ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 == 0);
            C.ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull == 0 && regFree1 != 0);
            C.ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 == 0);
            C.ASSERTCOVERAGE(jumpIfTrue == 0 && jumpIfNull != 0 && regFree1 != 0);
            C.ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 == 0);
            C.ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull == 0 && regFree1 != 0);
            C.ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 == 0);
            C.ASSERTCOVERAGE(jumpIfTrue != 0 && jumpIfNull != 0 && regFree1 != 0);
        }

        public void IfTrue(Parse parse, int dest, AFF jumpIfNull)
        {
            Debug.Assert(jumpIfNull == AFF.BIT_JUMPIFNULL || jumpIfNull == 0);
            Vdbe v = parse.V;
            if (C._NEVER(v == null)) return; // Existance of VDBE checked by caller

            int regFree1 = 0;
            int regFree2 = 0;
            int r1, r2;

            TK op = OP;
            switch (op)
            {
                case TK.AND:
                    {
                        int d2 = sqlite3VdbeMakeLabel(v);
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        CachePush(parse);
                        Left.IfFalse(parse, d2, jumpIfNull ^ AFF.BIT_JUMPIFNULL);
                        Right.IfTrue(parse, dest, jumpIfNull);
                        v.ResolveLabel(d2);
                        CachePop(parse, 1);
                        break;
                    }
                case TK.OR:
                    {
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        Left.IfTrue(parse, dest, jumpIfNull);
                        Right.IfTrue(parse, dest, jumpIfNull);
                        break;
                    }
                case TK.NOT:
                    {
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        Left.IfFalse(parse, dest, jumpIfNull);
                        break;
                    }
                case TK.LT:
                case TK.LE:
                case TK.GT:
                case TK.GE:
                case TK.NE:
                case TK.EQ:
                    {
                        Debug.Assert(TK.LT == OP.Lt);
                        Debug.Assert(TK.LE == OP.Le);
                        Debug.Assert(TK.GT == OP.Gt);
                        Debug.Assert(TK.GE == OP.Ge);
                        Debug.Assert(TK.EQ == OP.Eq);
                        Debug.Assert(TK.NE == OP.Ne);
                        C.ASSERTCOVERAGE(op == TK.LT);
                        C.ASSERTCOVERAGE(op == TK.LE);
                        C.ASSERTCOVERAGE(op == TK.GT);
                        C.ASSERTCOVERAGE(op == TK.GE);
                        C.ASSERTCOVERAGE(op == TK.EQ);
                        C.ASSERTCOVERAGE(op == TK.NE);
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        r2 = CodeTemp(parse, Right, ref regFree2);
                        CodeCompare(parse, Left, pRight, op, r1, r2, dest, jumpIfNull);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }
                case TK.IS:
                case TK.ISNOT:
                    {
                        C.ASSERTCOVERAGE(op == TK.IS);
                        C.ASSERTCOVERAGE(op == TK.ISNOT);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        r2 = CodeTemp(parse, Right, ref regFree2);
                        op = (op == TK.IS ? TK.EQ : TK.NE);
                        CodeCompare(parse, Left, Right, op, r1, r2, dest, AFF.BIT_NULLEQ);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }
                case TK.ISNULL:
                case TK.NOTNULL:
                    {
                        Debug.Assert(TK.ISNULL == OP.IsNull);
                        Debug.Assert(TK.NOTNULL == OP.NotNull);
                        C.ASSERTCOVERAGE(op == TK.ISNULL);
                        C.ASSERTCOVERAGE(op == TK.NOTNULL);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        v.AddOp2(op, r1, dest);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        break;
                    }
                case TK.BETWEEN:
                    {
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        ExprCodeBetween(parse, this, dest, 1, jumpIfNull);
                        break;
                    }
#if !OMIT_SUBQUERY
                case TK.IN:
                    {
                        int destIfFalse = v.MakeLabel();
                        int destIfNull = (jumpIfNull != 0 ? dest : destIfFalse);
                        CodeIN(parse, this, destIfFalse, destIfNull);
                        v.AddOp2(OP.Goto, 0, dest);
                        v.ResolveLabel(destIfFalse);
                        break;
                    }
#endif
                default:
                    {
                        r1 = CodeTemp(parse, this, ref regFree1);
                        v.AddOp3(OP.If, r1, dest, jumpIfNull != 0 ? 1 : 0);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        break;
                    }
            }
            ReleaseTempReg(parse, regFree1);
            ReleaseTempReg(parse, regFree2);
        }

        public void IfFalse(Parse parse, int dest, AFF jumpIfNull)
        {
            Vdbe v = parse.V;
            Debug.Assert(jumpIfNull == AFF.BIT_JUMPIFNULL || jumpIfNull == 0);
            if (C._NEVER(v == null)) return; // Existance of VDBE checked by caller

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
            TK op = ((OP + (TK.ISNULL & 1)) ^ 1) - (TK.ISNULL & 1);

            // Verify correct alignment of TK_ and OP_ constants
            Debug.Assert(OP != TK.ISNULL || op == OP.NotNull);
            Debug.Assert(OP != TK.NOTNULL || op == OP.IsNull);
            Debug.Assert(OP != TK.NE || op == OP.Eq);
            Debug.Assert(OP != TK.EQ || op == OP.Ne);
            Debug.Assert(OP != TK.LT || op == OP.Ge);
            Debug.Assert(OP != TK.LE || op == OP.Gt);
            Debug.Assert(OP != TK.GT || op == OP.Le);
            Debug.Assert(OP != TK.GE || op == OP.Lt);

            switch (OP)
            {
                case TK.AND:
                    {
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        Left.IfFalse(parse, dest, jumpIfNull);
                        Right.IfFalse(parse, dest, jumpIfNull);
                        break;
                    }
                case TK.OR:
                    {
                        int d2 = v.MakeLabel();
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        CachePush(parse);
                        Left.IfTrue(parse, d2, jumpIfNull ^ (int)AFF.BIT_JUMPIFNULL);
                        Right.IfFalse(parse, dest, jumpIfNull);
                        v.ResolveLabel(d2);
                        CachePop(parse, 1);
                        break;
                    }
                case TK.NOT:
                    {
                        testcase(jumpIfNull == 0);
                        Left.IfTrue(parse, dest, jumpIfNull);
                        break;
                    }
                case TK.LT:
                case TK.LE:
                case TK.GT:
                case TK.GE:
                case TK.NE:
                case TK.EQ:
                    {
                        C.ASSERTCOVERAGE(op == TK.LT);
                        C.ASSERTCOVERAGE(op == TK.LE);
                        C.ASSERTCOVERAGE(op == TK.GT);
                        C.ASSERTCOVERAGE(op == TK.GE);
                        C.ASSERTCOVERAGE(op == TK.EQ);
                        C.ASSERTCOVERAGE(op == TK.NE);
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        r2 = CodeTemp(parse, Right, ref regFree2);
                        CodeCompare(parse, Left, pRight, op, r1, r2, dest, jumpIfNull);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }
                case TK.IS:
                case TK.ISNOT:
                    {
                        C.ASSERTCOVERAGE(OP == TK.IS);
                        C.ASSERTCOVERAGE(OP == TK.ISNOT);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        r2 = CodeTemp(parse, Right, ref regFree2);
                        op = (OP == TK.IS ? TK.NE : TK.EQ);
                        CodeCompare(parse, Left, Right, op, r1, r2, dest, SQLITE_NULLEQ);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(regFree2 == 0);
                        break;
                    }
                case TK.ISNULL:
                case TK.NOTNULL:
                    {
                        C.ASSERTCOVERAGE(op == TK.ISNULL);
                        C.ASSERTCOVERAGE(op == TK.NOTNULL);
                        r1 = CodeTemp(parse, Left, ref regFree1);
                        v.AddOp2(op, r1, dest);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        break;
                    }
                case TK.BETWEEN:
                    {
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        ExprCodeBetween(parse, this, dest, 0, jumpIfNull);
                        break;
                    }
#if !OMIT_SUBQUERY
                case TK.IN:
                    {
                        if (jumpIfNull != 0)
                            CodeIN(parse, this, dest, dest);
                        else
                        {
                            int destIfNull = v.MakeLabel();
                            CodeIN(parse, this, dest, destIfNull);
                            v.ResolveLabel(destIfNull);
                        }
                        break;
                    }
#endif
                default:
                    {
                        r1 = CodeTemp(parse, this, ref regFree1);
                        v.AddOp3(OP.IfNot, r1, dest, jumpIfNull != 0 ? 1 : 0);
                        C.ASSERTCOVERAGE(regFree1 == 0);
                        C.ASSERTCOVERAGE(jumpIfNull == 0);
                        break;
                    }
            }
            ReleaseTempReg(parse, regFree1);
            ReleaseTempReg(parse, regFree2);
        }

        public static int Compare(Expr a, Expr b)
        {
            if (a == null || b == null)
                return (b == a ? 0 : 2);
            Debug.Assert(!E.ExprHasAnyProperty(a, EP.TokenOnly | EP.Reduced));
            Debug.Assert(!E.ExprHasAnyProperty(b, EP.TokenOnly | EP.Reduced));
            if (E.ExprHasProperty(a, EP.xIsSelect) || E.ExprHasProperty(b, EP.xIsSelect))
                return 2;
            if ((a.Flags & EP.Distinct) != (b.Flags & EP.Distinct)) return 2;
            if (a.OP != b.OP)
            {
                if (a.OP == TK.COLLATE && Compare(a->Left, b) < 2) return 1;
                if (b.OP == TK.COLLATE && Compare(a, b->Left) < 2) return 1;
                return 2;
            }
            if (Compare(a.Left, b.Left) != 0) return 2;
            if (Compare(a.Right, b.Right) != 0) return 2;
            if (ListCompare(a.x.List, b.x.List) != 0) return 2;
            if (a.TableId != b.TableId || a.ColumnIdx != b.ColumnIdx) return 2;
            if (E.ExprHasProperty(a, EP.IntValue))
            {
                if (!E.ExprHasProperty(b, EP.IntValue) || a.u.I != b.u.I) return 2;
            }
            else if (a.OP != TK.COLUMN && a.u.Token != null)
            {
                if (E.ExprHasProperty(b, EP.IntValue) || C._NEVER(b.u.Token == null)) return 2;
                if (!string.Equals(a.u.Token, b.u.Token, StringComparison.OrdinalIgnoreCase))
                    return (a.OP == TK.COLLATE ? 1 : 2);
            }
            return 0;
        }

        public static int ListCompare(ExprList a, ExprList b)
        {
            if (a == null && b == null) return 0;
            if (a == null || b == null) return 1;
            if (a.Exprs != b.Exprs) return 1;
            for (int i = 0; i < a.Exprs; i++)
            {
                Expr exprA = a.Ids[i].Expr;
                Expr exprB = b.Ids[i].Expr;
                if (a.Ids[i].sortOrder != b.Ids[i].SortOrder) return 1;
                if (Compare(exprA, exprB) != 0) return 1;
            }
            return 0;
        }

        private class SrcCount
        {
            SrcList[] Src; // One particular FROM clause in a nested query
            int This; // Number of references to columns in pSrcList
            int Other; // Number of references to columns in other FROM clauses
        }

        static WRC ExprSrcCount(Walker walker, Expr expr)
        {
            // The NEVER() on the second term is because sqlite3FunctionUsesThisSrc() is always called before sqlite3ExprAnalyzeAggregates() and so the
            // TK_COLUMNs have not yet been converted into TK_AGG_COLUMN.  If sqlite3FunctionUsesThisSrc() is used differently in the future, the
            // NEVER() will need to be removed.
            if (expr.OP == TK.COLUMN || C._NEVER(expr.OP == TK.AGG_COLUMN))
            {
                int i;
                SrcCount p = walker.u.SrcCount;
                SrcList src = p.Src;
                for (i = 0; i < src.Srcs; i++)
                    if (expr.TableId == src.Ids[i].Cursor) break;
                if (i < src.Srcs)
                    p.This++;
                else
                    p.Other++;
            }
            return WRC.Continue;
        }

        public bool FunctionUsesThisSrc(SrcList srcList)
        {
            Debug.Assert(OP == TK.AGG_FUNCTION);
            Walker w = new Walker();
            w.ExprCallback = ExprSrcCount;
            SrcCount cnt = new SrcCount();
            w.u.SrcCount = cnt;
            cnt.Src = srcList;
            cnt.This = 0;
            cnt.Other = 0;
            w.WalkExprList(x.List);
            return (cnt.This > 0 || cnt.Other == 0);
        }

        static int AddAggInfoColumn(Context ctx, AggInfo info)
        {
            int i = 0;
            info.aCol = sqlite3ArrayAllocate(ctx, info.aCol, -1, 3, ref info.nColumn, ref info.nColumnAlloc, ref i);
            return i;
        }

        static int AddAggInfoFunc(Context ctx, AggInfo info)
        {
            int i = 0;
            info.aFunc = sqlite3ArrayAllocate(ctx, info.aFunc, -1, 3, ref info.nFunc, ref info.nFuncAlloc, ref i);
            return i;
        }

        static int AnalyzeAggregate(Walker walker, ref Expr expr)
        {
            NameContext nc = walker.u.NC;
            Parse parse = nc.Parse;
            SrcList srcList = nc.SrcList;
            AggInfo aggInfo = nc.AggInfo;
            Context ctx = parse.Ctx;
            switch (expr.OP)
            {
                case TK.AGG_COLUMN:
                case TK.COLUMN:
                    {
                        C.ASSERTCOVERAGE(expr.OP == TK.AGG_COLUMN);
                        C.ASSERTCOVERAGE(expr.OP == TK.COLUMN);
                        // Check to see if the column is in one of the tables in the FROM clause of the aggregate query
                        if (C._ALWAYS(srcList != null))
                        {
                            int i;
                            SrcList.SrcListItem item;
                            for (i = 0, item = srcList.Ids[0]; i < srcList.Srcs; i++, item = srcList.Ids[i])
                            {
                                Debug.Assert(!E.ExprHasAnyProperty(expr, EP.TokenOnly | EP.Reduced));
                                if (expr.TableId == item.Cursor)
                                {
                                    // If we reach this point, it means that pExpr refers to a table that is in the FROM clause of the aggregate query.  
                                    //
                                    // Make an entry for the column in pAggInfo->aCol[] if there is not an entry there already.
                                    int k;
                                    AggInfo.AggInfoColumn col;
                                    for (k = 0, col = aggInfo.Columns.data[0]; k < aggInfo.Columns.length; k++, col = aggInfo.Columns.data[k])
                                        if (col.TableID == expr.TableId && col.Column == expr.ColumnIdx)
                                            break;
                                    if ((k >= aggInfo->Columns.length) && (k = AddAggInfoColumn(ctx, aggInfo)) >= 0)
                                    {
                                        col = aggInfo.Cols[k];
                                        col.Table = expr.Table;
                                        col.TableID = expr.TableId;
                                        col.Column = expr.ColumnIdx;
                                        col.Mem = ++parse.Mems;
                                        col.SorterColumn = -1;
                                        col.Expr = expr;
                                        if (aggInfo.GroupBy != null)
                                        {
                                            ExprList gb = aggInfo.GroupBy;
                                            ExprList.ExprListItem term = gb.Ids[0];
                                            int n = gb.Exprs;
                                            for (int j = 0; j < n; j++, term = gb.Ids[j])
                                            {
                                                Expr e = term.Expr;
                                                if (e.OP == TK.COLUMN && e.TableId == expr.TableId && e.iColumn == expr.iColumn)
                                                {
                                                    col.SorterColumn = j;
                                                    break;
                                                }
                                            }
                                        }
                                        if (col.SorterColumn < 0)
                                            col.SorterColumn = aggInfo.SortingColumns++;
                                    }
                                    // There is now an entry for pExpr in pAggInfo->aCol[] (either because it was there before or because we just created it).
                                    // Convert the pExpr to be a TK_AGG_COLUMN referring to that pAggInfo->aCol[] entry.
                                    E.ExprSetIrreducible(expr);
                                    expr.AggInfo = aggInfo;
                                    expr.OP = TK.AGG_COLUMN;
                                    expr.Agg = (short)k;
                                    break;
                                }
                            }
                        }
                        return WRC.Prune;
                    }

                case TK_AGG_FUNCTION:
                    {
                        if ((nc.NCFlags & NC.InAggFunc) == 0 && walker->WalkerDepth == expr.OP2)
                        {
                            // Check to see if pExpr is a duplicate of another aggregate function that is already in the pAggInfo structure
                            int i;
                            AggInfo.AggInfoFunc item = aggInfo.Funcs[0];
                            for (i = 0; i < aggInfo.Funcs.length; i++, item = aggInfo.Funcs[i])
                                if (Compare(item.Expr, expr) == 0)
                                    break;
                            if (i >= aggInfo->Funcs.length)
                            {
                                // pExpr is original.  Make a new entry in pAggInfo->aFunc[]
                                TEXTENCODE enc = parse.db.aDbStatic[0].pSchema.enc;// ENC(pParse.db);
                                i = AddAggInfoFunc(ctx, aggInfo);
                                if (i >= 0)
                                {
                                    Debug.Assert(!E.ExprHasProperty(expr, EP.xIsSelect));
                                    item = aggInfo.Funcs[i];
                                    item.Expr = expr;
                                    item.Mem = ++parse.Mems;
                                    Debug.Assert(!E.ExprHasProperty(expr, EP.IntValue));
                                    item.Func = Callback.FindFunction(ctx, expr.u.Token, expr.u.Token.Length, (expr.x.List != null ? expr.x.List.Exprs : 0), encode, false);
                                    item.Distinct = ((expr.Flags & EP.Distinct) != 0 ? parse.Tabs++ : -1);
                                }
                            }
                            // Make pExpr point to the appropriate pAggInfo->aFunc[] entry
                            Debug.Assert(!E.ExprHasAnyProperty(expr, EP.TokenOnly | EP.Reduced));
                            E.ExprSetIrreducible(expr);
                            expr.Agg = (short)i;
                            expr.AggInfo = aggInfo;
                            return WRC.Prune;
                        }
                        return WRC.Continue;
                    }
            }
            return WRC.Continue;
        }

        static WRC AnalyzeAggregatesInSelect(Walker walker, Select select)
        {
            return WRC.Continue;
        }

        public static void AnalyzeAggregates(NameContext nc, ref Expr expr)
        {
            Walker w = new Walker();
            w.ExprCallback = AnalyzeAggregate;
            w.SelectCallback = AnalyzeAggregatesInSelect;
            w.u.NC = nc;
            Debug.Assert(nc.SrcList != null);
            WalkExpr(w, ref expr);
        }

        public static void AnalyzeAggList(NameContext nc, ExprList list)
        {
            int i;
            ExprList.ExprListItem item;
            if (list != null)
                for (i = 0, item = list.Ids[0]; i < list.Exprs; i++, item = list.Ids[i])
                    AnalyzeAggregates(nc, ref item.Expr);
        }

        #region Registers

        public static int GetTempReg(Parse parse)
        {
            if (parse.TempReg.length == 0)
                return ++parse.Mems;
            return parse.TempReg[--parse.TempReg.length];
        }

        public static void ReleaseTempReg(Parse parse, int reg)
        {
            if (reg != 0 && parse.TempRegs.length < parse.TempReg.data.Length)
            {
                int i;
                Parse.ColCache p;
                for (i = 0, p = parse.ColCaches[0]; i < N_COLCACHE; i++, p = parse.ColCaches[i])
                {
                    if (p.Reg == reg)
                    {
                        p.TempReg = 1;
                        return;
                    }
                }
                parse.TempReg[parse.TempRegs++] = reg;
            }
        }

        public static int GetTempRange(Parse parse, int regs)
        {
            int i = parse.RangeReg;
            int n = parse.RangeReg;
            if (reg <= n)
            {
                Debug.Assert(!UsedAsColumnCache(parse, i, i + n - 1));
                parse.RangeRegIdx += regs;
                parse.RangeReg -= regs;
            }
            else
            {
                i = parse.Mems + 1;
                parse.Mems += regs;
            }
            return i;
        }

        public static void ReleaseTempRange(Parse parse, int reg, int regs)
        {
            CacheRemove(parse, reg, regs);
            if (regs > parse.RangeRegs)
            {
                parse.RangeRegs = regs;
                parse.RangeRegIdx = reg;
            }
        }

        public static void ClearTempRegCache(Parse parse)
        {
            parse.TempReg.length = 0;
            parse.RangeRegs = 0;
        }

        #endregion

    }
}
