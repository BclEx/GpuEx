using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;

namespace Core
{
    public partial class Walker
    {
        public WRC WalkExpr(Expr expr)
        {
            if (expr == null) return WRC.Continue;
            C.ASSERTCOVERAGE(E.ExprHasProperty(expr, EP.TokenOnly));
            C.ASSERTCOVERAGE(E.ExprHasProperty(expr, EP.Reduced));
            WRC rc = ExprCallback(this, expr);
            if (rc == WRC.Continue && !E.ExprHasAnyProperty(expr, EP.TokenOnly))
            {
                if (WalkExpr(expr.Left) != 0) return WRC.Abort;
                if (WalkExpr(expr.Right) != 0) return WRC.Abort;
                if (E.ExprHasProperty(expr, EP.xIsSelect))
                {
                    if (WalkSelect(expr.x.Select) != 0) return WRC.Abort;
                }
                else
                {
                    if (WalkExprList(expr.x.List) != 0) return WRC.Abort;
                }
            }
            return rc & WRC.Abort;
        }

        public WRC WalkExprList(ExprList p)
        {
            int i;
            ExprList.ExprListItem item;
            if (p != null)
                for (i = p.Exprs; i > 0; i--)
                {
                    item = p.Ids[p.Exprs - i];
                    if (WalkExpr(item.Expr) != 0) return WRC.Abort;
                }
            return WRC.Continue;
        }

        public WRC WalkSelectExpr(Select p)
        {
            if (WalkExprList(p.EList) != 0) return WRC.Abort;
            if (WalkExpr(p.Where) != 0) return WRC.Abort;
            if (WalkExprList(p.GroupBy) != 0) return WRC.Abort;
            if (WalkExpr(p.Having) != 0) return WRC.Abort;
            if (WalkExprList(p.OrderBy) != 0) return WRC.Abort;
            if (WalkExpr(p.Limit) != 0) return WRC.Abort;
            if (WalkExpr(p.Offset) != 0) return WRC.Abort;
            return WRC.Continue;
        }

        public WRC WalkSelectFrom(Select p)
        {
            SrcList src = p.Src;
            int i;
            SrcList.SrcListItem item;
            if (C._ALWAYS(src != null))
                for (i = src.Srcs; i > 0; i--)
                {
                    item = src.Ids[src.Srcs - i];
                    if (WalkSelect(item.Select) != 0) return WRC.Abort;
                }
            return WRC.Continue;
        }

        public WRC WalkSelect(Select p)
        {
            if (p == null || SelectCallback == null) return WRC.Continue;
            WRC rc = WRC.Continue;
            while (p != null)
            {
                rc = SelectCallback(this, p);
                if (rc != 0) break;
                if (WalkSelectExpr(p) != 0 || WalkSelectFrom(p) != 0)
                {
                    WalkerDepth--;
                    return WRC.Abort;
                }
                p = p.Prior;
            }
            return rc & WRC.Abort;
        }
    }
}
