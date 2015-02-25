// walker.c
#include "Core+Vdbe.cu.h"

namespace Core
{
	__device__ WRC Walker::WalkExpr(Expr *expr)
	{
		if (!expr) return WRC_Continue;
		ASSERTCOVERAGE(ExprHasProperty(expr, EP_TokenOnly));
		ASSERTCOVERAGE(ExprHasProperty(expr, EP_Reduced));
		int rc = ExprCallback(this, expr);
		if (rc == WRC_Continue && !ExprHasAnyProperty(expr, EP_TokenOnly))
		{
			if (WalkExpr(expr->Left)) return WRC_Abort;
			if (WalkExpr(expr->Right)) return WRC_Abort;
			if (ExprHasProperty(expr, EP_xIsSelect))
			{
				if (WalkSelect(expr->x.Select)) return WRC_Abort;
			}
			else
			{
				if (WalkExprList(expr->x.List)) return WRC_Abort;
			}
		}
		return (WRC)(rc & WRC_Abort);
	}

	__device__ WRC Walker::WalkExprList(ExprList *p)
	{
		int i;
		ExprList::ExprListItem *item;
		if (p)
			for (i = p->Exprs, item = p->Ids; i > 0; i--, item++)
				if (WalkExpr(item->Expr)) return WRC_Abort;
		return WRC_Continue;
	}

	__device__ WRC Walker::WalkSelectExpr(Select *p)
	{
		if (WalkExprList(p->EList)) return WRC_Abort;
		if (WalkExpr(p->Where)) return WRC_Abort;
		if (WalkExprList(p->GroupBy)) return WRC_Abort;
		if (WalkExpr(p->Having)) return WRC_Abort;
		if (WalkExprList(p->OrderBy)) return WRC_Abort;
		if (WalkExpr(p->Limit)) return WRC_Abort;
		if (WalkExpr(p->Offset)) return WRC_Abort;
		return WRC_Continue;
	}

	__device__ WRC Walker::WalkSelectFrom(Select *p)
	{
		SrcList *src = p->Src;
		int i;
		SrcList::SrcListItem *item;
		if (_ALWAYS(src))
			for (i = src->Srcs, item = src->Ids; i > 0; i--, item++)
				if (WalkSelect(item->Select)) return WRC_Abort;
		return WRC_Continue;
	} 

	__device__ WRC Walker::WalkSelect(Select *p)
	{
		if (!p || !SelectCallback) return WRC_Continue;
		WRC rc = WRC_Continue;
		WalkerDepth++;
		while (p)
		{
			rc = SelectCallback(this, p);
			if (rc) break;
			if (WalkSelectExpr(p) || WalkSelectFrom(p))
			{
				WalkerDepth--;
				return WRC_Abort;
			}
			p = p->Prior;
		}
		WalkerDepth--;
		return (WRC)(rc & WRC_Abort);
	}
}