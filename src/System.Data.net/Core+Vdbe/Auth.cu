#pragma region OMIT_AUTHORIZATION
#ifndef OMIT_AUTHORIZATION
#include "Core+Vdbe.cu.h"

namespace Core
{
	__device__ RC Auth::SetAuthorizer(Context *ctx, ARC (*auth)(void*,int,const char*,const char*,const char*,const char*), void *arg)
	{
		MutexEx::Enter(ctx->Mutex);
		ctx->Auth = auth;
		ctx->AuthArg = arg;
		Vdbe::ExpirePreparedStatements(ctx);
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

	__device__ void Auth::BadReturnCode(Parse *parse)
	{
		parse->ErrorMsg("authorizer malfunction");
		parse->RC = RC_ERROR;
	}

	__device__ ARC Auth::ReadColumn(Parse *parse, const char *table, const char *column, int db)
	{
		Context *ctx = parse->Ctx; // Database handle
		char *dbName = ctx->DBs[db].Name; // Name of attached database
		ARC rc = ctx->Auth(ctx->AuthArg, AUTH_READ, table, column, dbName, parse->AuthContext); // Auth callback return code
		if (rc == ARC_DENY)
		{
			if (ctx->DBs.length > 2 || db != 0)
				parse->ErrorMsg("access to %s.%s.%s is prohibited", dbName, table, column);
			else
				parse->ErrorMsg("access to %s.%s is prohibited", table, column);
			parse->RC = RC_AUTH;
		}
		else if (rc != ARC_IGNORE && rc != RC_OK)
			BadReturnCode(parse);
		return rc;
	}

	__device__ void Auth::Read(Parse *parse, Expr *expr, Schema *schema, SrcList *tableList)
	{
		Context *ctx = parse->Ctx;
		if (!ctx->Auth) return;
		int db = Prepare::SchemaToIndex(ctx, schema); // The index of the database the expression refers to
		if (db < 0)
			return; // An attempt to read a column out of a subquery or other temporary table.

		_assert(expr->OP == TK_COLUMN || expr->OP == TK_TRIGGER);
		Table *table = nullptr; // The table being read
		if (expr->OP == TK_TRIGGER)
			table = parse->TriggerTab;
		else
		{
			_assert(tableList);
			for (int src = 0; _ALWAYS(src < tableList->Srcs); src++)
				if (expr->TableId == tableList->Ids[src].Cursor)
				{
					table = tableList->Ids[src].Table;
					break;
				}
		}
		int col = expr->ColumnIdx; // Index of column in table
		if (_NEVER(table)) return;

		const char *colName; // Name of the column of the table
		if (col >= 0)
		{
			_assert(col < table->Cols.length);
			colName = table->Cols[col].Name;
		}
		else if (table->PKey >= 0)
		{
			_assert(table->PKey < table->Cols.length);
			colName = table->Cols[table->PKey].Name;
		}
		else
			colName = "ROWID";
		_assert(db >= 0 && db < ctx->DBs.length);
		if (ReadColumn(parse, table->Name, colName, db) == ARC_IGNORE)
			expr->OP = TK_NULL;
	}

	__device__ ARC Auth::Check(Parse *parse, AUTH code, const char *arg1, const char *arg2, const char *arg3)
	{
		Context *ctx = parse->Ctx;
		// Don't do any authorization checks if the database is initialising or if the parser is being invoked from within sqlite3_declare_vtab.
		if (ctx->Init.Busy || INDECLARE_VTABLE(parse))
			return ARC_OK;

		if (!ctx->Auth)
			return ARC_OK;
		ARC rc = (ARC)ctx->Auth(ctx->AuthArg, code, arg1, arg2, arg3, parse->AuthContext);
		if (rc == ARC_DENY)
		{
			parse->ErrorMsg("not authorized");
			parse->RC = RC_AUTH;
		}
		else if (rc != RC_OK && rc != ARC_IGNORE)
		{
			rc = ARC_DENY;
			BadReturnCode(parse);
		}
		return rc;
	}

	__device__ void Auth::ContextPush(Parse *parse, AuthContext *actx, const char *context)
	{
		_assert(parse);
		actx->Parse = parse;
		actx->AuthCtx = parse->AuthContext;
		parse->AuthContext = context;
	}

	__device__ void Auth::ContextPop(AuthContext *actx)
	{
		if (actx->Parse)
		{
			actx->Parse->AuthContext = actx->AuthCtx;
			actx->Parse = nullptr;
		}
	}
}
#endif
#pragma endregion
