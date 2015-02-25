// legacy.c
#include "Core+Vdbe.cu.h"
namespace Core
{
	__device__ RC Main::Exec(Context *ctx, const char *sql, bool (*callback)(void *, int, char **, char **), void *arg, char **errmsg)
	{
		RC rc = RC_OK; // Return code
		if (!SafetyCheckOk(ctx)) return SysEx_MISUSE_BKPT;
		if (!sql) sql = "";

		MutexEx::Enter(ctx->Mutex);
		Error(ctx, RC_OK, nullptr);
		Vdbe *stmt = nullptr; // The current SQL statement
		int retrys = 0; // Number of retry attempts
		char **colsNames = nullptr; // Names of result columns
		while ((rc == RC_OK || (rc == RC_SCHEMA && (++retrys) < 2)) && sql[0])
		{
			stmt = nullptr;
			const char *leftover; // Tail of unprocessed SQL
			rc = Prepare::Prepare_(ctx, sql, -1, &stmt, &leftover);
			_assert(rc == RC_OK || !stmt);
			if (rc != RC_OK)
				continue;
			if (!stmt)
			{
				sql = leftover; // this happens for a comment or white-space
				continue;
			}

			bool callbackIsInit = false; // True if callback data is initialized
			int cols = Vdbe::Column_Count(stmt);

			while (1)
			{
				rc = stmt->Step();

				// Invoke the callback function if required
				int i;
				if (callback && (rc == RC_ROW || (rc == RC_DONE && !callbackIsInit && (ctx->Flags & Context::FLAG_NullCallback) != 0)))
				{
					if (!callbackIsInit)
					{
						colsNames = (char **)_tagalloc2(ctx, 2*cols*sizeof(const char*) + 1, true);
						if (!colsNames)
							goto exec_out;
						for (i = 0; i < cols; i++)
						{
							colsNames[i] = (char *)Vdbe::Column_Name(stmt, i);
							// Vdbe::SetColName() installs column names as UTF8 strings so there is no way for sqlite3_column_name() to fail.
							_assert(colsNames[i] != 0);
						}
						callbackIsInit = true;
					}
					char **colsValues = nullptr;
					if (rc == RC_ROW)
					{
						colsValues = &colsNames[cols];
						for (i = 0; i < cols; i++)
						{
							colsValues[i] = (char *)Vdbe::Column_Text(stmt, i);
							if (!colsValues[i] && Vdbe::Column_Type(stmt, i) != TYPE_NULL)
							{
								ctx->MallocFailed = true;
								goto exec_out;
							}
						}
					}
					if (callback(arg, cols, colsValues, colsNames))
					{
						rc = RC_ABORT;
						stmt->Finalize();
						stmt = nullptr;
						Error(ctx, RC_ABORT, nullptr);
						goto exec_out;
					}
				}

				if (rc != RC_ROW)
				{
					stmt->Finalize();
					stmt = nullptr;
					if (rc != RC_SCHEMA)
					{
						retrys = 0;
						sql = leftover;
						while (_isspace(sql[0])) sql++;
					}
					break;
				}
			}

			_tagfree(ctx, colsNames);
			colsNames = nullptr;
		}

exec_out:
		if (stmt) stmt->Finalize();
		_tagfree(ctx, colsNames);

		rc = ApiExit(ctx, rc);
		if (rc != RC_OK && _ALWAYS(rc == ErrCode(ctx)) && errmsg)
		{
			int errMsgLength = 1 + _strlen30(ErrMsg(ctx));
			*errmsg = (char *)_alloc(errMsgLength);
			if (*errmsg)
				_memcpy(*errmsg, ErrMsg(ctx), errMsgLength);
			else
			{
				rc = RC_NOMEM;
				Error(ctx, RC_NOMEM, 0);
			}
		}
		else if (errmsg)
			*errmsg = nullptr;

		_assert((rc & ctx->ErrMask) == rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}
}