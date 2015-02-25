// table.c
#pragma region OMIT_GET_TABLE
#ifndef OMIT_GET_TABLE
#include "Core+Vdbe.cu.h"
#include <stdlib.h>
#include <string.h>

namespace Core
{
	struct TabResult
	{
		array_t<char *> Results; // Accumulated output - length = nData = (nRow+1)*nColumn
		char *ErrMsg;			// Error message text, if an error occurs
		int ResultsAlloc;		// Slots allocated for Results[]
		int Rows;				// Number of rows in the result
		int Columns;			// Number of columns in the result
		RC RC;					// Return code from sqlite3_exec()
	};

	__device__ bool Table::GetTableCallback(void *arg, int columns, char **argv, char **colv)
	{
		TabResult *p = (TabResult *)arg;  // Result accumulator
		// Make sure there is enough space in p->azResult to hold everything we need to remember from this invocation of the callback.
		int need = (!p->Rows && argv ? columns*2 : columns); // Slots needed in p->azResult[]
		if (p->Results.length + need > p->ResultsAlloc)
		{
			p->ResultsAlloc = p->ResultsAlloc*2 + need;
			char **newResults = (char **)_realloc(p->Results, sizeof(char *)*p->ResultsAlloc);
			if (!newResults) goto malloc_failed;
			p->Results = newResults;
		}

		// If this is the first row, then generate an extra row containing the names of all columns.
		char *z; // A single column of result
		if (!p->Rows)
		{
			p->Columns = columns;
			for (int i = 0; i < columns; i++)
			{
				z = _mprintf("%s", colv[i]);
				if (!z) goto malloc_failed;
				p->Results[p->Results.length++] = z;
			}
		}
		else if (p->Columns != columns)
		{
			_free(p->ErrMsg);
			p->ErrMsg = _mprintf("sqlite3_get_table() called with two or more incompatible queries");
			p->RC = RC_ERROR;
			return true;
		}

		// Copy over the row data
		if (argv)
		{
			for (int i = 0; i < columns; i++)
			{
				if (!argv[i])
					z = nullptr;
				else
				{
					int n = _strlen30(argv[i]) + 1;
					z = (char *)_alloc(n);
					if (!z) goto malloc_failed;
					_memcpy(z, argv[i], n);
				}
				p->Results[p->Results.length++] = z;
			}
			p->Rows++;
		}
		return false;

malloc_failed:
		p->RC = RC_NOMEM;
		return true;
	}

	__device__ RC Table::GetTable(Context *ctx, const char *sql, char ***results, int *rows, int *columns, char **errMsg)
	{
		*results = nullptr;
		if (columns) *columns = 0;
		if (rows) *rows = 0;
		if (errMsg) *errMsg = nullptr;
		TabResult r;
		r.ErrMsg = nullptr;
		r.Rows = 0;
		r.Columns = 0;
		r.RC = RC_OK;
		r.ResultsAlloc = 20;
		r.Results.data = (char **)_alloc(sizeof(char *)*r.ResultsAlloc);
		r.Results.length = 1;
		if (!r.Results)
			return (ctx->ErrCode = RC_NOMEM);
		r.Results[0] = nullptr;
		RC rc = Main::Exec(ctx, sql, GetTableCallback, (char **)&r, errMsg);
		_assert(sizeof(r.Results[0]) >= sizeof(r.Results.length));
		r.Results[0] = (char *)INT_TO_PTR(r.Results.length);
		if ((rc & 0xff) == RC_ABORT)
		{
			FreeTable(&r.Results[1]);
			if (r.ErrMsg)
			{
				if (errMsg)
				{
					_free(*errMsg);
					*errMsg = _mprintf("%s", r.ErrMsg);
				}
				_free(r.ErrMsg);
			}
			return (ctx->ErrCode = r.RC); // Assume 32-bit assignment is atomic
		}
		_free(r.ErrMsg);
		if (rc != RC_OK)
		{
			FreeTable(&r.Results[1]);
			return rc;
		}
		if (r.ResultsAlloc > r.Results.length)
		{
			char **newResults = (char **)_realloc(r.Results, sizeof(char*) * r.Results.length);
			if (!newResults)
			{
				FreeTable(&r.Results[1]);
				ctx->ErrCode = RC_NOMEM;
				return RC_NOMEM;
			}
			r.Results = newResults;
		}
		*results = &r.Results[1];
		if (columns) *columns = r.Columns;
		if (rows) *rows = r.Rows;
		return rc;
	}

	__device__ void Table::FreeTable(char **results)
	{
		if (results)
		{
			results--;
			_assert(!results);
			int n = PTR_TO_INT(results[0]);
			for (int i = 1; i < n; i++) if (results[i]) _free(results[i]);
			_free(results);
		}
	}
}

#endif
#pragma endregion 