// Test extension for testing the sqlite3_load_extension() function.
#include <string.h>
#include <Core+Vdbe\Core+Ext.cu.h>
EXTENSION_INIT1

	// The half() SQL function returns half of its input value.
	__device__ static void HalfFunc(FuncContext *fctx, int argc, Mem **args)
{
	Vdbe::Result_Double(fctx, 0.5 * Vdbe::Value_Double(args[0]));
}

// SQL functions to call the sqlite3_status function and return results.
__constant__ static const struct {
	const char *Name;
	STATUS OP;
} _ops[] = {
	{ "MEMORY_USED",         STATUS_MEMORY_USED         },
	{ "PAGECACHE_USED",      STATUS_PAGECACHE_USED      },
	{ "PAGECACHE_OVERFLOW",  STATUS_PAGECACHE_OVERFLOW  },
	{ "SCRATCH_USED",        STATUS_SCRATCH_USED        },
	{ "SCRATCH_OVERFLOW",    STATUS_SCRATCH_OVERFLOW    },
	{ "MALLOC_SIZE",         STATUS_MALLOC_SIZE         },
};

__device__ static void StatusFunc(FuncContext *fctx, int argc, Mem **args)
{
	STATUS op;
	if (Vdbe::Value_Type(args[0]) == TYPE_INTEGER)
		op = (STATUS)Vdbe::Value_Int(args[0]);
	else if (Vdbe::Value_Type(args[0]) == TYPE_TEXT)
	{
		int i;
		int opsLength = _lengthof(_ops);
		const char *name = (const char *)Vdbe::Value_Text(args[0]);
		for (i = 0; i < opsLength; i++)
		{
			if (!_strcmp(_ops[i].Name, name))
			{
				op = _ops[i].OP;
				break;
			}
		}
		if (i >= opsLength)
		{
			char *msg = _mprintf("unknown status property: %s", name);
			Vdbe::Result_Error(fctx, msg, -1);
			_free(msg);
			return;
		}
	}
	else
	{
		Vdbe::Result_Error(fctx, "unknown status type", -1);
		return;
	}
	bool resetFlag = (argc == 2 ? Vdbe::Value_Int(args[1]) != 0 : false);
	int cur, max;
	bool rc = _status(op, &cur, &max, resetFlag);
	if (!rc)
	{
		char *msg = _mprintf("sqlite3_status(%d,...) returns %d", op, rc);
		Vdbe::Result_Error(fctx, msg, -1);
		_free(msg);
		return;
	} 
	Vdbe::Result_Int(fctx, (argc == 2 ? max : cur));
}

// Extension load function.
__device__ RC testloadext_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	int err = 0;
	EXTENSION_INIT2(api);
	err |= Main::CreateFunction(ctx, "half", 1, TEXTENCODE_ANY, 0, HalfFunc, 0, 0);
	err |= Main::CreateFunction(ctx, "sqlite3_status", 1, TEXTENCODE_ANY, 0, StatusFunc, 0, 0);
	err |= Main::CreateFunction(ctx, "sqlite3_status", 2, TEXTENCODE_ANY, 0, StatusFunc, 0, 0);
	return (err ? RC_ERROR : RC_OK);
}

// Another extension entry point. This one always fails.
__device__ int testbrokenext_init(Context *ctx, char **errMsg, const core_api_routines *api)
{
	EXTENSION_INIT2(api);
	char *err = _mprintf("broken!");
	*errMsg = err;
	return 1;
}
