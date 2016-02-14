// Code for testing all sorts of SQLite interfaces.  This code implements new SQL functions used by the test scripts.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>
//#include <stdlib.h>
//#include <string.h>
//#include <assert.h>

// Allocate nByte bytes of space using sqlite3_malloc(). If the allocation fails, call sqlite3_result_error_nomem() to notify
// the database handle that malloc() has failed.
__device__ static void *testContextMalloc(FuncContext *fctx, int bytes)
{
	char *z = (char *)_alloc(bytes);
	if (!z && bytes > 0)
		Vdbe::Result_ErrorNoMem(fctx);
	return z;
}

// This function generates a string of random characters.  Used for generating test data.
__constant__ static const unsigned char _srcs[] = 
	"abcdefghijklmnopqrstuvwxyz"
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"0123456789"
	".-!,:*^+=_|?/<> ";
__device__ static void randStr(FuncContext *fctx, int argc, Mem **args)
{
	// It used to be possible to call randstr() with any number of arguments, but now it is registered with SQLite as requiring exactly 2.
	_assert(argc == 2);

	unsigned char buf[1000];
	int min = Vdbe::Value_Int(args[0]);
	if (min < 0) min = 0;
	if (min >= sizeof(buf)) min = sizeof(buf)-1;
	int max = Vdbe::Value_Int(args[1]);
	if (max < min) max = min;
	if (max >= sizeof(buf)) max = sizeof(buf)-1;
	int n = min;
	if (max > min)
	{
		int r;
		SysEx::PutRandom(sizeof(r), &r);
		r &= 0x7fffffff;
		n += r%(max + 1 - min);
	}
	_assert(n < sizeof(buf));
	SysEx::PutRandom(n, buf);
	for (int i = 0; i < n; i++)
		buf[i] = _srcs[buf[i]%(sizeof(_srcs)-1)];
	buf[n] = 0;
	Vdbe::Result_Text(fctx, (char *)buf, n, DESTRUCTOR_TRANSIENT);
}

// The following two SQL functions are used to test returning a text result with a destructor. Function 'test_destructor' takes one argument
// and returns the same argument interpreted as TEXT. A destructor is passed with the sqlite3_result_text() call.
//
// SQL function 'test_destructor_count' returns the number of outstanding allocations made by 'test_destructor';
//
// WARNING: Not threadsafe.
__device__ static int _test_destructor_count_var = 0;
__device__ static void destructor(void *p)
{
	char *val = (char *)p;
	_assert(val);
	val--;
	_free(val);
	_test_destructor_count_var--;
}
__device__ static void test_destructor(FuncContext *fctx, int argc, Mem **args)
{
	_test_destructor_count_var++;
	_assert(argc == 1);
	if (Vdbe::Value_Type(args[0]) == TYPE_NULL) return;
	int len = Vdbe::Value_Bytes(args[0]); 
	char *val = (char *)testContextMalloc(fctx, len+3);
	if (!val)
		return;
	val[len+1] = 0;
	val[len+2] = 0;
	val++;
	_memcpy(val, Vdbe::Value_Text(args[0]), len);
	Vdbe::Result_Text(fctx, val, -1, destructor);
}
#ifndef OMIT_UTF16
__device__ static void test_destructor16(FuncContext *fctx, int argc, Mem **args)
{
	_test_destructor_count_var++;
	_assert(argc == 1);
	if (Vdbe::Value_Type(args[0]) == TYPE_NULL) return;
	int len = Vdbe::Value_Bytes16(args[0]); 
	char *val = (char *)testContextMalloc(fctx, len+3);
	if (!val)
		return;
	val[len+1] = 0;
	val[len+2] = 0;
	val++;
	_memcpy(val, Vdbe::Value_Text16(args[0]), len);
	Vdbe::Result_Text16(fctx, val, -1, destructor);
}
#endif
__device__ static void test_destructor_count(FuncContext *fctx, int argc, Mem **args)
{
	Vdbe::Result_Int(fctx, _test_destructor_count_var);
}

// The following aggregate function, test_agg_errmsg16(), takes zero arguments. It returns the text value returned by the sqlite3_errmsg16()
// API function.
#ifndef OMIT_BUILTIN_TEST
__device__ void _benignalloc_begin();
__device__ void _benignalloc_end();
#else
#define _benignalloc_begin()
#define _benignalloc_end()
#endif
__device__ static void test_agg_errmsg16_step(FuncContext *a, int b, Mem **c)
{
}
__device__ static void test_agg_errmsg16_final(FuncContext *fctx)
{
#ifndef OMIT_UTF16
	Context *ctx = Vdbe::Context_Ctx(fctx);
	Vdbe::Aggregate_Context(fctx, 2048);
	_benignalloc_begin();
	const void *z = DataEx::ErrMsg16(ctx);
	_benignalloc_end();
	Vdbe::Result_Text16(fctx, z, -1, DESTRUCTOR_TRANSIENT);
#endif
}

// Routines for testing the sqlite3_get_auxdata() and sqlite3_set_auxdata() interface.
//
// The test_auxdata() SQL function attempts to register each of its arguments as auxiliary data.  If there are no prior registrations of aux data for
// that argument (meaning the argument is not a constant or this is its first call) then the result for that argument is 0.  If there is a prior
// registration, the result for that argument is 1.  The overall result is the individual argument results separated by spaces.
__device__ static void free_test_auxdata(void *p) { _free(p); }
__device__ static void test_auxdata(FuncContext *fctx, int argc, Mem **args)
{
	char *ret = (char *)testContextMalloc(fctx, argc*2);
	if (!ret) return;
	_memset(ret, 0, argc*2);
	for (int i = 0; i < argc; i++)
	{
		char const *z = (char *)Vdbe::Value_Text(args[i]);
		if (z)
		{
			int n;
			char *aux = (char *)Vdbe::get_Auxdata(fctx, i);
			if (aux)
			{
				ret[i*2] = '1';
				_assert(!_strcmp(aux, z));
			}
			else
				ret[i*2] = '0';
			n = (int)_strlen(z) + 1;
			aux = (char *)testContextMalloc(fctx, n);
			if (aux)
			{
				_memcpy(aux, z, n);
				Vdbe::set_Auxdata(fctx, i, aux, free_test_auxdata);
			}
			ret[i*2+1] = ' ';
		}
	}
	Vdbe::Result_Text(fctx, ret, 2*argc-1, free_test_auxdata);
}

// A function to test error reporting from user functions. This function returns a copy of its first argument as the error message.  If the
// second argument exists, it becomes the error code.
__device__ static void test_error(FuncContext *fctx, int argc, Mem **args)
{
	Vdbe::Result_Error(fctx, (char *)Vdbe::Value_Text(args[0]), -1);
	if (argc == 2)
		Vdbe::Result_ErrorCode(fctx, (RC)Vdbe::Value_Int(args[1]));
}

// Implementation of the counter(X) function.  If X is an integer constant, then the first invocation will return X.  The second X+1.
// and so forth.  Can be used (for example) to provide a sequence number in a result set.
__device__ static void counterFunc(FuncContext *fctx, int argc, Mem **args)
{
	int *counter = (int *)Vdbe::get_Auxdata(fctx, 0);
	if (!counter)
	{
		counter = (int *)_alloc(sizeof(*counter));
		if (!counter)
		{
			Vdbe::Result_ErrorNoMem(fctx);
			return;
		}
		*counter = Vdbe::Value_Int(args[0]);
		Vdbe::set_Auxdata(fctx, 0, counter, _free);
	}
	else
		++*counter;
	Vdbe::Result_Int(fctx, *counter);
}

// This function takes two arguments.  It performance UTF-8/16 type conversions on the first argument then returns a copy of the second
// argument.
//
// This function is used in cases such as the following:
//
//      SELECT test_isolation(x,x) FROM t1;
//
// We want to verify that the type conversions that occur on the first argument do not invalidate the second argument.
__device__ static void test_isolation(FuncContext *fctx, int argc, Mem **args)
{
#ifndef OMIT_UTF16
	Vdbe::Value_Text16(args[0]);
	Vdbe::Value_Text(args[0]);
	Vdbe::Value_Text16(args[0]);
	Vdbe::Value_Text(args[0]);
#endif
	Vdbe::Result_Value(fctx, args[1]);
}

// Invoke an SQL statement recursively.  The function result is the first column of the first row of the result set.
__device__ static void test_eval(FuncContext *fctx, int argc, Mem **args)
{
	Context *ctx = Vdbe::Context_Ctx(fctx);
	const char *sql = (char *)Vdbe::Value_Text(args[0]);
	Vdbe *stmt;
	RC rc = Prepare::Prepare_v2(ctx, sql, -1, &stmt, 0);
	if (rc == RC_OK)
	{
		rc = stmt->Step();
		if (rc == RC_ROW)
			Vdbe::Result_Value(fctx, Vdbe::Column_Value(stmt, 0));
		rc = Vdbe::Finalize(stmt);
	}
	if (rc)
	{
		_assert(!stmt);
		char *err = _mprintf("sqlite3_prepare_v2() error: %s", DataEx::ErrMsg(ctx));
		Vdbe::Result_Text(fctx, err, -1, _free);
		Vdbe::Result_ErrorCode(fctx, rc);
	}
}

// convert one character from hex to binary
__device__ static int testHexChar(char c)
{
	if (c >= '0' && c <= '9') return c - '0';
	else if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	else if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return 0;
}

// Convert hex to binary.
__device__ static void testHexToBin(const char *in, char *out)
{
	while (in[0] && in[1])
	{
		*(out++) = (testHexChar(in[0])<<4) + testHexChar(in[1]);
		in += 2;
	}
}

//      hex_to_utf16be(HEX)
//
// Convert the input string from HEX into binary.  Then return the result using sqlite3_result_text16le().
#ifndef OMIT_UTF16
__device__ static void testHexToUtf16be(FuncContext *fctx, int argc, Mem **args)
{
	_assert(argc == 1);
	int n = Vdbe::Value_Bytes(args[0]);
	const char *in = (const char *)Vdbe::Value_Text(args[0]);
	char *out = (char *)_alloc(n/2);
	if (!out)
		Vdbe::Result_ErrorNoMem(fctx);
	else
	{
		testHexToBin(in, out);
		Vdbe::Result_Text16be(fctx, out, n/2, _free);
	}
}
#endif

//      hex_to_utf8(HEX)
//
// Convert the input string from HEX into binary.  Then return the result using sqlite3_result_text16le().
__device__ static void testHexToUtf8(FuncContext *fctx, int argc, Mem **args)
{
	_assert(argc == 1);
	int n = Vdbe::Value_Bytes(args[0]);
	const char *in = (const char *)Vdbe::Value_Text(args[0]);
	char *out = (char *)_alloc(n/2);
	if (!out)
		Vdbe::Result_ErrorNoMem(fctx);
	else
	{
		testHexToBin(in, out);
		Vdbe::Result_Text(fctx, out, n/2, _free);
	}
}

//      hex_to_utf16le(HEX)
//
// Convert the input string from HEX into binary.  Then return the result using sqlite3_result_text16le().
#ifndef OMIT_UTF16
__device__ static void testHexToUtf16le(FuncContext *fctx, int argc, Mem **args)
{
	_assert(argc == 1);
	int n = Vdbe::Value_Bytes(args[0]);
	const char *in = (const char *)Vdbe::Value_Text(args[0]);
	char *out = (char *)_alloc(n/2);
	if (!out)
		Vdbe::Result_ErrorNoMem(fctx);
	else
	{
		testHexToBin(in, out);
		Vdbe::Result_Text16le(fctx, out, n/2, _free);
	}
}
#endif

// SQL function:   real2hex(X)
//
// If argument X is a real number, then convert it into a string which is the big-endian hexadecimal representation of the ieee754 encoding of
// that number.  If X is not a real number, return NULL.
__device__ static void real2hex(FuncContext *fctx, int argc, Mem **args)
{
	union
	{
		uint64 i;
		double r;
		unsigned char x[8];
	} v;
	v.i = 1;
	int bigEndian = (v.x[0]==0);
	v.r = Vdbe::Value_Double(args[0]);
	char out[20];
	for (int i = 0; i < 8; i++)
	{
		if (bigEndian)
		{
			out[i*2]   = "0123456789abcdef"[v.x[i]>>4];
			out[i*2+1] = "0123456789abcdef"[v.x[i]&0xf];
		}
		else
		{
			out[14-i*2]   = "0123456789abcdef"[v.x[i]>>4];
			out[14-i*2+1] = "0123456789abcdef"[v.x[i]&0xf];
		}
	}
	out[16] = 0;
	Vdbe::Result_Text(fctx, out, -1, DESTRUCTOR_TRANSIENT);
}

__constant__ static const struct {
	char *Name;
	signed char Args;
	TEXTENCODE TextRep; // 1: UTF-16.  0: UTF-8
	void (*Func)(FuncContext*,int,Mem**);
} _funcs[] = {
	{ "randstr",               2, TEXTENCODE_UTF8, randStr    },
	{ "test_destructor",       1, TEXTENCODE_UTF8, test_destructor},
#ifndef SQLITE_OMIT_UTF16
	{ "test_destructor16",     1, TEXTENCODE_UTF8, test_destructor16},
	{ "hex_to_utf16be",        1, TEXTENCODE_UTF8, testHexToUtf16be},
	{ "hex_to_utf16le",        1, TEXTENCODE_UTF8, testHexToUtf16le},
#endif
	{ "hex_to_utf8",           1, TEXTENCODE_UTF8, testHexToUtf8},
	{ "test_destructor_count", 0, TEXTENCODE_UTF8, test_destructor_count},
	{ "test_auxdata",         -1, TEXTENCODE_UTF8, test_auxdata},
	{ "test_error",            1, TEXTENCODE_UTF8, test_error},
	{ "test_error",            2, TEXTENCODE_UTF8, test_error},
	{ "test_eval",             1, TEXTENCODE_UTF8, test_eval},
	{ "test_isolation",        2, TEXTENCODE_UTF8, test_isolation},
	{ "test_counter",          1, TEXTENCODE_UTF8, counterFunc},
	{ "real2hex",              1, TEXTENCODE_UTF8, real2hex},
};
__device__ static int registerTestFunctions(Context *ctx)
{
	for (int i = 0; i < _lengthof(_funcs); i++)
		DataEx::CreateFunction(ctx, _funcs[i].Name, _funcs[i].Args, _funcs[i].TextRep, nullptr, _funcs[i].Func, nullptr, nullptr);
	DataEx::CreateFunction(ctx, "test_agg_errmsg16", 0, TEXTENCODE_ANY, nullptr, nullptr, test_agg_errmsg16_step, test_agg_errmsg16_final);
	return RC_OK;
}

// TCLCMD:  autoinstall_test_functions
//
// Invoke this TCL command to use sqlite3_auto_extension() to cause the standard set of test functions to be loaded into each new
// database connection.
__device__ static int autoinstall_test_funcs(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	extern int Md5_Register(Context *);
	RC rc = DataEx::AutoExtension((void(*)())registerTestFunctions);
	if (rc == RC_OK)
		rc = DataEx::AutoExtension((void(*)())Md5_Register);
	Jim_SetResultInt(interp, rc);
	return JIM_OK;
}

// A bogus step function and finalizer function.
__device__ static void tStep(FuncContext *a, int b, Mem **c) { }
__device__ static void tFinal(FuncContext *a) { }

// tclcmd:  abuse_create_function
//
// Make various calls to sqlite3_create_function that do not have valid parameters.  Verify that the error condition is detected and reported.
__device__ static int abuse_create_function(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	__device__ extern int GetDbPointer(Jim_Interp *, const char *, Context **);
	Context *ctx;
	if (GetDbPointer(interp, (char *)Jim_String(args[1]), &ctx)) return JIM_ERROR;

	RC rc = DataEx::CreateFunction(ctx, "tx", 1, TEXTENCODE_UTF8, nullptr, tStep, tStep, tFinal);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", 1, TEXTENCODE_UTF8, nullptr, tStep, tStep, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", 1, TEXTENCODE_UTF8, nullptr, tStep, nullptr, tFinal);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", 1, TEXTENCODE_UTF8, nullptr, nullptr, nullptr, tFinal);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", 1, TEXTENCODE_UTF8, nullptr, nullptr, tStep, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", -2, TEXTENCODE_UTF8, nullptr, tStep, nullptr, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "tx", 128, TEXTENCODE_UTF8, nullptr, tStep, nullptr, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;

	rc = DataEx::CreateFunction(ctx, "funcxx"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789",
		1, TEXTENCODE_UTF8, nullptr, tStep, nullptr, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;

	// This last function registration should actually work.  Generate a no-op function (that always returns NULL) and which has the
	// maximum-length function name and the maximum number of parameters.
	DataEx::Limit(ctx, LIMIT_FUNCTION_ARG, 10000);
	int maxArg = DataEx::Limit(ctx, LIMIT_FUNCTION_ARG, -1);
	rc = DataEx::CreateFunction(ctx, "nullx"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789"
		"_123456789_123456789_123456789_123456789_123456789",
		maxArg, TEXTENCODE_UTF8, nullptr, tStep, nullptr, nullptr);
	if (rc != RC_MISUSE) goto abuse_err;
	return JIM_OK;

abuse_err:
	Jim_AppendResult(interp, "sqlite3_create_function abused test failed", nullptr);
	return JIM_ERROR;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _objCmds[] = {
	{ "autoinstall_test_functions",    autoinstall_test_funcs },
	{ "abuse_create_function",         abuse_create_function  },
};
__device__ int Sqlitetest_func_Init(Jim_Interp *interp)
{
	extern int Md5_Register(Context*);
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, nullptr, nullptr);
	DataEx::Initialize();
	DataEx::AutoExtension((void(*)())registerTestFunctions);
	DataEx::AutoExtension((void(*)())Md5_Register);
	return JIM_OK;
}
