#include "Test.cu.h"
#include "..\System.Data.net\Core+Vdbe\VdbeInt.cu.h"

// The first argument is a TCL UTF-8 string. Return the byte array object with the encoded representation of the string, including the NULL terminator.
__device__ static int binarize(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	_assert(argc == 2);
	int len;
	const char *bytes = Jim_GetString(args[1], &len);
	Jim_SetResultByteArray(interp, (uint8 *)bytes, len+1);
	return JIM_OK;
}

// Usage: test_value_overhead <repeat-count> <do-calls>.
//
// This routine is used to test the overhead of calls to sqlite3_value_text(), on a value that contains a UTF-8 string. The idea
// is to figure out whether or not it is a problem to use sqlite3_value structures with collation sequence functions.
//
// If <do-calls> is 0, then the calls to sqlite3_value_text() are not actually made.
__device__ static int test_value_overhead(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " <repeat-count> <do-calls>", nullptr);
		return JIM_ERROR;
	}

	int repeat_count;
	int do_calls;
	if (Jim_GetInt(interp, args[1], &repeat_count)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[2], &do_calls)) return JIM_ERROR;

	Mem val;
	val.Flags = (MEM)(MEM_Str|MEM_Term|MEM_Static);
	val.Z = "hello world";
	val.Type = TYPE_TEXT;
	val.Encode = TEXTENCODE_UTF8;

	for (int i = 0; i < repeat_count; i++)
		if (do_calls)
			Vdbe::Value_Text(&val);
	return JIM_OK;
}

__constant__ struct EncName {
	char *Name;
	TEXTENCODE Encode;
} _encodeNames[] = {
	{ "UTF8", TEXTENCODE_UTF8 },
	{ "UTF16LE", TEXTENCODE_UTF16LE },
	{ "UTF16BE", TEXTENCODE_UTF16BE },
	{ "UTF16", TEXTENCODE_UTF16 },
	{ nullptr, (TEXTENCODE)0 }
};
__device__ static TEXTENCODE name_to_enc(Jim_Interp *interp, const char *arg)
{
	char *z = (char *)arg;
	struct EncName *encode;
	for (encode = &_encodeNames[0]; encode->Name; encode++)
		if (!_strcmp(z, encode->Name))
			break;
	if (!encode->Encode)
		Jim_AppendResult(interp, "No such encoding: ", z, nullptr);
	return (encode->Encode != TEXTENCODE_UTF16 ? encode->Encode : TEXTENCODE_UTF16NATIVE);
}

// Usage:   test_translate <string/blob> <from enc> <to enc> ?<transient>?
__device__ static int test_translate(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 4 && argc != 5)
	{
		Jim_AppendResult(interp, "wrong # args: should be \"", Jim_String(args[0]), " <string/blob> <from enc> <to enc>", nullptr);
		return JIM_ERROR;
	}
	void (*del)(void *p) = (argc == 5 ? _free : DESTRUCTOR_STATIC);

	TEXTENCODE enc_from = name_to_enc(interp, Jim_String(args[2]));
	if (!enc_from) return JIM_ERROR;
	TEXTENCODE enc_to = name_to_enc(interp, Jim_String(args[3]));
	if (!enc_to) return JIM_ERROR;

	const char *z;
	int len;
	Mem *val = Vdbe::ValueNew(nullptr);
	if (enc_from == TEXTENCODE_UTF8)
	{
		z = Jim_String(args[1]);
		if (argc == 5)
			z = _mprintf("%s", z);
		Vdbe::ValueSetStr(val, -1, z, enc_from, del);
	}
	else
	{
		z = Jim_GetByteArray(args[1], &len);
		if (argc == 5)
		{
			char *tmp = (char *)z;
			z = (const char *)_alloc(len);
			_memcpy((char *)z, tmp, len);
		}
		Vdbe::ValueSetStr(val, -1, z, enc_from, del);
	}

	z = (char *)Vdbe::ValueText(val, enc_to);
	len = Vdbe::ValueBytes(val, enc_to) + (enc_to==TEXTENCODE_UTF8?1:2);
	Jim_SetResultByteArray(interp, (uint8 *)z, len);

	Vdbe::ValueFree(val);
	return JIM_OK;
}

// Usage: translate_selftest
//
// Call _runtime_utfselftest() to run the internal tests for unicode translation. If there is a problem an assert() will fail.
__device__ void _runtime_utfselftest();
__device__ static int test_translate_selftest(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
#ifndef OMIT_UTF16
	_runtime_utfselftest();
#endif
	return RC_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "binarize",                binarize },
	{ "test_value_overhead",     test_value_overhead },
	{ "test_translate",          test_translate },
	{ "translate_selftest",      test_translate_selftest },
};
__device__ int Sqlitetest5_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	return RC_OK;
}
