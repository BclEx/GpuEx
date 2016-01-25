
// Code for testing all sorts of SQLite interfaces.  This code implements TCL commands for reading and writing the binary
// database files and displaying the content of those files as hexadecimal.  We could, in theory, use the built-in "binary"
// command of TCL to do a lot of this, but there are some issues with historical versions of the "binary" command.  So it seems
// easier and safer to build our own mechanism.
#include <Core+Vdbe\VdbeInt.cu.h>
#include "JimEx.h"

// Convert binary to hex.  The input zBuf[] contains N bytes of binary data.  zBuf[] is 2*n+1 bytes long.  Overwrite zBuf[]
// with a hexadecimal representation of its original binary input.
__device__ void sqlite3TestBinToHex(char *buf, int value)
{
	const char hex[] = "0123456789ABCDEF";
	char c;
	int i = value*2;
	buf[i--] = 0;
	for (int j = value-1; j >= 0; j--)
	{
		c = buf[j];
		buf[i--] = hex[c&0xf];
		buf[i--] = hex[c>>4];
	}
	_assert(i == -1);
}

// Convert hex to binary.  The input zIn[] contains N bytes of hexadecimal.  Convert this into binary and write aOut[] with
// the binary data.  Spaces in the original input are ignored. Return the number of bytes of binary rendered.
__device__ int sqlite3TestHexToBin(const char *in_, int value, char *out_)
{
	const unsigned char map[] = {
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		1, 2, 3, 4, 5, 6, 7, 8,  9,10, 0, 0, 0, 0, 0, 0,
		0,11,12,13,14,15,16, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0,11,12,13,14,15,16, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
	};
	int hi = 1;
	int i, j;
	for (i = j =0; i < value; i++)
	{
		unsigned char c = map[in_[i]];
		if (!c) continue;
		if (hi) { out_[j] = (c-1)<<4; hi = 0; }
		else { out_[j++] |= c-1; hi = 1; }
	}
	return j;
}

// Usage:   hexio_read  FILENAME  OFFSET  AMT
// Read AMT bytes from file FILENAME beginning at OFFSET from the beginning of the file.  Convert that information to hexadecimal
// and return the resulting HEX string.
__device__ static int hexio_read(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 4)
	{
		Jim_WrongNumArgs(interp, 1, args, "FILENAME OFFSET AMT");
		return JIM_ERROR;
	}
	int offset;
	int amt;
	if (Jim_GetInt(interp, args[2], &offset)) return JIM_ERROR;
	if (Jim_GetInt(interp, args[3], &amt)) return JIM_ERROR;
	const char *file = Jim_String(args[1]);
	char *buf = (char *)_alloc(amt*2+1 );
	if (!buf)
		return JIM_ERROR;
	FILE *in_ = _fopen(file, "rb");
	if (!in_)
		in_ = fopen(file, "r");
	if (!in_)
	{
		Jim_AppendResult(interp, "cannot open input file ", file, nullptr);
		return JIM_ERROR;
	}
	_fseek(in_, offset, SEEK_SET);
	int got = (int)_fread(buf, 1, amt, in_);
	_fclose(in_);
	if (got < 0)
		got = 0;
	sqlite3TestBinToHex(buf, got);
	Jim_AppendResult(interp, buf, nullptr);
	_free(buf);
	return JIM_OK;
}

// Usage:   hexio_write  FILENAME  OFFSET  DATA
// Write DATA into file FILENAME beginning at OFFSET from the beginning of the file.  DATA is expressed in hexadecimal.
__device__ static int hexio_write(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 4)
	{
		Jim_WrongNumArgs(interp, 1, args, "FILENAME OFFSET HEXDATA");
		return JIM_ERROR;
	}
	int offset;
	if (Jim_GetInt(interp, args[2], &offset)) return JIM_ERROR;
	const char *file = Jim_String(args[1]);
	int inLength;
	const char *in_ = (const char *)Jim_GetString(args[3], &inLength);
	char *out_ = (char *)_alloc(inLength/2);
	if (!out_)
		return JIM_ERROR;
	int outLength = sqlite3TestHexToBin(in_, inLength, out_);
	FILE *outFile = _fopen(file, "r+b");
	if (!outFile)
		outFile = _fopen(file, "r+");
	if (!outFile)
	{
		Jim_AppendResult(interp, "cannot open output file ", file, nullptr);
		return JIM_ERROR;
	}
	_fseek(outFile, offset, SEEK_SET);
	int written = (int)fwrite(out_, 1, outLength, outFile);
	_free(out_);
	fclose(outFile);
	Jim_SetResultInt(interp, written);
	return JIM_OK;
}

// USAGE:   hexio_get_int   HEXDATA
// Interpret the HEXDATA argument as a big-endian integer.  Return the value of that integer.  HEXDATA can contain between 2 and 8 hexadecimal digits.
__device__ static int hexio_get_int(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "HEXDATA");
		return JIM_ERROR;
	}
	int inLength;
	const char *in_ = (const char *)Jim_GetString(args[1], &inLength);
	char *out_ = (char *)_alloc(inLength/2);
	if (!out_)
		return JIM_ERROR;
	char num[4];
	int outLength = sqlite3TestHexToBin(in_, inLength, out_);
	if (outLength >= 4)
		_memcpy(num, out_, 4);
	else
	{
		_memset(num, 0, sizeof(num));
		_memcpy(&num[4-outLength], out_, outLength);
	}
	_free(out_);
	int val = (num[0]<<24) | (num[1]<<16) | (num[2]<<8) | num[3];
	Jim_SetResultInt(interp, val);
	return JIM_OK;
}

// USAGE:   hexio_render_int16   INTEGER
// Render INTEGER has a 16-bit big-endian integer in hexadecimal.
__device__ static int hexio_render_int16(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "INTEGER");
		return JIM_ERROR;
	}
	int val;
	if (Jim_GetInt(interp, args[1], &val)) return JIM_ERROR;
	char num[10];
	num[0] = val>>8;
	num[1] = val;
	sqlite3TestBinToHex(num, 2);
	Jim_SetResultString(interp, (char*)num, 4);
	return JIM_OK;
}


// USAGE:   hexio_render_int32   INTEGER
// Render INTEGER has a 32-bit big-endian integer in hexadecimal.
__device__ static int hexio_render_int32(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "INTEGER");
		return JIM_ERROR;
	}
	int val;
	if (Jim_GetInt(interp, args[1], &val)) return JIM_ERROR;
	char num[10];
	num[0] = val>>24;
	num[1] = val>>16;
	num[2] = val>>8;
	num[3] = val;
	sqlite3TestBinToHex(num, 4);
	Jim_SetResultString(interp, (char*)num, 8);
	return JIM_OK;
}

// USAGE:  utf8_to_utf8  HEX
// The argument is a UTF8 string represented in hexadecimal. The UTF8 might not be well-formed.  Run this string through sqlite3Utf8to8() convert it back to hex and return the result.
__device__ static int utf8_to_utf8(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[]){
#ifdef _DEBUG
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "HEX");
		return JIM_ERROR;
	}
	int n;
	const char *orig = (char *)Jim_GetString(args[1], &n);
	char *z = (char *)_alloc(n+3);
	n = sqlite3TestHexToBin(orig, n, z);
	z[n] = 0;
	int outLength = _utf8to8((unsigned char *)z);
	sqlite3TestBinToHex(z, outLength);
	Jim_AppendResult(interp, (char*)z, nullptr);
	_free(z);
	return JIM_OK;
#else
	Jim_AppendResult(interp, "[utf8_to_utf8] unavailable - _DEBUG not defined", nullptr);
	return JIM_ERROR;
#endif
}

__device__ static int getFts3Varint(const char *p, int64 *v)
{
	const unsigned char *q = (const unsigned char *)p;
	uint64 x = 0, y = 1;
	while ((*q & 0x80) == 0x80)
	{
		x += y * (*q++ & 0x7f);
		y <<= 7;
	}
	x += y * (*q++);
	*v = (int64)x;
	return (int)(q - (unsigned char *)p);
}

/*
** USAGE:  read_fts3varint BLOB VARNAME
** Read a varint from the start of BLOB. Set variable VARNAME to contain the interpreted value. Return the number of bytes of BLOB consumed.
*/
__device__ static int read_fts3varint(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "BLOB VARNAME");
		return JIM_ERROR;
	}
	int blobLength;
	unsigned char *blob = (unsigned char *)Jim_GetByteArray(args[1], &blobLength);
	int64 val;
	int valLength = getFts3Varint((char*)blob, (int64 *)(&val));
	Jim_ObjSetVar2(interp, args[2], nullptr, Jim_NewWideObj(interp, val));
	Jim_SetResultInt(interp, valLength);
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "hexio_read",                   hexio_read            },
	{ "hexio_write",                  hexio_write           },
	{ "hexio_get_int",                hexio_get_int         },
	{ "hexio_render_int16",           hexio_render_int16    },
	{ "hexio_render_int32",           hexio_render_int32    },
	{ "utf8_to_utf8",                 utf8_to_utf8          },
	{ "read_fts3varint",              read_fts3varint       },
};
__device__ int Sqlitetest_hexio_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	return JIM_OK;
}
