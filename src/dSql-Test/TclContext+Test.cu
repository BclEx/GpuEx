#include <RuntimeHost.h>
#include "TclContext.cu.h"

#pragma region MD5
#if defined(_TEST) || defined(_TCLMD5)

#ifndef uint32
#define uint32 unsigned int
#endif
struct MD5Context
{
	bool IsInit;
	uint32 Buf[4];
	uint32 Bits[2];
	unsigned char In[64];
};
typedef struct MD5Context MD5Context;

__device__ static void ByteReverse(unsigned char *buf, unsigned longs)
{
	uint32 t;
	do
	{
		t = (uint32)((unsigned)buf[3]<<8 | buf[2]) << 16 | ((unsigned)buf[1]<<8 | buf[0]);
		*(uint32 *)buf = t;
		buf += 4;
	} while (--longs);
}

#define F1(x, y, z) (z ^ (x & (y ^ z))) // #define F1(x, y, z) (x & y | ~x & z)
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))
#define MD5STEP(f, w, x, y, z, data, s) ( w += f(x, y, z) + data,  w = w<<s | w>>(32-s),  w += x )

__device__ static void MD5Transform(uint32 buf[4], const uint32 in[16])
{
	register uint32 a = buf[0];
	register uint32 b = buf[1];
	register uint32 c = buf[2];
	register uint32 d = buf[3];
	MD5STEP(F1, a, b, c, d, in[ 0]+0xd76aa478,  7);
	MD5STEP(F1, d, a, b, c, in[ 1]+0xe8c7b756, 12);
	MD5STEP(F1, c, d, a, b, in[ 2]+0x242070db, 17);
	MD5STEP(F1, b, c, d, a, in[ 3]+0xc1bdceee, 22);
	MD5STEP(F1, a, b, c, d, in[ 4]+0xf57c0faf,  7);
	MD5STEP(F1, d, a, b, c, in[ 5]+0x4787c62a, 12);
	MD5STEP(F1, c, d, a, b, in[ 6]+0xa8304613, 17);
	MD5STEP(F1, b, c, d, a, in[ 7]+0xfd469501, 22);
	MD5STEP(F1, a, b, c, d, in[ 8]+0x698098d8,  7);
	MD5STEP(F1, d, a, b, c, in[ 9]+0x8b44f7af, 12);
	MD5STEP(F1, c, d, a, b, in[10]+0xffff5bb1, 17);
	MD5STEP(F1, b, c, d, a, in[11]+0x895cd7be, 22);
	MD5STEP(F1, a, b, c, d, in[12]+0x6b901122,  7);
	MD5STEP(F1, d, a, b, c, in[13]+0xfd987193, 12);
	MD5STEP(F1, c, d, a, b, in[14]+0xa679438e, 17);
	MD5STEP(F1, b, c, d, a, in[15]+0x49b40821, 22);
	//
	MD5STEP(F2, a, b, c, d, in[ 1]+0xf61e2562,  5);
	MD5STEP(F2, d, a, b, c, in[ 6]+0xc040b340,  9);
	MD5STEP(F2, c, d, a, b, in[11]+0x265e5a51, 14);
	MD5STEP(F2, b, c, d, a, in[ 0]+0xe9b6c7aa, 20);
	MD5STEP(F2, a, b, c, d, in[ 5]+0xd62f105d,  5);
	MD5STEP(F2, d, a, b, c, in[10]+0x02441453,  9);
	MD5STEP(F2, c, d, a, b, in[15]+0xd8a1e681, 14);
	MD5STEP(F2, b, c, d, a, in[ 4]+0xe7d3fbc8, 20);
	MD5STEP(F2, a, b, c, d, in[ 9]+0x21e1cde6,  5);
	MD5STEP(F2, d, a, b, c, in[14]+0xc33707d6,  9);
	MD5STEP(F2, c, d, a, b, in[ 3]+0xf4d50d87, 14);
	MD5STEP(F2, b, c, d, a, in[ 8]+0x455a14ed, 20);
	MD5STEP(F2, a, b, c, d, in[13]+0xa9e3e905,  5);
	MD5STEP(F2, d, a, b, c, in[ 2]+0xfcefa3f8,  9);
	MD5STEP(F2, c, d, a, b, in[ 7]+0x676f02d9, 14);
	MD5STEP(F2, b, c, d, a, in[12]+0x8d2a4c8a, 20);
	//
	MD5STEP(F3, a, b, c, d, in[ 5]+0xfffa3942,  4);
	MD5STEP(F3, d, a, b, c, in[ 8]+0x8771f681, 11);
	MD5STEP(F3, c, d, a, b, in[11]+0x6d9d6122, 16);
	MD5STEP(F3, b, c, d, a, in[14]+0xfde5380c, 23);
	MD5STEP(F3, a, b, c, d, in[ 1]+0xa4beea44,  4);
	MD5STEP(F3, d, a, b, c, in[ 4]+0x4bdecfa9, 11);
	MD5STEP(F3, c, d, a, b, in[ 7]+0xf6bb4b60, 16);
	MD5STEP(F3, b, c, d, a, in[10]+0xbebfbc70, 23);
	MD5STEP(F3, a, b, c, d, in[13]+0x289b7ec6,  4);
	MD5STEP(F3, d, a, b, c, in[ 0]+0xeaa127fa, 11);
	MD5STEP(F3, c, d, a, b, in[ 3]+0xd4ef3085, 16);
	MD5STEP(F3, b, c, d, a, in[ 6]+0x04881d05, 23);
	MD5STEP(F3, a, b, c, d, in[ 9]+0xd9d4d039,  4);
	MD5STEP(F3, d, a, b, c, in[12]+0xe6db99e5, 11);
	MD5STEP(F3, c, d, a, b, in[15]+0x1fa27cf8, 16);
	MD5STEP(F3, b, c, d, a, in[ 2]+0xc4ac5665, 23);
	//
	MD5STEP(F4, a, b, c, d, in[ 0]+0xf4292244,  6);
	MD5STEP(F4, d, a, b, c, in[ 7]+0x432aff97, 10);
	MD5STEP(F4, c, d, a, b, in[14]+0xab9423a7, 15);
	MD5STEP(F4, b, c, d, a, in[ 5]+0xfc93a039, 21);
	MD5STEP(F4, a, b, c, d, in[12]+0x655b59c3,  6);
	MD5STEP(F4, d, a, b, c, in[ 3]+0x8f0ccc92, 10);
	MD5STEP(F4, c, d, a, b, in[10]+0xffeff47d, 15);
	MD5STEP(F4, b, c, d, a, in[ 1]+0x85845dd1, 21);
	MD5STEP(F4, a, b, c, d, in[ 8]+0x6fa87e4f,  6);
	MD5STEP(F4, d, a, b, c, in[15]+0xfe2ce6e0, 10);
	MD5STEP(F4, c, d, a, b, in[ 6]+0xa3014314, 15);
	MD5STEP(F4, b, c, d, a, in[13]+0x4e0811a1, 21);
	MD5STEP(F4, a, b, c, d, in[ 4]+0xf7537e82,  6);
	MD5STEP(F4, d, a, b, c, in[11]+0xbd3af235, 10);
	MD5STEP(F4, c, d, a, b, in[ 2]+0x2ad7d2bb, 15);
	MD5STEP(F4, b, c, d, a, in[ 9]+0xeb86d391, 21);
	//
	buf[0] += a;
	buf[1] += b;
	buf[2] += c;
	buf[3] += d;
}

__device__ static void MD5Init(MD5Context *ctx)
{
	ctx->IsInit = true;
	ctx->Buf[0] = 0x67452301;
	ctx->Buf[1] = 0xefcdab89;
	ctx->Buf[2] = 0x98badcfe;
	ctx->Buf[3] = 0x10325476;
	ctx->Bits[0] = 0;
	ctx->Bits[1] = 0;
}

__device__ static void MD5Update(MD5Context *ctx, const unsigned char *buf, unsigned int len)
{
	// Update bitcount
	uint32 t = ctx->Bits[0];
	if ((ctx->Bits[0] = t + ((uint32)len << 3)) < t)
		ctx->Bits[1]++; // Carry from low to high
	ctx->Bits[1] += len >> 29;

	t = (t >> 3) & 0x3f; // Bytes already in shsInfo->data

	// Handle any leading odd-sized chunks
	if (t)
	{
		unsigned char *p = (unsigned char *)ctx->In + t;
		t = 64-t;
		if (len < t)
		{
			_memcpy(p, buf, len);
			return;
		}
		_memcpy(p, buf, t);
		ByteReverse(ctx->In, 16);
		MD5Transform(ctx->Buf, (uint32 *)ctx->In);
		buf += t;
		len -= t;
	}

	// Process data in 64-byte chunks
	while (len >= 64)
	{
		_memcpy(ctx->In, buf, 64);
		ByteReverse(ctx->In, 16);
		MD5Transform(ctx->Buf, (uint32 *)ctx->In);
		buf += 64;
		len -= 64;
	}

	// Handle any remaining bytes of data.
	_memcpy(ctx->In, buf, len);
}

__device__ static void MD5Final(unsigned char digest[16], MD5Context *ctx)
{
	unsigned char *p;

	// Compute number of bytes mod 64
	unsigned count = (ctx->Bits[0] >> 3) & 0x3F;

	// Set the first char of padding to 0x80.  This is safe since there is always at least one byte free
	p = ctx->In + count;
	*p++ = 0x80;

	// Bytes of padding needed to make 64 bytes
	count = 64 - 1 - count;

	// Pad out to 56 mod 64
	if (count < 8)
	{
		// Two lots of padding:  Pad the first block to 64 bytes
		_memset(p, 0, count);
		ByteReverse(ctx->In, 16);
		MD5Transform(ctx->Buf, (uint32 *)ctx->In);
		_memset(ctx->In, 0, 56); // Now fill the next block with 56 bytes
	}
	else
		_memset(p, 0, count-8); // Pad block to 56 bytes
	ByteReverse(ctx->In, 14);

	// Append length in bits and transform
	((uint32 *)ctx->In)[14] = ctx->Bits[0];
	((uint32 *)ctx->In)[15] = ctx->Bits[1];

	MD5Transform(ctx->Buf, (uint32 *)ctx->In);
	ByteReverse((unsigned char *)ctx->Buf, 4);
	_memcpy(digest, ctx->Buf, 16);
	_memset(ctx, 0, sizeof(ctx)); // In case it is sensitive
}

// Convert a 128-bit MD5 digest into a 32-digit base-16 number.
__constant__ static char const _encode[] = "0123456789abcdef";
__device__ static void MD5DigestToBase16(unsigned char *digest, char *buf)
{
	int i, j;
	for (j = i = 0; i < 16; i++)
	{
		int a = digest[i];
		buf[j++] = _encode[(a>>4)&0xf];
		buf[j++] = _encode[a & 0xf];
	}
	buf[j] = 0;
}

// Convert a 128-bit MD5 digest into sequency of eight 5-digit integers each representing 16 bits of the digest and separated from each other by a "-" character.
__device__ static void MD5DigestToBase10x8(unsigned char digest[16], char digestAsText[50])
{
	int i, j;
	for (i = j = 0; i < 16; i+=2)
	{
		unsigned int x = digest[i]*256 + digest[i+1];
		if (i > 0) digestAsText[j++] = '-';
		_sprintf(&digestAsText[j], "%05u", x);
		j += 5;
	}
	digestAsText[j] = 0;
}

// A TCL command for md5.  The argument is the text to be hashed.  The Result is the hash in base64.  
__device__ static int md5_cmd(void *cd, Tcl_Interp *interp, int argc, const char **argv)
{
	if (argc != 2)
	{
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " TEXT\"", nullptr);
		return TCL_ERROR;
	}
	MD5Context ctx;
	MD5Init(&ctx);
	MD5Update(&ctx, (unsigned char *)argv[1], (unsigned)_strlen(argv[1]));
	unsigned char digest[16];
	MD5Final(digest, &ctx);
	void (*converter)(unsigned char*,char*);
	converter = (void(*)(unsigned char*,char*))cd;
	char buf[50];
	converter(digest, buf);
	Tcl_AppendResult(interp, buf, nullptr);
	return TCL_OK;
}

// A TCL command to take the md5 hash of a file.  The argument is the name of the file.
__device__ static int md5file_cmd(void *cd, Tcl_Interp *interp, int argc, const char **argv)
{
	if (argc != 2)
	{
		Tcl_AppendResult(interp,"wrong # args: should be \"", argv[0], " FILENAME\"", nullptr);
		return TCL_ERROR;
	}
	FILE *in = _fopen(argv[1], "rb");
	if (!in)
	{
		Tcl_AppendResult(interp,"unable to open file \"", argv[1], "\" for reading", nullptr);
		return TCL_ERROR;
	}
	MD5Context ctx;
	MD5Init(&ctx);
	char buf[10240];
	for (;;)
	{
		int n = (int)_fread(buf, 1, sizeof(buf), in);
		if (n <= 0) break;
		MD5Update(&ctx, (unsigned char *)buf, (unsigned)n);
	}
	_fclose(in);

	unsigned char digest[16];
	MD5Final(digest, &ctx);
	void (*converter)(unsigned char*,char*);
	converter = (void(*)(unsigned char*,char*))cd;
	converter(digest, buf);
	Tcl_AppendResult(interp, buf, nullptr);
	return TCL_OK;
}

// Register the four new TCL commands for generating MD5 checksums with the TCL interpreter.
__device__ int Md5_Init(Tcl_Interp *interp)
{
	Tcl_CreateCommand(interp, "md5", (Tcl_CmdProc *)md5_cmd, (ClientData)MD5DigestToBase16, nullptr);
	Tcl_CreateCommand(interp, "md5-10x8", (Tcl_CmdProc *)md5_cmd, (ClientData)MD5DigestToBase10x8, nullptr);
	Tcl_CreateCommand(interp, "md5file", (Tcl_CmdProc *)md5file_cmd, (ClientData)MD5DigestToBase16, nullptr);
	Tcl_CreateCommand(interp, "md5file-10x8", (Tcl_CmdProc *)md5file_cmd, (ClientData)MD5DigestToBase10x8, nullptr);
	return TCL_OK;
}

#endif

#if defined(_TEST)

// During testing, the special md5sum() aggregate function is available. inside SQLite.  The following routines implement that function.
__device__ static void md5step(FuncContext *fctx, int argc, Mem **argv)
{
	if (argc < 1) return;
	MD5Context *p = (MD5Context *)Vdbe::Aggregate_Context(fctx, sizeof(*p));
	if (!p) return;
	if (!p->IsInit)
		MD5Init(p);
	for (int i = 0; i < argc; i++)
	{
		const char *data = (char *)Vdbe::Value_Text(argv[i]);
		if (data)
			MD5Update(p, (unsigned char *)data, (int)_strlen(data));
	}
}
__device__ static void md5finalize(FuncContext *fctx)
{

	MD5Context *p = (MD5Context *)Vdbe::Aggregate_Context(fctx, sizeof(*p));
	unsigned char digest[16];
	MD5Final(digest, p);
	char buf[33];
	MD5DigestToBase16(digest, buf);
	Vdbe::Result_Text(fctx, buf, -1, DESTRUCTOR_TRANSIENT);
}

__device__ int Md5_Register(Context *ctx)
{
	int rc = Main::CreateFunction(ctx, "md5sum", -1, TEXTENCODE_UTF8, nullptr, nullptr, md5step, md5finalize);
	Main::OverloadFunction(ctx, "md5sum", -1); // To exercise this API
	return rc;
}
#endif
#pragma endregion

#pragma region Tests

#ifdef _TEST
__device__ static void init_all(Tcl_Interp *);

__device__ static int init_all_cmd(ClientData cd, Tcl_Interp *interp, int argc, char *args[])
{
	if (argc != 2)
	{
		Tcl_WrongNumArgs(interp, 1, args, "SLAVE");
		return TCL_ERROR;
	}
	Tcl_Interp *slave = interp; //Tcl_GetSlave(interp, args[1]);
	if (!slave)
		return TCL_ERROR;
	init_all(slave);
	return TCL_OK;
}

__device__ extern int Main_Init(Tcl_Interp *interp);
__device__ static void init_all(Tcl_Interp *interp)
{
	Main_Init(interp);
#if defined(_TEST) || defined(_TCLMD5)
	Md5_Init(interp);
#endif

	//	// Install the [register_dbstat_vtab] command to access the implementation of virtual table dbstat (source file test_stat.c). This command is
	//	// required for testfixture and sqlite3_analyzer, but not by the production Tcl extension.
	//#if defined(_TEST)
	//	extern int SqlitetestStat_Init(Tcl_Interp *);
	//	SqlitetestStat_Init(interp);
	//	extern int Sqliteconfig_Init(Tcl_Interp*);
	//	extern int Sqlitetest1_Init(Tcl_Interp*);
	//	extern int Sqlitetest2_Init(Tcl_Interp*);
	//	extern int Sqlitetest3_Init(Tcl_Interp*);
	//	extern int Sqlitetest4_Init(Tcl_Interp*);
	//	extern int Sqlitetest5_Init(Tcl_Interp*);
	//	extern int Sqlitetest6_Init(Tcl_Interp*);
	//	extern int Sqlitetest7_Init(Tcl_Interp*);
	//	extern int Sqlitetest8_Init(Tcl_Interp*);
	//	extern int Sqlitetest9_Init(Tcl_Interp*);
	//	extern int Sqlitetestasync_Init(Tcl_Interp*);
	//	extern int Sqlitetest_autoext_Init(Tcl_Interp*);
	//	extern int Sqlitetest_demovfs_Init(Tcl_Interp *);
	//	extern int Sqlitetest_func_Init(Tcl_Interp*);
	//	extern int Sqlitetest_hexio_Init(Tcl_Interp*);
	//	extern int Sqlitetest_init_Init(Tcl_Interp*);
	//	extern int Sqlitetest_malloc_Init(Tcl_Interp*);
	//	extern int Sqlitetest_mutex_Init(Tcl_Interp*);
	//	extern int Sqlitetestschema_Init(Tcl_Interp*);
	//	extern int Sqlitetestsse_Init(Tcl_Interp*);
	//	extern int Sqlitetesttclvar_Init(Tcl_Interp*);
	//	extern int Sqlitetestfs_Init(Tcl_Interp*);
	//	extern int SqlitetestThread_Init(Tcl_Interp*);
	//	extern int SqlitetestOnefile_Init(Tcl_Interp*);
	//	extern int SqlitetestOsinst_Init(Tcl_Interp*);
	//	extern int Sqlitetestbackup_Init(Tcl_Interp*);
	//	extern int Sqlitetestintarray_Init(Tcl_Interp*);
	//	extern int Sqlitetestvfs_Init(Tcl_Interp *);
	//	extern int Sqlitetestrtree_Init(Tcl_Interp*);
	//	extern int Sqlitequota_Init(Tcl_Interp*);
	//	extern int Sqlitemultiplex_Init(Tcl_Interp*);
	//	extern int SqliteSuperlock_Init(Tcl_Interp*);
	//	extern int SqlitetestSyscall_Init(Tcl_Interp*);
	//	extern int Sqlitetestfuzzer_Init(Tcl_Interp*);
	//	extern int Sqlitetestwholenumber_Init(Tcl_Interp*);
	//	extern int Sqlitetestregexp_Init(Tcl_Interp*);
	//#if defined(ENABLE_FTS3) || defined(ENABLE_FTS4)
	//	extern int Sqlitetestfts3_Init(Tcl_Interp *interp);
	//#endif
	//#ifdef ENABLE_ZIPVFS
	//	extern int Zipvfs_Init(Tcl_Interp*);
	//	Zipvfs_Init(interp);
	//#endif
	//	Sqliteconfig_Init(interp);
	//	Sqlitetest1_Init(interp);
	//	Sqlitetest2_Init(interp);
	//	Sqlitetest3_Init(interp);
	//	Sqlitetest4_Init(interp);
	//	Sqlitetest5_Init(interp);
	//	Sqlitetest6_Init(interp);
	//	Sqlitetest7_Init(interp);
	//	Sqlitetest8_Init(interp);
	//	Sqlitetest9_Init(interp);
	//	Sqlitetestasync_Init(interp);
	//	Sqlitetest_autoext_Init(interp);
	//	Sqlitetest_demovfs_Init(interp);
	//	Sqlitetest_func_Init(interp);
	//	Sqlitetest_hexio_Init(interp);
	//	Sqlitetest_init_Init(interp);
	//	Sqlitetest_malloc_Init(interp);
	//	Sqlitetest_mutex_Init(interp);
	//	Sqlitetestschema_Init(interp);
	//	Sqlitetesttclvar_Init(interp);
	//	Sqlitetestfs_Init(interp);
	//	SqlitetestThread_Init(interp);
	//	SqlitetestOnefile_Init(interp);
	//	SqlitetestOsinst_Init(interp);
	//	Sqlitetestbackup_Init(interp);
	//	Sqlitetestintarray_Init(interp);
	//	Sqlitetestvfs_Init(interp);
	//	Sqlitetestrtree_Init(interp);
	//	Sqlitequota_Init(interp);
	//	Sqlitemultiplex_Init(interp);
	//	SqliteSuperlock_Init(interp);
	//	SqlitetestSyscall_Init(interp);
	//	Sqlitetestfuzzer_Init(interp);
	//	Sqlitetestwholenumber_Init(interp);
	//	Sqlitetestregexp_Init(interp);
	//#if defined(ENABLE_FTS3) || defined(ENABLE_FTS4)
	//	Sqlitetestfts3_Init(interp);
	//#endif

	Tcl_CreateCommand(interp, "load_testfixture_extensions", (Tcl_CmdProc *)init_all_cmd, nullptr, nullptr);
#ifdef _SSE
	Sqlitetestsse_Init(interp);
#endif
#endif
}

#pragma endregion

#if 1
__device__ static char *tclsh_main_loop()
{
	return
		"set line {}\n"
		"while {![eof stdin]} {\n"
		"if {$line!=\"\"} {\n"
		"puts -nonewline \"> \"\n"
		"} else {\n"
		"puts -nonewline \"% \"\n"
		"}\n"
		"flush stdout\n"
		"append line [gets stdin]\n"
		"if {[info complete $line]} {\n"
		"if {[catch {uplevel #0 $line} result]} {\n"
		"puts stderr \"Error: $result\"\n"
		"} elseif {$result!=\"\"} {\n"
		"puts $result\n"
		"}\n"
		"set line {}\n"
		"} else {\n"
		"append line \\n\n"
		"}\n"
		"}\n";
}
#else
__device__ static char *tclsh_main_loop();
#endif

cudaDeviceHeap CudaInit()
{
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*3));
	return cudaDeviceHeapCreate(256, 4096);
}

static void CudaShutdown(cudaDeviceHeap deviceHeap)
{
	cudaDeviceHeapDestroy(deviceHeap);
	cudaDeviceReset();
}

int main(int argc, char **argv)
{
#if __CUDACC__
	cudaDeviceHeap deviceHeap = CudaInit();
#endif

	// Call sqlite3_shutdown() once before doing anything else. This is to test that sqlite3_shutdown() can be safely called by a process before sqlite3_initialize() is. */
	Main::Shutdown();

	//Tcl_FindExecutable(argv[0]);
	Tcl_Interp *interp = Tcl_CreateInterp();
	SysEx::Config(SysEx::CONFIG_SINGLETHREAD);

	init_all(interp);
	if (argc >= 2)
	{
		char b[32];
		__snprintf(b, sizeof(b), "%d", argc-2);
		Tcl_SetVar(interp, "argc", b, TCL_GLOBAL_ONLY);
		Tcl_SetVar(interp, "argv0", argv[1], TCL_GLOBAL_ONLY);
		Tcl_SetVar(interp, "argv", "", TCL_GLOBAL_ONLY);
		for (int i = 3; i < argc; i++)
			Tcl_SetVar(interp, "argv", argv[i], TCL_GLOBAL_ONLY | TCL_LIST_ELEMENT | TCL_APPEND_VALUE);
		if (Tcl_EvalFile(interp, argv[1]) != TCL_OK)
		{
			const char *info = Tcl_GetVar(interp, "errorInfo", TCL_GLOBAL_ONLY);
			if (!info) info = interp->result;
			fprintf(stderr, "%s: %s\n", *argv, info);
			return 1;
		}
	}
	if (argc <= 1)
		Tcl_GlobalEval(interp, tclsh_main_loop());

#if __CUDACC__
	CudaShutdown(deviceHeap);
#endif
	printf("\nEnd."); char c; scanf("%c", &c);
	return 0;
}

