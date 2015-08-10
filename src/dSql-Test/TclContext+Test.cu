#ifdef TCLSH
/*****************************************************************************
** All of the code that follows is used to build standalone TCL interpreters
** that are statically linked with SQLite.  Enable these by compiling
** with -DTCLSH=n where n can be 1 or 2.  An n of 1 generates a standard
** tclsh but with SQLite built in.  An n of 2 generates the SQLite space
** analysis program.
*/

#if defined(SQLITE_TEST) || defined(SQLITE_TCLMD5)
/*
* This code implements the MD5 message-digest algorithm.
* The algorithm is due to Ron Rivest.  This code was
* written by Colin Plumb in 1993, no copyright is claimed.
* This code is in the public domain; do with it what you wish.
*
* Equivalent code is available from RSA Data Security, Inc.
* This code has been tested against that, and is equivalent,
* except that you don't need to include two pages of legalese
* with every copy.
*
* To compute the message digest of a chunk of bytes, declare an
* MD5Context structure, pass it to MD5Init, call MD5Update as
* needed on buffers full of bytes, and then call MD5Final, which
* will fill a supplied 16-byte array with the digest.
*/

/*
* If compiled on a machine that doesn't have a 32-bit integer,
* you just set "uint32" to the appropriate datatype for an
* unsigned 32-bit integer.  For example:
*
*       cc -Duint32='unsigned long' md5.c
*
*/
#ifndef uint32
#  define uint32 unsigned int
#endif

struct MD5Context {
	int isInit;
	uint32 buf[4];
	uint32 bits[2];
	unsigned char in[64];
};
typedef struct MD5Context MD5Context;

/*
* Note: this code is harmless on little-endian machines.
*/
static void byteReverse (unsigned char *buf, unsigned longs){
	uint32 t;
	do {
		t = (uint32)((unsigned)buf[3]<<8 | buf[2]) << 16 |
			((unsigned)buf[1]<<8 | buf[0]);
		*(uint32 *)buf = t;
		buf += 4;
	} while (--longs);
}
/* The four core functions - F1 is optimized somewhat */

/* #define F1(x, y, z) (x & y | ~x & z) */
#define F1(x, y, z) (z ^ (x & (y ^ z)))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))

/* This is the central step in the MD5 algorithm. */
#define MD5STEP(f, w, x, y, z, data, s) \
	( w += f(x, y, z) + data,  w = w<<s | w>>(32-s),  w += x )

/*
* The core of the MD5 algorithm, this alters an existing MD5 hash to
* reflect the addition of 16 longwords of new data.  MD5Update blocks
* the data and converts bytes into longwords for this routine.
*/
static void MD5Transform(uint32 buf[4], const uint32 in[16]){
	register uint32 a, b, c, d;

	a = buf[0];
	b = buf[1];
	c = buf[2];
	d = buf[3];

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

	buf[0] += a;
	buf[1] += b;
	buf[2] += c;
	buf[3] += d;
}

/*
* Start MD5 accumulation.  Set bit count to 0 and buffer to mysterious
* initialization constants.
*/
static void MD5Init(MD5Context *ctx){
	ctx->isInit = 1;
	ctx->buf[0] = 0x67452301;
	ctx->buf[1] = 0xefcdab89;
	ctx->buf[2] = 0x98badcfe;
	ctx->buf[3] = 0x10325476;
	ctx->bits[0] = 0;
	ctx->bits[1] = 0;
}

/*
* Update context to reflect the concatenation of another buffer full
* of bytes.
*/
static 
	void MD5Update(MD5Context *ctx, const unsigned char *buf, unsigned int len){
		uint32 t;

		/* Update bitcount */

		t = ctx->bits[0];
		if ((ctx->bits[0] = t + ((uint32)len << 3)) < t)
			ctx->bits[1]++; /* Carry from low to high */
		ctx->bits[1] += len >> 29;

		t = (t >> 3) & 0x3f;    /* Bytes already in shsInfo->data */

		/* Handle any leading odd-sized chunks */

		if ( t ) {
			unsigned char *p = (unsigned char *)ctx->in + t;

			t = 64-t;
			if (len < t) {
				memcpy(p, buf, len);
				return;
			}
			memcpy(p, buf, t);
			byteReverse(ctx->in, 16);
			MD5Transform(ctx->buf, (uint32 *)ctx->in);
			buf += t;
			len -= t;
		}

		/* Process data in 64-byte chunks */

		while (len >= 64) {
			memcpy(ctx->in, buf, 64);
			byteReverse(ctx->in, 16);
			MD5Transform(ctx->buf, (uint32 *)ctx->in);
			buf += 64;
			len -= 64;
		}

		/* Handle any remaining bytes of data. */

		memcpy(ctx->in, buf, len);
}

/*
* Final wrapup - pad to 64-byte boundary with the bit pattern 
* 1 0* (64-bit count of bits processed, MSB-first)
*/
static void MD5Final(unsigned char digest[16], MD5Context *ctx){
	unsigned count;
	unsigned char *p;

	/* Compute number of bytes mod 64 */
	count = (ctx->bits[0] >> 3) & 0x3F;

	/* Set the first char of padding to 0x80.  This is safe since there is
	always at least one byte free */
	p = ctx->in + count;
	*p++ = 0x80;

	/* Bytes of padding needed to make 64 bytes */
	count = 64 - 1 - count;

	/* Pad out to 56 mod 64 */
	if (count < 8) {
		/* Two lots of padding:  Pad the first block to 64 bytes */
		memset(p, 0, count);
		byteReverse(ctx->in, 16);
		MD5Transform(ctx->buf, (uint32 *)ctx->in);

		/* Now fill the next block with 56 bytes */
		memset(ctx->in, 0, 56);
	} else {
		/* Pad block to 56 bytes */
		memset(p, 0, count-8);
	}
	byteReverse(ctx->in, 14);

	/* Append length in bits and transform */
	((uint32 *)ctx->in)[ 14 ] = ctx->bits[0];
	((uint32 *)ctx->in)[ 15 ] = ctx->bits[1];

	MD5Transform(ctx->buf, (uint32 *)ctx->in);
	byteReverse((unsigned char *)ctx->buf, 4);
	memcpy(digest, ctx->buf, 16);
	memset(ctx, 0, sizeof(ctx));    /* In case it is sensitive */
}

/*
** Convert a 128-bit MD5 digest into a 32-digit base-16 number.
*/
static void MD5DigestToBase16(unsigned char *digest, char *zBuf){
	static char const zEncode[] = "0123456789abcdef";
	int i, j;

	for(j=i=0; i<16; i++){
		int a = digest[i];
		zBuf[j++] = zEncode[(a>>4)&0xf];
		zBuf[j++] = zEncode[a & 0xf];
	}
	zBuf[j] = 0;
}


/*
** Convert a 128-bit MD5 digest into sequency of eight 5-digit integers
** each representing 16 bits of the digest and separated from each
** other by a "-" character.
*/
static void MD5DigestToBase10x8(unsigned char digest[16], char zDigest[50]){
	int i, j;
	unsigned int x;
	for(i=j=0; i<16; i+=2){
		x = digest[i]*256 + digest[i+1];
		if( i>0 ) zDigest[j++] = '-';
		sprintf(&zDigest[j], "%05u", x);
		j += 5;
	}
	zDigest[j] = 0;
}

/*
** A TCL command for md5.  The argument is the text to be hashed.  The
** Result is the hash in base64.  
*/
static int md5_cmd(void*cd, Tcl_Interp *interp, int argc, const char **argv){
	MD5Context ctx;
	unsigned char digest[16];
	char zBuf[50];
	void (*converter)(unsigned char*, char*);

	if( argc!=2 ){
		Tcl_AppendResult(interp,"wrong # args: should be \"", argv[0], 
			" TEXT\"", 0);
		return TCL_ERROR;
	}
	MD5Init(&ctx);
	MD5Update(&ctx, (unsigned char*)argv[1], (unsigned)strlen(argv[1]));
	MD5Final(digest, &ctx);
	converter = (void(*)(unsigned char*,char*))cd;
	converter(digest, zBuf);
	Tcl_AppendResult(interp, zBuf, (char*)0);
	return TCL_OK;
}

/*
** A TCL command to take the md5 hash of a file.  The argument is the
** name of the file.
*/
static int md5file_cmd(void*cd, Tcl_Interp*interp, int argc, const char **argv){
	FILE *in;
	MD5Context ctx;
	void (*converter)(unsigned char*, char*);
	unsigned char digest[16];
	char zBuf[10240];

	if( argc!=2 ){
		Tcl_AppendResult(interp,"wrong # args: should be \"", argv[0], 
			" FILENAME\"", 0);
		return TCL_ERROR;
	}
	in = fopen(argv[1],"rb");
	if( in==0 ){
		Tcl_AppendResult(interp,"unable to open file \"", argv[1], 
			"\" for reading", 0);
		return TCL_ERROR;
	}
	MD5Init(&ctx);
	for(;;){
		int n;
		n = (int)fread(zBuf, 1, sizeof(zBuf), in);
		if( n<=0 ) break;
		MD5Update(&ctx, (unsigned char*)zBuf, (unsigned)n);
	}
	fclose(in);
	MD5Final(digest, &ctx);
	converter = (void(*)(unsigned char*,char*))cd;
	converter(digest, zBuf);
	Tcl_AppendResult(interp, zBuf, (char*)0);
	return TCL_OK;
}

/*
** Register the four new TCL commands for generating MD5 checksums
** with the TCL interpreter.
*/
int Md5_Init(Tcl_Interp *interp){
	Tcl_CreateCommand(interp, "md5", (Tcl_CmdProc*)md5_cmd,
		MD5DigestToBase16, 0);
	Tcl_CreateCommand(interp, "md5-10x8", (Tcl_CmdProc*)md5_cmd,
		MD5DigestToBase10x8, 0);
	Tcl_CreateCommand(interp, "md5file", (Tcl_CmdProc*)md5file_cmd,
		MD5DigestToBase16, 0);
	Tcl_CreateCommand(interp, "md5file-10x8", (Tcl_CmdProc*)md5file_cmd,
		MD5DigestToBase10x8, 0);
	return TCL_OK;
}
#endif /* defined(SQLITE_TEST) || defined(SQLITE_TCLMD5) */

#if defined(SQLITE_TEST)
/*
** During testing, the special md5sum() aggregate function is available.
** inside SQLite.  The following routines implement that function.
*/
static void md5step(sqlite3_context *context, int argc, sqlite3_value **argv){
	MD5Context *p;
	int i;
	if( argc<1 ) return;
	p = sqlite3_aggregate_context(context, sizeof(*p));
	if( p==0 ) return;
	if( !p->isInit ){
		MD5Init(p);
	}
	for(i=0; i<argc; i++){
		const char *zData = (char*)sqlite3_value_text(argv[i]);
		if( zData ){
			MD5Update(p, (unsigned char*)zData, (int)strlen(zData));
		}
	}
}
static void md5finalize(sqlite3_context *context){
	MD5Context *p;
	unsigned char digest[16];
	char zBuf[33];
	p = sqlite3_aggregate_context(context, sizeof(*p));
	MD5Final(digest,p);
	MD5DigestToBase16(digest, zBuf);
	sqlite3_result_text(context, zBuf, -1, SQLITE_TRANSIENT);
}
int Md5_Register(sqlite3 *db){
	int rc = sqlite3_create_function(db, "md5sum", -1, SQLITE_UTF8, 0, 0, 
		md5step, md5finalize);
	sqlite3_overload_function(db, "md5sum", -1);  /* To exercise this API */
	return rc;
}
#endif /* defined(SQLITE_TEST) */


/*
** If the macro TCLSH is one, then put in code this for the
** "main" routine that will initialize Tcl and take input from
** standard input, or if a file is named on the command line
** the TCL interpreter reads and evaluates that file.
*/
#if TCLSH==1
static const char *tclsh_main_loop(void){
	static const char zMainloop[] =
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
		"}\n"
		;
	return zMainloop;
}
#endif
#if TCLSH==2
static const char *tclsh_main_loop(void);
#endif

#ifdef SQLITE_TEST
static void init_all(Tcl_Interp *);
static int init_all_cmd(
	ClientData cd,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){

	Tcl_Interp *slave;
	if( objc!=2 ){
		Tcl_WrongNumArgs(interp, 1, objv, "SLAVE");
		return TCL_ERROR;
	}

	slave = Tcl_GetSlave(interp, Tcl_GetString(objv[1]));
	if( !slave ){
		return TCL_ERROR;
	}

	init_all(slave);
	return TCL_OK;
}

/*
** Tclcmd: db_use_legacy_prepare DB BOOLEAN
**
**   The first argument to this command must be a database command created by
**   [sqlite3]. If the second argument is true, then the handle is configured
**   to use the sqlite3_prepare_v2() function to prepare statements. If it
**   is false, sqlite3_prepare().
*/
static int db_use_legacy_prepare_cmd(
	ClientData cd,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	Tcl_CmdInfo cmdInfo;
	SqliteDb *pDb;
	int bPrepare;

	if( objc!=3 ){
		Tcl_WrongNumArgs(interp, 1, objv, "DB BOOLEAN");
		return TCL_ERROR;
	}

	if( !Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &cmdInfo) ){
		Tcl_AppendResult(interp, "no such db: ", Tcl_GetString(objv[1]), (char*)0);
		return TCL_ERROR;
	}
	pDb = (SqliteDb*)cmdInfo.objClientData;
	if( Tcl_GetBooleanFromObj(interp, objv[2], &bPrepare) ){
		return TCL_ERROR;
	}

	pDb->bLegacyPrepare = bPrepare;

	Tcl_ResetResult(interp);
	return TCL_OK;
}
#endif

/*
** Configure the interpreter passed as the first argument to have access
** to the commands and linked variables that make up:
**
**   * the [sqlite3] extension itself, 
**
**   * If SQLITE_TCLMD5 or SQLITE_TEST is defined, the Md5 commands, and
**
**   * If SQLITE_TEST is set, the various test interfaces used by the Tcl
**     test suite.
*/
static void init_all(Tcl_Interp *interp){
	Sqlite3_Init(interp);

#if defined(SQLITE_TEST) || defined(SQLITE_TCLMD5)
	Md5_Init(interp);
#endif

	/* Install the [register_dbstat_vtab] command to access the implementation
	** of virtual table dbstat (source file test_stat.c). This command is
	** required for testfixture and sqlite3_analyzer, but not by the production
	** Tcl extension.  */
#if defined(SQLITE_TEST) || TCLSH==2
	{
		extern int SqlitetestStat_Init(Tcl_Interp*);
		SqlitetestStat_Init(interp);
	}
#endif

#ifdef SQLITE_TEST
	{
		extern int Sqliteconfig_Init(Tcl_Interp*);
		extern int Sqlitetest1_Init(Tcl_Interp*);
		extern int Sqlitetest2_Init(Tcl_Interp*);
		extern int Sqlitetest3_Init(Tcl_Interp*);
		extern int Sqlitetest4_Init(Tcl_Interp*);
		extern int Sqlitetest5_Init(Tcl_Interp*);
		extern int Sqlitetest6_Init(Tcl_Interp*);
		extern int Sqlitetest7_Init(Tcl_Interp*);
		extern int Sqlitetest8_Init(Tcl_Interp*);
		extern int Sqlitetest9_Init(Tcl_Interp*);
		extern int Sqlitetestasync_Init(Tcl_Interp*);
		extern int Sqlitetest_autoext_Init(Tcl_Interp*);
		extern int Sqlitetest_demovfs_Init(Tcl_Interp *);
		extern int Sqlitetest_func_Init(Tcl_Interp*);
		extern int Sqlitetest_hexio_Init(Tcl_Interp*);
		extern int Sqlitetest_init_Init(Tcl_Interp*);
		extern int Sqlitetest_malloc_Init(Tcl_Interp*);
		extern int Sqlitetest_mutex_Init(Tcl_Interp*);
		extern int Sqlitetestschema_Init(Tcl_Interp*);
		extern int Sqlitetestsse_Init(Tcl_Interp*);
		extern int Sqlitetesttclvar_Init(Tcl_Interp*);
		extern int Sqlitetestfs_Init(Tcl_Interp*);
		extern int SqlitetestThread_Init(Tcl_Interp*);
		extern int SqlitetestOnefile_Init();
		extern int SqlitetestOsinst_Init(Tcl_Interp*);
		extern int Sqlitetestbackup_Init(Tcl_Interp*);
		extern int Sqlitetestintarray_Init(Tcl_Interp*);
		extern int Sqlitetestvfs_Init(Tcl_Interp *);
		extern int Sqlitetestrtree_Init(Tcl_Interp*);
		extern int Sqlitequota_Init(Tcl_Interp*);
		extern int Sqlitemultiplex_Init(Tcl_Interp*);
		extern int SqliteSuperlock_Init(Tcl_Interp*);
		extern int SqlitetestSyscall_Init(Tcl_Interp*);
		extern int Sqlitetestfuzzer_Init(Tcl_Interp*);
		extern int Sqlitetestwholenumber_Init(Tcl_Interp*);
		extern int Sqlitetestregexp_Init(Tcl_Interp*);

#if defined(SQLITE_ENABLE_FTS3) || defined(SQLITE_ENABLE_FTS4)
		extern int Sqlitetestfts3_Init(Tcl_Interp *interp);
#endif

#ifdef SQLITE_ENABLE_ZIPVFS
		extern int Zipvfs_Init(Tcl_Interp*);
		Zipvfs_Init(interp);
#endif

		Sqliteconfig_Init(interp);
		Sqlitetest1_Init(interp);
		Sqlitetest2_Init(interp);
		Sqlitetest3_Init(interp);
		Sqlitetest4_Init(interp);
		Sqlitetest5_Init(interp);
		Sqlitetest6_Init(interp);
		Sqlitetest7_Init(interp);
		Sqlitetest8_Init(interp);
		Sqlitetest9_Init(interp);
		Sqlitetestasync_Init(interp);
		Sqlitetest_autoext_Init(interp);
		Sqlitetest_demovfs_Init(interp);
		Sqlitetest_func_Init(interp);
		Sqlitetest_hexio_Init(interp);
		Sqlitetest_init_Init(interp);
		Sqlitetest_malloc_Init(interp);
		Sqlitetest_mutex_Init(interp);
		Sqlitetestschema_Init(interp);
		Sqlitetesttclvar_Init(interp);
		Sqlitetestfs_Init(interp);
		SqlitetestThread_Init(interp);
		SqlitetestOnefile_Init(interp);
		SqlitetestOsinst_Init(interp);
		Sqlitetestbackup_Init(interp);
		Sqlitetestintarray_Init(interp);
		Sqlitetestvfs_Init(interp);
		Sqlitetestrtree_Init(interp);
		Sqlitequota_Init(interp);
		Sqlitemultiplex_Init(interp);
		SqliteSuperlock_Init(interp);
		SqlitetestSyscall_Init(interp);
		Sqlitetestfuzzer_Init(interp);
		Sqlitetestwholenumber_Init(interp);
		Sqlitetestregexp_Init(interp);

#if defined(SQLITE_ENABLE_FTS3) || defined(SQLITE_ENABLE_FTS4)
		Sqlitetestfts3_Init(interp);
#endif

		Tcl_CreateObjCommand(
			interp, "load_testfixture_extensions", init_all_cmd, 0, 0
			);
		Tcl_CreateObjCommand(
			interp, "db_use_legacy_prepare", db_use_legacy_prepare_cmd, 0, 0
			);

#ifdef SQLITE_SSE
		Sqlitetestsse_Init(interp);
#endif
	}
#endif
}

#define TCLSH_MAIN main   /* Needed to fake out mktclapp */
int TCLSH_MAIN(int argc, char **argv){
	Tcl_Interp *interp;

	/* Call sqlite3_shutdown() once before doing anything else. This is to
	** test that sqlite3_shutdown() can be safely called by a process before
	** sqlite3_initialize() is. */
	sqlite3_shutdown();

	Tcl_FindExecutable(argv[0]);
	interp = Tcl_CreateInterp();

#if TCLSH==2
	sqlite3_config(SQLITE_CONFIG_SINGLETHREAD);
#endif

	init_all(interp);
	if( argc>=2 ){
		int i;
		char zArgc[32];
		sqlite3_snprintf(sizeof(zArgc), zArgc, "%d", argc-(3-TCLSH));
		Tcl_SetVar(interp,"argc", zArgc, TCL_GLOBAL_ONLY);
		Tcl_SetVar(interp,"argv0",argv[1],TCL_GLOBAL_ONLY);
		Tcl_SetVar(interp,"argv", "", TCL_GLOBAL_ONLY);
		for(i=3-TCLSH; i<argc; i++){
			Tcl_SetVar(interp, "argv", argv[i],
				TCL_GLOBAL_ONLY | TCL_LIST_ELEMENT | TCL_APPEND_VALUE);
		}
		if( TCLSH==1 && Tcl_EvalFile(interp, argv[1])!=TCL_OK ){
			const char *zInfo = Tcl_GetVar(interp, "errorInfo", TCL_GLOBAL_ONLY);
			if( zInfo==0 ) zInfo = Tcl_GetStringResult(interp);
			fprintf(stderr,"%s: %s\n", *argv, zInfo);
			return 1;
		}
	}
	if( TCLSH==2 || argc<=1 ){
		Tcl_GlobalEval(interp, tclsh_main_loop());
	}
	return 0;
}
#endif /* TCLSH */
