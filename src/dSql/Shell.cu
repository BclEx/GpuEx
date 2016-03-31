//#define VISUAL
//ParserTrace(stdout, "p: ");
#pragma region PREAMBLE

#if (defined(_WIN32) || defined(WIN32)) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS // This needs to come before any includes for MSVC compiler
#endif

// Enable large-file support for fopen() and friends on unix.
#ifndef DISABLE_LFS
#define _LARGE_FILE 1
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#define _LARGEFILE_SOURCE 1
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <RuntimeHost.h>
#include <Core+Vdbe\VdbeInt.cu.h>

#if !defined(_WIN32) && !defined(WIN32)
#include <signal.h>
#if !defined(__RTP__) && !defined(_WRS_KERNEL)
#include <pwd.h>
#endif
#include <unistd.h>
#include <sys/types.h>
#endif

#ifdef HAVE_EDITLINE
#include <editline/editline.h>
#endif
#if defined(HAVE_READLINE) && HAVE_READLINE==1
#include <readline/readline.h>
#include <readline/history.h>
#endif
#if !defined(HAVE_EDITLINE) && (!defined(HAVE_READLINE) || HAVE_READLINE!=1)
#define readline(p) LocalGetLine(p, stdin, 0)
#define add_history(X)
#define ReadHistory(X)
#define write_history(X)
#define stifle_history(X)
#endif

#if defined(_WIN32) || defined(WIN32)
#include <io.h>
#define isatty(h) _isatty(h)
#define access(f,m) _access((f),(m))
#undef popen
#define popen(a,b) _popen((a),(b))
#undef pclose
#define pclose(x) _pclose(x)
#else
extern int isatty(int); // Make sure isatty() has a prototype.
#endif
#if defined(_WIN32_WCE)
// Windows CE (arm-wince-mingw32ce-gcc) does not provide isatty() thus we always assume that we have a console. That can be overridden with the -batch command line option.
#define isatty(x) 1
#endif

// ctype macros that work with signed characters
#define IsSpace(X)  _isspace((unsigned char)X)
#define IsDigit(X)  _isdigit((unsigned char)X)
#define ToLower(X)  (char)__tolower((unsigned char)X)
#pragma endregion

#pragma region TIMER
static bool _enableTimer = false; // True if the timer is enabled

#if !defined(_WIN32) && !defined(WIN32) && !defined(_WRS_KERNEL) && !defined(__minux)
#include <sys/time.h>
#include <sys/resource.h>

static struct rusage _sBegin; // Saved resource information for the beginning of an operation
static void BeginTimer()
{
	if (_enableTimer)
		getrusage(RUSAGE_SELF, &_sBegin);
}

static double TimeDiff(timeval *start, timeval *end)
{
	return (end->tv_usec - start->tv_usec)*0.000001 + (double)(end->tv_sec - start->tv_sec);
}

static void EndTimer()
{
	if (_enableTimer)
	{
		rusage sEnd;
		getrusage(RUSAGE_SELF, &sEnd);
		printf("CPU Time: user %f sys %f\n",
			TimeDiff(&_sBegin.ru_utime, &sEnd.ru_utime),
			TimeDiff(&_sBegin.ru_stime, &sEnd.ru_stime));
	}
}

#define BEGIN_TIMER BeginTimer()
#define END_TIMER EndTimer()
#define HAS_TIMER 1

#elif (defined(_WIN32) || defined(WIN32))

#include <windows.h>

// Saved resource information for the beginning of an operation
static HANDLE _hProcess;
static FILETIME _ftKernelBegin;
static FILETIME _ftUserBegin;
typedef BOOL (WINAPI *GETPROCTIMES)(HANDLE, LPFILETIME, LPFILETIME, LPFILETIME, LPFILETIME);
static GETPROCTIMES _getProcessTimesAddr = nullptr;

static int HasTimer()
{
	if (_getProcessTimesAddr)
		return true;
	// GetProcessTimes() isn't supported in WIN95 and some other Windows versions. See if the version we are running on has it, and if it does, save off
	// a pointer to it and the current process handle.
	_hProcess = GetCurrentProcess();
	if (_hProcess)
	{
		HINSTANCE hinstLib = LoadLibrary(TEXT("Kernel32.dll"));
		if (hinstLib)
		{
			_getProcessTimesAddr = (GETPROCTIMES)GetProcAddress(hinstLib, "GetProcessTimes");
			if (_getProcessTimesAddr)
				return true;
			FreeLibrary(hinstLib); 
		}
	}
	return false;
}

static void BeginTimer()
{
	if (_enableTimer && _getProcessTimesAddr)
	{
		FILETIME ftCreation, ftExit;
		_getProcessTimesAddr(_hProcess, &ftCreation, &ftExit, &_ftKernelBegin, &_ftUserBegin);
	}
}

static double TimeDiff(FILETIME *start, FILETIME *end)
{
	int64 i64Start = *((int64 *)start);
	int64 i64End = *((int64 *)end);
	return (double)((i64End - i64Start) / 10000000.0);
}

static void EndTimer()
{
	if (_enableTimer && _getProcessTimesAddr)
	{
		FILETIME ftCreation, ftExit, ftKernelEnd, ftUserEnd;
		_getProcessTimesAddr(_hProcess, &ftCreation, &ftExit, &ftKernelEnd, &ftUserEnd);
		printf("CPU Time: user %f sys %f\n", TimeDiff(&_ftUserBegin, &ftUserEnd), TimeDiff(&_ftKernelBegin, &ftKernelEnd));
	}
}

#define BEGIN_TIMER BeginTimer()
#define END_TIMER EndTimer()
#define HAS_TIMER HasTimer()
#else
#define BEGIN_TIMER 
#define END_TIMER
#define HAS_TIMER false
#endif
#pragma endregion

static int _bailOnError = 0; // If the following flag is set, then command execution stops at an error if we are not interactive.
static int _stdinIsInteractive = 1; // Threat stdin as an interactive input if the following variable is true.  Otherwise, assume stdin is connected to a file or pipe.
__device__ static Context *_ctx = nullptr; // The following is the open SQLite database.  We make a pointer to this database a static variable so that it can be accessed by the SIGINT handler to interrupt database processing.
static volatile int _seenInterrupt = 0; // True if an interrupt (Control-C) has been received.
static char *Argv0; // This is the name of our program. It is set in main(), used in a number of other places, mostly for error messages.

// Prompt strings. Initialized in main. Settable with .prompt main continue
static char _mainPrompt[20];     // First line prompt. default: "sqlite> "
static char _continuePrompt[20]; // Continuation prompt. default: "   ...> "

#pragma region Name

// Write I/O traces to the following stream.
#ifdef ENABLE_IOTRACE
static FILE *iotrace = nullptr;

static void iotracePrintf(const char *fmt, ...)
{
	if (!iotrace) return;
	_va_list args;
	_va_start(args, fmt);
	char *z = _vmprintf(fmt, ap);
	_va_end(ap);
	fprintf(iotrace, "%s", z);
	_free(z);
}
#endif

__device__ static bool isNumber(const char *z, int *realnum)
{
	if (*z == '-' || *z == '+') z++;
	if (!IsDigit(*z))
		return 0;
	z++;
	if (realnum) *realnum = 0;
	while (IsDigit(*z)) { z++; }
	if (*z == '.')
	{
		z++;
		if (!IsDigit(*z)) return 0;
		while (IsDigit(*z)) { z++; }
		if (realnum) *realnum = 1;
	}
	if (*z == 'e' || *z == 'E')
	{
		z++;
		if (*z == '+' || *z == '-') z++;
		if (!IsDigit(*z)) return 0;
		while (IsDigit(*z)) { z++; }
		if (realnum) *realnum = 1;
	}
	return (*z == 0);
}

__device__ static const char *_shellStatic = nullptr;
__device__ static void ShellStaticFunc(FuncContext *fctx, int argc, Mem **argv)
{
	_assert(argc == 0);
	_assert(_shellStatic);
	Vdbe::Result_Text(fctx, _shellStatic, -1, DESTRUCTOR_STATIC);
}

static char *LocalGetLine(char *prompt, FILE *in, int csvFlag)
{
	if (prompt && *prompt)
	{
		printf("%s", prompt);
		fflush(stdout);
	}
	int lineLength = 100;
	char *line = (char *)malloc(lineLength);
	if (!line) return nullptr;
	int n = 0;
	bool inQuote = false;
	while (1)
	{
		if (n+100 > lineLength)
		{
			lineLength = lineLength*2 + 100;
			line = (char *)realloc(line, lineLength);
			if (!line) return nullptr;
		}
		if (!fgets(&line[n], lineLength - n, in))
		{
			if (n == 0)
			{
				free(line);
				return nullptr;
			}
			line[n] = 0;
			break;
		}
		while (line[n])
		{
			if (line[n] == '"') inQuote = !inQuote;
			n++;
		}
		if (n > 0 && line[n-1] == '\n' && (!inQuote || !csvFlag))
		{
			n--;
			if (n > 0 && line[n-1] == '\r') n--;
			line[n] = 0;
			break;
		}
	}
	line = (char *)realloc(line, n+1);
	return line;
}

static char *OneInputLine(const char *prior, FILE *in)
{
	if (in != nullptr)
		return LocalGetLine(nullptr, in, 0);
	char *prompt = (prior && prior[0] ? _continuePrompt : _mainPrompt);
	char *result = readline(prompt);
#if defined(HAVE_READLINE) && HAVE_READLINE==1
	if (result && *result) AddHistory(result);
#endif
	return result;
}

enum MODE : uint8
{
	MODE_Line     = 0,		// One column per line.  Blank line between records
	MODE_Column   = 1,		// One record per line in neat columns
	MODE_List     = 2,		// One record per line with a separator
	MODE_Semi     = 3,		// Same as MODE_List but append ";" to each line
	MODE_Html     = 4,		// Generate an XHTML table
	MODE_Insert   = 5,		// Generate SQL "insert" statements
	MODE_Tcl      = 6,		// Generate ANSI-C or TCL quoted elements
	MODE_Csv      = 7,		// Quote strings, numbers are plain
	MODE_Explain  = 8,		// Like MODE_Column, but do not truncate data
};

struct PreviousModeData
{
	int Valid;        // Is there legit data in here?
	MODE Mode;
	int ShowHeader;
	int ColWidth[100];
};

static const char *_modeDescr[] =
{
	"line",
	"column",
	"list",
	"semi",
	"html",
	"insert",
	"tcl",
	"csv",
	"explain",
};

struct CallbackData
{
	int H_size;
	struct CallbackData *H_;
	struct CallbackData *D_;
	Context *Ctx;				// The database
	int EchoOn;					// True to echo input commands
	int StatsOn;				// True to display memory stats before each finalize
	int Cnt;					// Number of records displayed so far
	FILE *Out;					// Write results here
	FILE *TraceOut;				// Output for sqlite3_trace()
	int Errs;					// Number of errors seen
	MODE Mode;					// An output mode setting
	int WritableSchema;			// True if PRAGMA writable_schema=ON
	int ShowHeader;				// True to show column names in List or Column mode
	char *DestTable;			// Name of destination table when MODE_Insert
	char Separator[20];			// Separator character for MODE_List
	int ColWidth[100];			// Requested width of each column when in column mode
	int ActualWidth[100];		// Actual width of each column
	char NullValue[20];			// The text to print when a NULL comes back from the database
	struct PreviousModeData ExplainPrev;
	// Holds the mode information just before .explain ON
	char Outfile[FILENAME_MAX]; // Filename for *out_
	const char *DbFilename;		// name of the database file
	//const char *Vfs;			// Name of VFS to use
	Vdbe *Stmt;					// Current statement if any.
	FILE *Log;					// Write log output here
};

__device__ static void ShellLog(void *arg, int errCode, const char *msg)
{
	struct CallbackData *p = (struct CallbackData*)arg;
	if (!p->Log) return;
	_fprintf(p->Log, "(%d) %s\n", errCode, msg);
	_fflush(p->Log);
}

#pragma endregion

#pragma region CUDA
#if __CUDACC__
cudaDeviceHeap _deviceHeap;

void H_DIRTY(struct CallbackData *p)
{
	p->H_size = 0;
}

void D_FREE(struct CallbackData *p)
{
	if (p->D_)
	{
		free(p->H_);
		cudaFree(p->D_);
		p->H_ = p->D_ = nullptr;
	}
}

void D_DATA(struct CallbackData *p)
{
	CallbackData *h = p->H_;
	char *destTable;
	const char *dbFilename;
	if (!p->H_size)
	{
		D_FREE(p);
		// Allocate memory for the CallbackData structure, PCache object, the three file descriptors, the database file name and the journal file name.
		int destTableLength = (p->DestTable ? (int)strlen(p->DestTable) + 1 : 0);
		int dbFilenameLength = (p->DbFilename ? (int)strlen(p->DbFilename) + 1 : 0);
		int size = _ROUND8(sizeof(CallbackData)) + // CallbackData structure
			destTableLength + // DestTable
			dbFilenameLength; // DbFilename
		uint8 *ptr = (uint8 *)malloc(size);
		if (!ptr)
		{
			printf("D_DATA: RC_NOMEM");
			return;
		}
		memset(ptr, 0, size);
		// create clone to send to device
		h = p->H_ = (CallbackData *)(ptr);
		destTable = (char *)(ptr += _ROUND8(sizeof(CallbackData)));
		dbFilename = (char *)(ptr += destTableLength);
		memcpy((void *)destTable, p->DestTable, destTableLength);
		memcpy((void *)dbFilename, p->DbFilename, dbFilenameLength);
		p->H_size = size;
		//
		cudaErrorCheck(cudaMalloc((void **)&ptr, p->H_size));
		p->D_ = (CallbackData *)(ptr);
		destTable = (char *)(ptr += _ROUND8(sizeof(CallbackData)));
		dbFilename = (char *)(ptr += destTableLength);
		h->DestTable = (p->DestTable ? destTable : nullptr);
		h->DbFilename = (p->DbFilename ? dbFilename : nullptr);
		h->Out = cudaIobTranslate(p->Out, cudaMemcpyHostToDevice);
	}
	//
	destTable = h->DestTable;
	dbFilename = h->DbFilename;
	memcpy(h, p, sizeof(CallbackData));
	h->DestTable = destTable;
	h->DbFilename = dbFilename;
	cudaErrorCheck(cudaMemcpy(p->D_, h, p->H_size, cudaMemcpyHostToDevice));
}

void H_DATA(struct CallbackData *p)
{
	CallbackData *h = p->H_;
	cudaErrorCheck(cudaMemcpy(h, p->D_, p->H_size, cudaMemcpyDeviceToHost));
	char *destTable = p->DestTable;
	const char *dbFilename = p->DbFilename;
	memcpy(p, h, sizeof(CallbackData));
	p->DestTable = destTable;
	p->DbFilename = dbFilename;
}


__device__ long d_return;
long h_return;
void H_RETURN()
{
	cudaErrorCheck(cudaMemcpyFromSymbol(&h_return, d_return, sizeof(h_return), 0, cudaMemcpyDeviceToHost));
}

__global__ void d_DoMetaCommand(struct CallbackData *p, int argsLength, char **args, int tag, void *tag2);
void D_META(struct CallbackData *p, int argsLength, char *args[50], int tag = 0, void *tag2 = nullptr)
{
	int argsSize = _ROUND8(sizeof(char*)*argsLength);
	int size = argsSize;
	for (int i = 0; i < argsLength; i++) size += (int)strlen(args[i]) + 1;
	char *ptr = (char *)malloc(size);
	if (!ptr)
	{
		printf("D_META: RC_NOMEM");
		return;
	}
	memset(ptr, 0, size);
	// create clone to send to device
	char *h_ = ptr;
	ptr += argsSize;
	for (int i = 0; i < argsLength; i++)
	{
		int length = (int)strlen(args[i]) + 1;
		memcpy((void *)ptr, args[i], length);
		ptr += length;
	}
	//
	cudaErrorCheck(cudaMalloc((void**)&ptr, size));
	char *d_ = ptr;
	ptr += argsSize;
	char **h_args = (char **)h_;
	for (int i = 0; i < argsLength; i++)
	{
		int length = (int)strlen(args[i]) + 1;
		h_args[i] = ptr;
		ptr += length;
	}
	cudaErrorCheck(cudaMemcpy(d_, h_, size, cudaMemcpyHostToDevice));
	//
	d_DoMetaCommand<<<1,1>>>(p->D_, argsLength, (char **)d_, tag, tag2); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); cudaErrorCheck(cudaDeviceSynchronize()); 
	cudaFree(d_);
	free(h_);
}

#else
#define H_DIRTY(p) 0
#endif
#pragma endregion

#pragma region Output

__device__ static void OutputHexBlob(FILE *out_, const void *blob, int blobLength)
{
	char *blob2 = (char *)blob;
	_fprintf(out_, "X'");
	for (int i = 0; i < blobLength; i++) { _fprintf(out_, "%02x", blob2[i]&0xff); }
	_fprintf(out_, "'");
}

__device__ static void OutputQuotedString(FILE *out_, const char *z)
{
	int i;
	int singles = 0;
	for (i = 0; z[i]; i++)
		if (z[i] == '\'' ) singles++;
	if (singles == 0)
		_fprintf(out_, "'%s'", z);
	else
	{
		_fprintf(out_, "'");
		while (*z)
		{
			for (i = 0; z[i] && z[i] != '\''; i++) { }
			if (i == 0) { _fprintf(out_, "''"); z++; }
			else if (z[i] == '\'') { _fprintf(out_, "%.*s''", i, z); z += i+1; }
			else { _fprintf(out_, "%s", z); break; }
		}
		_fprintf(out_, "'");
	}
}

__device__ static void OutputCString(FILE *out_, const char *z)
{
	unsigned int c;
	_fputc('"', out_);
	while ((c = *(z++)) != 0)
	{
		if (c == '\\') { _fputc(c, out_); _fputc(c, out_); }
		else if (c == '"') { _fputc('\\', out_); _fputc('"', out_); }
		else if (c == '\t') { _fputc('\\', out_); _fputc('t', out_); }
		else if (c == '\n') { _fputc('\\', out_); _fputc('n', out_); }
		else if (c == '\r') { _fputc('\\', out_); _fputc('r', out_); }
		else if (!_isprint(c)) _fprintf(out_, "\\%03o", c&0xff);
		else _fputc(c, out_);
	}
	_fputc('"', out_);
}

static void fOutputCString(FILE *out_, const char *z)
{
	unsigned int c;
	fputc('"', out_);
	while ((c = *(z++)) != 0)
	{
		if (c == '\\') { fputc(c, out_); fputc(c, out_); }
		else if (c == '"') { fputc('\\', out_); fputc('"', out_); }
		else if (c == '\t') { fputc('\\', out_); fputc('t', out_); }
		else if (c == '\n') { fputc('\\', out_); fputc('n', out_); }
		else if (c == '\r') { fputc('\\', out_); fputc('r', out_); }
		else if (!_isprint(c)) fprintf(out_, "\\%03o", c&0xff);
		else fputc(c, out_);
	}
	fputc('"', out_);
}

__device__ static void OutputHtmlString(FILE *out_, const char *z)
{
	int i;
	while (*z)
	{
		for (i = 0; z[i] && z[i] != '<' && z[i] != '&' && z[i] != '>' && z[i] != '\"' && z[i] != '\''; i++) { }
		if (i > 0) _fprintf(out_, "%.*s", i, z);
		if (z[i] == '<') _fprintf(out_,"&lt;");
		else if (z[i] == '&') _fprintf(out_,"&amp;");
		else if (z[i] == '>') _fprintf(out_,"&gt;");
		else if (z[i] == '\"') _fprintf(out_,"&quot;");
		else if (z[i] == '\'') _fprintf(out_,"&#39;");
		else break;
		z += i + 1;
	}
}

__constant__ static const char _needCsvQuote[] = {
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 0, 1, 0, 0, 0, 0, 1,   0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 1, 
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
	1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,   
};

__device__ static void OutputCsv(struct CallbackData *p, const char *z, bool sep)
{
	FILE *out_ = p->Out;
	if (!z) _fprintf(out_, "%s", p->NullValue);
	else
	{
		int i;
		int sepLength = _strlen(p->Separator);
		for (i = 0; z[i]; i++)
		{
			if (_needCsvQuote[((unsigned char*)z)[i]] || (z[i] == p->Separator[0] && (sepLength == 1 || _memcmp(z, p->Separator, sepLength) == 0)))
			{
				i = 0;
				break;
			}
		}
		if (i == 0)
		{
			_fputc('"', out_);
			for (i = 0; z[i]; i++)
			{
				if (z[i] == '"') _fputc('"', out_);
				_fputc(z[i], out_);
			}
			_fputc('"', out_);
		}
		else _fprintf(out_, "%s", z);
	}
	if (sep) _fprintf(out_, "%s", p->Separator);
}

#pragma endregion

#pragma region Callback

#ifdef SIGINT
static void InterruptHandler(int notUsed)
{
	_seenInterrupt = 1;
	if (ctx) sqlite3_interrupt(ctx);
}
#endif

#define _lineSpacerLength 93
__device__ char *_lineSpacer = "---------------------------------------------------------------------------------------------";
__device__ static bool ShellCallback(void *arg, int colLength, char **colValues, char **colNames, int *colTypes)
{
	int i;
	struct CallbackData *p = (struct CallbackData *)arg;
	switch (p->Mode)
	{
	case MODE_Line: {
		int w = 5;
		if (colValues == 0) break;
		for (i = 0; i < colLength; i++)
		{
			int len = _strlen(colNames[i] ? colNames[i] : "");
			if (len > w) w = len;
		}
		if (p->Cnt++ > 0) _fprintf(p->Out, "\n");
		for (i = 0; i < colLength; i++)
			_fprintf(p->Out, "%*s = %s\n", w, colNames[i], colValues[i] ? colValues[i] : p->NullValue);
		break; }
	case MODE_Explain:
	case MODE_Column: {
		if (p->Cnt++ == 0)
		{
			for (i = 0; i < colLength; i++)
			{
				int w = (i < _lengthof(p->ColWidth) ? p->ColWidth[i] : 0);
				if (w == 0)
				{
					w = _strlen(colNames[i] ? colNames[i] : "");
					if (w < 10) w = 10;
					int n = _strlen(colValues && colValues[i] ? colValues[i] : p->NullValue);
					if (w < n) w = n;
				}
				if (i < _lengthof(p->ActualWidth))
					p->ActualWidth[i] = w;
				if (p->ShowHeader)
				{
					if (w < 0) w = -w;
#if __CUDACC__
					char w_;
					if (_strlen(colNames[i]) > w) { w_ = colNames[i][w]; colNames[i][w] = 0; } else w_ = 0;
					_fprintf(p->Out, "%-*s%s", w, colNames[i], (i == colLength-1 ? "\n": "  "));
					if (w_) colNames[i][w] = w_;
#else
					_fprintf(p->Out, "%-*.*s%s", w, w, colNames[i], (i == colLength-1 ? "\n": "  "));
#endif
				}
			}
			if (p->ShowHeader)
				for (i = 0; i < colLength; i++)
				{
					int w;
					if (i < _lengthof(p->ActualWidth))
					{
						w = p->ActualWidth[i];
						if (w < 0) w = -w;
					}
					else
						w = 10;
#if __CUDACC__
					char w_;
					if (_lineSpacerLength > w) { w_ = _lineSpacer[w]; _lineSpacer[w] = 0; } else w_ = 0;
					_fprintf(p->Out, "%-*s%s", w, _lineSpacer, (i == colLength-1 ? "\n" : "  "));
					if (w_) _lineSpacer[w] = '-';
#else
					_fprintf(p->Out, "%-*.*s%s", w, w, _lineSpacer, (i == colLength-1 ? "\n" : "  "));
#endif
				}
		}
		if (colValues == 0) break;
		for (i = 0; i < colLength; i++)
		{
			int w = (i < _lengthof(p->ActualWidth) ? p->ActualWidth[i] : 10);
			if (p->Mode == MODE_Explain && colValues[i] && _strlen(colValues[i]) > w)
				w = _strlen(colValues[i]);
			if (w < 0) w = -w;
#if __CUDACC__
			char w_;
			if (_strlen(colValues[i]) > w) { w_ = colValues[i][w]; colValues[i][w] = 0; } else w_ = 0;
			_fprintf(p->Out, "%-*s%s", w, (colValues[i] ? colValues[i] : p->NullValue), (i == colLength-1 ? "\n" : "  "));
			if (w_) colValues[i][w] = w_;
#else
			_fprintf(p->Out, "%-*.*s%s", w, w, (colValues[i] ? colValues[i] : p->NullValue), (i == colLength-1 ? "\n" : "  "));
#endif
		}
		break; }
	case MODE_Semi:
	case MODE_List: {
		if (p->Cnt++ == 0 && p->ShowHeader)
			for (i = 0; i < colLength; i++)
				_fprintf(p->Out, "%s%s", colNames[i], (i == colLength-1 ? "\n" : p->Separator));
		if (colValues == 0) break;
		for (i = 0; i < colLength; i++)
		{
			char *z = colValues[i];
			if (!z) z = p->NullValue;
			_fprintf(p->Out, "%s", z);
			if (i < colLength-1) _fprintf(p->Out, "%s", p->Separator);
			else if (p->Mode == MODE_Semi) _fprintf(p->Out, ";\n");
			else _fprintf(p->Out, "\n");
		}
		break; }
	case MODE_Html: {
		if (p->Cnt++ == 0 && p->ShowHeader)
		{
			_fprintf(p->Out, "<TR>");
			for (i = 0; i < colLength; i++)
			{
				_fprintf(p->Out, "<TH>");
				OutputHtmlString(p->Out, colNames[i]);
				_fprintf(p->Out, "</TH>\n");
			}
			_fprintf(p->Out, "</TR>\n");
		}
		if (colValues == 0) break;
		_fprintf(p->Out, "<TR>");
		for (i = 0; i < colLength; i++)
		{
			_fprintf(p->Out, "<TD>");
			OutputHtmlString(p->Out, (colValues[i] ? colValues[i] : p->NullValue));
			_fprintf(p->Out, "</TD>\n");
		}
		_fprintf(p->Out, "</TR>\n");
		break; }
	case MODE_Tcl: {
		if (p->Cnt++ == 0 && p->ShowHeader)
		{
			for (i = 0; i < colLength; i++)
			{
				OutputCString(p->Out, (colNames[i] ? colNames[i] : ""));
				if (i < colLength-1) _fprintf(p->Out, "%s", p->Separator);
			}
			_fprintf(p->Out, "\n");
		}
		if (colValues == 0) break;
		for (i = 0; i < colLength; i++)
		{
			OutputCString(p->Out, (colValues[i] ? colValues[i] : p->NullValue));
			if (i < colLength-1) _fprintf(p->Out, "%s", p->Separator);
		}
		_fprintf(p->Out, "\n");
		break; }
	case MODE_Csv: {
		if (p->Cnt++ == 0 && p->ShowHeader)
		{
			for (i = 0; i < colLength; i++)
				OutputCsv(p, (colNames[i] ? colNames[i] : ""), i < colLength-1);
			_fprintf(p->Out, "\n");
		}
		if (colValues == 0) break;
		for (i = 0; i < colLength; i++)
			OutputCsv(p, colValues[i], i < colLength-1);
		_fprintf(p->Out, "\n");
		break; }
	case MODE_Insert: {
		p->Cnt++;
		if (colValues == 0) break;
		_fprintf(p->Out, "INSERT INTO %s VALUES(", p->DestTable);
		for (i = 0; i < colLength; i++)
		{
			char *sep = (i > 0 ? "," : "");
			if ((colValues[i] == 0) || (colTypes && colTypes[i] == TYPE_NULL))
				_fprintf(p->Out, "%sNULL", sep);
			else if (colTypes && colTypes[i] == TYPE_TEXT)
			{
				if (sep[0]) _fprintf(p->Out, "%s", sep);
				OutputQuotedString(p->Out, colValues[i]);
			}
			else if (colTypes && (colTypes[i] == TYPE_INTEGER || colTypes[i] == TYPE_FLOAT))
				_fprintf(p->Out, "%s%s", sep, colValues[i]);
			else if (colTypes && colTypes[i] == TYPE_BLOB && p->Stmt)
			{
				const void *blob = Vdbe::Column_Blob(p->Stmt, i);
				int blobLength = Vdbe::Column_Bytes(p->Stmt, i);
				if (sep[0]) _fprintf(p->Out, "%s", sep);
				OutputHexBlob(p->Out, blob, blobLength);
			}
			else if (isNumber(colValues[i], 0))
				_fprintf(p->Out, "%s%s", sep, colValues[i]);
			else
			{
				if (sep[0]) _fprintf(p->Out, "%s", sep);
				OutputQuotedString(p->Out, colValues[i]);
			}
		}
		_fprintf(p->Out, ");\n");
		break; }
	}
	return false;
}

__device__ static bool Callback(void *arg, int colLength, char **colValues, char **colNames) { return ShellCallback(arg, colLength, colValues, colNames, nullptr);  } // since we don't have type info, call the ShellCallback with a NULL value

#pragma endregion

#pragma region Helpers

static void SetTableName(struct CallbackData *p, const char *name)
{
	H_DIRTY(p);
	if (p->DestTable)
	{
		free(p->DestTable);
		p->DestTable = nullptr;
	}
	if (!name) return;
	bool needQuote = (!isalpha((unsigned char)*name) && *name != '_');
	int i, n;
	for (i = n = 0; name[i]; i++, n++)
	{
		if (!isalnum((unsigned char)name[i]) && name[i] != '_')
		{
			needQuote = true;
			if (name[i] == '\'') n++;
		}
	}
	if (needQuote) n += 2;
	char *z = p->DestTable = (char *)malloc(n+1);
	if (!z)
	{
		fprintf(stderr, "Error: out_ of memory\n");
		exit(1);
	}
	n = 0;
	if (needQuote) z[n++] = '\'';
	for (i = 0; name[i]; i++)
	{
		z[n++] = name[i];
		if (name[i] == '\'') z[n++] = '\'';
	}
	if (needQuote) z[n++] = '\'';
	z[n] = 0;
}

__device__ static char *AppendText(char *in, char const *append, char quote)
{
	int i;
	int appendLength = _strlen(append);
	int inLength = (in ? _strlen(in) : 0);
	int newLength = appendLength+inLength+1;
	if (quote)
	{
		newLength += 2;
		for (i = 0; i < appendLength; i++)
			if (append[i] == quote) newLength++;
	}
	in = (char *)_realloc(in, newLength);
	if (!in)
		return nullptr;
	if (quote)
	{
		char *csr = &in[inLength];
		*csr++ = quote;
		for (i = 0; i < appendLength; i++)
		{
			*csr++ = append[i];
			if (append[i] == quote) *csr++ = quote;
		}
		*csr++ = quote;
		*csr++ = '\0';
		_assert((csr-in) == newLength);
	}
	else
	{
		_memcpy(&in[inLength], append, appendLength);
		in[newLength-1] = '\0';
	}
	return in;
}

__device__ static RC RunTableDumpQuery(struct CallbackData *p, const char *sql, const char *firstRow)
{
	int i;
	Vdbe *select;
	RC rc = Prepare::Prepare_(p->Ctx, sql, -1, &select, 0);
	if (rc != RC_OK || !select)
	{
		_fprintf(p->Out, "/**** ERROR: (%d) %s *****/\n", rc, DataEx::ErrMsg(p->Ctx));
		p->Errs++;
		return rc;
	}
	rc = select->Step();
	int results = Vdbe::Column_Count(select);
	while (rc == RC_ROW)
	{
		if (firstRow)
		{
			_fprintf(p->Out, "%s", firstRow);
			firstRow = nullptr;
		}
		const char *z = (const char *)Vdbe::Column_Text(select, 0);
		_fprintf(p->Out, "%s", z);
		for (i = 1; i < results; i++)
			_fprintf(p->Out, ",%s", Vdbe::Column_Text(select, i));
		if (!z) z = "";
		while (z[0] && (z[0] != '-' || z[1] != '-')) z++;
		if (z[0]) _fprintf(p->Out, "\n;\n");
		else _fprintf(p->Out, ";\n");
		rc = select->Step();
	}
	rc = Vdbe::Finalize(select);
	if (rc != RC_OK)
	{
		_fprintf(p->Out, "/**** ERROR: (%d) %s *****/\n", rc, DataEx::ErrMsg(p->Ctx));
		p->Errs++;
	}
	return rc;
}

__device__ static char *SaveErrMsg(Context *ctx)
{
	int errMsgLength = 1+_strlen(DataEx::ErrMsg(ctx));
	char *errMsg = (char *)_alloc(errMsgLength);
	if (errMsg)
		_memcpy(errMsg, DataEx::ErrMsg(ctx), errMsgLength);
	return errMsg;
}

__device__ static int DisplayStats(Context *ctx, struct CallbackData *arg, bool reset)
{
	int cur;
	int high;
	if (arg && arg->Out)
	{
		high = cur = -1;
		_status(STATUS_MEMORY_USED, &cur, &high, reset);
		_fprintf(arg->Out, "Memory Used:                         %d (max %d) bytes\n", cur, high);
		high = cur = -1;
		_status(STATUS_MALLOC_COUNT, &cur, &high, reset);
		_fprintf(arg->Out, "Number of Outstanding Allocations:   %d (max %d)\n", cur, high);
		// Not currently used by the CLI.
		//    high = cur = -1;
		//    _status(STATUS_PAGECACHE_USED, &cur, &high, reset);
		//    _fprintf(arg->Out, "Number of Pcache Pages Used:         %d (max %d) pages\n", cur, high);
		high = cur = -1;
		_status(STATUS_PAGECACHE_OVERFLOW, &cur, &high, reset);
		_fprintf(arg->Out, "Number of Pcache Overflow Bytes:     %d (max %d) bytes\n", cur, high);
		// Not currently used by the CLI.
		//    high = cur = -1;
		//    _status(STATUS_SCRATCH_USED, &cur, &high, reset);
		//    _fprintf(arg->Out, "Number of Scratch Allocations Used:  %d (max %d)\n", cur, high);
		high = cur = -1;
		_status(STATUS_SCRATCH_OVERFLOW, &cur, &high, reset);
		_fprintf(arg->Out, "Number of Scratch Overflow Bytes:    %d (max %d) bytes\n", cur, high);
		high = cur = -1;
		_status(STATUS_MALLOC_SIZE, &cur, &high, reset);
		_fprintf(arg->Out, "Largest Allocation:                  %d bytes\n", high);
		high = cur = -1;
		_status(STATUS_PAGECACHE_SIZE, &cur, &high, reset);
		_fprintf(arg->Out, "Largest Pcache Allocation:           %d bytes\n", high);
		high = cur = -1;
		_status(STATUS_SCRATCH_SIZE, &cur, &high, reset);
		_fprintf(arg->Out, "Largest Scratch Allocation:          %d bytes\n", high);
#ifdef YYTRACKMAXSTACKDEPTH
		high = cur = -1;
		_status(STATUS_PARSER_STACK, &cur, &high, reset);
		_fprintf(arg->Out, "Deepest Parser Stack:                %d (max %d)\n", cur, high);
#endif
	}

	if (arg && arg->Out && ctx)
	{
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_LOOKASIDE_USED, &cur, &high, reset);
		_fprintf(arg->Out, "Lookaside Slots Used:                %d (max %d)\n", cur, high);
		ctx->Status(Context::CTXSTATUS_LOOKASIDE_HIT, &cur, &high, reset);
		_fprintf(arg->Out, "Successful lookaside attempts:       %d\n", high);
		ctx->Status(Context::CTXSTATUS_LOOKASIDE_MISS_SIZE, &cur, &high, reset);
		_fprintf(arg->Out, "Lookaside failures due to size:      %d\n", high);
		ctx->Status(Context::CTXSTATUS_LOOKASIDE_MISS_FULL, &cur, &high, reset);
		_fprintf(arg->Out, "Lookaside failures due to OOM:       %d\n", high);
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_CACHE_USED, &cur, &high, reset);
		_fprintf(arg->Out, "Pager Heap Usage:                    %d bytes\n", cur);
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_CACHE_HIT, &cur, &high, 1);
		_fprintf(arg->Out, "Page cache hits:                     %d\n", cur);
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_CACHE_MISS, &cur, &high, 1);
		_fprintf(arg->Out, "Page cache misses:                   %d\n", cur); 
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_CACHE_WRITE, &cur, &high, 1);
		_fprintf(arg->Out, "Page cache writes:                   %d\n", cur); 
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_SCHEMA_USED, &cur, &high, reset);
		_fprintf(arg->Out, "Schema Heap Usage:                   %d bytes\n", cur); 
		high = cur = -1;
		ctx->Status(Context::CTXSTATUS_STMT_USED, &cur, &high, reset);
		_fprintf(arg->Out, "Statement Heap/Lookaside Usage:      %d bytes\n", cur); 
	}

	if (arg && arg->Out && ctx && arg->Stmt)
	{
		cur = Vdbe::Stmt_Status(arg->Stmt, Vdbe::STMTSTATUS_FULLSCAN_STEP, reset);
		_fprintf(arg->Out, "Fullscan Steps:                      %d\n", cur);
		cur = Vdbe::Stmt_Status(arg->Stmt, Vdbe::STMTSTATUS_SORT, reset);
		_fprintf(arg->Out, "Sort Operations:                     %d\n", cur);
		cur = Vdbe::Stmt_Status(arg->Stmt, Vdbe::STMTSTATUS_AUTOINDEX, reset);
		_fprintf(arg->Out, "Autoindex Inserts:                   %d\n", cur);
	}
	return 0;
}

// Execute a statement or set of statements.  Print any result rows/columns depending on the current mode set via the supplied callback.
// This is very similar to SQLite's built-in sqlite3_exec() function except it takes a slightly different callback and callback data argument.
// callback // (not the same as sqlite3_exec)
__device__ static int ShellExec(Context *ctx, const char *sql, bool (*callback)(void*,int,char**,char**,int*), struct CallbackData *arg, char **errMsg)
{
	Vdbe *stmt = nullptr; // Statement to execute.
	RC rc = RC_OK;
	int rc2;
	const char *leftover; // Tail of unprocessed SQL

	if (errMsg)
		*errMsg = nullptr;

	while (sql[0] && (rc == RC_OK))
	{
		rc = Prepare::Prepare_(ctx, sql, -1, &stmt, &leftover);
		if (rc != RC_OK)
		{
			if (errMsg)
				*errMsg = SaveErrMsg(ctx);
		}
		else
		{
			if (!stmt)
			{
				sql = leftover; // this happens for a comment or white-space
				while (IsSpace(sql[0])) sql++;
				continue;
			}

			// save off the prepared statment handle and reset row count
			if (arg)
			{
				arg->Stmt = stmt;
				arg->Cnt = 0;
			}

			// echo the sql statement if echo on
			if (arg && arg->EchoOn)
			{
				const char *stmtSql = Vdbe::Sql(stmt);
				_fprintf(arg->Out, "%s\n", (stmtSql ? stmtSql : sql));
			}

			// Output TESTCTRL_EXPLAIN text of requested
			if (arg && arg->Mode == MODE_Explain)
			{
				const char *explain = nullptr;
				//DataEx::TestControl(DataEx::TESTCTRL_EXPLAIN_STMT, stmt, &explain);
				if (explain && explain[0])
					_fprintf(arg->Out, "%s", explain);
			}

			// perform the first step.  this will tell us if we have a result set or not and how wide it is.
			rc = stmt->Step();
			// if we have a result set...
			if (rc == RC_ROW)
			{
				// if we have a callback...
				if (callback)
				{
					// allocate space for col name ptr, value ptr, and type 
					int colLength = Vdbe::Column_Count(stmt);
					void *data = _alloc(3*colLength*sizeof(const char *) + 1);
					if (!data)
						rc = RC_NOMEM;
					else
					{
						char **colNames = (char **)data; // Names of result columns
						char **colValues = &colNames[colLength];       // Results
						int *colTypes = (int *)&colValues[colLength]; // Result types
						int i;
						_assert(sizeof(int) <= sizeof(char *)); 
						// save off ptrs to column names
						for (i = 0; i < colLength; i++)
							colNames[i] = (char *)Vdbe::Column_Name(stmt, i);
						do
						{
							// extract the data and data types
							for (i = 0; i < colLength; i++)
							{
								colValues[i] = (char *)Vdbe::Column_Text(stmt, i);
								colTypes[i] = Vdbe::Column_Type(stmt, i);
								if (!colValues[i] && (colTypes[i] != TYPE_NULL))
								{
									rc = RC_NOMEM;
									break;
								}
							}
							// if data and types extracted successfully...call the supplied callback with the result row data
							if (rc == RC_ROW)
								rc = (callback(arg, colLength, colValues, colNames, colTypes) ? RC_ABORT : stmt->Step());
						} while (rc == RC_ROW);
						_free(data);
					}
				}
				else
					do { rc = stmt->Step(); }
					while (rc == RC_ROW);
			}

			// print Usage stats if stats on
			if (arg && arg->StatsOn)
				DisplayStats(ctx, arg, 0);

			// Finalize the statement just executed. If this fails, save a copy of the error message. Otherwise, set sql to point to the next statement to execute.
			rc2 = Vdbe::Finalize(stmt);
			if (rc != RC_NOMEM) rc = (RC)rc2;
			if (rc == RC_OK)
			{
				sql = leftover;
				while (IsSpace(sql[0])) sql++;
			}
			else if (errMsg)
				*errMsg = SaveErrMsg(ctx);

			// clear saved stmt handle
			if (arg)
				arg->Stmt = nullptr;
		}
	}
	return rc;
}

// This is a different callback routine used for dumping the database. Each row received by this callback consists of a table name,
// the table type ("index" or "table") and SQL to create the table. This routine should print text sufficient to recreate the table.
__device__ static bool DumpCallback(void *arg, int colLength, char **colValues, char **colNames)
{
	RC rc;
	const char *prepStmt = nullptr;
	struct CallbackData *p = (struct CallbackData *)arg;

	if (colLength != 3) return 1;
	const char *tableName = colValues[0];
	const char *typeName = colValues[1];
	const char *sql = colValues[2];

	if (!_strcmp(tableName, "sqlite_sequence")) prepStmt = "DELETE FROM sqlite_sequence;\n";
	else if (!_strcmp(tableName, "sqlite_stat1")) _fprintf(p->Out, "ANALYZE sqlite_master;\n");
	else if (!_strncmp(tableName, "sqlite_", 7)) return false;
	else if (!_strncmp(sql, "CREATE VIRTUAL TABLE", 20))
	{
		if (!p->WritableSchema)
		{
			_fprintf(p->Out, "PRAGMA writable_schema=ON;\n");
			p->WritableSchema = 1;
		}
		char *ins = _mprintf(
			"INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql)"
			"VALUES('table','%q','%q',0,'%q');",
			tableName, tableName, sql);
		_fprintf(p->Out, "%s\n", ins);
		_free(ins);
		return false;
	}
	else _fprintf(p->Out, "%s;\n", sql);

	if (!_strcmp(typeName, "table"))
	{
		char *tableInfoSql = nullptr;
		tableInfoSql = AppendText(tableInfoSql, "PRAGMA table_info(", 0);
		tableInfoSql = AppendText(tableInfoSql, tableName, '"');
		tableInfoSql = AppendText(tableInfoSql, ");", 0);

		Vdbe *tableInfo = nullptr;
		rc = Prepare::Prepare_(p->Ctx, tableInfoSql, -1, &tableInfo, 0);
		_free(tableInfoSql);
		if (rc != RC_OK || !tableInfo)
			return true;

		char *select = nullptr;
		select = AppendText(select, "SELECT 'INSERT INTO ' || ", 0);
		// Always quote the table name, even if it appears to be pure ascii, in case it is a keyword. Ex:  INSERT INTO "table" ...
		char *tmp = nullptr;
		tmp = AppendText(tmp, tableName, '"');
		if (tmp)
		{
			select = AppendText(select, tmp, '\'');
			_free(tmp);
		}
		select = AppendText(select, " || ' VALUES(' || ", 0);
		rc = tableInfo->Step();
		int rows = 0;
		while (rc == RC_ROW)
		{
			const char *text = (const char *)Vdbe::Column_Text(tableInfo, 1);
			select = AppendText(select, "quote(", 0);
			select = AppendText(select, text, '"');
			rc = tableInfo->Step();
			select = AppendText(select, (rc == RC_ROW ? "), " : ") "), 0);
			rows++;
		}
		rc = Vdbe::Finalize(tableInfo);
		if (rc != RC_OK || rows == 0)
		{
			_free(select);
			return 1;
		}
		select = AppendText(select, "|| ')' FROM  ", 0);
		select = AppendText(select, tableName, '"');

		rc = RunTableDumpQuery(p, select, prepStmt);
		if (rc == RC_CORRUPT)
		{
			select = AppendText(select, " ORDER BY rowid DESC", 0);
			RunTableDumpQuery(p, select, nullptr);
		}
		_free(select);
	}
	return false;
}

// Run zQuery.  Use DumpCallback() as the callback routine so that the contents of the query are output as SQL statements.
// If we get a SQLITE_CORRUPT error, rerun the query after appending "ORDER BY rowid DESC" to the end.
__device__ static int RunSchemaDumpQuery(struct CallbackData *p, const char *query)
{
	char *err = nullptr;
	RC rc = DataEx::Exec(p->Ctx, query, DumpCallback, p, &err);
	if (rc == RC_CORRUPT)
	{
		int length = _strlen(query);
		_fprintf(p->Out, "/****** CORRUPTION ERROR *******/\n");
		if (err)
		{
			_fprintf(p->Out, "/****** %s ******/\n", err);
			_free(err);
			err = nullptr;
		}
		char *q2 = (char *)malloc(length+100);
		if (!q2) return rc;
		__snprintf(q2, length+100, "%s ORDER BY rowid DESC", query);
		rc = DataEx::Exec(p->Ctx, q2, DumpCallback, p, &err);
		if (rc)
			_fprintf(p->Out, "/****** ERROR: %s ******/\n", err);
		else
			rc = RC_CORRUPT;
		_free(err);
		free(q2);
	}
	return rc;
}

// Text of a help message
static char _help[] =
	".backup ?DB? FILE      Backup DB (default \"main\") to FILE\n"
	".bail ON|OFF           Stop after hitting an error.  Default OFF\n"
	".databases             List names and files of attached databases\n"
	".dump ?TABLE? ...      Dump the database in an SQL text format\n"
	"                         If TABLE specified, only dump tables matching\n"
	"                         LIKE pattern TABLE.\n"
	".echo ON|OFF           Turn command echo on or off\n"
	".exit                  Exit this program\n"
	".explain ?ON|OFF?      Turn output mode suitable for EXPLAIN on or off.\n"
	"                         With no args, it turns EXPLAIN on.\n"
	".header(s) ON|OFF      Turn display of headers on or off\n"
	".help                  Show this message\n"
	".import FILE TABLE     Import data from FILE into TABLE\n"
	".indices ?TABLE?       Show names of all indices\n"
	"                         If TABLE specified, only show indices for tables\n"
	"                         matching LIKE pattern TABLE.\n"
#ifdef ENABLE_IOTRACE
	".iotrace FILE          Enable I/O diagnostic logging to FILE\n"
#endif
#ifndef OMIT_LOAD_EXTENSION
	".load FILE ?ENTRY?     Load an extension library\n"
#endif
	".log FILE|off          Turn logging on or off.  FILE can be stderr/stdout\n"
	".mode MODE ?TABLE?     Set output mode where MODE is one of:\n"
	"                         csv      Comma-separated values\n"
	"                         column   Left-aligned columns.  (See .width)\n"
	"                         html     HTML <table> code\n"
	"                         insert   SQL insert statements for TABLE\n"
	"                         line     One value per line\n"
	"                         list     Values delimited by .separator string\n"
	"                         tabs     Tab-separated values\n"
	"                         tcl      TCL list elements\n"
	".nullvalue STRING      Use STRING in place of NULL values\n"
	".output FILENAME       Send output to FILENAME\n"
	".output stdout         Send output to the screen\n"
	".print STRING...       Print literal STRING\n"
	".prompt MAIN CONTINUE  Replace the standard prompts\n"
	".quit                  Exit this program\n"
	".read FILENAME         Execute SQL in FILENAME\n"
	".restore ?DB? FILE     Restore content of DB (default \"main\") from FILE\n"
	".schema ?TABLE?        Show the CREATE statements\n"
	"                         If TABLE specified, only show tables matching\n"
	"                         LIKE pattern TABLE.\n"
	".separator STRING      Change separator used by output mode and .import\n"
	".show                  Show the current values for various settings\n"
	".stats ON|OFF          Turn stats on or off\n"
	".tables ?TABLE?        List names of tables\n"
	"                         If TABLE specified, only list tables matching\n"
	"                         LIKE pattern TABLE.\n"
	".timeout MS            Try opening locked tables for MS milliseconds\n"
	".trace FILE|off        Output each SQL statement as it is run\n"
	".vfsname ?AUX?         Print the name of the VFS stack\n"
	".width NUM1 NUM2 ...   Set column widths for \"column\" mode\n";

static char _timerHelp[] =
	".timer ON|OFF          Turn the CPU timer measurement on or off\n";

static bool ProcessInput(struct CallbackData *p, FILE *in); // Forward reference

// Make sure the database is open.  If it is not, then open it.  If the database fails to open, print an error message and exit.
__global__ static void OpenCtx(struct CallbackData *p)
{
	if (!p->Ctx)
	{
		DataEx::Initialize();
		DataEx::Open(p->DbFilename, &p->Ctx);
		_ctx = p->Ctx;
		if (_ctx && DataEx::ErrCode(_ctx) == RC_OK)
			DataEx::CreateFunction(_ctx, "shellstatic", 0, TEXTENCODE_UTF8, 0, ShellStaticFunc, 0, 0);
		if (!_ctx || DataEx::ErrCode(_ctx) != RC_OK)
		{
			_fprintf(_stderr, "Error: unable to open database \"%s\": %s\n", p->DbFilename, DataEx::ErrMsg(_ctx));
#if __CUDACC__
			d_return = -1; return;
#else
			exit(1);
#endif
		}
#ifndef OMIT_LOAD_EXTENSION
		//DataEx::enable_load_extension(p->db, 1);
#endif
#ifdef ENABLE_REGEXP
		{
			extern int sqlite3_add_regexp_func(sqlite3*);
			sqlite3_add_regexp_func(_ctx);
		}
#endif
#ifdef ENABLE_SPELLFIX
		{
			extern int sqlite3_spellfix1_register(sqlite3*);
			sqlite3_spellfix1_register(_ctx);
		}
#endif
	}
#if __CUDACC__
	d_return = 0;
#endif
}
#if __CUDACC__
static void _OpenCtx(struct CallbackData *p)
{
	D_DATA(p); OpenCtx<<<1,1>>>(p->D_); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(p);
	H_RETURN(); if (h_return) exit(1);
}
#else
#define _OpenCtx OpenCtx
#endif

// Do C-language style dequoting.
//
//    \t    -> tab
//    \n    -> newline
//    \r    -> carriage return
//    \NNN  -> ascii character NNN in octal
//    \\    -> backslash
static void ResolveBackslashes(char *z)
{
	int i, j;
	char c;
	for (i = j = 0; (c = z[i]) != 0; i++, j++)
	{
		if (c == '\\')
		{
			c = z[++i];
			if (c == 'n') c = '\n';
			else if (c == 't') c = '\t';
			else if (c == 'r') c = '\r';
			else if (c >= '0' && c <= '7')
			{
				c -= '0';
				if (z[i+1] >= '0' && z[i+1] <= '7')
				{
					i++; c = (c<<3) + z[i] - '0';
					if (z[i+1] >= '0' && z[i+1] <= '7') { i++; c = (c<<3) + z[i] - '0'; }
				}
			}
		}
		z[j] = c;
	}
	z[j] = 0;
}

// Interpret zArg as a boolean value.  Return either 0 or 1.
static bool BooleanValue(char *arg)
{
	int i;
	for (i = 0; arg[i] >= '0' && arg[i] <= '9'; i++) { }
	if (i > 0 && arg[i] == 0) return (atoi(arg) != 0);
	if (!strcmp(arg, "on") || !strcmp(arg, "yes")) return true;
	if (!strcmp(arg, "off") || !strcmp(arg, "no")) return false;
	fprintf(stderr, "ERROR: Not a boolean value: \"%s\". Assuming \"no\".\n", arg);
	return false;
}

// Close an output file, assuming it is not stderr or stdout
static void OutputFileClose(FILE *f)
{
	if (f && f != stdout && f != stderr) fclose(f);
}

// Try to open an output file.  The names "stdout" and "stderr" are recognized and do the right thing.  NULL is returned if the output filename is "off".
static FILE *OutputFileOpen(const char *file)
{
	FILE *f;
	if (!strcmp(file, "stdout")) f = stdout;
	else if (!strcmp(file, "stderr")) f = stderr;
	else if (!strcmp(file, "off")) f = 0;
	else
	{
		f = fopen(file, "wb");
		if (!f)
			fprintf(stderr, "Error: cannot open \"%s\"\n", file);
	}
	return f;
}

// A routine for handling output from sqlite3_trace().
__device__ static void SqlTraceCallback(void *arg, const char *z)
{
	FILE *f = (FILE *)arg;
	if (f) _fprintf(f, "%s\n", z);
}

// A no-op routine that runs with the ".breakpoint" doc-command.  This is a useful spot to set a debugger breakpoint.
static void TestBreakpoint()
{
	static int calls = 0;
	calls++;
}

#pragma endregion

#pragma region META

#if __CUDACC__
__global__ void d_DoMetaCommand(struct CallbackData *p, int argsLength, char **args, int tag, void *tag2)
{
#pragma region Preamble
	int n = (int)_strlen(args[0]);
	int c = args[0][0];
	int rc = 0;
#pragma endregion
#pragma region .databases
	if (c == 'd' && n > 1 && !_strncmp(args[0], "databases", n) && argsLength == 1)
	{
		struct CallbackData data;
		_memcpy(&data, p, sizeof(data));
		data.ShowHeader = 1;
		data.Mode = MODE_Column;
		data.ColWidth[0] = 3;
		data.ColWidth[1] = 15;
		data.ColWidth[2] = 58;
		data.Cnt = 0;
		char *errMsg = nullptr;
		DataEx::Exec(p->Ctx, "PRAGMA database_list;", ::Callback, &data, &errMsg);
		if (errMsg)
		{
			_fprintf(_stderr, "Error: %s\n", errMsg);
			_free(errMsg);
			rc = 1;
		}
	}
#pragma endregion
#pragma region .dump
	else if (c == 'd' && !_strncmp(args[0], "dump", n) && argsLength < 3)
	{
		// When playing back a "dump", the content might appear in an order which causes immediate foreign key constraints to be violated.
		// So disable foreign-key constraint enforcement to prevent problems.
		_fprintf(p->Out, "PRAGMA foreign_keys=OFF;\n");
		_fprintf(p->Out, "BEGIN TRANSACTION;\n");
		p->WritableSchema = 0;
		DataEx::Exec(p->Ctx, "SAVEPOINT dump; PRAGMA writable_schema=ON", 0, 0, 0);
		p->Errs = 0;
		if (argsLength == 1)
		{
			RunSchemaDumpQuery(p, 
				"SELECT name, type, sql FROM sqlite_master "
				"WHERE sql NOT NULL AND type=='table' AND name!='sqlite_sequence'");
			RunSchemaDumpQuery(p, 
				"SELECT name, type, sql FROM sqlite_master "
				"WHERE name=='sqlite_sequence'");
			RunTableDumpQuery(p,
				"SELECT sql FROM sqlite_master "
				"WHERE sql NOT NULL AND type IN ('index','trigger','view')", 0);
		}
		else
		{
			for (int i = 1; i < argsLength; i++)
			{
				_shellStatic = args[i];
				RunSchemaDumpQuery(p,
					"SELECT name, type, sql FROM sqlite_master "
					"WHERE tbl_name LIKE shellstatic() AND type=='table'"
					"  AND sql NOT NULL");
				RunTableDumpQuery(p,
					"SELECT sql FROM sqlite_master "
					"WHERE sql NOT NULL"
					"  AND type IN ('index','trigger','view')"
					"  AND tbl_name LIKE shellstatic()", 0);
				_shellStatic = nullptr;
			}
		}
		if (p->WritableSchema)
		{
			_fprintf(p->Out, "PRAGMA writable_schema=OFF;\n");
			p->WritableSchema = 0;
		}
		DataEx::Exec(p->Ctx, "PRAGMA writable_schema=OFF;", 0, 0, 0);
		DataEx::Exec(p->Ctx, "RELEASE dump;", 0, 0, 0);
		_fprintf(p->Out, (p->Errs ? "ROLLBACK; -- due to errors\n" : "COMMIT;\n"));
	}
#pragma endregion
#pragma region .import
#pragma endregion
#pragma region .indices
	else if (c == 'i' && !_strncmp(args[0], "indices", n) && argsLength < 3)
	{
		struct CallbackData data;
		char *errMsg = nullptr;
		_memcpy(&data, p, sizeof(data));
		data.ShowHeader = 0;
		data.Mode = MODE_List;
		if (argsLength == 1)
			rc = DataEx::Exec(p->Ctx,
			"SELECT name FROM sqlite_master "
			"WHERE type='index' AND name NOT LIKE 'sqlite_%' "
			"UNION ALL "
			"SELECT name FROM sqlite_temp_master "
			"WHERE type='index' "
			"ORDER BY 1",
			::Callback, &data, &errMsg);
		else
		{
			_shellStatic = args[1];
			rc = DataEx::Exec(p->Ctx,
				"SELECT name FROM sqlite_master "
				"WHERE type='index' AND tbl_name LIKE shellstatic() "
				"UNION ALL "
				"SELECT name FROM sqlite_temp_master "
				"WHERE type='index' AND tbl_name LIKE shellstatic() "
				"ORDER BY 1",
				::Callback, &data, &errMsg);
			_shellStatic = nullptr;
		}
		if (errMsg)
		{
			_fprintf(_stderr, "Error: %s\n", errMsg);
			_free(errMsg);
			rc = 1;
		}
		else if (rc != RC_OK)
		{
			_fprintf(_stderr, "Error: querying sqlite_master and sqlite_temp_master\n");
			rc = 1;
		}
	}
#pragma endregion
#pragma region .restore
#pragma endregion
#pragma region .schema
	else if (c == 's' && !_strncmp(args[0], "schema", n) && argsLength < 3)
	{
		struct CallbackData data;
		char *errMsg = 0;
		_memcpy(&data, p, sizeof(data));
		data.ShowHeader = 0;
		data.Mode = MODE_Semi;
		if (argsLength > 1)
		{
			for (int i = 0; args[1][i]; i++) args[1][i] = __tolower(args[1][i]);
			if (!_strcmp(args[1], "sqlite_master"))
			{
				char *new_argv[2], *new_colv[2];
				new_argv[0] = "CREATE TABLE sqlite_master (\n"
					"  type text,\n"
					"  name text,\n"
					"  tbl_name text,\n"
					"  rootpage integer,\n"
					"  sql text\n"
					")";
				new_argv[1] = 0;
				new_colv[0] = "sql";
				new_colv[1] = 0;
				::Callback(&data, 1, new_argv, new_colv);
				rc = RC_OK;
			}
			else if (!_strcmp(args[1], "sqlite_temp_master"))
			{
				char *new_argv[2], *new_colv[2];
				new_argv[0] = "CREATE TEMP TABLE sqlite_temp_master (\n"
					"  type text,\n"
					"  name text,\n"
					"  tbl_name text,\n"
					"  rootpage integer,\n"
					"  sql text\n"
					")";
				new_argv[1] = 0;
				new_colv[0] = "sql";
				new_colv[1] = 0;
				::Callback(&data, 1, new_argv, new_colv);
				rc = RC_OK;
			}
			else
			{
				_shellStatic = args[1];
				rc = DataEx::Exec(p->Ctx,
					"SELECT sql FROM "
					"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
					"     FROM sqlite_master UNION ALL"
					"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
					"WHERE lower(tbl_name) LIKE shellstatic()"
					"  AND type!='meta' AND sql NOTNULL "
					"ORDER BY substr(type,2,1), "
					" CASE type WHEN 'view' THEN rowid ELSE name END",
					::Callback, &data, &errMsg);
				_shellStatic = nullptr;
			}
		}
		else
		{
			rc = DataEx::Exec(p->Ctx,
				"SELECT sql FROM "
				"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
				"     FROM sqlite_master UNION ALL"
				"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
				"WHERE type!='meta' AND sql NOTNULL AND name NOT LIKE 'sqlite_%'"
				"ORDER BY substr(type,2,1),"
				" CASE type WHEN 'view' THEN rowid ELSE name END",
				::Callback, &data, &errMsg);
		}
		if (errMsg) { _fprintf(_stderr, "Error: %s\n", errMsg); _free(errMsg); rc = 1; }
		else if (rc != RC_OK) { _fprintf(_stderr, "Error: querying schema information\n"); rc = 1; }
		else rc = 0;
	}
#pragma endregion
#pragma region .tables
	else if (c == 't' && n > 1 && !_strncmp(args[0], "tables", n) && argsLength < 3)
	{
		Vdbe *stmt;
		rc = Prepare::Prepare_v2(p->Ctx, "PRAGMA database_list", -1, &stmt, 0);
		if (rc) goto _metaend;
		char *sql = _mprintf(
			"SELECT name FROM sqlite_master"
			" WHERE type IN ('table','view')"
			"   AND name NOT LIKE 'sqlite_%%'"
			"   AND name LIKE ?1");
		while (stmt->Step() == RC_ROW)
		{
			const char *dbName = (const char *)Vdbe::Column_Text(stmt, 1);
			if (!dbName || !_strcmp(dbName, "main")) continue;
			if (!_strcmp(dbName, "temp"))
				sql = _mprintf(
				"%z UNION ALL "
				"SELECT 'temp.' || name FROM sqlite_temp_master"
				" WHERE type IN ('table','view')"
				"   AND name NOT LIKE 'sqlite_%%'"
				"   AND name LIKE ?1", sql);
			else
				sql = _mprintf(
				"%z UNION ALL "
				"SELECT '%q.' || name FROM \"%w\".sqlite_master"
				" WHERE type IN ('table','view')"
				"   AND name NOT LIKE 'sqlite_%%'"
				"   AND name LIKE ?1", sql, dbName, dbName);
		}
		Vdbe::Finalize(stmt);
		sql = _mprintf("%z ORDER BY 1", sql);
		rc = Prepare::Prepare_v2(p->Ctx, sql, -1, &stmt, 0);
		_free(sql);
		if (rc) goto _metaend;
		int rows, allocs;
		rows = allocs = 0;
		char **results = nullptr;
		if (argsLength > 1)
			Vdbe::Bind_Text(stmt, 1, args[1], -1, DESTRUCTOR_TRANSIENT);
		else
			Vdbe::Bind_Text(stmt, 1, "%", -1, DESTRUCTOR_STATIC);
		while (stmt->Step() == RC_ROW)
		{
			if (rows >= allocs)
			{
				int n = allocs*2 + 10;
				char **newResults = (char **)_realloc(results, sizeof(results[0])*n);
				if (!newResults)
				{
					_fprintf(_stderr, "Error: out_ of memory\n");
					break;
				}
				allocs = n;
				results = newResults;
			}
			results[rows] = _mprintf("%s", Vdbe::Column_Text(stmt, 0));
			if (results[rows]) rows++;
		}
		Vdbe::Finalize(stmt);        
		if (rows > 0)
		{
			int i;
			int maxlen = 0;
			for (i = 0; i < rows; i++)
			{
				int len = _strlen(results[i]);
				if (len > maxlen) maxlen = len;
			}
			int printCols = 80/(maxlen+2);
			if (printCols < 1) printCols = 1;
			int printRows = (rows + printCols - 1)/printCols;
			for (i = 0; i < printRows; i++)
			{
				for (int j = i; j < rows; j += printRows)
				{
					char *sp = (j < printRows ? "" : "  ");
					_printf("%s%-*s", sp, maxlen, (results[j] ? results[j] : ""));
				}
				_printf("\n");
			}
		}
		for (int ii = 0; ii < rows; ii++) _free(results[ii]);
		_free(results);
	}
#pragma endregion
#pragma region .testctrl
#if _TEST
	else if (c == 't' && n >= 8 && !_strncmp(args[0], "testctrl", n) && argsLength >= 2)
	{
		DataEx::TESTCTRL testctrl = (DataEx::TESTCTRL)tag;
		switch (testctrl)
		{
		case DataEx::TESTCTRL_OPTIMIZATIONS:
		case DataEx::TESTCTRL_RESERVE:
			// DataEx::TestControl(int, db, int)
			if (argsLength == 3)
			{
				int opt = (int)_atoi(args[2]);
				rc = DataEx::TestControl(testctrl, p->Ctx, opt);
				_printf("%d (0x%08x)\n", rc, rc);
			}
			else
				_fprintf(_stderr, "Error: testctrl %s takes a single int option\n", args[1]);
			break;
		case DataEx::TESTCTRL_PRNG_SAVE:
		case DataEx::TESTCTRL_PRNG_RESTORE:
		case DataEx::TESTCTRL_PRNG_RESET:
			// DataEx::TestControl(int)
			if (argsLength == 2)
			{
				rc = DataEx::TestControl(testctrl);
				_printf("%d (0x%08x)\n", rc, rc);
			}
			else
				_fprintf(_stderr, "Error: testctrl %s takes no options\n", args[1]);
			break;
		case DataEx::TESTCTRL_PENDING_BYTE:
			// DataEx::TestControl(int, uint)
			if (argsLength == 3)
			{
				unsigned int opt = (unsigned int)_atoi(args[2]);
				rc = DataEx::TestControl(testctrl, opt);
				_printf("%d (0x%08x)\n", rc, rc);
			}
			else
				_fprintf(_stderr, "Error: testctrl %s takes a single unsigned int option\n", args[1]);
			break;
		case DataEx::TESTCTRL_ASSERT:
		case DataEx::TESTCTRL_ALWAYS:
			// DataEx::TestControl(int, int)
			if (argsLength == 3)
			{
				int opt = _atoi(args[2]);        
				rc = DataEx::TestControl(testctrl, opt);
				_printf("%d (0x%08x)\n", rc, rc);
			}
			else
				_fprintf(_stderr, "Error: testctrl %s takes a single int option\n", args[1]);
			break;
#ifdef N_KEYWORD
		case DataEx::TESTCTRL_ISKEYWORD:
			// DataEx::TestControl(int, char *)
			if (argsLength == 3)
			{
				const char *opt = args[2];
				rc = DataEx::TestControl(testctrl, opt);
				_printf("%d (0x%08x)\n", rc, rc);
			}
			else
				_fprintf(stderr, "Error: testctrl %s takes a single char * option\n", args[1]);
			break;
#endif
		case DataEx::TESTCTRL_BITVEC_TEST:         
		case DataEx::TESTCTRL_FAULT_INSTALL:       
		case DataEx::TESTCTRL_BENIGN_MALLOC_HOOKS: 
		case DataEx::TESTCTRL_SCRATCHMALLOC:       
		default:
			_fprintf(_stderr, "Error: CLI support for testctrl %s not implemented\n", args[1]);
			break;
		}
	}
#endif
#pragma endregion
#pragma region .timeout
	else if (c == 't' && n > 4 && !_strncmp(args[0], "timeout", n) && argsLength == 2)
	{
		DataEx::BusyTimeout(p->Ctx, _atoi(args[1]));
	}
#pragma endregion
#pragma region .trace
	else if (c == 't' && !_strncmp(args[0], "trace", n) && argsLength > 1)
	{
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
		if (!p->TraceOut)
			DataEx::Trace(p->Ctx, nullptr, nullptr);
		else
			DataEx::Trace(p->Ctx, SqlTraceCallback, p->TraceOut);
#endif
	}
#pragma endregion
#pragma region .vfsname
	else if (c == 'v' && !_strncmp(args[0], "vfsname", n))
	{
		const char *dbName = (argsLength == 2 ? args[1] : "main");
		char *vfsName = 0;
		if (p->Ctx)
		{
			DataEx::FileControl(p->Ctx, dbName, VFile::FCNTL_VFSNAME, &vfsName);
			if (vfsName)
			{
				printf("%s\n", vfsName);
				_free(vfsName);
			}
		}
	}
#pragma endregion
_metaend:
	d_return = rc;
}
#endif
static int DoMetaCommand(char *line, struct CallbackData *p)
{
#pragma region Preamble
	int i = 1;
	int argsLength = 0;
	char *args[50];

	// Parse the input line into tokens.
	while (line[i] && argsLength < _lengthof(args))
	{
		while (IsSpace(line[i])) { i++; }
		if (line[i] == 0) break;
		if (line[i] == '\'' || line[i] == '"')
		{
			int delim = line[i++];
			args[argsLength++] = &line[i];
			while (line[i] && line[i] != delim) { i++; }
			if (line[i] == delim)
				line[i++] = 0;
			if (delim == '"') ResolveBackslashes(args[argsLength-1]);
		}
		else
		{
			args[argsLength++] = &line[i];
			while (line[i] && !IsSpace(line[i])) { i++; }
			if (line[i]) line[i++] = 0;
			ResolveBackslashes(args[argsLength-1]);
		}
	}

	// Process the input line.
	if (argsLength == 0) return 0; // no tokens, no error
	int n = (int)strlen(args[0]);
	int c = args[0][0];
	int rc = 0;
#pragma endregion
	//
#pragma region .backup - todo
	if (c == 'b' && n >= 3 && !strncmp(args[0], "backup", n))
	{
#ifndef __CUDACC__
		const char *destFile = nullptr;
		const char *dbName = nullptr;
		const char *key = nullptr;
		Context *dest;
		Backup *backup;
		for (int j = 1; j < argsLength; j++)
		{
			const char *z = args[j];
			if (z[0] == '-')
			{
				while (z[0] == '-') z++;
				if (!strcmp(z, "key") && j < argsLength-1)
					key = args[++j];
				else
				{
					fprintf(stderr, "unknown option: %s\n", args[j]);
					return 1;
				}
			}
			else if (!destFile)
				destFile = args[j];
			else if (!dbName)
			{
				dbName = destFile;
				destFile = args[j];
			}
			else
			{
				fprintf(stderr, "too many arguments to .backup\n");
				return 1;
			}
		}
		if (!destFile)
		{
			fprintf(stderr, "missing FILENAME argument on .backup\n");
			return 1;
		}
		if (!dbName) dbName = "main";
		rc = DataEx::Open(destFile, &dest);
		if (rc != RC_OK)
		{
			fprintf(stderr, "Error: cannot open \"%s\"\n", destFile);
			DataEx::Close(dest);
			return 1;
		}
#ifdef HAS_CODEC
		sqlite3_key(dest, key, (int)strlen(key));
#endif
		_OpenCtx(p);
		backup = Backup::Init(dest, "main", p->Ctx, dbName);
		if (!backup)
		{
			fprintf(stderr, "Error: %s\n", DataEx::ErrMsg(dest));
			DataEx::Close(dest);
			return 1;
		}
		while ((rc = backup->Step(100)) == RC_OK) { }
		Backup::Finish(backup);
		if (rc == RC_DONE)
			rc = 0;
		else
		{
			fprintf(stderr, "Error: %s\n", DataEx::ErrMsg(dest));
			rc = 1;
		}
		DataEx::Close(dest);
#endif
	}
#pragma endregion
#pragma region .bail
	else if (c == 'b' && n >= 3 && !strncmp(args[0], "bail", n) && argsLength > 1 && argsLength < 3)
	{
		_bailOnError = BooleanValue(args[1]);
	}
#pragma endregion
#pragma region .breakpoint
	else if (c == 'b' && n >= 3 && !strncmp(args[0], "breakpoint", n))
	{
		// The undocumented ".breakpoint" command causes a call to the no-op routine named TestBreakpoint().
		TestBreakpoint();
	}
#pragma endregion
#pragma region .databases - done
	else if (c == 'd' && n > 1 && !strncmp(args[0], "databases", n) && argsLength == 1)
	{
		_OpenCtx(p);
#if __CUDACC__

		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		struct CallbackData data;
		memcpy(&data, p, sizeof(data));
		data.ShowHeader = 1;
		data.Mode = MODE_Column;
		data.ColWidth[0] = 3;
		data.ColWidth[1] = 15;
		data.ColWidth[2] = 58;
		data.Cnt = 0;
		char *errMsg = nullptr;
		DataEx::Exec(p->Ctx, "PRAGMA database_list;", ::Callback, &data, &errMsg);
		if (errMsg)
		{
			fprintf(stderr, "Error: %s\n", errMsg);
			free(errMsg);
			rc = 1;
		}
#endif
	}
#pragma endregion
#pragma region .dump - done
	else if (c == 'd' && !strncmp(args[0], "dump", n) && argsLength < 3)
	{
		_OpenCtx(p);
#if __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		// When playing back a "dump", the content might appear in an order which causes immediate foreign key constraints to be violated.
		// So disable foreign-key constraint enforcement to prevent problems.
		fprintf(p->Out, "PRAGMA foreign_keys=OFF;\n");
		fprintf(p->Out, "BEGIN TRANSACTION;\n");
		p->WritableSchema = 0;
		DataEx::Exec(p->Ctx, "SAVEPOINT dump; PRAGMA writable_schema=ON", 0, 0, 0);
		p->Errs = 0;
		if (argsLength == 1)
		{
			RunSchemaDumpQuery(p, 
				"SELECT name, type, sql FROM sqlite_master "
				"WHERE sql NOT NULL AND type=='table' AND name!='sqlite_sequence'");
			RunSchemaDumpQuery(p, 
				"SELECT name, type, sql FROM sqlite_master "
				"WHERE name=='sqlite_sequence'");
			RunTableDumpQuery(p,
				"SELECT sql FROM sqlite_master "
				"WHERE sql NOT NULL AND type IN ('index','trigger','view')", 0);
		}
		else
		{
			for (int i = 1; i < argsLength; i++)
			{
				_shellStatic = args[i];
				RunSchemaDumpQuery(p,
					"SELECT name, type, sql FROM sqlite_master "
					"WHERE tbl_name LIKE shellstatic() AND type=='table'"
					"  AND sql NOT NULL");
				RunTableDumpQuery(p,
					"SELECT sql FROM sqlite_master "
					"WHERE sql NOT NULL"
					"  AND type IN ('index','trigger','view')"
					"  AND tbl_name LIKE shellstatic()", 0);
				_shellStatic = nullptr;
			}
		}
		if (p->WritableSchema)
		{
			fprintf(p->Out, "PRAGMA writable_schema=OFF;\n");
			p->WritableSchema = 0;
		}
		DataEx::Exec(p->Ctx, "PRAGMA writable_schema=OFF;", 0, 0, 0);
		DataEx::Exec(p->Ctx, "RELEASE dump;", 0, 0, 0);
		fprintf(p->Out, (p->Errs ? "ROLLBACK; -- due to errors\n" : "COMMIT;\n"));
#endif
	}
#pragma endregion
#pragma region .echo
	else if (c == 'e' && !strncmp(args[0], "echo", n) && argsLength > 1 && argsLength < 3)
	{
		p->EchoOn = BooleanValue(args[1]);
	}
#pragma endregion
#pragma region .exit
	else if (c == 'e' && !strncmp(args[0], "exit", n))
	{
		if (argsLength > 1 && (rc = atoi(args[1])) != 0) exit(rc);
		rc = 2;
	}
#pragma endregion
#pragma region .explain
	else if (c == 'e' && !strncmp(args[0], "explain", n) && argsLength < 3)
	{
		int val = (argsLength >= 2 ? BooleanValue(args[1]) : 1);
		if (val == 1)
		{
			if (!p->ExplainPrev.Valid)
			{
				p->ExplainPrev.Valid = 1;
				p->ExplainPrev.Mode = p->Mode;
				p->ExplainPrev.ShowHeader = p->ShowHeader;
				memcpy(p->ExplainPrev.ColWidth, p->ColWidth, sizeof(p->ColWidth));
			}
			// We could put this code under the !p->explainValid condition so that it does not execute if we are already in
			// explain mode. However, always executing it allows us an easy was to reset to explain mode in case the user previously
			// did an .explain followed by a .width, .mode or .header command.
			p->Mode = MODE_Explain;
			p->ShowHeader = 1;
			memset(p->ColWidth, 0, _lengthof(p->ColWidth));
			p->ColWidth[0] = 4;		// addr
			p->ColWidth[1] = 13;	// opcode
			p->ColWidth[2] = 4;		// P1
			p->ColWidth[3] = 4;		// P2
			p->ColWidth[4] = 4;		// P3
			p->ColWidth[5] = 13;	// P4
			p->ColWidth[6] = 2;		// P5
			p->ColWidth[7] = 13;	// Comment
		}
		else if (p->ExplainPrev.Valid)
		{
			p->ExplainPrev.Valid = 0;
			p->Mode = p->ExplainPrev.Mode;
			p->ShowHeader = p->ExplainPrev.ShowHeader;
			memcpy(p->ColWidth, p->ExplainPrev.ColWidth, sizeof(p->ColWidth));
		}
	}
#pragma endregion
#pragma region .header(s)
	else if (c == 'h' && (!strncmp(args[0], "header", n) || !strncmp(args[0], "headers", n)) && argsLength > 1 && argsLength < 3)
	{
		p->ShowHeader = BooleanValue(args[1]);
	}
#pragma endregion
#pragma region .help
	else if (c == 'h' && !strncmp(args[0], "help", n))
	{
		fprintf(stderr, "%s", _help);
		if (HAS_TIMER)
			fprintf(stderr, "%s", _timerHelp);
	}
#pragma endregion
#pragma region .import - todo
	else if (c == 'i' && !strncmp(args[0], "import", n) && argsLength == 3)
	{
#ifndef __CUDACC__
		char *tableName = args[2]; // Insert data into this table
		char *file = args[1]; // The file from which to extract data
		Vdbe *stmt = nullptr; // A statement
		int i; 
		_OpenCtx(p);
		int sepLength = _strlen(p->Separator); // Number of bytes in p->separator[]
		if (sepLength == 0)
		{
			fprintf(stderr, "Error: non-null separator required for import\n");
			return 1;
		}
		char *sql = _mprintf("SELECT * FROM %s", tableName); // An SQL statement
		if (!sql)
		{
			fprintf(stderr, "Error: out_ of memory\n");
			return 1;
		}
		int bytes = _strlen(sql); // Number of bytes in an SQL string
		rc = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		_free(sql);
		if (rc)
		{
			if (stmt) Vdbe::Finalize(stmt);
			fprintf(stderr,"Error: %s\n", DataEx::ErrMsg(_ctx));
			return 1;
		}
		int colsLength = Vdbe::Column_Count(stmt); // Number of columns in the table
		Vdbe::Finalize(stmt);
		stmt = nullptr;
		if (colsLength == 0) return 0; // no columns, no error
		sql = (char *)malloc(bytes + 20 + colsLength*2);
		if (!sql)
		{
			fprintf(stderr, "Error: out_ of memory\n");
			return 1;
		}
		_snprintf(sql, bytes+20, "INSERT INTO %s VALUES(?", tableName);
		int j = _strlen(sql);
		for (i = 1; i < colsLength; i++)
		{
			sql[j++] = ',';
			sql[j++] = '?';
		}
		sql[j++] = ')';
		sql[j] = 0;
		rc = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		free(sql);
		if (rc)
		{
			fprintf(stderr, "Error: %s\n", DataEx::ErrMsg(_ctx));
			if (stmt) Vdbe::Finalize(stmt);
			return 1;
		}
		FILE *in = fopen(file, "rb"); // The input file;
		if (!in)
		{
			fprintf(stderr, "Error: cannot open \"%s\"\n", file);
			Vdbe::Finalize(stmt);
			return 1;
		}
		char **cols; // line[] broken up into columns
		cols = (char **)malloc(sizeof(cols[0])*(colsLength+1));
		if (!cols)
		{
			fprintf(stderr, "Error: out_ of memory\n");
			fclose(in);
			Vdbe::Finalize(stmt);
			return 1;
		}
		DataEx::Exec(p->Ctx, "BEGIN", 0, 0, 0);
		char *commit = "COMMIT"; // How to commit changes
		int lineno = 0; // Line number of input file
		char *line; // A single line of input from the file
		while ((line = LocalGetLine(0, in, 1)) != 0)
		{
			char *z, c;
			bool inQuote = false;
			lineno++;
			cols[0] = line;
			for (i = 0, z = line; (c = *z) != 0; z++)
			{
				if (c == '"') inQuote = !inQuote;
				if (c == '\n') lineno++;
				if (!inQuote && c == p->Separator[0] && !strncmp(z, p->Separator, sepLength))
				{
					*z = 0;
					i++;
					if (i < colsLength)
					{
						cols[i] = &z[sepLength];
						z += sepLength-1;
					}
				}
			} // end for
			*z = 0;
			if (i+1 != colsLength)
			{
				fprintf(stderr, "Error: %s line %d: expected %d columns of data but found %d\n", file, lineno, colsLength, i+1);
				commit = "ROLLBACK";
				free(line);
				rc = 1;
				break; // from while
			}
			for (i = 0; i < colsLength; i++)
			{
				if (cols[i][0]=='"')
				{
					int k;
					for (z = cols[i], j = 1, k = 0; z[j]; j++){
						if (z[j] == '"' ) { j++; if (z[j] == 0) break; }
						z[k++] = z[j];
					}
					z[k] = 0;
				}
				Vdbe::Bind_Text(stmt, i+1, cols[i], -1, DESTRUCTOR_STATIC);
			}
			stmt->Step();
			rc = stmt->Reset();
			free(line);
			if (rc != RC_OK)
			{
				fprintf(stderr,"Error: %s\n", DataEx::ErrMsg(_ctx));
				commit = "ROLLBACK";
				rc = 1;
				break; // from while
			}
		} // end while
		free(cols);
		fclose(in);
		Vdbe::Finalize(stmt);
		DataEx::Exec(p->Ctx, commit, 0, 0, 0);
#endif
	}
#pragma endregion
#pragma region .indices - done
	else if (c == 'i' && !strncmp(args[0], "indices", n) && argsLength < 3)
	{
		_OpenCtx(p);
#ifdef __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		struct CallbackData data;
		char *errMsg = nullptr;
		memcpy(&data, p, sizeof(data));
		data.ShowHeader = 0;
		data.Mode = MODE_List;
		if (argsLength == 1)
			rc = DataEx::Exec(p->Ctx,
			"SELECT name FROM sqlite_master "
			"WHERE type='index' AND name NOT LIKE 'sqlite_%' "
			"UNION ALL "
			"SELECT name FROM sqlite_temp_master "
			"WHERE type='index' "
			"ORDER BY 1",
			::Callback, &data, &errMsg);
		else
		{
			_shellStatic = args[1];
			rc = DataEx::Exec(p->Ctx,
				"SELECT name FROM sqlite_master "
				"WHERE type='index' AND tbl_name LIKE shellstatic() "
				"UNION ALL "
				"SELECT name FROM sqlite_temp_master "
				"WHERE type='index' AND tbl_name LIKE shellstatic() "
				"ORDER BY 1",
				::Callback, &data, &errMsg);
			_shellStatic = nullptr;
		}
		if (errMsg)
		{
			fprintf(stderr, "Error: %s\n", errMsg);
			free(errMsg);
			rc = 1;
		}
		else if (rc != RC_OK)
		{
			fprintf(stderr, "Error: querying sqlite_master and sqlite_temp_master\n");
			rc = 1;
		}
#endif
	}
#pragma endregion
#pragma region .iotrace
#ifdef ENABLE_IOTRACE
	else if (c == 'i' && !strncmp(args[0], "iotrace", n))
	{
		extern void (*sqlite3IoTrace)(const char*, ...);
		if (iotrace && iotrace != stdout) fclose(iotrace);
		iotrace = 0;
		if (argsLength < 2)
			sqlite3IoTrace = 0;
		else if (!strcmp(args[1], "-"))
		{
			sqlite3IoTrace = iotracePrintf;
			iotrace = stdout;
		}
		else
		{
			iotrace = fopen(args[1], "w");
			if (iotrace == 0)
			{
				fprintf(stderr, "Error: cannot open \"%s\"\n", args[1]);
				sqlite3IoTrace = 0;
				rc = 1;
			}
			else
				sqlite3IoTrace = iotracePrintf;
		}
	}
#endif
#pragma endregion
#pragma region .load
#ifndef OMIT_LOAD_EXTENSION
	//else if (c == 'l' && !strncmp(args[0], "load", n) && argsLength >= 2)
	//{
	//	const char *file = args[1];
	//	const char *proc = (argsLength >= 3 ? args[2] : 0);
	//	_OpenCtx(p);
	//	char *errMsg = 0;
	//	rc = DataEx::LoadExtension(p->Ctx, file, proc, &errMsg);
	//	if (rc != RC_OK)
	//	{
	//		fprintf(stderr, "Error: %s\n", errMsg);
	//		_free(errMsg);
	//		rc = 1;
	//	}
	//}
#endif
#pragma endregion
#pragma region .log
	else if (c == 'l' && !strncmp(args[0], "log", n) && argsLength >= 2)
	{
		const char *file = args[1];
		OutputFileClose(p->Log);
		p->Log = OutputFileOpen(file);
	}
#pragma endregion
#pragma region .mode
	else if (c == 'm' && !strncmp(args[0], "mode", n) && argsLength == 2)
	{
		int n2 = (int)strlen(args[1]);
		if ((n2 == 4 && !strncmp(args[1],"line",n2)) || (n2 == 5 && !strncmp(args[1],"lines",n2))) p->Mode = MODE_Line;
		else if ((n2 == 6 && !strncmp(args[1],"column",n2)) || (n2 == 7 && !strncmp(args[1],"columns",n2))) p->Mode = MODE_Column;
		else if (n2 == 4 && !strncmp(args[1],"list",n2)) p->Mode = MODE_List;
		else if (n2 == 4 && !strncmp(args[1],"html",n2)) p->Mode = MODE_Html;
		else if (n2 == 3 && !strncmp(args[1],"tcl",n2)) { p->Mode = MODE_Tcl; _snprintf(p->Separator, sizeof(p->Separator), " "); }
		else if (n2 == 3 && !strncmp(args[1],"csv",n2)) { p->Mode = MODE_Csv; _snprintf(p->Separator, sizeof(p->Separator), ","); }
		else if (n2 == 4 && !strncmp(args[1],"tabs",n2)) { p->Mode = MODE_List; _snprintf(p->Separator, sizeof(p->Separator), "\t"); }
		else if (n2 == 6 && !strncmp(args[1],"insert",n2)) { p->Mode = MODE_Insert; SetTableName(p, "table"); }
		else
		{
			fprintf(stderr,"Error: mode should be one of: column csv html insert line list tabs tcl\n");
			rc = 1;
		}
	}
	else if (c == 'm' && !strncmp(args[0], "mode", n) && argsLength == 3)
	{
		int n2 = (int)strlen(args[1]);
		if (n2 == 6 && !strncmp(args[1],"insert",n2)) { p->Mode = MODE_Insert; SetTableName(p, args[2]); }
		else
		{
			fprintf(stderr, "Error: invalid arguments:  \"%s\". Enter \".help\" for help\n", args[2]);
			rc = 1;
		}
	}
#pragma endregion
#pragma region .nullvalue
	else if (c == 'n' && !strncmp(args[0], "nullvalue", n) && argsLength == 2)
	{
		_snprintf(p->NullValue, sizeof(p->NullValue), "%.*s", (int)_lengthof(p->NullValue)-1, args[1]);
	}
#pragma endregion
#pragma region .output
	else if (c == 'o' && !strncmp(args[0], "output", n) && argsLength == 2)
	{
		if (p->Outfile[0] == '|') pclose(p->Out);
		else OutputFileClose(p->Out);
		p->Outfile[0] = 0;
		if (args[1][0] == '|')
		{
			p->Out = popen(&args[1][1], "w");
			if (!p->Out)
			{
				fprintf(stderr,"Error: cannot open pipe \"%s\"\n", &args[1][1]);
				p->Out = stdout;
				rc = 1;
			}
			else
				_snprintf(p->Outfile, sizeof(p->Outfile), "%s", args[1]);
		}
		else
		{
			p->Out = OutputFileOpen(args[1]);
			if (!p->Out)
			{
				if (strcmp(args[1], "off"))
					fprintf(stderr, "Error: cannot write to \"%s\"\n", args[1]);
				p->Out = stdout;
				rc = 1;
			}
			else
				_snprintf(p->Outfile, sizeof(p->Outfile), "%s", args[1]);
		}
		H_DIRTY(p);
	}
#pragma endregion
#pragma region .print
	else if (c == 'p' && n >= 3 && !strncmp(args[0], "print", n))
	{
		for (int i = 1; i < argsLength; i++)
		{
			if (i > 1) fprintf(p->Out, " ");
			fprintf(p->Out, "%s", args[i]);
		}
		fprintf(p->Out, "\n");
	}
#pragma endregion
#pragma region .prompt
	else if (c == 'p' && !strncmp(args[0], "prompt", n) && (argsLength == 2 || argsLength == 3))
	{
		if (argsLength >= 2)
			strncpy(_mainPrompt, args[1], (int)_lengthof(_mainPrompt)-1);
		if (argsLength >= 3)
			strncpy(_continuePrompt, args[2], (int)_lengthof(_continuePrompt)-1);
	}
#pragma endregion
#pragma region .quit
	else if (c == 'q' && !strncmp(args[0], "quit", n) && argsLength == 1)
	{
		rc = 2;
	}
#pragma endregion
#pragma region .read
	else if (c == 'r' && n >= 3 && !strncmp(args[0], "read", n) && argsLength == 2)
	{
		FILE *alt = fopen(args[1], "rb");
		if (!alt)
		{
			fprintf(stderr, "Error: cannot open \"%s\"\n", args[1]);
			rc = 1;
		}
		else
		{
			rc = ProcessInput(p, alt);
			fclose(alt);
		}
	}
#pragma endregion
#pragma region .restore - todo
	else if (c == 'r' && n >= 3 && !strncmp(args[0], "restore", n) && argsLength > 1 && argsLength < 4)
	{
#ifndef __CUDACC__
		const char *srcFile;
		const char *dbName;
		if (argsLength == 2)
		{
			srcFile = args[1];
			dbName = "main";
		}
		else
		{
			srcFile = args[2];
			dbName = args[1];
		}
		Context *src;
		rc = DataEx::Open(srcFile, &src);
		if (rc != RC_OK)
		{
			fprintf(stderr, "Error: cannot open \"%s\"\n", srcFile);
			DataEx::Close(src);
			return 1;
		}
		_OpenCtx(p);
		Backup *backup = Backup::Init(p->Ctx, dbName, src, "main");
		if (!backup)
		{
			fprintf(stderr, "Error: %s\n", DataEx::ErrMsg(p->Ctx));
			DataEx::Close(src);
			return 1;
		}
		int timeout = 0;
		while ((rc = backup->Step(100)) == RC_OK || rc == RC_BUSY)
		{
			if (rc == RC_BUSY)
			{
				if (timeout++ >= 3) break;
				_sleep(100);
			}
		}
		Backup::Finish(backup);
		if (rc == RC_DONE) rc = 0;
		else if (rc == RC_BUSY || rc == RC_LOCKED) { fprintf(stderr, "Error: source database is busy\n"); rc = 1; }
		else { fprintf(stderr, "Error: %s\n", DataEx::ErrMsg(p->Ctx)); rc = 1; }
		DataEx::Close(src);
#endif
	}
#pragma endregion
#pragma region .schema - done
	else if (c == 's' && !strncmp(args[0], "schema", n) && argsLength < 3)
	{
		_OpenCtx(p);
#ifdef __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		struct CallbackData data;
		char *errMsg = 0;
		memcpy(&data, p, sizeof(data));
		data.ShowHeader = 0;
		data.Mode = MODE_Semi;
		if (argsLength > 1)
		{
			for (int i = 0; args[1][i]; i++) args[1][i] = ToLower(args[1][i]);
			if (!strcmp(args[1], "sqlite_master"))
			{
				char *new_argv[2], *new_colv[2];
				new_argv[0] = "CREATE TABLE sqlite_master (\n"
					"  type text,\n"
					"  name text,\n"
					"  tbl_name text,\n"
					"  rootpage integer,\n"
					"  sql text\n"
					")";
				new_argv[1] = 0;
				new_colv[0] = "sql";
				new_colv[1] = 0;
				::Callback(&data, 1, new_argv, new_colv);
				rc = RC_OK;
			}
			else if (!strcmp(args[1], "sqlite_temp_master"))
			{
				char *new_argv[2], *new_colv[2];
				new_argv[0] = "CREATE TEMP TABLE sqlite_temp_master (\n"
					"  type text,\n"
					"  name text,\n"
					"  tbl_name text,\n"
					"  rootpage integer,\n"
					"  sql text\n"
					")";
				new_argv[1] = 0;
				new_colv[0] = "sql";
				new_colv[1] = 0;
				::Callback(&data, 1, new_argv, new_colv);
				rc = RC_OK;
			}
			else
			{
				_shellStatic = args[1];
				rc = DataEx::Exec(p->Ctx,
					"SELECT sql FROM "
					"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
					"     FROM sqlite_master UNION ALL"
					"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
					"WHERE lower(tbl_name) LIKE shellstatic()"
					"  AND type!='meta' AND sql NOTNULL "
					"ORDER BY substr(type,2,1), "
					" CASE type WHEN 'view' THEN rowid ELSE name END",
					::Callback, &data, &errMsg);
				_shellStatic = nullptr;
			}
		}
		else
		{
			rc = DataEx::Exec(p->Ctx,
				"SELECT sql FROM "
				"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
				"     FROM sqlite_master UNION ALL"
				"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
				"WHERE type!='meta' AND sql NOTNULL AND name NOT LIKE 'sqlite_%'"
				"ORDER BY substr(type,2,1),"
				" CASE type WHEN 'view' THEN rowid ELSE name END",
				::Callback, &data, &errMsg);
		}
		if (errMsg) { fprintf(stderr, "Error: %s\n", errMsg); _free(errMsg); rc = 1; }
		else if (rc != RC_OK) { fprintf(stderr, "Error: querying schema information\n"); rc = 1; }
		else rc = 0;
#endif	
	}
#pragma endregion
#pragma region .separator
	else if (c == 's' && !strncmp(args[0], "separator", n) && argsLength == 2)
	{
		_snprintf(p->Separator, sizeof(p->Separator), "%.*s", (int)sizeof(p->Separator)-1, args[1]);
	}
#pragma endregion
#pragma region .show
	else if (c == 's' && !strncmp(args[0], "show", n) && argsLength == 1)
	{
		fprintf(p->Out,"%9.9s: %s\n","echo", p->EchoOn ? "on" : "off");
		fprintf(p->Out,"%9.9s: %s\n","explain", p->ExplainPrev.Valid ? "on" :"off");
		fprintf(p->Out,"%9.9s: %s\n","headers", p->ShowHeader ? "on" : "off");
		fprintf(p->Out,"%9.9s: %s\n","mode", _modeDescr[p->Mode]);
		fprintf(p->Out,"%9.9s: ", "nullvalue");
		fOutputCString(p->Out, p->NullValue);
		fprintf(p->Out, "\n");
		fprintf(p->Out,"%9.9s: %s\n","output", strlen(p->Outfile) ? p->Outfile : "stdout");
		fprintf(p->Out,"%9.9s: ", "separator");
		fOutputCString(p->Out, p->Separator);
		fprintf(p->Out, "\n");
		fprintf(p->Out,"%9.9s: %s\n","stats", p->StatsOn ? "on" : "off");
		fprintf(p->Out,"%9.9s: ","width");
		for (int i = 0; i < (int)_lengthof(p->ColWidth) && p->ColWidth[i] != 0; i++)
			fprintf(p->Out, "%d ", p->ColWidth[i]);
		fprintf(p->Out,"\n");
	}
#pragma endregion
#pragma region .stats
	else if (c == 's' && !strncmp(args[0], "stats", n) && argsLength > 1 && argsLength < 3)
	{
		p->StatsOn = BooleanValue(args[1]);
	}
#pragma endregion
#pragma region .tables - done
	else if (c == 't' && n > 1 && !strncmp(args[0], "tables", n) && argsLength < 3)
	{
		_OpenCtx(p);
#if __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		Vdbe *stmt;
		rc = Prepare::Prepare_v2(p->Ctx, "PRAGMA database_list", -1, &stmt, 0);
		if (rc) return rc;
		char *sql = _mprintf(
			"SELECT name FROM sqlite_master"
			" WHERE type IN ('table','view')"
			"   AND name NOT LIKE 'sqlite_%%'"
			"   AND name LIKE ?1");
		while (stmt->Step() == RC_ROW)
		{
			const char *dbName = (const char *)Vdbe::Column_Text(stmt, 1);
			if (!dbName || !strcmp(dbName, "main")) continue;
			if (!strcmp(dbName, "temp"))
				sql = _mprintf(
				"%z UNION ALL "
				"SELECT 'temp.' || name FROM sqlite_temp_master"
				" WHERE type IN ('table','view')"
				"   AND name NOT LIKE 'sqlite_%%'"
				"   AND name LIKE ?1", sql);
			else
				sql = _mprintf(
				"%z UNION ALL "
				"SELECT '%q.' || name FROM \"%w\".sqlite_master"
				" WHERE type IN ('table','view')"
				"   AND name NOT LIKE 'sqlite_%%'"
				"   AND name LIKE ?1", sql, dbName, dbName);
		}
		Vdbe::Finalize(stmt);
		sql = _mprintf("%z ORDER BY 1", sql);
		rc = Prepare::Prepare_v2(p->Ctx, sql, -1, &stmt, 0);
		_free(sql);
		if (rc) return rc;
		int rows, allocs;
		rows = allocs = 0;
		char **results = nullptr;
		if (argsLength > 1)
			Vdbe::Bind_Text(stmt, 1, args[1], -1, DESTRUCTOR_TRANSIENT);
		else
			Vdbe::Bind_Text(stmt, 1, "%", -1, DESTRUCTOR_STATIC);
		while (stmt->Step() == RC_ROW)
		{
			if (rows >= allocs)
			{
				int n = allocs*2 + 10;
				char **newResults = (char **)_realloc(results, sizeof(results[0])*n);
				if (!newResults)
				{
					fprintf(stderr, "Error: out_ of memory\n");
					break;
				}
				allocs = n;
				results = newResults;
			}
			results[rows] = _mprintf("%s", Vdbe::Column_Text(stmt, 0));
			if (results[rows]) rows++;
		}
		Vdbe::Finalize(stmt);        
		if (rows > 0)
		{
			int i;
			int maxlen = 0;
			for (i = 0; i < rows; i++)
			{
				int len = _strlen(results[i]);
				if (len > maxlen) maxlen = len;
			}
			int printCols = 80/(maxlen+2);
			if (printCols < 1) printCols = 1;
			int printRows = (rows + printCols - 1)/printCols;
			for (i = 0; i < printRows; i++)
			{
				for (int j = i; j < rows; j += printRows)
				{
					char *sp = (j < printRows ? "" : "  ");
					printf("%s%-*s", sp, maxlen, (results[j] ? results[j] : ""));
				}
				printf("\n");
			}
		}
		for (int ii = 0; ii < rows; ii++) _free(results[ii]);
		_free(results);
#endif
	}
#pragma endregion
#pragma region .testctrl - done
#ifdef _TEST
	else if (c == 't' && n >= 8 && !strncmp(args[0], "testctrl", n) && argsLength >= 2)
	{
		static const struct
		{
			const char *CtrlName;   // Name of a test-control option
			DataEx::TESTCTRL CtrlCode;           // Integer code for that option
		} _ctrls[] = {
			{ "prng_save",             DataEx::TESTCTRL_PRNG_SAVE              },
			{ "prng_restore",          DataEx::TESTCTRL_PRNG_RESTORE           },
			{ "prng_reset",            DataEx::TESTCTRL_PRNG_RESET             },
			{ "bitvec_test",           DataEx::TESTCTRL_BITVEC_TEST            },
			{ "fault_install",         DataEx::TESTCTRL_FAULT_INSTALL          },
			{ "benign_malloc_hooks",   DataEx::TESTCTRL_BENIGN_MALLOC_HOOKS    },
			{ "pending_byte",          DataEx::TESTCTRL_PENDING_BYTE           },
			{ "assert",                DataEx::TESTCTRL_ASSERT                 },
			{ "always",                DataEx::TESTCTRL_ALWAYS                 },
			{ "reserve",               DataEx::TESTCTRL_RESERVE                },
			{ "optimizations",         DataEx::TESTCTRL_OPTIMIZATIONS          },
			{ "iskeyword",             DataEx::TESTCTRL_ISKEYWORD              },
			{ "scratchmalloc",         DataEx::TESTCTRL_SCRATCHMALLOC          },
		};
		DataEx::TESTCTRL testctrl = (DataEx::TESTCTRL)-1;

		// convert testctrl text option to value. allow any unique prefix of the option name, or a numerical value.
		int n = (int)strlen(args[1]);
		for (int i = 0; i < (int)_lengthof(_ctrls); i++)
		{
			if (!strncmp(args[1], _ctrls[i].CtrlName, n))
			{
				if (testctrl < 0)
					testctrl = _ctrls[i].CtrlCode;
				else
				{
					fprintf(stderr, "ambiguous option name: \"%s\"\n", args[1]);
					testctrl = (DataEx::TESTCTRL)-1;
					break;
				}
			}
		}
		if (testctrl < 0) testctrl = (DataEx::TESTCTRL)atoi(args[1]);
		if ((testctrl < DataEx::TESTCTRL_FIRST) || (testctrl > DataEx::TESTCTRL_LAST))
		{
			fprintf(stderr, "Error: invalid testctrl option: %s\n", args[1]);
			return 0;
		}
		_OpenCtx(p);
#ifdef __CUDACC__
		D_DATA(p); D_META(p, argsLength, args, testctrl); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		switch (testctrl)
		{
		case DataEx::TESTCTRL_OPTIMIZATIONS:
		case DataEx::TESTCTRL_RESERVE:
			// DataEx::TestControl(int, db, int)
			if (argsLength == 3)
			{
				int opt = (int)atoi(args[2]);
				rc = DataEx::TestControl(testctrl, p->Ctx, opt);
				printf("%d (0x%08x)\n", rc, rc);
			}
			else
				fprintf(stderr, "Error: testctrl %s takes a single int option\n", args[1]);
			break;
		case DataEx::TESTCTRL_PRNG_SAVE:
		case DataEx::TESTCTRL_PRNG_RESTORE:
		case DataEx::TESTCTRL_PRNG_RESET:
			// DataEx::TestControl(int)
			if (argsLength == 2)
			{
				rc = DataEx::TestControl(testctrl);
				printf("%d (0x%08x)\n", rc, rc);
			}
			else
				fprintf(stderr, "Error: testctrl %s takes no options\n", args[1]);
			break;
		case DataEx::TESTCTRL_PENDING_BYTE:
			// DataEx::TestControl(int, uint)
			if (argsLength == 3)
			{
				unsigned int opt = (unsigned int)atoi(args[2]);
				rc = DataEx::TestControl(testctrl, opt);
				printf("%d (0x%08x)\n", rc, rc);
			}
			else
				fprintf(stderr, "Error: testctrl %s takes a single unsigned int option\n", args[1]);
			break;
		case DataEx::TESTCTRL_ASSERT:
		case DataEx::TESTCTRL_ALWAYS:
			// DataEx::TestControl(int, int)
			if (argsLength == 3)
			{
				int opt = atoi(args[2]);        
				rc = DataEx::TestControl(testctrl, opt);
				printf("%d (0x%08x)\n", rc, rc);
			}
			else
				fprintf(stderr, "Error: testctrl %s takes a single int option\n", args[1]);
			break;
#ifdef N_KEYWORD
		case DataEx::TESTCTRL_ISKEYWORD:
			// DataEx::TestControl(int, char *)
			if (argsLength == 3)
			{
				const char *opt = args[2];
				rc = DataEx::TestControl(testctrl, opt);
				printf("%d (0x%08x)\n", rc, rc);
			}
			else
				fprintf(stderr, "Error: testctrl %s takes a single char * option\n", args[1]);
			break;
#endif
		case DataEx::TESTCTRL_BITVEC_TEST:         
		case DataEx::TESTCTRL_FAULT_INSTALL:       
		case DataEx::TESTCTRL_BENIGN_MALLOC_HOOKS: 
		case DataEx::TESTCTRL_SCRATCHMALLOC:       
		default:
			fprintf(stderr, "Error: CLI support for testctrl %s not implemented\n", args[1]);
			break;
		}
#endif
	}
#endif
#pragma endregion
#pragma region .timeout - done
	else if (c == 't' && n > 4 && !strncmp(args[0], "timeout", n) && argsLength == 2)
	{
		_OpenCtx(p);
#ifdef __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		DataEx::BusyTimeout(p->Ctx, atoi(args[1]));
#endif
	}
#pragma endregion
#pragma region .timer
	else if (HAS_TIMER && c == 't' && n >= 5 && !strncmp(args[0], "timer", n) && argsLength == 2)
	{
		_enableTimer = BooleanValue(args[1]);
	}
#pragma endregion
#pragma region .trace - done
	else if (c == 't' && !strncmp(args[0], "trace", n) && argsLength > 1)
	{
		_OpenCtx(p);
		OutputFileClose(p->TraceOut);
		p->TraceOut = OutputFileOpen(args[1]);
#ifdef __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
		if (!p->TraceOut)
			DataEx::Trace(p->Ctx, nullptr, nullptr);
		else
			DataEx::Trace(p->Ctx, SqlTraceCallback, p->TraceOut);
#endif
#endif
	}
#pragma endregion
#pragma region .version
	else if (c == 'v' && !strncmp(args[0], "version", n))
	{
		printf("SQLite %s %s\n", CORE_VERSION, CORE_SOURCE_ID);
	}
#pragma endregion
#pragma region .vfsname - done
	else if (c == 'v' && !strncmp(args[0], "vfsname", n))
	{
#if __CUDACC__
		D_DATA(p); D_META(p, argsLength, args); H_DATA(p);
		H_RETURN(); if (h_return) rc = h_return;
#else
		const char *dbName = (argsLength == 2 ? args[1] : "main");
		char *vfsName = 0;
		if (p->Ctx)
		{
			DataEx::FileControl(p->Ctx, dbName, VFile::FCNTL_VFSNAME, &vfsName);
			if (vfsName)
			{
				printf("%s\n", vfsName);
				_free(vfsName);
			}
		}
#endif
	}
#pragma endregion
#pragma region .wheretrace
#if defined(_DEBUG) && defined(ENABLE_WHERETRACE)
	else if (c == 'w' && !strncmp(args[0], "wheretrace", n))
	{
		extern int sqlite3WhereTrace;
		sqlite3WhereTrace = atoi(args[1]);
	}
#endif
#pragma endregion
#pragma region .width
	else if (c == 'w' && !strncmp(args[0], "width", n) && argsLength > 1)
	{
		assert(argsLength <= _lengthof(args));
		for (int j = 1; j < argsLength && j < _lengthof(p->ColWidth); j++)
			p->ColWidth[j-1] = atoi(args[j]);
	}
#pragma endregion
	else
	{
		fprintf(stderr, "Error: unknown command or invalid arguments:  \"%s\". Enter \".help\" for help\n", args[0]);
		rc = 1;
	}
	return rc;
}

#pragma endregion

#pragma region Parse

// Return TRUE if a semicolon occurs anywhere in the first N characters of string z[].
static bool _contains_semicolon(const char *z, int length)
{
	for (int i = 0; i < length; i++) { if (z[i] == ';') return true; }
	return false;
}

// Test to see if a line consists entirely of whitespace.
static bool _all_whitespace(const char *z)
{
	for (; *z; z++)
	{
		if (IsSpace(z[0])) continue;
		if (*z == '/' && z[1] == '*')
		{
			z += 2;
			while (*z && (*z != '*' || z[1] != '/')) { z++; }
			if (!*z) return false;
			z++;
			continue;
		}
		if (*z == '-' && z[1] == '-')
		{
			z += 2;
			while (*z && *z != '\n') { z++; }
			if (!*z) return true;
			continue;
		}
		return false;
	}
	return true;
}

// Return TRUE if the line typed in is an SQL command terminator other than a semi-colon.  The SQL Server style "go" command is understood as is the Oracle "/".
static bool _is_command_terminator(const char *line)
{
	while (IsSpace(line[0])) { line++; };
	if (line[0] == '/' && _all_whitespace(&line[1])) return true; // Oracle
	if (ToLower(line[0]) == 'g' && ToLower(line[1]) == 'o' && _all_whitespace(&line[2])) return true; // SQL Server
	return false;
}

// Return true if sql is a complete SQL statement.  Return false if it ends in the middle of a string literal or C-style comment.
static bool _is_complete(char *sql, int sqlLength)
{
	if (!sql) return 1;
	sql[sqlLength] = ';';
	sql[sqlLength+1] = 0;
	bool rc = Parse::Complete(sql);
	sql[sqlLength] = 0;
	return rc;
}

// Read input from *in and process it.  If *in==0 then input is interactive - the user is typing it it.  Otherwise, input
// is coming from a file or device.  A prompt is issued and history is saved only if input is interactive.  An interrupt signal will
// cause this routine to exit immediately, unless input is interactive.
//
// Return the number of errors.
#if __CUDACC__
__global__ void d_ProcessInput_0(struct CallbackData *p, char *sql, int startline)
{
	int rc;
	char *errMsg;
	rc = ShellExec(p->Ctx, sql, ShellCallback, p, &errMsg);
	if (rc || errMsg)
	{
		char prefix[100];
		if (startline != -1)
			__snprintf(prefix, sizeof(prefix), "Error: near line %d:", startline);
		else
			__snprintf(prefix, sizeof(prefix), "Error:");
		if (errMsg)
		{
			_fprintf(_stderr, "%s %s\n", prefix, errMsg);
			_free(errMsg);
			errMsg = nullptr;
		}
		else
			_fprintf(_stderr, "%s %s\n", prefix, DataEx::ErrMsg(p->Ctx));
		d_return = -1;
		return;
	}
	d_return = 0;
}
#endif
static bool ProcessInput(struct CallbackData *p, FILE *in)
{
	char *line = 0;
	char *sql = 0;
	int sqlLength = 0;
	int sqlLengthPrior = 0;
	int rc;
	int errCnt = 0;
	int lineno = 0;
	int startline = 0;
	while (errCnt == 0 || !_bailOnError || (in == 0 && _stdinIsInteractive))
	{
		fflush(p->Out);
		free(line);
		line = OneInputLine(sql, in);
		if (!line) // End of input
		{
			if (_stdinIsInteractive) printf("\n");
			break;
		}
		if (_seenInterrupt)
		{
			if (in) break;
			_seenInterrupt = 0;
		}
		lineno++;
		if ((!sql || sql[0] == 0) && _all_whitespace(line)) continue;
		if (line && line[0] == '.' && sqlLength == 0)
		{
			if (p->EchoOn) printf("%s\n", line);
			rc = DoMetaCommand(line, p);
			if (rc == 2) break; // exit requested
			else if (rc) errCnt++;
			continue;
		}
		if (_is_command_terminator(line) && _is_complete(sql, sqlLength))
			memcpy(line, ";", 2);
		sqlLengthPrior = sqlLength;
		if (!sql)
		{
			int i;
			for (i = 0; line[i] && IsSpace(line[i]); i++) { }
			if (line[i] != 0)
			{
				sqlLength = (int)strlen(line);
				sql = (char *)malloc(sqlLength+3);
				if (!sql)
				{
					fprintf(stderr, "Error: out_ of memory\n");
					exit(1);
				}
				memcpy(sql, line, sqlLength+1);
				startline = lineno;
			}
		}
		else
		{
			int lineLength = (int)strlen(line);
			sql = (char *)realloc(sql, sqlLength + lineLength + 4);
			if (!sql)
			{
				fprintf(stderr,"Error: out_ of memory\n");
				exit(1);
			}
			sql[sqlLength++] = '\n';
			memcpy(&sql[sqlLength], line, lineLength+1);
			sqlLength += lineLength;
		}
		if (sql && _contains_semicolon(&sql[sqlLengthPrior], sqlLength-sqlLengthPrior) && Parse::Complete(sql))
		{
			p->Cnt = 0;
			_OpenCtx(p);
			BEGIN_TIMER;
#if __CUDACC__
			char *d_sql;
			cudaMalloc((void**)&d_sql, sqlLength + 1);
			cudaMemcpy(d_sql, sql, sqlLength + 1, cudaMemcpyHostToDevice);
			int d_startline = (in != 0 || !_stdinIsInteractive ? startline : -1);
			D_DATA(p); d_ProcessInput_0<<<1,1>>>(p->D_, d_sql, d_startline); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(p);
			cudaFree(d_sql);
			H_RETURN(); if (h_return) errCnt++;
#else
			char *errMsg;
			rc = ShellExec(p->Ctx, sql, ShellCallback, p, &errMsg);
			if (rc || errMsg)
			{
				char prefix[100];
				if (in != 0 || !_stdinIsInteractive)
					_snprintf(prefix, sizeof(prefix), "Error: near line %d:", startline);
				else
					_snprintf(prefix, sizeof(prefix), "Error:");
				if (errMsg)
				{
					fprintf(stderr, "%s %s\n", prefix, errMsg);
					_free(errMsg);
					errMsg = nullptr;
				}
				else
					fprintf(stderr, "%s %s\n", prefix, DataEx::ErrMsg(p->Ctx));
				errCnt++;
			}
#endif
			END_TIMER;
			free(sql);
			sql = nullptr;
			sqlLength = 0;
		}
	}
	if (sql)
	{
		if (!_all_whitespace(sql))
			fprintf(stderr, "Error: incomplete SQL: %s\n", sql);
		free(sql);
	}
	free(line);
	return (errCnt > 0);
}

#pragma endregion

#pragma region Name2

// Return a pathname which is the user's home directory.  A 0 return indicates an error of some kind.
static char *FindHomeDir()
{
	static char *homeDir = NULL;
	if (homeDir) return homeDir;
#if !defined(_WIN32) && !defined(WIN32) && !defined(_WIN32_WCE) && !defined(__RTP__) && !defined(_WRS_KERNEL)
	{
		struct passwd *pwent;
		uid_t uid = getuid();
		if ((pwent = getpwuid(uid)) != NULL)
			homeDir = pwent->pw_dir;
	}
#endif
#if defined(_WIN32_WCE)
	homeDir = "/"; // Windows CE (arm-wince-mingw32ce-gcc) does not provide getenv()
#else
#if defined(_WIN32) || defined(WIN32)
	if (!homeDir) homeDir = getenv("USERPROFILE");
#endif
	if (!homeDir) homeDir = getenv("HOME");

#if defined(_WIN32) || defined(WIN32)
	if (!homeDir)
	{
		char *drive = getenv("HOMEDRIVE");
		char *path = getenv("HOMEPATH");
		if (drive && path)
		{
			int n = (int)strlen(drive) + (int)strlen(path) + 1;
			homeDir = (char *)malloc(n);
			if (!homeDir) return nullptr;
			_snprintf(homeDir, n, "%s%s", drive, path);
			return homeDir;
		}
		homeDir = "c:\\";
	}
#endif
#endif // !_WIN32_WCE
	if (homeDir)
	{
		int n = (int)strlen(homeDir) + 1;
		char *z = (char *)malloc(n);
		if (z) memcpy(z, homeDir, n);
		homeDir = z;
	}
	return homeDir;
}

// Read input from the file given by sqliterc_override.  Or if that parameter is NULL, take input from ~/.sqliterc
// Returns the number of errors.
static int ProcessSqliteRC(struct CallbackData *p, const char *sqliterc)
{
	char *homeDir = nullptr;
	char b[FILENAME_MAX];
	FILE *in = nullptr;
	int rc = 0;
	if (sqliterc == nullptr)
	{
		homeDir = FindHomeDir();
		if (!homeDir)
		{
#if !defined(__RTP__) && !defined(_WRS_KERNEL)
			fprintf(stderr, "%s: Error: cannot locate your home directory\n", Argv0);
#endif
			return 1;
		}
		_snprintf(b, sizeof(b), "%s/.sqliterc", homeDir);
		sqliterc = b;
	}
	in = fopen(sqliterc, "rb");
	if (in)
	{
		if (_stdinIsInteractive)
			fprintf(stderr, "-- Loading resources from %s\n", sqliterc);
		rc = ProcessInput(p, in);
		fclose(in);
	}
	return rc;
}

#pragma endregion

#pragma region DataEx

// Show available command line options
static const char _options[] = 
	"   -bail                stop after hitting an error\n"
	"   -batch               force batch I/O\n"
	"   -column              set output mode to 'column'\n"
	"   -cmd COMMAND         run \"COMMAND\" before reading stdin\n"
	"   -csv                 set output mode to 'csv'\n"
	"   -echo                print commands before execution\n"
	"   -init FILENAME       read/process named file\n"
	"   -[no]header          turn headers on or off\n"
#if defined(ENABLE_MEMSYS3) || defined(ENABLE_MEMSYS5)
	"   -heap SIZE           Size of heap for memsys3 or memsys5\n"
#endif
	"   -help                show this message\n"
	"   -html                set output mode to HTML\n"
	"   -interactive         force interactive I/O\n"
	"   -line                set output mode to 'line'\n"
	"   -list                set output mode to 'list'\n"
#ifdef ENABLE_MULTIPLEX
	"   -multiplex           enable the multiplexor VFS\n"
#endif
	"   -nullvalue TEXT      set text string for NULL values. Default ''\n"
	"   -separator SEP       set output field separator. Default: '|'\n"
	"   -stats               print memory stats before each finalize\n"
	"   -version             show SQLite version\n"
	"   -vfs NAME            use NAME as the default VFS\n"
#ifdef ENABLE_VFSTRACE
	"   -vfstrace            enable tracing of all VFS calls\n"
#endif
	;
static void Usage(bool showDetail)
{
	fprintf(stderr,
		"Usage: %s [OPTIONS] FILENAME [SQL]\n"  
		"FILENAME is the name of an SQLite database. A new database is created\n"
		"if the file does not previously exist.\n", Argv0);
	if (showDetail)
		fprintf(stderr, "OPTIONS include:\n%s", _options);
	else
		fprintf(stderr, "Use the -help option for additional information\n");
	exit(1);
}

// Initialize the state information in data
struct CallbackData _data;
#if __CUDACC__
__global__ void d_MainInit_0(struct CallbackData *data)
{
	SysEx::Config(SysEx::CONFIG_URI, 1);
	SysEx::Config(SysEx::CONFIG_LOG, ShellLog, data);
	SysEx::Config(SysEx::CONFIG_SINGLETHREAD);
}
#endif
static void MainInit()
{
	memset(&_data, 0, sizeof(_data));
	_data.Mode = MODE_List;
	memcpy(_data.Separator,"|", 2);
	_data.ShowHeader = 0;
	_snprintf(_mainPrompt, sizeof(_mainPrompt), "sqlite> ");
	_snprintf(_continuePrompt, sizeof(_continuePrompt), "   ...> ");
#if __CUDACC__
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*15));
	_deviceHeap = cudaDeviceHeapCreate(256, 100);
	cudaErrorCheck(cudaDeviceHeapSelect(_deviceHeap));
	//
	D_DATA(&_data); d_MainInit_0<<<1,1>>>(_data.D_); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(&_data);
	cudaDeviceHeapSynchronize(_deviceHeap);
	//cudaErrorCheck(cudaIobSelect());
#else
	SysEx::Config(SysEx::CONFIG_URI, 1);
	SysEx::Config(SysEx::CONFIG_LOG, ShellLog, _data);
	SysEx::Config(SysEx::CONFIG_SINGLETHREAD);
#endif
#if OS_MAP
	CoreS::VSystemSentinel::Initialize();
#endif
}

#if __CUDACC__
__global__ void d_MainShutdown_0(struct CallbackData *data)
{
	if (data->Ctx)
		DataEx::Close(data->Ctx);
	DataEx::Shutdown();
}
#endif
static void MainShutdown()
{
#if OS_MAP
	CoreS::VSystemSentinel::Shutdown();
#endif
#if __CUDACC__
	D_DATA(&_data); d_MainShutdown_0<<<1,1>>>(_data.D_); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(&_data);
	D_FREE(&_data);
	//
	cudaDeviceHeapDestroy(_deviceHeap);
	cudaDeviceReset();
#else
	if (_data.Ctx)
		DataEx::Close(_data.Ctx);
	DataEx::Shutdown();
#endif
}

// Get the argument to an --option.  Throw an error and die if no argument is available.
static char *CmdlineOptionValue(int argc, char **argv, int i)
{
	if (i == argc)
	{
		fprintf(stderr, "%s: Error: missing argument to %s\n", argv[0], argv[argc-1]);
		exit(1);
	}
	return argv[i];
}

#if __CUDACC__
__global__ void d_main_Vfs(char *name)
{
	VSystem *vfs = VSystem::FindVfs(name);
	if (vfs)
		VSystem::RegisterVfs(vfs, true);
	else
	{
		d_return = -1; return;
	}
	d_return = 0;
}

__global__ void d_main_ShellExec(struct CallbackData *p, char *sql)
{
	char *errMsg = 0;
	int rc = ShellExec(p->Ctx, sql, ShellCallback, p, &errMsg);
	if (errMsg)
	{
		_fprintf(_stderr, "Error: %s\n", errMsg);
		d_return = (rc ? rc : 1);
		return;
	}
	else if (rc)
	{
		_fprintf(_stderr, "Error: unable to process SQL \"%s\"\n", sql);
		d_return = rc;
		return;
	}
	d_return = 0;
}
#endif
int main(int argc, char **argv)
{
	//atexit(MainShutdown);
	char *errMsg = 0;
	const char *initFile = 0;
	char *firstCmd = 0;
	int i;
	int rc = 0;

	//if (strcmp(CORE_SOURCE_ID, SQLITE_SOURCE_ID))
	//{
	//	fprintf(stderr, "SQLite header and source version mismatch\n%s\n%s\n", CORE_SOURCE_ID, SQLITE_SOURCE_ID);
	//	exit(1);
	//}
	Argv0 = argv[0];
	MainInit();
	_stdinIsInteractive = isatty(0);

	// Make sure we have a valid signal handler early, before anything else is done.
#ifdef SIGINT
	signal(SIGINT, InterruptHandler);
#endif
	H_DIRTY(&_data);
	//_data.DbFilename = "\\T_\\t.db";

	// Do an initial pass through the command-line argument to locate the name of the database file, the name of the initialization file,
	// the size of the alternative malloc heap, and the first command to execute.
	for (i = 1; i < argc; i++)
	{
		char *z;
		z = argv[i];
		if (z[0] != '-')
		{
			if (!_data.DbFilename)
			{
				H_DIRTY(&_data);
				_data.DbFilename = z;
				continue;
			}
			if (!firstCmd)
			{
				firstCmd = z;
				continue;
			}
			fprintf(stderr, "%s: Error: too many options: \"%s\"\n", Argv0, argv[i]);
			fprintf(stderr, "Use -help for a list of options.\n");
			return 1;
		}
		if (z[1] == '-') z++;
		if (!strcmp(z, "-separator") || !strcmp(z, "-nullvalue") || !strcmp(z, "-cmd")) CmdlineOptionValue(argc, argv, ++i);
		else if (!strcmp(z, "-init")) initFile = CmdlineOptionValue(argc, argv, ++i);
		// Need to check for batch mode here to so we can avoid printing informational messages (like from ProcessSqliteRC) before 
		// we do the actual processing of arguments later in a second pass.
		else if (!strcmp(z, "-batch")) _stdinIsInteractive = 0;
#if defined(ENABLE_MEMSYS3) || defined(ENABLE_MEMSYS5)
		else if (!strcmp(z, "-heap"))
		{
			const char *sizeAsString = CmdlineOptionValue(argc, argv, ++i);
			int64 sizeHeap = atoi(sizeAsString);
			int c;
			for (int j = 0; (c = sizeAsString[j]) != 0; j++)
			{
				if (c == 'M') { sizeHeap *= 1000000; break; }
				if (c == 'K') { sizeHeap *= 1000; break; }
				if (c == 'G') { sizeHeap *= 1000000000; break; }
			}
			if (sizeHeap > 0x7fff0000) sizeHeap = 0x7fff0000;
#if __CUDACC__
#else
			DataEx::Config(CONFIG_HEAP, malloc((int)sizeHeap), (int)sizeHeap, 64);
#endif
		}
#else
		else if (!strcmp(z, "-heap")) { }
#endif
#ifdef ENABLE_VFSTRACE
		else if (!strcmp(z, "-vfstrace"))
		{
			extern int vfstrace_register(const char *zTraceName, const char *zOldVfsName, int (*xOut)(const char*,void*), void *pOutArg, int makeDefault);
			vfstrace_register("trace", 0, (int(*)(const char*,void*))fputs, stderr, 1);
		}
#endif
#ifdef ENABLE_MULTIPLEX
		else if (!strcmp(z, "-multiplex"))
		{
			extern int sqlite3_multiple_initialize(const char*,int);
			sqlite3_multiplex_initialize(0, 1);
		}
#endif
		else if (!strcmp(z, "-vfs"))
		{
#if __CUDACC__
			char *vfsName = CmdlineOptionValue(argc, argv, ++i);
			int vfsNameLength = (int)strlen(vfsName) + 1;
			char *d_vfsName;
			cudaMalloc((void**)&d_vfsName, vfsNameLength);
			cudaMemcpy(d_vfsName, vfsName, vfsNameLength, cudaMemcpyHostToDevice);
			d_main_Vfs<<<1,1>>>(d_vfsName); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); 
			cudaFree(d_vfsName);
			H_RETURN(); if (h_return) { fprintf(stderr, "no such VFS: \"%s\"\n", argv[i]); exit(1); }
#else
			DataEx::Initialize();
			VSystem *vfs = VSystem::FindVfs(CmdlineOptionValue(argc, argv, ++i));
			if (vfs)
				VSystem::RegisterVfs(vfs, true);
			else
			{
				fprintf(stderr, "no such VFS: \"%s\"\n", argv[i]);
				exit(1);
			}
#endif
		}
	}
	if (!_data.DbFilename)
	{
#ifndef OMIT_MEMORYDB
		H_DIRTY(&_data);
		_data.DbFilename = ":memory:";
#else
		fprintf(stderr,"%s: Error: no database filename specified\n", Argv0);
		return 1;
#endif
	}
	_data.Out = stdout;
	H_DIRTY(&_data);

	// Go ahead and open the database file if it already exists.  If the file does not exist, delay opening it.  This prevents empty database
	// files from being created if a user mistypes the database name argument to the sqlite command-line tool.
#ifndef __CUDACC__
	if (!access(_data.DbFilename, 0))
		_OpenCtx(&_data);
#endif

	// Process the initialization file if there is one.  If no -init option is given on the command line, look for a file named ~/.sqliterc and
	// try to process it.
	rc = ProcessSqliteRC(&_data, initFile);
	if (rc > 0)
		return rc;

	// Make a second pass through the command-line argument and set options.  This second pass is delayed until after the initialization
	// file is processed so that the command-line arguments will override
	// settings in the initialization file.
	for (i = 1; i < argc; i++)
	{
		char *z = argv[i];
		if (z[0] != '-' ) continue;
		if (z[1] == '-' ) z++;
		if (!strcmp(z, "-init")) i++;
		else if (!strcmp(z, "-html")) _data.Mode = MODE_Html;
		else if (!strcmp(z, "-list")) _data.Mode = MODE_List;
		else if (!strcmp(z, "-line")) _data.Mode = MODE_Line;
		else if (!strcmp(z, "-column")) _data.Mode = MODE_Column;
		else if (!strcmp(z, "-csv")) { _data.Mode = MODE_Csv; memcpy(_data.Separator, ",", 2); }
		else if (!strcmp(z, "-separator")) _snprintf(_data.Separator, sizeof(_data.Separator), "%s", CmdlineOptionValue(argc,argv,++i));
		else if (!strcmp(z, "-nullvalue")) _snprintf(_data.NullValue, sizeof(_data.NullValue), "%s", CmdlineOptionValue(argc,argv,++i));
		else if (!strcmp(z, "-header")) _data.ShowHeader = 1;
		else if (!strcmp(z, "-noheader")) _data.ShowHeader = 0;
		else if (!strcmp(z, "-echo")) _data.EchoOn = 1;
		else if (!strcmp(z, "-stats")) _data.StatsOn = 1;
		else if (!strcmp(z, "-bail")) _bailOnError = 1;
		else if (!strcmp(z, "-version")) { printf("%s %s\n", CORE_VERSION, CORE_SOURCE_ID); return 0; }
		else if (!strcmp(z, "-interactive")) _stdinIsInteractive = 1;
		else if (!strcmp(z, "-batch")) _stdinIsInteractive = 0;
		else if (!strcmp(z, "-heap")) i++;
		else if (!strcmp(z, "-vfs")) i++;
#ifdef ENABLE_VFSTRACE
		else if (!strcmp(z, "-vfstrace")) i++;
#endif
#ifdef ENABLE_MULTIPLEX
		else if (!strcmp(z, "-multiplex")) i++;
#endif
		else if (!strcmp(z, "-help")) Usage(1);
		else if (!strcmp(z, "-cmd"))
		{
			if (i == argc-1) break;
			z = CmdlineOptionValue(argc,argv,++i);
			if (z[0] == '.')
			{
				rc = DoMetaCommand(z, &_data);
				if (rc && _bailOnError) return rc;
			}
			else
			{
				_OpenCtx(&_data);
#if __CUDACC__
				int sqlLength = (int)strlen(z) + 1;
				char *d_sql;
				cudaMalloc((void**)&d_sql, sqlLength);
				cudaMemcpy(d_sql, z, sqlLength, cudaMemcpyHostToDevice);
				D_DATA(&_data); d_main_ShellExec<<<1,1>>>(_data.D_, d_sql); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(&_data);
				cudaFree(d_sql);
				H_RETURN(); if (_bailOnError && h_return) return h_return;
#else
				rc = ShellExec(_data.Ctx, z, ShellCallback, &_data, &errMsg);
				if (errMsg)
				{
					fprintf(stderr, "Error: %s\n", errMsg);
					if (_bailOnError) return (rc ? rc : 1);
				}
				else if (rc)
				{
					fprintf(stderr, "Error: unable to process SQL \"%s\"\n", z);
					if (_bailOnError) return rc;
				}
#endif
			}
		}
		else
		{
			fprintf(stderr, "%s: Error: unknown option: %s\n", Argv0, z);
			fprintf(stderr, "Use -help for a list of options.\n");
			return 1;
		}
	}

	if (firstCmd)
	{
		// Run just the command that follows the database name
		if (firstCmd[0] == '.')
			rc = DoMetaCommand(firstCmd, &_data);
		else
		{
			_OpenCtx(&_data);
#if __CUDACC__
			int sqlLength = (int)strlen(firstCmd) + 1;
			char *d_sql;
			cudaMalloc((void**)&d_sql, sqlLength);
			cudaMemcpy(d_sql, firstCmd, sqlLength, cudaMemcpyHostToDevice);
			D_DATA(&_data); d_main_ShellExec<<<1,1>>>(_data.D_, d_sql); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATA(&_data);
			cudaFree(d_sql);
			H_RETURN(); if (h_return) return h_return;
#else
			rc = ShellExec(_data.Ctx, firstCmd, ShellCallback, &_data, &errMsg);
			if (errMsg)
			{
				fprintf(stderr, "Error: %s\n", errMsg);
				return (rc ? rc : 1);
			}
			else if (rc)
			{
				fprintf(stderr, "Error: unable to process SQL \"%s\"\n", firstCmd);
				return rc;
			}
#endif
		}
	}
	else
	{
		// Run commands received from standard input
		if (_stdinIsInteractive)
		{
			printf(
				"SQLite version %s %.19s\n" /*extra-version-info*/
				"Enter \".help\" for instructions\n"
				"Enter SQL statements terminated with a \";\"\n",
				CORE_VERSION, CORE_SOURCE_ID);
			char *history = 0;
			char *home = FindHomeDir();
			if (home)
			{
				int historyLength = (int)strlen(home) + 20;
				if ((history = (char *)malloc(historyLength))!=0 ){
					_snprintf(history, historyLength, "%s/.sqlite_history", home);
				}
			}
#if defined(HAVE_READLINE) && HAVE_READLINE == 1
			if (history) ReadHistory(history);
#endif
			rc = ProcessInput(&_data, 0);
			if (history)
			{
				stifle_history(100);
				write_history(history);
				free(history);
			}
		}
		else
			rc = ProcessInput(&_data, stdin);
	}
	SetTableName(&_data, 0);
	MainShutdown(); //: called by atexit();
	return rc;
}

#pragma endregion
