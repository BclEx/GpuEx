//#define VISUAL
#pragma region PREAMBLE

#if (defined(_WIN32) || defined(WIN32)) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS // This needs to come before any includes for MSVC compiler
#endif

// Enable large-file support for fopen() and friends on unix.
#ifndef DISABLE_LFS
#define _LARGE_FILE       1
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#define _LARGEFILE_SOURCE 1
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "..\System.Data.net\Core+Vdbe\VdbeInt.cu.h"
#include <ctype.h>
#include <stdarg.h>

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
#define read_history(X)
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
#define ToLower(X)  (char)_tolower((unsigned char)X)
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

namespace Core
{
	static int _bailOnError = 0; // If the following flag is set, then command execution stops at an error if we are not interactive.
	static int _stdinIsInteractive = 1; // Threat stdin as an interactive input if the following variable is true.  Otherwise, assume stdin is connected to a file or pipe.
	static Context *_ctx = nullptr; // The following is the open SQLite database.  We make a pointer to this database a static variable so that it can be accessed by the SIGINT handler to interrupt database processing.
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
		va_list args;
		va_start(args, fmt);
		char *z = _vmprintf(fmt, ap);
		va_end(ap);
		_fprintf(iotrace, "%s", z);
		_free(z);
	}
#endif

	static bool isNumber(const char *z, int *realnum)
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

	static const char *_shellStatic = nullptr;
	static void ShellStaticFunc(FuncContext *fctx, int argc, Mem **argv)
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
			_fflush(stdout);
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

	struct PreviousModeData
	{
		int Valid;        // Is there legit data in here?
		int Mode;
		int ShowHeader;
		int ColWidth[100];
	};

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

	static const char *modeDescr[] =
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
		const char *Vfs;			// Name of VFS to use
		Vdbe *Stmt;					// Current statement if any.
		FILE *Log;					// Write log output here
	};

	//static int _strlen(const char *z)
	//{
	//	const char *z2 = z;
	//	while (*z2) { z2++; }
	//	return 0x3fffffff & (int)(z2 - z);
	//}

	static void ShellLog(void *arg, int errCode, const char *msg)
	{
		struct CallbackData *p = (struct CallbackData*)arg;
		if (!p->Log) return;
		_fprintf(p->Log, "(%d) %s\n", errCode, msg);
		_fflush(p->Log);
	}

#pragma endregion

#pragma region Output

	static void OutputHexBlob(FILE *out_, const void *blob, int blobLength)
	{
		char *blob2 = (char *)blob;
		_fprintf(out_, "X'");
		for (int i = 0; i < blobLength; i++) { _fprintf(out_, "%02x", blob2[i]&0xff); }
		_fprintf(out_, "'");
	}

	static void OutputQuotedString(FILE *out_, const char *z)
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

	static void OutputCString(FILE *out_, const char *z)
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
			else if (!isprint(c)) _fprintf(out_, "\\%03o", c&0xff);
			else fputc(c, out_);
		}
		fputc('"', out_);
	}

	static void OutputHtmlString(FILE *out_, const char *z)
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

	static const char _needCsvQuote[] = {
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

	static void OutputCsv(struct CallbackData *p, const char *z, bool sep)
	{
		FILE *out_ = p->Out;
		if (!z) _fprintf(out_, "%s", p->NullValue);
		else
		{
			int i;
			int sepLength = _strlen(p->Separator);
			for (i = 0; z[i]; i++)
			{
				if (_needCsvQuote[((unsigned char*)z)[i]] || (z[i] == p->Separator[0] &&  (sepLength == 1 || _memcmp(z, p->Separator, sepLength) == 0)))
				{
					i = 0;
					break;
				}
			}
			if (i == 0)
			{
				putc('"', out_);
				for (i = 0; z[i]; i++)
				{
					if (z[i] == '"') putc('"', out_);
					putc(z[i], out_);
				}
				putc('"', out_);
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

	static int ShellCallback(void *args, int argsLength, char **argNames, char **colNames, int *insertTypes)
	{
		int i;
		struct CallbackData *p = (struct CallbackData *)args;
		switch (p->Mode)
		{
		case MODE_Line: {
			int w = 5;
			if (argNames == 0) break;
			for (i = 0; i < argsLength; i++)
			{
				int len = _strlen(colNames[i] ? colNames[i] : "");
				if (len > w) w = len;
			}
			if (p->Cnt++ > 0) _fprintf(p->Out, "\n");
			for (i = 0; i < argsLength; i++)
				_fprintf(p->Out, "%*s = %s\n", w, colNames[i], argNames[i] ? argNames[i] : p->NullValue);
			break; }
		case MODE_Explain:
		case MODE_Column: {
			if (p->Cnt++ == 0)
			{
				for (i = 0; i < argsLength; i++)
				{
					int w = (i < _lengthof(p->ColWidth) ? p->ColWidth[i] : 0);
					if (w == 0)
					{
						w = _strlen(colNames[i] ? colNames[i] : "");
						if (w < 10) w = 10;
						int n = _strlen(argNames && argNames[i] ? argNames[i] : p->NullValue);
						if (w < n) w = n;
					}
					if (i < _lengthof(p->ActualWidth))
						p->ActualWidth[i] = w;
					if (p->ShowHeader)
					{
						if (w < 0) _fprintf(p->Out, "%*.*s%s", -w, -w, colNames[i], (i == argsLength-1 ? "\n" : "  "));
						else _fprintf(p->Out, "%-*.*s%s", w, w, colNames[i], (i == argsLength-1 ? "\n": "  "));
					}
				}
				if (p->ShowHeader)
					for (i = 0; i < argsLength; i++)
					{
						int w;
						if (i < _lengthof(p->ActualWidth))
						{
							w = p->ActualWidth[i];
							if (w < 0) w = -w;
						}
						else
							w = 10;
						_fprintf(p->Out, "%-*.*s%s", w, w, "---------------------------------------------------------------------------------------------", (i == argsLength-1 ? "\n" : "  "));
					}
			}
			if (argNames == 0) break;
			for (i = 0; i < argsLength; i++)
			{
				int w = (i < _lengthof(p->ActualWidth) ? p->ActualWidth[i] : 10);
				if (p->Mode == MODE_Explain && argNames[i] && _strlen(argNames[i]) > w)
					w = _strlen(argNames[i]);
				if (w < 0) _fprintf(p->Out, "%*.*s%s", -w, -w, (argNames[i] ? argNames[i] : p->NullValue), (i == argsLength-1 ? "\n" : "  "));
				else _fprintf(p->Out, "%-*.*s%s", w, w, (argNames[i] ? argNames[i] : p->NullValue), (i == argsLength-1 ? "\n" : "  "));
			}
			break; }
		case MODE_Semi:
		case MODE_List: {
			if (p->Cnt++ == 0 && p->ShowHeader)
				for (i = 0; i < argsLength; i++)
					_fprintf(p->Out, "%s%s", colNames[i], (i == argsLength-1 ? "\n" : p->Separator));
			if (argNames == 0) break;
			for (i = 0; i < argsLength; i++)
			{
				char *z = argNames[i];
				if (!z) z = p->NullValue;
				_fprintf(p->Out, "%s", z);
				if (i < argsLength-1) _fprintf(p->Out, "%s", p->Separator);
				else if (p->Mode == MODE_Semi) _fprintf(p->Out, ";\n");
				else _fprintf(p->Out, "\n");
			}
			break; }
		case MODE_Html: {
			if (p->Cnt++ == 0 && p->ShowHeader)
			{
				_fprintf(p->Out, "<TR>");
				for (i = 0; i < argsLength; i++)
				{
					_fprintf(p->Out, "<TH>");
					OutputHtmlString(p->Out, colNames[i]);
					_fprintf(p->Out, "</TH>\n");
				}
				_fprintf(p->Out, "</TR>\n");
			}
			if (argNames == 0) break;
			_fprintf(p->Out, "<TR>");
			for (i = 0; i < argsLength; i++)
			{
				_fprintf(p->Out, "<TD>");
				OutputHtmlString(p->Out, (argNames[i] ? argNames[i] : p->NullValue));
				_fprintf(p->Out, "</TD>\n");
			}
			_fprintf(p->Out, "</TR>\n");
			break; }
		case MODE_Tcl: {
			if (p->Cnt++ == 0 && p->ShowHeader)
			{
				for (i = 0; i < argsLength; i++)
				{
					OutputCString(p->Out, (colNames[i] ? colNames[i] : ""));
					if (i < argsLength-1) _fprintf(p->Out, "%s", p->Separator);
				}
				_fprintf(p->Out, "\n");
			}
			if (argNames == 0) break;
			for (i = 0; i < argsLength; i++)
			{
				OutputCString(p->Out, (argNames[i] ? argNames[i] : p->NullValue));
				if (i < argsLength-1) _fprintf(p->Out, "%s", p->Separator);
			}
			_fprintf(p->Out, "\n");
			break; }
		case MODE_Csv: {
			if (p->Cnt++ == 0 && p->ShowHeader)
			{
				for (i = 0; i < argsLength; i++)
					OutputCsv(p, (colNames[i] ? colNames[i] : ""), i < argsLength-1);
				_fprintf(p->Out, "\n");
			}
			if (argNames == 0) break;
			for (i = 0; i < argsLength; i++)
				OutputCsv(p, argNames[i], i < argsLength-1);
			_fprintf(p->Out, "\n");
			break; }
		case MODE_Insert: {
			p->Cnt++;
			if (argNames == 0) break;
			_fprintf(p->Out, "INSERT INTO %s VALUES(", p->DestTable);
			for (i = 0; i < argsLength; i++)
			{
				char *sep = (i > 0 ? "," : "");
				if ((argNames[i] == 0) || (insertTypes && insertTypes[i] == TYPE_NULL))
					_fprintf(p->Out, "%sNULL", sep);
				else if (insertTypes && insertTypes[i] == TYPE_TEXT)
				{
					if (sep[0]) _fprintf(p->Out, "%s", sep);
					OutputQuotedString(p->Out, argNames[i]);
				}
				else if (insertTypes && (insertTypes[i] == TYPE_INTEGER || insertTypes[i] == TYPE_FLOAT))
					_fprintf(p->Out, "%s%s", sep, argNames[i]);
				else if (insertTypes && insertTypes[i] == TYPE_BLOB && p->Stmt)
				{
					const void *blob = Vdbe::Column_Blob(p->Stmt, i);
					int blobLength = Vdbe::Column_Bytes(p->Stmt, i);
					if (sep[0]) _fprintf(p->Out, "%s", sep);
					OutputHexBlob(p->Out, blob, blobLength);
				}
				else if (isNumber(argNames[i], 0))
					_fprintf(p->Out, "%s%s", sep, argNames[i]);
				else
				{
					if (sep[0]) _fprintf(p->Out, "%s", sep);
					OutputQuotedString(p->Out, argNames[i]);
				}
			}
			_fprintf(p->Out, ");\n");
			break; }
		}
		return 0;
	}

	static int callback(void *args, int colLength, char **colValues, char **colNames)
	{
		return ShellCallback(args, colLength, colValues, colNames, nullptr); // since we don't have type info, call the ShellCallback with a NULL value
	}

#pragma endregion

#pragma region Helpers

	static void SetTableName(struct CallbackData *p, const char *name)
	{
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
			_fprintf(stderr, "Error: out_ of memory\n");
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

	static char *AppendText(char *in, char const *append, char quote)
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
		in = (char *)realloc(in, newLength);
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

	static int RunTableDumpQuery(struct CallbackData *p, const char *selectSql, const char *firstRow)
	{
		int i;
		Vdbe *select;
		RC rc = Prepare::Prepare_(p->Ctx, selectSql, -1, &select, 0);
		if (rc != RC_OK || !select)
		{
			_fprintf(p->Out, "/**** ERROR: (%d) %s *****/\n", rc, Main::ErrMsg(p->Ctx));
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
			_fprintf(p->Out, "/**** ERROR: (%d) %s *****/\n", rc, Main::ErrMsg(p->Ctx));
			p->Errs++;
		}
		return rc;
	}

	static char *save_err_msg(Context *ctx)
	{
		int errMsgLength = 1+_strlen(Main::ErrMsg(ctx));
		char *errMsg = (char *)_alloc(errMsgLength);
		if (errMsg)
			_memcpy(errMsg, Main::ErrMsg(ctx), errMsgLength);
		return errMsg;
	}

	static int display_stats(Context *ctx, struct CallbackData *arg, bool reset)
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
			cur = Vdbe::Status(arg->Stmt, Vdbe::STMTSTATUS_FULLSCAN_STEP, reset);
			_fprintf(arg->Out, "Fullscan Steps:                      %d\n", cur);
			cur = Vdbe::Status(arg->Stmt, Vdbe::STMTSTATUS_SORT, reset);
			_fprintf(arg->Out, "Sort Operations:                     %d\n", cur);
			cur = Vdbe::Status(arg->Stmt, Vdbe::STMTSTATUS_AUTOINDEX, reset);
			_fprintf(arg->Out, "Autoindex Inserts:                   %d\n", cur);
		}
		return 0;
	}

	// Execute a statement or set of statements.  Print any result rows/columns depending on the current mode set via the supplied callback.
	// This is very similar to SQLite's built-in sqlite3_exec() function except it takes a slightly different callback and callback data argument.
	// callback // (not the same as sqlite3_exec)
	static int ShellExec(Context *ctx, const char *sql, int (*callback)(void*,int,char**,char**,int*), struct CallbackData *arg, char **errMsgOut)
	{
		Vdbe *stmt = nullptr; // Statement to execute.
		RC rc = RC_OK;
		int rc2;
		const char *leftover; // Tail of unprocessed SQL

		if (errMsgOut)
			*errMsgOut = nullptr;

		while (sql[0] && (rc == RC_OK))
		{
			rc = Prepare::Preparev2(ctx, sql, -1, &stmt, &leftover);
			if (rc != RC_OK)
			{
				if (errMsgOut)
					*errMsgOut = save_err_msg(db);
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
					const char *stmtSql = Vdbe::Sql_(stmt);
					_fprintf(arg->Out, "%s\n", (stmtSql ? stmtSql : sql));
				}

				// Output TESTCTRL_EXPLAIN text of requested
				if (arg && arg->Mode == MODE_Explain)
				{
					const char *explain = nullptr;
					sqlite3_test_control(TESTCTRL_EXPLAIN_STMT, stmt, &explain);
					if (explain && explain[0])
						_fprintf(arg->Out, "%s", explain);
				}

				// perform the first step.  this will tell us if we have a result set or not and how wide it is.
				RC rc = stmt->Step();
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
					display_stats(ctx, arg, 0);

				// Finalize the statement just executed. If this fails, save a copy of the error message. Otherwise, set sql to point to the next statement to execute.
				rc2 = Vdbe::Finalize(stmt);
				if (rc != RC_NOMEM) rc = rc2;
				if (rc == RC_OK)
				{
					sql = leftover;
					while (IsSpace(sql[0])) sql++;
				}
				else if (errMsgOut)
					*errMsgOut = save_err_msg(db);

				// clear saved stmt handle
				if (arg)
					arg->Stmt = nullptr;
			}
		}
		return rc;
	}

	// This is a different callback routine used for dumping the database. Each row received by this callback consists of a table name,
	// the table type ("index" or "table") and SQL to create the table. This routine should print text sufficient to recreate the table.
	static int dump_callback(void *arg, int argsLength, char **args, char **cols)
	{
		RC rc;
		const char *prepStmt = nullptr;
		struct CallbackData *p = (struct CallbackData *)arg;

		if (argsLength != 3) return 1;
		const char *tableName = args[0];
		const char *typeName = args[1];
		const char *sql = args[2];

		if (!strcmp(tableName, "sqlite_sequence")) prepStmt = "DELETE FROM sqlite_sequence;\n";
		else if (!strcmp(tableName, "sqlite_stat1")) _fprintf(p->Out, "ANALYZE sqlite_master;\n");
		else if (!strncmp(tableName, "sqlite_", 7)) return 0;
		else if (!strncmp(sql, "CREATE VIRTUAL TABLE", 20))
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
			return 0;
		}
		else _fprintf(p->Out, "%s;\n", sql);

		if (!strcmp(typeName, "table"))
		{
			char *tableInfoSql = AppendText(zTableInfo, "PRAGMA table_info(", 0);
			tableInfoSql = AppendText(zTableInfo, tableName, '"');
			tableInfoSql = AppendText(zTableInfo, ");", 0);

			Vdbe *tableInfo = nullptr;
			rc = Prepare::Prepare_(p->Ctx, tableInfoSql, -1, &tableInfo, 0);
			free(tableInfoSql);
			if (rc != RC_OK || !tableInfo)
				return 1;

			char *select = nullptr;
			select = AppendText(select, "SELECT 'INSERT INTO ' || ", 0);
			// Always quote the table name, even if it appears to be pure ascii, in case it is a keyword. Ex:  INSERT INTO "table" ...
			char *tmp = nullptr;
			tmp = AppendText(tmp, tableName, '"');
			if (tmp)
			{
				select = AppendText(select, tmp, '\'');
				free(tmp);
			}
			select = AppendText(select, " || ' VALUES(' || ", 0);
			rc = tableInfo->Step();
			int rows = 0;
			while (rc == RC_ROW)
			{
				const char *text = (const char *)Vdbe::Column_Text(tableInfo, 1);
				select = AppendText(select, "quote(", nullptr);
				select = AppendText(select, text, '"');
				rc = tableInfo->Step();
				select = AppendText(select, (rc == RC_ROW ? "), " : ") "), nullptr);
				rows++;
			}
			rc = Vdbe::Finalize(tableInfo);
			if (rc != RC_OK || rows == 0)
			{
				free(select);
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
			free(select);
		}
		return 0;
	}

	// Run zQuery.  Use dump_callback() as the callback routine so that the contents of the query are output as SQL statements.
	// If we get a SQLITE_CORRUPT error, rerun the query after appending "ORDER BY rowid DESC" to the end.
	static int RunSchemaDumpQuery(struct CallbackData *p, const char *query)
	{
		char *err = nullptr;
		RC rc = Main::Exec(p->Ctx, query, dump_callback, p, &err);
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
			char *q2 = malloc(length+100);
			if (!q2) return rc;
			_snprintf(length+100, q2, "%s ORDER BY rowid DESC", query);
			rc = Main::Exec(p->Ctx, q2, dump_callback, p, &err);
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
		".width NUM1 NUM2 ...   Set column widths for \"column\" mode\n"
		;

	static char _timerHelp[] =
		".timer ON|OFF          Turn the CPU timer measurement on or off\n"
		;


	static int ProcessInput(struct CallbackData *p, FILE *in); // Forward reference

	// Make sure the database is open.  If it is not, then open it.  If the database fails to open, print an error message and exit.
	static void OpenCtx(struct CallbackData *p)
	{
		if (!p->Ctx)
		{
			Main::Initialize();
			Main::Open(p->DbFilename, &p->Ctx);
			ctx = p->Ctx;
			if (ctx && Main::ErrCode(ctx) == RC_OK)
				sqlite3_create_function(ctx, "shellstatic", 0, TEXTENCODE_UTF8, 0, shellstaticFunc, 0, 0);
			if (!ctx || Main::ErrCode(ctx) != RC_OK)
			{
				_fprintf(stderr,"Error: unable to open database \"%s\": %s\n", p->DbFilename, Main::ErrMsg(ctx));
				exit(1);
			}
#ifndef OMIT_LOAD_EXTENSION
			//Main::enable_load_extension(p->db, 1);
#endif
#ifdef ENABLE_REGEXP
			{
				extern int sqlite3_add_regexp_func(sqlite3*);
				sqlite3_add_regexp_func(db);
			}
#endif
#ifdef ENABLE_SPELLFIX
			{
				extern int sqlite3_spellfix1_register(sqlite3*);
				sqlite3_spellfix1_register(db);
			}
#endif
		}
	}

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
		if (i > 0 && arg[i] == 0) return atoi(arg);
		if (!_strcmp(arg, "on") || !strcmp(arg, "yes")) return true;
		if (!_strcmp(arg, "off") || !_strcmp(arg, "no")) return false;
		_fprintf(stderr, "ERROR: Not a boolean value: \"%s\". Assuming \"no\".\n", arg);
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
		if (!_strcmp(file, "stdout")) f = stdout;
		else if (!_strcmp(file, "stderr")) f = stderr;
		else if (!_strcmp(file, "off")) f = 0;
		else
		{
			f = fopen(file, "wb");
			if (!f)
				_fprintf(stderr, "Error: cannot open \"%s\"\n", file);
		}
		return f;
	}

	// A routine for handling output from sqlite3_trace().
	static void SqlTraceCallback(void *arg, const char *z)
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

	static int DoMetaCommand(char *line, struct CallbackData *p)
	{
		int i = 1;
		int argsLength = 0;
		int rc = 0;
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
		int n = _strlen(args[0]);
		int c = args[0][0];
		if (c == 'b' && n >= 3 && !strncmp(args[0], "backup", n))
		{
			const char *destFile = nullptr;
			const char *dbName = nullptr;
			const char *key = nullptr;
			Context *dest;
			sqlite3_backup *backup;
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
						_fprintf(stderr, "unknown option: %s\n", args[j]);
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
					_fprintf(stderr, "too many arguments to .backup\n");
					return 1;
				}
			}
			if (!destFile)
			{
				_fprintf(stderr, "missing FILENAME argument on .backup\n");
				return 1;
			}
			if (!dbName) dbName = "main";
			rc = sqlite3_open(destFile, &dest);
			if (rc != RC_OK)
			{
				_fprintf(stderr, "Error: cannot open \"%s\"\n", destFile);
				Main::Close(dest);
				return 1;
			}
#ifdef HAS_CODEC
			sqlite3_key(dest, key, (int)strlen(key));
#else
			(void)zKey;
#endif
			OpenCtx(p);
			backup = sqlite3_backup_init(pDest, "main", p->Ctx, dbName);
			if (!backup)
			{
				_fprintf(stderr, "Error: %s\n", Main::ErrMsg(dest));
				Main::Close(dest);
				return 1;
			}
			while ((rc = sqlite3_backup_step(backup, 100)) == RC_OK) { }
			sqlite3_backup_finish(backup);
			if (rc == RC_DONE)
				rc = 0;
			else
			{
				_fprintf(stderr, "Error: %s\n", Main::ErrMsg(dest));
				rc = 1;
			}
			Main::Close(dest);
		}
		else if (c == 'b' && n >= 3 && !strncmp(args[0], "bail", n) && argsLength > 1 && argsLength < 3)
		{
			_bailOnError = BooleanValue(args[1]);
		}
		else if (c == 'b' && n >= 3 && !strncmp(args[0], "breakpoint", n))
			// The undocumented ".breakpoint" command causes a call to the no-op routine named TestBreakpoint().
		{
			TestBreakpoint();
		}
		else if (c == 'd' && n > 1 && !strncmp(args[0], "databases", n) && argsLength == 1)
		{
			OpenCtx(p);
			struct CallbackData data;
			memcpy(&data, p, sizeof(data));
			data.ShowHeader = 1;
			data.Mode = MODE_Column;
			data.ColWidth[0] = 3;
			data.ColWidth[1] = 15;
			data.ColWidth[2] = 58;
			data.Cnt = 0;
			char *errMsg = 0;
			Main::Exec(p->Ctx, "PRAGMA database_list; ", callback, &data, &errMsg);
			if (errMsg)
			{
				_fprintf(stderr,"Error: %s\n", errMsg);
				_free(errMsg);
				rc = 1;
			}
		}
		else if (c == 'd' && !strncmp(args[0], "dump", n) && argsLength < 3)
		{
			OpenCtx(p);
			// When playing back a "dump", the content might appear in an order which causes immediate foreign key constraints to be violated.
			// So disable foreign-key constraint enforcement to prevent problems.
			_fprintf(p->Out, "PRAGMA foreign_keys=OFF;\n");
			_fprintf(p->Out, "BEGIN TRANSACTION;\n");
			p->WritableSchema = 0;
			Main::Exec(p->Ctx, "SAVEPOINT dump; PRAGMA writable_schema=ON", 0, 0, 0);
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
					shellStatic = args[i];
					RunSchemaDumpQuery(p,
						"SELECT name, type, sql FROM sqlite_master "
						"WHERE tbl_name LIKE shellstatic() AND type=='table'"
						"  AND sql NOT NULL");
					RunTableDumpQuery(p,
						"SELECT sql FROM sqlite_master "
						"WHERE sql NOT NULL"
						"  AND type IN ('index','trigger','view')"
						"  AND tbl_name LIKE shellstatic()", 0);
					shellStatic = nullptr;
				}
			}
			if (p->WritableSchema)
			{
				_fprintf(p->Out, "PRAGMA writable_schema=OFF;\n");
				p->WritableSchema = 0;
			}
			Main::Exec(p->Ctx, "PRAGMA writable_schema=OFF;", 0, 0, 0);
			Main::Exec(p->Ctx, "RELEASE dump;", 0, 0, 0);
			_fprintf(p->Out, (p->Errs ? "ROLLBACK; -- due to errors\n" : "COMMIT;\n"));
		}
		else if (c == 'e' && !strncmp(args[0], "echo", n) && argsLength > 1 && argsLength < 3)
		{
			p->EchoOn = BooleanValue(args[1]);
		}
		else if (c == 'e' && !strncmp(args[0], "exit", n))
		{
			if (argsLength > 1 && (rc = atoi(args[1])) != 0) exit(rc);
			rc = 2;
		}
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
		else if (c == 'h' && (!strncmp(args[0], "header", n) || !strncmp(args[0], "headers", n)) && argsLength > 1 && argsLength < 3)
		{
			p->ShowHeader = BooleanValue(args[1]);
		}
		else if (c == 'h' && !strncmp(args[0], "help", n))
		{
			_fprintf(stderr, "%s", _help);
			if (HAS_TIMER)
				_fprintf(stderr, "%s", _timerHelp);
		}
		else if (c == 'i' && !strncmp(args[0], "import", n) && argsLength == 3)
		{
			char *tableName = args[2];    // Insert data into this table
			char *file = args[1];     // The file from which to extract data
			Vdbe *stmt = nullptr; // A statement
			int i;                   // Loop counters
			OpenCtx(p);
			int sepLength = _strlen(p->Separator); // Number of bytes in p->separator[]
			if (sepLength == 0)
			{
				_fprintf(stderr, "Error: non-null separator required for import\n");
				return 1;
			}
			char *sql = _mprintf("SELECT * FROM %s", tableName); // An SQL statement
			if (!sql)
			{
				_fprintf(stderr, "Error: out_ of memory\n");
				return 1;
			}
			int bytes = _strlen(sql); // Number of bytes in an SQL string
			rc = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
			_free(sql);
			if (rc)
			{
				if (stmt) Vdbe::Finalize(stmt);
				_fprintf(stderr,"Error: %s\n", Main::ErrMsg(ctx));
				return 1;
			}
			int colsLength = Vdbe::Column_Count(stmt); // Number of columns in the table
			Vdbe::Finalize(stmt);
			stmt = nullptr;
			if (colsLength == 0) return 0; // no columns, no error
			sql = malloc(bytes + 20 + colsLength*2);
			if (!sql)
			{
				_fprintf(stderr, "Error: out_ of memory\n");
				return 1;
			}
			_snprintf(bytes+20, sql, "INSERT INTO %s VALUES(?", tableName);
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
				_fprintf(stderr, "Error: %s\n", Main::ErrMsg(ctx));
				if (stmt) Vdbe::Finalize(stmt);
				return 1;
			}
			FILE *in = fopen(file, "rb") // The input file;
				if (!in)
				{
					_fprintf(stderr, "Error: cannot open \"%s\"\n", file);
					Vdbe::Finalize(stmt);
					return 1;
				}
				char **cols; // line[] broken up into columns
				cols = malloc(sizeof(cols[0])*(colsLength+1));
				if (!cols)
				{
					_fprintf(stderr, "Error: out_ of memory\n");
					fclose(in);
					Vdbe::Finalize(stmt);
					return 1;
				}
				Main::Exec(p->Ctx, "BEGIN", 0, 0, 0);
				char *commit = "COMMIT"; // How to commit changes
				int lineno = 0; // Line number of input file
				char *line; // A single line of input from the file
				while ((line = local_getline(0, in, 1)) != 0)
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
						_fprintf(stderr, "Error: %s line %d: expected %d columns of data but found %d\n", file, lineno, colsLength, i+1);
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
						_fprintf(stderr,"Error: %s\n", Main::ErrMsg(ctx));
						commit = "ROLLBACK";
						rc = 1;
						break; // from while
					}
				} // end while
				free(cols);
				fclose(in);
				Vdbe::Finalize(stmt);
				Main::Exec(p->Ctx, commit, 0, 0, 0);
		}
		else if (c == 'i' && !strncmp(args[0], "indices", n) && argsLength < 3)
		{
			struct CallbackData data;
			char *errMsg = nullptr;
			OpenCtx(p);
			memcpy(&data, p, sizeof(data));
			data.ShowHeader = 0;
			data.Mode = MODE_List;
			if (argsLength == 1)
				rc = Main::Exec(p->Ctx,
				"SELECT name FROM sqlite_master "
				"WHERE type='index' AND name NOT LIKE 'sqlite_%' "
				"UNION ALL "
				"SELECT name FROM sqlite_temp_master "
				"WHERE type='index' "
				"ORDER BY 1",
				callback, &data, &errMsg);
			else
			{
				shellStatic = args[1];
				rc = Main::Exec(p->Ctx,
					"SELECT name FROM sqlite_master "
					"WHERE type='index' AND tbl_name LIKE shellstatic() "
					"UNION ALL "
					"SELECT name FROM sqlite_temp_master "
					"WHERE type='index' AND tbl_name LIKE shellstatic() "
					"ORDER BY 1",
					callback, &data, &errMsg);
				shellStatic = nullptr;
			}
			if (errMsg)
			{
				_fprintf(stderr,"Error: %s\n", errMsg);
				_free(errMsg);
				rc = 1;
			}
			else if( rc != RC_OK)
			{
				_fprintf(stderr,"Error: querying sqlite_master and sqlite_temp_master\n");
				rc = 1;
			}
		}
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
					_fprintf(stderr, "Error: cannot open \"%s\"\n", args[1]);
					sqlite3IoTrace = 0;
					rc = 1;
				}
				else
					sqlite3IoTrace = iotracePrintf;
			}
		}
#endif
#ifndef OMIT_LOAD_EXTENSION
		else if (c == 'l' && !strncmp(args[0], "load", n) && argsLength >= 2)
		{
			const char *file = args[1];
			const char *proc = (argsLength >= 3 ? args[2] : 0);
			OpenCtx(p);
			char *errMsg = 0;
			rc = sqlite3_load_extension(p->Ctx, file, proc, &errMsg);
			if (rc != RC_OK)
			{
				_fprintf(stderr, "Error: %s\n", errMsg);
				_free(errMsg);
				rc = 1;
			}
		}
#endif
		else if (c == 'l' && !strncmp(args[0], "log", n) && argsLength >= 2)
		{
			const char *file = args[1];
			OutputFileClose(p->Log);
			p->Log = OutputFileOpen(file);
		}
		else if (c == 'm' && !strncmp(args[0], "mode", n) && argsLength == 2)
		{
			int n2 = _strlen(args[1]);
			if ((n2 == 4 && !strncmp(args[1],"line",n2)) || (n2 == 5 && !strncmp(args[1],"lines",n2))) p->Mode = MODE_Line;
			else if ((n2 == 6 && !strncmp(args[1],"column",n2)) || (n2 == 7 && !strncmp(args[1],"columns",n2))) p->Mode = MODE_Column;
			else if (n2 == 4 && !strncmp(args[1],"list",n2)) p->Mode = MODE_List;
			else if (n2 == 4 && !strncmp(args[1],"html",n2)) p->Mode = MODE_Html;
			else if (n2 == 3 && !strncmp(args[1],"tcl",n2)) { p->Mode = MODE_Tcl; _snprintf(sizeof(p->Separator), p->Separator, " "); }
			else if (n2 == 3 && !strncmp(args[1],"csv",n2)) { p->Mode = MODE_Csv; _snprintf(sizeof(p->Separator), p->Separator, ","); }
			else if (n2 == 4 && !strncmp(args[1],"tabs",n2)) { p->Mode = MODE_List; _snprintf(sizeof(p->Separator), p->Separator, "\t"); }
			else if (n2 == 6 && !strncmp(args[1],"insert",n2)) { p->Mode = MODE_Insert; SetTableName(p, "table"); }
			else
			{
				_fprintf(stderr,"Error: mode should be one of: column csv html insert line list tabs tcl\n");
				rc = 1;
			}
		}
		else if (c == 'm' && !strncmp(args[0], "mode", n) && argsLength == 3)
		{
			int n2 = _strlen(args[1]);
			if (n2 == 6 && !strncmp(args[1],"insert",n2)) { p->Mode = MODE_Insert; SetTableName(p, args[2]); }
			else
			{
				_fprintf(stderr, "Error: invalid arguments:  \"%s\". Enter \".help\" for help\n", args[2]);
				rc = 1;
			}
		}
		else if (c == 'n' && !strncmp(args[0], "nullvalue", n) && argsLength == 2)
		{
			_snprintf(sizeof(p->NullValue), p->NullValue, "%.*s", (int)_lengthof(p->NullValue)-1, args[1]);
		}
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
					_fprintf(stderr,"Error: cannot open pipe \"%s\"\n", &args[1][1]);
					p->Out = stdout;
					rc = 1;
				}
				else
					_snprintf(sizeof(p->Outfile), p->Outfile, "%s", args[1]);
			}
			else
			{
				p->Out = OutputFileOpen(args[1]);
				if (!p->Out)
				{
					if (strcmp(args[1], "off"))
						_fprintf(stderr, "Error: cannot write to \"%s\"\n", args[1]);
					p->Out = stdout;
					rc = 1;
				}
				else
					_snprintf(sizeof(p->Outfile), p->Outfile, "%s", args[1]);
			}
		}
		else if (c == 'p' && n >= 3 && !strncmp(args[0], "print", n))
		{
			for (int i = 1; i < argsLength; i++)
			{
				if (i > 1) _fprintf(p->Out, " ");
				_fprintf(p->Out, "%s", args[i]);
			}
			_fprintf(p->Out, "\n");
		}
		else if (c == 'p' && !strncmp(args[0], "prompt", n) && (argsLength == 2 || argsLength == 3))
		{
			if (argsLength >= 2)
				strncpy(_mainPrompt, args[1], (int)_lengthof(_mainPrompt)-1);
			if (argsLength >= 3)
				strncpy(_continuePrompt, args[2], (int)_lengthof(_continuePrompt)-1);
		}
		else if (c == 'q' && !strncmp(args[0], "quit", n) && argsLength == 1)
		{
			rc = 2;
		}
		else if (c == 'r' && n >= 3 && !strncmp(args[0], "read", n) && argsLength == 2)
		{
			FILE *alt = fopen(args[1], "rb");
			if (!alt)
			{
				_fprintf(stderr, "Error: cannot open \"%s\"\n", args[1]);
				rc = 1;
			}
			else
			{
				rc = ProcessInput(p, alt);
				fclose(alt);
			}
		}
		else if (c == 'r' && n >= 3 && !strncmp(args[0], "restore", n) && argsLength > 1 && argsLength < 4)
		{
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
			rc = sqlite3_open(srcFile, &src);
			if (rc != RC_OK)
			{
				_fprintf(stderr, "Error: cannot open \"%s\"\n", srcFile);
				Main::Close(src);
				return 1;
			}
			OpenCtx(p);
			sqlite3_backup *backup = sqlite3_backup_init(p->db, dbName, src, "main");
			if (!backup)
			{
				_fprintf(stderr, "Error: %s\n", Main::ErrMsg(p->Ctx));
				Main::Close(src);
				return 1;
			}
			int timeout = 0;
			while ((rc = sqlite3_backup_step(backup, 100)) == RC_OK || rc == RC_BUSY)
			{
				if (rc == RC_BUSY)
				{
					if (timeout++ >= 3) break;
					sqlite3_sleep(100);
				}
			}
			sqlite3_backup_finish(backup);
			if (rc == RC_DONE) rc = 0;
			else if (rc == RC_BUSY || rc == RC_LOCKED) { _fprintf(stderr, "Error: source database is busy\n"); rc = 1; }
			else { _fprintf(stderr, "Error: %s\n", Main::ErrMsg(p->Ctx)); rc = 1; }
			Main::Close(src);
		}
		else if (c == 's' && !strncmp(args[0], "schema", n) && argsLength < 3)
		{
			struct CallbackData data;
			char *errMsg = 0;
			OpenCtx(p);
			memcpy(&data, p, sizeof(data));
			data.ShowHeader = 0;
			data.Mode = MODE_Semi;
			if ( rgsLength > 1)
			{
				int i;
				for (i = 0; args[1][i]; i++) args[1][i] = ToLower(args[1][i]);
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
					callback(&data, 1, new_argv, new_colv);
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
					callback(&data, 1, new_argv, new_colv);
					rc = RC_OK;
				}
				else
				{
					shellStatic = args[1];
					rc = Main:Exec(p->Ctx,
						"SELECT sql FROM "
						"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
						"     FROM sqlite_master UNION ALL"
						"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
						"WHERE lower(tbl_name) LIKE shellstatic()"
						"  AND type!='meta' AND sql NOTNULL "
						"ORDER BY substr(type,2,1), "
						" CASE type WHEN 'view' THEN rowid ELSE name END",
						callback, &data, &errMsg);
					shellStatic = nullptr;
				}
			}
			else
			{
				rc = Main::Exec(p->Ctx,
					"SELECT sql FROM "
					"  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
					"     FROM sqlite_master UNION ALL"
					"   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
					"WHERE type!='meta' AND sql NOTNULL AND name NOT LIKE 'sqlite_%'"
					"ORDER BY substr(type,2,1),"
					" CASE type WHEN 'view' THEN rowid ELSE name END",
					callback, &data, &errMsg);
			}
			if (errMsg) { _fprintf(stderr,"Error: %s\n", errMsg); _free(errMsg); rc = 1; }
			else if (rc != RC_OK) { _fprintf(stderr,"Error: querying schema information\n"); rc = 1; }
			else rc = 0;
		}
		else if (c == 's' && !strncmp(args[0], "separator", n) && argsLength == 2)
		{
			_snprintf(sizeof(p->Separator), p->Separator, "%.*s", (int)sizeof(p->Separator)-1, args[1]);
		}
		else if (c == 's' && !strncmp(args[0], "show", n) && argsLength == 1)
		{
			_fprintf(p->Out,"%9.9s: %s\n","echo", p->EchoOn ? "on" : "off");
			_fprintf(p->Out,"%9.9s: %s\n","explain", p->ExplainPrev.Valid ? "on" :"off");
			_fprintf(p->Out,"%9.9s: %s\n","headers", p->ShowHeader ? "on" : "off");
			_fprintf(p->Out,"%9.9s: %s\n","mode", modeDescr[p->Mode]);
			_fprintf(p->Out,"%9.9s: ", "nullvalue");
			OutputCString(p->Out, p->NullValue);
			_fprintf(p->Out, "\n");
			_fprintf(p->Out,"%9.9s: %s\n","output", _strlen(p->Outfile) ? p->Outfile : "stdout");
			_fprintf(p->Out,"%9.9s: ", "separator");
			OutputCString(p->Out, p->Separator);
			_fprintf(p->Out, "\n");
			_fprintf(p->Out,"%9.9s: %s\n","stats", p->StatsOn ? "on" : "off");
			_fprintf(p->Out,"%9.9s: ","width");
			for (int i = 0; i < (int)_lengthof(p->ColWidth) && p->ColWidth[i] != 0; i++)
				_fprintf(p->Out, "%d ", p->ColWidth[i]);
			_fprintf(p->Out,"\n");
		}
		else if (c == 's' && !strncmp(args[0], "stats", n) && argsLength > 1 && argsLength < 3)
		{
			p->StatsOn = BooleanValue(args[1]);
		}
		else if (c == 't' && n > 1 && !strncmp(args[0], "tables", n) && argsLength < 3)
		{
			OpenCtx(p);
			Vdbe *stmt;
			rc = Prepare::PreparevV(p->Ctx, "PRAGMA database_list", -1, &stmt, 0);
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
					sql = sqlite3_mprintf(
					"%z UNION ALL "
					"SELECT '%q.' || name FROM \"%w\".sqlite_master"
					" WHERE type IN ('table','view')"
					"   AND name NOT LIKE 'sqlite_%%'"
					"   AND name LIKE ?1", sql, dbName, dbName);
			}
			Vdbe::Finalize(stmt);
			sql = _mprintf("%z ORDER BY 1", sql);
			rc = Prepare::PrepareV2(p->Ctx, sql, -1, &stmt, 0);
			_free(sql);
			if (rc) return rc;
			int rows, allocs;
			rows = allocs = 0;
			char **newResults = nullptr;
			if (argsLength > 1)
				Vdbe::Bind_Text(stmt, 1, args[1], -1, DESTRUCTOR_TRANSIENT);
			else
				Vdbe::Bind_Text(stmt, 1, "%", -1, DESTRUCTOR_STATIC);
			while (stmt->Step() == RC_ROW)
			{
				if (rows >= allocs)
				{
					int n = allocs*2 + 10;
					char **newResults = _realloc(results, sizeof(results[0])*n);
					if (!newResults)
					{
						_fprintf(stderr, "Error: out_ of memory\n");
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
					for (int j = i; j < rows; j += pintRows)
					{
						char *sp = (j < printRow ? "" : "  ");
						printf("%s%-*s", sp, maxlen, (results[j] ? rResults[j] : ""));
					}
					printf("\n");
				}
			}
			for (int ii = 0; ii < rows; ii++) _free(results[ii]);
			_free(results);
		}
		else if (c == 't' && n >= 8 && !strncmp(args[0], "testctrl", n) && argsLength >= 2)
		{
			static const struct
			{
				const char *CtrlName;   // Name of a test-control option
				int CtrlCode;           // Integer code for that option
			} _ctrls[] = {
				{ "prng_save",             TESTCTRL_PRNG_SAVE              },
				{ "prng_restore",          TESTCTRL_PRNG_RESTORE           },
				{ "prng_reset",            TESTCTRL_PRNG_RESET             },
				{ "bitvec_test",           TESTCTRL_BITVEC_TEST            },
				{ "fault_install",         TESTCTRL_FAULT_INSTALL          },
				{ "benign_malloc_hooks",   TESTCTRL_BENIGN_MALLOC_HOOKS    },
				{ "pending_byte",          TESTCTRL_PENDING_BYTE           },
				{ "assert",                TESTCTRL_ASSERT                 },
				{ "always",                TESTCTRL_ALWAYS                 },
				{ "reserve",               TESTCTRL_RESERVE                },
				{ "optimizations",         TESTCTRL_OPTIMIZATIONS          },
				{ "iskeyword",             TESTCTRL_ISKEYWORD              },
				{ "scratchmalloc",         TESTCTRL_SCRATCHMALLOC          },
			};
			int testctrl = -1;
			int rc = 0;
			int i, n;
			OpenCtx(p);
			// convert testctrl text option to value. allow any unique prefix of the option name, or a numerical value.
			int n = _strlen(args[1]);
			for (i = 0; i < (int)_lengthof(_ctrls); i++)
			{
				if (!strncmp(args[1], _ctrls[i].CtrlName, n))
				{
					if (testctrl < 0)
						testctrl = _ctrls[i].CtrlCode;
					else
					{
						_fprintf(stderr, "ambiguous option name: \"%s\"\n", args[1]);
						testctrl = -1;
						break;
					}
				}
			}
			if (testctrl < 0) testctrl = atoi(args[1]);
			if ((testctrl < TESTCTRL_FIRST) || (testctrl > TESTCTRL_LAST))
				_fprintf(stderr,"Error: invalid testctrl option: %s\n", args[1]);
			else
			{
				switch (testctrl)
				{
				case TESTCTRL_OPTIMIZATIONS:
				case TESTCTRL_RESERVE:
					// sqlite3_test_control(int, db, int)
					if (argsLength == 3)
					{
						int opt = (int)strtol(args[2], 0, 0);
						rc = sqlite3_test_control(testctrl, p->Ctx, opt);
						printf("%d (0x%08x)\n", rc, rc);
					}
					else
						_fprintf(stderr,"Error: testctrl %s takes a single int option\n", args[1]);
					break;
				case TESTCTRL_PRNG_SAVE:
				case TESTCTRL_PRNG_RESTORE:
				case TESTCTRL_PRNG_RESET:
					// sqlite3_test_control(int)
					if (argsLength == 2)
					{
						rc = sqlite3_test_control(testctrl);
						printf("%d (0x%08x)\n", rc, rc);
					}
					else
						_fprintf(stderr,"Error: testctrl %s takes no options\n", args[1]);
					break;
				case TESTCTRL_PENDING_BYTE:
					// sqlite3_test_control(int, uint)
					if (argsLength == 3)
					{
						unsigned int opt = (unsigned int)atoi(args[2]);
						rc = sqlite3_test_control(testctrl, opt);
						printf("%d (0x%08x)\n", rc, rc);
					}
					else
						_fprintf(stderr,"Error: testctrl %s takes a single unsigned int option\n", args[1]);
					break;
				case TESTCTRL_ASSERT:
				case TESTCTRL_ALWAYS:
					// sqlite3_test_control(int, int)
					if (argsLength == 3)
					{
						int opt = atoi(args[2]);        
						rc = sqlite3_test_control(testctrl, opt);
						printf("%d (0x%08x)\n", rc, rc);
					}
					else
						_fprintf(stderr,"Error: testctrl %s takes a single int option\n", args[1]);
					break;
#ifdef SQLITE_N_KEYWORD
				case TESTCTRL_ISKEYWORD:
					// sqlite3_test_control(int, char *)
					if (argsLength == 3)
					{
						const char *opt = args[2];
						rc = sqlite3_test_control(testctrl, opt);
						printf("%d (0x%08x)\n", rc, rc);
					}
					else
						_fprintf(stderr,"Error: testctrl %s takes a single char * option\n", args[1]);
					break;
#endif
				case TESTCTRL_BITVEC_TEST:         
				case TESTCTRL_FAULT_INSTALL:       
				case TESTCTRL_BENIGN_MALLOC_HOOKS: 
				case TESTCTRL_SCRATCHMALLOC:       
				default:
					_fprintf(stderr,"Error: CLI support for testctrl %s not implemented\n", args[1]);
					break;
				}
			}
		}
		else if (c == 't' && n > 4 && !strncmp(args[0], "timeout", n) && argsLength == 2)
		{
			OpenCtx(p);
			sqlite3_busy_timeout(p->Ctx, atoi(args[1]));
		}
		else if (HAS_TIMER && c == 't' && n >= 5 && !strncmp(args[0], "timer", n) && argsLength == 2)
		{
			enableTimer = BooleanValue(args[1]);
		}
		else if (c == 't' && !strncmp(args[0], "trace", n) && argsLength > 1)
		{
			OpenCtx(p);
			OutputFileClose(p->traceOut);
			p->TraceOut = OutputFileOpen(args[1]);
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
			if (p->TraceOut == 0)
				sqlite3_trace(p->Ctx, 0, 0);
			else
				sqlite3_trace(p->Ctx, SqlTraceCallback, p->TraceOut);
#endif
		}
		else if (c == 'v' && !strncmp(args[0], "version", n))
		{
			printf("SQLite %s %s\n", sqlite3_libversion(), sqlite3_sourceid());
		}
		else if (c == 'v' && !strncmp(args[0], "vfsname", n))
		{
			const char *dbName = (argsLength == 2 ? args[1] : "main");
			char *vfsName = 0;
			if (p->Ctx)
			{
				sqlite3_file_control(p->Ctx, dbName, SQLITE_FCNTL_VFSNAME, &vfsName);
				if (vfsName)
				{
					printf("%s\n", vfsName);
					_free(vfsName);
				}
			}
		}
#if defined(_DEBUG) && defined(ENABLE_WHERETRACE)
		else if (c == 'w' && !strncmp(args[0], "wheretrace", n))
		{
			extern int sqlite3WhereTrace;
			sqlite3WhereTrace = atoi(args[1]);
		}
#endif
		else if (c == 'w' && !strncmp(args[0], "width", n) && argsLength > 1)
		{
			assert(argsLength <= _lengthof(args));
			for (int j = 1; j < argsLength && j < _lengthof(p->colWidth); j++)
				p->ColWidth[j-1] = atoi(args[j]);
		}
		else
		{
			_fprintf(stderr, "Error: unknown command or invalid arguments:  \"%s\". Enter \".help\" for help\n", args[0]);
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
	static int _is_complete(char *sql, int sqlLength)
	{
		if (!sql) return 1;
		sql[sqlLength] = ';';
		sql[sqlLength+1] = 0;
		RC rc = Complete::Complete_(sql);
		sql[sqlLength] = 0;
		return rc;
	}

	// Read input from *in and process it.  If *in==0 then input is interactive - the user is typing it it.  Otherwise, input
	// is coming from a file or device.  A prompt is issued and history is saved only if input is interactive.  An interrupt signal will
	// cause this routine to exit immediately, unless input is interactive.
	//
	// Return the number of errors.
	static bool ProcessInput(struct CallbackData *p, FILE *in)
	{
		char *line = 0;
		char *sql = 0;
		int sqlLength = 0;
		int sqlLengthPrior = 0;
		char *errMsg;
		int rc;
		int errCnt = 0;
		int lineno = 0;
		int startline = 0;
		while (errCnt == 0 || !_bailOnError || (in == 0 && _stdinIsInteractive))
		{
			_fflush(p->Out);
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
					sqlLength = _strlen(line);
					sql = malloc(sqlLength+3);
					if (!sql)
					{
						_fprintf(stderr, "Error: out_ of memory\n");
						exit(1);
					}
					memcpy(sql, line, sqlLength+1);
					startline = lineno;
				}
			}
			else
			{
				int lineLength = _strlen(line);
				sql = realloc(sql, sqlLength + lineLength + 4);
				if (!sql)
				{
					_fprintf(stderr,"Error: out_ of memory\n");
					exit(1);
				}
				sql[sqlLength++] = '\n';
				memcpy(&sql[sqlLength], line, lineLength+1);
				sqlLength += lineLength;
			}
			if (sql && _contains_semicolon(&sql[sqlLengthPrior], sqlLength-sqlLengthPrior) && Complete::Complete_(sql))
			{
				p->Cnt = 0;
				OpenCtx(p);
				BEGIN_TIMER;
				rc = ShellExec(p->Ctx, sql, ShellCallback, p, &errMsg);
				END_TIMER;
				if (rc || errMsg)
				{
					char prefix[100];
					if (in != 0 || !_stdinIsInteractive)
						_snprintf(sizeof(prefix), prefix, "Error: near line %d:", startline);
					else
						_snprintf(sizeof(prefix), prefix, "Error:");
					if (errMsg)
					{
						_fprintf(stderr, "%s %s\n", prefix, errMsg);
						_free(errMsg);
						errMsg = nullptr;
					}
					else
						_fprintf(stderr, "%s %s\n", prefix, Main::ErrMsg(p->Ctx));
					errCnt++;
				}
				free(sql);
				sql = nullptr;
				sqlLength = 0;
			}
		}
		if (sql)
		{
			if (!_all_whitespace(sql))
				_fprintf(stderr, "Error: incomplete SQL: %s\n", sql);
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
		static char *home_dir = NULL;
		if (home_dir) return home_dir;
#if !defined(_WIN32) && !defined(WIN32) && !defined(_WIN32_WCE) && !defined(__RTP__) && !defined(_WRS_KERNEL)
		{
			struct passwd *pwent;
			uid_t uid = getuid();
			if ((pwent = getpwuid(uid)) != NULL)
				home_dir = pwent->pw_dir;
		}
#endif
#if defined(_WIN32_WCE)
		home_dir = "/"; // Windows CE (arm-wince-mingw32ce-gcc) does not provide getenv()
#else
#if defined(_WIN32) || defined(WIN32)
		if (!home_dir) home_dir = getenv("USERPROFILE");
#endif
		if (!home_dir) home_dir = getenv("HOME");

#if defined(_WIN32) || defined(WIN32)
		if (!home_dir)
		{
			int n;
			char *drive = getenv("HOMEDRIVE");
			char *path = getenv("HOMEPATH");
			if (drive && path)
			{
				int n = _strlen(drive) + _strlen(path) + 1;
				home_dir = malloc(n);
				if (!home_dir) return nullptr;
				_snprintf(n, home_dir, "%s%s", drive, path);
				return home_dir;
			}
			home_dir = "c:\\";
		}
#endif
#endif // !_WIN32_WCE
		if (home_dir)
		{
			int n = _strlen(home_dir) + 1;
			char *z = malloc(n);
			if (z) memcpy(z, home_dir, n);
			home_dir = z;
		}
		return home_dir;
	}

	// Read input from the file given by sqliterc_override.  Or if that parameter is NULL, take input from ~/.sqliterc
	// Returns the number of errors.
	static int ProcessSqliteRC(struct CallbackData *p,  const char *sqliterc_override)
	{
		char *home_dir = nullptr;
		const char *sqliterc = sqliterc_override;
		char *buf = 0;
		FILE *in = nullptr;
		int rc = 0;
		if (sqliterc == nullptr)
		{
			home_dir = FindHomeDir();
			if (!home_dir)
			{
#if !defined(__RTP__) && !defined(_WRS_KERNEL)
				_fprintf(stderr,"%s: Error: cannot locate your home directory\n", Argv0);
#endif
				return 1;
			}
			Main::Initialize();
			buf = _mprintf("%s/.sqliterc", home_dir);
			sqliterc = buf;
		}
		in = fopen(sqliterc, "rb");
		if (in)
		{
			if (_stdinIsInteractive)
				_fprintf(stderr,"-- Loading resources from %s\n", sqliterc);
			rc = ProcessInput(p, in);
			fclose(in);
		}
		_free(buf);
		return rc;
	}

#pragma endregion

#pragma region Main

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
		_fprintf(stderr,
			"Usage: %s [OPTIONS] FILENAME [SQL]\n"  
			"FILENAME is the name of an SQLite database. A new database is created\n"
			"if the file does not previously exist.\n", Argv0);
		if (showDetail)
			_fprintf(stderr, "OPTIONS include:\n%s", _options);
		else
			_fprintf(stderr, "Use the -help option for additional information\n");
		exit(1);
	}

	// Initialize the state information in data
	static void MainInit(struct CallbackData *data)
	{
		memset(data, 0, sizeof(*data));
		data->Mode = MODE_List;
		memcpy(data->Separator,"|", 2);
		data->ShowHeader = 0;
		Main::Config(CONFIG_URI, 1);
		Main::Config(CONFIG_LOG, shellLog, data);
		_snprintf(sizeof(_mainPrompt), _mainPrompt, "sqlite> ");
		_snprintf(sizeof(_continuePrompt), _continuePrompt,"   ...> ");
		Main::Config(CONFIG_SINGLETHREAD);
	}

	// Get the argument to an --option.  Throw an error and die if no argument is available.
	static char *CmdlineOptionValue(int argc, char **argv, int i)
	{
		if (i == argc)
		{
			_fprintf(stderr, "%s: Error: missing argument to %s\n", argv[0], argv[argc-1]);
			exit(1);
		}
		return argv[i];
	}

	int main(int argc, char **argv)
	{
		char *errMsg = 0;
		struct CallbackData data;
		const char *initFile = 0;
		char *firstCmd = 0;
		int i;
		int rc = 0;

		if (strcmp(sqlite3_sourceid(),SQLITE_SOURCE_ID))
		{
			_fprintf(stderr, "SQLite header and source version mismatch\n%s\n%s\n", sqlite3_sourceid(), SQLITE_SOURCE_ID);
			exit(1);
		}
		Argv0 = argv[0];
		MainInit(&data);
		_stdinIsInteractive = isatty(0);

		// Make sure we have a valid signal handler early, before anything else is done.
#ifdef SIGINT
		signal(SIGINT, interrupt_handler);
#endif

		// Do an initial pass through the command-line argument to locate the name of the database file, the name of the initialization file,
		// the size of the alternative malloc heap, and the first command to execute.
		for (i = 1; i < argc; i++)
		{
			char *z;
			z = argv[i];
			if (z[0] != '-')
			{
				if (!data.DbFilename)
				{
					data.DbFilename = z;
					continue;
				}
				if (!firstCmd)
				{
					firstCmd = z;
					continue;
				}
				_fprintf(stderr, "%s: Error: too many options: \"%s\"\n", Argv0, argv[i]);
				_fprintf(stderr, "Use -help for a list of options.\n");
				return 1;
			}
			if (z[1] == '-') z++;
			if (!strcmp(z, "-separator") || !strcmp(z, "-nullvalue") || !strcmp(z, "-cmd")) CmdlineOptionValue(argc, argv, ++i);
			else if (!strcmp(z, "-init")) initFile = CmdlineOptionValue(argc, argv, ++i);
			// Need to check for batch mode here to so we can avoid printing informational messages (like from ProcessSqliteRC) before 
			// we do the actual processing of arguments later in a second pass.
			else if (!strcmp(z, "-batch")) _stdinIsInteractive = 0;
			else if (!strcmp(z, "-heap"))
			{
#if defined(ENABLE_MEMSYS3) || defined(ENABLE_MEMSYS5)
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
				Main::Config(CONFIG_HEAP, malloc((int)sizeHeap), (int)sizeHeap, 64);
			}
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
				VSystem *vfs = VSystem::Find(CmdlineOptionValue(argc, argv, ++i));
				if (vfs)
					sqlite3_vfs_register(vfs, 1);
				else
				{
					_fprintf(stderr, "no such VFS: \"%s\"\n", argv[i]);
					exit(1);
				}
			}
		}
		if (!data.DbFilename)
		{
#ifndef OMIT_MEMORYDB
			data.DbFilename = ":memory:";
#else
			_fprintf(stderr,"%s: Error: no database filename specified\n", Argv0);
			return 1;
#endif
		}
		data.out_ = stdout;

		// Go ahead and open the database file if it already exists.  If the file does not exist, delay opening it.  This prevents empty database
		// files from being created if a user mistypes the database name argument to the sqlite command-line tool.
		if (!access(data.DbFilename, 0))
			OpenCtx(&data);

		// Process the initialization file if there is one.  If no -init option is given on the command line, look for a file named ~/.sqliterc and
		// try to process it.
		rc = ProcessSqliteRC(&data, initFile);
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
			else if (!strcmp(z, "-html")) data.Mode = MODE_Html;
			else if (!strcmp(z, "-list")) data.Mode = MODE_List;
			else if (!strcmp(z, "-line")) data.Mode = MODE_Line;
			else if (!strcmp(z, "-column")) data.Mode = MODE_Column;
			else if (!strcmp(z, "-csv")) { data.Mode = MODE_Csv; memcpy(data.Separator, ",", 2); }
			else if (!strcmp(z, "-separator")) _snprintf(sizeof(data.Separator), data.Separator, "%s", CmdlineOptionValue(argc,argv,++i));
			else if (!strcmp(z, "-nullvalue")) _snprintf(sizeof(data.NullValue), data.NullValue, "%s", CmdlineOptionValue(argc,argv,++i));
			else if (!strcmp(z, "-header")) data.ShowHeader = 1;
			else if (!strcmp(z, "-noheader")) data.ShowHeader = 0;
			else if (!strcmp(z, "-echo")) data.EchoOn = 1;
			else if (!strcmp(z, "-stats")) data.StatsOn = 1;
			else if (!strcmp(z, "-bail")) _bailOnError = 1;
			else if (!strcmp(z, "-version")) { printf("%s %s\n", sqlite3_libversion(), sqlite3_sourceid()); return 0; }
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
			else if (!strcmp(z, "-cmd")=)
			{
				if (i == argc-1) break;
				z = CmdlineOptionValue(argc,argv,++i);
				if (z[0] == '.')
				{
					rc = DoMetaCommand(z, &data);
					if (rc && _bailOnError) return rc;
				}
				else
				{
					OpenCtx(&data);
					rc = ShellExec(data.Ctx, z, ShellCallback, &data, &errMsg);
					if (errMsg)
					{
						_fprintf(stderr,"Error: %s\n", errMsg);
						if (_bailOnError) return (rc ? rc : 1);
					}
					else if (rc)
					{
						_fprintf(stderr, "Error: unable to process SQL \"%s\"\n", z);
						if (_bailOnError) return rc;
					}
				}
			}
			else
			{
				_fprintf(stderr, "%s: Error: unknown option: %s\n", Argv0, z);
				_fprintf(stderr, "Use -help for a list of options.\n");
				return 1;
			}
		}

		if (firstCmd)
		{
			// Run just the command that follows the database name
			if (firstCmd[0] == '.')
				rc = DoMetaCommand(firstCmd, &data);
			else
			{
				OpenCtx(&data);
				rc = ShellExec(data.db, firstCmd, ShellCallback, &data, &errMsg);
				if (errMsg)
				{
					_fprintf(stderr,"Error: %s\n", errMsg);
					return (rc ? rc : 1);
				}
				else if (rc)
				{
					_fprintf(stderr,"Error: unable to process SQL \"%s\"\n", firstCmd);
					return rc;
				}
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
					sqlite3_libversion(), sqlite3_sourceid() );
				char *history = 0;
				char *home = FindHomeDir();
				if (home)
				{
					int historyLength = _strlen(home) + 20;
					if ((history = malloc(historyLength))!=0 ){
						sqlite3_snprintf(historyLength, history,"%s/.sqlite_history", home);
					}
				}
#if defined(HAVE_READLINE) && HAVE_READLINE == 1
				if (history) read_history(history);
#endif
				rc = ProcessInput(&data, 0);
				if (history)
				{
					stifle_history(100);
					write_history(history);
					free(history);
				}
			}
			else
				rc = ProcessInput(&data, stdin);
		}
		SetTableName(&data, 0);
		if (data.Ctx)
			Main::Close(data.Ctx);
		return rc;
	}

#pragma endregion
}
