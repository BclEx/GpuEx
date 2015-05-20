#ifndef __RUNTIMESENTINEL_H__
#define __RUNTIMESENTINEL_H__
#include "Runtime.h"

//////////////////////
// SENTINEL
#pragma region SENTINEL

struct RuntimeSentinelMessage
{
	char OP;
	void (*Prepare)(void*,char*,int);
	__device__ RuntimeSentinelMessage(char op, void (*prepare)(void*,char*,int))
		: OP(op), Prepare(prepare) { }
};
#define RUNTIMESENTINELPREPARE(P) ((void (*)(void*,char*,int))&P)

typedef struct
{
	volatile int Status;
	int Length;
	char Data[1024];
} RuntimeSentinelCommand;

typedef struct
{
	volatile unsigned int AddId;
	volatile unsigned int RunId;
	RuntimeSentinelCommand Commands[1];
} RuntimeSentinelMap;

typedef struct RuntimeSentinelExecutor
{
	RuntimeSentinelExecutor *Next;
	const char *Name;
	bool (*Executor)(void*,RuntimeSentinelMessage*,int);
	void *Tag;
} RuntimeSentinelExecutor;
#define RUNTIMESENTINELEXECUTOR(E) ((bool (*)(void*,RuntimeSentinelMessage*,int))&E)

typedef struct RuntimeSentinelContext
{
	RuntimeSentinelMap *Map;
	RuntimeSentinelExecutor *List;
} RuntimeSentinelContext;

struct RuntimeSentinel
{
public:
	static void Initialize(RuntimeSentinelExecutor *executor = nullptr);
	static void Shutdown();
	static RuntimeSentinelMap *GetMap();
	//
	static RuntimeSentinelExecutor *FindExecutor(const char *name);
	static void RegisterExecutor(RuntimeSentinelExecutor *exec, bool _default = false);
	static void UnregisterExecutor(RuntimeSentinelExecutor *exec);
	//
	__device__ static void Send(void *msg, int msgLength);
};

#pragma endregion

//////////////////////
// MESSAGES
#pragma region MESSAGES
namespace Messages
{
	struct Stdio_fprintf
	{
		__device__ inline static void Prepare(Stdio_fprintf *t, char *data, int length)
		{
			int formatLength = (t->Format ? _strlen(t->Format) + 1 : 0);
			char *format = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(format, t->Format, formatLength);
			t->Format = format;
		}
		RuntimeSentinelMessage Base;
		FILE *File; const char *Format;
		__device__ Stdio_fprintf(FILE *file, const char *format)
			: Base(10, RUNTIMESENTINELPREPARE(Prepare)), File(file), Format(format) { }
	};

	struct Stdio_fopen
	{
		__device__ inline static void Prepare(Stdio_fopen *t, char *data, int length)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(filename, t->Filename, filenameLength);
			t->Filename = filename;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; const char *Mode;
		__device__ Stdio_fopen(const char *filename, const char *mode)
			: Base(2, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), Mode(mode) { }
	};

	struct Stdio_fflush
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fflush(FILE *file)
			: Base(3, nullptr), File(file) { }
	};

	struct Stdio_fclose
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fclose(FILE *file)
			: Base(4, nullptr), File(file) { }
	};

	struct Stdio_fputc
	{
		RuntimeSentinelMessage Base;
		int Ch; FILE *File;
		__device__ Stdio_fputc(int ch, FILE *file)
			: Base(5, nullptr), Ch(ch), File(file) { }
	};

	struct Stdio_fputs
	{
		__device__ inline static void Prepare(Stdio_fputs *t, char *data, int length)
		{
			int strLength = (t->Str ? _strlen(t->Str) + 1 : 0);
			char *str = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(str, t->Str, strLength);
			t->Str = str;
		}
		RuntimeSentinelMessage Base;
		const char *Str; FILE *File;
		__device__ Stdio_fputs(const char *str, FILE *file)
			: Base(6, RUNTIMESENTINELPREPARE(Prepare)), Str(str), File(file) { }
	};
}
#pragma endregion

#endif // __RUNTIMESENTINEL_H__