// tcl.h - This header file describes the externally-visible facilities of the Tcl interpreter.
//
// Copyright 1987-1991 Regents of the University of California Permission to use, copy, modify, and distribute this
// software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright
// notice appear in all copies.  The University of California makes no representations about the suitability of this
// software for any purpose.  It is provided "as is" without express or implied warranty.

#ifndef __TCL_H__
#define __TCL_H__

#define TCL_VERSION "6.7"
#define TCL_MAJOR_VERSION 6
#define TCL_MINOR_VERSION 7
#include <Runtime.h>

#ifndef _CLIENTDATA
#define _CLIENTDATA
typedef int *ClientData;
#endif

// Data structures defined opaquely in this module.  The definitions below just provide dummy types.  A few fields are made visible in
// Tcl_Interp structures, namely those for returning string values.
// Note:  any change to the Tcl_Interp definition below must be mirrored in the "real" definition in tclInt.h.
typedef struct Tcl_Interp {
	char *result;		// Points to result string returned by last command.
	void (*freeProc)(char *blockPtr);
	// Zero means result is statically allocated. If non-zero, gives address of procedure to invoke to free the result.  Must be freed by Tcl_Eval before executing next command.
	int errorLine;		// When TCL_ERROR is returned, this gives the line number within the command where the error occurred (1 means first line).
} Tcl_Interp;

typedef struct Tcl_Obj
{
	char *Value;	// This points to the first byte of the object's string representation. The array must be followed by a null byte (i.e., at offset length) but may also contain
	int RefCount;	// When 0 the object will be freed.
	int Bytes;		// The number of bytes at *value, not including the terminating null.
	char *TypePtr;	// Denotes the object's type. Always corresponds to the type of the object's internal rep. NULL indicates the object has no internal rep (has no type).
	__device__ __forceinline operator char *() { return Value; }
} Tcl_Obj;



typedef int *Tcl_Trace;
typedef int *Tcl_CmdBuf;
typedef void *Tcl_Channel;

// When a TCL command returns, the string pointer interp->result points to a string containing return information from the command.  In addition,
// the command procedure returns an integer value, which is one of the following:
//
// TCL_OK			Command completed normally;  interp->result contains the command's result.
// TCL_ERROR		The command couldn't be completed successfully; interp->result describes what went wrong.
// TCL_RETURN		The command requests that the current procedure return;  interp->result contains the procedure's return value.
// TCL_BREAK		The command requests that the innermost loop be exited;  interp->result is meaningless.
// TCL_CONTINUE		Go on to the next iteration of the current loop; interp->result is meaninless.
// TCL_SIGNAL		A signal was caught. interp->result contains the signal name
#define TCL_OK		0
#define TCL_ERROR	1
#define TCL_RETURN	2
#define TCL_BREAK	3
#define TCL_CONTINUE 4
#define TCL_SIGNAL	5

#define TCL_RESULT_SIZE 199

// Procedure types defined by Tcl:
typedef void (Tcl_CmdDeleteProc)(ClientData clientData);
typedef int (Tcl_CmdProc)(ClientData clientData, Tcl_Interp *interp, int argc, char *argv[]);
typedef void (Tcl_CmdTraceProc)(ClientData clientData, Tcl_Interp *interp, int level, char *command, Tcl_CmdProc *proc, ClientData cmdClientData, int argc, char *argv[]);
typedef void (Tcl_FreeProc)(char *blockPtr);
typedef char *(Tcl_VarTraceProc)(ClientData clientData, Tcl_Interp *interp, char *part1, char *part2, int flags);

// Flag values passed to Tcl_Eval (see the man page for details;  also see tclInt.h for additional flags that are only used internally by Tcl):
#define TCL_BRACKET_TERM 1

// Flag that may be passed to Tcl_ConvertElement to force it not to output braces (careful!  if you change this flag be sure to change the definitions at the front of tclUtil.c).
#define TCL_DONT_USE_BRACES	1

// Flag value passed to Tcl_RecordAndEval to request no evaluation (record only).
#define TCL_NO_EVAL -1

// Specil freeProc values that may be passed to Tcl_SetResult (see the man page for details):
#define TCL_VOLATILE ((Tcl_FreeProc *)-1)
#define TCL_STATIC	((Tcl_FreeProc *)0)
#define TCL_DYNAMIC	((Tcl_FreeProc *)_free)

// Flag values passed to variable-related procedures.
#define TCL_GLOBAL_ONLY		1
#define TCL_APPEND_VALUE	2
#define TCL_LIST_ELEMENT	4
#define TCL_NO_SPACE		8
#define TCL_TRACE_READS		0x10
#define TCL_TRACE_WRITES	0x20
#define TCL_TRACE_UNSETS	0x40
#define TCL_TRACE_DESTROYED	0x80
#define TCL_INTERP_DESTROYED	0x100
#define TCL_LEAVE_ERR_MSG	0x200

// Additional flag passed back to variable watchers.  This flag must not overlap any of the TCL_TRACE_* flags defined above or the TRACE_* flags defined in tclInt.h.
#define TCL_VARIABLE_UNDEFINED	8

// The following declarations either map _allocFast and _freeFast to malloc and free, or they map them to procedures with all sorts of debugging hooks defined in tclCkalloc.c.
#ifdef TCL_MEM_DEBUG
extern __device__ char *Tcl_MemAlloc (unsigned int size, char *file, int line);
extern __device__ int Tcl_MemFree(char *ptr, char *file, int line);
extern __device__ char *Tcl_MemRealloc(char *ptr, unsigned int size, char *file, int line);
extern __device__ int Tcl_DumpActiveMemory(char *fileName);
extern __device__ void Tcl_ValidateAllMemory(char *file, int line);
#define _allocFast(x) Tcl_MemAlloc((x), __FILE__, __LINE__)
#define _freeFast(x) Tcl_MemFree((x), __FILE__, __LINE__)
#define _reallocFast(x,y) Tcl_MemRealloc((x), (y), __FILE__, __LINE__)
#else
#define _allocFast(x) _alloc(x)
#define _freeFast(x) _free(x)
#define _reallocFast(x,y) _realloc(x,y)
#define Tcl_DumpActiveMemory(x)
#define Tcl_ValidateAllMemory(x,y)
#endif
#define Tcl_Alloc(x) _allocFast(x)
#define Tcl_Free(x) _freeFast(x)

// Macro to free up result of interpreter.
#define Tcl_FreeResult(interp) \
	if ((interp)->freeProc != 0) { \
	if ((interp)->freeProc == (Tcl_FreeProc *)_free) { _freeFast((interp)->result); } \
	else { (*(interp)->freeProc)((interp)->result); } \
	(interp)->freeProc = 0; }

// Exported Tcl procedures:
extern __device__ void Tcl_AppendElement(Tcl_Interp *interp, const char *string, bool noSep);
#if __CUDACC__
extern __device__ void _Tcl_AppendResult(Tcl_Interp *interp, _va_list &args);
__device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp) { _va_list args; _va_start(args); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); _Tcl_AppendResult(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void Tcl_AppendResult(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); _Tcl_AppendResult(interp, args); _va_end(args); }
#else
extern __device__ void Tcl_AppendResult(Tcl_Interp *interp, ...);
#endif
extern __device__ char *Tcl_AssembleCmd(Tcl_CmdBuf buffer, char *string);
extern __device__ void Tcl_AddErrorInfo(Tcl_Interp *interp, char *message);
extern __device__ char Tcl_Backslash(const char *src, int *readPtr);
extern __device__ int Tcl_CommandComplete(char *cmd);
extern __device__ char *Tcl_Concat(int argc, char **argv);
extern __device__ int Tcl_ConvertElement(const char *src, char *dst, int flags);
extern __device__ Tcl_CmdBuf Tcl_CreateCmdBuf();
extern __device__ void Tcl_CreateCommand(Tcl_Interp *interp, char *cmdName, Tcl_CmdProc *proc, ClientData clientData, Tcl_CmdDeleteProc *deleteProc);
extern __device__ Tcl_Interp *Tcl_CreateInterp();
extern __device__ int Tcl_CreatePipeline(Tcl_Interp *interp, int argc, char **argv, int **pidArrayPtr, FILE **inPipePtr, FILE **outPipePtr, FILE **errFilePtr);
extern __device__ Tcl_Trace Tcl_CreateTrace(Tcl_Interp *interp, int level, Tcl_CmdTraceProc *proc, ClientData clientData);
extern __device__ void Tcl_DeleteCmdBuf(Tcl_CmdBuf buffer);
extern __device__ int Tcl_DeleteCommand(Tcl_Interp *interp, char *cmdName);
extern __device__ void Tcl_DeleteInterp(Tcl_Interp *interp);
extern __device__ void Tcl_DeleteTrace(Tcl_Interp *interp, Tcl_Trace trace);
extern __device__ void Tcl_DetachPids(int numPids, int *pidPtr);
extern __device__ char *Tcl_ErrnoId();
extern __device__ int Tcl_Eval(Tcl_Interp *interp, char *cmd, int flags, char **termPtr);
extern __device__ int Tcl_EvalFile(Tcl_Interp *interp, char *fileName);
extern __device__ int Tcl_ExprBoolean(Tcl_Interp *interp, char *string, int *ptr);
extern __device__ int Tcl_ExprDouble(Tcl_Interp *interp, char *string, double *ptr);
extern __device__ int Tcl_ExprLong(Tcl_Interp *interp, char *string, long *ptr);
extern __device__ int Tcl_ExprString(Tcl_Interp *interp, char *string);
extern __device__ int Tcl_Fork();
extern __device__ int Tcl_GetBoolean(Tcl_Interp *interp, char *string, bool *boolPtr);
extern __device__ int Tcl_GetDouble(Tcl_Interp *interp, char *string, double *doublePtr);
extern __device__ int Tcl_GetInt(Tcl_Interp *interp, char *string, int *intPtr);
extern __device__ int Tcl_GetWideInt(Tcl_Interp *interp, char *string, int64 *intPtr);
extern __device__ char *Tcl_GetByteArray(Tcl_Interp *interp, char *string, int *arrayLength);
extern __device__ char *Tcl_GetVar(Tcl_Interp *interp, char *varName, int flags);
extern __device__ char *Tcl_GetVar2(Tcl_Interp *interp, char *part1, char *part2, int flags);
extern __device__ int Tcl_GlobalEval(Tcl_Interp *interp, char *command);
extern __device__ void Tcl_InitHistory(Tcl_Interp *interp);
extern __device__ void Tcl_InitMemory(Tcl_Interp *interp);
extern __device__ void Tcl_InitUnix(Tcl_Interp *interp);
extern __device__ char *Tcl_Merge(int argc, char **argv);
extern __device__ char *Tcl_ParseVar(Tcl_Interp *interp, char *string, char **termPtr);
extern __device__ int Tcl_RecordAndEval(Tcl_Interp *interp, char *cmd, int flags);
extern __device__ void Tcl_ResetResult(Tcl_Interp *interp);
#define Tcl_Return Tcl_SetResult
extern __device__ int Tcl_ScanElement(const char *string, int *flagPtr);
extern __device__ void _Tcl_SetErrorCode(Tcl_Interp *interp, _va_list &args);
#if __CUDACC__
__device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp) { _va_list args; _va_start(args); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); _Tcl_SetErrorCode(interp, args); _va_end(args); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); _Tcl_SetErrorCode(interp, args); _va_end(args); }
#else
__device__ __forceinline void Tcl_SetErrorCode(Tcl_Interp *interp, ...) { _va_list args; _va_start(args, interp); _Tcl_SetErrorCode(interp, args); _va_end(args); }
#endif
extern __device__ void Tcl_SetResult(Tcl_Interp *interp, char *string, Tcl_FreeProc *freeProc);
extern __device__ char *Tcl_SetVar(Tcl_Interp *interp, char *varName, char *newValue, int flags);
extern __device__ char *Tcl_SetVar2(Tcl_Interp *interp, char *part1, char *part2, char *newValue, int flags);
extern __device__ char *Tcl_SignalId(int sig);
extern __device__ char *Tcl_SignalMsg(int sig);
extern __device__ int Tcl_SplitList(Tcl_Interp *interp, char *list, int *argcPtr, char ***argvPtr);
extern __device__ int Tcl_StringMatch(char *string, char *pattern);
extern __device__ char *Tcl_TildeSubst(Tcl_Interp *interp, char *name);
extern __device__ int Tcl_TraceVar(Tcl_Interp *interp, char *varName, int flags, Tcl_VarTraceProc *proc, ClientData clientData);
extern __device__ int Tcl_TraceVar2(Tcl_Interp *interp, char *part1, char *part2, int flags, Tcl_VarTraceProc *proc, ClientData clientData);
extern __device__ char *Tcl_OSError(Tcl_Interp *interp);
extern __device__ int Tcl_UnsetVar(Tcl_Interp *interp, char *varName, int flags);
extern __device__ int Tcl_UnsetVar2(Tcl_Interp *interp, char *part1, char *part2, int flags);
extern __device__ void Tcl_UntraceVar(Tcl_Interp *interp, char *varName, int flags, Tcl_VarTraceProc *proc, ClientData clientData);
extern __device__ void Tcl_UntraceVar2(Tcl_Interp *interp, char *part1, char *part2, int flags, Tcl_VarTraceProc *proc, ClientData clientData);
extern __device__ int _Tcl_VarEval(Tcl_Interp *interp, _va_list &args);
#if __CUDACC__
__device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp) { _va_list args; _va_start(args); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
#else
__device__ __forceinline int Tcl_VarEval(Tcl_Interp *interp, ...) { _va_list args; _va_start(args, interp); int r = _Tcl_VarEval(interp, args); _va_end(args); return r; }
#endif
extern __device__ ClientData Tcl_VarTraceInfo(Tcl_Interp *interp, char *varName, int flags, Tcl_VarTraceProc *procPtr, ClientData prevClientData);
extern __device__ ClientData Tcl_VarTraceInfo2(Tcl_Interp *interp, char *part1, char *part2, int flags, Tcl_VarTraceProc *procPtr, ClientData prevClientData);
extern __device__ int Tcl_WaitPids(int numPids, int *pidPtr, int *statusPtr);

// EXTRA
__device__ void Tcl_WrongNumArgs(Tcl_Interp *interp, int objc, char *objv[], const char *message);
inline __device__ char *Tcl_GetString(Tcl_Interp *interp, Tcl_Obj *obj, int *length) { *length = obj->Bytes; return (char *)obj; }
//inline __device__ Tcl_Obj *Tcl_DuplicateObj(Tcl_Obj *obj);
extern __device__ Tcl_Obj *Tcl_NewObj(const char *value, int length, char *typeName = nullptr);
//extern __device__ int Tcl_ObjAppendElement(Tcl_Interp *interp, Tcl_Obj *obj, Tcl_Obj *appendObj);
extern __device__ void Tcl_IncrRefCount(Tcl_Obj *obj);
extern __device__ void Tcl_DecrRefCount(Tcl_Obj *obj);
extern __device__ void Tcl_BackgroundError(Tcl_Interp *interp);

#endif /* __TCL_H__ */


