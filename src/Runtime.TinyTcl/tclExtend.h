/* 
* tclExtend.h
*
*    External declarations for the extended Tcl library.
*-----------------------------------------------------------------------------
* Copyright 1992 Karl Lehenbauer and Mark Diekhans.
*
* Permission to use, copy, modify, and distribute this software and its
* documentation for any purpose and without fee is hereby granted, provided
* that the above copyright notice appear in all copies.  Karl Lehenbauer and
* Mark Diekhans make no representations about the suitability of this
* software for any purpose.  It is provided "as is" without express or
* implied warranty.
*-----------------------------------------------------------------------------
* $Id: tclExtend.h,v 1.1.1.1 2001/04/29 20:34:45 karll Exp $
*-----------------------------------------------------------------------------
*/

#ifndef TCLEXTEND_H
#define TCLEXTEND_H

#include <stdio.h>
#include "tcl.h"

// Version suffix for extended Tcl, this is appended to the standard Tcl version to form the actual extended Tcl version.
#define TCL_EXTD_VERSION_SUFFIX "c"  // 6.1c, 6.2c or 6.3c
typedef void *void_pt;

// Flags for Tcl shell startup.
#define TCLSH_QUICK_STARTUP       1   // Don't process default & init files.
#define TCLSH_ABORT_STARTUP_ERR   2   // Abort on an error.
#define TCLSH_NO_INIT_FILE        4   // Don't process the init file.
#define TCLSH_NO_STACK_DUMP       8   // Don't dump the proc stack on error.

// These globals are used by the infox command.  Should be set by main.
extern char *tclxVersion;        // Extended Tcl version number.
extern int   tclxPatchlevel;     // Extended Tcl patch level.

extern char *tclAppName;         // Application name
extern char *tclAppLongname;     // Long, natural language application name
extern char *tclAppVersion;      // Version number of the application

// If set to be a pointer to the procedure Tcl_RecordAndEval, will link in history.  Should be set by main.
extern int (*tclShellCmdEvalProc) ();

// If non-zero, a signal was received.  Normally signals are handled in Tcl_Eval, but if an application does not return to eval for some period
// of time, then this should be checked and Tcl_CheckForSignal called if this is set.
extern int tclReceivedSignal;

// Exported Extended Tcl functions.
EXTERN int Tcl_CheckForSignal(Tcl_Interp *interp, int cmdResultCode);
EXTERN void Tcl_CommandLoop(Tcl_Interp *interp, FILE *inFile, FILE *outFile, int (*evalProc)(), unsigned options);
EXTERN Tcl_Interp *Tcl_CreateExtendedInterp();
EXTERN char *Tcl_DeleteKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *keyedList);
EXTERN char *Tcl_DownShift(char *targetStr, const char *sourceStr);
EXTERN void Tcl_ErrorAbort(Tcl_Interp *interp, int noStackDump, int exitCode);
EXTERN char *Tcl_UpShift(char *targetStr, const char *sourceStr);
EXTERN int Tcl_GetKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *keyedList, char **fieldValuePtr);
int Tcl_GetKeyedListKeys(Tcl_Interp *interp, const char *subFieldName, const char *keyedList, int *keyesArgcPtr, char ***keyesArgvPtr);
EXTERN int Tcl_GetLong(Tcl_Interp *interp, const char *string, long *longPtr);
EXTERN int Tcl_GetUnsigned(Tcl_Interp *interp, const char *string, unsigned *unsignedPtr);
EXTERN char *Tcl_SetKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *fieldvalue, const char *keyedList);
EXTERN int Tcl_StrToLong(const char *string, int base, long *longPtr);
EXTERN int Tcl_StrToInt(const char *string, int base, int *intPtr);
EXTERN int Tcl_StrToUnsigned(const char *string, int base, unsigned *unsignedPtr);
EXTERN int Tcl_StrToDouble(const char *string, double *doublePtr);
EXTERN void_pt Tcl_HandleAlloc(void_pt headerPtr, char *handlePtr);
EXTERN void Tcl_HandleFree(void_pt headerPtr, void_pt entryPtr);
EXTERN void_pt Tcl_HandleTblInit(const char *handleBase, int entrySize, int initEntries);
EXTERN void Tcl_HandleTblRelease(void_pt headerPtr);
EXTERN int Tcl_HandleTblUseCount(void_pt headerPtr, int amount);
EXTERN void_pt Tcl_HandleWalk(void_pt headerPtr, int *walkKeyPtr);
EXTERN void Tcl_WalkKeyToHandle(void_pt headerPtr, int walkKey, char *handlePtr);
EXTERN void_pt Tcl_HandleXlate(Tcl_Interp *interp, void_pt headerPtr, const char *handle);
EXTERN int Tcl_MathError(char *functionName, int errorType);
EXTERN void Tcl_Startup(Tcl_Interp *interp, int argc, const char **argv, const char *defaultFile, unsigned options);
EXTERN int Tcl_ShellEnvInit(Tcl_Interp *interp, unsigned options, const char *programName, int argc, const char **argv, int interactive, const char *defaultFile);
EXTERN int Tcl_System(Tcl_Interp *interp, char *command);

#endif
