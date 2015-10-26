// tclTest.c --
//
//	Test driver for TCL.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include <stdio.h>
#include <stdlib.h>
#include "Tcl.h"

#if 0

Tcl_Interp *_interp;
Tcl_CmdBuf _buffer;
bool _quitFlag = false;
char _initCmd[] = "if [file exists [info library]/init.tcl] {source [info library]/init.tcl}";

#ifdef TCL_MEM_DEBUG
char _dumpFile[100];
int cmdCheckmem(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " fileName\"", (char *)NULL);
		return TCL_ERROR;
	}
	strcpy(_dumpFile, args[1]);
	_quitFlag = true;
	return TCL_OK;
}
#endif

int cmdEcho(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	for (int i = 1; ; i++) {
		if (args[i] == NULL) {
			if (i != argc) {
echoError:
				sprintf(interp->result, "argument list wasn't properly NULL-terminated in \"%s\" command", args[0]);
			}
			break;
		}
		if (i >= argc) {
			goto echoError;
		}
		fputs(args[i], stdout);
		if (i < (argc-1)) {
			printf(" ");
		}
	}
	printf("\n");
	return TCL_OK;
}

int main()
{
	_interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
	Tcl_InitMemory(_interp);
#endif
	Tcl_CreateCommand(_interp, "echo", cmdEcho, (ClientData) "echo", (Tcl_CmdDeleteProc *)NULL);
#ifdef TCL_MEM_DEBUG
	Tcl_CreateCommand(_interp, "checkmem", cmdCheckmem, (ClientData) 0, (Tcl_CmdDeleteProc *)NULL);
#endif
	_buffer = Tcl_CreateCmdBuf();
	int result;
#ifndef TCL_GENERIC_ONLY
	result = Tcl_Eval(_interp, _initCmd, 0, (char **)NULL);
	if (result != TCL_OK) {
		printf("%s\n", _interp->result);
		exit(1);
	}
#endif

	bool gotPartial = false;
	while (true) {
		clearerr(stdin);
		if (!gotPartial) {
			fputs("% ", stdout);
			fflush(stdout);
		}
		char line[1000];
		if (fgets(line, 1000, stdin) == NULL) {
			if (!gotPartial) {
				exit(0);
			}
			line[0] = 0;
		}
		char *cmd = Tcl_AssembleCmd(_buffer, line);
		if (cmd == NULL) {
			gotPartial = true;
			continue;
		}

		gotPartial = false;
		result = Tcl_Eval(_interp, cmd, 0, (char **)NULL);
		if (result == TCL_OK) {
			if (*_interp->result != 0) {
				printf("%s\n", _interp->result);
			}
			if (_quitFlag) {
				Tcl_DeleteInterp(_interp);
				Tcl_DeleteCmdBuf(_buffer);
#ifdef TCL_MEM_DEBUG
				Tcl_DumpActiveMemory(_dumpFile);
#endif
				exit(0);
			}
		} else {
			if (result == TCL_ERROR) {
				printf("Error");
			} else {
				printf("Error %d", result);
			}
			if (*_interp->result != 0) {
				printf(": %s\n", _interp->result);
			} else {
				printf("\n");
			}
		}
	}
}

#endif