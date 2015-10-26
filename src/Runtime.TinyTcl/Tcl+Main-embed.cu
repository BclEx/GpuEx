// tclEmbed.c --
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
#if 0 && OS_UNIX

#pragma region Termios

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

struct termios saved_tio;
setup()
{
	struct termios tio;
	if (fcntl(0, F_SETFL, O_NONBLOCK) < 0) {
		perror("stdin fcntl");
	}

	tcgetattr(0, &tio);
	saved_tio = tio;
	tio.c_lflag &= ~(ECHO|ECHONL|ICANON|IEXTEN);
	tcsetattr (0, TCSANOW, &tio);
}

teardown()
{
	tcsetattr(0, TCSANOW, &saved_tio);
}

#pragma endregion

Tcl_Interp *_interp;
char _dumpFile[100];
bool _quitFlag = false;
char _initCmd[] = "echo \"procplace.com embedded tcl 6.7\"; source drongo.tcl";

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

int cmdGetkey(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	char c;
	if (read(0, &c, 1) < 0) {
		return TCL_OK;
	}
	sprintf(interp->result, "%d", c);
	return TCL_OK;
}

int cmdPause(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	usleep(50000);
	return TCL_OK;
}

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
	setup();
	_interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
	Tcl_InitMemory(_interp);
#endif
	Tcl_CreateCommand(_interp, "echo", cmdEcho, (ClientData)"echo", (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(_interp, "getkey", cmdGetkey, (ClientData)"getkey", (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(_interp, "pause", cmdPause, (ClientData)"pause", (Tcl_CmdDeleteProc *)NULL);
#ifdef TCL_MEM_DEBUG
	Tcl_CreateCommand(_interp, "checkmem", cmdCheckmem, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	char *result = Tcl_Eval(_interp, _initCmd, 0, (char **)NULL);
	if (result != TCL_OK) {
		printf("%s\n", interp->result);
		teardown();
		exit(1);
	}
	teardown();
	exit(0);
}

#endif