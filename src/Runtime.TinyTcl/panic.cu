/* 
* panic.c --
*
*	Source code for the "panic" library procedure for Tcl;
*	individual applications will probably override this with
*	an application-specific panic procedure.
*
* Copyright 1988-1991 Regents of the University of California
* Permission to use, copy, modify, and distribute this
* software and its documentation for any purpose and without
* fee is hereby granted, provided that the above copyright
* notice appears in all copies.  The University of California
* makes no representations about the suitability of this
* software for any purpose.  It is provided "as is" without
* express or implied warranty.
*
* $Id: panic.c,v 1.1.1.1 2001/04/29 20:34:05 karll Exp $
*/

#include <stdio.h>
#include <stdlib.h>

/*
*----------------------------------------------------------------------
*
* panic --
*
*	Print an error message and kill the process.
*
* Results:
*	None.
*
* Side effects:
*	The process dies, entering the debugger if possible.
*
*----------------------------------------------------------------------
*/

__device__ void panic(char *format, char *arg1, char *arg2, char *arg3, char *arg4, char *arg5, char *arg6, char *arg7, char *arg8)
{
	fprintf(stderr, format, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
	fflush(stderr);
	abort();
}
