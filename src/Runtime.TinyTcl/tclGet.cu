/* 
* tclGet.c --
*
*	This file contains procedures to convert strings into
*	other forms, like integers or floating-point numbers or
*	booleans, doing syntax checking along the way.
*
* Copyright 1990-1991 Regents of the University of California
* Permission to use, copy, modify, and distribute this
* software and its documentation for any purpose and without
* fee is hereby granted, provided that the above copyright
* notice appear in all copies.  The University of California
* makes no representations about the suitability of this
* software for any purpose.  It is provided "as is" without
* express or implied warranty.
*
* $Id: tclGet.c,v 1.1.1.1 2001/04/29 20:34:45 karll Exp $
*/

#include "tclInt.h"

/*
*----------------------------------------------------------------------
*
* Tcl_GetInt --
*
*	Given a string, produce the corresponding integer value.
*
* Results:
*	The return value is normally TCL_OK;  in this case *intPtr
*	will be set to the integer value equivalent to string.  If
*	string is improperly formed then TCL_ERROR is returned and
*	an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/

__device__ int Tcl_GetInt(Tcl_Interp *interp, char *string, int *intPtr)
{
	char *end;
	long i;

	i = strtol(string, &end, 0);
	while ((*end != '\0') && _isspace(*end)) {
		end++;
	}
	if ((end == string) || (*end != 0)) {
		Tcl_AppendResult(interp, "expected integer but got \"", string,
			"\"", (char *) NULL);
		return TCL_ERROR;
	}
	*intPtr = (int) i;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetDouble --
*
*	Given a string, produce the corresponding double-precision
*	floating-point value.
*
* Results:
*	The return value is normally TCL_OK;  in this case *doublePtr
*	will be set to the double-precision value equivalent to string.
*	If string is improperly formed then TCL_ERROR is returned and
*	an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/

__device__ int Tcl_GetDouble(Tcl_Interp *interp, char *string, double *doublePtr)
{
	char *end;
	double d;

	d = strtod(string, &end);
	while ((*end != '\0') && _isspace(*end)) {
		end++;
	}
	if ((end == string) || (*end != 0)) {
		Tcl_AppendResult(interp, "expected floating-point number but got \"",
			string, "\"", (char *) NULL);
		return TCL_ERROR;
	}
	*doublePtr = d;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetBoolean --
*
*	Given a string, return a 0/1 boolean value corresponding
*	to the string.
*
* Results:
*	The return value is normally TCL_OK;  in this case *boolPtr
*	will be set to the 0/1 value equivalent to string.  If
*	string is improperly formed then TCL_ERROR is returned and
*	an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/

__device__ int Tcl_GetBoolean(Tcl_Interp *interp, char *string, int *boolPtr)
{
	char c;
	char lowerCase[10];
	int i, length;

	/*
	* Convert the input string to all lower-case.
	*/

	for (i = 0; i < 9; i++) {
		c = string[i];
		if (c == 0) {
			break;
		}
		if ((c >= 'A') && (c <= 'Z')) {
			c += 'a' - 'A';
		}
		lowerCase[i] = c;
	}
	lowerCase[i] = 0;

	length = _strlen30(lowerCase);
	c = lowerCase[0];
	if ((c == '0') && (lowerCase[1] == '\0')) {
		*boolPtr = 0;
	} else if ((c == '1') && (lowerCase[1] == '\0')) {
		*boolPtr = 1;
	} else if ((c == 'y') && (_strncmp(lowerCase, "yes", length) == 0)) {
		*boolPtr = 1;
	} else if ((c == 'n') && (_strncmp(lowerCase, "no", length) == 0)) {
		*boolPtr = 0;
	} else if ((c == 't') && (_strncmp(lowerCase, "true", length) == 0)) {
		*boolPtr = 1;
	} else if ((c == 'f') && (_strncmp(lowerCase, "false", length) == 0)) {
		*boolPtr = 0;
	} else if ((c == 'o') && (length >= 2)) {
		if (_strncmp(lowerCase, "on", length) == 0) {
			*boolPtr = 1;
		} else if (_strncmp(lowerCase, "off", length) == 0) {
			*boolPtr = 0;
		}
	} else {
		Tcl_AppendResult(interp, "expected boolean value but got \"",
			string, "\"", (char *) NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}
