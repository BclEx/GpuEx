// tclUtil.c --
//
//	This file contains utility procedures that are used by many Tcl commands.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "Tcl+Int.h"

#ifndef __isascii
//#define __isascii isascii
#define __isascii(c) (c < 128)
#endif

// The following values are used in the flags returned by Tcl_ScanElement and used by Tcl_ConvertElement.  The value TCL_DONT_USE_BRACES is also
// defined in tcl.h;  make sure its value doesn't overlap with any of the values below.
//
// TCL_DONT_USE_BRACES  1 - 1 means the string mustn't be enclosed in braces (e.g. it contains unmatched braces, or ends in a backslash character, or user just doesn't want braces);  handle all special characters by adding backslashes.
#define USE_BRACES			2 // 1 means the string contains a special character that can be handled simply by enclosing the entire argument in braces.
#define BRACES_UNMATCHED	4 // 1 means that braces aren't properly matched in the argument. 

// Function prototypes for local procedures in this file:
__device__ static void SetupAppendBuffer(Interp *iPtr, int newSpace);

/*
*----------------------------------------------------------------------
*
* TclFindElement --
*	Given a pointer into a Tcl list, locate the first (or next) element in the list.
*
* Results:
*	The return value is normally TCL_OK, which means that the element was successfully located.  If TCL_ERROR is returned
*	it means that list didn't have proper list structure; interp->result contains a more detailed error message.
*
*	If TCL_OK is returned, then *elementPtr will be set to point to the first element of list, and *nextPtr will be set to point
*	to the character just after any white space following the last character that's part of the element.  If this is the last argument
*	in the list, then *nextPtr will point to the NULL character at the end of list.  If sizePtr is non-NULL, *sizePtr is filled in with
*	the number of characters in the element.  If the element is in braces, then *elementPtr will point to the character after the
*	opening brace and *sizePtr will not include either of the braces. If there isn't an element in the list, *sizePtr will be zero, and
*	both *elementPtr and *termPtr will refer to the null character at the end of list.  Note:  this procedure does NOT collapse backslash sequences.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int TclFindElement(Tcl_Interp *interp, register char *list, char **elementPtr, char **nextPtr, int *sizePtr, int *bracePtr)
{
	register char *p;
	int size;

	// Skim off leading white space and check for an opening brace or quote.   Note:  use of "isascii" below and elsewhere in this
	// procedure is a temporary hack (7/27/90) because Mx uses characters with the high-order bit set for some things.  This should probably
	// be changed back eventually, or all of Tcl should call isascii.
	while (__isascii(*list) && _isspace(*list)) {
		list++;
	}
	int openBraces = 0;
	bool inQuotes = false;
	if (*list == '{') {
		openBraces = 1;
		list++;
	} else if (*list == '"') {
		inQuotes = true;
		list++;
	}
	if (bracePtr != 0) {
		*bracePtr = openBraces;
	}
	p = list;

	// Find the end of the element (either a space or a close brace or the end of the string).
	while (true) {
		switch (*p) {
		case '{':
			// Open brace: don't treat specially unless the element is in braces.  In this case, keep a nesting count.
			if (openBraces != 0) {
				openBraces++;
			}
			break;
		case '}':
			// Close brace: if element is in braces, keep nesting count and quit when the last close brace is seen.
			if (openBraces == 1) {
				size = (int)(p - list);
				p++;
				if ((__isascii(*p) && _isspace(*p)) || *p == 0) {
					goto done;
				}
				char *p2;
				for (p2 = p; *p2 != 0 && !_isspace(*p2) && p2 < p+20; p2++) { } // null body
				Tcl_ResetResult(interp);
				_sprintf(interp->result, "list element in braces followed by \"%.*s\" instead of space", (int)(p2-p), p);
				return TCL_ERROR;
			} else if (openBraces != 0) {
				openBraces--;
			}
			break;
		case '\\': {
			// Backslash:  skip over everything up to the end of the backslash sequence.
			int size;
			Tcl_Backslash(p, &size);
			p += size - 1;
			break; }
		case ' ':
		case '\f':
		case '\n':
		case '\r':
		case '\t':
		case '\v':
			// Space: ignore if element is in braces or quotes;  otherwise terminate element.
			if (openBraces == 0 && !inQuotes) {
				size = (int)(p - list);
				goto done;
			}
			break;
		case '"':
			// Double-quote:  if element is in quotes then terminate it.
			if (inQuotes) {
				char *p2;
				size = (int)(p - list);
				p++;
				if ((__isascii(*p) && _isspace(*p)) || *p == 0) {
					goto done;
				}
				for (p2 = p; *p2 != 0 && !_isspace(*p2) && p2 < p+20; p2++) { } // null body
				Tcl_ResetResult(interp);
				_sprintf(interp->result, "list element in quotes followed by \"%.*s\" %s", (int)(p2-p), p, "instead of space");
				return TCL_ERROR;
			}
			break;
		case 0:
			// End of list:  terminate element.
			if (openBraces != 0) {
				Tcl_SetResult(interp, "unmatched open brace in list", TCL_STATIC);
				return TCL_ERROR;
			} else if (inQuotes) {
				Tcl_SetResult(interp, "unmatched open quote in list", TCL_STATIC);
				return TCL_ERROR;
			}
			size = (int)(p - list);
			goto done;
		}
		p++;
	}
done:
	while (__isascii(*p) && _isspace(*p)) {
		p++;
	}
	*elementPtr = list;
	*nextPtr = p;
	if (sizePtr != 0) {
		*sizePtr = size;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* TclCopyAndCollapse --
*	Copy a string and eliminate any backslashes that aren't in braces.
*
* Results:
*	There is no return value.  Count chars. get copied from src to dst.  Along the way, if backslash sequences are found outside
*	braces, the backslashes are eliminated in the copy. After scanning count chars. from source, a null character is placed at the end of dst.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ void TclCopyAndCollapse(int count, register char *src, register char *dst)
{
	for (register char c = *src; count > 0; src++, c = *src, count--) {
		if (c == '\\') {
			int numRead;
			*dst = Tcl_Backslash(src, &numRead);
			if (*dst != 0) {
				dst++;
			}
			src += numRead-1;
			count -= numRead-1;
		} else {
			*dst = c;
			dst++;
		}
	}
	*dst = 0;
}

/*
*----------------------------------------------------------------------
*
* Tcl_SplitList --
*	Splits a list up into its constituent fields.
*
* Results
*	The return value is normally TCL_OK, which means that the list was successfully split up.  If TCL_ERROR is
*	returned, it means that "list" didn't have proper list structure;  interp->result will contain a more detailed error message.
*
*	*argsPtr will be filled in with the address of an array whose elements point to the elements of list, in order.
*	*argcPtr will get filled in with the number of valid elements in the array.  A single block of memory is dynamically allocated
*	to hold both the args array and a copy of the list (with backslashes and braces removed in the standard way).
*	The caller must eventually free this memory by calling free() on *argsPtr.  Note:  *argsPtr and *argcPtr are only modified
*	if the procedure returns normally.
*
* Side effects:
*	Memory is allocated.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_SplitList(Tcl_Interp *interp, char *list, int *argcPtr, const char **argsPtr[])
{
	// Figure out how much space to allocate.  There must be enough space for both the array of pointers and also for a copy of
	// the list.  To estimate the number of pointers needed, count the number of space characters in the list.
	register char *p;
	int size, i;
	for (size = 1, p = list; *p != 0; p++) {
		if (_isspace(*p)) {
			size++;
		}
	}
	size++; // Leave space for final NULL pointer.
	const char **args = (const char **)_allocFast((unsigned)((size*sizeof(char *)) + (p - list) + 1));
	for (i = 0, p = ((char *)args) + size*sizeof(char *); *list != 0; i++) {
		char *element;
		int elSize, brace;
		int result = TclFindElement(interp, list, &element, &list, &elSize, &brace);
		if (result != TCL_OK) {
			_freeFast((char *)args);
			return result;
		}
		if (*element == 0) {
			break;
		}
		if (i >= size) {
			_freeFast((char *)args);
			Tcl_SetResult(interp, "internal error in Tcl_SplitList", TCL_STATIC);
			return TCL_ERROR;
		}
		args[i] = p;
		if (brace) {
			_strncpy(p, element, elSize);
			p += elSize;
			*p = 0;
			p++;
		} else {
			TclCopyAndCollapse(elSize, element, p);
			p += elSize+1;
		}
	}
	args[i] = NULL;
	*argsPtr = args;
	*argcPtr = i;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ScanElement --
*	This procedure is a companion procedure to Tcl_ConvertElement. It scans a string to see what needs to be done to it (e.g.
*	add backslashes or enclosing braces) to make the string into a valid Tcl list element.
*
* Results:
*	The return value is an overestimate of the number of characters that will be needed by Tcl_ConvertElement to produce a valid
*	list element from string.  The word at *flagPtr is filled in with a value needed by Tcl_ConvertElement when doing the actual conversion.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ScanElement(const char *string, int *flagPtr)
{
	/*
	* This procedure and Tcl_ConvertElement together do two things:
	*
	* 1. They produce a proper list, one that will yield back the
	* argument strings when evaluated or when disassembled with
	* Tcl_SplitList.  This is the most important thing.
	* 
	* 2. They try to produce legible output, which means minimizing the
	* use of backslashes (using braces instead).  However, there are
	* some situations where backslashes must be used (e.g. an element
	* like "{abc": the leading brace will have to be backslashed.  For
	* each element, one of three things must be done:
	*
	* (a) Use the element as-is (it doesn't contain anything special
	* characters).  This is the most desirable option.
	*
	* (b) Enclose the element in braces, but leave the contents alone.
	* This happens if the element contains embedded space, or if it
	* contains characters with special interpretation ($, [, ;, or \),
	* or if it starts with a brace or double-quote, or if there are
	* no characters in the element.
	*
	* (c) Don't enclose the element in braces, but add backslashes to
	* prevent special interpretation of special characters.  This is a
	* last resort used when the argument would normally fall under case
	* (b) but contains unmatched braces.  It also occurs if the last
	* character of the argument is a backslash or if the element contains
	* a backslash followed by newline.
	*
	* The procedure figures out how many bytes will be needed to store
	* the result (actually, it overestimates).  It also collects information
	* about the element in the form of a flags word.
	*/
	int nestingLevel = 0;
	int flags = 0;
	if (string == NULL) {
		string = "";
	}
	register const char *p = string;
	if (*p == '{' || *p == '"' || *p == 0) {
		flags |= USE_BRACES;
	}
	for (; *p != 0; p++) {
		switch (*p) {
		case '{':
			nestingLevel++;
			break;
		case '}':
			nestingLevel--;
			if (nestingLevel < 0) {
				flags |= TCL_DONT_USE_BRACES|BRACES_UNMATCHED;
			}
			break;
		case '[':
		case '$':
		case ';':
		case ' ':
		case '\f':
		case '\n':
		case '\r':
		case '\t':
		case '\v':
			flags |= USE_BRACES;
			break;
		case '\\':
			if (p[1] == 0 || p[1] == '\n') {
				flags = TCL_DONT_USE_BRACES;
			} else {
				int size;
				Tcl_Backslash(p, &size);
				p += size-1;
				flags |= USE_BRACES;
			}
			break;
		}
	}
	if (nestingLevel != 0) {
		flags = TCL_DONT_USE_BRACES | BRACES_UNMATCHED;
	}
	*flagPtr = flags;
	// Allow enough space to backslash every character plus leave two spaces for braces.
	return 2*(int)(p-string) + 2;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ConvertElement --
*	This is a companion procedure to Tcl_ScanElement.  Given the information produced by Tcl_ScanElement, this procedure converts
*	a string to a list element equal to that string.
*
* Results:
*	Information is copied to *dst in the form of a list element identical to src (i.e. if Tcl_SplitList is applied to dst it
*	will produce a string identical to src).  The return value is a count of the number of characters copied (not including the terminating NULL character).
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ConvertElement(register const char *src, char *dst, int flags)
{
	register char *p = dst;
	// See the comment block at the beginning of the Tcl_ScanElement code for details of how this works.
	if (src == NULL) {
		src = "";
	}
	if ((flags & USE_BRACES) && !(flags & TCL_DONT_USE_BRACES)) {
		*p = '{';
		p++;
		for (; *src != 0; src++, p++) {
			*p = *src;
		}
		*p = '}';
		p++;
	} else if (*src == 0) {
		// If string is empty but can't use braces, then use special backslash sequence that maps to empty string.
		p[0] = '\\';
		p[1] = '0';
		p += 2;
	} else {
		for (; *src != 0 ; src++) {
			switch (*src) {
			case ']':
			case '[':
			case '$':
			case ';':
			case ' ':
			case '\\':
			case '"':
				*p = '\\';
				p++;
				break;
			case '{':
			case '}':
				if (flags & BRACES_UNMATCHED) {
					*p = '\\';
					p++;
				}
				break;
			case '\f':
				*p = '\\';
				p++;
				*p = 'f';
				p++;
				continue;
			case '\n':
				*p = '\\';
				p++;
				*p = 'n';
				p++;
				continue;
			case '\r':
				*p = '\\';
				p++;
				*p = 'r';
				p++;
				continue;
			case '\t':
				*p = '\\';
				p++;
				*p = 't';
				p++;
				continue;
			case '\v':
				*p = '\\';
				p++;
				*p = 'v';
				p++;
				continue;
			}
			*p = *src;
			p++;
		}
	}
	*p = '\0';
	return (int)(p-dst);
}

/*
*----------------------------------------------------------------------
*
* Tcl_Merge --
*	Given a collection of strings, merge them together into a single string that has proper Tcl list structured (i.e.
*	Tcl_SplitList may be used to retrieve strings equal to the original elements, and Tcl_Eval will parse the string back
*	into its original elements).
*
* Results:
*	The return value is the address of a dynamically-allocated string containing the merged list.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_Merge(int argc, const char *args[])
{
#define LOCAL_SIZE 20
	// Pass 1: estimate space, gather flags.
	int localFlags[LOCAL_SIZE], *flagPtr;
	if (argc <= LOCAL_SIZE) {
		flagPtr = localFlags;
	} else {
		flagPtr = (int *)_allocFast((unsigned)argc*sizeof(int));
	}
	int numChars = 1;
	int i;
	for (i = 0; i < argc; i++) {
		numChars += Tcl_ScanElement(args[i], &flagPtr[i]) + 1;
	}
	// Pass two: copy into the result area.
	char *result = (char *)_allocFast((unsigned)numChars);
	register char *dst = result;
	for (i = 0; i < argc; i++) {
		numChars = Tcl_ConvertElement(args[i], dst, flagPtr[i]);
		dst += numChars;
		*dst = ' ';
		dst++;
	}
	if (dst == result) {
		*dst = 0;
	} else {
		dst[-1] = 0;
	}
	if (flagPtr != localFlags) {
		_freeFast((char *)flagPtr);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_Concat --
*	Concatenate a set of strings into a single large string.
*
* Results:
*	The return value is dynamically-allocated string containing a concatenation of all the strings in args, with spaces between the original args elements.
*
* Side effects:
*	Memory is allocated for the result;  the caller is responsible for freeing the memory.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_Concat(int argc, const char *args[])
{
	int totalSize, i;
	for (totalSize = 1, i = 0; i < argc; i++) {
		totalSize += _strlen(args[i]) + 1;
	}
	char *result = (char *)_allocFast((unsigned)totalSize);
	if (argc == 0) {
		*result = '\0';
		return result;
	}
	register char *p;
	for (p = result, i = 0; i < argc; i++) {
		// Clip white space off the front and back of the string to generate a neater result, and ignore any empty elements.
		char *element = (char *)args[i];
		while (_isspace(*element)) {
			element++;
		}
		int length;
		for (length = _strlen(element); length > 0 && _isspace(element[length-1]); length--) { } // Null loop body.
		if (length == 0) {
			continue;
		}
		_strncpy(p, element, length);
		p += length;
		*p = ' ';
		p++;
	}
	if (p != result) {
		p[-1] = 0;
	} else {
		*p = 0;
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_StringMatch --
*	See if a particular string matches a particular pattern.
*
* Results:
*	The return value is 1 if string matches pattern, and 0 otherwise.  The matching operation permits the following
*	special characters in the pattern: *?\[] (see the manual entry for details on what these mean).
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_StringMatch(register char *string, register char *pattern)
{
	while (true) {
		// See if we're at the end of both the pattern and the string. If so, we succeeded.  If we're at the end of the pattern but not at the end of the string, we failed.
		if (*pattern == 0) {
			if (*string == 0) {
				return 1;
			} else {
				return 0;
			}
		}
		if (*string == 0 && *pattern != '*') {
			return 0;
		}
		// Check for a "*" as the next pattern character.  It matches any substring.  We handle this by calling ourselves
		// recursively for each postfix of string, until either we match or we reach the end of the string.
		if (*pattern == '*') {
			pattern += 1;
			if (*pattern == 0) {
				return 1;
			}
			while (true) {
				if (Tcl_StringMatch(string, pattern)) {
					return 1;
				}
				if (*string == 0) {
					return 0;
				}
				string += 1;
			}
		}
		// Check for a "?" as the next pattern character.  It matches any single character.
		if (*pattern == '?') {
			goto thisCharOK;
		}
		// Check for a "[" as the next pattern character.  It is followed by a list of characters that are acceptable, or by a range (two characters separated by "-").
		if (*pattern == '[') {
			pattern += 1;
			while (true) {
				if (*pattern == ']' || *pattern == 0) {
					return 0;
				}
				if (*pattern == *string) {
					break;
				}
				if (pattern[1] == '-') {
					char c2 = pattern[2];
					if (c2 == 0) {
						return 0;
					}
					if (*pattern <= *string && c2 >= *string) {
						break;
					}
					if (*pattern >= *string && c2 <= *string) {
						break;
					}
					pattern += 2;
				}
				pattern += 1;
			}
			while (*pattern != ']' && *pattern != 0) {
				pattern += 1;
			}
			goto thisCharOK;
		}
		// If the next pattern character is '/', just strip off the '/' so we do exact matching on the character that follows.
		if (*pattern == '\\') {
			pattern += 1;
			if (*pattern == 0) {
				return 0;
			}
		}
		// There's no special character.  Just make sure that the next characters of each string match.
		if (*pattern != *string) {
			return 0;
		}
thisCharOK:
		pattern += 1;
		string += 1;
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_SetResult --
*	Arrange for "string" to be the Tcl return value.
*
* Results:
*	None.
*
* Side effects:
*	interp->result is left pointing either to "string" (if "copy" is 0) or to a copy of string.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_SetResult(Tcl_Interp *interp, char *string, Tcl_FreeProc *freeProc)
{
	register Interp *iPtr = (Interp *)interp;
	Tcl_FreeProc *oldFreeProc = iPtr->freeProc;
	char *oldResult = iPtr->result;
	iPtr->freeProc = freeProc;
	if (!string) {
		iPtr->resultSpace[0] = 0;
		iPtr->result = iPtr->resultSpace;
		iPtr->freeProc = nullptr;
	} else if (freeProc == TCL_VOLATILE) {
		int length = _strlen(string);
		if (length > TCL_RESULT_SIZE) {
			iPtr->result = (char *)_allocFast((unsigned)length+1);
			iPtr->freeProc = (Tcl_FreeProc *)_free;
		} else {
			iPtr->result = iPtr->resultSpace;
			iPtr->freeProc = nullptr;
		}
		_strcpy(iPtr->result, string);
	} else {
		iPtr->result = string;
	}
	// If the old result was dynamically-allocated, free it up.  Do it here, rather than at the beginning, in case the new result value was part of the old result value.
	if (oldFreeProc != 0) {
		if (oldFreeProc == (Tcl_FreeProc *)_free) {
			_freeFast(oldResult);
		} else {
			(*oldFreeProc)(oldResult);
		}
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_AppendResult --
*	Append a variable number of strings onto the result already present for an interpreter.
*
* Results:
*	None.
*
* Side effects:
*	The result in the interpreter given by the first argument is extended by the strings given by the second and following
*	arguments (up to a terminating NULL argument).
*
*----------------------------------------------------------------------
*/
#if __CUDACC__
__device__ void _Tcl_AppendResult(Tcl_Interp *interp, _va_list &argList)
{
#else
__device__ void Tcl_AppendResult(Tcl_Interp *interp, ...)
{
	_va_list argList;
	_va_start(argList, interp);
#endif
	register Interp *iPtr = (Interp *)interp;
	char *string;
	// First, scan through all the arguments to see how much space is needed.
	int newSpace = 0;
	while (true) {
		string = _va_arg(argList, char *);
		if (!string) {
			break;
		}
		newSpace += _strlen(string);
	}

	// If the append buffer isn't already setup and large enough to hold the new data, set it up.
	if (iPtr->result != iPtr->appendResult || (newSpace + iPtr->appendUsed) >= iPtr->appendAvl) {
		SetupAppendBuffer(iPtr, newSpace);
	}

#if __CUDACC__
	_va_restart(argList);
#else
	_va_end(argList);
	_va_start(argList, interp);
#endif

	// Final step:  go through all the argument strings again, copying them into the buffer.
	while (true) {
		string = _va_arg(argList, char *);
		if (string == NULL) {
			break;
		}
		_strcpy(iPtr->appendResult + iPtr->appendUsed, string);
		iPtr->appendUsed += _strlen(string);
	}
#ifndef __CUDACC__
	_va_end(argList);
#endif
}

/*
*----------------------------------------------------------------------
*
* Tcl_AppendElement --
*	Convert a string to a valid Tcl list element and append it to the current result (which is ostensibly a list).
*
* Results:
*	None.
*
* Side effects:
*	The result in the interpreter given by the first argument is extended with a list element converted from string.  If
*	the original result wasn't empty, then a blank is added before the converted list element.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_AppendElement(Tcl_Interp *interp, const char *string, bool noSep)
{
	register Interp *iPtr = (Interp *)interp;
	int flags;
	// See how much space is needed, and grow the append buffer if needed to accommodate the list element.
	int size = Tcl_ScanElement(string, &flags) + 1;
	if (iPtr->result != iPtr->appendResult || (size + iPtr->appendUsed) >= iPtr->appendAvl) {
		SetupAppendBuffer(iPtr, size+iPtr->appendUsed);
	}
	// Convert the string into a list element and copy it to the buffer that's forming.
	char *dst = iPtr->appendResult + iPtr->appendUsed;
	if (!noSep && iPtr->appendUsed != 0) {
		iPtr->appendUsed++;
		*dst = ' ';
		dst++;
	}
	iPtr->appendUsed += Tcl_ConvertElement(string, dst, flags);
}

/*
*----------------------------------------------------------------------
*
* SetupAppendBuffer --
*	This procedure makes sure that there is an append buffer properly initialized for interp, and that it has at least
*	enough room to accommodate newSpace new bytes of information.
*
* Results:
*	None.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ static void SetupAppendBuffer(register Interp *iPtr, int newSpace)
{
	// Make the append buffer larger, if that's necessary, then copy the current result into the append buffer and make the
	// append buffer the official Tcl result.
	if (iPtr->result != iPtr->appendResult) {
		// If an oversized buffer was used recently, then free it up so we go back to a smaller buffer.  This avoids tying up
		// memory forever after a large operation.
		if (iPtr->appendAvl > 500) {
			_freeFast(iPtr->appendResult);
			iPtr->appendResult = NULL;
			iPtr->appendAvl = 0;
		}
		iPtr->appendUsed = _strlen(iPtr->result);
	}
	int totalSpace = newSpace + iPtr->appendUsed;
	if (totalSpace >= iPtr->appendAvl) {
		if (totalSpace < 100) {
			totalSpace = 200;
		} else {
			totalSpace *= 2;
		}
		char *new_ = (char *)_allocFast((unsigned)totalSpace);
		_strcpy(new_, iPtr->result);
		if (iPtr->appendResult != NULL) {
			_freeFast(iPtr->appendResult);
		}
		iPtr->appendResult = new_;
		iPtr->appendAvl = totalSpace;
	} else if (iPtr->result != iPtr->appendResult) {
		_strcpy(iPtr->appendResult, iPtr->result);
	}
	Tcl_FreeResult(iPtr);
	iPtr->result = iPtr->appendResult;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ResetResult --
*	This procedure restores the result area for an interpreter to its default initialized state, freeing up any memory that
*	may have been allocated for the result and clearing any error information for the interpreter.
*
* Results:
*	None.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_ResetResult(Tcl_Interp *interp)
{
	register Interp *iPtr = (Interp *)interp;
	Tcl_FreeResult(iPtr);
	iPtr->result = iPtr->resultSpace;
	iPtr->resultSpace[0] = 0;
	iPtr->flags &= ~(ERR_ALREADY_LOGGED | ERR_IN_PROGRESS | ERROR_CODE_SET);
}

/*
*----------------------------------------------------------------------
*
* Tcl_SetErrorCode --
*	This procedure is called to record machine-readable information about an error that is about to be returned.
*
* Results:
*	None.
*
* Side effects:
*	The errorCode global variable is modified to hold all of the arguments to this procedure, in a list form with each argument
*	becoming one element of the list.  A flag is set internally to remember that errorCode has been set, so the variable doesn't
*	get set automatically when the error is returned.
*
*----------------------------------------------------------------------
*/
__device__ void _Tcl_SetErrorCode(Tcl_Interp *interp, _va_list &argList)
{
	register Interp *iPtr = (Interp *)interp;
	// Scan through the arguments one at a time, appending them to $errorCode as list elements.
	int flags = TCL_GLOBAL_ONLY | TCL_LIST_ELEMENT;
	while (true) {
		char *string = _va_arg(argList, char *);
		if (string == NULL) {
			break;
		}
		Tcl_SetVar2(interp, "errorCode", (char *)NULL, string, flags);
		flags |= TCL_APPEND_VALUE;
	}
	iPtr->flags |= ERROR_CODE_SET;
}

/*
*----------------------------------------------------------------------
*
* TclGetListIndex --
*	Parse a list index, which may be either an integer or the value "end".
*
* Results:
*	The return value is either TCL_OK or TCL_ERROR.  If it is TCL_OK, then the index corresponding to string is left in
*	*indexPtr.  If the return value is TCL_ERROR, then string was bogus;  an error message is returned in interp->result.
*	If a negative index is specified, it is rounded up to 0. The index value may be larger than the size of the list
*	(this happens when "end" is specified).
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int TclGetListIndex(Tcl_Interp *interp, char *string, int *indexPtr)
{
	if (_isdigit(*string) || *string == '-') {
		if (Tcl_GetInt(interp, string, indexPtr) != TCL_OK) {
			return TCL_ERROR;
		}
		if (*indexPtr < 0) {
			*indexPtr = 0;
		}
	} else if (!_strncmp(string, "end", _strlen(string))) {
		*indexPtr = 1<<30;
	} else {
		Tcl_AppendResult(interp, "bad index \"", string, "\": must be integer or \"end\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* TclCompileRegexp --
*	Compile a regular expression into a form suitable for fast matching.  This procedure retains a small cache of pre-compiled
*	regular expressions in the interpreter, in order to avoid compilation costs as much as possible.
*
* Results:
*	The return value is a pointer to the compiled form of string, suitable for passing to regexec.  If an error occurred while
*	compiling the pattern, then NULL is returned and an error message is left in interp->result.
*
* Side effects:
*	The cache of compiled regexp's in interp will be modified to hold information for string, if such information isn't already
*	present in the cache.
*
*----------------------------------------------------------------------
*/
__device__ regex_t *TclCompileRegexp(Tcl_Interp *interp, char *string, int nocase)
{
	register Interp *iPtr = (Interp *)interp;
	int length = _strlen(string);
	regex_t *result;
	int i;
	for (i = 0; i < iPtr->num_regexps; i++) {
		if (length == iPtr->regexps[i].length && nocase == iPtr->regexps[i].nocase && !_strcmp(string, iPtr->regexps[i].pattern)) {
			// Move the matched pattern to the first slot in the cache and shift the other patterns down one position.
			if (i != 0) {
				char *cachedString = iPtr->regexps[i].pattern;
				result = iPtr->regexps[i].regexp;
				for (int j = i-1; j >= 0; j--) {
					iPtr->regexps[j+1].pattern = iPtr->regexps[j].pattern;
					iPtr->regexps[j+1].length = iPtr->regexps[j].length;
					iPtr->regexps[j+1].nocase = iPtr->regexps[j].nocase;
					iPtr->regexps[j+1].regexp = iPtr->regexps[j].regexp;
				}
				iPtr->regexps[0].pattern = cachedString;
				iPtr->regexps[0].length = length;
				iPtr->regexps[0].nocase = nocase;
				iPtr->regexps[0].regexp = result;
			}
			return iPtr->regexps[0].regexp;
		}
	}

	// No match in the cache.  Compile the string and add it to the cache.
	result = (regex_t *)_allocFast(sizeof(*result));

	// Allocate the original string before compiling, since regcomp expects it to exist for the life of the pattern
	char *pattern = (char *)_allocFast((unsigned)(length+1));
	_strcpy(pattern, string);

#ifndef REG_ICASE
#define REG_ICASE 0
#endif
	int ret;
	if ((ret = regcomp(result, pattern, REG_EXTENDED | (nocase ? REG_ICASE : 0))) != 0) {
		char buf[100];
		regerror(ret, result, buf, sizeof(buf));
		Tcl_AppendResult(interp, "couldn't compile regular expression pattern: ", buf, (char *)NULL);
		_freeFast((char *)result);
		_freeFast(pattern);
		return NULL;
	}
	if (iPtr->regexps[iPtr->num_regexps-1].pattern != NULL) {
		_freeFast(iPtr->regexps[iPtr->num_regexps-1].pattern);
		regfree(iPtr->regexps[iPtr->num_regexps-1].regexp);
		_freeFast((char *)iPtr->regexps[iPtr->num_regexps-1].regexp);
		iPtr->regexps[iPtr->num_regexps-1].pattern = 0;
	}
	for (i = iPtr->num_regexps - 2; i >= 0; i--) {
		iPtr->regexps[i+1].pattern = iPtr->regexps[i].pattern;
		iPtr->regexps[i+1].length = iPtr->regexps[i].length;
		iPtr->regexps[i+1].nocase = iPtr->regexps[i].nocase;
		iPtr->regexps[i+1].regexp = iPtr->regexps[i].regexp;
	}
	iPtr->regexps[0].pattern = pattern;
	iPtr->regexps[0].nocase = nocase;
	iPtr->regexps[0].length = length;
	iPtr->regexps[0].regexp = result;
	return result;
}
