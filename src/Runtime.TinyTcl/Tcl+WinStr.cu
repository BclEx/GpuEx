#include "Tcl+Int.h"
#if OS_WIN
#include "Tcl+Win.h"

// usage
//     CHAR msgText[256];
//     getLastErrorText(msgText,sizeof(msgText));
static CHAR *getLastErrorText(CHAR *pBuf, ULONG bufSize)
{
	DWORD retSize;
	LPTSTR pTemp = NULL;

	if (bufSize < 16) {
		if (bufSize > 0) {
			pBuf[0]='\0';
		}
		return pBuf;
	}
	retSize = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_ARGUMENT_ARRAY,
		NULL, GetLastError(), LANG_NEUTRAL, (LPTSTR)&pTemp, 0, NULL);
	if (!retSize || pTemp == NULL) {
		pBuf[0]='\0';
	}
	else {
		pTemp[strlen(pTemp)-2]='\0'; //remove cr and newline character
		sprintf(pBuf,"%0.*s (0x%x)",bufSize-16,pTemp,GetLastError());
		LocalFree((HLOCAL)pTemp);
	}
	return pBuf;
}


/*
*----------------------------------------------------------------------
*
* Tcl_ErrnoId --
*	Return a textual identifier for the current errno value.
*
* Results:
*	This procedure returns a machine-readable textual identifier that corresponds to the current errno value (e.g. "EPERM").
*	The identifier is the same as the #define name in errno.h.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
char *Tcl_ErrnoId()
{
	static CHAR msgText[256];
	getLastErrorText(msgText, sizeof(msgText));
	return msgText;
}

/*
*----------------------------------------------------------------------
*
* Tcl_SignalId --
*	Return a textual identifier for a signal number.
*
* Results:
*	This procedure returns a machine-readable textual identifier that corresponds to sig.  The identifier is the same as the
*	#define name in signal.h.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
char *Tcl_SignalId(int sig)
{
	return "unknown signal";
}

/*
*----------------------------------------------------------------------
*
* Tcl_SignalMsg --
*	Return a human-readable message describing a signal.
*
* Results:
*	This procedure returns a string describing sig that should make sense to a human.  It may not be easy for a machine to parse.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
char *Tcl_SignalMsg(int sig)
{
	return "unknown signal";
}

#endif