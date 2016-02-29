#include "Tcl+Int.h"
#if OS_WIN
#include "Tcl+Win.h"
#include <io.h>

// Data structures of the following type are used by Tcl_Fork and Tcl_WaitPids to keep track of child processes.
#define WAIT_STATUS_TYPE int

typedef struct {
	HANDLE pid;					// Process id of child.
	WAIT_STATUS_TYPE status;	// Status returned when child exited or suspended.
	int flags;					// Various flag bits;  see below for definitions.
} WaitInfo;

// Flag bits in WaitInfo structures:
//
#define WI_READY	1 // Non-zero means process has exited or suspended since it was forked or last returned by Tcl_WaitPids.
#define WI_DETACHED	2 // Non-zero means no-one cares about the process anymore.  Ignore it until it exits, then forget about it.
static WaitInfo *waitTable = NULL;
static int waitTableSize = 0;	// Total number of entries available in waitTable.
static int waitTableUsed = 0;	// Number of entries in waitTable that are actually in use right now.  Active entries are always at the beginning of the table.
#define WAIT_TABLE_GROW_BY 4

/*
*----------------------------------------------------------------------
*
* Tcl_EvalFile --
*	Read in a file and process the entire file as one gigantic Tcl command.
*
* Results:
*	A standard Tcl result, which is either the result of executing the file or an error indicating why the file couldn't be read.
*
* Side effects:
*	Depends on the commands in the file.
*
*----------------------------------------------------------------------
*/
int Tcl_EvalFile(Tcl_Interp *interp, char *fileName)
{
	int result;
	Interp *iPtr = (Interp *)interp;
	char *oldScriptFile = iPtr->scriptFile;
	iPtr->scriptFile = fileName;
	fileName = Tcl_TildeSubst(interp, fileName);
	if (fileName == NULL) {
		goto error;
	}
	FILE *file = fopen(fileName, "rb");
	if (!file) {
		Tcl_AppendResult(interp, "couldn't read file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		goto error;
	}
	struct stat statBuf;
	if (fstat(fileno(file), &statBuf) == -1) {
		Tcl_AppendResult(interp, "couldn't stat file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		fclose(file);
		goto error;
	}
	int fileSize = statBuf.st_size;
	char *cmdBuffer = (char *)_allocFast((unsigned)fileSize+1);
	if (fread(cmdBuffer, 1, fileSize, file) != fileSize) {
		Tcl_AppendResult(interp, "error in reading file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		fclose(file);
		_freeFast(cmdBuffer);
		goto error;
	}
	if (fclose(file) != 0) {
		Tcl_AppendResult(interp, "error closing file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		_freeFast(cmdBuffer);
		goto error;
	}
	cmdBuffer[fileSize] = 0;
	char *end;
	result = Tcl_Eval(interp, cmdBuffer, 0, &end);
	if (result == TCL_RETURN) {
		result = TCL_OK;
	}
	if (result == TCL_ERROR) {
		// Record information telling where the error occurred.
		char msg[200];
		sprintf(msg, "\n    (file \"%.150s\" line %d)", fileName, interp->errorLine);
		Tcl_AddErrorInfo(interp, msg);
	}
	_freeFast(cmdBuffer);
	iPtr->scriptFile = oldScriptFile;
	return result;

error:
	iPtr->scriptFile = oldScriptFile;
	return TCL_ERROR;
}

static SECURITY_ATTRIBUTES *TclStdSecAttrs()
{
	static SECURITY_ATTRIBUTES secAtts;
	secAtts.nLength = sizeof(SECURITY_ATTRIBUTES);
	secAtts.lpSecurityDescriptor = NULL;
	secAtts.bInheritHandle = TRUE;
	return &secAtts;
}

static int TclWinFindExecutable(const char *originalName, char *fullPath)
{
	static char *_extensions[] = { ".exe", "", ".bat" };
	for (int i = 0; i < _lengthof(_extensions); i++)
	{
		lstrcpyn(fullPath, originalName, MAX_PATH - 5);
		lstrcat(fullPath, _extensions[i]);

		if (SearchPath(NULL, fullPath, NULL, MAX_PATH, fullPath, NULL) == 0)
			continue;
		if (GetFileAttributes(fullPath) & FILE_ATTRIBUTE_DIRECTORY)
			continue;
		return 0;
	}
	return -1;
}

static char *TclWinBuildCommandLine(const char *args[])
{
	TextBuilder b;
	TextBuilder::Init(&b);
	const char *start;
	bool quote;

	for (int i = 0; args[i]; i++)
	{
		if (i > 0)
			b.Append(" ", 1);
		if (args[i][0] == '\0')
			quote = true;
		else
		{
			quote = false;
			for (start = args[i]; *start; start++)
				if (_isspace(*start))
				{
					quote = true;
					break;
				}
		}
		if (quote)
			b.Append("\"" , 1);

		start = args[i];
		const char *special;
		for (special = args[i]; ; )
		{
			if ((*special == '\\' && special[1] == '\\') || special[1] == '"' || (quote && special[1] == '\0'))
			{
				b.Append(start, (int)(special - start));
				start = special;
				while (1)
				{
					special++;
					if (*special == '"' || (quote && *special == '\0'))
					{
						b.Append(start, (int)(special - start)); // N backslashes followed a quote -> insert, N * 2 + 1 backslashes then a quote.
						break;
					}
					if (*special != '\\')
						break;
				}
				b.Append(start, (int)(special - start));
				start = special;
			}
			if (*special == '"')
			{
				if (special == start)
					b.Append("\"", 1);
				else
					b.Append(start, (int)(special - start));
				b.Append("\\\"", 2);
				start = special + 1;
			}
			if (*special == '\0')
				break;
			special++;
		}
		b.Append(start, (int)(special - start));
		if (quote)
			b.Append("\"", 1);
	}
	return b.ToString();
}

/*
*----------------------------------------------------------------------
*
* Tcl_Fork --
*	Create a new process using the vfork system call, and keep track of it for "safe" waiting with Tcl_WaitPids.
*
* Results:
*	The return value is the value returned by the vfork system call (0 means child, > 0 means parent (value is child id), < 0 means error).
*
* Side effects:
*	A new process is created, and an entry is added to an internal table of child processes if the process is created successfully.
*
*----------------------------------------------------------------------
*/
static HANDLE TclStartWinProcess(Tcl_Interp *interp, const char *args[], char *env, HANDLE inputId, HANDLE outputId, HANDLE errorId)
{
	HANDLE pid = INVALID_HANDLE_VALUE;

	char execPath[MAX_PATH];
	if (TclWinFindExecutable(args[0], execPath) < 0)
		return INVALID_HANDLE_VALUE;
	args[0] = execPath;

	HANDLE hProcess = GetCurrentProcess();
	char *cmdLineObj = TclWinBuildCommandLine(args);

	// STARTF_USESTDHANDLES must be used to pass handles to child process. Using SetStdHandle() and/or dup2() only works when a console mode
	// parent process is spawning an attached console mode child process.
	STARTUPINFO startInfo;
	ZeroMemory(&startInfo, sizeof(startInfo));
	startInfo.cb = sizeof(startInfo);
	startInfo.dwFlags   = STARTF_USESTDHANDLES;
	startInfo.hStdInput = INVALID_HANDLE_VALUE;
	startInfo.hStdOutput= INVALID_HANDLE_VALUE;
	startInfo.hStdError = INVALID_HANDLE_VALUE;

	// Duplicate all the handles which will be passed off as stdin, stdout and stderr of the child process. The duplicate handles are set to
	// be inheritable, so the child process can use them.
	HANDLE h;
	if (inputId == INVALID_HANDLE_VALUE)
	{
		if (CreatePipe(&startInfo.hStdInput, &h, TclStdSecAttrs(), 0) != FALSE)
			CloseHandle(h);
	}
	else 
		DuplicateHandle(hProcess, inputId, hProcess, &startInfo.hStdInput, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdInput == INVALID_HANDLE_VALUE)
		goto end;

	if (outputId == INVALID_HANDLE_VALUE)
		startInfo.hStdOutput = CreateFile("NUL:", GENERIC_WRITE, 0, TclStdSecAttrs(), OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	else
		DuplicateHandle(hProcess, outputId, hProcess, &startInfo.hStdOutput, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdOutput == INVALID_HANDLE_VALUE)
		goto end;

	if (errorId == INVALID_HANDLE_VALUE) // If handle was not set, errors should be sent to an infinitely deep sink.
		startInfo.hStdError = CreateFile("NUL:", GENERIC_WRITE, 0, TclStdSecAttrs(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	else
		DuplicateHandle(hProcess, errorId, hProcess, &startInfo.hStdError, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdError == INVALID_HANDLE_VALUE)
		goto end;

	PROCESS_INFORMATION procInfo;
	if (!CreateProcess(NULL, cmdLineObj, NULL, NULL, TRUE, 0, env, NULL, &startInfo, &procInfo))
		goto end;

	// "When an application spawns a process repeatedly, a new thread instance will be created for each process but the previous
	// instances may not be cleaned up.  This results in a significant virtual memory loss each time the process is spawned.  If there
	// is a WaitForInputIdle() call between CreateProcess() and CloseHandle(), the problem does not occur." PSS ID Number: Q124121
	WaitForInputIdle(procInfo.hProcess, 5000);
	CloseHandle(procInfo.hThread);

	pid = procInfo.hProcess;

end:
	_free(cmdLineObj);
	if (startInfo.hStdInput != INVALID_HANDLE_VALUE)
		CloseHandle(startInfo.hStdInput);
	if (startInfo.hStdOutput != INVALID_HANDLE_VALUE)
		CloseHandle(startInfo.hStdOutput);
	if (startInfo.hStdError != INVALID_HANDLE_VALUE)
		CloseHandle(startInfo.hStdError);
	return pid;


	//WaitInfo *waitPtr;
	//pid_t pid;

	//// Enlarge the wait table if there isn't enough space for a new entry.
	//if (waitTableUsed == waitTableSize) {
	//	int newSize = waitTableSize + WAIT_TABLE_GROW_BY;
	//	WaitInfo *newWaitTable = (WaitInfo *)_allocFast((unsigned)(newSize * sizeof(WaitInfo)));
	//	memcpy(newWaitTable, waitTable, (waitTableSize * sizeof(WaitInfo)));
	//	if (waitTable != NULL) {
	//		_freeFast((char *)waitTable);
	//	}
	//	waitTable = newWaitTable;
	//	waitTableSize = newSize;
	//}

	//// Make a new process and enter it into the table if the fork is successful.
	//waitPtr = &waitTable[waitTableUsed];
	//pid = fork();
	//if (pid > 0) {
	//	waitPtr->pid = pid;
	//	waitPtr->flags = 0;
	//	waitTableUsed++;
	//}
	//return pid;
}

/*
*----------------------------------------------------------------------
*
* Tcl_WaitPids --
*	This procedure is used to wait for one or more processes created by Tcl_Fork to exit or suspend.  It records information about
*	all processes that exit or suspend, even those not waited for, so that later waits for them will be able to get the status information.
*
* Results:
*	-1 is returned if there is an error in the wait kernel call. Otherwise the pid of an exited/suspended process from *pidPtr
*	is returned and *statusPtr is set to the status value returned by the wait kernel call.
*
* Side effects:
*	Doesn't return until one of the pids at *pidPtr exits or suspends.
*
*----------------------------------------------------------------------
*/
int Tcl_WaitPids(int numPids, int *pidPtr, int *statusPtr)
{
	return -1;
	//DWORD ret = WaitForSingleObject(pid, nohang ? 0 : INFINITE);
	//if (ret == WAIT_TIMEOUT || ret == WAIT_FAILED) {
	//    /* WAIT_TIMEOUT can only happend with WNOHANG */
	//    return JIM_BAD_PID;
	//}
	//GetExitCodeProcess(pid, &ret);
	//*status = ret;
	//CloseHandle(pid);
	//return pid;
}


/*
*----------------------------------------------------------------------
*
* Tcl_DetachPids --
*	This procedure is called to indicate that one or more child processes have been placed in background and are no longer
*	cared about.  They should be ignored in future calls to Tcl_WaitPids.
*
* Results:
*	None.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
void Tcl_DetachPids(int numPids, HANDLE *pidPtr)
{
	int count;
	HANDLE pid;
	register WaitInfo *waitPtr;
	for (int i = 0; i < numPids; i++) {
		pid = pidPtr[i];
		for (waitPtr = waitTable, count = waitTableUsed; count > 0; waitPtr++, count--) {
			if (pid != waitPtr->pid) {
				continue;
			}
			//If the process has already exited then destroy its table entry now.
			if ((waitPtr->flags & WI_READY)) {
				*waitPtr = waitTable[waitTableUsed-1];
				waitTableUsed--;
			} else {
				waitPtr->flags |= WI_DETACHED;
			}
			goto nextPid;
		}
		_panic("Tcl_Detach couldn't find process");
nextPid:
		continue;
	}
}

void mktemp(char *buf, int size)
{
	TCHAR lpTempPathBuffer[MAX_PATH];
	DWORD dwRetVal = GetTempPath(MAX_PATH, lpTempPathBuffer); 
	if (dwRetVal > MAX_PATH || (dwRetVal == 0))
		_panic("mktemp failed");
	UINT uRetVal = GetTempFileName(lpTempPathBuffer, TEXT("tmp"), 0, buf); 
	if (uRetVal == 0)
		_panic("mktemp failed");
}

static int TclPipe(HANDLE pipefd[2])
{
	if (CreatePipe(&pipefd[0], &pipefd[1], NULL, 0)) {
		return 0;
	}
	return -1;
}

static HANDLE TclDupFd(HANDLE fd)
{
	HANDLE dupfd;
	HANDLE pid = GetCurrentProcess();
	if (DuplicateHandle(pid, fd, pid, &dupfd, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
		return dupfd;
	}
	return INVALID_HANDLE_VALUE;
}

static HANDLE TclFileno(FILE *f)
{
    return (HANDLE)_get_osfhandle(_fileno(f));
}

#define TclClose CloseHandle

/*
*----------------------------------------------------------------------
*
* Tcl_CreatePipeline --
*	Given an argc/args array, instantiate a pipeline of processes as described by the args.
*
* Results:
*	The return value is a count of the number of new processes created, or -1 if an error occurred while creating the pipeline.
*	*pidArrayPtr is filled in with the address of a dynamically allocated array giving the ids of all of the processes.  It
*	is up to the caller to free this array when it isn't needed anymore.  If inPipePtr is non-NULL, *inPipePtr is filled in
*	with the file id for the input pipe for the pipeline (if any): the caller must eventually close this file.  If outPipePtr
*	isn't NULL, then *outPipePtr is filled in with the file id for the output pipe from the pipeline:  the caller must close
*	this file.  If errFilePtr isn't NULL, then *errFilePtr is filled with a file id that may be used to read error output after the
*	pipeline completes.
*
* Side effects:
*	Processes and pipes are created.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_CreatePipeline(Tcl_Interp *interp, int argc, const char *args[], HANDLE **pidArrayPtr, HANDLE *inPipePtr, HANDLE *outPipePtr, HANDLE *errFilePtr)
{
	if (inPipePtr) {
		*inPipePtr = INVALID_HANDLE_VALUE;
	}
	if (outPipePtr) {
		*outPipePtr = INVALID_HANDLE_VALUE;
	}
	if (errFilePtr) {
		*errFilePtr = INVALID_HANDLE_VALUE;
	}
	HANDLE pipeIds[2]; // File ids for pipe that's being created.
	pipeIds[0] = pipeIds[1] = INVALID_HANDLE_VALUE;
	int numPids = 0; // Actual number of processes that exist at *pidPtr right now.

	// First, scan through all the arguments to figure out the structure of the pipeline.  Count the number of distinct processes (it's the number of "|" arguments).
	// If there are "<", "<<", or ">" arguments then make note of input and output redirection and remove these arguments and the arguments that follow them.
	int cmdCount = 1; // Count of number of distinct commands found in argc/args.
	int lastBar = -1;
	char *input = NULL; // Describes input for pipeline, depending on "inputFile".  NULL means take input from stdin/pipe.
	int inputFile = 0; // 1 means input is name of input file. 2 means input is filehandle name. 0 means input holds actual text to be input to command.
	int outputFile = 0; // 0 means output is the name of output file. 1 means output is the name of output file, and append. 2 means output is filehandle name. All this is ignored if output is NULL
	int errorFile = 0; // 0 means error is the name of error file. 1 means error is the name of error file, and append. 2 means error is filehandle name. All this is ignored if error is NULL
	char *output = NULL; // Holds name of output file to pipe to, or NULL if output goes to stdout/pipe.
	char *error = NULL; // Holds name of stderr file to pipe to, or NULL if stderr goes to stderr/pipe.
	int i;
	for (i = 0; i < argc; i++) {
		int removecount = 1;
		if (args[i][0] == '|' && args[i][1] == 0) {
			if (i == (lastBar+1) || i == (argc-1)) {
				interp->result = "illegal use of | in command";
				return -1;
			}
			lastBar = i;
			cmdCount++;
			continue;
		} else if (args[i][0] == '<') {
			input = (char *)args[i] + 1;
			inputFile = 1;
			if (*input == '<') {
				inputFile = 0;
				input++;
			}
			else if (*input == '@') {
				inputFile = 2;
				input++;
			}
			if (!*input) {
				input = (char *)args[i + 1];
				removecount++;
			}
		} else if (args[i][0] == '>') {
			output = (char *)args[i] + 1;
			outputFile = 0;
			if (*output == '@') {
				outputFile = 2;
				output++;
			}
			else if (*output == '>') {
				outputFile = 1;
				output++;
			}
			if (!*output) {
				output = (char *)args[i + 1];
				removecount++;
			}
		} else if (args[i][0] == '2' && args[i][1] == '>') {
			error = (char *)args[i] + 2;
			errorFile = 0;
			if (*error == '@') {
				errorFile = 2;
				error++;
			}
			else if (*error == '>') {
				errorFile = 1;
				error++;
			}
			if (!*error) {
				error = (char *)args[i + 1];
				removecount++;
			}
		} else {
			continue;
		}
		if (i + removecount > argc) {
			Tcl_AppendResult(interp, "can't specify \"", args[i], "\" as last word in command", (char *)NULL);
			return -1;
		}
		for (int j = i+removecount; j < argc; j++) {
			args[j-removecount] = args[j];
		}
		argc -= removecount;
		i -= removecount; // Process new arg from same position.
	}
	if (argc == 0) {
		interp->result =  "didn't specify command to execute";
		return -1;
	}

	// Set up the redirected input source for the pipeline, if so requested.
	HANDLE inputId = INVALID_HANDLE_VALUE; // Readable file id input to current command in pipeline (could be file or pipe).  -1 means use stdin.
	if (input != NULL) {
		if (inputFile == 0) {
			// Immediate data in command.  Create temporary file and put data into file.
			char inName[MAX_PATH];
			mktemp(inName, sizeof(inName));
			FILE *inputF = fopen(inName, "w+");
			inputId = TclFileno(inputF);
			//TclFdOpenForWrite(TclDupFd(handle))
			if (inputId < 0) {
				Tcl_AppendResult(interp, "couldn't create input file for command: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			int length = (int)strlen(input);
			if (fwrite(input, length, 1, inputF) != length) {
				Tcl_AppendResult(interp, "couldn't write file input for command: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			if (fseek(inputF, 0L, 0) == -1 || !DeleteFile(inName)) {
				Tcl_AppendResult(interp, "couldn't reset or remove input file for command: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		} else if (inputFile == 2) {
			// File redirection.  Just open the file.
			OpenFile_ *filePtr;
			if (TclGetOpenFile(interp, input, &filePtr) != TCL_OK) {
				goto error;
			}
			if (!filePtr->readable) {
				Tcl_AppendResult(interp, "\"", input, "\" wasn't opened for reading", (char *)NULL);
				goto error;
			}
			inputId = TclDupFd(TclFileno(filePtr->f2 ? filePtr->f2 : filePtr->f));
		} else {
			// File redirection.  Just open the file.
			inputId = TclFileno(fopen(input, "rb"));
			if (!inputId) {
				Tcl_AppendResult(interp, "couldn't read file \"", input, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (inPipePtr != NULL) {
		if (TclPipe(pipeIds) != 0) {
			Tcl_AppendResult(interp, "couldn't create input pipe for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		inputId = pipeIds[0];
		*inPipePtr = pipeIds[1];
		pipeIds[0] = pipeIds[1] = NULL;
	}

	// Set up the redirected output sink for the pipeline from one of two places, if requested.
	HANDLE lastOutputId = INVALID_HANDLE_VALUE; // Write file id for output from last command in pipeline (could be file or pipe). -1 means use stdout.
	if (output != NULL) {
		if (outputFile == 2) {
			OpenFile_ *filePtr;
			if (TclGetOpenFile(interp, output, &filePtr) != TCL_OK) {
				goto error;
			}
			if (!filePtr->writable) {
				Tcl_AppendResult(interp, "\"", input, "\" wasn't opened for writing", (char *)NULL);
				goto error;
			}
			fflush(filePtr->f2 ? filePtr->f2 : filePtr->f);
			lastOutputId = TclDupFd(TclFileno(filePtr->f2 ? filePtr->f2 : filePtr->f));
		}
		else {
			// Output is to go to a file.
			char *mode = "w";
			if (outputFile == 1) {
				mode = "w+";
			}
			lastOutputId = TclFileno(fopen(output, mode));
			if (lastOutputId < 0) {
				Tcl_AppendResult(interp, "couldn't write file \"", output, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (outPipePtr != NULL) {
		// Output is to go to a pipe.
		if (TclPipe(pipeIds) != 0) {
			Tcl_AppendResult(interp, "couldn't create output pipe: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		lastOutputId = pipeIds[1];
		*outPipePtr = pipeIds[0];
		pipeIds[0] = pipeIds[1] = NULL;
	}
	// If we are redirecting stderr with 2>filename or 2>@fileId, then we ignore errFilePtr
	HANDLE errorId = INVALID_HANDLE_VALUE; // Writable file id for all standard error output from all commands in pipeline.  -1 means use stderr.
	if (error != NULL) {
		if (errorFile == 2) {
			OpenFile_ *filePtr;
			if (TclGetOpenFile(interp, error, &filePtr) != TCL_OK) {
				goto error;
			}
			if (!filePtr->writable) {
				Tcl_AppendResult(interp, "\"", input, "\" wasn't opened for writing", (char *)NULL);
				goto error;
			}
			fflush(filePtr->f2 ? nullptr : filePtr->f);
			errorId = TclDupFd(TclFileno(filePtr->f2 ? nullptr : filePtr->f));
		}
		else {
			// Output is to go to a file.
			char *mode = "w";
			if (errorFile == 1) {
				mode = "w+";
			}
			errorId = TclFileno(fopen(error, mode));
			if (errorId < 0) {
				Tcl_AppendResult(interp, "couldn't write file \"", error, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (errFilePtr != NULL) {
		// Set up the standard error output sink for the pipeline, if requested.  Use a temporary file which is opened, then deleted.
		// Could potentially just use pipe, but if it filled up it could cause the pipeline to deadlock:  we'd be waiting for processes
		// to complete before reading stderr, and processes couldn't complete because stderr was backed up.
		char errName[MAX_PATH];
		mktemp(errName, sizeof(errName));
		errorId = TclFileno(fopen(errName, "w"));
		if (errorId < 0) {
errFileError:
			Tcl_AppendResult(interp, "couldn't create error file for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		*errFilePtr = TclFileno(fopen(errName, "rb"));
		if (*errFilePtr < 0) {
			goto errFileError;
		}
		if (!DeleteFile(errName)) {
			Tcl_AppendResult(interp, "couldn't remove error file for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
	}

	// Scan through the argc array, forking off a process for each group of arguments between "|" arguments.
	HANDLE *pidPtr = (HANDLE *)_allocFast((unsigned)(cmdCount * sizeof(HANDLE))); // Points to malloc-ed array holding all the pids of child processes.
	for (i = 0; i < numPids; i++) {
		pidPtr[i] = INVALID_HANDLE_VALUE;
	}
	HANDLE outputId = INVALID_HANDLE_VALUE; // Writable file id for output from current command in pipeline (could be file or pipe). -1 means use stdout.
	int lastArg;
	for (int firstArg = 0; firstArg < argc; numPids++, firstArg = lastArg+1) {
		for (lastArg = firstArg; lastArg < argc; lastArg++) {
			if (args[lastArg][0] == '|' && args[lastArg][1] == 0) {
				break;
			}
		}
		args[lastArg] = NULL;
		if (lastArg == argc) {
			outputId = lastOutputId;
		} else {
			if (TclPipe(pipeIds) != 0) {
				Tcl_AppendResult(interp, "couldn't create pipe: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			outputId = pipeIds[1];
		}
		char *execName = Tcl_TildeSubst(interp, (char *)args[firstArg]);

#if 0
		int pid = -1; //Tcl_Fork();
		if (pid == -1) {
			Tcl_AppendResult(interp, "couldn't fork child process: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		if (pid == 0) {
			char errSpace[200];
			//if ((inputId != -1 && dup2(inputId, 0) == -1) || (outputId != -1 && dup2(outputId, 1) == -1) || (errorId != -1 && dup2(errorId, 2) == -1)) {
			//	char *err = "forked process couldn't set up input/output\n";
			//	fwrite(err, strlen(err), 1, (!errorId ? 2 : errorId));
			//	_exit(1);
			//}
			//for (i = 3; i <= outputId || i <= inputId || i <= errorId; i++) {
			//	fclose(i);
			//}
			//execvp(execName, &args[firstArg]);
			sprintf(errSpace, "couldn't find \"%.150s\" to execute\n", args[firstArg]);
			fwrite(errSpace, strlen(errSpace), 1, stderr);
			_exit(1);
		} else {
			pidPtr[numPids] = pid;
		}
#endif

		// Close off our copies of file descriptors that were set up for this child, then set up the input for the next child.
		if (inputId) {
			TclClose(inputId);
		}
		if (outputId) {
			TclClose(outputId);
		}
		inputId = pipeIds[0];
		pipeIds[0] = pipeIds[1] = NULL;
	}
	*pidArrayPtr = pidPtr;

	// All done.  Cleanup open files lying around and then return.
cleanup:
	if (inputId) {
		TclClose(inputId);
	}
	if (lastOutputId) {
		TclClose(lastOutputId);
	}
	if (errorId) {
		TclClose(errorId);
	}
	return numPids;

	// An error occurred.  There could have been extra files open, such as pipes between children.  Clean them all up.  Detach any child processes that have been created.
error:
	if (inPipePtr != NULL && *inPipePtr) {
		TclClose(*inPipePtr);
		*inPipePtr = NULL;
	}
	if (outPipePtr != NULL && *outPipePtr) {
		TclClose(*outPipePtr);
		*outPipePtr = NULL;
	}
	if (errFilePtr != NULL && *errFilePtr) {
		TclClose(*errFilePtr);
		*errFilePtr = NULL;
	}
	if (pipeIds[0]) {
		TclClose(pipeIds[0]);
	}
	if (pipeIds[1]) {
		TclClose(pipeIds[1]);
	}
	if (pidPtr != NULL) {
		for (i = 0; i < numPids; i++) {
			if (pidPtr[i] != INVALID_HANDLE_VALUE) {
				Tcl_DetachPids(1, &pidPtr[i]);
			}
		}
		_freeFast((char *)pidPtr);
	}
	numPids = -1;
	goto cleanup;
}

/*
*----------------------------------------------------------------------
*
* Tcl_OSError --
*	This procedure is typically called after UNIX kernel calls return errors.  It stores machine-readable information about
*	the error in $errorCode returns an information string for the caller's use.
*
* Results:
*	The return value is a human-readable string describing the error, as returned by strerror.
*
* Side effects:
*	The global variable $errorCode is reset.
*
*----------------------------------------------------------------------
*/
char *Tcl_OSError(Tcl_Interp *interp)
{
	char *id = Tcl_ErrnoId();
	Tcl_SetErrorCode(interp, "UNIX", id, (char *)NULL);
	return id;
}

/*
*----------------------------------------------------------------------
*
* TclMakeFileTable --
*	Create or enlarge the file table for the interpreter, so that there is room for a given index.
*
* Results:
*	None.
*
* Side effects:
*	The file table for iPtr will be created if it doesn't exist (and entries will be added for stdin, stdout, and stderr).
*	If it already exists, then it will be grown if necessary.
*
*----------------------------------------------------------------------
*/
void TclMakeFileTable(Interp *iPtr, int index)
{
	int i;
	// If the table doesn't even exist, then create it and initialize entries for standard files.
#ifdef DEBUG_FDS
	syslog(LOG_INFO, "TclMakeFileTable() numFiles=%d, index=%d", iPtr->numFiles, index);
#endif
	if (iPtr->numFiles == 0) {
		if (index < 2) {
			iPtr->numFiles = 3;
		} else {
			iPtr->numFiles = index+1;
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() allocating table of size %d", iPtr->numFiles);
#endif
		iPtr->filePtrArray = (OpenFile_ **)_allocFast(iPtr->numFiles*sizeof(OpenFile_ *));
		for (i = iPtr->numFiles-1; i >= 0; i--) {
			iPtr->filePtrArray[i] = NULL;
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() after freopen(): stdin fd=%d, stdout fd=%d, stderr fd=%d", _fileno(stdin), _fileno(stdout), _fileno(stderr));
#endif
		OpenFile_ *filePtr;
		if (_fileno(stdin) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stdin;
			filePtr->f2 = NULL;
			filePtr->readable = 1;
			filePtr->writable = 0;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = NULL;
			iPtr->filePtrArray[_fileno(stdin)] = filePtr;
		}
		if (_fileno(stdout) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stdout;
			filePtr->f2 = NULL;
			filePtr->readable = 0;
			filePtr->writable = 1;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = NULL;
			iPtr->filePtrArray[_fileno(stdout)] = filePtr;
		}
		if (_fileno(stderr) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stderr;
			filePtr->f2 = NULL;
			filePtr->readable = 0;
			filePtr->writable = 1;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = NULL;
			iPtr->filePtrArray[_fileno(stderr)] = filePtr;
		}
	} else if (index >= iPtr->numFiles) {
		int newSize = index+1;
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() increasing size from %d to %d", iPtr->numFiles, newSize);
#endif
		OpenFile_ **newPtrArray = (OpenFile_ **)_allocFast(newSize*sizeof(OpenFile_ *));
		memcpy(newPtrArray, iPtr->filePtrArray, iPtr->numFiles*sizeof(OpenFile_ *));
		for (i = iPtr->numFiles; i < newSize; i++) {
			newPtrArray[i] = NULL;
		}
		_freeFast((char *)iPtr->filePtrArray);
		iPtr->numFiles = newSize;
		iPtr->filePtrArray = newPtrArray;
	}
}

/*
*----------------------------------------------------------------------
*
* TclGetOpenFile --
*	Given a string identifier for an open file, find the corresponding open file structure, if there is one.
*
* Results:
*	A standard Tcl return value.  If the open file is successfully located, *filePtrPtr is modified to point to its structure.
*	If TCL_ERROR is returned then interp->result contains an error message.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
int TclGetOpenFile(Tcl_Interp *interp, char *string, OpenFile_ **filePtrPtr)
{
	int fd = 0; // Initial value needed only to stop compiler warnings.
	Interp *iPtr = (Interp *)interp;
	if (string[0] == 'f' && string[1] == 'i' && string[2] == 'l' && string[3] == 'e') {
		char *end;
		fd = strtoul(string+4, &end, 10);
		if (end == string+4 || *end != 0) {
			goto badId;
		}
	} else if (string[0] == 's' && string[1] == 't' && string[2] == 'd') {
		if (!strcmp(string+3, "in")) {
			fd = _fileno(stdin);
		} else if (!strcmp(string+3, "out") ) {
			fd = _fileno(stdout);
		} else if (!strcmp(string+3, "err")) {
			fd = _fileno(stderr);
		} else {
			goto badId;
		}
	} else {
badId:
		Tcl_AppendResult(interp, "bad file identifier \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}

#ifdef DEBUG_FDS
	syslog(LOG_INFO, "TclGetOpenFile(%s), fd=%d, numFiles=%d", string, fd, iPtr->numFiles);
#endif
	if (iPtr->numFiles == 0) {
		TclMakeFileTable(iPtr, fd);
	}
	if (fd >= iPtr->numFiles || iPtr->filePtrArray[fd] == NULL) {
		Tcl_AppendResult(interp, "file \"", string, "\" isn't open", (char *)NULL);
		return TCL_ERROR;
	}
#ifdef DEBUG_FDS
	syslog(LOG_INFO, "TclGetOpenFile(%s): filePtrArray[%d]=%p", string, fd, iPtr->filePtrArray[fd]);
#endif
	*filePtrPtr = iPtr->filePtrArray[fd];
	return TCL_OK;
}
#endif