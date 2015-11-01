// tclUnixUtil.c --
//
//	This file contains a collection of utility procedures that are present in the Tcl's UNIX core but not in the generic
//	core.  For example, they do file manipulation and process manipulation.
//
// Copyright 1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that this copyright notice appears in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "Tcl+Int.h"
#if OS_UNIX
#include "Tcl+Unix.h"

#ifndef NO_FORK
// Data structures of the following type are used by Tcl_Fork and Tcl_WaitPids to keep track of child processes.
#define WAIT_STATUS_TYPE int

typedef struct {
	int pid;					// Process id of child.
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
#endif /* NO_FORK */

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
	int fileId = open(fileName, O_RDONLY, 0);
	if (fileId < 0) {
		Tcl_AppendResult(interp, "couldn't read file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		goto error;
	}
	struct stat statBuf;
	if (fstat(fileId, &statBuf) == -1) {
		Tcl_AppendResult(interp, "couldn't stat file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		close(fileId);
		goto error;
	}
	char *cmdBuffer = (char *)_allocFast((unsigned)statBuf.st_size+1);
	if (read(fileId, cmdBuffer, (int)statBuf.st_size) != statBuf.st_size) {
		Tcl_AppendResult(interp, "error in reading file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		close(fileId);
		_freeFast(cmdBuffer);
		goto error;
	}
	if (close(fileId) != 0) {
		Tcl_AppendResult(interp, "error closing file \"", fileName, "\": ", Tcl_OSError(interp), (char *)NULL);
		_freeFast(cmdBuffer);
		goto error;
	}
	cmdBuffer[statBuf.st_size] = 0;
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

#ifndef NO_FORK
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
int Tcl_Fork()
{
	WaitInfo *waitPtr;
	pid_t pid;

	// Disable SIGPIPE signals:  if they were allowed, this process might go away unexpectedly if children misbehave.  This code
	// can potentially interfere with other application code that expects to handle SIGPIPEs;  what's really needed is an
	// arbiter for signals to allow them to be "shared".
	if (waitTable == NULL) {
		signal(SIGPIPE, SIG_IGN);
	}

	// Enlarge the wait table if there isn't enough space for a new entry.
	if (waitTableUsed == waitTableSize) {
		int newSize = waitTableSize + WAIT_TABLE_GROW_BY;
		WaitInfo *newWaitTable = (WaitInfo *)_allocFast((unsigned)(newSize * sizeof(WaitInfo)));
		memcpy(newWaitTable, waitTable, (waitTableSize * sizeof(WaitInfo)));
		if (waitTable != NULL) {
			_freeFast((char *) waitTable);
		}
		waitTable = newWaitTable;
		waitTableSize = newSize;
	}

	// Make a new process and enter it into the table if the fork is successful.
	waitPtr = &waitTable[waitTableUsed];
	pid = fork();
	if (pid > 0) {
		waitPtr->pid = pid;
		waitPtr->flags = 0;
		waitTableUsed++;
	}
	return pid;
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
	int count, pid;
	register WaitInfo *waitPtr;
	while (true) {
		// Scan the table of child processes to see if one of the specified children has already exited or suspended.  If so, remove it from the table and return its status.
		bool anyProcesses = false;
		for (waitPtr = waitTable, count = waitTableUsed; count > 0; waitPtr++, count--) {
			for (int i = 0; i < numPids; i++) {
				if (pidPtr[i] != waitPtr->pid) {
					continue;
				}
				anyProcesses = true;
				if (waitPtr->flags & WI_READY) {
					*statusPtr = *((int *)&waitPtr->status);
					pid = waitPtr->pid;
					if (WIFEXITED(waitPtr->status) || WIFSIGNALED(waitPtr->status)) {
						*waitPtr = waitTable[waitTableUsed-1];
						waitTableUsed--;
					} else {
						waitPtr->flags &= ~WI_READY;
					}
					return pid;
				}
			}
		}

		// Make sure that the caller at least specified one valid process to wait for.
		if (!anyProcesses) {
			errno = ECHILD;
			return -1;
		}

		// Wait for a process to exit or suspend, then update its entry in the table and go back to the beginning of the
		// loop to see if it's one of the desired processes.
		WAIT_STATUS_TYPE status;
		pid = wait(&status);
		if (pid < 0) {
			return pid;
		}
		for (waitPtr = waitTable, count = waitTableUsed; ; waitPtr++, count--) {
			if (count == 0) {
				break;			// Ignore unknown processes.
			}
			if (pid != waitPtr->pid) {
				continue;
			}

			// If the process has been detached, then ignore anything other than an exit, and drop the entry on exit.
			if (waitPtr->flags & WI_DETACHED) {
				if (WIFEXITED(status) || WIFSIGNALED(status)) {
					*waitPtr = waitTable[waitTableUsed-1];
					waitTableUsed--;
				}
			} else {
				waitPtr->status = status;
				waitPtr->flags |= WI_READY;
			}
			break;
		}
	}
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
void Tcl_DetachPids(int numPids, int *pidPtr)
{
	int count, pid;
	register WaitInfo *waitPtr;
	for (int i = 0; i < numPids; i++) {
		pid = pidPtr[i];
		for (waitPtr = waitTable, count = waitTableUsed; count > 0; waitPtr++, count--) {
			if (pid != waitPtr->pid) {
				continue;
			}

			// If the process has already exited then destroy its table entry now.
			if ((waitPtr->flags & WI_READY) && (WIFEXITED(waitPtr->status) || WIFSIGNALED(waitPtr->status))) {
				*waitPtr = waitTable[waitTableUsed-1];
				waitTableUsed--;
			} else {
				waitPtr->flags |= WI_DETACHED;
			}
			goto nextPid;
		}
		panic("Tcl_Detach couldn't find process");
nextPid:
		continue;
	}
}

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
int Tcl_CreatePipeline(Tcl_Interp *interp, int argc, const char *args[], int **pidArrayPtr, int *inPipePtr, int *outPipePtr, int *errFilePtr)
{
	if (inPipePtr != NULL) {
		*inPipePtr = -1;
	}
	if (outPipePtr != NULL) {
		*outPipePtr = -1;
	}
	if (errFilePtr != NULL) {
		*errFilePtr = -1;
	}
	int pipeIds[2]; // File ids for pipe that's being created.
	pipeIds[0] = pipeIds[1] = -1;
	int numPids = 0; // Actual number of processes that exist at *pidPtr right now.

	// First, scan through all the arguments to figure out the structure of the pipeline.  Count the number of distinct processes (it's the number of "|" arguments).
	// If there are "<", "<<", or ">" argumentsthen make note of input and output redirection and remove these arguments and the arguments that follow them.
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
			input = args[i] + 1;
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
				input = args[i + 1];
				removecount++;
			}
		} else if (args[i][0] == '>') {
			output = args[i] + 1;
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
				output = args[i + 1];
				removecount++;
			}
		} else if (args[i][0] == '2' && args[i][1] == '>') {
			error = args[i] + 2;
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
				error = args[i + 1];
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
	int inputId = -1; // Readable file id input to current command in pipeline (could be file or pipe).  -1 means use stdin.
	if (input != NULL) {
		if (inputFile == 0) {
			// Immediate data in command.  Create temporary file and put data into file.
#define TMP_STDIN_NAME "/tmp/tcl.in.XXXXXX"
			char inName[sizeof(TMP_STDIN_NAME) + 1];
			strcpy(inName, TMP_STDIN_NAME);
#ifdef HAVE_MKSTEMP
			inputId = mkstemp(inName);
#else
			mktemp(inName);
			inputId = open(inName, O_RDWR|O_CREAT|O_TRUNC, 0600);
#endif
			if (inputId < 0) {
				Tcl_AppendResult(interp, "couldn't create input file for command: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			int length = strlen(input);
			if (write(inputId, input, length) != length) {
				Tcl_AppendResult(interp, "couldn't write file input for command: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			if (lseek(inputId, 0L, 0) == -1 || unlink(inName) == -1) {
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
			inputId = dup(fileno(filePtr->f2 ? : filePtr->f));
		} else {
			// File redirection.  Just open the file.
			inputId = open(input, O_RDONLY, 0);
			if (inputId < 0) {
				Tcl_AppendResult(interp, "couldn't read file \"", input, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (inPipePtr != NULL) {
		if (pipe(pipeIds) != 0) {
			Tcl_AppendResult(interp, "couldn't create input pipe for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		inputId = pipeIds[0];
		*inPipePtr = pipeIds[1];
		pipeIds[0] = pipeIds[1] = -1;
	}

	// Set up the redirected output sink for the pipeline from one of two places, if requested.
	int lastOutputId = -1;	// Write file id for output from last command in pipeline (could be file or pipe). -1 means use stdout.
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
			fflush(filePtr->f2 ?: filePtr->f);
			lastOutputId = dup(fileno(filePtr->f2 ?: filePtr->f));
		}
		else {
			// Output is to go to a file.
			int mode = O_WRONLY|O_CREAT|O_TRUNC;
			if (outputFile == 1) {
				mode = O_WRONLY|O_CREAT|O_APPEND;
			}
			lastOutputId = open(output, mode, 0666);
			if (lastOutputId < 0) {
				Tcl_AppendResult(interp, "couldn't write file \"", output, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (outPipePtr != NULL) {
		// Output is to go to a pipe.
		if (pipe(pipeIds) != 0) {
			Tcl_AppendResult(interp, "couldn't create output pipe: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		lastOutputId = pipeIds[1];
		*outPipePtr = pipeIds[0];
		pipeIds[0] = pipeIds[1] = -1;
	}
	// If we are redirecting stderr with 2>filename or 2>@fileId, then we ignore errFilePtr
	int errorId = -1; // Writable file id for all standard error output from all commands in pipeline.  -1 means use stderr.
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
			fflush(filePtr->f2 ?: filePtr->f);
			errorId = dup(fileno(filePtr->f2 ?: filePtr->f));
		}
		else {
			// Output is to go to a file.
			int mode = O_WRONLY|O_CREAT|O_TRUNC;
			if (errorFile == 1) {
				mode = O_WRONLY|O_CREAT|O_APPEND;
			}
			errorId = open(error, mode, 0666);
			if (errorId < 0) {
				Tcl_AppendResult(interp, "couldn't write file \"", error, "\": ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
		}
	} else if (errFilePtr != NULL) {
		// Set up the standard error output sink for the pipeline, if requested.  Use a temporary file which is opened, then deleted.
		// Could potentially just use pipe, but if it filled up it could cause the pipeline to deadlock:  we'd be waiting for processes
		// to complete before reading stderr, and processes couldn't complete because stderr was backed up.
#define TMP_STDERR_NAME "/tmp/tcl.err.XXXXXX"
		char errName[sizeof(TMP_STDERR_NAME) + 1];
		strcpy(errName, TMP_STDERR_NAME);
#ifdef HAVE_MKSTEMP
		errorId = mkstemp(errName);
#else
		mktemp(errName);
		errorId = open(errName, O_WRONLY|O_CREAT|O_TRUNC, 0600);
#endif
		if (errorId < 0) {
errFileError:
			Tcl_AppendResult(interp, "couldn't create error file for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		*errFilePtr = open(errName, O_RDONLY, 0);
		if (*errFilePtr < 0) {
			goto errFileError;
		}
		if (unlink(errName) == -1) {
			Tcl_AppendResult(interp, "couldn't remove error file for command: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
	}

	// Scan through the argc array, forking off a process for each group of arguments between "|" arguments.
	int *pidPtr = (int *)_allocFast((unsigned)(cmdCount * sizeof(int))); // Points to malloc-ed array holding all the pids of child processes.
	for (i = 0; i < numPids; i++) {
		pidPtr[i] = -1;
	}
	int outputId = -1; // Writable file id for output from current command in pipeline (could be file or pipe). -1 means use stdout.
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
			if (pipe(pipeIds) != 0) {
				Tcl_AppendResult(interp, "couldn't create pipe: ", Tcl_OSError(interp), (char *)NULL);
				goto error;
			}
			outputId = pipeIds[1];
		}
		char *execName = Tcl_TildeSubst(interp, args[firstArg]);
		int pid = Tcl_Fork();
		if (pid == -1) {
			Tcl_AppendResult(interp, "couldn't fork child process: ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
		if (pid == 0) {
			char errSpace[200];
			if ((inputId != -1 && dup2(inputId, 0) == -1) || (outputId != -1 && dup2(outputId, 1) == -1) || (errorId != -1 && dup2(errorId, 2) == -1)) {
				char *err = "forked process couldn't set up input/output\n";
				write(errorId < 0 ? 2 : errorId, err, strlen(err));
				_exit(1);
			}
			for (i = 3; i <= outputId || i <= inputId || i <= errorId; i++) {
				close(i);
			}
			execvp(execName, &args[firstArg]);
			sprintf(errSpace, "couldn't find \"%.150s\" to execute\n", args[firstArg]);
			write(2, errSpace, strlen(errSpace));
			_exit(1);
		} else {
			pidPtr[numPids] = pid;
		}
		// Close off our copies of file descriptors that were set up for this child, then set up the input for the next child.
		if (inputId != -1) {
			close(inputId);
		}
		if (outputId != -1) {
			close(outputId);
		}
		inputId = pipeIds[0];
		pipeIds[0] = pipeIds[1] = -1;
	}
	*pidArrayPtr = pidPtr;

	// All done.  Cleanup open files lying around and then return.
cleanup:
	if (inputId != -1) {
		close(inputId);
	}
	if (lastOutputId != -1) {
		close(lastOutputId);
	}
	if (errorId != -1) {
		close(errorId);
	}
	return numPids;

	// An error occurred.  There could have been extra files open, such as pipes between children.  Clean them all up.  Detach any child processes that have been created.
error:
	if (inPipePtr != NULL && *inPipePtr != -1) {
		close(*inPipePtr);
		*inPipePtr = -1;
	}
	if (outPipePtr != NULL && *outPipePtr != -1) {
		close(*outPipePtr);
		*outPipePtr = -1;
	}
	if (errFilePtr != NULL && *errFilePtr != -1) {
		close(*errFilePtr);
		*errFilePtr = -1;
	}
	if (pipeIds[0] != -1) {
		close(pipeIds[0]);
	}
	if (pipeIds[1] != -1) {
		close(pipeIds[1]);
	}
	if (pidPtr != NULL) {
		for (i = 0; i < numPids; i++) {
			if (pidPtr[i] != -1) {
				Tcl_DetachPids(1, &pidPtr[i]);
			}
		}
		_freeFast((char *)pidPtr);
	}
	numPids = -1;
	goto cleanup;
}
#endif /* NO_FORK */

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
	char *msg = strerror(errno);
	Tcl_SetErrorCode(interp, "UNIX", id, msg, (char *)NULL);
	return msg;
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
		long dummy;
		if (index < 2) {
			iPtr->numFiles = 3;
		} else {
			iPtr->numFiles = index+1;
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() allocating table of size %d", iPtr->numFiles);
#endif
		iPtr->filePtrArray = (OpenFile_ **)_allocFast((unsigned)(iPtr->numFiles*sizeof(OpenFile_ *)));
		for (i = iPtr->numFiles-1; i >= 0; i--) {
			iPtr->filePtrArray[i] = NULL;
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() stdin fd=%d, stdout fd=%d, stderr fd=%d", fileno(stdin), fileno(stdout), fileno(stderr));
#endif
		if (fcntl(fileno(stdout), F_GETFL, &dummy) == -1 && errno == EBADF) {
#ifdef DEBUG_FDS
			syslog(LOG_INFO, "TclMakeFileTable() reopening stdout");
#endif
			freopen("/dev/null", "w", stdout);
		}
		if (fcntl(fileno(stdin), F_GETFL, &dummy) == -1 && errno == EBADF) {
#ifdef DEBUG_FDS
			syslog(LOG_INFO, "TclMakeFileTable() reopening stdin");
#endif
			freopen("/dev/null", "r", stdin);
		}
		if (fcntl(fileno(stderr), F_GETFL, &dummy) == -1 && errno == EBADF) {
#ifdef DEBUG_FDS
			syslog(LOG_INFO, "TclMakeFileTable() reopening stderr");
#endif
			freopen("/dev/console", "w", stderr);
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() after freopen(): stdin fd=%d, stdout fd=%d, stderr fd=%d", fileno(stdin), fileno(stdout), fileno(stderr));
#endif
		OpenFile_ *filePtr;
		if (fileno(stdin) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stdin;
			filePtr->f2 = NULL;
			filePtr->readable = 1;
			filePtr->writable = 0;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = -1;
			iPtr->filePtrArray[fileno(stdin)] = filePtr;
		}
		if (fileno(stdout) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stdout;
			filePtr->f2 = NULL;
			filePtr->readable = 0;
			filePtr->writable = 1;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = -1;
			iPtr->filePtrArray[fileno(stdout)] = filePtr;
		}
		if (fileno(stderr) >= 0) {
			filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
			filePtr->f = stderr;
			filePtr->f2 = NULL;
			filePtr->readable = 0;
			filePtr->writable = 1;
			filePtr->numPids = 0;
			filePtr->pidPtr = NULL;
			filePtr->errorId = -1;
			iPtr->filePtrArray[fileno(stderr)] = filePtr;
		}
	} else if (index >= iPtr->numFiles) {
		int newSize = index+1;
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "TclMakeFileTable() increasing size from %d to %d", iPtr->numFiles, newSize);
#endif
		OpenFile_ **newPtrArray = (OpenFile_ **)_allocFast((unsigned)(newSize*sizeof(OpenFile_ *)));
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
			fd = fileno(stdin);
		} else if (!strcmp(string+3, "out") ) {
			fd = fileno(stdout);
		} else if (!strcmp(string+3, "err")) {
			fd = fileno(stderr);
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