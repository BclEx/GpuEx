#include "Tcl+Int.h"
#if 0 && OS_GPU
#include "Tcl+Gpu.h"

// The variable below caches the name of the current working directory in order to avoid repeated calls to getwd.  The string is malloc-ed. NULL means the cache needs to be refreshed.
__device__ static char *currentDir = NULL;

// Prototypes for local procedures defined in this file:
__device__ static int CleanupChildren(Tcl_Interp *interp, int numPids, int *pidPtr, FILE *errorId);
__device__ static char *GetFileType(int mode);
__device__ static int StoreStatData(Tcl_Interp *interp, char *varName, struct stat *statPtr);

/*
*----------------------------------------------------------------------
*
* Tcl_CdCmd --
*	This procedure is invoked to process the "cd" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_CdCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc > 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " dirName\"", (char *)NULL);
		return TCL_ERROR;
	}
	char *dirName;
	if (argc == 2) {
		dirName = argv[1];
	} else {
		dirName = "~";
	}
	dirName = Tcl_TildeSubst(interp, dirName);
	if (dirName == NULL) {
		return TCL_ERROR;
	}
	if (currentDir != NULL) {
		_freeFast(currentDir);
		currentDir = NULL;
	}
	if (_chdir(dirName) != 0) {
		Tcl_AppendResult(interp, "couldn't change working directory to \"", dirName, "\": ", Tcl_OSError(interp), (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_CloseCmd --
*	This procedure is invoked to process the "close" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_CloseCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	((Interp *)interp)->filePtrArray[fileno(filePtr->f)] = NULL;
	// First close the file (in the case of a process pipeline, there may be two files, one for the pipe at each end of the pipeline).
	int result = TCL_OK;
	if (filePtr->f2 != NULL) {
		if (_fclose(filePtr->f2)) {
			Tcl_AppendResult(interp, "error closing \"", argv[1], "\": ", Tcl_OSError(interp), "\n", (char *)NULL);
			result = TCL_ERROR;
		}
	}
	if (_fclose(filePtr->f)) {
		Tcl_AppendResult(interp, "error closing \"", argv[1], "\": ", Tcl_OSError(interp), "\n", (char *)NULL);
		result = TCL_ERROR;
	}
	// If the file was a connection to a pipeline, clean up everything associated with the child processes.
	if (filePtr->numPids > 0) {
		if (CleanupChildren(interp, filePtr->numPids, filePtr->pidPtr, filePtr->errorId) != TCL_OK) {
			result = TCL_ERROR;
		}
	}
	_freeFast((char *)filePtr);
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_EofCmd --
*	This procedure is invoked to process the "eof" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_EofCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	if (feof(filePtr->f)) {
		interp->result = "1";
	} else {
		interp->result = "0";
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ExecCmd --
*	This procedure is invoked to process the "exec" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ExecCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	int *pidPtr;
	int numPids;
	// See if the command is to be run in background;  if so, create the command, detach it, and return.
	if (argv[argc-1][0] == '&' && argv[argc-1][1] == 0) {
		argc--;
		argv[argc] = NULL;
		numPids = Tcl_CreatePipeline(interp, argc-1, argv+1, &pidPtr, (FILE **)NULL, (FILE **)NULL, (FILE **)NULL);
		if (numPids < 0) {
			return TCL_ERROR;
		}
		Tcl_DetachPids(numPids, pidPtr);
		_freeFast((char *)pidPtr);
		return TCL_OK;
	}

	// Create the command's pipeline.
	FILE *outputId; // File id for output pipe.  -1 means command overrode.
	FILE *errorId; // File id for temporary file containing error output.
	numPids = Tcl_CreatePipeline(interp, argc-1, argv+1, &pidPtr, (FILE **)NULL, &outputId, &errorId);
	if (numPids < 0) {
		return TCL_ERROR;
	}

	// Read the child's output (if any) and put it into the result.
	int result = TCL_OK;
	if (outputId) {
		while (true) {
#define BUFFER_SIZE 1000
			char buffer[BUFFER_SIZE+1];
			int count = fread(buffer, BUFFER_SIZE, 1, outputId);
			if (count == 0) {
				break;
			}
			if (count < 0) {
				Tcl_ResetResult(interp);
				Tcl_AppendResult(interp, "error reading from output pipe: ", Tcl_OSError(interp), (char *)NULL);
				result = TCL_ERROR;
				break;
			}
			buffer[count] = 0;
			Tcl_AppendResult(interp, buffer, (char *)NULL);
		}
		_fclose(outputId);
	}

	if (CleanupChildren(interp, numPids, pidPtr, errorId) != TCL_OK) {
		result = TCL_ERROR;
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ExitCmd --
*	This procedure is invoked to process the "exit" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ExitCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 1 && argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " ?returnCode?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (argc == 1) {
		exit(0);
	}
	int value;
	if (Tcl_GetInt(interp, argv[1], &value) != TCL_OK) {
		return TCL_ERROR;
	}
	exit(value);
	return TCL_OK; // Better not ever reach this!
}

/*
*----------------------------------------------------------------------
*
* Tcl_FileCmd --
*	This procedure is invoked to process the "file" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_FileCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	char *p;
	if (argc < 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " option name ?arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	char c = argv[1][0];
	int length = _strlen(argv[1]);

	// First handle operations on the file name.
	char *fileName = Tcl_TildeSubst(interp, argv[2]);
	if (fileName == NULL) {
		return TCL_ERROR;
	}
	if (c == 'd' && !_strncmp(argv[1], "dirname", length)) {
		if (argc != 3) {
			argv[1] = "dirname";
not3Args:
			Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " ", argv[1], " name\"", (char *)NULL);
			return TCL_ERROR;
		}
		p = _strrchr(fileName, '/');
		if (p == NULL) {
			interp->result = ".";
		} else if (p == fileName) {
			interp->result = "/";
		} else {
			*p = 0;
			Tcl_SetResult(interp, fileName, TCL_VOLATILE);
			*p = '/';
		}
		return TCL_OK;
	} else if (c == 'r' && !_strncmp(argv[1], "rootname", length) && length >= 2) {
		if (argc != 3) {
			argv[1] = "rootname";
			goto not3Args;
		}
		p = _strrchr(fileName, '.');
		char *lastSlash = _strrchr(fileName, '/');
		if (p == NULL || (lastSlash != NULL && lastSlash > p)) {
			Tcl_SetResult(interp, fileName, TCL_VOLATILE);
		} else {
			*p = 0;
			Tcl_SetResult(interp, fileName, TCL_VOLATILE);
			*p = '.';
		}
		return TCL_OK;
	} else if (c == 'e' && !_strncmp(argv[1], "extension", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "extension";
			goto not3Args;
		}
		p = _strrchr(fileName, '.');
		char *lastSlash = _strrchr(fileName, '/');
		if (p != NULL && (lastSlash == NULL || lastSlash < p)) {
			Tcl_SetResult(interp, p, TCL_VOLATILE);
		}
		return TCL_OK;
	} else if (c == 't' && !_strncmp(argv[1], "tail", length) && length >= 2) {
		if (argc != 3) {
			argv[1] = "tail";
			goto not3Args;
		}
		p = _strrchr(fileName, '/');
		if (p != NULL) {
			Tcl_SetResult(interp, p+1, TCL_VOLATILE);
		} else {
			Tcl_SetResult(interp, fileName, TCL_VOLATILE);
		}
		return TCL_OK;
	}

	// Next, handle operations that can be satisfied with the "access" kernel call.
	if (fileName == NULL) {
		return TCL_ERROR;
	}
	int mode = 0; // Initialized only to prevent compiler warning message.
	if (c == 'r' && !_strncmp(argv[1], "readable", length) && length >= 5) {
		if (argc != 3) {
			argv[1] = "readable";
			goto not3Args;
		}
		mode = R_OK;
checkAccess:
		if (_access(fileName, mode) == -1) {
			interp->result = "0";
		} else {
			interp->result = "1";
		}
		return TCL_OK;
	} else if (c == 'w' && !_strncmp(argv[1], "writable", length)) {
		if (argc != 3) {
			argv[1] = "writable";
			goto not3Args;
		}
		mode = W_OK;
		goto checkAccess;
	} else if (c == 'e' && !_strncmp(argv[1], "executable", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "executable";
			goto not3Args;
		}
		mode = X_OK;
		goto checkAccess;
	} else if (c == 'e' && !_strncmp(argv[1], "exists", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "exists";
			goto not3Args;
		}
		mode = F_OK;
		goto checkAccess;
	}

	// Next, handle operations on the file
	if (c == 'd' && !_strncmp(argv[1], "delete", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "delete";
			goto not3Args;
		}
		if (_unlink(fileName) == -1 && __errno != ENOENT) {
			Tcl_AppendResult(interp, "couldn't delete \"", argv[2], "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		return TCL_OK;
	}
	else if (!_strcmp(argv[1], "rename")) {
		if (argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " ", argv[1], " source target\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (!_rename(argv[2], argv[3])) {
			Tcl_AppendResult(interp, "couldn't rename \"", argv[2], "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		return TCL_OK;
	}

	// Lastly, check stuff that requires the file to be stat-ed.
	int statOp;
	struct stat statBuf;
	if (c == 'a' && !_strncmp(argv[1], "atime", length)) {
		if (argc != 3) {
			argv[1] = "atime";
			goto not3Args;
		}
		if (__stat(fileName, &statBuf) == -1) {
			goto badStat;
		}
		_sprintf(interp->result, "%ld", statBuf.st_atime);
		return TCL_OK;
	} else if (c == 'i' && !_strncmp(argv[1], "isdirectory", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "isdirectory";
			goto not3Args;
		}
		statOp = 2;
	} else if (c == 'i' && !_strncmp(argv[1], "isfile", length) && length >= 3) {
		if (argc != 3) {
			argv[1] = "isfile";
			goto not3Args;
		}
		statOp = 1;
	} else if (c == 'l' && !_strncmp(argv[1], "lstat", length)) {
		if (argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " lstat name varName\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (__stat(fileName, &statBuf) == -1) {
			Tcl_AppendResult(interp, "couldn't lstat \"", argv[2], "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		return StoreStatData(interp, argv[3], &statBuf);
	} else if (c == 'm' && !_strncmp(argv[1], "mtime", length)) {
		if (argc != 3) {
			argv[1] = "mtime";
			goto not3Args;
		}
		if (__stat(fileName, &statBuf) == -1) {
			goto badStat;
		}
		_sprintf(interp->result, "%ld", statBuf.st_mtime);
		return TCL_OK;
	} else if (c == 'o' && !_strncmp(argv[1], "owned", length)) {
		if (argc != 3) {
			argv[1] = "owned";
			goto not3Args;
		}
		statOp = 0;
#ifdef S_IFLNK
		// This option is only included if symbolic links exist on this system (in which case S_IFLNK should be defined).
	} else if (c == 'r' && !_strncmp(argv[1], "readlink", length) && length >= 5) {
		if (argc != 3) {
			argv[1] = "readlink";
			goto not3Args;
		}
		char linkValue[MAXPATHLEN+1];
		int linkLength = readlink(fileName, linkValue, sizeof(linkValue) - 1);
		if (linkLength == -1) {
			Tcl_AppendResult(interp, "couldn't readlink \"", argv[2], "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		linkValue[linkLength] = 0;
		Tcl_SetResult(interp, linkValue, TCL_VOLATILE);
		return TCL_OK;
#endif
	} else if (c == 's' && !_strncmp(argv[1], "size", length) && length >= 2) {
		if (argc != 3) {
			argv[1] = "size";
			goto not3Args;
		}
		if (__stat(fileName, &statBuf) == -1) {
			goto badStat;
		}
		_sprintf(interp->result, "%ld", statBuf.st_size);
		return TCL_OK;
	} else if (c == 's' && !_strncmp(argv[1], "stat", length) && length >= 2) {
		if (argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " stat name varName\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (__stat(fileName, &statBuf) == -1) {
badStat:
			Tcl_AppendResult(interp, "couldn't stat \"", argv[2], "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		return StoreStatData(interp, argv[3], &statBuf);
	} else if (c == 't' && !_strncmp(argv[1], "type", length) && length >= 2) {
		if (argc != 3) {
			argv[1] = "type";
			goto not3Args;
		}
		if (stat(fileName, &statBuf) == -1) {
			goto badStat;
		}
		interp->result = GetFileType((int)statBuf.st_mode);
		return TCL_OK;
	} else {
		Tcl_AppendResult(interp, "bad option \"", argv[1], "\": should be atime, dirname, executable, exists, ", "extension, isdirectory, isfile, lstat, mtime, owned, ", "readable, ",
#ifdef S_IFLNK
			"readlink, ",
#endif
			"root, size, stat, tail, type, ", "or writable", (char *)NULL);
		return TCL_ERROR;
	}
	if (__stat(fileName, &statBuf) == -1) {
		interp->result = "0";
		return TCL_OK;
	}
	switch (statOp) {
	case 0:
		//	mode = (geteuid() == statBuf.st_uid);
		break;
	case 1:
		mode = S_ISREG(statBuf.st_mode);
		break;
	case 2:
		mode = S_ISDIR(statBuf.st_mode);
		break;
	}
	if (mode) {
		interp->result = "1";
	} else {
		interp->result = "0";
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* StoreStatData --
*	This is a utility procedure that breaks out the fields of a "stat" structure and stores them in textual form into the elements of an associative array.
*
* Results:
*	Returns a standard Tcl return value.  If an error occurs then a message is left in interp->result.
*
* Side effects:
*	Elements of the associative array given by "varName" are modified.
*
*----------------------------------------------------------------------
*/
__device__ static int StoreStatData(Tcl_Interp *interp, char *varName, struct stat *statPtr)
{
	char string[30];
	_sprintf(string, "%d", (int)statPtr->st_dev);
	if (Tcl_SetVar2(interp, varName, "dev", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%d", (int)statPtr->st_ino);
	if (Tcl_SetVar2(interp, varName, "ino", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%d", statPtr->st_mode);
	if (Tcl_SetVar2(interp, varName, "mode", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%d", statPtr->st_nlink);
	if (Tcl_SetVar2(interp, varName, "nlink", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%d", statPtr->st_uid);
	if (Tcl_SetVar2(interp, varName, "uid", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%d", statPtr->st_gid);
	if (Tcl_SetVar2(interp, varName, "gid", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%ld", statPtr->st_size);
	if (Tcl_SetVar2(interp, varName, "size", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%ld", statPtr->st_atime);
	if (Tcl_SetVar2(interp, varName, "atime", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%ld", statPtr->st_mtime);
	if (Tcl_SetVar2(interp, varName, "mtime", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	_sprintf(string, "%ld", statPtr->st_ctime);
	if (Tcl_SetVar2(interp, varName, "ctime", string, TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	if (Tcl_SetVar2(interp, varName, "type", GetFileType((int)statPtr->st_mode), TCL_LEAVE_ERR_MSG) == NULL) {
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* GetFileType --
*	Given a mode word, returns a string identifying the type of a file.
*
* Results:
*	A static text string giving the file type from mode.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ static char *GetFileType(int mode)
{
	if (S_ISREG(mode)) return "file";
	else if (S_ISDIR(mode)) return "directory";
	//else if (S_ISCHR(mode)) return "characterSpecial";
	//else if (S_ISBLK(mode)) return "blockSpecial";
	//else if (S_ISFIFO(mode)) return "fifo";
	//else if (S_ISLNK(mode)) return "link";
	//else if (S_ISSOCK(mode)) return "socket";
	return "unknown";
}

/*
*----------------------------------------------------------------------
*
* Tcl_FlushCmd --
*	This procedure is invoked to process the "flush" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_FlushCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	if (!filePtr->writable) {
		Tcl_AppendResult(interp, "\"", argv[1], "\" wasn't opened for writing", (char *)NULL);
		return TCL_ERROR;
	}
	FILE *f = filePtr->f2;
	if (f == NULL) {
		f = filePtr->f;
	}
	if (_fflush(f) == EOF) {
		Tcl_AppendResult(interp, "error flushing \"", argv[1], "\": ", Tcl_OSError(interp), (char *)NULL);
		clearerr(f);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetsCmd --
*	This procedure is invoked to process the "gets" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GetsCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
#define BUF_SIZE 200
	if (argc != 2 && argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId ?varName?\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	if (!filePtr->readable) {
		Tcl_AppendResult(interp, "\"", argv[1], "\" wasn't opened for reading", (char *)NULL);
		return TCL_ERROR;
	}

	// We can't predict how large a line will be, so read it in pieces, appending to the current result or to a variable.
	int totalCount = 0;
	bool done = false;
	int flags = 0;
	register FILE *f = filePtr->f;
	while (!done) {
		char buffer[BUF_SIZE+1];
		register int c, count;
		register char *p;
		for (p = buffer, count = 0; count < BUF_SIZE-1; count++, p++) {
			c = _getc(f);
			if (c == EOF) {
				if (_ferror(filePtr->f)) {
					Tcl_ResetResult(interp);
					Tcl_AppendResult(interp, "error reading \"", argv[1], "\": ", Tcl_OSError(interp), (char *)NULL);
					clearerr(filePtr->f);
					return TCL_ERROR;
				} else if (_feof(filePtr->f)) {
					if (totalCount == 0 && count == 0) {
						totalCount = -1;
					}
					done = 1;
					break;
				}
			}
			if (c == '\n') {
				done = 1;
				break;
			}
			*p = c;
		}
		*p = 0;
		if (argc == 2) {
			Tcl_AppendResult(interp, buffer, (char *)NULL);
		} else {
			if (Tcl_SetVar(interp, argv[2], buffer, flags|TCL_LEAVE_ERR_MSG) == NULL) {
				return TCL_ERROR;
			}
			flags = TCL_APPEND_VALUE;
		}
		totalCount += count;
	}
	if (argc == 3) {
		_sprintf(interp->result, "%d", totalCount);
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_OpenCmd --
*	This procedure is invoked to process the "open" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_OpenCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
	Interp *iPtr = (Interp *)interp;
	char *access;
	if (argc == 2) {
		access = "r";
	} else if (argc == 3) {
		access = argv[2];
	} else {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " filename ?access?\"", (char *)NULL);
		return TCL_ERROR;
	}

	register OpenFile_ *filePtr = (OpenFile_ *)_allocFast(sizeof(OpenFile_));
	filePtr->f = NULL;
	filePtr->f2 = NULL;
	filePtr->readable = 0;
	filePtr->writable = 0;
	filePtr->numPids = 0;
	filePtr->pidPtr = NULL;
	filePtr->errorId = NULL;

	// Verify the requested form of access.
	int pipeline = 0;
	if (argv[1][0] == '|') {
#ifndef NO_FORK
		pipeline = 1;
#else
		Tcl_AppendResult(interp, "open with pipeline not supported in this version of Tcl", (char *)NULL);
		return TCL_ERROR;
#endif
	}
	switch (access[0]) {
	case 'r':
		filePtr->readable = 1;
		break;
	case 'w':
		filePtr->writable = 1;
		break;
	case 'a':
		filePtr->writable = 1;
		break;
	default:
badAccess:
		Tcl_AppendResult(interp, "illegal access mode \"", access, "\"", (char *)NULL);
		goto error;
	}
	if (access[1] == '+') {
		filePtr->readable = filePtr->writable = 1;
		if (access[2] != 0) {
			goto badAccess;
		}
	} else if (access[1] != 0) {
		goto badAccess;
	}

	// Before we open any files, make sure the file table is allocated so that stdin, etc. are sorted out
	TclMakeFileTable(iPtr, 0);

	// Open the file or create a process pipeline.
	if (!pipeline) {
		char *fileName = argv[1];
		if (fileName[0] == '~') {
			fileName = Tcl_TildeSubst(interp, fileName);
			if (fileName == NULL) {
				goto error;
			}
		}
		filePtr->f = fopen(fileName, access);
		if (filePtr->f == NULL) {
			Tcl_AppendResult(interp, "couldn't open \"", argv[1], "\": ", Tcl_OSError(interp), (char *)NULL);
			goto error;
		}
#ifdef DEBUG_FDS
		syslog(LOG_INFO, "Opened %s to give fd %d", fileName, fileno(filePtr->f));
#endif
	}
	else {
		int cmdArgc;
		char **cmdArgv;
		if (Tcl_SplitList(interp, argv[1]+1, &cmdArgc, &cmdArgv) != TCL_OK) {
			goto error;
		}
		int *inPipePtr = (filePtr->writable ? &inPipe : NULL);
		int *outPipePtr = (filePtr->readable ? &outPipe : NULL);
		int inPipe = -1, int outPipe = -1;
		filePtr->numPids = Tcl_CreatePipeline(interp, cmdArgc, cmdArgv, &filePtr->pidPtr, inPipePtr, outPipePtr, &filePtr->errorId);
		_freeFast((char *)cmdArgv);
		if (filePtr->numPids < 0) {
			goto error;
		}
		//if (filePtr->readable) {
		//	if (outPipe == -1) {
		//		if (inPipe != -1) {
		//			_close(inPipe);
		//		}
		//		Tcl_AppendResult(interp, "can't read output from command:", " standard output was redirected", (char *)NULL);
		//		goto error;
		//	}
		//	filePtr->f = _fdopen(outPipe, "r");
		//}
		//if (filePtr->writable) {
		//	if (inPipe == -1) {
		//		Tcl_AppendResult(interp, "can't write input to command:", " standard input was redirected", (char *)NULL);
		//		goto error;
		//	}
		//	if (filePtr->f != NULL) {
		//		filePtr->f2 = _fdopen(inPipe, "w");
		//	} else {
		//		filePtr->f = _fdopen(inPipe, "w");
		//	}
		//}
	}

	// Enter this new OpenFile_ structure in the table for the interpreter.  May have to expand the table to do this.
	int fd = _fileno(filePtr->f);
	TclMakeFileTable(iPtr, fd);
	if (iPtr->filePtrArray[fd] != NULL) {
		_panic("Tcl_OpenCmd found file already open");
	}
	iPtr->filePtrArray[fd] = filePtr;
	_sprintf(interp->result, "file%d", fd);
	return TCL_OK;

error:
	if (filePtr->f != NULL) {
		_fclose(filePtr->f);
	}
	if (filePtr->f2 != NULL) {
		_fclose(filePtr->f2);
	}
#ifndef NO_FORK
	if (filePtr->numPids > 0) {
		Tcl_DetachPids(filePtr->numPids, filePtr->pidPtr);
		_freeFast((char *)filePtr->pidPtr);
	}
#endif
	if (filePtr->errorId) {
		fclose(filePtr->errorId);
	}
	_freeFast((char *)filePtr);
	return TCL_ERROR;
}

/*
*----------------------------------------------------------------------
*
* Tcl_PwdCmd --
*	This procedure is invoked to process the "pwd" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
#define MAXPATHLEN 1024
__device__ int Tcl_PwdCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (currentDir == NULL) {
		char buffer[MAXPATHLEN+1];
		if (__getcwd(buffer, MAXPATHLEN) == NULL) {
			Tcl_AppendResult(interp, "error getting working directory name: ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		currentDir = (char *)_allocFast((unsigned)(strlen(buffer) + 1));
		_strcpy(currentDir, buffer);
	}
	interp->result = currentDir;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_PutsCmd --
*	This procedure is invoked to process the "puts" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_PutsCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	int i = 1;
	bool newline = true;
	if (argc >= 2 && !_strcmp(argv[1], "-nonewline")) {
		newline = false;
		i++;
	}
	if (i < (argc-3) || i >= argc) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], "\" ?-nonewline? ?fileId? string", (char *)NULL);
		return TCL_ERROR;
	}

	// The code below provides backwards compatibility with an old form of the command that is no longer recommended or documented.
	if (i == (argc-3)) {
		if (_strncmp(argv[i+2], "nonewline", _strlen(argv[i+2]))) {
			Tcl_AppendResult(interp, "bad argument \"", argv[i+2], "\": should be \"nonewline\"", (char *)NULL);
			return TCL_ERROR;
		}
		newline = 0;
	}
	char *fileId;
	if (i == (argc-1)) {
		fileId = "stdout";
	} else {
		fileId = argv[i];
		i++;
	}

	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, fileId, &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	if (!filePtr->writable) {
		Tcl_AppendResult(interp, "\"", fileId, "\" wasn't opened for writing", (char *)NULL);
		return TCL_ERROR;
	}
	FILE *f = filePtr->f2;
	if (f == NULL) {
		f = filePtr->f;
	}

	fputs(argv[i], f);
	if (newline) {
		fputc('\n', f);
	}
	if (ferror(f)) {
		Tcl_AppendResult(interp, "error writing \"", fileId, "\": ", Tcl_OSError(interp), (char *)NULL);
		clearerr(f);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ReadCmd --
*	This procedure is invoked to process the "read" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ReadCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
#define READ_BUF_SIZE 4096
	if (argc != 2 && argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId ?numBytes?\" or \"", argv[0], " ?-nonewline? fileId\"", (char *)NULL);
		return TCL_ERROR;
	}
	int i = 1;
	bool newline = true;
	if (argc == 3 && !_strcmp(argv[1], "-nonewline")) {
		newline = false;
		i++;
	}

	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[i], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	if (!filePtr->readable) {
		Tcl_AppendResult(interp, "\"", argv[i], "\" wasn't opened for reading", (char *)NULL);
		return TCL_ERROR;
	}

	// Compute how many bytes to read, and see whether the final newline should be dropped.
	int bytesLeft;
	if (argc >= (i + 2) && _isdigit(argv[i+1][0])) {
		if (Tcl_GetInt(interp, argv[i+1], &bytesLeft) != TCL_OK) {
			return TCL_ERROR;
		}
	} else {
		bytesLeft = 1<<30;

		// The code below provides backward compatibility for an archaic earlier version of this command.
		if (argc >= (i + 2)) {
			if (!_strncmp(argv[i+1], "nonewline", _strlen(argv[i+1]))) {
				newline = false;
			} else {
				Tcl_AppendResult(interp, "bad argument \"", argv[i+1], "\": should be \"nonewline\"", (char *)NULL);
				return TCL_ERROR;
			}
		}
	}

	// Read the file in one or more chunks.
	int bytesRead = 0;
	while (bytesLeft > 0) {
		int count = READ_BUF_SIZE;
		if (bytesLeft < READ_BUF_SIZE) {
			count = bytesLeft;
		}
		char buffer[READ_BUF_SIZE+1];
		count = fread(buffer, 1, count, filePtr->f);
		if (ferror(filePtr->f)) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "error reading \"", argv[i], "\": ", Tcl_OSError(interp), (char *)NULL);
			clearerr(filePtr->f);
			return TCL_ERROR;
		}
		if (count == 0) {
			break;
		}
		buffer[count] = 0;
		Tcl_AppendResult(interp, buffer, (char *)NULL);
		bytesLeft -= count;
		bytesRead += count;
	}
	if (!newline && bytesRead > 0 && interp->result[bytesRead-1] == '\n') {
		interp->result[bytesRead-1] = 0;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_SeekCmd --
*	This procedure is invoked to process the "seek" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_SeekCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 3 && argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId offset ?origin?\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	int offset;
	if (Tcl_GetInt(interp, argv[2], &offset) != TCL_OK) {
		return TCL_ERROR;
	}
	int mode = SEEK_SET;
	if (argc == 4) {
		int length = _strlen(argv[3]);
		char c = argv[3][0];
		if (c == 's' && !_strncmp(argv[3], "start", length)) {
			mode = SEEK_SET;
		} else if (c == 'c' && !_strncmp(argv[3], "current", length)) {
			mode = SEEK_CUR;
		} else if (c == 'e' && !_strncmp(argv[3], "end", length)) {
			mode = SEEK_END;
		} else {
			Tcl_AppendResult(interp, "bad origin \"", argv[3], "\": should be start, current, or end", (char *)NULL);
			return TCL_ERROR;
		}
	}
	if (fseek(filePtr->f, (long)offset, mode) == -1) {
		Tcl_AppendResult(interp, "error during seek: ", Tcl_OSError(interp), (char *)NULL);
		clearerr(filePtr->f);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_SourceCmd --
*	This procedure is invoked to process the "source" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_SourceCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileName\"", (char *)NULL);
		return TCL_ERROR;
	}
	return Tcl_EvalFile(interp, argv[1]);
}

/*
*----------------------------------------------------------------------
*
* Tcl_TellCmd --
*	This procedure is invoked to process the "tell" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_TellCmd(ClientData notUsed, Tcl_Interp *interp, int argc, char **argv)
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " fileId\"", (char *)NULL);
		return TCL_ERROR;
	}
	OpenFile_ *filePtr;
	if (TclGetOpenFile(interp, argv[1], &filePtr) != TCL_OK) {
		return TCL_ERROR;
	}
	_sprintf(interp->result, "%ld", ftell(filePtr->f));
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_TimeCmd --
*	This procedure is invoked to process the "time" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_TimeCmd(ClientData dummy, Tcl_Interp *interp, int argc, char **argv)
{
	int count;
	if (argc == 2) {
		count = 1;
	} else if (argc == 3) {
		if (Tcl_GetInt(interp, argv[2], &count) != TCL_OK) {
			return TCL_ERROR;
		}
	} else {
		Tcl_AppendResult(interp, "wrong # args: should be \"", argv[0], " command ?count?\"", (char *)NULL);
		return TCL_ERROR;
	}
	double timePer;
	clock_t start = clock();
	for (int i = count; i > 0; i--) {
		int result = Tcl_Eval(interp, argv[1], 0, (char **)NULL);
		if (result != TCL_OK) {
			if (result == TCL_ERROR) {
				char msg[60];
				_sprintf(msg, "\n    (\"time\" body line %d)", interp->errorLine);
				Tcl_AddErrorInfo(interp, msg);
			}
			return result;
		}
	}
	clock_t stop = clock();
	timePer = (((double)(stop - start))*1000000.0)/CLOCKS_PER_SEC;
	Tcl_ResetResult(interp);
	_sprintf(interp->result, "%.0f microseconds per iteration", timePer/count);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* CleanupChildren --
*	This is a utility procedure used to wait for child processes to exit, record information about abnormal exits, and then
*	collect any stderr output generated by them.
*
* Results:
*	The return value is a standard Tcl result.  If anything at weird happened with the child processes, TCL_ERROR is returned
*	and a message is left in interp->result.
*
* Side effects:
*	If the last character of interp->result is a newline, then it is removed.  File errorId gets closed, and pidPtr is freed
*	back to the storage allocator.
*
*----------------------------------------------------------------------
*/
__device__ static int CleanupChildren(Tcl_Interp *interp, int numPids, int *pidPtr, FILE *errorId)
{
	//	int result = TCL_OK;
	//	int i, pid;
	//#define WAIT_STATUS_TYPE int
	//	WAIT_STATUS_TYPE waitStatus;
	//	for (i = 0; i < numPids; i++) {
	//		pid = Tcl_WaitPids(1, &pidPtr[i], (int *) &waitStatus);
	//		if (pid == -1) {
	//			// This can happen if the process was already reaped, so just ignore it
	//#if 0
	//			Tcl_AppendResult(interp, "error waiting for process to exit: ", Tcl_OSError(interp), (char *)NULL);
	//#endif
	//			continue;
	//		}
	//
	//		// Create error messages for unusual process exits.  An extra newline gets appended to each error message, but
	//		// it gets removed below (in the same fashion that an extra newline in the command's output is removed).
	//		if (!WIFEXITED(waitStatus) || WEXITSTATUS(waitStatus) != 0) {
	//			char msg1[20], msg2[20];
	//			result = TCL_ERROR;
	//			sprintf(msg1, "%d", pid);
	//			if (WIFEXITED(waitStatus)) {
	//				sprintf(msg2, "%d", WEXITSTATUS(waitStatus));
	//				Tcl_SetErrorCode(interp, "CHILDSTATUS", msg1, msg2, (char *)NULL);
	//			} else if (WIFSIGNALED(waitStatus)) {
	//				char *p;
	//				p = Tcl_SignalMsg((int) (WTERMSIG(waitStatus)));
	//				Tcl_SetErrorCode(interp, "CHILDKILLED", msg1, Tcl_SignalId((int)(WTERMSIG(waitStatus))), p, (char *)NULL);
	//				Tcl_AppendResult(interp, "child killed: ", p, "\n", (char *)NULL);
	//			} else if (WIFSTOPPED(waitStatus)) {
	//				char *p;
	//				p = Tcl_SignalMsg((int) (WSTOPSIG(waitStatus)));
	//				Tcl_SetErrorCode(interp, "CHILDSUSP", msg1, Tcl_SignalId((int)(WSTOPSIG(waitStatus))), p, (char *)NULL);
	//				Tcl_AppendResult(interp, "child suspended: ", p, "\n", (char *)NULL);
	//			} else {
	//				Tcl_AppendResult(interp, "child wait status didn't make sense\n", (char *)NULL);
	//			}
	//		}
	//	}
	//	_freeFast((char *)pidPtr);
	//
	//	// Read the standard error file.  If there's anything there, then return an error and add the file's contents to the result string.
	//	if (errorId >= 0) {
	//		while (true) {
	//#define BUFFER_SIZE 1000
	//			char buffer[BUFFER_SIZE+1];
	//			int count;
	//			count = read(errorId, buffer, BUFFER_SIZE);
	//			if (count == 0) {
	//				break;
	//			}
	//			if (count < 0) {
	//				Tcl_AppendResult(interp, "error reading stderr output file: ", Tcl_OSError(interp), (char *)NULL);
	//				break;
	//			}
	//			buffer[count] = 0;
	//			Tcl_AppendResult(interp, buffer, (char *)NULL);
	//		}
	//		fclose(errorId);
	//	}
	//
	//	// If the last character of interp->result is a newline, then remove the newline character (the newline would just confuse things).
	//	int length = strlen(interp->result);
	//	if (length > 0 && interp->result[length-1] == '\n') {
	//		interp->result[length-1] = '\0';
	//	}
	//	return result;
	return 0;
}

/*
*-----------------------------------------------------------------------------
*
* Tcl_PidCmd --
*     Implements the pid TCL command:
*         pid
*
* Results:
*      Standard TCL result.
*-----------------------------------------------------------------------------
*/
#define GetCurrentProcessId 1
__device__ int Tcl_PidCmd(ClientData clientData, Tcl_Interp *interp, int argc, char **argv)
{
	char buf[10];
	if (argc != 1) {
		Tcl_AppendResult (interp, "bad # args: ", argv[0], (char *)NULL);
		return TCL_ERROR;
	}
	_sprintf(buf, "%d", GetCurrentProcessId);
	Tcl_AppendResult(interp, buf, (char *)NULL);
	return TCL_OK;
}

#endif