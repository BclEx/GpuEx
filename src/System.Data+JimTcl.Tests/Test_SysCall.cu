// The code in this file implements a Tcl interface used to test error handling in the os_unix.c module. Wrapper functions that support fault
// injection are registered as the low-level OS functions using the xSetSystemCall() method of the VFS. The Tcl interface is as follows:
//
//   test_syscall install LIST
//     Install wrapper functions for all system calls in argument LIST. LIST must be a list consisting of zero or more of the following literal values:
//
//         open        close      access   getcwd   stat      fstat    
//         ftruncate   fcntl      read     pread    pread64   write
//         pwrite      pwrite64   fchmod   fallocate
//
//   test_syscall uninstall
//     Uninstall all wrapper functions.
//
//   test_syscall fault ?COUNT PERSIST?
//     If [test_syscall fault] is invoked without the two arguments, fault injection is disabled. Otherwise, fault injection is configured to
//     cause a failure on the COUNT'th next call to a system call with a wrapper function installed. A COUNT value of 1 means fail the next system call. 
// 
//     Argument PERSIST is interpreted as a boolean. If true, the all system calls following the initial failure also fail. Otherwise, only
//     the single transient failure is injected.
//
//   test_syscall errno CALL ERRNO
//     Set the value that the global "errno" is set to following a fault in call CALL. Argument CALL must be one of the system call names
//     listed above (under [test_syscall install]). ERRNO is a symbolic name (i.e. "EACCES"). Not all errno codes are supported. Add extra
//     to the aErrno table in function test_syscall_errno() below as required.
//
//   test_syscall reset ?SYSTEM-CALL?
//     With no argument, this is an alias for the [uninstall] command. However, this command uses a VFS call of the form:
//
//       xSetSystemCall(pVfs, 0, 0);
//
//     To restore the default system calls. The [uninstall] command restores each system call individually by calling (i.e.):
//
//       xSetSystemCall(pVfs, "open", 0);
//
//     With an argument, this command attempts to reset the system call named by the parameter using the same method as [uninstall].
//
//   test_syscall exists SYSTEM-CALL
//     Return true if the named system call exists. Or false otherwise.
//
//   test_syscall list
//     Return a list of all system calls. The list is constructed using the xNextSystemCall() VFS method.

#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>
#include <stdlib.h>
#ifndef OS_UNIX

// From test1.c
__device__ extern const char *sqlite3TestErrorName(int);
#include <sys/types.h>
#include <errno.h>

static struct TestSyscallGlobal
{
	bool Persist;               // 1 for persistent errors, 0 for transient
	int Count;					// Fail after this many more calls
	int Fails;                  // Number of failures that have occurred
} g_Syscall = { false, 0, 0 };

static int ts_open(const char *, int, int);
static int ts_close(int fd);
static int ts_access(const char *zPath, int mode);
static char *ts_getcwd(char *zPath, size_t nPath);
static int ts_stat(const char *zPath, struct stat *p);
static int ts_fstat(int fd, struct stat *p);
static int ts_ftruncate(int fd, off_t n);
static int ts_fcntl(int fd, int cmd, ... );
static int ts_read(int fd, void *aBuf, size_t nBuf);
static int ts_pread(int fd, void *aBuf, size_t nBuf, off_t off);
static int ts_pread64(int fd, void *aBuf, size_t nBuf, off_t off);
static int ts_write(int fd, const void *aBuf, size_t nBuf);
static int ts_pwrite(int fd, const void *aBuf, size_t nBuf, off_t off);
static int ts_pwrite64(int fd, const void *aBuf, size_t nBuf, off_t off);
static int ts_fchmod(int fd, mode_t mode);
static int ts_fallocate(int fd, off_t off, off_t len);


struct TestSyscallArray {
	const char *Name;
	syscall_ptr Test;
	syscall_ptr Orig;
	int DefaultErrno; // Default value for errno following errors
	int CustomErrno; // Current value for errno if error
} _syscalls[] = {
	/*  0 */ { "open",      (syscall_ptr)ts_open,      nullptr, EACCES, 0 },
	/*  1 */ { "close",     (syscall_ptr)ts_close,     nullptr, 0, 0 },
	/*  2 */ { "access",    (syscall_ptr)ts_access,    nullptr, 0, 0 },
	/*  3 */ { "getcwd",    (syscall_ptr)ts_getcwd,    nullptr, 0, 0 },
	/*  4 */ { "stat",      (syscall_ptr)ts_stat,      nullptr, 0, 0 },
	/*  5 */ { "fstat",     (syscall_ptr)ts_fstat,     nullptr, 0, 0 },
	/*  6 */ { "ftruncate", (syscall_ptr)ts_ftruncate, nullptr, EIO, 0 },
	/*  7 */ { "fcntl",     (syscall_ptr)ts_fcntl,     nullptr, EACCES, 0 },
	/*  8 */ { "read",      (syscall_ptr)ts_read,      nullptr, 0, 0 },
	/*  9 */ { "pread",     (syscall_ptr)ts_pread,     nullptr, 0, 0 },
	/* 10 */ { "pread64",   (syscall_ptr)ts_pread64,   nullptr, 0, 0 },
	/* 11 */ { "write",     (syscall_ptr)ts_write,     nullptr, 0, 0 },
	/* 12 */ { "pwrite",    (syscall_ptr)ts_pwrite,    nullptr, 0, 0 },
	/* 13 */ { "pwrite64",  (syscall_ptr)ts_pwrite64,  nullptr, 0, 0 },
	/* 14 */ { "fchmod",    (syscall_ptr)ts_fchmod,    nullptr, 0, 0 },
	/* 15 */ { "fallocate", (syscall_ptr)ts_fallocate, nullptr, 0, 0 },
	{ nullptr, nullptr, nullptr, 0, 0 }
};

#define orig_open      ((int(*)(const char*, int, int))_syscalls[0].Orig)
#define orig_close     ((int(*)(int))_syscalls[1].Orig)
#define orig_access    ((int(*)(const char*,int))_syscalls[2].Orig)
#define orig_getcwd    ((char*(*)(char*,size_t))_syscalls[3].Orig)
#define orig_stat      ((int(*)(const char*,struct stat*))_syscalls[4].Orig)
#define orig_fstat     ((int(*)(int,struct stat*))_syscalls[5].Orig)
#define orig_ftruncate ((int(*)(int,off_t))_syscalls[6].Orig)
#define orig_fcntl     ((int(*)(int,int,...))_syscalls[7].Orig)
#define orig_read      ((ssize_t(*)(int,void*,size_t))_syscalls[8].Orig)
#define orig_pread     ((ssize_t(*)(int,void*,size_t,off_t))_syscalls[9].Orig)
#define orig_pread64   ((ssize_t(*)(int,void*,size_t,off_t))_syscalls[10].Orig)
#define orig_write     ((ssize_t(*)(int,const void*,size_t))_syscalls[11].Orig)
#define orig_pwrite    ((ssize_t(*)(int,const void*,size_t,off_t))_syscalls[12].Orig)
#define orig_pwrite64  ((ssize_t(*)(int,const void*,size_t,off_t))_syscalls[13].Orig)
#define orig_fchmod    ((int(*)(int,mode_t))_syscalls[14].Orig)
#define orig_fallocate ((int(*)(int,off_t,off_t))_syscalls[15].Orig)

// This function is called exactly once from within each invocation of a system call wrapper in this file. It returns 1 if the function should fail, or 0 if it should succeed.
__device__ static bool tsIsFail()
{
	g_Syscall.Count--;
	if (g_Syscall.Count == 0 || (g_Syscall.Fails && g_Syscall.Persist))
	{
		g_Syscall.Fails++;
		return true;
	}
	return false;
}

// Return the current error-number value for function zFunc. zFunc must be the name of a system call in the _syscalls[] table.
//
// Usually, the current error-number is the value that errno should be set to if the named system call fails. The exception is "fallocate". See 
// comments above the implementation of ts_fallocate() for details.
__device__ static int tsErrno(const char *func)
{
	int funcLength = (int)_strlen(func);
	for (int i = 0; _syscalls[i].Name; i++)
	{
		if (_strlen(_syscalls[i].Name) != funcLength) continue;
		if (_memcmp(_syscalls[i].Name, func, funcLength)) continue;
		return _syscalls[i].CustomErrno;
	}
	_assert(false);
	return 0;
}

// A wrapper around tsIsFail(). If tsIsFail() returns non-zero, set the value of errno before returning.
__device__ static bool tsIsFailErrno(const char *func) { if (tsIsFail()) { _errno = tsErrno(func); return true; } return false; }

// A wrapper around open().
__device__ static int ts_open(const char *file, int flags, int mode) { return (tsIsFailErrno("open") ? -1 : orig_open(file, flags, mode)); }

// A wrapper around close().
// Even if simulating an error, close the original file-descriptor. This is to stop the test process from running out of file-descriptors
// when running a long test. If a call to close() appears to fail, SQLite never attempts to use the file-descriptor afterwards (or even to close it a second time).
__device__ static int ts_close(int fd) { if (tsIsFail()) { orig_close(fd); return -1; } return orig_close(fd); }

// A wrapper around access().
__device__ static int ts_access(const char *path, int mode) { return (tsIsFail() ? -1 : orig_access(path, mode); }

// A wrapper around getcwd().
__device__ static char *ts_getcwd(char *path, size_t pathLength) { return (tsIsFail() ? NULL : orig_getcwd(path, pathLength)); }

// A wrapper around stat().
__device__ static int ts_stat(const char *path, struct stat *p) { return (tsIsFail() ? -1 : orig_stat(path, p)); }

// A wrapper around fstat().
__device__ static int ts_fstat(int fd, struct stat *p) { return (tsIsFailErrno("fstat") ? -1 :  orig_fstat(fd, p)); }

// A wrapper around ftruncate().
__device__ static int ts_ftruncate(int fd, off_t n) { return (tsIsFailErrno("ftruncate") ? -1 : orig_ftruncate(fd, n)); }

// A wrapper around fcntl().
__device__ static int ts_fcntl(int fd, int cmd, ... )
{
	if (tsIsFailErrno("fcntl")) return -1;
	_va_list args;
	_va_start(args, cmd);
	void *arg = _va_arg(args, void*);
	return orig_fcntl(fd, cmd, arg);
}

// A wrapper around read().
__device__ static int ts_read(int fd, void *buf, size_t bufLength) { return (tsIsFailErrno("read") ? -1 : orig_read(fd, buf, bufLength)); }

// A wrapper around pread().
__device__ static int ts_pread(int fd, void *buf, size_t bufLength, off_t off) { return (tsIsFailErrno("pread") ? -1 : orig_pread(fd, buf, bufLength, off)); }

// A wrapper around pread64().
__device__ static int ts_pread64(int fd, void *buf, size_t bufLength, off_t off) { return (tsIsFailErrno("pread64") ? -1 : orig_pread64(fd, buf, bufLength, off)); }

// A wrapper around write().
__device__ static int ts_write(int fd, const void *buf, size_t bufLength)
{
	if (tsIsFailErrno("write")) { if (tsErrno("write") == EINTR) orig_write(fd, buf, bufLength/2); return -1; }
	return orig_write(fd, buf, bufLength);
}

// A wrapper around pwrite().
__device__ static int ts_pwrite(int fd, const void *buf, size_t bufLength, off_t off) { return (tsIsFailErrno("pwrite") ? - 1 : orig_pwrite(fd, buf, bufLength, off)); }

// A wrapper around pwrite64().
__device__ static int ts_pwrite64(int fd, const void *buf, size_t bufLength, off_t off) { return (tsIsFailErrno("pwrite64") ? -1 : orig_pwrite64(fd, buf, bufLength, off)); }

// A wrapper around fchmod().
__device__ static int ts_fchmod(int fd, mode_t mode) { return (tsIsFail() ? -1 : orig_fchmod(fd, mode)); }

// A wrapper around fallocate().
// SQLite assumes that the fallocate() function is compatible with posix_fallocate(). According to the Linux man page (2009-09-30):
// posix_fallocate() returns  zero on success, or an error number on failure. Note that errno is not set.
__device__ static int ts_fallocate(int fd, off_t off, off_t len) { return (tsIsFail() ? tsErrno("fallocate") : orig_fallocate(fd, off, len)); }

__device__ static int test_syscall_install(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 2, args, "SYSCALL-LIST");
		return JIM_ERROR;
	}
	int elemsLength;
	Jim_Obj **elems;
	if (Jim_ListGetElements(interp, args[2], &elemsLength, &elems))
		return JIM_ERROR;
	VSystem *vfs = VSystem::FindVfs(nullptr);
	for (int i = 0; i < elemsLength; i++)
	{
		int call;
		int rc = Jim_GetEnumFromStruct(interp, elems[i], (const void **)_syscalls, sizeof(_syscalls[0]), &call, "system-call", 0);
		if (rc) return rc;
		if (!_syscalls[call].Orig)
		{
			_syscalls[call].Orig = vfs->GetSystemCall(_syscalls[call].Name);
			vfs->SetSystemCall(_syscalls[call].Name, _syscalls[call].Test);
		}
		_syscalls[call].CustomErrno = _syscalls[call].DefaultErrno;
	}
	return JIM_OK;
}

__device__ static int test_syscall_uninstall(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 2, args, "");
		return JIM_ERROR;
	}
	VSystem *vfs = VSystem::FindVfs(nullptr);
	for (int i = 0; _syscalls[i].Name; i++)
	{
		if (_syscalls[i].Orig)
		{
			vfs->SetSystemCall(_syscalls[i].Name, nullptr);
			_syscalls[i].Orig = nullptr;
		}
	}
	return JIM_OK;
}

__device__ static int test_syscall_reset(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2 && argc != 3)
	{
		Jim_WrongNumArgs(interp, 2, args, "");
		return JIM_ERROR;
	}
	VSystem *vfs = VSystem::FindVfs(nullptr);
	int i;
	int rc;
	if (argc == 2)
	{
		rc = vfs->SetSystemCall(nullptr, nullptr);
		for (i = 0; _syscalls[i].Name; i++) _syscalls[i].Orig = nullptr;
	}
	else
	{
		int funcLength;
		const char *func = Jim_GetString(args[2], &funcLength);
		rc = vfs->SetSystemCall(Jim_String(args[2]), nullptr);
		for (i = 0; rc == RC_OK && _syscalls[i].Name; i++)
		{
			if (_strlen(_syscalls[i].Name) != funcLength) continue;
			if (_memcmp(_syscalls[i].Name, func, funcLength)) continue;
			_syscalls[i].Orig = nullptr;
		}
	}
	if (rc != RC_OK)
	{
		Jim_SetResult(interp, Jim_NewStringObj(interp, sqlite3TestErrorName(rc), -1));
		return JIM_ERROR;
	}
	Jim_ResetResult(interp);
	return JIM_OK;
}

__device__ static int test_syscall_exists(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 3)
	{
		Jim_WrongNumArgs(interp, 2, args, "");
		return JIM_ERROR;
	}
	VSystem *vfs = VSystem::FindVfs(nullptr);
	syscall_ptr x = vfs->GetSystemCall(Jim_String(args[2]));
	Jim_SetResult(interp, Jim_NewBooleanObj(interp, x != nullptr));
	return JIM_OK;
}

__device__ static int test_syscall_fault(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2 && argc != 4)
	{
		Jim_WrongNumArgs(interp, 2, args, "?COUNT PERSIST?");
		return JIM_ERROR;
	}
	int count = 0;
	bool persist = false;
	if (argc == 4)
		if (Jim_GetInt(interp, args[2], &count) || Jim_GetBoolean(interp, args[3], &persist))
			return JIM_ERROR;
	Jim_SetResult(interp, Jim_NewIntObj(interp, g_Syscall.Fails));
	g_Syscall.Count = count;
	g_Syscall.Persist = persist;
	g_Syscall.Fails = 0;
	return JIM_OK;
}

__device__ static int test_syscall_errno(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	struct Errno {
		const char *z;
		int i;
	} errnos[] = {
		{ "EACCES",    EACCES },
		{ "EINTR",     EINTR },
		{ "EIO",       EIO },
		{ "EOVERFLOW", EOVERFLOW },
		{ "ENOMEM",    ENOMEM },
		{ "EAGAIN",    EAGAIN },
		{ "ETIMEDOUT", ETIMEDOUT },
		{ "EBUSY",     EBUSY },
		{ "EPERM",     EPERM },
		{ "EDEADLK",   EDEADLK },
		{ "ENOLCK",    ENOLCK },
		{ nullptr, 0 }
	};
	if (argc != 4)
	{
		Jim_WrongNumArgs(interp, 2, args, "SYSCALL ERRNO");
		return JIM_ERROR;
	}
	int call;
	int rc = Jim_GetEnumFromStruct(interp, args[2], (const void **)_syscalls, sizeof(_syscalls[0]), &call, "system-call", 0);
	if (rc != JIM_OK) return rc;
	int errno_;
	rc = Jim_GetEnumFromStruct(interp, args[3], (const void **)errnos, sizeof(errnos[0]), &errno_, "errno", 0);
	if (rc != JIM_OK) return rc;
	_syscalls[call].CustomErrno = errnos[errno_].i;
	return JIM_OK;
}

__device__ static int test_syscall_list(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 2, args, "");
		return JIM_ERROR;
	}
	VSystem *vfs = VSystem::FindVfs(nullptr);
	Jim_Obj *list = Jim_NewListObj(interp, nullptr, 0);
	Jim_IncrRefCount(list);
	for (const char *sys = vfs->NextSystemCall(nullptr); sys != nullptr; sys = vfs->NextSystemCall(sys))
		Jim_ListAppendElement(interp, list, Jim_NewStringObj(interp, sys, -1));
	Jim_SetResult(interp, list);
	Jim_DecrRefCount(interp, list);
	return JIM_OK;
}

__device__ static int test_syscall_defaultvfs(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 2, args, "");
		return JIM_ERROR;
	}
	VSystem *vfs = VSystem::FindVfs(nullptr);
	Jim_SetResult(interp, Jim_NewStringObj(interp, vfs->Name, -1));
	return JIM_OK;
}

__device__ static int test_syscall(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	struct SyscallCmd {
		const char *Name;
		Jim_CmdProc *Proc;
	} cmds[] = {
		{ "fault",      test_syscall_fault },
		{ "install",    test_syscall_install },
		{ "uninstall",  test_syscall_uninstall },
		{ "reset",      test_syscall_reset },
		{ "errno",      test_syscall_errno },
		{ "exists",     test_syscall_exists },
		{ "list",       test_syscall_list },
		{ "defaultvfs", test_syscall_defaultvfs },
		{ nullptr, 0 }
	};
	if (argc < 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "SUB-COMMAND ...");
		return JIM_ERROR;
	}
	int cmd;
	int rc = Jim_GetEnumFromStruct(interp, args[1], (void **)cmds, sizeof(cmds[0]), &cmd, "sub-command", 0);
	if (rc != JIM_OK) return rc;
	return cmds[cmd].Proc(clientData, interp, argc, args);
}
__constant__ struct SyscallCmd {
	const char *Name;
	Jim_CmdProc *Proc;
} _cmds[] = {
	{ "test_syscall", test_syscall},
};
__device__ int SqlitetestSyscall_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_cmds); i++)
		Jim_CreateCommand(interp, _cmds[i].Name, _cmds[i].Proc, nullptr, nullptr);
	return JIM_OK;
}
#else
__device__ int SqlitetestSyscall_Init(Jim_Interp *interp)
{
	return JIM_OK;
}
#endif
