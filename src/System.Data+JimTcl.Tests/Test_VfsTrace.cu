// This file contains code implements a VFS shim that writes diagnostic output for each VFS call, similar to "strace".
//
// USAGE:
//
// This source file exports a single symbol which is the name of a function:
//
//   int vfstrace_register(
//     const char *zTraceName,         // Name of the newly constructed VFS
//     const char *zOldVfsName,        // Name of the underlying VFS
//     int (*xOut)(const char*,void*), // Output routine.  ex: fputs
//     void *pOutArg,                  // 2nd argument to xOut.  ex: stderr
//     int makeDefault                 // Make the new VFS the default
//   );
//
// Applications that want to trace their VFS usage must provide a callback function with this prototype:
//
//   int traceOutput(const char *zMessage, void *pAppData);
//
// This function will "output" the trace messages, where "output" can mean different things to different applications.  The traceOutput function
// for the command-line shell (see shell.c) is "fputs" from the standard library, which means that all trace output is written on the stream
// specified by the second argument.  In the case of the command-line shell the second argument is stderr.  Other applications might choose to output
// trace information to a file, over a socket, or write it into a buffer.
//
// The vfstrace_register() function creates a new "shim" VFS named by the zTraceName parameter.  A "shim" VFS is an SQLite backend that does
// not really perform the duties of a true backend, but simply filters or interprets VFS calls before passing them off to another VFS which does
// the actual work.  In this case the other VFS - the one that does the real work - is identified by the second parameter, zOldVfsName.  If
// the 2nd parameter is NULL then the default VFS is used.  The common case is for the 2nd parameter to be NULL.
//
// The third and fourth parameters are the pointer to the output function and the second argument to the output function.  For the SQLite
// command-line shell, when the -vfstrace option is used, these parameters are fputs and stderr, respectively.
//
// The fifth argument is true (non-zero) to cause the newly created VFS to become the default VFS.  The common case is for the fifth parameter to be true.
//
// The call to vfstrace_register() simply creates the shim VFS that does tracing.  The application must also arrange to use the new VFS for
// all database connections that are created and for which tracing is desired.  This can be done by specifying the trace VFS using URI filename
// notation, or by specifying the trace VFS as the 4th parameter to sqlite3_open_v2() or by making the trace VFS be the default (by setting
// the 5th parameter of vfstrace_register() to 1).
//
//
// ENABLING VFSTRACE IN A COMMAND-LINE SHELL
//
// The SQLite command line shell implemented by the shell.c source file can be used with this module.  To compile in -vfstrace support, first
// gather this file (test_vfstrace.c), the shell source file (shell.c), and the SQLite amalgamation source files (sqlite3.c, sqlite3.h) into
// the working directory.  Then compile using a command like the following:
//
//    gcc -o sqlite3 -Os -I. -DSQLITE_ENABLE_VFSTRACE \
//        -DSQLITE_THREADSAFE=0 -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_RTREE \
//        -DHAVE_READLINE -DHAVE_USLEEP=1 \
//        shell.c test_vfstrace.c sqlite3.c -ldl -lreadline -lncurses
//
// The gcc command above works on Linux and provides (in addition to the -vfstrace option) support for FTS3 and FTS4, RTREE, and command-line
// editing using the readline library.  The command-line shell does not use threads so we added -DSQLITE_THREADSAFE=0 just to make the code
// run a little faster.   For compiling on a Mac, you'll probably need to omit the -DHAVE_READLINE, the -lreadline, and the -lncurses options.
// The compilation could be simplified to just this:
//
//    gcc -DSQLITE_ENABLE_VFSTRACE \
//         shell.c test_vfstrace.c sqlite3.c -ldl -lpthread
//
// In this second example, all unnecessary options have been removed Note that since the code is now threadsafe, we had to add the -lpthread
// option to pull in the pthreads library.
//
// To cross-compile for windows using MinGW, a command like this might work:
//
//    /opt/mingw/bin/i386-mingw32msvc-gcc -o sqlite3.exe -Os -I \
//         -DSQLITE_THREADSAFE=0 -DSQLITE_ENABLE_VFSTRACE \
//         shell.c test_vfstrace.c sqlite3.c
//
// Similar compiler commands will work on different systems.  The key invariants are (1) you must have -DSQLITE_ENABLE_VFSTRACE so that
// the shell.c source file will know to include the -vfstrace command-line option and (2) you must compile and link the three source files
// shell,c, test_vfstrace.c, and sqlite3.c.  
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <new.h>

// An instance of this structure is attached to the each trace VFS to provide auxiliary information.
typedef struct VfsTraceInfo VfsTraceInfo;
struct VfsTraceInfo
{
	VSystem *RootVfs;			// The underlying real VFS
	int (*Out)(const char*, void*); // Send output here
	void *OutArg;               // First argument to xOut
	const char *VfsName;        // Name of this trace-VFS
	VSystem *TraceVfs;          // Pointer back to the trace VFS
};

// Return a pointer to the tail of the pathname.  Examples:
//     /home/drh/xyzzy.txt -> xyzzy.txt
//     xyzzy.txt           -> xyzzy.txt
__device__ static const char *fileTail(const char *z)
{
	if (!z) return nullptr;
	int i = (int)_strlen(z)-1;
	while (i > 0 && z[i-1] != '/') { i--; }
	return &z[i];
}

// Send trace output defined by zFormat and subsequent arguments.
__device__ static void vfstrace_printf(VfsTraceInfo *info, const char *format, ...)
{
	_va_list args;
	_va_start(args, format);
	char *msg = _vmprintf(format, &args);
	_va_end(args);
	info->Out(msg, info->OutArg);
	_free(msg);
}

// Convert value rc into a string and print it using zFormat.  zFormat should have exactly one %s
__device__ static void vfstrace_print_errcode(VfsTraceInfo *info, const char *format, int rc)
{
	char buf[50];
	char *val;
	switch (rc) {
	case RC_OK:         val = "SQLITE_OK"; break;
	case RC_ERROR:      val = "SQLITE_ERROR"; break;
	case RC_PERM:       val = "SQLITE_PERM"; break;
	case RC_ABORT:      val = "SQLITE_ABORT"; break;
	case RC_BUSY:       val = "SQLITE_BUSY"; break;
	case RC_NOMEM:      val = "SQLITE_NOMEM"; break;
	case RC_READONLY:   val = "SQLITE_READONLY"; break;
	case RC_INTERRUPT:  val = "SQLITE_INTERRUPT"; break;
	case RC_IOERR:      val = "SQLITE_IOERR"; break;
	case RC_CORRUPT:    val = "SQLITE_CORRUPT"; break;
	case RC_FULL:       val = "SQLITE_FULL"; break;
	case RC_CANTOPEN:   val = "SQLITE_CANTOPEN"; break;
	case RC_PROTOCOL:   val = "SQLITE_PROTOCOL"; break;
	case RC_EMPTY:      val = "SQLITE_EMPTY"; break;
	case RC_SCHEMA:     val = "SQLITE_SCHEMA"; break;
	case RC_CONSTRAINT: val = "SQLITE_CONSTRAINT"; break;
	case RC_MISMATCH:   val = "SQLITE_MISMATCH"; break;
	case RC_MISUSE:     val = "SQLITE_MISUSE"; break;
	case RC_NOLFS:      val = "SQLITE_NOLFS"; break;
	case RC_IOERR_READ:         val = "SQLITE_IOERR_READ"; break;
	case RC_IOERR_SHORT_READ:   val = "SQLITE_IOERR_SHORT_READ"; break;
	case RC_IOERR_WRITE:        val = "SQLITE_IOERR_WRITE"; break;
	case RC_IOERR_FSYNC:        val = "SQLITE_IOERR_FSYNC"; break;
	case RC_IOERR_DIR_FSYNC:    val = "SQLITE_IOERR_DIR_FSYNC"; break;
	case RC_IOERR_TRUNCATE:     val = "SQLITE_IOERR_TRUNCATE"; break;
	case RC_IOERR_FSTAT:        val = "SQLITE_IOERR_FSTAT"; break;
	case RC_IOERR_UNLOCK:       val = "SQLITE_IOERR_UNLOCK"; break;
	case RC_IOERR_RDLOCK:       val = "SQLITE_IOERR_RDLOCK"; break;
	case RC_IOERR_DELETE:       val = "SQLITE_IOERR_DELETE"; break;
	case RC_IOERR_BLOCKED:      val = "SQLITE_IOERR_BLOCKED"; break;
	case RC_IOERR_NOMEM:        val = "SQLITE_IOERR_NOMEM"; break;
	case RC_IOERR_ACCESS:       val = "SQLITE_IOERR_ACCESS"; break;
	case RC_IOERR_CHECKRESERVEDLOCK: val = "SQLITE_IOERR_CHECKRESERVEDLOCK"; break;
	case RC_IOERR_LOCK:         val = "SQLITE_IOERR_LOCK"; break;
	case RC_IOERR_CLOSE:        val = "SQLITE_IOERR_CLOSE"; break;
	case RC_IOERR_DIR_CLOSE:    val = "SQLITE_IOERR_DIR_CLOSE"; break;
	case RC_IOERR_SHMOPEN:      val = "SQLITE_IOERR_SHMOPEN"; break;
	case RC_IOERR_SHMSIZE:      val = "SQLITE_IOERR_SHMSIZE"; break;
	case RC_IOERR_SHMLOCK:      val = "SQLITE_IOERR_SHMLOCK"; break;
	case RC_LOCKED_SHAREDCACHE: val = "SQLITE_LOCKED_SHAREDCACHE"; break;
	case RC_BUSY_RECOVERY:      val = "SQLITE_BUSY_RECOVERY"; break;
	case RC_CANTOPEN_NOTEMPDIR: val = "SQLITE_CANTOPEN_NOTEMPDIR"; break;
	default: {
		_snprintf(buf, sizeof(buf), "%d", rc);
		val = buf;
		break; }
	}
	vfstrace_printf(info, format, val);
}

// Append to a buffer.
__device__ static void strappend(char *z, int *i, const char *append)
{
	int i2 = *i;
	while (append[0]){ z[i2++] = *(append++); }
	z[i2] = 0;
	*i= i2;
}

class VfsTraceVFile : public VFile
{
public:
	VfsTraceInfo *Info; // The trace-VFS to which this file belongs
	const char *FName; // Base name of the file
	VFile *Real; // The real underlying file
public:
	__device__ virtual RC Close_()
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xClose(%s)", info->VfsName, FName);
		RC rc = Real->Close();
		vfstrace_print_errcode(info, " -> %s\n", rc);
		if (rc == RC_OK)
		{
			//_free((void*)p->base.pMethods);
			Opened = false;
		}
		return rc;
	}
	__device__ virtual RC Read(void *buffer, int amount, int64 offset)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xRead(%s,n=%d,ofst=%lld)", info->VfsName, FName, amount, offset);
		RC rc = Real->Read(buffer, amount, offset);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC Write(const void *buffer, int amount, int64 offset)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xWrite(%s,n=%d,ofst=%lld)", info->VfsName, FName, amount, offset);
		RC rc = Real->Write(buffer, amount, offset);;
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC Truncate(int64 size)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xTruncate(%s,%lld)", info->VfsName, FName, size);
		RC rc = Real->Truncate(size);
		vfstrace_printf(info, " -> %d\n", rc);
		return rc;
	}
	__device__ virtual RC Sync(SYNC flags)
	{
		char buf[100];
		memcpy(buf, "|0", 3);
		int i = 0;
		if (flags & SYNC_FULL) strappend(buf, &i, "|FULL");
		else if (flags & SYNC_NORMAL) strappend(buf, &i, "|NORMAL");
		if (flags & SYNC_DATAONLY) strappend(buf, &i, "|DATAONLY");
		if (flags & ~(SYNC_FULL|SYNC_DATAONLY)) _snprintf(&buf[i], sizeof(buf)-i, "|0x%x", flags);
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xSync(%s,%s)", info->VfsName, FName, &buf[1]);
		RC rc = Real->Sync(flags);
		vfstrace_printf(info, " -> %d\n", rc);
		return rc;
	}
	__device__ virtual RC get_FileSize(int64 &size)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xFileSize(%s)", info->VfsName, FName);
		RC rc = Real->get_FileSize(size);
		vfstrace_print_errcode(info, " -> %s,", rc);
		vfstrace_printf(info, " size=%lld\n", size);
		return rc;
	}
	__device__ static const char *lockName(int lock)
	{
		const char *lockNames[] = {
			"NONE", "SHARED", "RESERVED", "PENDING", "EXCLUSIVE"
		};
		return (lock < 0 || lock >= _lengthof(lockNames) ? "???" :lockNames[lock]);
	}
	__device__ virtual RC Lock(LOCK lock)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xLock(%s,%s)", info->VfsName, FName, lockName(lock));
		RC rc = Real->Lock(lock);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC Unlock(LOCK lock)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xUnlock(%s,%s)", info->VfsName, FName, lockName(lock));
		RC rc = Real->Unlock(lock);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC CheckReservedLock(int &lock)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xCheckReservedLock(%s,%d)", info->VfsName, FName);
		RC rc = Real->CheckReservedLock(lock);
		vfstrace_print_errcode(info, " -> %s", rc);
		vfstrace_printf(info, ", out=%d\n", lock);
		return rc;
	}
	__device__ virtual RC FileControl(FCNTL op, void *arg)
	{
		VfsTraceInfo *info = Info;
		char buf[100];
		char *opName;
		switch (op) {
		case FCNTL_LOCKSTATE:			opName = "LOCKSTATE"; break;
		case FCNTL_GET_LOCKPROXYFILE:	opName = "GET_LOCKPROXYFILE"; break;
		case FCNTL_SET_LOCKPROXYFILE:	opName = "SET_LOCKPROXYFILE"; break;
		case FCNTL_LAST_ERRNO:			opName = "LAST_ERRNO"; break;
		case FCNTL_SIZE_HINT: {
			_snprintf(buf, sizeof(buf), "SIZE_HINT,%lld", *(int64 *)arg);
			opName = buf;
			break; }
		case FCNTL_CHUNK_SIZE: {
			_snprintf(buf, sizeof(buf), "CHUNK_SIZE,%d", *(int *)arg);
			opName = buf;
			break; }
		case FCNTL_FILE_POINTER:	opName = "FILE_POINTER"; break;
		case FCNTL_SYNC_OMITTED:	opName = "SYNC_OMITTED"; break;
		case FCNTL_WIN32_AV_RETRY:	opName = "WIN32_AV_RETRY"; break;
		case FCNTL_PERSIST_WAL:		opName = "PERSIST_WAL"; break;
		case FCNTL_OVERWRITE:		opName = "OVERWRITE"; break;
		case FCNTL_VFSNAME:			opName = "VFSNAME"; break;
		case FCNTL_TEMPFILENAME:	opName = "TEMPFILENAME"; break;
		case 0xca093fa0:            opName = "DB_UNCHANGED"; break;
		case FCNTL_PRAGMA: {
			const char *const *a = (const char *const *)arg;
			_snprintf(buf, sizeof(buf), "PRAGMA,[%s,%s]", a[1], a[2]);
			opName = buf;
			break; }
		default: {
			_snprintf(buf, sizeof(buf), "%d", op);
			opName = buf;
			break; }
		}
		vfstrace_printf(info, "%s.xFileControl(%s,%s)", info->VfsName, FName, opName);
		RC rc = Real->FileControl(op, arg);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		if (op == FCNTL_VFSNAME && rc == RC_OK)
			*(char**)arg = _mprintf("vfstrace.%s/%z", info->VfsName, *(char **)arg);
		if ((op == FCNTL_PRAGMA || op == FCNTL_TEMPFILENAME) && rc == RC_OK && *(char **)arg)
			vfstrace_printf(info, "%s.xFileControl(%s,%s) returns %s", info->VfsName, FName, opName, *(char **)arg);
		return rc;
	}

	__device__ virtual uint get_SectorSize()
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xSectorSize(%s)", info->VfsName, FName);
		uint rc = Real->get_SectorSize();
		vfstrace_printf(info, " -> %d\n", rc);
		return rc;
	}
	__device__ virtual IOCAP get_DeviceCharacteristics()
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xDeviceCharacteristics(%s)", info->VfsName, FName);
		IOCAP rc = Real->get_DeviceCharacteristics();
		vfstrace_printf(info, " -> 0x%08x\n", rc);
		return rc;
	}

	__device__ virtual RC ShmLock(int offset, int n, SHM flags)
	{
		char lockName[100];
		memcpy(lockName, "|0", 3);
		int i = 0;
		if (flags & SHM_UNLOCK) strappend(lockName, &i, "|UNLOCK");
		if (flags & SHM_LOCK) strappend(lockName, &i, "|LOCK");
		if (flags & SHM_SHARED) strappend(lockName, &i, "|SHARED");
		if (flags & SHM_EXCLUSIVE) strappend(lockName, &i, "|EXCLUSIVE");
		if (flags & ~(0xf)) _snprintf(&lockName[i], sizeof(lockName)-i, "|0x%x", flags);
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xShmLock(%s,ofst=%d,n=%d,%s)", info->VfsName, FName, offset, n, &lockName[1]);
		RC rc = Real->ShmLock(offset, n, flags);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual void ShmBarrier()
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xShmBarrier(%s)\n", info->VfsName, FName);
		Real->ShmBarrier();
	}
	__device__ virtual RC ShmUnmap(bool deleteFlag)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xShmUnmap(%s,delFlag=%d)", info->VfsName, FName, deleteFlag);
		RC rc = Real->ShmUnmap(deleteFlag);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp)
	{
		VfsTraceInfo *info = Info;
		vfstrace_printf(info, "%s.xShmMap(%s,iRegion=%d,szRegion=%d,isWrite=%d,*)", info->VfsName, FName, region, sizeRegion, isWrite);
		RC rc = Real->ShmMap(region, sizeRegion, isWrite, pp);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
};

class VfsTraceVSystem : public VSystem
{
public:
	VfsTraceInfo *AppData;
public:
	__device__ virtual VFile *_AttachFile(void *buffer) { return new (buffer) VfsTraceVFile(); }
	__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
	{
		VfsTraceVFile *p = (VfsTraceVFile *)file;
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		p->Info = info;
		p->FName = (path ? fileTail(path) : "<temp>");
		p->Real = (VFile *)&p[1];
		RC rc = root->Open(path, p->Real, flags, outFlags);
		vfstrace_printf(info, "%s.xOpen(%s,flags=0x%x)", info->VfsName, p->FName, flags);
		if (p->Opened)
		{
		}
		vfstrace_print_errcode(info, " -> %s", rc);
		if (outFlags) vfstrace_printf(info, ", outFlags=0x%x\n", *outFlags);
		else vfstrace_printf(info, "\n");
		return rc;
	}
	__device__ virtual RC Delete(const char *path, bool syncDirectory)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDelete(\"%s\",%d)", info->VfsName, path, syncDirectory);
		RC rc = root->Delete(path, syncDirectory);
		vfstrace_print_errcode(info, " -> %s\n", rc);
		return rc;
	}
	__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDelete(\"%s\",%d)", info->VfsName, path, flags);
		RC rc = root->Access(path, flags, outRC);
		vfstrace_print_errcode(info, " -> %s", rc);
		vfstrace_printf(info, ", out=%d\n", *outRC);
		return rc;
	}
	__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xFullPathname(\"%s\")", info->VfsName, path);
		RC rc = root->FullPathname(path, pathOutLength, pathOut);
		vfstrace_print_errcode(info, " -> %s", rc);
		vfstrace_printf(info, ", out=\"%.*s\"\n", pathOutLength, pathOut);
		return rc;
	}

	__device__ virtual void *DlOpen(const char *filename)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDlOpen(\"%s\")\n", info->VfsName, filename);
		return root->DlOpen(filename);
	}
	__device__ virtual void DlError(int bufLength, char *buf)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDlError(%d)", info->VfsName, bufLength);
		root->DlError(bufLength, buf);
		vfstrace_printf(info, " -> \"%s\"", buf);
	}
	__device__ virtual void (*DlSym(void *handle, const char *symbol))()
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDlSym(\"%s\")\n", info->VfsName, symbol);
		return root->DlSym(handle, symbol);
	}
	__device__ virtual void DlClose(void *handle)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xDlOpen()\n", info->VfsName);
		root->DlClose(handle);
	}

	__device__ virtual int Randomness(int bufLength, char *buf)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		vfstrace_printf(info, "%s.xRandomness(%d)\n", info->VfsName, bufLength);
		return root->Randomness(bufLength, buf);
	}
	__device__ virtual int Sleep(int microseconds)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->Sleep(microseconds);
	}
	__device__ virtual RC CurrentTimeInt64(int64 *now)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->CurrentTimeInt64(now);
	}
	__device__ virtual RC CurrentTime(double *now)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->CurrentTime(now);
	}
	__device__ virtual RC GetLastError(int bufLength, char *buf)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->GetLastError(bufLength, buf);
	}

	__device__ virtual RC SetSystemCall(const char *name, syscall_ptr newFunc)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->SetSystemCall(name, newFunc);
	}
	__device__ virtual syscall_ptr GetSystemCall(const char *name)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->GetSystemCall(name);
	}
	__device__ virtual const char *NextSystemCall(const char *name)
	{
		VfsTraceInfo *info = AppData;
		VSystem *root = info->RootVfs;
		return root->NextSystemCall(name);
	}
};

// Clients invoke this routine to construct a new trace-vfs shim.
//
// Return SQLITE_OK on success.  
// SQLITE_NOMEM is returned in the case of a memory allocation error.
// SQLITE_NOTFOUND is returned if zOldVfsName does not exist.
__device__ int vfstrace_register(const char *traceName, const char *oldVfsName, int (*out)(const char*,void*), void *outArg, bool makeDefault)
{
	VSystem *root = VSystem::FindVfs(oldVfsName);
	if (!root) return RC_NOTFOUND;
	int nameLength = _strlen(traceName);
	int bytes = sizeof(VfsTraceVSystem) + sizeof(VfsTraceInfo) + nameLength + 1;
	VfsTraceVSystem *new_ = (VfsTraceVSystem *)_alloc(bytes);
	if (!new_) return RC_NOMEM;
	memset(new_, 0, bytes);
	VfsTraceInfo *info = (VfsTraceInfo *)&new_[1];
	new_->SizeOsFile = root->SizeOsFile + sizeof(VfsTraceVFile);
	new_->MaxPathname = root->MaxPathname;
	new_->Name = (char *)&info[1];
	memcpy((char *)&info[1], traceName, nameLength+1);
	new_->AppData = info;
	info->RootVfs = root;
	info->Out = out;
	info->OutArg = outArg;
	info->VfsName = new_->Name;
	info->TraceVfs = new_;
	vfstrace_printf(info, "%s.enabled_for(\"%s\")\n", info->VfsName, root->Name);
	return VSystem::RegisterVfs(new_, makeDefault);
}
