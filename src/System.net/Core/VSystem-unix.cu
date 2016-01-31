// os_unix.c
#include "Core.cu.h"

//#define OS_UNIX 1
#if OS_UNIX // This file is used on unix only

namespace CORE_NAME
{

#pragma region Preamble

#if defined(_TEST) || defined(_DEBUG)
	bool OsTrace = false;
#define OSTRACE(X, ...) if (OsTrace) { _dprintf("OS: "X, __VA_ARGS__); }
#else
#define OSTRACE(X, ...)
#endif

#ifdef _TEST
	__device__ int g_io_error_hit = 0;            // Total number of I/O Errors
	__device__ int g_io_error_hardhit = 0;        // Number of non-benign errors
	__device__ int g_io_error_pending = 0;        // Count down to first I/O error
	__device__ int g_io_error_persist = 0;        // True if I/O errors persist
	__device__ int g_io_error_benign = 0;         // True if errors are benign
	__device__ int g_diskfull_pending = 0;
	__device__ int g_diskfull = 0;
#define SimulateIOErrorBenign(X) g_io_error_benign=(X)
#define SimulateIOError(CODE) \
	if ((g_io_error_persist && g_io_error_hit) || g_io_error_pending-- == 1) { local_ioerr(); CODE; }
	__device__ static void local_ioerr() { OSTRACE("IOERR\n"); g_io_error_hit++; if (!g_io_error_benign) g_io_error_hardhit++; }
#define SimulateDiskfullError(CODE) \
	if (g_diskfull_pending) { if (g_diskfull_pending == 1) { \
	local_ioerr(); g_diskfull = 1; g_io_error_hit = 1; CODE; \
	} else g_diskfull_pending--; }
#else
#define SimulateIOErrorBenign(X)
#define SimulateIOError(A)
#define SimulateDiskfullError(A)
#endif

#ifdef _TEST
	__device__ static int _saved_cnt;
	__device__ void DisableSimulatedIOErrors(int *pending, int *hit) { if (!pending) pending = &_saved_cnt; *pending = g_io_error_pending; g_io_error_pending = -1; if (hit) { *hit = g_io_error_hit; g_io_error_hit = 0; } }
	__device__ void EnableSimulatedIOErrors(int *pending, int *hit) { if (!pending) pending = &_saved_cnt; g_io_error_pending = *pending; if (hit) g_io_error_hit = *hit; }
#endif

	// When testing, keep a count of the number of open files.
#ifdef _TEST
	__device__ int g_open_file_count = 0;
#define OpenCounter(X) g_open_file_count += (X)
#else
#define OpenCounter(X)
#endif

#pragma endregion

#pragma region Polyfill

	// Use posix_fallocate() if it is available
#if !defined(HAVE_POSIX_FALLOCATE) && (_XOPEN_SOURCE >= 600 || _POSIX_C_SOURCE >= 200112L)
#define HAVE_POSIX_FALLOCATE 1
#endif

#if !defined(ENABLE_LOCKING_STYLE)
#if defined(__APPLE__)
#define ENABLE_LOCKING_STYLE 1
#else
#define ENABLE_LOCKING_STYLE 0
#endif
#endif

#ifndef OS_VXWORKS
#if defined(__RTP__) || defined(_WRS_KERNEL)
#define OS_VXWORKS 1
#else
#define OS_VXWORKS 0
#endif
#endif

#ifndef DISABLE_LFS
#define _LARGE_FILE       1
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#define _LARGEFILE_SOURCE 1
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#ifndef OMIT_WAL
#include <sys/mman.h>
#endif

#if ENABLE_LOCKING_STYLE
#include <sys/ioctl.h>
#if OS_VXWORKS
#include <semaphore.h>
#include <limits.h>
#else
#include <sys/file.h>
#include <sys/param.h>
#endif
#endif

#if defined(__APPLE__) || (ENABLE_LOCKING_STYLE && !OS_VXWORKS)
#include <sys/mount.h>
#endif

#ifdef HAVE_UTIME
#include <utime.h>
#endif

#define FSFLAGS_IS_MSDOS     0x1

#if THREADSAFE
#include <pthread.h>
#define UNIX_THREADS 1
#endif

#ifndef DEFAULT_FILE_PERMISSIONS
#define DEFAULT_FILE_PERMISSIONS 0644
#endif

#ifndef DEFAULT_PROXYDIR_PERMISSIONS
#define DEFAULT_PROXYDIR_PERMISSIONS 0755
#endif

#define MAX_PATHNAME 512

	// Define various macros that are missing from some systems.
#ifndef O_LARGEFILE
# define O_LARGEFILE 0
#endif
#ifdef SQLITE_DISABLE_LFS
# undef O_LARGEFILE
# define O_LARGEFILE 0
#endif
#ifndef O_NOFOLLOW
# define O_NOFOLLOW 0
#endif
#ifndef O_BINARY
# define O_BINARY 0
#endif

	// The threadid macro resolves to the thread-id or to 0.  Used for testing and debugging only.
#if THREADSAFE
#define threadid pthread_self()
#else
#define threadid 0
#endif

	// Different Unix systems declare open() in different ways.  Same use open(const char*,int,mode_t).  Others use open(const char*,int,...).
	// The difference is important when using a pointer to the function.
	//
	// The safest way to deal with the problem is to always use this wrapper which always has the same well-defined interface.
	static int posixOpen(const char *zFile, int flags, int mode) { return open(zFile, flags, mode); }

	// On some systems, calls to fchown() will trigger a message in a security log if they come from non-root processes.  So avoid calling fchown() if
	// we are not running as root.
	static int posixFchown(int fd, uid_t uid, gid_t gid) { return geteuid() ? 0 : fchown(fd,uid,gid); }

#pragma endregion

#define IS_LOCK_ERROR(x)  ((x != RC_OK) && (x != RC_BUSY))

#pragma region UnixVFile

	// Forward references
	typedef struct unixShm unixShm;               // Connection shared memory
	typedef struct unixShmNode unixShmNode;       // Shared memory instance
	typedef struct unixInodeInfo unixInodeInfo;   // An i-node
	typedef struct UnixUnusedFd UnixUnusedFd;     // An unused file descriptor

	// Sometimes, after a file handle is closed by SQLite, the file descriptor cannot be closed immediately. In these cases, instances of the following
	// structure are used to store the file descriptor while waiting for an opportunity to either close or reuse it.
	struct UnixUnusedFd
	{
		int fd;                   // File descriptor to close
		int flags;                // Flags this file descriptor was opened with
		UnixUnusedFd *pNext;      // Next unused file descriptor on same file
	};

	// unixFile
	class UnixVFile : public VFile
	{
	public:
		enum UNIXFILE : uint8
		{
			UNIXFILE_EXCL = 0x01,			// Connections from one process only
			UNIXFILE_RDONLY = 0x02,			// Connection is read only
			UNIXFILE_PERSIST_WAL = 0x04,    // Persistent WAL mode
#ifndef DISABLE_DIRSYNC
			UNIXFILE_DIRSYNC = 0x08,		// Directory sync needed
#else
			UNIXFILE_DIRSYNC = 0x00,
#endif
			UNIXFILE_PSOW = 0x10,			// SQLITE_IOCAP_POWERSAFE_OVERWRITE
			UNIXFILE_DELETE = 0x20,			// Delete on close
			UNIXFILE_URI = 0x40,			// Filename might have query parameters
			UNIXFILE_NOLOCK = 0x80,			// Do no file locking
		};

		VSystem *pVfs;						// The VFS that created this unixFile
		unixInodeInfo *pInode;              // Info about locks on this inode
		int h;                              // The file descriptor
		unsigned char eFileLock;            // The type of lock held on this fd
		unsigned short int ctrlFlags;       // Behavioral bits.  UNIXFILE_* flags
		int lastErrno;                      // The unix errno from last I/O error
		void *lockingContext;               // Locking style specific state
		UnixUnusedFd *pUnused;              // Pre-allocated UnixUnusedFd
		const char *zPath;                  // Name of the file
		unixShm *pShm;                      // Shared memory segment information
		int szChunk;                        // Configured by FCNTL_CHUNK_SIZE
#ifdef __QNXNTO__
		int sectorSize;                     // Device sector size
		int deviceCharacteristics;          // Precomputed device characteristics
#endif
#if ENABLE_LOCKING_STYLE
		int openFlags;                      // The flags specified at open()
#endif
#if ENABLE_LOCKING_STYLE || defined(__APPLE__)
		unsigned fsFlags;                   // cached details from statfs()
#endif
#if OS_VXWORKS
		struct vxworksFileId *pId;          // Unique file ID
#endif
#ifdef _DEBUG
		// The next group of variables are used to track whether or not the transaction counter in bytes 24-27 of database files are updated
		// whenever any part of the database changes.  An assertion fault will occur if a file is updated without also updating the transaction
		// counter.  This test is made to avoid new problems similar to the one described by ticket #3584. 
		unsigned char transCntrChng;   // True if the transaction counter changed
		unsigned char dbUpdate;        // True if any part of database file changed
		unsigned char inNormalWrite;   // True if in a normal write operation
#endif
#ifdef _TEST
		// In test mode, increase the size of this structure a bit so that it is larger than the struct CrashFile defined in test6.c.
		char aPadding[32];
#endif
	};

	__device__ __forceinline void operator|=(UnixVFile::UNIXFILE &a, int b) { a = (UnixVFile::UNIXFILE)(a | b); }

#pragma endregion

#pragma region UnixVSystem

	class UnixVSystem : public VSystem
	{
	public:
		__device__ virtual VFile *_AttachFile(void *buffer);
		__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags);
		__device__ virtual RC Delete(const char *path, bool syncDirectory);
		__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC);
		__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut);

		__device__ virtual void *DlOpen(const char *filename);
		__device__ virtual void DlError(int bufLength, char *buf);
		__device__ virtual void (*DlSym(void *handle, const char *symbol))();
		__device__ virtual void DlClose(void *handle);

		__device__ virtual int Randomness(int bufLength, char *buf);
		__device__ virtual int Sleep(int microseconds);
		__device__ virtual RC CurrentTimeInt64(int64 *now);
		__device__ virtual RC CurrentTime(double *now);
		__device__ virtual RC GetLastError(int bufLength, char *buf);

		__device__ virtual RC SetSystemCall(const char *name, syscall_ptr newFunc);
		__device__ virtual syscall_ptr GetSystemCall(const char *name);
		__device__ virtual const char *NextSystemCall(const char *name);
	};

#pragma endregion

	// Forward reference
	static int openDirectory(const char*, int*);

#pragma region Syscall

	static struct unix_syscall
	{
		const char *Name;		// Name of the system call
		syscall_ptr Current;	// Current value of the system call
		syscall_ptr Default;	// Default value
	} Syscalls[] = {
		{ "open", (syscall_ptr)posixOpen, 0 },
#define osOpen ((int(*)(const char*,int,int))Syscalls[0].Current)
		{ "close", (syscall_ptr)close, 0 },
#define osClose ((int(*)(int))Syscalls[1].Current)
		{ "access", (syscall_ptr)access, 0 },
#define osAccess ((int(*)(const char*,int))Syscalls[2].Current)
		{ "getcwd", (syscall_ptr)getcwd, 0 },
#define osGetcwd ((char*(*)(char*,size_t))Syscalls[3].Current)
		{ "stat", (syscall_ptr)stat, 0  },
#define osStat ((int(*)(const char*,struct stat*))Syscalls[4].Current)
		// The DJGPP compiler environment looks mostly like Unix, but it lacks the fcntl() system call.  So redefine fcntl() to be something
		// that always succeeds.  This means that locking does not occur under DJGPP.  But it is DOS - what did you expect?
#ifdef __DJGPP__
		{ "fstat", 0, 0  },
#define osFstat(a,b,c) 0
#else     
		{ "fstat", (syscall_ptr)fstat, 0 },
#define osFstat ((int(*)(int,struct stat*))Syscalls[5].Current)
#endif
		{ "ftruncate", (syscall_ptr)ftruncate, 0 },
#define osFtruncate ((int(*)(int,off_t))Syscalls[6].Current)
		{ "fcntl", (syscall_ptr)fcntl, 0 },
#define osFcntl ((int(*)(int,int,...))Syscalls[7].Current)
		{ "read", (syscall_ptr)read, 0 },
#define osRead ((ssize_t(*)(int,void*,size_t))Syscalls[8].Current)
#if defined(USE_PREAD) || SQLITE_ENABLE_LOCKING_STYLE
		{ "pread", (syscall_ptr)pread, 0 },
#else
		{ "pread", (syscall_ptr)0, 0 },
#endif
#define osPread ((ssize_t(*)(int,void*,size_t,off_t))Syscalls[9].Current)
#if defined(USE_PREAD64)
		{ "pread64", (syscall_ptr)pread64, 0 },
#else
		{ "pread64", (syscall_ptr)0, 0 },
#endif
#define osPread64 ((ssize_t(*)(int,void*,size_t,off_t))Syscalls[10].Current)
		{ "write", (syscall_ptr)write, 0 },
#define osWrite ((ssize_t(*)(int,const void*,size_t))Syscalls[11].Current)
#if defined(USE_PREAD) || ENABLE_LOCKING_STYLE
		{ "pwrite", (syscall_ptr)pwrite, 0 },
#else
		{ "pwrite", (syscall_ptr)0, 0 },
#endif
#define osPwrite ((ssize_t(*)(int,const void*,size_t,off_t))Syscalls[12].Current)
#if defined(USE_PREAD64)
		{ "pwrite64", (syscall_ptr)pwrite64, 0 },
#else
		{ "pwrite64", (syscall_ptr)0, 0 },
#endif
#define osPwrite64 ((ssize_t(*)(int,const void*,size_t,off_t))Syscalls[13].Current)
		{ "fchmod", (syscall_ptr)fchmod, 0 },
#define osFchmod ((int(*)(int,mode_t))Syscalls[14].Current)
#if defined(HAVE_POSIX_FALLOCATE) && HAVE_POSIX_FALLOCATE
		{ "fallocate", (syscall_ptr)posix_fallocate, 0 },
#else
		{ "fallocate", (syscall_ptr)0, 0 },
#endif
#define osFallocate ((int(*)(int,off_t,off_t))Syscalls[15].Current)
		{ "unlink", (syscall_ptr)unlink, 0 },
#define osUnlink ((int(*)(const char*))Syscalls[16].Current)
		{ "openDirectory", (syscall_ptr)openDirectory, 0 },
#define osOpenDirectory ((int(*)(const char*,int*))Syscalls[17].Current)
		{ "mkdir", (syscall_ptr)mkdir, 0 },
#define osMkdir     ((int(*)(const char*,mode_t))Syscalls[18].Current)
		{ "rmdir", (syscall_ptr)rmdir, 0 },
#define osRmdir ((int(*)(const char*))Syscalls[19].Current)
		{ "fchown", (syscall_ptr)posixFchown, 0 },
#define osFchown ((int(*)(int,uid_t,gid_t))Syscalls[20].Current)

	}; /* End of the overrideable system calls */

	RC UnixVSystem::SetSystemCall(const char *name, syscall_ptr newFunc)
	{
		RC rc = RC_NOTFOUND;
		if (!name)
		{
			// If no zName is given, restore all system calls to their default settings and return NULL
			rc = RC_OK;
			for (int i =0 ; i < _lengthof(Syscalls); i++)
				if (Syscalls[i].Default)
					Syscalls[i].Current = Syscalls[i].Default;
			return rc;
		}
		// If zName is specified, operate on only the one system call specified.
		for (int i = 0; i < _lengthof(Syscalls); i++)
		{
			if (!_strcmp(name, Syscalls[i].Name))
			{
				if (!Syscalls[i].Default)
					Syscalls[i].Default = Syscalls[i].Current;
				rc = RC_OK;
				if (!newFunc) newFunc = Syscalls[i].Default;
				Syscalls[i].Current = newFunc;
				break;
			}
		}
		return rc;
	}

	syscall_ptr UnixVSystem::GetSystemCall(const char *name)
	{
		for (int i = 0; i < _lengthof(Syscalls); i++)
			if (!_strcmp(name, Syscalls[i].Name)) return Syscalls[i].Current;
		return nullptr;
	}

	const char *UnixVSystem::NextSystemCall(const char *name)
	{
		int i = -1;
		if (name)
			for (i = 0; i < _lengthof(Syscalls)-1; i++)
				if (!strcmp(name, Syscalls[i].Name)) break;
		for (i++; i < _lengthof(Syscalls); i++)
			if (Syscalls[i].Current) return Syscalls[i].Name;
		return 0;
	}

#pragma endregion

#pragma region Preamble

	static int robust_open(const char *z, int f, mode_t m)
	{
		int fd;
		mode_t m2 = (m ? m : DEFAULT_FILE_PERMISSIONS);
		do
		{
#if defined(O_CLOEXEC)
			fd = osOpen(z,f|O_CLOEXEC,m2);
#else
			fd = osOpen(z,f,m2);
#endif
		} while (fd < 0 && errno == EINTR);
		if (fd >= 0)
		{
			if (m != 0)
			{
				struct stat statbuf;
				if (osFstat(fd, &statbuf) == 0  && statbuf.st_size == 0 && (statbuf.st_mode&0777) != m)
					osFchmod(fd, m);
			}
#if defined(FD_CLOEXEC) && (!defined(O_CLOEXEC) || O_CLOEXEC == 0)
			osFcntl(fd, F_SETFD, osFcntl(fd, F_GETFD, 0) | FD_CLOEXEC);
#endif
		}
		return fd;
	}

	static void unixEnterMutex() { _mutex_enter(_mutex_alloc(MUTEX_STATIC_MASTER)); }
	static void unixLeaveMutex() { _mutex_leave(_mutex_alloc(MUTEX_STATIC_MASTER)); }
#ifdef _DEBUG
	static int unixMutexHeld() { return _mutex_held(_mutex_alloc(MUTEX_STATIC_MASTER)); }
#endif

#if defined(_TEST) && defined(_DEBUG)
	// Helper function for printing out trace information from debugging binaries. This returns the string represetation of the supplied integer lock-type.
	static const char *FileLockNames(int fileLock)
	{
		switch (fileLock)
		{
		case NO_LOCK: return "NONE";
		case SHARED_LOCK: return "SHARED";
		case RESERVED_LOCK: return "RESERVED";
		case PENDING_LOCK: return "PENDING";
		case EXCLUSIVE_LOCK: return "EXCLUSIVE";
		}
		return "ERROR";
	}
#endif

#ifdef LOCK_TRACE
	static int lockTrace(int fd, int op, struct flock *p)
	{
		char *opName, *typeName;
		int s;
		int savedErrno;
		if (op == F_GETLK) opName = "GETLK";
		else if (op == F_SETLK) opName = "SETLK";
		else
		{
			s = osFcntl(fd, op, p);
			sqlite3DebugPrintf("fcntl unknown %d %d %d\n", fd, op, s);
			return s;
		}
		if (p->l_type == F_RDLCK) typeName = "RDLCK";
		else if (p->l_type == F_WRLCK) typeName = "WRLCK";
		else if (p->l_type == F_UNLCK) typeName = "UNLCK";
		else _assert(0);
		_assert(p->l_whence == SEEK_SET);
		s = osFcntl(fd, op, p);
		savedErrno = errno;
		sqlite3DebugPrintf("fcntl %d %d %s %s %d %d %d %d\n", threadid, fd, opName, typeName, (int)p->l_start, (int)p->l_len, (int)p->l_pid, s);
		if (s == (-1) && op == F_SETLK && (p->l_type == F_RDLCK || p->l_type == F_WRLCK))
		{
			struct flock l2;
			l2 = *p;
			osFcntl(fd, F_GETLK, &l2);
			if (l2.l_type == F_RDLCK) typeName = "RDLCK";
			else if (l2.l_type == F_WRLCK) typeName = "WRLCK";
			else if (l2.l_type == F_UNLCK) typeName = "UNLCK";
			else _assert(0);
			sqlite3DebugPrintf("fcntl-failure-reason: %s %d %d %d\n", typeName, (int)l2.l_start, (int)l2.l_len, (int)l2.l_pid);
		}
		errno = savedErrno;
		return s;
	}
#undef osFcntl
#define osFcntl lockTrace
#endif


	static int robust_ftruncate(int h, _int64 sz)
	{
		int rc;
		do { rc = osFtruncate(h, sz); } while (rc < 0 && errno == EINTR);
		return rc;
	}

	static int sqliteErrorFromPosixError(int posixError, int sqliteIOErr)
	{
		switch (posixError)
		{
#if 0
			// At one point this code was not commented out. In theory, this branch should never be hit, as this function should only be called after
			// a locking-related function (i.e. fcntl()) has returned non-zero with the value of errno as the first argument. Since a system call has failed,
			// errno should be non-zero.
			//
			// Despite this, if errno really is zero, we still don't want to return SQLITE_OK. The system call failed, and *some* SQLite error should be
			// propagated back to the caller. Commenting this branch out means errno==0 will be handled by the "default:" case below.
		case 0: 
			return RC_OK;
#endif
		case EAGAIN:
		case ETIMEDOUT:
		case EBUSY:
		case EINTR:
		case ENOLCK:  
			return RC_BUSY; // random NFS retry error, unless during file system support introspection, in which it actually means what it says
		case EACCES: 
			// EACCES is like EAGAIN during locking operations, but not any other time
			if (sqliteIOErr == RC_IOERR_LOCK || sqliteIOErr == RC_IOERR_UNLOCK || sqliteIOErr == RC_IOERR_RDLOCK || sqliteIOErr == RC_IOERR_CHECKRESERVEDLOCK)
				return RC_BUSY;
			// else fall through
		case EPERM: 
			return RC_PERM;
#if 0
			// EDEADLK is only possible if a call to fcntl(F_SETLKW) is made. And this module never makes such a call. And the code in SQLite itself 
			// asserts that SQLITE_IOERR_BLOCKED is never returned. For these reasons this case is also commented out. If the system does set errno to EDEADLK,
			// the default SQLITE_IOERR_XXX code will be returned.
		case EDEADLK:
			return RC_IOERR_BLOCKED;
#endif
#if EOPNOTSUPP != ENOTSUP // something went terribly awry, unless during file system support introspection, in which it actually means what it says
		case EOPNOTSUPP: 
#endif
#ifdef ENOTSUP // invalid fd, unless during file system support introspection, in which it actually means what it says
		case ENOTSUP: 
#endif
		case EIO:
		case EBADF:
		case EINVAL:
		case ENOTCONN:
		case ENODEV:
		case ENXIO:
		case ENOENT:
#ifdef ESTALE // ESTALE is not defined on Interix systems
		case ESTALE:
#endif
		case ENOSYS: // these should force the client to close the file and reconnect
		default: 
			return sqliteIOErr;
		}
	}

#pragma endregion

#pragma region VxWorks
	struct vxworksFileId
	{
		struct vxworksFileId *Next;		// Next in a list of them all
		int Refs;						// Number of references to this one
		int Names;						// Length of the zCanonicalName[] string
		char *CanonicalName;			// Canonical filename
	};
#if OS_VXWORKS

	// All unique filenames are held on a linked list headed by this variable:
	static struct vxworksFileId *vxworksFileList = 0;

	static int vxworksSimplifyName(char *z, int n)
	{
		int i, j;
		while( n>1 && z[n-1]=='/' ){ n--; }
		for(i=j=0; i<n; i++){
			if( z[i]=='/' ){
				if( z[i+1]=='/' ) continue;
				if( z[i+1]=='.' && i+2<n && z[i+2]=='/' ){
					i += 1;
					continue;
				}
				if( z[i+1]=='.' && i+3<n && z[i+2]=='.' && z[i+3]=='/' ){
					while( j>0 && z[j-1]!='/' ){ j--; }
					if( j>0 ){ j--; }
					i += 2;
					continue;
				}
			}
			z[j++] = z[i];
		}
		z[j] = 0;
		return j;
	}

	static struct vxworksFileId *vxworksFindFileId(const char *zAbsoluteName)
	{
		struct vxworksFileId *pNew;         /* search key and new file ID */
		struct vxworksFileId *pCandidate;   /* For looping over existing file IDs */
		int n;                              /* Length of zAbsoluteName string */

		assert( zAbsoluteName[0]=='/' );
		n = (int)strlen(zAbsoluteName);
		pNew = sqlite3_malloc( sizeof(*pNew) + (n+1) );
		if( pNew==0 ) return 0;
		pNew->zCanonicalName = (char*)&pNew[1];
		memcpy(pNew->zCanonicalName, zAbsoluteName, n+1);
		n = vxworksSimplifyName(pNew->zCanonicalName, n);

		/* Search for an existing entry that matching the canonical name.
		** If found, increment the reference count and return a pointer to
		** the existing file ID.
		*/
		unixEnterMutex();
		for(pCandidate=vxworksFileList; pCandidate; pCandidate=pCandidate->pNext){
			if( pCandidate->nName==n 
				&& memcmp(pCandidate->zCanonicalName, pNew->zCanonicalName, n)==0
				){
					sqlite3_free(pNew);
					pCandidate->nRef++;
					unixLeaveMutex();
					return pCandidate;
			}
		}

		/* No match was found.  We will make a new file ID */
		pNew->nRef = 1;
		pNew->nName = n;
		pNew->pNext = vxworksFileList;
		vxworksFileList = pNew;
		unixLeaveMutex();
		return pNew;
	}

	static void vxworksReleaseFileId(struct vxworksFileId *pId)
	{
		unixEnterMutex();
		assert( pId->nRef>0 );
		pId->nRef--;
		if( pId->nRef==0 ){
			struct vxworksFileId **pp;
			for(pp=&vxworksFileList; *pp && *pp!=pId; pp = &((*pp)->pNext)){}
			assert( *pp==pId );
			*pp = pId->pNext;
			sqlite3_free(pId);
		}
		unixLeaveMutex();
	}

#endif /* OS_VXWORKS */
#pragma endregion

#pragma region LOCKING
	{

#pragma region Posix Advisory Locking

		struct unixFileId
		{
			dev_t dev;                  /* Device number */
#if OS_VXWORKS
			struct vxworksFileId *pId;  /* Unique file ID for vxworks. */
#else
			ino_t ino;                  /* Inode number */
#endif
		};

		struct unixInodeInfo
		{
			struct unixFileId fileId;       /* The lookup key */
			int nShared;                    /* Number of SHARED locks held */
			unsigned char eFileLock;        /* One of SHARED_LOCK, RESERVED_LOCK etc. */
			unsigned char bProcessLock;     /* An exclusive process lock is held */
			int nRef;                       /* Number of pointers to this structure */
			unixShmNode *pShmNode;          /* Shared memory associated with this inode */
			int nLock;                      /* Number of outstanding file locks */
			UnixUnusedFd *pUnused;          /* Unused file descriptors to close */
			unixInodeInfo *pNext;           /* List of all unixInodeInfo objects */
			unixInodeInfo *pPrev;           /*    .... doubly linked */
#if ENABLE_LOCKING_STYLE
			unsigned long long sharedByte;  /* for AFP simulated shared lock */
#endif
#if OS_VXWORKS
			sem_t *pSem;                    /* Named POSIX semaphore */
			char aSemName[MAX_PATHNAME+2];  /* Name of that semaphore */
#endif
		};

		static unixInodeInfo *inodeList = 0;

#define unixLogError(a,b,c) unixLogErrorAtLine(a,b,c,__LINE__)
		static int unixLogErrorAtLine(
			int errcode,                    /* SQLite error code */
			const char *zFunc,              /* Name of OS function that failed */
			const char *zPath,              /* File path associated with error */
			int iLine                       /* Source line number where error occurred */
			)
		{
			char *zErr;                     /* Message from strerror() or equivalent */
			int iErrno = errno;             /* Saved syscall error number */

			/* If this is not a threadsafe build (SQLITE_THREADSAFE==0), then use
			** the strerror() function to obtain the human-readable error message
			** equivalent to errno. Otherwise, use strerror_r().
			*/ 
#if THREADSAFE && defined(HAVE_STRERROR_R)
			char aErr[80];
			memset(aErr, 0, sizeof(aErr));
			zErr = aErr;

			/* If STRERROR_R_CHAR_P (set by autoconf scripts) or __USE_GNU is defined,
			** assume that the system provides the GNU version of strerror_r() that
			** returns a pointer to a buffer containing the error message. That pointer 
			** may point to aErr[], or it may point to some static storage somewhere. 
			** Otherwise, assume that the system provides the POSIX version of 
			** strerror_r(), which always writes an error message into aErr[].
			**
			** If the code incorrectly assumes that it is the POSIX version that is
			** available, the error message will often be an empty string. Not a
			** huge problem. Incorrectly concluding that the GNU version is available 
			** could lead to a segfault though.
			*/
#if defined(STRERROR_R_CHAR_P) || defined(__USE_GNU)
			zErr = 
# endif
				strerror_r(iErrno, aErr, sizeof(aErr)-1);

#elif THREADSAFE
			/* This is a threadsafe build, but strerror_r() is not available. */
			zErr = "";
#else
			/* Non-threadsafe build, use strerror(). */
			zErr = strerror(iErrno);
#endif

			assert( errcode!=SQLITE_OK );
			if( zPath==0 ) zPath = "";
			sqlite3_log(errcode,
				"os_unix.c:%d: (%d) %s(%s) - %s",
				iLine, iErrno, zFunc, zPath, zErr
				);

			return errcode;
		}

		static void robust_close(unixFile *pFile, int h, int lineno){
			if( osClose(h) ){
				unixLogErrorAtLine(SQLITE_IOERR_CLOSE, "close",
					pFile ? pFile->zPath : 0, lineno);
			}
		}

		static void closePendingFds(unixFile *pFile){
			unixInodeInfo *pInode = pFile->pInode;
			UnixUnusedFd *p;
			UnixUnusedFd *pNext;
			for(p=pInode->pUnused; p; p=pNext){
				pNext = p->pNext;
				robust_close(pFile, p->fd, __LINE__);
				sqlite3_free(p);
			}
			pInode->pUnused = 0;
		}

		static void releaseInodeInfo(unixFile *pFile){
			unixInodeInfo *pInode = pFile->pInode;
			assert( unixMutexHeld() );
			if( ALWAYS(pInode) ){
				pInode->nRef--;
				if( pInode->nRef==0 ){
					assert( pInode->pShmNode==0 );
					closePendingFds(pFile);
					if( pInode->pPrev ){
						assert( pInode->pPrev->pNext==pInode );
						pInode->pPrev->pNext = pInode->pNext;
					}else{
						assert( inodeList==pInode );
						inodeList = pInode->pNext;
					}
					if( pInode->pNext ){
						assert( pInode->pNext->pPrev==pInode );
						pInode->pNext->pPrev = pInode->pPrev;
					}
					sqlite3_free(pInode);
				}
			}
		}

		static int findInodeInfo(
			unixFile *pFile,               /* Unix file with file desc used in the key */
			unixInodeInfo **ppInode        /* Return the unixInodeInfo object here */
			){
				int rc;                        /* System call return code */
				int fd;                        /* The file descriptor for pFile */
				struct unixFileId fileId;      /* Lookup key for the unixInodeInfo */
				struct stat statbuf;           /* Low-level file information */
				unixInodeInfo *pInode = 0;     /* Candidate unixInodeInfo object */

				assert( unixMutexHeld() );

				/* Get low-level information about the file that we can used to
				** create a unique name for the file.
				*/
				fd = pFile->h;
				rc = osFstat(fd, &statbuf);
				if( rc!=0 ){
					pFile->lastErrno = errno;
#ifdef EOVERFLOW
					if( pFile->lastErrno==EOVERFLOW ) return SQLITE_NOLFS;
#endif
					return SQLITE_IOERR;
				}

#ifdef __APPLE__
				/* On OS X on an msdos filesystem, the inode number is reported
				** incorrectly for zero-size files.  See ticket #3260.  To work
				** around this problem (we consider it a bug in OS X, not SQLite)
				** we always increase the file size to 1 by writing a single byte
				** prior to accessing the inode number.  The one byte written is
				** an ASCII 'S' character which also happens to be the first byte
				** in the header of every SQLite database.  In this way, if there
				** is a race condition such that another thread has already populated
				** the first page of the database, no damage is done.
				*/
				if( statbuf.st_size==0 && (pFile->fsFlags & SQLITE_FSFLAGS_IS_MSDOS)!=0 ){
					do{ rc = osWrite(fd, "S", 1); }while( rc<0 && errno==EINTR );
					if( rc!=1 ){
						pFile->lastErrno = errno;
						return SQLITE_IOERR;
					}
					rc = osFstat(fd, &statbuf);
					if( rc!=0 ){
						pFile->lastErrno = errno;
						return SQLITE_IOERR;
					}
				}
#endif

				memset(&fileId, 0, sizeof(fileId));
				fileId.dev = statbuf.st_dev;
#if OS_VXWORKS
				fileId.pId = pFile->pId;
#else
				fileId.ino = statbuf.st_ino;
#endif
				pInode = inodeList;
				while( pInode && memcmp(&fileId, &pInode->fileId, sizeof(fileId)) ){
					pInode = pInode->pNext;
				}
				if( pInode==0 ){
					pInode = sqlite3_malloc( sizeof(*pInode) );
					if( pInode==0 ){
						return SQLITE_NOMEM;
					}
					memset(pInode, 0, sizeof(*pInode));
					memcpy(&pInode->fileId, &fileId, sizeof(fileId));
					pInode->nRef = 1;
					pInode->pNext = inodeList;
					pInode->pPrev = 0;
					if( inodeList ) inodeList->pPrev = pInode;
					inodeList = pInode;
				}else{
					pInode->nRef++;
				}
				*ppInode = pInode;
				return SQLITE_OK;
		}


		static int unixCheckReservedLock(sqlite3_file *id, int *pResOut){
			int rc = SQLITE_OK;
			int reserved = 0;
			unixFile *pFile = (unixFile*)id;

			SimulateIOError( return SQLITE_IOERR_CHECKRESERVEDLOCK; );

			assert( pFile );
			unixEnterMutex(); /* Because pFile->pInode is shared across threads */

			/* Check if a thread in this process holds such a lock */
			if( pFile->pInode->eFileLock>SHARED_LOCK ){
				reserved = 1;
			}

			/* Otherwise see if some other process holds it.
			*/
#ifndef __DJGPP__
			if( !reserved && !pFile->pInode->bProcessLock ){
				struct flock lock;
				lock.l_whence = SEEK_SET;
				lock.l_start = RESERVED_BYTE;
				lock.l_len = 1;
				lock.l_type = F_WRLCK;
				if( osFcntl(pFile->h, F_GETLK, &lock) ){
					rc = SQLITE_IOERR_CHECKRESERVEDLOCK;
					pFile->lastErrno = errno;
				} else if( lock.l_type!=F_UNLCK ){
					reserved = 1;
				}
			}
#endif

			unixLeaveMutex();
			OSTRACE(("_TEST WR-LOCK %d %d %d (unix)\n", pFile->h, rc, reserved));

			*pResOut = reserved;
			return rc;
		}

		static int unixFileLock(unixFile *pFile, struct flock *pLock){
			int rc;
			unixInodeInfo *pInode = pFile->pInode;
			assert( unixMutexHeld() );
			assert( pInode!=0 );
			if( ((pFile->ctrlFlags & UNIXFILE_EXCL)!=0 || pInode->bProcessLock)
				&& ((pFile->ctrlFlags & UNIXFILE_RDONLY)==0)
				){
					if( pInode->bProcessLock==0 ){
						struct flock lock;
						assert( pInode->nLock==0 );
						lock.l_whence = SEEK_SET;
						lock.l_start = SHARED_FIRST;
						lock.l_len = SHARED_SIZE;
						lock.l_type = F_WRLCK;
						rc = osFcntl(pFile->h, F_SETLK, &lock);
						if( rc<0 ) return rc;
						pInode->bProcessLock = 1;
						pInode->nLock++;
					}else{
						rc = 0;
					}
			}else{
				rc = osFcntl(pFile->h, F_SETLK, pLock);
			}
			return rc;
		}

		static int unixLock(sqlite3_file *id, int eFileLock)
		{
			/* The following describes the implementation of the various locks and
			** lock transitions in terms of the POSIX advisory shared and exclusive
			** lock primitives (called read-locks and write-locks below, to avoid
			** confusion with SQLite lock names). The algorithms are complicated
			** slightly in order to be compatible with windows systems simultaneously
			** accessing the same database file, in case that is ever required.
			**
			** Symbols defined in os.h indentify the 'pending byte' and the 'reserved
			** byte', each single bytes at well known offsets, and the 'shared byte
			** range', a range of 510 bytes at a well known offset.
			**
			** To obtain a SHARED lock, a read-lock is obtained on the 'pending
			** byte'.  If this is successful, a random byte from the 'shared byte
			** range' is read-locked and the lock on the 'pending byte' released.
			**
			** A process may only obtain a RESERVED lock after it has a SHARED lock.
			** A RESERVED lock is implemented by grabbing a write-lock on the
			** 'reserved byte'. 
			**
			** A process may only obtain a PENDING lock after it has obtained a
			** SHARED lock. A PENDING lock is implemented by obtaining a write-lock
			** on the 'pending byte'. This ensures that no new SHARED locks can be
			** obtained, but existing SHARED locks are allowed to persist. A process
			** does not have to obtain a RESERVED lock on the way to a PENDING lock.
			** This property is used by the algorithm for rolling back a journal file
			** after a crash.
			**
			** An EXCLUSIVE lock, obtained after a PENDING lock is held, is
			** implemented by obtaining a write-lock on the entire 'shared byte
			** range'. Since all other locks require a read-lock on one of the bytes
			** within this range, this ensures that no other locks are held on the
			** database. 
			**
			** The reason a single byte cannot be used instead of the 'shared byte
			** range' is that some versions of windows do not support read-locks. By
			** locking a random byte from a range, concurrent SHARED locks may exist
			** even if the locking primitive used is always a write-lock.
			*/
			int rc = SQLITE_OK;
			unixFile *pFile = (unixFile*)id;
			unixInodeInfo *pInode;
			struct flock lock;
			int tErrno = 0;

			assert( pFile );
			OSTRACE(("LOCK    %d %s was %s(%s,%d) pid=%d (unix)\n", pFile->h,
				azFileLock(eFileLock), azFileLock(pFile->eFileLock),
				azFileLock(pFile->pInode->eFileLock), pFile->pInode->nShared , getpid()));

			/* If there is already a lock of this type or more restrictive on the
			** unixFile, do nothing. Don't use the end_lock: exit path, as
			** unixEnterMutex() hasn't been called yet.
			*/
			if( pFile->eFileLock>=eFileLock ){
				OSTRACE(("LOCK    %d %s ok (already held) (unix)\n", pFile->h,
					azFileLock(eFileLock)));
				return SQLITE_OK;
			}

			/* Make sure the locking sequence is correct.
			**  (1) We never move from unlocked to anything higher than shared lock.
			**  (2) SQLite never explicitly requests a pendig lock.
			**  (3) A shared lock is always held when a reserve lock is requested.
			*/
			assert( pFile->eFileLock!=NO_LOCK || eFileLock==SHARED_LOCK );
			assert( eFileLock!=PENDING_LOCK );
			assert( eFileLock!=RESERVED_LOCK || pFile->eFileLock==SHARED_LOCK );

			/* This mutex is needed because pFile->pInode is shared across threads
			*/
			unixEnterMutex();
			pInode = pFile->pInode;

			/* If some thread using this PID has a lock via a different unixFile*
			** handle that precludes the requested lock, return BUSY.
			*/
			if( (pFile->eFileLock!=pInode->eFileLock && 
				(pInode->eFileLock>=PENDING_LOCK || eFileLock>SHARED_LOCK))
				){
					rc = SQLITE_BUSY;
					goto end_lock;
			}

			/* If a SHARED lock is requested, and some thread using this PID already
			** has a SHARED or RESERVED lock, then increment reference counts and
			** return SQLITE_OK.
			*/
			if( eFileLock==SHARED_LOCK && 
				(pInode->eFileLock==SHARED_LOCK || pInode->eFileLock==RESERVED_LOCK) ){
					assert( eFileLock==SHARED_LOCK );
					assert( pFile->eFileLock==0 );
					assert( pInode->nShared>0 );
					pFile->eFileLock = SHARED_LOCK;
					pInode->nShared++;
					pInode->nLock++;
					goto end_lock;
			}


			/* A PENDING lock is needed before acquiring a SHARED lock and before
			** acquiring an EXCLUSIVE lock.  For the SHARED lock, the PENDING will
			** be released.
			*/
			lock.l_len = 1L;
			lock.l_whence = SEEK_SET;
			if( eFileLock==SHARED_LOCK 
				|| (eFileLock==EXCLUSIVE_LOCK && pFile->eFileLock<PENDING_LOCK)
				){
					lock.l_type = (eFileLock==SHARED_LOCK?F_RDLCK:F_WRLCK);
					lock.l_start = PENDING_BYTE;
					if( unixFileLock(pFile, &lock) ){
						tErrno = errno;
						rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK);
						if( rc!=SQLITE_BUSY ){
							pFile->lastErrno = tErrno;
						}
						goto end_lock;
					}
			}


			/* If control gets to this point, then actually go ahead and make
			** operating system calls for the specified lock.
			*/
			if( eFileLock==SHARED_LOCK ){
				assert( pInode->nShared==0 );
				assert( pInode->eFileLock==0 );
				assert( rc==SQLITE_OK );

				/* Now get the read-lock */
				lock.l_start = SHARED_FIRST;
				lock.l_len = SHARED_SIZE;
				if( unixFileLock(pFile, &lock) ){
					tErrno = errno;
					rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK);
				}

				/* Drop the temporary PENDING lock */
				lock.l_start = PENDING_BYTE;
				lock.l_len = 1L;
				lock.l_type = F_UNLCK;
				if( unixFileLock(pFile, &lock) && rc==SQLITE_OK ){
					/* This could happen with a network mount */
					tErrno = errno;
					rc = SQLITE_IOERR_UNLOCK; 
				}

				if( rc ){
					if( rc!=SQLITE_BUSY ){
						pFile->lastErrno = tErrno;
					}
					goto end_lock;
				}else{
					pFile->eFileLock = SHARED_LOCK;
					pInode->nLock++;
					pInode->nShared = 1;
				}
			}else if( eFileLock==EXCLUSIVE_LOCK && pInode->nShared>1 ){
				/* We are trying for an exclusive lock but another thread in this
				** same process is still holding a shared lock. */
				rc = SQLITE_BUSY;
			}else{
				/* The request was for a RESERVED or EXCLUSIVE lock.  It is
				** assumed that there is a SHARED or greater lock on the file
				** already.
				*/
				assert( 0!=pFile->eFileLock );
				lock.l_type = F_WRLCK;

				assert( eFileLock==RESERVED_LOCK || eFileLock==EXCLUSIVE_LOCK );
				if( eFileLock==RESERVED_LOCK ){
					lock.l_start = RESERVED_BYTE;
					lock.l_len = 1L;
				}else{
					lock.l_start = SHARED_FIRST;
					lock.l_len = SHARED_SIZE;
				}

				if( unixFileLock(pFile, &lock) ){
					tErrno = errno;
					rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK);
					if( rc!=SQLITE_BUSY ){
						pFile->lastErrno = tErrno;
					}
				}
			}


#ifdef _DEBUG
			/* Set up the transaction-counter change checking flags when
			** transitioning from a SHARED to a RESERVED lock.  The change
			** from SHARED to RESERVED marks the beginning of a normal
			** write operation (not a hot journal rollback).
			*/
			if( rc==SQLITE_OK
				&& pFile->eFileLock<=SHARED_LOCK
				&& eFileLock==RESERVED_LOCK
				){
					pFile->transCntrChng = 0;
					pFile->dbUpdate = 0;
					pFile->inNormalWrite = 1;
			}
#endif

			if( rc==SQLITE_OK ){
				pFile->eFileLock = eFileLock;
				pInode->eFileLock = eFileLock;
			}else if( eFileLock==EXCLUSIVE_LOCK ){
				pFile->eFileLock = PENDING_LOCK;
				pInode->eFileLock = PENDING_LOCK;
			}

end_lock:
			unixLeaveMutex();
			OSTRACE(("LOCK    %d %s %s (unix)\n", pFile->h, azFileLock(eFileLock), 
				rc==SQLITE_OK ? "ok" : "failed"));
			return rc;
		}

		static void setPendingFd(unixFile *pFile){
			unixInodeInfo *pInode = pFile->pInode;
			UnixUnusedFd *p = pFile->pUnused;
			p->pNext = pInode->pUnused;
			pInode->pUnused = p;
			pFile->h = -1;
			pFile->pUnused = 0;
		}

		static int posixUnlock(sqlite3_file *id, int eFileLock, int handleNFSUnlock){
			unixFile *pFile = (unixFile*)id;
			unixInodeInfo *pInode;
			struct flock lock;
			int rc = SQLITE_OK;

			assert( pFile );
			OSTRACE(("UNLOCK  %d %d was %d(%d,%d) pid=%d (unix)\n", pFile->h, eFileLock,
				pFile->eFileLock, pFile->pInode->eFileLock, pFile->pInode->nShared,
				getpid()));

			assert( eFileLock<=SHARED_LOCK );
			if( pFile->eFileLock<=eFileLock ){
				return SQLITE_OK;
			}
			unixEnterMutex();
			pInode = pFile->pInode;
			assert( pInode->nShared!=0 );
			if( pFile->eFileLock>SHARED_LOCK ){
				assert( pInode->eFileLock==pFile->eFileLock );

#ifdef _DEBUG
				/* When reducing a lock such that other processes can start
				** reading the database file again, make sure that the
				** transaction counter was updated if any part of the database
				** file changed.  If the transaction counter is not updated,
				** other connections to the same file might not realize that
				** the file has changed and hence might not know to flush their
				** cache.  The use of a stale cache can lead to database corruption.
				*/
				pFile->inNormalWrite = 0;
#endif

				/* downgrading to a shared lock on NFS involves clearing the write lock
				** before establishing the readlock - to avoid a race condition we downgrade
				** the lock in 2 blocks, so that part of the range will be covered by a 
				** write lock until the rest is covered by a read lock:
				**  1:   [WWWWW]
				**  2:   [....W]
				**  3:   [RRRRW]
				**  4:   [RRRR.]
				*/
				if( eFileLock==SHARED_LOCK ){

#if !defined(__APPLE__) || !SQLITE_ENABLE_LOCKING_STYLE
					(void)handleNFSUnlock;
					assert( handleNFSUnlock==0 );
#endif
#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
					if( handleNFSUnlock ){
						int tErrno;               /* Error code from system call errors */
						off_t divSize = SHARED_SIZE - 1;

						lock.l_type = F_UNLCK;
						lock.l_whence = SEEK_SET;
						lock.l_start = SHARED_FIRST;
						lock.l_len = divSize;
						if( unixFileLock(pFile, &lock)==(-1) ){
							tErrno = errno;
							rc = SQLITE_IOERR_UNLOCK;
							if( IS_LOCK_ERROR(rc) ){
								pFile->lastErrno = tErrno;
							}
							goto end_unlock;
						}
						lock.l_type = F_RDLCK;
						lock.l_whence = SEEK_SET;
						lock.l_start = SHARED_FIRST;
						lock.l_len = divSize;
						if( unixFileLock(pFile, &lock)==(-1) ){
							tErrno = errno;
							rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_RDLOCK);
							if( IS_LOCK_ERROR(rc) ){
								pFile->lastErrno = tErrno;
							}
							goto end_unlock;
						}
						lock.l_type = F_UNLCK;
						lock.l_whence = SEEK_SET;
						lock.l_start = SHARED_FIRST+divSize;
						lock.l_len = SHARED_SIZE-divSize;
						if( unixFileLock(pFile, &lock)==(-1) ){
							tErrno = errno;
							rc = SQLITE_IOERR_UNLOCK;
							if( IS_LOCK_ERROR(rc) ){
								pFile->lastErrno = tErrno;
							}
							goto end_unlock;
						}
					}else
#endif /* defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE */
					{
						lock.l_type = F_RDLCK;
						lock.l_whence = SEEK_SET;
						lock.l_start = SHARED_FIRST;
						lock.l_len = SHARED_SIZE;
						if( unixFileLock(pFile, &lock) ){
							/* In theory, the call to unixFileLock() cannot fail because another
							** process is holding an incompatible lock. If it does, this 
							** indicates that the other process is not following the locking
							** protocol. If this happens, return SQLITE_IOERR_RDLOCK. Returning
							** SQLITE_BUSY would confuse the upper layer (in practice it causes 
							** an assert to fail). */ 
							rc = SQLITE_IOERR_RDLOCK;
							pFile->lastErrno = errno;
							goto end_unlock;
						}
					}
				}
				lock.l_type = F_UNLCK;
				lock.l_whence = SEEK_SET;
				lock.l_start = PENDING_BYTE;
				lock.l_len = 2L;  assert( PENDING_BYTE+1==RESERVED_BYTE );
				if( unixFileLock(pFile, &lock)==0 ){
					pInode->eFileLock = SHARED_LOCK;
				}else{
					rc = SQLITE_IOERR_UNLOCK;
					pFile->lastErrno = errno;
					goto end_unlock;
				}
			}
			if( eFileLock==NO_LOCK ){
				/* Decrement the shared lock counter.  Release the lock using an
				** OS call only when all threads in this same process have released
				** the lock.
				*/
				pInode->nShared--;
				if( pInode->nShared==0 ){
					lock.l_type = F_UNLCK;
					lock.l_whence = SEEK_SET;
					lock.l_start = lock.l_len = 0L;
					if( unixFileLock(pFile, &lock)==0 ){
						pInode->eFileLock = NO_LOCK;
					}else{
						rc = SQLITE_IOERR_UNLOCK;
						pFile->lastErrno = errno;
						pInode->eFileLock = NO_LOCK;
						pFile->eFileLock = NO_LOCK;
					}
				}

				/* Decrement the count of locks against this same file.  When the
				** count reaches zero, close any other file descriptors whose close
				** was deferred because of outstanding locks.
				*/
				pInode->nLock--;
				assert( pInode->nLock>=0 );
				if( pInode->nLock==0 ){
					closePendingFds(pFile);
				}
			}

end_unlock:
			unixLeaveMutex();
			if( rc==SQLITE_OK ) pFile->eFileLock = eFileLock;
			return rc;
		}

		static int unixUnlock(sqlite3_file *id, int eFileLock){
			return posixUnlock(id, eFileLock, 0);
		}

		static int closeUnixFile(sqlite3_file *id){
			unixFile *pFile = (unixFile*)id;
			if( pFile->h>=0 ){
				robust_close(pFile, pFile->h, __LINE__);
				pFile->h = -1;
			}
#if OS_VXWORKS
			if( pFile->pId ){
				if( pFile->ctrlFlags & UNIXFILE_DELETE ){
					osUnlink(pFile->pId->zCanonicalName);
				}
				vxworksReleaseFileId(pFile->pId);
				pFile->pId = 0;
			}
#endif
			OSTRACE(("CLOSE   %-3d\n", pFile->h));
			OpenCounter(-1);
			sqlite3_free(pFile->pUnused);
			memset(pFile, 0, sizeof(unixFile));
			return SQLITE_OK;
		}

		static int unixClose(sqlite3_file *id){
			int rc = SQLITE_OK;
			unixFile *pFile = (unixFile *)id;
			unixUnlock(id, NO_LOCK);
			unixEnterMutex();

			/* unixFile.pInode is always valid here. Otherwise, a different close
			** routine (e.g. nolockClose()) would be called instead.
			*/
			assert( pFile->pInode->nLock>0 || pFile->pInode->bProcessLock==0 );
			if( ALWAYS(pFile->pInode) && pFile->pInode->nLock ){
				/* If there are outstanding locks, do not actually close the file just
				** yet because that would clear those locks.  Instead, add the file
				** descriptor to pInode->pUnused list.  It will be automatically closed 
				** when the last lock is cleared.
				*/
				setPendingFd(pFile);
			}
			releaseInodeInfo(pFile);
			rc = closeUnixFile(id);
			unixLeaveMutex();
			return rc;
		}

#pragma endregion

#pragma region No-op Locking

		static int nolockCheckReservedLock(sqlite3_file *NotUsed, int *pResOut)
		{
			*pResOut = 0;
			return SQLITE_OK;
		}
		static int nolockLock(sqlite3_file *NotUsed, int NotUsed2)
		{
			return SQLITE_OK;
		}

		static int nolockUnlock(sqlite3_file *NotUsed, int NotUsed2)
		{
			return SQLITE_OK;
		}

		static int nolockClose(sqlite3_file *id)
		{
			return closeUnixFile(id);
		}

#pragma endregion

#pragma region Dot-file Locking

#define DOTLOCK_SUFFIX ".lock"

		static int dotlockCheckReservedLock(sqlite3_file *id, int *pResOut)
		{
			int rc = SQLITE_OK;
			int reserved = 0;
			unixFile *pFile = (unixFile*)id;

			SimulateIOError( return SQLITE_IOERR_CHECKRESERVEDLOCK; );

			assert( pFile );

			/* Check if a thread in this process holds such a lock */
			if( pFile->eFileLock>SHARED_LOCK ){
				/* Either this connection or some other connection in the same process
				** holds a lock on the file.  No need to check further. */
				reserved = 1;
			}else{
				/* The lock is held if and only if the lockfile exists */
				const char *zLockFile = (const char*)pFile->lockingContext;
				reserved = osAccess(zLockFile, 0)==0;
			}
			OSTRACE(("_TEST WR-LOCK %d %d %d (dotlock)\n", pFile->h, rc, reserved));
			*pResOut = reserved;
			return rc;
		}

		static int dotlockLock(sqlite3_file *id, int eFileLock) {
			unixFile *pFile = (unixFile*)id;
			char *zLockFile = (char *)pFile->lockingContext;
			int rc = SQLITE_OK;


			/* If we have any lock, then the lock file already exists.  All we have
			** to do is adjust our internal record of the lock level.
			*/
			if( pFile->eFileLock > NO_LOCK ){
				pFile->eFileLock = eFileLock;
				/* Always update the timestamp on the old file */
#ifdef HAVE_UTIME
				utime(zLockFile, NULL);
#else
				utimes(zLockFile, NULL);
#endif
				return SQLITE_OK;
			}

			/* grab an exclusive lock */
			rc = osMkdir(zLockFile, 0777);
			if( rc<0 ){
				/* failed to open/create the lock directory */
				int tErrno = errno;
				if( EEXIST == tErrno ){
					rc = SQLITE_BUSY;
				} else {
					rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK);
					if( IS_LOCK_ERROR(rc) ){
						pFile->lastErrno = tErrno;
					}
				}
				return rc;
			} 

			/* got it, set the type and return ok */
			pFile->eFileLock = eFileLock;
			return rc;
		}

		static int dotlockUnlock(sqlite3_file *id, int eFileLock) {
			unixFile *pFile = (unixFile*)id;
			char *zLockFile = (char *)pFile->lockingContext;
			int rc;

			assert( pFile );
			OSTRACE(("UNLOCK  %d %d was %d pid=%d (dotlock)\n", pFile->h, eFileLock,
				pFile->eFileLock, getpid()));
			assert( eFileLock<=SHARED_LOCK );

			/* no-op if possible */
			if( pFile->eFileLock==eFileLock ){
				return SQLITE_OK;
			}

			/* To downgrade to shared, simply update our internal notion of the
			** lock state.  No need to mess with the file on disk.
			*/
			if( eFileLock==SHARED_LOCK ){
				pFile->eFileLock = SHARED_LOCK;
				return SQLITE_OK;
			}

			/* To fully unlock the database, delete the lock file */
			assert( eFileLock==NO_LOCK );
			rc = osRmdir(zLockFile);
			if( rc<0 && errno==ENOTDIR ) rc = osUnlink(zLockFile);
			if( rc<0 ){
				int tErrno = errno;
				rc = 0;
				if( ENOENT != tErrno ){
					rc = SQLITE_IOERR_UNLOCK;
				}
				if( IS_LOCK_ERROR(rc) ){
					pFile->lastErrno = tErrno;
				}
				return rc; 
			}
			pFile->eFileLock = NO_LOCK;
			return SQLITE_OK;
		}

		static int dotlockClose(sqlite3_file *id) {
			int rc = SQLITE_OK;
			if( id ){
				unixFile *pFile = (unixFile*)id;
				dotlockUnlock(id, NO_LOCK);
				sqlite3_free(pFile->lockingContext);
				rc = closeUnixFile(id);
			}
			return rc;
		}

#pragma endregion

#pragma region Flock Locking

#if ENABLE_LOCKING_STYLE && !OS_VXWORKS

#ifdef EINTR
		static int robust_flock(int fd, int op){
			int rc;
			do{ rc = flock(fd,op); }while( rc<0 && errno==EINTR );
			return rc;
		}
#else
# define robust_flock(a,b) flock(a,b)
#endif

		static int flockCheckReservedLock(sqlite3_file *id, int *pResOut){
			int rc = SQLITE_OK;
			int reserved = 0;
			unixFile *pFile = (unixFile*)id;

			SimulateIOError( return SQLITE_IOERR_CHECKRESERVEDLOCK; );

			assert( pFile );

			/* Check if a thread in this process holds such a lock */
			if( pFile->eFileLock>SHARED_LOCK ){
				reserved = 1;
			}

			/* Otherwise see if some other process holds it. */
			if( !reserved ){
				/* attempt to get the lock */
				int lrc = robust_flock(pFile->h, LOCK_EX | LOCK_NB);
				if( !lrc ){
					/* got the lock, unlock it */
					lrc = robust_flock(pFile->h, LOCK_UN);
					if ( lrc ) {
						int tErrno = errno;
						/* unlock failed with an error */
						lrc = SQLITE_IOERR_UNLOCK; 
						if( IS_LOCK_ERROR(lrc) ){
							pFile->lastErrno = tErrno;
							rc = lrc;
						}
					}
				} else {
					int tErrno = errno;
					reserved = 1;
					/* someone else might have it reserved */
					lrc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK); 
					if( IS_LOCK_ERROR(lrc) ){
						pFile->lastErrno = tErrno;
						rc = lrc;
					}
				}
			}
			OSTRACE(("_TEST WR-LOCK %d %d %d (flock)\n", pFile->h, rc, reserved));

#ifdef SQLITE_IGNORE_FLOCK_LOCK_ERRORS
			if( (rc & SQLITE_IOERR) == SQLITE_IOERR ){
				rc = SQLITE_OK;
				reserved=1;
			}
#endif /* SQLITE_IGNORE_FLOCK_LOCK_ERRORS */
			*pResOut = reserved;
			return rc;
		}

		static int flockLock(sqlite3_file *id, int eFileLock) {
			int rc = SQLITE_OK;
			unixFile *pFile = (unixFile*)id;

			assert( pFile );

			/* if we already have a lock, it is exclusive.  
			** Just adjust level and punt on outta here. */
			if (pFile->eFileLock > NO_LOCK) {
				pFile->eFileLock = eFileLock;
				return SQLITE_OK;
			}

			/* grab an exclusive lock */

			if (robust_flock(pFile->h, LOCK_EX | LOCK_NB)) {
				int tErrno = errno;
				/* didn't get, must be busy */
				rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_LOCK);
				if( IS_LOCK_ERROR(rc) ){
					pFile->lastErrno = tErrno;
				}
			} else {
				/* got it, set the type and return ok */
				pFile->eFileLock = eFileLock;
			}
			OSTRACE(("LOCK    %d %s %s (flock)\n", pFile->h, azFileLock(eFileLock), 
				rc==SQLITE_OK ? "ok" : "failed"));
#ifdef SQLITE_IGNORE_FLOCK_LOCK_ERRORS
			if( (rc & SQLITE_IOERR) == SQLITE_IOERR ){
				rc = SQLITE_BUSY;
			}
#endif /* SQLITE_IGNORE_FLOCK_LOCK_ERRORS */
			return rc;
		}

		static int flockUnlock(sqlite3_file *id, int eFileLock) {
			unixFile *pFile = (unixFile*)id;

			assert( pFile );
			OSTRACE(("UNLOCK  %d %d was %d pid=%d (flock)\n", pFile->h, eFileLock,
				pFile->eFileLock, getpid()));
			assert( eFileLock<=SHARED_LOCK );

			/* no-op if possible */
			if( pFile->eFileLock==eFileLock ){
				return SQLITE_OK;
			}

			/* shared can just be set because we always have an exclusive */
			if (eFileLock==SHARED_LOCK) {
				pFile->eFileLock = eFileLock;
				return SQLITE_OK;
			}

			/* no, really, unlock. */
			if( robust_flock(pFile->h, LOCK_UN) ){
#ifdef SQLITE_IGNORE_FLOCK_LOCK_ERRORS
				return SQLITE_OK;
#endif /* SQLITE_IGNORE_FLOCK_LOCK_ERRORS */
				return SQLITE_IOERR_UNLOCK;
			}else{
				pFile->eFileLock = NO_LOCK;
				return SQLITE_OK;
			}
		}


		static int flockClose(sqlite3_file *id) {
			int rc = SQLITE_OK;
			if( id ){
				flockUnlock(id, NO_LOCK);
				rc = closeUnixFile(id);
			}
			return rc;
		}

#endif
#pragma endregion

#pragma region Named Semaphore Locking
#if OS_VXWORKS

		static int semCheckReservedLock(sqlite3_file *id, int *pResOut) {
			int rc = SQLITE_OK;
			int reserved = 0;
			unixFile *pFile = (unixFile*)id;

			SimulateIOError( return SQLITE_IOERR_CHECKRESERVEDLOCK; );

			assert( pFile );

			/* Check if a thread in this process holds such a lock */
			if( pFile->eFileLock>SHARED_LOCK ){
				reserved = 1;
			}

			/* Otherwise see if some other process holds it. */
			if( !reserved ){
				sem_t *pSem = pFile->pInode->pSem;
				struct stat statBuf;

				if( sem_trywait(pSem)==-1 ){
					int tErrno = errno;
					if( EAGAIN != tErrno ){
						rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_CHECKRESERVEDLOCK);
						pFile->lastErrno = tErrno;
					} else {
						/* someone else has the lock when we are in NO_LOCK */
						reserved = (pFile->eFileLock < SHARED_LOCK);
					}
				}else{
					/* we could have it if we want it */
					sem_post(pSem);
				}
			}
			OSTRACE(("_TEST WR-LOCK %d %d %d (sem)\n", pFile->h, rc, reserved));

			*pResOut = reserved;
			return rc;
		}

		static int semLock(sqlite3_file *id, int eFileLock) {
			unixFile *pFile = (unixFile*)id;
			int fd;
			sem_t *pSem = pFile->pInode->pSem;
			int rc = SQLITE_OK;

			/* if we already have a lock, it is exclusive.  
			** Just adjust level and punt on outta here. */
			if (pFile->eFileLock > NO_LOCK) {
				pFile->eFileLock = eFileLock;
				rc = SQLITE_OK;
				goto sem_end_lock;
			}

			/* lock semaphore now but bail out when already locked. */
			if( sem_trywait(pSem)==-1 ){
				rc = SQLITE_BUSY;
				goto sem_end_lock;
			}

			/* got it, set the type and return ok */
			pFile->eFileLock = eFileLock;

sem_end_lock:
			return rc;
		}

		static int semUnlock(sqlite3_file *id, int eFileLock) {
			unixFile *pFile = (unixFile*)id;
			sem_t *pSem = pFile->pInode->pSem;

			assert( pFile );
			assert( pSem );
			OSTRACE(("UNLOCK  %d %d was %d pid=%d (sem)\n", pFile->h, eFileLock,
				pFile->eFileLock, getpid()));
			assert( eFileLock<=SHARED_LOCK );

			/* no-op if possible */
			if( pFile->eFileLock==eFileLock ){
				return SQLITE_OK;
			}

			/* shared can just be set because we always have an exclusive */
			if (eFileLock==SHARED_LOCK) {
				pFile->eFileLock = eFileLock;
				return SQLITE_OK;
			}

			/* no, really unlock. */
			if ( sem_post(pSem)==-1 ) {
				int rc, tErrno = errno;
				rc = sqliteErrorFromPosixError(tErrno, SQLITE_IOERR_UNLOCK);
				if( IS_LOCK_ERROR(rc) ){
					pFile->lastErrno = tErrno;
				}
				return rc; 
			}
			pFile->eFileLock = NO_LOCK;
			return SQLITE_OK;
		}

		static int semClose(sqlite3_file *id) {
			if( id ){
				unixFile *pFile = (unixFile*)id;
				semUnlock(id, NO_LOCK);
				assert( pFile );
				unixEnterMutex();
				releaseInodeInfo(pFile);
				unixLeaveMutex();
				closeUnixFile(id);
			}
			return SQLITE_OK;
		}

#endif
#pragma endregion

#pragma region AFP Locking
#if defined(__APPLE__) && ENABLE_LOCKING_STYLE
		typedef struct afpLockingContext afpLockingContext;
		struct afpLockingContext {
			int reserved;
			const char *dbPath;             /* Name of the open file */
		};

		struct ByteRangeLockPB2
		{
			unsigned long long offset;        /* offset to first byte to lock */
			unsigned long long length;        /* nbr of bytes to lock */
			unsigned long long retRangeStart; /* nbr of 1st byte locked if successful */
			unsigned char unLockFlag;         /* 1 = unlock, 0 = lock */
			unsigned char startEndFlag;       /* 1=rel to end of fork, 0=rel to start */
			int fd;                           /* file desc to assoc this lock with */
		};

#define afpfsByteRangeLock2FSCTL        _IOWR('z', 23, struct ByteRangeLockPB2)

		static int afpSetLock(
			const char *path,              /* Name of the file to be locked or unlocked */
			unixFile *pFile,               /* Open file descriptor on path */
			unsigned long long offset,     /* First byte to be locked */
			unsigned long long length,     /* Number of bytes to lock */
			int setLockFlag                /* True to set lock.  False to clear lock */
			){
				struct ByteRangeLockPB2 pb;
				int err;

				pb.unLockFlag = setLockFlag ? 0 : 1;
				pb.startEndFlag = 0;
				pb.offset = offset;
				pb.length = length; 
				pb.fd = pFile->h;

				OSTRACE(("AFPSETLOCK [%s] for %d%s in range %llx:%llx\n", 
					(setLockFlag?"ON":"OFF"), pFile->h, (pb.fd==-1?"[testval-1]":""),
					offset, length));
				err = fsctl(path, afpfsByteRangeLock2FSCTL, &pb, 0);
				if ( err==-1 ) {
					int rc;
					int tErrno = errno;
					OSTRACE(("AFPSETLOCK failed to fsctl() '%s' %d %s\n",
						path, tErrno, strerror(tErrno)));
#ifdef SQLITE_IGNORE_AFP_LOCK_ERRORS
					rc = SQLITE_BUSY;
#else
					rc = sqliteErrorFromPosixError(tErrno,
						setLockFlag ? SQLITE_IOERR_LOCK : SQLITE_IOERR_UNLOCK);
#endif /* SQLITE_IGNORE_AFP_LOCK_ERRORS */
					if( IS_LOCK_ERROR(rc) ){
						pFile->lastErrno = tErrno;
					}
					return rc;
				} else {
					return SQLITE_OK;
				}
		}

		static int afpCheckReservedLock(sqlite3_file *id, int *pResOut){
			int rc = SQLITE_OK;
			int reserved = 0;
			unixFile *pFile = (unixFile*)id;
			afpLockingContext *context;

			SimulateIOError( return SQLITE_IOERR_CHECKRESERVEDLOCK; );

			assert( pFile );
			context = (afpLockingContext *) pFile->lockingContext;
			if( context->reserved ){
				*pResOut = 1;
				return SQLITE_OK;
			}
			unixEnterMutex(); /* Because pFile->pInode is shared across threads */

			/* Check if a thread in this process holds such a lock */
			if( pFile->pInode->eFileLock>SHARED_LOCK ){
				reserved = 1;
			}

			/* Otherwise see if some other process holds it.
			*/
			if( !reserved ){
				/* lock the RESERVED byte */
				int lrc = afpSetLock(context->dbPath, pFile, RESERVED_BYTE, 1,1);  
				if( SQLITE_OK==lrc ){
					/* if we succeeded in taking the reserved lock, unlock it to restore
					** the original state */
					lrc = afpSetLock(context->dbPath, pFile, RESERVED_BYTE, 1, 0);
				} else {
					/* if we failed to get the lock then someone else must have it */
					reserved = 1;
				}
				if( IS_LOCK_ERROR(lrc) ){
					rc=lrc;
				}
			}

			unixLeaveMutex();
			OSTRACE(("_TEST WR-LOCK %d %d %d (afp)\n", pFile->h, rc, reserved));

			*pResOut = reserved;
			return rc;
		}

		static int afpLock(sqlite3_file *id, int eFileLock){
			int rc = SQLITE_OK;
			unixFile *pFile = (unixFile*)id;
			unixInodeInfo *pInode = pFile->pInode;
			afpLockingContext *context = (afpLockingContext *) pFile->lockingContext;

			assert( pFile );
			OSTRACE(("LOCK    %d %s was %s(%s,%d) pid=%d (afp)\n", pFile->h,
				azFileLock(eFileLock), azFileLock(pFile->eFileLock),
				azFileLock(pInode->eFileLock), pInode->nShared , getpid()));

			/* If there is already a lock of this type or more restrictive on the
			** unixFile, do nothing. Don't use the afp_end_lock: exit path, as
			** unixEnterMutex() hasn't been called yet.
			*/
			if( pFile->eFileLock>=eFileLock ){
				OSTRACE(("LOCK    %d %s ok (already held) (afp)\n", pFile->h,
					azFileLock(eFileLock)));
				return SQLITE_OK;
			}

			/* Make sure the locking sequence is correct
			**  (1) We never move from unlocked to anything higher than shared lock.
			**  (2) SQLite never explicitly requests a pendig lock.
			**  (3) A shared lock is always held when a reserve lock is requested.
			*/
			assert( pFile->eFileLock!=NO_LOCK || eFileLock==SHARED_LOCK );
			assert( eFileLock!=PENDING_LOCK );
			assert( eFileLock!=RESERVED_LOCK || pFile->eFileLock==SHARED_LOCK );

			/* This mutex is needed because pFile->pInode is shared across threads
			*/
			unixEnterMutex();
			pInode = pFile->pInode;

			/* If some thread using this PID has a lock via a different unixFile*
			** handle that precludes the requested lock, return BUSY.
			*/
			if( (pFile->eFileLock!=pInode->eFileLock && 
				(pInode->eFileLock>=PENDING_LOCK || eFileLock>SHARED_LOCK))
				){
					rc = SQLITE_BUSY;
					goto afp_end_lock;
			}

			/* If a SHARED lock is requested, and some thread using this PID already
			** has a SHARED or RESERVED lock, then increment reference counts and
			** return SQLITE_OK.
			*/
			if( eFileLock==SHARED_LOCK && 
				(pInode->eFileLock==SHARED_LOCK || pInode->eFileLock==RESERVED_LOCK) ){
					assert( eFileLock==SHARED_LOCK );
					assert( pFile->eFileLock==0 );
					assert( pInode->nShared>0 );
					pFile->eFileLock = SHARED_LOCK;
					pInode->nShared++;
					pInode->nLock++;
					goto afp_end_lock;
			}

			/* A PENDING lock is needed before acquiring a SHARED lock and before
			** acquiring an EXCLUSIVE lock.  For the SHARED lock, the PENDING will
			** be released.
			*/
			if( eFileLock==SHARED_LOCK 
				|| (eFileLock==EXCLUSIVE_LOCK && pFile->eFileLock<PENDING_LOCK)
				){
					int failed;
					failed = afpSetLock(context->dbPath, pFile, PENDING_BYTE, 1, 1);
					if (failed) {
						rc = failed;
						goto afp_end_lock;
					}
			}

			/* If control gets to this point, then actually go ahead and make
			** operating system calls for the specified lock.
			*/
			if( eFileLock==SHARED_LOCK ){
				int lrc1, lrc2, lrc1Errno = 0;
				long lk, mask;

				assert( pInode->nShared==0 );
				assert( pInode->eFileLock==0 );

				mask = (sizeof(long)==8) ? LARGEST_INT64 : 0x7fffffff;
				/* Now get the read-lock SHARED_LOCK */
				/* note that the quality of the randomness doesn't matter that much */
				lk = random(); 
				pInode->sharedByte = (lk & mask)%(SHARED_SIZE - 1);
				lrc1 = afpSetLock(context->dbPath, pFile, 
					SHARED_FIRST+pInode->sharedByte, 1, 1);
				if( IS_LOCK_ERROR(lrc1) ){
					lrc1Errno = pFile->lastErrno;
				}
				/* Drop the temporary PENDING lock */
				lrc2 = afpSetLock(context->dbPath, pFile, PENDING_BYTE, 1, 0);

				if( IS_LOCK_ERROR(lrc1) ) {
					pFile->lastErrno = lrc1Errno;
					rc = lrc1;
					goto afp_end_lock;
				} else if( IS_LOCK_ERROR(lrc2) ){
					rc = lrc2;
					goto afp_end_lock;
				} else if( lrc1 != SQLITE_OK ) {
					rc = lrc1;
				} else {
					pFile->eFileLock = SHARED_LOCK;
					pInode->nLock++;
					pInode->nShared = 1;
				}
			}else if( eFileLock==EXCLUSIVE_LOCK && pInode->nShared>1 ){
				/* We are trying for an exclusive lock but another thread in this
				** same process is still holding a shared lock. */
				rc = SQLITE_BUSY;
			}else{
				/* The request was for a RESERVED or EXCLUSIVE lock.  It is
				** assumed that there is a SHARED or greater lock on the file
				** already.
				*/
				int failed = 0;
				assert( 0!=pFile->eFileLock );
				if (eFileLock >= RESERVED_LOCK && pFile->eFileLock < RESERVED_LOCK) {
					/* Acquire a RESERVED lock */
					failed = afpSetLock(context->dbPath, pFile, RESERVED_BYTE, 1,1);
					if( !failed ){
						context->reserved = 1;
					}
				}
				if (!failed && eFileLock == EXCLUSIVE_LOCK) {
					/* Acquire an EXCLUSIVE lock */

					/* Remove the shared lock before trying the range.  we'll need to 
					** reestablish the shared lock if we can't get the  afpUnlock
					*/
					if( !(failed = afpSetLock(context->dbPath, pFile, SHARED_FIRST +
						pInode->sharedByte, 1, 0)) ){
							int failed2 = SQLITE_OK;
							/* now attemmpt to get the exclusive lock range */
							failed = afpSetLock(context->dbPath, pFile, SHARED_FIRST, 
								SHARED_SIZE, 1);
							if( failed && (failed2 = afpSetLock(context->dbPath, pFile, 
								SHARED_FIRST + pInode->sharedByte, 1, 1)) ){
									/* Can't reestablish the shared lock.  Sqlite can't deal, this is
									** a critical I/O error
									*/
									rc = ((failed & SQLITE_IOERR) == SQLITE_IOERR) ? failed2 : 
										SQLITE_IOERR_LOCK;
									goto afp_end_lock;
							} 
					}else{
						rc = failed; 
					}
				}
				if( failed ){
					rc = failed;
				}
			}

			if( rc==SQLITE_OK ){
				pFile->eFileLock = eFileLock;
				pInode->eFileLock = eFileLock;
			}else if( eFileLock==EXCLUSIVE_LOCK ){
				pFile->eFileLock = PENDING_LOCK;
				pInode->eFileLock = PENDING_LOCK;
			}

afp_end_lock:
			unixLeaveMutex();
			OSTRACE(("LOCK    %d %s %s (afp)\n", pFile->h, azFileLock(eFileLock), 
				rc==SQLITE_OK ? "ok" : "failed"));
			return rc;
		}

		static int afpUnlock(sqlite3_file *id, int eFileLock) {
			int rc = SQLITE_OK;
			unixFile *pFile = (unixFile*)id;
			unixInodeInfo *pInode;
			afpLockingContext *context = (afpLockingContext *) pFile->lockingContext;
			int skipShared = 0;
#ifdef _TEST
			int h = pFile->h;
#endif

			assert( pFile );
			OSTRACE(("UNLOCK  %d %d was %d(%d,%d) pid=%d (afp)\n", pFile->h, eFileLock,
				pFile->eFileLock, pFile->pInode->eFileLock, pFile->pInode->nShared,
				getpid()));

			assert( eFileLock<=SHARED_LOCK );
			if( pFile->eFileLock<=eFileLock ){
				return SQLITE_OK;
			}
			unixEnterMutex();
			pInode = pFile->pInode;
			assert( pInode->nShared!=0 );
			if( pFile->eFileLock>SHARED_LOCK ){
				assert( pInode->eFileLock==pFile->eFileLock );
				SimulateIOErrorBenign(1);
				SimulateIOError( h=(-1) )
					SimulateIOErrorBenign(0);

#ifdef _DEBUG
				/* When reducing a lock such that other processes can start
				** reading the database file again, make sure that the
				** transaction counter was updated if any part of the database
				** file changed.  If the transaction counter is not updated,
				** other connections to the same file might not realize that
				** the file has changed and hence might not know to flush their
				** cache.  The use of a stale cache can lead to database corruption.
				*/
				assert( pFile->inNormalWrite==0
					|| pFile->dbUpdate==0
					|| pFile->transCntrChng==1 );
				pFile->inNormalWrite = 0;
#endif

				if( pFile->eFileLock==EXCLUSIVE_LOCK ){
					rc = afpSetLock(context->dbPath, pFile, SHARED_FIRST, SHARED_SIZE, 0);
					if( rc==SQLITE_OK && (eFileLock==SHARED_LOCK || pInode->nShared>1) ){
						/* only re-establish the shared lock if necessary */
						int sharedLockByte = SHARED_FIRST+pInode->sharedByte;
						rc = afpSetLock(context->dbPath, pFile, sharedLockByte, 1, 1);
					} else {
						skipShared = 1;
					}
				}
				if( rc==SQLITE_OK && pFile->eFileLock>=PENDING_LOCK ){
					rc = afpSetLock(context->dbPath, pFile, PENDING_BYTE, 1, 0);
				} 
				if( rc==SQLITE_OK && pFile->eFileLock>=RESERVED_LOCK && context->reserved ){
					rc = afpSetLock(context->dbPath, pFile, RESERVED_BYTE, 1, 0);
					if( !rc ){ 
						context->reserved = 0; 
					}
				}
				if( rc==SQLITE_OK && (eFileLock==SHARED_LOCK || pInode->nShared>1)){
					pInode->eFileLock = SHARED_LOCK;
				}
			}
			if( rc==SQLITE_OK && eFileLock==NO_LOCK ){

				/* Decrement the shared lock counter.  Release the lock using an
				** OS call only when all threads in this same process have released
				** the lock.
				*/
				unsigned long long sharedLockByte = SHARED_FIRST+pInode->sharedByte;
				pInode->nShared--;
				if( pInode->nShared==0 ){
					SimulateIOErrorBenign(1);
					SimulateIOError( h=(-1) )
						SimulateIOErrorBenign(0);
					if( !skipShared ){
						rc = afpSetLock(context->dbPath, pFile, sharedLockByte, 1, 0);
					}
					if( !rc ){
						pInode->eFileLock = NO_LOCK;
						pFile->eFileLock = NO_LOCK;
					}
				}
				if( rc==SQLITE_OK ){
					pInode->nLock--;
					assert( pInode->nLock>=0 );
					if( pInode->nLock==0 ){
						closePendingFds(pFile);
					}
				}
			}

			unixLeaveMutex();
			if( rc==SQLITE_OK ) pFile->eFileLock = eFileLock;
			return rc;
		}

		static int afpClose(sqlite3_file *id) {
			int rc = SQLITE_OK;
			if( id ){
				unixFile *pFile = (unixFile*)id;
				afpUnlock(id, NO_LOCK);
				unixEnterMutex();
				if( pFile->pInode && pFile->pInode->nLock ){
					/* If there are outstanding locks, do not actually close the file just
					** yet because that would clear those locks.  Instead, add the file
					** descriptor to pInode->aPending.  It will be automatically closed when
					** the last lock is cleared.
					*/
					setPendingFd(pFile);
				}
				releaseInodeInfo(pFile);
				sqlite3_free(pFile->lockingContext);
				rc = closeUnixFile(id);
				unixLeaveMutex();
			}
			return rc;
		}

#endif
#pragma endregion

#pragma region NFS Locking
#if defined(__APPLE__) && ENABLE_LOCKING_STYLE
		static int nfsUnlock(sqlite3_file *id, int eFileLock){
			return posixUnlock(id, eFileLock, 1);
		}

#endif
#pragma endregion

#pragma region Non-locking sqlite3_file methods

		static int seekAndRead(unixFile *id, sqlite3_int64 offset, void *pBuf, int cnt){
			int got;
			int prior = 0;
#if (!defined(USE_PREAD) && !defined(USE_PREAD64))
			i64 newOffset;
#endif
			TIMER_START;
			assert( cnt==(cnt&0x1ffff) );
			cnt &= 0x1ffff;
			do{
#if defined(USE_PREAD)
				got = osPread(id->h, pBuf, cnt, offset);
				SimulateIOError( got = -1 );
#elif defined(USE_PREAD64)
				got = osPread64(id->h, pBuf, cnt, offset);
				SimulateIOError( got = -1 );
#else
				newOffset = lseek(id->h, offset, SEEK_SET);
				SimulateIOError( newOffset-- );
				if( newOffset!=offset ){
					if( newOffset == -1 ){
						((unixFile*)id)->lastErrno = errno;
					}else{
						((unixFile*)id)->lastErrno = 0;
					}
					return -1;
				}
				got = osRead(id->h, pBuf, cnt);
#endif
				if( got==cnt ) break;
				if( got<0 ){
					if( errno==EINTR ){ got = 1; continue; }
					prior = 0;
					((unixFile*)id)->lastErrno = errno;
					break;
				}else if( got>0 ){
					cnt -= got;
					offset += got;
					prior += got;
					pBuf = (void*)(got + (char*)pBuf);
				}
			}while( got>0 );
			TIMER_END;
			OSTRACE(("READ    %-3d %5d %7lld %llu\n",
				id->h, got+prior, offset-prior, TIMER_ELAPSED));
			return got+prior;
		}

		static int unixRead(
			sqlite3_file *id, 
			void *pBuf, 
			int amt,
			sqlite3_int64 offset
			){
				unixFile *pFile = (unixFile *)id;
				int got;
				assert( id );

				/* If this is a database file (not a journal, master-journal or temp
				** file), the bytes in the locking range should never be read or written. */
#if 0
				assert( pFile->pUnused==0
					|| offset>=PENDING_BYTE+512
					|| offset+amt<=PENDING_BYTE 
					);
#endif

				got = seekAndRead(pFile, offset, pBuf, amt);
				if( got==amt ){
					return SQLITE_OK;
				}else if( got<0 ){
					/* lastErrno set by seekAndRead */
					return SQLITE_IOERR_READ;
				}else{
					pFile->lastErrno = 0; /* not a system error */
					/* Unread parts of the buffer must be zero-filled */
					memset(&((char*)pBuf)[got], 0, amt-got);
					return SQLITE_IOERR_SHORT_READ;
				}
		}

		static int seekAndWrite(unixFile *id, i64 offset, const void *pBuf, int cnt){
			int got;
#if (!defined(USE_PREAD) && !defined(USE_PREAD64))
			i64 newOffset;
#endif
			assert( cnt==(cnt&0x1ffff) );
			cnt &= 0x1ffff;
			TIMER_START;
#if defined(USE_PREAD)
			do{ got = osPwrite(id->h, pBuf, cnt, offset); }while( got<0 && errno==EINTR );
#elif defined(USE_PREAD64)
			do{ got = osPwrite64(id->h, pBuf, cnt, offset);}while( got<0 && errno==EINTR);
#else
			do{
				newOffset = lseek(id->h, offset, SEEK_SET);
				SimulateIOError( newOffset-- );
				if( newOffset!=offset ){
					if( newOffset == -1 ){
						((unixFile*)id)->lastErrno = errno;
					}else{
						((unixFile*)id)->lastErrno = 0;
					}
					return -1;
				}
				got = osWrite(id->h, pBuf, cnt);
			}while( got<0 && errno==EINTR );
#endif
			TIMER_END;
			if( got<0 ){
				((unixFile*)id)->lastErrno = errno;
			}

			OSTRACE(("WRITE   %-3d %5d %7lld %llu\n", id->h, got, offset, TIMER_ELAPSED));
			return got;
		}

		static int unixWrite(
			sqlite3_file *id, 
			const void *pBuf, 
			int amt,
			sqlite3_int64 offset 
			){
				unixFile *pFile = (unixFile*)id;
				int wrote = 0;
				assert( id );
				assert( amt>0 );

				/* If this is a database file (not a journal, master-journal or temp
				** file), the bytes in the locking range should never be read or written. */
#if 0
				assert( pFile->pUnused==0
					|| offset>=PENDING_BYTE+512
					|| offset+amt<=PENDING_BYTE 
					);
#endif

#ifdef _DEBUG
				/* If we are doing a normal write to a database file (as opposed to
				** doing a hot-journal rollback or a write to some file other than a
				** normal database file) then record the fact that the database
				** has changed.  If the transaction counter is modified, record that
				** fact too.
				*/
				if( pFile->inNormalWrite ){
					pFile->dbUpdate = 1;  /* The database has been modified */
					if( offset<=24 && offset+amt>=27 ){
						int rc;
						char oldCntr[4];
						SimulateIOErrorBenign(1);
						rc = seekAndRead(pFile, 24, oldCntr, 4);
						SimulateIOErrorBenign(0);
						if( rc!=4 || memcmp(oldCntr, &((char*)pBuf)[24-offset], 4)!=0 ){
							pFile->transCntrChng = 1;  /* The transaction counter has changed */
						}
					}
				}
#endif

				while( amt>0 && (wrote = seekAndWrite(pFile, offset, pBuf, amt))>0 ){
					amt -= wrote;
					offset += wrote;
					pBuf = &((char*)pBuf)[wrote];
				}
				SimulateIOError(( wrote=(-1), amt=1 ));
				SimulateDiskfullError(( wrote=0, amt=1 ));

				if( amt>0 ){
					if( wrote<0 && pFile->lastErrno!=ENOSPC ){
						/* lastErrno set by seekAndWrite */
						return SQLITE_IOERR_WRITE;
					}else{
						pFile->lastErrno = 0; /* not a system error */
						return SQLITE_FULL;
					}
				}

				return SQLITE_OK;
		}

#ifdef _TEST
		int sqlite3_sync_count = 0;
		int sqlite3_fullsync_count = 0;
#endif

#if !defined(fdatasync)
# define fdatasync fsync
#endif

#ifdef F_FULLFSYNC
# define HAVE_FULLFSYNC 1
#else
# define HAVE_FULLFSYNC 0
#endif

		static int full_fsync(int fd, int fullSync, int dataOnly){
			int rc;

			/* The following "ifdef/elif/else/" block has the same structure as
			** the one below. It is replicated here solely to avoid cluttering 
			** up the real code with the UNUSED_PARAMETER() macros.
			*/
#ifdef SQLITE_NO_SYNC
			UNUSED_PARAMETER(fd);
			UNUSED_PARAMETER(fullSync);
			UNUSED_PARAMETER(dataOnly);
#elif HAVE_FULLFSYNC
			UNUSED_PARAMETER(dataOnly);
#else
			UNUSED_PARAMETER(fullSync);
			UNUSED_PARAMETER(dataOnly);
#endif

			/* Record the number of times that we do a normal fsync() and 
			** FULLSYNC.  This is used during testing to verify that this procedure
			** gets called with the correct arguments.
			*/
#ifdef _TEST
			if( fullSync ) sqlite3_fullsync_count++;
			sqlite3_sync_count++;
#endif

			/* If we compiled with the SQLITE_NO_SYNC flag, then syncing is a
			** no-op
			*/
#ifdef SQLITE_NO_SYNC
			rc = SQLITE_OK;
#elif HAVE_FULLFSYNC
			if( fullSync ){
				rc = osFcntl(fd, F_FULLFSYNC, 0);
			}else{
				rc = 1;
			}
			/* If the FULLFSYNC failed, fall back to attempting an fsync().
			** It shouldn't be possible for fullfsync to fail on the local 
			** file system (on OSX), so failure indicates that FULLFSYNC
			** isn't supported for this file system. So, attempt an fsync 
			** and (for now) ignore the overhead of a superfluous fcntl call.  
			** It'd be better to detect fullfsync support once and avoid 
			** the fcntl call every time sync is called.
			*/
			if( rc ) rc = fsync(fd);

#elif defined(__APPLE__)
			/* fdatasync() on HFS+ doesn't yet flush the file size if it changed correctly
			** so currently we default to the macro that redefines fdatasync to fsync
			*/
			rc = fsync(fd);
#else 
			rc = fdatasync(fd);
#if OS_VXWORKS
			if( rc==-1 && errno==ENOTSUP ){
				rc = fsync(fd);
			}
#endif /* OS_VXWORKS */
#endif /* ifdef SQLITE_NO_SYNC elif HAVE_FULLFSYNC */

			if( OS_VXWORKS && rc!= -1 ){
				rc = 0;
			}
			return rc;
		}

		static int openDirectory(const char *zFilename, int *pFd){
			int ii;
			int fd = -1;
			char zDirname[MAX_PATHNAME+1];

			sqlite3_snprintf(MAX_PATHNAME, zDirname, "%s", zFilename);
			for(ii=(int)strlen(zDirname); ii>1 && zDirname[ii]!='/'; ii--);
			if( ii>0 ){
				zDirname[ii] = '\0';
				fd = robust_open(zDirname, O_RDONLY|O_BINARY, 0);
				if( fd>=0 ){
					OSTRACE(("OPENDIR %-3d %s\n", fd, zDirname));
				}
			}
			*pFd = fd;
			return (fd>=0?SQLITE_OK:unixLogError(SQLITE_CANTOPEN_BKPT, "open", zDirname));
		}

		static int unixSync(sqlite3_file *id, int flags){
			int rc;
			unixFile *pFile = (unixFile*)id;

			int isDataOnly = (flags&SQLITE_SYNC_DATAONLY);
			int isFullsync = (flags&0x0F)==SQLITE_SYNC_FULL;

			/* Check that one of SQLITE_SYNC_NORMAL or FULL was passed */
			assert((flags&0x0F)==SQLITE_SYNC_NORMAL
				|| (flags&0x0F)==SQLITE_SYNC_FULL
				);

			/* Unix cannot, but some systems may return SQLITE_FULL from here. This
			** line is to test that doing so does not cause any problems.
			*/
			SimulateDiskfullError( return SQLITE_FULL );

			assert( pFile );
			OSTRACE(("SYNC    %-3d\n", pFile->h));
			rc = full_fsync(pFile->h, isFullsync, isDataOnly);
			SimulateIOError( rc=1 );
			if( rc ){
				pFile->lastErrno = errno;
				return unixLogError(SQLITE_IOERR_FSYNC, "full_fsync", pFile->zPath);
			}

			/* Also fsync the directory containing the file if the DIRSYNC flag
			** is set.  This is a one-time occurrence.  Many systems (examples: AIX)
			** are unable to fsync a directory, so ignore errors on the fsync.
			*/
			if( pFile->ctrlFlags & UNIXFILE_DIRSYNC ){
				int dirfd;
				OSTRACE(("DIRSYNC %s (have_fullfsync=%d fullsync=%d)\n", pFile->zPath,
					HAVE_FULLFSYNC, isFullsync));
				rc = osOpenDirectory(pFile->zPath, &dirfd);
				if( rc==SQLITE_OK && dirfd>=0 ){
					full_fsync(dirfd, 0, 0);
					robust_close(pFile, dirfd, __LINE__);
				}else if( rc==SQLITE_CANTOPEN ){
					rc = SQLITE_OK;
				}
				pFile->ctrlFlags &= ~UNIXFILE_DIRSYNC;
			}
			return rc;
		}

		static int unixTruncate(sqlite3_file *id, i64 nByte){
			unixFile *pFile = (unixFile *)id;
			int rc;
			assert( pFile );
			SimulateIOError( return SQLITE_IOERR_TRUNCATE );

			/* If the user has configured a chunk-size for this file, truncate the
			** file so that it consists of an integer number of chunks (i.e. the
			** actual file size after the operation may be larger than the requested
			** size).
			*/
			if( pFile->szChunk>0 ){
				nByte = ((nByte + pFile->szChunk - 1)/pFile->szChunk) * pFile->szChunk;
			}

			rc = robust_ftruncate(pFile->h, (off_t)nByte);
			if( rc ){
				pFile->lastErrno = errno;
				return unixLogError(SQLITE_IOERR_TRUNCATE, "ftruncate", pFile->zPath);
			}else{
#ifdef SQLITE_DEBUG
				/* If we are doing a normal write to a database file (as opposed to
				** doing a hot-journal rollback or a write to some file other than a
				** normal database file) and we truncate the file to zero length,
				** that effectively updates the change counter.  This might happen
				** when restoring a database using the backup API from a zero-length
				** source.
				*/
				if( pFile->inNormalWrite && nByte==0 ){
					pFile->transCntrChng = 1;
				}
#endif

				return SQLITE_OK;
			}
		}

		static int unixFileSize(sqlite3_file *id, i64 *pSize){
			int rc;
			struct stat buf;
			assert( id );
			rc = osFstat(((unixFile*)id)->h, &buf);
			SimulateIOError( rc=1 );
			if( rc!=0 ){
				((unixFile*)id)->lastErrno = errno;
				return SQLITE_IOERR_FSTAT;
			}
			*pSize = buf.st_size;

			/* When opening a zero-size database, the findInodeInfo() procedure
			** writes a single byte into that file in order to work around a bug
			** in the OS-X msdos filesystem.  In order to avoid problems with upper
			** layers, we need to report this file size as zero even though it is
			** really 1.   Ticket #3260.
			*/
			if( *pSize==1 ) *pSize = 0;


			return SQLITE_OK;
		}

#if ENABLE_LOCKING_STYLE && defined(__APPLE__)
		static int proxyFileControl(sqlite3_file*,int,void*);
#endif

		static int fcntlSizeHint(unixFile *pFile, i64 nByte){
			if( pFile->szChunk>0 ){
				i64 nSize;                    /* Required file size */
				struct stat buf;              /* Used to hold return values of fstat() */

				if( osFstat(pFile->h, &buf) ) return SQLITE_IOERR_FSTAT;

				nSize = ((nByte+pFile->szChunk-1) / pFile->szChunk) * pFile->szChunk;
				if( nSize>(i64)buf.st_size ){

#if defined(HAVE_POSIX_FALLOCATE) && HAVE_POSIX_FALLOCATE
					/* The code below is handling the return value of osFallocate() 
					** correctly. posix_fallocate() is defined to "returns zero on success, 
					** or an error number on  failure". See the manpage for details. */
					int err;
					do{
						err = osFallocate(pFile->h, buf.st_size, nSize-buf.st_size);
					}while( err==EINTR );
					if( err ) return SQLITE_IOERR_WRITE;
#else
					/* If the OS does not have posix_fallocate(), fake it. First use
					** ftruncate() to set the file size, then write a single byte to
					** the last byte in each block within the extended region. This
					** is the same technique used by glibc to implement posix_fallocate()
					** on systems that do not have a real fallocate() system call.
					*/
					int nBlk = buf.st_blksize;  /* File-system block size */
					i64 iWrite;                 /* Next offset to write to */

					if( robust_ftruncate(pFile->h, nSize) ){
						pFile->lastErrno = errno;
						return unixLogError(SQLITE_IOERR_TRUNCATE, "ftruncate", pFile->zPath);
					}
					iWrite = ((buf.st_size + 2*nBlk - 1)/nBlk)*nBlk-1;
					while( iWrite<nSize ){
						int nWrite = seekAndWrite(pFile, iWrite, "", 1);
						if( nWrite!=1 ) return SQLITE_IOERR_WRITE;
						iWrite += nBlk;
					}
#endif
				}
			}

			return SQLITE_OK;
		}

		static void unixModeBit(unixFile *pFile, unsigned char mask, int *pArg){
			if( *pArg<0 ){
				*pArg = (pFile->ctrlFlags & mask)!=0;
			}else if( (*pArg)==0 ){
				pFile->ctrlFlags &= ~mask;
			}else{
				pFile->ctrlFlags |= mask;
			}
		}

		// Forward declaration
		static int unixGetTempname(int nBuf, char *zBuf);

		static int unixFileControl(sqlite3_file *id, int op, void *pArg){
			unixFile *pFile = (unixFile*)id;
			switch( op ){
			case SQLITE_FCNTL_LOCKSTATE: {
				*(int*)pArg = pFile->eFileLock;
				return SQLITE_OK;
										 }
			case SQLITE_LAST_ERRNO: {
				*(int*)pArg = pFile->lastErrno;
				return SQLITE_OK;
									}
			case SQLITE_FCNTL_CHUNK_SIZE: {
				pFile->szChunk = *(int *)pArg;
				return SQLITE_OK;
										  }
			case SQLITE_FCNTL_SIZE_HINT: {
				int rc;
				SimulateIOErrorBenign(1);
				rc = fcntlSizeHint(pFile, *(i64 *)pArg);
				SimulateIOErrorBenign(0);
				return rc;
										 }
			case SQLITE_FCNTL_PERSIST_WAL: {
				unixModeBit(pFile, UNIXFILE_PERSIST_WAL, (int*)pArg);
				return SQLITE_OK;
										   }
			case SQLITE_FCNTL_POWERSAFE_OVERWRITE: {
				unixModeBit(pFile, UNIXFILE_PSOW, (int*)pArg);
				return SQLITE_OK;
												   }
			case SQLITE_FCNTL_VFSNAME: {
				*(char**)pArg = sqlite3_mprintf("%s", pFile->pVfs->zName);
				return SQLITE_OK;
									   }
			case SQLITE_FCNTL_TEMPFILENAME: {
				char *zTFile = sqlite3_malloc( pFile->pVfs->mxPathname );
				if( zTFile ){
					unixGetTempname(pFile->pVfs->mxPathname, zTFile);
					*(char**)pArg = zTFile;
				}
				return SQLITE_OK;
											}
#ifdef SQLITE_DEBUG
											/* The pager calls this method to signal that it has done
											** a rollback and that the database is therefore unchanged and
											** it hence it is OK for the transaction change counter to be
											** unchanged.
											*/
			case SQLITE_FCNTL_DB_UNCHANGED: {
				((unixFile*)id)->dbUpdate = 0;
				return SQLITE_OK;
											}
#endif
#if SQLITE_ENABLE_LOCKING_STYLE && defined(__APPLE__)
			case SQLITE_SET_LOCKPROXYFILE:
			case SQLITE_GET_LOCKPROXYFILE: {
				return proxyFileControl(id,op,pArg);
										   }
#endif /* SQLITE_ENABLE_LOCKING_STYLE && defined(__APPLE__) */
			}
			return SQLITE_NOTFOUND;
		}


#ifndef __QNXNTO__ 
		static int unixSectorSize(sqlite3_file *NotUsed){
			UNUSED_PARAMETER(NotUsed);
			return SQLITE_DEFAULT_SECTOR_SIZE;
		}
#endif

#ifdef __QNXNTO__
#include <sys/dcmd_blk.h>
#include <sys/statvfs.h>
		static int unixSectorSize(sqlite3_file *id){
			unixFile *pFile = (unixFile*)id;
			if( pFile->sectorSize == 0 ){
				struct statvfs fsInfo;

				/* Set defaults for non-supported filesystems */
				pFile->sectorSize = SQLITE_DEFAULT_SECTOR_SIZE;
				pFile->deviceCharacteristics = 0;
				if( fstatvfs(pFile->h, &fsInfo) == -1 ) {
					return pFile->sectorSize;
				}

				if( !strcmp(fsInfo.f_basetype, "tmp") ) {
					pFile->sectorSize = fsInfo.f_bsize;
					pFile->deviceCharacteristics =
						SQLITE_IOCAP_ATOMIC4K |       /* All ram filesystem writes are atomic */
						SQLITE_IOCAP_SAFE_APPEND |    /* growing the file does not occur until
													  ** the write succeeds */
													  SQLITE_IOCAP_SEQUENTIAL |     /* The ram filesystem has no write behind
																					** so it is ordered */
																					0;
				}else if( strstr(fsInfo.f_basetype, "etfs") ){
					pFile->sectorSize = fsInfo.f_bsize;
					pFile->deviceCharacteristics =
						/* etfs cluster size writes are atomic */
						(pFile->sectorSize / 512 * SQLITE_IOCAP_ATOMIC512) |
						SQLITE_IOCAP_SAFE_APPEND |    /* growing the file does not occur until
													  ** the write succeeds */
													  SQLITE_IOCAP_SEQUENTIAL |     /* The ram filesystem has no write behind
																					** so it is ordered */
																					0;
				}else if( !strcmp(fsInfo.f_basetype, "qnx6") ){
					pFile->sectorSize = fsInfo.f_bsize;
					pFile->deviceCharacteristics =
						SQLITE_IOCAP_ATOMIC |         /* All filesystem writes are atomic */
						SQLITE_IOCAP_SAFE_APPEND |    /* growing the file does not occur until
													  ** the write succeeds */
													  SQLITE_IOCAP_SEQUENTIAL |     /* The ram filesystem has no write behind
																					** so it is ordered */
																					0;
				}else if( !strcmp(fsInfo.f_basetype, "qnx4") ){
					pFile->sectorSize = fsInfo.f_bsize;
					pFile->deviceCharacteristics =
						/* full bitset of atomics from max sector size and smaller */
						((pFile->sectorSize / 512 * SQLITE_IOCAP_ATOMIC512) << 1) - 2 |
						SQLITE_IOCAP_SEQUENTIAL |     /* The ram filesystem has no write behind
													  ** so it is ordered */
													  0;
				}else if( strstr(fsInfo.f_basetype, "dos") ){
					pFile->sectorSize = fsInfo.f_bsize;
					pFile->deviceCharacteristics =
						/* full bitset of atomics from max sector size and smaller */
						((pFile->sectorSize / 512 * SQLITE_IOCAP_ATOMIC512) << 1) - 2 |
						SQLITE_IOCAP_SEQUENTIAL |     /* The ram filesystem has no write behind
													  ** so it is ordered */
													  0;
				}else{
					pFile->deviceCharacteristics =
						SQLITE_IOCAP_ATOMIC512 |      /* blocks are atomic */
						SQLITE_IOCAP_SAFE_APPEND |    /* growing the file does not occur until
													  ** the write succeeds */
													  0;
				}
			}
			/* Last chance verification.  If the sector size isn't a multiple of 512
			** then it isn't valid.*/
			if( pFile->sectorSize % 512 != 0 ){
				pFile->deviceCharacteristics = 0;
				pFile->sectorSize = SQLITE_DEFAULT_SECTOR_SIZE;
			}
			return pFile->sectorSize;
		}
#endif

		static int unixDeviceCharacteristics(sqlite3_file *id){
			unixFile *p = (unixFile*)id;
			int rc = 0;
#ifdef __QNXNTO__
			if( p->sectorSize==0 ) unixSectorSize(id);
			rc = p->deviceCharacteristics;
#endif
			if( p->ctrlFlags & UNIXFILE_PSOW ){
				rc |= SQLITE_IOCAP_POWERSAFE_OVERWRITE;
			}
			return rc;
		}

#ifndef OMIT_WAL

		struct unixShmNode {
			unixInodeInfo *pInode;     /* unixInodeInfo that owns this SHM node */
			sqlite3_mutex *mutex;      /* Mutex to access this object */
			char *zFilename;           /* Name of the mmapped file */
			int h;                     /* Open file descriptor */
			int szRegion;              /* Size of shared-memory regions */
			u16 nRegion;               /* Size of array apRegion */
			u8 isReadonly;             /* True if read-only */
			char **apRegion;           /* Array of mapped shared-memory regions */
			int nRef;                  /* Number of unixShm objects pointing to this */
			unixShm *pFirst;           /* All unixShm objects pointing to this */
#ifdef SQLITE_DEBUG
			u8 exclMask;               /* Mask of exclusive locks held */
			u8 sharedMask;             /* Mask of shared locks held */
			u8 nextShmId;              /* Next available unixShm.id value */
#endif
		};

		struct unixShm {
			unixShmNode *pShmNode;     /* The underlying unixShmNode object */
			unixShm *pNext;            /* Next unixShm with the same unixShmNode */
			u8 hasMutex;               /* True if holding the unixShmNode mutex */
			u8 id;                     /* Id of this connection within its unixShmNode */
			u16 sharedMask;            /* Mask of shared locks held */
			u16 exclMask;              /* Mask of exclusive locks held */
		};

#define UNIX_SHM_BASE   ((22+SQLITE_SHM_NLOCK)*4)         /* first lock byte */
#define UNIX_SHM_DMS    (UNIX_SHM_BASE+SQLITE_SHM_NLOCK)  /* deadman switch */

		static int unixShmSystemLock(
			unixShmNode *pShmNode, /* Apply locks to this open shared-memory segment */
			int lockType,          /* F_UNLCK, F_RDLCK, or F_WRLCK */
			int ofst,              /* First byte of the locking range */
			int n                  /* Number of bytes to lock */
			){
				struct flock f;       /* The posix advisory locking structure */
				int rc = SQLITE_OK;   /* Result code form fcntl() */

				/* Access to the unixShmNode object is serialized by the caller */
				assert( sqlite3_mutex_held(pShmNode->mutex) || pShmNode->nRef==0 );

				/* Shared locks never span more than one byte */
				assert( n==1 || lockType!=F_RDLCK );

				/* Locks are within range */
				assert( n>=1 && n<SQLITE_SHM_NLOCK );

				if( pShmNode->h>=0 ){
					/* Initialize the locking parameters */
					memset(&f, 0, sizeof(f));
					f.l_type = lockType;
					f.l_whence = SEEK_SET;
					f.l_start = ofst;
					f.l_len = n;

					rc = osFcntl(pShmNode->h, F_SETLK, &f);
					rc = (rc!=(-1)) ? SQLITE_OK : SQLITE_BUSY;
				}

				/* Update the global lock state and do debug tracing */
#ifdef _DEBUG
				{ u16 mask;
				OSTRACE(("SHM-LOCK "));
				mask = (1<<(ofst+n)) - (1<<ofst);
				if( rc==SQLITE_OK ){
					if( lockType==F_UNLCK ){
						OSTRACE(("unlock %d ok", ofst));
						pShmNode->exclMask &= ~mask;
						pShmNode->sharedMask &= ~mask;
					}else if( lockType==F_RDLCK ){
						OSTRACE(("read-lock %d ok", ofst));
						pShmNode->exclMask &= ~mask;
						pShmNode->sharedMask |= mask;
					}else{
						assert( lockType==F_WRLCK );
						OSTRACE(("write-lock %d ok", ofst));
						pShmNode->exclMask |= mask;
						pShmNode->sharedMask &= ~mask;
					}
				}else{
					if( lockType==F_UNLCK ){
						OSTRACE(("unlock %d failed", ofst));
					}else if( lockType==F_RDLCK ){
						OSTRACE(("read-lock failed"));
					}else{
						assert( lockType==F_WRLCK );
						OSTRACE(("write-lock %d failed", ofst));
					}
				}
				OSTRACE((" - afterwards %03x,%03x\n",
					pShmNode->sharedMask, pShmNode->exclMask));
				}
#endif

				return rc;        
		}

		static void unixShmPurge(unixFile *pFd){
			unixShmNode *p = pFd->pInode->pShmNode;
			assert( unixMutexHeld() );
			if( p && p->nRef==0 ){
				int i;
				assert( p->pInode==pFd->pInode );
				sqlite3_mutex_free(p->mutex);
				for(i=0; i<p->nRegion; i++){
					if( p->h>=0 ){
						munmap(p->apRegion[i], p->szRegion);
					}else{
						sqlite3_free(p->apRegion[i]);
					}
				}
				sqlite3_free(p->apRegion);
				if( p->h>=0 ){
					robust_close(pFd, p->h, __LINE__);
					p->h = -1;
				}
				p->pInode->pShmNode = 0;
				sqlite3_free(p);
			}
		}

		static int unixOpenSharedMemory(unixFile *pDbFd){
			struct unixShm *p = 0;          /* The connection to be opened */
			struct unixShmNode *pShmNode;   /* The underlying mmapped file */
			int rc;                         /* Result code */
			unixInodeInfo *pInode;          /* The inode of fd */
			char *zShmFilename;             /* Name of the file used for SHM */
			int nShmFilename;               /* Size of the SHM filename in bytes */

			/* Allocate space for the new unixShm object. */
			p = sqlite3_malloc( sizeof(*p) );
			if( p==0 ) return SQLITE_NOMEM;
			memset(p, 0, sizeof(*p));
			assert( pDbFd->pShm==0 );

			/* Check to see if a unixShmNode object already exists. Reuse an existing
			** one if present. Create a new one if necessary.
			*/
			unixEnterMutex();
			pInode = pDbFd->pInode;
			pShmNode = pInode->pShmNode;
			if( pShmNode==0 ){
				struct stat sStat;                 /* fstat() info for database file */

				/* Call fstat() to figure out the permissions on the database file. If
				** a new *-shm file is created, an attempt will be made to create it
				** with the same permissions.
				*/
				if( osFstat(pDbFd->h, &sStat) && pInode->bProcessLock==0 ){
					rc = SQLITE_IOERR_FSTAT;
					goto shm_open_err;
				}

#ifdef SQLITE_SHM_DIRECTORY
				nShmFilename = sizeof(SQLITE_SHM_DIRECTORY) + 31;
#else
				nShmFilename = 6 + (int)strlen(pDbFd->zPath);
#endif
				pShmNode = sqlite3_malloc( sizeof(*pShmNode) + nShmFilename );
				if( pShmNode==0 ){
					rc = SQLITE_NOMEM;
					goto shm_open_err;
				}
				memset(pShmNode, 0, sizeof(*pShmNode)+nShmFilename);
				zShmFilename = pShmNode->zFilename = (char*)&pShmNode[1];
#ifdef SQLITE_SHM_DIRECTORY
				sqlite3_snprintf(nShmFilename, zShmFilename, 
					SQLITE_SHM_DIRECTORY "/sqlite-shm-%x-%x",
					(u32)sStat.st_ino, (u32)sStat.st_dev);
#else
				sqlite3_snprintf(nShmFilename, zShmFilename, "%s-shm", pDbFd->zPath);
				sqlite3FileSuffix3(pDbFd->zPath, zShmFilename);
#endif
				pShmNode->h = -1;
				pDbFd->pInode->pShmNode = pShmNode;
				pShmNode->pInode = pDbFd->pInode;
				pShmNode->mutex = sqlite3_mutex_alloc(SQLITE_MUTEX_FAST);
				if( pShmNode->mutex==0 ){
					rc = SQLITE_NOMEM;
					goto shm_open_err;
				}

				if( pInode->bProcessLock==0 ){
					int openFlags = O_RDWR | O_CREAT;
					if( sqlite3_uri_boolean(pDbFd->zPath, "readonly_shm", 0) ){
						openFlags = O_RDONLY;
						pShmNode->isReadonly = 1;
					}
					pShmNode->h = robust_open(zShmFilename, openFlags, (sStat.st_mode&0777));
					if( pShmNode->h<0 ){
						rc = unixLogError(SQLITE_CANTOPEN_BKPT, "open", zShmFilename);
						goto shm_open_err;
					}

					/* If this process is running as root, make sure that the SHM file
					** is owned by the same user that owns the original database.  Otherwise,
					** the original owner will not be able to connect.
					*/
					osFchown(pShmNode->h, sStat.st_uid, sStat.st_gid);

					/* Check to see if another process is holding the dead-man switch.
					** If not, truncate the file to zero length. 
					*/
					rc = SQLITE_OK;
					if( unixShmSystemLock(pShmNode, F_WRLCK, UNIX_SHM_DMS, 1)==SQLITE_OK ){
						if( robust_ftruncate(pShmNode->h, 0) ){
							rc = unixLogError(SQLITE_IOERR_SHMOPEN, "ftruncate", zShmFilename);
						}
					}
					if( rc==SQLITE_OK ){
						rc = unixShmSystemLock(pShmNode, F_RDLCK, UNIX_SHM_DMS, 1);
					}
					if( rc ) goto shm_open_err;
				}
			}

			/* Make the new connection a child of the unixShmNode */
			p->pShmNode = pShmNode;
#ifdef SQLITE_DEBUG
			p->id = pShmNode->nextShmId++;
#endif
			pShmNode->nRef++;
			pDbFd->pShm = p;
			unixLeaveMutex();

			/* The reference count on pShmNode has already been incremented under
			** the cover of the unixEnterMutex() mutex and the pointer from the
			** new (struct unixShm) object to the pShmNode has been set. All that is
			** left to do is to link the new object into the linked list starting
			** at pShmNode->pFirst. This must be done while holding the pShmNode->mutex 
			** mutex.
			*/
			sqlite3_mutex_enter(pShmNode->mutex);
			p->pNext = pShmNode->pFirst;
			pShmNode->pFirst = p;
			sqlite3_mutex_leave(pShmNode->mutex);
			return SQLITE_OK;

			/* Jump here on any error */
shm_open_err:
			unixShmPurge(pDbFd);       /* This call frees pShmNode if required */
			sqlite3_free(p);
			unixLeaveMutex();
			return rc;
		}

		static int unixShmMap(
			sqlite3_file *fd,               /* Handle open on database file */
			int iRegion,                    /* Region to retrieve */
			int szRegion,                   /* Size of regions */
			int bExtend,                    /* True to extend file if necessary */
			void volatile **pp              /* OUT: Mapped memory */
			){
				unixFile *pDbFd = (unixFile*)fd;
				unixShm *p;
				unixShmNode *pShmNode;
				int rc = SQLITE_OK;

				/* If the shared-memory file has not yet been opened, open it now. */
				if( pDbFd->pShm==0 ){
					rc = unixOpenSharedMemory(pDbFd);
					if( rc!=SQLITE_OK ) return rc;
				}

				p = pDbFd->pShm;
				pShmNode = p->pShmNode;
				sqlite3_mutex_enter(pShmNode->mutex);
				assert( szRegion==pShmNode->szRegion || pShmNode->nRegion==0 );
				assert( pShmNode->pInode==pDbFd->pInode );
				assert( pShmNode->h>=0 || pDbFd->pInode->bProcessLock==1 );
				assert( pShmNode->h<0 || pDbFd->pInode->bProcessLock==0 );

				if( pShmNode->nRegion<=iRegion ){
					char **apNew;                      /* New apRegion[] array */
					int nByte = (iRegion+1)*szRegion;  /* Minimum required file size */
					struct stat sStat;                 /* Used by fstat() */

					pShmNode->szRegion = szRegion;

					if( pShmNode->h>=0 ){
						/* The requested region is not mapped into this processes address space.
						** Check to see if it has been allocated (i.e. if the wal-index file is
						** large enough to contain the requested region).
						*/
						if( osFstat(pShmNode->h, &sStat) ){
							rc = SQLITE_IOERR_SHMSIZE;
							goto shmpage_out;
						}

						if( sStat.st_size<nByte ){
							/* The requested memory region does not exist. If bExtend is set to
							** false, exit early. *pp will be set to NULL and SQLITE_OK returned.
							**
							** Alternatively, if bExtend is true, use ftruncate() to allocate
							** the requested memory region.
							*/
							if( !bExtend ) goto shmpage_out;
#if defined(HAVE_POSIX_FALLOCATE) && HAVE_POSIX_FALLOCATE
							if( osFallocate(pShmNode->h, sStat.st_size, nByte)!=0 ){
								rc = unixLogError(SQLITE_IOERR_SHMSIZE, "fallocate",
									pShmNode->zFilename);
								goto shmpage_out;
							}
#else
							if( robust_ftruncate(pShmNode->h, nByte) ){
								rc = unixLogError(SQLITE_IOERR_SHMSIZE, "ftruncate",
									pShmNode->zFilename);
								goto shmpage_out;
							}
#endif
						}
					}

					/* Map the requested memory region into this processes address space. */
					apNew = (char **)sqlite3_realloc(
						pShmNode->apRegion, (iRegion+1)*sizeof(char *)
						);
					if( !apNew ){
						rc = SQLITE_IOERR_NOMEM;
						goto shmpage_out;
					}
					pShmNode->apRegion = apNew;
					while(pShmNode->nRegion<=iRegion){
						void *pMem;
						if( pShmNode->h>=0 ){
							pMem = mmap(0, szRegion,
								pShmNode->isReadonly ? PROT_READ : PROT_READ|PROT_WRITE, 
								MAP_SHARED, pShmNode->h, szRegion*(i64)pShmNode->nRegion
								);
							if( pMem==MAP_FAILED ){
								rc = unixLogError(SQLITE_IOERR_SHMMAP, "mmap", pShmNode->zFilename);
								goto shmpage_out;
							}
						}else{
							pMem = sqlite3_malloc(szRegion);
							if( pMem==0 ){
								rc = SQLITE_NOMEM;
								goto shmpage_out;
							}
							memset(pMem, 0, szRegion);
						}
						pShmNode->apRegion[pShmNode->nRegion] = pMem;
						pShmNode->nRegion++;
					}
				}

shmpage_out:
				if( pShmNode->nRegion>iRegion ){
					*pp = pShmNode->apRegion[iRegion];
				}else{
					*pp = 0;
				}
				if( pShmNode->isReadonly && rc==SQLITE_OK ) rc = SQLITE_READONLY;
				sqlite3_mutex_leave(pShmNode->mutex);
				return rc;
		}

		static int unixShmLock(
			sqlite3_file *fd,          /* Database file holding the shared memory */
			int ofst,                  /* First lock to acquire or release */
			int n,                     /* Number of locks to acquire or release */
			int flags                  /* What to do with the lock */
			){
				unixFile *pDbFd = (unixFile*)fd;      /* Connection holding shared memory */
				unixShm *p = pDbFd->pShm;             /* The shared memory being locked */
				unixShm *pX;                          /* For looping over all siblings */
				unixShmNode *pShmNode = p->pShmNode;  /* The underlying file iNode */
				int rc = SQLITE_OK;                   /* Result code */
				u16 mask;                             /* Mask of locks to take or release */

				assert( pShmNode==pDbFd->pInode->pShmNode );
				assert( pShmNode->pInode==pDbFd->pInode );
				assert( ofst>=0 && ofst+n<=SQLITE_SHM_NLOCK );
				assert( n>=1 );
				assert( flags==(SQLITE_SHM_LOCK | SQLITE_SHM_SHARED)
					|| flags==(SQLITE_SHM_LOCK | SQLITE_SHM_EXCLUSIVE)
					|| flags==(SQLITE_SHM_UNLOCK | SQLITE_SHM_SHARED)
					|| flags==(SQLITE_SHM_UNLOCK | SQLITE_SHM_EXCLUSIVE) );
				assert( n==1 || (flags & SQLITE_SHM_EXCLUSIVE)!=0 );
				assert( pShmNode->h>=0 || pDbFd->pInode->bProcessLock==1 );
				assert( pShmNode->h<0 || pDbFd->pInode->bProcessLock==0 );

				mask = (1<<(ofst+n)) - (1<<ofst);
				assert( n>1 || mask==(1<<ofst) );
				sqlite3_mutex_enter(pShmNode->mutex);
				if( flags & SQLITE_SHM_UNLOCK ){
					u16 allMask = 0; /* Mask of locks held by siblings */

					/* See if any siblings hold this same lock */
					for(pX=pShmNode->pFirst; pX; pX=pX->pNext){
						if( pX==p ) continue;
						assert( (pX->exclMask & (p->exclMask|p->sharedMask))==0 );
						allMask |= pX->sharedMask;
					}

					/* Unlock the system-level locks */
					if( (mask & allMask)==0 ){
						rc = unixShmSystemLock(pShmNode, F_UNLCK, ofst+UNIX_SHM_BASE, n);
					}else{
						rc = SQLITE_OK;
					}

					/* Undo the local locks */
					if( rc==SQLITE_OK ){
						p->exclMask &= ~mask;
						p->sharedMask &= ~mask;
					} 
				}else if( flags & SQLITE_SHM_SHARED ){
					u16 allShared = 0;  /* Union of locks held by connections other than "p" */

					/* Find out which shared locks are already held by sibling connections.
					** If any sibling already holds an exclusive lock, go ahead and return
					** SQLITE_BUSY.
					*/
					for(pX=pShmNode->pFirst; pX; pX=pX->pNext){
						if( (pX->exclMask & mask)!=0 ){
							rc = SQLITE_BUSY;
							break;
						}
						allShared |= pX->sharedMask;
					}

					/* Get shared locks at the system level, if necessary */
					if( rc==SQLITE_OK ){
						if( (allShared & mask)==0 ){
							rc = unixShmSystemLock(pShmNode, F_RDLCK, ofst+UNIX_SHM_BASE, n);
						}else{
							rc = SQLITE_OK;
						}
					}

					/* Get the local shared locks */
					if( rc==SQLITE_OK ){
						p->sharedMask |= mask;
					}
				}else{
					/* Make sure no sibling connections hold locks that will block this
					** lock.  If any do, return SQLITE_BUSY right away.
					*/
					for(pX=pShmNode->pFirst; pX; pX=pX->pNext){
						if( (pX->exclMask & mask)!=0 || (pX->sharedMask & mask)!=0 ){
							rc = SQLITE_BUSY;
							break;
						}
					}

					/* Get the exclusive locks at the system level.  Then if successful
					** also mark the local connection as being locked.
					*/
					if( rc==SQLITE_OK ){
						rc = unixShmSystemLock(pShmNode, F_WRLCK, ofst+UNIX_SHM_BASE, n);
						if( rc==SQLITE_OK ){
							assert( (p->sharedMask & mask)==0 );
							p->exclMask |= mask;
						}
					}
				}
				sqlite3_mutex_leave(pShmNode->mutex);
				OSTRACE(("SHM-LOCK shmid-%d, pid-%d got %03x,%03x\n",
					p->id, getpid(), p->sharedMask, p->exclMask));
				return rc;
		}

		static void unixShmBarrier(
			sqlite3_file *fd                /* Database file holding the shared memory */
			){
				UNUSED_PARAMETER(fd);
				unixEnterMutex();
				unixLeaveMutex();
		}

		static int unixShmUnmap(
			sqlite3_file *fd,               /* The underlying database file */
			int deleteFlag                  /* Delete shared-memory if true */
			){
				unixShm *p;                     /* The connection to be closed */
				unixShmNode *pShmNode;          /* The underlying shared-memory file */
				unixShm **pp;                   /* For looping over sibling connections */
				unixFile *pDbFd;                /* The underlying database file */

				pDbFd = (unixFile*)fd;
				p = pDbFd->pShm;
				if( p==0 ) return SQLITE_OK;
				pShmNode = p->pShmNode;

				assert( pShmNode==pDbFd->pInode->pShmNode );
				assert( pShmNode->pInode==pDbFd->pInode );

				/* Remove connection p from the set of connections associated
				** with pShmNode */
				sqlite3_mutex_enter(pShmNode->mutex);
				for(pp=&pShmNode->pFirst; (*pp)!=p; pp = &(*pp)->pNext){}
				*pp = p->pNext;

				/* Free the connection p */
				sqlite3_free(p);
				pDbFd->pShm = 0;
				sqlite3_mutex_leave(pShmNode->mutex);

				/* If pShmNode->nRef has reached 0, then close the underlying
				** shared-memory file, too */
				unixEnterMutex();
				assert( pShmNode->nRef>0 );
				pShmNode->nRef--;
				if( pShmNode->nRef==0 ){
					if( deleteFlag && pShmNode->h>=0 ) osUnlink(pShmNode->zFilename);
					unixShmPurge(pDbFd);
				}
				unixLeaveMutex();

				return SQLITE_OK;
		}


#else
# define unixShmMap     0
# define unixShmLock    0
# define unixShmBarrier 0
# define unixShmUnmap   0
#endif /* #ifndef SQLITE_OMIT_WAL */

#pragma endregion

#define IOMETHODS(FINDER, METHOD, VERSION, CLOSE, LOCK, UNLOCK, CKLOCK)      \
	static const sqlite3_io_methods METHOD = {                                   \
	VERSION,                    /* iVersion */                                \
	CLOSE,                      /* xClose */                                  \
	unixRead,                   /* xRead */                                   \
	unixWrite,                  /* xWrite */                                  \
	unixTruncate,               /* xTruncate */                               \
	unixSync,                   /* xSync */                                   \
	unixFileSize,               /* xFileSize */                               \
	LOCK,                       /* xLock */                                   \
	UNLOCK,                     /* xUnlock */                                 \
	CKLOCK,                     /* xCheckReservedLock */                      \
	unixFileControl,            /* xFileControl */                            \
	unixSectorSize,             /* xSectorSize */                             \
	unixDeviceCharacteristics,  /* xDeviceCapabilities */                     \
	unixShmMap,                 /* xShmMap */                                 \
	unixShmLock,                /* xShmLock */                                \
	unixShmBarrier,             /* xShmBarrier */                             \
	unixShmUnmap                /* xShmUnmap */                               \
		};                                                                           \
		static const sqlite3_io_methods *FINDER##Impl(const char *z, unixFile *p){   \
		UNUSED_PARAMETER(z); UNUSED_PARAMETER(p);                                  \
		return &METHOD;                                                            \
		}                                                                            \
		static const sqlite3_io_methods *(*const FINDER)(const char*,unixFile *p)    \
		= FINDER##Impl;


		IOMETHODS(
			posixIoFinder,            /* Finder function name */
			posixIoMethods,           /* sqlite3_io_methods object name */
			2,                        /* shared memory is enabled */
			unixClose,                /* xClose method */
			unixLock,                 /* xLock method */
			unixUnlock,               /* xUnlock method */
			unixCheckReservedLock     /* xCheckReservedLock method */
			)
			IOMETHODS(
			nolockIoFinder,           /* Finder function name */
			nolockIoMethods,          /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			nolockClose,              /* xClose method */
			nolockLock,               /* xLock method */
			nolockUnlock,             /* xUnlock method */
			nolockCheckReservedLock   /* xCheckReservedLock method */
			)
			IOMETHODS(
			dotlockIoFinder,          /* Finder function name */
			dotlockIoMethods,         /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			dotlockClose,             /* xClose method */
			dotlockLock,              /* xLock method */
			dotlockUnlock,            /* xUnlock method */
			dotlockCheckReservedLock  /* xCheckReservedLock method */
			)

#if SQLITE_ENABLE_LOCKING_STYLE && !OS_VXWORKS
			IOMETHODS(
			flockIoFinder,            /* Finder function name */
			flockIoMethods,           /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			flockClose,               /* xClose method */
			flockLock,                /* xLock method */
			flockUnlock,              /* xUnlock method */
			flockCheckReservedLock    /* xCheckReservedLock method */
			)
#endif

#if OS_VXWORKS
			IOMETHODS(
			semIoFinder,              /* Finder function name */
			semIoMethods,             /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			semClose,                 /* xClose method */
			semLock,                  /* xLock method */
			semUnlock,                /* xUnlock method */
			semCheckReservedLock      /* xCheckReservedLock method */
			)
#endif

#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
			IOMETHODS(
			afpIoFinder,              /* Finder function name */
			afpIoMethods,             /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			afpClose,                 /* xClose method */
			afpLock,                  /* xLock method */
			afpUnlock,                /* xUnlock method */
			afpCheckReservedLock      /* xCheckReservedLock method */
			)
#endif

#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
			static int proxyClose(sqlite3_file*);
		static int proxyLock(sqlite3_file*, int);
		static int proxyUnlock(sqlite3_file*, int);
		static int proxyCheckReservedLock(sqlite3_file*, int*);
		IOMETHODS(
			proxyIoFinder,            /* Finder function name */
			proxyIoMethods,           /* sqlite3_io_methods object name */
			1,                        /* shared memory is disabled */
			proxyClose,               /* xClose method */
			proxyLock,                /* xLock method */
			proxyUnlock,              /* xUnlock method */
			proxyCheckReservedLock    /* xCheckReservedLock method */
			)
#endif

			/* nfs lockd on OSX 10.3+ doesn't clear write locks when a read lock is set */
#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
			IOMETHODS(
			nfsIoFinder,               /* Finder function name */
			nfsIoMethods,              /* sqlite3_io_methods object name */
			1,                         /* shared memory is disabled */
			unixClose,                 /* xClose method */
			unixLock,                  /* xLock method */
			nfsUnlock,                 /* xUnlock method */
			unixCheckReservedLock      /* xCheckReservedLock method */
			)
#endif
#if defined(__APPLE__) && ENABLE_LOCKING_STYLE

			static const sqlite3_io_methods *autolockIoFinderImpl(
			const char *filePath,    /* name of the database file */
			unixFile *pNew           /* open file object for the database file */
			){
				static const struct Mapping {
					const char *zFilesystem;              /* Filesystem type name */
					const sqlite3_io_methods *pMethods;   /* Appropriate locking method */
				} aMap[] = {
					{ "hfs",    &posixIoMethods },
					{ "ufs",    &posixIoMethods },
					{ "afpfs",  &afpIoMethods },
					{ "smbfs",  &afpIoMethods },
					{ "webdav", &nolockIoMethods },
					{ 0, 0 }
				};
				int i;
				struct statfs fsInfo;
				struct flock lockInfo;

				if( !filePath ){
					/* If filePath==NULL that means we are dealing with a transient file
					** that does not need to be locked. */
					return &nolockIoMethods;
				}
				if( statfs(filePath, &fsInfo) != -1 ){
					if( fsInfo.f_flags & MNT_RDONLY ){
						return &nolockIoMethods;
					}
					for(i=0; aMap[i].zFilesystem; i++){
						if( strcmp(fsInfo.f_fstypename, aMap[i].zFilesystem)==0 ){
							return aMap[i].pMethods;
						}
					}
				}

				/* Default case. Handles, amongst others, "nfs".
				** Test byte-range lock using fcntl(). If the call succeeds, 
				** assume that the file-system supports POSIX style locks. 
				*/
				lockInfo.l_len = 1;
				lockInfo.l_start = 0;
				lockInfo.l_whence = SEEK_SET;
				lockInfo.l_type = F_RDLCK;
				if( osFcntl(pNew->h, F_GETLK, &lockInfo)!=-1 ) {
					if( strcmp(fsInfo.f_fstypename, "nfs")==0 ){
						return &nfsIoMethods;
					} else {
						return &posixIoMethods;
					}
				}else{
					return &dotlockIoMethods;
				}
		}
		static const sqlite3_io_methods 
			*(*const autolockIoFinder)(const char*,unixFile*) = autolockIoFinderImpl;

#endif

#if OS_VXWORKS && ENABLE_LOCKING_STYLE
		static const sqlite3_io_methods *autolockIoFinderImpl(
			const char *filePath,    /* name of the database file */
			unixFile *pNew           /* the open file object */
			){
				struct flock lockInfo;

				if( !filePath ){
					/* If filePath==NULL that means we are dealing with a transient file
					** that does not need to be locked. */
					return &nolockIoMethods;
				}

				/* Test if fcntl() is supported and use POSIX style locks.
				** Otherwise fall back to the named semaphore method.
				*/
				lockInfo.l_len = 1;
				lockInfo.l_start = 0;
				lockInfo.l_whence = SEEK_SET;
				lockInfo.l_type = F_RDLCK;
				if( osFcntl(pNew->h, F_GETLK, &lockInfo)!=-1 ) {
					return &posixIoMethods;
				}else{
					return &semIoMethods;
				}
		}
		static const sqlite3_io_methods 
			*(*const autolockIoFinder)(const char*,unixFile*) = autolockIoFinderImpl;

#endif
	}
#pragma endregion

#pragma region UnixVSystem

	static int fillInUnixFile(
		sqlite3_vfs *pVfs,      /* Pointer to vfs object */
		int h,                  /* Open file descriptor of file being opened */
		sqlite3_file *pId,      /* Write to the unixFile structure here */
		const char *zFilename,  /* Name of the file being opened */
		int ctrlFlags           /* Zero or more UNIXFILE_* values */
		){
			const sqlite3_io_methods *pLockingStyle;
			unixFile *pNew = (unixFile *)pId;
			int rc = SQLITE_OK;

			assert( pNew->pInode==NULL );

			/* Usually the path zFilename should not be a relative pathname. The
			** exception is when opening the proxy "conch" file in builds that
			** include the special Apple locking styles.
			*/
#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
			assert( zFilename==0 || zFilename[0]=='/' 
				|| pVfs->pAppData==(void*)&autolockIoFinder );
#else
			assert( zFilename==0 || zFilename[0]=='/' );
#endif

			/* No locking occurs in temporary files */
			assert( zFilename!=0 || (ctrlFlags & UNIXFILE_NOLOCK)!=0 );

			OSTRACE(("OPEN    %-3d %s\n", h, zFilename));
			pNew->h = h;
			pNew->pVfs = pVfs;
			pNew->zPath = zFilename;
			pNew->ctrlFlags = (u8)ctrlFlags;
			if( sqlite3_uri_boolean(((ctrlFlags & UNIXFILE_URI) ? zFilename : 0),
				"psow", SQLITE_POWERSAFE_OVERWRITE) ){
					pNew->ctrlFlags |= UNIXFILE_PSOW;
			}
			if( strcmp(pVfs->zName,"unix-excl")==0 ){
				pNew->ctrlFlags |= UNIXFILE_EXCL;
			}

#if OS_VXWORKS
			pNew->pId = vxworksFindFileId(zFilename);
			if( pNew->pId==0 ){
				ctrlFlags |= UNIXFILE_NOLOCK;
				rc = SQLITE_NOMEM;
			}
#endif

			if( ctrlFlags & UNIXFILE_NOLOCK ){
				pLockingStyle = &nolockIoMethods;
			}else{
				pLockingStyle = (**(finder_type*)pVfs->pAppData)(zFilename, pNew);
#if SQLITE_ENABLE_LOCKING_STYLE
				/* Cache zFilename in the locking context (AFP and dotlock override) for
				** proxyLock activation is possible (remote proxy is based on db name)
				** zFilename remains valid until file is closed, to support */
				pNew->lockingContext = (void*)zFilename;
#endif
			}

			if( pLockingStyle == &posixIoMethods
#if defined(__APPLE__) && SQLITE_ENABLE_LOCKING_STYLE
				|| pLockingStyle == &nfsIoMethods
#endif
				){
					unixEnterMutex();
					rc = findInodeInfo(pNew, &pNew->pInode);
					if( rc!=SQLITE_OK ){
						/* If an error occurred in findInodeInfo(), close the file descriptor
						** immediately, before releasing the mutex. findInodeInfo() may fail
						** in two scenarios:
						**
						**   (a) A call to fstat() failed.
						**   (b) A malloc failed.
						**
						** Scenario (b) may only occur if the process is holding no other
						** file descriptors open on the same file. If there were other file
						** descriptors on this file, then no malloc would be required by
						** findInodeInfo(). If this is the case, it is quite safe to close
						** handle h - as it is guaranteed that no posix locks will be released
						** by doing so.
						**
						** If scenario (a) caused the error then things are not so safe. The
						** implicit assumption here is that if fstat() fails, things are in
						** such bad shape that dropping a lock or two doesn't matter much.
						*/
						robust_close(pNew, h, __LINE__);
						h = -1;
					}
					unixLeaveMutex();
			}

#if SQLITE_ENABLE_LOCKING_STYLE && defined(__APPLE__)
			else if( pLockingStyle == &afpIoMethods ){
				/* AFP locking uses the file path so it needs to be included in
				** the afpLockingContext.
				*/
				afpLockingContext *pCtx;
				pNew->lockingContext = pCtx = sqlite3_malloc( sizeof(*pCtx) );
				if( pCtx==0 ){
					rc = SQLITE_NOMEM;
				}else{
					/* NB: zFilename exists and remains valid until the file is closed
					** according to requirement F11141.  So we do not need to make a
					** copy of the filename. */
					pCtx->dbPath = zFilename;
					pCtx->reserved = 0;
					srandomdev();
					unixEnterMutex();
					rc = findInodeInfo(pNew, &pNew->pInode);
					if( rc!=SQLITE_OK ){
						sqlite3_free(pNew->lockingContext);
						robust_close(pNew, h, __LINE__);
						h = -1;
					}
					unixLeaveMutex();        
				}
			}
#endif

			else if( pLockingStyle == &dotlockIoMethods ){
				/* Dotfile locking uses the file path so it needs to be included in
				** the dotlockLockingContext 
				*/
				char *zLockFile;
				int nFilename;
				assert( zFilename!=0 );
				nFilename = (int)strlen(zFilename) + 6;
				zLockFile = (char *)sqlite3_malloc(nFilename);
				if( zLockFile==0 ){
					rc = SQLITE_NOMEM;
				}else{
					sqlite3_snprintf(nFilename, zLockFile, "%s" DOTLOCK_SUFFIX, zFilename);
				}
				pNew->lockingContext = zLockFile;
			}

#if OS_VXWORKS
			else if( pLockingStyle == &semIoMethods ){
				/* Named semaphore locking uses the file path so it needs to be
				** included in the semLockingContext
				*/
				unixEnterMutex();
				rc = findInodeInfo(pNew, &pNew->pInode);
				if( (rc==SQLITE_OK) && (pNew->pInode->pSem==NULL) ){
					char *zSemName = pNew->pInode->aSemName;
					int n;
					sqlite3_snprintf(MAX_PATHNAME, zSemName, "/%s.sem",
						pNew->pId->zCanonicalName);
					for( n=1; zSemName[n]; n++ )
						if( zSemName[n]=='/' ) zSemName[n] = '_';
					pNew->pInode->pSem = sem_open(zSemName, O_CREAT, 0666, 1);
					if( pNew->pInode->pSem == SEM_FAILED ){
						rc = SQLITE_NOMEM;
						pNew->pInode->aSemName[0] = '\0';
					}
				}
				unixLeaveMutex();
			}
#endif

			pNew->lastErrno = 0;
#if OS_VXWORKS
			if( rc!=SQLITE_OK ){
				if( h>=0 ) robust_close(pNew, h, __LINE__);
				h = -1;
				osUnlink(zFilename);
				isDelete = 0;
			}
			if( isDelete ) pNew->ctrlFlags |= UNIXFILE_DELETE;
#endif
			if( rc!=SQLITE_OK ){
				if( h>=0 ) robust_close(pNew, h, __LINE__);
			}else{
				pNew->pMethod = pLockingStyle;
				OpenCounter(+1);
			}
			return rc;
	}

	/*
	** Return the name of a directory in which to put temporary files.
	** If no suitable temporary file directory can be found, return NULL.
	*/
	static const char *unixTempFileDir(void){
		static const char *azDirs[] = {
			0,
			0,
			"/var/tmp",
			"/usr/tmp",
			"/tmp",
			0        /* List terminator */
		};
		unsigned int i;
		struct stat buf;
		const char *zDir = 0;

		azDirs[0] = sqlite3_temp_directory;
		if( !azDirs[1] ) azDirs[1] = getenv("TMPDIR");
		for(i=0; i<sizeof(azDirs)/sizeof(azDirs[0]); zDir=azDirs[i++]){
			if( zDir==0 ) continue;
			if( osStat(zDir, &buf) ) continue;
			if( !S_ISDIR(buf.st_mode) ) continue;
			if( osAccess(zDir, 07) ) continue;
			break;
		}
		return zDir;
	}

	/*
	** Create a temporary file name in zBuf.  zBuf must be allocated
	** by the calling process and must be big enough to hold at least
	** pVfs->mxPathname bytes.
	*/
	static int unixGetTempname(int nBuf, char *zBuf){
		static const unsigned char zChars[] =
			"abcdefghijklmnopqrstuvwxyz"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"0123456789";
		unsigned int i, j;
		const char *zDir;

		/* It's odd to simulate an io-error here, but really this is just
		** using the io-error infrastructure to test that SQLite handles this
		** function failing. 
		*/
		SimulateIOError( return SQLITE_IOERR );

		zDir = unixTempFileDir();
		if( zDir==0 ) zDir = ".";

		/* Check that the output buffer is large enough for the temporary file 
		** name. If it is not, return SQLITE_ERROR.
		*/
		if( (strlen(zDir) + strlen(SQLITE_TEMP_FILE_PREFIX) + 18) >= (size_t)nBuf ){
			return SQLITE_ERROR;
		}

		do{
			sqlite3_snprintf(nBuf-18, zBuf, "%s/"SQLITE_TEMP_FILE_PREFIX, zDir);
			j = (int)strlen(zBuf);
			sqlite3_randomness(15, &zBuf[j]);
			for(i=0; i<15; i++, j++){
				zBuf[j] = (char)zChars[ ((unsigned char)zBuf[j])%(sizeof(zChars)-1) ];
			}
			zBuf[j] = 0;
			zBuf[j+1] = 0;
		}while( osAccess(zBuf,0)==0 );
		return SQLITE_OK;
	}

#if SQLITE_ENABLE_LOCKING_STYLE && defined(__APPLE__)
	/*
	** Routine to transform a unixFile into a proxy-locking unixFile.
	** Implementation in the proxy-lock division, but used by unixOpen()
	** if SQLITE_PREFER_PROXY_LOCKING is defined.
	*/
	static int proxyTransformUnixFile(unixFile*, const char*);
#endif

	/*
	** Search for an unused file descriptor that was opened on the database 
	** file (not a journal or master-journal file) identified by pathname
	** zPath with SQLITE_OPEN_XXX flags matching those passed as the second
	** argument to this function.
	**
	** Such a file descriptor may exist if a database connection was closed
	** but the associated file descriptor could not be closed because some
	** other file descriptor open on the same file is holding a file-lock.
	** Refer to comments in the unixClose() function and the lengthy comment
	** describing "Posix Advisory Locking" at the start of this file for 
	** further details. Also, ticket #4018.
	**
	** If a suitable file descriptor is found, then it is returned. If no
	** such file descriptor is located, -1 is returned.
	*/
	static UnixUnusedFd *findReusableFd(const char *zPath, int flags){
		UnixUnusedFd *pUnused = 0;

		/* Do not search for an unused file descriptor on vxworks. Not because
		** vxworks would not benefit from the change (it might, we're not sure),
		** but because no way to test it is currently available. It is better 
		** not to risk breaking vxworks support for the sake of such an obscure 
		** feature.  */
#if !OS_VXWORKS
		struct stat sStat;                   /* Results of stat() call */

		/* A stat() call may fail for various reasons. If this happens, it is
		** almost certain that an open() call on the same path will also fail.
		** For this reason, if an error occurs in the stat() call here, it is
		** ignored and -1 is returned. The caller will try to open a new file
		** descriptor on the same path, fail, and return an error to SQLite.
		**
		** Even if a subsequent open() call does succeed, the consequences of
		** not searching for a resusable file descriptor are not dire.  */
		if( 0==osStat(zPath, &sStat) ){
			unixInodeInfo *pInode;

			unixEnterMutex();
			pInode = inodeList;
			while( pInode && (pInode->fileId.dev!=sStat.st_dev
				|| pInode->fileId.ino!=sStat.st_ino) ){
					pInode = pInode->pNext;
			}
			if( pInode ){
				UnixUnusedFd **pp;
				for(pp=&pInode->pUnused; *pp && (*pp)->flags!=flags; pp=&((*pp)->pNext));
				pUnused = *pp;
				if( pUnused ){
					*pp = pUnused->pNext;
				}
			}
			unixLeaveMutex();
		}
#endif    /* if !OS_VXWORKS */
		return pUnused;
	}

	/*
	** This function is called by unixOpen() to determine the unix permissions
	** to create new files with. If no error occurs, then SQLITE_OK is returned
	** and a value suitable for passing as the third argument to open(2) is
	** written to *pMode. If an IO error occurs, an SQLite error code is 
	** returned and the value of *pMode is not modified.
	**
	** In most cases cases, this routine sets *pMode to 0, which will become
	** an indication to robust_open() to create the file using
	** SQLITE_DEFAULT_FILE_PERMISSIONS adjusted by the umask.
	** But if the file being opened is a WAL or regular journal file, then 
	** this function queries the file-system for the permissions on the 
	** corresponding database file and sets *pMode to this value. Whenever 
	** possible, WAL and journal files are created using the same permissions 
	** as the associated database file.
	**
	** If the SQLITE_ENABLE_8_3_NAMES option is enabled, then the
	** original filename is unavailable.  But 8_3_NAMES is only used for
	** FAT filesystems and permissions do not matter there, so just use
	** the default permissions.
	*/
	static int findCreateFileMode(
		const char *zPath,              /* Path of file (possibly) being created */
		int flags,                      /* Flags passed as 4th argument to xOpen() */
		mode_t *pMode,                  /* OUT: Permissions to open file with */
		uid_t *pUid,                    /* OUT: uid to set on the file */
		gid_t *pGid                     /* OUT: gid to set on the file */
		){
			int rc = SQLITE_OK;             /* Return Code */
			*pMode = 0;
			*pUid = 0;
			*pGid = 0;
			if( flags & (SQLITE_OPEN_WAL|SQLITE_OPEN_MAIN_JOURNAL) ){
				char zDb[MAX_PATHNAME+1];     /* Database file path */
				int nDb;                      /* Number of valid bytes in zDb */
				struct stat sStat;            /* Output of stat() on database file */

				/* zPath is a path to a WAL or journal file. The following block derives
				** the path to the associated database file from zPath. This block handles
				** the following naming conventions:
				**
				**   "<path to db>-journal"
				**   "<path to db>-wal"
				**   "<path to db>-journalNN"
				**   "<path to db>-walNN"
				**
				** where NN is a decimal number. The NN naming schemes are 
				** used by the test_multiplex.c module.
				*/
				nDb = sqlite3Strlen30(zPath) - 1; 
#ifdef SQLITE_ENABLE_8_3_NAMES
				while( nDb>0 && sqlite3Isalnum(zPath[nDb]) ) nDb--;
				if( nDb==0 || zPath[nDb]!='-' ) return SQLITE_OK;
#else
				while( zPath[nDb]!='-' ){
					assert( nDb>0 );
					assert( zPath[nDb]!='\n' );
					nDb--;
				}
#endif
				memcpy(zDb, zPath, nDb);
				zDb[nDb] = '\0';

				if( 0==osStat(zDb, &sStat) ){
					*pMode = sStat.st_mode & 0777;
					*pUid = sStat.st_uid;
					*pGid = sStat.st_gid;
				}else{
					rc = SQLITE_IOERR_FSTAT;
				}
			}else if( flags & SQLITE_OPEN_DELETEONCLOSE ){
				*pMode = 0600;
			}
			return rc;
	}

	/*
	** Open the file zPath.
	** 
	** Previously, the SQLite OS layer used three functions in place of this
	** one:
	**
	**     sqlite3OsOpenReadWrite();
	**     sqlite3OsOpenReadOnly();
	**     sqlite3OsOpenExclusive();
	**
	** These calls correspond to the following combinations of flags:
	**
	**     ReadWrite() ->     (READWRITE | CREATE)
	**     ReadOnly()  ->     (READONLY) 
	**     OpenExclusive() -> (READWRITE | CREATE | EXCLUSIVE)
	**
	** The old OpenExclusive() accepted a boolean argument - "delFlag". If
	** true, the file was configured to be automatically deleted when the
	** file handle closed. To achieve the same effect using this new 
	** interface, add the DELETEONCLOSE flag to those specified above for 
	** OpenExclusive().
	*/
	static int unixOpen(
		sqlite3_vfs *pVfs,           /* The VFS for which this is the xOpen method */
		const char *zPath,           /* Pathname of file to be opened */
		sqlite3_file *pFile,         /* The file descriptor to be filled in */
		int flags,                   /* Input flags to control the opening */
		int *pOutFlags               /* Output flags returned to SQLite core */
		){
			unixFile *p = (unixFile *)pFile;
			int fd = -1;                   /* File descriptor returned by open() */
			int openFlags = 0;             /* Flags to pass to open() */
			int eType = flags&0xFFFFFF00;  /* Type of file to open */
			int noLock;                    /* True to omit locking primitives */
			int rc = SQLITE_OK;            /* Function Return Code */
			int ctrlFlags = 0;             /* UNIXFILE_* flags */

			int isExclusive  = (flags & SQLITE_OPEN_EXCLUSIVE);
			int isDelete     = (flags & SQLITE_OPEN_DELETEONCLOSE);
			int isCreate     = (flags & SQLITE_OPEN_CREATE);
			int isReadonly   = (flags & SQLITE_OPEN_READONLY);
			int isReadWrite  = (flags & SQLITE_OPEN_READWRITE);
#if SQLITE_ENABLE_LOCKING_STYLE
			int isAutoProxy  = (flags & SQLITE_OPEN_AUTOPROXY);
#endif
#if defined(__APPLE__) || SQLITE_ENABLE_LOCKING_STYLE
			struct statfs fsInfo;
#endif

			/* If creating a master or main-file journal, this function will open
			** a file-descriptor on the directory too. The first time unixSync()
			** is called the directory file descriptor will be fsync()ed and close()d.
			*/
			int syncDir = (isCreate && (
				eType==SQLITE_OPEN_MASTER_JOURNAL 
				|| eType==SQLITE_OPEN_MAIN_JOURNAL 
				|| eType==SQLITE_OPEN_WAL
				));

			/* If argument zPath is a NULL pointer, this function is required to open
			** a temporary file. Use this buffer to store the file name in.
			*/
			char zTmpname[MAX_PATHNAME+2];
			const char *zName = zPath;

			/* Check the following statements are true: 
			**
			**   (a) Exactly one of the READWRITE and READONLY flags must be set, and 
			**   (b) if CREATE is set, then READWRITE must also be set, and
			**   (c) if EXCLUSIVE is set, then CREATE must also be set.
			**   (d) if DELETEONCLOSE is set, then CREATE must also be set.
			*/
			assert((isReadonly==0 || isReadWrite==0) && (isReadWrite || isReadonly));
			assert(isCreate==0 || isReadWrite);
			assert(isExclusive==0 || isCreate);
			assert(isDelete==0 || isCreate);

			/* The main DB, main journal, WAL file and master journal are never 
			** automatically deleted. Nor are they ever temporary files.  */
			assert( (!isDelete && zName) || eType!=SQLITE_OPEN_MAIN_DB );
			assert( (!isDelete && zName) || eType!=SQLITE_OPEN_MAIN_JOURNAL );
			assert( (!isDelete && zName) || eType!=SQLITE_OPEN_MASTER_JOURNAL );
			assert( (!isDelete && zName) || eType!=SQLITE_OPEN_WAL );

			/* Assert that the upper layer has set one of the "file-type" flags. */
			assert( eType==SQLITE_OPEN_MAIN_DB      || eType==SQLITE_OPEN_TEMP_DB 
				|| eType==SQLITE_OPEN_MAIN_JOURNAL || eType==SQLITE_OPEN_TEMP_JOURNAL 
				|| eType==SQLITE_OPEN_SUBJOURNAL   || eType==SQLITE_OPEN_MASTER_JOURNAL 
				|| eType==SQLITE_OPEN_TRANSIENT_DB || eType==SQLITE_OPEN_WAL
				);

			memset(p, 0, sizeof(unixFile));

			if( eType==SQLITE_OPEN_MAIN_DB ){
				UnixUnusedFd *pUnused;
				pUnused = findReusableFd(zName, flags);
				if( pUnused ){
					fd = pUnused->fd;
				}else{
					pUnused = sqlite3_malloc(sizeof(*pUnused));
					if( !pUnused ){
						return SQLITE_NOMEM;
					}
				}
				p->pUnused = pUnused;

				/* Database filenames are double-zero terminated if they are not
				** URIs with parameters.  Hence, they can always be passed into
				** sqlite3_uri_parameter(). */
				assert( (flags & SQLITE_OPEN_URI) || zName[strlen(zName)+1]==0 );

			}else if( !zName ){
				/* If zName is NULL, the upper layer is requesting a temp file. */
				assert(isDelete && !syncDir);
				rc = unixGetTempname(MAX_PATHNAME+2, zTmpname);
				if( rc!=SQLITE_OK ){
					return rc;
				}
				zName = zTmpname;

				/* Generated temporary filenames are always double-zero terminated
				** for use by sqlite3_uri_parameter(). */
				assert( zName[strlen(zName)+1]==0 );
			}

			/* Determine the value of the flags parameter passed to POSIX function
			** open(). These must be calculated even if open() is not called, as
			** they may be stored as part of the file handle and used by the 
			** 'conch file' locking functions later on.  */
			if( isReadonly )  openFlags |= O_RDONLY;
			if( isReadWrite ) openFlags |= O_RDWR;
			if( isCreate )    openFlags |= O_CREAT;
			if( isExclusive ) openFlags |= (O_EXCL|O_NOFOLLOW);
			openFlags |= (O_LARGEFILE|O_BINARY);

			if( fd<0 ){
				mode_t openMode;              /* Permissions to create file with */
				uid_t uid;                    /* Userid for the file */
				gid_t gid;                    /* Groupid for the file */
				rc = findCreateFileMode(zName, flags, &openMode, &uid, &gid);
				if( rc!=SQLITE_OK ){
					assert( !p->pUnused );
					assert( eType==SQLITE_OPEN_WAL || eType==SQLITE_OPEN_MAIN_JOURNAL );
					return rc;
				}
				fd = robust_open(zName, openFlags, openMode);
				OSTRACE(("OPENX   %-3d %s 0%o\n", fd, zName, openFlags));
				if( fd<0 && errno!=EISDIR && isReadWrite && !isExclusive ){
					/* Failed to open the file for read/write access. Try read-only. */
					flags &= ~(SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE);
					openFlags &= ~(O_RDWR|O_CREAT);
					flags |= SQLITE_OPEN_READONLY;
					openFlags |= O_RDONLY;
					isReadonly = 1;
					fd = robust_open(zName, openFlags, openMode);
				}
				if( fd<0 ){
					rc = unixLogError(SQLITE_CANTOPEN_BKPT, "open", zName);
					goto open_finished;
				}

				/* If this process is running as root and if creating a new rollback
				** journal or WAL file, set the ownership of the journal or WAL to be
				** the same as the original database.
				*/
				if( flags & (SQLITE_OPEN_WAL|SQLITE_OPEN_MAIN_JOURNAL) ){
					osFchown(fd, uid, gid);
				}
			}
			assert( fd>=0 );
			if( pOutFlags ){
				*pOutFlags = flags;
			}

			if( p->pUnused ){
				p->pUnused->fd = fd;
				p->pUnused->flags = flags;
			}

			if( isDelete ){
#if OS_VXWORKS
				zPath = zName;
#else
				osUnlink(zName);
#endif
			}
#if SQLITE_ENABLE_LOCKING_STYLE
			else{
				p->openFlags = openFlags;
			}
#endif

			noLock = eType!=SQLITE_OPEN_MAIN_DB;


#if defined(__APPLE__) || SQLITE_ENABLE_LOCKING_STYLE
			if( fstatfs(fd, &fsInfo) == -1 ){
				((unixFile*)pFile)->lastErrno = errno;
				robust_close(p, fd, __LINE__);
				return SQLITE_IOERR_ACCESS;
			}
			if (0 == strncmp("msdos", fsInfo.f_fstypename, 5)) {
				((unixFile*)pFile)->fsFlags |= SQLITE_FSFLAGS_IS_MSDOS;
			}
#endif

			/* Set up appropriate ctrlFlags */
			if( isDelete )                ctrlFlags |= UNIXFILE_DELETE;
			if( isReadonly )              ctrlFlags |= UNIXFILE_RDONLY;
			if( noLock )                  ctrlFlags |= UNIXFILE_NOLOCK;
			if( syncDir )                 ctrlFlags |= UNIXFILE_DIRSYNC;
			if( flags & SQLITE_OPEN_URI ) ctrlFlags |= UNIXFILE_URI;

#if SQLITE_ENABLE_LOCKING_STYLE
#if SQLITE_PREFER_PROXY_LOCKING
			isAutoProxy = 1;
#endif
			if( isAutoProxy && (zPath!=NULL) && (!noLock) && pVfs->xOpen ){
				char *envforce = getenv("SQLITE_FORCE_PROXY_LOCKING");
				int useProxy = 0;

				/* SQLITE_FORCE_PROXY_LOCKING==1 means force always use proxy, 0 means 
				** never use proxy, NULL means use proxy for non-local files only.  */
				if( envforce!=NULL ){
					useProxy = atoi(envforce)>0;
				}else{
					if( statfs(zPath, &fsInfo) == -1 ){
						/* In theory, the close(fd) call is sub-optimal. If the file opened
						** with fd is a database file, and there are other connections open
						** on that file that are currently holding advisory locks on it,
						** then the call to close() will cancel those locks. In practice,
						** we're assuming that statfs() doesn't fail very often. At least
						** not while other file descriptors opened by the same process on
						** the same file are working.  */
						p->lastErrno = errno;
						robust_close(p, fd, __LINE__);
						rc = SQLITE_IOERR_ACCESS;
						goto open_finished;
					}
					useProxy = !(fsInfo.f_flags&MNT_LOCAL);
				}
				if( useProxy ){
					rc = fillInUnixFile(pVfs, fd, pFile, zPath, ctrlFlags);
					if( rc==SQLITE_OK ){
						rc = proxyTransformUnixFile((unixFile*)pFile, ":auto:");
						if( rc!=SQLITE_OK ){
							/* Use unixClose to clean up the resources added in fillInUnixFile 
							** and clear all the structure's references.  Specifically, 
							** pFile->pMethods will be NULL so sqlite3OsClose will be a no-op 
							*/
							unixClose(pFile);
							return rc;
						}
					}
					goto open_finished;
				}
			}
#endif

			rc = fillInUnixFile(pVfs, fd, pFile, zPath, ctrlFlags);

open_finished:
			if( rc!=SQLITE_OK ){
				sqlite3_free(p->pUnused);
			}
			return rc;
	}


	/*
	** Delete the file at zPath. If the dirSync argument is true, fsync()
	** the directory after deleting the file.
	*/
	static int unixDelete(
		sqlite3_vfs *NotUsed,     /* VFS containing this as the xDelete method */
		const char *zPath,        /* Name of file to be deleted */
		int dirSync               /* If true, fsync() directory after deleting file */
		){
			int rc = SQLITE_OK;
			UNUSED_PARAMETER(NotUsed);
			SimulateIOError(return SQLITE_IOERR_DELETE);
			if( osUnlink(zPath)==(-1) ){
				if( errno==ENOENT ){
					rc = SQLITE_IOERR_DELETE_NOENT;
				}else{
					rc = unixLogError(SQLITE_IOERR_DELETE, "unlink", zPath);
				}
				return rc;
			}
#ifndef SQLITE_DISABLE_DIRSYNC
			if( (dirSync & 1)!=0 ){
				int fd;
				rc = osOpenDirectory(zPath, &fd);
				if( rc==SQLITE_OK ){
#if OS_VXWORKS
					if( fsync(fd)==-1 )
#else
					if( fsync(fd) )
#endif
					{
						rc = unixLogError(SQLITE_IOERR_DIR_FSYNC, "fsync", zPath);
					}
					robust_close(0, fd, __LINE__);
				}else if( rc==SQLITE_CANTOPEN ){
					rc = SQLITE_OK;
				}
			}
#endif
			return rc;
	}

	/*
	** Test the existence of or access permissions of file zPath. The
	** test performed depends on the value of flags:
	**
	**     SQLITE_ACCESS_EXISTS: Return 1 if the file exists
	**     SQLITE_ACCESS_READWRITE: Return 1 if the file is read and writable.
	**     SQLITE_ACCESS_READONLY: Return 1 if the file is readable.
	**
	** Otherwise return 0.
	*/
	static int unixAccess(
		sqlite3_vfs *NotUsed,   /* The VFS containing this xAccess method */
		const char *zPath,      /* Path of the file to examine */
		int flags,              /* What do we want to learn about the zPath file? */
		int *pResOut            /* Write result boolean here */
		){
			int amode = 0;
			UNUSED_PARAMETER(NotUsed);
			SimulateIOError( return SQLITE_IOERR_ACCESS; );
			switch( flags ){
			case SQLITE_ACCESS_EXISTS:
				amode = F_OK;
				break;
			case SQLITE_ACCESS_READWRITE:
				amode = W_OK|R_OK;
				break;
			case SQLITE_ACCESS_READ:
				amode = R_OK;
				break;

			default:
				assert(!"Invalid flags argument");
			}
			*pResOut = (osAccess(zPath, amode)==0);
			if( flags==SQLITE_ACCESS_EXISTS && *pResOut ){
				struct stat buf;
				if( 0==osStat(zPath, &buf) && buf.st_size==0 ){
					*pResOut = 0;
				}
			}
			return SQLITE_OK;
	}


	/*
	** Turn a relative pathname into a full pathname. The relative path
	** is stored as a nul-terminated string in the buffer pointed to by
	** zPath. 
	**
	** zOut points to a buffer of at least sqlite3_vfs.mxPathname bytes 
	** (in this case, MAX_PATHNAME bytes). The full-path is written to
	** this buffer before returning.
	*/
	static int unixFullPathname(
		sqlite3_vfs *pVfs,            /* Pointer to vfs object */
		const char *zPath,            /* Possibly relative input path */
		int nOut,                     /* Size of output buffer in bytes */
		char *zOut                    /* Output buffer */
		){

			/* It's odd to simulate an io-error here, but really this is just
			** using the io-error infrastructure to test that SQLite handles this
			** function failing. This function could fail if, for example, the
			** current working directory has been unlinked.
			*/
			SimulateIOError( return SQLITE_ERROR );

			assert( pVfs->mxPathname==MAX_PATHNAME );
			UNUSED_PARAMETER(pVfs);

			zOut[nOut-1] = '\0';
			if( zPath[0]=='/' ){
				sqlite3_snprintf(nOut, zOut, "%s", zPath);
			}else{
				int nCwd;
				if( osGetcwd(zOut, nOut-1)==0 ){
					return unixLogError(SQLITE_CANTOPEN_BKPT, "getcwd", zPath);
				}
				nCwd = (int)strlen(zOut);
				sqlite3_snprintf(nOut-nCwd, &zOut[nCwd], "/%s", zPath);
			}
			return SQLITE_OK;
	}


#ifndef SQLITE_OMIT_LOAD_EXTENSION
	/*
	** Interfaces for opening a shared library, finding entry points
	** within the shared library, and closing the shared library.
	*/
#include <dlfcn.h>
	static void *unixDlOpen(sqlite3_vfs *NotUsed, const char *zFilename){
		UNUSED_PARAMETER(NotUsed);
		return dlopen(zFilename, RTLD_NOW | RTLD_GLOBAL);
	}

	/*
	** SQLite calls this function immediately after a call to unixDlSym() or
	** unixDlOpen() fails (returns a null pointer). If a more detailed error
	** message is available, it is written to zBufOut. If no error message
	** is available, zBufOut is left unmodified and SQLite uses a default
	** error message.
	*/
	static void unixDlError(sqlite3_vfs *NotUsed, int nBuf, char *zBufOut){
		const char *zErr;
		UNUSED_PARAMETER(NotUsed);
		unixEnterMutex();
		zErr = dlerror();
		if( zErr ){
			sqlite3_snprintf(nBuf, zBufOut, "%s", zErr);
		}
		unixLeaveMutex();
	}
	static void (*unixDlSym(sqlite3_vfs *NotUsed, void *p, const char*zSym))(void){
		/* 
		** GCC with -pedantic-errors says that C90 does not allow a void* to be
		** cast into a pointer to a function.  And yet the library dlsym() routine
		** returns a void* which is really a pointer to a function.  So how do we
		** use dlsym() with -pedantic-errors?
		**
		** Variable x below is defined to be a pointer to a function taking
		** parameters void* and const char* and returning a pointer to a function.
		** We initialize x by assigning it a pointer to the dlsym() function.
		** (That assignment requires a cast.)  Then we call the function that
		** x points to.  
		**
		** This work-around is unlikely to work correctly on any system where
		** you really cannot cast a function pointer into void*.  But then, on the
		** other hand, dlsym() will not work on such a system either, so we have
		** not really lost anything.
		*/
		void (*(*x)(void*,const char*))(void);
		UNUSED_PARAMETER(NotUsed);
		x = (void(*(*)(void*,const char*))(void))dlsym;
		return (*x)(p, zSym);
	}
	static void unixDlClose(sqlite3_vfs *NotUsed, void *pHandle){
		UNUSED_PARAMETER(NotUsed);
		dlclose(pHandle);
	}
#else /* if SQLITE_OMIT_LOAD_EXTENSION is defined: */
#define unixDlOpen  0
#define unixDlError 0
#define unixDlSym   0
#define unixDlClose 0
#endif

	int UnixVSystem::Randomness(int bufLength, char *buf)
	{
		assert((size_t)bufLength>=(sizeof(time_t)+sizeof(int)));

		// We have to initialize buf to prevent valgrind from reporting errors.  The reports issued by valgrind are incorrect - we would
		// prefer that the randomness be increased by making use of the uninitialized space in buf - but valgrind errors tend to worry
		// some users.  Rather than argue, it seems easier just to initialize the whole array and silence valgrind, even if that means less randomness
		// in the random seed.
		//
		// When testing, initializing buf[] to zero is all we do.  That means that we always use the same random number sequence.  This makes the
		// tests repeatable.
		memset(buf, 0, bufLength);
#if !defined(_TEST)
		{
			int pid, fd, got;
			fd = robust_open("/dev/urandom", O_RDONLY, 0);
			if (fd < 0)
			{
				time_t t;
				time(&t);
				memcpy(buf, &t, sizeof(t));
				pid = getpid();
				memcpy(&buf[sizeof(t)], &pid, sizeof(pid));
				assert( sizeof(t)+sizeof(pid)<=(size_t)bufLength );
				bufLength = sizeof(t) + sizeof(pid);
			}
			else
			{
				do { got = osRead(fd, buf, bufLength); }while( got<0 && errno==EINTR );
				robust_close(0, fd, __LINE__);
			}
		}
#endif
		return bufLength;
	}

	int UnixVSystem::Sleep(int microseconds)
	{
#if OS_VXWORKS
		struct timespec sp;
		sp.tv_sec = microseconds / 1000000;
		sp.tv_nsec = (microseconds % 1000000) * 1000;
		nanosleep(&sp, NULL);
		return microseconds;
#elif defined(HAVE_USLEEP) && HAVE_USLEEP
		usleep(microseconds);
		return microseconds;
#else
		int seconds = (microseconds+999999)/1000000;
		sleep(seconds);
		return seconds*1000000;
#endif
	}

#ifdef _TEST
	int _current_time = 0; // Fake system time in seconds since 1970.
#endif
	RC UnixVSystem::CurrentTimeInt64(int64 *now)
	{
		static const int64 unixEpoch = 24405875*(int64)8640000;
		int rc = RC_OK;
#if defined(NO_GETTOD)
		time_t t;
		time(&t);
		*now = ((int64)t)*1000 + unixEpoch;
#elif OS_VXWORKS
		struct timespec sNow;
		clock_gettime(CLOCK_REALTIME, &sNow);
		*now = unixEpoch + 1000*(int64)sNow.tv_sec + sNow.tv_nsec/1000000;
#else
		struct timeval sNow;
		if (!gettimeofday(&sNow, 0))
			*now = unixEpoch + 1000*(int64)sNow.tv_sec + sNow.tv_usec/1000;
		else
			rc = RC_ERROR;
#endif
#ifdef _TEST
		if (_current_time)
			*now = 1000*(int64)_current_time + unixEpoch;
#endif
		return rc;
	}


	RC UnixVSystem::CurrentTime(double *now)
	{
		int64 i = 0;
		int rc = CurrentTimeInt64(&i);
		*now = i/86400000.0;
		return rc;
	}

	RC UnixVSystem::GetLastError(int bufLength, char *buf)
	{
		return RC_OK;
	}

#pragma region Proxy Locking
	// Proxy locking is only available on MacOSX 
#if defined(__APPLE__) && ENABLE_LOCKING_STYLE

	typedef struct proxyLockingContext proxyLockingContext;
	struct proxyLockingContext {
		unixFile *conchFile;         /* Open conch file */
		char *conchFilePath;         /* Name of the conch file */
		unixFile *lockProxy;         /* Open proxy lock file */
		char *lockProxyPath;         /* Name of the proxy lock file */
		char *dbPath;                /* Name of the open file */
		int conchHeld;               /* 1 if the conch is held, -1 if lockless */
		void *oldLockingContext;     /* Original lockingcontext to restore on close */
		sqlite3_io_methods const *pOldMethod;     /* Original I/O methods for close */
	};

	static int proxyGetLockPath(const char *dbPath, char *lPath, size_t maxLen){
		int len;
		int dbLen;
		int i;

#ifdef LOCKPROXYDIR
		len = strlcpy(lPath, LOCKPROXYDIR, maxLen);
#else
# ifdef _CS_DARWIN_USER_TEMP_DIR
		{
			if( !confstr(_CS_DARWIN_USER_TEMP_DIR, lPath, maxLen) ){
				OSTRACE(("GETLOCKPATH  failed %s errno=%d pid=%d\n",
					lPath, errno, getpid()));
				return SQLITE_IOERR_LOCK;
			}
			len = strlcat(lPath, "sqliteplocks", maxLen);    
		}
# else
		len = strlcpy(lPath, "/tmp/", maxLen);
# endif
#endif

		if( lPath[len-1]!='/' ){
			len = strlcat(lPath, "/", maxLen);
		}

		/* transform the db path to a unique cache name */
		dbLen = (int)strlen(dbPath);
		for( i=0; i<dbLen && (i+len+7)<(int)maxLen; i++){
			char c = dbPath[i];
			lPath[i+len] = (c=='/')?'_':c;
		}
		lPath[i+len]='\0';
		strlcat(lPath, ":auto:", maxLen);
		OSTRACE(("GETLOCKPATH  proxy lock path=%s pid=%d\n", lPath, getpid()));
		return SQLITE_OK;
	}

	static int proxyCreateLockPath(const char *lockPath){
		int i, len;
		char buf[MAXPATHLEN];
		int start = 0;

		assert(lockPath!=NULL);
		/* try to create all the intermediate directories */
		len = (int)strlen(lockPath);
		buf[0] = lockPath[0];
		for( i=1; i<len; i++ ){
			if( lockPath[i] == '/' && (i - start > 0) ){
				/* only mkdir if leaf dir != "." or "/" or ".." */
				if( i-start>2 || (i-start==1 && buf[start] != '.' && buf[start] != '/') 
					|| (i-start==2 && buf[start] != '.' && buf[start+1] != '.') ){
						buf[i]='\0';
						if( osMkdir(buf, SQLITE_DEFAULT_PROXYDIR_PERMISSIONS) ){
							int err=errno;
							if( err!=EEXIST ) {
								OSTRACE(("CREATELOCKPATH  FAILED creating %s, "
									"'%s' proxy lock path=%s pid=%d\n",
									buf, strerror(err), lockPath, getpid()));
								return err;
							}
						}
				}
				start=i+1;
			}
			buf[i] = lockPath[i];
		}
		OSTRACE(("CREATELOCKPATH  proxy lock path=%s pid=%d\n", lockPath, getpid()));
		return 0;
	}

	static int proxyCreateUnixFile(
		const char *path,        /* path for the new unixFile */
		unixFile **ppFile,       /* unixFile created and returned by ref */
		int islockfile           /* if non zero missing dirs will be created */
		) {
			int fd = -1;
			unixFile *pNew;
			int rc = SQLITE_OK;
			int openFlags = O_RDWR | O_CREAT;
			sqlite3_vfs dummyVfs;
			int terrno = 0;
			UnixUnusedFd *pUnused = NULL;

			/* 1. first try to open/create the file
			** 2. if that fails, and this is a lock file (not-conch), try creating
			** the parent directories and then try again.
			** 3. if that fails, try to open the file read-only
			** otherwise return BUSY (if lock file) or CANTOPEN for the conch file
			*/
			pUnused = findReusableFd(path, openFlags);
			if( pUnused ){
				fd = pUnused->fd;
			}else{
				pUnused = sqlite3_malloc(sizeof(*pUnused));
				if( !pUnused ){
					return SQLITE_NOMEM;
				}
			}
			if( fd<0 ){
				fd = robust_open(path, openFlags, 0);
				terrno = errno;
				if( fd<0 && errno==ENOENT && islockfile ){
					if( proxyCreateLockPath(path) == SQLITE_OK ){
						fd = robust_open(path, openFlags, 0);
					}
				}
			}
			if( fd<0 ){
				openFlags = O_RDONLY;
				fd = robust_open(path, openFlags, 0);
				terrno = errno;
			}
			if( fd<0 ){
				if( islockfile ){
					return SQLITE_BUSY;
				}
				switch (terrno) {
				case EACCES:
					return SQLITE_PERM;
				case EIO: 
					return SQLITE_IOERR_LOCK; /* even though it is the conch */
				default:
					return SQLITE_CANTOPEN_BKPT;
				}
			}

			pNew = (unixFile *)sqlite3_malloc(sizeof(*pNew));
			if( pNew==NULL ){
				rc = SQLITE_NOMEM;
				goto end_create_proxy;
			}
			memset(pNew, 0, sizeof(unixFile));
			pNew->openFlags = openFlags;
			memset(&dummyVfs, 0, sizeof(dummyVfs));
			dummyVfs.pAppData = (void*)&autolockIoFinder;
			dummyVfs.zName = "dummy";
			pUnused->fd = fd;
			pUnused->flags = openFlags;
			pNew->pUnused = pUnused;

			rc = fillInUnixFile(&dummyVfs, fd, (sqlite3_file*)pNew, path, 0);
			if( rc==SQLITE_OK ){
				*ppFile = pNew;
				return SQLITE_OK;
			}
end_create_proxy:    
			robust_close(pNew, fd, __LINE__);
			sqlite3_free(pNew);
			sqlite3_free(pUnused);
			return rc;
	}

#ifdef _TEST
	int sqlite3_hostid_num = 0; // simulate multiple hosts by creating unique hostid file paths
#endif

#define PROXY_HOSTIDLEN    16  /* conch file host id length */

	extern int gethostuuid(uuid_t id, const struct timespec *wait); // Not always defined in the headers as it ought to be

	static int proxyGetHostID(unsigned char *pHostID, int *pError){
		assert(PROXY_HOSTIDLEN == sizeof(uuid_t));
		memset(pHostID, 0, PROXY_HOSTIDLEN);
#if defined(__MAX_OS_X_VERSION_MIN_REQUIRED)\
	&& __MAC_OS_X_VERSION_MIN_REQUIRED<1050
		{
			static const struct timespec timeout = {1, 0}; /* 1 sec timeout */
			if( gethostuuid(pHostID, &timeout) ){
				int err = errno;
				if( pError ){
					*pError = err;
				}
				return SQLITE_IOERR;
			}
		}
#else
		UNUSED_PARAMETER(pError);
#endif
#ifdef _TEST
		/* simulate multiple hosts by creating unique hostid file paths */
		if( sqlite3_hostid_num != 0){
			pHostID[0] = (char)(pHostID[0] + (char)(sqlite3_hostid_num & 0xFF));
		}
#endif

		return SQLITE_OK;
	}

	// The conch file contains the header, host id and lock file path
#define PROXY_CONCHVERSION 2   /* 1-byte header, 16-byte host id, path */
#define PROXY_HEADERLEN    1   /* conch file header length */
#define PROXY_PATHINDEX    (PROXY_HEADERLEN+PROXY_HOSTIDLEN)
#define PROXY_MAXCONCHLEN  (PROXY_HEADERLEN+PROXY_HOSTIDLEN+MAXPATHLEN)

	static int proxyBreakConchLock(unixFile *pFile, uuid_t myHostID){
		proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext; 
		unixFile *conchFile = pCtx->conchFile;
		char tPath[MAXPATHLEN];
		char buf[PROXY_MAXCONCHLEN];
		char *cPath = pCtx->conchFilePath;
		size_t readLen = 0;
		size_t pathLen = 0;
		char errmsg[64] = "";
		int fd = -1;
		int rc = -1;
		UNUSED_PARAMETER(myHostID);

		/* create a new path by replace the trailing '-conch' with '-break' */
		pathLen = strlcpy(tPath, cPath, MAXPATHLEN);
		if( pathLen>MAXPATHLEN || pathLen<6 || 
			(strlcpy(&tPath[pathLen-5], "break", 6) != 5) ){
				sqlite3_snprintf(sizeof(errmsg),errmsg,"path error (len %d)",(int)pathLen);
				goto end_breaklock;
		}
		/* read the conch content */
		readLen = osPread(conchFile->h, buf, PROXY_MAXCONCHLEN, 0);
		if( readLen<PROXY_PATHINDEX ){
			sqlite3_snprintf(sizeof(errmsg),errmsg,"read error (len %d)",(int)readLen);
			goto end_breaklock;
		}
		/* write it out to the temporary break file */
		fd = robust_open(tPath, (O_RDWR|O_CREAT|O_EXCL), 0);
		if( fd<0 ){
			sqlite3_snprintf(sizeof(errmsg), errmsg, "create failed (%d)", errno);
			goto end_breaklock;
		}
		if( osPwrite(fd, buf, readLen, 0) != (ssize_t)readLen ){
			sqlite3_snprintf(sizeof(errmsg), errmsg, "write failed (%d)", errno);
			goto end_breaklock;
		}
		if( rename(tPath, cPath) ){
			sqlite3_snprintf(sizeof(errmsg), errmsg, "rename failed (%d)", errno);
			goto end_breaklock;
		}
		rc = 0;
		fprintf(stderr, "broke stale lock on %s\n", cPath);
		robust_close(pFile, conchFile->h, __LINE__);
		conchFile->h = fd;
		conchFile->openFlags = O_RDWR | O_CREAT;

end_breaklock:
		if( rc ){
			if( fd>=0 ){
				osUnlink(tPath);
				robust_close(pFile, fd, __LINE__);
			}
			fprintf(stderr, "failed to break stale lock on %s, %s\n", cPath, errmsg);
		}
		return rc;
	}

	static int proxyConchLock(unixFile *pFile, uuid_t myHostID, int lockType){
		proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext; 
		unixFile *conchFile = pCtx->conchFile;
		int rc = SQLITE_OK;
		int nTries = 0;
		struct timespec conchModTime;

		memset(&conchModTime, 0, sizeof(conchModTime));
		do {
			rc = conchFile->pMethod->xLock((sqlite3_file*)conchFile, lockType);
			nTries ++;
			if( rc==SQLITE_BUSY ){
				/* If the lock failed (busy):
				* 1st try: get the mod time of the conch, wait 0.5s and try again. 
				* 2nd try: fail if the mod time changed or host id is different, wait 
				*           10 sec and try again
				* 3rd try: break the lock unless the mod time has changed.
				*/
				struct stat buf;
				if( osFstat(conchFile->h, &buf) ){
					pFile->lastErrno = errno;
					return SQLITE_IOERR_LOCK;
				}

				if( nTries==1 ){
					conchModTime = buf.st_mtimespec;
					usleep(500000); /* wait 0.5 sec and try the lock again*/
					continue;  
				}

				assert( nTries>1 );
				if( conchModTime.tv_sec != buf.st_mtimespec.tv_sec || 
					conchModTime.tv_nsec != buf.st_mtimespec.tv_nsec ){
						return SQLITE_BUSY;
				}

				if( nTries==2 ){  
					char tBuf[PROXY_MAXCONCHLEN];
					int len = osPread(conchFile->h, tBuf, PROXY_MAXCONCHLEN, 0);
					if( len<0 ){
						pFile->lastErrno = errno;
						return SQLITE_IOERR_LOCK;
					}
					if( len>PROXY_PATHINDEX && tBuf[0]==(char)PROXY_CONCHVERSION){
						/* don't break the lock if the host id doesn't match */
						if( 0!=memcmp(&tBuf[PROXY_HEADERLEN], myHostID, PROXY_HOSTIDLEN) ){
							return SQLITE_BUSY;
						}
					}else{
						/* don't break the lock on short read or a version mismatch */
						return SQLITE_BUSY;
					}
					usleep(10000000); /* wait 10 sec and try the lock again */
					continue; 
				}

				assert( nTries==3 );
				if( 0==proxyBreakConchLock(pFile, myHostID) ){
					rc = SQLITE_OK;
					if( lockType==EXCLUSIVE_LOCK ){
						rc = conchFile->pMethod->xLock((sqlite3_file*)conchFile, SHARED_LOCK);          
					}
					if( !rc ){
						rc = conchFile->pMethod->xLock((sqlite3_file*)conchFile, lockType);
					}
				}
			}
		} while( rc==SQLITE_BUSY && nTries<3 );

		return rc;
	}

	static int proxyTakeConch(unixFile *pFile){
		proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext; 

		if( pCtx->conchHeld!=0 ){
			return SQLITE_OK;
		}else{
			unixFile *conchFile = pCtx->conchFile;
			uuid_t myHostID;
			int pError = 0;
			char readBuf[PROXY_MAXCONCHLEN];
			char lockPath[MAXPATHLEN];
			char *tempLockPath = NULL;
			int rc = SQLITE_OK;
			int createConch = 0;
			int hostIdMatch = 0;
			int readLen = 0;
			int tryOldLockPath = 0;
			int forceNewLockPath = 0;

			OSTRACE(("TAKECONCH  %d for %s pid=%d\n", conchFile->h,
				(pCtx->lockProxyPath ? pCtx->lockProxyPath : ":auto:"), getpid()));

			rc = proxyGetHostID(myHostID, &pError);
			if( (rc&0xff)==SQLITE_IOERR ){
				pFile->lastErrno = pError;
				goto end_takeconch;
			}
			rc = proxyConchLock(pFile, myHostID, SHARED_LOCK);
			if( rc!=SQLITE_OK ){
				goto end_takeconch;
			}
			/* read the existing conch file */
			readLen = seekAndRead((unixFile*)conchFile, 0, readBuf, PROXY_MAXCONCHLEN);
			if( readLen<0 ){
				/* I/O error: lastErrno set by seekAndRead */
				pFile->lastErrno = conchFile->lastErrno;
				rc = SQLITE_IOERR_READ;
				goto end_takeconch;
			}else if( readLen<=(PROXY_HEADERLEN+PROXY_HOSTIDLEN) || 
				readBuf[0]!=(char)PROXY_CONCHVERSION ){
					/* a short read or version format mismatch means we need to create a new 
					** conch file. 
					*/
					createConch = 1;
			}
			/* if the host id matches and the lock path already exists in the conch
			** we'll try to use the path there, if we can't open that path, we'll 
			** retry with a new auto-generated path 
			*/
			do { /* in case we need to try again for an :auto: named lock file */

				if( !createConch && !forceNewLockPath ){
					hostIdMatch = !memcmp(&readBuf[PROXY_HEADERLEN], myHostID, 
						PROXY_HOSTIDLEN);
					/* if the conch has data compare the contents */
					if( !pCtx->lockProxyPath ){
						/* for auto-named local lock file, just check the host ID and we'll
						** use the local lock file path that's already in there
						*/
						if( hostIdMatch ){
							size_t pathLen = (readLen - PROXY_PATHINDEX);

							if( pathLen>=MAXPATHLEN ){
								pathLen=MAXPATHLEN-1;
							}
							memcpy(lockPath, &readBuf[PROXY_PATHINDEX], pathLen);
							lockPath[pathLen] = 0;
							tempLockPath = lockPath;
							tryOldLockPath = 1;
							/* create a copy of the lock path if the conch is taken */
							goto end_takeconch;
						}
					}else if( hostIdMatch
						&& !strncmp(pCtx->lockProxyPath, &readBuf[PROXY_PATHINDEX],
						readLen-PROXY_PATHINDEX)
						){
							/* conch host and lock path match */
							goto end_takeconch; 
					}
				}

				/* if the conch isn't writable and doesn't match, we can't take it */
				if( (conchFile->openFlags&O_RDWR) == 0 ){
					rc = SQLITE_BUSY;
					goto end_takeconch;
				}

				/* either the conch didn't match or we need to create a new one */
				if( !pCtx->lockProxyPath ){
					proxyGetLockPath(pCtx->dbPath, lockPath, MAXPATHLEN);
					tempLockPath = lockPath;
					/* create a copy of the lock path _only_ if the conch is taken */
				}

				/* update conch with host and path (this will fail if other process
				** has a shared lock already), if the host id matches, use the big
				** stick.
				*/
				futimes(conchFile->h, NULL);
				if( hostIdMatch && !createConch ){
					if( conchFile->pInode && conchFile->pInode->nShared>1 ){
						/* We are trying for an exclusive lock but another thread in this
						** same process is still holding a shared lock. */
						rc = SQLITE_BUSY;
					} else {          
						rc = proxyConchLock(pFile, myHostID, EXCLUSIVE_LOCK);
					}
				}else{
					rc = conchFile->pMethod->xLock((sqlite3_file*)conchFile, EXCLUSIVE_LOCK);
				}
				if( rc==SQLITE_OK ){
					char writeBuffer[PROXY_MAXCONCHLEN];
					int writeSize = 0;

					writeBuffer[0] = (char)PROXY_CONCHVERSION;
					memcpy(&writeBuffer[PROXY_HEADERLEN], myHostID, PROXY_HOSTIDLEN);
					if( pCtx->lockProxyPath!=NULL ){
						strlcpy(&writeBuffer[PROXY_PATHINDEX], pCtx->lockProxyPath, MAXPATHLEN);
					}else{
						strlcpy(&writeBuffer[PROXY_PATHINDEX], tempLockPath, MAXPATHLEN);
					}
					writeSize = PROXY_PATHINDEX + strlen(&writeBuffer[PROXY_PATHINDEX]);
					robust_ftruncate(conchFile->h, writeSize);
					rc = unixWrite((sqlite3_file *)conchFile, writeBuffer, writeSize, 0);
					fsync(conchFile->h);
					/* If we created a new conch file (not just updated the contents of a 
					** valid conch file), try to match the permissions of the database 
					*/
					if( rc==SQLITE_OK && createConch ){
						struct stat buf;
						int err = osFstat(pFile->h, &buf);
						if( err==0 ){
							mode_t cmode = buf.st_mode&(S_IRUSR|S_IWUSR | S_IRGRP|S_IWGRP |
								S_IROTH|S_IWOTH);
							/* try to match the database file R/W permissions, ignore failure */
#ifndef SQLITE_PROXY_DEBUG
							osFchmod(conchFile->h, cmode);
#else
							do{
								rc = osFchmod(conchFile->h, cmode);
							}while( rc==(-1) && errno==EINTR );
							if( rc!=0 ){
								int code = errno;
								fprintf(stderr, "fchmod %o FAILED with %d %s\n",
									cmode, code, strerror(code));
							} else {
								fprintf(stderr, "fchmod %o SUCCEDED\n",cmode);
							}
						}else{
							int code = errno;
							fprintf(stderr, "STAT FAILED[%d] with %d %s\n", 
								err, code, strerror(code));
#endif
						}
					}
				}
				conchFile->pMethod->xUnlock((sqlite3_file*)conchFile, SHARED_LOCK);

end_takeconch:
				OSTRACE(("TRANSPROXY: CLOSE  %d\n", pFile->h));
				if( rc==SQLITE_OK && pFile->openFlags ){
					int fd;
					if( pFile->h>=0 ){
						robust_close(pFile, pFile->h, __LINE__);
					}
					pFile->h = -1;
					fd = robust_open(pCtx->dbPath, pFile->openFlags, 0);
					OSTRACE(("TRANSPROXY: OPEN  %d\n", fd));
					if( fd>=0 ){
						pFile->h = fd;
					}else{
						rc=SQLITE_CANTOPEN_BKPT; /* SQLITE_BUSY? proxyTakeConch called
												 during locking */
					}
				}
				if( rc==SQLITE_OK && !pCtx->lockProxy ){
					char *path = tempLockPath ? tempLockPath : pCtx->lockProxyPath;
					rc = proxyCreateUnixFile(path, &pCtx->lockProxy, 1);
					if( rc!=SQLITE_OK && rc!=SQLITE_NOMEM && tryOldLockPath ){
						/* we couldn't create the proxy lock file with the old lock file path
						** so try again via auto-naming 
						*/
						forceNewLockPath = 1;
						tryOldLockPath = 0;
						continue; /* go back to the do {} while start point, try again */
					}
				}
				if( rc==SQLITE_OK ){
					/* Need to make a copy of path if we extracted the value
					** from the conch file or the path was allocated on the stack
					*/
					if( tempLockPath ){
						pCtx->lockProxyPath = sqlite3DbStrDup(0, tempLockPath);
						if( !pCtx->lockProxyPath ){
							rc = SQLITE_NOMEM;
						}
					}
				}
				if( rc==SQLITE_OK ){
					pCtx->conchHeld = 1;

					if( pCtx->lockProxy->pMethod == &afpIoMethods ){
						afpLockingContext *afpCtx;
						afpCtx = (afpLockingContext *)pCtx->lockProxy->lockingContext;
						afpCtx->dbPath = pCtx->lockProxyPath;
					}
				} else {
					conchFile->pMethod->xUnlock((sqlite3_file*)conchFile, NO_LOCK);
				}
				OSTRACE(("TAKECONCH  %d %s\n", conchFile->h,
					rc==SQLITE_OK?"ok":"failed"));
				return rc;
			} while (1); /* in case we need to retry the :auto: lock file - 
						 ** we should never get here except via the 'continue' call. */
		}
	}

	static int proxyReleaseConch(unixFile *pFile){
		int rc = SQLITE_OK;         /* Subroutine return code */
		proxyLockingContext *pCtx;  /* The locking context for the proxy lock */
		unixFile *conchFile;        /* Name of the conch file */

		pCtx = (proxyLockingContext *)pFile->lockingContext;
		conchFile = pCtx->conchFile;
		OSTRACE(("RELEASECONCH  %d for %s pid=%d\n", conchFile->h,
			(pCtx->lockProxyPath ? pCtx->lockProxyPath : ":auto:"), 
			getpid()));
		if( pCtx->conchHeld>0 ){
			rc = conchFile->pMethod->xUnlock((sqlite3_file*)conchFile, NO_LOCK);
		}
		pCtx->conchHeld = 0;
		OSTRACE(("RELEASECONCH  %d %s\n", conchFile->h,
			(rc==SQLITE_OK ? "ok" : "failed")));
		return rc;
	}

	static int proxyCreateConchPathname(char *dbPath, char **pConchPath){
		int i;                        /* Loop counter */
		int len = (int)strlen(dbPath); /* Length of database filename - dbPath */
		char *conchPath;              /* buffer in which to construct conch name */

		/* Allocate space for the conch filename and initialize the name to
		** the name of the original database file. */  
		*pConchPath = conchPath = (char *)sqlite3_malloc(len + 8);
		if( conchPath==0 ){
			return SQLITE_NOMEM;
		}
		memcpy(conchPath, dbPath, len+1);

		/* now insert a "." before the last / character */
		for( i=(len-1); i>=0; i-- ){
			if( conchPath[i]=='/' ){
				i++;
				break;
			}
		}
		conchPath[i]='.';
		while ( i<len ){
			conchPath[i+1]=dbPath[i];
			i++;
		}

		/* append the "-conch" suffix to the file */
		memcpy(&conchPath[i+1], "-conch", 7);
		assert( (int)strlen(conchPath) == len+7 );

		return SQLITE_OK;
	}

	static int switchLockProxyPath(unixFile *pFile, const char *path) {
		proxyLockingContext *pCtx = (proxyLockingContext*)pFile->lockingContext;
		char *oldPath = pCtx->lockProxyPath;
		int rc = SQLITE_OK;

		if( pFile->eFileLock!=NO_LOCK ){
			return SQLITE_BUSY;
		}  

		/* nothing to do if the path is NULL, :auto: or matches the existing path */
		if( !path || path[0]=='\0' || !strcmp(path, ":auto:") ||
			(oldPath && !strncmp(oldPath, path, MAXPATHLEN)) ){
				return SQLITE_OK;
		}else{
			unixFile *lockProxy = pCtx->lockProxy;
			pCtx->lockProxy=NULL;
			pCtx->conchHeld = 0;
			if( lockProxy!=NULL ){
				rc=lockProxy->pMethod->xClose((sqlite3_file *)lockProxy);
				if( rc ) return rc;
				sqlite3_free(lockProxy);
			}
			sqlite3_free(oldPath);
			pCtx->lockProxyPath = sqlite3DbStrDup(0, path);
		}

		return rc;
	}

	static int proxyGetDbPathForUnixFile(unixFile *pFile, char *dbPath){
#if defined(__APPLE__)
		if( pFile->pMethod == &afpIoMethods ){
			/* afp style keeps a reference to the db path in the filePath field 
			** of the struct */
			assert( (int)strlen((char*)pFile->lockingContext)<=MAXPATHLEN );
			strlcpy(dbPath, ((afpLockingContext *)pFile->lockingContext)->dbPath, MAXPATHLEN);
		} else
#endif
			if( pFile->pMethod == &dotlockIoMethods ){
				/* dot lock style uses the locking context to store the dot lock
				** file path */
				int len = strlen((char *)pFile->lockingContext) - strlen(DOTLOCK_SUFFIX);
				memcpy(dbPath, (char *)pFile->lockingContext, len + 1);
			}else{
				/* all other styles use the locking context to store the db file path */
				assert( strlen((char*)pFile->lockingContext)<=MAXPATHLEN );
				strlcpy(dbPath, (char *)pFile->lockingContext, MAXPATHLEN);
			}
			return SQLITE_OK;
	}

	static int proxyTransformUnixFile(unixFile *pFile, const char *path) {
		proxyLockingContext *pCtx;
		char dbPath[MAXPATHLEN+1];       /* Name of the database file */
		char *lockPath=NULL;
		int rc = SQLITE_OK;

		if( pFile->eFileLock!=NO_LOCK ){
			return SQLITE_BUSY;
		}
		proxyGetDbPathForUnixFile(pFile, dbPath);
		if( !path || path[0]=='\0' || !strcmp(path, ":auto:") ){
			lockPath=NULL;
		}else{
			lockPath=(char *)path;
		}

		OSTRACE(("TRANSPROXY  %d for %s pid=%d\n", pFile->h,
			(lockPath ? lockPath : ":auto:"), getpid()));

		pCtx = sqlite3_malloc( sizeof(*pCtx) );
		if( pCtx==0 ){
			return SQLITE_NOMEM;
		}
		memset(pCtx, 0, sizeof(*pCtx));

		rc = proxyCreateConchPathname(dbPath, &pCtx->conchFilePath);
		if( rc==SQLITE_OK ){
			rc = proxyCreateUnixFile(pCtx->conchFilePath, &pCtx->conchFile, 0);
			if( rc==SQLITE_CANTOPEN && ((pFile->openFlags&O_RDWR) == 0) ){
				/* if (a) the open flags are not O_RDWR, (b) the conch isn't there, and
				** (c) the file system is read-only, then enable no-locking access.
				** Ugh, since O_RDONLY==0x0000 we test for !O_RDWR since unixOpen asserts
				** that openFlags will have only one of O_RDONLY or O_RDWR.
				*/
				struct statfs fsInfo;
				struct stat conchInfo;
				int goLockless = 0;

				if( osStat(pCtx->conchFilePath, &conchInfo) == -1 ) {
					int err = errno;
					if( (err==ENOENT) && (statfs(dbPath, &fsInfo) != -1) ){
						goLockless = (fsInfo.f_flags&MNT_RDONLY) == MNT_RDONLY;
					}
				}
				if( goLockless ){
					pCtx->conchHeld = -1; /* read only FS/ lockless */
					rc = SQLITE_OK;
				}
			}
		}  
		if( rc==SQLITE_OK && lockPath ){
			pCtx->lockProxyPath = sqlite3DbStrDup(0, lockPath);
		}

		if( rc==SQLITE_OK ){
			pCtx->dbPath = sqlite3DbStrDup(0, dbPath);
			if( pCtx->dbPath==NULL ){
				rc = SQLITE_NOMEM;
			}
		}
		if( rc==SQLITE_OK ){
			/* all memory is allocated, proxys are created and assigned, 
			** switch the locking context and pMethod then return.
			*/
			pCtx->oldLockingContext = pFile->lockingContext;
			pFile->lockingContext = pCtx;
			pCtx->pOldMethod = pFile->pMethod;
			pFile->pMethod = &proxyIoMethods;
		}else{
			if( pCtx->conchFile ){ 
				pCtx->conchFile->pMethod->xClose((sqlite3_file *)pCtx->conchFile);
				sqlite3_free(pCtx->conchFile);
			}
			sqlite3DbFree(0, pCtx->lockProxyPath);
			sqlite3_free(pCtx->conchFilePath); 
			sqlite3_free(pCtx);
		}
		OSTRACE(("TRANSPROXY  %d %s\n", pFile->h,
			(rc==SQLITE_OK ? "ok" : "failed")));
		return rc;
	}

	static int proxyFileControl(sqlite3_file *id, int op, void *pArg){
		switch( op ){
		case SQLITE_GET_LOCKPROXYFILE: {
			unixFile *pFile = (unixFile*)id;
			if( pFile->pMethod == &proxyIoMethods ){
				proxyLockingContext *pCtx = (proxyLockingContext*)pFile->lockingContext;
				proxyTakeConch(pFile);
				if( pCtx->lockProxyPath ){
					*(const char **)pArg = pCtx->lockProxyPath;
				}else{
					*(const char **)pArg = ":auto: (not held)";
				}
			} else {
				*(const char **)pArg = NULL;
			}
			return SQLITE_OK;
									   }
		case SQLITE_SET_LOCKPROXYFILE: {
			unixFile *pFile = (unixFile*)id;
			int rc = SQLITE_OK;
			int isProxyStyle = (pFile->pMethod == &proxyIoMethods);
			if( pArg==NULL || (const char *)pArg==0 ){
				if( isProxyStyle ){
					/* turn off proxy locking - not supported */
					rc = SQLITE_ERROR /*SQLITE_PROTOCOL? SQLITE_MISUSE?*/;
				}else{
					/* turn off proxy locking - already off - NOOP */
					rc = SQLITE_OK;
				}
			}else{
				const char *proxyPath = (const char *)pArg;
				if( isProxyStyle ){
					proxyLockingContext *pCtx = 
						(proxyLockingContext*)pFile->lockingContext;
					if( !strcmp(pArg, ":auto:") 
						|| (pCtx->lockProxyPath &&
						!strncmp(pCtx->lockProxyPath, proxyPath, MAXPATHLEN))
						){
							rc = SQLITE_OK;
					}else{
						rc = switchLockProxyPath(pFile, proxyPath);
					}
				}else{
					/* turn on proxy file locking */
					rc = proxyTransformUnixFile(pFile, proxyPath);
				}
			}
			return rc;
									   }
		default: {
			assert( 0 );  /* The call assures that only valid opcodes are sent */
				 }
		}
		/*NOTREACHED*/
		return SQLITE_ERROR;
	}

	static int proxyCheckReservedLock(sqlite3_file *id, int *pResOut) {
		unixFile *pFile = (unixFile*)id;
		int rc = proxyTakeConch(pFile);
		if( rc==SQLITE_OK ){
			proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext;
			if( pCtx->conchHeld>0 ){
				unixFile *proxy = pCtx->lockProxy;
				return proxy->pMethod->xCheckReservedLock((sqlite3_file*)proxy, pResOut);
			}else{ /* conchHeld < 0 is lockless */
				pResOut=0;
			}
		}
		return rc;
	}

	static int proxyLock(sqlite3_file *id, int eFileLock) {
		unixFile *pFile = (unixFile*)id;
		int rc = proxyTakeConch(pFile);
		if( rc==SQLITE_OK ){
			proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext;
			if( pCtx->conchHeld>0 ){
				unixFile *proxy = pCtx->lockProxy;
				rc = proxy->pMethod->xLock((sqlite3_file*)proxy, eFileLock);
				pFile->eFileLock = proxy->eFileLock;
			}else{
				/* conchHeld < 0 is lockless */
			}
		}
		return rc;
	}


	static int proxyUnlock(sqlite3_file *id, int eFileLock) {
		unixFile *pFile = (unixFile*)id;
		int rc = proxyTakeConch(pFile);
		if( rc==SQLITE_OK ){
			proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext;
			if( pCtx->conchHeld>0 ){
				unixFile *proxy = pCtx->lockProxy;
				rc = proxy->pMethod->xUnlock((sqlite3_file*)proxy, eFileLock);
				pFile->eFileLock = proxy->eFileLock;
			}else{
				/* conchHeld < 0 is lockless */
			}
		}
		return rc;
	}

	static int proxyClose(sqlite3_file *id) {
		if( id ){
			unixFile *pFile = (unixFile*)id;
			proxyLockingContext *pCtx = (proxyLockingContext *)pFile->lockingContext;
			unixFile *lockProxy = pCtx->lockProxy;
			unixFile *conchFile = pCtx->conchFile;
			int rc = SQLITE_OK;

			if( lockProxy ){
				rc = lockProxy->pMethod->xUnlock((sqlite3_file*)lockProxy, NO_LOCK);
				if( rc ) return rc;
				rc = lockProxy->pMethod->xClose((sqlite3_file*)lockProxy);
				if( rc ) return rc;
				sqlite3_free(lockProxy);
				pCtx->lockProxy = 0;
			}
			if( conchFile ){
				if( pCtx->conchHeld ){
					rc = proxyReleaseConch(pFile);
					if( rc ) return rc;
				}
				rc = conchFile->pMethod->xClose((sqlite3_file*)conchFile);
				if( rc ) return rc;
				sqlite3_free(conchFile);
			}
			sqlite3DbFree(0, pCtx->lockProxyPath);
			sqlite3_free(pCtx->conchFilePath);
			sqlite3DbFree(0, pCtx->dbPath);
			/* restore the original locking context and pMethod then close it */
			pFile->lockingContext = pCtx->oldLockingContext;
			pFile->pMethod = pCtx->pOldMethod;
			sqlite3_free(pCtx);
			return pFile->pMethod->xClose(id);
		}
		return SQLITE_OK;
	}

#endif
#pragma endregion

	//#define UNIXVFS(VFSNAME, FINDER) {                        \
	//	3,                    /* iVersion */                    \
	//	sizeof(unixFile),     /* szOsFile */                    \
	//	MAX_PATHNAME,         /* mxPathname */                  \
	//	0,                    /* pNext */                       \
	//	VFSNAME,              /* zName */                       \
	//	(void*)&FINDER,       /* pAppData */                    \
	//	unixOpen,             /* xOpen */                       \
	//	unixDelete,           /* xDelete */                     \
	//	unixAccess,           /* xAccess */                     \
	//	unixFullPathname,     /* xFullPathname */               \
	//	unixDlOpen,           /* xDlOpen */                     \
	//	unixDlError,          /* xDlError */                    \
	//	unixDlSym,            /* xDlSym */                      \
	//	unixDlClose,          /* xDlClose */                    \
	//	unixRandomness,       /* xRandomness */                 \
	//	unixSleep,            /* xSleep */                      \
	//	unixCurrentTime,      /* xCurrentTime */                \
	//	unixCurrentTimeInt64, /* xCurrentTimeInt64 */           \
	//	unixSetSystemCall,    /* xSetSystemCall */              \
	//	unixGetSystemCall,    /* xGetSystemCall */              \
	//	unixNextSystemCall,   /* xNextSystemCall */             \
	//}

	static sqlite3_vfs aVfs[] = {
#if SQLITE_ENABLE_LOCKING_STYLE && (OS_VXWORKS || defined(__APPLE__))
		UNIXVFS("unix",          autolockIoFinder ),
#else
		UNIXVFS("unix",          posixIoFinder ),
#endif
		UNIXVFS("unix-none",     nolockIoFinder ),
		UNIXVFS("unix-dotfile",  dotlockIoFinder ),
		UNIXVFS("unix-excl",     posixIoFinder ),
#if OS_VXWORKS
		UNIXVFS("unix-namedsem", semIoFinder ),
#endif
#if SQLITE_ENABLE_LOCKING_STYLE
		UNIXVFS("unix-posix",    posixIoFinder ),
#if !OS_VXWORKS
		UNIXVFS("unix-flock",    flockIoFinder ),
#endif
#endif
#if SQLITE_ENABLE_LOCKING_STYLE && defined(__APPLE__)
		UNIXVFS("unix-afp",      afpIoFinder ),
		UNIXVFS("unix-nfs",      nfsIoFinder ),
		UNIXVFS("unix-proxy",    proxyIoFinder ),
#endif
	};

	RC VSystem::Initialize()
	{
		// Double-check that the Syscalls[] array has been constructed correctly.  See ticket [bb3a86e890c8e96ab]
		_assert(_lengthof(Syscalls) == 21);
		// Register all VFSes defined in the aVfs[] array
		for (int i = 0; i<(sizeof(aVfs)/sizeof(sqlite3_vfs)); i++)
			sqlite3_vfs_register(&aVfs[i], i==0);
		return RC_OK; 
	}

	void VSystem::Shutdown()
	{ 
	}

#pragma endregion
}
#endif