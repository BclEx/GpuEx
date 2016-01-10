// This file implements an example of a simple VFS implementation that omits complex features often not required or not possible on embedded
// platforms.  Code is included to buffer writes to the journal file, which can be a significant performance improvement on some embedded platforms.
//
// OVERVIEW
//   The code in this file implements a minimal SQLite VFS that can be used on Linux and other posix-like operating systems. The following 
//   system calls are used:
//
//    File-system: access(), unlink(), getcwd()
//    File IO:     open(), read(), write(), fsync(), close(), fstat()
//    Other:       sleep(), usleep(), time()
//
//   The following VFS features are omitted:
//
//     1. File locking. The user must ensure that there is at most one connection to each database when using this VFS. Multiple
//        connections to a single shared-cache count as a single connection for the purposes of the previous statement.
//
//     2. The loading of dynamic extensions (shared libraries).
//
//     3. Temporary files. The user must configure SQLite to use in-memory temp files when using this VFS. The easiest way to do this is to
//        compile with:
//
//          -DSQLITE_TEMP_STORE=3
//
//     4. File truncation. As of version 3.6.24, SQLite may run without a working xTruncate() call, providing the user does not configure
//        SQLite to use "journal_mode=truncate", or use both "journal_mode=persist" and ATTACHed databases.
//
//   It is assumed that the system uses UNIX-like path-names. Specifically, that '/' characters are used to separate path components and that
//   a path-name is a relative path unless it begins with a '/'. And that no UTF-8 encoded paths are greater than 512 bytes in length.
//
// JOURNAL WRITE-BUFFERING
//   To commit a transaction to the database, SQLite first writes rollback information into the journal file. This usually consists of 4 steps:
//
//     1. The rollback information is sequentially written into the journal file, starting at the start of the file.
//     2. The journal file is synced to disk.
//     3. A modification is made to the first few bytes of the journal file.
//     4. The journal file is synced to disk again.
//
//   Most of the data is written in step 1 using a series of calls to the VFS xWrite() method. The buffers passed to the xWrite() calls are of
//   various sizes. For example, as of version 3.6.24, when committing a transaction that modifies 3 pages of a database file that uses 4096 
//   byte pages residing on a media with 512 byte sectors, SQLite makes eleven calls to the xWrite() method to create the rollback journal, 
//   as follows:
//
//             Write offset | Bytes written
//             ----------------------------
//                        0            512
//                      512              4
//                      516           4096
//                     4612              4
//                     4616              4
//                     4620           4096
//                     8716              4
//                     8720              4
//                     8724           4096
//                    12820              4
//             ++++++++++++SYNC+++++++++++
//                        0             12
//             ++++++++++++SYNC+++++++++++
//
//   On many operating systems, this is an efficient way to write to a file. However, on some embedded systems that do not cache writes in OS 
//   buffers it is much more efficient to write data in blocks that are an integer multiple of the sector-size in size and aligned at the
//   start of a sector.
//
//   To work around this, the code in this file allocates a fixed size buffer of SQLITE_DEMOVFS_BUFFERSZ using sqlite3_malloc() whenever a 
//   journal file is opened. It uses the buffer to coalesce sequential writes into aligned SQLITE_DEMOVFS_BUFFERSZ blocks. When SQLite
//   invokes the xSync() method to sync the contents of the file to disk, all accumulated data is written out, even if it does not constitute
//   a complete block. This means the actual IO to create the rollback journal for the example transaction above is this:
//
//             Write offset | Bytes written
//             ----------------------------
//                        0           8192
//                     8192           4632
//             ++++++++++++SYNC+++++++++++
//                        0             12
//             ++++++++++++SYNC+++++++++++
//
//   Much more efficient if the underlying OS is not caching write operations.
#if !defined(_TEST) || OS_UNIX

#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <new.h>
//#include <assert.h>
//#include <string.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <sys/file.h>
//#include <sys/param.h>
//#include <unistd.h>
//#include <time.h>
//#include <errno.h>
//#include <fcntl.h>

// Size of the write buffer used by journal files in bytes.
#ifndef DEMOVFS_BUFFERSZ
#define DEMOVFS_BUFFERSZ 8192
#endif
// The maximum pathname length supported by this VFS.
#define MAXPATHNAME 512

// When using this VFS, the sqlite3_file* handles that SQLite uses are actually pointers to instances of type DemoFile.
class DemoVFile : public VFile
{
public:
	int Fd;                     // File descriptor
	char *Buffer;               // Pointer to malloc'd buffer
	int BufferLength;           // Valid bytes of data in zBuffer
	int64 BufferOffset;			// Offset in file of zBuffer[0]
public:
	// Write directly to the file passed as the first argument. Even if the file has a write-buffer (DemoFile.aBuffer), ignore it.
	__device__ int DemoDirectWrite(const void *buffer, int amount, int64 offset)
	{
		off_t ofst = _lseek(Fd, offset, SEEK_SET);
		if (ofst != offset)
			return RC_IOERR_WRITE;
		size_t write = write(Fd, buffer, amount);
		if (write != amount)
			return RC_IOERR_WRITE;
		return RC_OK;
	}

	// Flush the contents of the DemoFile.aBuffer buffer to disk. This is a no-op if this particular file does not have a buffer (i.e. it is not
	// a journal file) or if the buffer is currently empty.
	__device__ static int DemoFlushBuffer()
	{
		RC rc = RC_OK;
		if (BufferLength)
		{
			rc = DemoDirectWrite(Buffer, BufferLength, BufferOffset);
			BufferLength = 0;
		}
		return rc;
	}

	__device__ virtual RC Close_()
	{ 
		RC rc = DemoFlushBuffer();
		_free(p->Buffer);
		_close(p->Fd);
		return rc;
	}
	__device__ virtual RC Read(void *buffer, int amount, int64 offset)
	{
		// Flush any data in the write buffer to disk in case this operation is trying to read data the file-region currently cached in the buffer.
		// It would be possible to detect this case and possibly save an unnecessary write here, but in practice SQLite will rarely read from
		// a journal file when there is data cached in the write-buffer.
		RC rc = DemoFlushBuffer();
		if (rc != RC_OK)
			return rc;
		off_t ofst = _lseek(Fd, offset, SEEK_SET);
		if (ofst != offset)
			return RC_IOERR_READ;
		int read = _read(Fd, buffer, amount);
		if (read == amount)
			return RC_OK;
		else if (read >= 0)
			return RC_IOERR_SHORT_READ;
		return RC_IOERR_READ;
	}
	__device__ virtual RC Write(const void *buffer, int amount, int64 offset)
	{
		if (Buffer)
		{
			char *z = (char *)buffer; // Pointer to remaining data to write
			int n = amount; // Number of bytes at z
			int64 i = offset; // File offset to write to
			while (n > 0)
			{
				// If the buffer is full, or if this data is not being written directly following the data already buffered, flush the buffer. Flushing
				// the buffer is a no-op if it is empty.  
				if (BufferLength == DEMOVFS_BUFFERSZ || BufferOffset+BufferLength != i)
				{
					RC rc = DemoFlushBuffer();
					if (rc != RC_OK)
						return rc;
				}
				_assert(BufferLength == 0 || BufferOffset+BufferLength == i);
				BufferOffset = i - BufferLength;
				// Copy as much data as possible into the buffer.
				int copy = DEMOVFS_BUFFERSZ - BufferLength; // Number of bytes to copy into buffer
				if (copy > n)
					copy = n;
				memcpy(&Buffer[BufferLength], z, copy);
				BufferLength += copy;
				n -= copy;
				i += copy;
				z += copy;
			}
		}
		else
			return DemoDirectWrite(buffer, amount, offset);
		return RC_OK;
	}
	__device__ virtual RC Truncate(int64 size)
	{
#if 0
		if (ftruncate(Fd, size)) return RC_IOERR_TRUNCATE;
#endif
		return RC_OK;
	}
	__device__ virtual RC Sync(SYNC flags)
	{
		RC rc = DemoFlushBuffer();
		if (rc != RC_OK)
			return rc;
		rc = fsync(Fd);
		return (rc == 0 ? RC_OK : RC_IOERR_FSYNC);
	}
	__device__ virtual RC get_FileSize(int64 &size)
	{
		// Flush the contents of the buffer to disk. As with the flush in the demoRead() method, it would be possible to avoid this and save a write
		// here and there. But in practice this comes up so infrequently it is not worth the trouble.
		RC rc = DemoFlushBuffer();
		if (rc != RC_OK)
			return rc;
		struct stat sStat; // Output of fstat() call
		rc = _fstat(Fd, &sStat);
		if (rc != 0) return RC_IOERR_FSTAT;
		size = sStat.st_size;
		return RC_OK;
	}

	__device__ virtual RC Lock(LOCK lock) { return RC_OK; }
	__device__ virtual RC Unlock(LOCK lock) { return RC_OK; }
	__device__ virtual RC CheckReservedLock(int &lock) { lock = 0; return RC_OK; }
	__device__ virtual RC FileControl(FCNTL op, void *arg) { return RC_OK; }

	__device__ virtual uint get_SectorSize() { return 0; }
	__device__ virtual IOCAP get_DeviceCharacteristics() { return (IOCAP)0; }

	__device__ virtual RC ShmLock(int offset, int n, SHM flags) { return Real->ShmLock(offset, n, flags); }
	__device__ virtual void ShmBarrier() { Real->ShmBarrier(); }
	__device__ virtual RC ShmUnmap(bool deleteFlag) { return Real->ShmUnmap(deleteFlag); }
	__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp) { return Real->ShmMap(region, sizeRegion, isWrite, pp); }

};

#ifndef F_OK
#define F_OK 0
#endif
#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif

class DemoVSystem : public VSystem
{
public:
	__device__ virtual VFile *_AttachFile(void *buffer) { return new (buffer) DemoVFile(); }
	__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
	{
		if (!path)
			return RC_IOERR;
		char *buf = nullptr;
		if (flags & VSystem::OPEN_MAIN_JOURNAL)
		{
			buf = (char *)_alloc(DEMOVFS_BUFFERSZ);
			if (!buf)
				return RC_NOMEM;
		}
		int oflags = 0; // flags to pass to open() call
		if (flags & VSystem::OPEN_EXCLUSIVE) oflags |= O_EXCL;
		if (flags & VSystem::OPEN_CREATE)    oflags |= O_CREAT;
		if (flags & VSystem::OPEN_READONLY)  oflags |= O_RDONLY;
		if (flags & VSystem::OPEN_READWRITE) oflags |= O_RDWR;
		DemoVFile *p = (DemoVFile *)file;
		memset(p, 0, sizeof(DemoVFile));
		p->Fd = open(path, oflags, 0600);
		if (p->Fd < 0)
		{
			_free(buf);
			return RC_CANTOPEN;
		}
		p->Buffer = buf;
		if (outFlags) *outFlags = flags;
		p->Opened = true;
		return RC_OK;
	}
	__device__ virtual RC Delete(const char *path, bool syncDirectory)
	{
		RC rc = unlink(zPath);
		if (rc != 0 && errno == ENOENT) return RC_OK;
		if (rc == 0 && syncDirectory)
		{
			// Figure out the directory name from the path of the file deleted.
			char dir[MAXPATHNAME+1]; // Name of directory containing file zPath
			_snprintf(MAXPATHNAME, dir, "%s", path);
			zDir[MAXPATHNAME] = '\0';
			int i;
			for (i = _strlen(dir); i > 1 && dir[i] != '/'; i++) { }
			dir[i] = '\0';
			// Open a file-descriptor on the directory. Sync. Close.
			int dfd = open(zDir, O_RDONLY, 0); // File descriptor open on directory
			if (dfd < 0)
				rc = -1;
			else
			{
				rc = _fsync(dfd);
				_close(dfd);
			}
		}
		return (rc == 0 ? RC_OK : RC_IOERR_DELETE);
	}
	__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC)
	{
		_assert(flags == ACCESS_EXISTS || flags == ACCESS_READ || flags == ACCESS_READWRITE);
		int eAccess = F_OK;
		if (flags == ACCESS_READWRITE) eAccess = R_OK|W_OK;
		if (flags == ACCESS_READ) eAccess = R_OK;
		RC rc = _access(path, eAccess);
		*outRC = (rc == 0);
		return RC_OK;
	}
	__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut)
	{
		char dir[MAXPATHNAME+1];
		if (path[0] == '/')
			dir[0] = '\0';
		else
			getcwd(dir, sizeof(dir));
		dir[MAXPATHNAME] = '\0';
		_snprintf(pathOutLength, pathOut, "%s/%s", dir, path);
		pathOut[pathOutLength-1] = '\0';
		return RC_OK;
	}

	__device__ virtual void *DlOpen(const char *filename) { return nullptr; }
	__device__ virtual void DlError(int bufLength, char *buf) { _snprintf(bufLength, buf, "Loadable extensions are not supported"); buf[bufLength-1] = '\0'; }
	__device__ virtual void (*DlSym(void *handle, const char *symbol))() { return nullptr; }
	__device__ virtual void DlClose(void *handle) { }
	__device__ virtual int Randomness(int bufLength, char *buf) { return RC_OK; }
	__device__ virtual int Sleep(int microseconds)
	{
		sleep(microseconds / 1000000);
		usleep(microseconds % 1000000);
		return microseconds;
	}
	//__device__ virtual RC CurrentTimeInt64(int64 *now);
	__device__ virtual RC CurrentTime(double *now)
	{
		// Set *pTime to the current UTC time expressed as a Julian day. Return SQLITE_OK if successful, or an error code otherwise.
		//   http://en.wikipedia.org/wiki/Julian_day
		// This implementation is not very good. The current time is rounded to an integer number of seconds. Also, assuming time_t is a signed 32-bit 
		// value, it will stop working some time in the year 2038 AD (the so-called "year 2038" problem that afflicts systems that store time this way). 
		time_t t = time(0);
		*now = t/86400.0 + 2440587.5; 
		return RC_OK;
	}
	//__device__ virtual RC GetLastError(int bufLength, char *buf);
};

__device__ static unsigned char _demoVfsBuf[sizeof(DemoVSystem)];
__device__ static DemoVSystem *_demoVfs = new (_demoVfsBuf) DemoVSystem();
__device__ VSystem *sqlite3_demovfs()
{
	_demoVfs->SizeOsFile = sizeof(DemoVFile);
	_demoVfs->MaxPathname = MAXPATHNAME;
	_demoVfs->Name = "demo";
	return _demoVfs;
}

#endif

#ifdef _TEST

#include <Jim.h>
#if OS_UNIX
__device__ static int register_demovfs(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	VSystem::RegisterVfs(sqlite3_demovfs(), true);
	return JIM_OK;
}
__device__ static int unregister_demovfs(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	VSystem::Unregister(sqlite3_demovfs());
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__device__ int Sqlitetest_demovfs_Init(Jim_Interp *interp)
{
	Jim_CreateObjCommand(interp, "register_demovfs", register_demovfs, nullptr, nullptr);
	Jim_CreateObjCommand(interp, "unregister_demovfs", unregister_demovfs, nullptr, nullptr);
	return JIM_OK;
}
#else
__device__ int Sqlitetest_demovfs_Init(Jim_Interp *interp) { return JIM_OK; }
#endif

#endif
