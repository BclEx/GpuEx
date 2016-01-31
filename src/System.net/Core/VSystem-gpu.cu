// os_gpu.c
#include "Core.cu.h"
#include <new.h>

#if OS_GPUx

namespace Core
{

#pragma region Preamble

#if defined(_TEST) || defined(_DEBUG)
	__device__ bool OsTrace = true;
#define OSTRACE(X, ...) if (OsTrace) { _dprintf(X, __VA_ARGS__); }
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

#define MAX_PATH 100

	// LOCKFILE_FAIL_IMMEDIATELY is undefined on some Windows systems.
#ifndef LOCKFILE_FAIL_IMMEDIATELY
#define LOCKFILE_FAIL_IMMEDIATELY 1
#endif
#ifndef LOCKFILE_EXCLUSIVE_LOCK
#define LOCKFILE_EXCLUSIVE_LOCK 2
#endif

	// Historically, SQLite has used both the LockFile and LockFileEx functions. When the LockFile function was used, it was always expected to fail
	// immediately if the lock could not be obtained.  Also, it always expected to obtain an exclusive lock.  These flags are used with the LockFileEx function
	// and reflect those expectations; therefore, they should not be changed.
#ifndef LOCKFILE_FLAGS
#define LOCKFILE_FLAGS (LOCKFILE_FAIL_IMMEDIATELY | LOCKFILE_EXCLUSIVE_LOCK)
#endif
#ifndef LOCKFILEEX_FLAGS
#define LOCKFILEEX_FLAGS (LOCKFILE_FAIL_IMMEDIATELY)
#endif

#pragma endregion

#pragma region GpuVFile

	typedef struct gpuLock
	{
		int Readers;       // Number of reader locks obtained
		bool Pending;      // Indicates a pending lock has been obtained
		bool Reserved;     // Indicates a reserved lock has been obtained
		bool Exclusive;    // Indicates an exclusive lock has been obtained
	} gpuLock;

	// gpuFile
	class GpuVFile : public VFile
	{
	public:
		VSystem *Vfs;			// The VFS used to open this file
		HANDLE H;				// Handle for accessing the file
		LOCK Lock_;				// Type of lock currently held on this file
		short SharedLockByte;   // Randomly chosen byte used as a shared lock
		DWORD LastErrno;		// The Windows errno from the last I/O error
		const char *Path;		// Full pathname of this file
		int SizeChunk;          // Chunk size configured by FCNTL_CHUNK_SIZE
		//
		char *DeleteOnClose;	// Name of file to delete when closing
		HANDLE Mutex;			// Mutex used to control access to shared lock
		HANDLE SharedHandle;	// Shared memory segment used for locking
		gpuLock Local;			// Locks obtained by this instance of winFile
		gpuLock *Shared;		// Global shared lock memory for the file

	public:
		__device__ virtual RC Read(void *buffer, int amount, int64 offset);
		__device__ virtual RC Write(const void *buffer, int amount, int64 offset);
		__device__ virtual RC Truncate(int64 size);
		__device__ virtual RC Close_();
		__device__ virtual RC Sync(SYNC flags);
		__device__ virtual RC get_FileSize(int64 &size);

		__device__ virtual RC Lock(LOCK lock);
		__device__ virtual RC Unlock(LOCK lock);
		__device__ virtual RC CheckReservedLock(int &lock);
		__device__ virtual RC FileControl(FCNTL op, void *arg);

		__device__ virtual uint get_SectorSize();
		__device__ virtual IOCAP get_DeviceCharacteristics();
	};

#pragma endregion

#pragma region GpuVSystem

	class GpuVSystem : public VSystem
	{
	public:
		Hash FS; // The FS
	public:
		//__device__ GpuVSystem() { }
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

#pragma region Gpu

#ifndef GPU_DATA_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the data directory should be changed.
#define GPU_DATA_DIRECTORY_TYPE (1)
#endif
#ifndef GPU_TEMP_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the temporary directory should be changed.
#define GPU_TEMP_DIRECTORY_TYPE (2) 
#endif

#ifndef TEMP_FILE_PREFIX
#define TEMP_FILE_PREFIX "etilqs_"
#endif

#pragma endregion

#pragma region Gpu

	__device__ char *g_data_directory;
	__device__ char *g_temp_directory;
	__device__ RC gpu_SetDirectory(DWORD type, void *value)
	{
#ifndef OMIT_AUTOINIT
		RC rc = SysEx::AutoInitialize();
		if (rc) return rc;
#endif
		char **directory = nullptr;
		if (type == GPU_DATA_DIRECTORY_TYPE)
			directory = &g_data_directory;
		else if (type == GPU_TEMP_DIRECTORY_TYPE)
			directory = &g_temp_directory;
		_assert(!directory || type == GPU_DATA_DIRECTORY_TYPE || type == GPU_TEMP_DIRECTORY_TYPE);
		_assert(!directory || _memdbg_hastype(*directory, MEMTYPE_HEAP));
		if (directory)
		{
			_free(*directory);
			*directory = (char *)value;
			return RC_OK;
		}
		return RC_ERROR;
	}

#pragma endregion

#pragma region OS Errors

	__device__ static RC getLastErrorMsg(DWORD lastErrno, int bufLength, char *buf)
	{
		// FormatMessage returns 0 on failure.  Otherwise it returns the number of TCHARs written to the output buffer, excluding the terminating null char.
		DWORD dwLen = 0;
		char *out = nullptr;
		//LPWSTR tempWide = NULL;
		//dwLen = osFormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPWSTR)&tempWide, 0, 0);
		//if (dwLen > 0)
		//{
		//	// allocate a buffer and convert to UTF8
		//	_benignalloc_begin();
		//	out = UnicodeToUtf8(tempWide);
		//	_benignalloc_end();
		//}
		if (!dwLen)
			__snprintf(buf, bufLength, "OsError 0x%x (%u)", lastErrno, lastErrno);
		else
		{
			// copy a maximum of nBuf chars to output buffer
			__snprintf(buf, bufLength, "%s", out);
			// free the UTF8 buffer
			_free(out);
		}
		return RC_OK;
	}

#define gpuLogError(a,b,c,d) gpuLogErrorAtLine(a,b,c,d,__LINE__)
	__device__ static RC gpuLogErrorAtLine(RC errcode, DWORD lastErrno, const char *func, const char *path, int line)
	{
		char msg[500]; // Human readable error text
		msg[0] = 0;
		getLastErrorMsg(lastErrno, sizeof(msg), msg);
		_assert(errcode != RC_OK);
		if (!path) path = "";
		int i;
		for (i = 0; msg[i] && msg[i] != '\r' && msg[i] != '\n'; i++) { }
		msg[i] = 0;
		SysEx_LOG(errcode, "os_win.c:%d: (%d) %s(%s) - %s", line, lastErrno, func, path, msg);
		return errcode;
	}

#ifndef GPU_IOERR_RETRY
#define GPU_IOERR_RETRY 1
#endif
#ifndef GPU_IOERR_RETRY_DELAY
#define GPU_IOERR_RETRY_DELAY 10
#endif
	__device__ static int gpuIoerrRetry = GPU_IOERR_RETRY;
	__device__ static int gpuIoerrRetryDelay = GPU_IOERR_RETRY_DELAY;

	__device__ static int retryIoerr(int *retry, DWORD *error)
	{
		DWORD e = osGetLastError();
		if (*retry >= gpuIoerrRetry)
		{
			if (error)
				*error = e;
			return 0;
		}
		if (e == ERROR_ACCESS_DENIED || e == ERROR_LOCK_VIOLATION || e == ERROR_SHARING_VIOLATION)
		{
			osSleep(gpuIoerrRetryDelay*(1+*retry));
			++*retry;
			return 1;
		}
		if (error)
			*error = e;
		return 0;
	}

	__device__ static void logIoerr(int retry)
	{
		if (retry)
			SysEx_LOG(RC_IOERR, "delayed %dms for lock/sharing conflict", gpuIoerrRetryDelay*retry*(retry+1)/2);
	}

#pragma endregion

#pragma region GPU Only

	__device__ static void *ConvertFilename(const char *name)
	{
		void *converted = nullptr;
		int length = _strlen(name);
		converted = _alloc(length);
		_memcpy(converted, name, length);
		return converted;
	}

#define HANDLE_TO_GPUFILE(a) (GpuVFile*)&((char*)a)[-(int)offsetof(GpuVFile,H)]

	__device__ static void gpuMutexAcquire(HANDLE h)
	{
		DWORD err;
		do
		{
			err = osWaitForSingleObject(h, INFINITE);
		} while (err != WAIT_OBJECT_0 && err != WAIT_ABANDONED);
	}

#define gpuMutexRelease(h) osReleaseMutex(h)

	__device__ static RC gpuCreateLock(const char *filename, GpuVFile *file)
	{
		char *name = (char *)ConvertFilename(filename);
		if (!name)
			return RC_IOERR_NOMEM;
		// Initialize the local lockdata
		_memset(&file->Local, 0, sizeof(file->Local));
		// Replace the backslashes from the filename and lowercase it to derive a mutex name.
		//LPWSTR tok = osCharLowerW(name);
		//for (; *tok; tok++)
		//	if (*tok == '\\') *tok = '_';
		// Create/open the named mutex
		file->Mutex = osCreateMutexA(NULL, false, name);
		if (!file->Mutex)
		{
			file->LastErrno = osGetLastError();
			gpuLogError(RC_IOERR, file->LastErrno, "gpuCreateLock1", filename);
			_free(name);
			return RC_IOERR;
		}
		// Acquire the mutex before continuing
		gpuMutexAcquire(file->Mutex);
		// Since the names of named mutexes, semaphores, file mappings etc are case-sensitive, take advantage of that by uppercasing the mutex name
		// and using that as the shared filemapping name.
		//osCharUpperW(name);
		file->SharedHandle = osCreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(gpuLock), name);  
		// Set a flag that indicates we're the first to create the memory so it must be zero-initialized
		bool init = true;
		DWORD lastErrno = osGetLastError();
		if (lastErrno == ERROR_ALREADY_EXISTS)
			init = false;
		_free(name);

		// If we succeeded in making the shared memory handle, map it.
		bool logged = false;
		if (file->SharedHandle)
		{
			file->Shared = (gpuLock *)osMapViewOfFile(file->SharedHandle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(gpuLock));
			// If mapping failed, close the shared memory handle and erase it
			if (!file->Shared)
			{
				file->LastErrno = osGetLastError();
				gpuLogError(RC_IOERR, file->LastErrno, "gpuCreateLock2", filename);
				logged = true;
				osCloseHandle(file->SharedHandle);
				file->SharedHandle = NULL;
			}
		}
		// If shared memory could not be created, then close the mutex and fail
		if (!file->SharedHandle)
		{
			if (!logged)
			{
				file->LastErrno = lastErrno;
				gpuLogError(RC_IOERR, file->LastErrno, "gpuCreateLock3", filename);
				logged = true;
			}
			gpuMutexRelease(file->Mutex);
			osCloseHandle(file->Mutex);
			file->Mutex = NULL;
			return RC_IOERR;
		}
		// Initialize the shared memory if we're supposed to
		if (init)
			_memset(file->Shared, 0, sizeof(gpuLock));
		gpuMutexRelease(file->Mutex);
		return RC_OK;
	}

	__device__ static void gpuDestroyLock(GpuVFile *file)
	{
		if (file->Mutex)
		{
			// Acquire the mutex
			gpuMutexAcquire(file->Mutex);
			// The following blocks should probably assert in debug mode, but they are to cleanup in case any locks remained open
			if (file->Local.Readers)
				file->Shared->Readers--;
			if (file->Local.Reserved)
				file->Shared->Reserved = false;
			if (file->Local.Pending)
				file->Shared->Pending = false;
			if (file->Local.Exclusive)
				file->Shared->Exclusive = false;
			// De-reference and close our copy of the shared memory handle
			osUnmapViewOfFile(file->Shared);
			osCloseHandle(file->SharedHandle);
			// Done with the mutex
			gpuMutexRelease(file->Mutex);    
			osCloseHandle(file->Mutex);
			file->Mutex = NULL;
		}
	}

	__device__ static bool gpuLockFile(LPHANDLE fileHandle, DWORD flags, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh)
	{
		GpuVFile *file = HANDLE_TO_GPUFILE(fileHandle);
		bool r = false;
		if (!file->Mutex) return true;
		gpuMutexAcquire(file->Mutex);
		// Wanting an exclusive lock?
		if (offsetLow == (DWORD)SHARED_FIRST && numBytesLow == (DWORD)SHARED_SIZE)
		{
			if (file->Shared->Readers == 0 && !file->Shared->Exclusive)
			{
				file->Shared->Exclusive = true;
				file->Local.Exclusive = true;
				r = true;
			}
		}
		// Want a read-only lock? 
		else if (offsetLow == (DWORD)SHARED_FIRST && numBytesLow == 1)
		{
			if (!file->Shared->Exclusive)
			{
				file->Local.Readers++;
				if (file->Local.Readers == 1)
					file->Shared->Readers++;
				r = true;
			}
		}
		// Want a pending lock?
		else if (offsetLow == (DWORD)PENDING_BYTE && numBytesLow == 1)
		{
			// If no pending lock has been acquired, then acquire it
			if (!file->Shared->Pending) 
			{
				file->Shared->Pending = true;
				file->Local.Pending = true;
				r = true;
			}
		}
		// Want a reserved lock?
		else if (offsetLow == (DWORD)RESERVED_BYTE && numBytesLow == 1)
		{
			if (!file->Shared->Reserved)
			{
				file->Shared->Reserved = true;
				file->Local.Reserved = true;
				r = true;
			}
		}
		gpuMutexRelease(file->Mutex);
		return r;
	}

	__device__ static bool gpuUnlockFile(LPHANDLE fileHandle, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh)
	{
		GpuVFile *file = HANDLE_TO_GPUFILE(fileHandle);
		bool r = false;
		if (!file->Mutex) return true;
		gpuMutexAcquire(file->Mutex);
		// Releasing a reader lock or an exclusive lock
		if (offsetLow == (DWORD)SHARED_FIRST)
		{
			// Did we have an exclusive lock?
			if (file->Local.Exclusive)
			{
				_assert(numBytesLow == (DWORD)SHARED_SIZE);
				file->Local.Exclusive = false;
				file->Shared->Exclusive = false;
				r = true;
			}
			// Did we just have a reader lock?
			else if (file->Local.Readers)
			{
				_assert(numBytesLow == (DWORD)SHARED_SIZE || numBytesLow == 1);
				file->Local.Readers--;
				if (file->Local.Readers == 0)
					file->Shared->Readers--;
				r = true;
			}
		}
		// Releasing a pending lock
		else if (offsetLow == (DWORD)PENDING_BYTE && numBytesLow == 1)
		{
			if (file->Local.Pending)
			{
				file->Local.Pending = false;
				file->Shared->Pending = false;
				r = true;
			}
		}
		// Releasing a reserved lock
		else if (offsetLow == (DWORD)RESERVED_BYTE && numBytesLow == 1)
		{
			if (file->Local.Reserved)
			{
				file->Local.Reserved = false;
				file->Shared->Reserved = false;
				r = true;
			}
		}
		gpuMutexRelease(file->Mutex);
		return r;
	}

#pragma endregion

#pragma region GpuVFile

	__device__ static int seekGpuFile(GpuVFile *file, int64 offset)
	{
		bool ret = osSetFilePointer(file->H, offset, 0, FILE_BEGIN); // Value returned by SetFilePointerEx()
		if (!ret)
		{
			file->LastErrno = osGetLastError();
			gpuLogError(RC_IOERR_SEEK, file->LastErrno, "seekGpuFile", file->Path);
			return 1;
		}
		return 0;
	}

#define MAX_CLOSE_ATTEMPT 3
	__device__ RC GpuVFile::Close_()
	{
		OSTRACE("CLOSE %d\n", H);
		_assert(H != NULL && H != INVALID_HANDLE_VALUE);
		int rc;
		int cnt = 0;
		do
		{
			rc = osCloseHandle(H);
		} while (!rc && ++cnt < MAX_CLOSE_ATTEMPT && (osSleep(100), 1));
#define GPU_DELETION_ATTEMPTS 3
		gpuDestroyLock(this);
		if (DeleteOnClose)
		{
			int cnt = 0;
			while (osDeleteFileA(DeleteOnClose) == 0 && osGetFileAttributesA(DeleteOnClose) != 0xffffffff && cnt++ < GPU_DELETION_ATTEMPTS)
				osSleep(100); // Wait a little before trying again
			_free(DeleteOnClose);
		}
		OSTRACE("CLOSE %d %s\n", H, rc ? "ok" : "failed");
		if (rc)
			H = NULL;
		Opened = false;
		return (rc ? RC_OK : gpuLogError(RC_IOERR_CLOSE, osGetLastError(), "gpuClose", Path));
	}

	__device__ RC GpuVFile::Read(void *buffer, int amount, int64 offset)
	{
		OSOVERLAPPED overlapped; // The offset for ReadFile.
		int retry = 0; // Number of retrys
		SimulateIOError(return RC_IOERR_READ);
		OSTRACE("READ %d lock=%d\n", H, Lock_);
		DWORD read; // Number of bytes actually read from file
		_memset(&overlapped, 0, sizeof(OSOVERLAPPED));
		overlapped.Offset = (DWORD)(offset & 0xffffffff);
		overlapped.OffsetHigh = (DWORD)((offset>>32) & 0x7fffffff);
		while (!osReadFile(H, buffer, amount, &read, &overlapped) && osGetLastError() != ERROR_HANDLE_EOF)
		{
			DWORD lastErrno;
			if (retryIoerr(&retry, &lastErrno)) continue;
			LastErrno = lastErrno;
			return gpuLogError(RC_IOERR_READ, LastErrno, "gpuRead", Path);
		}
		logIoerr(retry);
		if (read < (DWORD)amount)
		{
			// Unread parts of the buffer must be zero-filled
			_memset(&((char *)buffer)[read], 0, amount - read);
			return RC_IOERR_SHORT_READ;
		}
		return RC_OK;
	}

	__device__ RC GpuVFile::Write(const void *buffer, int amount, int64 offset)
	{
		_assert(amount > 0);
		SimulateIOError(return RC_IOERR_WRITE);
		SimulateDiskfullError(return RC_FULL);
		OSTRACE("WRITE %d lock=%d\n", H, Lock_);
		int rc = 0; // True if error has occurred, else false
		int retry = 0; // Number of retries
		{
			OSOVERLAPPED overlapped; // The offset for WriteFile.
			_memset(&overlapped, 0, sizeof(OSOVERLAPPED));
			overlapped.Offset = (DWORD)(offset & 0xffffffff);
			overlapped.OffsetHigh = (DWORD)((offset>>32) & 0x7fffffff);

			uint8 *remain = (uint8 *)buffer; // Data yet to be written
			int remainLength = amount; // Number of bytes yet to be written
			DWORD write; // Bytes written by each WriteFile() call
			DWORD lastErrno = NO_ERROR; // Value returned by GetLastError()
			while (remainLength > 0)
			{
				if (!osWriteFile(H, remain, remainLength, &write, &overlapped))
				{
					if (retryIoerr(&retry, &lastErrno)) continue;
					break;
				}
				_assert(write == 0 || write <= (DWORD)remainLength);
				if (write == 0 || write > (DWORD)remainLength)
				{
					lastErrno = osGetLastError();
					break;
				}
				offset += write;
				overlapped.Offset = (DWORD)(offset & 0xffffffff);
				overlapped.OffsetHigh = (DWORD)((offset>>32) & 0x7fffffff);

				remain += write;
				remainLength -= write;
			}
			if (remainLength > 0)
			{
				LastErrno = lastErrno;
				rc = 1;
			}
		}
		if (rc)
		{
			if (LastErrno == ERROR_HANDLE_DISK_FULL ||  LastErrno == ERROR_DISK_FULL)
				return RC_FULL;
			return gpuLogError(RC_IOERR_WRITE, LastErrno, "gpuWrite", Path);
		}
		else
			logIoerr(retry);
		return RC_OK;
	}

	__device__ RC GpuVFile::Truncate(int64 size)
	{
		RC rc = RC_OK;
		OSTRACE("TRUNCATE %d %lld\n", H, size);
		SimulateIOError(return RC_IOERR_TRUNCATE);
		// If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
		// actual file size after the operation may be larger than the requested size).
		if (SizeChunk > 0)
			size = ((size+SizeChunk-1)/SizeChunk)*SizeChunk;
		// SetEndOfFile() returns non-zero when successful, or zero when it fails.
		if (seekGpuFile(this, size))
			rc = gpuLogError(RC_IOERR_TRUNCATE, LastErrno, "gpuTruncate1", Path);
		else if (!osSetEndOfFile(H))
		{
			LastErrno = osGetLastError();
			rc = gpuLogError(RC_IOERR_TRUNCATE, LastErrno, "gpuTruncate2", Path);
		}
		OSTRACE("TRUNCATE %d %lld %s\n", H, size, rc ? "failed" : "ok");
		return rc;
	}

#ifdef _TEST
	// Count the number of fullsyncs and normal syncs.  This is used to test that syncs and fullsyncs are occuring at the right times.
	__device__ int g_sync_count = 0;
	__device__ int g_fullsync_count = 0;
#endif
	__device__ RC GpuVFile::Sync(SYNC flags)
	{
		// Check that one of SQLITE_SYNC_NORMAL or FULL was passed
		_assert((flags&0x0F) == SYNC_NORMAL || (flags&0x0F) == SYNC_FULL);
		OSTRACE("SYNC %d lock=%d\n", H, Lock_);
		// Unix cannot, but some systems may return SQLITE_FULL from here. This line is to test that doing so does not cause any problems.
		SimulateDiskfullError(return RC_FULL);
#ifdef _TEST
		if ((flags&0x0F) == SYNC_FULL)
			g_fullsync_count++;
		g_sync_count++;
#endif
#ifdef NO_SYNC // If we compiled with the SQLITE_NO_SYNC flag, then syncing is a no-op
		return RC_OK;
#else
		bool rc = osFlushFileBuffers(H);
		SimulateIOError(rc = false);
		if (rc)
			return RC_OK;
		LastErrno = osGetLastError();
		return gpuLogError(RC_IOERR_FSYNC, LastErrno, "gpuSync", Path);
#endif
	}

	__device__ RC GpuVFile::get_FileSize(int64 &size)
	{
		RC rc = RC_OK;
		SimulateIOError(return RC_IOERR_FSTAT);
		DWORD upperBits;
		DWORD lowerBits = osGetFileSize(H, &upperBits);
		size = (((int64)upperBits)<<32) + lowerBits;
		DWORD lastErrno;
		if (lowerBits == INVALID_FILE_SIZE && (lastErrno = osGetLastError()) != NO_ERROR)
		{
			LastErrno = lastErrno;
			rc = gpuLogError(RC_IOERR_FSTAT, LastErrno, "gpuFileSize", Path);
		}
		return rc;
	}

	__device__ static int getReadLock(GpuVFile *file)
	{
		int res;
		res = gpuLockFile(&file->H, 0, SHARED_FIRST, 0, 1, 0);
		//int lock;
		//SysEx::PutRandom(sizeof(lock), &lock);
		//file->SharedLockByte = (short)((lock & 0x7fffffff)%(SHARED_SIZE - 1));
		//res = gpuLockFile(&file->H, LOCKFILE_FLAGS, SHARED_FIRST + file->SharedLockByte, 0, 1, 0);
		if (res == 0)
			file->LastErrno = osGetLastError();
		// No need to log a failure to lock
		return res;
	}

	__device__ static int unlockReadLock(GpuVFile *file)
	{
		int res;
		res = gpuUnlockFile(&file->H, SHARED_FIRST, 0, 1, 0);
		//res = gpuUnlockFile(&file->H, SHARED_FIRST + file->SharedLockByte, 0, 1, 0);
		DWORD lastErrno;
		if (res == 0 && (lastErrno = osGetLastError()) != ERROR_NOT_LOCKED)
		{
			file->LastErrno = lastErrno;
			gpuLogError(RC_IOERR_UNLOCK, file->LastErrno, "unlockReadLock", file->Path);
		}
		return res;
	}

	__device__ RC GpuVFile::Lock(LOCK lock)
	{
		OSTRACE("LOCK %d %d was %d(%d)\n", H, lock, Lock_, SharedLockByte);

		// If there is already a lock of this type or more restrictive on the OsFile, do nothing. Don't use the end_lock: exit path, as
		// sqlite3OsEnterMutex() hasn't been called yet.
		if (Lock_ >= lock)
			return RC_OK;

		// Make sure the locking sequence is correct
		_assert(Lock_ != LOCK_NO || lock == LOCK_SHARED);
		_assert(lock != LOCK_PENDING);
		_assert(lock != LOCK_RESERVED || Lock_ == LOCK_SHARED);

		// Lock the PENDING_LOCK byte if we need to acquire a PENDING lock or a SHARED lock.  If we are acquiring a SHARED lock, the acquisition of
		// the PENDING_LOCK byte is temporary.
		LOCK newLock = Lock_; // Set pFile->locktype to this value before exiting
		int res = 1; // Result of a Windows lock call
		bool gotPendingLock = false; // True if we acquired a PENDING lock this time
		DWORD lastErrno = NO_ERROR;
		if (Lock_ == LOCK_NO || (lock == LOCK_EXCLUSIVE && Lock_ == LOCK_RESERVED))
		{
			int cnt = 3;
			while (cnt-- > 0 && (res = gpuLockFile(&H, LOCKFILE_FLAGS, PENDING_BYTE, 0, 1, 0)) == 0)
			{
				// Try 3 times to get the pending lock.  This is needed to work around problems caused by indexing and/or anti-virus software on Windows systems.
				// If you are using this code as a model for alternative VFSes, do not copy this retry logic.  It is a hack intended for Windows only.
				OSTRACE("could not get a PENDING lock. cnt=%d\n", cnt);
				if (cnt) osSleep(1);
			}
			gotPendingLock = (res != 0);
			if (!res)
				lastErrno = osGetLastError();
		}

		// Acquire a SHARED lock
		if (lock == LOCK_SHARED && res)
		{
			_assert(Lock_ == LOCK_NO);
			res = getReadLock(this);
			if (res)
				newLock = LOCK_SHARED;
			else
				lastErrno = osGetLastError();
		}

		// Acquire a RESERVED lock
		if (lock == LOCK_RESERVED && res)
		{
			_assert(Lock_ == LOCK_SHARED);
			res = gpuLockFile(&H, LOCKFILE_FLAGS, RESERVED_BYTE, 0, 1, 0);
			if (res)
				newLock = LOCK_RESERVED;
			else
				lastErrno = osGetLastError();
		}

		// Acquire a PENDING lock
		if (lock == LOCK_EXCLUSIVE && res)
		{
			newLock = LOCK_PENDING;
			gotPendingLock = false;
		}

		// Acquire an EXCLUSIVE lock
		if (lock == LOCK_EXCLUSIVE && res)
		{
			_assert(Lock_ >= LOCK_SHARED);
			res = unlockReadLock(this);
			OSTRACE("unreadlock = %d\n", res);
			res = gpuLockFile(&H, LOCKFILE_FLAGS, SHARED_FIRST, 0, SHARED_SIZE, 0);
			if (res)
				newLock = LOCK_EXCLUSIVE;
			else
			{
				lastErrno = osGetLastError();
				OSTRACE("error-code = %d\n", lastErrno);
				getReadLock(this);
			}
		}

		// If we are holding a PENDING lock that ought to be released, then release it now.
		if (gotPendingLock && lock == LOCK_SHARED)
			gpuUnlockFile(&H, PENDING_BYTE, 0, 1, 0);

		// Update the state of the lock has held in the file descriptor then return the appropriate result code.
		RC rc;
		if (res)
			rc = RC_OK;
		else
		{
			OSTRACE("LOCK FAILED %d trying for %d but got %d\n", H, lock, newLock);
			LastErrno = lastErrno;
			rc = RC_BUSY;
		}
		Lock_ = newLock;
		return rc;
	}

	__device__ RC GpuVFile::CheckReservedLock(int &lock)
	{
		SimulateIOError(return RC_IOERR_CHECKRESERVEDLOCK;);
		int rc;
		if (Lock_ >= LOCK_RESERVED)
		{
			rc = 1;
			OSTRACE("_TEST WR-LOCK %d %d (local)\n", H, rc);
		}
		else
		{
			rc = gpuLockFile(&H, LOCKFILEEX_FLAGS, RESERVED_BYTE, 0, 1, 0);
			if (rc)
				gpuUnlockFile(&H, RESERVED_BYTE, 0, 1, 0);
			rc = !rc;
			OSTRACE("_TEST WR-LOCK %d %d (remote)\n", H, rc);
		}
		lock = rc;
		return RC_OK;
	}

	__device__ RC GpuVFile::Unlock(LOCK lock)
	{
		_assert(lock <= LOCK_SHARED);
		OSTRACE("UNLOCK %d to %d was %d(%d)\n", H, lock, Lock_, SharedLockByte);
		RC rc = RC_OK;
		LOCK type = Lock_;
		if (type >= LOCK_EXCLUSIVE)
		{
			gpuUnlockFile(&H, SHARED_FIRST, 0, SHARED_SIZE, 0);
			if (lock == LOCK_SHARED && !getReadLock(this)) // This should never happen.  We should always be able to reacquire the read lock
				rc = gpuLogError(RC_IOERR_UNLOCK, osGetLastError(), "gpuUnlock", Path);
		}
		if (type >= LOCK_RESERVED)
			gpuUnlockFile(&H, RESERVED_BYTE, 0, 1, 0);
		if (lock == LOCK_NO && type >= LOCK_SHARED)
			unlockReadLock(this);
		if (type >= LOCK_PENDING)
			gpuUnlockFile(&H, PENDING_BYTE, 0, 1, 0);
		Lock_ = lock;
		return rc;
	}

	//__device__ static void gpuModeBit(GpuVFile *file, uint8 mask, int *arg)
	//{
	//	if (*arg < 0)
	//		*arg = ((file->CtrlFlags & mask) != 0);
	//	else if ((*arg) == 0)
	//		file->CtrlFlags = (GpuVFile::WINFILE)(file->CtrlFlags & ~mask);
	//	else
	//		file->CtrlFlags |= mask;
	//}

	__device__ static RC getTempname(int bufLength, char *buf);
	__device__ RC GpuVFile::FileControl(FCNTL op, void *arg)
	{
		int *a;
		char *tfile;
		switch (op)
		{
		case FCNTL_LOCKSTATE:
			*(int*)arg = Lock_;
			return RC_OK;
		case FCNTL_LAST_ERRNO:
			*(int*)arg = (int)LastErrno;
			return RC_OK;
		case FCNTL_CHUNK_SIZE:
			SizeChunk = *(int *)arg;
			return RC_OK;
		case FCNTL_SIZE_HINT:
			if (SizeChunk > 0)
			{
				int64 oldSize;
				RC rc = get_FileSize(oldSize);
				if (rc == RC_OK)
				{
					int64 newSize = *(int64 *)arg;
					if (newSize > oldSize)
					{
						SimulateIOErrorBenign(true);
						rc = Truncate(newSize);
						SimulateIOErrorBenign(false);
					}
				}
				return rc;
			}
			return RC_OK;
			//case FCNTL_PERSIST_WAL:
			//	winModeBit(this, (uint8)WINFILE_PERSIST_WAL, (int*)arg);
			//	return RC_OK;
			//case FCNTL_POWERSAFE_OVERWRITE:
			//	winModeBit(this, (uint8)WINFILE_PSOW, (int*)arg);
			//	return RC_OK;
		case FCNTL_VFSNAME:
			*(char**)arg = "win32";
			return RC_OK;
		case FCNTL_WIN32_AV_RETRY:
			a = (int*)arg;
			if (a[0] > 0)
				gpuIoerrRetry = a[0];
			else
				a[0] = gpuIoerrRetry;
			if (a[1] > 0)
				gpuIoerrRetryDelay = a[1];
			else
				a[1] = gpuIoerrRetryDelay;
			return RC_OK;
		case FCNTL_TEMPFILENAME:
			tfile = (char *)_allocZero(Vfs->MaxPathname);
			if (tfile)
			{
				getTempname(Vfs->MaxPathname, tfile);
				*(char**)arg = tfile;
			}
			return RC_OK;
		}
		return RC_NOTFOUND;
	}

	__device__ uint GpuVFile::get_SectorSize()
	{
		return 512;
	}

	__device__ VFile::IOCAP GpuVFile::get_DeviceCharacteristics()
	{
		return (VFile::IOCAP)0;
	}

#pragma endregion

#pragma region GpuVSystem

	__constant__ static char _chars[] =
		"abcdefghijklmnopqrstuvwxyz"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"0123456789";
	__device__ static RC getTempname(int bufLength, char *buf)
	{
		// It's odd to simulate an io-error here, but really this is just using the io-error infrastructure to test that SQLite handles this function failing.
		SimulateIOError(return RC_IOERR);
		char tempPath[MAX_PATH+2];
		_memset(tempPath, 0, MAX_PATH+2);
		if (g_temp_directory)
			__snprintf(tempPath, MAX_PATH-30, "%s", g_temp_directory);
		// Check that the output buffer is large enough for the temporary file name. If it is not, return SQLITE_ERROR.
		int tempPathLength = _strlen(tempPath);
		if ((tempPathLength + _strlen(TEMP_FILE_PREFIX) + 18) >= bufLength)
			return RC_ERROR;
		size_t i;
		for (i = tempPathLength; i > 0 && tempPath[i-1] == '\\'; i--) { }
		tempPath[i] = 0;
		size_t j;
		__snprintf(buf, bufLength-18, (tempPathLength > 0 ? "%s\\"TEMP_FILE_PREFIX : TEMP_FILE_PREFIX), tempPath);
		j = _strlen(buf);
		SysEx::PutRandom(15, &buf[j]);
		for (i = 0; i < 15; i++, j++)
			buf[j] = (char)_chars[((unsigned char)buf[j])%(sizeof(_chars)-1)];
		buf[j] = 0;
		buf[j+1] = 0;
		OSTRACE("TEMP FILENAME: %s\n", buf);
		return RC_OK; 
	}

	__device__ static bool gpuIsDir(const void *converted)
	{
		DWORD attr = osGetFileAttributesA((char *)converted);
		return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
	}

	__device__ VFile *GpuVSystem::_AttachFile(void *buffer)
	{
		return new (buffer) GpuVFile();
	}

	__device__ RC GpuVSystem::Open(const char *name, VFile *id, OPEN flags, OPEN *outFlags)
	{
		// 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
		// SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
		flags = (OPEN)((uint)flags & 0x87f7f);

		RC rc = RC_OK;
		OPEN type = (OPEN)(flags & 0xFFFFFF00);  // Type of file to open
		bool isExclusive = ((flags & OPEN_EXCLUSIVE) != 0);
		bool isDelete = ((flags & OPEN_DELETEONCLOSE) != 0);
		bool isCreate = ((flags & OPEN_CREATE) != 0);
		bool isReadonly = ((flags & OPEN_READONLY) != 0);
		bool isReadWrite = ((flags & OPEN_READWRITE) != 0);
		bool isOpenJournal = (isCreate && (type == OPEN_MASTER_JOURNAL || type == OPEN_MAIN_JOURNAL || type == OPEN_WAL));

		// Check the following statements are true: 
		//
		//   (a) Exactly one of the READWRITE and READONLY flags must be set, and 
		//   (b) if CREATE is set, then READWRITE must also be set, and
		//   (c) if EXCLUSIVE is set, then CREATE must also be set.
		//   (d) if DELETEONCLOSE is set, then CREATE must also be set.
		_assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
		_assert(!isCreate || isReadWrite);
		_assert(!isExclusive || isCreate);
		_assert(!isDelete || isCreate);

		// The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
		_assert((!isDelete && name) || type != OPEN_MAIN_DB);
		_assert((!isDelete && name) || type != OPEN_MAIN_JOURNAL);
		_assert((!isDelete && name) || type != OPEN_MASTER_JOURNAL);
		_assert((!isDelete && name) || type != OPEN_WAL);

		// Assert that the upper layer has set one of the "file-type" flags.
		_assert(type == OPEN_MAIN_DB || type == OPEN_TEMP_DB ||
			type == OPEN_MAIN_JOURNAL || type == OPEN_TEMP_JOURNAL ||
			type == OPEN_SUBJOURNAL || type == OPEN_MASTER_JOURNAL ||
			type == OPEN_TRANSIENT_DB || type == OPEN_WAL);

		GpuVFile *file = (GpuVFile *)id;
		_assert(file != nullptr);
		_memset(file, 0, sizeof(GpuVFile));
		file = new (file) GpuVFile();
		file->H = INVALID_HANDLE_VALUE;

		// If the second argument to this function is NULL, generate a temporary file name to use 
		const char *utf8Name = name; // Filename in UTF-8 encoding
		char tmpname[MAX_PATH+2];     // Buffer used to create temp filename
		if (!utf8Name)
		{
			_assert(isDelete && !isOpenJournal);
			_memset(tmpname, 0, MAX_PATH+2);
			rc = getTempname(MAX_PATH+2, tmpname);
			if (rc != RC_OK)
				return rc;
			utf8Name = tmpname;
		}

		// Database filenames are double-zero terminated if they are not URIs with parameters.  Hence, they can always be passed into sqlite3_uri_parameter().
		_assert(type != OPEN_MAIN_DB || (flags & OPEN_URI) || utf8Name[_strlen(utf8Name)+1] == 0);

		// Convert the filename to the system encoding.
		void *converted = ConvertFilename(utf8Name); // Filename in OS encoding
		if (!converted)
			return RC_IOERR_NOMEM;

		if (gpuIsDir(converted))
		{
			_free(converted);
			return RC_CANTOPEN_ISDIR;
		}

		DWORD dwDesiredAccess;
		if (isReadWrite)
			dwDesiredAccess = GENERIC_READ | GENERIC_WRITE;
		else
			dwDesiredAccess = GENERIC_READ;

		// SQLITE_OPEN_EXCLUSIVE is used to make sure that a new file is created. SQLite doesn't use it to indicate "exclusive access" as it is usually understood.
		DWORD dwCreationDisposition;
		if (isExclusive) // Creates a new file, only if it does not already exist. If the file exists, it fails.
			dwCreationDisposition = CREATE_NEW;
		else if (isCreate) // Open existing file, or create if it doesn't exist
			dwCreationDisposition = OPEN_ALWAYS;
		else // Opens a file, only if it exists.
			dwCreationDisposition = OPEN_EXISTING;

		DWORD dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;

		DWORD dwFlagsAndAttributes = 0;
		int isTemp = 0;
		if (isDelete)
		{
			dwFlagsAndAttributes = FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE;
			isTemp = 1;
		}
		else
			dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;

		HANDLE h;
		DWORD lastErrno = 0;
		int cnt = 0;
		while ((h = osCreateFileA((char *)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
		logIoerr(cnt);

		OSTRACE("OPEN %d %s 0x%lx %s\n", h, name, dwDesiredAccess, h == INVALID_HANDLE_VALUE ? "failed" : "ok");
		if (h == INVALID_HANDLE_VALUE)
		{
			file->LastErrno = lastErrno;
			gpuLogError(RC_CANTOPEN, file->LastErrno, "winOpen", utf8Name);
			_free(converted);
			if (isReadWrite && !isExclusive)
				return Open(name, id, (OPEN)((flags|OPEN_READONLY) & ~(OPEN_CREATE|OPEN_READWRITE)), outFlags);
			else
				return SysEx_CANTOPEN_BKPT;
		}

		if (outFlags)
			*outFlags = (isReadWrite ? OPEN_READWRITE : OPEN_READONLY);
		if (isReadWrite && type == OPEN_MAIN_DB && (rc = gpuCreateLock(name, file)) != RC_OK)
		{
			osCloseHandle(h);
			_free(converted);
			return rc;
		}
		if (isTemp)
			file->DeleteOnClose = (char *)converted;
		else
			_free(converted);
		file->Opened = true;
		file->Vfs = this;
		file->H = h;
		//if (VSystem::UriBoolean(name, "psow", POWERSAFE_OVERWRITE))
		//	file->CtrlFlags |= WinVFile::WINFILE_PSOW;
		file->LastErrno = NO_ERROR;
		file->Path = name;
		OpenCounter(+1);
		return rc;
	}

	__device__ RC GpuVSystem::Delete(const char *filename, bool syncDir)
	{
		SimulateIOError(return RC_IOERR_DELETE;);
		void *converted = ConvertFilename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		RC rc;
		DWORD lastErrno;
		int cnt = 0;
		do {
			attr = osGetFileAttributesA((char *)converted);
			if (attr == INVALID_FILE_ATTRIBUTES)
			{
				lastErrno = osGetLastError();
				rc = (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND ? RC_IOERR_DELETE_NOENT : RC_ERROR); // Already gone?
				break;
			}
			if (attr & FILE_ATTRIBUTE_DIRECTORY)
			{
				rc = RC_ERROR; // Files only.
				break;
			}
			if (osDeleteFileA((char *)converted))
			{
				rc = RC_OK; // Deleted OK.
				break;
			}
			if (!retryIoerr(&cnt, &lastErrno))
			{
				rc = RC_ERROR; // No more retries.
				break;
			}
		} while (1);
		if (rc && rc != RC_IOERR_DELETE_NOENT)
			rc = gpuLogError(RC_IOERR_DELETE, lastErrno, "gpuDelete", filename);
		else
			logIoerr(cnt);
		_free(converted);
		OSTRACE("DELETE \"%s\" %s\n", filename, rc ? "failed" : "ok" );
		return rc;
	}

	__device__ RC GpuVSystem::Access(const char *filename, ACCESS flags, int *resOut)
	{
		SimulateIOError(return RC_IOERR_ACCESS;);
		void *converted = ConvertFilename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		int rc = 0;
		//DWORD lastErrno;
		attr = osGetFileAttributesA((char *)converted);
		_free(converted);
		switch (flags)
		{
		case ACCESS_READ:
		case ACCESS_EXISTS:
			rc = attr != INVALID_FILE_ATTRIBUTES;
			break;
		case ACCESS_READWRITE:
			rc = attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_READONLY) == 0;
			break;
		default:
			_assert(!"Invalid flags argument");
		}
		*resOut = rc;
		return RC_OK;
	}

	__device__ static bool gpuIsVerbatimPathname(const char *pathname)
	{
		// If the path name starts with a forward slash or a backslash, it is either a legal UNC name, a volume relative path, or an absolute path name in the
		// "Unix" format on Windows.  There is no easy way to differentiate between the final two cases; therefore, we return the safer return value of TRUE
		// so that callers of this function will simply use it verbatim.
		if (pathname[0] == '/' || pathname[0] == '\\')
			return true;
		// If the path name starts with a letter and a colon it is either a volume relative path or an absolute path.  Callers of this function must not
		// attempt to treat it as a relative path name (i.e. they should simply use it verbatim).
		if (_isalpha2(pathname[0]) && pathname[1] == ':')
			return true;
		// If we get to this point, the path name should almost certainly be a purely relative one (i.e. not a UNC name, not absolute, and not volume relative).
		return false;
	}

	__device__ RC GpuVSystem::FullPathname(const char *relative, int fullLength, char *full)
	{
		SimulateIOError(return RC_ERROR);
		if (g_data_directory && !gpuIsVerbatimPathname(relative))
			__snprintf(full, MIN(fullLength, MaxPathname), "%s\\%s", g_data_directory, relative);
		else
			__snprintf(full, MIN(fullLength, MaxPathname), "%s", relative);
		return RC_OK;
	}

#ifndef OMIT_LOAD_EXTENSION
	__device__ void *GpuVSystem::DlOpen(const char *filename)
	{
		return nullptr;
	}

	__device__ void GpuVSystem::DlError(int bufLength, char *buf)
	{
	}

	__device__ void (*GpuVSystem::DlSym(void *handle, const char *symbol))()
	{
		return nullptr;
	}

	__device__ void GpuVSystem::DlClose(void *handle)
	{
	}
#else
#define winDlOpen  0
#define winDlError 0
#define winDlSym   0
#define winDlClose 0
#endif

	__device__ int GpuVSystem::Randomness(int bufLength, char *buf)
	{
		int n = 0;
#if _TEST
		n = bufLength;
		_memset(buf, 0, bufLength);
#else
		if (sizeof(DWORD) <= bufLength - n)
		{
			DWORD cnt = clock();
			memcpy(&buf[n], &cnt, sizeof(cnt));
			n += sizeof(cnt);
		}
		if (sizeof(DWORD) <= bufLength - n)
		{
			DWORD cnt = clock();
			memcpy(&buf[n], &cnt, sizeof(cnt));
			n += sizeof(cnt);
		}
#endif
		return n;
	}

	__device__ int GpuVSystem::Sleep(int microseconds)
	{
		osSleep((microseconds+999)/1000);
		return ((microseconds+999)/1000)*1000;
	}

	__device__ RC GpuVSystem::CurrentTimeInt64(int64 *now)
	{
		*now = clock();
		return RC_OK;
	}

	__device__ RC GpuVSystem::CurrentTime(double *now)
	{
		int64 i;
		RC rc = CurrentTimeInt64(&i);
		if (rc == RC_OK)
			*now = i/86400000.0;
		return rc;
	}

	__device__ RC GpuVSystem::GetLastError(int bufLength, char *buf)
	{
		return getLastErrorMsg(osGetLastError(), bufLength, buf);
	}

	__device__ RC GpuVSystem::SetSystemCall(const char *name, syscall_ptr newFunc)
	{
		return RC_ERROR;
	}
	__device__ syscall_ptr GpuVSystem::GetSystemCall(const char *name)
	{
		return nullptr;
	}
	__device__ const char *GpuVSystem::NextSystemCall(const char *name)
	{
		return nullptr;
	}

	__device__ static unsigned char _gpuVfsBuf[sizeof(GpuVSystem)];
	__device__ static GpuVSystem *_gpuVfs;
	__device__ RC VSystem::Initialize()
	{
		_gpuVfs = new (_gpuVfsBuf) GpuVSystem();
		_gpuVfs->SizeOsFile = sizeof(GpuVFile);
		_gpuVfs->MaxPathname = 260;
		_gpuVfs->Name = "gpu";
		RegisterVfs(_gpuVfs, true);
		return RC_OK; 
	}

	__device__ void VSystem::Shutdown()
	{
	}

#pragma endregion

}
#endif