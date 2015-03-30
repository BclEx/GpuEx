// os_win.c
#define OS_GPU 1
#if OS_GPU
#include "Core.cu.h"
#include <new.h>

namespace Core
{
#pragma region Preamble

#if defined(TEST) || defined(_DEBUG)
	__device__ bool OsTrace = true;
#define OSTRACE(X, ...) if (OsTrace) { _dprintf(X, __VA_ARGS__); }
#else
#define OSTRACE(X, ...)
#endif

#define HANDLE int
#define DWORD unsigned long
#define INVALID_HANDLE_VALUE -1

#ifdef TEST
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

	// When testing, keep a count of the number of open files.
#ifdef TEST
	__device__ int g_open_file_count = 0;
#define OpenCounter(X) g_open_file_count += (X)
#else
#define OpenCounter(X)
#endif

#define MAX_PATH 100


#define INVALID_FILE_ATTRIBUTES ((DWORD)-1) 
#define GENERIC_READ                     (0x80000000L)
#define GENERIC_WRITE                    (0x40000000L)
#define GENERIC_EXECUTE                  (0x20000000L)
#define GENERIC_ALL                      (0x10000000L)

#define CREATE_NEW          1
#define CREATE_ALWAYS       2
#define OPEN_EXISTING       3
#define OPEN_ALWAYS         4
#define TRUNCATE_EXISTING   5
#define FILE_SHARE_READ                 0x00000001
#define FILE_SHARE_WRITE                0x00000002
#define FILE_SHARE_DELETE               0x00000004
#define FILE_ATTRIBUTE_READONLY             0x00000001
#define FILE_ATTRIBUTE_HIDDEN               0x00000002
#define FILE_ATTRIBUTE_SYSTEM               0x00000004
#define FILE_ATTRIBUTE_DIRECTORY            0x00000010
	//#define FILE_ATTRIBUTE_ARCHIVE              0x00000020
	//#define FILE_ATTRIBUTE_DEVICE               0x00000040
#define FILE_ATTRIBUTE_NORMAL               0x00000080
#define FILE_ATTRIBUTE_TEMPORARY            0x00000100
	//#define FILE_ATTRIBUTE_SPARSE_FILE          0x00000200
	//#define FILE_ATTRIBUTE_REPARSE_POINT        0x00000400
	//#define FILE_ATTRIBUTE_COMPRESSED           0x00000800
	//#define FILE_ATTRIBUTE_OFFLINE              0x00001000
	//#define FILE_ATTRIBUTE_NOT_CONTENT_INDEXED  0x00002000
	//#define FILE_ATTRIBUTE_ENCRYPTED            0x00004000
	//#define FILE_ATTRIBUTE_INTEGRITY_STREAM     0x00008000
	//#define FILE_ATTRIBUTE_VIRTUAL              0x00010000
	//#define FILE_ATTRIBUTE_NO_SCRUB_DATA        0x00020000
#define FILE_FLAG_DELETE_ON_CLOSE       0x04000000

#define NO_ERROR 0L // dderror

#pragma endregion

#pragma region GpuVFile

	// gpuFile
	class GpuVFile : public VFile
	{
	public:
		VSystem *Vfs;			// The VFS used to open this file
		HANDLE H;				// Handle for accessing the file
		LOCK Lock_;				// Type of lock currently held on this file
		DWORD LastErrno;		// The Windows errno from the last I/O error
		const char *Path;		// Full pathname of this file
		int SizeChunk;          // Chunk size configured by FCNTL_CHUNK_SIZE

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
			*directory = value;
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
		LPWSTR tempWide = NULL;
		dwLen = osFormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPWSTR)&tempWide, 0, 0);
		if (dwLen > 0)
		{
			// allocate a buffer and convert to UTF8
			_benignalloc_begin();
			out = UnicodeToUtf8(tempWide);
			_benignalloc_end();
		}
		if (!dwLen)
			_snprintf(buf, bufLength, "OsError 0x%x (%u)", lastErrno, lastErrno);
		else
		{
			// copy a maximum of nBuf chars to output buffer
			_snprintf(buf, bufLength, "%s", out);
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

#ifndef WIN32_IOERR_RETRY
#define WIN32_IOERR_RETRY 1
#endif
#ifndef WIN32_IOERR_RETRY_DELAY
#define WIN32_IOERR_RETRY_DELAY 10
#endif
	__device__ static int gpuIoerrRetry = WIN32_IOERR_RETRY;
	__device__ static int gpuIoerrRetryDelay = WIN32_IOERR_RETRY_DELAY;

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
			__sleep(gpuIoerrRetryDelay*(1+*retry));
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
			SysEx_LOG(RC_IOERR, "delayed %dms for lock/sharing conflict", win32IoerrRetryDelay*retry*(retry+1)/2);
	}

#pragma endregion

#pragma region GpuVFile

	__device__ RC GpuVFile::Close_()
	{
		OSTRACE("CLOSE %d\n", H);
		return RC_OK;
		//_assert(H != NULL && H != INVALID_HANDLE_VALUE);
		//int rc;
		//rc = osCloseHandle(H);
		//OSTRACE("CLOSE %d %s\n", H, rc ? "ok" : "failed");
		//if (rc)
		//	H = NULL;
		//return (rc ? RC_OK : gpuLogError(RC_IOERR_CLOSE, gpuGetLastError(), "gpuClose", Path));
	}

	__device__ RC GpuVFile::Read(void *buffer, int amount, int64 offset)
	{
		OSTRACE("READ %d lock=%d\n", H, Lock_);
		return RC_OK;
		//int retry = 0; // Number of retrys
		//DWORD read; // Number of bytes actually read from file
		//if (seekGpuFile(this, offset))
		//	return RC_FULL;
		//while (!gpuReadFile(H, buffer, amount, &read, 0))
		//{
		//	DWORD lastErrno;
		//	if (retryIoerr(&retry, &lastErrno)) continue;
		//	LastErrno = lastErrno;
		//	return winLogError(RC_IOERR_READ, LastErrno, "winRead", Path);
		//}
		//logIoerr(retry);
		//if (read < (DWORD)amount)
		//{
		//	// Unread parts of the buffer must be zero-filled
		//	memset(&((char *)buffer)[read], 0, amount - read);
		//	return RC_IOERR_SHORT_READ;
		//}
		//return RC_OK;
	}

	__device__ RC GpuVFile::Write(const void *buffer, int amount, int64 offset)
	{
		_assert(amount > 0);
		OSTRACE("WRITE %d lock=%d\n", H, Lock_);
		return RC_OK;
		//int rc = 0; // True if error has occurred, else false
		//int retry = 0; // Number of retries
		//{
		//	uint8 *remain = (uint8 *)buffer; // Data yet to be written
		//	int remainLength = amount; // Number of bytes yet to be written
		//	DWORD write; // Bytes written by each WriteFile() call
		//	DWORD lastErrno = NO_ERROR; // Value returned by GetLastError()
		//	while (remainLength > 0)
		//	{
		//		if (!osWriteFile(H, remain, remainLength, &write, 0)) {
		//			if (retryIoerr(&retry, &lastErrno)) continue;
		//			break;
		//		}
		//		_assert(write == 0 || write <= (DWORD)remainLength);
		//		if (write == 0 || write > (DWORD)remainLength)
		//		{
		//			lastErrno = osGetLastError();
		//			break;
		//		}
		//		remain += write;
		//		remainLength -= write;
		//	}
		//	if (remainLength > 0)
		//	{
		//		LastErrno = lastErrno;
		//		rc = 1;
		//	}
		//}
		//if (rc)
		//{
		//	if (LastErrno == ERROR_HANDLE_DISK_FULL ||  LastErrno == ERROR_DISK_FULL)
		//		return RC_FULL;
		//	return winLogError(RC_IOERR_WRITE, LastErrno, "winWrite", Path);
		//}
		//else
		//	logIoerr(retry);
		//return RC_OK;
	}

	__device__ RC GpuVFile::Truncate(int64 size)
	{
		OSTRACE("TRUNCATE %d %lld\n", H, size);
		return RC_OK;
		//RC rc = RC_OK;
		//// If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
		//// actual file size after the operation may be larger than the requested size).
		//if (SizeChunk > 0)
		//	size = ((size+SizeChunk-1)/SizeChunk)*SizeChunk;
		//// SetEndOfFile() returns non-zero when successful, or zero when it fails.
		//if (seekWinFile(this, size))
		//	rc = winLogError(RC_IOERR_TRUNCATE, LastErrno, "winTruncate1", Path);
		//else if (!osSetEndOfFile(H))
		//{
		//	LastErrno = osGetLastError();
		//	rc = winLogError(RC_IOERR_TRUNCATE, LastErrno, "winTruncate2", Path);
		//}
		//OSTRACE("TRUNCATE %d %lld %s\n", H, size, rc ? "failed" : "ok");
		//return rc;
	}

	__device__ RC GpuVFile::Sync(SYNC flags)
	{
		// Check that one of SQLITE_SYNC_NORMAL or FULL was passed
		_assert((flags&0x0F) == SYNC_NORMAL || (flags&0x0F) == SYNC_FULL);
		OSTRACE("SYNC %d lock=%d\n", H, Lock_);
		return RC_OK;
	}

	__device__ RC GpuVFile::get_FileSize(int64 &size)
	{
		return RC_OK;
		//RC rc = RC_OK;
		//FILE_STANDARD_INFO info;
		//if (osGetFileInformationByHandleEx(H, FileStandardInfo, &info, sizeof(info)))
		//	size = info.EndOfFile.QuadPart;
		//else
		//{
		//	LastErrno = osGetLastError();
		//	rc = winLogError(RC_IOERR_FSTAT, LastErrno, "winFileSize", Path);
		//}
		//return rc;
	}

	__device__ RC GpuVFile::Lock(LOCK lock)
	{
		return RC_OK;
	}

	__device__ RC GpuVFile::CheckReservedLock(int &lock)
	{
		return RC_OK;
	}

	__device__ RC GpuVFile::Unlock(LOCK lock)
	{
		return RC_OK;
	}

	__device__ RC GpuVFile::FileControl(FCNTL op, void *arg)
	{
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

	__device__ HANDLE osCreateFileA(void *converted, DWORD dwDesiredAccess, DWORD dwShareMode, DWORD dummy1, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, DWORD dummy2)
	{
		return INVALID_HANDLE_VALUE;
	}

#pragma region GpuVSystem

	__device__ static void *ConvertUtf8Filename(const char *name)
	{
		void *converted = nullptr;
		int length = _strlen30(name);
		converted = _alloc(length);
		_memcpy(converted, name, length);
		return converted;
	}

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
		int tempPathLength = _strlen30(tempPath);
		if ((tempPathLength + _strlen30(TEMP_FILE_PREFIX) + 18) >= bufLength)
			return RC_ERROR;
		size_t i;
		for (i = tempPathLength; i > 0 && tempPath[i-1] == '\\'; i--) { }
		tempPath[i] = 0;
		size_t j;
		__snprintf(buf, bufLength-18, (tempPathLength > 0 ? "%s\\"TEMP_FILE_PREFIX : TEMP_FILE_PREFIX, tempPath));
		j = _strlen30(buf);
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
		return false;
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
		_assert(type != OPEN_MAIN_DB || (flags & OPEN_URI) || utf8Name[_strlen30(utf8Name)+1] == 0);

		// Convert the filename to the system encoding.
		void *converted = ConvertUtf8Filename(utf8Name); // Filename in OS encoding
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
		if (isDelete)
			dwFlagsAndAttributes = FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE;
		else
			dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;

		HANDLE h;
		DWORD lastErrno = 0;
		int cnt = 0;
		while ((h = osCreateFileA(converted, dwDesiredAccess, dwShareMode, nullptr, dwCreationDisposition, dwFlagsAndAttributes, nullptr)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
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
		void *converted = ConvertUtf8Filename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		RC rc;
		DWORD lastErrno;
		int cnt = 0;
		do {
			attr = osGetFileAttributesA(converted);
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
			if (osDeleteFileA(converted))
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
		void *converted = ConvertUtf8Filename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		int rc = 0;
		DWORD lastErrno;
		attr = osGetFileAttributesA((char*)converted);
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
			_snprintf(full, MIN(fullLength, MaxPathname), "%s\\%s", g_data_directory, relative);
		else
			_snprintf(full, MIN(fullLength, MaxPathname), "%s", relative);
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
#if TEST
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
		_sleep((microseconds+999)/1000);
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