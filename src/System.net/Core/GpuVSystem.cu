// os_win.c
#define OS_GPU 1
#if OS_GPU
#include <new.h>
#include "Core.cu.h"

namespace Core
{
#pragma region Preamble

#if defined(TEST) || defined(_DEBUG)
	__device__ bool OsTrace = true;
#define OSTRACE(X, ...) if (OsTrace) { _fprintf(nullptr, X, __VA_ARGS__); }
#else
#define OSTRACE(X, ...)
#endif

#define INVALID_HANDLE_VALUE -1

#pragma endregion

#pragma region GpuVFile

	// gpuFile
	class GpuVFile : public VFile
	{
	public:
		VSystem *Vfs;			// The VFS used to open this file
		int H;					// Handle for accessing the file
		LOCK Lock_;				// Type of lock currently held on this file
		int	LastErrno;			// The Windows errno from the last I/O error
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
		__device__ GpuVSystem() { }
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

#pragma region GpuVSystem

	__device__ VFile *GpuVSystem::_AttachFile(void *buffer)
	{
		return new (buffer) GpuVFile();
	}

	__device__ RC GpuVSystem::Open(const char *name, VFile *id, OPEN flags, OPEN *outFlags)
	{
		//		// 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
		//		// SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
		//		flags = (OPEN)((uint)flags & 0x87f7f);
		//
		//		RC rc = RC_OK;
		//		OPEN type = (OPEN)(flags & 0xFFFFFF00);  // Type of file to open
		//		bool isExclusive = (flags & OPEN_EXCLUSIVE);
		//		bool isDelete = (flags & OPEN_DELETEONCLOSE);
		//		bool isCreate = (flags & OPEN_CREATE);
		//		bool isReadonly = (flags & OPEN_READONLY);
		//		bool isReadWrite = (flags & OPEN_READWRITE);
		//		bool isOpenJournal = (isCreate && (type == OPEN_MASTER_JOURNAL || type == OPEN_MAIN_JOURNAL || type == OPEN_WAL));
		//
		//		// Check the following statements are true: 
		//		//
		//		//   (a) Exactly one of the READWRITE and READONLY flags must be set, and 
		//		//   (b) if CREATE is set, then READWRITE must also be set, and
		//		//   (c) if EXCLUSIVE is set, then CREATE must also be set.
		//		//   (d) if DELETEONCLOSE is set, then CREATE must also be set.
		//		_assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
		//		_assert(!isCreate || isReadWrite);
		//		_assert(!isExclusive || isCreate);
		//		_assert(!isDelete || isCreate);
		//
		//		// The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
		//		_assert((!isDelete && name) || type != OPEN_MAIN_DB);
		//		_assert((!isDelete && name) || type != OPEN_MAIN_JOURNAL);
		//		_assert((!isDelete && name) || type != OPEN_MASTER_JOURNAL);
		//		_assert((!isDelete && name) || type != OPEN_WAL);
		//
		//		// Assert that the upper layer has set one of the "file-type" flags.
		//		_assert(type == OPEN_MAIN_DB || type == OPEN_TEMP_DB ||
		//			type == OPEN_MAIN_JOURNAL || type == OPEN_TEMP_JOURNAL ||
		//			type == OPEN_SUBJOURNAL || type == OPEN_MASTER_JOURNAL ||
		//			type == OPEN_TRANSIENT_DB || type == OPEN_WAL);
		//
		//		GpuVFile *file = (GpuVFile *)id;
		//		_assert(file != nullptr);
		//		_memset(file, 0, sizeof(GpuVFile));
		//		file = new (file) GpuVFile();
		//		file->H = INVALID_HANDLE_VALUE;
		//
		//		// If the second argument to this function is NULL, generate a temporary file name to use 
		//		const char *utf8Name = name; // Filename in UTF-8 encoding
		//		//char tmpname[MAX_PATH+2];     // Buffer used to create temp filename
		//		//if (!utf8Name)
		//		//{
		//		//	_assert(isDelete && !isOpenJournal);
		//		//	_memset(tmpname, 0, MAX_PATH+2);
		//		//	rc = getTempname(MAX_PATH+2, tmpname);
		//		//	if (rc != RC_OK)
		//		//		return rc;
		//		//	utf8Name = tmpname;
		//		//}
		//
		//		// Database filenames are double-zero terminated if they are not URIs with parameters.  Hence, they can always be passed into
		//		// sqlite3_uri_parameter().
		//		_assert(type != OPEN_MAIN_DB || (flags & OPEN_URI) || utf8Name[_strlen30(utf8Name)+1]==0);
		//
		//		// Convert the filename to the system encoding.
		//		void *converted = ConvertUtf8Filename(utf8Name); // Filename in OS encoding
		//		if (!converted)
		//			return RC_IOERR_NOMEM;
		//
		//		if (winIsDir(converted))
		//		{
		//			_free(converted);
		//			return RC_CANTOPEN_ISDIR;
		//		}
		//
		//		DWORD dwDesiredAccess;
		//		if (isReadWrite)
		//			dwDesiredAccess = GENERIC_READ | GENERIC_WRITE;
		//		else
		//			dwDesiredAccess = GENERIC_READ;
		//
		//		// SQLITE_OPEN_EXCLUSIVE is used to make sure that a new file is created. SQLite doesn't use it to indicate "exclusive access"
		//		// as it is usually understood.
		//		DWORD dwCreationDisposition;
		//		if (isExclusive) // Creates a new file, only if it does not already exist. If the file exists, it fails.
		//			dwCreationDisposition = CREATE_NEW;
		//		else if (isCreate) // Open existing file, or create if it doesn't exist
		//			dwCreationDisposition = OPEN_ALWAYS;
		//		else // Opens a file, only if it exists.
		//			dwCreationDisposition = OPEN_EXISTING;
		//
		//		DWORD dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
		//
		//		DWORD dwFlagsAndAttributes = 0;
		//		if (isDelete)
		//			dwFlagsAndAttributes = FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE;
		//		else
		//			dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
		//
		//		HANDLE h;
		//		DWORD lastErrno;
		//		int cnt = 0;
		//		if (isNT())
		//			while ((h = osCreateFileW((LPCWSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
		//#ifdef WIN32_HAS_ANSI
		//		else
		//			while ((h = osCreateFileA((LPCSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
		//#endif
		//			logIoerr(cnt);
		//
		//			OSTRACE("OPEN %d %s 0x%lx %s\n", h, name, dwDesiredAccess, h == INVALID_HANDLE_VALUE ? "failed" : "ok");
		//			if (h == INVALID_HANDLE_VALUE)
		//			{
		//				file->LastErrno = lastErrno;
		//				winLogError(RC_CANTOPEN, file->LastErrno, "winOpen", utf8Name);
		//				_free(converted);
		//				if (isReadWrite && !isExclusive)
		//					return Open(name, id, (OPEN)((flags|OPEN_READONLY) & ~(OPEN_CREATE|OPEN_READWRITE)), outFlags);
		//				else
		//					return SysEx_CANTOPEN_BKPT;
		//			}
		//
		//			if (outFlags)
		//				*outFlags = (isReadWrite ? OPEN_READWRITE : OPEN_READONLY);
		//			_free(converted);
		//			file->Opened = true;
		//			file->Vfs = this;
		//			file->H = h;
		//			//if (sqlite3_uri_boolean(name, "psow", POWERSAFE_OVERWRITE))
		//			//	file->CtrlFlags |= WinVFile::WINFILE_PSOW;
		//			file->LastErrno = NO_ERROR;
		//			file->Path = name;
		//			OpenCounter(+1);
		//			return rc;
		return RC_ERROR;
	}

	__device__ RC GpuVSystem::Delete(const char *filename, bool syncDir)
	{
		return RC_ERROR;
	}

	__device__ RC GpuVSystem::Access(const char *filename, ACCESS flags, int *resOut)
	{
		return RC_ERROR;
	}

	__device__ RC GpuVSystem::FullPathname(const char *relative, int fullLength, char *full)
	{
		return RC_ERROR;
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
		return 0;
	}

	__device__ int GpuVSystem::Sleep(int microseconds)
	{
		return 0;
	}

	__device__ RC GpuVSystem::CurrentTimeInt64(int64 *now)
	{
		return RC_ERROR;
	}

	__device__ RC GpuVSystem::CurrentTime(double *now)
	{
		return RC_ERROR;
	}

	__device__ RC GpuVSystem::GetLastError(int bufLength, char *buf)
	{
		return RC_ERROR;
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

	__device__ static char _gpuVfsBuf[sizeof(GpuVSystem)];
	__device__ static GpuVSystem *_gpuVfs;
	__device__ RC VSystem::Initialize()
	{
		_gpuVfs = new (_gpuVfsBuf) GpuVSystem();
		_gpuVfs->SizeOsFile = 0;
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