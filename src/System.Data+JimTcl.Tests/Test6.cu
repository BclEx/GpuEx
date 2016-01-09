#include "Test.cu.h"

#ifndef OMIT_DISKIO  // This file is a no-op if disk I/O is disabled

//#define TRACE_CRASHTEST

typedef class CrashVFile CrashVFile;
typedef struct CrashGlobal CrashGlobal;
typedef struct WriteBuffer WriteBuffer;

// Method:
//
//   This layer is implemented as a wrapper around the "real" sqlite3_file object for the host system. Each time data is 
//   written to the file object, instead of being written to the underlying file, the write operation is stored in an in-memory 
//   structure (type WriteBuffer). This structure is placed at the end of a global ordered list (the write-list).
//
//   When data is read from a file object, the requested region is first retrieved from the real file. The write-list is then 
//   traversed and data copied from any overlapping WriteBuffer structures to the output buffer. i.e. a read() operation following
//   one or more write() operations works as expected, even if no data has actually been written out to the real file.
//
//   When a fsync() operation is performed, an operating system crash may be simulated, in which case exit(-1) is called (the call to 
//   xSync() never returns). Whether or not a crash is simulated, the data associated with a subset of the WriteBuffer structures 
//   stored in the write-list is written to the real underlying files and the entries removed from the write-list. If a crash is simulated,
//   a subset of the buffers may be corrupted before the data is written.
//
//   The exact subset of the write-list written and/or corrupted is determined by the simulated device characteristics and sector-size.
//
// "Normal" mode:
//
//   Normal mode is used when the simulated device has none of the SQLITE_IOCAP_XXX flags set.
//
//   In normal mode, if the fsync() is not a simulated crash, the write-list is traversed from beginning to end. Each WriteBuffer
//   structure associated with the file handle used to call xSync() is written to the real file and removed from the write-list.
//
//   If a crash is simulated, one of the following takes place for each WriteBuffer in the write-list, regardless of which 
//   file-handle it is associated with:
//
//     1. The buffer is correctly written to the file, just as if a crash were not being simulated.
//
//     2. Nothing is done.
//
//     3. Garbage data is written to all sectors of the file that overlap the region specified by the WriteBuffer. Or garbage
//        data is written to some contiguous section within the overlapped sectors.
//
// Device Characteristic flag handling:
//
//   If the IOCAP_ATOMIC flag is set, then option (3) above is never selected.
//
//   If the IOCAP_ATOMIC512 flag is set, and the WriteBuffer represents an aligned write() of an integer number of 512 byte regions, then
//   option (3) above is never selected. Instead, each 512 byte region is either correctly written or left completely untouched. Similar
//   logic governs the behavior if any of the other ATOMICXXX flags is set.
//
//   If either the IOCAP_SAFEAPPEND or IOCAP_SEQUENTIAL flags are set and a crash is being simulated, then an entry of the write-list is
//   selected at random. Everything in the list after the selected entry is discarded before processing begins.
//
//   If IOCAP_SEQUENTIAL is set and a crash is being simulated, option (1) is selected for all write-list entries except the last. If a 
//   crash is not being simulated, then all entries in the write-list that occur before at least one write() on the file-handle specified
//   as part of the xSync() are written to their associated real files.
//
//   If IOCAP_SAFEAPPEND is set and the first byte written by the write() operation is one byte past the current end of the file, then option
//   (1) is always selected.

// Each write operation in the write-list is represented by an instance of the following structure.
//
// If zBuf is 0, then this structure represents a call to xTruncate(), not xWrite(). In that case, iOffset is the size that the file is
// truncated to.

struct WriteBuffer
{
	int64 Offset;               // Byte offset of the start of this write()
	int BufLength;              // Number of bytes written
	uint8 *Buf;                 // Pointer to copy of written data
	CrashVFile *File;            // File this write() applies to

	WriteBuffer *Next;          // Next in CrashGlobal.pWriteList
};

class CrashVFile : public VFile
{
public:
	VFile *RealFile;				// Underlying "real" file handle
	char *Name;
	VSystem::OPEN Flags;            // Flags the file was opened with

	// Cache of the entire file. This is used to speed up OsRead() and OsFileSize() calls. Although both could be done by traversing the
	// write-list, in practice this is impractically slow.
	int Size;						// Size of file in bytes
	int DataLength;					// Size of buffer allocated at zData
	uint8 *Data;					// Buffer containing file contents
public:
	//: VFILE::Opened
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

	__device__ virtual RC ShmLock(int offset, int n, SHM flags);
	__device__ virtual void ShmBarrier();
	__device__ virtual RC ShmUnmap(bool deleteFlag);
	__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp);
};

struct CrashGlobal
{
	WriteBuffer *WriteList;     // Head of write-list
	WriteBuffer *WriteListEnd;  // End of write-list

	int SectorSize;             // Value of simulated sector size
	VFile::IOCAP DeviceCharacteristics;  // Value of simulated device characteristics

	int CrashAt;                // Crash on the iCrash'th call to xSync()
	char CrashFile[500];        // Crash during an xSync() on this file
};

static CrashGlobal _g = { nullptr, nullptr, DEFAULT_SECTOR_SIZE, (VFile::IOCAP)0, 0, 0 };

// Set this global variable to 1 to enable crash testing.
__device__ static bool _crashTestEnable = false;

__device__ static void *crash_malloc(int bytes)
{
	return (void *)Jim_Alloc((size_t)bytes);
}
__device__ static void crash_free(void *p)
{
	Jim_Free(p);
}
__device__ static void *crash_realloc(void *p, int n)
{
	return (void *)Jim_Realloc(p, (size_t)n);
}

// Wrapper around the sqlite3OsWrite() function that avoids writing to the 512 byte block begining at offset PENDING_BYTE.
__device__ static RC writeDbFile(CrashVFile *p, uint8 *z, int64 amount, int64 offset)
{
	RC rc = RC_OK;
	int skip = (offset == PENDING_BYTE && (p->Flags & VSystem::OPEN_MAIN_DB) ? 512 : 0);
	if ((amount-skip) > 0)
		rc = p->RealFile->Write(&z[skip], (int)(amount-skip), offset+skip);
	return rc;
}

// Flush the write-list as if xSync() had been called on file handle pFile. If isCrash is true, simulate a crash.
__device__ static RC writeListSync(CrashVFile *file, bool isCrash)
{
	RC rc = RC_OK;
	VFile::IOCAP dc = _g.DeviceCharacteristics;

	// If this is not a crash simulation, set pFinal to point to the last element of the write-list that is associated with file handle pFile.
	//
	// If this is a crash simulation, set pFinal to an arbitrarily selected element of the write-list.
	WriteBuffer *write;
	WriteBuffer *final_ = nullptr;
	if (!isCrash)
	{
		for (write = _g.WriteList; write; write = write->Next)
			if (write->File == file)
				final_ = write;
	}
	else if (dc & (VFile::IOCAP_SEQUENTIAL|VFile::IOCAP_SAFE_APPEND))
	{
		int writes = 0;
		for (write = _g.WriteList; write; write = write->Next) writes++;
		int finalId;
		SysEx::PutRandom(sizeof(int), &finalId);
		finalId = (finalId < 0?-1*finalId:finalId)%writes;
		for (write = _g.WriteList; finalId > 0; write = write->Next) finalId--;
		final_ = write;
	}

#ifdef TRACE_CRASHTEST
	_printf("Sync %s (is %s crash)\n", file->Name, (isCrash?"a":"not a"));
#endif

	WriteBuffer **ptr = &_g.WriteList;
	for (write = *ptr; rc == RC_OK && write; write = *ptr)
	{
		VFile *realFile = write->File->RealFile;
		// (eAction == 1)      -> write block out normally,
		// (eAction == 2)      -> do nothing,
		// (eAction == 3)      -> trash sectors.
		int eAction = 0;
		if (!isCrash)
			eAction = (write->File == file || dc & VFile::IOCAP_SEQUENTIAL ? 1 : 2);
		else
		{
			char random;
			SysEx::PutRandom(1, &random);
			// Do not select option 3 (sector trashing) if the IOCAP_ATOMIC flag is set or this is an OsTruncate(), not an Oswrite().
			if ((dc & VFile::IOCAP_ATOMIC) || !write->Buf)
				random &= 0x01;
			// If IOCAP_SEQUENTIAL is set and this is not the final entry in the truncated write-list, always select option 1 (write out correctly).
			if (dc & VFile::IOCAP_SEQUENTIAL && write != final_)
				random = 0;
			// If IOCAP_SAFE_APPEND is set and this OsWrite() operation is an append (first byte of the written region is 1 byte past the
			// current EOF), always select option 1 (write out correctly).
			if (dc & VFile::IOCAP_SAFE_APPEND && write->Buf)
			{
				int64 size;
				realFile->get_FileSize(size);
				if (size == write->Offset)
					random = 0;
			}
			if ((random&0x06) == 0x06)
				eAction = 3;
			else
				eAction = ((random & 0x01) ? 2 : 1);
		}

		switch (eAction)
		{
		case 1: { // Write out correctly
			if (write->Buf)
				rc = writeDbFile(write->File, write->Buf, write->BufLength, write->Offset);
			else
				rc = realFile->Truncate(write->Offset);
			*ptr = write->Next;
#ifdef TRACE_CRASHTEST
			if (isCrash)
				_printf("Writing %d bytes @ %d (%s)\n", write->BufLength, (int)write->Offset, write->File->Name);
#endif
			crash_free(write);
			break; }
		case 2: { // Do nothing
			ptr = &write->Next;
#ifdef TRACE_CRASHTEST
			if (isCrash)
				_printf("Omiting %d bytes @ %d (%s)\n", write->BufLength, (int)write->Offset, write->File->Name);
#endif
			break; }
		case 3: { // Trash sectors
			int firstId = (int)(write->Offset/_g.SectorSize);
			int lastId = (int)((write->Offset+write->BufLength-1)/_g.SectorSize);
			_assert(write->Buf);
#ifdef TRACE_CRASHTEST
			_printf("Trashing %d sectors @ %lld (sector %d) (%s)\n", 1+lastId-firstId, write->Offset, firstId, write->File->Name);
#endif
			uint8 *garbage = (uint8 *)crash_malloc(_g.SectorSize);
			if (garbage)
			{
				int64 i;
				for (i = firstId; rc == RC_OK && i <= lastId; i++)
				{
					SysEx::PutRandom(_g.SectorSize, garbage); 
					rc = writeDbFile(write->File, garbage, _g.SectorSize, i*_g.SectorSize);
				}
				crash_free(garbage);
			}
			else
				rc = RC_NOMEM;

			ptr = &write->Next;
			break; }
		default:
			_assert(!"Cannot happen");
		}
		if (write == final_) break;
	}

	if (rc == RC_OK && isCrash)
		exit(-1);

	for (write = _g.WriteList; write && write->Next; write = write->Next);
	_g.WriteListEnd = write;
	return rc;
}

// Add an entry to the end of the write-list.
__device__ static RC writeListAppend(VFile *file, int64 offset, const uint8 *buf, int bufLength)
{
	_assert((buf && bufLength) || (!bufLength && !buf));
	WriteBuffer *new_ = (WriteBuffer *)crash_malloc(sizeof(WriteBuffer) + bufLength);
	if (!new_)
		_fprintf(stderr, "out of memory in the crash simulator\n");
	_memset(new_, 0, sizeof(WriteBuffer) + bufLength);
	new_->Offset = offset;
	new_->BufLength = bufLength;
	new_->File = (CrashVFile *)file;
	if (buf)
	{
		new_->Buf = (uint8 *)&new_[1];
		_memcpy(new_->Buf, buf, bufLength);
	}
	if (_g.WriteList)
	{
		_assert(_g.WriteListEnd);
		_g.WriteListEnd->Next = new_;
	}
	else
		_g.WriteList = new_;
	_g.WriteListEnd = new_;
	return RC_OK;
}

// Close a crash-file.
__device__ RC CrashVFile::Close_()
{
	writeListSync(this, false);
	RealFile->Close();
	return RC_OK;
}

// Read data from a crash-file.
__device__ RC CrashVFile::Read(void *buffer, int amount, int64 offset)
{
	// Check the file-size to see if this is a short-read
	if (Size < (offset+amount))
		return RC_IOERR_SHORT_READ;
	_memcpy(buffer, &Data[offset], amount);
	return RC_OK;
}

// Write data to a crash-file.
__device__ RC CrashVFile::Write(const void *buffer, int amount, int64 offset)
{
	if (amount+offset > Size)
		Size = (int)(amount+offset);
	while (Size > DataLength)
	{
		int newLength = (DataLength*2) + 4096;
		uint8 *new_ = (uint8 *)crash_realloc(Data, newLength);
		if (!new_)
			return RC_NOMEM;
		_memset(&new_[DataLength], 0, newLength-DataLength);
		DataLength = newLength;
		Data = new_;
	}
	_memcpy(&Data[offset], buffer, amount);
	return writeListAppend(this, offset, (const uint8 *)buffer, amount);
}

// Truncate a crash-file.
__device__ RC CrashVFile::Truncate(int64 size)
{
	_assert(size >= 0);
	if (Size > size)
		Size = (int)size;
	return writeListAppend(this, size, nullptr, 0);
}

// Sync a crash-file.
__device__ RC CrashVFile::Sync(SYNC flags)
{
	bool isCrash = false;

	const char *name = Name;
	const char *crashFile = _g.CrashFile;
	int nameLength = (int)_strlen(name);
	int crashFileLength = (int)_strlen(crashFile);

	if (crashFileLength > 0 && crashFile[crashFileLength-1] == '*')
	{
		crashFileLength--;
		if (nameLength > crashFileLength) nameLength = crashFileLength;
	}

#ifdef TRACE_CRASHTEST
	_printf("cfSync(): nName = %d, nCrashFile = %d, zName = %s, zCrashFile = %s\n", nameLength, crashFileLength, name, crashFile);
#endif

	if (nameLength == crashFileLength && !_memcmp(name, crashFile, nameLength))
	{
#ifdef TRACE_CRASHTEST
		_printf("cfSync(): name matched, g.iCrash = %d\n", _g.CrashAt);
#endif
		if ((--_g.CrashAt) == 0) isCrash = true;
	}

	return writeListSync(this, isCrash);
}

// Return the current file-size of the crash-file.
__device__ RC CrashVFile::get_FileSize(int64 &size)
{
	size = Size;
	return RC_OK;
}

// Calls related to file-locks are passed on to the real file handle.
__device__ RC CrashVFile::Lock(LOCK lock)
{
	return RealFile->Lock(lock);
}
__device__ RC CrashVFile::Unlock(LOCK lock)
{
	return RealFile->Unlock(lock);
}
__device__ RC CrashVFile::CheckReservedLock(int &lock)
{
	return RealFile->CheckReservedLock(lock);
}
__device__ RC CrashVFile::FileControl(FCNTL op, void *arg)
{
	if (op == FCNTL_SIZE_HINT)
	{
		int64 bytes = *(int64 *)arg;
		if (bytes > Size)
			if (writeListAppend(this, bytes, nullptr, 0) == RC_OK)
				Size = (int)bytes;
		return RC_OK;
	}
	return RealFile->FileControl(op, arg);
}

// The xSectorSize() and xDeviceCharacteristics() functions return the global values configured by the [sqlite_crashparams] tcl interface.
__device__ uint CrashVFile::get_SectorSize()
{
	return _g.SectorSize;
}
__device__ VFile::IOCAP CrashVFile::get_DeviceCharacteristics()
{
	return _g.DeviceCharacteristics;
}

// Pass-throughs for WAL support.
__device__ RC CrashVFile::ShmLock(int offset, int n, SHM flags)
{
	return RealFile->ShmLock(offset, n, flags);
}
__device__ void CrashVFile::ShmBarrier()
{
	RealFile->ShmBarrier();
}
__device__ RC CrashVFile::ShmUnmap(bool deleteFlag)
{
	return RealFile->ShmUnmap(deleteFlag);
}
__device__ RC CrashVFile::ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp)
{
	return RealFile->ShmMap(region, sizeRegion, isWrite, pp);
}

// Application data for the crash VFS
class CrashVfs : public VSystem
{
public:
	VSystem *Orig;				// Wrapped vfs structure
public:
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
	//__device__ virtual RC CurrentTimeInt64(int64 *now);
	__device__ virtual RC CurrentTime(double *now);
	//__device__ virtual RC GetLastError(int bufLength, char *buf);
};

// Open a crash-file file handle.
//
// The caller will have allocated pVfs->szOsFile bytes of space at pFile. This file uses this space for the CrashVFile structure
// and allocates space for the "real" file structure using sqlite3_malloc(). The assumption here is (pVfs->szOsFile) is
// equal or greater than sizeof(CrashVFile).
__device__ RC CrashVfs::Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
{
	CrashVFile *wrapper = (CrashVFile *)file;
	VFile *real = (VFile *)&wrapper[1];

	_memset(wrapper, 0, sizeof(CrashVFile));
	RC rc = Orig->Open(path, real, flags, outFlags);

	if (rc == RC_OK)
	{
		wrapper->Opened = true;
		wrapper->Name = (char *)path;
		wrapper->RealFile = real;
		int64 size;
		rc = real->get_FileSize(size);
		wrapper->Size = (int)size;
		wrapper->Flags = flags;
	}
	if (rc == RC_OK)
	{
		wrapper->DataLength = (4096 + wrapper->Size);
		wrapper->Data = (uint8 *)crash_malloc(wrapper->DataLength);
		if (wrapper->Data)
		{
			// os_unix.c contains an assert() that fails if the caller attempts to read data from the 512-byte locking region of a file opened
			// with the SQLITE_OPEN_MAIN_DB flag. This region of a database file never contains valid data anyhow. So avoid doing such a read here.
			//
			// UPDATE: It also contains an assert() verifying that each call to the xRead() method reads less than 128KB of data.
			const int isDb = (flags & VSystem::OPEN_MAIN_DB);
			int64 offset;
			_memset(wrapper->Data, 0, wrapper->DataLength);
			for (offset = 0; offset < wrapper->Size; offset += 512)
			{
				int read = wrapper->Size - (int)offset;
				if (read > 512) read = 512;
				if (isDb && offset == PENDING_BYTE) continue;
				rc = real->Read(&wrapper->Data[offset], read, offset);
			}
		}
		else
			rc = RC_NOMEM;
	}
	if (rc != RC_OK && wrapper->Opened)
		file->Close();
	return rc;
}

__device__ RC CrashVfs::Delete(const char *path, bool syncDirectory)
{
	return Orig->Delete(path, syncDirectory);
}
__device__ RC CrashVfs::Access(const char *path, ACCESS flags, int *outRC)
{
	return Orig->Access(path, flags, outRC);
}
__device__ RC CrashVfs::FullPathname(const char *path, int pathOutLength, char *pathOut)
{
	return Orig->FullPathname(path, pathOutLength, pathOut);
}

__device__ void *CrashVfs::DlOpen(const char *filename)
{
	return Orig->DlOpen(filename);
}
__device__ void CrashVfs::DlError(int bufLength, char *buf)
{
	Orig->DlError(bufLength, buf);
}
__device__ void (*CrashVfs::DlSym(void *handle, const char *symbol))()
{
	return Orig->DlSym(handle, symbol);
}
__device__ void CrashVfs::DlClose(void *handle)
{
	Orig->DlClose(handle);
}

__device__ int CrashVfs::Randomness(int bufLength, char *buf)
{
	return Orig->Randomness(bufLength, buf);
}
__device__ int CrashVfs::Sleep(int microseconds)
{
	return Orig->Sleep(microseconds);
}
RC CrashVfs::CurrentTime(double *now)
{
	return Orig->CurrentTime(now);
}

__constant__ struct DeviceFlag
{
	char *Name;
	VFile::IOCAP Value;
} _flags[] = {
	{ "atomic",              VFile::IOCAP_ATOMIC                },
	{ "atomic512",           VFile::IOCAP_ATOMIC512             },
	{ "atomic1k",            VFile::IOCAP_ATOMIC1K              },
	{ "atomic2k",            VFile::IOCAP_ATOMIC2K              },
	{ "atomic4k",            VFile::IOCAP_ATOMIC4K              },
	{ "atomic8k",            VFile::IOCAP_ATOMIC8K              },
	{ "atomic16k",           VFile::IOCAP_ATOMIC16K             },
	{ "atomic32k",           VFile::IOCAP_ATOMIC32K             },
	{ "atomic64k",           VFile::IOCAP_ATOMIC64K             },
	{ "sequential",          VFile::IOCAP_SEQUENTIAL            },
	{ "safe_append",         VFile::IOCAP_SAFE_APPEND           },
	{ "powersafe_overwrite", VFile::IOCAP_POWERSAFE_OVERWRITE   },
	{ nullptr, (VFile::IOCAP)0 }
};
__device__ static int processDevSymArgs(Jim_Interp *interp, char argc, Jim_Obj *const args[], VFile::IOCAP *deviceCharOut, int *sectorSizeOut)
{
	int sectorSize = 0;
	int deviceChar = 0;
	bool setSectorsize = false;
	bool setDeviceChar = false;
	for (int i = 0; i < argc; i += 2)
	{
		int optLength;
		const char *opt = Jim_GetString(args[i], &optLength);
		if ((optLength > 11 || optLength < 2 || _strncmp("-sectorsize", opt, optLength)) && (optLength > 16 || optLength < 2 || _strncmp("-characteristics", opt, optLength)))
		{
			Jim_AppendResult(interp, "Bad option: \"", opt, "\" - must be \"-characteristics\" or \"-sectorsize\"", nullptr);
			return JIM_ERROR;
		}
		if (i == argc-1)
		{
			Jim_AppendResult(interp, "Option requires an argument: \"", opt, "\"", nullptr);
			return JIM_ERROR;
		}

		if (opt[1] == 's')
		{
			if (Jim_GetInt(interp, args[i+1], &sectorSize))
				return JIM_ERROR;
			setSectorsize = true;
		}
		else
		{
			Jim_Obj **objs;
			int objc;
			if (Jim_ListObjGetElements(interp, args[i+1], &objc, &objs))
				return JIM_ERROR;
			for (int j = 0; j < objc; j++)
			{
				Jim_Obj *flag = objs[j];
				Jim_IncrRefCount(flag);
				Jim_UtfToLower(Jim_String(flag));

				int choice;
				int rc = Jim_GetEnumFromObjStruct(interp, flag, (const void **)_flags, sizeof(_flags[0]), &choice, "no such flag", 0);
				if (rc)
					return JIM_ERROR;
				deviceChar |= _flags[choice].Value;
			}
			setDeviceChar = true;
		}
	}
	if (setDeviceChar)
		*deviceCharOut = (VFile::IOCAP)deviceChar;
	if (setSectorsize)
		*sectorSizeOut = sectorSize;
	return JIM_OK;
}

// tclcmd:   sqlite_crash_enable ENABLE
//
// Parameter ENABLE must be a boolean value. If true, then the "crash" vfs is added to the system. If false, it is removed.
__device__ CrashVfs _crashVfs;
__device__ static int crashEnableCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "ENABLE");
		return JIM_ERROR;
	}

	bool isEnable;
	if (Jim_GetBoolean(interp, args[1], &isEnable))
		return JIM_ERROR;

	if ((isEnable && _crashVfs.Orig) || (!isEnable && !_crashVfs.Orig))
		return JIM_OK;

	if (!_crashVfs.Orig)
	{
		VSystem *orig = VSystem::FindVfs(nullptr);
		_crashVfs.MaxPathname = orig->MaxPathname;
		_crashVfs.Orig = orig;
		_crashVfs.SizeOsFile = sizeof(CrashVFile) + orig->SizeOsFile;
		_crashVfs.Name = "crash";
		VSystem::RegisterVfs(&_crashVfs, false);
	}
	else
	{
		_crashVfs.Orig = nullptr;
		VSystem::UnregisterVfs(&_crashVfs);
	}
	return JIM_OK;
}

// tclcmd:   sqlite_crashparams ?OPTIONS? DELAY CRASHFILE
//
// This procedure implements a TCL command that enables crash testing in testfixture.  Once enabled, crash testing cannot be disabled.
//
// Available options are "-characteristics" and "-sectorsize". Both require an argument. For -sectorsize, this is the simulated sector size in
// bytes. For -characteristics, the argument must be a list of io-capability flags to simulate. Valid flags are "atomic", "atomic512", "atomic1K",
// "atomic2K", "atomic4K", "atomic8K", "atomic16K", "atomic32K", "atomic64K", "sequential" and "safe_append".
//
// Example:
//   sqlite_crashparams -sect 1024 -char {atomic sequential} ./test.db 1
__device__ static int crashParamsObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc < 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "?OPTIONS? DELAY CRASHFILE");
		goto error;
	}

	int crashFileLength;
	const char *crashFile = Jim_GetString(args[argc-1], &crashFileLength);
	if (crashFileLength >= sizeof(_g.CrashFile) ){
		Jim_AppendResult(interp, "Filename is too long: \"", crashFile, "\"", 0);
		goto error;
	}
	int delayAt;
	if (Jim_GetInt(interp, args[argc-2], &delayAt))
		goto error;

	VFile::IOCAP deviceChar = (VFile::IOCAP)-1;
	int sectorSize = -1;
	if (processDevSymArgs(interp, argc-3, &args[1], &deviceChar, &sectorSize))
		return JIM_ERROR;

	if (deviceChar >= 0)
		_g.DeviceCharacteristics = deviceChar;
	if (sectorSize >= 0)
		_g.SectorSize = sectorSize;

	_g.CrashAt = delayAt;
	_memcpy(_g.CrashFile, crashFile, crashFileLength+1);
	_crashTestEnable = true;
	return JIM_OK;

error:
	return JIM_ERROR;
}

__device__ extern void devsym_register(int deviceChar, int sectorSize);
__device__ static int devSymObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	VFile::IOCAP deviceChar = (VFile::IOCAP)-1;
	int sectorSize = -1;
	if (processDevSymArgs(interp, argc-1, &args[1], &deviceChar, &sectorSize))
		return JIM_ERROR;
	devsym_register(deviceChar, sectorSize);
	return JIM_OK;
}

// tclcmd: register_jt_vfs ?-default? PARENT-VFS
__device__ extern int jt_register(char *, int);
__device__ static int jtObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2 && argc != 3)
	{
		Jim_WrongNumArgs(interp, 1, args, "?-default? PARENT-VFS");
		return JIM_ERROR;
	}
	const char *parent = Jim_String(args[1]);
	if (argc == 3)
	{
		if (!_strcmp(parent, "-default"))
		{
			Jim_AppendResult(interp, "bad option \"", parent, "\": must be -default", nullptr);
			return JIM_ERROR;
		}
		parent = (char *)args[2];
	}
	if (!(*parent))
		parent = 0;
	if (jt_register((char *)parent, argc == 3))
	{
		Jim_AppendResult(interp, "Error in jt_register", nullptr);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// tclcmd: unregister_jt_vfs
__device__ extern void jt_unregister(void);
__device__ static int jtUnregisterObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "");
		return JIM_ERROR;
	}
	jt_unregister();
	return JIM_OK;
}

#endif

// This procedure registers the TCL procedures defined in this file.
__device__ int Sqlitetest6_Init(Jim_Interp *interp)
{
#ifndef OMIT_DISKIO
	Jim_CreateCommand(interp, "sqlite3_crash_enable", crashEnableCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_crashparams", crashParamsObjCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "sqlite3_simulate_device", devSymObjCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "register_jt_vfs", jtObjCmd, nullptr, nullptr);
	Jim_CreateCommand(interp, "unregister_jt_vfs", jtUnregisterObjCmd, nullptr, nullptr);
#endif
	return JIM_OK;
}
