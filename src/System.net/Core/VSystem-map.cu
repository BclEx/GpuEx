﻿#include "VSystemSentinel.cu.h"
#include <new.h>

#if OS_MAP
namespace Core
{

#pragma region Preamble

	//#if defined(_TEST) || defined(_DEBUG)
	//	__device__ bool OsTrace = true;
	//#define OSTRACE(X, ...) if (OsTrace) { _dprintf(X, __VA_ARGS__); }
	//#else
	//#define OSTRACE(X, ...)
	//#endif

#pragma endregion

#pragma region MapVFile

	class MapVFile : public VFile
	{
	public:
		VSystem *Vfs;	// The VFS used to open this file
		VFile *F;		// The native VFile
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

#pragma region MapVSystem

	class MapVSystem : public VSystem
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

#pragma region MapVFile

	__device__ RC MapVFile::Close_()
	{
		Messages::File_Close msg(F);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		Opened = false;
		return msg.RC;
	}

	__device__ RC MapVFile::Read(void *buffer, int amount, int64 offset)
	{
		Messages::File_Read msg(F, amount, offset);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		_memcpy(buffer, msg.Buffer, amount);
		return msg.RC;
	}

	__device__ RC MapVFile::Write(const void *buffer, int amount, int64 offset)
	{
		Messages::File_Write msg(F, buffer, amount, offset);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVFile::Truncate(int64 size)
	{
		Messages::File_Truncate msg(F, size);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVFile::Sync(SYNC flags)
	{
		Messages::File_Sync msg(F, flags);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVFile::get_FileSize(int64 &size)
	{
		Messages::File_get_FileSize msg(F);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		size = msg.Size;
		return msg.RC;
	}

	__device__ RC MapVFile::Lock(LOCK lock)
	{
		Messages::File_Lock msg(F, lock);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVFile::CheckReservedLock(int &lock)
	{
		Messages::File_CheckReservedLock msg(F);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		lock = msg.Lock;
		return msg.RC;
	}

	__device__ RC MapVFile::Unlock(LOCK lock)
	{
		Messages::File_Unlock msg(F, lock);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVFile::FileControl(FCNTL op, void *arg)
	{
		return RC_NOTFOUND;
	}

	__device__ uint MapVFile::get_SectorSize()
	{
		return 512;
	}

	__device__ VFile::IOCAP MapVFile::get_DeviceCharacteristics()
	{
		return (VFile::IOCAP)0;
	}

#pragma endregion

#pragma region MapVSystem

	__device__ VFile *MapVSystem::_AttachFile(void *buffer)
	{
		return new (buffer) MapVFile();
	}

	__device__ RC MapVSystem::Open(const char *name, VFile *id, OPEN flags, OPEN *outFlags)
	{
		// 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
		// SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
		flags = (OPEN)((uint)flags & 0x87f7f);

		OPEN type = (OPEN)(flags & 0xFFFFFF00);  // Type of file to open
		bool isExclusive = ((flags & OPEN_EXCLUSIVE) != 0);
		bool isDelete = ((flags & OPEN_DELETEONCLOSE) != 0);
		bool isCreate = ((flags & OPEN_CREATE) != 0);
		bool isReadonly = ((flags & OPEN_READONLY) != 0);
		bool isReadWrite = ((flags & OPEN_READWRITE) != 0);
		//bool isOpenJournal = (isCreate && (type == OPEN_MASTER_JOURNAL || type == OPEN_MAIN_JOURNAL || type == OPEN_WAL));

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

		MapVFile *file = (MapVFile *)id;
		_assert(file != nullptr);
		_memset(file, 0, sizeof(MapVFile));
		file = new (file) MapVFile();
		//
		Messages::System_Open msg(name, flags);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		if (outFlags)
			*outFlags = msg.OutFlags;
		file->Opened = true;
		file->Vfs = this;
		file->F = msg.F;
		return msg.RC;
	}

	__device__ RC MapVSystem::Delete(const char *filename, bool syncDir)
	{
		Messages::System_Delete msg(filename, syncDir);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		return msg.RC;
	}

	__device__ RC MapVSystem::Access(const char *filename, ACCESS flags, int *resOut)
	{
		Messages::System_Access msg(filename, flags);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		*resOut = msg.ResOut;
		return msg.RC;
	}

	__device__ RC MapVSystem::FullPathname(const char *relative, int fullLength, char *full)
	{
		Messages::System_FullPathname msg(relative, fullLength);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		full = _mprintf("%s", msg.Full);
		return msg.RC;
	}

#ifndef OMIT_LOAD_EXTENSION
	__device__ void *MapVSystem::DlOpen(const char *filename)
	{
		return nullptr;
	}

	__device__ void MapVSystem::DlError(int bufLength, char *buf)
	{
	}

	__device__ void (*MapVSystem::DlSym(void *handle, const char *symbol))()
	{
		return nullptr;
	}

	__device__ void MapVSystem::DlClose(void *handle)
	{
	}
#else
#define winDlOpen  0
#define winDlError 0
#define winDlSym   0
#define winDlClose 0
#endif

	__device__ int MapVSystem::Randomness(int bufLength, char *buf)
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

	__device__ int MapVSystem::Sleep(int milliseconds)
	{
#if __CUDACC__
		clock_t start = clock();
		clock_t end = milliseconds * 10;
		for (;;)
		{
			clock_t now = clock();
			clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
			if (cycles >= end) break;
		}
		return ((milliseconds+999)/1000)*1000;
#else
		return 0;
#endif
	}

	__device__ RC MapVSystem::CurrentTimeInt64(int64 *now)
	{
#if __CUDACC__
		*now = clock();
#endif
		return RC_OK;
	}

	__device__ RC MapVSystem::CurrentTime(double *now)
	{
#if __CUDACC__
		int64 i;
		RC rc = CurrentTimeInt64(&i);
		if (rc == RC_OK)
			*now = i/86400000.0;
		return rc;
#else
		return RC_OK;
#endif
	}

	__device__ RC MapVSystem::GetLastError(int bufLength, char *buf)
	{
		Messages::System_GetLastError msg(bufLength);
		RuntimeSentinel::Send(&msg, sizeof(msg));
		buf = _mprintf("%", msg.Buf);
		return msg.RC;
	}

	__device__ RC MapVSystem::SetSystemCall(const char *name, syscall_ptr newFunc)
	{
		return RC_ERROR;
	}
	__device__ syscall_ptr MapVSystem::GetSystemCall(const char *name)
	{
		return nullptr;
	}
	__device__ const char *MapVSystem::NextSystemCall(const char *name)
	{
		return nullptr;
	}

	__device__ static unsigned char _mapVfsBuf[sizeof(MapVSystem)];
	__device__ static MapVSystem *_mapVfs;
#ifdef _CPU
	__device__ RC MapVSystem_Initialize()
	{
		_mapVfs = new (_mapVfsBuf) MapVSystem();
		_mapVfs->SizeOsFile = sizeof(MapVFile);
		_mapVfs->MaxPathname = 260;
		_mapVfs->Name = "map";
		VSystem::RegisterVfs(_mapVfs, true);
		return RC_OK; 
	}
#else
	__device__ RC VSystem::Initialize()
	{
		_mapVfs = new (_mapVfsBuf) MapVSystem();
		_mapVfs->SizeOsFile = sizeof(MapVFile);
		_mapVfs->MaxPathname = 260;
		_mapVfs->Name = "map";
		RegisterVfs(_mapVfs, true);
		return RC_OK; 
	}

	__device__ void VSystem::Shutdown()
	{
	}
#endif

#pragma endregion

}
#endif