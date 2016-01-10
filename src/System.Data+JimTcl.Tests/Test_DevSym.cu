// This file contains code that modified the OS layer in order to simulate different device types (by overriding the return values of the  xDeviceCharacteristics() and xSectorSize() methods).
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <new.h>

// Maximum pathname length supported by the devsym backend.
#define DEVSYM_MAX_PATHNAME 512
// Name used to identify this VFS.
#define DEVSYM_VFS_NAME "devsym"

struct DevsymGlobal
{
	VSystem *Vfs;
	VFile::IOCAP DeviceChar;
	int SectorSize;
};
__device__ struct DevsymGlobal _g = { nullptr, (VFile::IOCAP)0, 512 };

class DevSymVFile : public VFile
{
public:
	VFile *Real; // The "real" underlying file descriptor
public:
	__device__ virtual RC Close_() { return Real->Close(); }
	__device__ virtual RC Read(void *buffer, int amount, int64 offset) { return Real->Read(buffer, amount, offset); }
	__device__ virtual RC Write(const void *buffer, int amount, int64 offset) { return Real->Write(buffer, amount, offset); }
	__device__ virtual RC Truncate(int64 size) { return Real->Truncate(size); }
	__device__ virtual RC Sync(SYNC flags) { return Real->Sync(flags); }
	__device__ virtual RC get_FileSize(int64 &size) { return Real->get_FileSize(size); }

	__device__ virtual RC Lock(LOCK lock) { return Real->Lock(lock); }
	__device__ virtual RC Unlock(LOCK lock) { return Real->Unlock(lock); }
	__device__ virtual RC CheckReservedLock(int &lock) { return Real->CheckReservedLock(lock); }
	__device__ virtual RC FileControl(FCNTL op, void *arg) { return Real->FileControl(op, arg); }

	__device__ virtual uint get_SectorSize() { return _g.SectorSize; }
	__device__ virtual IOCAP get_DeviceCharacteristics() { return _g.DeviceChar; }

	__device__ virtual RC ShmLock(int offset, int n, SHM flags) { return Real->ShmLock(offset, n, flags); }
	__device__ virtual void ShmBarrier() { Real->ShmBarrier(); }
	__device__ virtual RC ShmUnmap(bool deleteFlag) { return Real->ShmUnmap(deleteFlag); }
	__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp) { return Real->ShmMap(region, sizeRegion, isWrite, pp); }
};

class DevSymVSystem : public VSystem
{
public:
	VFile *Real; // The "real" underlying file descriptor
public:
	__device__ virtual VFile *_AttachFile(void *buffer) { return new (buffer) DevSymVFile(); }
	__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
	{
		DevSymVFile *p = (DevSymVFile *)file;
		p->Real = (VFile *)&p[1];
		RC rc = _g.Vfs->Open(path, p->Real, flags, outFlags);
		p->Opened = p->Real->Opened;
		return rc;
	}
	__device__ virtual RC Delete(const char *path, bool syncDirectory) { return _g.Vfs->Delete(path, syncDirectory); }
	__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC) { return _g.Vfs->Access(path, flags, outRC); }
	__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut) { return _g.Vfs->FullPathname(path, pathOutLength, pathOut); }

#ifndef OMIT_LOAD_EXTENSION
	__device__ virtual void *DlOpen(const char *filename) { return _g.Vfs->DlOpen(filename); }
	__device__ virtual void DlError(int bufLength, char *buf) { return _g.Vfs->DlError(bufLength, buf); }
	__device__ virtual void (*DlSym(void *handle, const char *symbol))() { return _g.Vfs->DlSym(handle, symbol); }
	__device__ virtual void DlClose(void *handle) { return _g.Vfs->DlClose(handle); }
#endif
	__device__ virtual int Randomness(int bufLength, char *buf) { return _g.Vfs->Randomness(bufLength, buf); }
	__device__ virtual int Sleep(int microseconds) { return _g.Vfs->Sleep(microseconds); }
	//__device__ virtual RC CurrentTimeInt64(int64 *now);
	__device__ virtual RC CurrentTime(double *now) { return _g.Vfs->CurrentTime(now); }
	//__device__ virtual RC GetLastError(int bufLength, char *buf);
};

// This procedure registers the devsym vfs with SQLite. If the argument is true, the devsym vfs becomes the new default vfs. It is the only publicly
// available function in this file.
__device__ static unsigned char _devsymVfsBuf[sizeof(DevSymVSystem)];
__device__ static DevSymVSystem *_devsymVfs;
__device__ void devsym_register(VFile::IOCAP deviceChar, int sectorSize)
{
	if (!_g.Vfs)
	{
		_g.Vfs = VSystem::FindVfs(nullptr);
		_devsymVfs = new (_devsymVfsBuf) DevSymVSystem();
		_devsymVfs->SizeOsFile = sizeof(DevSymVFile) + _g.Vfs->SizeOsFile;
		_devsymVfs->MaxPathname = DEVSYM_MAX_PATHNAME;
		_devsymVfs->Name = DEVSYM_VFS_NAME;
		VSystem::RegisterVfs(_devsymVfs, false);
	}
	_g.DeviceChar = ((int)deviceChar >= 0 ? deviceChar : (VFile::IOCAP)0);
	_g.SectorSize = (sectorSize >= 0 ? sectorSize : 512);
}
