#include "../Core.cu.h"

namespace Core { namespace IO
{
	__device__ RC VFile::Lock(LOCK lock) { return RC_OK; }
	__device__ RC VFile::Unlock(LOCK lock) { return RC_OK; }
	__device__ RC VFile::CheckReservedLock(int &lock) { return RC_OK; }
	__device__ RC VFile::FileControl(FCNTL op, void *arg) { return RC_NOTFOUND; }

	__device__ uint VFile::get_SectorSize() { return 0; }
	__device__ VFile::IOCAP VFile::get_DeviceCharacteristics() { return (VFile::IOCAP)0; }

	__device__ RC VFile::ShmLock(int offset, int n, SHM flags) { return RC_OK; }
	__device__ void VFile::ShmBarrier() { }
	__device__ RC VFile::ShmUnmap(bool deleteFlag) { return RC_OK; }
	__device__ RC VFile::ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp) { return RC_OK; }
}}