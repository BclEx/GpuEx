#include <RuntimeHost.h>
#include "..\System.net\Core\Core.cu.h"

__device__ static void TestVFS()
{
	auto vfs = VSystem::FindVfs(nullptr);
	auto file = (VFile *)_allocZero(vfs->SizeOsFile);
	auto rc = vfs->Open("\\T_\\Test.db", file, (VSystem::OPEN)(VSystem::OPEN_CREATE|VSystem::OPEN_READWRITE|VSystem::OPEN_MAIN_DB), nullptr);
	file->Write4(0, 123145);
	file->Close();
}

// NATIVE: assert
__global__ static void testSystem0(void *r)
{
	_runtimeSetHeap(r);
	MutexEx masterMutex;
	RC rc = SysEx::PreInitialize(masterMutex);
	SysEx::PostInitialize(masterMutex);
	//
	TestVFS();
	//
	SysEx::Shutdown();
	printf("System: 0\n");
}

#if __CUDACC__
void __testSystem(cudaDeviceHeap &r)
{
	testSystem0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystem(cudaDeviceHeap &r)
{
	testSystem0(r.heap);
}
#endif