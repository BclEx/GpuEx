#include "..\System.net\Core\Core.cu.h"

__device__ static void TestVFS()
{
}

// NATIVE: assert
__global__ static void testSystemIO0(void *r)
{
	_runtimeSetHeap(r);
	MutexEx masterMutex;
	RC rc = SysEx::PreInitialize(masterMutex);
	SysEx::PostInitialize(masterMutex);
	//
	TestVFS();
	//
	SysEx::Shutdown();
	printf("System.IO: 0\n");
}

#if __CUDACC__
void __testSystemIO(cudaDeviceHeap &r)
{
	testSystemIO0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystemIO(cudaDeviceHeap &r)
{
	testSystemIO0(r.heap);
}
#endif