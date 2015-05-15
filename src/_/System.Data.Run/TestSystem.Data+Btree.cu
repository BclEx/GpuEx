#include "..\System.Data.net\Core+Vdbe\Core+Vdbe.cu.h"

// NATIVE: assert
__global__ static void testSystemDataBtree0(void *r)
{
	_runtimeSetHeap(r);
	Main::Initialize();
	//
	//TestDB();
	//
	Main::Shutdown();
	printf("System.Data+Btree: 0\n");
}

#if __CUDACC__
void __testSystemDataBtree(cudaDeviceHeap &r)
{
	testSystemDataBtree0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystemDataBtree(cudaDeviceHeap &r)
{
	testSystemDataBtree0(r.heap);
}
#endif