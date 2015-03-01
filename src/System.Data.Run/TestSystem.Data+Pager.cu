#include "..\System.Data.net\Core+Vdbe\Core+Vdbe.cu.h"

// NATIVE: assert
__global__ static void testSystemDataPager0(void *r)
{
	_runtimeSetHeap(r);
	Main::Initialize();
	//
	//TestDB();
	//
	Main::Shutdown();
	printf("System.Data+Pager: 0\n");
}

#if __CUDACC__
void __testSystemDataPager(cudaDeviceHeap &r)
{
	testSystemDataPager0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystemDataPager(cudaDeviceHeap &r)
{
	testSystemDataPager0(r.heap);
}
#endif