#include <RuntimeHost.h>
#include <RuntimeEx.h>

// NATIVE: assert
__global__ static void runtime0(void *r)
{
	_runtimeSetHeap(r);
	_assert(true);
	printf("Example: 0\n");
}


#if __CUDACC__
void __testRuntimeEx(cudaDeviceHeap &r)
{
	RuntimeSentinel::ServerInitialize();
	runtime0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	RuntimeSentinel::ServerShutdown();
}
#else
void __testRuntimeEx(cudaDeviceHeap &r)
{
#if OS_MAP
	RuntimeSentinel::ServerInitialize();
#endif
	runtime0(r.heap);
#if OS_MAP
	RuntimeSentinel::ServerShutdown();
#endif
}
#endif