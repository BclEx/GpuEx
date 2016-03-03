#pragma warning(disable: 4996)
#ifndef _LIB

#include <stdio.h>
#include "RuntimeHost.h"
#include "Runtime.h"
//#include "..\Runtime.Tests\TestRuntime.cu"

__global__ static void runtime13(void *r)
{
	FILE *f = _fopen("C:\\T_\\fopen.txt", "w");
	_fprintfR(f, "The quick brown fox jumps over the lazy dog");
	_fcloseR(f);
}

cudaDeviceHeap _deviceHeap;
int main(int argc, char **argv)
{
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	//cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*4));
	_deviceHeap = cudaDeviceHeapCreate(256, 100);
	RuntimeSentinel::ServerInitialize();

	//__testRuntime(_deviceHeap);
	runtime13<<<1, 1>>>(_deviceHeap.heap); cudaDeviceHeapSynchronize(_deviceHeap);

	RuntimeSentinel::ServerShutdown();
	cudaDeviceHeapDestroy(_deviceHeap);
	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}

#endif