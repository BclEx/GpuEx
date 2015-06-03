#include "Runtime.h"
#include "RuntimeOS.h"

__device__ void osSleep(DWORD milliseconds)
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
#endif
}

__device__ DWORD osGetLastError()
{
	return 0;	
}

__device__ DWORD osWaitForSingleObject(HANDLE hHandle, DWORD dwMilliseconds)
{
	return 0;
}

__device__ HANDLE osCreateMutexA(void *dummy1, bool dummy2, const char *name)
{
	return 0;
}

__device__ void osReleaseMutex(HANDLE h)
{
}