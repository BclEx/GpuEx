#include "Runtime.h"
#include "RuntimeOS.h"

#if OS_GPU
__device__ void _sleep(int milliseconds)
{
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;)
	{
		clock_t now = clock();
		clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
		if (cycles >= end) break;
	}
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
#endif