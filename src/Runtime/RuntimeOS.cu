#include "Runtime.h"
#include "RuntimeOS.h"

__device__ void _sleep(int milliseconds)
{
}

__device__ DWORD osGetLastError()
{
	return 0;	
}

__device__ DWORD osWaitForSingleObject(HANDLE h, int time)
{
	return 0;
}

__device__ HANDLE osCreateMutexA(void *dummy1, bool dummy2, const char *name)
{
	return 0;
}

__device__ void ReleaseMutex(HANDLE h)
{
}
