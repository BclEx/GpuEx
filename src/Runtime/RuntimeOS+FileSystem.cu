#include "Runtime.h"
#include "RuntimeOS.h"

__device__ HANDLE osCreateFileA(char *path, DWORD dwDesiredAccess, DWORD dwShareMode, DWORD dummy1, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, DWORD dummy2)
{
	return INVALID_HANDLE_VALUE;
}

__device__ DWORD osGetFileAttributesA(char *path)
{
	return 0;
}

__device__ DWORD osGetFileSize(HANDLE h, LPDWORD upper)
{
	*upper = 0;
	return 0;
}

__device__ DWORD osDeleteFileA(char *path)
{
	return 0;
}

__device__ bool osCloseHandle(HANDLE h)
{
	return true;
}

__device__ bool osReadFile(HANDLE h, void *buffer, DWORD amount, DWORD *read, OSOVERLAPPED *overlapped)
{
	return true;
}

__device__ bool osWriteFile(HANDLE h, const void *buffer, DWORD amount, DWORD *write, OSOVERLAPPED *overlapped)
{
	return true;
}

__device__ bool osSetFilePointer(HANDLE h, __int64 offset, int *reserved, int seekType)
{
	return true;
}

__device__ bool osSetEndOfFile(HANDLE h)
{
	return true;
}

__device__ bool osFlushFileBuffers(HANDLE h)
{
	return true;
}

//
__device__ HANDLE osCreateFileMappingA(HANDLE h, void *dummy1, int flags, void *dummy2, int length, const char *name)
{
	return 0;
}

__device__ HANDLE osMapViewOfFile(HANDLE h, int flags, void *dummy1, void *dummy2, int length)
{
	return 0;
}

__device__ void osUnmapViewOfFile(HANDLE h)
{
}