#ifndef __RUNTIMEOS_H__
#define __RUNTIMEOS_H__

//////////////////////
// OPERATING SYSTEM
#pragma region OPERATING SYSTEM

//#define DWORD unsigned long
typedef unsigned long DWORD;
typedef DWORD *LPDWORD;

#define NULL 0 // ((void *)0)

typedef void *HANDLE;
typedef HANDLE *LPHANDLE;
#ifndef INVALID_HANDLE_VALUE
#define INVALID_HANDLE_VALUE ((HANDLE)(__int64)-1)
#endif

#define ERROR_SUCCESS                    0L
#define NO_ERROR						 0L // dderror
#define ERROR_INVALID_FUNCTION           1L // dderror
#define ERROR_FILE_NOT_FOUND             2L
#define ERROR_PATH_NOT_FOUND             3L
#define ERROR_TOO_MANY_OPEN_FILES        4L
#define ERROR_ACCESS_DENIED              5L
#define ERROR_INVALID_HANDLE             6L
#define ERROR_SHARING_VIOLATION          32L
#define ERROR_LOCK_VIOLATION             33L
#define ERROR_HANDLE_EOF                 38L
#define ERROR_HANDLE_DISK_FULL           39L
#define ERROR_DISK_FULL                  112L
#define ERROR_NOT_LOCKED                 158L
#define ERROR_ALREADY_EXISTS             183L

//typedef struct
//{
//	unsigned __int64 *Internal;
//	unsigned __int64 *InternalHigh;
//	union
//	{
//		struct
//		{
//			DWORD Offset;
//			DWORD OffsetHigh;
//		};
//		void *Pointer;
//	};
//	HANDLE hEvent;
//} OSOVERLAPPED;

//__device__ void osSleep(DWORD milliseconds);
//__device__ DWORD osGetLastError();
//__device__ DWORD osWaitForSingleObject(HANDLE handle, DWORD milliseconds);
//__device__ HANDLE osCreateMutexA(void *dummy1, bool initialOwner, const char *name);
//__device__ void osReleaseMutex(HANDLE h);

#pragma endregion

//////////////////////
// FILE SYSTEM
#pragma region FILE SYSTEM

#define FILE_BEGIN           0
#define FILE_CURRENT         1
#define FILE_END             2

#ifndef WAIT_FAILED
#define WAIT_FAILED			((DWORD)0xFFFFFFFF)
#define WAIT_OBJECT_0       ((DWORD)0x00000000L)
#define WAIT_ABANDONED      ((DWORD)0x00000080L)
#endif
#define INFINITE 0xFFFFFFFF  // Infinite timeout

#define GENERIC_READ                     (0x80000000L)
#define GENERIC_WRITE                    (0x40000000L)
#define GENERIC_EXECUTE                  (0x20000000L)
#define GENERIC_ALL                      (0x10000000L)

#define CREATE_NEW          1
#define CREATE_ALWAYS       2
#define OPEN_EXISTING       3
#define OPEN_ALWAYS         4
#define TRUNCATE_EXISTING   5

#define INVALID_FILE_SIZE ((DWORD)0xFFFFFFFF)
#define INVALID_SET_FILE_POINTER ((DWORD)-1)
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)

#define FILE_SHARE_READ                 0x00000001
#define FILE_SHARE_WRITE                0x00000002
#define FILE_SHARE_DELETE               0x00000004

#define FILE_ATTRIBUTE_READONLY             0x00000001
#define FILE_ATTRIBUTE_HIDDEN               0x00000002
#define FILE_ATTRIBUTE_SYSTEM               0x00000004
#define FILE_ATTRIBUTE_DIRECTORY            0x00000010
#define FILE_ATTRIBUTE_NORMAL               0x00000080
#define FILE_ATTRIBUTE_TEMPORARY            0x00000100
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)

#define FILE_FLAG_DELETE_ON_CLOSE       0x04000000

#define PAGE_NOACCESS          0x01     
#define PAGE_READONLY          0x02     
#define PAGE_READWRITE         0x04     
#define PAGE_WRITECOPY         0x08     

#ifndef FILE_MAP_WRITE
#define FILE_MAP_WRITE      0x0002
#define FILE_MAP_READ       0x0004
#define FILE_MAP_ALL_ACCESS FILE_MAP_WRITE|FILE_MAP_READ
#endif

//__device__ HANDLE osCreateFileA(char *path, DWORD dwDesiredAccess, DWORD dwShareMode, DWORD dummy1, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, DWORD dummy2);
//__device__ DWORD osGetFileAttributesA(char *path);
//__device__ DWORD osGetFileSize(HANDLE h, LPDWORD upper);
//__device__ DWORD osDeleteFileA(char *path);
//__device__ bool osCloseHandle(HANDLE h);
//__device__ bool osReadFile(HANDLE h, void *buffer, DWORD amount, LPDWORD read, OSOVERLAPPED *overlapped);
//__device__ bool osWriteFile(HANDLE h, const void *buffer, DWORD amount, LPDWORD write, OSOVERLAPPED *overlapped);
//__device__ bool osSetFilePointer(HANDLE h, __int64 offset, int *reserved, int seekType);
//__device__ bool osSetEndOfFile(HANDLE h);
//__device__ bool osFlushFileBuffers(HANDLE h);
//
////
//__device__ HANDLE osCreateFileMappingA(HANDLE h, void *dummy1, int flags, void *dummy2, int length, const char *name);
//__device__ HANDLE osMapViewOfFile(HANDLE h, int flags, void *dummy1, void *dummy2, int length);
//__device__ void osUnmapViewOfFile(HANDLE h);

#pragma endregion

//////////////////////
// DIR ENT
#pragma region DIR ENT

typedef struct DIR DIR;

struct dirent
{
	char *d_name;
};

__device__ DIR *_opendir(const char *);
__device__ int _closedir(DIR *);
__device__ struct dirent *_readdir(DIR *);
__device__ void _rewinddir(DIR *);

#pragma endregion

#endif // __RUNTIMEOS_H__