// os_win.c
#define OS_WIN 1
#define POWERSAFE_OVERWRITE 1
#if OS_WIN // This file is used for Windows only
#include <windows.h>
#include <new.h>
#include "Core.cu.h"

namespace Core
{
#pragma region Preamble

#if defined(TEST) || defined(_DEBUG)
	bool OsTrace = true;
#define OSTRACE(X, ...) if (OsTrace) { _fprintf(nullptr, "OS: "X, __VA_ARGS__); }
#else
#define OSTRACE(X, ...)
#endif

#ifdef TEST
	int io_error_hit = 0;            // Total number of I/O Errors
	int io_error_hardhit = 0;        // Number of non-benign errors
	int io_error_pending = 0;        // Count down to first I/O error
	int io_error_persist = 0;        // True if I/O errors persist
	int io_error_benign = 0;         // True if errors are benign
	int diskfull_pending = 0;
	int diskfull = 0;
#define SimulateIOErrorBenign(X) io_error_benign=(X)
#define SimulateIOError(CODE) \
	if ((io_error_persist && io_error_hit) || io_error_pending-- == 1) { local_ioerr(); CODE; }
	static void local_ioerr() { OSTRACE("IOERR\n"); io_error_hit++; if (!io_error_benign) io_error_hardhit++; }
#define SimulateDiskfullError(CODE) \
	if (diskfull_pending) { if (diskfull_pending == 1) { \
	local_ioerr(); diskfull = 1; io_error_hit = 1; CODE; \
	} else diskfull_pending--; }
#else
#define SimulateIOErrorBenign(X)
#define SimulateIOError(A)
#define SimulateDiskfullError(A)
#endif

	// When testing, keep a count of the number of open files.
#ifdef TEST
	int open_file_count = 0;
#define OpenCounter(X) open_file_count += (X)
#else
#define OpenCounter(X)
#endif

#if OS_WIN && !defined(OS_WINNT)
#define OS_WINNT 1
#endif
#if defined(_WIN32_WCE)
#define OS_WINCE 1
#else
#define OS_WINCE 0
#endif
#if !defined(OS_WINRT)
#define OS_WINRT 0
#endif

#pragma endregion

#pragma region Polyfill

#if !OS_WINNT && !defined(OMIT_WAL) // Compiling and using WAL mode requires several APIs that are only available in Windows platforms based on the NT kernel.
#error "WAL mode requires support from the Windows NT kernel, compile with OMIT_WAL."
#endif

	// Are most of the Win32 ANSI APIs available (i.e. with certain exceptions based on the sub-platform)?
#ifdef __CYGWIN__
#include <sys/cygwin.h>
#endif
#if !OS_WINCE && !OS_WINRT 
#define WIN32_HAS_ANSI
#endif
#if OS_WINCE || OS_WINNT || OS_WINRT
#define WIN32_HAS_WIDE
#endif
#if WIN32_FILEMAPPING_API && !defined(OMIT_WAL)
#if OS_WINRT
	WINBASEAPI HANDLE WINAPI CreateFileMappingFromApp(HANDLE, LPSECURITY_ATTRIBUTES, ULONG, ULONG64, LPCWSTR);
	WINBASEAPI LPVOID WINAPI MapViewOfFileFromApp(HANDLE, ULONG, ULONG64, SIZE_T);
#else
#if defined(WIN32_HAS_ANSI)
	WINBASEAPI HANDLE WINAPI CreateFileMappingA(HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCSTR);
#endif
#if defined(WIN32_HAS_WIDE)
	WINBASEAPI HANDLE WINAPI CreateFileMappingW(HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCWSTR);
#endif
	WINBASEAPI LPVOID WINAPI MapViewOfFile(HANDLE, DWORD, DWORD, DWORD, SIZE_T);
#endif
	WINBASEAPI BOOL WINAPI UnmapViewOfFile(LPCVOID);
#endif
#if OS_WINCE // WinCE lacks native support for file locking so we have to fake it with some code of our own.
	typedef struct winceLock
	{
		int Readers;       // Number of reader locks obtained
		bool Pending;      // Indicates a pending lock has been obtained
		bool Reserved;     // Indicates a reserved lock has been obtained
		bool Exclusive;    // Indicates an exclusive lock has been obtained
	} winceLock;
#endif

	// Some Microsoft compilers lack this definition.
#ifndef INVALID_FILE_ATTRIBUTES
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1) 
#endif
#ifndef FILE_FLAG_MASK
#define FILE_FLAG_MASK (0xFF3C0000)
#endif
#ifndef FILE_ATTRIBUTE_MASK
#define FILE_ATTRIBUTE_MASK (0x0003FFF7)
#endif
#ifndef INVALID_SET_FILE_POINTER
#define INVALID_SET_FILE_POINTER ((DWORD)-1)
#endif

	// LOCKFILE_FAIL_IMMEDIATELY is undefined on some Windows systems.
#ifndef LOCKFILE_FAIL_IMMEDIATELY
#define LOCKFILE_FAIL_IMMEDIATELY 1
#endif
#ifndef LOCKFILE_EXCLUSIVE_LOCK
#define LOCKFILE_EXCLUSIVE_LOCK 2
#endif

	// Historically, SQLite has used both the LockFile and LockFileEx functions. When the LockFile function was used, it was always expected to fail
	// immediately if the lock could not be obtained.  Also, it always expected to obtain an exclusive lock.  These flags are used with the LockFileEx function
	// and reflect those expectations; therefore, they should not be changed.
#ifndef LOCKFILE_FLAGS
#define LOCKFILE_FLAGS (LOCKFILE_FAIL_IMMEDIATELY | LOCKFILE_EXCLUSIVE_LOCK)
#endif
#ifndef LOCKFILEEX_FLAGS
#define LOCKFILEEX_FLAGS (LOCKFILE_FAIL_IMMEDIATELY)
#endif

	// other
#undef ERROR

#pragma endregion

#pragma region WinVFile

#ifndef OMIT_WAL // Forward references
	typedef struct winShm winShm;           // A connection to shared-memory
	typedef struct winShmNode winShmNode;   // A region of shared-memory
#endif

	// winFile
	class WinVFile : public VFile
	{
	public:
		enum WINFILE : uint8
		{
			WINFILE_PERSIST_WAL = 0x04,  // Persistent WAL mode
			WINFILE_PSOW = 0x10,		// SQLITE_IOCAP_POWERSAFE_OVERWRITE
		};

		VSystem *Vfs;			// The VFS used to open this file
		HANDLE H;               // Handle for accessing the file
		LOCK Lock_;				// Type of lock currently held on this file
		short SharedLockByte;   // Randomly chosen byte used as a shared lock
		WINFILE CtrlFlags;      // Flags.  See WINFILE_* below
		DWORD LastErrno;        // The Windows errno from the last I/O error
#ifndef OMIT_WAL
		winShm *Shm;			// Instance of shared memory on this file
#endif
		const char *Path;		// Full pathname of this file
		int SizeChunk;          // Chunk size configured by FCNTL_CHUNK_SIZE
#if OS_WINCE
		LPWSTR DeleteOnClose;  // Name of file to delete when closing
		HANDLE Mutex;			// Mutex used to control access to shared lock
		HANDLE SharedHandle;	// Shared memory segment used for locking
		winceLock Local;        // Locks obtained by this instance of winFile
		winceLock *Shared;      // Global shared lock memory for the file
#endif
	public:
		__device__ virtual RC Read(void *buffer, int amount, int64 offset);
		__device__ virtual RC Write(const void *buffer, int amount, int64 offset);
		__device__ virtual RC Truncate(int64 size);
		__device__ virtual RC Close_();
		__device__ virtual RC Sync(SYNC flags);
		__device__ virtual RC get_FileSize(int64 &size);

		__device__ virtual RC Lock(LOCK lock);
		__device__ virtual RC Unlock(LOCK lock);
		__device__ virtual RC CheckReservedLock(int &lock);
		__device__ virtual RC FileControl(FCNTL op, void *arg);

		__device__ virtual uint get_SectorSize();
		__device__ virtual IOCAP get_DeviceCharacteristics();

		//__device__ virtual RC ShmLock(int offset, int n, SHM flags);
		//__device__ virtual void ShmBarrier();
		//__device__ virtual RC ShmUnmap(bool deleteFlag);
		//__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp);
	};

	__device__ inline void operator|=(WinVFile::WINFILE &a, int b) { a = (WinVFile::WINFILE)(a | b); }

#pragma endregion

#pragma region WinVSystem

	class WinVSystem : public VSystem
	{
	public:
		__device__ virtual VFile *_AttachFile(void *buffer);
		__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags);
		__device__ virtual RC Delete(const char *path, bool syncDirectory);
		__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC);
		__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut);

		__device__ virtual void *DlOpen(const char *filename);
		__device__ virtual void DlError(int bufLength, char *buf);
		__device__ virtual void (*DlSym(void *handle, const char *symbol))();
		__device__ virtual void DlClose(void *handle);

		__device__ virtual int Randomness(int bufLength, char *buf);
		__device__ virtual int Sleep(int microseconds);
		__device__ virtual RC CurrentTimeInt64(int64 *now);
		__device__ virtual RC CurrentTime(double *now);
		__device__ virtual RC GetLastError(int bufLength, char *buf);

		__device__ virtual RC SetSystemCall(const char *name, syscall_ptr newFunc);
		__device__ virtual syscall_ptr GetSystemCall(const char *name);
		__device__ virtual const char *NextSystemCall(const char *name);
	};

#pragma endregion

#pragma region Win32

#ifndef WIN32_DBG_BUF_SIZE // The size of the buffer used by sqlite3_win32_write_debug().
#define WIN32_DBG_BUF_SIZE ((int)(4096 - sizeof(DWORD)))
#endif
#ifndef WIN32_DATA_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the data directory should be changed.
#define WIN32_DATA_DIRECTORY_TYPE (1)
#endif
#ifndef WIN32_TEMP_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the temporary directory should be changed.
#define WIN32_TEMP_DIRECTORY_TYPE (2) 
#endif

#ifndef DEFAULT_SECTOR_SIZE
#define DEFAULT_SECTOR_SIZE 4096
#endif
#ifndef TEMP_FILE_PREFIX
#define TEMP_FILE_PREFIX "etilqs_"
#endif

#pragma endregion

#pragma region Syscall

#ifndef SYSCALL
#define SYSCALL syscall_ptr
#endif

	static struct win_syscall
	{
		const char *Name;            // Name of the system call
		syscall_ptr Current; // Current value of the system call
		syscall_ptr Default; // Default value
	} Syscalls[] = 
	{
#if !OS_WINCE && !OS_WINRT
		{"AreFileApisANSI", (SYSCALL)AreFileApisANSI, nullptr},
#else
		{"AreFileApisANSI", (SYSCALL)nullptr, nullptr},
#endif
#if OS_WINCE || OS_WINRT // This function is not available on Windows CE or WinRT.
#define osAreFileApisANSI() 1
#else
#define osAreFileApisANSI ((BOOL(WINAPI*)(VOID))Syscalls[0].Current)
#endif
#if OS_WINCE && defined(WIN32_HAS_WIDE)
		{"CharLowerW", (SYSCALL)CharLowerW, nullptr},
#else
		{"CharLowerW", (SYSCALL)nullptr, nullptr},
#endif
#define osCharLowerW ((LPWSTR(WINAPI*)(LPWSTR))aSyscall[1].pCurrent)
#if OS_WINCE && defined(WIN32_HAS_WIDE)
		{"CharUpperW", (SYSCALL)CharUpperW, nullptr},
#else
		{"CharUpperW", (SYSCALL)nullptr, nullptr},
#endif
#define osCharUpperW ((LPWSTR(WINAPI *)(LPWSTR))Syscalls[2].Current)
		{"CloseHandle", (SYSCALL)CloseHandle, nullptr},
#define osCloseHandle ((BOOL(WINAPI *)(HANDLE))Syscalls[3].Current)
#if defined(WIN32_HAS_ANSI)
		{"CreateFileA", (SYSCALL)CreateFileA, nullptr},
#else
		{"CreateFileA", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFileA ((HANDLE(WINAPI *)(LPCSTR,DWORD,DWORD,LPSECURITY_ATTRIBUTES,DWORD,DWORD,HANDLE))Syscalls[4].Current)
#if !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"CreateFileW", (SYSCALL)CreateFileW, nullptr},
#else
		{"CreateFileW", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFileW ((HANDLE(WINAPI*)(LPCWSTR,DWORD,DWORD,LPSECURITY_ATTRIBUTES,DWORD,DWORD,HANDLE))Syscalls[5].Current)
#if (!OS_WINRT && defined(WIN32_HAS_ANSI) && !defined(OMIT_WAL))
		{"CreateFileMappingA", (SYSCALL)CreateFileMappingA, nullptr},
#else
		{"CreateFileMappingA", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFileMappingA ((HANDLE(WINAPI *)(HANDLE,LPSECURITY_ATTRIBUTES,DWORD,DWORD,DWORD,LPCSTR))Syscalls[6].Current)

#if OS_WINCE || (!OS_WINRT && defined(WIN32_HAS_WIDE) && !defined(OMIT_WAL))
		{"CreateFileMappingW", (SYSCALL)CreateFileMappingW, nullptr},
#else
		{"CreateFileMappingW", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFileMappingW ((HANDLE(WINAPI *)(HANDLE,LPSECURITY_ATTRIBUTES,DWORD,DWORD,DWORD,LPCWSTR))Syscalls[7].Current)
#if !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"CreateMutexW", (SYSCALL)CreateMutexW, nullptr},
#else
		{"CreateMutexW", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateMutexW ((HANDLE(WINAPI *)(LPSECURITY_ATTRIBUTES,BOOL,LPCWSTR))Syscalls[8].Current)
#if defined(WIN32_HAS_ANSI)
		{"DeleteFileA", (SYSCALL)DeleteFileA, nullptr},
#else
		{"DeleteFileA", (SYSCALL)nullptr, nullptr},
#endif
#define osDeleteFileA ((BOOL(WINAPI *)(LPCSTR))Syscalls[9].Current)
#if defined(SQLITE_WIN32_HAS_WIDE)
		{"DeleteFileW", (SYSCALL)DeleteFileW, nullptr},
#else
		{"DeleteFileW", (SYSCALL)nullptr, nullptr},
#endif
#define osDeleteFileW ((BOOL(WINAPI *)(LPCWSTR))Syscalls[10].Current)
#if OS_WINCE
		{"FileTimeToLocalFileTime", (SYSCALL)FileTimeToLocalFileTime, nullptr},
#else
		{"FileTimeToLocalFileTime", (SYSCALL)nullptr, nullptr},
#endif
#define osFileTimeToLocalFileTime ((BOOL(WINAPI *)(CONST FILETIME*,LPFILETIME))Syscalls[11].Current)
#if OS_WINCE
		{"FileTimeToSystemTime", (SYSCALL)FileTimeToSystemTime, nullptr},
#else
		{"FileTimeToSystemTime", (SYSCALL)nullptr, nullptr},
#endif
#define osFileTimeToSystemTime ((BOOL(WINAPI *)(CONST FILETIME*,LPSYSTEMTIME))Syscalls[12].Current)
		{"FlushFileBuffers", (SYSCALL)FlushFileBuffers, nullptr},
#define osFlushFileBuffers ((BOOL(WINAPI *)(HANDLE))Syscalls[13].Current)
#if defined(WIN32_HAS_ANSI)
		{"FormatMessageA", (SYSCALL)FormatMessageA, nullptr},
#else
		{"FormatMessageA", (SYSCALL)nullptr, nullptr},
#endif
#define osFormatMessageA ((DWORD(WINAPI *)(DWORD,LPCVOID,DWORD,DWORD,LPSTR,DWORD,va_list*))Syscalls[14].Current)
#if defined(WIN32_HAS_WIDE)
		{"FormatMessageW", (SYSCALL)FormatMessageW, nullptr},
#else
		{"FormatMessageW", (SYSCALL)nullptr, nullptr},
#endif
#define osFormatMessageW ((DWORD(WINAPI *)(DWORD,LPCVOID,DWORD,DWORD,LPWSTR,DWORD,va_list*))Syscalls[15].Current)
#if !defined(OMIT_LOAD_EXTENSION)
		{"FreeLibrary", (SYSCALL)FreeLibrary, nullptr},
#else
		{"FreeLibrary", (SYSCALL)nullptr, nullptr},
#endif
#define osFreeLibrary ((BOOL(WINAPI *)(HMODULE))Syscalls[16].Current)
		{"GetCurrentProcessId", (SYSCALL)GetCurrentProcessId, nullptr},
#define osGetCurrentProcessId ((DWORD(WINAPI *)(VOID))Syscalls[17].Current)
#if !OS_WINCE && defined(WIN32_HAS_ANSI)
		{"GetDiskFreeSpaceA", (SYSCALL)GetDiskFreeSpaceA, nullptr},
#else
		{"GetDiskFreeSpaceA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetDiskFreeSpaceA ((BOOL(WINAPI *)(LPCSTR,LPDWORD,LPDWORD,LPDWORD,LPDWORD))Syscalls[18].Current)
#if !OS_WINCE && !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"GetDiskFreeSpaceW", (SYSCALL)GetDiskFreeSpaceW, nullptr},
#else
		{"GetDiskFreeSpaceW", (SYSCALL)nullptr, nullptr},
#endif
#define osGetDiskFreeSpaceW ((BOOL(WINAPI*)(LPCWSTR,LPDWORD,LPDWORD,LPDWORD,LPDWORD))Syscalls[19].Current)
#if defined(WIN32_HAS_ANSI)
		{"GetFileAttributesA", (SYSCALL)GetFileAttributesA, nullptr},
#else
		{"GetFileAttributesA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFileAttributesA ((DWORD(WINAPI *)(LPCSTR))Syscalls[20].Current)
#if !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"GetFileAttributesW", (SYSCALL)GetFileAttributesW, nullptr},
#else
		{"GetFileAttributesW", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFileAttributesW ((DWORD(WINAPI *)(LPCWSTR))Syscalls[21].Current)

#if defined(WIN32_HAS_WIDE)
		{"GetFileAttributesExW", (SYSCALL)GetFileAttributesExW, nullptr},
#else
		{"GetFileAttributesExW", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFileAttributesExW ((BOOL(WINAPI*)(LPCWSTR,GET_FILEEX_INFO_LEVELS,LPVOID))Syscalls[22].Current)
#if !OS_WINRT
		{"GetFileSize", (SYSCALL)GetFileSize, nullptr},
#else
		{"GetFileSize", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFileSize ((DWORD(WINAPI *)(HANDLE,LPDWORD))Syscalls[23].Current)
#if !OS_WINCE && defined(WIN32_HAS_ANSI)
		{"GetFullPathNameA", (SYSCALL)GetFullPathNameA, nullptr},
#else
		{"GetFullPathNameA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFullPathNameA ((DWORD(WINAPI*)(LPCSTR,DWORD,LPSTR,LPSTR*))Syscalls[24].Current)
#if !OS_WINCE && !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"GetFullPathNameW", (SYSCALL)GetFullPathNameW, nullptr},
#else
		{"GetFullPathNameW", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFullPathNameW ((DWORD(WINAPI *)(LPCWSTR,DWORD,LPWSTR,LPWSTR*))Syscalls[25].Current)
		{"GetLastError", (SYSCALL)GetLastError, nullptr},
#define osGetLastError ((DWORD(WINAPI *)(VOID))Syscalls[26].Current)
#if !defined(OMIT_LOAD_EXTENSION)
#if OS_WINCE
		// The GetProcAddressA() routine is only available on Windows CE.
		{"GetProcAddressA", (SYSCALL)GetProcAddressA, nullptr},
#else
		// All other Windows platforms expect GetProcAddress() to take an ANSI string regardless of the _UNICODE setting
		{"GetProcAddressA", (SYSCALL)GetProcAddress, nullptr},
#endif
#else
		{"GetProcAddressA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetProcAddressA ((FARPROC(WINAPI *)(HMODULE,LPCSTR))Syscalls[27].Current)
#if !OS_WINRT
		{"GetSystemInfo", (SYSCALL)GetSystemInfo, nullptr},
#else
		{"GetSystemInfo", (SYSCALL)nullptr, nullptr},
#endif
#define osGetSystemInfo ((VOID(WINAPI *)(LPSYSTEM_INFO))Syscalls[28].Current)
		{"GetSystemTime", (SYSCALL)GetSystemTime, nullptr},
#define osGetSystemTime ((VOID(WINAPI *)(LPSYSTEMTIME))Syscalls[29].Current)
#if !OS_WINCE
		{"GetSystemTimeAsFileTime", (SYSCALL)GetSystemTimeAsFileTime, nullptr},
#else
		{"GetSystemTimeAsFileTime", (SYSCALL)nullptr, nullptr},
#endif
#define osGetSystemTimeAsFileTime ((VOID(WINAPI *)(LPFILETIME))Syscalls[30].Current)
#if defined(WIN32_HAS_ANSI)
		{"GetTempPathA", (SYSCALL)GetTempPathA, nullptr},
#else
		{"GetTempPathA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetTempPathA ((DWORD(WINAPI*)(DWORD,LPSTR))Syscalls[31].Current)
#if !OS_WINRT && defined(WIN32_HAS_WIDE)
		{"GetTempPathW", (SYSCALL)GetTempPathW, nullptr},
#else
		{"GetTempPathW", (SYSCALL)nullptr, nullptr},
#endif
#define osGetTempPathW ((DWORD(WINAPI *)(DWORD,LPWSTR))Syscalls[32].Current)
#if !OS_WINRT
		{"GetTickCount", (SYSCALL)GetTickCount, nullptr},
#else
		{"GetTickCount", (SYSCALL)nullptr, nullptr},
#endif
#define osGetTickCount ((DWORD(WINAPI *)(VOID))Syscalls[33].Current)
#if defined(WIN32_HAS_ANSI)
		{"GetVersionExA", (SYSCALL)GetVersionExA, nullptr},
#else
		{"GetVersionExA", (SYSCALL)nullptr, nullptr},
#endif
#define osGetVersionExA ((BOOL(WINAPI *)(LPOSVERSIONINFOA))Syscalls[34].Current)
		{"HeapAlloc", (SYSCALL)HeapAlloc, nullptr},
#define osHeapAlloc ((LPVOID(WINAPI *)(HANDLE,DWORD,SIZE_T))Syscalls[35].Current)
#if !OS_WINRT
		{"HeapCreate", (SYSCALL)HeapCreate, nullptr},
#else
		{"HeapCreate", (SYSCALL)nullptr, nullptr},
#endif
#define osHeapCreate ((HANDLE(WINAPI *)(DWORD,SIZE_T,SIZE_T))Syscalls[36].Current)
#if !OS_WINRT
		{"HeapDestroy", (SYSCALL)HeapDestroy, nullptr},
#else
		{"HeapDestroy", (SYSCALL)nullptr, nullptr},
#endif
#define osHeapDestroy ((BOOL(WINAPI *)(HANDLE))Syscalls[37].Current)
		{"HeapFree", (SYSCALL)HeapFree, nullptr},
#define osHeapFree ((BOOL(WINAPI *)(HANDLE,DWORD,LPVOID))Syscalls[38].Current)
		{"HeapReAlloc", (SYSCALL)HeapReAlloc, nullptr},
#define osHeapReAlloc ((LPVOID(WINAPI *)(HANDLE,DWORD,LPVOID,SIZE_T))Syscalls[39].Current)
		{"HeapSize", (SYSCALL)HeapSize, nullptr},

#define osHeapSize ((SIZE_T(WINAPI *)(HANDLE,DWORD,LPCVOID))Syscalls[40].Current)
#if !OS_WINRT
		{"HeapValidate", (SYSCALL)HeapValidate, nullptr},
#else
		{"HeapValidate", (SYSCALL)nullptr, nullptr},
#endif
#define osHeapValidate ((BOOL(WINAPI *)(HANDLE,DWORD,LPCVOID))Syscalls[41].Current)
#if defined(WIN32_HAS_ANSI) && !defined(OMIT_LOAD_EXTENSION)
		{"LoadLibraryA", (SYSCALL)LoadLibraryA, nullptr},
#else
		{"LoadLibraryA", (SYSCALL)nullptr, nullptr},
#endif
#define osLoadLibraryA ((HMODULE(WINAPI *)(LPCSTR))Syscalls[42].Current)
#if !OS_WINRT && defined(WIN32_HAS_WIDE) && !defined(OMIT_LOAD_EXTENSION)
		{"LoadLibraryW", (SYSCALL)LoadLibraryW, nullptr},
#else
		{"LoadLibraryW", (SYSCALL)nullptr, nullptr},
#endif
#define osLoadLibraryW ((HMODULE(WINAPI *)(LPCWSTR))Syscalls[43].Current)
#if !OS_WINRT
		{"LocalFree", (SYSCALL)LocalFree, nullptr},
#else
		{"LocalFree", (SYSCALL)nullptr, nullptr},
#endif
#define osLocalFree ((HLOCAL(WINAPI *)(HLOCAL))Syscalls[44].Current)
#if !OS_WINCE && !OS_WINRT
		{"LockFile", (SYSCALL)LockFile, nullptr},
#else
		{"LockFile", (SYSCALL)nullptr, nullptr},
#endif
#ifndef osLockFile
#define osLockFile ((BOOL(WINAPI *)(HANDLE,DWORD,DWORD,DWORD,DWORD))Syscalls[45].Current)
#endif
#if !OS_WINCE
		{"LockFileEx", (SYSCALL)LockFileEx, nullptr},
#else
		{"LockFileEx", (SYSCALL)nullptr, nullptr},
#endif
#ifndef osLockFileEx
#define osLockFileEx ((BOOL(WINAPI *)(HANDLE,DWORD,DWORD,DWORD,DWORD,LPOVERLAPPED))Syscalls[46].Current)
#endif
#if OS_WINCE || (!OS_WINRT && !defined(OMIT_WAL))
		{"MapViewOfFile", (SYSCALL)MapViewOfFile,nullptr},
#else
		{"MapViewOfFile", (SYSCALL)nullptr,nullptr},
#endif
#define osMapViewOfFile ((LPVOID(WINAPI *)(HANDLE,DWORD,DWORD,DWORD,SIZE_T))Syscalls[47].Current)
		{"MultiByteToWideChar", (SYSCALL)MultiByteToWideChar, nullptr},
#define osMultiByteToWideChar ((int(WINAPI *)(UINT,DWORD,LPCSTR,int,LPWSTR,int))Syscalls[48].Current)
		{"QueryPerformanceCounter", (SYSCALL)QueryPerformanceCounter, nullptr},
#define osQueryPerformanceCounter ((BOOL(WINAPI *)(LARGE_INTEGER*))Syscalls[49].Current)
		{"ReadFile", (SYSCALL)ReadFile, nullptr},
#define osReadFile ((BOOL(WINAPI *)(HANDLE,LPVOID,DWORD,LPDWORD,LPOVERLAPPED))Syscalls[50].Current)
		{"SetEndOfFile", (SYSCALL)SetEndOfFile, nullptr},
#define osSetEndOfFile ((BOOL(WINAPI *)(HANDLE))Syscalls[51].Current)
#if !OS_WINRT
		{"SetFilePointer", (SYSCALL)SetFilePointer, nullptr},
#else
		{"SetFilePointer", (SYSCALL)nullptr, nullptr},
#endif
#define osSetFilePointer ((DWORD(WINAPI *)(HANDLE,LONG,PLONG,DWORD))Syscalls[52].Current)
#if !OS_WINRT
		{"Sleep", (SYSCALL)Sleep, nullptr},
#else
		{"Sleep", (SYSCALL)nullptr, nullptr},
#endif
#define osSleep ((VOID(WINAPI *)(DWORD))Syscalls[53].Current)
		{"SystemTimeToFileTime", (SYSCALL)SystemTimeToFileTime, nullptr},
#define osSystemTimeToFileTime ((BOOL(WINAPI *)(CONST SYSTEMTIME*,LPFILETIME))Syscalls[54].Current)
#if !OS_WINCE && !OS_WINRT
		{"UnlockFile", (SYSCALL)UnlockFile, nullptr},
#else
		{"UnlockFile", (SYSCALL)nullptr, nullptr},
#endif
#ifndef osUnlockFile
#define osUnlockFile ((BOOL(WINAPI *)(HANDLE,DWORD,DWORD,DWORD,DWORD))Syscalls[55].Current)
#endif
#if !OS_WINCE
		{"UnlockFileEx", (SYSCALL)UnlockFileEx, nullptr},
#else
		{"UnlockFileEx", (SYSCALL)nullptr, nullptr},
#endif
#define osUnlockFileEx ((BOOL(WINAPI *)(HANDLE,DWORD,DWORD,DWORD,LPOVERLAPPED))Syscalls[56].Current)
#if OS_WINCE || !defined(OMIT_WAL)
		{"UnmapViewOfFile", (SYSCALL)UnmapViewOfFile, nullptr},
#else
		{"UnmapViewOfFile", (SYSCALL)nullptr, nullptr},
#endif
#define osUnmapViewOfFile ((BOOL(WINAPI *)(LPCVOID))Syscalls[57].Current)
		{"WideCharToMultiByte", (SYSCALL)WideCharToMultiByte, nullptr},
#define osWideCharToMultiByte ((int(WINAPI *)(UINT,DWORD,LPCWSTR,int,LPSTR,int,LPCSTR,LPBOOL))Syscalls[58].Current)
		{"WriteFile", (SYSCALL)WriteFile, nullptr},
#define osWriteFile ((BOOL(WINAPI *)(HANDLE,LPCVOID,DWORD,LPDWORD,LPOVERLAPPED))Syscalls[59].Current)
#if OS_WINRT
		{"CreateEventExW", (SYSCALL)CreateEventExW, nullptr},
#else
		{"CreateEventExW", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateEventExW ((HANDLE(WINAPI *)(LPSECURITY_ATTRIBUTES,LPCWSTR,DWORD,DWORD))Syscalls[60].Current)
#if !OS_WINRT
		{"WaitForSingleObject", (SYSCALL)WaitForSingleObject, nullptr},
#else
		{"WaitForSingleObject", (SYSCALL)nullptr, nullptr},
#endif
#define osWaitForSingleObject ((DWORD(WINAPI *)(HANDLE,DWORD))Syscalls[61].Current)
#if OS_WINRT
		{"WaitForSingleObjectEx", (SYSCALL)WaitForSingleObjectEx, nullptr},
#else
		{"WaitForSingleObjectEx", (SYSCALL)nullptr, nullptr},
#endif
#define osWaitForSingleObjectEx ((DWORD(WINAPI *)(HANDLE,DWORD,BOOL))Syscalls[62].Current)
#if OS_WINRT
		{"SetFilePointerEx", (SYSCALL)SetFilePointerEx, nullptr},
#else
		{"SetFilePointerEx", (SYSCALL)nullptr, nullptr},
#endif
#define osSetFilePointerEx ((BOOL(WINAPI *)(HANDLE,LARGE_INTEGER,PLARGE_INTEGER,DWORD))Syscalls[63].Current)
#if OS_WINRT
		{"GetFileInformationByHandleEx", (SYSCALL)GetFileInformationByHandleEx, nullptr},
#else
		{"GetFileInformationByHandleEx", (SYSCALL)nullptr, nullptr},
#endif
#define osGetFileInformationByHandleEx ((BOOL(WINAPI *)(HANDLE,FILE_INFO_BY_HANDLE_CLASS,LPVOID,DWORD))Syscalls[64].Current)
#if OS_WINRT && !defined(OMIT_WAL)
		{"MapViewOfFileFromApp", (SYSCALL)MapViewOfFileFromApp, nullptr},
#else
		{"MapViewOfFileFromApp", (SYSCALL)nullptr, nullptr},
#endif
#define osMapViewOfFileFromApp ((LPVOID(WINAPI *)(HANDLE,ULONG,ULONG64,SIZE_T))Syscalls[65].Current)
#if OS_WINRT
		{"CreateFile2", (SYSCALL)CreateFile2, nullptr},
#else
		{"CreateFile2", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFile2 ((HANDLE(WINAPI *)(LPCWSTR,DWORD,DWORD,DWORD,LPCREATEFILE2_EXTENDED_PARAMETERS))Syscalls[66].Current)
#if OS_WINRT && !defined(OMIT_LOAD_EXTENSION)
		{"LoadPackagedLibrary", (SYSCALL)LoadPackagedLibrary, nullptr},
#else
		{"LoadPackagedLibrary", (SYSCALL)nullptr, nullptr},
#endif
#define osLoadPackagedLibrary ((HMODULE(WINAPI *)(LPCWSTR,DWORD))Syscalls[67].Current)
#if OS_WINRT
		{"GetTickCount64", (SYSCALL)GetTickCount64, nullptr},
#else
		{"GetTickCount64", (SYSCALL)nullptr, nullptr},
#endif
#define osGetTickCount64 ((ULONGLONG(WINAPI *)(VOID))Syscalls[68].Current)
#if OS_WINRT
		{"GetNativeSystemInfo", (SYSCALL)GetNativeSystemInfo, nullptr},
#else
		{"GetNativeSystemInfo", (SYSCALL)nullptr, nullptr},
#endif
#define osGetNativeSystemInfo ((VOID(WINAPI *)(LPSYSTEM_INFO))Syscalls[69].Current)
#if defined(WIN32_HAS_ANSI)
		{"OutputDebugStringA", (SYSCALL)OutputDebugStringA, nullptr},
#else
		{"OutputDebugStringA", (SYSCALL)nullptr, nullptr},
#endif
#define osOutputDebugStringA ((VOID(WINAPI *)(LPCSTR))Syscalls[70].Current)
#if defined(WIN32_HAS_WIDE)
		{"OutputDebugStringW", (SYSCALL)OutputDebugStringW, nullptr},
#else
		{"OutputDebugStringW", (SYSCALL)nullptr, nullptr},
#endif
#define osOutputDebugStringW ((VOID(WINAPI *)(LPCWSTR))Syscalls[71].Current)
		{"GetProcessHeap", (SYSCALL)GetProcessHeap, nullptr},
#define osGetProcessHeap ((HANDLE(WINAPI *)(VOID))Syscalls[72].Current)
#if OS_WINRT && !defined(OMIT_WAL)
		{"CreateFileMappingFromApp", (SYSCALL)CreateFileMappingFromApp, nullptr},
#else
		{"CreateFileMappingFromApp", (SYSCALL)nullptr, nullptr},
#endif
#define osCreateFileMappingFromApp ((HANDLE(WINAPI *)(HANDLE,LPSECURITY_ATTRIBUTES,ULONG,ULONG64,LPCWSTR))Syscalls[73].Current)
	}; // End of the overrideable system calls

	// The following variable is (normally) set once and never changes thereafter.  It records whether the operating system is Win9x or WinNT.
	// 0:   Operating system unknown.
	// 1:   Operating system is Win9x.
	// 2:   Operating system is WinNT.
	// In order to facilitate testing on a WinNT system, the test fixture can manually set this value to 1 to emulate Win98 behavior.
#ifdef TEST
	int os_type = 0;
#else
	static int os_type = 0;
#endif
#if OS_WINCE || OS_WINRT
#define isNT() (true)
#elif !defined(WIN32_HAS_WIDE)
#define isNT() (false)
#else
	static bool isNT()
	{
		if (os_type == 0)
		{
			OSVERSIONINFOA sInfo;
			sInfo.dwOSVersionInfoSize = sizeof(sInfo);
			osGetVersionExA(&sInfo);
			os_type = (sInfo.dwPlatformId == VER_PLATFORM_WIN32_NT ? 2 : 1);
		}
		return (os_type == 2);
	}
#endif

	RC WinVSystem::SetSystemCall(const char *name, syscall_ptr newFunc)
	{
		RC rc = RC_NOTFOUND;
		if (name == nullptr)
		{
			/// If no zName is given, restore all system calls to their default settings and return NULL
			rc = RC_OK;
			for (int i = 0; i < _lengthof(Syscalls); i++)
				if (Syscalls[i].Default)
					Syscalls[i].Current = Syscalls[i].Default;
			return rc;
		}
		// If zName is specified, operate on only the one system call specified.
		for (int i = 0; i < _lengthof(Syscalls); i++)
		{
			if (!strcmp(name, Syscalls[i].Name))
			{
				if (!Syscalls[i].Default)
					Syscalls[i].Default = Syscalls[i].Current;
				rc = RC_OK;
				if (!newFunc) newFunc = Syscalls[i].Default;
				Syscalls[i].Current = newFunc;
				break;
			}
		}
		return rc;
	}

	syscall_ptr WinVSystem::GetSystemCall(const char *name)
	{
		for (int i = 0; i < _lengthof(Syscalls); i++)
			if (!strcmp(name, Syscalls[i].Name)) return Syscalls[i].Current;
		return nullptr;
	}

	const char *WinVSystem::NextSystemCall(const char *name)
	{
		int i = -1;
		if (name)
			for (i = 0; i < _lengthof(Syscalls)-1; i++)
				if (!strcmp(name, Syscalls[i].Name)) break;
		for (i++; i < _lengthof(Syscalls); i++)
			if (Syscalls[i].Current) return Syscalls[i].Name;
		return 0;
	}

#pragma endregion

#pragma region Win32

	void win32_WriteDebug(const char *buf, int bufLength)
	{
		char dbgBuf[WIN32_DBG_BUF_SIZE];
		int min = MIN(bufLength, (WIN32_DBG_BUF_SIZE - 1)); // may be negative.
		if (min < -1) min = -1; // all negative values become -1.
		_assert(min == -1 || min == 0 || min < WIN32_DBG_BUF_SIZE);
#if defined(WIN32_HAS_ANSI)
		if (min > 0)
		{
			memset(dbgBuf, 0, WIN32_DBG_BUF_SIZE);
			memcpy(dbgBuf, buf, min);
			osOutputDebugStringA(dbgBuf);
		}
		else
			osOutputDebugStringA(buf);
#elif defined(WIN32_HAS_WIDE)
		memset(dbgBuf, 0, WIN32_DBG_BUF_SIZE);
		if (osMultiByteToWideChar(osAreFileApisANSI() ? CP_ACP : CP_OEMCP, 0, buf, min, (LPWSTR)dbgBuf, WIN32_DBG_BUF_SIZE/sizeof(WCHAR)) <= 0)
			return;
		osOutputDebugStringW((LPCWSTR)dbgBuf);
#else
		if (min > 0)
		{
			memset(dbgBuf, 0, WIN32_DBG_BUF_SIZE);
			memcpy(dbgBuf, buf, min);
			fprintf(stderr, "%s", dbgBuf);
		}
		else
			fprintf(stderr, "%s", buf);
#endif
	}

#if OS_WINRT
	static HANDLE sleepObj = NULL;
#endif
	void win32_Sleep(DWORD milliseconds)
	{
#if OS_WINRT
		if (sleepObj == NULL)
			sleepObj = osCreateEventExW(NULL, NULL, CREATE_EVENT_MANUAL_RESET, SYNCHRONIZE);
		_assert(sleepObj != NULL);
		osWaitForSingleObjectEx(sleepObj, milliseconds, FALSE);
#else
		osSleep(milliseconds);
#endif
	}

#pragma endregion

#pragma region WIN32_MALLOC
#define WIN32_MALLOC
#ifdef WIN32_MALLOC

	// If compiled with WIN32_MALLOC on Windows, we will use the various Win32 API heap functions instead of our own.

	// If this is non-zero, an isolated heap will be created by the native Win32 allocator subsystem; otherwise, the default process heap will be used.  This
	// setting has no effect when compiling for WinRT.  By default, this is enabled and an isolated heap will be created to store all allocated data.
	//
	//*****************************************************************************
	// WARNING: It is important to note that when this setting is non-zero and the winMemShutdown function is called (e.g. by the sqlite3_shutdown
	//          function), all data that was allocated using the isolated heap will be freed immediately and any attempt to access any of that freed
	//          data will almost certainly result in an immediate access violation.
	//*****************************************************************************
#define DEFAULT_CACHE_SIZE 0
#define DEFAULT_PAGE_SIZE 0
#ifndef WIN32_HEAP_CREATE
#define WIN32_HEAP_CREATE (TRUE)
#endif
#ifndef WIN32_HEAP_INIT_SIZE // The initial size of the Win32-specific heap.  This value may be zero.
#define WIN32_HEAP_INIT_SIZE ((DEFAULT_CACHE_SIZE) * (DEFAULT_PAGE_SIZE) + 4194304)
#endif
#ifndef WIN32_HEAP_MAX_SIZE // The maximum size of the Win32-specific heap.  This value may be zero.
#define WIN32_HEAP_MAX_SIZE (0)
#endif
#ifndef WIN32_HEAP_FLAGS // The extra flags to use in calls to the Win32 heap APIs. This value may be zero for the default behavior.
#define WIN32_HEAP_FLAGS (0)
#endif

	// The winMemData structure stores information required by the Win32-specific sqlite3_mem_methods implementation.
	typedef struct WinMemData
	{
#ifdef _DEBUG
		uint32 Magic;    // Magic number to detect structure corruption.
#endif
		HANDLE Heap; // The handle to our heap.
		BOOL Owned;  // Do we own the heap (i.e. destroy it on shutdown)?
	} WinMemData;

#ifdef _DEBUG
#define WINMEM_MAGIC 0x42b2830b
#endif

	static struct WinMemData winMemData_ = {
#ifdef _DEBUG
		WINMEM_MAGIC,
#endif
		NULL, FALSE
	};

#ifdef _DEBUG
#define winMemAssertMagic() _assert(winMemData_.Magic == WINMEM_MAGIC)
#else
#define winMemAssertMagic()
#endif
#define winMemGetHeap() winMemData_.Heap

	class WinVAlloc : public VAlloc
	{
	public:
		__device__ virtual void *Alloc(int bytes);
		__device__ virtual void Free(void *prior);
		__device__ virtual void *Realloc(void *prior, int bytes);
		__device__ virtual int Size(void *p);
		__device__ virtual int Roundup(int bytes);
		__device__ virtual RC Init(void *appData);
		__device__ virtual void Shutdown(void *appData);
	};

	void *WinVAlloc::Alloc(int bytes)
	{
		winMemAssertMagic();
		HANDLE heap = winMemGetHeap();
		_assert(heap != 0);
		_assert(heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(WIN32_MALLOC_VALIDATE)
		_assert(osHeapValidate(heap, WIN32_HEAP_FLAGS, NULL));
#endif
		_assert(bytes >=0);
		void *p = osHeapAlloc(heap, WIN32_HEAP_FLAGS, (SIZE_T)bytes);
		if (!p)
			SysEx_LOG(RC_NOMEM, "failed to HeapAlloc %u bytes (%d), heap=%p", bytes, osGetLastError(), (void*)heap);
		return p;
	}

	void WinVAlloc::Free(void *prior)
	{
		winMemAssertMagic();
		HANDLE heap = winMemGetHeap();
		_assert(heap != 0);
		_assert(heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(WIN32_MALLOC_VALIDATE)
		_assert(osHeapValidate(heap, WIN32_HEAP_FLAGS, prior));
#endif
		if (!prior) return; // Passing NULL to HeapFree is undefined.
		if (!osHeapFree(heap, WIN32_HEAP_FLAGS, prior))
			SysEx_LOG(RC_NOMEM, "failed to HeapFree block %p (%d), heap=%p", prior, osGetLastError(), (void*)heap);
	}

	void *WinVAlloc::Realloc(void *prior, int bytes)
	{
		winMemAssertMagic();
		HANDLE heap = winMemGetHeap();
		_assert(heap != NULL);
		_assert(heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(SQLITE_WIN32_MALLOC_VALIDATE)
		_assert(osHeapValidate(heap, WIN32_HEAP_FLAGS, prior));
#endif
		_assert(bytes >= 0);
		void *p;
		if (!prior)
			p = osHeapAlloc(heap, WIN32_HEAP_FLAGS, (SIZE_T)bytes);
		else
			p = osHeapReAlloc(heap, WIN32_HEAP_FLAGS, prior, (SIZE_T)bytes);
		if (!p)
			SysEx_LOG(RC_NOMEM, "failed to %s %u bytes (%d), heap=%p", (prior ? "HeapReAlloc" : "HeapAlloc"), bytes, osGetLastError(), (void*)heap);
		return p;
	}

	int WinVAlloc::Size(void *p)
	{
		winMemAssertMagic();
		HANDLE heap = winMemGetHeap();
		_assert(heap != NULL);
		_assert(heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(SQLITE_WIN32_MALLOC_VALIDATE)
		_assert(osHeapValidate(heap, WIN32_HEAP_FLAGS, NULL));
#endif
		if (!p) return 0;
		SIZE_T n = osHeapSize(heap, WIN32_HEAP_FLAGS, p);
		if (n == (SIZE_T)-1)
		{
			SysEx_LOG(RC_NOMEM, "failed to HeapSize block %p (%d), heap=%p", p, osGetLastError(), (void*)heap);
			return 0;
		}
		return (int)n;
	}

	int WinVAlloc::Roundup(int bytes)
	{
		return bytes;
	}

	RC WinVAlloc::Init(void *appData)
	{
		WinMemData *winMemData = (WinMemData *)appData;
		if (!winMemData) return RC_ERROR;
#if _DEBUG
		_assert(winMemData->Magic == WINMEM_MAGIC);
#endif
#if !OS_WINRT && WIN32_HEAP_CREATE
		if (!winMemData->Heap)
		{
			winMemData->Heap = osHeapCreate(WIN32_HEAP_FLAGS, WIN32_HEAP_INIT_SIZE, WIN32_HEAP_MAX_SIZE);
			if (!winMemData->Heap)
			{
				SysEx_LOG(RC_NOMEM, "failed to HeapCreate (%d), flags=%u, initSize=%u, maxSize=%u", osGetLastError(), WIN32_HEAP_FLAGS, WIN32_HEAP_INIT_SIZE, WIN32_HEAP_MAX_SIZE);
				return RC_NOMEM;
			}
			winMemData->Owned = TRUE;
			_assert(winMemData->Owned);
		}
#else
		winMemData->Heap = osGetProcessHeap();
		if (!winMemData->Heap)
		{
			SysEx_LOG(RC_NOMEM, "failed to GetProcessHeap (%d)", osGetLastError());
			return RC_NOMEM;
		}
		winMemData->Owned = FALSE;
		_assert(!winMemData->Owned);
#endif
		_assert(winMemData->Heap != 0);
		_assert(winMemData->Heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(WIN32_MALLOC_VALIDATE)
		_assert(osHeapValidate(winMemData->Heap, WIN32_HEAP_FLAGS, NULL));
#endif
		return RC_OK;
	}

	void WinVAlloc::Shutdown(void *appData)
	{
		WinMemData *winMemData = (WinMemData *)appData;
		if (!winMemData) return;
		if (winMemData->Heap)
		{
			_assert(winMemData->Heap != INVALID_HANDLE_VALUE);
#if !OS_WINRT && defined(WIN32_MALLOC_VALIDATE)
			_assert(osHeapValidate(winMemData->Heap, WIN32_HEAP_FLAGS, NULL));
#endif
			if (winMemData->Owned)
			{
				if (!osHeapDestroy(winMemData->Heap))
					SysEx_LOG(RC_NOMEM, "failed to HeapDestroy (%d), heap=%p", osGetLastError(), (void*)winMemData->Heap);
				winMemData->Owned = FALSE;
			}
			winMemData->Heap = NULL;
		}
	}

#endif
#pragma endregion

#pragma region String Converters

	static LPWSTR Utf8ToUnicode(const char *name)
	{
		int c = osMultiByteToWideChar(CP_UTF8, 0, name, -1, NULL, 0);
		if (!c)
			return nullptr;
		LPWSTR wideName = (LPWSTR)_alloc2(c*sizeof(wideName[0]), true);
		if (!wideName)
			return nullptr;
		c = osMultiByteToWideChar(CP_UTF8, 0, name, -1, wideName, c);
		if (!c)
		{
			_free(wideName);
			wideName = nullptr;
		}
		return wideName;
	}

	static char *UnicodeToUtf8(LPCWSTR wideName)
	{
		int c = osWideCharToMultiByte(CP_UTF8, 0, wideName, -1, 0, 0, 0, 0);
		if (!c)
			return nullptr;
		char *name = (char *)_alloc2(c, true);
		if (!name)
			return nullptr;
		c = osWideCharToMultiByte(CP_UTF8, 0, wideName, -1, name, c, 0, 0);
		if (!c)
		{
			_free(name);
			name = nullptr;
		}
		return name;
	}

	static LPWSTR MbcsToUnicode(const char *name)
	{
		int codepage = (osAreFileApisANSI() ? CP_ACP : CP_OEMCP);
		int c = osMultiByteToWideChar(codepage, 0, name, -1, NULL, 0)*sizeof(WCHAR);
		if (!c)
			return nullptr;
		LPWSTR mbcsName = (LPWSTR)_alloc2(c*sizeof(mbcsName[0]), true);
		if (!mbcsName)
			return nullptr;
		c = osMultiByteToWideChar(codepage, 0, name, -1, mbcsName, c);
		if (!c)
		{
			_free(mbcsName);
			mbcsName = nullptr;
		}
		return mbcsName;
	}

	static char *UnicodeToMbcs(LPCWSTR wideName)
	{
		int codepage = (osAreFileApisANSI() ? CP_ACP : CP_OEMCP);
		int c = osWideCharToMultiByte(codepage, 0, wideName, -1, 0, 0, 0, 0);
		if (!c)
			return nullptr;
		char *name = (char *)_alloc2(c, true);
		if (!name)
			return nullptr;
		c = osWideCharToMultiByte(codepage, 0, wideName, -1, name, c, 0, 0);
		if (!c)
		{
			_free(name);
			name = nullptr;
		}
		return name;
	}

	char *win32_MbcsToUtf8(const char *name)
	{
		LPWSTR tmpWide = MbcsToUnicode(name);
		if (!tmpWide)
			return nullptr;
		char *nameUtf8 = UnicodeToUtf8(tmpWide);
		_free(tmpWide);
		return nameUtf8;
	}

	char *win32_Utf8ToMbcs(const char *name)
	{
		LPWSTR tmpWide = Utf8ToUnicode(name);
		if (!tmpWide)
			return nullptr;
		char *nameMbcs = UnicodeToMbcs(tmpWide);
		_free(tmpWide);
		return nameMbcs;
	}

#pragma endregion

#pragma region Win32

	char *data_directory;
	char *temp_directory;
	RC win32_SetDirectory(DWORD type, LPCWSTR value)
	{
#ifndef OMIT_AUTOINIT
		RC rc = SysEx::AutoInitialize();
		if (rc) return rc;
#endif
		char **directory = nullptr;
		if (type == WIN32_DATA_DIRECTORY_TYPE)
			directory = &data_directory;
		else if (type == WIN32_TEMP_DIRECTORY_TYPE)
			directory = &temp_directory;
		_assert(!directory || type == WIN32_DATA_DIRECTORY_TYPE || type == WIN32_TEMP_DIRECTORY_TYPE);
		_assert(!directory || _memdbg_hastype(*directory, MEMTYPE_HEAP));
		if (directory)
		{
			char *valueUtf8 = nullptr;
			if (value && value[0])
			{
				valueUtf8 = UnicodeToUtf8(value);
				if (!valueUtf8)
					return RC_NOMEM;
			}
			_free(*directory);
			*directory = valueUtf8;
			return RC_OK;
		}
		return RC_ERROR;
	}

#pragma endregion

#pragma region OS Errors

	static RC getLastErrorMsg(DWORD lastErrno, int bufLength, char *buf)
	{
		// FormatMessage returns 0 on failure.  Otherwise it returns the number of TCHARs written to the output
		// buffer, excluding the terminating null char.
		DWORD dwLen = 0;
		char *out = nullptr;
		if (isNT())
		{
#if OS_WINRT
			WCHAR tempWide[MAX_PATH + 1]; // NOTE: Somewhat arbitrary.
			dwLen = osFormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, tempWide, MAX_PATH, 0);
#else
			LPWSTR tempWide = NULL;
			dwLen = osFormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPWSTR)&tempWide, 0, 0);
#endif
			if (dwLen > 0)
			{
				// allocate a buffer and convert to UTF8
				_benignalloc_begin();
				out = UnicodeToUtf8(tempWide);
				_benignalloc_end();
#if !OS_WINRT
				// free the system buffer allocated by FormatMessage
				osLocalFree(tempWide);
#endif
			}
		}
#ifdef WIN32_HAS_ANSI
		else
		{
			char *temp = NULL;
			dwLen = osFormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPSTR)&temp, 0, 0);
			if (dwLen > 0)
			{
				// allocate a buffer and convert to UTF8
				_benignalloc_begin();
				out = win32_MbcsToUtf8(temp);
				_benignalloc_end();
				// free the system buffer allocated by FormatMessage
				osLocalFree(temp);
			}
		}
#endif
		if (!dwLen)
			_snprintf(buf, bufLength, "OsError 0x%x (%u)", lastErrno, lastErrno);
		else
		{
			// copy a maximum of nBuf chars to output buffer
			_snprintf(buf, bufLength, "%s", out);
			// free the UTF8 buffer
			_free(out);
		}
		return RC_OK;
	}

#define winLogError(a,b,c,d) winLogErrorAtLine(a,b,c,d,__LINE__)
	static RC winLogErrorAtLine(RC errcode, DWORD lastErrno, const char *func, const char *path, int line)
	{
		char msg[500]; // Human readable error text
		msg[0] = 0;
		getLastErrorMsg(lastErrno, sizeof(msg), msg);
		_assert(errcode != RC_OK);
		if (!path) path = "";
		int i;
		for (i = 0; msg[i] && msg[i] != '\r' && msg[i] != '\n'; i++) { }
		msg[i] = 0;
		SysEx_LOG(errcode, "os_win.c:%d: (%d) %s(%s) - %s", line, lastErrno, func, path, msg);
		return errcode;
	}

#ifndef WIN32_IOERR_RETRY
#define WIN32_IOERR_RETRY 10
#endif
#ifndef WIN32_IOERR_RETRY_DELAY
#define WIN32_IOERR_RETRY_DELAY 25
#endif
	static int win32IoerrRetry = WIN32_IOERR_RETRY;
	static int win32IoerrRetryDelay = WIN32_IOERR_RETRY_DELAY;

	static int retryIoerr(int *retry, DWORD *error)
	{
		DWORD e = osGetLastError();
		if (*retry >= win32IoerrRetry)
		{
			if (error)
				*error = e;
			return 0;
		}
		if (e == ERROR_ACCESS_DENIED || e == ERROR_LOCK_VIOLATION || e == ERROR_SHARING_VIOLATION)
		{
			win32_Sleep(win32IoerrRetryDelay*(1+*retry));
			++*retry;
			return 1;
		}
		if (error)
			*error = e;
		return 0;
	}

	static void logIoerr(int retry)
	{
		if (retry)
			SysEx_LOG(RC_IOERR, "delayed %dms for lock/sharing conflict", win32IoerrRetryDelay*retry*(retry+1)/2);
	}

#pragma endregion

#pragma region WinCE Only
#if OS_WINCE

#define HANDLE_TO_WINFILE(a) (WinVFile*)&((char*)a)[-(int)offsetof(WinVFile,h)]

#if !defined(MSVC_LOCALTIME_API) || !MSVC_LOCALTIME_API
	// The MSVC CRT on Windows CE may not have a localtime() function.  So create a substitute.
#include <time.h>
	struct tm *__cdecl localtime(const time_t *t)
	{
		static struct tm y;
		FILETIME uTm, lTm;
		SYSTEMTIME pTm;
		sqlite3_int64 t64;
		t64 = *t;
		t64 = (t64 + 11644473600)*10000000;
		uTm.dwLowDateTime = (DWORD)(t64 & 0xFFFFFFFF);
		uTm.dwHighDateTime= (DWORD)(t64 >> 32);
		osFileTimeToLocalFileTime(&uTm,&lTm);
		osFileTimeToSystemTime(&lTm,&pTm);
		y.tm_year = pTm.wYear - 1900;
		y.tm_mon = pTm.wMonth - 1;
		y.tm_wday = pTm.wDayOfWeek;
		y.tm_mday = pTm.wDay;
		y.tm_hour = pTm.wHour;
		y.tm_min = pTm.wMinute;
		y.tm_sec = pTm.wSecond;
		return &y;
	}
#endif

	static void winceMutexAcquire(HANDLE h)
	{
		DWORD err;
		do
		{
			err = osWaitForSingleObject(h, INFINITE);
		} while (err != WAIT_OBJECT_0 && err != WAIT_ABANDONED);
	}

#define winceMutexRelease(h) ReleaseMutex(h)

	static RC winceCreateLock(const char *filename, WinVFile *file)
	{
		LPWSTR name = Utf8ToUnicode(filename);
		if (!name)
			return RC_IOERR_NOMEM;
		// Initialize the local lockdata
		memset(&file->Local, 0, sizeof(file->Local));
		// Replace the backslashes from the filename and lowercase it to derive a mutex name.
		LPWSTR tok = osCharLowerW(name);
		for (; *tok; tok++)
			if (*tok == '\\') *tok = '_';
		// Create/open the named mutex
		file->Mutex = osCreateMutexW(NULL, FALSE, name);
		if (!file->Mutex)
		{
			file->LastErrno = osGetLastError();
			winLogError(RC_IOERR, file->LastErrno, "winceCreateLock1", filename);
			_free(name);
			return RC_IOERR;
		}
		// Acquire the mutex before continuing
		winceMutexAcquire(file->Mutex);
		// Since the names of named mutexes, semaphores, file mappings etc are case-sensitive, take advantage of that by uppercasing the mutex name
		// and using that as the shared filemapping name.
		osCharUpperW(name);
		file->SharedHandle = osCreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(winceLock), name);  
		// Set a flag that indicates we're the first to create the memory so it must be zero-initialized
		bool init = true;
		DWORD lastErrno = osGetLastError();
		if (lastErrno == ERROR_ALREADY_EXISTS)
			init = false;
		_free(name);

		// If we succeeded in making the shared memory handle, map it.
		bool logged = false;
		if (file->SharedHandle)
		{
			file->Shared = (winceLock *)osMapViewOfFile(file->SharedHandle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(winceLock));
			// If mapping failed, close the shared memory handle and erase it
			if (!file->Shared)
			{
				file->LastErrno = osGetLastError();
				winLogError(RC_IOERR, file->LastErrno, "winceCreateLock2", filename);
				logged = true;
				osCloseHandle(file->SharedHandle);
				file->SharedHandle = NULL;
			}
		}
		// If shared memory could not be created, then close the mutex and fail
		if (!file->SharedHandle)
		{
			if (!logged)
			{
				file->LastErrno = lastErrno;
				winLogError(RC_IOERR, file->LastErrno, "winceCreateLock3", filename);
				logged = true;
			}
			winceMutexRelease(file->Mutex);
			osCloseHandle(file->Mutex);
			file->Mutex = NULL;
			return RC_IOERR;
		}
		// Initialize the shared memory if we're supposed to
		if (init)
			memset(file->Shared, 0, sizeof(winceLock));
		winceMutexRelease(file->Mutex);
		return RC_OK;
	}

	static void winceDestroyLock(WinVFile *file)
	{
		if (file->Mutex)
		{
			// Acquire the mutex
			winceMutexAcquire(file->Mutex);
			// The following blocks should probably assert in debug mode, but they are to cleanup in case any locks remained open
			if (file->Local.Readers)
				file->Shared->Readers--;
			if (file->Local.Reserved)
				file->Shared->Reserved = FALSE;
			if (file->Local.Pending)
				file->Shared->Pending = FALSE;
			if (file->Local.Exclusive)
				file->Shared->Exclusive = FALSE;
			// De-reference and close our copy of the shared memory handle
			osUnmapViewOfFile(file->Shared);
			osCloseHandle(file->SharedHandle);
			// Done with the mutex
			winceMutexRelease(file->Mutex);    
			osCloseHandle(file->Mutex);
			file->Mutex = NULL;
		}
	}

	static BOOL winceLockFile(LPHANDLE fileHandle, DWORD fileOffsetLow, DWORD fileOffsetHigh, DWORD numberOfBytesToLockLow, DWORD numberOfBytesToLockHigh)
	{
		WinVFile *file = HANDLE_TO_WINFILE(fileHandle);
		BOOL r = FALSE;
		if (!file->Mutex) return true;
		winceMutexAcquire(file->Mutex);
		// Wanting an exclusive lock?
		if (fileOffsetLow == (DWORD)SHARED_FIRST && numberOfBytesToLockLow == (DWORD)SHARED_SIZE)
		{
			if (file->Shared->Readers == 0 && !file->Shared->Exclusive)
			{
				file->Shared->Exclusive = true;
				file->Local.Exclusive = true;
				r = TRUE;
			}
		}
		// Want a read-only lock? 
		else if (fileOffsetLow == (DWORD)SHARED_FIRST && numberOfBytesToLockLow == 1)
		{
			if (!file->Shared->Exclusive)
			{
				file->Local.Readers++;
				if (file->Local.Readers == 1)
					file->Shared->Readers++;
				r = TRUE;
			}
		}
		// Want a pending lock?
		else if (fFileOffsetLow == (DWORD)PENDING_BYTE && numberOfBytesToLockLow == 1)
		{
			// If no pending lock has been acquired, then acquire it
			if (!file->Shared->Pending) 
			{
				file->Shared->Pending = true;
				file->Local.Pending = true;
				r = TRUE;
			}
		}
		// Want a reserved lock?
		else if (fileOffsetLow == (DWORD)RESERVED_BYTE && numberOfBytesToLockLow == 1)
		{
			if (!file->Shared->Reserved)
			{
				file->Shared->Reserved = true;
				file->Local.Reserved = true;
				r = TRUE;
			}
		}
		winceMutexRelease(file->Mutex);
		return r;
	}

	static BOOL winceUnlockFile(LPHANDLE fileHandle, DWORD fileOffsetLow, DWORD fileOffsetHigh, DWORD numberOfBytesToUnlockLow, DWORD numberOfBytesToUnlockHigh)
	{
		WinVFile *file = HANDLE_TO_WINFILE(fileHandle);
		BOOL r = FALSE;
		if (!file->Mutex) return true;
		winceMutexAcquire(file->Mutex);
		// Releasing a reader lock or an exclusive lock
		if (fileOffsetLow == (DWORD)SHARED_FIRST)
		{
			// Did we have an exclusive lock?
			if (file->Local.Exclusive)
			{
				_assert(numberOfBytesToUnlockLow == (DWORD)SHARED_SIZE);
				file->Local.Exclusive = false;
				file->Shared->Exclusive = false;
				r = TRUE;
			}
			// Did we just have a reader lock?
			else if (file->Local.Readers)
			{
				_assert(numberOfBytesToUnlockLow == (DWORD)SHARED_SIZE || numberOfBytesToUnlockLow == 1);
				file->Local.Readers--;
				if (file->Local.Readers == 0)
					file->Shared->Readers--;
				r = TRUE;
			}
		}
		// Releasing a pending lock
		else if (fileOffsetLow == (DWORD)PENDING_BYTE && numberOfBytesToUnlockLow == 1)
		{
			if (file->Local.Pending)
			{
				file->Local.Pending = false;
				file->Shared->Pending = false;
				r = TRUE;
			}
		}
		// Releasing a reserved lock
		else if (fileOffsetLow == (DWORD)RESERVED_BYTE && numberOfBytesToUnlockLow == 1)
		{
			if (file->Local.Reserved)
			{
				file->Local.Reserved = false;
				file->Shared->Reserved = false;
				r = TRUE;
			}
		}
		winceMutexRelease(file->Mutex);
		return r;
	}

#endif
#pragma endregion

#pragma region Locking

	static BOOL winLockFile(LPHANDLE fileHandle, DWORD flags, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh)
	{
#if OS_WINCE
		// NOTE: Windows CE is handled differently here due its lack of the Win32 API LockFile.
		return winceLockFile(fileHandle, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#else
		if (isNT())
		{
			OVERLAPPED ovlp;
			memset(&ovlp, 0, sizeof(OVERLAPPED));
			ovlp.Offset = offsetLow;
			ovlp.OffsetHigh = offsetHigh;
			return osLockFileEx(*fileHandle, flags, 0, numBytesLow, numBytesHigh, &ovlp);
		}
		else
			return osLockFile(*fileHandle, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#endif
	}

	static BOOL winUnlockFile(LPHANDLE fileHandle, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh)
	{
#if OS_WINCE
		// NOTE: Windows CE is handled differently here due its lack of the Win32 API UnlockFile.
		return winceUnlockFile(fileHandle, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#else
		if (isNT())
		{
			OVERLAPPED ovlp;
			memset(&ovlp, 0, sizeof(OVERLAPPED));
			ovlp.Offset = offsetLow;
			ovlp.OffsetHigh = offsetHigh;
			return osUnlockFileEx(*fileHandle, 0, numBytesLow, numBytesHigh, &ovlp);
		}
		else
			return osUnlockFile(*fileHandle, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#endif
	}

#pragma endregion

#pragma region WinVFile

	static int seekWinFile(WinVFile *file, int64 offset)
	{
#if !OS_WINRT
		LONG upperBits = (LONG)((offset>>32) & 0x7fffffff); // Most sig. 32 bits of new offset
		LONG lowerBits = (LONG)(offset & 0xffffffff); // Least sig. 32 bits of new offset
		// API oddity: If successful, SetFilePointer() returns a dword containing the lower 32-bits of the new file-offset. Or, if it fails,
		// it returns INVALID_SET_FILE_POINTER. However according to MSDN, INVALID_SET_FILE_POINTER may also be a valid new offset. So to determine 
		// whether an error has actually occurred, it is also necessary to call GetLastError().
		DWORD dwRet = osSetFilePointer(file->H, lowerBits, &upperBits, FILE_BEGIN); // Value returned by SetFilePointer()
		DWORD lastErrno; // Value returned by GetLastError()
		if (dwRet == INVALID_SET_FILE_POINTER && (lastErrno = osGetLastError()) != NO_ERROR)
		{
			file->LastErrno = lastErrno;
			winLogError(RC_IOERR_SEEK, file->LastErrno, "seekWinFile", file->Path);
			return 1;
		}
		return 0;
#else
		// Same as above, except that this implementation works for WinRT.
		LARGE_INTEGER x; // The new offset
		x.QuadPart = offset; 
		BOOL ret = osSetFilePointerEx(file->H, x, 0, FILE_BEGIN); // Value returned by SetFilePointerEx()
		if (!ret)
		{
			file->LastErrno = osGetLastError();
			winLogError(RC_IOERR_SEEK, file->LastErrno, "seekWinFile", file->Path);
			return 1;
		}
		return 0;
#endif
	}

#define MX_CLOSE_ATTEMPT 3
	RC WinVFile::Close_()
	{
#ifndef OMIT_WAL
		_assert(Shm == 0);
#endif
		OSTRACE("CLOSE %d\n", H);
		_assert(H != NULL && H != INVALID_HANDLE_VALUE);
		int rc;
		int cnt = 0;
		do
		{
			rc = osCloseHandle(H);
		} while (!rc && ++cnt < MX_CLOSE_ATTEMPT && (win32_Sleep(100), 1));
#if OS_WINCE
#define WINCE_DELETION_ATTEMPTS 3
		winceDestroyLock(this);
		if (DeleteOnClose)
		{
			int cnt = 0;
			while (osDeleteFileW(DeleteOnClose) == 0 && osGetFileAttributesW(DeleteOnClose) != 0xffffffff && cnt++ < WINCE_DELETION_ATTEMPTS)
				win32_Sleep(100);  // Wait a little before trying again
			_free(DeleteOnClose);
		}
#endif
		OSTRACE("CLOSE %d %s\n", H, rc ? "ok" : "failed");
		if (rc)
			H = NULL;
		OpenCounter(-1);
		return (rc ? RC_OK : winLogError(RC_IOERR_CLOSE, osGetLastError(), "winClose", Path));
	}

	RC WinVFile::Read(void *buffer, int amount, int64 offset)
	{
#if !OS_WINCE
		OVERLAPPED overlapped; // The offset for ReadFile.
#endif
		int retry = 0; // Number of retrys
		SimulateIOError(return RC_IOERR_READ);
		OSTRACE("READ %d lock=%d\n", H, Lock_);
		DWORD read; // Number of bytes actually read from file
#if OS_WINCE
		if (seekWinFile(this, offset))
			return RC_FULL;
		while (!osReadFile(H, buffer, amount, &read, 0))
		{
#else
		_memset(&overlapped, 0, sizeof(OVERLAPPED));
		overlapped.Offset = (LONG)(offset & 0xffffffff);
		overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
		while (!osReadFile(H, buffer, amount, &read, &overlapped) && osGetLastError() != ERROR_HANDLE_EOF)
		{
#endif
			DWORD lastErrno;
			if (retryIoerr(&retry, &lastErrno)) continue;
			LastErrno = lastErrno;
			return winLogError(RC_IOERR_READ, LastErrno, "winRead", Path);
		}
		logIoerr(retry);
		if (read < (DWORD)amount)
		{
			// Unread parts of the buffer must be zero-filled
			_memset(&((char *)buffer)[read], 0, amount - read);
			return RC_IOERR_SHORT_READ;
		}
		return RC_OK;
	}

	RC WinVFile::Write(const void *buffer, int amount, int64 offset)
	{
		_assert(amount > 0);
		SimulateIOError(return RC_IOERR_WRITE);
		SimulateDiskfullError(return RC_FULL);
		OSTRACE("WRITE %d lock=%d\n", H, Lock_);
		int rc = 0; // True if error has occurred, else false
		int retry = 0; // Number of retries
#if OS_WINCE
		rc = seekWinFile(this, offset);
		if (!rc)
		{
#else
		{
#endif
#if !OS_WINCE
			OVERLAPPED overlapped; // The offset for WriteFile.
			memset(&overlapped, 0, sizeof(OVERLAPPED));
			overlapped.Offset = (LONG)(offset & 0xffffffff);
			overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
#endif
			uint8 *remain = (uint8 *)buffer; // Data yet to be written
			int remainLength = amount; // Number of bytes yet to be written
			DWORD write; // Bytes written by each WriteFile() call
			DWORD lastErrno = NO_ERROR; // Value returned by GetLastError()
			while (remainLength > 0)
			{
#if OS_WINCE
				if (!osWriteFile(H, remain, remainLength, &write, 0)) {
#else
				if (!osWriteFile(H, remain, remainLength, &write, &overlapped)) {
#endif
					if (retryIoerr(&retry, &lastErrno)) continue;
					break;
				}
				_assert(write == 0 || write <= (DWORD)remainLength);
				if (write == 0 || write > (DWORD)remainLength)
				{
					lastErrno = osGetLastError();
					break;
				}
#if !OS_WINCE
				offset += write;
				overlapped.Offset = (LONG)(offset & 0xffffffff);
				overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
#endif
				remain += write;
				remainLength -= write;
			}
			if (remainLength > 0)
			{
				LastErrno = lastErrno;
				rc = 1;
			}
		}
		if (rc)
		{
			if (LastErrno == ERROR_HANDLE_DISK_FULL ||  LastErrno == ERROR_DISK_FULL)
				return RC_FULL;
			return winLogError(RC_IOERR_WRITE, LastErrno, "winWrite", Path);
		}
		else
			logIoerr(retry);
		return RC_OK;
	}

	RC WinVFile::Truncate(int64 size)
	{
		RC rc = RC_OK;
		OSTRACE("TRUNCATE %d %lld\n", H, size);
		SimulateIOError(return RC_IOERR_TRUNCATE);
		// If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
		// actual file size after the operation may be larger than the requested size).
		if (SizeChunk > 0)
			size = ((size+SizeChunk-1)/SizeChunk)*SizeChunk;
		// SetEndOfFile() returns non-zero when successful, or zero when it fails.
		if (seekWinFile(this, size))
			rc = winLogError(RC_IOERR_TRUNCATE, LastErrno, "winTruncate1", Path);
		else if (!osSetEndOfFile(H))
		{
			LastErrno = osGetLastError();
			rc = winLogError(RC_IOERR_TRUNCATE, LastErrno, "winTruncate2", Path);
		}
		OSTRACE("TRUNCATE %d %lld %s\n", H, size, rc ? "failed" : "ok");
		return rc;
	}

#ifdef TEST
	// Count the number of fullsyncs and normal syncs.  This is used to test that syncs and fullsyncs are occuring at the right times.
	int sync_count = 0;
	int fullsync_count = 0;
#endif

	RC WinVFile::Sync(SYNC flags)
	{
		// Check that one of SQLITE_SYNC_NORMAL or FULL was passed
		_assert((flags&0x0F) == SYNC_NORMAL || (flags&0x0F) == SYNC_FULL);
		OSTRACE("SYNC %d lock=%d\n", H, Lock_);
		// Unix cannot, but some systems may return SQLITE_FULL from here. This line is to test that doing so does not cause any problems.
		SimulateDiskfullError(return RC_FULL);
#ifdef TEST
		if ((flags&0x0F) == SYNC_FULL)
			fullsync_count++;
		sync_count++;
#endif
#ifdef NO_SYNC // If we compiled with the SQLITE_NO_SYNC flag, then syncing is a no-op
		return RC_OK;
#else
		BOOL rc = osFlushFileBuffers(H);
		SimulateIOError(rc = FALSE);
		if (rc)
			return RC_OK;
		LastErrno = osGetLastError();
		return winLogError(RC_IOERR_FSYNC, LastErrno, "winSync", Path);
#endif
	}

	RC WinVFile::get_FileSize(int64 &size)
	{
		RC rc = RC_OK;
		SimulateIOError(return RC_IOERR_FSTAT);
#if OS_WINRT
		{
			FILE_STANDARD_INFO info;
			if (osGetFileInformationByHandleEx(H, FileStandardInfo, &info, sizeof(info)))
				size = info.EndOfFile.QuadPart;
			else
			{
				LastErrno = osGetLastError();
				rc = winLogError(RC_IOERR_FSTAT, LastErrno, "winFileSize", Path);
			}
		}
#else
		{
			DWORD upperBits;
			DWORD lowerBits = osGetFileSize(H, &upperBits);
			size = (((int64)upperBits)<<32) + lowerBits;
			DWORD lastErrno;
			if (lowerBits == INVALID_FILE_SIZE && (lastErrno = osGetLastError()) != NO_ERROR)
			{
				LastErrno = lastErrno;
				rc = winLogError(RC_IOERR_FSTAT, LastErrno, "winFileSize", Path);
			}
		}
#endif
		return rc;
	}

	static int getReadLock(WinVFile *file)
	{
		int res;
		if (isNT())
		{
#if OS_WINCE
			// NOTE: Windows CE is handled differently here due its lack of the Win32 API LockFileEx.
			res = winceLockFile(&file->H, SHARED_FIRST, 0, 1, 0);
#else
			res = winLockFile(&file->H, LOCKFILEEX_FLAGS, SHARED_FIRST, 0, SHARED_SIZE, 0);
#endif
		}
#ifdef WIN32_HAS_ANSI
		else
		{
			int lock;
			SysEx::PutRandom(sizeof(lock), &lock);
			file->SharedLockByte = (short)((lock & 0x7fffffff)%(SHARED_SIZE - 1));
			res = winLockFile(&file->H, LOCKFILE_FLAGS, SHARED_FIRST + file->SharedLockByte, 0, 1, 0);
		}
#endif
		if (res == 0)
			file->LastErrno = osGetLastError();
		// No need to log a failure to lock
		return res;
	}

	static int unlockReadLock(WinVFile *file)
	{
		int res;
		if (isNT())
			res = winUnlockFile(&file->H, SHARED_FIRST, 0, SHARED_SIZE, 0);
#ifdef WIN32_HAS_ANSI
		else
			res = winUnlockFile(&file->H, SHARED_FIRST + file->SharedLockByte, 0, 1, 0);
#endif
		DWORD lastErrno;
		if (res == 0 && (lastErrno = osGetLastError()) != ERROR_NOT_LOCKED)
		{
			file->LastErrno = lastErrno;
			winLogError(RC_IOERR_UNLOCK, file->LastErrno, "unlockReadLock", file->Path);
		}
		return res;
	}

	RC WinVFile::Lock(LOCK lock)
	{
		OSTRACE("LOCK %d %d was %d(%d)\n", H, lock, Lock_, SharedLockByte);

		// If there is already a lock of this type or more restrictive on the OsFile, do nothing. Don't use the end_lock: exit path, as
		// sqlite3OsEnterMutex() hasn't been called yet.
		if (Lock_ >= lock)
			return RC_OK;

		// Make sure the locking sequence is correct
		_assert(Lock_ != LOCK_NO || lock == LOCK_SHARED);
		_assert(lock != LOCK_PENDING);
		_assert(lock != LOCK_RESERVED || Lock_ == LOCK_SHARED);

		// Lock the PENDING_LOCK byte if we need to acquire a PENDING lock or a SHARED lock.  If we are acquiring a SHARED lock, the acquisition of
		// the PENDING_LOCK byte is temporary.
		LOCK newLock = Lock_; // Set pFile->locktype to this value before exiting
		int res = 1; // Result of a Windows lock call
		bool gotPendingLock = false; // True if we acquired a PENDING lock this time
		DWORD lastErrno = NO_ERROR;
		if (Lock_ == LOCK_NO || (lock == LOCK_EXCLUSIVE && Lock_ == LOCK_RESERVED))
		{
			int cnt = 3;
			while (cnt-- > 0 && (res = winLockFile(&H, LOCKFILE_FLAGS, PENDING_BYTE, 0, 1, 0)) == 0)
			{
				// Try 3 times to get the pending lock.  This is needed to work around problems caused by indexing and/or anti-virus software on Windows systems.
				// If you are using this code as a model for alternative VFSes, do not copy this retry logic.  It is a hack intended for Windows only.
				OSTRACE("could not get a PENDING lock. cnt=%d\n", cnt);
				if (cnt) win32_Sleep(1);
			}
			gotPendingLock = (res != 0);
			if (!res)
				lastErrno = osGetLastError();
		}

		// Acquire a SHARED lock
		if (lock == LOCK_SHARED && res)
		{
			_assert(Lock_ == LOCK_NO);
			res = getReadLock(this);
			if (res)
				newLock = LOCK_SHARED;
			else
				lastErrno = osGetLastError();
		}

		// Acquire a RESERVED lock
		if (lock == LOCK_RESERVED && res)
		{
			_assert(Lock_ == LOCK_SHARED);
			res = winLockFile(&H, LOCKFILE_FLAGS, RESERVED_BYTE, 0, 1, 0);
			if (res)
				newLock = LOCK_RESERVED;
			else
				lastErrno = osGetLastError();
		}

		// Acquire a PENDING lock
		if (lock == LOCK_EXCLUSIVE && res)
		{
			newLock = LOCK_PENDING;
			gotPendingLock = false;
		}

		// Acquire an EXCLUSIVE lock
		if (lock == LOCK_EXCLUSIVE && res)
		{
			_assert(Lock_ >= LOCK_SHARED);
			res = unlockReadLock(this);
			OSTRACE("unreadlock = %d\n", res);
			res = winLockFile(&H, LOCKFILE_FLAGS, SHARED_FIRST, 0, SHARED_SIZE, 0);
			if (res)
				newLock = LOCK_EXCLUSIVE;
			else
			{
				lastErrno = osGetLastError();
				OSTRACE("error-code = %d\n", lastErrno);
				getReadLock(this);
			}
		}

		// If we are holding a PENDING lock that ought to be released, then release it now.
		if (gotPendingLock && lock == LOCK_SHARED)
			winUnlockFile(&H, PENDING_BYTE, 0, 1, 0);

		// Update the state of the lock has held in the file descriptor then return the appropriate result code.
		RC rc;
		if (res)
			rc = RC_OK;
		else
		{
			OSTRACE("LOCK FAILED %d trying for %d but got %d\n", H, lock, newLock);
			LastErrno = lastErrno;
			rc = RC_BUSY;
		}
		Lock_ = newLock;
		return rc;
	}

	RC WinVFile::CheckReservedLock(int &lock)
	{
		SimulateIOError(return RC_IOERR_CHECKRESERVEDLOCK;);
		int rc;
		if (Lock_ >= LOCK_RESERVED)
		{
			rc = 1;
			OSTRACE("TEST WR-LOCK %d %d (local)\n", H, rc);
		}
		else
		{
			rc = winLockFile(&H, LOCKFILEEX_FLAGS, RESERVED_BYTE, 0, 1, 0);
			if (rc)
				winUnlockFile(&H, RESERVED_BYTE, 0, 1, 0);
			rc = !rc;
			OSTRACE("TEST WR-LOCK %d %d (remote)\n", H, rc);
		}
		lock = rc;
		return RC_OK;
	}

	RC WinVFile::Unlock(LOCK lock)
	{
		_assert(lock <= LOCK_SHARED);
		OSTRACE("UNLOCK %d to %d was %d(%d)\n", H, lock, Lock_, SharedLockByte);
		RC rc = RC_OK;
		LOCK type = Lock_;
		if (type >= LOCK_EXCLUSIVE)
		{
			winUnlockFile(&H, SHARED_FIRST, 0, SHARED_SIZE, 0);
			if (lock == LOCK_SHARED && !getReadLock(this)) // This should never happen.  We should always be able to reacquire the read lock
				rc = winLogError(RC_IOERR_UNLOCK, osGetLastError(), "winUnlock", Path);
		}
		if (type >= LOCK_RESERVED)
			winUnlockFile(&H, RESERVED_BYTE, 0, 1, 0);
		if (lock == LOCK_NO && type >= LOCK_SHARED)
			unlockReadLock(this);
		if (type >= LOCK_PENDING)
			winUnlockFile(&H, PENDING_BYTE, 0, 1, 0);
		Lock_ = lock;
		return rc;
	}

	static void winModeBit(WinVFile *file, uint8 mask, int *arg)
	{
		if (*arg < 0)
			*arg = ((file->CtrlFlags & mask) != 0);
		else if ((*arg) == 0)
			file->CtrlFlags = (WinVFile::WINFILE)(file->CtrlFlags & ~mask);
		else
			file->CtrlFlags |= mask;
	}

	static RC getTempname(int bufLength, char *buf);
	RC WinVFile::FileControl(FCNTL op, void *arg)
	{
		int *a;
		char *tfile;
		switch (op)
		{
		case FCNTL_LOCKSTATE:
			*(int*)arg = Lock_;
			return RC_OK;
		case FCNTL_LAST_ERRNO:
			*(int*)arg = (int)LastErrno;
			return RC_OK;
		case FCNTL_CHUNK_SIZE:
			SizeChunk = *(int *)arg;
			return RC_OK;
		case FCNTL_SIZE_HINT:
			if (SizeChunk > 0)
			{
				int64 oldSize;
				RC rc = get_FileSize(oldSize);
				if (rc == RC_OK)
				{
					int64 newSize = *(int64 *)arg;
					if (newSize > oldSize)
					{
						SimulateIOErrorBenign(true);
						rc = Truncate(newSize);
						SimulateIOErrorBenign(false);
					}
				}
				return rc;
			}
			return RC_OK;
		case FCNTL_PERSIST_WAL:
			winModeBit(this, (uint8)WINFILE_PERSIST_WAL, (int*)arg);
			return RC_OK;
		case FCNTL_POWERSAFE_OVERWRITE:
			winModeBit(this, (uint8)WINFILE_PSOW, (int*)arg);
			return RC_OK;
		case FCNTL_VFSNAME:
			*(char**)arg = "win32";
			return RC_OK;
		case FCNTL_WIN32_AV_RETRY:
			a = (int*)arg;
			if (a[0] > 0)
				win32IoerrRetry = a[0];
			else
				a[0] = win32IoerrRetry;
			if (a[1] > 0)
				win32IoerrRetryDelay = a[1];
			else
				a[1] = win32IoerrRetryDelay;
			return RC_OK;
		case FCNTL_TEMPFILENAME:
			tfile = (char *)_alloc2(Vfs->MaxPathname, true);
			if (tfile)
			{
				getTempname(Vfs->MaxPathname, tfile);
				*(char**)arg = tfile;
			}
			return RC_OK;
		}
		return RC_NOTFOUND;
	}

	uint WinVFile::get_SectorSize()
	{
		return DEFAULT_SECTOR_SIZE;
	}

	VFile::IOCAP WinVFile::get_DeviceCharacteristics()
	{
		return (VFile::IOCAP)(VFile::IOCAP_UNDELETABLE_WHEN_OPEN | ((CtrlFlags & WINFILE_PSOW) ? VFile::IOCAP_POWERSAFE_OVERWRITE : 0));
	}

#ifndef OMIT_WAL

	SYSTEM_INFO winSysInfo;
	static void winShmEnterMutex() { MutexEx::Enter(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER)); }
	static void winShmLeaveMutex() { MutexEx::Leave(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER)); }
#ifdef _DEBUG
	static bool winShmMutexHeld() { return MutexEx::Held(MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER)); }
#endif

	struct winShmNode
	{
		MutexEx Mutex;			// Mutex to access this object
		char *Filename;			// Name of the file
		WinVFile File;			// File handle from winOpen
		int SizeRegion;         // Size of shared-memory regions
		int RegionsLength;		// Size of array apRegion
		struct ShmRegion
		{
			HANDLE MapHandle;   // File handle from CreateFileMapping
			void *Map;
		} *Regions;
		DWORD LastErrno;		// The Windows errno from the last I/O error
		int Refs;               // Number of winShm objects pointing to this
		winShm *First;          // All winShm objects pointing to this
		winShmNode *Next;       // Next in list of all winShmNode objects
#ifdef _DEBUG
		uint8 NextShmID;        // Next available winShm.id value
#endif
	};

	static winShmNode *_winShmNodeList = 0;

	struct winShm
	{
		winShmNode *ShmNode;    // The underlying winShmNode object
		winShm *Next;           // Next winShm with the same winShmNode
		bool HasMutex;          // True if holding the winShmNode mutex
		uint16 SharedMask;      // Mask of shared locks held
		uint16 ExclMask;        // Mask of exclusive locks held
#ifdef _DEBUG
		uint8 ID;               // Id of this connection with its winShmNode
#endif
	};

#define WIN_SHM_BASE ((22+SHM_NLOCK)*4)        // first lock byte
#define WIN_SHM_DMS (WIN_SHM_BASE+SHM_NLOCK)  // deadman switch

	enum _SHM
	{
		_SHM_UNLCK = 1,
		_SHM_RDLCK = 2,
		_SHM_WRLCK = 3,
	};

	static int winShmSystemLock(winShmNode *file, _SHM lock, int offset, int bytes)
	{
		// Access to the winShmNode object is serialized by the caller
		_assert(MutexEx::Held(file->Mutex) || file->Refs == 0);
		// Release/Acquire the system-level lock
		int rc = 0; // Result code form Lock/UnlockFileEx()
		if (lock == _SHM_UNLCK)
			rc = winUnlockFile(&file->File.H, offset, 0, bytes, 0);
		else
		{
			// Initialize the locking parameters
			DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
			if (lock == _SHM_WRLCK) flags |= LOCKFILE_EXCLUSIVE_LOCK;
			rc = winLockFile(&file->File.H, flags, offset, 0, bytes, 0);
		}
		if (rc)
			rc = RC_OK;
		else
		{
			file->LastErrno = osGetLastError();
			rc = RC_BUSY;
		}
		OSTRACE("SHM-LOCK %d %s %s 0x%08lx\n", file->File.H, rc == RC_OK ? "ok" : "failed", lock == _SHM_UNLCK ? "UnlockFileEx" : "LockFileEx", file->LastErrno);
		return rc;
	}

	//// Forward references to VFS methods
	//static int winOpen(VSystem *, const char *, VFile *, int, int *);
	//static int winDelete(VSystem **, const char *, int);

	static void winShmPurge(VSystem *vfs, bool deleteFlag)
	{
		_assert(winShmMutexHeld());
		winShmNode **pp = &_winShmNodeList;
		winShmNode *p;
		while ((p = *pp) != nullptr)
			if (p->Refs == 0)
			{
				//if (p->Mutex) MutexEx::Free(p->Mutex);
				for (int i = 0; i < p->RegionLength; i++)
				{
					BOOL rc = osUnmapViewOfFile(p->Regions[i].Map);
					OSTRACE("SHM-PURGE pid-%d unmap region=%d %s\n", (int)osGetCurrentProcessId(), i, rc ? "ok" : "failed");
					rc = osCloseHandle(p->Regions[i].Map);
					OSTRACE("SHM-PURGE pid-%d close region=%d %s\n", (int)osGetCurrentProcessId(), i, rc ? "ok" : "failed");
				}
				if (p->File.H != NULL && p->File.H != INVALID_HANDLE_VALUE)
				{
					SimulateIOErrorBenign(true);
					p->File.Close_();
					SimulateIOErrorBenign(false);
				}
				if (deleteFlag)
				{
					SimulateIOErrorBenign(true);
					_benignalloc_begin();
					vfs->Delete(p->Filename, false);
					_benignalloc_end();
					SimulateIOErrorBenign(false);
				}
				*pp = p->Next;
				_free(p->Regions);
				_free(p);
			}
			else
				pp = &p->Next;
	}

	static RC winOpenSharedMemory(WinVFile *file)
	{
		_assert(file->Shm == nullptr); // Not previously opened

		// Allocate space for the new sqlite3_shm object.  Also speculatively allocate space for a new winShmNode and filename.
		struct winShm *p = (struct winShm *)_alloc(sizeof(*p), true); // The connection to be opened */
		if (!p) return RC_IOERR_NOMEM;
		int nameLength = _strlen30(file->Path); // Size of zName in bytes
		struct winShmNode *shmNode; // The underlying mmapped file
		struct winShmNode *newNode = (struct winShmNode *)_alloc(sizeof(*shmNode) + nameLength + 17, true); // Newly allocated winShmNode
		if (!newNode)
		{
			_free(p);
			return RC_IOERR_NOMEM;
		}
		newNode->Filename = (char *)&newNode[1];
		_snprintf(newNode->Filename, nameLength + 15, "%s-shm", file->Path);
		sqlite3FileSuffix3(file->Path, newNode->Filename); 

		// Look to see if there is an existing winShmNode that can be used. If no matching winShmNode currently exists, create a new one.
		winShmEnterMutex();
		for (shmNode = _winShmNodeList; shmNode; shmNode = shmNode->Next) // TBD need to come up with better match here.  Perhaps use FILE_ID_BOTH_DIR_INFO Structure.
			if (_strICmp(shmNode->Filename, newNode->Filename) == 0) break;
		RC rc;
		if (shmNode)
			_free(newNode);
		else
		{
			shmNode = newNode;
			newNode = nullptr;
			shmNode->File->H = INVALID_HANDLE_VALUE;
			shmNode->Next = _winShmNodeList;
			_winShmNodeList = shmNode;
			shmNode->Mutex = MutexEx::Alloc(MutexEx::MUTEX_FAST);
			rc = file->Vfs->Open(shmNode->Filename, &shmNode->File, VSystem::OPEN_WAL | VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE, nullptr);
			if (rc != RC_OK)
				goto shm_open_err;
			// Check to see if another process is holding the dead-man switch. If not, truncate the file to zero length. 
			if (winShmSystemLock(shmNode, _SHM_WRLCK, WIN_SHM_DMS, 1) == RC_OK)
			{
				rc = shmNode->File.Truncate(0);
				if (rc != RC_OK)
					rc = winLogError(RC_IOERR_SHMOPEN, osGetLastError(), "winOpenShm", file->Path);
			}
			if (rc == RC_OK)
			{
				winShmSystemLock(shmNode, _SHM_UNLCK, WIN_SHM_DMS, 1);
				rc = winShmSystemLock(shmNode, _SHM_RDLCK, WIN_SHM_DMS, 1);
			}
			if (rc) goto shm_open_err;
		}
		// Make the new connection a child of the winShmNode
		p->ShmNode = shmNode;
#ifdef _DEBUG
		p->ID = shmNode->NextShmID++;
#endif
		shmNode->Refs++;
		file->Shm = p;
		winShmLeaveMutex();

		// The reference count on pShmNode has already been incremented under the cover of the winShmEnterMutex() mutex and the pointer from the
		// new (struct winShm) object to the pShmNode has been set. All that is left to do is to link the new object into the linked list starting
		// at pShmNode->pFirst. This must be done while holding the pShmNode->mutex mutex.
		MutexEx::Enter(shmNode->Mutex);
		p->Next = shmNode->First;
		shmNode->First = p;
		MutexEx::Leave(shmNode->Mutex);
		return RC_OK;

		// Jump here on any error
shm_open_err:
		winShmSystemLock(shmNode, _SHM_UNLCK, WIN_SHM_DMS, 1);
		winShmPurge(file->Vfs, 0); // This call frees pShmNode if required
		_free(p);
		_free(newNode);
		winShmLeaveMutex();
		return rc;
	}

	RC WinVFile::ShmUnmap(bool deleteFlag)
	{
		winShm *p = Shm; // The connection to be closed
		if (p == nullptr) return RC_OK;
		winShmNode *shmNode = p->ShmNode; // The underlying shared-memory file

		// Remove connection p from the set of connections associated with pShmNode
		MutexEx::Enter(shmNode->Mutex);
		winShm **pp;
		for (pp = &shmNode->First; (*pp) != p; pp = &(*pp)->Next) { }
		*pp = p->Next;
		_free(p); // Free the connection p
		Shm = nullptr;
		MutexEx::Leave(shmNode->Mutex);

		// If pShmNode->nRef has reached 0, then close the underlying shared-memory file, too
		winShmEnterMutex();
		_assert(shmNode->Refs > 0);
		shmNode->Refs--;
		if (shmNode->Refs == 0)
			winShmPurge(Vfs, deleteFlag);
		winShmLeaveMutex();
		return RC_OK;
	}

	RC WinVFile::ShmLock(int offset, int count, _SHM flags)
	{
		_assert(offset >= 0 && offset+count <= _SHM_NLOCK);
		_assert(count >= 1);
		_assert(flags == (SHM_LOCK|SHM_SHARED) || flags == (SHM_LOCK|SHM_EXCLUSIVE) ||
			flags == (SHM_UNLOCK|SHM_SHARED) || flags == (SHM_UNLOCK|SHM_EXCLUSIVE));
		_assert(count == 1 || (flags & SHM_EXCLUSIVE) != 0);
		uint16 mask = (uint16)((1U<<(offset+count)) - (1U<<offset)); // Mask of locks to take or release
		_assert(count > 1 || mask == (1<<offset));
		RC rc = RC_OK;
		winShm *p = Shm; // The shared memory being locked
		winShmNode *shmNode = p->ShmNode;
		MutexEx::Enter(shmNode->Mutex);
		winShm *x;
		if (flags & SHM_UNLOCK)
		{
			// See if any siblings hold this same lock
			uint16 allMask = 0; // Mask of locks held by siblings
			for (x = shmNode->First; x; x = x->Next)
			{
				if (x == p) continue;
				_assert((x->ExclMask & (p->ExclMask|p->SharedMask)) == 0);
				allMask |= x->SharedMask;
			}
			// Unlock the system-level locks
			if ((mask & allMask) == 0)
				rc = winShmSystemLock(shmNode, _SHM_UNLCK, offset+WIN_SHM_BASE, count);
			else
				rc = RC_OK;
			// Undo the local locks
			if (rc == RC_OK)
			{
				p->ExclMask &= ~mask;
				p->SharedMask &= ~mask;
			} 
		}
		else if (flags & SHM_SHARED)
		{
			// Find out which shared locks are already held by sibling connections. If any sibling already holds an exclusive lock, go ahead and return SQLITE_BUSY.
			uint16 allShared = 0; // Union of locks held by connections other than "p"
			for (x = shmNode->First; x; x = x->Next)
			{
				if ((x->ExclMask & mask) != 0)
				{
					rc = RC_BUSY;
					break;
				}
				allShared |= x->SharedMask;
			}
			// Get shared locks at the system level, if necessary
			if (rc == RC_OK)
			{
				if ((allShared & mask) == 0)
					rc = winShmSystemLock(shmNode, _SHM_RDLCK, offset+WIN_SHM_BASE, count);
				else
					rc = RC_OK;
			}
			// Get the local shared locks
			if (rc == RC_OK)
				p->SharedMask |= mask;
		}
		else
		{
			// Make sure no sibling connections hold locks that will block this lock.  If any do, return SQLITE_BUSY right away.
			for (x = shmNode->First; x; x = x->Next)
				if ((x->ExclMask & mask) != 0 || (x->SharedMask & mask) != 0)
				{
					rc = RC_BUSY;
					break;
				}
				// Get the exclusive locks at the system level.  Then if successful also mark the local connection as being locked.
				if (rc == RC_OK)
				{
					rc = winShmSystemLock(shmNode, _SHM_WRLCK, offset+WIN_SHM_BASE, count);
					if (rc == RC_OK)
					{
						_assert((p->SharedMask & mask) == 0);
						p->ExclMask |= mask;
					}
				}
		}
		MutexEx::Leave(shmNode->Mutex);
		OSTRACE("SHM-LOCK shmid-%d, pid-%d got %03x,%03x %s\n", p->ID, (int)osGetCurrentProcessId(), p->SharedMask, p->ExclMask, rc ? "failed" : "ok");
		return rc;
	}

	void WinVFile::ShmBarrier()
	{
		// MemoryBarrier(); // does not work -- do not know why not
		winShmEnterMutex();
		winShmLeaveMutex();
	}

	int WinVFile::ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp)
	{
		RC rc = RC_OK;
		winShm *p = Shm;
		if (!p)
		{
			rc = winOpenSharedMemory(this);
			if (rc != RC_OK) return rc;
			p = Shm;
		}
		winShmNode *shmNode = p->ShmNode;

		MutexEx::Enter(shmNode->Mutex);
		_assert(sizeRegion == shmNode->SizeRegion || shmNode->RegionLength == 0);
		if (shmNode->RegionLength <= region)
		{
			shmNode->SizeRegion = sizeRegion;
			// The requested region is not mapped into this processes address space. Check to see if it has been allocated (i.e. if the wal-index file is
			// large enough to contain the requested region).
			int64 size; // Current size of wal-index file
			rc = shmNode->File->get_FileSize(&size);
			if (rc != RC_OK)
			{
				rc = winLogError(RC_IOERR_SHMSIZE, osGetLastError(), "winShmMap1", Path);
				goto shmpage_out;
			}
			int bytes = (region+1)*sizeRegion; // Minimum required file size
			if (size < bytes)
			{
				// The requested memory region does not exist. If isWrite is set to zero, exit early. *pp will be set to NULL and SQLITE_OK returned.
				// Alternatively, if isWrite is non-zero, use ftruncate() to allocate the requested memory region.
				if (!isWrite) goto shmpage_out;
				rc = shmNode->File->Truncate(bytes);
				if (rc != RC_OK)
				{
					rc = winLogError(RC_IOERR_SHMSIZE, osGetLastError(), "winShmMap2", Path);
					goto shmpage_out;
				}
			}

			// Map the requested memory region into this processes address space.
			struct ShmRegion *newRegions = (struct ShmRegion *)SysEx::Realloc(shmNode->Regions, (region+1)*sizeof(newRegions[0])); // New aRegion[] array
			if (!newRegions)
			{
				rc = RC_IOERR_NOMEM;
				goto shmpage_out;
			}
			shmNode->Regions = newRegions;
			while (shmNode->RegionLength <= region)
			{
				HANDLE mapHandle = NULL; // file-mapping handle
#if OS_WINRT
				mapHandle = osCreateFileMappingFromApp(shmNode->File.H, NULL, PAGE_READWRITE, bytes, NULL );
#elif defined(WIN32_HAS_WIDE)
				mapHandle = osCreateFileMappingW(shmNode->File.H, NULL, PAGE_READWRITE, 0, bytes, NULL);
#elif defined(WIN32_HAS_ANSI)
				mapHandle = osCreateFileMappingA(shmNode->File.H, NULL, PAGE_READWRITE, 0, bytes, NULL);
#endif
				OSTRACE("SHM-MAP pid-%d create region=%d nbyte=%d %s\n", (int)osGetCurrentProcessId(), shmNode->RegionLength, bytes, mapHandle ? "ok" : "failed");
				void *map = nullptr; // Mapped memory region
				if (mapHandle)
				{
					int offset = shmNode->RegionLength*sizeRegion;
					int offsetShift = offset % winSysInfo.dwAllocationGranularity;
#if OS_WINRT
					map = osMapViewOfFileFromApp(mapHandle, FILE_MAP_WRITE | FILE_MAP_READ, offset - offsetShift, sizeRegion + offsetShift);
#else
					map = osMapViewOfFile(mapHandle, FILE_MAP_WRITE | FILE_MAP_READ, 0, offset - offsetShift, sizeRegion + offsetShift);
#endif
					OSTRACE("SHM-MAP pid-%d map region=%d offset=%d size=%d %s\n", (int)osGetCurrentProcessId(), shmNode->RegionLength, offset, sizeRegion, map ? "ok" : "failed");
				}
				if (!map)
				{
					shmNode->LastErrno = osGetLastError();
					rc = winLogError(RC_IOERR_SHMMAP, shmNode->LastErrno, "winShmMap3", Path);
					if (mapHandle) osCloseHandle(mapHandle);
					goto shmpage_out;
				}
				shmNode->Regions[shmNode->RegionLength].Map = map;
				shmNode->Regions[shmNode->RegionLength].MapHandle = mapHandle;
				shmNode->RegionLength++;
			}
		}

shmpage_out:
		if (shmNode->RegionLength > region)
		{
			int offset = region*sizeRegion;
			int offsetShift = offset % winSysInfo.dwAllocationGranularity;
			char *p = (char *)shmNode->Regions[region].Map;
			*pp = (void *)&p[offsetShift];
		}
		else
			*pp = nullptr;
		MutexEx::Leave(shmNode->Mutex);
		return rc;
	}

#endif

#pragma endregion

#pragma region WinVSystem

	static void *ConvertUtf8Filename(const char *name)
	{
		void *converted = nullptr;
		if (isNT())
			converted = Utf8ToUnicode(name);
#ifdef WIN32_HAS_ANSI
		else
			converted = win32_Utf8ToMbcs(name);
#endif
		// caller will handle out of memory
		return converted;
	}

	static RC getTempname(int bufLength, char *buf)
	{
		static char chars[] =
			"abcdefghijklmnopqrstuvwxyz"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"0123456789";
		// It's odd to simulate an io-error here, but really this is just using the io-error infrastructure to test that SQLite handles this function failing.
		SimulateIOError(return RC_IOERR);
		char tempPath[MAX_PATH+2];
		memset(tempPath, 0, MAX_PATH+2);
		if (temp_directory)
			__snprintf(tempPath, MAX_PATH-30, "%s", temp_directory);
#if !OS_WINRT
		else if (isNT())
		{
			char *multi;
			WCHAR widePath[MAX_PATH];
			osGetTempPathW(MAX_PATH-30, widePath);
			multi = UnicodeToUtf8(widePath);
			if (multi)
			{
				__snprintf(tempPath, MAX_PATH-30, "%s", multi);
				_free(multi);
			}
			else
				return RC_IOERR_NOMEM;
		}
#ifdef WIN32_HAS_ANSI
		else
		{
			char *utf8;
			char mbcsPath[MAX_PATH];
			osGetTempPathA(MAX_PATH-30, mbcsPath);
			utf8 = win32_MbcsToUtf8(mbcsPath);
			if (utf8)
			{
				__snprintf(tempPath, MAX_PATH-30, "%s", utf8);
				_free(utf8);
			}
			else
				return RC_IOERR_NOMEM;
		}
#endif
#endif
		// Check that the output buffer is large enough for the temporary file name. If it is not, return SQLITE_ERROR.
		int tempPathLength = _strlen30(tempPath);
		if ((tempPathLength + _strlen30(TEMP_FILE_PREFIX) + 18) >= bufLength)
			return RC_ERROR;
		size_t i;
		for (i = tempPathLength; i > 0 && tempPath[i-1] == '\\'; i--) { }
		tempPath[i] = 0;
		size_t j;
		__snprintf(buf, bufLength-18, (tempPathLength > 0 ? "%s\\"TEMP_FILE_PREFIX : TEMP_FILE_PREFIX, tempPath));
		j = _strlen30(buf);
		SysEx::PutRandom(15, &buf[j]);
		for (i = 0; i < 15; i++, j++)
			buf[j] = (char)chars[((unsigned char)buf[j])%(sizeof(chars)-1)];
		buf[j] = 0;
		buf[j+1] = 0;
		OSTRACE("TEMP FILENAME: %s\n", buf);
		return RC_OK; 
	}

	static int winIsDir(const void *converted)
	{
		DWORD attr;
		if (isNT())
		{
			int cnt = 0;
			WIN32_FILE_ATTRIBUTE_DATA sAttrData;
			memset(&sAttrData, 0, sizeof(sAttrData));
			int rc = 0;
			DWORD lastErrno;
			while (!(rc = osGetFileAttributesExW((LPCWSTR)converted, GetFileExInfoStandard, &sAttrData)) && retryIoerr(&cnt, &lastErrno)) { }
			if (!rc)
				return 0; // Invalid name?
			attr = sAttrData.dwFileAttributes;
#if OS_WINCE == 0
		}
		else
		{
			attr = osGetFileAttributesA((char*)converted);
#endif
		}
		return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
	}

	VFile *WinVSystem::_AttachFile(void *buffer)
	{
		return new (buffer) WinVFile();
	}

	RC WinVSystem::Open(const char *name, VFile *id, OPEN flags, OPEN *outFlags)
	{
		// 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
		// SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
		flags = (OPEN)((uint)flags & 0x87f7f);

		RC rc = RC_OK;
		OPEN type = (OPEN)(flags & 0xFFFFFF00);  // Type of file to open
		bool isExclusive = ((flags & OPEN_EXCLUSIVE) != 0);
		bool isDelete = ((flags & OPEN_DELETEONCLOSE) != 0);
		bool isCreate = ((flags & OPEN_CREATE) != 0);
		bool isReadonly = ((flags & OPEN_READONLY) != 0);
		bool isReadWrite = ((flags & OPEN_READWRITE) != 0);
		bool isOpenJournal = (isCreate && (type == OPEN_MASTER_JOURNAL || type == OPEN_MAIN_JOURNAL || type == OPEN_WAL));

		// Check the following statements are true: 
		//
		//   (a) Exactly one of the READWRITE and READONLY flags must be set, and 
		//   (b) if CREATE is set, then READWRITE must also be set, and
		//   (c) if EXCLUSIVE is set, then CREATE must also be set.
		//   (d) if DELETEONCLOSE is set, then CREATE must also be set.
		_assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
		_assert(!isCreate || isReadWrite);
		_assert(!isExclusive || isCreate);
		_assert(!isDelete || isCreate);

		// The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
		_assert((!isDelete && name) || type != OPEN_MAIN_DB);
		_assert((!isDelete && name) || type != OPEN_MAIN_JOURNAL);
		_assert((!isDelete && name) || type != OPEN_MASTER_JOURNAL);
		_assert((!isDelete && name) || type != OPEN_WAL);

		// Assert that the upper layer has set one of the "file-type" flags.
		_assert(type == OPEN_MAIN_DB || type == OPEN_TEMP_DB ||
			type == OPEN_MAIN_JOURNAL || type == OPEN_TEMP_JOURNAL ||
			type == OPEN_SUBJOURNAL || type == OPEN_MASTER_JOURNAL ||
			type == OPEN_TRANSIENT_DB || type == OPEN_WAL);

		WinVFile *file = (WinVFile *)id;
		_assert(file != nullptr);
		memset(file, 0, sizeof(WinVFile));
		file = new (file) WinVFile();
		file->H = INVALID_HANDLE_VALUE;

#if OS_WINRT
		if (!temp_directory)
			SysEx_LOG(RC_ERROR, "sqlite3_temp_directory variable should be set for WinRT");
#endif

		// If the second argument to this function is NULL, generate a temporary file name to use 
		const char *utf8Name = name; // Filename in UTF-8 encoding
		char tmpname[MAX_PATH+2];     // Buffer used to create temp filename
		if (!utf8Name)
		{
			_assert(isDelete && !isOpenJournal);
			memset(tmpname, 0, MAX_PATH+2);
			rc = getTempname(MAX_PATH+2, tmpname);
			if (rc != RC_OK)
				return rc;
			utf8Name = tmpname;
		}

		// Database filenames are double-zero terminated if they are not URIs with parameters.  Hence, they can always be passed into
		// sqlite3_uri_parameter().
		_assert(type != OPEN_MAIN_DB || (flags & OPEN_URI) || utf8Name[strlen(utf8Name)+1]==0);

		// Convert the filename to the system encoding.
		void *converted = ConvertUtf8Filename(utf8Name); // Filename in OS encoding
		if (!converted)
			return RC_IOERR_NOMEM;

		if (winIsDir(converted))
		{
			_free(converted);
			return RC_CANTOPEN_ISDIR;
		}

		DWORD dwDesiredAccess;
		if (isReadWrite)
			dwDesiredAccess = GENERIC_READ | GENERIC_WRITE;
		else
			dwDesiredAccess = GENERIC_READ;

		// SQLITE_OPEN_EXCLUSIVE is used to make sure that a new file is created. SQLite doesn't use it to indicate "exclusive access"
		// as it is usually understood.
		DWORD dwCreationDisposition;
		if (isExclusive) // Creates a new file, only if it does not already exist. If the file exists, it fails.
			dwCreationDisposition = CREATE_NEW;
		else if (isCreate) // Open existing file, or create if it doesn't exist
			dwCreationDisposition = OPEN_ALWAYS;
		else // Opens a file, only if it exists.
			dwCreationDisposition = OPEN_EXISTING;

		DWORD dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;

		DWORD dwFlagsAndAttributes = 0;
#if OS_WINCE
		int isTemp = 0;
#endif
		if (isDelete)
		{
#if OS_WINCE
			dwFlagsAndAttributes = FILE_ATTRIBUTE_HIDDEN;
			isTemp = 1;
#else
			dwFlagsAndAttributes = FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE;
#endif
		}
		else
			dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
		// Reports from the internet are that performance is always better if FILE_FLAG_RANDOM_ACCESS is used.  Ticket #2699.
#if OS_WINCE
		dwFlagsAndAttributes |= FILE_FLAG_RANDOM_ACCESS;
#endif

		HANDLE h;
		DWORD lastErrno;
		int cnt = 0;
		if (isNT())
		{
#if OS_WINRT
			CREATEFILE2_EXTENDED_PARAMETERS extendedParameters;
			extendedParameters.dwSize = sizeof(CREATEFILE2_EXTENDED_PARAMETERS);
			extendedParameters.dwFileAttributes = dwFlagsAndAttributes & FILE_ATTRIBUTE_MASK;
			extendedParameters.dwFileFlags = dwFlagsAndAttributes & FILE_FLAG_MASK;
			extendedParameters.dwSecurityQosFlags = SECURITY_ANONYMOUS;
			extendedParameters.lpSecurityAttributes = NULL;
			extendedParameters.hTemplateFile = NULL;
			while ((h = osCreateFile2((LPCWSTR)converted, dwDesiredAccess, dwShareMode, dwCreationDisposition, &extendedParameters)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
#else
			while ((h = osCreateFileW((LPCWSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
#endif
		}
#ifdef WIN32_HAS_ANSI
		else
			while ((h = osCreateFileA((LPCSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && retryIoerr(&cnt, &lastErrno)) { }
#endif
			logIoerr(cnt);

			OSTRACE("OPEN %d %s 0x%lx %s\n", h, name, dwDesiredAccess, h == INVALID_HANDLE_VALUE ? "failed" : "ok");
			if (h == INVALID_HANDLE_VALUE)
			{
				file->LastErrno = lastErrno;
				winLogError(RC_CANTOPEN, file->LastErrno, "winOpen", utf8Name);
				_free(converted);
				if (isReadWrite && !isExclusive)
					return Open(name, id, (OPEN)((flags|OPEN_READONLY) & ~(OPEN_CREATE|OPEN_READWRITE)), outFlags);
				else
					return SysEx_CANTOPEN_BKPT;
			}

			if (outFlags)
				*outFlags = (isReadWrite ? OPEN_READWRITE : OPEN_READONLY);
#if OS_WINCE
			if (isReadWrite && type == OPEN_MAIN_DB && (rc = winceCreateLock(name, file)) != RC_OK)
			{
				osCloseHandle(h);
				_free(converted);
				return rc;
			}
			if (isTemp)
				file->DeleteOnClose = converted;
			else
#endif
				_free(converted);
			file->Opened = true;
			file->Vfs = this;
			file->H = h;
			if (VSystem::UriBoolean(name, "psow", POWERSAFE_OVERWRITE))
				file->CtrlFlags |= WinVFile::WINFILE_PSOW;
			file->LastErrno = NO_ERROR;
			file->Path = name;
			OpenCounter(+1);
			return rc;
	}

	RC WinVSystem::Delete(const char *filename, bool syncDir)
	{
		SimulateIOError(return RC_IOERR_DELETE;);
		void *converted = ConvertUtf8Filename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		RC rc;
		DWORD lastErrno;
		int cnt = 0;
		if (isNT())
			do {
#if OS_WINRT
				WIN32_FILE_ATTRIBUTE_DATA sAttrData;
				memset(&sAttrData, 0, sizeof(sAttrData));
				if (osGetFileAttributesExW(converted, GetFileExInfoStandard, &sAttrData))
					attr = sAttrData.dwFileAttributes;
				else
				{
					lastErrno = osGetLastError();
					rc = (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND ? RC_IOERR_DELETE_NOENT : RC_ERROR); // Already gone?
					break;
				}
#else
				attr = osGetFileAttributesW((LPCWSTR)converted);
#endif
				if (attr == INVALID_FILE_ATTRIBUTES)
				{
					lastErrno = osGetLastError();
					rc = (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND ? RC_IOERR_DELETE_NOENT : RC_ERROR); // Already gone?
					break;
				}
				if (attr & FILE_ATTRIBUTE_DIRECTORY)
				{
					rc = RC_ERROR; // Files only.
					break;
				}
				if (osDeleteFileW((LPCWSTR)converted))
				{
					rc = RC_OK; // Deleted OK.
					break;
				}
				if (!retryIoerr(&cnt, &lastErrno))
				{
					rc = RC_ERROR; // No more retries.
					break;
				}
			} while(1);
#ifdef WIN32_HAS_ANSI
		else
			do {
				attr = osGetFileAttributesA((LPCSTR)converted);
				if (attr == INVALID_FILE_ATTRIBUTES)
				{
					lastErrno = osGetLastError();
					rc = (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND ? RC_IOERR_DELETE_NOENT : RC_ERROR); // Already gone?
					break;
				}
				if (attr & FILE_ATTRIBUTE_DIRECTORY)
				{
					rc = RC_ERROR; // Files only.
					break;
				}
				if (osDeleteFileA((LPCSTR)converted))
				{
					rc = RC_OK; // Deleted OK.
					break;
				}
				if (!retryIoerr(&cnt, &lastErrno))
				{
					rc = RC_ERROR; // No more retries.
					break;
				}
			} while(1);
#endif
		if (rc && rc != RC_IOERR_DELETE_NOENT)
			rc = winLogError(RC_IOERR_DELETE, lastErrno, "winDelete", filename);
		else
			logIoerr(cnt);
		_free(converted);
		OSTRACE("DELETE \"%s\" %s\n", filename, rc ? "failed" : "ok" );
		return rc;
	}

	RC WinVSystem::Access(const char *filename, ACCESS flags, int *resOut)
	{
		SimulateIOError(return RC_IOERR_ACCESS;);
		void *converted = ConvertUtf8Filename(filename);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD attr;
		int rc = 0;
		DWORD lastErrno;
		if (isNT())
		{
			int cnt = 0;
			WIN32_FILE_ATTRIBUTE_DATA sAttrData;
			memset(&sAttrData, 0, sizeof(sAttrData));
			while (!(rc = osGetFileAttributesExW((LPCWSTR)converted, GetFileExInfoStandard, &sAttrData)) && retryIoerr(&cnt, &lastErrno)) { }
			if (rc)
			{
				// For an SQLITE_ACCESS_EXISTS query, treat a zero-length file as if it does not exist.
				if (flags == ACCESS_EXISTS && sAttrData.nFileSizeHigh == 0  && sAttrData.nFileSizeLow == 0)
					attr = INVALID_FILE_ATTRIBUTES;
				else
					attr = sAttrData.dwFileAttributes;
			}
			else
			{
				logIoerr(cnt);
				if (lastErrno != ERROR_FILE_NOT_FOUND && lastErrno != ERROR_PATH_NOT_FOUND)
				{
					winLogError(RC_IOERR_ACCESS, lastErrno, "winAccess", filename);
					_free(converted);
					return RC_IOERR_ACCESS;
				}
				else
					attr = INVALID_FILE_ATTRIBUTES;
			}
		}
#ifdef WIN32_HAS_ANSI
		else
			attr = osGetFileAttributesA((char*)converted);
#endif
		_free(converted);
		switch (flags)
		{
		case ACCESS_READ:
		case ACCESS_EXISTS:
			rc = attr != INVALID_FILE_ATTRIBUTES;
			break;
		case ACCESS_READWRITE:
			rc = attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_READONLY) == 0;
			break;
		default:
			_assert(!"Invalid flags argument");
		}
		*resOut = rc;
		return RC_OK;
	}

	static BOOL winIsVerbatimPathname(const char *pathname)
	{
		// If the path name starts with a forward slash or a backslash, it is either a legal UNC name, a volume relative path, or an absolute path name in the
		// "Unix" format on Windows.  There is no easy way to differentiate between the final two cases; therefore, we return the safer return value of TRUE
		// so that callers of this function will simply use it verbatim.
		if (pathname[0] == '/' || pathname[0] == '\\')
			return TRUE;
		// If the path name starts with a letter and a colon it is either a volume relative path or an absolute path.  Callers of this function must not
		// attempt to treat it as a relative path name (i.e. they should simply use it verbatim).
		if (_isalpha2(pathname[0]) && pathname[1] == ':')
			return TRUE;
		// If we get to this point, the path name should almost certainly be a purely relative one (i.e. not a UNC name, not absolute, and not volume relative).
		return FALSE;
	}

	RC WinVSystem::FullPathname(const char *relative, int fullLength, char *full)
	{
#if defined(__CYGWIN__)
		SimulateIOError(return RC_ERROR);
		_assert(MaxPathname >= MAX_PATH);
		_assert(fullLength >= MaxPathname);
		// NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
		//       for converting the relative path name to an absolute one by prepending the data directory and a slash.
		if (data_directory && !winIsVerbatimPathname(relative))
		{
			char out[MAX_PATH+1];
			memset(out, 0, MAX_PATH+1);
			cygwin_conv_path(CCP_POSIX_TO_WIN_A|CCP_RELATIVE, relative, out, MAX_PATH+1);
			_snprintf(full, MIN(fullLength, MaxPathname), "%s\\%s", data_directory, out);
		}
		else
			cygwin_conv_path(CCP_POSIX_TO_WIN_A, relative, full, fullLength);
		return RC_OK;
#endif
#if (OS_WINCE || OS_WINRT) && !defined(__CYGWIN__)
		SimulateIOError(return RC_ERROR);
		// WinCE has no concept of a relative pathname, or so I am told.
		// WinRT has no way to convert a relative path to an absolute one.
		// NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
		//       for converting the relative path name to an absolute one by prepending the data directory and a backslash.
		if (data_directory && !winIsVerbatimPathname(relative))
			_snprintf(full, MIN(fullLength, MaxPathname), "%s\\%s", data_directory, relative);
		else
			_snprintf(full, MIN(fullLength, MaxPathname), "%s", relative);
		return RC_OK;
#endif
#if !OS_WINCE && !OS_WINRT && !defined(__CYGWIN__)
		// If this path name begins with "/X:", where "X" is any alphabetic character, discard the initial "/" from the pathname.
		if (relative[0] == '/' && _isalpha2(relative[1]) && relative[2] == ':')
			relative++;
		// It's odd to simulate an io-error here, but really this is just using the io-error infrastructure to test that SQLite handles this
		// function failing. This function could fail if, for example, the current working directory has been unlinked.
		SimulateIOError(return RC_ERROR);
		// NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
		//       for converting the relative path name to an absolute one by prepending the data directory and a backslash.
		if (data_directory && !winIsVerbatimPathname(relative))
		{
			_snprintf(full, MIN(fullLength, MaxPathname), "%s\\%s", data_directory, relative);
			return RC_OK;
		}
		void *converted = ConvertUtf8Filename(relative);
		if (!converted)
			return RC_IOERR_NOMEM;
		DWORD bytes;
		char *out;
		if (isNT())
		{
			LPWSTR temp;
			bytes = osGetFullPathNameW((LPCWSTR)converted, 0, 0, 0);
			if (bytes == 0)
			{
				winLogError(RC_ERROR, osGetLastError(), "GetFullPathNameW1", (char *)converted);
				_free(converted);
				return RC_CANTOPEN_FULLPATH;
			}
			bytes += 3;
			temp = (LPWSTR)_alloc2(bytes*sizeof(temp[0]), true);
			if (!temp)
			{
				_free(converted);
				return RC_IOERR_NOMEM;
			}
			bytes = osGetFullPathNameW((LPCWSTR)converted, bytes, temp, 0);
			if (bytes == 0)
			{
				winLogError(RC_ERROR, osGetLastError(), "GetFullPathNameW2", (char *)converted);
				_free(converted);
				_free(temp);
				return RC_CANTOPEN_FULLPATH;
			}
			_free(converted);
			out = UnicodeToUtf8(temp);
			_free(temp);
		}
#ifdef WIN32_HAS_ANSI
		else
		{
			char *temp;
			bytes = osGetFullPathNameA((char*)converted, 0, 0, 0);
			if (bytes == 0)
			{
				winLogError(RC_ERROR, osGetLastError(), "GetFullPathNameA1", (char *)converted);
				_free(converted);
				return RC_CANTOPEN_FULLPATH;
			}
			bytes += 3;
			temp = (char *)_alloc2(bytes*sizeof(temp[0]), true);
			if (!temp)
			{
				_free(converted);
				return RC_IOERR_NOMEM;
			}
			bytes = osGetFullPathNameA((char*)converted, bytes, temp, 0);
			if (bytes == 0)
			{
				winLogError(RC_ERROR, osGetLastError(), "GetFullPathNameA2", (char *)converted);
				_free(converted);
				_free(temp);
				return RC_CANTOPEN_FULLPATH;
			}
			_free(converted);
			out = win32_MbcsToUtf8(temp);
			_free(temp);
		}
#endif
		if (out)
		{
			_snprintf(full, MIN(fullLength, MaxPathname), "%s", out);
			_free(out);
			return RC_OK;
		}
		return RC_IOERR_NOMEM;
#endif
	}

#ifndef OMIT_LOAD_EXTENSION
	void *WinVSystem::DlOpen(const char *filename)
	{
		void *converted = ConvertUtf8Filename(filename);
		if (!converted)
			return nullptr;
		HANDLE h;
		if (isNT())
#if OS_WINRT
			h = osLoadPackagedLibrary((LPCWSTR)converted, 0);
#else
			h = osLoadLibraryW((LPCWSTR)converted);
#endif
#ifdef WIN32_HAS_ANSI
		else
			h = osLoadLibraryA((char*)converted);
#endif
		_free(converted);
		return (void *)h;
	}

	void WinVSystem::DlError(int bufLength, char *buf)
	{
		getLastErrorMsg(osGetLastError(), bufLength, buf);
	}

	void (*WinVSystem::DlSym(void *handle, const char *symbol))()
	{
		return (void(*)())osGetProcAddressA((HMODULE)handle, symbol);
	}

	void WinVSystem::DlClose(void *handle)
	{
		osFreeLibrary((HMODULE)handle);
	}
#else
#define winDlOpen  0
#define winDlError 0
#define winDlSym   0
#define winDlClose 0
#endif

	int WinVSystem::Randomness(int bufLength, char *buf)
	{
		int n = 0;
#if TEST
		n = bufLength;
		memset(buf, 0, bufLength);
#else
		if (sizeof(SYSTEMTIME) <= bufLength - n)
		{
			SYSTEMTIME x;
			osGetSystemTime(&x);
			memcpy(&buf[n], &x, sizeof(x));
			n += sizeof(x);
		}
		if (sizeof(DWORD) <= bufLength - n)
		{
			DWORD pid = osGetCurrentProcessId();
			memcpy(&buf[n], &pid, sizeof(pid));
			n += sizeof(pid);
		}
#if OS_WINRT
		if (sizeof(ULONGLONG) <= bufLength - n)
		{
			ULONGLONG cnt = osGetTickCount64();
			memcpy(&buf[n], &cnt, sizeof(cnt));
			n += sizeof(cnt);
		}
#else
		if (sizeof(DWORD) <= bufLength - n)
		{
			DWORD cnt = osGetTickCount();
			memcpy(&buf[n], &cnt, sizeof(cnt));
			n += sizeof(cnt);
		}
#endif
		if (sizeof(LARGE_INTEGER) <= bufLength - n)
		{
			LARGE_INTEGER i;
			osQueryPerformanceCounter(&i);
			memcpy(&buf[n], &i, sizeof(i));
			n += sizeof(i);
		}
#endif
		return n;
	}

	int WinVSystem::Sleep(int microseconds)
	{
		win32_Sleep((microseconds+999)/1000);
		return ((microseconds+999)/1000)*1000;
	}

#ifdef TEST
	int current_time = 0; // Fake system time in seconds since 1970.
#endif
	RC WinVSystem::CurrentTimeInt64(int64 *now)
	{
		// FILETIME structure is a 64-bit value representing the number of 100-nanosecond intervals since January 1, 1601 (= JD 2305813.5). 
		FILETIME ft;
		static const int64 winFiletimeEpoch = 23058135*(int64)8640000;
#ifdef TEST
		static const int64 unixEpoch = 24405875*(int64)8640000;
#endif
		// 2^32 - to avoid use of LL and warnings in gcc
		static const int64 max32BitValue = (int64)2000000000 + (int64)2000000000 + (int64)294967296;
#if OS_WINCE
		SYSTEMTIME time;
		osGetSystemTime(&time);
		// if SystemTimeToFileTime() fails, it returns zero.
		if (!osSystemTimeToFileTime(&time,&ft))
			return RC_ERROR;
#else
		osGetSystemTimeAsFileTime(&ft);
#endif
		*now = winFiletimeEpoch + ((((int64)ft.dwHighDateTime)*max32BitValue) + (int64)ft.dwLowDateTime)/(int64)10000;
#ifdef TEST
		if (current_time)
			*now = 1000*(int64)current_time + unixEpoch;
#endif
		return RC_OK;
	}

	RC WinVSystem::CurrentTime(double *now)
	{
		int64 i;
		RC rc = CurrentTimeInt64(&i);
		if (rc == RC_OK)
			*now = i/86400000.0;
		return rc;
	}

	RC WinVSystem::GetLastError(int bufLength, char *buf)
	{
		return getLastErrorMsg(osGetLastError(), bufLength, buf);
	}

	static WinVSystem _winVfs;
	RC VSystem::Initialize()
	{
		_winVfs.SizeOsFile = sizeof(WinVFile);
		_winVfs.MaxPathname = 260;
		_winVfs.Name = "win32";
		// Double-check that the aSyscall[] array has been constructed correctly.  See ticket [bb3a86e890c8e96ab]
		_assert(_lengthof(Syscalls) == 74);
#ifndef OMIT_WAL
		// get memory map allocation granularity
		memset(&winSysInfo, 0, sizeof(SYSTEM_INFO));
#if OS_WINRT
		osGetNativeSystemInfo(&winSysInfo);
#else
		osGetSystemInfo(&winSysInfo);
#endif
		_assert(winSysInfo.dwAllocationGranularity > 0);
#endif
		RegisterVfs(&_winVfs, true);
		return RC_OK; 
	}

	void VSystem::Shutdown()
	{ 
#if OS_WINRT
		if (sleepObj != NULL)
		{
			osCloseHandle(sleepObj);
			sleepObj = NULL;
		}
#endif
	}

#pragma endregion
}
#endif