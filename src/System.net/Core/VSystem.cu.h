// sqlite.h
#pragma once
namespace CORE_NAME
{
	// The default size of a disk sector
#ifndef DEFAULT_SECTOR_SIZE
#define DEFAULT_SECTOR_SIZE 4096
#endif

	// Temporary files are named starting with this prefix followed by 16 random alphanumeric characters, and no file extension. They are stored in the
	// OS's standard temporary file directory, and are deleted prior to exit. If sqlite is being embedded in another program, you may wish to change the
	// prefix to reflect your program's name, so that if your program exits prematurely, old temporary files can be easily identified. This can be done
	// using -DSQLITE_TEMP_FILE_PREFIX=myprefix_ on the compiler command line.
	//
	// 2006-10-31:  The default prefix used to be "sqlite_".  But then Mcafee started using SQLite in their anti-virus product and it
	// started putting files with the "sqlite" name in the c:/temp folder. This annoyed many windows users.  Those users would then do a 
	// Google search for "sqlite", find the telephone numbers of the developers and call to wake them up at night and complain.
	// For this reason, the default name prefix is changed to be "sqlite" spelled backwards.  So the temp files are still identified, but
	// anybody smart enough to figure out the code is also likely smart enough to know that calling the developer will not help get rid
	// of the file.
#ifndef TEMP_FILE_PREFIX
#define TEMP_FILE_PREFIX "etilqs_"
#endif

#ifdef _TESTX
	__device__ void DisableSimulatedIOErrors(int *pending = nullptr, int *hit = nullptr);
	__device__ void EnableSimulatedIOErrors(int *pending = nullptr, int *hit = nullptr);
#else
#define DisableSimulatedIOErrors(...)
#define EnableSimulatedIOErrors(...)
#endif

	typedef void (*syscall_ptr)();

	class VFile;
	class VSystem
	{
	public:
		class VAlloc
		{
		public:
			__device__ virtual void *Alloc(int bytes);			// Memory allocation function
			__device__ virtual void Free(void *prior);			// Free a prior allocation
			__device__ virtual void *Realloc(void *prior, int bytes);	// Resize an allocation
			__device__ virtual int Size(void *p);				// Return the size of an allocation
			__device__ virtual int Roundup(int bytes);			// Round up request size to allocation size
			__device__ virtual RC Init(void *appData);			// Initialize the memory allocator
			__device__ virtual void Shutdown(void *appData);	// Deinitialize the memory allocator
			void *AppData;										// Argument to xInit() and xShutdown()
		};

		enum OPEN : int
		{
			OPEN_READONLY = 0x00000001,          // Ok for sqlite3_open_v2() 
			OPEN_READWRITE = 0x00000002,        // Ok for sqlite3_open_v2() 
			OPEN_CREATE = 0x00000004,            // Ok for sqlite3_open_v2() 
			OPEN_DELETEONCLOSE = 0x00000008,     // VFS only 
			OPEN_EXCLUSIVE = 0x00000010,         // VFS only 
			OPEN_AUTOPROXY = 0x00000020,         // VFS only 
			OPEN_URI = 0x00000040,               // Ok for sqlite3_open_v2() 
			OPEN_MEMORY = 0x00000080,            // Ok for sqlite3_open_v2()
			OPEN_MAIN_DB = 0x00000100,           // VFS only 
			OPEN_TEMP_DB = 0x00000200,           // VFS only 
			OPEN_TRANSIENT_DB = 0x00000400,      // VFS only 
			OPEN_MAIN_JOURNAL = 0x00000800,      // VFS only 
			OPEN_TEMP_JOURNAL = 0x00001000,      // VFS only 
			OPEN_SUBJOURNAL = 0x00002000,        // VFS only 
			OPEN_MASTER_JOURNAL = 0x00004000,    // VFS only 
			OPEN_NOMUTEX = 0x00008000,           // Ok for sqlite3_open_v2() 
			OPEN_FULLMUTEX = 0x00010000,         // Ok for sqlite3_open_v2() 
			OPEN_SHAREDCACHE = 0x00020000,       // Ok for sqlite3_open_v2() 
			OPEN_PRIVATECACHE = 0x00040000,      // Ok for sqlite3_open_v2() 
			OPEN_WAL = 0x00080000,               // VFS only 
		};

		enum ACCESS
		{
			ACCESS_EXISTS = 0,
			ACCESS_READWRITE = 1,	// Used by PRAGMA temp_store_directory
			ACCESS_READ = 2,		// Unused
		};

		VSystem *Next;	// Next registered VFS
		const char *Name;	// Name of this virtual file system
		void *Tag;			// Pointer to application-specific data
		int SizeOsFile;     // Size of subclassed VirtualFile
		int MaxPathname;	// Maximum file pathname length

		__device__ static RC Initialize();
		__device__ static void Shutdown();

		__device__ static VSystem *FindVfs(const char *name);
		__device__ static RC RegisterVfs(VSystem *vfs, bool _default);
		__device__ static RC UnregisterVfs(VSystem *vfs);

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

		__device__ inline RC OpenAndAlloc(const char *path, VFile **file, OPEN flags, OPEN *outFlags)
		{
			VFile *file2 = (VFile *)_alloc(SizeOsFile);
			if (!file2)
				return RC_NOMEM;
			RC rc = Open(path, file2, flags, outFlags);
			if (rc != RC_OK)
				_free(file2);
			else
				*file = file2;
			return rc;
		}

#pragma region File
#ifdef ENABLE_8_3_NAMES
		__device__ static void FileSuffix3(const char *baseFilename, char *z)
#else
		__device__ inline static void FileSuffix3(const char *baseFilename, char *z) { }
#endif
		__device__ static RC ParseUri(const char *defaultVfsName, const char *uri, VSystem::OPEN *flagsRef, VSystem **vfsOut, char **fileNameOut, char **errMsgOut);
		__device__ static const char *UriParameter(const char *filename, const char *param);
		__device__ static bool UriBoolean(const char *filename, const char *param, bool dflt);
		__device__ static int64 UriInt64(const char *filename, const char *param, int64 dflt);

#pragma endregion
	};

	__device__ __forceinline void operator|=(VSystem::OPEN &a, int b) { a = (VSystem::OPEN)(a | b); }
	__device__ __forceinline void operator&=(VSystem::OPEN &a, int b) { a = (VSystem::OPEN)(a & b); }
}