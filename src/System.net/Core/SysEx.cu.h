// os.h
namespace Core
{
#pragma region Log & Trace

#if _DEBUG
	__device__ extern bool IOTrace;
#define SysEx_LOG(RC, X, ...) { _printf(X, __VA_ARGS__); }
#define SysEx_IOTRACE(X, ...) if (IOTrace) { _printf("IO: "X, __VA_ARGS__); }
#else
#define SysEx_LOG(RC, X, ...)
#define SysEx_IOTRACE(X, ...)
#endif

	//#ifdef _DEBUG
	//	extern bool IOTrace;
	//#ifdef __CUDACC__
	//	__device__ inline static void SysEx_LOG(RC rc, const char *fmt) { }
	//	template <typename T1> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1) { }
	//	template <typename T1, typename T2> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2) { }
	//	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	//	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	//	__device__ inline static void SysEx_IOTRACE(const char *fmt) { }
	//	template <typename T1> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1) { }
	//	template <typename T1, typename T2> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2) { }
	//	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	//	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	//#else
	//	__device__ inline static void SysEx_LOG(RC rc, const char *fmt, ...) { }
	//	__device__ inline static void SysEx_IOTRACE(const char *fmt, ...) { }
	//#endif
	//#else
	//#define SysEx_LOG(X, ...) ((void)0)
	//#define SysEx_IOTRACE(X, ...) ((void)0)
	//#endif

#pragma endregion

#define CORE_VERSION        "--VERS--"
#define CORE_VERSION_NUMBER 3007016
#define CORE_SOURCE_ID      "--SOURCE-ID--"

	class TagBase
	{
	public:
		struct LookasideSlot
		{
			LookasideSlot *Next;    // Next buffer in the list of free buffers
		};

		struct Lookaside
		{
			uint16 Size;            // Size of each buffer in bytes
			bool Enabled;           // False to disable new lookaside allocations
			bool Malloced;          // True if pStart obtained from sqlite3_malloc()
			int Outs;               // Number of buffers currently checked out
			int MaxOuts;            // Highwater mark for nOut
			int Stats[3];			// 0: hits.  1: size misses.  2: full misses
			LookasideSlot *Free;	// List of available buffers
			void *Start;			// First byte of available memory space
			void *End;				// First byte past end of available space
		};

		MutexEx Mutex;			// Connection mutex 
		bool MallocFailed;		// True if we have seen a malloc failure
		RC ErrCode;				// Most recent error code (RC_*)
		int ErrMask;			// & result codes with this before returning
		Lookaside Lookaside;	// Lookaside malloc configuration
	};

	class SysEx
	{
	public:

#pragma region Initialize/Shutdown/Config
		struct GlobalStatics
		{
			bool Memstat;						// True to enable memory status
			bool CoreMutex;						// True to enable core mutexing
			bool FullMutex;						// True to enable full mutexing
			bool OpenUri;						// True to interpret filenames as URIs
			//Main::bool UseCis;						// Use covering indices for full-scans
			int MaxStrlen;						// Maximum string length
			int LookasideSize;					// Default lookaside buffer size
			int Lookasides;						// Default lookaside buffer count
			//mem_methods m;					// Low-level memory allocation interface
			//mutex_methods mutex;				// Low-level mutex interface
			//Main::sqlite3_pcache_methods2 pcache2;	// Low-level page-cache interface
			//array_t<void> Heap;				// Heap storage space
			//int MinReq, MaxReq;				// Min and max heap requests sizes
			void *Scratch;						// Scratch memory
			int ScratchSize;					// Size of each scratch buffer
			int Scratchs;						// Number of scratch buffers
			//Main::void *Page;						// Page cache memory
			//Main::int PageSize;						// Size of each page in pPage[]
			//Main::int Pages;						// Number of pages in pPage[]
			//Main::int MaxParserStack;				// maximum depth of the parser stack
			bool SharedCacheEnabled;			// true if shared-cache mode enabled
			// The above might be initialized to non-zero.  The following need to always initially be zero, however.
			bool IsInit;						// True after initialization has finished
			bool InProgress;					// True while initialization in progress
			bool IsMutexInit;					// True after mutexes are initialized
			bool IsMallocInit;					// True after malloc is initialized
			//bool IsPCacheInit;				// True after malloc is initialized
			MutexEx InitMutex;					// Mutex used by sqlite3_initialize()
			int InitMutexRefs;					// Number of users of pInitMutex
			void (*Log)(void *, int, const char *); // Function for logging
			void *LogArg;						// First argument to xLog()
			bool LocaltimeFault;				// True to fail localtime() calls
#ifdef ENABLE_SQLLOG
			void (*Sqllog)(void*,TagBase*,const char*, int);
			void *SqllogArg;
#endif
		};

		__device__ inline static RC AutoInitialize() { return RC_OK; }
		__device__ static RC PreInitialize(MutexEx &masterMutex);
		__device__ static void PostInitialize(MutexEx masterMutex);
		__device__ static RC Shutdown();

		enum CONFIG
		{
			CONFIG_SINGLETHREAD = 1,	// nil
			CONFIG_MULTITHREAD = 2,		// nil
			CONFIG_SERIALIZED = 3,		// nil
			CONFIG_MALLOC = 4,			// sqlite3_mem_methods*
			CONFIG_GETMALLOC = 5,		// sqlite3_mem_methods*
			CONFIG_SCRATCH = 6,			// void*, int sz, int N
			CONFIG_HEAP = 8,			// void*, int nByte, int min
			CONFIG_MEMSTATUS = 9,		// boolean
			CONFIG_MUTEX = 10,			// sqlite3_mutex_methods*
			CONFIG_GETMUTEX = 11,		// sqlite3_mutex_methods*
			CONFIG_LOOKASIDE = 13,		// int int
			CONFIG_LOG = 16,			// xFunc, void*
			CONFIG_URI = 17,			// int
			CONFIG_SQLLOG = 21,			// xSqllog, void*
		};
		__device__ static RC Config(CONFIG op, va_list *args);
#if __CUDACC__
		__device__ inline static void Config(CONFIG op) { va_list args; va_start(args, nullptr); Config(op, &args); va_end(args); }
		template <typename T1> __device__ inline static void Config(CONFIG op, T1 arg1) { va_list args; va_start(args, arg1); Config(op, &args); va_end(args); }
		template <typename T1, typename T2> __device__ inline static void Config(CONFIG op, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); Config(op, &args); va_end(args); }
		template <typename T1, typename T2, typename T3> __device__ inline static void Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); Config(op, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); Config(op, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static void Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); Config(op, &args); va_end(args); }
#else
		__device__ inline void static Config(CONFIG op, ...) { va_list args; va_start(args, op); Config(op, &args); va_end(args); }
#endif
#pragma endregion

#pragma region Func
		__device__ static RC SysEx::SetupLookaside(TagBase *tag, void *buf, int size, int count);
		// random
		__device__ static void PutRandom(int length, void *buffer);
#pragma endregion

#pragma region BKPT
#if _DEBUG
		__device__ inline static RC CORRUPT_BKPT_(int line)
		{
			SysEx_LOG(RC_CORRUPT, "database corruption at line %d of [%.10s]", line, "src");
			return RC_CORRUPT;
		}
		__device__ inline static RC MISUSE_BKPT_(int line)
		{
			SysEx_LOG(RC_MISUSE, "misuse at line %d of [%.10s]", line, "src");
			return RC_MISUSE;
		}
		__device__ inline static RC CANTOPEN_BKPT_(int line)
		{
			SysEx_LOG(RC_CANTOPEN, "cannot open file at line %d of [%.10s]", line, "src");
			return RC_CANTOPEN;
		}
#define SysEx_CORRUPT_BKPT SysEx::CORRUPT_BKPT_(__LINE__)
#define SysEx_MISUSE_BKPT SysEx::MISUSE_BKPT_(__LINE__)
#define SysEx_CANTOPEN_BKPT SysEx::CANTOPEN_BKPT_(__LINE__)
#else
#define SysEx_CORRUPT_BKPT RC_CORRUPT
#define SysEx_MISUSE_BKPT RC_MISUSE
#define SysEx_CANTOPEN_BKPT RC_CANTOPEN
#endif
#pragma endregion
	};

	__device__ extern _WSD SysEx::GlobalStatics g_GlobalStatics;
#define SysEx_GlobalStatics _GLOBAL(SysEx::GlobalStatics, g_GlobalStatics)
}
