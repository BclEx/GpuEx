using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace Core
{
    public class TagBase
    {
        public class LookasideSlot
        {
            public LookasideSlot Next;    // Next buffer in the list of free buffers
        }

        public class Lookaside_
        {
            public ushort Size;            // Size of each buffer in bytes
            public bool Enabled;           // False to disable new lookaside allocations
            public bool Malloced;          // True if pStart obtained from sqlite3_malloc()
            public int Outs;               // Number of buffers currently checked out
            public int MaxOuts;            // Highwater mark for nOut
            public int[] Stats = new int[3];			// 0: hits.  1: size misses.  2: full misses
            public LookasideSlot Free;	// List of available buffers
            public byte[] Start;			// First byte of available memory space
            public LookasideSlot End;				// First byte past end of available space
        }

        public MutexEx Mutex;       // Connection mutex
        public bool MallocFailed;   // True if we have seen a malloc failure
        public RC ErrCode;          // Most recent error code (RC_*)
        public int ErrMask;         // & result codes with this before returning
        public Lookaside_ Lookaside;	// Lookaside malloc configuration
    }

    public partial class SysEx
    {
        #region Log & Trace

#if DEBUG
        internal static bool IOTrace = true;
        internal static void LOG(RC rc, string format, params object[] args) { Console.WriteLine("l:" + string.Format(format, args)); }
        internal static void IOTRACE(string format, params object[] args) { if (IOTrace) Console.WriteLine("i:" + string.Format(format, args)); }
#else
        internal static void LOG(RC rc, string x, params object[] args) { }
        internal static void IOTRACE(string format, params object[] args) { }
#endif
        //internal static RC LOG2(RC rc, string func, string path)
        //{
        //    var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
        //    var errorID = (uint)Marshal.GetLastWin32Error();
        //    var message = Marshal.GetLastWin32Error().ToString();
        //    Debug.Assert(rc != RC.OK);
        //    if (path == null)
        //        path = string.Empty;
        //    int i;
        //    for (i = 0; i < message.Length && message[i] != '\r' && message[i] != '\n'; i++) ;
        //    message = message.Substring(0, i);
        //    //sqlite3_log("os_win.c:%d: (%d) %s(%s) - %s", sf.GetFileLineNumber(), errorID, func, sf.GetFileName(), message);
        //    return rc;
        //}

        #endregion

        internal const string CORE_VERSION = "--VERS--";
        internal const int CORE_VERSION_NUMBER = 3007016;
        internal const string CORE_SOURCE_ID = "--SOURCE-ID--";

        #region WSD
        #endregion

        #region Initialize/Shutdown/Config

        public class GlobalStatics
        {
            public bool Memstat;                        // True to enable memory status
            public bool CoreMutex;                      // True to enable core mutexing
            public bool FullMutex;                      // True to enable full mutexing
            public bool OpenUri;                        // True to interpret filenames as URIs
            //MainLLbool UseCis;						// Use covering indices for full-scans
            public int MaxStrlen;                       // Maximum string length
            public int LookasideSize;                   // Default lookaside buffer size
            public int Lookasides;                      // Default lookaside buffer count
            //public sqlite3_mem_methods m;             // Low-level memory allocation interface
            //public sqlite3_mutex_methods mutex;       // Low-level mutex interface
            //Main::sqlite3_pcache_methods pcache;      // Low-level page-cache interface
            //public array_t<byte[]> Heap;              // Heap storage space
            //public int MaxReq, MaxReq;                // Min and max heap requests sizes
            public byte[][] Scratch, Scratch2;          // Scratch memory
            public int ScratchSize;                     // Size of each scratch buffer
            public int Scratchs;                        // Number of scratch buffers
            //Main::MemPage Page;                       // Page cache memory
            //Main::int PageSize;                       // Size of each page in pPage[]
            //Main::int Pages;                          // Number of pages in pPage[]
            //Main::int MaxParserStack;                 // maximum depth of the parser stack
            public bool SharedCacheEnabled;             // true if shared-cache mode enabled
            // The above might be initialized to non-zero.  The following need to always initially be zero, however.
            public bool IsInit;                         // True after initialization has finished
            public bool InProgress;                     // True while initialization in progress
            public bool IsMutexInit;                    // True after mutexes are initialized
            public bool IsMallocInit;                   // True after malloc is initialized
            //Main::bool IsPCacheInit;                  // True after malloc is initialized
            public MutexEx InitMutex;                   // Mutex used by sqlite3_initialize()
            public int InitMutexRefs;                   // Number of users of pInitMutex
            public Action<object, int, string> Log;     // Function for logging
            public object LogArg;                       // First argument to xLog()
            public bool LocaltimeFault;                 // True to fail localtime() calls
#if ENABLE_SQLLOG
            public Action<object, TagBase, string, int> Sqllog;
			public object SqllogArg;
#endif

            public GlobalStatics(
                bool memstat,
                bool coreMutex,
                bool fullMutex,
                bool openUri,
                int maxStrlen,
                int lookasideSize,
                int lookasides,
                //sqlite3_mem_methods m,
                //sqlite3_mutex_methods mutex,
                //array_t<byte[]> heap,
                //int minReq, int maxReq,
                byte[][] scratch,
                int scratchSize,
                int scratchs,
                bool sharedCacheEnabled,
                bool isInit,
                bool inProgress,
                bool isMutexInit,
                bool isMallocInit,
                MutexEx initMutex,
                int initMutexRefs,
                Action<object, int, string> log,
                object logArg,
                bool localtimeFault
#if ENABLE_SQLLOG
                , Action<object, TagBase, string, int> Sqllog,
                object SqllogArg
#endif
)
            {
                Memstat = memstat;
                CoreMutex = coreMutex;
                OpenUri = openUri;
                FullMutex = fullMutex;
                MaxStrlen = maxStrlen;
                LookasideSize = lookasideSize;
                Lookasides = lookasides;
                //m = m;
                //mutex = mutex;
                //Heap = heap;
                //MaxReq = minReq; MinReq = maxReq;
                Scratch = scratch;
                ScratchSize = scratchSize;
                Scratchs = scratchs;
                SharedCacheEnabled = sharedCacheEnabled;
                IsInit = isInit;
                InProgress = inProgress;
                IsMutexInit = isMutexInit;
                IsMallocInit = isMallocInit;
                InitMutex = initMutex;
                InitMutexRefs = initMutexRefs;
                Log = log;
                LogArg = logArg;
                LocaltimeFault = localtimeFault;
#if ENABLE_SQLLOG
                Sqllog = sqllog;
                SqllogArg = sqllogArg;
#endif
            }
        }

        public enum CONFIG
        {
            SINGLETHREAD = 1,	// nil
            MULTITHREAD = 2,	// nil
            SERIALIZED = 3,		// nil
            MALLOC = 4,			// sqlite3_mem_methods*
            GETMALLOC = 5,		// sqlite3_mem_methods*
            SCRATCH = 6,		// void*, int sz, int N
            HEAP = 8,			// void*, int nByte, int min
            MEMSTATUS = 9,		// boolean
            MUTEX = 10,			// sqlite3_mutex_methods*
            GETMUTEX = 11,		// sqlite3_mutex_methods*
            LOOKASIDE = 13,		// int int
            LOG = 16,			// xFunc, void*
            URI = 17,			// int
            SQLLOG = 21,		// xSqllog, void*
        }

        const bool CORE_DEFAULT_MEMSTATUS = true;
        const bool CORE_USE_URI = false;

        // The following singleton contains the global configuration for the SQLite library.
        public static readonly GlobalStatics _GlobalStatics = new GlobalStatics(
            CORE_DEFAULT_MEMSTATUS,		// Memstat
            true,						// CoreMutex
#if THREADSAFE
            true,						// FullMutex
#else
		    false,						// FullMutex
#endif
            CORE_USE_URI,				// OpenUri
            // Main::UseCis
            0x7ffffffe,					// MaxStrlen
            128,						// LookasideSize
            500,						// Lookasides
            //{0,0,0,0,0,0,0,0},		// m
            //{0,0,0,0,0,0,0,0,0},		// mutex
            // pcache2
            //array_t(void *)nullptr, 0)// Heap
            //0, 0,						// MinHeap, MaxHeap
            null,			            // Scratch
            0,							// ScratchSize
            0,							// Scratchs
            // Main::Page
            // Main::PageSize
            // Main::Pages
            // Main::MaxParserStack
            false,						// SharedCacheEnabled
            // All the rest should always be initialized to zero
            false,						// IsInit
            false,						// InProgress
            false,						// IsMutexInit
            false,						// IsMallocInit
            // Main::IsPCacheInit
            default(MutexEx),			// InitMutex
            0,							// InitMutexRefs
            null,					    // Log
            null,						// LogArg
            false						// LocaltimeFault    
#if ENABLE_SQLLOG
		    , null,					    // Sqllog
		    null						// SqllogArg
#endif
        );

        public static RC PreInitialize(out MutexEx masterMutex)
        {
            masterMutex = default(MutexEx);
            // If SQLite is already completely initialized, then this call to sqlite3_initialize() should be a no-op.  But the initialization
            // must be complete.  So isInit must not be set until the very end of this routine.
            if (_GlobalStatics.IsInit) return RC.OK;

            // The following is just a sanity check to make sure SQLite has been compiled correctly.  It is important to run this code, but
            // we don't want to run it too often and soak up CPU cycles for no reason.  So we run it once during initialization.
#if !NDEBUG && !OMIT_FLOATING_POINT
            // This section of code's only "output" is via assert() statements.
            //ulong x = (((ulong)1)<<63)-1;
            //double y;
            //Debug.Assert(sizeof(ulong) == 8);
            //Debug.Assert(sizeof(ulong) == sizeof(double));
            //_memcpy<void>(&y, &x, 8);
            //Debug.Assert(double.IsNaN(y));
#endif

            RC rc;
#if ENABLE_SQLLOG
		{
			Init_Sqllog();
		}
#endif

            // Make sure the mutex subsystem is initialized.  If unable to initialize the mutex subsystem, return early with the error.
            // If the system is so sick that we are unable to allocate a mutex, there is not much SQLite is going to be able to do.
            // The mutex subsystem must take care of serializing its own initialization.
            rc = RC.OK; //MutexEx::Init();
            if (rc != 0) return rc;

            // Initialize the malloc() system and the recursive pInitMutex mutex. This operation is protected by the STATIC_MASTER mutex.  Note that
            // MutexAlloc() is called for a static mutex prior to initializing the malloc subsystem - this implies that the allocation of a static
            // mutex must not require support from the malloc subsystem.
            masterMutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER); // The main static mutex
            MutexEx.Enter(masterMutex);
            _GlobalStatics.IsMutexInit = true;
            //if (!SysEx_GlobalStatics.IsMallocInit)
            //	rc = sqlite3MallocInit();
            if (rc == RC.OK)
            {
                _GlobalStatics.IsMallocInit = true;
                if (_GlobalStatics.InitMutex.Tag == null)
                {
                    _GlobalStatics.InitMutex = MutexEx.Alloc(MutexEx.MUTEX.RECURSIVE);
                    if (_GlobalStatics.CoreMutex && _GlobalStatics.InitMutex.Tag == null)
                        rc = RC.NOMEM;
                }
            }
            if (rc == RC.OK)
                _GlobalStatics.InitMutexRefs++;
            MutexEx.Leave(masterMutex);

            // If rc is not SQLITE_OK at this point, then either the malloc subsystem could not be initialized or the system failed to allocate
            // the pInitMutex mutex. Return an error in either case.
            if (rc != RC.OK)
                return rc;

            // Do the rest of the initialization under the recursive mutex so that we will be able to handle recursive calls into
            // sqlite3_initialize().  The recursive calls normally come through sqlite3_os_init() when it invokes sqlite3_vfs_register(), but other
            // recursive calls might also be possible.
            //
            // IMPLEMENTATION-OF: R-00140-37445 SQLite automatically serializes calls to the xInit method, so the xInit method need not be threadsafe.
            //
            // The following mutex is what serializes access to the appdef pcache xInit methods.  The sqlite3_pcache_methods.xInit() all is embedded in the
            // call to sqlite3PcacheInitialize().
            MutexEx.Enter(_GlobalStatics.InitMutex);
            if (!_GlobalStatics.IsInit && !_GlobalStatics.InProgress)
            {
                _GlobalStatics.InProgress = true;
                rc = VSystem.Initialize();
            }
            if (rc != RC.OK)
                MutexEx.Leave(_GlobalStatics.InitMutex);
            return rc;
        }

        public static void PostInitialize(MutexEx masterMutex)
        {
            MutexEx.Leave(_GlobalStatics.InitMutex);

            // Go back under the static mutex and clean up the recursive mutex to prevent a resource leak.
            MutexEx.Enter(masterMutex);
            _GlobalStatics.InitMutexRefs--;
            if (_GlobalStatics.InitMutexRefs <= 0)
            {
                Debug.Assert(_GlobalStatics.InitMutexRefs == 0);
                MutexEx.Free(_GlobalStatics.InitMutex);
                _GlobalStatics.InitMutex.Tag = null;
            }
            MutexEx.Leave(masterMutex);
        }

        public static RC Shutdown()
        {
            if (_GlobalStatics.IsInit)
            {
                VSystem.Shutdown();
                //sqlite3_reset_auto_extension();
                _GlobalStatics.IsInit = false;
            }
            //if (SysEx_GlobalStatics.IsMallocInit)
            //{
            //	sqlite3MallocEnd();
            //	SysEx_GlobalStatics.IsMallocInit = false;
            //}
            if (_GlobalStatics.IsMutexInit)
            {
                //MutexEx::End();
                _GlobalStatics.IsMutexInit = false;
            }
            return RC.OK;
        }

        public static RC Config(CONFIG op, params object[] args)
        {
            // sqlite3_config() shall return SQLITE_MISUSE if it is invoked while the SQLite library is in use.
            if (_GlobalStatics.IsInit) return SysEx.MISUSE_BKPT();
            RC rc = RC.OK;
            switch (op)
            {
#if THREADSAFE
                // Mutex configuration options are only available in a threadsafe compile. 
                case CONFIG.SINGLETHREAD:
                    { // Disable all mutexing
                        _GlobalStatics.CoreMutex = false;
                        _GlobalStatics.FullMutex = false;
                        break;
                    }
                case CONFIG.MULTITHREAD:
                    { // Disable mutexing of database connections, Enable mutexing of core data structures
                        _GlobalStatics.CoreMutex = true;
                        _GlobalStatics.FullMutex = false;
                        break;
                    }
                case CONFIG.SERIALIZED:
                    { // Enable all mutexing
                        _GlobalStatics.CoreMutex = true;
                        _GlobalStatics.FullMutex = true;
                        break;
                    }
                case CONFIG.MUTEX:
                    { // Specify an alternative mutex implementation
                        //_GlobalStatics.Mutex = (sqlite3_mutex_methods)args[0];
                        break;
                    }
                case CONFIG.GETMUTEX:
                    { // Retrieve the current mutex implementation
                        //args[0] = _GlobalStatics.Mutex;
                        break;
                    }
#endif
                case CONFIG.MALLOC:
                    { // Specify an alternative malloc implementation
                        //_GlobalStatics.m = *va_arg(args, sqlite3_mem_methods*);
                        break;
                    }
                case CONFIG.GETMALLOC:
                    { // Retrieve the current malloc() implementation
                        //if (_GlobalStatics.m.xMalloc==0) sqlite3MemSetDefault();
                        //args[0]= _GlobalStatics.m;
                        break;
                    }
                case CONFIG.MEMSTATUS:
                    { // Enable or disable the malloc status collection
                        _GlobalStatics.Memstat = (bool)args[0];
                        break;
                    }
                case CONFIG.SCRATCH:
                    { // Designate a buffer for scratch memory space
                        _GlobalStatics.Scratch = (byte[][])args[0];
                        _GlobalStatics.ScratchSize = (int)args[1];
                        _GlobalStatics.Scratchs = (int)args[2];
                        break;
                    }
#if ENABLE_MEMSYS3 || ENABLE_MEMSYS5
		case CONFIG_HEAP: {
			// Designate a buffer for heap memory space
			_GlobalStatics.Heap.data = va_arg(args, void*);
			_GlobalStatics.Heap.length = va_arg(args, int);
			_GlobalStatics.MinReq = va_arg(ap, int);
			if (_GlobalStatics.MinReq < 1)
				_GlobalStatics.MinReq = 1;
			else if (SysEx_GlobalStatics.MinReq > (1<<12)) // cap min request size at 2^12
				_GlobalStatics.MinReq = (1<<12);
			if (!_GlobalStatics.Heap.data)
				// If the heap pointer is NULL, then restore the malloc implementation back to NULL pointers too.  This will cause the malloc to go back to its default implementation when sqlite3_initialize() is run.
					memset(&_GlobalStatics.m, 0, sizeof(_GlobalStatics.m));
			else
				// The heap pointer is not NULL, then install one of the mem5.c/mem3.c methods. If neither ENABLE_MEMSYS3 nor ENABLE_MEMSYS5 is defined, return an error.
#if ENABLE_MEMSYS3
				_GlobalStatics.m = sqlite3MemGetMemsys3();
#endif
#if ENABLE_MEMSYS5
			_GlobalStatics.m = sqlite3MemGetMemsys5();
#endif
			break; }
#endif
                case CONFIG.LOOKASIDE:
                    {
                        _GlobalStatics.LookasideSize = (int)args[0];
                        _GlobalStatics.Lookasides = (int)args[1];
                        break;
                    }
                case CONFIG.LOG:
                    { // Record a pointer to the logger function and its first argument. The default is NULL.  Logging is disabled if the function pointer is NULL.
                        // MSVC is picky about pulling func ptrs from va lists.
                        // http://support.microsoft.com/kb/47961
                        _GlobalStatics.Log = (Action<object, int, string>)args[0];
                        _GlobalStatics.LogArg = (object)args[1];
                        break;
                    }
                case CONFIG.URI:
                    {
                        _GlobalStatics.OpenUri = (bool)args[0];
                        break;
                    }
#if ENABLE_SQLLOG
		case CONFIG.SQLLOG: {
			_GlobalStatics.Sqllog = (Action<object, TagBase, int, string>)args[0];;
			_GlobalStatics.SqllogArg = (object)args[1];
			break; }
#endif
                default:
                    {
                        rc = RC.ERROR;
                        break;
                    }
            }
            return rc;
        }

        #endregion

        #region Func
        #endregion

        #region BKPT
#if DEBUG
        internal static RC CORRUPT_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "database corruption at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.CORRUPT;
        }
        internal static RC MISUSE_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "misuse at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.MISUSE;
        }
        internal static RC CANTOPEN_BKPT()
        {
            var sf = new StackTrace(new StackFrame(true)).GetFrame(0);
            LOG(RC.CANTOPEN, "cannot open file at line {0} of [{1}]", sf.GetFileLineNumber(), sf.GetFileName());
            return RC.CANTOPEN;
        }
#else
        internal static RC CORRUPT_BKPT() { return RC.CORRUPT; }
        internal static RC MISUSE_BKPT() { return RC.MISUSE; }
        internal static RC CANTOPEN_BKPT() { return RC.CANTOPEN; }
#endif
        #endregion

        public static RC SetupLookaside(TagBase tag, byte[] buf, int size, int count)
        {
            if (tag.Lookaside.Outs != 0)
                return RC.BUSY;
            // Free any existing lookaside buffer for this handle before allocating a new one so we don't have to have space for both at the same time.
            if (tag.Lookaside.Malloced)
                C._free(ref tag.Lookaside.Start);
            // The size of a lookaside slot after ROUNDDOWN8 needs to be larger than a pointer to be useful.
            size = C._ROUNDDOWN8(size); // IMP: R-33038-09382
            if (size <= (int)4) size = 0;
            if (count < 0) count = 0;
            byte[] start;
            if (size == 0 || count == 0)
            {
                size = 0;
                start = null;
            }
            else if (buf == null)
            {
                C._benignalloc_begin();
                start = new byte[size * count]; // IMP: R-61949-35727
                C._benignalloc_end();
            }
            else
                start = buf;
            tag.Lookaside.Start = start;
            tag.Lookaside.Free = null;
            tag.Lookaside.Size = (ushort)size;
            if (start != null)
            {
                Debug.Assert(size > 4);
                TagBase.LookasideSlot p = (TagBase.LookasideSlot)null; //: start;
                for (int i = count - 1; i >= 0; i--)
                {
                    p.Next = tag.Lookaside.Free;
                    tag.Lookaside.Free = p;
                    p = (TagBase.LookasideSlot)null; //: &((uint8 *)p)[size];
                }
                tag.Lookaside.End = p;
                tag.Lookaside.Enabled = true;
                tag.Lookaside.Malloced = (buf == null);
            }
            else
            {
                tag.Lookaside.End = null;
                tag.Lookaside.Enabled = false;
                tag.Lookaside.Malloced = false;
            }
            return RC.OK;
        }
    }
}