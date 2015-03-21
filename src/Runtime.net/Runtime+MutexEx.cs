namespace Core
{
    public struct MutexEx
    {
        public static bool WantsCoreMutex { get; set; }

        public enum MUTEX
        {
            FAST = 0,
            RECURSIVE = 1,
            STATIC_MASTER = 2,
            STATIC_MEM = 3,  // sqlite3_malloc()
            STATIC_MEM2 = 4,  // NOT USED
            STATIC_OPEN = 4,  // sqlite3BtreeOpen()
            STATIC_PRNG = 5,  // sqlite3_random()
            STATIC_LRU = 6,   // lru page list
            STATIC_LRU2 = 7,  // NOT USED
            STATIC_PMEM = 7, // sqlite3PageMalloc()
        }

        public static MutexEx Alloc(MUTEX id)
        {
            //if (!SysEx_GlobalStatics.bCoreMutex)
            //    return null;
            //Debug.Assert(mutexIsInit != 0);
            //return SysEx_GlobalStatics.mutex.xMutexAlloc(id);
            return default(MutexEx);
        }

        public static void Enter(MutexEx mutex) { }
        public static void Leave(MutexEx mutex) { }
        public static bool Held(MutexEx mutex) { return true; }
        public static bool NotHeld(MutexEx mutex) { return true; }
        public static void Free(MutexEx mutex) { }

        public object Tag;      // Mutex controlling the lock
        public int id;          // Mutex type
        public int nRef;        // Number of enterances
        public int owner;       // Thread holding this mutex
#if DEBUG
        public int trace;       // True to trace changes
#endif


        //        public MutexEx(MutexEx mutex, int id, int nRef, int owner
        //#if DEBUG
        //, int trace
        //#endif
        //)
        //        {
        //            this.mutex = mutex;
        //            this.id = id;
        //            this.nRef = nRef;
        //            this.owner = owner;
        //#if DEBUG
        //            this.trace = 0;
        //#endif
        //        }
    }
}