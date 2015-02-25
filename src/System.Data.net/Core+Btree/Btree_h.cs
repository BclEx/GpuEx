// btree.h
using Pid = System.UInt32;
using Mem = System.Object;
using System;
namespace Core
{
    public enum SO : byte
    {
        ASC = 0, // Sort in ascending order
        DESC = 1, // Sort in ascending order
    }

    public class KeyInfo
    {
        public BContext Ctx;		// The database connection
        public TEXTENCODE Encode;	// Text encoding - one of the SQLITE_UTF* values
        public ushort Fields;       // Number of entries in aColl[]
        public SO[] SortOrders;   // Sort order for each column.  May be NULL
        public CollSeq[] Colls = new CollSeq[1];  // Collating sequence for each term of the key

        public KeyInfo _memcpy() { return (KeyInfo)MemberwiseClone(); }
    }

    [Flags]
    public enum UNPACKED : byte
    {
        INCRKEY = 0x01,			// Make this key an epsilon larger
        PREFIX_MATCH = 0x02,	// A prefix match is considered OK
        PREFIX_SEARCH = 0x04,	// Ignore final (rowid) field
    }

    public class UnpackedRecord
    {
        public KeyInfo KeyInfo;	// Collation and sort-order information
        public ushort Fields;      // Number of entries in apMem[]
        public UNPACKED Flags;     // Boolean settings.  UNPACKED_... below
        public long Rowid;        // Used by UNPACKED_PREFIX_SEARCH
        public Mem[] Mems;          // Values
    }

    public enum LOCK : byte
    {
        READ = 1,
        WRITE = 2,
    }

    public enum TRANS : byte
    {
        NONE = 0,
        READ = 1,
        WRITE = 2,
    }

    public partial class Btree
    {
        public const int N_BTREE_META = 10;
        const AUTOVACUUM DEFAULT_AUTOVACUUM = AUTOVACUUM.NONE;

        public enum AUTOVACUUM : byte
        {
            NONE = 0,               // Do not do auto-vacuum
            FULL = 1,               // Do full auto-vacuum
            INCR = 2,               // Incremental vacuum
        }

        [Flags]
        public enum OPEN : byte
        {
            OMIT_JOURNAL = 1,       // Do not create or use a rollback journal
            MEMORY = 2,             // This is an in-memory DB
            SINGLE = 4,             // The file contains at most 1 b-tree
            UNORDERED = 8,          // Use of a hash implementation is OK
        }

        public class BtLock
        {
            public Btree Btree;     // Btree handle holding this lock
            public Pid Table;       // Root page of table
            public LOCK Lock;       // READ_LOCK or WRITE_LOCK
            public BtLock Next;     // Next in BtShared.pLock list
        }

        public BContext Ctx;        // The database connection holding this Btree
        public BtShared Bt;         // Sharable content of this Btree
        public TRANS InTrans;       // TRANS_NONE, TRANS_READ or TRANS_WRITE
        public bool Sharable_;       // True if we can share pBt with another db
        public bool Locked;         // True if db currently has pBt locked
        public int WantToLock;      // Number of nested calls to sqlite3BtreeEnter()
        public int Backups;         // Number of backup operations reading this btree
        public Btree Next;          // List of other sharable Btrees from the same db
        public Btree Prev;          // Back pointer of the same list
#if !OMIT_SHARED_CACHE
        public BtLock Lock;         // Object used to lock page 1
#endif

        const int BTREE_INTKEY = 1;     // Table has only 64-bit signed integer keys
        const int BTREE_BLOBKEY = 2;    // Table has keys only - no data

        public enum META : byte
        {
            FREE_PAGE_COUNT = 0,
            SCHEMA_VERSION = 1,
            FILE_FORMAT = 2,
            DEFAULT_CACHE_SIZE = 3,
            LARGEST_ROOT_PAGE = 4,
            TEXT_ENCODING = 5,
            USER_VERSION = 6,
            INCR_VACUUM = 7,
        }

        const int BTREE_BULKLOAD = 0x00000001;

#if !OMIT_SHARED_CACHE
        public void Enter() { }
        public static void EnterAll(BContext ctx) { }
        public bool Sharable() { return Sharable_; }
        public void Leave() { }
        public static void LeaveAll(BContext ctx) { }
        //#if !DEBUG
        // These routines are used inside Debug.Assert() statements only.
        public bool HoldsMutex() { return true; }
        public static bool HoldsAllMutexes(BContext ctx) { return true; }
        public static bool SchemaMutexHeld(BContext ctx, int db, Schema schema) { return true; }
        //#endif
#else
        static void Enter() { }
        static void EnterAll(BContext ctx) { }
        static bool Sharable(Btree b) { return false; }
        static void Leave(Btree b) { }
        static void LeaveAll(object b) { }
        static bool HoldsMutex(Btree b) { return true; }
        static bool HoldsAllMutexes(BContext ctx) { return true; }
        static bool SchemaMutexHeld(BContext ctx, int db, Schema schema) { return true; }
#endif
    }
}
