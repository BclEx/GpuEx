using System;
namespace Core
{
    public partial class E
    {
        public static TEXTENCODE CTXENCODE(BContext ctx) { return ctx.DBs[0].Schema.Encode; }
        public static void CTXENCODE(BContext ctx, TEXTENCODE encode) { ctx.DBs[0].Schema.Encode = encode; }
    }

    public class BContext : TagBase
    {
        const int MAX_ATTACHED = 10;

        public class BusyHandlerType
        {
            public Func<object, int, int> Func; // The busy callback
            public object Arg;                  // First arg to busy callback
            public int Busys;                   // Incremented with each busy call
        }

        public class DB
        {
            public string Name;					// Name of this database
            public Btree Bt;					// The B*Tree structure for this database file
            public byte InTrans;				// 0: not writable.  1: Transaction.  2: Checkpoint
            public byte SafetyLevel;			// How aggressive at syncing data to disk
            public Schema Schema;			    // Pointer to database schema (possibly shared)
        }

        [Flags]
        public enum FLAG : uint
        {
            VdbeTrace = 0x00000001,	// True to trace VDBE execution
            InternChanges = 0x00000002,	// Uncommitted Hash table changes
            FullColNames = 0x00000004,	// Show full column names on SELECT
            ShortColNames = 0x00000008,	// Show short columns names
            CountRows = 0x00000010,	// Count rows changed by INSERT, DELETE, or UPDATE and return the count using a callback.
            NullCallback = 0x00000020,	// Invoke the callback once if the result set is empty
            SqlTrace = 0x00000040,	// Debug print SQL as it executes
            VdbeListing = 0x00000080,	// Debug listings of VDBE programs
            WriteSchema = 0x00000100,	// OK to update SQLITE_MASTER
            VdbeAddopTrace = 0x00000200,	// Trace sqlite3VdbeAddOp() calls
            IgnoreChecks = 0x00000400,	// Do not enforce check constraints
            ReadUncommitted = 0x0000800,	// For shared-cache mode
            LegacyFileFmt = 0x00001000,	// Create new databases in format 1
            FullFSync = 0x00002000,	// Use full fsync on the backend
            CkptFullFSync = 0x00004000,	// Use full fsync for checkpoint
            RecoveryMode = 0x00008000,	// Ignore schema errors
            ReverseOrder = 0x00010000,	// Reverse unordered SELECTs
            RecTriggers = 0x00020000,	// Enable recursive triggers
            ForeignKeys = 0x00040000,	// Enforce foreign key constraints
            AutoIndex = 0x00080000,	// Enable automatic indexes
            PreferBuiltin = 0x00100000,	// Preference to built-in funcs
            LoadExtension = 0x00200000,	// Enable load_extension
            EnableTrigger = 0x00400000,	// True to enable triggers
        }

        public array_t<DB> DBs = new array_t<DB>(new DB[MAX_ATTACHED]); // All backends / Number of backends currently in use
        public FLAG Flags;                      // Miscellaneous flags. See below
        public int ActiveVdbeCnt;               // Number of VDBEs currently executing
        public BusyHandlerType BusyHandler; // Busy callback
        public DB[] DBStatics = new[] { new DB(), new DB() }; // Static space for the 2 default backends
        public Savepoint Savepoints;            // List of active savepoints
        public int BusyTimeout;				    // Busy handler timeout, in msec
        public int SavepointsLength;            // Number of non-transaction savepoints
#if ENABLE_UNLOCK_NOTIFY
        public Context BlockingConnection;          // Connection that caused SQLITE_LOCKED
        public Context UnlockConnection;            // Connection to watch for unlock
        public object UnlockArg;                    // Argument to xUnlockNotify
        public Action<object[], int> UnlockNotify;  // Unlock notify callback
        public Context NextBlocked;                 // Next in list of all blocked connections
#endif

        #region From: Main_c
        public bool TempInMemory()
        {
            //#if TEMP_STORE = 1
            //            return (TempStore == 2);
            //#endif
            //#if TEMP_STORE = 2
            //            return (TempStore != 1);
            //#endif
            //#if TEMP_STORE = 3
            //            return true;
            //#endif
            return false;
        }
        #endregion

        #region From: Notify_c
#if ENABLE_UNLOCK_NOTIFY
#else
        public void ConnectionBlocked(BContext blocked) { }
        internal void ConnectionUnlocked() { }
        internal void ConnectionClosed() { }
#endif
        #endregion
    }
}