namespace Core
{
	class Btree;
	struct Savepoint;

#define CTXENCODE(ctx) ((ctx)->DBs[0].Schema->Encode)

	class BContext : public TagBase
	{
	public:
		static const int MAX_ATTACHED = 10;

		struct BusyHandlerType
		{
			int (*Func)(void *, int);	// The busy callback
			void *Arg;					// First arg to busy callback
			int Busys;					// Incremented with each busy call
		};

		struct DB
		{
			char *Name;					// Name of this database
			Btree *Bt;					// The B*Tree structure for this database file
			uint8 InTrans;				// 0: not writable.  1: Transaction.  2: Checkpoint
			uint8 SafetyLevel;			// How aggressive at syncing data to disk
			Schema *Schema;				// Pointer to database schema (possibly shared)
		};

		enum FLAG : uint32
		{
			FLAG_VdbeTrace      = 0x00000001,	// True to trace VDBE execution
			FLAG_InternChanges  = 0x00000002,	// Uncommitted Hash table changes
			FLAG_FullColNames   = 0x00000004,	// Show full column names on SELECT
			FLAG_ShortColNames  = 0x00000008,	// Show short columns names
			FLAG_CountRows      = 0x00000010,	// Count rows changed by INSERT, DELETE, or UPDATE and return the count using a callback.
			FLAG_NullCallback   = 0x00000020,	// Invoke the callback once if the result set is empty
			FLAG_SqlTrace       = 0x00000040,	// Debug print SQL as it executes
			FLAG_VdbeListing    = 0x00000080,	// Debug listings of VDBE programs
			FLAG_WriteSchema    = 0x00000100,	// OK to update SQLITE_MASTER
			FLAG_VdbeAddopTrace = 0x00000200,	// Trace sqlite3VdbeAddOp() calls
			FLAG_IgnoreChecks   = 0x00000400,	// Do not enforce check constraints
			FLAG_ReadUncommitted = 0x0000800,	// For shared-cache mode
			FLAG_LegacyFileFmt  = 0x00001000,	// Create new databases in format 1
			FLAG_FullFSync      = 0x00002000,	// Use full fsync on the backend
			FLAG_CkptFullFSync  = 0x00004000,	// Use full fsync for checkpoint
			FLAG_RecoveryMode   = 0x00008000,	// Ignore schema errors
			FLAG_ReverseOrder   = 0x00010000,	// Reverse unordered SELECTs
			FLAG_RecTriggers    = 0x00020000,	// Enable recursive triggers
			FLAG_ForeignKeys    = 0x00040000,	// Enforce foreign key constraints
			FLAG_AutoIndex      = 0x00080000,	// Enable automatic indexes
			FLAG_PreferBuiltin  = 0x00100000,	// Preference to built-in funcs
			FLAG_LoadExtension  = 0x00200000,	// Enable load_extension
			FLAG_EnableTrigger  = 0x00400000,	// True to enable triggers
		};

		array_t<DB> DBs;					// All backends / Number of backends currently in use
		FLAG Flags;							// Miscellaneous flags. See below
		int ActiveVdbeCnt;					// Number of VDBEs currently executing
		BusyHandlerType BusyHandler;		// Busy callback
		DB DBStatics[2];					// Static space for the 2 default backends
		Savepoint *Savepoints;				// List of active savepoints
		int BusyTimeout;					// Busy handler timeout, in msec
		int SavepointsLength;				// Number of non-transaction savepoints
#ifdef ENABLE_UNLOCK_NOTIFY
		BContext *BlockingConnection;		// Connection that caused SQLITE_LOCKED
		BContext *UnlockConnection;			// Connection to watch for unlock
		void *UnlockArg;					// Argument to xUnlockNotify
		void (*UnlockNotify)(void**,int);	// Unlock notify callback
		BContext *NextBlocked;				// Next in list of all blocked connections
#endif

#pragma region From: Main_c
		__device__ int InvokeBusyHandler();
		__device__ inline bool TempInMemory()
		{
#if TEMP_STORE == 1
			return (TempStore == 2);
#endif
#if TEMP_STORE == 2
			return (TempStore != 1);
#endif
#if TEMP_STORE == 3
			return true;
#endif
			return false;
		}
#pragma endregion

#pragma region From: Notify_c
#if ENABLE_UNLOCK_NOTIFY
		__device__ RC UnlockNotify_(void (*notify)(void **, int), void *arg, void (*error)(BContext *, RC, const char *));
		__device__ void ConnectionBlocked(BContext *blocker);
		__device__ void ConnectionUnlocked();
		__device__ void ConnectionClosed();
#else
		__device__ static void ConnectionBlocked(BContext *blocker) { }
		__device__ static void ConnectionUnlocked() { }
		__device__ static void ConnectionClosed() { }
#endif
#pragma endregion

	};

	__device__ inline void operator|=(BContext::FLAG &a, BContext::FLAG b) { a = (BContext::FLAG)(a | b); }
	__device__ inline void operator&=(BContext::FLAG &a, BContext::FLAG b) { a = (BContext::FLAG)(a & b); }
	__device__ inline BContext::FLAG operator|(BContext::FLAG a, BContext::FLAG b) { return (BContext::FLAG)((int)a | (int)b); }
	__device__ inline BContext::FLAG operator&(BContext::FLAG a, BContext::FLAG b) { return (BContext::FLAG)((int)a & (int)b); }
	__device__ inline BContext::FLAG operator~(BContext::FLAG a) { return (BContext::FLAG)(~(int)a); }
}