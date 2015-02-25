// btree.h
namespace Core
{
#define N_BTREE_META 10

#ifndef DEFAULT_AUTOVACUUM
#define DEFAULT_AUTOVACUUM Btree::AUTOVACUUM_NONE
#endif

	struct Mem;
	struct BtCursor;
	struct BtShared;

	enum SO : uint8
	{
		SO_ASC = 0, // Sort in ascending order
		SO_DESC = 1, // Sort in ascending order
	};

	struct KeyInfo
	{
		BContext *Ctx;		// The database connection
		TEXTENCODE Encode;	// Text encoding - one of the SQLITE_UTF* values
		uint16 Fields;      // Number of entries in aColl[]
		SO *SortOrders;		// Sort order for each column.  May be NULL
		CollSeq *Colls[1];  // Collating sequence for each term of the key
	};

	enum UNPACKED : uint8
	{
		UNPACKED_INCRKEY = 0x01,			// Make this key an epsilon larger
		UNPACKED_PREFIX_MATCH = 0x02,	// A prefix match is considered OK
		UNPACKED_PREFIX_SEARCH = 0x04,	// Ignore final (rowid) field
	};
	__device__ inline void operator|=(UNPACKED &a, int b) { a = (UNPACKED)(a | b); }
	__device__ inline void operator&=(UNPACKED &a, int b) { a = (UNPACKED)(a & b); }
	__device__ inline UNPACKED operator|(UNPACKED a, UNPACKED b) { return (UNPACKED)((int)a | (int)b); }

	struct UnpackedRecord
	{
		KeyInfo *KeyInfo;	// Collation and sort-order information
		uint16 Fields;      // Number of entries in apMem[]
		UNPACKED Flags;     // Boolean settings.  UNPACKED_... below
		int64 Rowid;        // Used by UNPACKED_PREFIX_SEARCH
		Mem *Mems;          // Values
	};

	enum LOCK : uint8
	{
		LOCK_READ = 1,
		LOCK_WRITE = 2,
	};

	enum TRANS : uint8
	{
		TRANS_NONE = 0,
		TRANS_READ = 1,
		TRANS_WRITE = 2,
	};

	class Btree
	{
	public:
		enum AUTOVACUUM : uint8
		{
			AUTOVACUUM_NONE = 0,	// Do not do auto-vacuum
			AUTOVACUUM_FULL = 1,    // Do full auto-vacuum
			AUTOVACUUM_INCR = 2,    // Incremental vacuum
		};

		enum OPEN : uint8
		{
			OPEN_OMIT_JOURNAL = 1,	// Do not create or use a rollback journal
			OPEN_MEMORY = 2,		// This is an in-memory DB
			OPEN_SINGLE = 4,		// The file contains at most 1 b-tree
			OPEN_UNORDERED = 8,		// Use of a hash implementation is OK
		};

		struct BtLock
		{
			Btree *Btree;			// Btree handle holding this lock
			Pid Table;				// Root page of table
			LOCK Lock;				// READ_LOCK or WRITE_LOCK
			BtLock *Next;			// Next in BtShared.pLock list
		};

		BContext *Ctx;				// The database connection holding this btree
		BtShared *Bt;				// Sharable content of this btree
		TRANS InTrans;				// TRANS_NONE, TRANS_READ or TRANS_WRITE
		bool Sharable_;				// True if we can share pBt with another db
		bool Locked;				// True if db currently has pBt locked
		int WantToLock;				// Number of nested calls to sqlite3BtreeEnter()
		int Backups;				// Number of backup operations reading this btree
		Btree *Next;				// List of other sharable Btrees from the same db
		Btree *Prev;				// Back pointer of the same list
#ifndef OMIT_SHARED_CACHE
		BtLock Lock;				// Object used to lock page 1
#endif

		__device__ static RC Open(VSystem *vfs, const char *filename, BContext *ctx, Btree **btree, OPEN flags, VSystem::OPEN vfsFlags);
		__device__ RC Close();
		__device__ RC SetCacheSize(int maxPage);
		__device__ RC SetSafetyLevel(int level, bool fullSync, bool ckptFullSync);
		__device__ bool SyncDisabled();
		__device__ RC SetPageSize(int pageSize, int reserves, bool fix);
		__device__ int GetPageSize();
		__device__ int MaxPageCount(int maxPage);
		__device__ Pid LastPage();
		__device__ bool SecureDelete(bool newFlag);
		__device__ int GetReserve();
#if defined(HAS_CODEC) || defined(_DEBUG)
		__device__ int GetReserveNoMutex();
#endif
		__device__ RC SetAutoVacuum(AUTOVACUUM autoVacuum);
		__device__ AUTOVACUUM GetAutoVacuum();
		__device__ RC BeginTrans(int wrflag);
		__device__ RC CommitPhaseOne(const char *master);
		__device__ RC CommitPhaseTwo(bool cleanup);
		__device__ RC Commit();
		__device__ RC Rollback(RC tripCode);
		__device__ RC BeginStmt(int statement);
		__device__ RC CreateTable(int *tableID, int flags);
		__device__ bool IsInTrans();
		__device__ bool IsInReadTrans();
		__device__ bool IsInBackup();
		__device__ Schema *Schema(int bytes, void (*free)(void *));
		__device__ RC SchemaLocked();
		__device__ RC LockTable(Pid tableID, bool isWriteLock);
		__device__ RC Savepoint(IPager::SAVEPOINT op, int savepoint);

		__device__ const char *get_Filename();
		__device__ const char *get_Journalname();
		//__device__ int CopyFile(Btree *, Btree *);

		__device__ RC IncrVacuum();

#define BTREE_INTKEY 1		// Table has only 64-bit signed integer keys
#define BTREE_BLOBKEY 2		// Table has keys only - no data

		__device__ RC DropTable(int tableID, int *movedID);
		__device__ RC ClearTable(int tableID, int *changes);
		__device__ void TripAllCursors(RC errCode);

		enum META : uint8
		{
			META_FREE_PAGE_COUNT = 0,
			META_SCHEMA_VERSION = 1,
			META_FILE_FORMAT = 2,
			META_DEFAULT_CACHE_SIZE  = 3,
			META_LARGEST_ROOT_PAGE = 4,
			META_TEXT_ENCODING = 5,
			META_USER_VERSION = 6,
			META_INCR_VACUUM = 7,
		};
		__device__ void GetMeta(META id, uint32 *meta);
		__device__ RC UpdateMeta(META id, uint32 meta);

		__device__ RC NewDb();

#define BTREE_BULKLOAD 0x00000001

		__device__ RC Cursor(Pid tableID, bool wrFlag, struct KeyInfo *keyInfo, BtCursor *cur);
		__device__ static int CursorSize();
		__device__ static void CursorZero(BtCursor *p);

		__device__ static RC CloseCursor(BtCursor *cur);
		__device__ static RC MovetoUnpacked(BtCursor *cur, UnpackedRecord *idxKey, int64 intKey, int biasRight, int *eof);
		__device__ static RC CursorHasMoved(BtCursor *cur, bool *hasMoved);
		__device__ static RC Delete(BtCursor *cur);
		__device__ static RC Insert(BtCursor *cur, const void *key, int64 keyLength, const void *data, int dataLength, int zero, int appendBias, int seekResult);
		__device__ static RC First(BtCursor *cur, int *eof);
		__device__ static RC Last(BtCursor *cur, int *eof);
		__device__ static RC Next_(BtCursor *cur, int *eof);
		__device__ static bool Eof(BtCursor *cur);
		__device__ static RC Previous(BtCursor *cur, int *eof);
		__device__ static RC KeySize(BtCursor *cur, int64 *size);
		__device__ static RC Key(BtCursor *cur, uint32 offset, uint32 amount, void *buf);
		__device__ static const void *KeyFetch(BtCursor *cur, int *amount);
		__device__ static const void *DataFetch(BtCursor *cur, int *amount);
		__device__ static RC DataSize(BtCursor *cur, uint32 *size);
		__device__ static RC Data(BtCursor *cur, uint32 offset, uint32 amount, void *buf);
		__device__ static void SetCachedRowID(BtCursor *cur, int64 rowid);
		__device__ static int64 GetCachedRowID(BtCursor *cur);

		__device__ char *IntegrityCheck(Pid *roots, int rootsLength, int maxErrors, int *errors);
		__device__ Pager *get_Pager();

		__device__ static RC PutData(BtCursor *cur, uint32 offset, uint32 amount, void *z);
		__device__ static void CacheOverflow(BtCursor *cur);
		__device__ static void ClearCursor(BtCursor *cur);
		__device__ RC SetVersion(int version);
		__device__ static void CursorHints(BtCursor *cur, unsigned int mask);

#ifndef DEBUG
		__device__ static bool CursorIsValid(BtCursor *cur);
#endif

#ifndef OMIT_BTREECOUNT
		__device__ static RC Count(BtCursor *cur, int64 *entrysOut);
#endif

#ifdef TEST
		//__device__ int CursorInfo(BtCursor*, int*, int);
		//__device__ void CursorList(Btree*);
#endif

#ifndef OMIT_WAL
		__device__ RC Checkpoint(int mode, int *logs, int *checkpoints);
#endif

#ifndef OMIT_SHARED_CACHE
		__device__ inline void Enter() { }
		__device__ static void EnterAll(BContext *ctx) { }
		__device__ inline bool Sharable() { return Sharable_; }
		__device__ inline void Leave() { }
		__device__ static void LeaveAll(BContext *ctx) { }
		//#ifndef _DEBUG
		// These routines are used inside assert() statements only.
		__device__ inline bool HoldsMutex() { return true; }
		__device__ inline static bool HoldsAllMutexes(BContext *ctx) { return true; }
		__device__ inline static bool SchemaMutexHeld(BContext *ctx, int db, Core::Schema *schema) { return true; }
		//#endif
#else
		//#define EnterAll(X)
#define Sharable(X) 0
#define Leave(X)
#define LeaveAll(X)
#ifndef _DEBUG
		// These routines are used inside assert() statements only.
#define HoldsMutex(X) 1
#define HoldsAllMutexes(X) 1
#define SchemaMutexHeld(X,Y,Z) 1
#endif
#endif

	};

	typedef struct Btree::BtLock BtLock;
	__device__ inline void operator|=(Btree::OPEN &a, int b) { a = (Btree::OPEN)(a | b); }
	__device__ inline Btree::OPEN operator|(Btree::OPEN a, Btree::OPEN b) { return (Btree::OPEN)((int)a | (int)b); }
}