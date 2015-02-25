// pager.h
namespace Core
{
	typedef class Pager Pager;
	typedef struct Wal Wal;
	typedef struct PgHdr IPage;
	typedef struct PagerSavepoint PagerSavepoint;
	typedef struct PCache PCache;

	class IPager
	{
	public:
		// NOTE: These values must match the corresponding BTREE_ values in btree.h.
		enum PAGEROPEN : char
		{
			PAGEROPEN_OMIT_JOURNAL = 0x0001,	// Do not use a rollback journal
			PAGEROPEN_MEMORY = 0x0002,			// In-memory database
		};

		enum LOCKINGMODE : char
		{
			LOCKINGMODE_QUERY = -1,
			LOCKINGMODE_NORMAL = 0,
			LOCKINGMODE_EXCLUSIVE = 1,
		};

		enum JOURNALMODE : char
		{
			JOURNALMODE_JQUERY = -1,    // Query the value of journalmode
			JOURNALMODE_DELETE = 0,     // Commit by deleting journal file
			JOURNALMODE_PERSIST = 1,    // Commit by zeroing journal header
			JOURNALMODE_OFF = 2,        // Journal omitted.
			JOURNALMODE_TRUNCATE = 3,	// Commit by truncating journal
			JOURNALMODE_JMEMORY = 4,    // In-memory journal file
			JOURNALMODE_WAL = 5,        // Use write-ahead logging
		};

		// sqlite3.h
		enum CHECKPOINT : char
		{
			CHECKPOINT_PASSIVE = 0,
			CHECKPOINT_FULL = 1,
			CHECKPOINT_RESTART = 2,
		};

		// sqliteInt.h
		enum SAVEPOINT : char
		{
			SAVEPOINT_BEGIN = 0,
			SAVEPOINT_RELEASE = 1,
			SAVEPOINT_ROLLBACK = 2,
		};
	};

	// sqliteLimit.h
#ifdef MAX_PAGE_SIZE
#undef MAX_PAGE_SIZE
#endif
#define MAX_PAGE_SIZE 65536

	class Pager
	{
	public:
		enum PAGER : char
		{
			PAGER_OPEN = 0,
			PAGER_READER = 1,
			PAGER_WRITER_LOCKED = 2,
			PAGER_WRITER_CACHEMOD = 3,
			PAGER_WRITER_DBMOD = 4,
			PAGER_WRITER_FINISHED = 5,
			PAGER_ERROR = 6,
		};

		VSystem *Vfs;				// OS functions to use for IO
		IPager::LOCKINGMODE ExclusiveMode;	// Boolean. True if locking_mode==EXCLUSIVE
		IPager::JOURNALMODE JournalMode; // One of the PAGER_JOURNALMODE_* values
		bool UseJournal;			// Use a rollback journal on this file
		bool NoSync;				// Do not sync the journal if true
		bool FullSync;				// Do extra syncs of the journal for robustness
		VFile::SYNC CheckpointSyncFlags;	// SYNC_NORMAL or SYNC_FULL for checkpoint
		VFile::SYNC WalSyncFlags;	// SYNC_NORMAL or SYNC_FULL for wal writes
		VFile::SYNC SyncFlags;		// SYNC_NORMAL or SYNC_FULL otherwise
		bool TempFile;				// zFilename is a temporary file
		bool ReadOnly;				// True for a read-only database
		bool MemoryDB;				// True to inhibit all file I/O
		// The following block contains those class members that change during routine opertion.  Class members not in this block are either fixed
		// when the pager is first created or else only change when there is a significant mode change (such as changing the page_size, locking_mode,
		// or the journal_mode).  From another view, these class members describe the "state" of the pager, while other class members describe the "configuration" of the pager.
		PAGER State;                // Pager state (OPEN, READER, WRITER_LOCKED..)
		VFile::LOCK Lock;           // Current lock held on database file
		bool ChangeCountDone;       // Set after incrementing the change-counter
		bool SetMaster;             // True if a m-j name has been written to jrnl
		uint8 DoNotSpill;           // Do not spill the cache when non-zero
		uint8 DoNotSyncSpill;       // Do not do a spill that requires jrnl sync
		bool SubjInMemory;          // True to use in-memory sub-journals
		Pid DBSize;					// Number of pages in the database
		Pid DBOrigSize;				// dbSize before the current transaction
		Pid DBFileSize;				// Number of pages in the database file
		Pid DBHintSize;				// Value passed to FCNTL_SIZE_HINT call
		RC ErrorCode;               // One of several kinds of errors
		int Records;                // Pages journalled since last j-header written
		uint32 ChecksumInit;        // Quasi-random value added to every checksum
		uint32 SubRecords;          // Number of records written to sub-journal
		Bitvec *InJournal;			// One bit for each page in the database file
		VFile *File;				// File descriptor for database
		VFile *JournalFile;			// File descriptor for main journal
		VFile *SubJournalFile;		// File descriptor for sub-journal
		int64 JournalOffset;        // Current write offset in the journal file
		int64 JournalHeader;        // Byte offset to previous journal header
		IBackup *Backup;			// Pointer to list of ongoing backup processes
		array_t<PagerSavepoint> Savepoints;	// Array of active savepoints
		char DBFileVersion[16];		// Changes whenever database file changes
		// End of the routinely-changing class members
		uint16 ExtraBytes;          // Add this many bytes to each in-memory page
		int16 ReserveBytes;         // Number of unused bytes at end of each page
		VSystem::OPEN VfsFlags;		// Flags for sqlite3_vfs.xOpen()
		uint32 SectorSize;          // Assumed sector size during rollback
		int PageSize;               // Number of bytes in a page
		Pid MaxPid;					// Maximum allowed size of the database
		int64 JournalSizeLimit;     // Size limit for persistent journal files
		char *Filename;				// Name of the database file
		char *Journal;				// Name of the journal file
		int (*BusyHandler)(void*);	// Function to call when busy
		void *BusyHandlerArg;		// BContext argument for xBusyHandler
		int Stats[3];               // Total cache hits, misses and writes
#ifdef TEST
		int Reads;                  // Database pages read
#endif
		void (*Reiniter)(IPage *);	// Call this routine when reloading pages
#ifdef HAS_CODEC
		void *(*Codec)(void *, void *, Pid, int);	// Routine for en/decoding data
		void (*CodecSizeChange)(void *, int, int);	// Notify of page size changes
		void (*CodecFree)(void *);					// Destructor for the codec
		void *CodecArg;								// First argument to xCodec... methods
#endif
		void *TmpSpace;				// Pager.pageSize bytes of space for tmp use
		PCache *PCache;				// Pointer to page cache object
#ifndef OMIT_WAL
		Wal *Wal;					// Write-ahead log used by "journal_mode=wal"
		char *WalName;              // File name for write-ahead log
#else
		Wal *Wal;
#endif
		// Open and close a Pager connection. 
		__device__ static RC Open(VSystem *vfs, Pager **pagerOut, const char *filename, int extraBytes, IPager::PAGEROPEN flags, VSystem::OPEN vfsFlags, void (*reinit)(IPage *));
		__device__ RC Close();
		__device__ RC ReadFileheader(int n, unsigned char *dest);

		// Functions used to configure a Pager object.
		__device__ void SetBusyhandler(int (*busyHandler)(void *), void *busyHandlerArg);
		__device__ RC SetPageSize(uint32 *pageSizeRef, int reserveBytes);
		__device__ int MaxPages(int maxPages);
		__device__ void SetCacheSize(int maxPages);
		__device__ void Shrink();
		__device__ void SetSafetyLevel(int level, bool fullFsync, bool checkpointFullFsync);
		__device__ IPager::LOCKINGMODE LockingMode(IPager::LOCKINGMODE mode);
		__device__ IPager::JOURNALMODE SetJournalMode(IPager::JOURNALMODE mode);
		__device__ IPager::JOURNALMODE Pager::GetJournalMode();
		__device__ bool OkToChangeJournalMode();
		__device__ int64 SetJournalSizeLimit(int64 limit);
		//__device__ IBackup **BackupPtr();

		// Functions used to obtain and release page references.
		__device__ RC Acquire(Pid id, IPage **pageOut, bool noContent);
		__device__ IPage *Lookup(Pid id);
		__device__ static void Ref(IPage *pg);
		__device__ static void Unref(IPage *pg);

		// Operations on page references.
		__device__ static RC Write(IPage *page);
		__device__ static void DontWrite(IPage *page);
		__device__ RC Movepage(IPage *pg, Pid id, bool isCommit);
		__device__ static int get_PageRefs(IPage *page);
		__device__ static void *GetData(IPage *pg);
		__device__ static void *GetExtra(IPage *pg);

		// Functions used to manage pager transactions and savepoints.
		__device__ void Pages(Pid *pagesOut);
		__device__ RC Begin(bool exFlag, bool subjInMemory);
		__device__ RC CommitPhaseOne(const char *master, bool noSync);
		__device__ RC ExclusiveLock();
		__device__ RC Sync();
		__device__ RC CommitPhaseTwo();
		__device__ RC Rollback();
		__device__ RC OpenSavepoint(int savepoints);
		__device__ RC Savepoint(IPager::SAVEPOINT op, int savepoints);
		__device__ RC SharedLock();
#ifndef OMIT_WAL
		__device__ RC Checkpoint(int mode, int *logs, int *checkpoints);
		__device__ bool WalSupported();
		__device__ RC WalCallback();
		__device__ RC OpenWal(bool *opened);
		__device__ RC CloseWal();
#endif
#ifdef ENABLE_ZIPVFS
		__device__ int WalFramesize();
#endif

		// Functions used to query pager state and configuration.
		__device__ bool get_Readonly();
		__device__ int get_Refs();
		__device__ int get_MemUsed();
		__device__ const char *get_Filename(bool nullIfMemDb);
		__device__ const VSystem *get_Vfs();
		__device__ VFile *get_File();
		__device__ const char *get_Journalname();
		__device__ bool get_NoSync();
#ifdef HAS_CODEC
		__device__ void SetCodec(void *(*codec)(void *,void *, Pid, int), void (*codecSizeChange)(void *, int, int), void (*codecFree)(void *), void *codecArg);
		__device__ void *GetCodec();
#endif
		__device__ void *get_TempSpace();
		__device__ bool get_MemoryDB();
		__device__ void CacheStat(int dbStatus, bool reset, int *value);
		__device__ void ClearCache();
		__device__ static int get_SectorSize(VFile *file);

		// Functions used to truncate the database file.
		__device__ void TruncateImage(Pid pages);

#if defined(HAS_CODEC) && !defined(OMIT_WAL)
		__device__ void *get_Codec(IPage *pg);
#endif
		// Functions to support testing and debugging.
#if !defined(_DEBUG) || defined(TEST)
		__device__ static Pid get_PageID(IPage *pg);
		__device__ static bool Iswriteable(IPage *pg);
#endif
#ifdef TEST
		__device__ int *get_Stats();
#endif
	};

#ifdef TEST
	__device__ void disable_simulated_io_errors();
	__device__ void enable_simulated_io_errors();
#else
#define disable_simulated_io_errors()
#define enable_simulated_io_errors()
#endif
}