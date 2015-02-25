// wal.h
namespace Core
{
	struct Wal
	{
#ifdef OMIT_WAL

		__device__ inline static RC Open(VSystem *vfs, VFile *dbFile, const char *walName, bool noShm, int64 maxWalSize, Wal **walOut) { return RC_OK; }
		__device__ inline void Limit(int64 limit) { }
		__device__ inline RC Close(VFile::SYNC sync_flags, int bufLength, uint8 *buf) { return RC_OK; }
		__device__ inline RC BeginReadTransaction(bool *changed) { return RC_OK; }
		__device__ inline void EndReadTransaction() { }
		__device__ inline RC Read(Pid id, bool *inWal, int bufLength, uint8 *buf) { return RC_OK; }
		__device__ inline Pid DBSize() { return 0; }
		__device__ inline RC BeginWriteTransaction() { return RC_OK; }
		__device__ inline RC EndWriteTransaction() { return RC_OK; }
		__device__ inline RC Undo(RC (*undo)(void *, Pid), void *undoCtx) { return RC_OK; }
		__device__ inline void Savepoint(uint32 *walData) { }
		__device__ inline RC SavepointUndo(uint32 *walData) { return RC_OK; }
		__device__ inline RC Frames(int sizePage, PgHdr *list, Pid truncate, bool isCommit, VFile::SYNC sync_flags) { return RC_OK; }
		__device__ inline RC Checkpoint(int mode, int (*busy)(void*), void *busyArg, VFile::SYNC sync_flags, int bufLength, uint8 *buf, int *logs, int *checkpoints) { *logs = 0, *checkpoints = 0; return RC_OK; }
		__device__ inline int get_Callback() { return 0; }
		__device__ inline bool ExclusiveMode(int op) { return false; }
		__device__ inline bool get_HeapMemory() { return false; }
#ifdef ENABLE_ZIPVFS
		__device__ inline int get_Framesize() { return 0; }
#endif

#else
		enum MODE : uint8
		{
			MODE_NORMAL = 0,
			MODE_EXCLUSIVE = 1,
			MODE_HEAPMEMORY = 2,
		};

		enum RDONLY : uint8
		{
			RDONLY_RDWR = 0,			// Normal read/write connection
			RDONLY_RDONLY = 1,			// The WAL file is readonly
			RDONLY_SHM_RDONLY = 2,		// The SHM file is readonly
		};

		VSystem *Vfs;					// The VFS used to create pDbFd
		VFile *DBFile;					// File handle for the database file
		VFile *WalFile;					// File handle for WAL file
		uint32 Callback;				// Value to pass to log callback (or 0)
		int64 MaxWalSize;				// Truncate WAL to this size upon reset
		int SizeFirstBlock;				// Size of first block written to WAL file
		//int nWiData;					// Size of array apWiData
		volatile uint32 **WiData;		// Pointer to wal-index content in memory
		uint32 SizePage;                // Database page size
		int16 ReadLock;					// Which read lock is being held.  -1 for none
		uint8 SyncFlags;				// Flags to use to sync header writes
		MODE ExclusiveMode_;			// Non-zero if connection is in exclusive mode
		bool WriteLock;					// True if in a write transaction
		bool CheckpointLock;			// True if holding a checkpoint lock
		RDONLY ReadOnly;				// WAL_RDWR, WAL_RDONLY, or WAL_SHM_RDONLY
		bool TruncateOnCommit;			// True to truncate WAL file on commit
		uint8 SyncHeader;				// Fsync the WAL header if true
		uint8 PadToSectorBoundary;		// Pad transactions out to the next sector
		struct IndexHeader
		{
			uint32 Version;                 // Wal-index version
			uint32 Unused;					// Unused (padding) field
			uint32 Change;                  // Counter incremented each transaction
			bool IsInit;					// 1 when initialized
			bool BigEndianChecksum;			// True if checksums in WAL are big-endian
			uint16 SizePage;                // Database page size in bytes. 1==64K
			uint32 MaxFrame;                // Index of last valid frame in the WAL
			uint32 Pages;                   // Size of database in pages
			uint32 FrameChecksum[2];		// Checksum of last frame in log
			uint32 Salt[2];					// Two salt values copied from WAL header
			uint32 Checksum[2];				// Checksum over all prior fields
		} Header; // Wal-index header for current transaction
		const char *WalName;			// Name of WAL file
		uint32 Checkpoints;				// Checkpoint sequence counter in the wal-header
#ifdef _DEBUG
		uint8 LockError;				// True if a locking error has occurred
#endif
		//
		static RC Open(VSystem *vfs, VFile *dbFile, const char *walName, bool noShm, int64 maxWalSize, Wal **walOut);
		void Limit(int64 limit);
		RC Close(VFile::SYNC sync_flags, int bufLength, uint8 *buf);
		RC BeginReadTransaction(bool *changed);
		void EndReadTransaction();
		RC Read(Pid id, bool *inWal, int bufLength, uint8 *buf);
		Pid DBSize();
		RC BeginWriteTransaction();
		RC EndWriteTransaction();
		RC Undo(int (*undo)(void *, Pid), void *undoCtx);
		void Savepoint(uint32 *walData);
		RC SavepointUndo(uint32 *walData);
		RC Frames(int sizePage, PgHdr *list, Pid truncate, bool isCommit, VFile::SYNC sync_flags);
		RC Checkpoint(IPager::CHECKPOINT mode, int (*busy)(void*), void *busyArg, VFile::SYNC sync_flags, int bufLength, uint8 *buf, int *logs, int *checkpoints);
		int get_Callback();
		bool ExclusiveMode(int op);
		bool get_HeapMemory();
#ifdef ENABLE_ZIPVFS
		int get_Framesize();
#endif

#endif
	};
}