using Pid = System.UInt32;
using IPage = Core.PgHdr;
using System;
using Core.IO;

namespace Core
{
    public partial class IPager
    {
        // NOTE: These values must match the corresponding BTREE_ values in btree.h.
        [Flags]
        public enum PAGEROPEN : byte
        {
            OMIT_JOURNAL = 0x0001,  // Do not use a rollback journal
            MEMORY = 0x0002,		// In-memory database
        }

        public enum LOCKINGMODE : sbyte
        {
            QUERY = -1,
            NORMAL = 0,
            EXCLUSIVE = 1,
        }

        [Flags]
        public enum JOURNALMODE : sbyte
        {
            JQUERY = -1,     // Query the value of journalmode
            DELETE = 0,     // Commit by deleting journal file
            PERSIST = 1,    // Commit by zeroing journal header
            OFF = 2,        // Journal omitted.
            TRUNCATE = 3,   // Commit by truncating journal
            JMEMORY = 4,     // In-memory journal file
            WAL = 5,        // Use write-ahead logging
        }

        // sqlite3.h
        public enum CHECKPOINT : byte
        {
            PASSIVE = 0,
            FULL = 1,
            RESTART = 2,
        }

        // sqliteInt.h
        public enum SAVEPOINT : byte
        {
            BEGIN = 0,
            RELEASE = 1,
            ROLLBACK = 2,
        }
    }

    public partial class Pager
    {
        // sqliteLimit.h
        public const int MAX_PAGE_SIZE = 65535;

        enum PAGER : byte
        {
            OPEN = 0,
            READER = 1,
            WRITER_LOCKED = 2,
            WRITER_CACHEMOD = 3,
            WRITER_DBMOD = 4,
            WRITER_FINISHED = 5,
            ERROR = 6,
        }

        VSystem Vfs;             // OS functions to use for IO
        IPager.LOCKINGMODE ExclusiveMode;   // Boolean. True if locking_mode==EXCLUSIVE
        IPager.JOURNALMODE JournalMode;     // One of the PAGER_JOURNALMODE_* values
        bool UseJournal;             // Use a rollback journal on this file
        bool NoSync;                 // Do not sync the journal if true
        bool FullSync;               // Do extra syncs of the journal for robustness
        VFile.SYNC CheckpointSyncFlags;    // SYNC_NORMAL or SYNC_FULL for checkpoint
        VFile.SYNC WalSyncFlags;     // SYNC_NORMAL or SYNC_FULL otherwise
        VFile.SYNC SyncFlags;        // SYNC_NORMAL or SYNC_FULL otherwise
        bool TempFile;               // zFilename is a temporary file
        bool ReadOnly;               // True for a read-only database
        bool MemoryDB;               // True to inhibit all file I/O
        // The following block contains those class members that change during routine opertion.  Class members not in this block are either fixed
        // when the pager is first created or else only change when there is a significant mode change (such as changing the page_size, locking_mode,
        // or the journal_mode).  From another view, these class members describe the "state" of the pager, while other class members describe the "configuration" of the pager.
        PAGER State;                 // Pager state (OPEN, READER, WRITER_LOCKED..) 
        VFile.LOCK Lock;             // Current lock held on database file 
        bool ChangeCountDone;        // Set after incrementing the change-counter 
        bool SetMaster;              // True if a m-j name has been written to jrnl 
        byte DoNotSpill;             // Do not spill the cache when non-zero 
        byte DoNotSyncSpill;         // Do not do a spill that requires jrnl sync 
        bool SubjInMemory;           // True to use in-memory sub-journals 
        Pid DBSize;                  // Number of pages in the database 
        Pid DBOrigSize;              // dbSize before the current transaction 
        Pid DBFileSize;              // Number of pages in the database file 
        Pid DBHintSize;              // Value passed to FCNTL_SIZE_HINT call 
        RC ErrorCode;                // One of several kinds of errors 
        int Records;                 // Pages journalled since last j-header written 
        uint ChecksumInit;           // Quasi-random value added to every checksum 
        uint SubRecords;             // Number of records written to sub-journal 
        Bitvec InJournal;            // One bit for each page in the database file 
        VFile File;                  // File descriptor for database 
        VFile JournalFile;           // File descriptor for main journal 
        VFile SubJournalFile;        // File descriptor for sub-journal 
        long JournalOffset;          // Current write offset in the journal file 
        long JournalHeader;          // Byte offset to previous journal header 
        public IBackup Backup;          // Pointer to list of ongoing backup processes 
        PagerSavepoint[] Savepoints; // Array of active savepoints 
        byte[] DBFileVersion = new byte[16];    // Changes whenever database file changes
        // End of the routinely-changing class members
        ushort ExtraBytes;           // Add this many bytes to each in-memory page
        short ReserveBytes;          // Number of unused bytes at end of each page
        VSystem.OPEN VfsFlags;   // Flags for VirtualFileSystem.xOpen() 
        uint SectorSize;             // Assumed sector size during rollback 
        int PageSize;                // Number of bytes in a page 
        Pid MaxPid;                  // Maximum allowed size of the database 
        long JournalSizeLimit;       // Size limit for persistent journal files 
        string Filename;             // Name of the database file 
        string Journal;              // Name of the journal file 
        Func<object, int> BusyHandler;  // Function to call when busy 
        object BusyHandlerArg;       // BContext argument for xBusyHandler 
        int[] Stats = new int[3];    // Total cache hits, misses and writes
#if TEST
        int Reads;                   // Database pages read
#endif
        Action<IPage> Reiniter;	    // Call this routine when reloading pages
#if HAS_CODEC
        Func<object, object, Pid, int, object> Codec;    // Routine for en/decoding data
        Action<object, int, int> CodecSizeChange;        // Notify of page size changes
        Action<object> CodecFree;                        // Destructor for the codec
        object CodecArg;                                 // First argument to xCodec... methods
#endif
        byte[] TmpSpace;				// Pager.pageSize bytes of space for tmp use
        PCache PCache;				// Pointer to page cache object
#if !OMIT_WAL
        Wal Wal;					    // Write-ahead log used by "journal_mode=wal"
        char WalName;                // File name for write-ahead log
#else
        // For C#
        Wal Wal;
#endif

    }
}