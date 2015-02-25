using Pid = System.UInt32;
using IPage = Core.PgHdr;
using System;
using System.Diagnostics;
using Core.IO;

namespace Core
{
    public partial class Btree
    {
        static ushort MX_CELL_SIZE(BtShared bt) { return (ushort)(bt.PageSize - 8); }
        static ushort MX_CELL(BtShared bt) { return (ushort)((bt.PageSize - 8) / 6); }

        const string FILE_HEADER = "SQLite format 3\0";
        const int DEFAULT_CACHE_SIZE = 2000;
        const int MASTER_ROOT = 1; // The root-page of the master database table.

        const byte PTF_INTKEY = 0x01;
        const byte PTF_ZERODATA = 0x02;
        const byte PTF_LEAFDATA = 0x04;
        const byte PTF_LEAF = 0x08;

        public struct OverflowCell // Cells that will not fit on aData[]
        {
            internal byte[] Cell;     // Pointers to the body of the overflow cell
            internal OverflowCell Copy()
            {
                var cp = new OverflowCell();
                if (Cell != null)
                {
                    cp.Cell = C._alloc(Cell.Length);
                    Buffer.BlockCopy(Cell, 0, cp.Cell, 0, Cell.Length);
                }
                return cp;
            }
        };

        public class MemPage
        {
            internal bool IsInit;               // True if previously initialized. MUST BE FIRST!
            internal byte Overflows;            // Number of overflow cell bodies in aCell[]
            internal bool IntKey;               // True if intkey flag is set
            internal bool Leaf;                 // True if leaf flag is set
            internal bool HasData;              // True if this page stores data
            internal byte HdrOffset;            // 100 for page 1.  0 otherwise
            internal byte ChildPtrSize;         // 0 if leaf.  4 if !leaf
            internal byte Max1bytePayload;      // min(maxLocal,127)
            internal ushort MaxLocal;           // Copy of BtShared.maxLocal or BtShared.maxLeaf
            internal ushort MinLocal;           // Copy of BtShared.minLocal or BtShared.minLeaf
            internal ushort CellOffset;         // Index in aData of first cell pou16er
            internal ushort Frees;              // Number of free bytes on the page
            internal ushort Cells;              // Number of cells on this page, local and ovfl
            internal ushort MaskPage;           // Mask for page offset
            internal ushort[] OvflIdxs = new ushort[5];
            internal OverflowCell[] Ovfls = new OverflowCell[5];
            internal BtShared Bt;               // Pointer to BtShared that this page is part of
            internal byte[] Data;               // Pointer to disk image of the page data
            //object byte[] DataEnd;			// One byte past the end of usable data
            //object byte[] CellIdx;			// The cell index area
            internal IPage DBPage;              // Pager page handle
            internal Pid ID;                    // Page number for this page

            internal MemPage memcopy()
            {
                var cp = (MemPage)MemberwiseClone();
                //if (Overflows != null)
                //{
                //    cp.Overflows = new OverflowCell[Overflows.Length];
                //    for (int i = 0; i < Overflows.Length; i++)
                //        cp.Overflows[i] = Overflows[i].Copy();
                //}
                //if (Data != null)
                //{
                //    cp.Data = C._alloc(Data.Length);
                //    Buffer.BlockCopy(Data, 0, cp.Data, 0, Data.Length);
                //}
                return cp;
            }
        };

        const int EXTRA_SIZE = 0; // No used in C#, since we use create a class; was MemPage.Length;

        public enum BTS : ushort
        {
            READ_ONLY = 0x0001,		// Underlying file is readonly
            PAGESIZE_FIXED = 0x0002,// Page size can no longer be changed
            SECURE_DELETE = 0x0004, // PRAGMA secure_delete is enabled
            INITIALLY_EMPTY = 0x0008, // Database was empty at trans start
            NO_WAL = 0x0010,		// Do not open write-ahead-log files
            EXCLUSIVE = 0x0020,		// pWriter has an exclusive lock
            PENDING = 0x0040,		// Waiting for read-locks to clear
        }

        public class BtShared
        {
            static int _autoID; // For C#
            internal int AutoID; // For C#
            public BtShared() { AutoID = _autoID++; } // For C#

            internal Pager Pager;             // The page cache
            internal BContext Ctx;             // Database connection currently using this Btree
            internal BtCursor Cursor;         // A list of all open cursors
            internal MemPage Page1;           // First page of the database
            internal Btree.OPEN OpenFlags;    // Flags to sqlite3BtreeOpen()
#if !OMIT_AUTOVACUUM
            internal bool AutoVacuum;         // True if auto-vacuum is enabled
            internal bool IncrVacuum;         // True if incr-vacuum is enabled
            internal bool DoTruncate;		    // True to truncate db on commit
#endif
            internal TRANS InTransaction;     // Transaction state
            internal byte Max1bytePayload;	// Maximum first byte of cell for a 1-byte payload
            internal BTS BtsFlags;			// Boolean parameters.  See BTS_* macros below
            internal ushort MaxLocal;         // Maximum local payload in non-LEAFDATA tables
            internal ushort MinLocal;         // Minimum local payload in non-LEAFDATA tables
            internal ushort MaxLeaf;          // Maximum local payload in a LEAFDATA table
            internal ushort MinLeaf;          // Minimum local payload in a LEAFDATA table
            internal uint PageSize;           // Total number of bytes on a page
            internal uint UsableSize;         // Number of usable bytes on each page
            internal int Transactions;        // Number of open transactions (read + write)
            internal Pid Pages;               // Number of pages in the database
            internal Schema Schema;           // Pointer to space allocated by sqlite3BtreeSchema()
            internal Action<Schema> FreeSchema; // Destructor for BtShared.pSchema
            internal MutexEx Mutex;           // Non-recursive mutex required to access this object
            internal Bitvec HasContent;       // Set of pages moved to free-list this transaction
#if !OMIT_SHARED_CACHE
            internal int Refs;                // Number of references to this structure
            internal BtShared Next;           // Next on a list of sharable BtShared structs
            internal BtLock Lock;             // List of locks held on this shared-btree struct
            internal Btree Writer;            // Btree with currently open write transaction
#endif
            internal byte[] TmpSpace;         // BtShared.pageSize bytes of space for tmp use
        }

        public struct CellInfo
        {
            internal uint Cell_;    // Offset to start of cell content -- Needed for C#
            internal long Key;        // The key for INTKEY tables, or number of bytes in key
            internal byte[] Cell;     // Pointer to the start of cell content
            internal uint Data;       // Number of bytes of data
            internal uint Payload;    // Total amount of payload
            internal ushort Header;   // Size of the cell content header in bytes
            internal ushort Local;    // Amount of payload held locally
            internal ushort Overflow; // Offset to overflow page number.  Zero if no overflow
            internal ushort Size;     // Size of the cell content on the main b-tree page
            internal bool Equals(CellInfo ci)
            {
                if (ci.Cell_ >= ci.Cell.Length || Cell_ >= this.Cell.Length)
                    return false;
                if (ci.Cell[ci.Cell_] != this.Cell[Cell_])
                    return false;
                if (ci.Key != this.Key || ci.Data != this.Data || ci.Payload != this.Payload)
                    return false;
                if (ci.Header != this.Header || ci.Local != this.Local)
                    return false;
                if (ci.Overflow != this.Overflow || ci.Size != this.Size)
                    return false;
                return true;
            }
        }

        const int BTCURSOR_MAX_DEPTH = 20;

        public enum CURSOR : byte
        {
            INVALID = 0,
            VALID = 1,
            REQUIRESEEK = 2,
            FAULT = 3,
        }

        public class BtCursor
        {
            internal Btree Btree;             // The Btree to which this cursor belongs
            internal BtShared Bt;             // The BtShared this cursor points to
            internal BtCursor Next, Prev;     // Forms a linked list of all cursors
            internal KeyInfo KeyInfo;         // Argument passed to comparison function
#if !OMIT_INCRBLOB
            internal Pid[] Overflows;         // Cache of overflow page locations
#endif
            internal Pid RootID;              // The root page of this tree
            internal long CachedRowID;        // Next rowid cache.  0 means not valid
            internal CellInfo Info = new CellInfo();           // A parse of the cell we are pointing at
            internal long KeyLength;          // Size of pKey, or last integer key
            internal byte[] Key;              // Saved key that was cursor's last known position
            internal int SkipNext;            // Prev() is noop if negative. Next() is noop if positive
            internal bool WrFlag;             // True if writable
            internal byte AtLast;             // VdbeCursor pointing to the last entry
            internal bool ValidNKey;          // True if info.nKey is valid
            internal CURSOR State;            // One of the CURSOR_XXX constants (see below)
#if !OMIT_INCRBLOB
            internal bool IsIncrblobHandle;   // True if this cursor is an incr. io handle
#endif
            internal byte Hints;			    // As configured by CursorSetHints()
            internal short ID;           // Index of current page in apPage
            internal ushort[] Idxs = new ushort[BTCURSOR_MAX_DEPTH]; // Current index in apPage[i]
            internal MemPage[] Pages = new MemPage[BTCURSOR_MAX_DEPTH]; // Pages from root to current page
            internal void memset()
            {
                Next = Prev = null;
                KeyInfo = null;
                RootID = 0;
                CachedRowID = 0;
                Info = new CellInfo();
                WrFlag = false;
                AtLast = 0;
                ValidNKey = false;
                State = 0;
                KeyLength = 0;
                Key = null;
                SkipNext = 0;
#if !OMIT_INCRBLOB
                IsIncrblobHandle = false;
                Overflows = null;
#endif
                ID = 0;
            }
            public BtCursor Copy()
            {
                var cp = (BtCursor)MemberwiseClone();
                return cp;
            }

#if !OMIT_INCRBLOB
            public void EnterCursor() { Btree.Enter(); }
            public void LeaveCursor() { Btree.Leave(); }
#endif
        }

        public static Pid PENDING_BYTE_PAGE(BtShared bt) { return Pager.MJ_PID(bt.Pager); }

        static Pid PTRMAP_PAGENO(BtShared bt, Pid id) { return ptrmapPageno(bt, id); }
        static Pid PTRMAP_PTROFFSET(Pid ptrmapID, Pid id) { return (5 * (id - ptrmapID - 1)); }
        static bool PTRMAP_ISPAGE(BtShared bt, Pid id) { return (PTRMAP_PAGENO((bt), (id)) == (id)); }

        internal enum PTRMAP : byte
        {
            ROOTPAGE = 1,
            FREEPAGE = 2,
            OVERFLOW1 = 3,
            OVERFLOW2 = 4,
            BTREE = 5,
        }

#if DEBUG
        static void btreeIntegrity(Btree p)
        {
            Debug.Assert(p.Bt.InTransaction != TRANS.NONE || p.Bt.Transactions == 0);
            Debug.Assert(p.Bt.InTransaction >= p.InTrans);
        }
#else
static void btreeIntegrity(Btree p) { }
#endif

        internal class IntegrityCk
        {
            public BtShared Bt;         // The tree being checked out
            public Pager Pager;         // The associated pager.  Also accessible by pBt.pPager
            public byte[] PgRefs;		// 1 bit per page in the db (see above)
            public Pid Pages;           // Number of pages in the database
            public int MaxErrors;       // Stop accumulating errors when this reaches zero
            public int Errors;          // Number of messages written to zErrMsg so far
            public bool MallocFailed;   // A memory allocation error has occurred
            public Text.StringBuilder ErrMsg = new Text.StringBuilder(); // Accumulate the error message text here
        };
    }
}
