using Pid = System.UInt32;
using IPage = Core.PgHdr;
using System;
using Core.IO;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Core
{
    public partial class Pager
    {
#if DEBUG
        static bool PagerTrace = false;
        static void PAGERTRACE(string x, params object[] args) { if (PagerTrace) Console.WriteLine("p:" + string.Format(x, args)); }
#else
        static void PAGERTRACE(string x, params object[] args) { }
#endif
        static int PAGERID(Pager p) { return p.GetHashCode(); }
        static int FILEHANDLEID(VFile fd) { return fd.GetHashCode(); }

        #region Struct

        static readonly byte[] _journalMagic = new byte[] { 0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7 };

        // sqliteLimit.h
        const int DEFAULT_PAGE_SIZE = 1024;
        const int MAX_DEFAULT_PAGE_SIZE = 8192;
        const int MAX_PAGE_COUNT = 1073741823;
        // pager.h
        const int DEFAULT_JOURNAL_SIZE_LIMIT = -1;

#if HAS_CODEC
        static bool CODEC1(Pager p, byte[] d, Pid id, int x)
        {
            return (p.Codec != null && p.Codec(p.CodecArg, d, id, x) == null);
        }
        static bool CODEC2(ref byte[] o, Pager p, byte[] d, Pid id, int x)
        {
            if (p.Codec == null) { o = d; return false; } else { return ((o = (byte[])p.Codec(p.CodecArg, d, id, x)) == null); }
        }
#else
        static bool CODEC1(Pager p, byte[] d, Pid id, int x) { return false; }
        static bool CODEC2(Pager p, byte[] d, Pid id, int x, ref byte[] o) { o = d; return false; }
#endif

        const int MAX_SECTOR_SIZE = 0x10000;

        class PagerSavepoint
        {
            public long Offset;             // Starting offset in main journal
            public long HdrOffset;          // See above
            public Bitvec InSavepoint;      // Set of pages in this savepoint
            public Pid Orig;                // Original number of pages in file
            public Pid SubRecords;              // Index of first record in sub-journal
#if !OMIT_WAL
            public uint[] WalData = new uint[WAL_SAVEPOINT_NDATA];        // WAL savepoint context
#else
            // For C#
            public object WalData = null;
#endif
        }

        enum STAT : byte
        {
            HIT = 0,
            MISS = 1,
            WRITE = 2,
        }

#if TEST
        static int _readdb_count = 0;    // Number of full pages read from DB
        static int _writedb_count = 0;   // Number of full pages written to DB
        static int _writej_count = 0;    // Number of pages written to journal
        static void PAGER_INCR(ref int v) { v++; }
#else
        static void PAGER_INCR(ref int v) { }
#endif

        static uint JOURNAL_PG_SZ(Pager pager) { return (uint)pager.PageSize + 8; }
        static uint JOURNAL_HDR_SZ(Pager pager) { return pager.SectorSize; }
        internal static Pid MJ_PID(Pager pager) { return ((Pid)((VFile.PENDING_BYTE / ((pager).PageSize)) + 1)); }

        const int MAX_PID = 2147483647;

#if !OMIT_WAL
        internal static bool UseWal(Pager pager) { return (pager.Wal != null); }
#else
        internal bool UseWal() { return false; }
        internal RC pagerRollbackWal() { return RC.OK; }
        internal RC pagerWalFrames(PgHdr w, Pid x, bool y) { return RC.OK; }
        internal RC pagerOpenWalIfPresent() { return RC.OK; }
        internal RC pagerBeginReadTransaction() { return RC.OK; }
#endif

        #endregion

        #region Debug
#if DEBUG

        internal static bool assert_pager_state(Pager p)
        {
            // State must be valid.
            Debug.Assert(p.State == PAGER.OPEN ||
                p.State == PAGER.READER ||
                p.State == PAGER.WRITER_LOCKED ||
                p.State == PAGER.WRITER_CACHEMOD ||
                p.State == PAGER.WRITER_DBMOD ||
                p.State == PAGER.WRITER_FINISHED ||
                p.State == PAGER.ERROR);

            // Regardless of the current state, a temp-file connection always behaves as if it has an exclusive lock on the database file. It never updates
            // the change-counter field, so the changeCountDone flag is always set.
            Debug.Assert(!p.TempFile || p.Lock == VFile.LOCK.EXCLUSIVE);
            Debug.Assert(!p.TempFile || p.ChangeCountDone);

            // If the useJournal flag is clear, the journal-mode must be "OFF". And if the journal-mode is "OFF", the journal file must not be open.
            Debug.Assert(p.JournalMode == IPager.JOURNALMODE.OFF || p.UseJournal);
            Debug.Assert(p.JournalMode != IPager.JOURNALMODE.OFF || !p.JournalFile.Opened);

            // Check that MEMDB implies noSync. And an in-memory journal. Since  this means an in-memory pager performs no IO at all, it cannot encounter 
            // either SQLITE_IOERR or SQLITE_FULL during rollback or while finalizing a journal file. (although the in-memory journal implementation may 
            // return SQLITE_IOERR_NOMEM while the journal file is being written). It is therefore not possible for an in-memory pager to enter the ERROR state.
            if (p.MemoryDB)
            {
                Debug.Assert(p.NoSync);
                Debug.Assert(p.JournalMode == IPager.JOURNALMODE.OFF || p.JournalMode == IPager.JOURNALMODE.JMEMORY);
                Debug.Assert(p.State != PAGER.ERROR && p.State != PAGER.OPEN);
                Debug.Assert(!p.UseWal());
            }

            // If changeCountDone is set, a RESERVED lock or greater must be held on the file.
            Debug.Assert(!p.ChangeCountDone || p.Lock >= VFile.LOCK.RESERVED);
            Debug.Assert(p.Lock != VFile.LOCK.PENDING);

            switch (p.State)
            {
                case PAGER.OPEN:
                    Debug.Assert(!p.MemoryDB);
                    Debug.Assert(p.ErrorCode == RC.OK);
                    Debug.Assert(p.PCache.get_Refs() == 0 || p.TempFile);
                    break;

                case PAGER.READER:
                    Debug.Assert(p.ErrorCode == RC.OK);
                    Debug.Assert(p.Lock != VFile.LOCK.UNKNOWN);
                    Debug.Assert(p.Lock >= VFile.LOCK.SHARED);
                    break;

                case PAGER.WRITER_LOCKED:
                    Debug.Assert(p.Lock != VFile.LOCK.UNKNOWN);
                    Debug.Assert(p.ErrorCode == RC.OK);
                    if (!p.UseWal())
                        Debug.Assert(p.Lock >= VFile.LOCK.RESERVED);
                    Debug.Assert(p.DBSize == p.DBOrigSize);
                    Debug.Assert(p.DBOrigSize == p.DBFileSize);
                    Debug.Assert(p.DBOrigSize == p.DBHintSize);
                    Debug.Assert(!p.SetMaster);
                    break;

                case PAGER.WRITER_CACHEMOD:
                    Debug.Assert(p.Lock != VFile.LOCK.UNKNOWN);
                    Debug.Assert(p.ErrorCode == RC.OK);
                    if (!p.UseWal())
                    {
                        // It is possible that if journal_mode=wal here that neither the journal file nor the WAL file are open. This happens during
                        // a rollback transaction that switches from journal_mode=off to journal_mode=wal.
                        Debug.Assert(p.Lock >= VFile.LOCK.RESERVED);
                        Debug.Assert(p.JournalFile.Opened || p.JournalMode == IPager.JOURNALMODE.OFF || p.JournalMode == IPager.JOURNALMODE.WAL);
                    }
                    Debug.Assert(p.DBOrigSize == p.DBFileSize);
                    Debug.Assert(p.DBOrigSize == p.DBHintSize);
                    break;

                case PAGER.WRITER_DBMOD:
                    Debug.Assert(p.Lock == VFile.LOCK.EXCLUSIVE);
                    Debug.Assert(p.ErrorCode == RC.OK);
                    Debug.Assert(!p.UseWal());
                    Debug.Assert(p.Lock >= VFile.LOCK.EXCLUSIVE);
                    Debug.Assert(p.JournalFile.Opened || p.JournalMode == IPager.JOURNALMODE.OFF || p.JournalMode == IPager.JOURNALMODE.WAL);
                    Debug.Assert(p.DBOrigSize <= p.DBHintSize);
                    break;

                case PAGER.WRITER_FINISHED:
                    Debug.Assert(p.Lock == VFile.LOCK.EXCLUSIVE);
                    Debug.Assert(p.ErrorCode == RC.OK);
                    Debug.Assert(!p.UseWal());
                    Debug.Assert(p.JournalFile.Opened || p.JournalMode == IPager.JOURNALMODE.OFF || p.JournalMode == IPager.JOURNALMODE.WAL);
                    break;

                case PAGER.ERROR:
                    // There must be at least one outstanding reference to the pager if in ERROR state. Otherwise the pager should have already dropped back to OPEN state.
                    Debug.Assert(p.ErrorCode != RC.OK);
                    Debug.Assert(p.PCache.get_Refs() > 0);
                    break;
            }

            return true;
        }

        internal static string print_pager_state(Pager p)
        {
            return string.Format(@"
Filename:      {0}
State:         {1} errCode={2}
Lock:          {3}
Locking mode:  locking_mode={4}
Journal mode:  journal_mode={5}
Backing store: tempFile={6} memDb={7} useJournal={8}
Journal:       journalOff={9.11} journalHdr={10.11}
Size:          dbsize={11} dbOrigSize={12} dbFileSize={13}"
          , p.Filename
          , p.State == PAGER.OPEN ? "OPEN" :
              p.State == PAGER.READER ? "READER" :
              p.State == PAGER.WRITER_LOCKED ? "WRITER_LOCKED" :
              p.State == PAGER.WRITER_CACHEMOD ? "WRITER_CACHEMOD" :
              p.State == PAGER.WRITER_DBMOD ? "WRITER_DBMOD" :
              p.State == PAGER.WRITER_FINISHED ? "WRITER_FINISHED" :
              p.State == PAGER.ERROR ? "ERROR" : "?error?"
          , (int)p.ErrorCode
          , p.Lock == VFile.LOCK.NO ? "NO_LOCK" :
              p.Lock == VFile.LOCK.RESERVED ? "RESERVED" :
              p.Lock == VFile.LOCK.EXCLUSIVE ? "EXCLUSIVE" :
              p.Lock == VFile.LOCK.SHARED ? "SHARED" :
              p.Lock == VFile.LOCK.UNKNOWN ? "UNKNOWN" : "?error?"
          , p.ExclusiveMode ? "exclusive" : "normal"
          , p.JournalMode == IPager.JOURNALMODE.JMEMORY ? "memory" :
              p.JournalMode == IPager.JOURNALMODE.OFF ? "off" :
              p.JournalMode == IPager.JOURNALMODE.DELETE ? "delete" :
              p.JournalMode == IPager.JOURNALMODE.PERSIST ? "persist" :
              p.JournalMode == IPager.JOURNALMODE.TRUNCATE ? "truncate" :
              p.JournalMode == IPager.JOURNALMODE.WAL ? "wal" : "?error?"
          , (p.TempFile ? 1 : 0), (p.MemoryDB ? 1 : 0), (p.UseJournal ? 1 : 0)
          , p.JournalOffset, p.JournalHeader
          , (int)p.DBSize, (int)p.DBOrigSize, (int)p.DBFileSize);
        }

#endif
        #endregion

        #region Name1

        private static bool subjRequiresPage(PgHdr pg)
        {
            var id = pg.ID;
            var pager = pg.Pager;
            for (var i = 0; i < pager.Savepoints.Length; i++)
            {
                var p = pager.Savepoints[i];
                if (p.Orig >= id && !p.InSavepoint.Get(id))
                    return true;
            }
            return false;
        }

        private static bool pageInJournal(PgHdr pg)
        {
            return pg.Pager.InJournal.Get(pg.ID);
        }

        private RC pagerUnlockDb(VFile.LOCK lock_)
        {
            Debug.Assert(!ExclusiveMode || Lock == lock_);
            Debug.Assert(lock_ == VFile.LOCK.NO || lock_ == VFile.LOCK.SHARED);
            Debug.Assert(lock_ != VFile.LOCK.NO || !UseWal());
            var rc = RC.OK;
            if (File.Opened)
            {
                Debug.Assert(Lock >= lock_);
                rc = File.Unlock(lock_);
                if (Lock != VFile.LOCK.UNKNOWN)
                    Lock = lock_;
                SysEx.IOTRACE("UNLOCK {0:x} {1}", this, lock_);
            }
            return rc;
        }

        private RC pagerLockDb(VFile.LOCK lock_)
        {
            Debug.Assert(lock_ == VFile.LOCK.SHARED || lock_ == VFile.LOCK.RESERVED || lock_ == VFile.LOCK.EXCLUSIVE);
            var rc = RC.OK;
            if (Lock < lock_ || Lock == VFile.LOCK.UNKNOWN)
            {
                rc = File.Lock(lock_);
                if (rc == RC.OK && (Lock != VFile.LOCK.UNKNOWN || lock_ == VFile.LOCK.EXCLUSIVE))
                {
                    Lock = lock_;
                    SysEx.IOTRACE("LOCK {0:x} {1}", this, lock_);
                }
            }
            return rc;
        }

#if ENABLE_ATOMIC_WRITE
        internal static int jrnlBufferSize(Pager pager)
        {
            Debug.Assert(!pager.MemoryDB);
            if (!pager.TempFile)
            {
                Debug.Assert(pager.File.Opened);
                var dc = pager.File.get_DeviceCharacteristics();
                var sectorSize = pager.SectorSize;
                var pageSize = pager.PageSize;
                Debug.Assert((int)VFile.IOCAP.ATOMIC512 == (512 >> 8));
                Debug.Assert((int)VFile.IOCAP.ATOMIC64K == (65536 >> 8));
                if (!((dc & (VFile.IOCAP.ATOMIC | (VFile.IOCAP)(pageSize >> 8))) != 0 || sectorSize > pageSize))
                    return 0;
            }
            return (int)(JOURNAL_HDR_SZ(pager) + JOURNAL_PG_SZ(pager));
        }
#endif

#if CHECK_PAGES
        internal static uint pager_datahash(int bytes, byte[] data)
        {
            uint hash = 0;
            for (var i = 0; i < bytes; i++)
                hash = (hash * 1039) + data[i];
            return hash;
        }
        internal static uint pager_pagehash(PgHdr page) { return pager_datahash(page.Pager.PageSize, page.Data); }
        internal static void pager_set_pagehash(PgHdr page) { page.PageHash = pager_pagehash(page); }
        internal static void checkPage(PgHdr page)
        {
            var pager = page.Pager;
            Debug.Assert(pager.State != PAGER.ERROR);
            Debug.Assert((page.Flags & PgHdr.PGHDR.DIRTY) != 0 || page.PageHash == pager_pagehash(page));
        }
#else
        internal static uint pager_pagehash(PgHdr x) { return 0; }
        internal static uint pager_datahash(int x, byte[] y) { return 0; }
        internal static void pager_set_pagehash(PgHdr x) { }
        internal static void checkPage(PgHdr x) { }
#endif

        #endregion

        #region Journal1

        private static RC readMasterJournal(VFile journalFile, out string master, uint masterLength)
        {
            int nameLength = 0; // Length in bytes of master journal name 
            long fileSize = 0; // Total size in bytes of journal file pJrnl 
            uint checksum = 0; // MJ checksum value read from journal
            var magic = new byte[8]; // A buffer to hold the magic header
            var master2 = new byte[masterLength];
            master2[0] = 0;
            RC rc;
            if ((rc = journalFile.get_FileSize(out fileSize)) != RC.OK ||
                fileSize < 16 ||
                (rc = journalFile.Read4((int)(fileSize - 16), out nameLength)) != RC.OK ||
                nameLength >= masterLength ||
                (rc = journalFile.Read4(fileSize - 12, out checksum)) != RC.OK ||
                (rc = journalFile.Read(magic, 8, fileSize - 8)) != RC.OK ||
                Enumerable.SequenceEqual(magic, _journalMagic) ||
                (rc = journalFile.Read(master2, nameLength, (long)(fileSize - 16 - nameLength))) != RC.OK)
            {
                master = null;
                return rc;
            }
            // See if the checksum matches the master journal name
            for (var u = 0U; u < nameLength; u++)
                checksum -= master2[u];
            if (checksum != 0)
            {
                // If the checksum doesn't add up, then one or more of the disk sectors containing the master journal filename is corrupted. This means
                // definitely roll back, so just return SQLITE.OK and report a (nul) master-journal filename.
                nameLength = 0;
            }
            master2[nameLength] = 0;
            master = Encoding.UTF8.GetString(master2);
            return RC.OK;
        }

        private long journalHdrOffset()
        {
            long offset = 0;
            var c = JournalOffset;
            if (c != 0)
                offset = (((c - 1) / JOURNAL_HDR_SZ(this) + 1) * JOURNAL_HDR_SZ(this));
            Debug.Assert(offset % JOURNAL_HDR_SZ(this) == 0);
            Debug.Assert(offset >= c);
            Debug.Assert((offset - c) < JOURNAL_HDR_SZ(this));
            return offset;
        }

        private RC zeroJournalHdr(bool doTruncate)
        {
            Debug.Assert(JournalFile.Opened);
            var rc = RC.OK;
            if (JournalOffset != 0)
            {
                var zeroHeader = new byte[28];
                var limit = JournalSizeLimit; // Local cache of jsl
                SysEx.IOTRACE("JZEROHDR {0:x}", this);
                if (doTruncate || limit == 0)
                    rc = JournalFile.Truncate(0);
                else
                    rc = JournalFile.Write(zeroHeader, zeroHeader.Length, 0);
                if (rc == RC.OK && !NoSync)
                    rc = JournalFile.Sync(VFile.SYNC.DATAONLY | SyncFlags);
                // At this point the transaction is committed but the write lock is still held on the file. If there is a size limit configured for
                // the persistent journal and the journal file currently consumes more space than that limit allows for, truncate it now. There is no need
                // to sync the file following this operation.
                if (rc == RC.OK && limit > 0)
                {
                    long fileSize;
                    rc = JournalFile.get_FileSize(out fileSize);
                    if (rc == RC.OK && fileSize > limit)
                        rc = JournalFile.Truncate(limit);
                }
            }
            return rc;
        }

        private RC writeJournalHdr()
        {
            Debug.Assert(JournalFile.Opened);
            var header = TmpSpace;                  // Temporary space used to build header
            var headerSize = (uint)PageSize;        // Size of buffer pointed to by zHeader
            if (headerSize > JOURNAL_HDR_SZ(this))
                headerSize = JOURNAL_HDR_SZ(this);

            // If there are active savepoints and any of them were created since the most recent journal header was written, update the
            // PagerSavepoint.iHdrOffset fields now.
            for (var ii = 0; ii < Savepoints.Length; ii++)
                if (Savepoints[ii].HdrOffset == 0)
                    Savepoints[ii].HdrOffset = JournalOffset;
            JournalHeader = JournalOffset = journalHdrOffset();

            // Write the nRec Field - the number of page records that follow this journal header. Normally, zero is written to this value at this time.
            // After the records are added to the journal (and the journal synced, if in full-sync mode), the zero is overwritten with the true number
            // of records (see syncJournal()).
            //
            // A faster alternative is to write 0xFFFFFFFF to the nRec field. When reading the journal this value tells SQLite to assume that the
            // rest of the journal file contains valid page records. This assumption is dangerous, as if a failure occurred whilst writing to the journal
            // file it may contain some garbage data. There are two scenarios where this risk can be ignored:
            //   * When the pager is in no-sync mode. Corruption can follow a power failure in this case anyway.
            //   * When the SQLITE_IOCAP_SAFE_APPEND flag is set. This guarantees that garbage data is never appended to the journal file.
            Debug.Assert(File.Opened || NoSync);
            if (NoSync || (JournalMode == IPager.JOURNALMODE.JMEMORY) || (File.get_DeviceCharacteristics() & VFile.IOCAP.SAFE_APPEND) != 0)
            {
                _journalMagic.CopyTo(header, 0);
                ConvertEx.Put4(header, _journalMagic.Length, 0xffffffff);
            }
            else
                Array.Clear(header, 0, _journalMagic.Length + 4);
            SysEx.MakeRandomness(sizeof(long), ref ChecksumInit);
            ConvertEx.Put4(header, _journalMagic.Length + 4, ChecksumInit); // The random check-hash initializer
            ConvertEx.Put4(header, _journalMagic.Length + 8, DBOrigSize);   // The initial database size
            ConvertEx.Put4(header, _journalMagic.Length + 12, SectorSize);  // The assumed sector size for this process
            ConvertEx.Put4(header, _journalMagic.Length + 16, PageSize);    // The page size
            // Initializing the tail of the buffer is not necessary.  Everything works find if the following memset() is omitted.  But initializing
            // the memory prevents valgrind from complaining, so we are willing to take the performance hit.
            Array.Clear(header, _journalMagic.Length + 20, (int)headerSize - _journalMagic.Length + 20);

            // In theory, it is only necessary to write the 28 bytes that the journal header consumes to the journal file here. Then increment the 
            // Pager.journalOff variable by JOURNAL_HDR_SZ so that the next record is written to the following sector (leaving a gap in the file
            // that will be implicitly filled in by the OS).
            //
            // However it has been discovered that on some systems this pattern can be significantly slower than contiguously writing data to the file,
            // even if that means explicitly writing data to the block of (JOURNAL_HDR_SZ - 28) bytes that will not be used. So that is what is done. 
            //
            // The loop is required here in case the sector-size is larger than the database page size. Since the zHeader buffer is only Pager.pageSize
            // bytes in size, more than one call to sqlite3OsWrite() may be required to populate the entire journal header sector.
            RC rc = RC.OK;
            for (var headerWritten = 0U; rc == RC.OK && headerWritten < JOURNAL_HDR_SZ(this); headerWritten += headerSize)
            {
                SysEx.IOTRACE("JHDR {0:x} {1,11} {2}", this, JournalHeader, headerSize);
                rc = JournalFile.Write(header, (int)headerSize, JournalOffset);
                Debug.Assert(JournalHeader <= JournalOffset);
                JournalOffset += (int)headerSize;
            }
            return rc;
        }

        private RC readJournalHdr(bool isHot, long journalSize, ref uint recordsOut, ref uint dbSizeOut)
        {
            Debug.Assert(JournalFile.Opened);

            // Advance Pager.journalOff to the start of the next sector. If the journal file is too small for there to be a header stored at this
            // point, return SQLITE_DONE.
            JournalOffset = journalHdrOffset();
            if (JournalOffset + JOURNAL_HDR_SZ(this) > journalSize)
                return RC.DONE;
            var headerOffset = JournalOffset;

            // Read in the first 8 bytes of the journal header. If they do not match the  magic string found at the start of each journal header, return
            // SQLITE_DONE. If an IO error occurs, return an error code. Otherwise, proceed.
            RC rc;
            var magic = new byte[8];
            if (isHot || headerOffset != JournalHeader)
            {
                rc = JournalFile.Read(magic, magic.Length, headerOffset);
                if (rc != RC.OK)
                    return rc;
                if (Enumerable.SequenceEqual(magic, _journalMagic))
                    return RC.DONE;
            }
            // Read the first three 32-bit fields of the journal header: The nRec field, the checksum-initializer and the database size at the start
            // of the transaction. Return an error code if anything goes wrong.
            if ((rc = JournalFile.Read4(headerOffset + 8, out recordsOut)) != RC.OK ||
                (rc = JournalFile.Read4(headerOffset + 12, out ChecksumInit)) != RC.OK ||
                (rc = JournalFile.Read4(headerOffset + 16, out dbSizeOut)) != RC.OK)
                return rc;

            if (JournalOffset == 0)
            {
                uint pageSize = 0; // Page-size field of journal header
                uint sectorSize = 0; // Sector-size field of journal header
                // Read the page-size and sector-size journal header fields.
                if ((rc = JournalFile.Read4(headerOffset + 20, out sectorSize)) != RC.OK ||
                    (rc = JournalFile.Read4(headerOffset + 24, out pageSize)) != RC.OK)
                    return rc;

                // Versions of SQLite prior to 3.5.8 set the page-size field of the journal header to zero. In this case, assume that the Pager.pageSize
                // variable is already set to the correct page size.
                if (pageSize == 0)
                    pageSize = (uint)PageSize;

                // Check that the values read from the page-size and sector-size fields are within range. To be 'in range', both values need to be a power
                // of two greater than or equal to 512 or 32, and not greater than their respective compile time maximum limits.
                if (pageSize < 512 || sectorSize < 32 ||
                    pageSize > MAX_PAGE_SIZE || sectorSize > MAX_SECTOR_SIZE ||
                    ((pageSize - 1) & pageSize) != 0 || ((sectorSize - 1) & sectorSize) != 0)
                    // If the either the page-size or sector-size in the journal-header is invalid, then the process that wrote the journal-header must have
                    // crashed before the header was synced. In this case stop reading the journal file here.
                    return RC.DONE;

                // Update the page-size to match the value read from the journal. Use a testcase() macro to make sure that malloc failure within PagerSetPagesize() is tested.
                rc = SetPageSize(ref pageSize, -1);

                // Update the assumed sector-size to match the value used by the process that created this journal. If this journal was
                // created by a process other than this one, then this routine is being called from within pager_playback(). The local value
                // of Pager.sectorSize is restored at the end of that routine.
                SectorSize = sectorSize;
            }

            JournalOffset += (int)JOURNAL_HDR_SZ(this);
            return rc;
        }

        private RC writeMasterJournal(string master)
        {
            Debug.Assert(!SetMaster);
            Debug.Assert(!UseWal());

            if (master == null ||
                JournalMode == IPager.JOURNALMODE.JMEMORY ||
                JournalMode == IPager.JOURNALMODE.OFF)
                return RC.OK;
            SetMaster = true;
            Debug.Assert(JournalFile.Opened);
            Debug.Assert(JournalHeader <= JournalOffset);

            // Calculate the length in bytes and the checksum of zMaster
            uint checksum = 0;  // Checksum of string zMaster
            int masterLength; // Length of string zMaster
            for (masterLength = 0; masterLength < master.Length && master[masterLength] != 0; masterLength++)
                checksum += master[masterLength];

            // If in full-sync mode, advance to the next disk sector before writing the master journal name. This is in case the previous page written to
            // the journal has already been synced.
            if (FullSync)
                JournalOffset = journalHdrOffset();
            var headerOffset = JournalOffset; // Offset of header in journal file

            // Write the master journal data to the end of the journal file. If an error occurs, return the error code to the caller.
            RC rc;
            if ((rc = JournalFile.Write4(headerOffset, (uint)MJ_PID(this))) != RC.OK ||
                (rc = JournalFile.Write(Encoding.UTF8.GetBytes(master), masterLength, headerOffset + 4)) != RC.OK ||
                (rc = JournalFile.Write4(headerOffset + 4 + masterLength, (uint)masterLength)) != RC.OK ||
                (rc = JournalFile.Write4(headerOffset + 4 + masterLength + 4, checksum)) != RC.OK ||
                (rc = JournalFile.Write(_journalMagic, 8, headerOffset + 4 + masterLength + 8)) != RC.OK)
                return rc;
            JournalOffset += (masterLength + 20);

            // If the pager is in peristent-journal mode, then the physical journal-file may extend past the end of the master-journal name
            // and 8 bytes of magic data just written to the file. This is dangerous because the code to rollback a hot-journal file
            // will not be able to find the master-journal name to determine whether or not the journal is hot. 
            //
            // Easiest thing to do in this scenario is to truncate the journal file to the required size.
            long journalSize = 0;  // Size of journal file on disk
            if ((rc = JournalFile.get_FileSize(out journalSize)) == RC.OK && journalSize > JournalOffset)
                rc = JournalFile.Truncate(JournalOffset);
            return rc;
        }

        #endregion

        #region Name2

        private PgHdr pager_lookup(Pid id)
        {
            // It is not possible for a call to PcacheFetch() with createFlag==0 to fail, since no attempt to allocate dynamic memory will be made.
            PgHdr p;
            PCache.Fetch(id, false, out p);
            return p;
        }

        private void pager_reset()
        {
            if (Backup != null)
                Backup.Restart();
            PCache.Clear();
        }

        private void releaseAllSavepoints()
        {
            for (var ii = 0; ii < Savepoints.Length; ii++)
                Bitvec.Destroy(ref Savepoints[ii].InSavepoint);
            if (!ExclusiveMode || SubJournalFile is MemoryVFile)
                SubJournalFile.Close();
            Savepoints = null;
            SubRecords = 0;
        }

        private RC addToSavepointBitvecs(Pid id)
        {
            var rc = RC.OK;
            for (var ii = 0; ii < Savepoints.Length; ii++)
            {
                var p = Savepoints[ii];
                if (id <= p.Orig)
                {
                    rc |= p.InSavepoint.Set(id);
                    Debug.Assert(rc == RC.OK || rc == RC.NOMEM);
                }
            }
            return rc;
        }

        private void pager_unlock()
        {
            Debug.Assert(State == PAGER.READER ||
                State == PAGER.OPEN ||
                State == PAGER.ERROR);

            Bitvec.Destroy(ref InJournal);
            InJournal = null;
            releaseAllSavepoints();

            if (UseWal())
            {
                Debug.Assert(!JournalFile.Opened);
                Wal.EndReadTransaction();
                State = PAGER.OPEN;
            }
            else if (!ExclusiveMode)
            {
                // If the operating system support deletion of open files, then close the journal file when dropping the database lock.  Otherwise
                // another connection with journal_mode=delete might delete the file out from under us.
                Debug.Assert(((int)IPager.JOURNALMODE.JMEMORY & 5) != 1);
                Debug.Assert(((int)IPager.JOURNALMODE.OFF & 5) != 1);
                Debug.Assert(((int)IPager.JOURNALMODE.WAL & 5) != 1);
                Debug.Assert(((int)IPager.JOURNALMODE.DELETE & 5) != 1);
                Debug.Assert(((int)IPager.JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)IPager.JOURNALMODE.PERSIST & 5) == 1);
                var dc = (File.Opened ? File.get_DeviceCharacteristics() : 0);
                if ((dc & VFile.IOCAP.UNDELETABLE_WHEN_OPEN) == 0 || ((int)JournalMode & 5) != 1)
                    JournalFile.Close();

                // If the pager is in the ERROR state and the call to unlock the database file fails, set the current lock to UNKNOWN_LOCK. See the comment
                // above the #define for UNKNOWN_LOCK for an explanation of why this is necessary.
                var rc = pagerUnlockDb(VFile.LOCK.NO);
                if (rc != RC.OK && State == PAGER.ERROR)
                    Lock = VFile.LOCK.UNKNOWN;

                // The pager state may be changed from PAGER_ERROR to PAGER_OPEN here without clearing the error code. This is intentional - the error
                // code is cleared and the cache reset in the block below.
                Debug.Assert(ErrorCode != 0 || State != PAGER.ERROR);
                ChangeCountDone = false;
                State = PAGER.OPEN;
            }

            // If Pager.errCode is set, the contents of the pager cache cannot be trusted. Now that there are no outstanding references to the pager,
            // it can safely move back to PAGER_OPEN state. This happens in both normal and exclusive-locking mode.
            if (ErrorCode != 0)
            {
                Debug.Assert(!MemoryDB);
                pager_reset();
                ChangeCountDone = TempFile;
                State = PAGER.OPEN;
                ErrorCode = RC.OK;
            }
            JournalOffset = 0;
            JournalHeader = 0;
            SetMaster = false;
        }

        internal RC pager_error(RC rc)
        {
            var rc2 = (RC)((int)rc & 0xff);
            Debug.Assert(rc == RC.OK || !MemoryDB);
            Debug.Assert(ErrorCode == RC.FULL ||
                ErrorCode == RC.OK ||
                ((int)ErrorCode & 0xff) == (int)RC.IOERR);
            if (rc2 == RC.FULL || rc2 == RC.IOERR)
            {
                ErrorCode = rc;
                State = PAGER.ERROR;
            }
            return rc;
        }

        #endregion

        #region Transaction1

        private RC pager_end_transaction(bool hasMaster, bool commit)
        {
            // Do nothing if the pager does not have an open write transaction or at least a RESERVED lock. This function may be called when there
            // is no write-transaction active but a RESERVED or greater lock is held under two circumstances:
            //
            //   1. After a successful hot-journal rollback, it is called with eState==PAGER_NONE and eLock==EXCLUSIVE_LOCK.
            //
            //   2. If a connection with locking_mode=exclusive holding an EXCLUSIVE lock switches back to locking_mode=normal and then executes a
            //      read-transaction, this function is called with eState==PAGER_READER and eLock==EXCLUSIVE_LOCK when the read-transaction is closed.
            Debug.Assert(assert_pager_state(this));
            Debug.Assert(State != PAGER.ERROR);
            if (State < PAGER.WRITER_LOCKED && Lock < VFile.LOCK.RESERVED)
                return RC.OK;

            releaseAllSavepoints();
            Debug.Assert(JournalFile.Opened || InJournal == null);
            var rc = RC.OK;
            if (JournalFile.Opened)
            {
                Debug.Assert(!UseWal());

                // Finalize the journal file.
                if (VFile.HasMemoryVFile(JournalFile))
                {
                    Debug.Assert(JournalMode == IPager.JOURNALMODE.JMEMORY);
                    JournalFile.Close();
                }
                else if (JournalMode == IPager.JOURNALMODE.TRUNCATE)
                {
                    rc = (JournalOffset == 0 ? RC.OK : JournalFile.Truncate(0));
                    JournalOffset = 0;
                }
                else if (JournalMode == IPager.JOURNALMODE.PERSIST || (ExclusiveMode && JournalMode != IPager.JOURNALMODE.WAL))
                {
                    rc = zeroJournalHdr(hasMaster);
                    JournalOffset = 0;
                }
                else
                {
                    // This branch may be executed with Pager.journalMode==MEMORY if a hot-journal was just rolled back. In this case the journal
                    // file should be closed and deleted. If this connection writes to the database file, it will do so using an in-memory journal.
                    var delete_ = (!TempFile && VFile.HasJournalVFile(JournalFile));
                    Debug.Assert(JournalMode == IPager.JOURNALMODE.DELETE ||
                        JournalMode == IPager.JOURNALMODE.JMEMORY ||
                        JournalMode == IPager.JOURNALMODE.WAL);
                    JournalFile.Close();
                    if (delete_)
                        Vfs.Delete(Journal, false);
                }
            }

#if CHECK_PAGES
            PCache.IterateDirty(pager_set_pagehash);
            if (DBSize == 0 && PCache.get_Refs() > 0)
            {
                var p = pager_lookup(1);
                if (p != null)
                {
                    p.PageHash = 0;
                    Pager.Unref(p);
                }
            }
#endif

            Bitvec.Destroy(ref InJournal); InJournal = null;
            Records = 0;
            PCache.CleanAll();
            PCache.Truncate(DBSize);

            var rc2 = RC.OK;    // Error code from db file unlock operation
            if (UseWal())
            {
                // Drop the WAL write-lock, if any. Also, if the connection was in locking_mode=exclusive mode but is no longer, drop the EXCLUSIVE 
                // lock held on the database file.
                rc2 = Wal.EndWriteTransaction();
                Debug.Assert(rc2 == RC.OK);
            }
            if (!ExclusiveMode && (!UseWal() || Wal.ExclusiveMode(0)))
            {
                rc2 = pagerUnlockDb(VFile.LOCK.SHARED);
                ChangeCountDone = false;
            }
            State = PAGER.READER;
            SetMaster = false;

            return (rc == RC.OK ? rc2 : rc);
        }

        private void pagerUnlockAndRollback()
        {
            if (State != PAGER.ERROR && State != PAGER.OPEN)
            {
                Debug.Assert(assert_pager_state(this));
                if (State >= PAGER.WRITER_LOCKED)
                {
                    C._benignalloc_begin();
                    Rollback();
                    C._benignalloc_end();
                }
                else if (!ExclusiveMode)
                {
                    Debug.Assert(State == PAGER.READER);
                    pager_end_transaction(false, false);
                }
            }
            pager_unlock();
        }

        private uint pager_cksum(byte[] data)
        {
            var checksum = ChecksumInit;
            var i = PageSize - 200;
            while (i > 0)
            {
                checksum += data[i];
                i -= 200;
            }
            return checksum;
        }

#if HAS_CODEC
        private void pagerReportSize()
        {
            if (CodecSizeChange != null)
                CodecSizeChange(Codec, PageSize, ReserveBytes);
        }
#else
        private void pagerReportSize() { }
#endif

        private RC pager_playback_one_page(ref long offset, Bitvec done, bool isMainJournal, bool isSavepoint)
        {
            Debug.Assert(isMainJournal || done != null);   // pDone always used on sub-journals
            Debug.Assert(isSavepoint || done == null);     // pDone never used on non-savepoint

            var data = TmpSpace; // Temporary storage for the page
            Debug.Assert(data != null); // Temp storage must have already been allocated
            Debug.Assert(!UseWal() || (!isMainJournal && isSavepoint));

            // Either the state is greater than PAGER_WRITER_CACHEMOD (a transaction or savepoint rollback done at the request of the caller) or this is
            // a hot-journal rollback. If it is a hot-journal rollback, the pager is in state OPEN and holds an EXCLUSIVE lock. Hot-journal rollback
            // only reads from the main journal, not the sub-journal.
            Debug.Assert(State >= PAGER.WRITER_CACHEMOD || (State == PAGER.OPEN && Lock == VFile.LOCK.EXCLUSIVE));
            Debug.Assert(State >= PAGER.WRITER_CACHEMOD || isMainJournal);

            // Read the page number and page data from the journal or sub-journal file. Return an error code to the caller if an IO error occurs.
            var journalFile = (isMainJournal ? JournalFile : SubJournalFile); // The file descriptor for the journal file
            Pid id; // The page number of a page in journal
            var rc = journalFile.Read4(offset, out id);
            if (rc != RC.OK) return rc;
            rc = journalFile.Read(data, PageSize, offset + 4);
            if (rc != RC.OK) return rc;
            offset += PageSize + 4 + (isMainJournal ? 4 : 0); //TODO: CHECK THIS

            // Sanity checking on the page.  This is more important that I originally thought.  If a power failure occurs while the journal is being written,
            // it could cause invalid data to be written into the journal.  We need to detect this invalid data (with high probability) and ignore it.
            if (id == 0 || id == MJ_PID(this))
            {
                Debug.Assert(!isSavepoint);
                return RC.DONE;
            }
            if (id > DBSize || done.Get(id))
                return RC.OK;
            if (isMainJournal)
            {
                uint checksum; // Checksum used for sanity checking
                rc = journalFile.Read4(offset - 4, out checksum);
                if (rc != RC.OK) return rc;
                if (!isSavepoint && pager_cksum(data) != checksum)
                    return RC.DONE;
            }

            // If this page has already been played by before during the current rollback, then don't bother to play it back again.
            if (done != null && (rc = done.Set(id)) != RC.OK)
                return rc;

            // When playing back page 1, restore the nReserve setting
            if (id == 1 && ReserveBytes != data[20])
            {
                ReserveBytes = (data)[20];
                pagerReportSize();
            }

            // If the pager is in CACHEMOD state, then there must be a copy of this page in the pager cache. In this case just update the pager cache,
            // not the database file. The page is left marked dirty in this case.
            //
            // An exception to the above rule: If the database is in no-sync mode and a page is moved during an incremental vacuum then the page may
            // not be in the pager cache. Later: if a malloc() or IO error occurs during a Movepage() call, then the page may not be in the cache
            // either. So the condition described in the above paragraph is not assert()able.
            //
            // If in WRITER_DBMOD, WRITER_FINISHED or OPEN state, then we update the pager cache if it exists and the main file. The page is then marked 
            // not dirty. Since this code is only executed in PAGER_OPEN state for a hot-journal rollback, it is guaranteed that the page-cache is empty
            // if the pager is in OPEN state.
            //
            // Ticket #1171:  The statement journal might contain page content that is different from the page content at the start of the transaction.
            // This occurs when a page is changed prior to the start of a statement then changed again within the statement.  When rolling back such a
            // statement we must not write to the original database unless we know for certain that original page contents are synced into the main rollback
            // journal.  Otherwise, a power loss might leave modified data in the database file without an entry in the rollback journal that can
            // restore the database to its original form.  Two conditions must be met before writing to the database files. (1) the database must be
            // locked.  (2) we know that the original page content is fully synced in the main journal either because the page is not in cache or else
            // the page is marked as needSync==0.
            //
            // 2008-04-14:  When attempting to vacuum a corrupt database file, it is possible to fail a statement on a database that does not yet exist.
            // Do not attempt to write if database file has never been opened.
            var pg = (UseWal() ? null : pager_lookup(id)); // An existing page in the cache
            Debug.Assert(pg != null || !MemoryDB);
            Debug.Assert(State != PAGER.OPEN || pg == null);
            PAGERTRACE("PLAYBACK {0} page {1} hash({2,08:x}) {3}", PAGERID(this), id, pager_datahash(PageSize, data), (isMainJournal ? "main-journal" : "sub-journal"));
            bool isSynced; // True if journal page is synced
            if (isMainJournal)
                isSynced = NoSync || (offset <= JournalHeader);
            else
                isSynced = (pg == null || 0 == (pg.Flags & PgHdr.PGHDR.NEED_SYNC));
            if (File.Opened && (State >= PAGER.WRITER_DBMOD || State == PAGER.OPEN) && isSynced)
            {
                long ofst = (id - 1) * PageSize;
                Debug.Assert(!UseWal());
                rc = File.Write(data, PageSize, ofst);
                if (id > DBFileSize)
                    DBFileSize = id;
                if (Backup != null)
                {
                    if (CODEC1(this, data, id, 3)) rc = RC.NOMEM;
                    Backup.Update(id, data);
                    if (CODEC2(ref data, this, data, id, 7)) rc = RC.NOMEM;
                }
            }
            else if (isMainJournal && pg == null)
            {
                // If this is a rollback of a savepoint and data was not written to the database and the page is not in-memory, there is a potential
                // problem. When the page is next fetched by the b-tree layer, it will be read from the database file, which may or may not be 
                // current. 
                //
                // There are a couple of different ways this can happen. All are quite obscure. When running in synchronous mode, this can only happen 
                // if the page is on the free-list at the start of the transaction, then populated, then moved using sqlite3PagerMovepage().
                //
                // The solution is to add an in-memory page to the cache containing the data just read from the sub-journal. Mark the page as dirty 
                // and if the pager requires a journal-sync, then mark the page as requiring a journal-sync before it is written.
                Debug.Assert(isSavepoint);
                Debug.Assert(DoNotSpill == 0);
                DoNotSpill++;
                rc = Acquire(id, ref pg, true);
                Debug.Assert(DoNotSpill == 1);
                DoNotSpill--;
                if (rc != RC.OK) return rc;
                pg.Flags &= ~PgHdr.PGHDR.NEED_READ;
                PCache.MakeDirty(pg);
            }
            if (pg != null)
            {
                // No page should ever be explicitly rolled back that is in use, except for page 1 which is held in use in order to keep the lock on the
                // database active. However such a page may be rolled back as a result of an internal error resulting in an automatic call to
                // sqlite3PagerRollback().
                var pageData = pg.Data;
                Buffer.BlockCopy(data, 0, pageData, 0, PageSize);
                Reiniter(pg);
                if (isMainJournal && (!isSavepoint || offset <= JournalHeader))
                {
                    // If the contents of this page were just restored from the main journal file, then its content must be as they were when the 
                    // transaction was first opened. In this case we can mark the page as clean, since there will be no need to write it out to the
                    // database.
                    //
                    // There is one exception to this rule. If the page is being rolled back as part of a savepoint (or statement) rollback from an 
                    // unsynced portion of the main journal file, then it is not safe to mark the page as clean. This is because marking the page as
                    // clean will clear the PGHDR_NEED_SYNC flag. Since the page is already in the journal file (recorded in Pager.pInJournal) and
                    // the PGHDR_NEED_SYNC flag is cleared, if the page is written to again within this transaction, it will be marked as dirty but
                    // the PGHDR_NEED_SYNC flag will not be set. It could then potentially be written out into the database file before its journal file
                    // segment is synced. If a crash occurs during or following this, database corruption may ensue.
                    Debug.Assert(!UseWal());
                    PCache.MakeClean(pg);
                }
                pager_set_pagehash(pg);

                // If this was page 1, then restore the value of Pager.dbFileVers. Do this before any decoding.
                if (id == 1)
                    Buffer.BlockCopy(pageData, 24, DBFileVersion, 0, DBFileVersion.Length);

                // Decode the page just read from disk
                if (CODEC1(this, pageData, pg.ID, 3)) rc = RC.NOMEM;
                PCache.Release(pg);
            }
            return rc;
        }

        private RC pager_delmaster(string master)
        {
            throw new NotImplementedException();
            //var vfs = Vfs;

            //// Allocate space for both the pJournal and pMaster file descriptors. If successful, open the master journal file for reading.
            //var masterFile = new CoreVFile(); // Malloc'd master-journal file descriptor
            //var journalFile = new CoreVFile(); // Malloc'd child-journal file descriptor
            //VSystem.OPEN dummy;
            //var rc = vfs.Open(master, masterFile, VSystem.OPEN.READONLY | VSystem.OPEN.MASTER_JOURNAL, out dummy);
            //if (rc != RC.OK) goto delmaster_out;

            //Debugger.Break();

            //// Load the entire master journal file into space obtained from sqlite3_malloc() and pointed to by zMasterJournal.   Also obtain
            //// sufficient space (in zMasterPtr) to hold the names of master journal files extracted from regular rollback-journals.
            //long masterJournalSize; // Size of master journal file
            //rc = masterFile.get_FileSize(out masterJournalSize);
            //if (rc != RC.OK) goto delmaster_out;
            //var masterPtrSize = vfs.MaxPathname + 1; // Amount of space allocated to zMasterPtr[]
            //var masterJournal = new byte[masterJournalSize + 1];
            //string masterPtr;
            //rc = masterFile.Read(masterJournal, (int)masterJournalSize, 0);
            //if (rc != RC.OK) goto delmaster_out;
            //masterJournal[masterJournalSize] = 0;

            //var journalIdx = 0; // Pointer to one journal within MJ file
            //while (journalIdx < masterJournalSize)
            //{
            //    var journal = masterJournal;
            //    int exists;
            //    rc = vfs.Access(journal, VSystem.ACCESS.EXISTS, out exists);
            //    if (rc != RC.OK)
            //        goto delmaster_out;
            //    if (exists != 0)
            //    {
            //        // One of the journals pointed to by the master journal exists. Open it and check if it points at the master journal. If
            //        // so, return without deleting the master journal file.
            //        VSystem.OPEN dummy2;
            //        rc = vfs.Open(journal, journalFile, VSystem.OPEN.READONLY | VSystem.OPEN.MAIN_JOURNAL, out dummy2);
            //        if (rc != RC.OK)
            //            goto delmaster_out;

            //        rc = readMasterJournal(journalFile, out masterPtr, (uint)masterPtrSize);
            //        journalFile.Close();
            //        if (rc != RC.OK)
            //            goto delmaster_out;

            //        var c = string.Equals(master, masterPtr);
            //        if (c) // We have a match. Do not delete the master journal file.
            //            goto delmaster_out;
            //    }
            //    journalIdx += (sqlite3Strlen30(journal) + 1);
            //}

            //masterFile.Close();
            //rc = vfs.Delete(master, false);

            //delmaster_out:
            //    //masterJournal = null;
            //    if (masterFile != null)
            //    {
            //        masterFile.Close();
            //        Debug.Assert(!masterFile.Opened);
            //        masterFile = null;
            //    }
            //    return rc;
        }

        private RC pager_truncate(Pid pages)
        {
            Debug.Assert(State != PAGER.ERROR);
            Debug.Assert(State != PAGER.READER);

            var rc = RC.OK;
            if (File.Opened && (State >= PAGER.WRITER_DBMOD || State == PAGER.OPEN))
            {
                var sizePage = PageSize;
                Debug.Assert(Lock == VFile.LOCK.EXCLUSIVE);
                // TODO: Is it safe to use Pager.dbFileSize here?
                long currentSize;
                rc = File.get_FileSize(out currentSize);
                var newSize = sizePage * pages;
                if (rc == RC.OK && currentSize != newSize)
                {
                    if (currentSize > newSize)
                        rc = File.Truncate(newSize);
                    else
                    {
                        var tmp = TmpSpace;
                        Array.Clear(tmp, 0, sizePage);
                        rc = File.Write(tmp, sizePage, newSize - sizePage);
                    }
                    if (rc == RC.OK)
                        DBSize = pages;
                }
            }
            return rc;
        }

        #endregion

        #region Transaction2

        public uint get_SectorSize(VFile file)
        {
            var ret = file.get_SectorSize();
            if (ret < 32)
                ret = 512;
            else if (ret > MAX_SECTOR_SIZE)
            {
                Debug.Assert(MAX_SECTOR_SIZE >= 512);
                ret = MAX_SECTOR_SIZE;
            }
            return ret;
        }

        void setSectorSize()
        {
            Debug.Assert(File.Opened || TempFile);
            if (TempFile || (File.get_DeviceCharacteristics() & VFile.IOCAP.IOCAP_POWERSAFE_OVERWRITE) != 0)
                SectorSize = 512; // Sector size doesn't matter for temporary files. Also, the file may not have been opened yet, in which case the OsSectorSize() call will segfault.
            else
                SectorSize = File.get_SectorSize();
        }

        private RC pager_playback(bool isHot)
        {
            var res = 1;
            string master = null; // Name of master journal file if any

            // Figure out how many records are in the journal.  Abort early if the journal is empty.
            Debug.Assert(JournalFile.Opened);
            long sizeJournal; // Size of the journal file in bytes
            RC rc = JournalFile.get_FileSize(out sizeJournal);
            if (rc != RC.OK)
                goto end_playback;

            // Read the master journal name from the journal, if it is present. If a master journal file name is specified, but the file is not
            // present on disk, then the journal is not hot and does not need to be played back.
            //
            // TODO: Technically the following is an error because it assumes that buffer Pager.pTmpSpace is (mxPathname+1) bytes or larger. i.e. that
            // (pPager->pageSize >= pPager->pVfs->mxPathname+1). Using os_unix.c, mxPathname is 512, which is the same as the minimum allowable value
            // for pageSize.
            var vfs = Vfs;
            rc = readMasterJournal(JournalFile, out master, (uint)vfs.MaxPathname + 1);
            if (rc == RC.OK && master[0] != 0)
                rc = vfs.Access(master, VSystem.ACCESS.EXISTS, out res);
            master = null;
            if (rc != RC.OK || res == 0)
                goto end_playback;
            JournalOffset = 0;
            bool needPagerReset = isHot; // True to reset page prior to first page rollback

            // This loop terminates either when a readJournalHdr() or pager_playback_one_page() call returns SQLITE_DONE or an IO error occurs.
            while (true)
            {
                // Read the next journal header from the journal file.  If there are not enough bytes left in the journal file for a complete header, or
                // it is corrupted, then a process must have failed while writing it. This indicates nothing more needs to be rolled back.
                uint records = 0; // Number of Records in the journal
                Pid maxPage = 0; // Size of the original file in pages
                rc = readJournalHdr(isHot, sizeJournal, ref records, ref maxPage);
                if (rc != RC.OK)
                {
                    if (rc == RC.DONE)
                        rc = RC.OK;
                    goto end_playback;
                }

                // If nRec is 0xffffffff, then this journal was created by a process working in no-sync mode. This means that the rest of the journal
                // file consists of pages, there are no more journal headers. Compute the value of nRec based on this assumption.
                if (records == 0xffffffff)
                {
                    Debug.Assert(JournalOffset == JOURNAL_HDR_SZ(this));
                    records = (uint)((sizeJournal - JOURNAL_HDR_SZ(this)) / JOURNAL_PG_SZ(this));
                }

                // If nRec is 0 and this rollback is of a transaction created by this process and if this is the final header in the journal, then it means
                // that this part of the journal was being filled but has not yet been synced to disk.  Compute the number of pages based on the remaining
                // size of the file.
                //
                // The third term of the test was added to fix ticket #2565. When rolling back a hot journal, nRec==0 always means that the next
                // chunk of the journal contains zero pages to be rolled back.  But when doing a ROLLBACK and the nRec==0 chunk is the last chunk in
                // the journal, it means that the journal might contain additional pages that need to be rolled back and that the number of pages 
                // should be computed based on the journal file size.
                if (records == 0 && !isHot && JournalHeader + JOURNAL_HDR_SZ(this) == JournalOffset)
                    records = (uint)((sizeJournal - JournalOffset) / JOURNAL_PG_SZ(this));

                // If this is the first header read from the journal, truncate the database file back to its original size.
                if (JournalOffset == JOURNAL_HDR_SZ(this))
                {
                    rc = pager_truncate(maxPage);
                    if (rc != RC.OK)
                        goto end_playback;
                    DBSize = maxPage;
                }

                // Copy original pages out of the journal and back into the database file and/or page cache.
                for (var u = 0U; u < records; u++)
                {
                    if (needPagerReset)
                    {
                        pager_reset();
                        needPagerReset = false;
                    }
                    rc = pager_playback_one_page(ref JournalOffset, null, true, false);
                    if (rc != RC.OK)
                        if (rc == RC.DONE)
                        {
                            JournalOffset = sizeJournal;
                            break;
                        }
                        else if (rc == RC.IOERR_SHORT_READ)
                        {
                            // If the journal has been truncated, simply stop reading and processing the journal. This might happen if the journal was
                            // not completely written and synced prior to a crash.  In that case, the database should have never been written in the
                            // first place so it is OK to simply abandon the rollback.
                            rc = RC.OK;
                            goto end_playback;
                        }
                        else
                        {
                            // If we are unable to rollback, quit and return the error code.  This will cause the pager to enter the error state
                            // so that no further harm will be done.  Perhaps the next process to come along will be able to rollback the database.
                            goto end_playback;
                        }
                }
            }

        end_playback:
            // Following a rollback, the database file should be back in its original state prior to the start of the transaction, so invoke the
            // SQLITE_FCNTL_DB_UNCHANGED file-control method to disable the assertion that the transaction counter was modified.
#if DEBUG
            long dummy = 0;
            File.FileControl(VFile.FCNTL.DB_UNCHANGED, ref dummy);
            // TODO: verfiy this, because its different then the c# version
#endif

            // If this playback is happening automatically as a result of an IO or malloc error that occurred after the change-counter was updated but 
            // before the transaction was committed, then the change-counter modification may just have been reverted. If this happens in exclusive 
            // mode, then subsequent transactions performed by the connection will not update the change-counter at all. This may lead to cache inconsistency
            // problems for other processes at some point in the future. So, just in case this has happened, clear the changeCountDone flag now.
            ChangeCountDone = TempFile;

            if (rc == RC.OK)
                rc = readMasterJournal(JournalFile, out master, (uint)Vfs.MaxPathname + 1);
            if (rc == RC.OK && (State >= PAGER.WRITER_DBMOD || State == PAGER.OPEN))
                rc = Sync();
            if (rc == RC.OK)
                rc = pager_end_transaction(master[0] != '\0', false);
            if (rc == RC.OK && master[0] != '\0' && res != 0)
                // If there was a master journal and this routine will return success, see if it is possible to delete the master journal.
                rc = pager_delmaster(master);

            // The Pager.sectorSize variable may have been updated while rolling back a journal created by a process with a different sector size
            // value. Reset it to the correct value for this process.
            setSectorSize();
            return rc;
        }

        private static RC readDbPage(PgHdr page)
        {
            var pager = page.Pager; // Pager object associated with page pPg

            Debug.Assert(pager.State >= PAGER.READER && !pager.MemoryDB);
            Debug.Assert(pager.File.Opened);

            if (C._NEVER(!pager.File.Opened))
            {
                Debug.Assert(pager.TempFile);
                Array.Clear(page.Data, 0, pager.PageSize);
                return RC.OK;
            }

            var rc = RC.OK;
            var id = page.ID; // Page number to read
            var isInWal = 0; // True if page is in log file
            var pageSize = pager.PageSize; // Number of bytes to read
            if (pager.UseWal()) // Try to pull the page from the write-ahead log.
                rc = pager.Wal.Read(id, ref isInWal, pageSize, page.Data);
            if (rc == RC.OK && isInWal == 0)
            {
                var offset = (id - 1) * (long)pager.PageSize;
                rc = pager.File.Read(page.Data, pageSize, offset);
                if (rc == RC.IOERR_SHORT_READ)
                    rc = RC.OK;
            }

            if (id == 1)
            {
                // If the read is unsuccessful, set the dbFileVers[] to something that will never be a valid file version.  dbFileVers[] is a copy
                // of bytes 24..39 of the database.  Bytes 28..31 should always be zero or the size of the database in page. Bytes 32..35 and 35..39
                // should be page numbers which are never 0xffffffff.  So filling pPager->dbFileVers[] with all 0xff bytes should suffice.
                //
                // For an encrypted database, the situation is more complex:  bytes 24..39 of the database are white noise.  But the probability of
                // white noising equaling 16 bytes of 0xff is vanishingly small so we should still be ok.
                if (rc != 0)
                    for (int i = 0; i < pager.DBFileVersion.Length; pager.DBFileVersion[i++] = 0xff) ; //_memset(pager->DBFileVersion, 0xff, sizeof(pager->DBFileVersion));
                else
                    Buffer.BlockCopy(page.Data, 24, pager.DBFileVersion, 0, pager.DBFileVersion.Length);
            }
            if (CODEC1(pager, page.Data, id, 3)) rc = RC.NOMEM;

            PAGER_INCR(ref _readdb_count);
            PAGER_INCR(ref pager.Reads);
            SysEx.IOTRACE("PGIN {0:x} {1}", pager.GetHashCode(), id);
            PAGERTRACE("FETCH {0} page {1}% hash({2,08:x})", PAGERID(pager), id, pager_pagehash(page));

            return rc;
        }

        private static void pager_write_changecounter(PgHdr pg)
        {
            // Increment the value just read and write it back to byte 24.
            uint change_counter = ConvertEx.Get4(pg.Pager.DBFileVersion, 0) + 1;
            ConvertEx.Put4(pg.Data, 24, change_counter);

            // Also store the SQLite version number in bytes 96..99 and in bytes 92..95 store the change counter for which the version number is valid.
            ConvertEx.Put4(pg.Data, 92, change_counter);
            ConvertEx.Put4(pg.Data, 96, SysEx.CORE_VERSION_NUMBER);
        }

#if !OMIT_WAL
        static RC pagerUndoCallback(object ctx, Pid id)
        {
            var rc = RC.OK;
            var pager = (Pager)ctx;
            var pg = pager.Lookup(id);
            if (pg != null)
            {
                if (PCache.get_PageRefs(pg) == 1)
                    PCache.Drop(pg);
                else
                {
                    rc = readDbPage(pg);
                    if (rc == RC.OK)
                        pager.Reiniter(pg);
                    Pager.Unref(pg);
                }
            }
            // Normally, if a transaction is rolled back, any backup processes are updated as data is copied out of the rollback journal and into the
            // database. This is not generally possible with a WAL database, as rollback involves simply truncating the log file. Therefore, if one
            // or more frames have already been written to the log (and therefore also copied into the backup databases) as part of this transaction,
            // the backups must be restarted.
            if (pager.Backup != null)
                pager.Backup.Restart();
            return rc;
        }

        static RC pagerRollbackWal(Pager pager)
        {
            // For all pages in the cache that are currently dirty or have already been written (but not committed) to the log file, do one of the following:
            //
            //   + Discard the cached page (if refcount==0), or
            //   + Reload page content from the database (if refcount>0).
            pager.DBSize = pager.DBOrigSize;
            var rc = pager.Wal.Undo(pagerUndoCallback, pager);
            var list = pager.PCache.DirtyList(); // List of dirty pages to revert 
            while (list != null && rc == RC.OK)
            {
                var next = list.Dirty;
                rc = pagerUndoCallback(pager, list.ID);
                list = next;
            }

            return rc;
        }

        static RC pagerWalFrames(Pager pager, PgHdr list, Pid truncate, bool isCommit)
        {
            Debug.Assert(pager.Wal != null);
            Debug.Assert(list != null);
            PgHdr p; // For looping over pages
#if DEBUG
            // Verify that the page list is in accending order
            for (p = list; p != null && p.Dirty != null; p = p.Dirty)
                Debug.Assert(p.ID < p.Dirty.ID);
#endif
            int listPages; // Number of pages in pList
            Debug.Assert(list.Dirty == null || isCommit);
            if (isCommit)
            {
                // If a WAL transaction is being committed, there is no point in writing any pages with page numbers greater than nTruncate into the WAL file.
                // They will never be read by any client. So remove them from the pDirty list here.
                PgHdr next = list;
                listPages = 0;
                for (p = list; (next = p) != null; p = p.Dirty)
                    if (p.ID <= truncate)
                    {
                        next = p.Dirty;
                        listPages++;
                    }
                Debug.Assert(list != null);
            }
            else
                listPages = 1;
            pager.Stats[(int)STAT.WRITE] += listPages;

            if (list.ID == 1) pager_write_changecounter(list);
            var rc = pager.Wal.Frames(pager.PageSize, list, truncate, isCommit, pager.WalSyncFlags);
            if (rc == RC.OK && pager.Backup != null)
                for (p = list; p != null; p = p.Dirty)
                    pager.Backup.Update(p.ID, p.Data);

#if CHECK_PAGES
            list = pager.PCache.DirtyList());
            for (p = list; p != null; p = p.Dirty)
                pager_set_pagehash(p);
#endif

            return rc;
        }

        static RC pagerBeginReadTransaction(Pager pager)
        {
            Debug.Assert(UseWal(pager));
            Debug.Assert(pager.State == PAGER.OPEN || pager.State == PAGER.READER);

            // sqlite3WalEndReadTransaction() was not called for the previous transaction in locking_mode=EXCLUSIVE.  So call it now.  If we
            // are in locking_mode=NORMAL and EndRead() was previously called, the duplicate call is harmless.
            pager.Wal.EndReadTransaction();

            int changed = 0; // True if cache must be reset
            var rc = pager.Wal.BeginReadTransaction(out changed);
            if (rc != RC.OK || changed)
                pager_reset(pager);

            return rc;
        }
#endif

        private RC pagerPagecount(out Pid pagesRef)
        {
            // Query the WAL sub-system for the database size. The WalDbsize() function returns zero if the WAL is not open (i.e. Pager.pWal==0), or
            // if the database size is not available. The database size is not available from the WAL sub-system if the log file is empty or
            // contains no valid committed transactions.
            Debug.Assert(State == PAGER.OPEN);
            Debug.Assert(Lock >= VFile.LOCK.SHARED);
            var pages = Wal.DBSize();

            // If the database size was not available from the WAL sub-system, determine it based on the size of the database file. If the size
            // of the database file is not an integer multiple of the page-size, round down to the nearest page. Except, any file larger than 0
            // bytes in size is considered to contain at least one page.
            if (pages == 0)
            {
                Debug.Assert(File.Opened || TempFile);
                var n = 0L; // Size of db file in bytes
                if (File.Opened)
                {
                    var rc = File.get_FileSize(out n);
                    if (rc != RC.OK)
                    {
                        pagesRef = 0;
                        return rc;
                    }
                }
                pages = (Pid)((n + PageSize - 1) / PageSize);
            }

            // If the current number of pages in the file is greater than the configured maximum pager number, increase the allowed limit so
            // that the file can be read.
            if (pages > MaxPid)
                MaxPid = (Pid)pages;

            pagesRef = pages;
            return RC.OK;
        }

#if !OMIT_WAL
        static RC pagerOpenWalIfPresent(Pager pager)
        {
            Debug.Assert(pager.State == PAGER.OPEN);
            Debug.Assert(pager.Lock >= VFile.LOCK.SHARED);
            var rc = RC.OK;

            if (!pager.TempFile)
            {
                int isWal;                    /* True if WAL file exists */
                Pid pages; // Size of the database file
                rc = pagerPagecount(pager, &pages);
                if (rc != RC.OK) return rc;
                int isWal; // True if WAL file exists
                if (pages == 0)
                {
                    rc = pager.Vfs.Delete(pager.WalName, false);
                    if (rc == RC.IOERR_DELETE_NOENT) rc = RC.OK;
                    isWal = 0;
                }
                else
                    rc = pager.Vfs.Access(pager.WalName, VSystem.ACCESS.EXISTS, out isWal);
                if (rc == RC.OK)
                {
                    if (isWal)
                        rc = pager.pagerOpenWal(0);
                    else if (pager.JournalMode == IPager.JOURNALMODE.WAL)
                        pager.JournalMode = IPager.JOURNALMODE.DELETE;
                }
            }
            return rc;
        }
#endif

        private RC pagerPlaybackSavepoint(PagerSavepoint savepoint)
        {
            Debug.Assert(State != PAGER.ERROR);
            Debug.Assert(State >= PAGER.WRITER_LOCKED);

            // Allocate a bitvec to use to store the set of pages rolled back
            Bitvec done = null; // Bitvec to ensure pages played back only once
            if (savepoint != null)
                done = new Bitvec(savepoint.Orig);

            // Set the database size back to the value it was before the savepoint being reverted was opened.
            DBSize = (savepoint != null ? savepoint.Orig : DBOrigSize);
            ChangeCountDone = TempFile;

            if (savepoint != null && UseWal())
                return pagerRollbackWal();

            // Use pPager->journalOff as the effective size of the main rollback journal.  The actual file might be larger than this in
            // PAGER_JOURNALMODE_TRUNCATE or PAGER_JOURNALMODE_PERSIST.  But anything past pPager->journalOff is off-limits to us.
            var sizeJournal = JournalOffset; // Effective size of the main journal
            Debug.Assert(!UseWal() || sizeJournal == 0);

            // Begin by rolling back records from the main journal starting at PagerSavepoint.iOffset and continuing to the next journal header.
            // There might be records in the main journal that have a page number greater than the current database size (pPager->dbSize) but those
            // will be skipped automatically.  Pages are added to pDone as they are played back.
            var rc = RC.OK;
            if (savepoint != null && !UseWal())
            {
                long hdrOffset = (savepoint.HdrOffset != 0 ? savepoint.HdrOffset : sizeJournal); // End of first segment of main-journal records
                JournalOffset = savepoint.Offset;
                while (rc == RC.OK && JournalOffset < hdrOffset)
                    rc = pager_playback_one_page(ref JournalOffset, done, true, true);
                Debug.Assert(rc != RC.DONE);
            }
            else
                JournalOffset = 0;

            // Continue rolling back records out of the main journal starting at the first journal header seen and continuing until the effective end
            // of the main journal file.  Continue to skip out-of-range pages and continue adding pages rolled back to pDone.
            while (rc == RC.OK && JournalOffset < sizeJournal)
            {
                uint records = 0; // Number of Journal Records
                uint dummy = 0;
                rc = readJournalHdr(false, sizeJournal, ref records, ref dummy);
                Debug.Assert(rc != RC.DONE);

                // The "pPager.journalHdr+JOURNAL_HDR_SZ(pPager)==pPager.journalOff" test is related to ticket #2565.  See the discussion in the
                // pager_playback() function for additional information.
                if (records == 0 && JournalHeader + JOURNAL_HDR_SZ(this) >= JournalOffset)
                    records = (uint)((sizeJournal - JournalOffset) / JOURNAL_PG_SZ(this));
                for (var ii = 0U; rc == RC.OK && ii < records && JournalOffset < sizeJournal; ii++)
                    rc = pager_playback_one_page(ref JournalOffset, done, true, true);
                Debug.Assert(rc != RC.DONE);
            }
            Debug.Assert(rc != RC.OK || JournalOffset >= sizeJournal);

            // Finally,  rollback pages from the sub-journal.  Page that were previously rolled back out of the main journal (and are hence in pDone)
            // will be skipped.  Out-of-range pages are also skipped.
            if (savepoint != null)
            {
                long offset = savepoint.SubRecords * (4 + PageSize);

                if (UseWal())
                    rc = Wal.SavepointUndo(savepoint.WalData);
                for (var ii = savepoint.SubRecords; rc == RC.OK && ii < SubRecords; ii++)
                {
                    Debug.Assert(offset == ii * (4 + PageSize));
                    rc = pager_playback_one_page(ref offset, done, false, true);
                }
                Debug.Assert(rc != RC.DONE);
            }

            Bitvec.Destroy(ref done);
            if (rc == RC.OK)
                JournalOffset = (int)sizeJournal;

            return rc;
        }

        #endregion

        #region Name3

        // was:sqlite3PagerSetCachesize
        public void SetCacheSize(int maxPage)
        {
            PCache.set_CacheSize(maxPage);
        }

        public void Shrink()
        {
            PCache.Shrink();
        }

#if !OMIT_PAGER_PRAGMAS
        public void SetSafetyLevel(int level, bool fullFsync, bool checkpointFullFsync)
        {
            Debug.Assert(level >= 1 && level <= 3);
            NoSync = (level == 1 || TempFile);
            FullSync = (level == 3 && !TempFile);
            if (NoSync)
            {
                SyncFlags = 0;
                CheckpointSyncFlags = 0;
            }
            else if (fullFsync)
            {
                SyncFlags = VFile.SYNC.FULL;
                CheckpointSyncFlags = VFile.SYNC.FULL;
            }
            else if (checkpointFullFsync)
            {
                SyncFlags = VFile.SYNC.NORMAL;
                CheckpointSyncFlags = VFile.SYNC.FULL;
            }
            else
            {
                SyncFlags = VFile.SYNC.NORMAL;
                CheckpointSyncFlags = VFile.SYNC.NORMAL;
            }
            WalSyncFlags = SyncFlags;
            if (FullSync)
                WalSyncFlags |= VFile.SYNC.WAL_TRANSACTIONS;
        }
#endif

#if TEST
        // The following global variable is incremented whenever the library attempts to open a temporary file.  This information is used for testing and analysis only.  
        int _opentemp_count = 0;
#endif

        private RC pagerOpentemp(ref VFile file, VSystem.OPEN vfsFlags)
        {
#if TEST
            _opentemp_count++; // Used for testing and analysis only
#endif
            vfsFlags |= VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE | VSystem.OPEN.EXCLUSIVE | VSystem.OPEN.DELETEONCLOSE;
            VSystem.OPEN dummy = 0;
            var rc = Vfs.Open(null, file, vfsFlags, out dummy);
            Debug.Assert(rc != RC.OK || file.Opened);
            return rc;
        }

        // was:sqlite3PagerSetBusyhandler
        public void SetBusyHandler(Func<object, int> busyHandler, object busyHandlerArg)
        {
            BusyHandler = busyHandler;
            BusyHandlerArg = busyHandlerArg;

            if (File.Opened)
            {
                //object ap = BusyHandler;
                //Debug.Assert(ap[0] == busyHandler);
                //Debug.Assert(ap[1] == busyHandlerArg);
                // TODO: This
                //File.FileControl(VFile.FCNTL.BUSYHANDLER, (void *)ap);
            }
        }

        public RC SetPageSize(ref uint pageSizeRef, int reserveBytes)
        {
            // It is not possible to do a full assert_pager_state() here, as this function may be called from within PagerOpen(), before the state
            // of the Pager object is internally consistent.
            //
            // At one point this function returned an error if the pager was in PAGER_ERROR state. But since PAGER_ERROR state guarantees that
            // there is at least one outstanding page reference, this function is a no-op for that case anyhow.
            var pageSize = pageSizeRef;
            Debug.Assert(pageSize == 0 || (pageSize >= 512 && pageSize <= MAX_PAGE_SIZE));
            var rc = RC.OK;
            if ((!MemoryDB || DBSize == 0) &&
                PCache.get_Refs() == 0 &&
                pageSize != 0 && pageSize != (uint)PageSize)
            {
                long bytes = 0;
                if (State > PAGER.OPEN && File.Opened)
                    rc = File.get_FileSize(out bytes);
                byte[] tempSpace = null; // New temp space
                if (rc == RC.OK)
                    tempSpace = PCache.PageAlloc2((int)pageSize);
                if (rc == RC.OK)
                {
                    pager_reset();
                    DBSize = (Pid)((bytes + pageSize - 1) / pageSize);
                    PageSize = (int)pageSize;
                    PCache.PageFree2(ref TmpSpace);
                    TmpSpace = tempSpace;
                    PCache.SetPageSize((int)pageSize);
                }
            }
            pageSizeRef = (uint)PageSize;
            if (rc == RC.OK)
            {
                if (reserveBytes < 0) reserveBytes = ReserveBytes;
                Debug.Assert(reserveBytes >= 0 && reserveBytes < 1000);
                ReserveBytes = (short)reserveBytes;
                pagerReportSize();
            }
            return rc;
        }

        public byte[] get_TempSpace()
        {
            return TmpSpace;
        }

        public Pid MaxPages(int maxPage)
        {
            if (maxPage > 0)
                MaxPid = (Pid)maxPage;
            Debug.Assert(State != PAGER.OPEN);  // Called only by OP_MaxPgcnt
            Debug.Assert(MaxPid >= DBSize);     // OP_MaxPgcnt enforces this
            return MaxPid;
        }

#if TEST
        static int _io_error_pending;
        static int _io_error_hit;
        static int saved_cnt;

        internal static void disable_simulated_io_errors()
        {
            saved_cnt = _io_error_pending;
            _io_error_pending = -1;
        }

        internal static void enable_simulated_io_errors()
        {
            _io_error_pending = saved_cnt;
        }
#else
        internal static void disable_simulated_io_errors() { }
        internal static void enable_simulated_io_errors() { }
#endif

        public RC ReadFileHeader(int n, byte[] dest)
        {
            Array.Clear(dest, 0, n);
            Debug.Assert(File.Opened || TempFile);

            // This routine is only called by btree immediately after creating the Pager object.  There has not been an opportunity to transition to WAL mode yet.
            Debug.Assert(!UseWal());

            var rc = RC.OK;
            if (File.Opened)
            {
                SysEx.IOTRACE("DBHDR {0} 0 {1}", GetHashCode(), n);
                rc = File.Read(dest, n, 0);
                if (rc == RC.IOERR_SHORT_READ)
                    rc = RC.OK;
            }
            return rc;
        }

        // was:sqlite3PagerPagecount
        public void Pages(out Pid pagesOut)
        {
            Debug.Assert(State >= PAGER.READER);
            Debug.Assert(State != PAGER.WRITER_FINISHED);
            pagesOut = DBSize;
        }

        private RC pager_wait_on_lock(VFile.LOCK locktype)
        {
            // Check that this is either a no-op (because the requested lock is already held, or one of the transistions that the busy-handler
            // may be invoked during, according to the comment above sqlite3PagerSetBusyhandler().
            Debug.Assert((Lock >= locktype) ||
                (Lock == VFile.LOCK.NO && locktype == VFile.LOCK.SHARED) ||
                (Lock == VFile.LOCK.RESERVED && locktype == VFile.LOCK.EXCLUSIVE));

            RC rc;
            do
                rc = pagerLockDb(locktype);
            while (rc == RC.BUSY && BusyHandler(BusyHandlerArg) != 0);
            return rc;
        }

#if DEBUG
        static void assertTruncateConstraintCb(PgHdr page)
        {
            Debug.Assert((page.Flags & PgHdr.PGHDR.DIRTY) != 0);
            Debug.Assert(!subjRequiresPage(page) || page.ID <= page.Pager.DBSize);
        }

        void assertTruncateConstraint()
        {
            PCache.IterateDirty(assertTruncateConstraintCb);
        }
#else
        void assertTruncateConstraint() { }
#endif

        // was:sqlite3PagerTruncateImage
        public void TruncateImage(Pid pages)
        {
            Debug.Assert(DBSize >= pages);
            Debug.Assert(State >= PAGER.WRITER_CACHEMOD);
            DBSize = pages;

            // At one point the code here called assertTruncateConstraint() to ensure that all pages being truncated away by this operation are,
            // if one or more savepoints are open, present in the savepoint journal so that they can be restored if the savepoint is rolled
            // back. This is no longer necessary as this function is now only called right before committing a transaction. So although the 
            // Pager object may still have open savepoints (Pager.nSavepoint!=0), they cannot be rolled back. So the assertTruncateConstraint() call
            // is no longer correct.
        }

        private RC pagerSyncHotJournal()
        {
            var rc = RC.OK;
            if (!NoSync)
                rc = JournalFile.Sync(VFile.SYNC.NORMAL);
            if (rc == RC.OK)
                rc = JournalFile.get_FileSize(out JournalHeader);
            return rc;
        }

        public RC Close()
        {
            Debug.Assert(assert_pager_state(this));
            disable_simulated_io_errors();
            C._benignalloc_begin();
            ErrorCode = RC.OK;
            ExclusiveMode = false;
            var tmp = TmpSpace;
#if !OMIT_WAL
            Wal.Close(CheckpointSyncFlags, PageSize, tmp);
            Wal = null;
#endif
            pager_reset();
            if (MemoryDB)
                pager_unlock();
            else
            {
                // If it is open, sync the journal file before calling UnlockAndRollback. If this is not done, then an unsynced portion of the open journal 
                // file may be played back into the database. If a power failure occurs while this is happening, the database could become corrupt.
                //
                // If an error occurs while trying to sync the journal, shift the pager into the ERROR state. This causes UnlockAndRollback to unlock the
                // database and close the journal file without attempting to roll it back or finalize it. The next database user will have to do hot-journal
                // rollback before accessing the database file.
                if (JournalFile.Opened)
                    pager_error(pagerSyncHotJournal());
                pagerUnlockAndRollback();
            }
            C._benignalloc_end();
            PAGERTRACE("CLOSE {0}", PAGERID(this));
            SysEx.IOTRACE("CLOSE {0:x}", GetHashCode());
            JournalFile.Close();
            File.Close();
            PCache.PageFree2(ref tmp);
            PCache.Close();

#if HAS_CODEC
            if (CodecFree != null) CodecFree(Codec);
#endif

            Debug.Assert(Savepoints == null && !InJournal);
            Debug.Assert(!JournalFile.Opened && !SubJournalFile.Opened);

            //this = null;
            return RC.OK;
        }

#if !DEBUG || TEST
        // was:sqlite3PagerPagenumber
        public static Pid get_PageID(IPage pg)
        {
            return pg.ID;
        }
#endif

        // was:sqlite3PagerRef
        public static void AddPageRef(IPage pg)
        {
            PCache.Ref(pg);
        }

        #endregion

        #region Main

        private RC syncJournal(bool newHeader)
        {

            Debug.Assert(State == PAGER.WRITER_CACHEMOD || State == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(this));
            Debug.Assert(!UseWal());

            var rc = ExclusiveLock();
            if (rc != RC.OK) return rc;

            if (!NoSync)
            {
                Debug.Assert(!TempFile);
                if (JournalFile.Opened && JournalMode != IPager.JOURNALMODE.JMEMORY)
                {
                    var dc = File.get_DeviceCharacteristics();
                    Debug.Assert(JournalFile.Opened);

                    if ((dc & VFile.IOCAP.SAFE_APPEND) == 0)
                    {
                        // This block deals with an obscure problem. If the last connection that wrote to this database was operating in persistent-journal
                        // mode, then the journal file may at this point actually be larger than Pager.journalOff bytes. If the next thing in the journal
                        // file happens to be a journal-header (written as part of the previous connection's transaction), and a crash or power-failure 
                        // occurs after nRec is updated but before this connection writes anything else to the journal file (or commits/rolls back its 
                        // transaction), then SQLite may become confused when doing the hot-journal rollback following recovery. It may roll back all
                        // of this connections data, then proceed to rolling back the old, out-of-date data that follows it. Database corruption.
                        //
                        // To work around this, if the journal file does appear to contain a valid header following Pager.journalOff, then write a 0x00
                        // byte to the start of it to prevent it from being recognized.
                        //
                        // Variable iNextHdrOffset is set to the offset at which this problematic header will occur, if it exists. aMagic is used 
                        // as a temporary buffer to inspect the first couple of bytes of the potential journal header.
                        var header = new byte[_journalMagic.Length + 4];
                        _journalMagic.CopyTo(header, 0);
                        ConvertEx.Put4(header, _journalMagic.Length, Records);

                        var magic = new byte[8];
                        var nextHdrOffset = journalHdrOffset();
                        rc = JournalFile.Read(magic, 8, nextHdrOffset);

                        if (rc == RC.OK && Enumerable.SequenceEqual(magic, _journalMagic))
                        {
                            var zerobyte = new byte[1];
                            rc = JournalFile.Write(zerobyte, 1, nextHdrOffset);
                        }
                        if (rc != RC.OK && rc != RC.IOERR_SHORT_READ)
                            return rc;

                        // Write the nRec value into the journal file header. If in full-synchronous mode, sync the journal first. This ensures that
                        // all data has really hit the disk before nRec is updated to mark it as a candidate for rollback.
                        //
                        // This is not required if the persistent media supports the SAFE_APPEND property. Because in this case it is not possible 
                        // for garbage data to be appended to the file, the nRec field is populated with 0xFFFFFFFF when the journal header is written
                        // and never needs to be updated.
                        if (FullSync && (dc & VFile.IOCAP.SEQUENTIAL) == 0)
                        {
                            PAGERTRACE("SYNC journal of {0}", PAGERID(this));
                            SysEx.IOTRACE("JSYNC {0:x}", GetHashCode());
                            rc = JournalFile.Sync(SyncFlags);
                            if (rc != RC.OK) return rc;
                        }
                        SysEx.IOTRACE("JHDR {0:x} {1,11}", GetHashCode(), JournalHeader);
                        rc = JournalFile.Write(header, header.Length, JournalHeader);
                        if (rc != RC.OK) return rc;
                    }
                    if ((dc & VFile.IOCAP.SEQUENTIAL) == 0)
                    {
                        PAGERTRACE("SYNC journal of {0}", PAGERID(this));
                        SysEx.IOTRACE("JSYNC {0:x}", GetHashCode());
                        rc = JournalFile.Sync(SyncFlags | (SyncFlags == VFile.SYNC.FULL ? VFile.SYNC.DATAONLY : 0));
                        if (rc != RC.OK) return rc;
                    }

                    JournalHeader = JournalOffset;
                    if (newHeader && (dc & VFile.IOCAP.SAFE_APPEND) == 0)
                    {
                        Records = 0;
                        rc = writeJournalHdr();
                        if (rc != RC.OK) return rc;
                    }
                }
                else
                    JournalHeader = JournalOffset;
            }

            // Unless the pager is in noSync mode, the journal file was just successfully synced. Either way, clear the PGHDR_NEED_SYNC flag on all pages.
            PCache.ClearSyncFlags();
            State = PAGER.WRITER_DBMOD;
            Debug.Assert(assert_pager_state(this));
            return RC.OK;
        }

        private RC pager_write_pagelist(PgHdr list)
        {
            // This function is only called for rollback pagers in WRITER_DBMOD state.
            Debug.Assert(!UseWal());
            Debug.Assert(State == PAGER.WRITER_DBMOD);
            Debug.Assert(Lock == VFile.LOCK.EXCLUSIVE);

            // If the file is a temp-file has not yet been opened, open it now. It is not possible for rc to be other than SQLITE_OK if this branch
            // is taken, as pager_wait_on_lock() is a no-op for temp-files.
            var rc = RC.OK;
            if (!File.Opened)
            {
                Debug.Assert(TempFile && rc == RC.OK);
                rc = pagerOpentemp(ref File, VfsFlags);
            }

            // Before the first write, give the VFS a hint of what the final file size will be.
            Debug.Assert(rc != RC.OK || File.Opened);
            if (rc == RC.OK && DBSize > DBHintSize)
            {
                long sizeFile = PageSize * (long)DBSize;
                File.FileControl(VFile.FCNTL.SIZE_HINT, ref sizeFile);
                DBHintSize = DBSize;
            }

            while (rc == RC.OK && list != null)
            {
                var id = list.ID;

                // If there are dirty pages in the page cache with page numbers greater than Pager.dbSize, this means sqlite3PagerTruncateImage() was called to
                // make the file smaller (presumably by auto-vacuum code). Do not write any such pages to the file.
                //
                // Also, do not write out any page that has the PGHDR_DONT_WRITE flag set (set by sqlite3PagerDontWrite()).
                if (id <= DBSize && (list.Flags & PgHdr.PGHDR.DONT_WRITE) == 0)
                {
                    Debug.Assert((list.Flags & PgHdr.PGHDR.NEED_SYNC) == 0);
                    if (id == 1) pager_write_changecounter(list);

                    // Encode the database
                    byte[] data = null; // Data to write
                    if (CODEC2(ref data, this, list.Data, id, 6)) return RC.NOMEM;

                    // Write out the page data.
                    long offset = (id - 1) * (long)PageSize; // Offset to write
                    rc = File.Write(data, PageSize, offset);

                    // If page 1 was just written, update Pager.dbFileVers to match the value now stored in the database file. If writing this 
                    // page caused the database file to grow, update dbFileSize. 
                    if (id == 1)
                        Buffer.BlockCopy(data, 24, DBFileVersion, 0, DBFileVersion.Length);
                    if (id > DBFileSize)
                        DBFileSize = id;
                    Stats[(int)STAT.WRITE]++;

                    // Update any backup objects copying the contents of this pager.
                    if (Backup != null) Backup.Update(id, list.Data);

                    PAGERTRACE("STORE {0} page {1} hash({2,08:x})", PAGERID(this), id, pager_pagehash(list));
                    SysEx.IOTRACE("PGOUT {0:x} {1}", GetHashCode(), id);
                    PAGER_INCR(ref _writedb_count);
                }
                else
                    PAGERTRACE("NOSTORE {0} page {1}", PAGERID(this), id);
                pager_set_pagehash(list);
                list = list.Dirty;
            }

            return rc;
        }

        private RC openSubJournal()
        {
            var rc = RC.OK;
            if (!SubJournalFile.Opened)
            {
                if (JournalMode == IPager.JOURNALMODE.JMEMORY || SubjInMemory)
                    VFile.MemoryVFileOpen(ref SubJournalFile);
                else
                    rc = pagerOpentemp(ref SubJournalFile, VSystem.OPEN.SUBJOURNAL);
            }
            return rc;
        }

        private static RC subjournalPage(PgHdr pg)
        {
            var rc = RC.OK;
            var pager = pg.Pager;
            if (pager.JournalMode != IPager.JOURNALMODE.OFF)
            {
                // Open the sub-journal, if it has not already been opened
                Debug.Assert(pager.UseJournal);
                Debug.Assert(pager.JournalFile.Opened || pager.UseWal());
                Debug.Assert(pager.SubJournalFile.Opened || pager.SubRecords == 0);
                Debug.Assert(pager.UseWal() || pageInJournal(pg) || pg.ID > pager.DBOrigSize);
                rc = pager.openSubJournal();
                // If the sub-journal was opened successfully (or was already open), write the journal record into the file. 
                if (rc == RC.OK)
                {
                    var data = pg.Data;
                    long offset = pager.SubRecords * (4 + pager.PageSize);
                    byte[] data2 = null;
                    if (CODEC2(ref data2, pager, data, pg.ID, 7)) return RC.NOMEM;
                    PAGERTRACE("STMT-JOURNAL {0} page {1}", PAGERID(pager), pg.ID);
                    rc = pager.SubJournalFile.Write4(offset, pg.ID);
                    if (rc == RC.OK)
                        rc = pager.SubJournalFile.Write(data2, pager.PageSize, offset + 4);
                }
            }
            if (rc == RC.OK)
            {
                pager.SubRecords++;
                Debug.Assert(pager.Savepoints.Length > 0);
                rc = pager.addToSavepointBitvecs(pg.ID);
            }
            return rc;
        }

        private static RC pagerStress(object p, IPage pg)
        {
            var pager = (Pager)p;
            Debug.Assert(pg.Pager == pager);
            Debug.Assert((pg.Flags & PgHdr.PGHDR.DIRTY) != 0);

            // The doNotSyncSpill flag is set during times when doing a sync of journal (and adding a new header) is not allowed. This occurs
            // during calls to sqlite3PagerWrite() while trying to journal multiple pages belonging to the same sector.
            //
            // The doNotSpill flag inhibits all cache spilling regardless of whether or not a sync is required.  This is set during a rollback.
            //
            // Spilling is also prohibited when in an error state since that could lead to database corruption.   In the current implementaton it 
            // is impossible for sqlite3PcacheFetch() to be called with createFlag==1 while in the error state, hence it is impossible for this routine to
            // be called in the error state.  Nevertheless, we include a NEVER() test for the error state as a safeguard against future changes.
            if (C._NEVER(pager.ErrorCode != 0)) return RC.OK;
            if (pager.DoNotSpill != 0) return RC.OK;
            if (pager.DoNotSyncSpill != 0 && (pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0) return RC.OK;

            pg.Dirty = null;
            var rc = RC.OK;
            if (pager.UseWal())
            {
                // Write a single frame for this page to the log.
                if (subjRequiresPage(pg))
                    rc = subjournalPage(pg);
                if (rc == RC.OK)
                    rc = pager.pagerWalFrames(pg, 0, false);
            }
            else
            {
                // Sync the journal file if required. 
                if ((pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0 || pager.State == PAGER.WRITER_CACHEMOD)
                    rc = pager.syncJournal(true);

                // If the page number of this page is larger than the current size of the database image, it may need to be written to the sub-journal.
                // This is because the call to pager_write_pagelist() below will not actually write data to the file in this case.
                //
                // Consider the following sequence of events:
                //
                //   BEGIN;
                //     <journal page X>
                //     <modify page X>
                //     SAVEPOINT sp;
                //       <shrink database file to Y pages>
                //       pagerStress(page X)
                //     ROLLBACK TO sp;
                //
                // If (X>Y), then when pagerStress is called page X will not be written out to the database file, but will be dropped from the cache. Then,
                // following the "ROLLBACK TO sp" statement, reading page X will read data from the database file. This will be the copy of page X as it
                // was when the transaction started, not as it was when "SAVEPOINT sp" was executed.
                //
                // The solution is to write the current data for page X into the sub-journal file now (if it is not already there), so that it will
                // be restored to its current value when the "ROLLBACK TO sp" is executed.
                if (C._NEVER(rc == RC.OK && pg.ID > pager.DBSize && subjRequiresPage(pg)))
                    rc = subjournalPage(pg);

                // Write the contents of the page out to the database file.
                if (rc == RC.OK)
                {
                    Debug.Assert((pg.Flags & PgHdr.PGHDR.NEED_SYNC) == 0);
                    rc = pager.pager_write_pagelist(pg);
                }
            }

            // Mark the page as clean.
            if (rc == RC.OK)
            {
                PAGERTRACE("STRESS {0} page {1}", PAGERID(pager), pg.ID);
                PCache.MakeClean(pg);
            }

            return pager.pager_error(rc);
        }

        // was:sqlite3PagerOpen
        public static RC Open(VSystem vfs, out Pager pagerOut, string filename, int extraBytes, IPager.PAGEROPEN flags, VSystem.OPEN vfsFlags, Action<PgHdr> reinit, Func<object> memPageBuilder)
        {
            // Figure out how much space is required for each journal file-handle (there are two of them, the main journal and the sub-journal). This
            // is the maximum space required for an in-memory journal file handle and a regular journal file-handle. Note that a "regular journal-handle"
            // may be a wrapper capable of caching the first portion of the journal file in memory to implement the atomic-write optimization (see 
            // source file journal.c).
            int journalFileSize = SysEx.ROUND8(VFile.JournalVFileSize(vfs) > VFile.MemoryVFileSize() ? VFile.JournalVFileSize(vfs) : VFile.MemoryVFileSize()); // Bytes to allocate for each journal fd

            // Set the output variable to NULL in case an error occurs.
            pagerOut = null;

            bool memoryDB = false;      // True if this is an in-memory file
            string pathname = null;     // Full path to database file
            string uri = null;          // URI args to copy
#if !OMIT_MEMORYDB
            if ((flags & IPager.PAGEROPEN.MEMORY) != 0)
            {
                memoryDB = true;
                if (!string.IsNullOrEmpty(filename))
                {
                    pathname = filename;
                    filename = null;
                }
            }
#endif

            // Compute and store the full pathname in an allocated buffer pointed to by zPathname, length nPathname. Or, if this is a temporary file,
            // leave both nPathname and zPathname set to 0.
            var rc = RC.OK;
            if (!string.IsNullOrEmpty(filename))
            {
                rc = vfs.FullPathname(filename, out pathname);
                var z = uri = filename;
                // TODO: CONVERT
                //const char* z = uri = &filename[_strlen30(filename) + 1];
                //while (*z)
                //{
                //    z += _strlen30(z) + 1;
                //    z += _strlen30(z) + 1;
                //}
                //uriLength = (int)(&z[1] - uri);
                Debug.Assert(uri.Length >= 0);
                // This branch is taken when the journal path required by the database being opened will be more than pVfs->mxPathname
                // bytes in length. This means the database cannot be opened, as it will not be possible to open the journal file or even
                // check for a hot-journal before reading.
                if (rc == RC.OK && pathname.Length + 8 > vfs.MaxPathname)
                    rc = SysEx.CANTOPEN_BKPT();
                if (rc != RC.OK)
                    return rc;
            }

            // Allocate memory for the Pager structure, PCache object, the three file descriptors, the database file name and the journal file name.
            var pager = new Pager(); //memPageBuilder);
            pager.PCache = new PCache();
            pager.File = vfs.CreateOsFile();
            pager.SubJournalFile = new MemoryVFile();
            pager.JournalFile = new MemoryVFile();

            // Fill in the Pager.zFilename and Pager.zJournal buffers, if required.
            if (pathname != null)
            {
                Debug.Assert(pathname.Length > 0);
                pager.Filename = pathname;
                if (string.IsNullOrEmpty(uri)) pager.Filename += uri;
                pager.Journal = pager.Filename + "-journal";
#if !OMIT_WAL
                pager.WalName = pager.Filename + "-wal";
#endif
            }
            else
                pager.Filename = string.Empty;
            pager.Vfs = vfs;
            pager.VfsFlags = vfsFlags;

            // Open the pager file.
            var tempFile = false; // True for temp files (incl. in-memory files)
            var readOnly = false; // True if this is a read-only file
            uint sizePage = DEFAULT_PAGE_SIZE;  // Default page size
            if (!string.IsNullOrEmpty(filename))
            {
                VSystem.OPEN fout = 0; // VFS flags returned by xOpen()
                rc = vfs.Open(filename, pager.File, vfsFlags, out fout);
                Debug.Assert(!memoryDB);
                readOnly = ((fout & VSystem.OPEN.READONLY) != 0);

                // If the file was successfully opened for read/write access, choose a default page size in case we have to create the
                // database file. The default page size is the maximum of:
                //
                //    + SQLITE_DEFAULT_PAGE_SIZE,
                //    + The value returned by sqlite3OsSectorSize()
                //    + The largest page size that can be written atomically.
                if (rc == RC.OK && !readOnly)
                {
                    pager.setSectorSize();
                    Debug.Assert(DEFAULT_PAGE_SIZE <= MAX_DEFAULT_PAGE_SIZE);
                    if (sizePage < pager.SectorSize)
                        sizePage = (pager.SectorSize > MAX_DEFAULT_PAGE_SIZE ? MAX_DEFAULT_PAGE_SIZE : (uint)pager.SectorSize);
#if ENABLE_ATOMIC_WRITE
                    Debug.Assert((int)VFile.IOCAP.ATOMIC512 == (512 >> 8));
                    Debug.Assert((int)VFile.IOCAP.ATOMIC64K == (65536 >> 8));
                    Debug.Assert(MAX_DEFAULT_PAGE_SIZE <= 65536);
                    var dc = (uint)pager.File.get_DeviceCharacteristics();
                    for (var ii = sizePage; ii <= MAX_DEFAULT_PAGE_SIZE; ii = ii * 2)
                        if ((dc & ((uint)VFile.IOCAP.ATOMIC | (ii >> 8))) != 0)
                            sizePage = ii;
#endif
                }
            }
            else
            {
                // If a temporary file is requested, it is not opened immediately. In this case we accept the default page size and delay actually
                // opening the file until the first call to OsWrite().
                //
                // This branch is also run for an in-memory database. An in-memory database is the same as a temp-file that is never written out to
                // disk and uses an in-memory rollback journal.
                tempFile = true;
                pager.State = PAGER.READER;
                pager.Lock = VFile.LOCK.EXCLUSIVE;
                readOnly = (vfsFlags & VSystem.OPEN.READONLY) != 0;
            }

            // The following call to PagerSetPagesize() serves to set the value of Pager.pageSize and to allocate the Pager.pTmpSpace buffer.
            if (rc == RC.OK)
            {
                Debug.Assert(!pager.MemoryDB);
                rc = pager.SetPageSize(ref sizePage, -1);
            }

            // If an error occurred in either of the blocks above, free the Pager structure and close the file.
            if (rc != RC.OK)
            {
                Debug.Assert(pager.TmpSpace == null);
                pager.File.Close();
                return rc;
            }
            // Initialize the PCache object.
            Debug.Assert(extraBytes < 1000);
            extraBytes = SysEx.ROUND8(extraBytes);
            PCache.Open((int)sizePage, extraBytes, !memoryDB, (!memoryDB ? (Func<object, IPage, RC>)pagerStress : null), pager, pager.PCache);

            PAGERTRACE("OPEN {0} {1}", FILEHANDLEID(pager.File), pager.Filename);
            SysEx.IOTRACE("OPEN {0:x} {1}", pager.GetHashCode(), pager.Filename);

            bool useJournal = (flags & IPager.PAGEROPEN.OMIT_JOURNAL) == 0; // False to omit journal
            pager.UseJournal = useJournal;
            pager.MaxPid = MAX_PAGE_COUNT;
            pager.TempFile = tempFile;
            //Debug.Assert(tempFile == IPager.LOCKINGMODE.NORMAL || tempFile == IPager.LOCKINGMODE.EXCLUSIVE);
            //Debug.Assert(IPager.LOCKINGMODE.EXCLUSIVE == 1);
            pager.ExclusiveMode = tempFile;
            pager.ChangeCountDone = tempFile;
            pager.MemoryDB = memoryDB;
            pager.ReadOnly = readOnly;
            Debug.Assert(useJournal || tempFile);
            pager.NoSync = tempFile;
            if (pager.NoSync)
            {
                Debug.Assert(!pager.FullSync);
                Debug.Assert(pager.SyncFlags == 0);
                Debug.Assert(pager.WalSyncFlags == 0);
                Debug.Assert(pager.CheckpointSyncFlags == 0);
            }
            else
            {
                pager.FullSync = true;
                pager.SyncFlags = VFile.SYNC.NORMAL;
                pager.WalSyncFlags = VFile.SYNC.NORMAL | VFile.SYNC.WAL_TRANSACTIONS;
                pager.CheckpointSyncFlags = VFile.SYNC.NORMAL;
            }
            pager.ExtraBytes = (ushort)extraBytes;
            pager.JournalSizeLimit = DEFAULT_JOURNAL_SIZE_LIMIT;
            Debug.Assert(pager.File.Opened || tempFile);
            pager.setSectorSize();
            if (!useJournal)
                pager.JournalMode = IPager.JOURNALMODE.OFF;
            else if (memoryDB)
                pager.JournalMode = IPager.JOURNALMODE.JMEMORY;
            pager.Reiniter = reinit;

            pagerOut = pager;
            return RC.OK;
        }

        private RC hasHotJournal(out bool existsOut)
        {
            Debug.Assert(UseJournal);
            Debug.Assert(File.Opened);
            Debug.Assert(State == PAGER.OPEN);
            var journalOpened = JournalFile.Opened;
            Debug.Assert(!journalOpened || (JournalFile.get_DeviceCharacteristics() & VFile.IOCAP.UNDELETABLE_WHEN_OPEN) != 0);

            existsOut = false;
            var vfs = Vfs;
            var rc = RC.OK;
            var exists = 1; // True if a journal file is present
            if (!journalOpened)
                rc = vfs.Access(Journal, VSystem.ACCESS.EXISTS, out exists);
            if (rc == RC.OK && exists != 0)
            {
                // Race condition here:  Another process might have been holding the the RESERVED lock and have a journal open at the sqlite3OsAccess() 
                // call above, but then delete the journal and drop the lock before we get to the following sqlite3OsCheckReservedLock() call.  If that
                // is the case, this routine might think there is a hot journal when in fact there is none.  This results in a false-positive which will
                // be dealt with by the playback routine.  Ticket #3883.
                var locked = 0; // True if some process holds a RESERVED lock
                rc = File.CheckReservedLock(ref locked);
                if (rc == RC.OK && locked == 0)
                {
                    // Check the size of the database file. If it consists of 0 pages, then delete the journal file. See the header comment above for 
                    // the reasoning here.  Delete the obsolete journal file under a RESERVED lock to avoid race conditions and to avoid violating [H33020].
                    Pid pages; // Number of pages in database file
                    rc = pagerPagecount(out pages);
                    if (rc == RC.OK)
                        if (pages == 0)
                        {
                            C._benignalloc_begin();
                            if (pagerLockDb(VFile.LOCK.RESERVED) == RC.OK)
                            {
                                vfs.Delete(Journal, false);
                                if (!ExclusiveMode) pagerUnlockDb(VFile.LOCK.SHARED);
                            }
                            C._benignalloc_end();
                        }
                        else
                        {
                            // The journal file exists and no other connection has a reserved or greater lock on the database file. Now check that there is
                            // at least one non-zero bytes at the start of the journal file. If there is, then we consider this journal to be hot. If not, 
                            // it can be ignored.
                            if (journalOpened)
                            {
                                var f = VSystem.OPEN.READONLY | VSystem.OPEN.MAIN_JOURNAL;
                                rc = vfs.Open(Journal, JournalFile, f, out f);
                            }
                            if (rc == RC.OK)
                            {
                                var first = new byte[1];
                                rc = JournalFile.Read(first, 1, 0);
                                if (rc == RC.IOERR_SHORT_READ)
                                    rc = RC.OK;
                                if (!journalOpened)
                                    JournalFile.Close();
                                existsOut = (first[0] != 0);
                            }
                            else if (rc == RC.CANTOPEN)
                            {
                                // If we cannot open the rollback journal file in order to see if its has a zero header, that might be due to an I/O error, or
                                // it might be due to the race condition described above and in ticket #3883.  Either way, assume that the journal is hot.
                                // This might be a false positive.  But if it is, then the automatic journal playback and recovery mechanism will deal
                                // with it under an EXCLUSIVE lock where we do not need to worry so much with race conditions.
                                existsOut = true;
                                rc = RC.OK;
                            }
                        }
                }
            }

            return rc;
        }

        public RC SharedLock()
        {
            // This routine is only called from b-tree and only when there are no outstanding pages. This implies that the pager state should either
            // be OPEN or READER. READER is only possible if the pager is or was in exclusive access mode.
            Debug.Assert(PCache.get_Refs() == 0);
            Debug.Assert(assert_pager_state(this));
            Debug.Assert(State == PAGER.OPEN || State == PAGER.READER);
            if (C._NEVER(MemoryDB && ErrorCode != 0)) return ErrorCode;

            var rc = RC.OK;
            if (!UseWal() && State == PAGER.OPEN)
            {
                Debug.Assert(!MemoryDB);

                rc = pager_wait_on_lock(VFile.LOCK.SHARED);
                if (rc != RC.OK)
                {
                    Debug.Assert(Lock == VFile.LOCK.NO || Lock == VFile.LOCK.UNKNOWN);
                    goto failed;
                }

                // If a journal file exists, and there is no RESERVED lock on the database file, then it either needs to be played back or deleted.
                var hotJournal = true; // True if there exists a hot journal-file
                if (Lock <= VFile.LOCK.SHARED)
                    rc = hasHotJournal(out hotJournal);
                if (rc != RC.OK)
                    goto failed;
                if (hotJournal)
                {
                    if (ReadOnly)
                    {
                        rc = RC.READONLY;
                        goto failed;
                    }

                    // Get an EXCLUSIVE lock on the database file. At this point it is important that a RESERVED lock is not obtained on the way to the
                    // EXCLUSIVE lock. If it were, another process might open the database file, detect the RESERVED lock, and conclude that the
                    // database is safe to read while this process is still rolling the hot-journal back.
                    // 
                    // Because the intermediate RESERVED lock is not requested, any other process attempting to access the database file will get to 
                    // this point in the code and fail to obtain its own EXCLUSIVE lock on the database file.
                    //
                    // Unless the pager is in locking_mode=exclusive mode, the lock is downgraded to SHARED_LOCK before this function returns.
                    rc = pagerLockDb(VFile.LOCK.EXCLUSIVE);
                    if (rc != RC.OK)
                        goto failed;

                    // If it is not already open and the file exists on disk, open the journal for read/write access. Write access is required because 
                    // in exclusive-access mode the file descriptor will be kept open and possibly used for a transaction later on. Also, write-access 
                    // is usually required to finalize the journal in journal_mode=persist mode (and also for journal_mode=truncate on some systems).
                    //
                    // If the journal does not exist, it usually means that some other connection managed to get in and roll it back before 
                    // this connection obtained the exclusive lock above. Or, it may mean that the pager was in the error-state when this
                    // function was called and the journal file does not exist.
                    if (!JournalFile.Opened)
                    {
                        var vfs = Vfs;
                        int exists; // True if journal file exists
                        rc = vfs.Access(Journal, VSystem.ACCESS.EXISTS, out exists);
                        if (rc == RC.OK && exists != 0)
                        {
                            Debug.Assert(!TempFile);
                            VSystem.OPEN fout = 0;
                            rc = vfs.Open(Journal, JournalFile, VSystem.OPEN.READWRITE | VSystem.OPEN.MAIN_JOURNAL, out fout);
                            Debug.Assert(rc != RC.OK || JournalFile.Opened);
                            if (rc == RC.OK && (fout & VSystem.OPEN.READONLY) != 0)
                            {
                                rc = SysEx.CANTOPEN_BKPT();
                                JournalFile.Close();
                            }
                        }
                    }

                    // Playback and delete the journal.  Drop the database write lock and reacquire the read lock. Purge the cache before
                    // playing back the hot-journal so that we don't end up with an inconsistent cache.  Sync the hot journal before playing
                    // it back since the process that crashed and left the hot journal probably did not sync it and we are required to always sync
                    // the journal before playing it back.
                    if (JournalFile.Opened)
                    {
                        Debug.Assert(rc == RC.OK);
                        rc = pagerSyncHotJournal();
                        if (rc == RC.OK)
                        {
                            rc = pager_playback(true);
                            State = PAGER.OPEN;
                        }
                    }
                    else if (!ExclusiveMode)
                        pagerUnlockDb(VFile.LOCK.SHARED);

                    if (rc != RC.OK)
                    {
                        // This branch is taken if an error occurs while trying to open or roll back a hot-journal while holding an EXCLUSIVE lock. The
                        // pager_unlock() routine will be called before returning to unlock the file. If the unlock attempt fails, then Pager.eLock must be
                        // set to UNKNOWN_LOCK (see the comment above the #define for UNKNOWN_LOCK above for an explanation). 
                        //
                        // In order to get pager_unlock() to do this, set Pager.eState to PAGER_ERROR now. This is not actually counted as a transition
                        // to ERROR state in the state diagram at the top of this file, since we know that the same call to pager_unlock() will very
                        // shortly transition the pager object to the OPEN state. Calling assert_pager_state() would fail now, as it should not be possible
                        // to be in ERROR state when there are zero outstanding page references.
                        pager_error(rc);
                        goto failed;
                    }

                    Debug.Assert(State == PAGER.OPEN);
                    Debug.Assert((Lock == VFile.LOCK.SHARED) || (ExclusiveMode && Lock > VFile.LOCK.SHARED));
                }

                if (!TempFile && (Backup != null || PCache.get_Pages() > 0))
                {
                    // The shared-lock has just been acquired on the database file and there are already pages in the cache (from a previous
                    // read or write transaction).  Check to see if the database has been modified.  If the database has changed, flush the cache.
                    //
                    // Database changes is detected by looking at 15 bytes beginning at offset 24 into the file.  The first 4 of these 16 bytes are
                    // a 32-bit counter that is incremented with each change.  The other bytes change randomly with each file change when
                    // a codec is in use.
                    // 
                    // There is a vanishingly small chance that a change will not be detected.  The chance of an undetected change is so small that
                    // it can be neglected.
                    Pid pages;
                    var dbFileVersion = new byte[DBFileVersion.Length];

                    rc = pagerPagecount(out pages);
                    if (rc != RC.OK)
                        goto failed;

                    if (pages > 0)
                    {
                        SysEx.IOTRACE("CKVERS {0} {1}\n", this, dbFileVersion.Length);
                        rc = File.Read(dbFileVersion, dbFileVersion.Length, 24);
                        if (rc != RC.OK)
                            goto failed;
                    }
                    else
                        Array.Clear(dbFileVersion, 0, dbFileVersion.Length);

                    if (!Enumerable.SequenceEqual(DBFileVersion, dbFileVersion))
                        pager_reset();
                }

                // If there is a WAL file in the file-system, open this database in WAL mode. Otherwise, the following function call is a no-op.
                rc = pagerOpenWalIfPresent();
#if !OMIT_WAL
                Debug.Assert(pager.Wal == null || rc == RC.OK);
#endif
            }

            if (UseWal())
            {
                Debug.Assert(rc == RC.OK);
                rc = pagerBeginReadTransaction();
            }

            if (State == PAGER.OPEN && rc == RC.OK)
                rc = pagerPagecount(out DBSize);

        failed:
            if (rc != RC.OK)
            {
                Debug.Assert(MemoryDB);
                pager_unlock();
                Debug.Assert(State == PAGER.OPEN);
            }
            else
                State = PAGER.READER;
            return rc;
        }

        private void UnlockIfUnused()
        {
            if (PCache.get_Refs() == 0)
                pagerUnlockAndRollback();
        }

        public RC Acquire(Pid id, ref PgHdr pageOut, bool noContent)
        {
            Debug.Assert(State >= PAGER.READER);
            Debug.Assert(assert_pager_state(this));

            if (id == 0)
                return SysEx.CORRUPT_BKPT();

            // If the pager is in the error state, return an error immediately.  Otherwise, request the page from the PCache layer.
            RC rc;
            if (ErrorCode != RC.OK)
                rc = ErrorCode;
            else
                rc = PCache.Fetch(id, true, out pageOut);

            PgHdr pg;
            if (rc != RC.OK)
            {
                pg = null;
                goto pager_get_err;
            }
            Debug.Assert(pageOut.ID == id);
            Debug.Assert(pageOut.Pager == this || pageOut.Pager == null);

            if (pageOut.Pager != null && !noContent)
            {
                // In this case the pcache already contains an initialized copy of the page. Return without further ado.
                Debug.Assert(id <= MAX_PID && id != MJ_PID(this));
                return RC.OK;
            }
            // The pager cache has created a new page. Its content needs to be initialized.
            pg = pageOut;
            pg.Pager = this;

            // The maximum page number is 2^31. Return CORRUPT if a page number greater than this, or the unused locking-page, is requested.
            if (id > MAX_PID || id == MJ_PID(this))
            {
                rc = SysEx.CORRUPT_BKPT();
                goto pager_get_err;
            }

            if (MemoryDB || DBSize < id || noContent || !File.Opened)
            {
                if (id > MaxPid)
                {
                    rc = RC.FULL;
                    goto pager_get_err;
                }
                if (noContent)
                {
                    // Failure to set the bits in the InJournal bit-vectors is benign. It merely means that we might do some extra work to journal a 
                    // page that does not need to be journaled.  Nevertheless, be sure to test the case where a malloc error occurs while trying to set 
                    // a bit in a bit vector.
                    C._benignalloc_begin();
                    if (id <= DBOrigSize)
                        InJournal.Set(id);
                    addToSavepointBitvecs(id);
                    C._benignalloc_end();
                }
                Array.Clear(pg.Data, 0, PageSize);
                SysEx.IOTRACE("ZERO {0:x} {1}\n", GetHashCode(), id);
            }
            else
            {
                Debug.Assert(pg.Pager == this);
                Stats[(int)STAT.MISS]++;
                rc = readDbPage(pg);
                if (rc != RC.OK)
                    goto pager_get_err;
            }
            pager_set_pagehash(pg);
            return RC.OK;

        pager_get_err:
            Debug.Assert(rc != RC.OK);
            if (pg != null)
                PCache.Drop(pg);
            UnlockIfUnused();

            pageOut = null;
            return rc;
        }

        public IPage Lookup(Pid id)
        {
            Debug.Assert(id != 0);
            Debug.Assert(PCache != null);
            Debug.Assert(State >= PAGER.READER && State != PAGER.ERROR);
            PgHdr pg;
            PCache.Fetch(id, false, out pg);
            return pg;
        }

        public static void Unref(IPage pg)
        {
            if (pg != null)
            {
                var pager = pg.Pager;
                PCache.Release(pg);
                pager.UnlockIfUnused();
            }
        }

        private RC pager_open_journal()
        {
            Debug.Assert(State == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(this));
            Debug.Assert(InJournal == null);

            // If already in the error state, this function is a no-op.  But on the other hand, this routine is never called if we are already in
            // an error state.
            if (C._NEVER(ErrorCode != RC.OK)) return ErrorCode;

            var rc = RC.OK;
            if (!UseWal() && JournalMode != IPager.JOURNALMODE.OFF)
            {
                InJournal = new Bitvec(DBSize);

                // Open the journal file if it is not already open.
                if (!JournalFile.Opened)
                {
                    if (JournalMode == IPager.JOURNALMODE.JMEMORY)
                        //sqlite3MemJournalOpen(pager->JournalFile);
                        JournalFile = new MemoryVFile();
                    else
                    {
                        var flags = VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE | (TempFile ? VSystem.OPEN.DELETEONCLOSE | VSystem.OPEN.TEMP_JOURNAL : VSystem.OPEN.MAIN_JOURNAL);
#if ENABLE_ATOMIC_WRITE
                        rc = VFile.JournalVFileOpen(Vfs, Journal, ref JournalFile, flags, jrnlBufferSize(this));
#else
                        VSystem.OPEN dummy;
                        rc = Vfs.Open(Journal, JournalFile, flags, out dummy);
#endif
                    }
                    Debug.Assert(rc != RC.OK || JournalFile.Opened);
                }

                // Write the first journal header to the journal file and open the sub-journal if necessary.
                if (rc == RC.OK)
                {
                    // TODO: Check if all of these are really required.
                    Records = 0;
                    JournalOffset = 0;
                    SetMaster = false;
                    JournalHeader = 0;
                    rc = writeJournalHdr();
                }
            }

            if (rc != RC.OK)
            {
                Bitvec.Destroy(ref InJournal);
                InJournal = null;
            }
            else
            {
                Debug.Assert(State == PAGER.WRITER_LOCKED);
                State = PAGER.WRITER_CACHEMOD;
            }

            return rc;
        }

        // was:sqlite3PagerBegin
        public RC Begin(bool exFlag, bool subjInMemory)
        {
            if (ErrorCode != 0) return ErrorCode;
            Debug.Assert(State >= PAGER.READER && State < PAGER.ERROR);
            SubjInMemory = subjInMemory;

            var rc = RC.OK;
            if (C._ALWAYS(State == PAGER.READER))
            {
                Debug.Assert(InJournal == null);

                if (UseWal())
                {
                    // If the pager is configured to use locking_mode=exclusive, and an exclusive lock on the database is not already held, obtain it now.
                    if (ExclusiveMode && Wal.ExclusiveMode(-1))
                    {
                        rc = pagerLockDb(VFile.LOCK.EXCLUSIVE);
                        if (rc != RC.OK)
                            return rc;
                        Wal.ExclusiveMode(1);
                    }

                    // Grab the write lock on the log file. If successful, upgrade to PAGER_RESERVED state. Otherwise, return an error code to the caller.
                    // The busy-handler is not invoked if another connection already holds the write-lock. If possible, the upper layer will call it.
                    rc = Wal.BeginWriteTransaction();
                }
                else
                {
                    // Obtain a RESERVED lock on the database file. If the exFlag parameter is true, then immediately upgrade this to an EXCLUSIVE lock. The
                    // busy-handler callback can be used when upgrading to the EXCLUSIVE lock, but not when obtaining the RESERVED lock.
                    rc = pagerLockDb(VFile.LOCK.RESERVED);
                    if (rc == RC.OK && exFlag)
                        rc = pager_wait_on_lock(VFile.LOCK.EXCLUSIVE);
                }

                if (rc == RC.OK)
                {
                    // Change to WRITER_LOCKED state.
                    //
                    // WAL mode sets Pager.eState to PAGER_WRITER_LOCKED or CACHEMOD when it has an open transaction, but never to DBMOD or FINISHED.
                    // This is because in those states the code to roll back savepoint transactions may copy data from the sub-journal into the database 
                    // file as well as into the page cache. Which would be incorrect in WAL mode.
                    State = PAGER.WRITER_LOCKED;
                    DBHintSize = DBSize;
                    DBFileSize = DBSize;
                    DBOrigSize = DBSize;
                    JournalOffset = 0;
                }

                Debug.Assert(rc == RC.OK || State == PAGER.READER);
                Debug.Assert(rc != RC.OK || State == PAGER.WRITER_LOCKED);
                Debug.Assert(assert_pager_state(this));
            }

            PAGERTRACE("TRANSACTION {0}", PAGERID(this));
            return rc;
        }

        private static RC pager_write(PgHdr pg)
        {
            var data = pg.Data;
            var pager = pg.Pager;

            // This routine is not called unless a write-transaction has already been started. The journal file may or may not be open at this point. It is never called in the ERROR state.
            Debug.Assert(pager.State == PAGER.WRITER_LOCKED ||
                pager.State == PAGER.WRITER_CACHEMOD ||
                pager.State == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pager));

            // If an error has been previously detected, report the same error again. This should not happen, but the check provides robustness. 
            if (C._NEVER(pager.ErrorCode != RC.OK)) return pager.ErrorCode;

            // Higher-level routines never call this function if database is not writable.  But check anyway, just for robustness.
            if (C._NEVER(pager.ReadOnly)) return RC.PERM;

            checkPage(pg);

            // The journal file needs to be opened. Higher level routines have already obtained the necessary locks to begin the write-transaction, but the
            // rollback journal might not yet be open. Open it now if this is the case.
            //
            // This is done before calling sqlite3PcacheMakeDirty() on the page. Otherwise, if it were done after calling sqlite3PcacheMakeDirty(), then
            // an error might occur and the pager would end up in WRITER_LOCKED state with pages marked as dirty in the cache.
            var rc = RC.OK;
            if (pager.State == PAGER.WRITER_LOCKED)
            {
                rc = pager.pager_open_journal();
                if (rc != RC.OK) return rc;
            }
            Debug.Assert(pager.State >= PAGER.WRITER_CACHEMOD);
            Debug.Assert(assert_pager_state(pager));

            // Mark the page as dirty.  If the page has already been written to the journal then we can return right away.
            PCache.MakeDirty(pg);
            if (pageInJournal(pg) && !subjRequiresPage(pg))
                Debug.Assert(!pager.UseWal());
            else
            {
                // The transaction journal now exists and we have a RESERVED or an EXCLUSIVE lock on the main database file.  Write the current page to the transaction journal if it is not there already.
                if (!pageInJournal(pg) && !pager.UseWal())
                {
                    Debug.Assert(!pager.UseWal());
                    if (pg.ID <= pager.DBOrigSize && pager.JournalFile.Opened)
                    {
                        // We should never write to the journal file the page that contains the database locks.  The following Debug.Assert verifies that we do not.
                        Debug.Assert(pg.ID != MJ_PID(pager));

                        Debug.Assert(pager.JournalHeader <= pager.JournalOffset);
                        byte[] data2 = null;
                        if (CODEC2(ref data2, pager, data, pg.ID, 7)) return RC.NOMEM;
                        var checksum = pager.pager_cksum(data2);

                        // Even if an IO or diskfull error occurs while journalling the page in the block above, set the need-sync flag for the page.
                        // Otherwise, when the transaction is rolled back, the logic in playback_one_page() will think that the page needs to be restored
                        // in the database file. And if an IO error occurs while doing so, then corruption may follow.
                        pg.Flags |= PgHdr.PGHDR.NEED_SYNC;

                        var offset = pager.JournalOffset;
                        rc = pager.JournalFile.Write4(offset, pg.ID);
                        if (rc != RC.OK) return rc;
                        rc = pager.JournalFile.Write(data2, pager.PageSize, offset + 4);
                        if (rc != RC.OK) return rc;
                        rc = pager.JournalFile.Write4(offset + pager.PageSize + 4, checksum);
                        if (rc != RC.OK) return rc;

                        SysEx.IOTRACE("JOUT {0:x} {1} {2,11} {3}", pager.GetHashCode(), pg.ID, pager.JournalOffset, pager.PageSize);
                        PAGER_INCR(ref _writej_count);
                        PAGERTRACE("JOURNAL {0} page {1} needSync={2} hash({3,08:x})", PAGERID(pager), pg.ID, (pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0, pager_pagehash(pg));

                        pager.JournalOffset += 8 + pager.PageSize;
                        pager.Records++;
                        Debug.Assert(pager.InJournal != null);
                        rc = pager.InJournal.Set(pg.ID);
                        Debug.Assert(rc == RC.OK || rc == RC.NOMEM);
                        rc |= pager.addToSavepointBitvecs(pg.ID);
                        if (rc != RC.OK)
                        {
                            Debug.Assert(rc == RC.NOMEM);
                            return rc;
                        }
                    }
                    else
                    {
                        if (pager.State != PAGER.WRITER_DBMOD)
                            pg.Flags |= PgHdr.PGHDR.NEED_SYNC;
                        PAGERTRACE("APPEND {0} page {1} needSync={2}", PAGERID(pager), pg.ID, (pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0);
                    }
                }

                // If the statement journal is open and the page is not in it, then write the current page to the statement journal.  Note that
                // the statement journal format differs from the standard journal format in that it omits the checksums and the header.
                if (subjRequiresPage(pg))
                    rc = subjournalPage(pg);
            }

            // Update the database size and return.
            if (pager.DBSize < pg.ID)
                pager.DBSize = pg.ID;
            return rc;
        }

        // was:sqlite3PagerWrite
        public static RC Write(IPage page)
        {
            var pg = page;
            var pager = pg.Pager;
            var pagePerSector = (Pid)(pager.SectorSize / pager.PageSize);

            Debug.Assert(pager.State >= PAGER.WRITER_LOCKED);
            Debug.Assert(pager.State != PAGER.ERROR);
            Debug.Assert(assert_pager_state(pager));

            var rc = RC.OK;

            if (pagePerSector > 1)
            {
                // Set the doNotSyncSpill flag to 1. This is because we cannot allow a journal header to be written between the pages journaled by this function.
                Debug.Assert(!pager.MemoryDB);
                Debug.Assert(pager.DoNotSyncSpill == 0);
                pager.DoNotSyncSpill++;

                // This trick assumes that both the page-size and sector-size are an integer power of 2. It sets variable pg1 to the identifier
                // of the first page of the sector pPg is located on.
                var pg1 = (Pid)((pg.ID - 1) & ~(pagePerSector - 1)) + 1; // First page of the sector pPg is located on.

                Pid pages = 0; // Number of pages starting at pg1 to journal
                var pageCount = pager.DBSize; // Total number of pages in database file
                if (pg.ID > pageCount)
                    pages = (pg.ID - pg1) + 1;
                else if ((pg1 + pagePerSector - 1) > pageCount)
                    pages = pageCount + 1 - pg1;
                else
                    pages = pagePerSector;
                Debug.Assert(pages > 0);
                Debug.Assert(pg1 <= pg.ID);
                Debug.Assert((pg1 + pages) > pg.ID);

                bool needSync = false;   // True if any page has PGHDR_NEED_SYNC
                for (var ii = 0; ii < pages && rc == RC.OK; ii++)
                {
                    var id = (Pid)(pg1 + ii);
                    var page2 = new PgHdr();
                    if (id == pg.ID || !pager.InJournal.Get(id))
                    {
                        if (id != MJ_PID(pager))
                        {
                            rc = pager.Acquire(id, ref page2, false);
                            if (rc == RC.OK)
                            {
                                rc = pager_write(page2);
                                if ((page2.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                                    needSync = true;
                                Unref(page2);
                            }
                        }
                    }
                    else if ((page2 = pager.pager_lookup(id)) != null)
                    {
                        if ((page2.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                            needSync = true;
                        Unref(page2);
                    }
                }

                // If the PGHDR_NEED_SYNC flag is set for any of the nPage pages starting at pg1, then it needs to be set for all of them. Because
                // writing to any of these nPage pages may damage the others, the journal file must contain sync()ed copies of all of them
                // before any of them can be written out to the database file.
                if (rc == RC.OK && needSync)
                {
                    Debug.Assert(!pager.MemoryDB);
                    for (var ii = 0; ii < pages; ii++)
                    {
                        var page2 = pager.pager_lookup((Pid)(pg1 + ii));
                        if (page2 != null)
                        {
                            page2.Flags |= PgHdr.PGHDR.NEED_SYNC;
                            Unref(page2);
                        }
                    }
                }

                Debug.Assert(pager.DoNotSyncSpill == 1);
                pager.DoNotSyncSpill--;
            }
            else
                rc = pager_write(page);
            return rc;
        }

#if DEBUG
        public static bool Iswriteable(IPage pg)
        {
            return ((pg.Flags & PgHdr.PGHDR.DIRTY) != 0);
        }
#endif

        public static void DontWrite(PgHdr pg)
        {
            var pager = pg.Pager;
            if ((pg.Flags & PgHdr.PGHDR.DIRTY) != 0 && pager.Savepoints.Length == 0)
            {
                PAGERTRACE("DONT_WRITE page {0} of {1}", pg.ID, PAGERID(pager));
                SysEx.IOTRACE("CLEAN {0:x} {1}", pager.GetHashCode(), pg.ID);
                pg.Flags |= PgHdr.PGHDR.DONT_WRITE;
                pager_set_pagehash(pg);
            }
        }

        private RC pager_incr_changecounter(bool isDirectMode)
        {
            Debug.Assert(State == PAGER.WRITER_CACHEMOD ||
                State == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(this));

            // Declare and initialize constant integer 'isDirect'. If the atomic-write optimization is enabled in this build, then isDirect
            // is initialized to the value passed as the isDirectMode parameter to this function. Otherwise, it is always set to zero.
            //
            // The idea is that if the atomic-write optimization is not enabled at compile time, the compiler can omit the tests of
            // 'isDirect' below, as well as the block enclosed in the "if( isDirect )" condition.
#if !ENABLE_ATOMIC_WRITE
            var DIRECT_MODE = false;
            Debug.Assert(!isDirectMode);
#else
            var DIRECT_MODE = isDirectMode;
#endif
            var rc = RC.OK;
            if (!ChangeCountDone && C._ALWAYS(DBSize > 0))
            {
                Debug.Assert(!TempFile && File.Opened);

                // Open page 1 of the file for writing.
                PgHdr pgHdr = null; // Reference to page 1
                rc = Acquire(1, ref pgHdr, false);
                Debug.Assert(pgHdr == null || rc == RC.OK);

                // If page one was fetched successfully, and this function is not operating in direct-mode, make page 1 writable.  When not in 
                // direct mode, page 1 is always held in cache and hence the PagerGet() above is always successful - hence the ALWAYS on rc==SQLITE_OK.
                if (!DIRECT_MODE && C._ALWAYS(rc == RC.OK))
                    rc = Write(pgHdr);

                if (rc == RC.OK)
                {
                    // Actually do the update of the change counter
                    pager_write_changecounter(pgHdr);

                    // If running in direct mode, write the contents of page 1 to the file.
                    if (DIRECT_MODE)
                    {
                        byte[] buf = null;
                        Debug.Assert(DBFileSize > 0);
                        if (CODEC2(ref buf, this, pgHdr.Data, 1, 6)) return rc = RC.NOMEM;
                        if (rc == RC.OK)
                        {
                            rc = File.Write(buf, PageSize, 0);
                            Stats[(int)STAT.WRITE]++;
                        }
                        if (rc == RC.OK)
                            ChangeCountDone = true;
                    }
                    else
                        ChangeCountDone = true;
                }

                // Release the page reference.
                Unref(pgHdr);
            }
            return rc;
        }

        public RC Sync()
        {
            var rc = RC.OK;
            if (!NoSync)
            {
                Debug.Assert(!MemoryDB);
                rc = File.Sync(SyncFlags);
            }
            else if (File.Opened)
            {
                Debug.Assert(!MemoryDB);
                var refArg = 0L;
                rc = File.FileControl(VFile.FCNTL.SYNC_OMITTED, ref refArg);
                if (rc == RC.NOTFOUND)
                    rc = RC.OK;
            }
            return rc;
        }

        #endregion

        #region Commit

        public RC ExclusiveLock()
        {
            Debug.Assert(State == PAGER.WRITER_CACHEMOD ||
                State == PAGER.WRITER_DBMOD ||
                State == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(this));
            var rc = RC.OK;
            if (!UseWal())
                rc = pager_wait_on_lock(VFile.LOCK.EXCLUSIVE);
            return rc;
        }

        public RC CommitPhaseOne(string master, bool noSync)
        {
            Debug.Assert(State == PAGER.WRITER_LOCKED ||
                State == PAGER.WRITER_CACHEMOD ||
                State == PAGER.WRITER_DBMOD ||
                State == PAGER.ERROR);
            Debug.Assert(assert_pager_state(this));

            // If a prior error occurred, report that error again.
            if (C._NEVER(ErrorCode != 0)) return ErrorCode;

            PAGERTRACE("DATABASE SYNC: File={0} zMaster={1} nSize={2}", Filename, master, DBSize);

            // If no database changes have been made, return early.
            if (State < PAGER.WRITER_CACHEMOD) return RC.OK;

            var rc = RC.OK;
            if (MemoryDB)
            {
                // If this is an in-memory db, or no pages have been written to, or this function has already been called, it is mostly a no-op.  However, any
                // backup in progress needs to be restarted.
                if (Backup != null) Backup.Restart();
            }
            else
            {
                if (UseWal())
                {
                    var list = PCache.DirtyList();
                    PgHdr pageOne = null;
                    if (list == null)
                    {
                        // Must have at least one page for the WAL commit flag. Ticket [2d1a5c67dfc2363e44f29d9bbd57f] 2011-05-18
                        rc = Acquire(1, ref pageOne, false);
                        list = pageOne;
                        list.Dirty = null;
                    }
                    Debug.Assert(rc == RC.OK);
                    if (C._ALWAYS(list != null))
                        rc = pagerWalFrames(list, DBSize, true);
                    Unref(pageOne);
                    if (rc == RC.OK)
                        PCache.CleanAll();
                }
                else
                {
                    // The following block updates the change-counter. Exactly how it does this depends on whether or not the atomic-update optimization
                    // was enabled at compile time, and if this transaction meets the runtime criteria to use the operation: 
                    //
                    //    * The file-system supports the atomic-write property for blocks of size page-size, and 
                    //    * This commit is not part of a multi-file transaction, and
                    //    * Exactly one page has been modified and store in the journal file.
                    //
                    // If the optimization was not enabled at compile time, then the pager_incr_changecounter() function is called to update the change
                    // counter in 'indirect-mode'. If the optimization is compiled in but is not applicable to this transaction, call sqlite3JournalCreate()
                    // to make sure the journal file has actually been created, then call pager_incr_changecounter() to update the change-counter in indirect mode. 
                    //
                    // Otherwise, if the optimization is both enabled and applicable, then call pager_incr_changecounter() to update the change-counter
                    // in 'direct' mode. In this case the journal file will never be created for this transaction.
#if ENABLE_ATOMIC_WRITE
                    PgHdr pg;
                    Debug.Assert(JournalFile.Opened ||
                        JournalMode == IPager.JOURNALMODE.OFF ||
                        JournalMode == IPager.JOURNALMODE.WAL);
                    if (master == null && JournalFile.Opened &&
                        JournalOffset == jrnlBufferSize(this) &&
                        DBSize >= DBOrigSize &&
                        ((pg = PCache.DirtyList()) == null || pg.Dirty == null))
                    {
                        // Update the db file change counter via the direct-write method. The following call will modify the in-memory representation of page 1 
                        // to include the updated change counter and then write page 1 directly to the database file. Because of the atomic-write 
                        // property of the host file-system, this is safe.
                        rc = pager_incr_changecounter(true);
                    }
                    else
                    {
                        rc = VFile.JournalVFileCreate(JournalFile);
                        if (rc == RC.OK)
                            rc = pager_incr_changecounter(false);
                    }
#else
                    rc = pager_incr_changecounter(false);
#endif
                    if (rc != RC.OK) goto commit_phase_one_exit;

                    // Write the master journal name into the journal file. If a master journal file name has already been written to the journal file, 
                    // or if zMaster is NULL (no master journal), then this call is a no-op.
                    rc = writeMasterJournal(master);
                    if (rc != RC.OK) goto commit_phase_one_exit;

                    // Sync the journal file and write all dirty pages to the database. If the atomic-update optimization is being used, this sync will not 
                    // create the journal file or perform any real IO.
                    //
                    // Because the change-counter page was just modified, unless the atomic-update optimization is used it is almost certain that the
                    // journal requires a sync here. However, in locking_mode=exclusive on a system under memory pressure it is just possible that this is 
                    // not the case. In this case it is likely enough that the redundant xSync() call will be changed to a no-op by the OS anyhow. 
                    rc = syncJournal(false);
                    if (rc != RC.OK) goto commit_phase_one_exit;

                    rc = pager_write_pagelist(PCache.DirtyList());
                    if (rc != RC.OK)
                    {
                        Debug.Assert(rc != RC.IOERR_BLOCKED);
                        goto commit_phase_one_exit;
                    }
                    PCache.CleanAll();

                    // If the file on disk is smaller than the database image, use pager_truncate to grow the file here. This can happen if the database
                    // image was extended as part of the current transaction and then the last page in the db image moved to the free-list. In this case the
                    // last page is never written out to disk, leaving the database file undersized. Fix this now if it is the case.
                    if (DBSize != DBFileSize)
                    {
                        var newID = (Pid)(DBSize - (DBSize == MJ_PID(this) ? 1 : 0));
                        Debug.Assert(State >= PAGER.WRITER_DBMOD);
                        rc = pager_truncate(newID);
                        if (rc != RC.OK) goto commit_phase_one_exit;
                    }

                    // Finally, sync the database file.
                    if (!noSync)
                        rc = Sync();
                    SysEx.IOTRACE("DBSYNC {0:x}", GetHashCode());
                }
            }

        commit_phase_one_exit:
            if (rc == RC.OK && !UseWal())
                State = PAGER.WRITER_FINISHED;
            return rc;
        }

        public RC CommitPhaseTwo()
        {
            // This routine should not be called if a prior error has occurred. But if (due to a coding error elsewhere in the system) it does get
            // called, just return the same error code without doing anything.
            if (C._NEVER(ErrorCode != RC.OK)) return ErrorCode;
            Debug.Assert(State == PAGER.WRITER_LOCKED ||
                State == PAGER.WRITER_FINISHED ||
                (UseWal() && State == PAGER.WRITER_CACHEMOD));
            Debug.Assert(assert_pager_state(this));

            // An optimization. If the database was not actually modified during this transaction, the pager is running in exclusive-mode and is
            // using persistent journals, then this function is a no-op.
            //
            // The start of the journal file currently contains a single journal header with the nRec field set to 0. If such a journal is used as
            // a hot-journal during hot-journal rollback, 0 changes will be made to the database file. So there is no need to zero the journal 
            // header. Since the pager is in exclusive mode, there is no need to drop any locks either.
            if (State == PAGER.WRITER_LOCKED &&
                ExclusiveMode &&
                JournalMode == IPager.JOURNALMODE.PERSIST)
            {
                Debug.Assert(JournalOffset == JOURNAL_HDR_SZ(this) || JournalOffset == 0);
                State = PAGER.READER;
                return RC.OK;
            }

            PAGERTRACE("COMMIT {0}", PAGERID(this));
            var rc = pager_end_transaction(SetMaster, true);
            return pager_error(rc);
        }

        public RC Rollback()
        {
            PAGERTRACE("ROLLBACK {0}", PAGERID(this));
            // PagerRollback() is a no-op if called in READER or OPEN state. If the pager is already in the ERROR state, the rollback is not attempted here. Instead, the error code is returned to the caller.
            Debug.Assert(assert_pager_state(this));
            if (State == PAGER.ERROR) return ErrorCode;
            if (State <= PAGER.READER) return RC.OK;

            var rc = RC.OK;
            if (UseWal())
            {
                rc = Savepoint(IPager.SAVEPOINT.ROLLBACK, -1);
                var rc2 = pager_end_transaction(SetMaster, false);
                if (rc == RC.OK) rc = rc2;
            }
            else if (!JournalFile.Opened || State == PAGER.WRITER_LOCKED)
            {
                var eState = State;
                rc = pager_end_transaction(false, false);
                if (MemoryDB && eState > PAGER.WRITER_LOCKED)
                {
                    // This can happen using journal_mode=off. Move the pager to the error state to indicate that the contents of the cache may not be trusted. Any active readers will get SQLITE_ABORT.
                    ErrorCode = RC.ABORT;
                    State = PAGER.ERROR;
                    return rc;
                }
            }
            else
                rc = pager_playback(false);

            Debug.Assert(State == PAGER.READER || rc != RC.OK);
            Debug.Assert(rc == RC.OK || rc == RC.FULL || ((int)rc & 0xFF) == (int)RC.IOERR);

            // If an error occurs during a ROLLBACK, we can no longer trust the pager cache. So call pager_error() on the way out to make any error persistent.
            return pager_error(rc);
        }

        #endregion

        #region Name4

        public bool get_Readonly()
        {
            return ReadOnly;
        }

        public int get_Refs()
        {
            return PCache.get_Refs();
        }

        public int get_MemUsed()
        {
            var perPageSize = PageSize + ExtraBytes + 20;
            return perPageSize * PCache.get_Pages() + 0 + PageSize;
        }

        public static int get_PageRefs(IPage page)
        {
            return PCache.get_PageRefs(page);
        }

#if TEST
        int[] sqlite3PagerStats()
        {
            var a = new int[11];
            a[0] = PCache.get_Refs();
            a[1] = PCache.get_Pages();
            a[2] = (int)PCache.get_CacheSize();
            a[3] = (State == PAGER.OPEN ? -1 : (int)DBSize);
            a[4] = (int)State;
            a[5] = (int)ErrorCode;
            a[6] = Stats[(int)STAT.HIT];
            a[7] = Stats[(int)STAT.MISS];
            a[8] = 0;  // Used to be pager->nOvfl
            a[9] = Reads;
            a[10] = Stats[(int)STAT.WRITE];
            return a;
        }
#endif
        public void CacheStat(int dbStatus, bool reset, ref int value)
        {
            //Debug.Assert(dbStatus == DBSTATUS::CACHE_HIT ||
            //    dbStatus == DBSTATUS::CACHE_MISS ||
            //    dbStatus == DBSTATUS::CACHE_WRITE);
            //Debug.Assert(DBSTATUS::CACHE_HIT + 1 == DBSTATUS::CACHE_MISS);
            //Debug.Assert(DBSTATUS::CACHE_HIT + 2 == DBSTATUS::CACHE_WRITE);
            //Debug.Assert(STAT::HIT == 0 &&
            //    STAT::MISS == 1 &&
            //    STAT::WRITE == 2);

            //value += Stats[dbStatus - DBSTATUS::CACHE_HIT];
            //if (reset)
            //    Stats[dbStatus - DBSTATUS::CACHE_HIT] = 0;
        }

        public bool get_MemoryDB
        {
            get { return MemoryDB; }
        }

        public RC OpenSavepoint(int savepoints)
        {
            Debug.Assert(State >= PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(this));

            var rc = RC.OK;
            var currentSavepoints = Savepoints.Length; // Current number of savepoints 
            if (savepoints > currentSavepoints && UseJournal)
            {
                // Grow the Pager.aSavepoint array using realloc(). Return SQLITE_NOMEM if the allocation fails. Otherwise, zero the new portion in case a 
                // malloc failure occurs while populating it in the for(...) loop below.
                Array.Resize(ref Savepoints, savepoints);
                var newSavepoints = Savepoints;

                // Populate the PagerSavepoint structures just allocated.
                for (var ii = currentSavepoints; ii < Savepoints.Length; ii++)
                {
                    newSavepoints[ii] = new PagerSavepoint();
                    newSavepoints[ii].Orig = DBSize;
                    newSavepoints[ii].Offset = (JournalFile.Opened && JournalOffset > 0 ? JournalOffset : (int)JOURNAL_HDR_SZ(this));
                    newSavepoints[ii].SubRecords = SubRecords;
                    newSavepoints[ii].InSavepoint = new Bitvec(DBSize);
                    if (UseWal())
                        Wal.Savepoint(newSavepoints[ii].WalData);
                    var savepointsLength = ii + 1;
                    if (Savepoints.Length != savepointsLength)
                    {
                        var lastSavepoints = Savepoints;
                        Savepoints = new PagerSavepoint[savepointsLength];
                        lastSavepoints.CopyTo(Savepoints, Math.Min(lastSavepoints.Length, Savepoints.Length));
                    }
                }
                Debug.Assert(Savepoints.Length == savepoints);
                assertTruncateConstraint();
            }
            return rc;
        }

        public RC Savepoint(IPager.SAVEPOINT op, int savepoints)
        {
            Debug.Assert(op == IPager.SAVEPOINT.RELEASE || op == IPager.SAVEPOINT.ROLLBACK);
            Debug.Assert(savepoints >= 0 || op == IPager.SAVEPOINT.ROLLBACK);
            var rc = ErrorCode;
            if (rc == RC.OK && savepoints < Savepoints.Length)
            {
                // Figure out how many savepoints will still be active after this operation. Store this value in nNew. Then free resources associated 
                // with any savepoints that are destroyed by this operation.
                var newLength = savepoints + ((op == IPager.SAVEPOINT.RELEASE) ? 0 : 1); // Number of remaining savepoints after this op.
                for (var ii = newLength; ii < Savepoints.Length; ii++)
                    Bitvec.Destroy(ref Savepoints[ii].InSavepoint);
                //SavepointLength = newLength;

                // If this is a release of the outermost savepoint, truncate the sub-journal to zero bytes in size.
                if (op == IPager.SAVEPOINT.RELEASE)
                    if (newLength == 0 && SubJournalFile.Opened)
                    {
                        // Only truncate if it is an in-memory sub-journal.
                        if (SubJournalFile is MemoryVFile)
                        {
                            rc = SubJournalFile.Truncate(0);
                            Debug.Assert(rc == RC.OK);
                        }
                        SubRecords = 0;
                    }
                    // Else this is a rollback operation, playback the specified savepoint. If this is a temp-file, it is possible that the journal file has
                    // not yet been opened. In this case there have been no changes to the database file, so the playback operation can be skipped.
                    else if (UseWal() || JournalFile.Opened)
                    {
                        var savepoint = (newLength == 0 ? (PagerSavepoint)null : Savepoints[newLength - 1]);
                        rc = pagerPlaybackSavepoint(savepoint);
                        Debug.Assert(rc != RC.DONE);
                    }
            }
            return rc;
        }

        public string get_Filename(bool nullIfMemDb)
        {
            return (nullIfMemDb && MemoryDB ? string.Empty : Filename);
        }

        public VSystem get_Vfs()
        {
            return Vfs;
        }

        public VFile get_File()
        {
            return File;
        }

        public string get_Journalname()
        {
            return Journal;
        }

        public bool get_NoSync()
        {
            return NoSync;
        }

#if HAS_CODEC
        public void sqlite3PagerSetCodec(Func<object, object, Pid, int, object> codec, Action<object, int, int> codecSizeChange, Action<object> codecFree, object codecArg)
        {
            if (CodecFree != null) CodecFree(Codec);
            Codec = (MemoryDB ? null : codec);
            CodecSizeChange = codecSizeChange;
            CodecFree = codecFree;
            CodecArg = codecArg;
            pagerReportSize();
        }

        public object sqlite3PagerGetCodec()
        {
            return Codec;
        }
#endif

#if !OMIT_AUTOVACUUM
        public RC Movepage(IPage pg, Pid id, bool isCommit)
        {
            Debug.Assert(pg.Refs > 0);
            Debug.Assert(State == PAGER.WRITER_CACHEMOD ||
                State == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(this));

            // In order to be able to rollback, an in-memory database must journal the page we are moving from.
            var rc = RC.OK;
            if (!MemoryDB)
            {
                rc = Write(pg);
                if (rc != RC.OK) return rc;
            }

            // If the page being moved is dirty and has not been saved by the latest savepoint, then save the current contents of the page into the 
            // sub-journal now. This is required to handle the following scenario:
            //
            //   BEGIN;
            //     <journal page X, then modify it in memory>
            //     SAVEPOINT one;
            //       <Move page X to location Y>
            //     ROLLBACK TO one;
            //
            // If page X were not written to the sub-journal here, it would not be possible to restore its contents when the "ROLLBACK TO one"
            // statement were is processed.
            //
            // subjournalPage() may need to allocate space to store pPg->pgno into one or more savepoint bitvecs. This is the reason this function
            // may return SQLITE_NOMEM.
            if ((pg.Flags & PgHdr.PGHDR.DIRTY) != 0 &&
                subjRequiresPage(pg) &&
                (rc = subjournalPage(pg)) != RC.OK)
                return rc;

            PAGERTRACE("MOVE {0} page {1} (needSync={2}) moves to {3}", PAGERID(this), pg.ID, ((pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0), id);
            Console.WriteLine("MOVE {0} {1} {2}", GetHashCode(), pg.ID, id);

            // If the journal needs to be sync()ed before page pPg->pgno can be written to, store pPg->pgno in local variable needSyncPgno.
            //
            // If the isCommit flag is set, there is no need to remember that the journal needs to be sync()ed before database page pPg->pgno 
            // can be written to. The caller has already promised not to write to it.
            Pid needSyncID = 0; // Old value of pPg.pgno, if sync is required
            if (((pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0) && !isCommit)
            {
                needSyncID = pg.ID;
                Debug.Assert(JournalMode == IPager.JOURNALMODE.OFF || pageInJournal(pg) || pg.ID > DBOrigSize);
                Debug.Assert((pg.Flags & PgHdr.PGHDR.DIRTY) != 0);
            }

            // If the cache contains a page with page-number pgno, remove it from its hash chain. Also, if the PGHDR_NEED_SYNC flag was set for 
            // page pgno before the 'move' operation, it needs to be retained for the page moved there.
            pg.Flags &= ~PgHdr.PGHDR.NEED_SYNC;
            var pgOld = pager_lookup(id); // The page being overwritten.
            Debug.Assert(pgOld == null || pgOld.Refs == 1);
            if (pgOld != null)
            {
                pg.Flags |= (pgOld.Flags & PgHdr.PGHDR.NEED_SYNC);
                if (!MemoryDB)
                    // Do not discard pages from an in-memory database since we might need to rollback later.  Just move the page out of the way.
                    PCache.Move(pgOld, DBSize + 1);
                else
                    PCache.Drop(pgOld);
            }

            var origID = pg.ID; // The original page number
            PCache.Move(pg, id);
            PCache.MakeDirty(pg);

            // For an in-memory database, make sure the original page continues to exist, in case the transaction needs to roll back.  Use pPgOld
            // as the original page since it has already been allocated.
            if (!MemoryDB)
            {
                Debug.Assert(pgOld != null);
                PCache.Move(pgOld, origID);
                Unref(pgOld);
            }

            if (needSyncID != 0)
            {
                // If needSyncPgno is non-zero, then the journal file needs to be sync()ed before any data is written to database file page needSyncPgno.
                // Currently, no such page exists in the page-cache and the "is journaled" bitvec flag has been set. This needs to be remedied by
                // loading the page into the pager-cache and setting the PGHDR_NEED_SYNC flag.
                //
                // If the attempt to load the page into the page-cache fails, (due to a malloc() or IO failure), clear the bit in the pInJournal[]
                // array. Otherwise, if the page is loaded and written again in this transaction, it may be written to the database file before
                // it is synced into the journal file. This way, it may end up in the journal file twice, but that is not a problem.
                PgHdr pgHdr = null;
                rc = Acquire(needSyncID, ref pgHdr, false);
                if (rc != RC.OK)
                {
                    if (needSyncID <= DBOrigSize)
                    {
                        //Debug.Assert(TmpSpace != null);
                        var temp = new uint[TmpSpace.Length];
                        InJournal.Clear(needSyncID, temp);
                    }
                    return rc;
                }
                pgHdr.Flags |= PgHdr.PGHDR.NEED_SYNC;
                PCache.MakeDirty(pgHdr);
                Unref(pgHdr);
            }
            return RC.OK;
        }
#endif

        public static byte[] GetData(IPage pg)
        {
            Debug.Assert(pg.Refs > 0 || pg.Pager.MemoryDB);
            return pg.Data;
        }

        public static T GetExtra<T>(IPage pg)
        {
            return (T)pg.Extra;
        }

        public bool LockingMode(IPager.LOCKINGMODE mode)
        {
            Debug.Assert(mode == IPager.LOCKINGMODE.QUERY ||
                mode == IPager.LOCKINGMODE.NORMAL ||
                mode == IPager.LOCKINGMODE.EXCLUSIVE);
            Debug.Assert(IPager.LOCKINGMODE.QUERY < 0);
            Debug.Assert(IPager.LOCKINGMODE.NORMAL >= 0 && IPager.LOCKINGMODE.EXCLUSIVE >= 0);
            Debug.Assert(ExclusiveMode || !Wal.get_HeapMemory());
            if (mode >= 0 && !TempFile && !Wal.get_HeapMemory())
                ExclusiveMode = (mode != 0);
            return ExclusiveMode;
        }

        public IPager.JOURNALMODE SetJournalMode(IPager.JOURNALMODE mode)
        {
#if DEBUG
            // The print_pager_state() routine is intended to be used by the debugger only.  We invoke it once here to suppress a compiler warning. */
            print_pager_state(this);
#endif

            // The eMode parameter is always valid
            Debug.Assert(mode == IPager.JOURNALMODE.DELETE ||
                mode == IPager.JOURNALMODE.TRUNCATE ||
                mode == IPager.JOURNALMODE.PERSIST ||
                mode == IPager.JOURNALMODE.OFF ||
                mode == IPager.JOURNALMODE.WAL ||
                mode == IPager.JOURNALMODE.JMEMORY);

            // This routine is only called from the OP_JournalMode opcode, and the logic there will never allow a temporary file to be changed to WAL mode.
            Debug.Assert(!TempFile || mode != IPager.JOURNALMODE.WAL);

            // Do allow the journalmode of an in-memory database to be set to anything other than MEMORY or OFF
            var old = JournalMode; // Prior journalmode
            if (MemoryDB)
            {
                Debug.Assert(old == IPager.JOURNALMODE.JMEMORY || old == IPager.JOURNALMODE.OFF);
                if (mode != IPager.JOURNALMODE.JMEMORY && mode != IPager.JOURNALMODE.OFF)
                    mode = old;
            }

            if (mode != old)
            {
                // Change the journal mode.
                Debug.Assert(State != PAGER.ERROR);
                JournalMode = mode;

                // When transistioning from TRUNCATE or PERSIST to any other journal mode except WAL, unless the pager is in locking_mode=exclusive mode,
                // delete the journal file.
                Debug.Assert(((int)IPager.JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)IPager.JOURNALMODE.PERSIST & 5) == 1);
                Debug.Assert(((int)IPager.JOURNALMODE.DELETE & 5) == 0);
                Debug.Assert(((int)IPager.JOURNALMODE.JMEMORY & 5) == 4);
                Debug.Assert(((int)IPager.JOURNALMODE.OFF & 5) == 0);
                Debug.Assert(((int)IPager.JOURNALMODE.WAL & 5) == 5);
                Debug.Assert(File.Opened || ExclusiveMode);
                if (!ExclusiveMode && ((int)old & 5) == 1 && ((int)mode & 1) == 0)
                {
                    // In this case we would like to delete the journal file. If it is not possible, then that is not a problem. Deleting the journal file
                    // here is an optimization only.
                    //
                    // Before deleting the journal file, obtain a RESERVED lock on the database file. This ensures that the journal file is not deleted
                    // while it is in use by some other client.
                    JournalFile.Close();
                    if (Lock >= VFile.LOCK.RESERVED)
                        Vfs.Delete(Journal, false);
                    else
                    {
                        var rc = RC.OK;
                        var state = State;
                        Debug.Assert(state == PAGER.OPEN || state == PAGER.READER);
                        if (state == PAGER.OPEN)
                            rc = SharedLock();
                        if (State == PAGER.READER)
                        {
                            Debug.Assert(rc == RC.OK);
                            rc = pagerLockDb(VFile.LOCK.RESERVED);
                        }
                        if (rc == RC.OK)
                            Vfs.Delete(Journal, false);
                        if (rc == RC.OK && state == PAGER.READER)
                            pagerUnlockDb(VFile.LOCK.SHARED);
                        else if (state == PAGER.OPEN)
                            pager_unlock();
                        Debug.Assert(state == State);
                    }
                }
            }

            // Return the new journal mode
            return JournalMode;
        }

        public IPager.JOURNALMODE GetJournalMode()
        {
            return JournalMode;
        }

        private bool OkToChangeJournalMode()
        {
            Debug.Assert(assert_pager_state(this));
            if (State >= PAGER.WRITER_CACHEMOD) return false;
            return (C._NEVER(JournalFile.Opened && JournalOffset > 0) ? false : true);
        }

        public long SetJournalSizeLimit(long limit)
        {
            if (limit >= -1)
            {
                JournalSizeLimit = limit;
                Wal.Limit(limit);
            }
            return JournalSizeLimit;
        }

        //public IBackup BackupPtr()
        //{
        //    return Backup;
        //}

#if !OMIT_VACUUM
        void ClearCache()
        {
            if (!MemoryDB && !TempFile)
                pager_reset();
        }
#endif

        #endregion

        #region Wal
#if !OMIT_WAL

        public RC Checkpoint(int mode, ref int logs, ref int checkpoints)
        {
            var rc = RC.OK;
            if (Wal)
            {
                rc = Wal.Checkpoint(mode,
                pager.BusyHandler, pager.BusyHandlerArg,
                pager.CheckpointSyncFlags, pager.PageSize, pager.TmpSpace,
                logs, checkpoints);
            }
            return rc;
        }

        public RC WalCallback()
        {
            return Wal.Callback();
        }

        public bool WalSupported()
        {
            return false;
            //const sqlite3_io_methods* pMethods = pPager.fd->pMethods;
            //return pPager.exclusiveMode || (pMethods->iVersion >= 2 && pMethods->xShmMap);
        }

        static RC pagerExclusiveLock(Pager pager)
        {
            Debug.Assert(pager.Lock == VFile.LOCK.SHARED || pager.Lock == VFile.LOCK.EXCLUSIVE);
            var rc = pagerLockDb(pager, VFile.LOCK.EXCLUSIVE);
            if (rc != RC.OK) // If the attempt to grab the exclusive lock failed, release the pending lock that may have been obtained instead.
                pagerUnlockDb(pager, VFile.LOCK.SHARED);
            return rc;
        }

        static RC pagerOpenWal(Pager pager)
        {
            Debug.Assert(pager.Wal == null && !pager.TempFile);
            Debug.Assert(pager.Lock == VFile.LOCK.SHARED || pager.Lock == VFile.LOCK.EXCLUSIVE || pager.NoReadlock);

            // If the pager is already in exclusive-mode, the WAL module will use heap-memory for the wal-index instead of the VFS shared-memory 
            // implementation. Take the exclusive lock now, before opening the WAL file, to make sure this is safe.
            var rc = RC.OK;
            if (pager.ExclusiveMode)
                rc = pagerExclusiveLock(pager);

            // Open the connection to the log file. If this operation fails, (e.g. due to malloc() failure), return an error code.
            if (rc == RC.OK)
                rc = sqlite3WalOpen(pager.Vfs, pager.File, pager.WalName, pager.ExclusiveMode, ref pager.Wal, pager.JournalSizeLimit, ref pager.Wal);

            return rc;
        }

        public RC OpenWal(out bool opened)
        {
            Debug.Assert(assert_pager_state(this));
            Debug.Assert(State == PAGER.OPEN || opened);
            Debug.Assert(State == PAGER.READER || !opened);
            Debug.Assert((TempFile == null && Wal == null));

            var rc = RC.OK;
            if (!TempFile && Wal == null)
            {
                if (!WalSupported()) return RC.CANTOPEN;

                // Close any rollback journal previously open
                JournalFile.Close();

                rc = pagerOpenWal(this);
                if (rc == RC.OK)
                {
                    JournalMode = IPager.JOURNALMODE.WAL;
                    State = PAGER.OPEN;
                }
            }
            else
                opened = true;

            return rc;
        }

        public RC CloseWal()
        {
            Debug.Assert(JournalMode == IPager.JOURNALMODE.WAL);

            // If the log file is not already open, but does exist in the file-system, it may need to be checkpointed before the connection can switch to
            // rollback mode. Open it now so this can happen.
            var rc = RC.OK;
            if (Wal == null)
            {
                rc = pagerLockDb(this, VFile.LOCK.SHARED);
                int logexists = 0;
                if (rc == RC.OK)
                    rc = Vfs.Access(WalName, VSystem.ACCESS.EXISTS, out logexists);
                if (rc == RC.OK && logexists)
                    rc = pagerOpenWal(this);
            }

            // Checkpoint and close the log. Because an EXCLUSIVE lock is held on the database file, the log and log-summary files will be deleted.
            if (rc == RC.OK && Wal != null)
            {
                rc = pagerExclusiveLock(this);
                if (rc == RC.OK)
                {
                    rc = Wal.Close(CheckpointSyncFlags, PageSize, TmpSpace);
                    Wal = null;
                }
            }
            return rc;
        }

#endif
        #endregion

        #region Misc

#if ENABLE_ZIPVFS
		int WalFramesize()
		{
			Debug.Asset(State == PAGER.READER);
			return Wal.Frames();
		}
#endif

#if HAS_CODEC
        public byte[] get_Codec(PgHdr pg)
        {
            byte[] data = null;
            if (CODEC2(ref data, pg.Pager, pg.Data, pg.ID, 6)) return null;
            return data;
        }
#endif

        #endregion
    }
}
