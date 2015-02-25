#if !OMIT_WAL
using System;
using Core.IO;
using Pid = System.UInt32;
using ht_slot = System.UInt16;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace Core
{
    public partial class Wal
    {
#if DEBUG
        internal static bool WalTrace = false;
        internal static void WALTRACE(string x, params object[] args) { if (WalTrace) Console.WriteLine("a:" + string.Format(x, args)); }
#else
        internal static void WALTRACE(string x, params object[] args) { }
#endif

        #region Struct

        const int WAL_MAX_VERSION = 3007000;
        const uint WALINDEX_MAX_VERSION = 3007000;
        const int WAL_WRITE_LOCK = 0;
        const int WAL_ALL_BUT_WRITE = 1;
        const int WAL_CKPT_LOCK = 1;
        const int WAL_RECOVER_LOCK = 2;
        static int WAL_READ_LOCK(int I) { return (3 + (I)); }
        const int WAL_NREADER = ((int)VFile.SHM.MAX - 3);

        class WalIndexHeader
        {
            public uint Version;                // Wal-index version
            public uint Unused;				    // Unused (padding) field
            public uint Change;                 // Counter incremented each transaction
            public bool IsInit;				    // 1 when initialized
            public bool BigEndianChecksum;		// True if checksums in WAL are big-endian
            public ushort SizePage;             // Database page size in bytes. 1==64K
            public uint MaxFrame;               // Index of last valid frame in the WAL
            public uint Pages;                  // Size of database in pages
            public byte[] FrameChecksum = new byte[2 * 4];	// Checksum of last frame in log
            public byte[] Salt = new byte[2 * 4];			// Two salt values copied from WAL header
            public byte[] Checksum = new byte[2 * 4];		// Checksum over all prior fields

            public WalIndexHeader copy()
            {
                var x = (WalIndexHeader)MemberwiseClone();
                x.FrameChecksum = new byte[2 * 4]; Buffer.BlockCopy(FrameChecksum, 0, x.FrameChecksum, 0, FrameChecksum.Length);
                x.Salt = new byte[2 * 4]; Buffer.BlockCopy(Salt, 0, x.Salt, 0, Salt.Length);
                x.Checksum = new byte[2 * 4]; Buffer.BlockCopy(Checksum, 0, x.Checksum, 0, Checksum.Length);
                return x;
            }
        }

        class WalCheckpointInfo
        {
            public uint Backfills; // Number of WAL frames backfilled into DB
            public uint[] ReadMarks = new uint[WAL_NREADER]; // Reader marks

            public WalCheckpointInfo copy()
            {
                var x = (WalCheckpointInfo)MemberwiseClone();
                x.ReadMarks = new uint[WAL_NREADER]; Buffer.BlockCopy(ReadMarks, 0, x.ReadMarks, 0, ReadMarks.Length);
                return x;
            }
        };

        const long READMARK_NOT_USED = 0xffffffff;
        const int WALINDEX_LOCK_OFFSET = (1 * 2 + 1);
        const int WALINDEX_LOCK_RESERVED = 16;
        const int WALINDEX_HDR_SIZE = (WALINDEX_LOCK_OFFSET + WALINDEX_LOCK_RESERVED);
        const int WAL_FRAME_HDRSIZE = 24;
        const int WAL_HDRSIZE = 32;
        const int WAL_MAGIC = 0x377f0682;
        static long walFrameOffset(int frame, long sizePage) { return (WAL_HDRSIZE + ((frame) - 1) * (long)((sizePage) + WAL_FRAME_HDRSIZE)); }

        enum MODE : byte
        {
            NORMAL = 0,
            EXCLUSIVE = 1,
            HEAPMEMORY = 2,
        }

        enum READONLY : byte
        {
            RDWR = 0,			// Normal read/write connection
            RDONLY = 1,			// The WAL file is readonly
            SHM_RDONLY = 2,		// The SHM file is readonly
        }

        VSystem Vfs;			// The VFS used to create pDbFd
        VFile DBFile;				// File handle for the database file
        VFile WalFile;				// File handle for WAL file
        uint Callback_;				// Value to pass to log callback (or 0)
        long MaxWalSize;			// Truncate WAL to this size upon reset
        int SizeFirstBlock;			// Size of first block written to WAL file
        //int nWiData;				// Size of array apWiData
        volatile object[] WiData;	// Pointer to wal-index content in memory
        uint SizePage;              // Database page size
        short ReadLock;				// Which read lock is being held.  -1 for none
        byte SyncFlags;				// Flags to use to sync header writes
        MODE ExclusiveMode_;		// Non-zero if connection is in exclusive mode
        bool WriteLock;				// True if in a write transaction
        bool CheckpointLock;        // True if holding a checkpoint lock
        READONLY ReadOnly;			// WAL_RDWR, WAL_RDONLY, or WAL_SHM_RDONLY
        byte TruncateOnCommit;		// True to truncate WAL file on commit
        byte SyncHeader;			// Fsync the WAL header if true
        byte PadToSectorBoundary;	// Pad transactions out to the next sector
        WalIndexHeader Header;		// Wal-index header for current transaction
        string WalName;		        // Name of WAL file
        uint Checkpoints;			// Checkpoint sequence counter in the wal-header
#if DEBUG
        byte LockError;				// True if a locking error has occurred
#endif

        class WalIterator
        {
            public int Prior;					// Last result returned from the iterator
            public int SegmentsLength;         // Number of entries in aSegment[]
            public class WalSegment
            {
                public int Next;				// Next slot in aIndex[] not yet returned
                public ht_slot[] Indexs;        // i0, i1, i2... such that aPgno[iN] ascend
                public uint[] IDs;				// Array of page numbers.
                public int Entrys;             // Nr. of entries in aPgno[] and aIndex[]
                public int Zero;				// Frame number associated with aPgno[0]
            };
            public WalSegment[] Segments = new WalSegment[1]; // One for every 32KB page in the wal-index
        };

        const int HASHTABLE_NPAGE = 4096;                 // Must be power of 2
        const int HASHTABLE_HASH_1 = 383;                  // Should be prime
        const int HASHTABLE_NSLOT = (HASHTABLE_NPAGE * 2);  // Must be a power of 2
        const int HASHTABLE_NPAGE_ONE = (HASHTABLE_NPAGE - (WALINDEX_HDR_SIZE / sizeof(uint)));
        const int WALINDEX_PGSZ = (sizeof(ht_slot) * HASHTABLE_NSLOT + HASHTABLE_NPAGE * sizeof(uint));

        #endregion

        #region Alloc

        static RC walIndexPage(Wal wal, Pid id, ref object idOut)
        {
            // Enlarge the pWal->apWiData[] array if required
            if (wal.WiData.Length <= id)
            {
                var bytes = (int)(1 * (id + 1));
                var newWiData = SysEx.Realloc<object>(1, wal.WiData, bytes);
                if (newWiData != null)
                {
                    idOut = null;
                    return RC.NOMEM;
                }
                Array.Clear(newWiData, wal.WiData.Length, (int)(id + 1 - wal.WiData.Length));
                wal.WiData = newWiData;
            }

            // Request a pointer to the required page from the VFS
            var rc = RC.OK;
            if (wal.WiData[id] == null)
            {
                if (wal.ExclusiveMode_ == Wal.MODE.HEAPMEMORY)
                {
                    wal.WiData[id] = C._alloc<Pid>(4, WALINDEX_PGSZ, true);
                    if (wal.WiData[id] != null) rc = RC.NOMEM;
                }
                else
                {
                    rc = wal.DBFile.ShmMap((int)id, WALINDEX_PGSZ, (int)wal.WriteLock, wal.WiData, (int)id);
                    if (rc == RC.READONLY)
                    {
                        wal.ReadOnly_ |= READONLY.SHM_RDONLY;
                        rc = RC.OK;
                    }
                }
            }

            idOut = wal.WiData[id];
            Debug.Assert(id == 0 || idOut != null || rc != RC.OK);
            return rc;
        }

        static WalCheckpointInfo walCkptInfo(Wal wal, int idx)
        {
            Debug.Assert(wal.WiData.Length > 0 && wal.WiData[0] != null);
            return (WalCheckpointInfo)wal.WiData[0];
        }

        static WalIndexHeader[] walIndexHeader(Wal wal, int idx)
        {
            Debug.Assert(wal.WiData.Length > 0 && wal.WiData[0] != null);
            return (WalIndexHeader[])wal.WiData[0];
        }

        #endregion

        #region Checksum

        static uint BYTESWAP32(uint x) { return ((((x) & 0x000000FF) << 24) + (((x) & 0x0000FF00) << 8) + (((x) & 0x00FF0000) >> 8) + (((x) & 0xFF000000) >> 24)); }

        static void walChecksumBytes(bool nativeChecksum, WalIndexHeader header, byte[] checksum, byte[] checksumOut)
        {
            int size = Marshal.SizeOf(header);
            var b = new byte[size];
            var p = Marshal.AllocHGlobal(size);
            Marshal.StructureToPtr(header, p, true);
            Marshal.Copy(p, b, 0, size);
            Marshal.FreeHGlobal(p);
            walChecksumBytes(nativeChecksum, b, size, checksum, checksumOut);
        }
        static void walChecksumBytes(bool nativeChecksum, byte[] b, int length, byte[] checksum, byte[] checksumOut)
        {
            uint s1, s2;
            if (checksum != null)
            {
                s1 = ConvertEx.Get4(checksum, 0);
                s2 = ConvertEx.Get4(checksum, 4);
            }
            else
                s1 = s2 = 0;

            Debug.Assert(length >= 8);
            Debug.Assert((length & 0x00000007) == 0);

            uint data = 0;
            uint end = (uint)length * 4;
            if (nativeChecksum)
                do
                {
                    s1 += ConvertEx.Get4(b, data) + s2; data += 4;
                    s2 += ConvertEx.Get4(b, data) + s1; data += 4;
                } while (data < end);
            else
                do
                {
                    s1 += BYTESWAP32(ConvertEx.Get4(b, data)) + s2; data += 4;
                    s2 += BYTESWAP32(ConvertEx.Get4(b, data)) + s1; data += 4;
                } while (data < end);

            ConvertEx.Put4(checksumOut, 0, s1);
            ConvertEx.Put4(checksumOut, 4, s2);
        }

        static void walShmBarrier(Wal wal)
        {
            if (wal.ExclusiveMode_ != MODE.HEAPMEMORY)
                wal.DBFile.ShmBarrier();
        }

        static void walIndexWriteHdr(Wal wal)
        {
            var header = walIndexHeader(wal, 0);

            Debug.Assert(wal.WriteLock != 0);
            wal.Header.IsInit = true;
            wal.Header.Version = WALINDEX_MAX_VERSION;
            walChecksumBytes(true, wal.Header, null, wal.Header.Checksum);
            wal.Header = header[1].copy();
            walShmBarrier(wal);
            wal.Header = header[0].copy();
        }

        static void walEncodeFrame(Wal wal, Pid id, uint truncate, byte[] data, byte[] frame)
        {
            var checksum = wal.Header.FrameChecksum;
            Debug.Assert(WAL_FRAME_HDRSIZE == 24);
            ConvertEx.Put4(frame, 0, id);
            ConvertEx.Put4(frame, 4, truncate);
            Buffer.BlockCopy(frame, 8, wal.Header.Salt, 0, 8);

            bool nativeChecksum = true; // (wal.Header.BigEndianChecksum == TYPE_BIGENDIAN); // True for native byte-order checksums
            walChecksumBytes(nativeChecksum, frame, 8, checksum, checksum);
            walChecksumBytes(nativeChecksum, data, (int)wal.SizePage, checksum, checksum);

            ConvertEx.Put4(frame, 16, checksum[0]);
            ConvertEx.Put4(frame, 20, checksum[1]);
        }

        static bool walDecodeFrame(Wal wal, Pid idOut, uint truncateOut, byte[] data, byte[] frame)
        {
            var checksum = wal.Header.FrameChecksum;
            Debug.Assert(WAL_FRAME_HDRSIZE == 24);

            // A frame is only valid if the salt values in the frame-header match the salt values in the wal-header. 
            var testFrame = new byte[8];
            Buffer.BlockCopy(frame, 8, testFrame, 0, 8);
            if (Enumerable.SequenceEqual(wal.Header.Salt, testFrame))
                return false;

            // A frame is only valid if the page number is creater than zero.
            Pid id = ConvertEx.Get4(frame, 0); // Page number of the frame
            if (id == 0)
                return false;

            // A frame is only valid if a checksum of the WAL header, all prior frams, the first 16 bytes of this frame-header, 
            // and the frame-data matches the checksum in the last 8 bytes of this frame-header.
            bool nativeChecksum = true; // (wal.Header.BigEndianChecksum == TYPE_BIGENDIAN); // True for native byte-order checksums
            walChecksumBytes(nativeChecksum, frame, 8, checksum, checksum);
            walChecksumBytes(nativeChecksum, data, (int)wal.SizePage, checksum, checksum);
            if (checksum[0] != ConvertEx.Get4(frame, 16) || checksum[1] != ConvertEx.Get4(frame, 20)) // Checksum failed.
                return false;

            // If we reach this point, the frame is valid.  Return the page number and the new database size.
            idOut = id;
            truncateOut = ConvertEx.Get4(frame, 4);
            return true;
        }


        #endregion

        #region Lock

        static string walLockName(int lockIdx)
        {
            if (lockIdx == WAL_WRITE_LOCK)
                return "WRITE-LOCK";
            else if (lockIdx == WAL_CKPT_LOCK)
                return "CKPT-LOCK";
            else if (lockIdx == WAL_RECOVER_LOCK)
                return "RECOVER-LOCK";
            else
                return string.Format("READ-LOCK[{0}]", lockIdx - WAL_READ_LOCK(0));
        }

        static RC walLockShared(Wal wal, int lockIdx)
        {
            if (wal.ExclusiveMode_ != MODE.NORMAL) return RC.OK;
            RC rc = wal.DBFile.ShmLock(lockIdx, 1, VFile.SHM.LOCK | VFile.SHM.SHARED);
            WALTRACE("WAL%p: acquire SHARED-%s %s\n", wal, walLockName(lockIdx), rc != 0 ? "failed" : "ok");
            return rc;
        }

        static void walUnlockShared(Wal wal, int lockIdx)
        {
            if (wal.ExclusiveMode_ != MODE.NORMAL) return;
            wal.DBFile.ShmLock(lockIdx, 1, VFile.SHM.UNLOCK | VFile.SHM.SHARED);
            WALTRACE("WAL%p: release SHARED-%s\n", wal, walLockName(lockIdx));
        }

        static RC walLockExclusive(Wal wal, int lockIdx, int n)
        {
            if (wal.ExclusiveMode_ != MODE.NORMAL) return RC.OK;
            RC rc = wal.DBFile.ShmLock(lockIdx, n, VFile.SHM.LOCK | VFile.SHM.EXCLUSIVE);
            WALTRACE("WAL%p: acquire EXCLUSIVE-%s cnt=%d %s\n", wal, walLockName(lockIdx), n, rc != 0 ? "failed" : "ok");
            return rc;
        }

        static void walUnlockExclusive(Wal wal, int lockIdx, int n)
        {
            if (wal.ExclusiveMode_ != MODE.NORMAL) return;
            wal.DBFile.ShmLock(lockIdx, n, VFile.SHM.UNLOCK | VFile.SHM.EXCLUSIVE);
            WALTRACE("WAL%p: release EXCLUSIVE-%s cnt=%d\n", wal, walLockName(lockIdx), n);
        }

        #endregion

        #region Hash

        static int walHash(uint id)
        {
            Debug.Assert(id > 0);
            Debug.Assert((HASHTABLE_NSLOT & (HASHTABLE_NSLOT - 1)) == 0);
            return (int)(id * HASHTABLE_HASH_1) & (HASHTABLE_NSLOT - 1);
        }

        static int walNextHash(int priorHash)
        {
            return (priorHash + 1) & (HASHTABLE_NSLOT - 1);
        }

        static RC walHashGet(Wal wal, Pid id, ht_slot[] hashOut, Pid[] idsOut, out uint zeroOut)
        {
            object ids;
            RC rc = walIndexPage(wal, id, ref ids);
            Debug.Assert(rc == RC.OK || id > 0);

            if (rc == RC.OK)
            {
                Pid zero;
                var hash = (ht_slot)ids[HASHTABLE_NPAGE];
                if (id == 0)
                {
                    ids = &ids[WALINDEX_HDR_SIZE / sizeof(Pid)];
                    zero = 0;
                }
                else
                    zero = (uint)(HASHTABLE_NPAGE_ONE + (id - 1) * HASHTABLE_NPAGE);

                idsOut = &ids[-1];
                hashOut = hash;
                zeroOut = zero;
            }
            return rc;
        }

        static int walFramePage(uint frame)
        {
            int hash = (int)(frame + HASHTABLE_NPAGE - HASHTABLE_NPAGE_ONE - 1) / HASHTABLE_NPAGE;
            Debug.Assert((hash == 0 || frame > HASHTABLE_NPAGE_ONE) &&
                (hash >= 1 || frame <= HASHTABLE_NPAGE_ONE) &&
                (hash <= 1 || frame > (HASHTABLE_NPAGE_ONE + HASHTABLE_NPAGE)) &&
                (hash >= 2 || frame <= HASHTABLE_NPAGE_ONE + HASHTABLE_NPAGE) &&
                (hash <= 2 || frame > (HASHTABLE_NPAGE_ONE + 2 * HASHTABLE_NPAGE)));
            return hash;
        }

        static uint walFramePgno(Wal wal, uint frame)
        {
            int hash = walFramePage(frame);
            if (hash == 0)
                return wal.WiData[0][WALINDEX_HDR_SIZE / sizeof(uint) + frame - 1];
            return wal.WiData[hash][(frame - 1 - HASHTABLE_NPAGE_ONE) % HASHTABLE_NPAGE];
        }

        static void walCleanupHash(Wal wal)
        {
            Debug.Assert(wal.WriteLock);

            if (wal.Header.MaxFrame == 0) return;

            // Obtain pointers to the hash-table and page-number array containing the entry that corresponds to frame pWal->hdr.mxFrame. It is guaranteed
            // that the page said hash-table and array reside on is already mapped.
            Debug.Assert(wal.WiData.Length > walFramePage(wal.Header.MaxFrame));
            Debug.Assert(wal.WiData[walFramePage(wal.Header.MaxFrame)] != 0);
            ht_slot[] hash = null; // Pointer to hash table to clear
            Pid[] ids = null; // Page number array for hash table
            int zero = 0; // frame == (aHash[x]+iZero)
            walHashGet(wal, walFramePage(wal.Header.MaxFrame), ref hash, ref ids, ref zero);

            // Zero all hash-table entries that correspond to frame numbers greater than pWal->hdr.mxFrame.
            int limit = wal.Header.MaxFrame - zero; // Zero values greater than this
            Debug.Assert(limit > 0);
            for (int i = 0; i < HASHTABLE_NSLOT; i++)
                if (hash[i] > limit)
                    hash[i] = 0;

            // Zero the entries in the aPgno array that correspond to frames with frame numbers greater than pWal->hdr.mxFrame. 
            int bytes = (int)((char*)hash - (char*)&ids[limit + 1]); // Number of bytes to zero in aPgno[]
            _memset((void*)&ids[limit + 1], 0, bytes);

#if ENABLE_EXPENSIVE_ASSERT
		// Verify that the every entry in the mapping region is still reachable via the hash table even after the cleanup.
		int key; // Hash key
		if (limit)
			for (int i = 1; i <= limit; i++)
			{
				for (key = walHash(ids[i]); hash[key]; key = walNextHash(key))
					if (hash[key] == i) break;
				_assert(hash[key] == i);
			}
#endif
        }

        #endregion
    }
}
#endif