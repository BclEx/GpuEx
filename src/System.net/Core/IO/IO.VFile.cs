using System;
using System.IO;
namespace Core.IO
{
    // sqliteInt.h
    public abstract class VFile
    {
        public static int PENDING_BYTE = 0x40000000;
        public static int RESERVED_BYTE = (PENDING_BYTE + 1);
        public static int SHARED_FIRST = (PENDING_BYTE + 2);
        public static int SHARED_SIZE = 510;

        public enum LOCK : byte
        {
            NO = 0,
            SHARED = 1,
            RESERVED = 2,
            PENDING = 3,
            EXCLUSIVE = 4,
            UNKNOWN = 5,
        }

        // sqlite3.h
        [Flags]
        public enum SYNC : byte
        {
            NORMAL = 0x00002,
            FULL = 0x00003,
            DATAONLY = 0x00010,
            // wal.h
            WAL_TRANSACTIONS = 0x20,    // Sync at the end of each transaction
            WAL_MASK = 0x13,            // Mask off the SQLITE_SYNC_* values
        }

        // sqlite3.h
        public enum FCNTL : uint
        {
            LOCKSTATE = 1,
            GET_LOCKPROXYFILE = 2,
            SET_LOCKPROXYFILE = 3,
            LAST_ERRNO = 4,
            SIZE_HINT = 5,
            CHUNK_SIZE = 6,
            FILE_POINTER = 7,
            SYNC_OMITTED = 8,
            WIN32_AV_RETRY = 9,
            PERSIST_WAL = 10,
            OVERWRITE = 11,
            VFSNAME = 12,
            POWERSAFE_OVERWRITE = 13,
            PRAGMA = 14,
            BUSYHANDLER = 15,
            TEMPFILENAME = 16,
            MMAP_SIZE = 18,
            // os.h
            DB_UNCHANGED = 0xca093fa0,
        }

        // sqlite3.h
        [Flags]
        public enum IOCAP : uint
        {
            ATOMIC = 0x00000001,
            ATOMIC512 = 0x00000002,
            ATOMIC1K = 0x00000004,
            ATOMIC2K = 0x00000008,
            ATOMIC4K = 0x00000010,
            ATOMIC8K = 0x00000020,
            ATOMIC16K = 0x00000040,
            ATOMIC32K = 0x00000080,
            ATOMIC64K = 0x00000100,
            SAFE_APPEND = 0x00000200,
            SEQUENTIAL = 0x00000400,
            UNDELETABLE_WHEN_OPEN = 0x00000800,
            IOCAP_POWERSAFE_OVERWRITE = 0x00001000,
        }

        // sqlite3.h
        [Flags]
        public enum SHM : byte
        {
            UNLOCK = 1,
            LOCK = 2,
            SHARED = 4,
            EXCLUSIVE = 8,
            MAX = 8,
        };

        public byte Type;
        public bool Opened;

        public abstract RC Read(byte[] buffer, int amount, long offset);
        public abstract RC Write(byte[] buffer, int amount, long offset);
        public abstract RC Truncate(long size);
        public abstract RC Close();
        public abstract RC Sync(SYNC flags);
        public abstract RC get_FileSize(out long size);

        public virtual RC Lock(LOCK lock_) { return RC.OK; }
        public virtual RC CheckReservedLock(ref int resOut) { return RC.OK; }
        public virtual RC Unlock(LOCK lock_) { return RC.OK; }
        public virtual RC FileControl(FCNTL op, ref long arg) { return RC.NOTFOUND; }

        public virtual uint get_SectorSize() { return 0; }
        public virtual IOCAP get_DeviceCharacteristics() { return 0; }

        public virtual RC ShmMap(int region, int sizeRegion, bool isWrite, out object pp) { pp = null; return RC.OK; }
        public virtual RC ShmLock(int offset, int count, SHM flags) { return RC.OK; }
        public virtual void ShmBarrier() { }
        public virtual RC ShmUnmap(bool deleteFlag) { return RC.OK; }

        public RC Read4(int offset, out int valueOut)
        {
            uint u32_pRes = 0;
            var rc = Read4(offset, out u32_pRes);
            valueOut = (int)u32_pRes;
            return rc;
        }
        public RC Read4(long offset, out uint valueOut) { return Read4((int)offset, out valueOut); }
        public RC Read4(int offset, out uint valueOut)
        {
            var b = new byte[4];
            var rc = Read(b, b.Length, offset);
            valueOut = (rc == RC.OK ? ConvertEx.Get4(b) : 0);
            return rc;
        }

        public RC Write4(long offset, uint val)
        {
            var ac = new byte[4];
            ConvertEx.Put4(ac, val);
            return Write(ac, 4, offset);
        }

        public RC CloseAndFree()
        {
            RC rc = Close();
            //C._free(ref this);
            return rc;
        }

        // extensions
#if ENABLE_ATOMIC_WRITE
        internal static RC JournalVFileOpen(VSystem vfs, string name, ref VFile file, VSystem.OPEN flags, int bufferLength)
        {
            var p = new JournalVFile();
            if (bufferLength > 0)
                p.Buffer = C._alloc2(bufferLength, true);
            else
            {
                VSystem.OPEN dummy;
                return vfs.Open(name, p, flags, out dummy);
            }
            p.Type = 2;
            p.BufferLength = bufferLength;
            p.Flags = flags;
            p.Journal = name;
            p.Vfs = vfs;
            file = p;
            return RC.OK;
        }
        internal static int JournalVFileSize(VSystem vfs)
        {
            return 0;
        }
        internal static RC JournalVFileCreate(VFile file)
        {
            if (file.Type != 2)
                return RC.OK;
            return ((JournalVFile)file).CreateFile();
        }
        internal static bool HasJournalVFile(VFile file)
        {
            return (file.Type != 2 || ((JournalVFile)file).Real != null);
        }
#else
		internal static int JournalVFileSize(VSystem vfs) { return vfs.SizeOsFile; }
		internal bool HasJournalVFile(VFile file) { return true; }
#endif
        internal static void MemoryVFileOpen(ref VFile file)
        {
            file = new MemoryVFile();
            file.Type = 1;
        }
        internal static bool HasMemoryVFile(VFile file)
        {
            return (file.Type == 1);
        }
        internal static int MemoryVFileSize()
        {
            return 0;
        }

        //#if ENABLE_ATOMIC_WRITE
        //        static int JournalOpen(VSystem vfs, string a, VFile b, int c, int d) { return 0; }
        //        static int JournalSize(VSystem vfs) { return 0; }
        //        static int JournalCreate(VFile v) { return 0; }
        //        static bool JournalExists(VFile v) { return true; }
        //#else
        //        static int JournalSize(VSystem vfs) { return vfs.SizeOsFile; }
        //        static bool JournalExists(VFile v) { return true; }
        //#endif
    }
}