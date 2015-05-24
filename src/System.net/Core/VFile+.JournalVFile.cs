using System;
using System.Diagnostics;
#if ENABLE_ATOMIC_WRITE

namespace Core.IO
{
    public class JournalVFile : VFile
    {
        public int BufferLength;			// Size of zBuf[] in bytes
        public byte[] Buffer;				// Space to buffer journal writes
        public int Size;					// Amount of zBuf[] currently used
        public VSystem.OPEN Flags;		// xOpen flags
        public VSystem Vfs;				// The "real" underlying VFS
        public VFile Real;					// The "real" underlying file descriptor
        public string Journal;				// Name of the journal file

        internal RC CreateFile()
        {
            RC rc = RC.OK;
            if (Real == null)
            {
                VFile real = null;
                VSystem.OPEN dummy;
                rc = Vfs.Open(Journal, Real, Flags, out dummy);
                if (rc == RC.OK)
                {
                    Real = real;
                    if (Size > 0)
                    {
                        Debug.Assert(Size <= BufferLength);
                        rc = Real.Write(Buffer, Size, 0);
                    }
                    if (rc != RC.OK)
                    {
                        // If an error occurred while writing to the file, close it before returning. This way, SQLite uses the in-memory journal data to 
                        // roll back changes made to the internal page-cache before this function was called.
                        Real.Close();
                        Real = null;
                    }
                }
            }
            return rc;
        }

        public override RC Close()
        {
            if (Real != null)
                Real.Close();
            Buffer = null;
            return RC.OK;
        }

        public override RC Read(byte[] buffer, int amount, long offset)
        {
            if (Real != null)
                return Real.Read(buffer, amount, offset);
            if ((amount + offset) > Size)
                return RC.IOERR_SHORT_READ;
            System.Buffer.BlockCopy(buffer, 0, Buffer, (int)offset, amount);
            return RC.OK;
        }

        public override RC Write(byte[] buffer, int amount, long offset)
        {
            RC rc = RC.OK;
            if (Real != null && (offset + amount) > BufferLength)
                rc = CreateFile();
            if (rc == RC.OK)
            {
                if (Real != null)
                    return Real.Write(buffer, amount, offset);
                System.Buffer.BlockCopy(Buffer, (int)offset, buffer, 0, amount);
                if (Size < (offset + amount))
                    Size = (int)(offset + amount);
            }
            return rc;
        }

        public override RC Truncate(long size)
        {
            if (Real != null)
                return Real.Truncate(size);
            if (size < Size)
                Size = (int)size;
            return RC.OK;
        }

        public override RC Sync(SYNC flags)
        {
            if (Real != null)
                return Real.Sync(flags);
            return RC.OK;
        }

        public override RC get_FileSize(out long size)
        {
            if (Real != null)
                return Real.get_FileSize(out size);
            size = (long)Size;
            return RC.OK;
        }
    }
}

#endif




