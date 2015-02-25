using Core.IO;
using System;
using System.Diagnostics;

namespace Core
{
    public partial class Vdbe
    {
        #region Structs

        public class VdbeSorter
        {
            public long WriteOffset;			// Current write offset within file pTemp1
            public long ReadOffset;				// Current read offset within file pTemp1
            public int InMemory;				// Current size of pRecord list as PMA
            public int PMAs;                    // Number of PMAs stored in pTemp1
            public int MinPmaSize;              // Minimum PMA size, in bytes
            public int MaxPmaSize;              // Maximum PMA size, in bytes.  0==no limit
            public VdbeSorterIter Iters;        // Array of iterators to merge
            public array_t<int> Trees;          // Current state of incremental merge / Used size of aTree/aIter (power of 2)
            public VFile Temp1;					// PMA file 1
            public SorterRecord Record;			// Head of in-memory record list
            public UnpackedRecord Unpacked;		// Used to unpack keys
        }

        public class VdbeSorterIter
        {
            public long ReadOffset;				// Current read offset
            public long Eof;					// 1 byte past EOF for this iterator
            public VFile File;					// File iterator is reading from
            public array_t<byte> Alloc;			// Allocated space
            public array_t<byte> Key;			// Pointer to current key
            public array_t<byte> Buffer;		// Current read buffer

            internal void _memset()
            {
            }
        }

        public class FileWriter
        {
            public int FWErr;					// Non-zero if in an error state
            public array_t<byte> Buffer;		// Pointer to write buffer
            public int BufStart;				// First byte of buffer to write
            public int BufEnd;					// Last byte of buffer to write
            public long WriteOffset;			// Offset of start of buffer in file
            public VFile File;					// File to write to
        }

        public class SorterRecord
        {
            public object P;
            public int N;
            public SorterRecord Next;
        }

        public const int SORTER_MIN_WORKING = 10; // Minimum allowable value for the VdbeSorter.nWorking variable
        public const int SORTER_MAX_MERGE_COUNT = 16; // Maximum number of segments to merge in a single pass.

        #endregion

        #region Sorter Iter

        static void VdbeSorterIterZero(Context ctx, VdbeSorterIter iter)
        {
            C._tagfree(ctx, ref iter.Alloc.data);
            C._tagfree(ctx, ref iter.Buffer.data);
            iter._memset();
        }

        static RC VdbeSorterIterRead(Context ctx, VdbeSorterIter p, int bytes, ref byte[] out_)
        {
            Debug.Assert(p.Buffer.data != null);

            // If there is no more data to be read from the buffer, read the next p->nBuffer bytes of data from the file into it. Or, if there are less
            // than p->nBuffer bytes remaining in the PMA, read all remaining data.
            int bufferIdx = (int)(p.ReadOffset % p.Buffer.length); // Offset within buffer to read from
            if (bufferIdx == 0)
            {
                // Determine how many bytes of data to read.
                int read = (int)((p.Eof - p.ReadOffset) > p.Buffer.length ? p.Buffer.length : p.Eof - p.ReadOffset); // Bytes to read from disk
                Debug.Assert(read > 0);

                // Read data from the file. Return early if an error occurs.
                RC rc = p.File.Read(p.Buffer, read, p.ReadOffset);
                Debug.Assert(rc != RC.IOERR_SHORT_READ);
                if (rc != RC.OK) return rc;
            }
            int avail = p.Buffer.length - bufferIdx; // Bytes of data available in buffer

            if (bytes <= avail)
            {
                // The requested data is available in the in-memory buffer. In this case there is no need to make a copy of the data, just return a 
                // pointer into the buffer to the caller.
                out_ = p.Buffer[bufferIdx];
                p.ReadOffset += bytes;
            }
            // The requested data is not all available in the in-memory buffer. In this case, allocate space at p->aAlloc[] to copy the requested
            // range into. Then return a copy of pointer p->aAlloc to the caller.
            else
            {
                // Extend the p->aAlloc[] allocation if required.
                if (p.Alloc.length < bytes)
                {
                    int newSize = p.Alloc.length * 2;
                    while (bytes > newSize) newSize = newSize * 2;
                    p.Alloc.data = (byte[])C._tagrealloc(ctx, 0, p.Alloc.data, newSize);
                    if (p.Alloc.data == null) return RC.NOMEM;
                    p.Alloc.length = newSize;
                }

                // Copy as much data as is available in the buffer into the start of p->aAlloc[].
                C._memcpy(p.Alloc.data, p.Buffer[bufferIdx], avail);
                p.ReadOffset += avail;

                // The following loop copies up to p->nBuffer bytes per iteration into the p->aAlloc[] buffer.
                int remaining = bytes - avail; // Bytes remaining to copy
                while (remaining > 0)
                {
                    int copy = remaining; // Number of bytes to copy
                    if (remaining > p.Buffer.length) copy = p.Buffer.length;
                    byte[] next; // Pointer to buffer to copy data from
                    RC rc = VdbeSorterIterRead(ctx, p, copy, ref next);
                    if (rc != RC.OK) return rc;
                    Debug.Assert(next != p.Alloc);
                    C._memcpy(p.Alloc[bytes - remaining], next, copy);
                    remaining -= copy;
                }

                out_ = p.Alloc;
            }
            return RC.OK;
        }

        static RC VdbeSorterIterVarint(Context ctx, VdbeSorterIter p, out ulong out_)
        {
            int bufferIdx = (int)(p.ReadOffset % p.Buffer.length);
            if (bufferIdx != 0 && (p.Buffer.length - bufferIdx) >= 9)
                p.ReadOffset += ConvertEx.GetVarint(p.Buffer[bufferIdx], out out_);
            else
            {
                byte[] varint = new byte[16]; byte[] a;
                int i = 0;
                do
                {
                    RC rc = VdbeSorterIterRead(ctx, p, 1, ref a);
                    if (rc != 0) return rc;
                    varint[(i++) & 0xf] = a[0];
                } while ((a[0] & 0x80) != 0);
                ConvertEx.GetVarint(varint, out out_);
            }
            return RC.OK;
        }

        static RC VdbeSorterIterNext(Context ctx, VdbeSorterIter iter)
        {
            if (iter.ReadOffset >= iter.Eof)
            {
                VdbeSorterIterZero(ctx, iter); // This is an EOF condition
                return RC.OK;
            }
            ulong recordSize; // Size of record in bytes
            RC rc = VdbeSorterIterVarint(ctx, iter, ref recordSize);
            if (rc == RC.OK)
            {
                iter.Key.length = (int)recordSize;
                rc = VdbeSorterIterRead(ctx, iter, (int)recordSize, ref iter.Key.data);
            }
            return rc;
        }

        static RC VdbeSorterIterInit(Context ctx, VdbeSorter sorter, long start, VdbeSorterIter iter, ref long bytes)
        {
            Debug.Assert(sorter.WriteOffset > start);
            Debug.Assert(iter.Alloc.data == null);
            Debug.Assert(iter.Buffer.data == null);
            int bufferLength = ctx.DBs[0].Bt.GetPageSize();
            iter.File = sorter.Temp1;
            iter.ReadOffset = start;
            iter.Alloc.length = 128;
            iter.Alloc.data = (byte[])C._tagalloc(ctx, iter.Alloc.length);
            iter.Buffer.length = bufferLength;
            iter.Buffer.data = (byte[])C._tagalloc(ctx, bufferLength);
            RC rc = RC.OK;
            if (iter.Buffer.data == null)
                rc = RC.NOMEM;
            else
            {
                int bufferIdx = start % bufferLength;
                if (bufferIdx != 0)
                {
                    int read = bufferLength - bufferIdx;
                    if ((start + read) > sorter.WriteOffset)
                        read = (int)(sorter.WriteOffset - start);
                    rc = sorter.Temp1.Read(iter.Buffer[bufferIdx], read, start);
                    Debug.Assert(rc != RC.IOERR_SHORT_READ);
                }
                if (rc == RC.OK)
                {
                    iter.Eof = sorter.WriteOffset;
                    ulong bytes2; // Size of PMA in bytes
                    rc = VdbeSorterIterVarint(ctx, iter, ref bytes2);
                    iter.Eof = iter.ReadOffset + bytes2;
                    bytes += bytes2;
                }
            }
            if (rc == RC.OK)
                rc = VdbeSorterIterNext(ctx, iter);
            return rc;
        }

        #endregion

        #region Sorter Compare/Merge

        static void VdbeSorterCompare(VdbeCursor cursor, bool omitRowid, string key1, int key1Length, string key2, int key2Length, ref int out_)
        {
            KeyInfo keyInfo = cursor.KeyInfo;
            VdbeSorter sorter = (VdbeSorter)cursor.Sorter;
            UnpackedRecord r2 = sorter.Unpacked;
            if (key2 != null)
                Vdbe.RecordUnpack(keyInfo, key2Length, key2, r2);
            if (omitRowid)
            {
                r2.Fields = keyInfo.Fields;
                Debug.Assert(r2.Fields > 0);
                for (int i = 0; i < r2.Fields; i++)
                    if ((r2.Mems[i].Flags & MEM.Null) != 0)
                    {
                        out_ = -1;
                        return;
                    }
                r2.Flags |= UNPACKED.PREFIX_MATCH;
            }
            out_ = Vdbe.RecordCompare(key1Length, key1, r2);
        }

        static RC VdbeSorterDoCompare(VdbeCursor cursor, int idx)
        {
            VdbeSorter sorter = cursor.Sorter;
            Debug.Assert(idx < sorter.Trees.length && idx > 0);

            int i1;
            int i2;
            if (idx >= (sorter.Trees.length / 2))
            {
                i1 = (idx - sorter.Trees.length / 2) * 2;
                i2 = i1 + 1;
            }
            else
            {
                i1 = sorter.Trees[idx * 2];
                i2 = sorter.Trees[idx * 2 + 1];
            }
            VdbeSorterIter p1 = sorter.Iters[i1];
            VdbeSorterIter p2 = sorter.Iters[i2];

            int i;
            if (p1.File == null)
                i = i2;
            else if (p2.File == null)
                i = i1;
            else
            {
                Debug.Assert(sorter.Unpacked != null); // allocated in vdbeSorterMerge()
                int r;
                VdbeSorterCompare(cursor, 0, p1.Key, p1.Key.length, p2.Key, p2.Key.length, ref r);
                i = (r <= 0 ? i1 : i2);
            }
            sorter.Trees[idx] = i;
            return RC.OK;
        }

        public RC SorterInit(Context ctx, VdbeCursor cursor)
        {
            Debug.Assert(cursor.KeyInfo != null && cursor.Bt == null);
            VdbeSorter sorter; // The new sorter
            cursor.Sorter = sorter = (VdbeSorter)C._tagalloc(ctx, sizeof(VdbeSorter));
            if (sorter == null)
                return RC.NOMEM;

            object d;
            sorter.Unpacked = AllocUnpackedRecord(cursor.KeyInfo, 0, 0, ref d);
            if (sorter.Unpacked == null) return RC.NOMEM;
            Debug.Assert(sorter.Unpacked == (UnpackedRecord)d);

            if (ctx.TempInMemory() == null)
            {
                int pageSize = ctx.DBs[0].Bt.GetPageSize(); // Page size of main database
                sorter.MinPmaSize = SORTER_MIN_WORKING * pageSize;
                int maxCache = ctx.DBs[0].Schema.CacheSize; // Cache size
                if (maxCache < SORTER_MIN_WORKING) maxCache = SORTER_MIN_WORKING;
                sorter.MaxPmaSize = maxCache * pageSize;
            }
            return RC.OK;
        }

        static void VdbeSorterRecordFree(Context ctx, SorterRecord record)
        {
            SorterRecord next;
            for (SorterRecord p = record; p != null; p = next)
            {
                next = p.Next;
                C._tagfree(ctx, ref p);
            }
        }

        public void SorterClose(Context ctx, VdbeCursor cursor)
        {
            VdbeSorter sorter = cursor.Sorter;
            if (sorter != null)
            {
                if (sorter.Iters != null)
                {
                    for (int i = 0; i < sorter.Trees.length; i++)
                        VdbeSorterIterZero(ctx, sorter.Iters[i]);
                    C._tagfree(ctx, ref sorter.Iters);
                }
                if (sorter.Temp1 != null)
                    sorter.Temp1.CloseAndFree();
                VdbeSorterRecordFree(ctx, sorter.Record);
                C._tagfree(ctx, sorter.Unpacked);
                C._tagfree(ctx, sorter);
                cursor.Sorter = null;
            }
        }

        static RC VdbeSorterOpenTempFile(Context ctx, ref VFile file)
        {
            VSystem.OPEN outFlags;
            return ctx.Vfs.OpenAndAlloc(null, file, (VSystem.OPEN)(VSystem.OPEN_TEMP_JOURNAL | VSystem.OPEN_READWRITE | VSystem.OPEN_CREATE | VSystem.OPEN_EXCLUSIVE | VSystem.OPEN_DELETEONCLOSE), ref outFlags);
        }

        static void VdbeSorterMerge(VdbeCursor cursor, SorterRecord p1, SorterRecord p2, ref SorterRecord out_)
        {
            SorterRecord result = null;
            SorterRecord pp = result;
            object p2P = (p2 != null ? p2.P : null);
            while (p1 != null && p2 != null)
            {
                int r;
                VdbeSorterCompare(cursor, false, p1.P, p1.N, p2P, p2.N, ref r);
                if (r <= 0)
                {
                    pp = p1;
                    pp = p1.Next;
                    p1 = p1.Next;
                    p2P = null;
                }
                else
                {
                    pp = p2;
                    pp = p2.Next;
                    p2 = p2.Next;
                    if (p2 != null) break;
                    p2P = p2.P;
                }
            }
            pp = (p1 != null ? p1 : p2);
            out_ = result;
        }

        static RC VdbeSorterSort(VdbeCursor cursor)
        {
            SorterRecord slots = new SorterRecord[64];
            if (slots == null)
                return RC.NOMEM;
            VdbeSorter sorter = cursor.Sorter;
            SorterRecord p = sorter.Record;
            int i;
            while (p != null)
            {
                SorterRecord next = p.Next;
                p.Next = null;
                for (i = 0; slots[i] != null; i++)
                {
                    VdbeSorterMerge(cursor, p, slots[i], ref p);
                    slots[i] = null;
                }
                slots[i] = p;
                p = next;
            }
            p = null;
            for (i = 0; i < 64; i++)
                VdbeSorterMerge(cursor, p, slots[i], ref p);
            sorter.Record = p;
            C._free(ref slots);
            return RC.OK;
        }

        #endregion

        #region FileWriter

        static void FileWriterInit(Context ctx, VFile file, FileWriter p, long start)
        {
            p._memset();
            int pageSize = ctx.DBs[0].Bt.GetPageSize();
            p.Buffer.data = (byte[])C._tagalloc(ctx, pageSize);
            if (p.Buffer.data == null)
                p.FWErr = RC.NOMEM;
            else
            {
                p.BufEnd = p.BufStart = (start % pageSize);
                p.WriteOffset = start - p.BufStart;
                p.Buffer.length = pageSize;
                p.File = file;
            }
        }

        static void FileWriterWrite(FileWriter p, byte[] data, int dataLength)
        {
            int remain = dataLength;
            while (remain > 0 && p.FWErr == 0)
            {
                int copy = remain;
                if (copy > (p.Buffer.length - p.BufEnd))
                    copy = p.Buffer.length - p.BufEnd;
                C._memcpy(p.Buffer[p.BufEnd], data[dataLength - remain], copy);
                p.BufEnd += copy;
                if (p.BufEnd == p.Buffer.length)
                {
                    p.FWErr = p.File.Write(p.Buffer[p.BufStart], p.BufEnd - p.BufStart, p.WriteOffset + p.BufStart);
                    p.BufStart = p.BufEnd = 0;
                    p.WriteOffset += p.Buffer.length;
                }
                Debug.Assert(p.BufEnd < p.Buffer.length);
                remain -= copy;
            }
        }

        static RC FileWriterFinish(Context ctx, FileWriter p, ref long eof)
        {
            if (p.FWErr == 0 && C._ALWAYS(p.Buffer) && p.BufEnd > p.BufStart)
                p.FWErr = p.File.Write(p.Buffer[p.BufStart], p.BufEnd - p.BufStart, p.WriteOffset + p.BufStart);
            eof = (p.WriteOffset + p.BufEnd);
            C._tagfree(ctx, ref p.Buffer.data);
            RC rc = (RC)p.FWErr;
            p._memset();
            return rc;
        }

        static void FileWriterWriteVarint(FileWriter p, ulong value)
        {
            byte[] bytes = new byte[10];
            int length = ConvertEx.PutVarint(bytes, value);
            FileWriterWrite(p, bytes, length);
        }

        static RC VdbeSorterListToPMA(Context ctx, VdbeCursor cursor)
        {
            VdbeSorter sorter = cursor.Sorter;
            FileWriter writer;
            writer._memset();
            if (sorter.InMemory == 0)
            {
                Debug.Assert(sorter.Record == null);
                return RC.OK;
            }
            RC rc = VdbeSorterSort(cursor);
            // If the first temporary PMA file has not been opened, open it now.
            if (rc == RC.OK && sorter.Temp1 == null)
            {
                rc = VdbeSorterOpenTempFile(ctx, ref sorter.Temp1);
                Debug.Assert(rc != RC.OK || sorter.Temp1 != null);
                Debug.Assert(sorter.WriteOffset == 0);
                Debug.Assert(sorter.PMAs == 0);
            }
            if (rc == RC.OK)
            {
                FileWriterInit(ctx, sorter.Temp1, writer, sorter.WriteOffset);
                sorter.PMAs++;
                FileWriterWriteVarint(writer, sorter.InMemory);
                SorterRecord p;
                SorterRecord next = null;
                for (p = sorter.Record; p; p = next)
                {
                    next = p.Next;
                    FileWriterWriteVarint(writer, p.N);
                    FileWriterWrite(writer, (byte[])p.P, p.N);
                    C._tagfree(ctx, ref p);
                }
                sorter.Record = p;
                rc = FileWriterFinish(ctx, writer, sorter.WriteOffset);
            }
            return rc;
        }

        public RC SorterWrite(Context ctx, VdbeCursor cursor, Mem mem)
        {
            VdbeSorter sorter = cursor.Sorter;
            Debug.Assert(sorter != null);
            sorter.InMemory += ConvertEx.GetVarintLength(mem.N) + mem.N;
            SorterRecord newRecord = (SorterRecord)C._tagalloc(ctx, mem.N + sizeof(SorterRecord)); // New list element
            RC rc = RC.OK;
            if (newRecord == null)
                rc = RC.NOMEM;
            else
            {
                newRecord.P = newRecord[1];
                C._memcpy(newRecord.P, mem.Z, mem.N);
                newRecord.N = mem.N;
                newRecord.Next = sorter.Record;
                sorter.Record = newRecord;
            }
            // See if the contents of the sorter should now be written out. They are written out when either of the following are true:
            //   * The total memory allocated for the in-memory list is greater than (page-size * cache-size), or
            //   * The total memory allocated for the in-memory list is greater than (page-size * 10) and sqlite3HeapNearlyFull() returns true.
            if (rc == RC.OK && sorter.MaxPmaSize > 0 && ((sorter.InMemory > sorter.MaxPmaSize) || (sorter.InMemory > sorter.MaxPmaSize && C._heapnearlyfull())))
            {
#if DEBUG
                long expect = sorter.WriteOffset + ConvertEx.GetVarintLength(sorter.InMemory) + sorter.InMemory;
#endif
                rc = VdbeSorterListToPMA(ctx, cursor);
                sorter.InMemory = 0;
                Debug.Assert(rc != RC.OK || expect == sorter.WriteOffset);
            }
            return rc;
        }

        static RC VdbeSorterInitMerge(Context ctx, VdbeCursor cursor, ref long bytes)
        {
            VdbeSorter sorter = cursor, Sorter;
            long bytes2 = 0; // Total bytes in all opened PMAs
            RC rc = RC.OK;
            // Initialize the iterators.
            int i; // Used to iterator through aIter[]
            for (i = 0; i < SORTER_MAX_MERGE_COUNT; i++)
            {
                VdbeSorterIter iter = sorter.Iters[i];
                rc = VdbeSorterIterInit(ctx, sorter, sorter.ReadOffset, iter, ref bytes2);
                sorter.ReadOffset = iter.Eof;
                Debug.Assert(rc != RC.OK || sorter.ReadOffset <= sorter.WriteOffset);
                if (rc != RC.OK || sorter.ReadOffset >= sorter.WriteOffset) break;
            }
            // Initialize the aTree[] array.
            for (i = sorter.Trees.length - 1; rc == RC.OK && i > 0; i--)
                rc = VdbeSorterDoCompare(cursor, i);
            bytes = bytes2;
            return rc;
        }

        public RC SorterRewind(Context ctx, VdbeCursor cursor, ref bool eof)
        {
            VdbeSorter sorter = cursor.Sorter;
            Debug.Assert(sorter != null);

            // If no data has been written to disk, then do not do so now. Instead, sort the VdbeSorter.pRecord list. The vdbe layer will read data directly
            // from the in-memory list.
            if (sorter.PMAs == 0)
            {
                eof = (sorter.Record == null);
                Debug.Assert(sorter.Trees.data = null);
                return VdbeSorterSort(cursor);
            }

            // Write the current in-memory list to a PMA.
            RC rc = VdbeSorterListToPMA(ctx, cursor);
            if (rc != RC.OK) return rc;

            // Allocate space for aIter[] and aTree[].
            int iters = sorter.PMAs; // Number of iterators used
            if (iters > SORTER_MAX_MERGE_COUNT) iters = SORTER_MAX_MERGE_COUNT;
            Debug.Assert(iters > 0);
            int n = 2; while (n < iters) n += n; // Power of 2 >= iters
            int bytes = n * (sizeof(int) + sizeof(VdbeSorterIter)); // Bytes of space required for aIter/aTree
            sorter.Iters = (VdbeSorterIter)C._tagalloc(ctx, bytes);
            if (sorter.Iters == null) return RC.NOMEM;
            sorter.Trees = sorter.Iters[n];
            sorter.Trees.length = n;

            int newIdx; // Index of new, merged, PMA
            VFile temp2 = null; // Second temp file to use
            long write2 = 0; // Write offset for pTemp2
            do
            {
                for (newIdx = 0; rc == RC.OK && newIdx * SORTER_MAX_MERGE_COUNT < sorter.PMAs; newIdx++)
                {
                    FileWriter writer; writer._memset(); // Object used to write to disk
                    // If there are SORTER_MAX_MERGE_COUNT or less PMAs in file pTemp1, initialize an iterator for each of them and break out of the loop.
                    // These iterators will be incrementally merged as the VDBE layer calls sqlite3VdbeSorterNext().
                    //
                    // Otherwise, if pTemp1 contains more than SORTER_MAX_MERGE_COUNT PMAs, initialize interators for SORTER_MAX_MERGE_COUNT of them. These PMAs
                    // are merged into a single PMA that is written to file pTemp2.
                    long writes; // Number of bytes in new PMA
                    rc = VdbeSorterInitMerge(ctx, cursor, ref writes);
                    Debug.Assert(rc != RC.OK || sorter.Iters[sorter.Trees[1]].File);
                    if (rc != RC.OK || sorter.PMAs <= SORTER_MAX_MERGE_COUNT)
                        break;
                    // Open the second temp file, if it is not already open.
                    if (temp2 == null)
                    {
                        Debug.Assert(write2 == 0);
                        rc = VdbeSorterOpenTempFile(ctx, ref temp2);
                    }
                    if (rc == RC.OK)
                    {
                        bool eof = false;
                        FileWriterInit(ctx, temp2, ref writer, write2);
                        FileWriterWriteVarint(writer, writes);
                        while (rc == RC.OK && !eof)
                        {
                            VdbeSorterIter iter = sorter.Iters[sorter.Trees[1]];
                            Debug.Assert(iter.File);
                            FileWriterWriteVarint(writer, iter.Key.length);
                            FileWriterWrite(writer, iter.Key, iter.Key.length);
                            rc = SorterNext(ctx, cursor, eof);
                        }
                        RC rc2 = FileWriterFinish(ctx, writer, write2);
                        if (rc == RC.OK) rc = rc2;
                    }
                }
                if (sorter.PMAs <= SORTER_MAX_MERGE_COUNT)
                    break;
                else
                {
                    VFile tmp = sorter.Temp1;
                    sorter.PMAs = newIdx;
                    sorter.Temp1 = temp2;
                    temp2 = tmp;
                    sorter.WriteOffset = write2;
                    sorter.ReadOffset = 0;
                    write2 = 0;
                }
            } while (rc == RC.OK);
            if (temp2)
                temp2.CloseAndFree();
            eof = (sorter.Iters[sorter.Trees[1]].File == null);
            return rc;
        }

        public RC SorterNext(Context ctx, VdbeCursor cursor, ref bool eof)
        {
            VdbeSorter sorter = cursor.Sorter;
            if (sorter.Trees)
            {
                int prevI = sorter.Trees[1]; // Index of iterator to advance
                RC rc = VdbeSorterIterNext(ctx, sorter.Iters[prevI]);
                for (int i = (sorter.Trees.length + prevI) / 2; rc == RC.OK && i > 0; i /= 2) // Index of aTree[] to recalculate
                    rc = VdbeSorterDoCompare(cursor, i);
                eof = (sorter.Iters[sorter.Trees[1]].File == null);
                return rc;
            }
            SorterRecord free = sorter.Record;
            sorter.Record = free.Next;
            free.Next = nullptr;
            VdbeSorterRecordFree(ctx, free);
            eof = (sorter.Record == null);
            return RC.OK;
        }

        static object VdbeSorterRowkey(VdbeSorter sorter, out int keyLength)
        {
            if (sorter.Trees.data != null)
            {
                VdbeSorterIter iter = sorter.Iters[sorter.Trees.data[1]];
                keyLength = iter.Key.length;
                return iter.Key;
            }
            keyLength = sorter.Record.N;
            return sorter.Record.P;
        }

        public RC SorterRowkey(VdbeCursor cursor, Mem mem)
        {
            int keyLength;
            void* key = VdbeSorterRowkey(cursor.Sorter, out keyLength);
            if (MemGrow(mem, keyLength, 0))
                return RC.NOMEM;
            mem.N = keyLength;
            MemSetTypeFlag(mem, MEM.Blob);
            C._memcpy(mem.Z, key, keyLength);
            return RC.OK;
        }

        public RC SorterCompare(VdbeCursor cursor, Mem mem, out int r)
        {
            int keyLength;
            object key = VdbeSorterRowkey(cursor.Sorter, out keyLength);
            VdbeSorterCompare(cursor, 1, mem.Z, mem.N, key, keyLength, out r);
            return RC.OK;
        }

        #endregion
    }
}
