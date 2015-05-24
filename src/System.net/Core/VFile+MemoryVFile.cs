using System;
using System.Diagnostics;
namespace Core.IO
{
    public class MemoryVFile : VFile
    {
        const int JOURNAL_CHUNKSIZE = 4096;

        public class FileChunk
        {
            public FileChunk Next;                             // Next chunk in the journal 
            public byte[] Chunk = new byte[JOURNAL_CHUNKSIZE]; // Content of this chunk 
        }

        public class FilePoint
        {
            public long Offset;     // Offset from the beginning of the file 
            public FileChunk Chunk; // Specific chunk into which cursor points 
        }

        public FileChunk First;              // Head of in-memory chunk-list
        public FilePoint _endpoint = new FilePoint();            // Pointer to the end of the file
        public FilePoint _readpoint = new FilePoint();           // Pointer to the end of the last xRead()

        public override RC Read(byte[] buffer, int amount, long offset)
        {
            // SQLite never tries to read past the end of a rollback journal file 
            Debug.Assert(offset + amount <= _endpoint.Offset);
            FileChunk chunk;
            if (_readpoint.Offset != offset || offset == 0)
            {
                var offset2 = 0;
                for (chunk = First; C._ALWAYS(chunk != null) && (offset2 + JOURNAL_CHUNKSIZE) <= offset; chunk = chunk.Next)
                    offset2 += JOURNAL_CHUNKSIZE;
            }
            else
                chunk = _readpoint.Chunk;
            var chunkOffset = (int)(offset % JOURNAL_CHUNKSIZE);
            var @out = 0;
            var read = amount;
            do
            {
                var space = JOURNAL_CHUNKSIZE - chunkOffset;
                var copy = Math.Min(read, space);
                Buffer.BlockCopy(chunk.Chunk, chunkOffset, buffer, @out, copy);
                @out += copy;
                read -= space;
                chunkOffset = 0;
            } while (read >= 0 && (chunk = chunk.Next) != null && read > 0);
            _readpoint.Offset = (int)(offset + amount);
            _readpoint.Chunk = chunk;
            return RC.OK;
        }

        public override RC Write(byte[] buffer, int amount, long offset)
        {
            // An in-memory journal file should only ever be appended to. Random access writes are not required by sqlite.
            Debug.Assert(offset == _endpoint.Offset);
            var b = 0;
            while (amount > 0)
            {
                var chunk = _endpoint.Chunk;
                var chunkOffset = (int)(_endpoint.Offset % JOURNAL_CHUNKSIZE);
                var space = Math.Min(amount, JOURNAL_CHUNKSIZE - chunkOffset);
                if (chunkOffset == 0)
                {
                    // new chunk is required to extend the file.
                    var newChunk = new FileChunk();
                    if (newChunk == null)
                        return RC.IOERR_NOMEM;
                    newChunk.Next = null;
                    if (chunk != null) { Debug.Assert(First != null); chunk.Next = newChunk; }
                    else { Debug.Assert(First == null); First = newChunk; }
                    _endpoint.Chunk = newChunk;
                }
                Buffer.BlockCopy(buffer, b, _endpoint.Chunk.Chunk, chunkOffset, space);
                b += space;
                amount -= space;
                _endpoint.Offset += space;
            }
            return RC.OK;
        }

        public override RC Truncate(long size)
        {
            Debug.Assert(size == 0);
            //var chunk = First;
            //while (chunk != null)
            //{
            //    var tmp = chunk;
            //    chunk = chunk.Next;
            //}
            Open();
            return RC.OK;
        }

        public override RC Close()
        {
            Truncate(0);
            return RC.OK;
        }

        public override RC Sync(SYNC flags)
        {
            return RC.OK;
        }

        public override RC get_FileSize(out long size)
        {
            size = _endpoint.Offset;
            return RC.OK;
        }

        private void Open()
        {
            Opened = true;
            //Debug.Assert(SysEx.HASALIGNMENT8(this));
            //_memset(this, 0, sizeof(MemoryVFile));
            // clear
            First = null;
            _endpoint = new FilePoint();
            _readpoint = new FilePoint();
        }
    }
}