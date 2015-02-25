using Pid = System.UInt32;
using IPage = Core.PgHdr;
using System;
using System.Diagnostics;
using System.Text;
using Core.IO;

namespace Core
{
    public partial class Btree
    {
#if DEBUG
        static bool BtreeTrace = false;
        static void TRACE(string x, params object[] args) { if (BtreeTrace) Console.WriteLine("b:" + string.Format(x, args)); }
#else
        static void TRACE(string x, params object[] args) { }
#endif

        #region Struct

        static byte[] _magicHeader = System.Text.Encoding.UTF8.GetBytes(FILE_HEADER);
        static IVdbe _vdbe;

        enum BTALLOC : byte
        {
            ANY = 0,        // Allocate any page
            EXACT = 1,      // Allocate exact page if possible
            LE = 2,         // Allocate any page <= the parameter
        }

        static void ASSERTCOVERAGE(bool p) { }
#if !OMIT_AUTOVACUUM
        static bool IFAUTOVACUUM(bool x) { return x; }
#else
        static bool IFAUTOVACUUM(bool x) { return false; }
#endif

#if !OMIT_SHARED_CACHE
        static BtShared _sharedCacheList = null;
        RC enable_shared_cache(bool enable)
        {
            _globalMainStatics.SharedCacheEnabled = enable;
            return RC.OK;
        }
#else
        static RC querySharedCacheTableLock(Btree p, Pid table, LOCK lock_) { return RC.OK; }
        static void clearAllSharedCacheTableLocks(Btree a) { }
        static void downgradeAllSharedCacheTableLocks(Btree a) { }
        static bool hasSharedCacheTableLock(Btree a, Pid b, int c, int d) { return true; }
        static bool hasReadConflicts(Btree a, Pid b) { return false; }
#endif

        #endregion

        #region Shared Code1
#if !OMIT_SHARED_CACHE

#if DEBUG
        static bool hasSharedCacheTableLock(Btree btree, Pid root, bool isIndex, LOCK lockType)
        {
            // If this database is not shareable, or if the client is reading and has the read-uncommitted flag set, then no lock is required. 
            // Return true immediately.
            if (!btree.Sharable_ || (lockType == LOCK.READ && (btree.Ctx.Flags & BContext.FLAG.ReadUncommitted) != 0))
                return true;

            // If the client is reading  or writing an index and the schema is not loaded, then it is too difficult to actually check to see if
            // the correct locks are held.  So do not bother - just return true. This case does not come up very often anyhow.
            var schema = btree.Bt.Schema;
            if (isIndex && (schema == null || (schema.Flags & SCHEMA_.SchemaLoaded) == 0))
                return true;

            // Figure out the root-page that the lock should be held on. For table b-trees, this is just the root page of the b-tree being read or
            // written. For index b-trees, it is the root page of the associated table.
            Pid table = 0;
            if (isIndex)
                throw new NotImplementedException();
            //for (var p = sqliteHashFirst(schema.IdxHash); p != null; p = sqliteHashNext(p))
            //{
            //    var idx = (Index)sqliteHashData(p);
            //    if (idx.TID == (int)root)
            //        table = idx.Table.TID;
            //}
            else
                table = root;

            // Search for the required lock. Either a write-lock on root-page iTab, a write-lock on the schema table, or (if the client is reading) a
            // read-lock on iTab will suffice. Return 1 if any of these are found.
            for (var lock_ = btree.Bt.Lock; lock_ != null; lock_ = lock_.Next)
                if (lock_.Btree == btree &&
                    (lock_.Table == table || (lock_.Lock == LOCK.WRITE && lock_.Table == 1)) &&
                    lock_.Lock >= lockType)
                    return true;

            // Failed to find the required lock.
            return false;
        }

        static bool hasReadConflicts(Btree btree, Pid root)
        {
            for (var p = btree.Bt.Cursor; p != null; p = p.Next)
                if (p.RootID == root &&
                    p.Btree != btree &&
                    (p.Btree.Ctx.Flags & BContext.FLAG.ReadUncommitted) == 0)
                    return true;
            return false;
        }
#endif

        static RC querySharedCacheTableLock(Btree p, Pid table, LOCK lockType)
        {
            Debug.Assert(p.HoldsMutex());
            Debug.Assert(lockType == LOCK.READ || lockType == LOCK.WRITE);
            Debug.Assert(p.Ctx != null);
            Debug.Assert((p.Ctx.Flags & BContext.FLAG.ReadUncommitted) == 0 || lockType == LOCK.WRITE || table == 1);

            // If requesting a write-lock, then the Btree must have an open write transaction on this file. And, obviously, for this to be so there
            // must be an open write transaction on the file itself.
            var bt = p.Bt;
            Debug.Assert(lockType == LOCK.READ || (p == bt.Writer && p.InTrans == TRANS.WRITE));
            Debug.Assert(lockType == LOCK.READ || bt.InTransaction == TRANS.WRITE);

            // This routine is a no-op if the shared-cache is not enabled
            if (!p.Sharable_)
                return RC.OK;

            // If some other connection is holding an exclusive lock, the requested lock may not be obtained.
            if (bt.Writer != p && (bt.BtsFlags & BTS.EXCLUSIVE) != 0)
            {
                BContext.ConnectionBlocked(p.Ctx, bt.Writer.Ctx);
                return RC.LOCKED_SHAREDCACHE;
            }

            for (var iter = bt.Lock; iter != null; iter = iter.Next)
            {
                // The condition (pIter->eLock!=eLock) in the following if(...) statement is a simplification of:
                //
                //   (eLock==WRITE_LOCK || pIter->eLock==WRITE_LOCK)
                //
                // since we know that if eLock==WRITE_LOCK, then no other connection may hold a WRITE_LOCK on any table in this file (since there can
                // only be a single writer).
                Debug.Assert(iter.Lock == LOCK.READ || iter.Lock == LOCK.WRITE);
                Debug.Assert(lockType == LOCK.READ || iter.Btree == p || iter.Lock == LOCK.READ);
                if (iter.Btree != p && iter.Table == table && iter.Lock != lockType)
                {
                    BContext.ConnectionBlocked(p.Ctx, iter.Btree.Ctx);
                    if (lockType == LOCK.WRITE)
                    {
                        Debug.Assert(p == bt.Writer);
                        bt.BtsFlags |= BTS.PENDING;
                    }
                    return RC.LOCKED_SHAREDCACHE;
                }
            }
            return RC.OK;
        }

        static RC setSharedCacheTableLock(Btree p, Pid table, LOCK lock_)
        {
            Debug.Assert(p.HoldsMutex());
            Debug.Assert(lock_ == LOCK.READ || lock_ == LOCK.WRITE);
            Debug.Assert(p.Ctx != null);

            // A connection with the read-uncommitted flag set will never try to obtain a read-lock using this function. The only read-lock obtained
            // by a connection in read-uncommitted mode is on the sqlite_master table, and that lock is obtained in BtreeBeginTrans().
            Debug.Assert((p.Ctx.Flags & BContext.FLAG.ReadUncommitted) == 0 || lock_ == LOCK.WRITE);

            // This function should only be called on a sharable b-tree after it has been determined that no other b-tree holds a conflicting lock.
            var bt = p.Bt;

            Debug.Assert(p.Sharable_);
            Debug.Assert(RC.OK == querySharedCacheTableLock(p, table, lock_));

            // First search the list for an existing lock on this table.
            BtLock newLock = null;
            for (var iter = bt.Lock; iter != null; iter = iter.Next)
                if (iter.Table == table && iter.Btree == p)
                {
                    newLock = iter;
                    break;
                }

            // If the above search did not find a BtLock struct associating Btree p with table iTable, allocate one and link it into the list.
            if (newLock == null)
            {
                newLock = new BtLock();
                newLock.Table = table;
                newLock.Btree = p;
                newLock.Next = bt.Lock;
                bt.Lock = newLock;
            }

            // Set the BtLock.eLock variable to the maximum of the current lock and the requested lock. This means if a write-lock was already held
            // and a read-lock requested, we don't incorrectly downgrade the lock.
            Debug.Assert(LOCK.WRITE > LOCK.READ);
            if (lock_ > newLock.Lock)
                newLock.Lock = lock_;

            return RC.OK;
        }

        static void clearAllSharedCacheTableLocks(Btree p)
        {
            var bt = p.Bt;
            var iter = bt.Lock;

            Debug.Assert(p.HoldsMutex());
            Debug.Assert(p.Sharable_ || iter == null);
            Debug.Assert((int)p.InTrans > 0);

            while (iter != null)
            {
                BtLock lock_ = iter;
                Debug.Assert((bt.BtsFlags & BTS.EXCLUSIVE) == 0 || bt.Writer == lock_.Btree);
                Debug.Assert((int)lock_.Btree.InTrans >= (int)lock_.Lock);
                if (lock_.Btree == p)
                {
                    iter = lock_.Next;
                    Debug.Assert(lock_.Table != 1 || lock_ == p.Lock);
                    if (lock_.Table != 1)
                        lock_ = null;
                }
                else
                    iter = lock_.Next;
            }

            Debug.Assert((bt.BtsFlags & BTS.PENDING) == 0 || bt.Writer != null);
            if (bt.Writer == p)
            {
                bt.Writer = null;
                bt.BtsFlags &= ~(BTS.EXCLUSIVE | BTS.PENDING);
            }
            else if (bt.Transactions == 2)
            {
                // This function is called when Btree p is concluding its transaction. If there currently exists a writer, and p is not
                // that writer, then the number of locks held by connections other than the writer must be about to drop to zero. In this case
                // set the BTS_PENDING flag to 0.
                //
                // If there is not currently a writer, then BTS_PENDING must be zero already. So this next line is harmless in that case.
                bt.BtsFlags &= ~BTS.PENDING;
            }
        }

        static void downgradeAllSharedCacheTableLocks(Btree p)
        {
            var bt = p.Bt;
            if (bt.Writer == p)
            {
                bt.Writer = null;
                bt.BtsFlags &= ~(BTS.EXCLUSIVE | BTS.PENDING);
                for (var lock_ = bt.Lock; lock_ != null; lock_ = lock_.Next)
                {
                    Debug.Assert(lock_.Lock == LOCK.READ || lock_.Btree == p);
                    lock_.Lock = LOCK.READ;
                }
            }
        }

#endif
        #endregion

        #region Name2

#if DEBUG
        static bool cursorHoldsMutex(BtCursor p)
        {
            return MutexEx.Held(p.Bt.Mutex);
        }
#endif

#if !OMIT_INCRBLOB
        static void invalidateOverflowCache(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            cur.Overflows = null;
        }

        static void invalidateAllOverflowCache(BtShared bt)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            for (var p = bt.Cursor; p != null; p = p.Next)
                invalidateOverflowCache(p);
        }

        static void invalidateIncrblobCursors(Btree btree, long rowid, bool isClearTable)
        {
            var bt = btree.Bt;
            Debug.Assert(btree.HoldsMutex());
            for (var p = bt.Cursor; p != null; p = p.Next)
                if (p.IsIncrblobHandle && (isClearTable || p.Info.Key == rowid))
                    p.State = CURSOR.INVALID;
        }
#else
        static void invalidateOverflowCache(BtCursor cur) { }
        static void invalidateAllOverflowCache(BtShared bt) { }
        static void invalidateIncrblobCursors(Btree x, long y, bool z) { }
#endif

        #endregion

        #region Name3

        static RC btreeSetHasContent(BtShared bt, Pid id)
        {
            var rc = RC.OK;
            if (bt.HasContent == null)
            {
                Debug.Assert(id <= bt.Pages);
                bt.HasContent = new Bitvec(bt.Pages);
            }
            if (rc == RC.OK && id <= bt.HasContent.Length)
                rc = bt.HasContent.Set(id);
            return rc;
        }

        static bool btreeGetHasContent(BtShared bt, Pid id)
        {
            var p = bt.HasContent;
            return (p != null && (id > p.Length || p.Get(id)));
        }

        static void btreeClearHasContent(BtShared bt)
        {
            Bitvec.Destroy(ref bt.HasContent);
            bt.HasContent = null;
        }

        static void btreeReleaseAllCursorPages(BtCursor cur)
        {
            for (int i = 0; i <= cur.ID; i++)
            {
                releasePage(cur.Pages[i]);
                cur.Pages[i] = null;
            }
            cur.ID = -1;
        }

        static RC saveCursorPosition(BtCursor cur)
        {
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cur.Key == null);
            Debug.Assert(cursorHoldsMutex(cur));

            var rc = KeySize(cur, ref cur.KeyLength);
            Debug.Assert(rc == RC.OK);  // KeySize() cannot fail

            // If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur.nKey. In this case this value is
            // all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur.nKey bytes of key data.
            if (!cur.Pages[0].IntKey)
            {
                var key = C._alloc((int)cur.KeyLength);
                rc = Key(cur, 0, (uint)cur.KeyLength, key);
                if (rc == RC.OK)
                    cur.Key = key;
            }
            Debug.Assert(!cur.Pages[0].IntKey || cur.Key == null);

            if (rc == RC.OK)
            {
                btreeReleaseAllCursorPages(cur);
                cur.State = CURSOR.REQUIRESEEK;
            }

            invalidateOverflowCache(cur);
            return rc;
        }

        static RC saveAllCursors(BtShared bt, Pid root, BtCursor except)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(except == null || except.Bt == bt);
            for (var p = bt.Cursor; p != null; p = p.Next)
            {
                if (p != except && (root == 0 || p.RootID == root) && p.State == CURSOR.VALID)
                {
                    var rc = saveCursorPosition(p);
                    if (rc != RC.OK)
                        return rc;
                }
            }
            return RC.OK;
        }

        public static void ClearCursor(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            C._free(ref cur.Key);
            cur.State = CURSOR.INVALID;
        }

        static RC btreeMoveto(BtCursor cur, byte[] key, long keyLength, int bias, ref int res)
        {
            UnpackedRecord idxKey; // Unpacked index key
            object free;
            if (key != null)
            {
                Debug.Assert(keyLength == (long)(int)keyLength);
                idxKey = _vdbe.AllocUnpackedRecord(cur.KeyInfo, null, 0, out free);
                if (idxKey == null) return RC.NOMEM;
                _vdbe.RecordUnpack(cur.KeyInfo, (int)keyLength, key, idxKey);
            }
            else
                idxKey = null;
            var rc = MovetoUnpacked(cur, idxKey, keyLength, (bias != 0 ? 1 : 0), out res);
            return rc;
        }

        static RC btreeRestoreCursorPosition(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State >= CURSOR.REQUIRESEEK);
            if (cur.State == CURSOR.FAULT)
                return (RC)cur.SkipNext;
            cur.State = CURSOR.INVALID;
            var rc = btreeMoveto(cur, cur.Key, cur.KeyLength, 0, ref cur.SkipNext);
            if (rc == RC.OK)
            {
                cur.Key = null;
                Debug.Assert(cur.State == CURSOR.VALID || cur.State == CURSOR.INVALID);
            }
            return rc;
        }

        static RC restoreCursorPosition(BtCursor cur)
        {
            return (cur.State >= CURSOR.REQUIRESEEK ? btreeRestoreCursorPosition(cur) : RC.OK);
        }

        public static RC CursorHasMoved(BtCursor cur, out bool hasMoved)
        {
            var rc = restoreCursorPosition(cur);
            if (rc != RC.OK)
            {
                hasMoved = true;
                return rc;
            }
            hasMoved = (cur.State != CURSOR.VALID || cur.SkipNext != 0);
            return RC.OK;
        }

        #endregion

        #region Parse Cell

#if !OMIT_AUTOVACUUM
        static Pid ptrmapPageno(BtShared bt, Pid id)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            if (id < 2) return 0;
            var pagesPerMapPage = (int)(bt.UsableSize / 5 + 1);
            var ptrMap = (Pid)((id - 2) / pagesPerMapPage);
            var ret = (Pid)(ptrMap * pagesPerMapPage) + 2;
            if (ret == PENDING_BYTE_PAGE(bt))
                ret++;
            return ret;
        }

        static void ptrmapPut(BtShared bt, Pid key, PTRMAP type, Pid parent, ref RC rcRef)
        {
            if (rcRef != RC.OK) return;

            Debug.Assert(MutexEx.Held(bt.Mutex));
            // The master-journal page number must never be used as a pointer map page
            Debug.Assert(!PTRMAP_ISPAGE(bt, PENDING_BYTE_PAGE(bt)));

            Debug.Assert(bt.AutoVacuum);
            if (key == 0)
            {
                rcRef = SysEx.CORRUPT_BKPT();
                return;
            }
            var ptrmapIdx = PTRMAP_PAGENO(bt, key); // The pointer map page number
            var page = (IPage)new PgHdr(); // The pointer map page
            var rc = bt.Pager.Acquire(ptrmapIdx, ref page, false);
            if (rc != RC.OK)
            {
                rcRef = rc;
                return;
            }
            var offset = (int)PTRMAP_PTROFFSET(ptrmapIdx, key); // Offset in pointer map page
            if (offset < 0)
            {
                rcRef = SysEx.CORRUPT_BKPT();
                goto ptrmap_exit;
            }
            Debug.Assert(offset <= (int)bt.UsableSize - 5);
            var ptrmap = Pager.GetData(page); // The pointer map page

            if (type != (PTRMAP)ptrmap[offset] || ConvertEx.Get4(ptrmap, offset + 1) != parent)
            {
                TRACE("PTRMAP_UPDATE: %d->(%d,%d)\n", key, type, parent);
                rcRef = rc = Pager.Write(page);
                if (rc == RC.OK)
                {
                    ptrmap[offset] = (byte)type;
                    ConvertEx.Put4(ptrmap, offset + 1, parent);
                }
            }

        ptrmap_exit:
            Pager.Unref(page);
        }

        static RC ptrmapGet(BtShared bt, Pid key, ref PTRMAP type, ref Pid id)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));

            var page = (IPage)new PgHdr(); // The pointer map page
            var ptrmapIdx = (Pid)PTRMAP_PAGENO(bt, key); // Pointer map page index
            var rc = bt.Pager.Acquire(ptrmapIdx, ref page, false);
            if (rc != RC.OK)
                return rc;
            var ptrmap = Pager.GetData(page); // Pointer map page data

            var offset = (int)PTRMAP_PTROFFSET(ptrmapIdx, key); // Offset of entry in pointer map
            if (offset < 0)
            {
                Pager.Unref(page);
                return SysEx.CORRUPT_BKPT();
            }
            Debug.Assert(offset <= (int)bt.UsableSize - 5);
            Debug.Assert(type != 0);
            type = (PTRMAP)ptrmap[offset];
            id = ConvertEx.Get4(ptrmap, offset + 1);

            Pager.Unref(page);
            if ((byte)type < 1 || (byte)type > 5) return SysEx.CORRUPT_BKPT();
            return RC.OK;
        }

#else
//#define ptrmapPut(w,x,y,z,rc)
//#define ptrmapGet(w,x,y,z) RC.OK
//#define ptrmapPutOvflPtr(x, y, rc)
#endif

        static uint findCell(MemPage page, uint cell) { return ConvertEx.Get2(page.Data, page.CellOffset + 2 * cell); }
        static byte[] findCellv2(byte[] page, ushort cell, ushort o, int i) { Debugger.Break(); return page; }

        static uint findOverflowCell(MemPage page, uint cell)
        {
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            for (var i = page.Overflows - 1; i >= 0; i--)
            {
                ushort k = page.OvflIdxs[i];
                if (k <= cell)
                {
                    if (k == cell)
                        return (uint)-i - 1; //page.Ovfls[i]; // Negative Offset means overflow cells

                    cell--;
                }
            }
            return findCell(page, (uint)cell);
        }

        static void btreeParseCellPtr(MemPage page, uint cellIdx, ref CellInfo info) { btreeParseCellPtr(page, page.Data, cellIdx, ref info); }
        static void btreeParseCellPtr(MemPage page, byte[] cell, ref CellInfo info) { btreeParseCellPtr(page, cell, 0, ref info); }
        static void btreeParseCellPtr(MemPage page, byte[] cell, uint cellIdx, ref CellInfo info)
        {
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));

            if (info.Cell != cell) info.Cell = cell;
            info.Cell_ = cellIdx;
            ushort n = page.ChildPtrSize; // Number bytes in cell content header
            Debug.Assert(n == (page.Leaf ? 0 : 4));
            uint payloadLength = 0; // Number of bytes of cell payload
            if (page.IntKey)
            {
                if (page.HasData)
                    n += (ushort)ConvertEx.GetVarint32(cell, cellIdx + n, out payloadLength);
                else
                    payloadLength = 0;
                n += (ushort)ConvertEx.GetVarint(cell, cellIdx + n, out info.Key);
                info.Data = payloadLength;
            }
            else
            {
                info.Data = 0;
                n += (ushort)ConvertEx.GetVarint32(cell, cellIdx + n, out payloadLength);
                info.Key = payloadLength;
            }
            info.Payload = payloadLength;
            info.Header = n;
            if (payloadLength <= page.MaxLocal)
            {
                // This is the (easy) common case where the entire payload fits on the local page.  No overflow is required.
                if ((info.Size = (ushort)(n + payloadLength)) < 4) info.Size = 4;
                info.Local = (ushort)payloadLength;
                info.Overflow = 0;
            }
            else
            {
                // If the payload will not fit completely on the local page, we have to decide how much to store locally and how much to spill onto
                // overflow pages.  The strategy is to minimize the amount of unused space on overflow pages while keeping the amount of local storage
                // in between minLocal and maxLocal.
                //
                // Warning:  changing the way overflow payload is distributed in any way will result in an incompatible file format.
                int minLocal = page.MinLocal; // Minimum amount of payload held locally
                int maxLocal = page.MaxLocal; // Maximum amount of payload held locally
                int surplus = (int)(minLocal + (payloadLength - minLocal) % (page.Bt.UsableSize - 4)); // Overflow payload available for local storage
                ASSERTCOVERAGE(surplus == maxLocal);
                ASSERTCOVERAGE(surplus == maxLocal + 1);
                if (surplus <= maxLocal)
                    info.Local = (ushort)surplus;
                else
                    info.Local = (ushort)minLocal;
                info.Overflow = (ushort)(info.Local + n);
                info.Size = (ushort)(info.Overflow + 4);
            }
        }

        static void parseCell(MemPage page, uint cell_, ref CellInfo info) { btreeParseCellPtr(page, findCell(page, cell_), ref info); }
        static void btreeParseCell(MemPage page, uint cell, ref CellInfo info) { parseCell(page, cell, ref info); }

        static ushort cellSizePtr(MemPage page, uint cell_) // For C#
        {
            var info = new CellInfo();
            var cell2 = new byte[13];
            // Minimum Size = (2 bytes of Header  or (4) Child Pointer) + (maximum of) 9 bytes data
            if (cell_ < 0) // Overflow Cell
                Buffer.BlockCopy(page.Ovfls[-(cell_ + 1)].Cell, 0, cell2, 0, cell2.Length < page.Ovfls[-(cell_ + 1)].Cell.Length ? cell2.Length : page.Ovfls[-(cell_ + 1)].Cell.Length);
            else if (cell_ >= page.Data.Length + 1 - cell2.Length)
                Buffer.BlockCopy(page.Data, (int)cell_, cell2, 0, (int)(page.Data.Length - cell_));
            else
                Buffer.BlockCopy(page.Data, (int)cell_, cell2, 0, cell2.Length);
            btreeParseCellPtr(page, cell2, ref info);
            return info.Size;
        }
        static ushort cellSizePtr(MemPage page, byte[] cell, uint offset_) // For C#
        {
            var info = new CellInfo();
            info.Cell = C._alloc(cell.Length);
            Buffer.BlockCopy(cell, (int)offset_, info.Cell, 0, (int)(cell.Length - offset_));
            btreeParseCellPtr(page, info.Cell, ref info);
            return info.Size;
        }
        static ushort cellSizePtr(MemPage page, byte[] cell)
        {
#if DEBUG
            // The value returned by this function should always be the same as the (CellInfo.nSize) value found by doing a full parse of the
            // cell. If SQLITE_DEBUG is defined, an assert() at the bottom of this function verifies that this invariant is not violated.
            var debuginfo = new CellInfo();
            btreeParseCellPtr(page, cell, ref debuginfo);
#else
            var debuginfo = new CellInfo();
#endif
            var iter = page.ChildPtrSize;
            uint size = 0;
            if (page.IntKey)
            {
                if (page.HasData)
                    iter += ConvertEx.GetVarint32(cell, out size); // iter += ConvertEx.GetVarint32(iter, out size);
                else
                    size = 0;

                // pIter now points at the 64-bit integer key value, a variable length integer. The following block moves pIter to point at the first byte
                // past the end of the key value.
                int end = iter + 9; // end = &pIter[9];
                while (((cell[iter++]) & 0x80) != 0 && iter < end) { } // while ((iter++) & 0x80 && iter < end);
            }
            else
                iter += ConvertEx.GetVarint32(cell, iter, out size); //pIter += getVarint32( pIter, out nSize );

            if (size > page.MaxLocal)
            {
                int minLocal = page.MinLocal;
                size = (uint)(minLocal + (size - minLocal) % (page.Bt.UsableSize - 4));
                if (size > page.MaxLocal)
                    size = (uint)minLocal;
                size += 4;
            }
            size += (uint)iter; // size += (uint32)(iter - cell);

            // The minimum size of any cell is 4 bytes.
            if (size < 4)
                size = 4;

            Debug.Assert(size == debuginfo.Size);
            return (ushort)size;
        }

#if DEBUG
        static ushort cellSize(MemPage page, uint cell)
        {
            return cellSizePtr(page, findCell(page, cell));
        }
#else
        static ushort cellSize(MemPage page, int cell) { return -1; }
#endif

#if !OMIT_AUTOVACUUM
        static void ptrmapPutOvflPtr(MemPage page, uint cell_, ref RC rcRef) // For C#
        {
            if (rcRef != RC.OK) return;
            var info = new CellInfo();
            Debug.Assert(cell_ != 0);
            btreeParseCellPtr(page, cell_, ref info);
            Debug.Assert((info.Data + (page.IntKey ? 0 : info.Key)) == info.Payload);
            if (info.Overflow != 0)
            {
                Pid ovfl = ConvertEx.Get4(page.Data, cell_ + info.Overflow);
                ptrmapPut(page.Bt, ovfl, PTRMAP.OVERFLOW1, page.ID, ref rcRef);
            }
        }
        static void ptrmapPutOvflPtr(MemPage page, byte[] cell, ref RC rcRef)
        {
            if (rcRef != RC.OK) return;
            Debug.Assert(cell != null);
            var info = new CellInfo();
            btreeParseCellPtr(page, cell, ref info);
            Debug.Assert((info.Data + (page.IntKey ? 0 : info.Key)) == info.Payload);
            if (info.Overflow != 0)
            {
                Pid ovfl = ConvertEx.Get4(cell, info.Overflow);
                ptrmapPut(page.Bt, ovfl, PTRMAP.OVERFLOW1, page.ID, ref rcRef);
            }
        }
#endif

        #endregion

        #region Allocate / Defragment

        static RC defragmentPage(MemPage page)
        {
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            Debug.Assert(page.Bt != null);
            Debug.Assert(page.Bt.UsableSize <= Pager.MAX_PAGE_SIZE);
            Debug.Assert(page.Overflows == 0);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            byte[] temp = page.Bt.Pager.get_TempSpace(); // Temp area for cell content
            byte[] data = page.Data; // The page data
            int hdr = page.HdrOffset; // Offset to the page header
            int cellOffset = page.CellOffset; // Offset to the cell pointer array
            int cells = page.Cells; // Number of cells on the page
            Debug.Assert(cells == ConvertEx.Get2(data, hdr + 3));
            uint usableSize = page.Bt.UsableSize; // Number of usable bytes on a page
            uint cbrk = (uint)ConvertEx.Get2(data, hdr + 5); // Offset to the cell content area
            Buffer.BlockCopy(data, (int)cbrk, temp, (int)cbrk, (int)(usableSize - cbrk)); // memcpy(temp[cbrk], ref data[cbrk], usableSize - cbrk);
            cbrk = usableSize;
            ushort cellFirst = (ushort)(cellOffset + 2 * cells); // First allowable cell index
            ushort cellLast = (ushort)(usableSize - 4); // Last possible cell index
            var addr = 0;  // The i-th cell pointer
            for (var i = 0; i < cells; i++)
            {
                addr = cellOffset + i * 2;
                uint pc = ConvertEx.Get2(data, addr); // Address of a i-th cell
#if !ENABLE_OVERSIZE_CELL_CHECK
                // These conditions have already been verified in btreeInitPage() if ENABLE_OVERSIZE_CELL_CHECK is defined
                if (pc < cellFirst || pc > cellLast)
                    return SysEx.CORRUPT_BKPT();
#endif
                Debug.Assert(pc >= cellFirst && pc <= cellLast);
                uint size = cellSizePtr(page, temp, pc); // Size of a cell
                cbrk -= size;
#if ENABLE_OVERSIZE_CELL_CHECK
                if (cbrk < cellFirst || pc + size > usableSize)
                    return SysEx.CORRUPT_BKPT();
#else
                if (cbrk < cellFirst || pc + size > usableSize)
                    return SysEx.CORRUPT_BKPT();
#endif
                Debug.Assert(cbrk + size <= usableSize && cbrk >= cellFirst);
                Buffer.BlockCopy(temp, (int)pc, data, (int)cbrk, (int)size);
                ConvertEx.Put2(data, addr, cbrk);
            }
            Debug.Assert(cbrk >= cellFirst);
            ConvertEx.Put2(data, hdr + 5, cbrk);
            data[hdr + 1] = 0;
            data[hdr + 2] = 0;
            data[hdr + 7] = 0;
            addr = cellOffset + 2 * cells;
            Array.Clear(data, addr, (int)(cbrk - addr));
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            if (cbrk - cellFirst != page.Frees)
                return SysEx.CORRUPT_BKPT();
            return RC.OK;
        }

        static RC allocateSpace(MemPage page, int bytes, ref uint idx)
        {
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            Debug.Assert(page.Bt != null);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(bytes >= 0);  // Minimum cell size is 4
            Debug.Assert(page.Frees >= bytes);
            Debug.Assert(page.Overflows == 0);
            var usableSize = page.Bt.UsableSize; // Usable size of the page
            Debug.Assert(bytes < usableSize - 8);

            var hdr = page.HdrOffset;  // Local cache of pPage.hdrOffset
            var data = page.Data;    // Local cache of pPage.aData
            var frags = data[hdr + 7]; // Number of fragmented bytes on pPage
            Debug.Assert(page.CellOffset == hdr + 12 - 4 * (page.Leaf ? 1 : 0));
            var gap = page.CellOffset + 2 * page.Cells; // First byte of gap between cell pointers and cell content
            var top = (uint)ConvertEx.Get2nz(data, hdr + 5); // First byte of cell content area
            if (gap > top) return SysEx.CORRUPT_BKPT();

            RC rc;
            if (frags >= 60)
            {
                // Always defragment highly fragmented pages
                rc = defragmentPage(page);
                if (rc != RC.OK) return rc;
                top = ConvertEx.Get2nz(data, hdr + 5);
            }
            else if (gap + 2 <= top)
            {
                // Search the freelist looking for a free slot big enough to satisfy the request. The allocation is made from the first free slot in 
                // the list that is large enough to accomadate it.
                int pc;
                for (int addr = hdr + 1; (pc = ConvertEx.Get2(data, addr)) > 0; addr = pc)
                {
                    if (pc > usableSize - 4 || pc < addr + 4)
                        return SysEx.CORRUPT_BKPT();
                    int size = ConvertEx.Get2(data, pc + 2); // Size of free slot
                    if (size >= bytes)
                    {
                        int x = size - bytes;
                        if (x < 4)
                        {
                            // Remove the slot from the free-list. Update the number of fragmented bytes within the page.
                            data[addr + 0] = data[pc + 0]; // memcpy( data[addr], ref data[pc], 2 );
                            data[addr + 1] = data[pc + 1];
                            data[hdr + 7] = (byte)(frags + x);
                        }
                        else if (size + pc > usableSize)
                            return SysEx.CORRUPT_BKPT();
                        else // The slot remains on the free-list. Reduce its size to account for the portion used by the new allocation.
                            ConvertEx.Put2(data, pc + 2, x);
                        idx = (uint)(pc + x);
                        return RC.OK;
                    }
                }
            }

            // Check to make sure there is enough space in the gap to satisfy the allocation.  If not, defragment.
            if (gap + 2 + bytes > top)
            {
                rc = defragmentPage(page);
                if (rc != RC.OK) return rc;
                top = ConvertEx.Get2nz(data, hdr + 5);
                Debug.Assert(gap + bytes <= top);
            }

            // Allocate memory from the gap in between the cell pointer array and the cell content area.  The btreeInitPage() call has already
            // validated the freelist.  Given that the freelist is valid, there is no way that the allocation can extend off the end of the page.
            // The assert() below verifies the previous sentence.
            top -= (uint)bytes;
            ConvertEx.Put2(data, hdr + 5, top);
            Debug.Assert(top + bytes <= (int)page.Bt.UsableSize);
            idx = top;
            return RC.OK;
        }

        static RC freeSpace(MemPage page, uint start, int size) { return freeSpace(page, (int)start, size); }
        static RC freeSpace(MemPage page, int start, int size)
        {
            Debug.Assert(page.Bt != null);
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            Debug.Assert(start >= page.HdrOffset + 6 + page.ChildPtrSize);
            Debug.Assert((start + size) <= (int)page.Bt.UsableSize);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(size >= 0); // Minimum cell size is 4

            var data = page.Data;
            if ((page.Bt.BtsFlags & BTS.SECURE_DELETE) != 0) // Overwrite deleted information with zeros when the secure_delete option is enabled
                Array.Clear(data, start, size);

            // Add the space back into the linked list of freeblocks.  Note that even though the freeblock list was checked by btreeInitPage(),
            // btreeInitPage() did not detect overlapping cells or freeblocks that overlapped cells.   Nor does it detect when the
            // cell content area exceeds the value in the page header.  If these situations arise, then subsequent insert operations might corrupt
            // the freelist.  So we do need to check for corruption while scanning the freelist.
            int hdr = page.HdrOffset;
            int addr = hdr + 1;
            int last = (int)page.Bt.UsableSize - 4; // Largest possible freeblock offset
            Debug.Assert(start <= last);
            int pbegin;
            while ((pbegin = ConvertEx.Get2(data, addr)) < start && pbegin > 0)
            {
                if (pbegin < addr + 4)
                    return SysEx.CORRUPT_BKPT();
                addr = pbegin;
            }
            if (pbegin > last)
                return SysEx.CORRUPT_BKPT();
            Debug.Assert(pbegin > addr || pbegin == 0);
            ConvertEx.Put2(data, addr, start);
            ConvertEx.Put2(data, start, pbegin);
            ConvertEx.Put2(data, start + 2, size);
            page.Frees = (ushort)(page.Frees + size);

            // Coalesce adjacent free blocks
            addr = hdr + 1;
            while ((pbegin = ConvertEx.Get2(data, addr)) > 0)
            {
                Debug.Assert(pbegin > addr);
                Debug.Assert(pbegin <= (int)page.Bt.UsableSize - 4);
                int pnext = ConvertEx.Get2(data, pbegin);
                int psize = ConvertEx.Get2(data, pbegin + 2);
                if (pbegin + psize + 3 >= pnext && pnext > 0)
                {
                    int frag = pnext - (pbegin + psize);
                    if (frag < 0 || frag > (int)data[hdr + 7])
                        return SysEx.CORRUPT_BKPT();
                    data[hdr + 7] -= (byte)frag;
                    int x = ConvertEx.Get2(data, pnext);
                    ConvertEx.Put2(data, pbegin, x);
                    x = pnext + ConvertEx.Get2(data, pnext + 2) - pbegin;
                    ConvertEx.Put2(data, pbegin + 2, x);
                }
                else
                    addr = pbegin;
            }

            // If the cell content area begins with a freeblock, remove it.
            if (data[hdr + 1] == data[hdr + 5] && data[hdr + 2] == data[hdr + 6])
            {
                pbegin = ConvertEx.Get2(data, hdr + 1);
                ConvertEx.Put2(data, hdr + 1, ConvertEx.Get2(data, pbegin)); // memcpy( data[hdr + 1], ref data[pbegin], 2 );
                int top = ConvertEx.Get2(data, hdr + 5) + ConvertEx.Get2(data, pbegin + 2);
                ConvertEx.Put2(data, hdr + 5, top);
            }
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            return RC.OK;
        }

        static RC decodeFlags(MemPage page, int flagByte)
        {
            Debug.Assert(page.HdrOffset == (page.ID == 1 ? 100 : 0));
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            page.Leaf = ((flagByte >> 3) != 0); Debug.Assert(PTF_LEAF == 1 << 3);
            flagByte &= ~PTF_LEAF;
            page.ChildPtrSize = (byte)(page.Leaf ? 0 : 4);
            BtShared bt = page.Bt; // A copy of pPage.pBt
            if (flagByte == (PTF_LEAFDATA | PTF_INTKEY))
            {
                page.IntKey = true;
                page.HasData = page.Leaf;
                page.MaxLocal = bt.MaxLeaf;
                page.MinLocal = bt.MinLeaf;
            }
            else if (flagByte == PTF_ZERODATA)
            {
                page.IntKey = false;
                page.HasData = false;
                page.MaxLocal = bt.MaxLocal;
                page.MinLocal = bt.MinLocal;
            }
            else
                return SysEx.CORRUPT_BKPT();
            page.Max1bytePayload = bt.Max1bytePayload;
            return RC.OK;
        }

        static RC btreeInitPage(MemPage page)
        {
            Debug.Assert(page.Bt != null);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(page.ID == Pager.get_PageID(page.DBPage));
            Debug.Assert(page == Pager.GetExtra<MemPage>(page.DBPage));
            Debug.Assert(page.Data == Pager.GetData(page.DBPage));

            if (!page.IsInit)
            {
                var bt = page.Bt; // The main btree structure

                var hdr = page.HdrOffset; // Offset to beginning of page header 
                var data = page.Data; // Equal to pPage.aData
                if (decodeFlags(page, data[hdr]) != RC.OK) return SysEx.CORRUPT_BKPT();
                Debug.Assert(bt.PageSize >= 512 && bt.PageSize <= 65536);
                page.MaskPage = (ushort)(bt.PageSize - 1);
                page.Overflows = 0;
                int usableSize = (int)bt.UsableSize; // Amount of usable space on each page
                ushort cellOffset; // Offset from start of page to first cell pointer
                page.CellOffset = (cellOffset = (ushort)(hdr + 12 - 4 * (page.Leaf ? 1 : 0)));
                int top = ConvertEx.Get2nz(data, hdr + 5); // First byte of the cell content area
                page.Cells = (ushort)(ConvertEx.Get2(data, hdr + 3));
                if (page.Cells > MX_CELL(bt))
                    // To many cells for a single page.  The page must be corrupt
                    return SysEx.CORRUPT_BKPT();

                // A malformed database page might cause us to read past the end of page when parsing a cell.  
                //
                // The following block of code checks early to see if a cell extends past the end of a page boundary and causes SQLITE_CORRUPT to be 
                // returned if it does.
                int cellFirst = cellOffset + 2 * page.Cells; // First allowable cell or freeblock offset
                int cellLast = usableSize - 4; // Last possible cell or freeblock offset
                ushort pc; // Address of a freeblock within pPage.aData[]
#if ENABLE_OVERSIZE_CELL_CHECK
                {
                    if (!page.Leaf) cellLast--;
                    for (var i = 0; i < page.Cells; i++)
                    {
                        pc = (ushort)ConvertEx.Get2(data, cellOffset + i * 2);
                        if (pc < cellFirst || pc > cellLast)
                            return SysEx.CORRUPT_BKPT();
                        int sz = cellSizePtr(page, data, pc); // Size of a cell
                        if (pc + sz > usableSize)
                            return SysEx.CORRUPT_BKPT();
                    }
                    if (!page.Leaf) cellLast++;
                }
#endif

                // Compute the total free space on the page
                pc = (ushort)ConvertEx.Get2(data, hdr + 1);
                int free = (ushort)(data[hdr + 7] + top); // Number of unused bytes on the page
                while (pc > 0)
                {
                    if (pc < cellFirst || pc > cellLast)
                        // Start of free block is off the page
                        return SysEx.CORRUPT_BKPT();
                    var next = (ushort)ConvertEx.Get2(data, pc);
                    var size = (ushort)ConvertEx.Get2(data, pc + 2);
                    if ((next > 0 && next <= pc + size + 3) || pc + size > usableSize)
                        // Free blocks must be in ascending order. And the last byte of the free-block must lie on the database page.
                        return SysEx.CORRUPT_BKPT();
                    free = (ushort)(free + size);
                    pc = next;
                }

                // At this point, nFree contains the sum of the offset to the start of the cell-content area plus the number of free bytes within
                // the cell-content area. If this is greater than the usable-size of the page, then the page must be corrupted. This check also
                // serves to verify that the offset to the start of the cell-content area, according to the page header, lies within the page.
                if (free > usableSize)
                    return SysEx.CORRUPT_BKPT();
                page.Frees = (ushort)(free - cellFirst);
                page.IsInit = true;
            }
            return RC.OK;
        }

        static void zeroPage(MemPage page, int flags)
        {
            var bt = page.Bt;
            var data = page.Data;
            Debug.Assert(Pager.get_PageID(page.DBPage) == page.ID);
            Debug.Assert(Pager.GetExtra<MemPage>(page.DBPage) == page);
            Debug.Assert(Pager.GetData(page.DBPage) == data);
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            Debug.Assert(MutexEx.Held(bt.Mutex));
            var hdr = page.HdrOffset;
            if ((bt.BtsFlags & BTS.SECURE_DELETE) != 0)
                Array.Clear(data, hdr, (int)(bt.UsableSize - hdr));
            data[hdr] = (byte)flags;
            var first = (ushort)(hdr + 8 + 4 * ((flags & PTF_LEAF) == 0 ? 1 : 0));
            Array.Clear(data, hdr + 1, 4);
            data[hdr + 7] = 0;
            ConvertEx.Put2(data, hdr + 5, bt.UsableSize);
            page.Frees = (ushort)(bt.UsableSize - first);
            decodeFlags(page, flags);
            page.HdrOffset = hdr;
            //page.DataEnd = &data[pt.UsableSize];
            //page.CellIdx = &data[first];
            page.CellOffset = first;
            page.Overflows = 0;
            Debug.Assert(bt.PageSize >= 512 && bt.PageSize <= 65536);
            page.MaskPage = (ushort)(bt.PageSize - 1);
            page.Cells = 0;
            page.IsInit = true;
        }

        #endregion

        #region Page

        static MemPage btreePageFromDbPage(IPage dbPage, Pid id, BtShared bt)
        {
            MemPage page = Pager.GetExtra<MemPage>(dbPage);
            page.Data = Pager.GetData(dbPage);
            page.DBPage = dbPage;
            page.Bt = bt;
            page.ID = id;
            page.HdrOffset = (byte)(page.ID == 1 ? 100 : 0);
            return page;
        }

        static RC btreeGetPage(BtShared bt, Pid id, ref MemPage page, bool noContent)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            IPage dbPage = null;
            var rc = bt.Pager.Acquire(id, ref dbPage, noContent);
            if (rc != RC.OK) return rc;
            page = btreePageFromDbPage(dbPage, id, bt);
            return RC.OK;
        }

        static MemPage btreePageLookup(BtShared bt, Pid id)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            var dbPage = bt.Pager.Lookup(id);
            return (dbPage != null ? btreePageFromDbPage(dbPage, id, bt) : null);
        }

        static Pid btreePagecount(BtShared bt)
        {
            return bt.Pages;
        }

        public Pid LastPage()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(((Bt.Pages) & 0x8000000) == 0);
            return (Pid)btreePagecount(Bt);
        }

        static RC getAndInitPage(BtShared bt, Pid id, ref MemPage page)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));

            RC rc;
            if (id > btreePagecount(bt))
                rc = SysEx.CORRUPT_BKPT();
            else
            {
                rc = btreeGetPage(bt, id, ref page, false);
                if (rc == RC.OK)
                {
                    rc = btreeInitPage(page);
                    if (rc != RC.OK)
                        releasePage(page);
                }
            }

            Debug.Assert(id != 0 || rc == RC.CORRUPT);
            return rc;
        }

        static void releasePage(MemPage page)
        {
            if (page != null)
            {
                Debug.Assert(page.Data != null);
                Debug.Assert(page.Bt != null);
                Debug.Assert(Pager.GetExtra<MemPage>(page.DBPage) == page);
                Debug.Assert(Pager.GetData(page.DBPage) == page.Data);
                Debug.Assert(MutexEx.Held(page.Bt.Mutex));
                Pager.Unref(page.DBPage);
            }
        }

        static void pageReinit(IPage dbPage)
        {
            MemPage page = Pager.GetExtra<MemPage>(dbPage);
            Debug.Assert(Pager.get_PageRefs(dbPage) > 0);
            if (page.IsInit)
            {
                Debug.Assert(MutexEx.Held(page.Bt.Mutex));
                page.IsInit = false;
                if (Pager.get_PageRefs(dbPage) > 1)
                {
                    // pPage might not be a btree page;  it might be an overflow page or ptrmap page or a free page.  In those cases, the following
                    // call to btreeInitPage() will likely return SQLITE_CORRUPT. But no harm is done by this.  And it is very important that
                    // btreeInitPage() be called on every btree page so we make the call for every page that comes in for re-initing.
                    btreeInitPage(page);
                }
            }
        }

        #endregion

        #region Open / Close

        static int btreeInvokeBusyHandler(object arg)
        {
            var bt = (BtShared)arg;
            Debug.Assert(bt.Ctx != null);
            Debug.Assert(MutexEx.Held(bt.Ctx.Mutex));
            return bt.Ctx.InvokeBusyHandler();
        }

        public static RC Open(VSystem vfs, string filename, BContext ctx, ref Btree btree, OPEN flags, VSystem.OPEN vfsFlags)
        {
            // True if opening an ephemeral, temporary database
            bool tempDB = string.IsNullOrEmpty(filename);

            // Set the variable isMemdb to true for an in-memory database, or false for a file-based database.
            bool memoryDB = (filename == ":memory:") ||
                (tempDB && ctx.TempInMemory()) ||
                (vfsFlags & VSystem.OPEN.MEMORY) != 0;

            Debug.Assert(ctx != null);
            Debug.Assert(vfs != null);
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            Debug.Assert(((uint)flags & 0xff) == (uint)flags); // flags fit in 8 bits

            // Only a BTREE_SINGLE database can be BTREE_UNORDERED
            Debug.Assert((flags & OPEN.UNORDERED) == 0 || (flags & OPEN.SINGLE) != 0);

            // A BTREE_SINGLE database is always a temporary and/or ephemeral
            Debug.Assert((flags & OPEN.SINGLE) == 0 || tempDB);

            if (memoryDB)
                flags |= OPEN.MEMORY;
            if ((vfsFlags & VSystem.OPEN.MAIN_DB) != 0 && (memoryDB || tempDB))
                vfsFlags = (vfsFlags & ~VSystem.OPEN.MAIN_DB) | VSystem.OPEN.TEMP_DB;
            var p = new Btree(); // Handle to return
            if (p == null)
                return RC.NOMEM;
            p.InTrans = TRANS.NONE;
            p.Ctx = ctx;
#if !OMIT_SHARED_CACHE
            p.Lock.Btree = p;
            p.Lock.Table = 1;
#endif

            RC rc = RC.OK; // Result code from this function
            BtShared bt = null; // Shared part of btree structure
            MutexEx mutexOpen = null;
#if !OMIT_SHARED_CACHE && !OMIT_DISKIO
            // If this Btree is a candidate for shared cache, try to find an existing BtShared object that we can share with
            if (!tempDB && (!memoryDB || (vfsFlags & VSystem.OPEN.URI) != 0))
                if ((vfsFlags & VSystem.OPEN.SHAREDCACHE) != 0)
                {
                    string fullPathname;
                    p.Sharable_ = true;
                    if (memoryDB)
                        fullPathname = filename;
                    else
                        vfs.FullPathname(filename, out fullPathname);
                    MutexEx mutexShared;
#if THREADSAFE
                    mutexOpen = MutexEx.Alloc(MutexEx.MUTEX.STATIC_OPEN); // Prevents a race condition. Ticket #3537
                    MutexEx.Enter(mutexOpen);
                    mutexShared = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
                    MutexEx.Enter(mutexShared);
#endif
                    for (bt = _sharedCacheList; bt != null; bt = bt.Next)
                    {
                        Debug.Assert(bt.Refs > 0);
                        if (fullPathname == bt.Pager.get_Filename(false) && bt.Pager.get_Vfs() == vfs)
                        {
                            for (var i = ctx.DBs.length - 1; i >= 0; i--)
                            {
                                var existing = ctx.DBs[i].Bt;
                                if (existing != null && existing.Bt == bt)
                                {
                                    MutexEx.Leave(mutexShared);
                                    MutexEx.Leave(mutexOpen);
                                    fullPathname = null;
                                    p = null;
                                    return RC.CONSTRAINT;
                                }
                            }
                            p.Bt = bt;
                            bt.Refs++;
                            break;
                        }
                    }
                    MutexEx.Leave(mutexShared);
                    fullPathname = null;
                }
#if DEBUG
                else
                    // In debug mode, we mark all persistent databases as sharable even when they are not.  This exercises the locking code and
                    // gives more opportunity for asserts(sqlite3_mutex_held()) statements to find locking problems.
                    p.Sharable_ = true;
#endif
#endif

            byte reserves; // Byte of unused space on each page
            var dbHeader = new byte[100]; // Database header content
            if (bt == null)
            {
                // The following asserts make sure that structures used by the btree are the right size.  This is to guard against size changes that result
                // when compiling on a different architecture.
                Debug.Assert(sizeof(long) == 8 || sizeof(long) == 4);
                Debug.Assert(sizeof(ulong) == 8 || sizeof(ulong) == 4);
                Debug.Assert(sizeof(uint) == 4);
                Debug.Assert(sizeof(ushort) == 2);
                Debug.Assert(sizeof(Pid) == 4);

                bt = new BtShared();
                if (bt == null)
                {
                    rc = RC.NOMEM;
                    goto btree_open_out;
                }
                rc = Pager.Open(vfs, out bt.Pager, filename, EXTRA_SIZE, (IPager.PAGEROPEN)flags, vfsFlags, pageReinit, null);
                if (rc == RC.OK)
                    rc = bt.Pager.ReadFileHeader(dbHeader.Length, dbHeader);
                if (rc != RC.OK)
                    goto btree_open_out;
                bt.OpenFlags = flags;
                bt.Ctx = ctx;
                bt.Pager.SetBusyHandler(btreeInvokeBusyHandler, bt);
                p.Bt = bt;

                bt.Cursor = null;
                bt.Page1 = null;
                if (bt.Pager.get_Readonly()) bt.BtsFlags |= BTS.READ_ONLY;
#if SECURE_DELETE
                bt.BtsFlags |= BTS.SECURE_DELETE;
#endif
                bt.PageSize = (Pid)((dbHeader[16] << 8) | (dbHeader[17] << 16));
                if (bt.PageSize < 512 || bt.PageSize > Pager.MAX_PAGE_SIZE || ((bt.PageSize - 1) & bt.PageSize) != 0)
                {
                    bt.PageSize = 0;
#if !OMIT_AUTOVACUUM
                    // If the magic name ":memory:" will create an in-memory database, then leave the autoVacuum mode at 0 (do not auto-vacuum), even if
                    // SQLITE_DEFAULT_AUTOVACUUM is true. On the other hand, if SQLITE_OMIT_MEMORYDB has been defined, then ":memory:" is just a
                    // regular file-name. In this case the auto-vacuum applies as per normal.
                    if (filename != null && !memoryDB)
                    {
                        bt.AutoVacuum = (DEFAULT_AUTOVACUUM != 0);
                        bt.IncrVacuum = (DEFAULT_AUTOVACUUM == AUTOVACUUM.INCR);
                    }
#endif
                    reserves = 0;
                }
                else
                {
                    reserves = dbHeader[20];
                    bt.BtsFlags |= BTS.PAGESIZE_FIXED;
#if !OMIT_AUTOVACUUM
                    bt.AutoVacuum = (ConvertEx.Get4(dbHeader, 36 + 4 * 4) != 0);
                    bt.IncrVacuum = (ConvertEx.Get4(dbHeader, 36 + 7 * 4) != 0);
#endif
                }
                rc = bt.Pager.SetPageSize(ref bt.PageSize, reserves);
                if (rc != RC.OK) goto btree_open_out;
                bt.UsableSize = (ushort)(bt.PageSize - reserves);
                Debug.Assert((bt.PageSize & 7) == 0); // 8-byte alignment of pageSize

#if !SHARED_CACHE && !OMIT_DISKIO
                // Add the new BtShared object to the linked list sharable BtShareds.
                if (p.Sharable_)
                {
                    bt.Refs = 1;
                    MutexEx mutexShared;
#if THREADSAFE
                    mutexShared = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
                    bt.Mutex = MutexEx.Alloc(MutexEx.MUTEX.FAST);
#endif
                    MutexEx.Enter(mutexShared);
                    bt.Next = _sharedCacheList;
                    _sharedCacheList = bt;
                    MutexEx.Leave(mutexShared);
                }
#endif
            }

#if !OMIT_SHARED_CACHE && !OMIT_DISKIO
            // If the new Btree uses a sharable pBtShared, then link the new Btree into the list of all sharable Btrees for the same connection.
            // The list is kept in ascending order by pBt address.
            if (p.Sharable_)
            {
                Btree sib;
                for (var i = 0; i < ctx.DBs.length; i++)
                    if ((sib = ctx.DBs[i].Bt) != null && sib.Sharable_)
                    {
                        while (sib.Prev != null) { sib = sib.Prev; }
                        if (p.Bt.AutoID < sib.Bt.AutoID)
                        {
                            p.Next = sib;
                            p.Prev = null;
                            sib.Prev = p;
                        }
                        else
                        {
                            while (sib.Next != null && sib.Next.Bt.AutoID < p.Bt.AutoID)
                                sib = sib.Next;
                            p.Next = sib.Next;
                            p.Prev = sib;
                            if (p.Next != null)
                                p.Next.Prev = p;
                            sib.Next = p;
                        }
                        break;
                    }
            }
#endif
            btree = p;

        btree_open_out:
            if (rc != RC.OK)
            {
                if (bt != null && bt.Pager != null)
                    bt.Pager.Close();
                bt = null;
                p = null;
                btree = null;
            }
            else
                // If the B-Tree was successfully opened, set the pager-cache size to the default value. Except, when opening on an existing shared pager-cache,
                // do not change the pager-cache size.
                if (p.Schema(0, null) == null)
                    p.Bt.Pager.SetCacheSize(DEFAULT_CACHE_SIZE);
#if THREADSAFE
            Debug.Assert(MutexEx.Held(mutexOpen));
            MutexEx.Leave(mutexOpen);
#endif
            return rc;
        }

        static bool removeFromSharingList(BtShared bt)
        {
#if !OMIT_SHARED_CACHE
            Debug.Assert(MutexEx.Held(bt.Mutex));
#if THREADSAFE
            var master = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
            var removed = false;
            MutexEx.Enter(master);
            bt.Refs--;
            if (bt.Refs <= 0)
            {
                if (_sharedCacheList == bt)
                    _sharedCacheList = bt.Next;
                else
                {
                    var list = _sharedCacheList;
                    while (C._ALWAYS(list != null) && list.Next != bt)
                        list = list.Next;
                    if (C._ALWAYS(list != null))
                        list.Next = bt.Next;
                }
#if THREADSAFE
                MutexEx.Free(bt.Mutex);
#endif
                removed = true;
            }
            MutexEx.Leave(master);
            return removed;
#else
            return true;
#endif
        }

        static void allocateTempSpace(BtShared bt)
        {
            if (bt.TmpSpace == null)
                bt.TmpSpace = PCache.PageAlloc2((int)bt.PageSize);
        }

        static void freeTempSpace(BtShared bt)
        {
            PCache.PageFree2(ref bt.TmpSpace);
            bt.TmpSpace = null;
        }

        public RC Close()
        {
            // Close all cursors opened via this handle.
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            Enter();
            var bt = Bt;
            var cur = bt.Cursor;
            while (cur != null)
            {
                var tmp = cur;
                cur = cur.Next;
                if (tmp.Btree == this)
                    CloseCursor(tmp);
            }

            // Rollback any active transaction and free the handle structure. The call to sqlite3BtreeRollback() drops any table-locks held by this handle.
            Rollback(RC.OK);
            Leave();

            // If there are still other outstanding references to the shared-btree structure, return now. The remainder of this procedure cleans up the shared-btree.
            Debug.Assert(WantToLock == 0 && !Locked);
            if (!Sharable_ || removeFromSharingList(bt))
            {
                // The pBt is no longer on the sharing list, so we can access it without having to hold the mutex.
                //
                // Clean out and delete the BtShared object.
                Debug.Assert(bt.Cursor == null);
                bt.Pager.Close();
                if (bt.FreeSchema != null && bt.Schema != null)
                    bt.FreeSchema(bt.Schema);
                bt.Schema = null;
                freeTempSpace(bt);
                bt = null;
            }

#if !OMIT_SHARED_CACHE
            Debug.Assert(WantToLock == 0 && !Locked);
            if (Prev != null) Prev.Next = Next;
            if (Next != null) Next.Prev = Prev;
#endif

            return RC.OK;
        }

        #endregion

        #region Settings1

        public RC SetCacheSize(int maxPage)
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            Enter();
            Bt.Pager.SetCacheSize(maxPage);
            Leave();
            return RC.OK;
        }

#if !OMIT_PAGER_PRAGMAS
        public RC SetSafetyLevel(int level, bool fullSync, bool ckptFullSync)
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            Debug.Assert(level >= 1 && level <= 3);
            Enter();
            Bt.Pager.SetSafetyLevel(level, fullSync, ckptFullSync);
            Leave();
            return RC.OK;
        }
#endif

        public bool SyncDisabled()
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            Enter();
            Debug.Assert(Bt != null && Bt.Pager != null);
            var rc = Bt.Pager.get_NoSync();
            Leave();
            return rc;
        }

        public RC SetPageSize(int pageSize, int reserves, bool fix)
        {
            Debug.Assert(reserves >= -1 && reserves <= 255);
            Enter();
            BtShared bt = Bt;
            if ((bt.BtsFlags & BTS.PAGESIZE_FIXED) != 0)
            {
                Leave();
                return RC.READONLY;
            }
            if (reserves < 0)
                reserves = (int)(bt.PageSize - bt.UsableSize);
            Debug.Assert(reserves >= 0 && reserves <= 255);
            if (pageSize >= 512 && pageSize <= Pager.MAX_PAGE_SIZE && ((pageSize - 1) & pageSize) == 0)
            {
                Debug.Assert((pageSize & 7) == 0);
                Debug.Assert(bt.Page1 == null && bt.Cursor == null);
                bt.PageSize = (uint)pageSize;
                freeTempSpace(bt);
            }
            var rc = bt.Pager.SetPageSize(ref bt.PageSize, reserves);
            bt.UsableSize = (ushort)(bt.PageSize - reserves);
            if (fix) bt.BtsFlags |= BTS.PAGESIZE_FIXED;
            Leave();
            return rc;
        }

        public int GetPageSize()
        {
            return (int)Bt.PageSize;
        }

#if HAS_CODEC || _DEBUG
        public int GetReserveNoMutex()
        {
            Debug.Assert(MutexEx.Held(Bt.Mutex));
            return (int)(Bt.PageSize - Bt.UsableSize);
        }
#endif

#if !OMIT_PAGER_PRAGMAS || !OMIT_VACUUM
        public int GetReserve()
        {
            Enter();
            var n = (int)(Bt.PageSize - Bt.UsableSize);
            Leave();
            return n;
        }

        public int MaxPageCount(int maxPage)
        {
            Enter();
            var n = (int)Bt.Pager.MaxPages(maxPage);
            Leave();
            return n;
        }

        public bool SecureDelete(bool newFlag)
        {
            Enter();
            Bt.BtsFlags &= ~BTS.SECURE_DELETE;
            if (newFlag) Bt.BtsFlags |= BTS.SECURE_DELETE;
            bool b = (Bt.BtsFlags & BTS.SECURE_DELETE) != 0;
            Leave();
            return b;
        }
#endif

        public RC SetAutoVacuum(AUTOVACUUM autoVacuum)
        {
#if OMIT_AUTOVACUUM
            return RC.READONLY;
#else
            var rc = RC.OK;
            Enter();
            var bt = Bt;
            if ((bt.BtsFlags & BTS.PAGESIZE_FIXED) != 0 && (autoVacuum != 0) != bt.AutoVacuum)
                rc = RC.READONLY;
            else
            {
                bt.AutoVacuum = (autoVacuum != 0);
                bt.IncrVacuum = (autoVacuum == AUTOVACUUM.INCR);
            }
            Leave();
            return rc;
#endif
        }

        public AUTOVACUUM GetAutoVacuum()
        {
#if OMIT_AUTOVACUUM
            return AUTOVACUUM.NONE;
#else
            Enter();
            var bt = Bt;
            var rc = (!bt.AutoVacuum ? AUTOVACUUM.NONE :
                !bt.IncrVacuum ? AUTOVACUUM.FULL :
                AUTOVACUUM.INCR);
            Leave();
            return rc;
#endif
        }

        #endregion

        #region Lock / Unlock

        static RC lockBtree(BtShared bt)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(bt.Page1 == null);
            var rc = bt.Pager.SharedLock();
            if (rc != RC.OK) return rc;
            MemPage page1 = null; // Page 1 of the database file
            rc = btreeGetPage(bt, 1, ref page1, false);
            if (rc != RC.OK) return rc;

            // Do some checking to help insure the file we opened really is a valid database file. 
            Pid pagesHeader; // Number of pages in the database according to hdr
            Pid pages = pagesHeader = ConvertEx.Get4(page1.Data, 28); // Number of pages in the database
            Pid pagesFile = 0; // Number of pages in the database file
            bt.Pager.Pages(out pagesFile);
            if (pages == 0 || C.memcmp(page1.Data, 24, page1.Data, 92, 4) != 0)
                pages = pagesFile;
            if (pages > 0)
            {
                var page1Data = page1.Data;
                rc = RC.NOTADB;
                if (C.memcmp(page1Data, _magicHeader, 16) != 0)
                    goto page1_init_failed;

#if OMIT_WAL
                if (page1Data[18] > 1)
                    bt.BtsFlags |= BTS.READ_ONLY;
                if (page1Data[19] > 1)
                    goto page1_init_failed;
#else
                if (page1Data[18] > 2)
                    bt.BtsFlags |= BTS.READ_ONLY;
                if (page1Data[19] > 2)
                    goto page1_init_failed;

                // return SQLITE_OK and return without populating BtShared.pPage1. The caller detects this and calls this function again. This is
                // required as the version of page 1 currently in the page1 buffer may not be the latest version - there may be a newer one in the log file.
                if (page1Data[19] == 2 && (bt.BtsFlags & BTS.NO_WAL) == 0)
                {
                    int isOpen = 0;
                    rc = bt.Pager.OpenWal(ref isOpen);
                    if (rc != RC.OK)
                        goto page1_init_failed;
                    else if (isOpen == 0)
                    {
                        releasePage(page1);
                        return RC.OK;
                    }
                    rc = RC.NOTADB;
                }
#endif

                // The maximum embedded fraction must be exactly 25%.  And the minimum embedded fraction must be 12.5% for both leaf-data and non-leaf-data.
                // The original design allowed these amounts to vary, but as of version 3.6.0, we require them to be fixed.
                if (C.memcmp(page1Data, 21, "\x0040\x0020\x0020", 3) != 0) // "\100\040\040"
                    goto page1_init_failed;
                uint pageSize = (uint)((page1Data[16] << 8) | (page1Data[17] << 16));
                if (((pageSize - 1) & pageSize) != 0 ||
                    pageSize > Pager.MAX_PAGE_SIZE ||
                    pageSize <= 256)
                    goto page1_init_failed;
                Debug.Assert((pageSize & 7) == 0);
                uint usableSize = pageSize - page1Data[20];
                if (pageSize != bt.PageSize)
                {
                    // After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
                    // actually pageSize. Unlock the database, leave pBt->pPage1 at zero and return SQLITE_OK. The caller will call this function
                    // again with the correct page-size.
                    releasePage(page1);
                    bt.UsableSize = usableSize;
                    bt.PageSize = pageSize;
                    freeTempSpace(bt);
                    rc = bt.Pager.SetPageSize(ref bt.PageSize, (int)(pageSize - usableSize));
                    return rc;
                }
                if ((bt.Ctx.Flags & BContext.FLAG.RecoveryMode) == 0 && pages > pagesFile)
                {
                    rc = SysEx.CORRUPT_BKPT();
                    goto page1_init_failed;
                }
                if (usableSize < 480)
                    goto page1_init_failed;
                bt.PageSize = pageSize;
                bt.UsableSize = usableSize;
#if !OMIT_AUTOVACUUM
                bt.AutoVacuum = (ConvertEx.Get4(page1Data, 36 + 4 * 4) != 0);
                bt.IncrVacuum = (ConvertEx.Get4(page1Data, 36 + 7 * 4) != 0);
#endif
            }

            // maxLocal is the maximum amount of payload to store locally for a cell.  Make sure it is small enough so that at least minFanout
            // cells can will fit on one page.  We assume a 10-byte page header. Besides the payload, the cell must store:
            //     2-byte pointer to the cell
            //     4-byte child pointer
            //     9-byte nKey value
            //     4-byte nData value
            //     4-byte overflow page pointer
            // So a cell consists of a 2-byte pointer, a header which is as much as 17 bytes long, 0 to N bytes of payload, and an optional 4 byte overflow
            // page pointer.
            bt.MaxLocal = (ushort)((bt.UsableSize - 12) * 64 / 255 - 23);
            bt.MinLocal = (ushort)((bt.UsableSize - 12) * 32 / 255 - 23);
            bt.MaxLeaf = (ushort)(bt.UsableSize - 35);
            bt.MinLeaf = (ushort)((bt.UsableSize - 12) * 32 / 255 - 23);
            Debug.Assert(bt.MaxLeaf + 23 <= MX_CELL_SIZE(bt));
            bt.Page1 = page1;
            bt.Pages = pages;
            return RC.OK;

        page1_init_failed:
            releasePage(page1);
            bt.Page1 = null;
            return rc;
        }

        static void unlockBtreeIfUnused(BtShared bt)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(bt.Cursor == null || bt.InTransaction > TRANS.NONE);
            if (bt.InTransaction == TRANS.NONE && bt.Page1 != null)
            {
                Debug.Assert(bt.Page1.Data != null);
                Debug.Assert(bt.Pager.get_Refs() == 1);
                releasePage(bt.Page1);
                bt.Page1 = null;
            }
        }

        #endregion

        #region NewDB

        static RC newDatabase(BtShared bt)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            if (bt.Pages > 0)
                return RC.OK;
            var p1 = bt.Page1;
            Debug.Assert(p1 != null);
            var data = p1.Data;
            var rc = Pager.Write(p1.DBPage);
            if (rc != RC.OK) return rc;
            Buffer.BlockCopy(_magicHeader, 0, data, 0, _magicHeader.Length);
            Debug.Assert(_magicHeader.Length == 16);
            data[16] = (byte)((bt.PageSize >> 8) & 0xff);
            data[17] = (byte)((bt.PageSize >> 16) & 0xff);
            data[18] = 1;
            data[19] = 1;
            Debug.Assert(bt.UsableSize <= bt.PageSize && bt.UsableSize + 255 >= bt.PageSize);
            data[20] = (byte)(bt.PageSize - bt.UsableSize);
            data[21] = 64;
            data[22] = 32;
            data[23] = 32;
            //_memset(&data[24], 0, 100 - 24);
            zeroPage(p1, PTF_INTKEY | PTF_LEAF | PTF_LEAFDATA);
            bt.BtsFlags |= BTS.PAGESIZE_FIXED;
#if !SQLITE_OMIT_AUTOVACUUM
            ConvertEx.Put4(data, 36 + 4 * 4, bt.AutoVacuum ? 1 : 0);
            ConvertEx.Put4(data, 36 + 7 * 4, bt.IncrVacuum ? 1 : 0);
#endif
            bt.Pages = 1;
            data[31] = 1;
            return RC.OK;
        }

        public RC NewDb()
        {
            Enter();
            Bt.Pages = 0;
            RC rc = newDatabase(Bt);
            Leave();
            return rc;
        }

        #endregion

        #region Transactions

        public RC BeginTrans(int wrflag)
        {
            Enter();
            btreeIntegrity(this);

            // If the btree is already in a write-transaction, or it is already in a read-transaction and a read-transaction
            // is requested, this is a no-op.
            var bt = Bt;
            RC rc = RC.OK;
            if (InTrans == TRANS.WRITE || (InTrans == TRANS.READ && wrflag == 0))
                goto trans_begun;
            Debug.Assert(!IFAUTOVACUUM(Bt.DoTruncate));

            // Write transactions are not possible on a read-only database
            if ((bt.BtsFlags & BTS.READ_ONLY) != 0 && wrflag != 0)
            {
                rc = RC.READONLY;
                goto trans_begun;
            }

#if !OMIT_SHARED_CACHE
            // If another database handle has already opened a write transaction on this shared-btree structure and a second write transaction is
            // requested, return SQLITE_LOCKED.
            BContext blockingCtx = null;
            if ((wrflag != 0 && bt.InTransaction == TRANS.WRITE) || (bt.BtsFlags & BTS.PENDING) != 0)
                blockingCtx = bt.Writer.Ctx;
            else if (wrflag > 1)
            {
                for (var iter = bt.Lock; iter != null; iter = iter.Next)
                    if (iter.Btree != this)
                    {
                        blockingCtx = iter.Btree.Ctx;
                        break;
                    }
            }

            if (blockingCtx != null)
            {
                BContext.ConnectionBlocked(Ctx, blockingCtx);
                rc = RC.LOCKED_SHAREDCACHE;
                goto trans_begun;
            }
#endif

            // Any read-only or read-write transaction implies a read-lock on page 1. So if some other shared-cache client already has a write-lock 
            // on page 1, the transaction cannot be opened. */
            rc = querySharedCacheTableLock(this, MASTER_ROOT, LOCK.READ);
            if (rc != RC.OK) goto trans_begun;

            bt.BtsFlags &= ~BTS.INITIALLY_EMPTY;
            if (bt.Pages == 0) bt.BtsFlags |= BTS.INITIALLY_EMPTY;
            do
            {
                // Call lockBtree() until either pBt->pPage1 is populated or lockBtree() returns something other than SQLITE_OK. lockBtree()
                // may return SQLITE_OK but leave pBt->pPage1 set to 0 if after reading page 1 it discovers that the page-size of the database 
                // file is not pBt->pageSize. In this case lockBtree() will update pBt->pageSize to the page-size of the file on disk.
                while (bt.Page1 == null && (rc = lockBtree(bt)) == RC.OK) ;

                if (rc == RC.OK && wrflag != 0)
                {
                    if ((bt.BtsFlags & BTS.READ_ONLY) != 0)
                        rc = RC.READONLY;
                    else
                    {
                        rc = bt.Pager.Begin(wrflag > 1, Ctx.TempInMemory());
                        if (rc == RC.OK)
                            rc = newDatabase(bt);
                    }
                }

                if (rc != RC.OK)
                    unlockBtreeIfUnused(bt);
            } while (((int)rc & 0xFF) == (int)RC.BUSY && bt.InTransaction == TRANS.NONE && btreeInvokeBusyHandler(bt) != 0);

            if (rc == RC.OK)
            {
                if (InTrans == TRANS.NONE)
                {
                    bt.Transactions++;
#if !OMIT_SHARED_CACHE
                    if (Sharable_)
                    {
                        Debug.Assert(Lock.Btree == this && Lock.Table == 1);
                        Lock.Lock = LOCK.READ;
                        Lock.Next = bt.Lock;
                        bt.Lock = Lock;
                    }
#endif
                }
                InTrans = (wrflag != 0 ? TRANS.WRITE : TRANS.READ);
                if (InTrans > bt.InTransaction)
                    bt.InTransaction = InTrans;
                if (wrflag != 0)
                {
                    var page1 = bt.Page1;
#if !OMIT_SHARED_CACHE
                    Debug.Assert(bt.Writer == null);
                    bt.Writer = this;
                    bt.BtsFlags &= ~BTS.EXCLUSIVE;
                    if (wrflag > 1) bt.BtsFlags |= BTS.EXCLUSIVE;
#endif

                    // If the db-size header field is incorrect (as it may be if an old client has been writing the database file), update it now. Doing
                    // this sooner rather than later means the database size can safely re-read the database size from page 1 if a savepoint or transaction
                    // rollback occurs within the transaction.
                    if (bt.Pages != ConvertEx.Get4(page1.Data, 28))
                    {
                        rc = Pager.Write(page1.DBPage);
                        if (rc == RC.OK)
                            ConvertEx.Put4(page1.Data, 28, bt.Pages);
                    }
                }
            }

        trans_begun:
            if (rc == RC.OK && wrflag != 0)
            {
                // This call makes sure that the pager has the correct number of open savepoints. If the second parameter is greater than 0 and
                // the sub-journal is not already open, then it will be opened here.
                rc = bt.Pager.OpenSavepoint(Ctx.Savepoints);
            }

            btreeIntegrity(this);
            Leave();
            return rc;
        }

        #endregion

        #region Autovacuum
#if !SQLITE_OMIT_AUTOVACUUM

        static RC setChildPtrmaps(MemPage page)
        {
            var isInitOrig = page.IsInit;

            var bt = page.Bt;
            Debug.Assert(MutexEx.Held(bt.Mutex));
            var rc = btreeInitPage(page);
            if (rc != RC.OK)
                goto set_child_ptrmaps_out;
            var cells = page.Cells; // Number of cells in page pPage

            var id = page.ID;
            for (var i = 0U; i < cells; i++)
            {
                uint cell_ = findCell(page, i);
                ptrmapPutOvflPtr(page, cell_, ref rc);
                if (!page.Leaf)
                {
                    Pid childID = ConvertEx.Get4(page.Data, cell_);
                    ptrmapPut(bt, childID, PTRMAP.BTREE, id, ref rc);
                }
            }

            if (!page.Leaf)
            {
                Pid childID = ConvertEx.Get4(page.Data, page.HdrOffset + 8);
                ptrmapPut(bt, childID, PTRMAP.BTREE, id, ref rc);
            }

        set_child_ptrmaps_out:
            page.IsInit = isInitOrig;
            return rc;
        }

        static RC modifyPagePointer(MemPage page, Pid from, Pid to, PTRMAP type)
        {
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            if (type == PTRMAP.OVERFLOW2)
            {
                // The pointer is always the first 4 bytes of the page in this case.
                if (ConvertEx.Get4(page.Data) != from)
                    return SysEx.CORRUPT_BKPT();
                ConvertEx.Put4(page.Data, to);
            }
            else
            {
                var isInitOrig = page.IsInit;

                btreeInitPage(page);
                ushort cells = page.Cells;

                uint i;
                for (i = 0U; i < cells; i++)
                {
                    uint cell_ = findCell(page, i);
                    if (type == PTRMAP.OVERFLOW1)
                    {
                        var info = new CellInfo();
                        btreeParseCellPtr(page, cell_, ref info);
                        if (info.Overflow != 0 &&
                            //cell + info.Overflow + 3 <= page.Data + page.MaskPage &&
                            from == ConvertEx.Get4(page.Data, cell_ + info.Overflow))
                        {
                            ConvertEx.Put4(page.Data, cell_ + info.Overflow, (int)to);
                            break;
                        }
                    }
                    else
                        if (ConvertEx.Get4(page.Data, cell_) == from)
                        {
                            ConvertEx.Put4(page.Data, cell_, (int)to);
                            break;
                        }
                }

                if (i == cells)
                {
                    if (type != PTRMAP.BTREE || ConvertEx.Get4(page.Data, page.HdrOffset + 8) != from)
                        return SysEx.CORRUPT_BKPT();
                    ConvertEx.Put4(page.Data, page.HdrOffset + 8, to);
                }

                page.IsInit = isInitOrig;
            }
            return RC.OK;
        }

        static RC relocatePage(BtShared bt, MemPage page, PTRMAP type, Pid ptrPageID, Pid freePageID, bool isCommit)
        {
            Debug.Assert(type == PTRMAP.OVERFLOW2 || type == PTRMAP.OVERFLOW1 || type == PTRMAP.BTREE || type == PTRMAP.ROOTPAGE);
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(page.Bt == bt);

            // Move page iDbPage from its current location to page number iFreePage
            var lastID = page.ID;
            TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", lastID, freePageID, ptrPageID, type);
            Pager pager = bt.Pager;
            var rc = pager.Movepage(page.DBPage, freePageID, isCommit);
            if (rc != RC.OK)
                return rc;
            page.ID = freePageID;

            // If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
            // pages need to be changed.
            //
            // If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
            // the pointer map needs to be updated for the subsequent overflow page.
            if (type == PTRMAP.BTREE || type == PTRMAP.ROOTPAGE)
            {
                rc = setChildPtrmaps(page);
                if (rc != RC.OK)
                    return rc;
            }
            else
            {
                Pid nextOvfl = ConvertEx.Get4(page.Data);
                if (nextOvfl != 0)
                {
                    ptrmapPut(bt, nextOvfl, PTRMAP.OVERFLOW2, freePageID, ref rc);
                    if (rc != RC.OK)
                        return rc;
                }
            }

            // Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
            if (type != PTRMAP.ROOTPAGE)
            {
                var ptrPage = new MemPage(); // The page that contains a pointer to pDbPage
                rc = btreeGetPage(bt, ptrPageID, ref ptrPage, false);
                if (rc != RC.OK)
                    return rc;
                rc = Pager.Write(ptrPage.DBPage);
                if (rc != RC.OK)
                {
                    releasePage(ptrPage);
                    return rc;
                }
                rc = modifyPagePointer(ptrPage, lastID, freePageID, type);
                releasePage(ptrPage);
                if (rc == RC.OK)
                    ptrmapPut(bt, freePageID, type, ptrPageID, ref rc);
            }
            return rc;
        }

        static RC incrVacuumStep(BtShared bt, Pid fins, Pid lastPageID, bool commit)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(lastPageID > fins);

            if (!PTRMAP_ISPAGE(bt, lastPageID) && lastPageID != PENDING_BYTE_PAGE(bt))
            {
                Pid freesList = ConvertEx.Get4(bt.Page1.Data, 36); // Number of pages still on the free-list
                if (freesList == 0)
                    return RC.DONE;

                PTRMAP type = 0;
                Pid ptrPageID = 0;
                var rc = ptrmapGet(bt, lastPageID, ref type, ref ptrPageID);
                if (rc != RC.OK)
                    return rc;
                if (type == PTRMAP.ROOTPAGE)
                    return SysEx.CORRUPT_BKPT();

                if (type == PTRMAP.FREEPAGE)
                {
                    if (!commit)
                    {
                        // Remove the page from the files free-list. This is not required if bCommit is non-zero. In that case, the free-list will be
                        // truncated to zero after this function returns, so it doesn't matter if it still contains some garbage entries.
                        Pid freePageID = 0;
                        var freePage = new MemPage();
                        rc = allocateBtreePage(bt, ref freePage, ref freePageID, lastPageID, BTALLOC.EXACT);
                        if (rc != RC.OK)
                            return rc;
                        Debug.Assert(freePageID == lastPageID);
                        releasePage(freePage);
                    }
                }
                else
                {
                    MemPage lastPage = new MemPage();
                    rc = btreeGetPage(bt, lastPageID, ref lastPage, false);
                    if (rc != RC.OK)
                        return rc;

                    // If bCommit is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
                    //
                    // On the other hand, if bCommit is greater than zero, then keep looping until a free-page located within the first nFin pages
                    // of the file is found.
                    BTALLOC mode = BTALLOC.ANY; // Mode parameter for allocateBtreePage()
                    Pid nearID = 0; // nearby parameter for allocateBtreePage()
                    if (!commit)
                    {
                        mode = BTALLOC.LE;
                        nearID = fins;
                    }
                    Pid freePageID = 0; // Index of free page to move pLastPg to
                    do
                    {
                        MemPage freePage = new MemPage();
                        rc = allocateBtreePage(bt, ref freePage, ref freePageID, nearID, mode);
                        if (rc != RC.OK)
                        {
                            releasePage(lastPage);
                            return rc;
                        }
                        releasePage(freePage);
                    } while (commit && freePageID > fins);
                    Debug.Assert(freePageID < lastPageID);

                    rc = relocatePage(bt, lastPage, type, ptrPageID, freePageID, commit);
                    releasePage(lastPage);
                    if (rc != RC.OK)
                        return rc;
                }
            }

            if (!commit)
            {
                do
                {
                    lastPageID--;
                } while (lastPageID == PENDING_BYTE_PAGE(bt) || PTRMAP_ISPAGE(bt, lastPageID));
                bt.DoTruncate = true;
                bt.Pages = lastPageID;
            }
            return RC.OK;
        }

        static Pid finalDbSize(BtShared bt, Pid origs, Pid frees)
        {
            int entrys = (int)(bt.UsableSize / 5); // Number of entries on one ptrmap page
            Pid ptrmaps = (Pid)((frees - origs + PTRMAP_PAGENO(bt, origs) + entrys) / entrys); // Number of PtrMap pages to be freed
            Pid fins = origs - frees - ptrmaps; // Return value
            if (origs > PENDING_BYTE_PAGE(bt) && fins < PENDING_BYTE_PAGE(bt))
                fins--;
            while (PTRMAP_ISPAGE(bt, fins) || fins == PENDING_BYTE_PAGE(bt))
                fins--;
            return fins;
        }

        public RC IncrVacuum()
        {
            var bt = Bt;

            Enter();
            Debug.Assert(bt.InTransaction == TRANS.WRITE && InTrans == TRANS.WRITE);
            RC rc;
            if (!bt.AutoVacuum)
                rc = RC.DONE;
            else
            {
                Pid origs = btreePagecount(bt);
                Pid frees = ConvertEx.Get4(bt.Page1.Data, 36);
                Pid fins = finalDbSize(bt, origs, frees);

                if (origs < fins)
                    rc = SysEx.CORRUPT_BKPT();
                else if (frees > 0)
                {
                    invalidateAllOverflowCache(bt);
                    rc = incrVacuumStep(bt, fins, origs, false);
                    if (rc == RC.OK)
                    {
                        rc = Pager.Write(bt.Page1.DBPage);
                        ConvertEx.Put4(bt.Page1.Data, 28, bt.Pages);
                    }
                }
                else
                    rc = RC.DONE;
            }
            Leave();
            return rc;
        }

        static RC autoVacuumCommit(BtShared bt)
        {
            var pager = bt.Pager;
#if DEBUG
            int refs = pager.get_Refs();
#endif

            Debug.Assert(MutexEx.Held(bt.Mutex));
            invalidateAllOverflowCache(bt);
            Debug.Assert(bt.AutoVacuum);
            var rc = RC.OK;
            if (!bt.IncrVacuum)
            {
                Pid origs = btreePagecount(bt); // Database size before freeing
                if (PTRMAP_ISPAGE(bt, origs) || origs == PENDING_BYTE_PAGE(bt))
                {
                    // It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
                    // is encountered, this indicates corruption.
                    return SysEx.CORRUPT_BKPT();
                }

                Pid frees = ConvertEx.Get4(bt.Page1.Data, 36); // Number of pages on the freelist initially
                Pid fins = finalDbSize(bt, origs, frees); // Number of pages in database after autovacuuming
                if (fins > origs) return SysEx.CORRUPT_BKPT();

                for (var freeID = origs; freeID > fins && rc == RC.OK; freeID--) // The next page to be freed
                    rc = incrVacuumStep(bt, fins, freeID, true);
                if ((rc == RC.DONE || rc == RC.OK) && frees > 0)
                {
                    rc = Pager.Write(bt.Page1.DBPage);
                    ConvertEx.Put4(bt.Page1.Data, 32, 0);
                    ConvertEx.Put4(bt.Page1.Data, 36, 0);
                    ConvertEx.Put4(bt.Page1.Data, 28, fins);
                    bt.DoTruncate = true;
                    bt.Pages = fins;
                }
                if (rc != RC.OK)
                    pager.Rollback();
            }
#if DEBUG
            Debug.Assert(refs == pager.get_Refs());
#endif
            return rc;
        }

#else
//# define setChildPtrmaps(x) RC_OK
#endif
        #endregion

        #region Commit / Rollback

        public RC CommitPhaseOne(string master)
        {
            var rc = RC.OK;
            if (InTrans == TRANS.WRITE)
            {
                var bt = Bt;
                Enter();
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum)
                {
                    rc = autoVacuumCommit(bt);
                    if (rc != RC.OK)
                    {
                        Leave();
                        return rc;
                    }
                }
                if (bt.DoTruncate)
                    bt.Pager.TruncateImage(bt.Pages);
#endif
                rc = bt.Pager.CommitPhaseOne(master, false);
                Leave();
            }
            return rc;
        }

        static void btreeEndTransaction(Btree p)
        {
            var bt = p.Bt;
            Debug.Assert(p.HoldsMutex());

#if !OMIT_AUTOVACUUM
            bt.DoTruncate = false;
#endif
            btreeClearHasContent(bt);
            if (p.InTrans > TRANS.NONE && p.Ctx.ActiveVdbeCnt > 1)
            {
                // If there are other active statements that belong to this database handle, downgrade to a read-only transaction. The other statements
                // may still be reading from the database.
                downgradeAllSharedCacheTableLocks(p);
                p.InTrans = TRANS.READ;
            }
            else
            {
                // If the handle had any kind of transaction open, decrement the transaction count of the shared btree. If the transaction count 
                // reaches 0, set the shared state to TRANS_NONE. The unlockBtreeIfUnused() call below will unlock the pager.
                if (p.InTrans != TRANS.NONE)
                {
                    clearAllSharedCacheTableLocks(p);
                    bt.Transactions--;
                    if (bt.Transactions == 0)
                        bt.InTransaction = TRANS.NONE;
                }

                // Set the current transaction state to TRANS_NONE and unlock the  pager if this call closed the only read or write transaction.
                p.InTrans = TRANS.NONE;
                unlockBtreeIfUnused(bt);
            }

            btreeIntegrity(p);
        }

        public RC CommitPhaseTwo(bool cleanup)
        {
            if (InTrans == TRANS.NONE) return RC.OK;
            Enter();
            btreeIntegrity(this);

            // If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
            if (InTrans == TRANS.WRITE)
            {
                var bt = Bt;
                Debug.Assert(bt.InTransaction == TRANS.WRITE);
                Debug.Assert(bt.Transactions > 0);
                var rc = bt.Pager.CommitPhaseTwo();
                if (rc != RC.OK && !cleanup)
                {
                    Leave();
                    return rc;
                }
                bt.InTransaction = TRANS.READ;
            }

            btreeEndTransaction(this);
            Leave();
            return RC.OK;
        }

        public RC Commit()
        {
            Enter();
            var rc = CommitPhaseOne(null);
            if (rc == RC.OK)
                rc = CommitPhaseTwo(false);
            Leave();
            return rc;
        }

#if DEBUG
        static int countWriteCursors(BtShared bt)
        {
            int r = 0;
            for (var cur = bt.Cursor; cur != null; cur = cur.Next)
                if (cur.WrFlag && cur.State != CURSOR.FAULT) r++;
            return r;
        }
#endif

        public void TripAllCursors(RC errCode)
        {
            Enter();
            for (var p = Bt.Cursor; p != null; p = p.Next)
            {
                ClearCursor(p);
                p.State = CURSOR.FAULT;
                p.SkipNext = (int)errCode;
                for (var i = 0; i <= p.ID; i++)
                {
                    releasePage(p.Pages[i]);
                    p.Pages[i] = null;
                }
            }
            Leave();
        }

        public RC Rollback(RC tripCode)
        {
            var bt = Bt;

            Enter();
            RC rc;
            if (tripCode == RC.OK)
                rc = tripCode = saveAllCursors(bt, 0, null);
            else
                rc = RC.OK;
            if (tripCode != RC.OK)
                TripAllCursors(tripCode);

            btreeIntegrity(this);

            if (InTrans == TRANS.WRITE)
            {
                Debug.Assert(bt.InTransaction == TRANS.WRITE);
                var rc2 = bt.Pager.Rollback();
                if (rc2 != RC.OK)
                    rc = rc2;

                // The rollback may have destroyed the pPage1->aData value. So call btreeGetPage() on page 1 again to make
                // sure pPage1->aData is set correctly.
                MemPage page1 = new MemPage();
                if (btreeGetPage(bt, 1, ref page1, false) == RC.OK)
                {
                    Pid pages = ConvertEx.Get4(page1.Data, 28);
                    if (pages == 0) bt.Pager.Pages(out pages);
                    bt.Pages = pages;
                    releasePage(page1);
                }
                Debug.Assert(countWriteCursors(bt) == 0);
                bt.InTransaction = TRANS.READ;
            }

            btreeEndTransaction(this);
            Leave();
            return rc;
        }

        public RC BeginStmt(int statements)
        {
            BtShared bt = Bt;
            Enter();
            Debug.Assert(InTrans == TRANS.WRITE);
            Debug.Assert((bt.BtsFlags & BTS.READ_ONLY) == 0);
            Debug.Assert(statements > 0);
            Debug.Assert(statements > Ctx.Savepoints);
            Debug.Assert(bt.InTransaction == TRANS.WRITE);
            // At the pager level, a statement transaction is a savepoint with an index greater than all savepoints created explicitly using
            // SQL statements. It is illegal to open, release or rollback any such savepoints while the statement transaction savepoint is active.
            var rc = bt.Pager.OpenSavepoint(statements);
            Leave();
            return rc;
        }

        public RC Savepoint(IPager.SAVEPOINT op, int savepoints)
        {
            var rc = RC.OK;
            if (InTrans == TRANS.WRITE)
            {
                BtShared bt = Bt;
                Debug.Assert(op == IPager.SAVEPOINT.RELEASE || op == IPager.SAVEPOINT.ROLLBACK);
                Debug.Assert(savepoints >= 0 || (savepoints == -1 && op == IPager.SAVEPOINT.ROLLBACK));
                Enter();
                rc = bt.Pager.Savepoint(op, savepoints);
                if (rc == RC.OK)
                {
                    if (savepoints < 0 && (bt.BtsFlags & BTS.INITIALLY_EMPTY) != 0)
                        bt.Pages = 0;
                    rc = newDatabase(bt);
                    bt.Pages = ConvertEx.Get4(bt.Page1.Data, 28);

                    // The database size was written into the offset 28 of the header when the transaction started, so we know that the value at offset
                    // 28 is nonzero.
                    Debug.Assert(bt.Pages > 0);
                }
                Leave();
            }
            return rc;
        }

        #endregion

        #region Cursors

        static RC btreeCursor(Btree p, Pid tableID, bool wrFlag, KeyInfo keyInfo, BtCursor cur)
        {
            var bt = p.Bt; // Shared b-tree handle

            Debug.Assert(p.HoldsMutex());

            // The following assert statements verify that if this is a sharable b-tree database, the connection is holding the required table locks, 
            // and that no other connection has any open cursor that conflicts with this lock.
            Debug.Assert(hasSharedCacheTableLock(p, (uint)tableID, keyInfo != null, (LOCK)(wrFlag ? 1 : 0) + 1));
            Debug.Assert(!wrFlag || !hasReadConflicts(p, (uint)tableID));

            // Assert that the caller has opened the required transaction.
            Debug.Assert(p.InTrans > TRANS.NONE);
            Debug.Assert(!wrFlag || p.InTrans == TRANS.WRITE);
            Debug.Assert(bt.Page1 != null && bt.Page1.Data != null);

            if (C._NEVER(wrFlag && (bt.BtsFlags & BTS.READ_ONLY) != 0))
                return RC.READONLY;
            if (tableID == 1 && btreePagecount(bt) == 0)
            {
                Debug.Assert(!wrFlag);
                tableID = 0;
            }

            // Now that no other errors can occur, finish filling in the BtCursor variables and link the cursor into the BtShared list.
            cur.RootID = tableID;
            cur.ID = -1;
            cur.KeyInfo = keyInfo;
            cur.Btree = p;
            cur.Bt = bt;
            cur.WrFlag = wrFlag;
            cur.Next = bt.Cursor;
            if (cur.Next != null)
                cur.Next.Prev = cur;
            bt.Cursor = cur;
            cur.State = CURSOR.INVALID;
            cur.CachedRowID = 0;
            return RC.OK;
        }

        public RC Cursor(Pid tableID, bool wrFlag, KeyInfo keyInfo, BtCursor cur)
        {
            Enter();
            var rc = btreeCursor(this, tableID, wrFlag, keyInfo, cur);
            Leave();
            return rc;
        }

        public static int CursorSize()
        {
            return -1; // Not Used
        }

        public static void CursorZero(BtCursor p)
        {
            p.memset();
        }

        public static void SetCachedRowid(BtCursor cur, long rowid)
        {
            for (var p = cur.Bt.Cursor; p != null; p = p.Next)
                if (p.RootID == cur.RootID)
                    p.CachedRowID = rowid;
            Debug.Assert(cur.CachedRowID == rowid);
        }

        public static long GetCachedRowid(BtCursor cur)
        {
            return cur.CachedRowID;
        }

        public static RC CloseCursor(BtCursor cur)
        {
            var btree = cur.Btree;
            if (btree != null)
            {
                var bt = cur.Bt;
                btree.Enter();
                ClearCursor(cur);
                if (cur.Prev != null)
                    cur.Prev.Next = cur.Next;
                else
                    bt.Cursor = cur.Next;
                if (cur.Next != null)
                    cur.Next.Prev = cur.Prev;
                for (var i = 0; i <= cur.ID; i++)
                    releasePage(cur.Pages[i]);
                unlockBtreeIfUnused(bt);
                invalidateOverflowCache(cur);
                btree.Leave();
            }
            return RC.OK;
        }

#if DEBUG
        static void assertCellInfo(BtCursor cur)
        {
            int id = cur.ID;
            var info = new CellInfo();
            btreeParseCell(cur.Pages[id], cur.Idxs[id], ref info);
            Debug.Assert(info.GetHashCode() == cur.Info.GetHashCode() || info.Equals(cur.Info)); // _memcmp(info, cur.Info, sizeof(info)) == 0);
        }
#else
        static void assertCellInfo(BtCursor cur) { }
#endif
        static void getCellInfo(BtCursor cur)
        {
            if (cur.Info.Size == 0)
            {
                int id = cur.ID;
                btreeParseCell(cur.Pages[id], cur.Idxs[id], ref cur.Info);
                cur.ValidNKey = true;
            }
            else
                assertCellInfo(cur);
        }

#if DEBUG
        public static bool CursorIsValid(BtCursor cur)
        {
            return cur != null && cur.State == CURSOR.VALID;
        }
#endif

        public static RC KeySize(BtCursor cur, ref long size)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.INVALID || cur.State == CURSOR.VALID);
            if (cur.State != CURSOR.VALID)
                size = 0;
            else
            {
                getCellInfo(cur);
                size = cur.Info.Key;
            }
            return RC.OK;
        }

        public static RC DataSize(BtCursor cur, ref uint size)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            getCellInfo(cur);
            size = cur.Info.Data;
            return RC.OK;
        }

        #endregion

        #region Payload / Overflow

        static RC getOverflowPage(BtShared bt, Pid ovfl, out MemPage pageOut, out Pid idNextOut)
        {
            Pid next = 0;
            MemPage page = null;
            pageOut = null;
            var rc = RC.OK;

            Debug.Assert(MutexEx.Held(bt.Mutex));

#if !OMIT_AUTOVACUUM
            // Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in 
            // the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page 
            // number ovfl to determine the next page number.
            if (bt.AutoVacuum)
            {
                Pid guess = ovfl + 1;

                while (PTRMAP_ISPAGE(bt, guess) || guess == PENDING_BYTE_PAGE(bt))
                    guess++;
                if (guess <= btreePagecount(bt))
                {
                    Pid id = 0;
                    PTRMAP type = 0;
                    rc = ptrmapGet(bt, guess, ref type, ref id);
                    if (rc == RC.OK && type == PTRMAP.OVERFLOW2 && id == ovfl)
                    {
                        next = guess;
                        rc = RC.DONE;
                    }
                }
            }
#endif

            Debug.Assert(next == 0 || rc == RC.DONE);
            if (rc == RC.OK)
            {
                rc = btreeGetPage(bt, ovfl, ref page, false);
                Debug.Assert(rc == RC.OK || page == null);
                if (rc == RC.OK)
                    next = ConvertEx.Get4(page.Data);
            }

            idNextOut = next;
            if (pageOut != null)
                pageOut = page;
            else
                releasePage(page);
            return (rc == RC.DONE ? RC.OK : rc);
        }

        static RC copyPayload(byte[] payload, uint payloadOffset, byte[] buf, uint bufOffset, uint bytes, int op, IPage dbPage)
        {
            if (op != 0)
            {
                // Copy data from buffer to page (a write operation)
                var rc = Pager.Write(dbPage);
                if (rc != RC.OK)
                    return rc;
                Buffer.BlockCopy(buf, (int)bufOffset, payload, (int)payloadOffset, (int)bytes);
            }
            else
                // Copy data from page to buffer (a read operation)
                Buffer.BlockCopy(payload, (int)payloadOffset, buf, (int)bufOffset, (int)bytes);
            return RC.OK;
        }

        static RC accessPayload(BtCursor cur, uint offset, uint amount, byte[] buf, int op)
        {
            MemPage page = cur.Pages[cur.ID]; // Btree page of current entry

            Debug.Assert(page != null);
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cur.Idxs[cur.ID] < page.Cells);
            Debug.Assert(cursorHoldsMutex(cur));

            getCellInfo(cur);
            var payload = cur.Info.Cell; //cur.Info.Cell + cur.Info.Header;
            var key = (uint)(page.IntKey ? 0 : (int)cur.Info.Key);

            BtShared bt = cur.Bt; // Btree this cursor belongs to
            if (C._NEVER(offset + amount > key + cur.Info.Data) || cur.Info.Local > bt.UsableSize)
                // Trying to read or write past the end of the data is an error
                return SysEx.CORRUPT_BKPT();

            // Check if data must be read/written to/from the btree page itself.
            var idx = 0U;
            var rc = RC.OK;
            var buf_ = 0U;
            if (offset < cur.Info.Local)
            {
                int a = (int)amount;
                if (a + offset > cur.Info.Local)
                    a = (int)(cur.Info.Local - offset);
                rc = copyPayload(payload, (uint)(offset + cur.Info.Cell_ + cur.Info.Header), buf, buf_, (uint)a, op, page.DBPage);
                offset = 0;
                buf_ += (uint)a;
                amount -= (uint)a;
            }
            else
                offset -= cur.Info.Local;

            if (rc == RC.OK && amount > 0)
            {
                var ovflSize = (uint)(bt.UsableSize - 4); // Bytes content per ovfl page
                Pid nextPage = ConvertEx.Get4(payload, cur.Info.Local + cur.Info.Cell_ + cur.Info.Header);

#if !OMIT_INCRBLOB
                // If the isIncrblobHandle flag is set and the BtCursor.aOverflow[] has not been allocated, allocate it now. The array is sized at
                // one entry for each overflow page in the overflow chain. The page number of the first overflow page is stored in aOverflow[0],
                // etc. A value of 0 in the aOverflow[] array means "not yet known" (the cache is lazily populated).
                if (cur.IsIncrblobHandle && cur.Overflows == null)
                {
                    uint ovfl = (cur.Info.Payload - cur.Info.Local + ovflSize - 1) / ovflSize;
                    cur.Overflows = new Pid[ovfl];
                    // nOvfl is always positive.  If it were zero, fetchPayload would have been used instead of this routine. */
                    if (C._ALWAYS(ovfl != 0) && cur.Overflows == null)
                        rc = RC.NOMEM;
                }

                // If the overflow page-list cache has been allocated and the entry for the first required overflow page is valid, skip
                // directly to it.
                if (cur.Overflows != null && cur.Overflows[offset / ovflSize] != 0)
                {
                    idx = (offset / ovflSize);
                    nextPage = cur.Overflows[idx];
                    offset = (offset % ovflSize);
                }
#endif

                for (; rc == RC.OK && amount > 0 && nextPage != 0; idx++)
                {
#if !OMIT_INCRBLOB
                    // If required, populate the overflow page-list cache.
                    if (cur.Overflows != null)
                    {
                        Debug.Assert(cur.Overflows[idx] == 0 || cur.Overflows[idx] == nextPage);
                        cur.Overflows[idx] = nextPage;
                    }
#endif

                    MemPage dummy = null;
                    if (offset >= ovflSize)
                    {
                        // The only reason to read this page is to obtain the page number for the next page in the overflow chain. The page
                        // data is not required. So first try to lookup the overflow page-list cache, if any, then fall back to the getOverflowPage() function.
#if !OMIT_INCRBLOB
                        if (cur.Overflows == null && cur.Overflows[idx + 1] != 0)
                            nextPage = cur.Overflows[idx + 1];
                        else
#endif
                            rc = getOverflowPage(bt, nextPage, out dummy, out nextPage);
                        offset -= ovflSize;
                    }
                    else
                    {
                        // Need to read this page properly. It contains some of the range of data that is being read (eOp==0) or written (eOp!=0).
                        int a = (int)amount;
                        if (a + offset > ovflSize)
                            a = (int)(ovflSize - offset);

#if DIRECT_OVERFLOW_READ
                        // If all the following are true:
                        //
                        //   1) this is a read operation, and 
                        //   2) data is required from the start of this overflow page, and
                        //   3) the database is file-backed, and
                        //   4) there is no open write-transaction, and
                        //   5) the database is not a WAL database,
                        //
                        // then data can be read directly from the database file into the output buffer, bypassing the page-cache altogether. This speeds
                        // up loading large records that span many overflow pages.
                        VFile fd;
                        if (op == 0 && // (1)
                            offset == 0 && // (2)
                            bt.InTransaction == TRANS.READ && // (4)
                            (fd = bt.Pager.File())->Methods && // (3)
                            bt->Page1->Data[19] == 0x01) // (5)
                        {
                            var save = new byte[4];
                            var writeOffset = bufOffset - 4;
                            Buffer.BlockCopy(buf, writeOffset, save, 0, 4);
                            rc = fd.Read(buf, a + 4, writeOffset + (long)bt.PageSize * (nextPage - 1));
                            nextPage = ConvertEx.Get4(buf, writeOffset);
                            Buffer.BlockCopy(save, 0, buf, writeOffset, 4);
                        }
                        else
#endif
                        {
                            var dbPage = new PgHdr();
                            rc = bt.Pager.Acquire(nextPage, ref dbPage, false);
                            if (rc == RC.OK)
                            {
                                payload = Pager.GetData(dbPage);
                                nextPage = ConvertEx.Get4(payload);
                                rc = copyPayload(payload, offset + 4, buf, buf_, (uint)a, op, dbPage);
                                Pager.Unref(dbPage);
                                offset = 0;
                            }
                        }
                        amount -= (uint)a;
                        buf_ += (uint)a;
                    }
                }
            }

            if (rc == RC.OK && amount > 0)
                return SysEx.CORRUPT_BKPT();
            return rc;
        }

        public static RC Key(BtCursor cur, uint offset, uint amount, byte[] buf)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cur.ID >= 0 && cur.Pages[cur.ID] != null);
            Debug.Assert(cur.Idxs[cur.ID] < cur.Pages[cur.ID].Cells);
            return accessPayload(cur, offset, amount, buf, 0);
        }

        public static RC Data(BtCursor cur, uint offset, uint amount, byte[] buf)
        {
#if !OMIT_INCRBLOB
            if (cur.State == CURSOR.INVALID)
                return RC.ABORT;
#endif

            Debug.Assert(cursorHoldsMutex(cur));
            var rc = restoreCursorPosition(cur);
            if (rc == RC.OK)
            {
                Debug.Assert(cur.State == CURSOR.VALID);
                Debug.Assert(cur.ID >= 0 && cur.Pages[cur.ID] != null);
                Debug.Assert(cur.Idxs[cur.ID] < cur.Pages[cur.ID].Cells);
                rc = accessPayload(cur, offset, amount, buf, 0);
            }
            return rc;
        }

        static byte[] fetchPayload(BtCursor cur, ref int amount, bool skipKey, out uint payload_)
        {
            Debug.Assert(cur != null && cur.ID >= 0 && cur.Pages[cur.ID] != null);
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cursorHoldsMutex(cur));
            var page = cur.Pages[cur.ID];
            Debug.Assert(cur.Idxs[cur.ID] < page.Cells);
            if (C._NEVER(cur.Info.Size == 0))
                btreeParseCell(cur.Pages[cur.ID], cur.Idxs[cur.ID], ref cur.Info);
            var payload = C._alloc(cur.Info.Size - cur.Info.Header); //cur.Info.Cell + cur.Info.Header;
            payload_ = (cur.Info.Cell_ + cur.Info.Header);
            var key = (page.IntKey ? 0U : (uint)cur.Info.Key);
            uint local;
            if (skipKey)
            {
                payload_ += key;
                Buffer.BlockCopy(cur.Info.Cell, (int)payload_, payload, 0, (int)(cur.Info.Size - cur.Info.Header - key));
                local = cur.Info.Local - key;
            }
            else
            {
                Buffer.BlockCopy(cur.Info.Cell, (int)payload_, payload, 0, cur.Info.Size - cur.Info.Header);
                local = cur.Info.Local;
                Debug.Assert(local <= key);
            }
            amount = (int)local;
            return payload;
        }

        public static byte[] KeyFetch(BtCursor cur, ref int amount, out uint offset_)
        {
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            Debug.Assert(cursorHoldsMutex(cur));
            byte[] p = null;
            offset_ = 0U;
            if (C._ALWAYS(cur.State == CURSOR.VALID))
                p = fetchPayload(cur, ref amount, false, out offset_);
            return p;
        }

        public static byte[] DataFetch(BtCursor cur, ref int amount, out uint offset_)
        {
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            Debug.Assert(cursorHoldsMutex(cur));
            byte[] p = null;
            offset_ = 0U;
            if (C._ALWAYS(cur.State == CURSOR.VALID))
                p = fetchPayload(cur, ref amount, true, out offset_);
            return p;
        }

        #endregion

        #region Move Cursor

        static RC moveToChild(BtCursor cur, uint newID)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cur.ID < BTCURSOR_MAX_DEPTH);
            if (cur.ID >= (BTCURSOR_MAX_DEPTH - 1))
                return SysEx.CORRUPT_BKPT();
            var bt = cur.Bt;
            var newPage = new MemPage();
            var rc = getAndInitPage(bt, newID, ref newPage);
            if (rc != RC.OK)
                return rc;
            var i = cur.ID;
            cur.Pages[i + 1] = newPage;
            cur.Idxs[i + 1] = 0;
            cur.ID++;

            cur.Info.Size = 0;
            cur.ValidNKey = false;
            if (newPage.Cells < 1 || newPage.IntKey != cur.Pages[i].IntKey)
                return SysEx.CORRUPT_BKPT();
            return RC.OK;
        }

#if false
        static void assertParentIndex(MemPage parent, int idx, Pid child)
        {
            Debug.Assert(idx <= parent.nCell);
            if (idx == parent.nCell)
                Debug.Assert(ConvertEx.Get4(parent.Data, parent.HdrOffset + 8) == child);
            else
                Debug.Assert(ConvertEx.Get4(parent.Data, findCell(parent, idx)) == child);
        }
#else
        static void assertParentIndex(MemPage parent, int idx, Pid child) { }
#endif

        static void moveToParent(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            Debug.Assert(cur.ID > 0);
            Debug.Assert(cur.Pages[cur.ID] != null);

            // UPDATE: It is actually possible for the condition tested by the assert below to be untrue if the database file is corrupt. This can occur if
            // one cursor has modified page pParent while a reference to it is held by a second cursor. Which can only happen if a single page is linked
            // into more than one b-tree structure in a corrupt database.
#if false
            assertParentIndex(cur.Pages[cur.ID - 1], cur.Idxs[cur.ID - 1], cur.Pages[cur.ID].ID);
#endif
            ASSERTCOVERAGE(cur.Idxs[cur.ID - 1] > cur.Pages[cur.ID - 1].Cells);

            releasePage(cur.Pages[cur.ID]);
            cur.ID--;
            cur.Info.Size = 0;
            cur.ValidNKey = false;
        }

        static RC moveToRoot(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(CURSOR.INVALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.VALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.FAULT > CURSOR.REQUIRESEEK);
            if (cur.State >= CURSOR.REQUIRESEEK)
            {
                if (cur.State == CURSOR.FAULT)
                {
                    Debug.Assert(cur.SkipNext != 0);
                    return (RC)cur.SkipNext;
                }
                ClearCursor(cur);
            }

            var rc = RC.OK;
            var bt = cur.Btree.Bt;
            if (cur.ID >= 0)
            {
                for (var i = 1; i <= cur.ID; i++)
                    releasePage(cur.Pages[i]);
                cur.ID = 0;
            }
            else if (cur.RootID == 0)
            {
                cur.State = CURSOR.INVALID;
                return RC.OK;
            }
            else
            {
                rc = getAndInitPage(bt, cur.RootID, ref cur.Pages[0]);
                if (rc != RC.OK)
                {
                    cur.State = CURSOR.INVALID;
                    return rc;
                }
                cur.ID = 0;

                // If pCur.pKeyInfo is not NULL, then the caller that opened this cursor expected to open it on an index b-tree. Otherwise, if pKeyInfo is
                // NULL, the caller expects a table b-tree. If this is not the case, return an SQLITE_CORRUPT error.
                if ((cur.KeyInfo == null) != cur.Pages[0].IntKey)
                    return SysEx.CORRUPT_BKPT();
            }

            // Assert that the root page is of the correct type. This must be the case as the call to this function that loaded the root-page (either
            // this call or a previous invocation) would have detected corruption if the assumption were not true, and it is not possible for the flags
            // byte to have been modified while this cursor is holding a reference to the page.
            var root = cur.Pages[0];
            Debug.Assert(root.ID == cur.RootID);
            Debug.Assert(root.IsInit && (cur.KeyInfo == null) == root.IntKey);

            cur.Idxs[0] = 0;
            cur.Info.Size = 0;
            cur.AtLast = 0;
            cur.ValidNKey = false;

            if (root.Cells == 0 && !root.Leaf)
            {
                if (root.ID != 1)
                    return SysEx.CORRUPT_BKPT();
                Pid subpage = ConvertEx.Get4(root.Data, root.HdrOffset + 8);
                cur.State = CURSOR.VALID;
                rc = moveToChild(cur, subpage);
            }
            else
                cur.State = (root.Cells > 0 ? CURSOR.VALID : CURSOR.INVALID);
            return rc;
        }

        static RC moveToLeftmost(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            MemPage page;
            var rc = RC.OK;
            while (rc == RC.OK && !(page = cur.Pages[cur.ID]).Leaf)
            {
                Debug.Assert(cur.Idxs[cur.ID] < page.Cells);
                Pid id = ConvertEx.Get4(page.Data, findCell(page, cur.Idxs[cur.ID]));
                rc = moveToChild(cur, id);
            }
            return rc;
        }

        static RC moveToRightmost(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.State == CURSOR.VALID);
            var rc = RC.OK;
            MemPage page = null;
            while (rc == RC.OK && !(page = cur.Pages[cur.ID]).Leaf)
            {
                Pid id = ConvertEx.Get4(page.Data, page.HdrOffset + 8);
                cur.Idxs[cur.ID] = page.Cells;
                rc = moveToChild(cur, id);
            }
            if (rc == RC.OK)
            {
                cur.Idxs[cur.ID] = (ushort)(page.Cells - 1);
                cur.Info.Size = 0;
                cur.ValidNKey = false;
            }
            return rc;
        }

        public static RC First(BtCursor cur, ref int res)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            var rc = moveToRoot(cur);
            if (rc == RC.OK)
            {
                if (cur.State == CURSOR.INVALID)
                {
                    Debug.Assert(cur.Pages[cur.ID].Cells == 0);
                    res = 1;
                }
                else
                {
                    Debug.Assert(cur.Pages[cur.ID].Cells > 0);
                    res = 0;
                    rc = moveToLeftmost(cur);
                }
            }
            return rc;
        }

        public static RC Last(BtCursor cur, ref int res)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));

            // If the cursor already points to the last entry, this is a no-op.
            if (cur.State == CURSOR.VALID && cur.AtLast != 0)
            {
#if DEBUG
                // This block serves to Debug.Assert() that the cursor really does point to the last entry in the b-tree.
                for (var ii = 0; ii < cur.ID; ii++)
                    Debug.Assert(cur.Idxs[ii] == cur.Pages[ii].Cells);
                Debug.Assert(cur.Idxs[cur.ID] == cur.Pages[cur.ID].Cells - 1);
                Debug.Assert(cur.Pages[cur.ID].Leaf);
#endif
                return RC.OK;
            }

            var rc = moveToRoot(cur);
            if (rc == RC.OK)
            {
                if (cur.State == CURSOR.INVALID)
                {
                    Debug.Assert(cur.Pages[cur.ID].Cells == 0);
                    res = 1;
                }
                else
                {
                    Debug.Assert(cur.State == CURSOR.VALID);
                    res = 0;
                    rc = moveToRightmost(cur);
                    cur.AtLast = (byte)(rc == RC.OK ? 1 : 0);
                }
            }
            return rc;
        }

        public static RC MovetoUnpacked(BtCursor cur, UnpackedRecord idxKey, long intKey, int biasRight, out int res)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            Debug.Assert((idxKey == null) == (cur.KeyInfo == null));

            // If the cursor is already positioned at the point we are trying to move to, then just return without doing any work
            if (cur.State == CURSOR.VALID && cur.ValidNKey && cur.Pages[0].IntKey)
            {
                if (cur.Info.Key == intKey)
                {
                    res = 0;
                    return RC.OK;
                }
                if (cur.AtLast != 0 && cur.Info.Key < intKey)
                {
                    res = -1;
                    return RC.OK;
                }
            }

            res = -1;
            var rc = moveToRoot(cur);
            if (rc != RC.OK)
                return rc;
            Debug.Assert(cur.RootID == 0 || cur.Pages[cur.ID] != null);
            Debug.Assert(cur.RootID == 0 || cur.Pages[cur.ID].IsInit);
            Debug.Assert(cur.State == CURSOR.INVALID || cur.Pages[cur.ID].Cells > 0);
            if (cur.State == CURSOR.INVALID)
            {
                res = -1;
                Debug.Assert(cur.RootID == 0 || cur.Pages[cur.ID].Cells == 0);
                return RC.OK;
            }
            Debug.Assert(cur.Pages[0].IntKey || idxKey != null);
            int c;
            for (; ; )
            {
                // pPage->nCell must be greater than zero. If this is the root-page the cursor would have been INVALID above and this for(;;) loop
                // not run. If this is not the root-page, then the moveToChild() routine would have already detected db corruption. Similarly, pPage must
                // be the right kind (index or table) of b-tree page. Otherwise a moveToChild() or moveToRoot() call would have detected corruption.
                MemPage page = cur.Pages[cur.ID];
                Debug.Assert(page.Cells > 0);
                Debug.Assert(page.IntKey == (idxKey == null));
                uint idx;
                uint lwr = 0;
                uint upr = (uint)(page.Cells - 1);
                if (biasRight != 0)
                    cur.Idxs[cur.ID] = (ushort)(idx = upr);
                else
                    cur.Idxs[cur.ID] = (ushort)(idx = (upr + lwr) / 2);
                for (; ; )
                {
                    Debug.Assert(idx == cur.Idxs[cur.ID]);
                    cur.Info.Size = 0;
                    uint cell_ = findCell(page, idx) + page.ChildPtrSize; // Pointer to current cell in pPage
                    if (page.IntKey)
                    {
                        if (page.HasData)
                        {
                            uint dummy = 0;
                            cell_ += ConvertEx.GetVarint32(page.Data, cell_, out dummy);
                        }
                        long cellKeyLength = 0;
                        ConvertEx.GetVarint(page.Data, cell_, out cellKeyLength);
                        if (cellKeyLength == intKey)
                            c = 0;
                        else if (cellKeyLength < intKey)
                            c = -1;
                        else
                            c = +1;
                        cur.ValidNKey = true;
                        cur.Info.Key = cellKeyLength;
                    }
                    else
                    {
                        // The maximum supported page-size is 65536 bytes. This means that the maximum number of record bytes stored on an index B-Tree
                        // page is less than 16384 bytes and may be stored as a 2-byte varint. This information is used to attempt to avoid parsing 
                        // the entire cell by checking for the cases where the record is stored entirely within the b-tree page by inspecting the first 
                        // 2 bytes of the cell.
                        int cellLength = page.Data[cell_ + 0];
                        if (cellLength <= page.Max1bytePayload)
                        {
                            // This branch runs if the record-size field of the cell is a single byte varint and the record fits entirely on the main b-tree page.
                            c = _vdbe.RecordCompare(cellLength, page.Data, cell_ + 1, idxKey);
                        }
                        else if ((page.Data[cell_ + 1] & 0x80) == 0 && (cellLength = ((cellLength & 0x7f) << 7) + page.Data[cell_ + 1]) <= page.MaxLocal)
                        {
                            // The record-size field is a 2 byte varint and the record fits entirely on the main b-tree page.
                            c = _vdbe.RecordCompare(cellLength, page.Data, cell_ + 2, idxKey);
                        }
                        else
                        {
                            // The record flows over onto one or more overflow pages. In this case the whole cell needs to be parsed, a buffer allocated
                            // and accessPayload() used to retrieve the record into the buffer before VdbeRecordCompare() can be called.
                            var cellBody = new byte[page.Data.Length - cell_ + page.ChildPtrSize];
                            Buffer.BlockCopy(page.Data, (int)cell_ - page.ChildPtrSize, cellBody, 0, cellBody.Length);
                            btreeParseCellPtr(page, cellBody, ref cur.Info);
                            cellLength = (int)cur.Info.Key;
                            var cellKey = C._alloc(cellLength);
                            rc = accessPayload(cur, 0, (uint)cellLength, cellKey, 0);
                            if (rc != RC.OK)
                            {
                                cellKey = null;
                                goto moveto_finish;
                            }
                            c = _vdbe.RecordCompare(cellLength, cellKey, 0, idxKey);
                            cellKey = null;
                        }
                    }
                    if (c == 0)
                    {
                        if (page.IntKey && !page.Leaf)
                        {
                            lwr = idx;
                            upr = lwr - 1;
                            break;
                        }
                        else
                        {
                            res = 0;
                            rc = RC.OK;
                            goto moveto_finish;
                        }
                    }
                    if (c < 0)
                        lwr = idx + 1;
                    else
                        upr = idx - 1;
                    if (lwr > upr)
                        break;
                    cur.Idxs[cur.ID] = (ushort)(idx = (lwr + upr) / 2);
                }
                Debug.Assert(lwr == upr + 1 || (page.IntKey && !page.Leaf));
                Debug.Assert(page.IsInit);
                Pid childID;
                if (page.Leaf)
                    childID = 0;
                else if (lwr >= page.Cells)
                    childID = ConvertEx.Get4(page.Data, page.HdrOffset + 8);
                else
                    childID = ConvertEx.Get4(page.Data, findCell(page, lwr));
                if (childID == 0)
                {
                    Debug.Assert(cur.Idxs[cur.ID] < cur.Pages[cur.ID].Cells);
                    res = (int)c;
                    rc = RC.OK;
                    goto moveto_finish;
                }
                cur.Idxs[cur.ID] = (ushort)lwr;
                cur.Info.Size = 0;
                cur.ValidNKey = false;
                rc = moveToChild(cur, childID);
                if (rc != RC.OK) goto moveto_finish;
            }
        moveto_finish:
            return rc;
        }

        public static bool Eof(BtCursor cur)
        {
            // TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries have been deleted? This API will need to change to return an error code
            // as well as the boolean result value.
            return (cur.State != CURSOR.VALID);
        }

        public static RC Next_(BtCursor cur, ref int res)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            var rc = restoreCursorPosition(cur);
            if (rc != RC.OK)
                return rc;
            if (cur.State == CURSOR.INVALID)
            {
                res = 1;
                return RC.OK;
            }
            if (cur.SkipNext > 0)
            {
                cur.SkipNext = 0;
                res = 0;
                return RC.OK;
            }
            cur.SkipNext = 0;

            MemPage page = cur.Pages[cur.ID];
            uint idx = ++cur.Idxs[cur.ID];
            Debug.Assert(page.IsInit);

            cur.Info.Size = 0;
            cur.ValidNKey = false;
            if (idx >= page.Cells)
            {
                if (!page.Leaf)
                {
                    rc = moveToChild(cur, ConvertEx.Get4(page.Data, page.HdrOffset + 8));
                    if (rc != RC.OK) return rc;
                    rc = moveToLeftmost(cur);
                    res = 0;
                    return rc;
                }
                do
                {
                    if (cur.ID == 0)
                    {
                        res = 1;
                        cur.State = CURSOR.INVALID;
                        return RC.OK;
                    }
                    moveToParent(cur);
                    page = cur.Pages[cur.ID];
                } while (cur.Idxs[cur.ID] >= page.Cells);
                res = 0;
                if (page.IntKey)
                    rc = Next_(cur, ref res);
                else
                    rc = RC.OK;
                return rc;
            }
            res = 0;
            if (page.Leaf)
                return RC.OK;
            rc = moveToLeftmost(cur);
            return rc;
        }

        public static RC Previous(BtCursor cur, ref int res)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            var rc = restoreCursorPosition(cur);
            if (rc != RC.OK)
                return rc;
            cur.AtLast = 0;
            if (cur.State == CURSOR.INVALID)
            {
                res = 1;
                return RC.OK;
            }
            if (cur.SkipNext < 0)
            {
                cur.SkipNext = 0;
                res = 0;
                return RC.OK;
            }
            cur.SkipNext = 0;

            MemPage page = cur.Pages[cur.ID];
            Debug.Assert(page.IsInit);
            if (!page.Leaf)
            {
                uint idx = cur.Idxs[cur.ID];
                rc = moveToChild(cur, ConvertEx.Get4(page.Data, findCell(page, idx)));
                if (rc != RC.OK)
                    return rc;
                rc = moveToRightmost(cur);
            }
            else
            {
                while (cur.Idxs[cur.ID] == 0)
                {
                    if (cur.ID == 0)
                    {
                        cur.State = CURSOR.INVALID;
                        res = 1;
                        return RC.OK;
                    }
                    moveToParent(cur);
                }
                cur.Info.Size = 0;
                cur.ValidNKey = false;

                cur.Idxs[cur.ID]--;
                page = cur.Pages[cur.ID];
                if (page.IntKey && !page.Leaf)
                    rc = Previous(cur, ref res);
                else
                    rc = RC.OK;
            }
            res = 0;
            return rc;
        }

        #endregion

        #region Allocate Page

        static RC allocateBtreePage(BtShared bt, ref MemPage page, ref Pid id, Pid nearby, BTALLOC mode)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(mode == BTALLOC.ANY || (nearby > 0 && IFAUTOVACUUM(bt.AutoVacuum)));
            MemPage page1 = bt.Page1;
            Pid maxPage = btreePagecount(bt); // Total size of the database file
            uint n = ConvertEx.Get4(page1.Data, 36); // Number of pages on the freelist
            ASSERTCOVERAGE(n == maxPage - 1);
            if (n >= maxPage)
                return SysEx.CORRUPT_BKPT();
            RC rc;
            MemPage trunk = null;
            MemPage prevTrunk = null;
            if (n > 0)
            {
                // There are pages on the freelist.  Reuse one of those pages.
                bool searchList = false; // If the free-list must be searched for

                // If eMode==BTALLOC_EXACT and a query of the pointer-map shows that the page 'nearby' is somewhere on the free-list, then
                // the entire-list will be searched for that page.
#if !OMIT_AUTOVACUUM
                if (mode == BTALLOC.EXACT)
                {
                    if (nearby <= maxPage)
                    {
                        Debug.Assert(nearby > 0);
                        Debug.Assert(bt.AutoVacuum);
                        PTRMAP type = 0;
                        Pid dummy = 0;
                        rc = ptrmapGet(bt, nearby, ref type, ref dummy);
                        if (rc != RC.OK) return rc;
                        if (type == PTRMAP.FREEPAGE)
                            searchList = true;
                    }
                }
                else if (mode == BTALLOC.LE)
                    searchList = true;
#endif

                // Decrement the free-list count by 1. Set iTrunk to the index of the first free-list trunk page. iPrevTrunk is initially 1.
                rc = Pager.Write(page1.DBPage);
                if (rc != RC.OK) return rc;
                ConvertEx.Put4(page1.Data, (uint)36, n - 1);

                // The code within this loop is run only once if the 'searchList' variable is not true. Otherwise, it runs once for each trunk-page on the
                // free-list until the page 'nearby' is located (eMode==BTALLOC_EXACT) or until a page less than 'nearby' is located (eMode==BTALLOC_LT)
                Pid trunkID;
                do
                {
                    prevTrunk = trunk;
                    if (prevTrunk != null)
                        trunkID = ConvertEx.Get4(prevTrunk.Data, 0);
                    else
                        trunkID = ConvertEx.Get4(page1.Data, 32);

                    ASSERTCOVERAGE(trunkID == maxPage);
                    if (trunkID > maxPage)
                        rc = SysEx.CORRUPT_BKPT();
                    else
                        rc = btreeGetPage(bt, trunkID, ref trunk, false);
                    if (rc != RC.OK)
                    {
                        trunk = null;
                        goto end_allocate_page;
                    }
                    Debug.Assert(trunk != null);
                    Debug.Assert(trunk.Data != null);

                    uint k = ConvertEx.Get4(trunk.Data, 4); // # of leaves on this trunk page, Number of leaves on the trunk of the freelist
                    if (k == 0 && !searchList)
                    {
                        // The trunk has no leaves and the list is not being searched. So extract the trunk page itself and use it as the newly allocated page
                        Debug.Assert(prevTrunk == null);
                        rc = Pager.Write(trunk.DBPage);
                        if (rc != RC.OK)
                            goto end_allocate_page;
                        id = trunkID;
                        Buffer.BlockCopy(trunk.Data, 0, page1.Data, 32, 4);
                        page = trunk;
                        trunk = null;
                        TRACE("ALLOCATE: %d trunk - %d free pages left\n", id, n - 1);
                    }
                    else if (k > (uint)(bt.UsableSize / 4 - 2))
                    {
                        // Value of k is out of range.  Database corruption
                        rc = SysEx.CORRUPT_BKPT();
                        goto end_allocate_page;
#if !OMIT_AUTOVACUUM
                    }
                    else if (searchList && (nearby == trunkID || (trunkID < nearby && mode == BTALLOC.LE)))
                    {
                        // The list is being searched and this trunk page is the page to allocate, regardless of whether it has leaves.
                        id = trunkID;
                        page = trunk;
                        searchList = false;
                        rc = Pager.Write(trunk.DBPage);
                        if (rc != RC.OK)
                            goto end_allocate_page;
                        if (k == 0)
                        {
                            if (prevTrunk == null)
                            {
                                //memcpy(page1.Data[32], trunk.Data[0], 4);
                                page1.Data[32 + 0] = trunk.Data[0 + 0];
                                page1.Data[32 + 1] = trunk.Data[0 + 1];
                                page1.Data[32 + 2] = trunk.Data[0 + 2];
                                page1.Data[32 + 3] = trunk.Data[0 + 3];
                            }
                            else
                            {
                                rc = Pager.Write(prevTrunk.DBPage);
                                if (rc != RC.OK)
                                    goto end_allocate_page;
                                //memcpy(prevTrunk.Data[0], trunk.Data[0], 4);
                                prevTrunk.Data[0 + 0] = trunk.Data[0 + 0];
                                prevTrunk.Data[0 + 1] = trunk.Data[0 + 1];
                                prevTrunk.Data[0 + 2] = trunk.Data[0 + 2];
                                prevTrunk.Data[0 + 3] = trunk.Data[0 + 3];
                            }
                        }
                        else
                        {
                            // The trunk page is required by the caller but it contains pointers to free-list leaves. The first leaf becomes a trunk
                            // page in this case.
                            Pid newTrunkID = ConvertEx.Get4(trunk.Data, 8);
                            if (newTrunkID > maxPage)
                            {
                                rc = SysEx.CORRUPT_BKPT();
                                goto end_allocate_page;
                            }
                            ASSERTCOVERAGE(newTrunkID == maxPage);
                            var newTrunk = new MemPage();
                            rc = btreeGetPage(bt, newTrunkID, ref newTrunk, false);
                            if (rc != RC.OK)
                                goto end_allocate_page;
                            rc = Pager.Write(newTrunk.DBPage);
                            if (rc != RC.OK)
                            {
                                releasePage(newTrunk);
                                goto end_allocate_page;
                            }
                            //memcpy(newTrunk.Data[0], trunk.Data[0], 4);
                            newTrunk.Data[0 + 0] = trunk.Data[0 + 0];
                            newTrunk.Data[0 + 1] = trunk.Data[0 + 1];
                            newTrunk.Data[0 + 2] = trunk.Data[0 + 2];
                            newTrunk.Data[0 + 3] = trunk.Data[0 + 3];
                            ConvertEx.Put4(newTrunk.Data, (uint)4, (uint)(k - 1));
                            Buffer.BlockCopy(trunk.Data, 12, newTrunk.Data, 8, (int)(k - 1) * 4);
                            releasePage(newTrunk);
                            if (prevTrunk == null)
                            {
                                Debug.Assert(Pager.Iswriteable(page1.DBPage));
                                ConvertEx.Put4(page1.Data, (uint)32, newTrunkID);
                            }
                            else
                            {
                                rc = Pager.Write(prevTrunk.DBPage);
                                if (rc != RC.OK)
                                    goto end_allocate_page;
                                ConvertEx.Put4(prevTrunk.Data, (uint)0, newTrunkID);
                            }
                        }
                        trunk = null;
                        TRACE("ALLOCATE: %d trunk - %d free pages left\n", id, n - 1);
#endif
                    }
                    else if (k > 0)
                    {
                        // Extract a leaf from the trunk
                        byte[] data = trunk.Data;
                        Pid pageID;
                        uint closest;
                        if (nearby > 0)
                        {
                            closest = 0;
                            if (mode == BTALLOC.LE)
                            {
                                for (var i = 0U; i < k; i++)
                                {
                                    pageID = ConvertEx.Get4(data, 8 + i * 4);
                                    if (pageID <= nearby)
                                    {
                                        closest = i;
                                        break;
                                    }
                                }
                            }
                            else
                            {
                                int dist = Math.Abs((int)(ConvertEx.Get4(data, 8) - nearby));
                                for (var i = 1U; i < k; i++)
                                {
                                    int d2 = Math.Abs((int)(ConvertEx.Get4(data, 8 + i * 4) - nearby));
                                    if (d2 < dist)
                                    {
                                        closest = i;
                                        dist = d2;
                                    }
                                }
                            }
                        }
                        else
                            closest = 0;

                        pageID = ConvertEx.Get4(data, 8 + closest * 4);
                        ASSERTCOVERAGE(pageID == maxPage);
                        if (pageID > maxPage)
                        {
                            rc = SysEx.CORRUPT_BKPT();
                            goto end_allocate_page;
                        }
                        ASSERTCOVERAGE(pageID == maxPage);
                        if (!searchList || (pageID == nearby || (pageID < nearby && mode == BTALLOC.LE)))
                        {
                            id = pageID;
                            TRACE("ALLOCATE: %d was leaf %d of %d on trunk %d: %d more free pages\n", id, closest + 1, k, trunk.ID, n - 1);
                            rc = Pager.Write(trunk.DBPage);
                            if (rc != RC.OK) goto end_allocate_page;
                            if (closest < k - 1)
                                Buffer.BlockCopy(data, (int)(4 + k * 4), data, 8 + (int)closest * 4, 4);//memcpy( aData[8 + closest * 4], ref aData[4 + k * 4], 4 );
                            ConvertEx.Put4(data, (uint)4, (k - 1));
                            bool noContent = !btreeGetHasContent(bt, id);
                            rc = btreeGetPage(bt, id, ref page, noContent);
                            if (rc == RC.OK)
                            {
                                rc = Pager.Write((page).DBPage);
                                if (rc != RC.OK)
                                    releasePage(page);
                            }
                            searchList = false;
                        }
                    }
                    releasePage(prevTrunk);
                    prevTrunk = null;
                } while (searchList);
            }
            else
            {
                // Normally, new pages allocated by this block can be requested from the pager layer with the 'no-content' flag set. This prevents the pager
                // from trying to read the pages content from disk. However, if the current transaction has already run one or more incremental-vacuum
                // steps, then the page we are about to allocate may contain content that is required in the event of a rollback. In this case, do
                // not set the no-content flag. This causes the pager to load and journal the current page content before overwriting it.
                //
                // Note that the pager will not actually attempt to load or journal content for any page that really does lie past the end of the database
                // file on disk. So the effects of disabling the no-content optimization here are confined to those pages that lie between the end of the
                // database image and the end of the database file.
                bool noContent = !IFAUTOVACUUM(bt.DoTruncate);

                // There are no pages on the freelist, so append a new page to the database image.
                rc = Pager.Write(bt.Page1.DBPage);
                if (rc != RC.OK) return rc;
                bt.Pages++;
                if (bt.Pages == PENDING_BYTE_PAGE(bt)) bt.Pages++;

#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum && PTRMAP_ISPAGE(bt, bt.Pages))
                {
                    // If *pPgno refers to a pointer-map page, allocate two new pages at the end of the file instead of one. The first allocated page
                    // becomes a new pointer-map page, the second is used by the caller.
                    TRACE("ALLOCATE: %d from end of file (pointer-map page)\n", bt.Pages);
                    Debug.Assert(bt.Pages != PENDING_BYTE_PAGE(bt));
                    MemPage pg = null;
                    rc = btreeGetPage(bt, bt.Pages, ref pg, noContent);
                    if (rc == RC.OK)
                    {
                        rc = Pager.Write(pg.DBPage);
                        releasePage(pg);
                    }
                    if (rc != RC.OK) return rc;
                    bt.Pages++;
                    if (bt.Pages == PENDING_BYTE_PAGE(bt)) bt.Pages++;
                }
#endif
                ConvertEx.Put4(bt.Page1.Data, (uint)28, bt.Pages);
                id = bt.Pages;

                Debug.Assert(id != PENDING_BYTE_PAGE(bt));
                rc = btreeGetPage(bt, id, ref page, noContent);
                if (rc != RC.OK) return rc;
                rc = Pager.Write((page).DBPage);
                if (rc != RC.OK)
                    releasePage(page);
                TRACE("ALLOCATE: %d from end of file\n", id);
            }

            Debug.Assert(id != PENDING_BYTE_PAGE(bt));

        end_allocate_page:
            releasePage(trunk);
            releasePage(prevTrunk);
            if (rc == RC.OK)
            {
                if (Pager.get_PageRefs((page).DBPage) > 1)
                {
                    releasePage(page);
                    return SysEx.CORRUPT_BKPT();
                }
                (page).IsInit = false;
            }
            else
                page = null;
            Debug.Assert(rc != RC.OK || Pager.Iswriteable((page).DBPage));
            return rc;
        }

        static RC freePage2(BtShared bt, MemPage memPage, Pid pageID)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(pageID > 1);
            Debug.Assert(memPage == null || memPage.ID == pageID);

            MemPage page; // Page being freed. May be NULL.
            MemPage trunk = null; // Free-list trunk page
            if (memPage != null)
            {
                page = memPage;
                Pager.get_PageRefs(page.DBPage);
            }
            else
                page = btreePageLookup(bt, pageID);

            // Increment the free page count on pPage1
            MemPage page1 = bt.Page1; // Local reference to page 1
            RC rc = Pager.Write(page1.DBPage);
            if (rc != RC.OK) goto freepage_out;
            int frees = (int)ConvertEx.Get4(page1.Data, 36); // Initial number of pages on free-list
            ConvertEx.Put4(page1.Data, 36, frees + 1);

            if ((bt.BtsFlags & BTS.SECURE_DELETE) != 0)
            {
                // If the secure_delete option is enabled, then always fully overwrite deleted information with zeros.
                if ((page == null && (rc = btreeGetPage(bt, pageID, ref page, false)) != RC.OK) || (rc = Pager.Write(page.DBPage)) != RC.OK)
                    goto freepage_out;
                Array.Clear(page.Data, 0, (int)page.Bt.PageSize);
            }

            // If the database supports auto-vacuum, write an entry in the pointer-map to indicate that the page is free.
#if !OMIT_AUTOVACUUM
            if (bt.AutoVacuum)
            {
                ptrmapPut(bt, pageID, PTRMAP.FREEPAGE, 0, ref rc);
                if (rc != RC.OK) goto freepage_out;
            }
#endif

            // Now manipulate the actual database free-list structure. There are two possibilities. If the free-list is currently empty, or if the first
            // trunk page in the free-list is full, then this page will become a new free-list trunk page. Otherwise, it will become a leaf of the
            // first trunk page in the current free-list. This block tests if it is possible to add the page as a new free-list leaf.
            Pid trunkID = 0; // Page number of free-list trunk page
            if (frees != 0)
            {
                trunkID = ConvertEx.Get4(page1.Data, 32);
                rc = btreeGetPage(bt, trunkID, ref trunk, false);
                if (rc != RC.OK)
                    goto freepage_out;

                uint leafs = ConvertEx.Get4(trunk.Data, 4); // Initial number of leaf cells on trunk page
                Debug.Assert(bt.UsableSize > 32);
                if (leafs > (uint)bt.UsableSize / 4 - 2)
                {
                    rc = SysEx.CORRUPT_BKPT();
                    goto freepage_out;
                }
                if (leafs < (uint)bt.UsableSize / 4 - 8)
                {
                    // In this case there is room on the trunk page to insert the page being freed as a new leaf.
                    //
                    // Note that the trunk page is not really full until it contains usableSize/4 - 2 entries, not usableSize/4 - 8 entries as we have
                    // coded.  But due to a coding error in versions of SQLite prior to 3.6.0, databases with freelist trunk pages holding more than
                    // usableSize/4 - 8 entries will be reported as corrupt.  In order to maintain backwards compatibility with older versions of SQLite,
                    // we will continue to restrict the number of entries to usableSize/4 - 8 for now.  At some point in the future (once everyone has upgraded
                    // to 3.6.0 or later) we should consider fixing the conditional above to read "usableSize/4-2" instead of "usableSize/4-8".
                    rc = Pager.Write(trunk.DBPage);
                    if (rc == RC.OK)
                    {
                        ConvertEx.Put4(trunk.Data, (uint)4, leafs + 1);
                        ConvertEx.Put4(trunk.Data, (uint)8 + leafs * 4, pageID);
                        if (page != null && (bt.BtsFlags & BTS.SECURE_DELETE) == 0)
                            Pager.DontWrite(page.DBPage);
                        rc = btreeSetHasContent(bt, pageID);
                    }
                    TRACE("FREE-PAGE: %d leaf on trunk page %d\n", pageID, trunk.ID);
                    goto freepage_out;
                }
            }

            // If control flows to this point, then it was not possible to add the the page being freed as a leaf page of the first trunk in the free-list.
            // Possibly because the free-list is empty, or possibly because the first trunk in the free-list is full. Either way, the page being freed
            // will become the new first trunk page in the free-list.
            if (page == null && (rc = btreeGetPage(bt, pageID, ref page, false)) != RC.OK)
                goto freepage_out;
            rc = Pager.Write(page.DBPage);
            if (rc != RC.OK)
                goto freepage_out;
            ConvertEx.Put4(page.Data, trunkID);
            ConvertEx.Put4(page.Data, 4, 0);
            ConvertEx.Put4(page1.Data, (uint)32, pageID);
            TRACE("FREE-PAGE: %d new trunk page replacing %d\n", page.ID, trunkID);

        freepage_out:
            if (page != null)
                page.IsInit = false;
            releasePage(page);
            releasePage(trunk);
            return rc;
        }

        static void freePage(MemPage page, ref RC rc)
        {
            if ((rc) == RC.OK)
                rc = freePage2(page.Bt, page, page.ID);
        }

        static RC clearCell(MemPage page, uint cell_)
        {
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            var info = new CellInfo();
            btreeParseCellPtr(page, cell_, ref info);
            if (info.Overflow == 0)
                return RC.OK; // No overflow pages. Return without doing anything
            //if (cell_ + info.Overflow + 3 > page.Data + page.MaskPage)
            //    return SysEx.CORRUPT_BKPT(); // Cell extends past end of page
            Pid ovflID = ConvertEx.Get4(page.Data, cell_ + info.Overflow);
            var bt = page.Bt;
            Debug.Assert(bt.UsableSize > 4);
            var ovflPageSize = (uint)(bt.UsableSize - 4);
            var ovfls = (int)((info.Payload - info.Local + ovflPageSize - 1) / ovflPageSize);
            Debug.Assert(ovflID == 0 || ovfls > 0);
            RC rc;
            while (ovfls-- != 0)
            {
                if (ovflID < 2 || ovflID > btreePagecount(bt))
                {
                    // 0 is not a legal page number and page 1 cannot be an overflow page. Therefore if ovflPgno<2 or past the end of the 
                    // file the database must be corrupt.
                    return SysEx.CORRUPT_BKPT();
                }
                Pid nextID = 0;
                MemPage ovfl = null;
                if (ovfls != 0)
                {
                    rc = getOverflowPage(bt, ovflID, out ovfl, out nextID);
                    if (rc != RC.OK) return rc;
                }

                if ((ovfl != null || (ovfl = btreePageLookup(bt, ovflID)) != null) && Pager.get_PageRefs(ovfl.DBPage) != 1)
                {
                    // There is no reason any cursor should have an outstanding reference to an overflow page belonging to a cell that is being deleted/updated.
                    // So if there exists more than one reference to this page, then it must not really be an overflow page and the database must be corrupt. 
                    // It is helpful to detect this before calling freePage2(), as freePage2() may zero the page contents if secure-delete mode is
                    // enabled. If this 'overflow' page happens to be a page that the caller is iterating through or using in some other way, this can be problematic.
                    rc = SysEx.CORRUPT_BKPT();
                }
                else
                    rc = freePage2(bt, ovfl, ovflID);

                if (ovfl != null)
                    Pager.Unref(ovfl.DBPage);
                if (rc != RC.OK) return rc;
                ovflID = nextID;
            }
            return RC.OK;
        }

        static RC fillInCell(MemPage page, byte[] cell, byte[] key, long keyLength, byte[] data, int dataLength, int zeros, ref ushort sizeOut)
        {
            BtShared bt = page.Bt;
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));

            // pPage is not necessarily writeable since pCell might be auxiliary buffer space that is separate from the pPage buffer area
            //Skipped//Debug.Assert(cell < page->Data || cell >= &page.Data[bt.PageSize] || Pager.Iswriteable(page.DBPage));

            // Fill in the header.
            uint header = 0U;
            if (!page.Leaf)
                header += 4;
            if (page.HasData)
                header += (uint)ConvertEx.PutVarint(cell, header, (int)(dataLength + zeros));
            else
                dataLength = zeros = 0;
            header += ConvertEx.PutVarint(cell, header, (ulong)keyLength);
            var info = new CellInfo();
            btreeParseCellPtr(page, cell, ref info);
            Debug.Assert(info.Header == header);
            Debug.Assert(info.Key == keyLength);
            Debug.Assert(info.Data == (uint)(dataLength + zeros));

            // Fill in the payload
            byte[] src;
            uint src_ = 0U;
            int srcLength;
            int payloadLength = dataLength + zeros;
            if (page.IntKey)
            {
                src = data;
                srcLength = dataLength;
                dataLength = 0;
            }
            else
            {
                if (C._NEVER(keyLength > 0x7fffffff || key == null))
                    return SysEx.CORRUPT_BKPT();
                payloadLength += (int)keyLength;
                src = key;
                srcLength = (int)keyLength;
            }
            sizeOut = info.Size;
            int spaceLeft = info.Local;
            byte[] payload = cell; uint payload_ = header;
            byte[] prior = cell; uint prior_ = info.Overflow;

            RC rc;
            MemPage toRelease = null;
            while (payloadLength > 0)
            {
                if (spaceLeft == 0)
                {
                    Pid idOvfl = 0;
#if !OMIT_AUTOVACUUM
                    Pid idPtrmap = idOvfl; // Overflow page pointer-map entry page
                    if (bt.AutoVacuum)
                    {
                        do
                        {
                            idOvfl++;
                        } while (PTRMAP_ISPAGE(bt, idOvfl) || idOvfl == PENDING_BYTE_PAGE(bt));
                    }
#endif
                    MemPage ovfl = null;
                    rc = allocateBtreePage(bt, ref ovfl, ref idOvfl, idOvfl, BTALLOC.ANY);
#if !OMIT_AUTOVACUUM
                    // If the database supports auto-vacuum, and the second or subsequent overflow page is being allocated, add an entry to the pointer-map
                    // for that page now. 
                    //
                    // If this is the first overflow page, then write a partial entry to the pointer-map. If we write nothing to this pointer-map slot,
                    // then the optimistic overflow chain processing in clearCell() may misinterpret the uninitialized values and delete the
                    // wrong pages from the database.
                    if (bt.AutoVacuum && rc == RC.OK)
                    {
                        var type = (idPtrmap != 0 ? PTRMAP.OVERFLOW2 : PTRMAP.OVERFLOW1);
                        ptrmapPut(bt, idOvfl, type, idPtrmap, ref rc);
                        if (rc != RC.OK)
                            releasePage(ovfl);
                    }
#endif
                    if (rc != RC.OK)
                    {
                        releasePage(toRelease);
                        return rc;
                    }

                    // If pToRelease is not zero than pPrior points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                    Debug.Assert(toRelease == null || Pager.Iswriteable(toRelease.DBPage));

                    // If pPrior is part of the data area of pPage, then make sure pPage is still writeable
                    //skipped//Debug.Assert(prior < page->Data || prior >= page->Data[bt.PageSize] || Pager.Iswriteable(page.DBPage));

                    ConvertEx.Put4(prior, prior_, idOvfl);
                    releasePage(toRelease);
                    toRelease = ovfl;
                    prior = ovfl.Data; prior_ = 0;
                    ConvertEx.Put4(prior, 0);
                    payload = ovfl.Data; payload_ = 4;
                    spaceLeft = (int)bt.UsableSize - 4;
                }
                int n = payloadLength;
                if (n > spaceLeft) n = spaceLeft;

                // If pToRelease is not zero than pPrior points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                Debug.Assert(toRelease == null || Pager.Iswriteable(toRelease.DBPage));

                // If pPrior is part of the data area of pPage, then make sure pPage is still writeable
                //skipped//Debug.Assert(prior < page.Data || prior >= &page.Data[bt.PageSize] || Pager.Iswriteable(pageDBPage));

                if (srcLength > 0)
                {
                    if (n > srcLength) n = srcLength;
                    Debug.Assert(src != null);
                    Buffer.BlockCopy(src, (int)src_, payload, (int)payload_, n);
                }
                else
                {

                    var zeroBlob = new byte[n];
                    Buffer.BlockCopy(zeroBlob, 0, payload, (int)payload_, n);
                }
                payloadLength -= n;
                payload_ += (uint)n;
                src_ += (uint)n;
                srcLength -= n;
                spaceLeft -= n;
                if (srcLength == 0)
                {
                    srcLength = dataLength;
                    src = data;
                }
            }
            releasePage(toRelease);
            return RC.OK;
        }

        static void dropCell(MemPage page, uint idx, ushort size, ref RC rcRef)
        {
            if (rcRef != RC.OK) return;

            Debug.Assert(idx < page.Cells);
            Debug.Assert(size == cellSize(page, idx));
            Debug.Assert(Pager.Iswriteable(page.DBPage));
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            var data = page.Data;
            var ptr = page.CellOffset + 2 * idx; // Used to move bytes around within data[]
            var pc = (uint)ConvertEx.Get2(data, ptr); // Offset to cell content of cell being deleted
            var hdr = page.HdrOffset; // Beginning of the header.  0 most pages.  100 page 1
            ASSERTCOVERAGE(pc == ConvertEx.Get2(data, hdr + 5));
            ASSERTCOVERAGE(pc + size == page.Bt.UsableSize);
            if (pc < (uint)ConvertEx.Get2(data, hdr + 5) || pc + size > page.Bt.UsableSize)
            {
                rcRef = SysEx.CORRUPT_BKPT();
                return;
            }
            var rc = freeSpace(page, pc, size);
            if (rc != RC.OK)
            {
                rcRef = rc;
                return;
            }
            {
                //uint8* endPtr = &page->CellIdx[2 * page->Cells - 2]; // End of loop
                //_assert((PTR_TO_INT(ptr) & 1) == 0); // ptr is always 2-byte aligned
                //while (ptr < endPtr)
                //{
                //    *(uint16*)ptr = *(uint16*)&ptr[2];
                //    ptr += 2;
                //}
                Buffer.BlockCopy(data, (int)ptr + 2, data, (int)ptr, (int)(page.Cells - 1 - idx) * 2);
            }
            page.Cells--;
            // ConvertEx.Put2(&data[hdr + 3], page->Cells);
            data[page.HdrOffset + 3] = (byte)(page.Cells >> 8);
            data[page.HdrOffset + 4] = (byte)(page.Cells);
            page.Frees += 2;
        }

        static void insertCell(MemPage page, uint i, byte[] cell, ushort size, byte[] temp, Pid childID, ref RC rcRef)
        {
            if (rcRef != RC.OK) return;

            Debug.Assert(i <= page.Cells + page.Overflows);
            Debug.Assert(page.Cells <= MX_CELL(page.Bt) && MX_CELL(page.Bt) <= 10921);
            Debug.Assert(page.Overflows <= page.Ovfls.Length);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            // The cell should normally be sized correctly.  However, when moving a malformed cell from a leaf page to an interior page, if the cell size
            // wanted to be less than 4 but got rounded up to 4 on the leaf, then size might be less than 8 (leaf-size + pointer) on the interior node.  Hence
            // the term after the || in the following assert().
            Debug.Assert(size == cellSizePtr(page, cell) || (size == 8 && childID > 0));
            int skip = (childID != 0 ? 4 : 0);
            if (page.Overflows != 0 || size + 2 > page.Frees)
            {
                if (temp != null)
                {
                    Buffer.BlockCopy(cell, skip, temp, skip, size - skip);
                    cell = temp;
                }
                if (childID != 0)
                    ConvertEx.Put4(cell, childID);
                int j = page.Overflows++;
                Debug.Assert(j < page.Ovfls.Length);
                page.Ovfls[j].Cell = cell;
                page.OvflIdxs[j] = (ushort)i;
            }
            else
            {
                RC rc = Pager.Write(page.DBPage);
                if (rc != RC.OK)
                {
                    rcRef = rc;
                    return;
                }
                Debug.Assert(Pager.Iswriteable(page.DBPage));
                var data = page.Data; // The content of the whole page
                uint cellOffset = page.CellOffset; // Address of first cell pointer in data[]
                uint end = cellOffset + 2U * page.Cells; // First byte past the last cell pointer in data[]
                uint ins = cellOffset + 2U * i; // Index in data[] where new cell pointer is inserted
                uint idx = 0; // Where to write new cell content in data[]
                rc = allocateSpace(page, size, ref idx);
                if (rc != RC.OK) { rcRef = rc; return; }
                // The allocateSpace() routine guarantees the following two properties if it returns success
                Debug.Assert(idx >= end + 2);
                Debug.Assert(idx + size <= (int)page.Bt.UsableSize);
                page.Cells++;
                page.Frees -= (ushort)(2 + size);
                Buffer.BlockCopy(cell, skip, data, (int)(idx + skip), size - skip);
                if (childID != 0)
                    ConvertEx.Put4(data, idx, childID);
                {
                    //uint8 *ptr = &data[end]; // Used for moving information around in data[]
                    //uint8 *endPtr = &data[ins]; // End of the loop
                    //_assert((PTR_TO_INT(ptr) & 1) == 0); // ptr is always 2-byte aligned
                    //while (ptr > endPtr)
                    //{
                    //    *(uint16*)ptr = *(uint16*)&ptr[-2];
                    //    ptr -= 2;
                    //}
                    for (uint j = end; j > ins; j -= 2)
                    {
                        data[j + 0] = data[j - 2];
                        data[j + 1] = data[j - 1];
                    }
                }
                ConvertEx.Put2(data, ins, idx);
                ConvertEx.Put2(data, page.HdrOffset + 3, page.Cells);
#if !OMIT_AUTOVACUUM
                if (page.Bt.AutoVacuum)
                {
                    // The cell may contain a pointer to an overflow page. If so, write the entry for the overflow page into the pointer map.
                    ptrmapPutOvflPtr(page, cell, ref rcRef);
                }
#endif
            }
        }

        static void assemblePage(MemPage page, int cells, byte[] cellSet, int[] sizes)
        {
            Debug.Assert(page.Overflows == 0);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(cells >= 0 && cells <= (int)MX_CELL(page.Bt) && (int)MX_CELL(page.Bt) <= 10921);
            Debug.Assert(Pager.Iswriteable(page.DBPage));

            // Check that the page has just been zeroed by zeroPage()
            byte[] data = page.Data; // Pointer to data for pPage
            int hdr = page.HdrOffset; // Offset of header on pPage
            int usable = (int)page.Bt.UsableSize; // Usable size of page
            Debug.Assert(ConvertEx.Get2nz(data, hdr + 5) == usable);

            int cellptr = page.CellOffset + cells * 2; // Address of next cell pointer
            int cellbody = usable; // Address of next cell body
            for (int i = cells - 1; i >= 0; i--)
            {
                var sz = (ushort)sizes[i];
                cellptr -= 2;
                cellbody -= sz;
                ConvertEx.Put2(data, cellptr, cellbody);
                Buffer.BlockCopy(cellSet, 0, data, cellbody, sz);
            }
            ConvertEx.Put2(data, hdr + 3, cells);
            ConvertEx.Put2(data, hdr + 5, cellbody);
            page.Frees -= (ushort)(cells * 2 + usable - cellbody);
            page.Cells = (ushort)cells;
        }
        static void assemblePage(MemPage page, int cells, byte[] cellSet, ushort[] sizes)
        {
            Debug.Assert(page.Overflows == 0);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(cells >= 0 && cells <= MX_CELL(page.Bt) && MX_CELL(page.Bt) <= 5460);
            Debug.Assert(Pager.Iswriteable(page.DBPage));

            // Check that the page has just been zeroed by zeroPage()
            byte[] data = page.Data; // Pointer to data for pPage
            int hdr = page.HdrOffset; // Offset of header on pPage
            int usable = (int)page.Bt.UsableSize; // Usable size of page
            Debug.Assert(ConvertEx.Get2nz(data, hdr + 5) == usable);

            int cellptr = page.CellOffset + cells * 2; // Address of next cell pointer
            int cellbody = usable; // Address of next cell body
            for (int i = cells - 1; i >= 0; i--)
            {
                var sz = (ushort)sizes[i];
                cellptr -= 2;
                cellbody -= sz;
                ConvertEx.Put2(data, cellptr, cellbody);
                Buffer.BlockCopy(cellSet, 0, data, cellbody, sz);
            }
            ConvertEx.Put2(data, hdr + 3, cells);
            ConvertEx.Put2(data, hdr + 5, cellbody);
            page.Frees -= (ushort)(cells * 2 + usable - cellbody);
            page.Cells = (ushort)cells;
        }
        static void assemblePage(MemPage page, int cells, byte[][] cellSet, ushort[] size, int offset)
        {
            Debug.Assert(page.Overflows == 0);
            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(cells >= 0 && cells <= MX_CELL(page.Bt) && MX_CELL(page.Bt) <= 5460);
            Debug.Assert(Pager.Iswriteable(page.DBPage));

            // Check that the page has just been zeroed by zeroPage()
            byte[] data = page.Data; // Pointer to data for pPage
            int hdr = page.HdrOffset; // Offset of header on pPage
            int usable = (int)page.Bt.UsableSize; // Usable size of page
            Debug.Assert(ConvertEx.Get2(data, hdr + 5) == usable);

            int cellptr = page.CellOffset + cells * 2; // Address of next cell pointer
            int cellbody = usable; // Address of next cell body
            for (int i = cells - 1; i >= 0; i--)
            {
                var sz = (ushort)size[i + offset];
                cellptr -= 2;
                cellbody -= sz;
                ConvertEx.Put2(data, cellptr, cellbody);
                Buffer.BlockCopy(cellSet[offset + i], 0, data, cellbody, sz);
            }
            ConvertEx.Put2(data, hdr + 3, cells);
            ConvertEx.Put2(data, hdr + 5, cellbody);
            page.Frees -= (ushort)(cells * 2 + usable - cellbody);
            page.Cells = (ushort)cells;
        }

        #endregion

        #region Balance

        static int NN = 1;              // Number of neighbors on either side of pPage
        static int NB = (NN * 2 + 1);   // Total pages involved in the balance

#if !OMIT_QUICKBALANCE
        static RC balance_quick(MemPage parent, MemPage page, byte[] space)
        {
            BtShared bt = page.Bt; // B-Tree Database

            Debug.Assert(MutexEx.Held(page.Bt.Mutex));
            Debug.Assert(Pager.Iswriteable(parent.DBPage));
            Debug.Assert(page.Overflows == 1);

            // This error condition is now caught prior to reaching this function
            if (page.Cells <= 0)
                return SysEx.CORRUPT_BKPT();

            // Allocate a new page. This page will become the right-sibling of pPage. Make the parent page writable, so that the new divider cell
            // may be inserted. If both these operations are successful, proceed.

            MemPage newPage = new MemPage(); // Newly allocated page
            Pid newPageID = 0; // Page number of pNew
            var rc = allocateBtreePage(bt, ref newPage, ref newPageID, 0, BTALLOC.ANY);

            if (rc == RC.OK)
            {
                ushort out_ = 4; //byte[] out_ = &space[4];
                byte[] cell = page.Ovfls[0].Cell;
                ushort[] sizeCell = new ushort[1];
                sizeCell[0] = cellSizePtr(page, cell);

                Debug.Assert(Pager.Iswriteable(newPage.DBPage));
                Debug.Assert(page.Data[0] == (PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF));
                zeroPage(newPage, PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF);
                assemblePage(newPage, 1, cell, sizeCell);

                // If this is an auto-vacuum database, update the pointer map with entries for the new page, and any pointer from the 
                // cell on the page to an overflow page. If either of these operations fails, the return code is set, but the contents
                // of the parent page are still manipulated by thh code below. That is Ok, at this point the parent page is guaranteed to
                // be marked as dirty. Returning an error code will cause a rollback, undoing any changes made to the parent page.
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum)
                {
                    ptrmapPut(bt, newPageID, PTRMAP.BTREE, parent.ID, ref rc);
                    if (sizeCell[0] > newPage.MinLocal)
                        ptrmapPutOvflPtr(newPage, cell, ref rc);
                }
#endif

                // Create a divider cell to insert into pParent. The divider cell consists of a 4-byte page number (the page number of pPage) and
                // a variable length key value (which must be the same value as the largest key on pPage).
                //
                // To find the largest key value on pPage, first find the right-most cell on pPage. The first two fields of this cell are the 
                // record-length (a variable length integer at most 32-bits in size) and the key value (a variable length integer, may have any value).
                // The first of the while(...) loops below skips over the record-length field. The second while(...) loop copies the key value from the
                // cell on pPage into the pSpace buffer.
                uint cell_ = findCell(page, page.Cells - 1U);
                cell = page.Data;
                uint stop = cell_ + 9;
                while (((cell[cell_++]) & 0x80) != 0 && cell_ < stop) ;
                stop = cell_ + 9;
                while (((space[out_++] = cell[cell_++]) & 0x80) != 0 && cell_ < stop) ;

                // Insert the new divider cell into pParent.
                insertCell(parent, parent.Cells, space, out_, null, page.ID, ref rc);

                // Set the right-child pointer of pParent to point to the new page.
                ConvertEx.Put4(parent.Data, parent.HdrOffset + 8, newPageID);

                // Release the reference to the new page.
                releasePage(newPage);
            }

            return rc;
        }
#endif

#if false
        static int ptrmapCheckPages(MemPage[] pageSet, int pages)
        {
            for (int i = 0; i < pages; i++)
            {
                MemPage page = pageSet[i];
                BtShared bt = page.Bt;
                Debug.Assert(page.IsInit);

                Pid n;
                PTRMAP e;
                for (uint j = 0U; j < page.Cells; j++)
                {
                    uint z_ = findCell(page, j);
                    CellInfo info = new CellInfo();
                    btreeParseCellPtr(page, z_, ref info);
                    if (info.Overflow != 0)
                    {
                        Pid ovfl = ConvertEx.Get4(page.Data, z_ + info.Overflow);
                        ptrmapGet(bt, ovfl, ref e, ref n);
                        Debug.Assert(n == page.ID && e == PTRMAP.OVERFLOW1);
                    }
                    if (!page.Leaf)
                    {
                        Pid child = ConvertEx.Get4(page.Data, z_);
                        ptrmapGet(bt, child, ref e, ref n);
                        Debug.Assert(n == page.ID && e == PTRMAP.BTREE);
                    }
                }
                if (!page.Leaf)
                {
                    Pid child = ConvertEx.Get4(page.Data, page.HdrOffset + 8);
                    ptrmapGet(bt, child, ref e, ref n);
                    Debug.Assert(n == page.ID && e == PTRMAP.BTREE);
                }
            }
            return 1;
        }
#endif

        static void copyNodeContent(MemPage from, MemPage to, ref RC rcRef)
        {
            if (rcRef == RC.OK)
            {
                BtShared bt = from.Bt;
                var fromData = from.Data;
                var toData = to.Data;
                int fromHdr = from.HdrOffset;
                int toHdr = (to.ID == 1 ? 100 : 0);

                Debug.Assert(from.IsInit);
                Debug.Assert(from.Frees >= toHdr);
                Debug.Assert(ConvertEx.Get2(fromData, fromHdr + 5) <= (int)bt.UsableSize);

                // Copy the b-tree node content from page pFrom to page pTo.
                int data = ConvertEx.Get2(fromData, fromHdr + 5);
                Buffer.BlockCopy(fromData, data, toData, data, (int)bt.UsableSize - data);
                Buffer.BlockCopy(fromData, fromHdr, toData, toHdr, from.CellOffset + 2 * from.Cells);

                // Reinitialize page pTo so that the contents of the MemPage structure match the new data. The initialization of pTo can actually fail under
                // fairly obscure circumstances, even though it is a copy of initialized page pFrom.
                to.IsInit = false;
                var rc = btreeInitPage(to);
                if (rc != RC.OK)
                {
                    rcRef = rc;
                    return;
                }

                // If this is an auto-vacuum database, update the pointer-map entries for any b-tree or overflow pages that pTo now contains the pointers to.
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum)
                    rcRef = setChildPtrmaps(to);
#endif
            }
        }

        // under C#; Try to reuse Memory
        static RC balance_nonroot(MemPage parent, uint parentIdx, byte[] ovflSpace, bool isRoot, bool bulk)
        {
            BtShared bt = parent.Bt; // The whole database
            Debug.Assert(MutexEx.Held(bt.Mutex));
            Debug.Assert(Pager.Iswriteable(parent.DBPage));

#if false
            TRACE("BALANCE: begin page %d child of %d\n", page.ID, parent.ID);
#endif

            // At this point pParent may have at most one overflow cell. And if this overflow cell is present, it must be the cell with 
            // index iParentIdx. This scenario comes about when this function is called (indirectly) from sqlite3BtreeDelete().
            Debug.Assert(parent.Overflows == 0 || parent.Overflows == 1);
            Debug.Assert(parent.Overflows == 0 || parent.OvflIdxs[0] == parentIdx);

            if (ovflSpace == null)
                return RC.NOMEM;

            // Find the sibling pages to balance. Also locate the cells in pParent that divide the siblings. An attempt is made to find NN siblings on 
            // either side of pPage. More siblings are taken from one side, however, if there are fewer than NN siblings on the other side. If pParent
            // has NB or fewer children then all children of pParent are taken.  
            //
            // This loop also drops the divider cells from the parent page. This way, the remainder of the function does not have to deal with any
            // overflow cells in the parent page, since if any existed they will have already been removed.
            uint i = (uint)parent.Overflows + parent.Cells;
            uint nxDiv; // Next divider slot in pParent.aCell[]
            if (i < 2)
                nxDiv = 0;
            else
            {
                if (parentIdx == 0)
                    nxDiv = 0;
                else if (parentIdx == i)
                    nxDiv = i - 2 + (bulk ? 1U : 0U);
                else
                {
                    Debug.Assert(!bulk);
                    nxDiv = parentIdx - 1;
                }
                i = 2 - (bulk ? 1U : 0U);
            }
            uint right_; // Location in parent of right-sibling pointer
            if ((i + nxDiv - parent.Overflows) == parent.Cells)
                right_ = parent.HdrOffset + 8U;
            else
                right_ = findCell(parent, i + nxDiv - parent.Overflows);
            Pid id = ConvertEx.Get4(parent.Data, right_); // Temp var to store a page number in
            RC rc;
            MemPage[] oldPages = new MemPage[NB]; // pPage and up to two siblings
            MemPage[] copyPages = new MemPage[NB]; // Private copies of apOld[] pages
            MemPage[] newPages = new MemPage[NB + 2]; // pPage and up to NB siblings after balancing
            uint oldPagesUsed = i + 1; // Number of pages in apOld[]
            uint newPagesUsed = 0; // Number of pages in apNew[]
            uint maxCells = 0; // Allocated size of apCell, szCell, aFrom.
            uint[] divs_ = new uint[NB - 1]; // Divider cells in pParent
            Pid[] countNew = new Pid[NB + 2]; // Index in aCell[] of cell after i-th page
            ushort[] sizeNew = new ushort[NB + 2]; // Combined size of cells place on i-th page
            byte[][] cell = null; // All cells begin balanced
            while (true)
            {
                rc = getAndInitPage(bt, id, ref oldPages[i]);
                if (rc != RC.OK)
                {
                    //_memset(oldPages, 0, (i + 1) * sizeof(MemPage *));
                    goto balance_cleanup;
                }
                maxCells += 1U + oldPages[i].Cells + oldPages[i].Overflows;
                if (i-- == 0) break;

                if (i + nxDiv == parent.OvflIdxs[0] && parent.Overflows != 0)
                {
                    divs_[i] = 0; //parent.Ovfls[0];
                    id = (Pid)ConvertEx.Get4(parent.Ovfls[0].Cell, divs_[i]);
                    sizeNew[i] = cellSizePtr(parent, divs_[i]);
                    parent.Overflows = 0;
                }
                else
                {
                    divs_[i] = findCell(parent, i + nxDiv - parent.Overflows);
                    id = ConvertEx.Get4(parent.Data, divs_[i]);
                    sizeNew[i] = cellSizePtr(parent, divs_[i]);

                    // Drop the cell from the parent page. apDiv[i] still points to the cell within the parent, even though it has been dropped.
                    // This is safe because dropping a cell only overwrites the first four bytes of it, and this function does not need the first
                    // four bytes of the divider cell. So the pointer is safe to use later on.
                    //
                    // But not if we are in secure-delete mode. In secure-delete mode, the dropCell() routine will overwrite the entire cell with zeroes.
                    // In this case, temporarily copy the cell into the aOvflSpace[] buffer. It will be copied out again as soon as the aSpace[] buffer is allocated.
                    //if ((bt.BtsFlags & BTS.SECURE_DELETE) != 0)
                    //{
                    //    int off = (int)(divs[i]) - (int)(parent.Data);
                    //    if ((off + newPages[i]) > (int)bt.UsableSize)
                    //    {
                    //        rc = SysEx.CORRUPT_BKPT();
                    //        Array.Clear(oldPages[0].Data, 0, oldPages[0].Data.Length);
                    //        goto balance_cleanup;
                    //    }
                    //    else
                    //    {
                    //        memcpy(ovflSpace,off, divs,i,, sizeNew[i]);
                    //        divs[i] = ovflSpace[divs[i] - parent.Data];
                    //    }
                    //}
                    dropCell(parent, i + nxDiv - parent.Overflows, sizeNew[i], ref rc);
                }
            }

            // Make nMaxCells a multiple of 4 in order to preserve 8-byte alignment
            maxCells = (maxCells + 3U) & ~3U;

            // Allocate space for memory structures
            uint j, k;
            //uint k = bt.PageSize + SysEx.ROUND8(sizeof(MemPage));
            //int szScratch = // Size of scratch memory requested
            //     maxCells * sizeof(byte *) // apCell
            //   + maxCells * sizeof(ushort) // szCell
            //   + bt.PageSize // aSpace1
            //   + k * oldPagesUsed; // Page copies (apCopy)
            Pid cells = 0; // Number of cells in apCell[]
            cell = SysEx.ScratchAlloc(cell, (int)maxCells);
            if (cell == null)
            {
                rc = RC.NOMEM;
                goto balance_cleanup;
            }

            var sizeCell = new ushort[1]; // Local size of all cells in apCell[]
            if (sizeCell.Length < maxCells)
                Array.Resize(ref sizeCell, (int)maxCells); // sizeCell = (ushort *)&cell[maxCells];
            //var space1 = new byte[bt.PageSize * maxCells]; // Space for copies of dividers cells
            //Debug.Assert(_HASALIGNMENT8(space1));

            // Load pointers to all cells on sibling pages and the divider cells into the local apCell[] array.  Make copies of the divider cells
            // into space obtained from aSpace1[] and remove the divider cells from pParent.
            //
            // If the siblings are on leaf pages, then the child pointers of the divider cells are stripped from the cells before they are copied
            // into aSpace1[].  In this way, all cells in apCell[] are without child pointers.  If siblings are not leaves, then all cell in
            // apCell[] include child pointers.  Either way, all cells in apCell[] are alike.
            //
            // leafCorrection:  4 if pPage is a leaf.  0 if pPage is not a leaf.
            // leafData:  1 if pPage holds key+data and pParent holds only keys.
            ushort leafCorrection = (ushort)(oldPages[0].Leaf ? 4 : 0); // 4 if pPage is a leaf.  0 if not
            uint leafData = (oldPages[0].HasData ? 1U : 0U); // True if pPage is a leaf of a LEAFDATA tree
            for (i = 0U; i < oldPagesUsed; i++)
            {
                // Before doing anything else, take a copy of the i'th original sibling The rest of this function will use data from the copies rather
                // that the original pages since the original pages will be in the process of being overwritten.
                //MemPage oldPage = copyPages[i] = (MemPage *)&space1[bt.PageSize + k * i];
                //_memcpy(oldPage, oldPages[i], sizeof(MemPage));
                //oldPage.Data = (void *)&oldPages[1];
                //_memcpy(oldPage.Data, oldPages[i].Data, bt.PageSize);
                MemPage oldPage = copyPages[i] = oldPages[i].memcopy();

                int limit = oldPage.Cells + oldPage.Overflows;
                if (oldPage.Overflows > 0 || true)
                {
                    for (j = 0U; j < limit; j++)
                    {
                        Debug.Assert(cells < maxCells);
                        uint fofc = findOverflowCell(oldPage, j);
                        sizeCell[cells] = cellSizePtr(oldPage, fofc);
                        // Copy the Data Locally
                        if (cell[cells] == null)
                            cell[cells] = new byte[sizeCell[cells]];
                        else if (cell[cells].Length < sizeCell[cells])
                            Array.Resize(ref cell[cells], sizeCell[cells]);
                        if (fofc < 0)  // Overflow Cell
                            Buffer.BlockCopy(oldPage.Ovfls[-(fofc + 1)].Cell, 0, cell[cells], 0, sizeCell[cells]);
                        else
                            Buffer.BlockCopy(oldPage.Data, (int)fofc, cell[cells], 0, sizeCell[cells]);
                        cells++;
                    }
                }
                //else
                //{
                //    var data = oldPage.Data;
                //    var maskPage = oldPage.MaskPage;
                //    var cellOffset = oldPage.CellOffset;
                //    for (int j = 0; j < limit; j++)
                //    {
                //        Debug.Assert(cells < maxCells);
                //        cell[cells] = findCellv2(data, maskPage, cellOffset, j);
                //        sizeCell[cells] = cellSizePtr(oldPage, cell[cells]);
                //        cells++;
                //    }
                //}
                if (i < oldPagesUsed - 1 && leafData == 0)
                {
                    var size = (ushort)sizeNew[i];
                    Debug.Assert(cells < maxCells);
                    sizeCell[cells] = size;
                    var temp = new byte[size + leafCorrection];
                    Debug.Assert(size <= bt.MaxLocal + 23);
                    Buffer.BlockCopy(parent.Data, (int)divs_[i], temp, 0, size);
                    if (cell[cells] == null || cell[cells].Length < size)
                        Array.Resize(ref cell[cells], size);
                    Buffer.BlockCopy(temp, leafCorrection, cell[cells], 0, size);
                    Debug.Assert(leafCorrection == 0 || leafCorrection == 4);
                    sizeCell[cells] = (ushort)(sizeCell[cells] - leafCorrection);
                    if (!oldPage.Leaf)
                    {
                        Debug.Assert(leafCorrection == 0);
                        Debug.Assert(oldPage.HdrOffset == 0);
                        // The right pointer of the child page pOld becomes the left pointer of the divider cell
                        Buffer.BlockCopy(oldPage.Data, 8, cell[cells], 0, 4);
                    }
                    else
                    {
                        Debug.Assert(leafCorrection == 4);
                        if (sizeCell[cells] < 4) // Do not allow any cells smaller than 4 bytes.
                            sizeCell[cells] = 4;
                    }
                    cells++;
                }
            }

            // Figure out the number of pages needed to hold all nCell cells. Store this number in "k".  Also compute szNew[] which is the total
            // size of all cells on the i-th page and cntNew[] which is the index in apCell[] of the cell that divides page i from page i+1.  
            // cntNew[k] should equal nCell.
            //
            // Values computed by this block:
            //           k: The total number of sibling pages
            //    szNew[i]: Spaced used on the i-th sibling page.
            //   cntNew[i]: Index in apCell[] and szCell[] for the first cell to the right of the i-th sibling page.
            // usableSpace: Number of bytes of space available on each sibling.
            ushort subtotal; // Subtotal of bytes in cells on one page
            var usableSpace = (uint)bt.UsableSize - 12 + leafCorrection; // Bytes in pPage beyond the header
            for (subtotal = 0, k = i = 0; i < cells; i++)
            {
                Debug.Assert(i < maxCells);
                subtotal += (ushort)(sizeCell[i] + 2U);
                if (subtotal > usableSpace)
                {
                    sizeNew[k] = (ushort)(subtotal - sizeCell[i]);
                    countNew[k] = i;
                    if (leafData != 0) i--;
                    subtotal = 0;
                    k++;
                    if (k > NB + 1) { rc = SysEx.CORRUPT_BKPT(); goto balance_cleanup; }
                }
            }
            sizeNew[k] = subtotal;
            countNew[k] = cells;
            k++;

            // The packing computed by the previous block is biased toward the siblings on the left side.  The left siblings are always nearly full, while the
            // right-most sibling might be nearly empty.  This block of code attempts to adjust the packing of siblings to get a better balance.
            //
            // This adjustment is more than an optimization.  The packing above might be so out of balance as to be illegal.  For example, the right-most
            // sibling might be completely empty.  This adjustment is not optional.
            for (i = k - 1; i > 0; i--)
            {
                ushort sizeRight = sizeNew[i];  // Size of sibling on the right
                ushort sizeLeft = sizeNew[i - 1]; // Size of sibling on the left
                Pid r = countNew[i - 1] - 1; // Index of right-most cell in left sibling
                uint d = r + 1U - leafData; // Index of first cell to the left of right sibling
                Debug.Assert(d < maxCells);
                Debug.Assert(r < maxCells);
                while (sizeRight == 0 || (!bulk && sizeRight + sizeCell[d] + 2 <= sizeLeft - (sizeCell[r] + 2)))
                {
                    sizeRight += (ushort)(sizeCell[d] + 2U);
                    sizeLeft -= (ushort)(sizeCell[r] + 2U);
                    countNew[i - 1]--;
                    r = countNew[i - 1] - 1;
                    d = r + 1U - leafData;
                }
                sizeNew[i] = sizeRight;
                sizeNew[i - 1] = sizeLeft;
            }

            // Either we found one or more cells (cntnew[0])>0) or pPage is a virtual root page.  A virtual root page is when the real root
            // page is page 1 and we are the only child of that page.
            //
            // UPDATE:  The assert() below is not necessarily true if the database file is corrupt.  The corruption will be detected and reported later
            // in this procedure so there is no need to act upon it now.
#if false
            Debug.Assert(countNew[0] > 0 || (parent.ID == 1 && parent.Cells == 0));
#endif

            TRACE("BALANCE: old: %d %d %d  ",
                oldPages[0].ID,
                oldPagesUsed >= 2 ? oldPages[1].ID : 0,
                oldPagesUsed >= 3 ? oldPages[2].ID : 0);

            // Allocate k new pages.  Reuse old pages where possible.
            if (oldPages[0].ID <= 1)
            {
                rc = SysEx.CORRUPT_BKPT();
                goto balance_cleanup;
            }
            int pageFlags = oldPages[0].Data[0]; // Value of pPage->aData[0]
            for (i = 0; i < k; i++)
            {
                var newPage = new MemPage();
                if (i < oldPagesUsed)
                {
                    newPage = newPages[i] = oldPages[i];
                    oldPages[i] = null;
                    rc = Pager.Write(newPage.DBPage);
                    newPagesUsed++;
                    if (rc != RC.OK) goto balance_cleanup;
                }
                else
                {
                    Debug.Assert(i > 0);
                    rc = allocateBtreePage(bt, ref newPage, ref id, (bulk ? 1 : id), BTALLOC.ANY);
                    if (rc != RC.OK) goto balance_cleanup;
                    newPages[i] = newPage;
                    newPagesUsed++;

                    // Set the pointer-map entry for the new sibling page.
#if !OMIT_AUTOVACUUM
                    if (bt.AutoVacuum)
                    {
                        ptrmapPut(bt, newPage.ID, PTRMAP.BTREE, parent.ID, ref rc);
                        if (rc != RC.OK)
                            goto balance_cleanup;
                    }
#endif
                }
            }

            // Free any old pages that were not reused as new pages.
            while (i < oldPagesUsed)
            {
                freePage(oldPages[i], ref rc);
                if (rc != RC.OK) goto balance_cleanup;
                releasePage(oldPages[i]);
                oldPages[i] = null;
                i++;
            }

            // Put the new pages in accending order.  This helps to keep entries in the disk file in order so that a scan
            // of the table is a linear scan through the file.  That in turn helps the operating system to deliver pages
            // from the disk more rapidly.
            //
            // An O(n^2) insertion sort algorithm is used, but since n is never more than NB (a small constant), that should
            // not be a problem.
            //
            // When NB==3, this one optimization makes the database about 25% faster for large insertions and deletions.
            for (i = 0; i < k - 1; i++)
            {
                Pid minV = newPages[i].ID;
                Pid minI = (Pid)i;
                for (j = i + 1; j < k; j++)
                {
                    if (newPages[j].ID < minV)
                    {
                        minI = j;
                        minV = newPages[j].ID;
                    }
                }
                if (minI > i)
                {
                    var t = newPages[i];
                    newPages[i] = newPages[minI];
                    newPages[minI] = t;
                }
            }
            TRACE("new: %d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n",
                newPages[0].ID, sizeNew[0],
                newPagesUsed >= 2 ? newPages[1].ID : 0, newPagesUsed >= 2 ? sizeNew[1] : 0,
                newPagesUsed >= 3 ? newPages[2].ID : 0, newPagesUsed >= 3 ? sizeNew[2] : 0,
                newPagesUsed >= 4 ? newPages[3].ID : 0, newPagesUsed >= 4 ? sizeNew[3] : 0,
                newPagesUsed >= 5 ? newPages[4].ID : 0, newPagesUsed >= 5 ? sizeNew[4] : 0);

            Debug.Assert(Pager.Iswriteable(parent.DBPage));
            ConvertEx.Put4(parent.Data, right_, newPages[newPagesUsed - 1].ID);

            // Evenly distribute the data in apCell[] across the new pages. Insert divider cells into pParent as necessary.
            int ovflSpaceID = 0; // First unused byte of aOvflSpace[]
            j = 0;
            for (i = 0; i < newPagesUsed; i++)
            {
                // Assemble the new sibling page.
                MemPage newPage = newPages[i];
                Debug.Assert(j < maxCells);
                zeroPage(newPage, pageFlags);
                assemblePage(newPage, (int)(countNew[i] - j), cell, sizeCell, (int)j);
                Debug.Assert(newPage.Cells > 0 || (newPagesUsed == 1 && countNew[0] == 0));
                Debug.Assert(newPage.Overflows == 0);

                j = countNew[i];

                // If the sibling page assembled above was not the right-most sibling, insert a divider cell into the parent page.
                Debug.Assert(i < newPagesUsed - 1 || j == cells);
                if (j < cells)
                {
                    Debug.Assert(j < maxCells);
                    byte[] pCell = cell[j];
                    int size = sizeCell[j] + leafCorrection;
                    byte[] pTemp = new byte[size]; //&aOvflSpace[iOvflSpace];
                    if (!newPage.Leaf)
                        Buffer.BlockCopy(pCell, 0, newPage.Data, 8, 4);
                    else if (leafData != 0)
                    {
                        // If the tree is a leaf-data tree, and the siblings are leaves, then there is no divider cell in apCell[]. Instead, the divider 
                        // cell consists of the integer key for the right-most cell of the sibling-page assembled above only.
                        j--;
                        CellInfo info = new CellInfo();
                        btreeParseCellPtr(newPage, cell[j], ref info);
                        pCell = pTemp;
                        size = 4 + ConvertEx.PutVarint(pCell, 4, (ulong)info.Key);
                        pTemp = null;
                    }
                    else
                    {
                        { byte[] cell_4 = new byte[pCell.Length + 4]; Buffer.BlockCopy(pCell, 0, cell_4, 4, pCell.Length); pCell = cell_4; } //pCell -= 4;  
                        // Obscure case for non-leaf-data trees: If the cell at pCell was previously stored on a leaf node, and its reported size was 4
                        // bytes, then it may actually be smaller than this (see btreeParseCellPtr(), 4 bytes is the minimum size of
                        // any cell). But it is important to pass the correct size to insertCell(), so reparse the cell now.
                        //
                        // Note that this can never happen in an SQLite data file, as all cells are at least 4 bytes. It only happens in b-trees used
                        // to evaluate "IN (SELECT ...)" and similar clauses.
                        if (sizeCell[j] == 4)
                        {
                            Debug.Assert(leafCorrection == 4);
                            size = cellSizePtr(parent, pCell);
                        }
                    }
                    ovflSpaceID += size;
                    Debug.Assert(size <= bt.MaxLocal + 23);
                    Debug.Assert(ovflSpaceID <= (int)bt.PageSize);
                    insertCell(parent, nxDiv, pCell, (ushort)size, pTemp, newPage.ID, ref rc);
                    if (rc != RC.OK)
                        goto balance_cleanup;
                    Debug.Assert(Pager.Iswriteable(parent.DBPage));

                    j++;
                    nxDiv++;
                }
            }
            Debug.Assert(j == cells);
            Debug.Assert(oldPagesUsed > 0);
            Debug.Assert(newPagesUsed > 0);
            if ((pageFlags & PTF_LEAF) == 0)
                Buffer.BlockCopy(copyPages[oldPagesUsed - 1].Data, 8, newPages[newPagesUsed - 1].Data, 8, 4); //u8* zChild = &apCopy[nOld - 1].aData[8]; memcpy( apNew[nNew - 1].aData[8], zChild, 4 );

            if (isRoot && parent.Cells == 0 && parent.HdrOffset <= newPages[0].Frees)
            {
                // The root page of the b-tree now contains no cells. The only sibling page is the right-child of the parent. Copy the contents of the
                // child page into the parent, decreasing the overall height of the b-tree structure by one. This is described as the "balance-shallower"
                // sub-algorithm in some documentation.
                //
                // If this is an auto-vacuum database, the call to copyNodeContent() sets all pointer-map entries corresponding to database image pages 
                // for which the pointer is stored within the content being copied.
                //
                // The second assert below verifies that the child page is defragmented (it must be, as it was just reconstructed using assemblePage()). This
                // is important if the parent page happens to be page 1 of the database image.
                Debug.Assert(newPagesUsed == 1);
                Debug.Assert(newPages[0].Frees == (ConvertEx.Get2(newPages[0].Data, 5) - newPages[0].CellOffset - newPages[0].Cells * 2));
                copyNodeContent(newPages[0], parent, ref rc);
                freePage(newPages[0], ref rc);
            }
#if !OMIT_AUTOVACUUM
            else if (bt.AutoVacuum)
            {
                // Fix the pointer-map entries for all the cells that were shifted around. There are several different types of pointer-map entries that need to
                // be dealt with by this routine. Some of these have been set already, but many have not. The following is a summary:
                //
                //   1) The entries associated with new sibling pages that were not siblings when this function was called. These have already
                //      been set. We don't need to worry about old siblings that were moved to the free-list - the freePage() code has taken care
                //      of those.
                //
                //   2) The pointer-map entries associated with the first overflow page in any overflow chains used by new divider cells. These 
                //      have also already been taken care of by the insertCell() code.
                //
                //   3) If the sibling pages are not leaves, then the child pages of cells stored on the sibling pages may need to be updated.
                //
                //   4) If the sibling pages are not internal intkey nodes, then any overflow pages used by these cells may need to be updated
                //      (internal intkey nodes never contain pointers to overflow pages).
                //
                //   5) If the sibling pages are not leaves, then the pointer-map entries for the right-child pages of each sibling may need
                //      to be updated.
                //
                // Cases 1 and 2 are dealt with above by other code. The next block deals with cases 3 and 4 and the one after that, case 5. Since
                // setting a pointer map entry is a relatively expensive operation, this code only sets pointer map entries for child or overflow pages that have
                // actually moved between pages.
                MemPage newPage = newPages[0];
                MemPage oldPage = copyPages[0];
                uint overflows = oldPage.Overflows;
                Pid nextOldID = oldPage.Cells + overflows;
                int overflowID = (overflows != 0 ? oldPage.OvflIdxs[0] : -1);
                j = 0; // Current 'old' sibling page
                k = 0; // Current 'new' sibling page
                for (i = 0; i < cells; i++)
                {
                    bool isDivider = false;
                    while (i == nextOldID)
                    {
                        // Cell i is the cell immediately following the last cell on old sibling page j. If the siblings are not leaf pages of an
                        // intkey b-tree, then cell i was a divider cell.
                        Debug.Assert(j + 1 < copyPages.Length);
                        Debug.Assert(j + 1 < oldPagesUsed);
                        oldPage = copyPages[++j];
                        nextOldID = (i + (leafData == 0 ? 1U : 0U) + oldPage.Cells + oldPage.Overflows);
                        if (oldPage.Overflows != 0)
                        {
                            overflows = oldPage.Overflows;
                            overflowID = (int)(i + (leafData == 0 ? 1U : 0U) + oldPage.OvflIdxs[0]);
                        }
                        isDivider = (leafData == 0);
                    }

                    Debug.Assert(overflows > 0 || overflowID < i);
                    Debug.Assert(overflows < 2 || oldPage.OvflIdxs[0] == oldPage.OvflIdxs[1] - 1);
                    Debug.Assert(overflows < 3 || oldPage.OvflIdxs[1] == oldPage.OvflIdxs[2] - 1);
                    if (i == overflowID)
                    {
                        isDivider = true;
                        if (--overflows > 0)
                            overflowID++;
                    }

                    if (i == countNew[k])
                    {
                        // Cell i is the cell immediately following the last cell on new sibling page k. If the siblings are not leaf pages of an
                        // intkey b-tree, then cell i is a divider cell.
                        newPage = newPages[++k];
                        if (leafData == 0)
                            continue;
                    }
                    Debug.Assert(j < oldPagesUsed);
                    Debug.Assert(k < newPagesUsed);

                    // If the cell was originally divider cell (and is not now) or an overflow cell, or if the cell was located on a different sibling
                    // page before the balancing, then the pointer map entries associated with any child or overflow pages need to be updated.
                    if (isDivider || oldPage.ID != newPage.ID)
                    {
                        if (leafCorrection == 0)
                            ptrmapPut(bt, ConvertEx.Get4(cell[i]), PTRMAP.BTREE, newPage.ID, ref rc);
                        if (sizeCell[i] > newPage.MinLocal)
                            ptrmapPutOvflPtr(newPage, cell[i], ref rc);
                    }
                }
#endif

                if (leafCorrection == 0)
                {
                    for (i = 0; i < newPagesUsed; i++)
                    {
                        uint key = ConvertEx.Get4(newPages[i].Data, 8);
                        ptrmapPut(bt, key, PTRMAP.BTREE, newPages[i].ID, ref rc);
                    }
                }

#if false
			// The ptrmapCheckPages() contains assert() statements that verify that all pointer map pages are set correctly. This is helpful while 
			// debugging. This is usually disabled because a corrupt database may cause an assert() statement to fail.
            ptrmapCheckPages(newPages, newPagesUsed);
            ptrmapCheckPages(parent, 1);
#endif
            }

            Debug.Assert(parent.IsInit);
            TRACE("BALANCE: finished: old=%d new=%d cells=%d\n", oldPagesUsed, newPagesUsed, cells);

        // Cleanup before returning.
        balance_cleanup:
            SysEx.ScratchFree(cell);
            for (i = 0; i < oldPagesUsed; i++)
                releasePage(oldPages[i]);
            for (i = 0; i < newPagesUsed; i++)
                releasePage(newPages[i]);

            return rc;
        }

        static RC balance_deeper(MemPage root, ref MemPage childOut)
        {
            var bt = root.Bt; // The BTree

            Debug.Assert(root.Overflows > 0);
            Debug.Assert(MutexEx.Held(bt.Mutex));

            // Make pRoot, the root page of the b-tree, writable. Allocate a new page that will become the new right-child of pPage. Copy the contents
            // of the node stored on pRoot into the new child page.
            MemPage child = null; // Pointer to a new child page
            Pid childID = 0; // Page number of the new child page
            var rc = Pager.Write(root.DBPage);
            if (rc == RC.OK)
            {
                rc = allocateBtreePage(bt, ref child, ref childID, root.ID, BTALLOC.ANY);
                copyNodeContent(root, child, ref rc);
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum)
                    ptrmapPut(bt, childID, PTRMAP.BTREE, root.ID, ref rc);
#endif
            }
            if (rc != RC.OK)
            {
                childOut = null;
                releasePage(child);
                return rc;
            }
            Debug.Assert(Pager.Iswriteable(child.DBPage));
            Debug.Assert(Pager.Iswriteable(root.DBPage));
            Debug.Assert(child.Cells == root.Cells);

            TRACE("BALANCE: copy root %d into %d\n", root.ID, child.ID);

            // Copy the overflow cells from pRoot to pChild
            Array.Copy(root.OvflIdxs, child.OvflIdxs, root.Overflows);
            Array.Copy(root.Ovfls, child.Ovfls, root.Overflows);
            child.Overflows = root.Overflows;

            // Zero the contents of pRoot. Then install pChild as the right-child.
            zeroPage(root, child.Data[0] & ~PTF_LEAF);
            ConvertEx.Put4(root.Data, root.HdrOffset + 8, childID);

            childOut = child;
            return RC.OK;
        }

        static byte[] _balanceQuickSpace = new byte[13];
        static RC balance(BtCursor cur)
        {
#if DEBUG
            int balance_quick_called = 0;
            int balance_deeper_called = 0;
#else
            int balance_quick_called = 0;
            int balance_deeper_called = 0;
#endif

            byte[] free = null;
            int min = (int)cur.Bt.UsableSize * 2 / 3;
            RC rc = RC.OK;
            do
            {
                int pageID = cur.ID;
                MemPage page = cur.Pages[pageID];

                if (pageID == 0)
                {
                    if (page.Overflows != 0)
                    {
                        // The root page of the b-tree is overfull. In this case call the balance_deeper() function to create a new child for the root-page
                        // and copy the current contents of the root-page to it. The next iteration of the do-loop will balance the child page.
                        Debug.Assert(balance_deeper_called++ == 0);
                        rc = balance_deeper(page, ref cur.Pages[1]);
                        if (rc == RC.OK)
                        {
                            cur.ID = 1;
                            cur.Idxs[0] = 0;
                            cur.Idxs[1] = 0;
                            Debug.Assert(cur.Pages[1].Overflows != 0);
                        }
                    }
                    else
                        break;
                }
                else if (page.Overflows == 0 && page.Frees <= min)
                    break;
                else
                {
                    MemPage parent = cur.Pages[pageID - 1];
                    ushort idx = cur.Idxs[pageID - 1];

                    rc = Pager.Write(parent.DBPage);
                    if (rc == RC.OK)
                    {
#if !OMIT_QUICKBALANCE
                        if (page.HasData && page.Overflows == 1 && page.OvflIdxs[0] == page.Cells && parent.ID != 1 && parent.Cells == idx)
                        {
                            // Call balance_quick() to create a new sibling of pPage on which to store the overflow cell. balance_quick() inserts a new cell
                            // into pParent, which may cause pParent overflow. If this happens, the next interation of the do-loop will balance pParent 
                            // use either balance_nonroot() or balance_deeper(). Until this happens, the overflow cell is stored in the aBalanceQuickSpace[] buffer. 
                            //
                            // The purpose of the following assert() is to check that only a single call to balance_quick() is made for each call to this
                            // function. If this were not verified, a subtle bug involving reuse of the aBalanceQuickSpace[] might sneak in.
                            Debug.Assert((balance_quick_called++) == 0);
                            rc = balance_quick(parent, page, _balanceQuickSpace);
                        }
                        else
#endif
                        {
                            // In this case, call balance_nonroot() to redistribute cells between pPage and up to 2 of its sibling pages. This involves
                            // modifying the contents of pParent, which may cause pParent to become overfull or underfull. The next iteration of the do-loop
                            // will balance the parent page to correct this.
                            // 
                            // If the parent page becomes overfull, the overflow cell or cells are stored in the pSpace buffer allocated immediately below. 
                            // A subsequent iteration of the do-loop will deal with this by calling balance_nonroot() (balance_deeper() may be called first,
                            // but it doesn't deal with overflow cells - just moves them to a different page). Once this subsequent call to balance_nonroot() 
                            // has completed, it is safe to release the pSpace buffer used by the previous call, as the overflow cell data will have been 
                            // copied either into the body of a database page or into the new pSpace buffer passed to the latter call to balance_nonroot().
                            byte[] space = null; //PCache.PageAlloc2((int)cur.Bt.PageSize);
                            rc = balance_nonroot(parent, idx, space, pageID == 1, cur.Hints != 0);
                            if (free != null)
                            {
                                // If pFree is not NULL, it points to the pSpace buffer used  by a previous call to balance_nonroot(). Its contents are
                                // now stored either on real database pages or within the new pSpace buffer, so it may be safely freed here.
                                PCache.PageFree2(ref free);
                            }

                            // The pSpace buffer will be freed after the next call to balance_nonroot(), or just before this function returns, whichever comes first.
                            free = space;
                        }
                    }

                    page.Overflows = 0;

                    // The next iteration of the do-loop balances the parent page.
                    releasePage(page);
                    cur.ID--;
                }
            } while (rc == RC.OK);

            if (free != null)
                PCache.PageFree2(ref free);
            return rc;
        }

        #endregion

        #region Insert / Delete / CreateTable / DropTable

        public static RC Insert(BtCursor cur, byte[] key, long keyLength, byte[] data, int dataLength, int zero, int appendBias, int seekResult)
        {
            if (cur.State == CURSOR.FAULT)
            {
                Debug.Assert(cur.SkipNext != 0);
                return (RC)cur.SkipNext;
            }

            Btree p = cur.Btree;
            BtShared bt = p.Bt;
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(cur.WrFlag && bt.InTransaction == TRANS.WRITE && (bt.BtsFlags & BTS.READ_ONLY) == 0);
            Debug.Assert(hasSharedCacheTableLock(p, cur.RootID, cur.KeyInfo != null, LOCK.WRITE));

            // Assert that the caller has been consistent. If this cursor was opened expecting an index b-tree, then the caller should be inserting blob
            // keys with no associated data. If the cursor was opened expecting an intkey table, the caller should be inserting integer keys with a
            // blob of associated data.
            Debug.Assert((key == null) == (cur.KeyInfo == null));

            // Save the positions of any other cursors open on this table.
            //
            // In some cases, the call to btreeMoveto() below is a no-op. For example, when inserting data into a table with auto-generated integer
            // keys, the VDBE layer invokes sqlite3BtreeLast() to figure out the integer key to use. It then calls this function to actually insert the 
            // data into the intkey B-Tree. In this case btreeMoveto() recognizes that the cursor is already where it needs to be and returns without
            // doing any work. To avoid thwarting these optimizations, it is important not to clear the cursor here.
            RC rc = saveAllCursors(bt, cur.RootID, cur);
            if (rc != RC.OK) return rc;

            // If this is an insert into a table b-tree, invalidate any incrblob cursors open on the row being replaced (assuming this is a replace
            // operation - if it is not, the following is a no-op).
            if (cur.KeyInfo == null)
                invalidateIncrblobCursors(p, keyLength, false);

            int loc = seekResult; // -1: before desired location  +1: after
            if (loc == 0)
            {
                rc = btreeMoveto(cur, key, keyLength, appendBias, ref loc);
                if (rc != RC.OK) return rc;
            }
            Debug.Assert(cur.State == CURSOR.VALID || (cur.State == CURSOR.INVALID && loc != 0));

            MemPage page = cur.Pages[cur.ID];
            Debug.Assert(page.IntKey || keyLength >= 0);
            Debug.Assert(page.Leaf || !page.IntKey);

            TRACE("INSERT: table=%d nkey=%lld ndata=%d page=%d %s\n", cur.RootID, keyLength, dataLength, page.ID, loc == 0 ? "overwrite" : "new entry");
            Debug.Assert(page.IsInit);
            allocateTempSpace(bt);
            byte[] newCell = bt.TmpSpace;
            if (newCell == null) return RC.NOMEM;
            ushort sizeNew = 0;
            rc = fillInCell(page, newCell, key, keyLength, data, dataLength, zero, ref sizeNew);
            if (rc != RC.OK) goto end_insert;
            Debug.Assert(sizeNew == cellSizePtr(page, newCell));
            Debug.Assert(sizeNew <= MX_CELL_SIZE(bt));
            uint idx = cur.Idxs[cur.ID];
            if (loc == 0)
            {
                Debug.Assert(idx < page.Cells);
                rc = Pager.Write(page.DBPage);
                if (rc != RC.OK)
                    goto end_insert;
                uint oldCell_ = findCell(page, idx);
                if (!page.Leaf)
                {
                    //_memcpy(newCell, oldCell, 4);
                    newCell[0] = page.Data[oldCell_ + 0];
                    newCell[1] = page.Data[oldCell_ + 1];
                    newCell[2] = page.Data[oldCell_ + 2];
                    newCell[3] = page.Data[oldCell_ + 3];
                }
                ushort sizeOld = cellSizePtr(page, oldCell_);
                rc = clearCell(page, oldCell_);
                dropCell(page, idx, sizeOld, ref rc);
                if (rc != RC.OK) goto end_insert;
            }
            else if (loc < 0 && page.Cells > 0)
            {
                Debug.Assert(page.Leaf);
                idx = ++cur.Idxs[cur.ID];
            }
            else
                Debug.Assert(page.Leaf);
            insertCell(page, idx, newCell, sizeNew, null, 0, ref rc);
            Debug.Assert(rc != RC.OK || page.Cells > 0 || page.Overflows > 0);

            // If no error has occurred and pPage has an overflow cell, call balance() to redistribute the cells within the tree. Since balance() may move
            // the cursor, zero the BtCursor.info.nSize and BtCursor.validNKey variables.
            //
            // Previous versions of SQLite called moveToRoot() to move the cursor back to the root page as balance() used to invalidate the contents
            // of BtCursor.apPage[] and BtCursor.aiIdx[]. Instead of doing that, set the cursor state to "invalid". This makes common insert operations
            // slightly faster.
            //
            // There is a subtle but important optimization here too. When inserting multiple records into an intkey b-tree using a single cursor (as can
            // happen while processing an "INSERT INTO ... SELECT" statement), it is advantageous to leave the cursor pointing to the last entry in
            // the b-tree if possible. If the cursor is left pointing to the last entry in the table, and the next row inserted has an integer key
            // larger than the largest existing key, it is possible to insert the row without seeking the cursor. This can be a big performance boost.
            cur.Info.Size = 0;
            cur.ValidNKey = false;
            if (rc == RC.OK && page.Overflows != 0)
            {
                rc = balance(cur);

                // Must make sure nOverflow is reset to zero even if the balance() fails. Internal data structure corruption will result otherwise. 
                // Also, set the cursor state to invalid. This stops saveCursorPosition() from trying to save the current position of the cursor.
                cur.Pages[cur.ID].Overflows = 0;
                cur.State = CURSOR.INVALID;
            }
            Debug.Assert(cur.Pages[cur.ID].Overflows == 0);

        end_insert:
            return rc;
        }

        public static RC Delete(BtCursor cur)
        {
            Btree p = cur.Btree;
            BtShared bt = p.Bt;
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(bt.InTransaction == TRANS.WRITE);
            Debug.Assert((bt.BtsFlags & BTS.READ_ONLY) == 0);
            Debug.Assert(cur.WrFlag);
            Debug.Assert(hasSharedCacheTableLock(p, cur.RootID, cur.KeyInfo != null, LOCK.WRITE));
            Debug.Assert(!hasReadConflicts(p, cur.RootID));

            if (C._NEVER(cur.Idxs[cur.ID] >= cur.Pages[cur.ID].Cells) || C._NEVER(cur.State != CURSOR.VALID))
                return RC.ERROR; // Something has gone awry.

            int cellDepth = cur.ID; // Depth of node containing pCell
            uint cellIdx = cur.Idxs[cellDepth]; // Index of cell to delete
            MemPage page = cur.Pages[cellDepth]; // Page to delete cell from
            uint cell_ = findCell(page, cellIdx); // Pointer to cell to delete

            // If the page containing the entry to delete is not a leaf page, move the cursor to the largest entry in the tree that is smaller than
            // the entry being deleted. This cell will replace the cell being deleted from the internal node. The 'previous' entry is used for this instead
            // of the 'next' entry, as the previous entry is always a part of the sub-tree headed by the child page of the cell being deleted. This makes
            // balancing the tree following the delete operation easier.
            RC rc;
            if (!page.Leaf)
            {
                int notUsed = 0;
                rc = Previous(cur, ref notUsed);
                if (rc != RC.OK) return rc;
            }

            // Save the positions of any other cursors open on this table before making any modifications. Make the page containing the entry to be 
            // deleted writable. Then free any overflow pages associated with the entry and finally remove the cell itself from within the page.  
            rc = saveAllCursors(bt, cur.RootID, cur);
            if (rc != RC.OK) return rc;

            // If this is a delete operation to remove a row from a table b-tree, invalidate any incrblob cursors open on the row being deleted.
            if (cur.KeyInfo == null)
                invalidateIncrblobCursors(p, cur.Info.Key, false);

            rc = Pager.Write(page.DBPage);
            if (rc != RC.OK) return rc;
            rc = clearCell(page, cell_);
            dropCell(page, cellIdx, cellSizePtr(page, cell_), ref rc);
            if (rc != RC.OK) return rc;

            // If the cell deleted was not located on a leaf page, then the cursor is currently pointing to the largest entry in the sub-tree headed
            // by the child-page of the cell that was just deleted from an internal node. The cell from the leaf node needs to be moved to the internal
            // node to replace the deleted cell.
            if (!page.Leaf)
            {
                MemPage leaf = cur.Pages[cur.ID];
                Pid n = cur.Pages[cellDepth + 1].ID;

                cell_ = findCell(leaf, (uint)leaf.Cells - 1);
                ushort sizeCell = cellSizePtr(leaf, cell_);
                Debug.Assert(MX_CELL_SIZE(bt) >= sizeCell);

                //allocateTempSpace(bt);
                //byte[] tmp = Bt.TmpSpace;

                rc = Pager.Write(leaf.DBPage);
                { byte[] cell_4 = new byte[sizeCell + 4]; Buffer.BlockCopy(leaf.Data, (int)cell_ - 4, cell_4, 0, (int)sizeCell + 4); insertCell(page, cellIdx, cell_4, (ushort)(sizeCell + 4), null, n, ref rc); }
                dropCell(leaf, (uint)(leaf.Cells - 1), sizeCell, ref rc);
                if (rc != RC.OK) return rc;
            }

            // Balance the tree. If the entry deleted was located on a leaf page, then the cursor still points to that page. In this case the first
            // call to balance() repairs the tree, and the if(...) condition is never true.
            //
            // Otherwise, if the entry deleted was on an internal node page, then pCur is pointing to the leaf page from which a cell was removed to
            // replace the cell deleted from the internal node. This is slightly tricky as the leaf node may be underfull, and the internal node may
            // be either under or overfull. In this case run the balancing algorithm on the leaf node first. If the balance proceeds far enough up the
            // tree that we can be sure that any problem in the internal node has been corrected, so be it. Otherwise, after balancing the leaf node,
            // walk the cursor up the tree to the internal node and balance it as well.
            rc = balance(cur);
            if (rc == RC.OK && cur.ID > cellDepth)
            {
                while (cur.ID > cellDepth)
                    releasePage(cur.Pages[cur.ID--]);
                rc = balance(cur);
            }

            if (rc == RC.OK)
                moveToRoot(cur);
            return rc;
        }

        static RC btreeCreateTable(Btree p, ref int tableID, int createTabFlags)
        {
            BtShared bt = p.Bt;
            Debug.Assert(p.HoldsMutex());
            Debug.Assert(bt.InTransaction == TRANS.WRITE);
            Debug.Assert((bt.BtsFlags & BTS.READ_ONLY) == 0);

            RC rc;
            MemPage root = new MemPage();
            Pid rootID = 0;
#if OMIT_AUTOVACUUM
            rc = allocateBtreePage(bt, ref root, ref rootID, 1, BTALLOC.ANY);
            if (rc != RC.OK)
                return rc;
#else
            if (bt.AutoVacuum)
            {
                // Creating a new table may probably require moving an existing database to make room for the new tables root page. In case this page turns
                // out to be an overflow page, delete all overflow page-map caches held by open cursors.
                invalidateAllOverflowCache(bt);

                // Read the value of meta[3] from the database to determine where the root page of the new table should go. meta[3] is the largest root-page
                // created so far, so the new root-page is (meta[3]+1).
                p.GetMeta(META.LARGEST_ROOT_PAGE, ref rootID);
                rootID++;

                // The new root-page may not be allocated on a pointer-map page, or the PENDING_BYTE page.
                while (rootID == PTRMAP_PAGENO(bt, rootID) || rootID == PENDING_BYTE_PAGE(bt))
                    rootID++;
                Debug.Assert(rootID >= 3);

                // Allocate a page. The page that currently resides at pgnoRoot will be moved to the allocated page (unless the allocated page happens
                // to reside at pgnoRoot).
                Pid moveID = 0; // Move a page here to make room for the root-page
                MemPage pageMove = new MemPage();  // The page to move to.
                rc = allocateBtreePage(bt, ref pageMove, ref moveID, rootID, BTALLOC.EXACT);
                if (rc != RC.OK)
                    return rc;

                if (moveID != rootID)
                {
                    releasePage(pageMove);

                    // Move the page currently at pgnoRoot to pgnoMove.
                    rc = btreeGetPage(bt, rootID, ref root, false);
                    if (rc != RC.OK)
                        return rc;

                    // pgnoRoot is the page that will be used for the root-page of the new table (assuming an error did not occur). But we were
                    // allocated pgnoMove. If required (i.e. if it was not allocated by extending the file), the current page at position pgnoMove
                    // is already journaled.
                    PTRMAP type = 0;
                    Pid ptrPageID = 0;
                    rc = ptrmapGet(bt, rootID, ref type, ref ptrPageID);
                    if (type == PTRMAP.ROOTPAGE || type == PTRMAP.FREEPAGE)
                        rc = SysEx.CORRUPT_BKPT();
                    if (rc != RC.OK)
                    {
                        releasePage(root);
                        return rc;
                    }
                    Debug.Assert(type != PTRMAP.ROOTPAGE);
                    Debug.Assert(type != PTRMAP.FREEPAGE);
                    rc = relocatePage(bt, root, type, ptrPageID, moveID, false);
                    releasePage(root);

                    // Obtain the page at pgnoRoot
                    if (rc != RC.OK)
                        return rc;
                    rc = btreeGetPage(bt, rootID, ref root, false);
                    if (rc != RC.OK)
                        return rc;
                    rc = Pager.Write(root.DBPage);
                    if (rc != RC.OK)
                    {
                        releasePage(root);
                        return rc;
                    }
                }
                else
                    root = pageMove;

                // Update the 0pointer-map and meta-data with the new root-page number.
                ptrmapPut(bt, rootID, PTRMAP.ROOTPAGE, 0, ref rc);
                if (rc != RC.OK)
                {
                    releasePage(root);
                    return rc;
                }

                // When the new root page was allocated, page 1 was made writable in order either to increase the database filesize, or to decrement the
                // freelist count.  Hence, the sqlite3BtreeUpdateMeta() call cannot fail.
                Debug.Assert(Pager.Iswriteable(bt.Page1.DBPage));
                rc = p.UpdateMeta(META.LARGEST_ROOT_PAGE, rootID);
                if (C._NEVER(rc != RC.OK))
                {
                    releasePage(root);
                    return rc;
                }
            }
            else
            {
                rc = allocateBtreePage(bt, ref root, ref rootID, 1, BTALLOC.ANY);
                if (rc != RC.OK) return rc;
            }
#endif
            Debug.Assert(Pager.Iswriteable(root.DBPage));
            int ptfFlags; // Page-type flage for the root page of new table
            if ((createTabFlags & BTREE_INTKEY) != 0)
                ptfFlags = PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF;
            else
                ptfFlags = PTF_ZERODATA | PTF_LEAF;
            zeroPage(root, ptfFlags);
            Pager.Unref(root.DBPage);
            Debug.Assert((bt.OpenFlags & OPEN.SINGLE) == 0 || rootID == 2);
            tableID = (int)rootID;
            return RC.OK;
        }

        public RC CreateTable(ref int tableID, int flags)
        {
            Enter();
            var rc = btreeCreateTable(this, ref tableID, flags);
            Leave();
            return rc;
        }

        static RC clearDatabasePage(BtShared bt, Pid id, bool freePageFlag, ref int changes)
        {
            Debug.Assert(MutexEx.Held(bt.Mutex));
            if (id > btreePagecount(bt))
                return SysEx.CORRUPT_BKPT();

            MemPage page = new MemPage();
            var rc = getAndInitPage(bt, id, ref page);
            if (rc != RC.OK) return rc;
            for (uint i = 0U; i < page.Cells; i++)
            {
                uint cell_ = findCell(page, i);
                if (!page.Leaf)
                {
                    rc = clearDatabasePage(bt, ConvertEx.Get4(page.Data, cell_), true, ref changes);
                    if (rc != RC.OK) goto cleardatabasepage_out;
                }
                rc = clearCell(page, cell_);
                if (rc != RC.OK) goto cleardatabasepage_out;
            }
            if (!page.Leaf)
            {
                rc = clearDatabasePage(bt, ConvertEx.Get4(page.Data, 8), true, ref changes);
                if (rc != RC.OK) goto cleardatabasepage_out;
            }
            else
            {
                Debug.Assert(page.IntKey);
                changes += page.Cells;
            }
            if (freePageFlag)
                freePage(page, ref rc);
            else if ((rc = Pager.Write(page.DBPage)) == 0)
                zeroPage(page, page.Data[0] | PTF_LEAF);

        cleardatabasepage_out:
            releasePage(page);
            return rc;
        }

        public RC ClearTable(int tableID, ref int changes)
        {
            BtShared bt = Bt;
            Enter();
            Debug.Assert(InTrans == TRANS.WRITE);
            RC rc = saveAllCursors(bt, (Pid)tableID, null);
            if (rc == RC.OK)
            {
                // Invalidate all incrblob cursors open on table iTable (assuming iTable is the root of a table b-tree - if it is not, the following call is a no-op).
                invalidateIncrblobCursors(this, 0, true);
                rc = clearDatabasePage(bt, (Pid)tableID, false, ref changes);
            }
            Leave();
            return rc;
        }

        static RC btreeDropTable(Btree p, Pid tableID, ref int movedID)
        {
            BtShared bt = p.Bt;
            Debug.Assert(p.HoldsMutex());
            Debug.Assert(p.InTrans == TRANS.WRITE);

            // It is illegal to drop a table if any cursors are open on the database. This is because in auto-vacuum mode the backend may
            // need to move another root-page to fill a gap left by the deleted root page. If an open cursor was using this page a problem would occur.
            //
            // This error is caught long before control reaches this point.
            if (C._NEVER(bt.Cursor != null))
            {
                BContext.ConnectionBlocked(p.Ctx, bt.Cursor.Btree.Ctx);
                return RC.LOCKED_SHAREDCACHE;
            }

            MemPage page = null;
            RC rc = btreeGetPage(bt, (Pid)tableID, ref page, false);
            if (rc != RC.OK) return rc;
            int dummy0 = 0;
            rc = p.ClearTable((int)tableID, ref dummy0);
            if (rc != RC.OK)
            {
                releasePage(page);
                return rc;
            }

            movedID = 0;

            if (tableID > 1)
            {
#if OMIT_AUTOVACUUM
                freePage(page, ref rc);
                releasePage(page);
#else
                if (bt.AutoVacuum)
                {
                    Pid maxRootID = 0;
                    p.GetMeta(META.LARGEST_ROOT_PAGE, ref maxRootID);

                    if (tableID == maxRootID)
                    {
                        // If the table being dropped is the table with the largest root-page number in the database, put the root page on the free list. 
                        freePage(page, ref rc);
                        releasePage(page);
                        if (rc != RC.OK)
                            return rc;
                    }
                    else
                    {
                        // The table being dropped does not have the largest root-page number in the database. So move the page that does into the 
                        // gap left by the deleted root-page.
                        releasePage(page);
                        MemPage move = new MemPage();
                        rc = btreeGetPage(bt, maxRootID, ref move, false);
                        if (rc != RC.OK)
                            return rc;
                        rc = relocatePage(bt, move, PTRMAP.ROOTPAGE, 0, tableID, false);
                        releasePage(move);
                        if (rc != RC.OK)
                            return rc;
                        move = null;
                        rc = btreeGetPage(bt, maxRootID, ref move, false);
                        freePage(move, ref rc);
                        releasePage(move);
                        if (rc != RC.OK)
                            return rc;
                        movedID = (int)maxRootID;
                    }

                    // Set the new 'max-root-page' value in the database header. This is the old value less one, less one more if that happens to
                    // be a root-page number, less one again if that is the PENDING_BYTE_PAGE.
                    maxRootID--;
                    while (maxRootID == PENDING_BYTE_PAGE(bt) || PTRMAP_ISPAGE(bt, maxRootID))
                        maxRootID--;
                    Debug.Assert(maxRootID != PENDING_BYTE_PAGE(bt));

                    rc = p.UpdateMeta(META.LARGEST_ROOT_PAGE, maxRootID);
                }
                else
                {
                    freePage(page, ref rc);
                    releasePage(page);
                }
#endif
            }
            else
            {
                // If sqlite3BtreeDropTable was called on page 1. This really never should happen except in a corrupt database. 
                zeroPage(page, PTF_INTKEY | PTF_LEAF);
                releasePage(page);
            }
            return rc;
        }

        public RC DropTable(int tableID, ref int movedID)
        {
            Enter();
            RC rc = btreeDropTable(this, (Pid)tableID, ref movedID);
            Leave();
            return rc;
        }

        #endregion

        #region Meta / Count

        public void GetMeta(META id, ref uint meta)
        {
            BtShared bt = Bt;
            Enter();
            Debug.Assert(InTrans > TRANS.NONE);
            Debug.Assert(querySharedCacheTableLock(this, MASTER_ROOT, LOCK.READ) == RC.OK);
            Debug.Assert(bt.Page1 != null);
            Debug.Assert(id >= 0 && (int)id <= 15);
            meta = ConvertEx.Get4(bt.Page1.Data, 36 + (int)id * 4);
            // If auto-vacuum is disabled in this build and this is an auto-vacuum database, mark the database as read-only.
#if OMIT_AUTOVACUUM
            if (idx == META.LARGEST_ROOT_PAGE && meta > 0)
                bt.BtsFlags |= BTS.READ_ONLY;
#endif
            Leave();
        }

        public RC UpdateMeta(META id, uint meta)
        {
            BtShared bt = Bt;
            Debug.Assert((int)id >= 1 && (int)id <= 15);
            Enter();
            Debug.Assert(InTrans == TRANS.WRITE);
            Debug.Assert(bt.Page1 != null);
            byte[] p1 = bt.Page1.Data;
            var rc = Pager.Write(bt.Page1.DBPage);
            if (rc == RC.OK)
            {
                ConvertEx.Put4(p1, 36 + (int)id * 4, meta);
#if !OMIT_AUTOVACUUM
                if (id == META.INCR_VACUUM)
                {
                    Debug.Assert(bt.AutoVacuum || meta == 0);
                    Debug.Assert(meta == 0 || meta == 1);
                    bt.IncrVacuum = (meta != 0);
                }
#endif
            }
            Leave();
            return rc;
        }

#if !OMIT_BTREECOUNT
        public static RC Count(BtCursor cur, ref long entrysRef)
        {
            if (cur.RootID == 0)
            {
                entrysRef = 0;
                return RC.OK;
            }
            var rc = moveToRoot(cur);

            // Unless an error occurs, the following loop runs one iteration for each page in the B-Tree structure (not including overflow pages).
            long entrys = 0; // Value to return in pnEntry
            while (rc == RC.OK)
            {
                // If this is a leaf page or the tree is not an int-key tree, then this page contains countable entries. Increment the entry counter accordingly.
                MemPage page = cur.Pages[cur.ID]; // Current page of the b-tree
                if (page.Leaf || !page.IntKey)
                    entrys += page.Cells;

                // pPage is a leaf node. This loop navigates the cursor so that it points to the first interior cell that it points to the parent of
                // the next page in the tree that has not yet been visited. The pCur->aiIdx[pCur->iPage] value is set to the index of the parent cell
                // of the page, or to the number of cells in the page if the next page to visit is the right-child of its parent.
                //
                // If all pages in the tree have been visited, return SQLITE_OK to the caller.
                if (page.Leaf)
                {
                    do
                    {
                        if (cur.ID == 0)
                        {
                            // All pages of the b-tree have been visited. Return successfully.
                            entrysRef = entrys;
                            return RC.OK;
                        }
                        moveToParent(cur);
                    } while (cur.Idxs[cur.ID] >= cur.Pages[cur.ID].Cells);

                    cur.Idxs[cur.ID]++;
                    page = cur.Pages[cur.ID];
                }

                // Descend to the child node of the cell that the cursor currently points at. This is the right-child if (iIdx==pPage->nCell).
                uint idx = cur.Idxs[cur.ID]; // Index of child node in parent
                if (idx == page.Cells)
                    rc = moveToChild(cur, ConvertEx.Get4(page.Data, page.HdrOffset + 8));
                else
                    rc = moveToChild(cur, ConvertEx.Get4(page.Data, findCell(page, idx)));
            }

            // An error has occurred. Return an error code.
            return rc;
        }
#endif

        public Pager get_Pager()
        {
            return Bt.Pager;
        }

        #endregion

        #region Integrity Check
#if !OMIT_INTEGRITY_CHECK

        static void checkAppendMsg(IntegrityCk check, string msg1, string format, params object[] args)
        {
            if (check.MaxErrors == 0) return;
            //va_list ap;
            //lock (_va_listLock)
            {
                check.MaxErrors--;
                check.Errors++;
                // implement
                //va_start(args, format);
                //if (check.errMsg.zText.Length != 0)
                //    sqlite3StrAccumAppend(check.errMsg, "\n", 1);
                //if (msg1.Length > 0)
                //    sqlite3StrAccumAppend(check.errMsg, msg1.ToString(), -1);
                //sqlite3VXPrintf(check.errMsg, 1, format, args);
                //va_end(ref args);
            }
        }
        static void checkAppendMsg(IntegrityCk check, StringBuilder msg1, string format, params object[] args)
        {
            if (check.MaxErrors == 0) return;
            //va_list ap;
            //lock (_va_listLock)
            {
                check.MaxErrors--;
                check.Errors++;
                //va_start(args, format);
                //if (check.errMsg.zText.Length != 0)
                //    sqlite3StrAccumAppend(check.errMsg, "\n", 1);
                //if (msg1.Length > 0)
                //    sqlite3StrAccumAppend(check.errMsg, msg1.ToString(), -1);
                //sqlite3VXPrintf(check.errMsg, 1, format, args);
                //va_end(ref args);
            }
        }

        static bool getPageReferenced(IntegrityCk check, Pid pageID)
        {
            Debug.Assert(pageID <= check.Pages && sizeof(byte) == 1);
            return ((check.PgRefs[pageID / 8] & (1 << (int)(pageID & 0x07))) != 0);
        }

        static void setPageReferenced(IntegrityCk check, Pid pageID)
        {
            Debug.Assert(pageID <= check.Pages && sizeof(byte) == 1);
            check.PgRefs[pageID / 8] |= (byte)(1 << (int)(pageID & 0x07));
        }

        static bool checkRef(IntegrityCk check, Pid pageID, string context)
        {
            if (pageID == 0) return true;
            if (pageID > check.Pages)
            {
                checkAppendMsg(check, context, "invalid page number %d", pageID);
                return true;
            }
            if (getPageReferenced(check, pageID))
            {
                checkAppendMsg(check, context, "2nd reference to page %d", pageID);
                return true;
            }
            setPageReferenced(check, pageID);
            return true;
        }

#if !OMIT_AUTOVACUUM
        static void checkPtrmap(IntegrityCk check, Pid childID, PTRMAP type, Pid parentID, string context)
        {
            PTRMAP ptrmapType = 0;
            Pid ptrmapParentID = 0;
            RC rc = ptrmapGet(check.Bt, childID, ref ptrmapType, ref ptrmapParentID);
            if (rc != RC.OK)
            {
                if (rc == RC.NOMEM || rc == RC.IOERR_NOMEM) check.MallocFailed = true;
                checkAppendMsg(check, context, "Failed to read ptrmap key=%d", childID);
                return;
            }

            if (ptrmapType != type || ptrmapParentID != parentID)
                checkAppendMsg(check, context, "Bad ptr map entry key=%d expected=(%d,%d) got=(%d,%d)", childID, type, parentID, ptrmapType, ptrmapParentID);
        }
#endif

        static void checkList(IntegrityCk check, bool isFreeList, Pid pageID, int length, string context)
        {
            int expected = length;
            Pid firstID = pageID;
            while (length-- > 0 && check.MaxErrors != 0)
            {
                if (pageID < 1)
                {
                    checkAppendMsg(check, context, "%d of %d pages missing from overflow list starting at %d", length + 1, expected, firstID);
                    break;
                }
                if (checkRef(check, pageID, context)) break;
                PgHdr ovflPage = new PgHdr();
                if (check.Pager.Acquire((Pid)pageID, ref ovflPage, false) != RC.OK)
                {
                    checkAppendMsg(check, context, "failed to get page %d", pageID);
                    break;
                }
                byte[] ovflData = Pager.GetData(ovflPage);
                if (isFreeList)
                {
                    int n = (int)ConvertEx.Get4(ovflData, 4);
#if !OMIT_AUTOVACUUM
                    if (check.Bt.AutoVacuum)
                        checkPtrmap(check, (uint)pageID, PTRMAP.FREEPAGE, 0, context);
#endif
                    if (n > (int)check.Bt.UsableSize / 4 - 2)
                    {
                        checkAppendMsg(check, context, "freelist leaf count too big on page %d", pageID);
                        length--;
                    }
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            Pid freePageID = ConvertEx.Get4(ovflData, 8 + i * 4);
#if !OMIT_AUTOVACUUM
                            if (check.Bt.AutoVacuum)
                                checkPtrmap(check, freePageID, PTRMAP.FREEPAGE, 0, context);
#endif
                            checkRef(check, freePageID, context);
                        }
                        length -= n;
                    }
                }
#if !OMIT_AUTOVACUUM
                else
                {
                    // If this database supports auto-vacuum and iPage is not the last page in this overflow list, check that the pointer-map entry for
                    // the following page matches iPage.
                    if (check.Bt.AutoVacuum && length > 0)
                    {
                        int i = (int)ConvertEx.Get4(ovflData);
                        checkPtrmap(check, (uint)i, PTRMAP.OVERFLOW2, (uint)pageID, context);
                    }
                }
#endif
                pageID = (Pid)ConvertEx.Get4(ovflData);
                Pager.Unref(ovflPage);
            }
        }

        static long _nullRef_;
        static int checkTreePage(IntegrityCk check, Pid pageID, string parentContext, ref long parentMinKey, bool hasParentMinKey, ref long parentMaxKey, bool hasParentMaxKey)
        {
            var msg = new StringBuilder(100);
            msg.AppendFormat("Page {0}: ", pageID);

            // Check that the page exists
            var bt = check.Bt;
            var usableSize = (int)bt.UsableSize;
            if (pageID == 0) return 0;
            if (checkRef(check, pageID, parentContext)) return 0;
            RC rc;
            MemPage page = new MemPage();
            if ((rc = btreeGetPage(bt, pageID, ref page, false)) != RC.OK)
            {
                checkAppendMsg(check, msg.ToString(), "unable to get the page. error code=%d", rc);
                return 0;
            }

            // Clear MemPage.isInit to make sure the corruption detection code in btreeInitPage() is executed.
            page.IsInit = false;
            if ((rc = btreeInitPage(page)) != RC.OK)
            {
                Debug.Assert(rc == RC.CORRUPT); // The only possible error from InitPage
                checkAppendMsg(check, msg.ToString(), "btreeInitPage() returns error code %d", rc);
                releasePage(page);
                return 0;
            }

            // Check out all the cells.
            Pid id;
            uint i;
            int depth = 0;
            long minKey = 0;
            long maxKey = 0;
            for (i = 0U; i < page.Cells && check.MaxErrors != 0; i++)
            {
                // Check payload overflow pages
                msg.AppendFormat("On tree page {0} cell {1}: ", pageID, i);
                uint cell_ = findCell(page, i);
                var info = new CellInfo();
                btreeParseCellPtr(page, cell_, ref info);
                uint sizeCell = info.Data;
                if (!page.IntKey) sizeCell += (uint)info.Key;
                // For intKey pages, check that the keys are in order.
                else if (i == 0) minKey = maxKey = info.Key;
                else
                {
                    if (info.Key <= maxKey)
                        checkAppendMsg(check, msg.ToString(), "Rowid %lld out of order (previous was %lld)", info.Key, maxKey);
                    maxKey = info.Key;
                }
                Debug.Assert(sizeCell == info.Payload);
                if (sizeCell > info.Local) //&& pCell[info.iOverflow]<=&pPage.aData[pBt.usableSize]
                {
                    int pages = (int)(sizeCell - info.Local + usableSize - 5) / (usableSize - 4);
                    Pid ovflID = ConvertEx.Get4(page.Data, cell_ + info.Overflow);
#if !OMIT_AUTOVACUUM
                    if (bt.AutoVacuum)
                        checkPtrmap(check, ovflID, PTRMAP.OVERFLOW1, pageID, msg.ToString());
#endif
                    checkList(check, false, ovflID, pages, msg.ToString());
                }

                // Check sanity of left child page.
                if (!page.Leaf)
                {
                    id = (Pid)ConvertEx.Get4(page.Data, cell_);
#if !OMIT_AUTOVACUUM
                    if (bt.AutoVacuum)
                        checkPtrmap(check, id, PTRMAP.BTREE, pageID, msg.ToString());
#endif
                    int depth2;
                    if (i == 0)
                        depth2 = checkTreePage(check, id, msg.ToString(), ref minKey, true, ref _nullRef_, false);
                    else
                        depth2 = checkTreePage(check, id, msg.ToString(), ref minKey, true, ref maxKey, true);

                    if (i > 0 && depth2 != depth)
                        checkAppendMsg(check, msg, "Child page depth differs");
                    depth = depth2;
                }
            }
            if (!page.Leaf)
            {
                id = (Pid)ConvertEx.Get4(page.Data, page.HdrOffset + 8);
                msg.AppendFormat("On page {0} at right child: ", pageID);
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum)
                    checkPtrmap(check, id, PTRMAP.BTREE, pageID, msg.ToString());
#endif
                if (page.Cells == 0)
                    checkTreePage(check, id, msg.ToString(), ref _nullRef_, false, ref _nullRef_, false);
                else
                    checkTreePage(check, id, msg.ToString(), ref _nullRef_, false, ref maxKey, true);
            }

            // For intKey leaf pages, check that the min/max keys are in order with any left/parent/right pages.
            if (page.Leaf && page.IntKey)
            {
                // if we are a left child page
                if (hasParentMinKey)
                {
                    // if we are the left most child page
                    if (!hasParentMaxKey)
                    {
                        if (maxKey > parentMinKey)
                            checkAppendMsg(check, msg, "Rowid %lld out of order (max larger than parent min of %lld)", maxKey, parentMinKey);
                    }
                    else
                    {
                        if (minKey <= parentMinKey)
                            checkAppendMsg(check, msg, "Rowid %lld out of order (min less than parent min of %lld)", minKey, parentMinKey);
                        if (maxKey > parentMaxKey)
                            checkAppendMsg(check, msg, "Rowid %lld out of order (max larger than parent max of %lld)", maxKey, parentMaxKey);
                        parentMinKey = maxKey;
                    }
                }
                // else if we're a right child page
                else if (hasParentMaxKey)
                {
                    if (minKey <= parentMaxKey)
                        checkAppendMsg(check, msg, "Rowid %lld out of order (min less than parent max of %lld)", minKey, parentMaxKey);
                }
            }

            // Check for complete coverage of the page
            byte[] data = page.Data;
            uint hdr = page.HdrOffset;
            byte[] hit = PCache.PageAlloc2((int)bt.PageSize);
            if (hit == null)
                check.MallocFailed = true;
            else
            {
                uint contentOffset = ConvertEx.Get2nz(data, hdr + 5);
                Debug.Assert(contentOffset <= usableSize); // Enforced by btreeInitPage()
                Array.Clear(hit, (int)contentOffset, (int)(usableSize - contentOffset));
                { for (uint z = contentOffset - 1; z >= 0; z--) hit[z] = 1; }// memset(hit, 1, contentOffset);
                uint cells = ConvertEx.Get2(data, hdr + 3);
                uint cellStart = hdr + 12 - 4 * (page.Leaf ? 1U : 0U);
                for (i = 0; i < cells; i++)
                {
                    var sizeCell = 65536U;
                    uint pc = ConvertEx.Get2(data, cellStart + i * 2);
                    if (pc <= usableSize - 4)
                        sizeCell = cellSizePtr(page, data, pc);
                    if ((int)(pc + sizeCell - 1) >= usableSize)
                        checkAppendMsg(check, (string)null, "Corruption detected in cell %d on page %d", i, pageID);
                    else
                        for (var j = (int)(pc + sizeCell - 1); j >= pc; j--) hit[j]++;
                }
                i = ConvertEx.Get2(data, hdr + 1);
                while (i > 0)
                {
                    Debug.Assert(i <= usableSize - 4); // Enforced by btreeInitPage()
                    uint size = ConvertEx.Get2(data, i + 2);
                    Debug.Assert(i + size <= usableSize); // Enforced by btreeInitPage()
                    uint j;
                    for (j = i + size - 1; j >= i; j--) hit[j]++;
                    j = ConvertEx.Get2(data, i);
                    Debug.Assert(j == 0 || j > i + size); // Enforced by btreeInitPage()
                    Debug.Assert(j <= usableSize - 4); // Enforced by btreeInitPage()
                    i = j;
                }
                uint cnt;
                for (i = cnt = 0; i < usableSize; i++)
                {
                    if (hit[i] == 0)
                        cnt++;
                    else if (hit[i] > 1)
                    {
                        checkAppendMsg(check, (string)null, "Multiple uses for byte %d of page %d", i, pageID);
                        break;
                    }
                }
                if (cnt != data[hdr + 7])
                    checkAppendMsg(check, (string)null, "Fragmentation of %d bytes reported as %d on page %d", cnt, data[hdr + 7], pageID);
            }
            PCache.PageFree2(ref hit);
            releasePage(page);
            return depth + 1;
        }

        public string IntegrityCheck(Pid[] roots, int rootsLength, int maxErrors, out int errors)
        {
            BtShared bt = Bt;
            Enter();
            Debug.Assert(InTrans > TRANS.NONE && bt.InTransaction > TRANS.NONE);
            int refs = bt.Pager.get_Refs();
            IntegrityCk check = new IntegrityCk();
            check.Bt = bt;
            check.Pager = bt.Pager;
            check.Pages = btreePagecount(check.Bt);
            check.MaxErrors = maxErrors;
            check.Errors = 0;
            check.MallocFailed = false;
            errors = 0;
            if (check.Pages == 0)
            {
                Leave();
                return null;
            }
            check.PgRefs = C._alloc((int)(check.Pages / 8) + 1);
            if (check.PgRefs == null)
            {
                errors = 1;
                Leave();
                return null;
            }
            Pid i = PENDING_BYTE_PAGE(bt);
            if (i <= check.Pages) setPageReferenced(check, i);
            TextBuilder.Init(check.ErrMsg, 1000, 20000);

            // Check the integrity of the freelist
            checkList(check, true, (Pid)ConvertEx.Get4(bt.Page1.Data, 32), (int)ConvertEx.Get4(bt.Page1.Data, 36), "Main freelist: ");

            // Check all the tables.
            for (i = 0; (int)i < rootsLength && check.MaxErrors != 0; i++)
            {
                if (roots[i] == 0) continue;
#if !OMIT_AUTOVACUUM
                if (bt.AutoVacuum && roots[i] > 1)
                    checkPtrmap(check, roots[i], PTRMAP.ROOTPAGE, 0, null);
#endif
                checkTreePage(check, roots[i], "List of tree roots: ", ref _nullRef_, false, ref _nullRef_, false);
            }

            // Make sure every page in the file is referenced
            for (i = 1; i <= check.Pages && check.MaxErrors != 0; i++)
            {
#if OMIT_AUTOVACUUM
                if (check.PgRefs[i] == null)
                    checkAppendMsg(check, null, "Page %d is never used", i);
#else
                // If the database supports auto-vacuum, make sure no tables contain references to pointer-map pages.
                if (!getPageReferenced(check, i) && (PTRMAP_PAGENO(bt, i) != i || !bt.AutoVacuum))
                    checkAppendMsg(check, (string)null, "Page %d is never used", i);
                if (getPageReferenced(check, i) && (PTRMAP_PAGENO(bt, i) == i && bt.AutoVacuum))
                    checkAppendMsg(check, (string)null, "Pointer map page %d is referenced", i);
#endif
            }

            // Make sure this analysis did not leave any unref() pages. This is an internal consistency check; an integrity check
            // of the integrity check.
            if (C._NEVER(refs != bt.Pager.get_Refs()))
                checkAppendMsg(check, (string)null, "Outstanding page count goes from %d to %d during this analysis", refs, bt.Pager.get_Refs());

            // Clean  up and report errors.
            Leave();
            check.PgRefs = null;
            if (check.MallocFailed)
            {
                check.ErrMsg.Reset();
                errors = check.Errors + 1;
                return null;
            }
            errors = check.Errors;
            if (check.Errors == 0)
                check.ErrMsg.Reset();
            return check.ErrMsg.ToString();
        }

#endif
        #endregion

        #region Settings2

        public string get_Filename()
        {
            Debug.Assert(Bt.Pager != null);
            return Bt.Pager.get_Filename(true);
        }

        public string get_Journalname()
        {
            Debug.Assert(Bt.Pager != null);
            return Bt.Pager.get_Journalname();
        }

        public bool IsInTrans()
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            return (InTrans == TRANS.WRITE);
        }

#if !OMIT_WAL
        public RC Checkpoint(int mode, int* logs, int* checkpoints)
        {
            BtShared bt = Bt;
            Enter();
            RC rc;
            if (bt.InTransaction != TRANS.NONE)
                rc = RC.LOCKED;
            else
                rc = bt.Pager.Checkpoint(mode, logs, checkpoints);
            Leave();
            return rc;
        }
#endif

        public bool IsInReadTrans()
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            return (InTrans != TRANS.NONE);
        }

        public bool IsInBackup()
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            return (Backups != 0);
        }

        public Schema Schema(int bytes, Action<Schema> free)
        {
            BtShared bt = Bt;
            Enter();
            if (bt.Schema == null && bytes != 0)
            {
                bt.Schema = new Schema();
                bt.FreeSchema = free;
            }
            Leave();
            return bt.Schema;
        }

        public RC SchemaLocked()
        {
            Debug.Assert(MutexEx.Held(Ctx.Mutex));
            Enter();
            var rc = querySharedCacheTableLock(this, MASTER_ROOT, LOCK.READ);
            Debug.Assert(rc == RC.OK || rc == RC.LOCKED_SHAREDCACHE);
            Leave();
            return rc;
        }


#if !OMIT_SHARED_CACHE
        public RC LockTable(Pid tableID, bool isWriteLock)
        {
            Debug.Assert(InTrans != TRANS.NONE);
            RC rc = RC.OK;
            if (Sharable_)
            {
                LOCK lockType = (isWriteLock ? LOCK.READ : LOCK.WRITE);
                Enter();
                rc = querySharedCacheTableLock(this, tableID, lockType);
                if (rc == RC.OK)
                    rc = setSharedCacheTableLock(this, tableID, lockType);
                Leave();
            }
            return rc;
        }
#endif

#if !OMIT_INCRBLOB
        public static RC PutData(BtCursor cur, uint offset, uint amount, byte[] z)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            Debug.Assert(cur.IsIncrblobHandle);

            RC rc = restoreCursorPosition(cur);
            if (rc != RC.OK)
                return rc;
            Debug.Assert(cur.State != CURSOR.REQUIRESEEK);
            if (cur.State != CURSOR.VALID)
                return RC.ABORT;

            // Check some assumptions: 
            //   (a) the cursor is open for writing,
            //   (b) there is a read/write transaction open,
            //   (c) the connection holds a write-lock on the table (if required),
            //   (d) there are no conflicting read-locks, and
            //   (e) the cursor points at a valid row of an intKey table.
            if (!cur.WrFlag)
                return RC.READONLY;
            Debug.Assert((cur.Bt.BtsFlags & BTS.READ_ONLY) == 0 && cur.Bt.InTransaction == TRANS.WRITE);
            Debug.Assert(hasSharedCacheTableLock(cur.Btree, cur.RootID, false, LOCK.WRITE));
            Debug.Assert(!hasReadConflicts(cur.Btree, cur.RootID));
            Debug.Assert(cur.Pages[cur.ID].IntKey);

            return accessPayload(cur, offset, amount, z, 1);
        }

        public static void CacheOverflow(BtCursor cur)
        {
            Debug.Assert(cursorHoldsMutex(cur));
            Debug.Assert(MutexEx.Held(cur.Btree.Ctx.Mutex));
            invalidateOverflowCache(cur);
            cur.IsIncrblobHandle = true;
        }
#endif

        public RC SetVersion(int version)
        {
            Debug.Assert(version == 1 || version == 2);

            // If setting the version fields to 1, do not automatically open the WAL connection, even if the version fields are currently set to 2.
            BtShared bt = Bt;
            bt.BtsFlags &= ~BTS.NO_WAL;
            if (version == 1) bt.BtsFlags |= BTS.NO_WAL;

            var rc = BeginTrans(0);
            if (rc == RC.OK)
            {
                var data = bt.Page1.Data;
                if (data[18] != (byte)version || data[19] != (byte)version)
                {
                    rc = BeginTrans(2);
                    if (rc == RC.OK)
                    {
                        rc = Pager.Write(bt.Page1.DBPage);
                        if (rc == RC.OK)
                        {
                            data[18] = (byte)version;
                            data[19] = (byte)version;
                        }
                    }
                }
            }

            bt.BtsFlags &= ~BTS.NO_WAL;
            return rc;
        }

        public static void CursorHints(BtCursor cur, uint mask)
        {
            Debug.Assert(mask == BTREE_BULKLOAD || mask == 0);
            cur.Hints = (byte)mask;
        }

        #endregion
    }
}

