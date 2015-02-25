using Pid = System.UInt32;
using System;
using System.Diagnostics;

namespace Core
{
    partial class PCache
    {
        #region Linked List

#if EXPENSIVE_ASSERT
        private static bool CheckSynced(PCache cache)
        {
            PgHdr p;
            for (p = cache.DirtyTail; p != cache.Synced; p = p.DirtyPrev)
                Debug.Assert(p.Refs != 0 || (p.Flags & PgHdr.PGHDR.NEED_SYNC) != 0);
            return (p == null || p.Refs != 0 || (p.Flags & PgHdr.PGHDR.NEED_SYNC) == 0);
        }
#endif

        private static void RemoveFromDirtyList(PgHdr page)
        {
            var p = page.Cache;
            Debug.Assert(page.DirtyNext != null || page == p.DirtyTail);
            Debug.Assert(page.DirtyPrev != null || page == p.Dirty);
            // Update the PCache1.pSynced variable if necessary.
            if (p.Synced == page)
            {
                var synced = page.DirtyPrev;
                while (synced != null && (synced.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                    synced = synced.DirtyPrev;
                p.Synced = synced;
            }
            if (page.DirtyNext != null)
                page.DirtyNext.DirtyPrev = page.DirtyPrev;
            else
            {
                Debug.Assert(page == p.DirtyTail);
                p.DirtyTail = page.DirtyPrev;
            }
            if (page.DirtyPrev != null)
                page.DirtyPrev.DirtyNext = page.DirtyNext;
            else
            {
                Debug.Assert(page == p.Dirty);
                p.Dirty = page.DirtyNext;
            }
            page.DirtyNext = null;
            page.DirtyPrev = null;
#if EXPENSIVE_ASSERT
            Debug.Assert(CheckSynced(p));
#endif
        }

        private static void AddToDirtyList(PgHdr page)
        {
            var p = page.Cache;
            Debug.Assert(page.DirtyNext == null && page.DirtyPrev == null && p.Dirty != page);
            page.DirtyNext = p.Dirty;
            if (page.DirtyNext != null)
            {
                Debug.Assert(page.DirtyNext.DirtyPrev == null);
                page.DirtyNext.DirtyPrev = page;
            }
            p.Dirty = page;
            if (p.DirtyTail == null)
                p.DirtyTail = page;
            if (p.Synced == null && (page.Flags & PgHdr.PGHDR.NEED_SYNC) == 0)
                p.Synced = page;
#if EXPENSIVE_ASSERT
            Debug.Assert(CheckSynced(p));
#endif
        }

        private static void Unpin(PgHdr p)
        {
            var cache = p.Cache;
            if (cache.Purgeable)
            {
                if (p.ID == 1)
                    cache.Page1 = null;
                cache.Cache.Unpin(p.Page, false);
            }
        }

        #endregion

        #region Interface

        static IPCache _pcache;

        public static RC Initialize()
        {
            if (_pcache == null)
                _pcache = new PCache1();
            return _pcache.Init();
        }
        public static void Shutdown()
        {
            _pcache.Shutdown();
        }
        //public static int SizeOf()
        //{
        //    return 4;
        //}

        public static void Open(int sizePage, int sizeExtra, bool purgeable, Func<object, PgHdr, RC> stress, object stressArg, PCache p)
        {
            p.memset();
            p.SizePage = sizePage;
            p.SizeExtra = sizeExtra;
            p.Purgeable = purgeable;
            p.Stress = stress;
            p.StressArg = stressArg;
            p.SizeCache = 100;
        }

        public void SetPageSize(int sizePage)
        {
            Debug.Assert(Refs == 0 && Dirty == null);
            if (Cache != null)
            {
                _pcache.Destroy(ref Cache);
                Cache = null;
                Page1 = null;
            }
            SizePage = sizePage;
        }

        public uint get_CacheSize() // NumberOfCachePages
        {
            if (SizeCache >= 0)
                return (uint)SizeCache;
            return (uint)((-1024 * (long)SizeCache) / (SizePage + SizeExtra));
        }

        public RC Fetch(Pid id, bool createFlag, out PgHdr pageOut)
        {
            Debug.Assert(id > 0);
            // If the pluggable cache (sqlite3_pcache*) has not been allocated, allocate it now.
            if (Cache == null && createFlag)
            {
                var p = _pcache.Create(SizePage, SizeExtra + 0, Purgeable);
                p.Cachesize(get_CacheSize());
                Cache = p;
            }
            ICachePage page = null;
            var create = (createFlag ? 1 : 0) * (1 + ((!Purgeable || Dirty == null) ? 1 : 0));
            if (Cache != null)
                page = Cache.Fetch(id, create > 0);
            if (page == null && create == 1)
            {
                // Find a dirty page to write-out and recycle. First try to find a page that does not require a journal-sync (one with PGHDR_NEED_SYNC
                // cleared), but if that is not possible settle for any other unreferenced dirty page.
#if EXPENSIVE_ASSERT
                CheckSynced(this);
#endif
                PgHdr pg;
                for (pg = Synced; pg != null && (pg.Refs != 0 || (pg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0); pg = pg.DirtyPrev) ;
                Synced = pg;
                if (pg == null)
                    for (pg = DirtyTail; pg != null && pg.Refs != 0; pg = pg.DirtyPrev) ;
                if (pg != null)
                {
#if LOG_CACHE_SPILL
                    SysEx.Log(RC.FULL, "spill page %d making room for %d - cache used: %d/%d", pg.ID, id, _pcache->Pagecount(Cache), NumberOfCachePages(this));
#endif
                    var rc = Stress(StressArg, pg);
                    if (rc != RC.OK && rc != RC.BUSY)
                    {
                        pageOut = null;
                        return rc;
                    }
                }
                page = Cache.Fetch(id, true);
            }
            PgHdr pgHdr = null;
            if (page != null)
            {
                //pgHdr = page.Extra;
                if (page.Data == null)
                {
                    //page.Page = page;
                    page.Data = C._alloc(SizePage);
                    //page.Extra = this;
                    page.Cache = this;
                    page.ID = id;
                }
                Debug.Assert(page.Cache == Cache);
                Debug.Assert(page.ID == id);
                //Debug.Assert(page.Data == page.Buffer);
                //Debug.Assert(page.Extra == this);
                if (page.Refs == 0)
                    Refs++;
                page.Refs++;
                if (id == 1)
                    Page1 = pgHdr;
            }
            pageOut = pgHdr;
            return (pgHdr == null && create != 0 ? RC.NOMEM : RC.OK);
        }

        public static void Release(PgHdr p)
        {
            Debug.Assert(p.Refs > 0);
            p.Refs--;
            if (p.Refs == 0)
            {
                var cache = p.Cache;
                cache.Refs--;
                if ((p.Flags & PgHdr.PGHDR.DIRTY) == 0)
                    Unpin(p);
                else
                {
                    // Move the page to the head of the dirty list.
                    RemoveFromDirtyList(p);
                    AddToDirtyList(p);
                }
            }
        }

        public static void Ref(PgHdr p)
        {
            Debug.Assert(p.Refs > 0);
            p.Refs++;
        }

        public static void Drop(PgHdr p)
        {
            Debug.Assert(p.Refs == 1);
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0)
                RemoveFromDirtyList(p);
            var cache = p.Cache;
            cache.Refs--;
            if (p.ID == 1)
                cache.Page1 = null;
            cache.Cache.Unpin(p.Page, true);
        }

        public static void MakeDirty(PgHdr p)
        {
            p.Flags &= ~PgHdr.PGHDR.DONT_WRITE;
            Debug.Assert(p.Refs > 0);
            if ((p.Flags & PgHdr.PGHDR.DIRTY) == 0)
            {
                p.Flags |= PgHdr.PGHDR.DIRTY;
                AddToDirtyList(p);
            }
        }

        public static void MakeClean(PgHdr p)
        {
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0)
            {
                RemoveFromDirtyList(p);
                p.Flags &= ~(PgHdr.PGHDR.DIRTY | PgHdr.PGHDR.NEED_SYNC);
                if (p.Refs == 0)
                    Unpin(p);
            }
        }

        public void CleanAll()
        {
            PgHdr p;
            while ((p = Dirty) != null)
                MakeClean(p);
        }

        public void ClearSyncFlags()
        {
            for (var p = Dirty; p != null; p = p.DirtyNext)
                p.Flags &= ~PgHdr.PGHDR.NEED_SYNC;
            Synced = DirtyTail;
        }

        public static void Move(PgHdr p, Pid newID)
        {
            PCache cache = p.Cache;
            Debug.Assert(p.Refs > 0);
            Debug.Assert(newID > 0);
            cache.Cache.Rekey(p.Page, p.ID, newID);
            p.ID = newID;
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0 && (p.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
            {
                RemoveFromDirtyList(p);
                AddToDirtyList(p);
            }
        }

        public void Truncate(Pid id)
        {
            if (Cache != null)
            {
                PgHdr p;
                PgHdr next;
                for (p = Dirty; p != null; p = next)
                {
                    next = p.DirtyNext;
                    // This routine never gets call with a positive pgno except right after sqlite3PcacheCleanAll().  So if there are dirty pages, it must be that pgno==0.
                    Debug.Assert(p.ID > 0);
                    if (C._ALWAYS(p.ID > id))
                    {
                        Debug.Assert((p.Flags & PgHdr.PGHDR.DIRTY) != 0);
                        MakeClean(p);
                    }
                }
                if (id == 0 && Page1 != null)
                {
                    Page1.memsetData(SizePage);
                    id = 1;
                }
                Cache.Truncate(id + 1);
            }
        }

        public void Close()
        {
            if (Cache != null)
                _pcache.Destroy(ref Cache);
        }

        public void Clear()
        {
            Truncate(0);
        }

        private static PgHdr MergeDirtyList(PgHdr a, PgHdr b)
        {
            var result = new PgHdr();
            var tail = result;
            while (a != null && b != null)
            {
                if (a.ID < b.ID)
                {
                    tail.Dirty = a;
                    tail = a;
                    a = a.Dirty;
                }
                else
                {
                    tail.Dirty = b;
                    tail = b;
                    b = b.Dirty;
                }
            }
            if (a != null)
                tail.Dirty = a;
            else if (b != null)
                tail.Dirty = b;
            else
                tail.Dirty = null;
            return result.Dirty;
        }

        private const int N_SORT_BUCKET = 32;

        private static PgHdr SortDirtyList(PgHdr @in)
        {
            var a = new PgHdr[N_SORT_BUCKET];
            PgHdr p;
            int i;
            while (@in != null)
            {
                p = @in;
                @in = p.Dirty;
                p.Dirty = null;
                for (i = 0; C._ALWAYS(i < N_SORT_BUCKET - 1); i++)
                {
                    if (a[i] == null)
                    {
                        a[i] = p;
                        break;
                    }
                    else
                    {
                        p = MergeDirtyList(a[i], p);
                        a[i] = null;
                    }
                }
                if (C._NEVER(i == N_SORT_BUCKET - 1))
                    // To get here, there need to be 2^(N_SORT_BUCKET) elements in the input list.  But that is impossible.
                    a[i] = MergeDirtyList(a[i], p);
            }
            p = a[0];
            for (i = 1; i < N_SORT_BUCKET; i++)
                p = MergeDirtyList(p, a[i]);
            return p;
        }

        public PgHdr DirtyList()
        {
            for (var p = Dirty; p != null; p = p.DirtyNext)
                p.Dirty = p.DirtyNext;
            return SortDirtyList(Dirty);
        }

        public int get_Refs()
        {
            return Refs;
        }

        public static int get_PageRefs(PgHdr p)
        {
            return p.Refs;
        }

        public int get_Pages()
        {
            return (Cache != null ? Cache.get_Pages() : 0);
        }

        public void set_CacheSize(int maxPage)
        {
            SizeCache = maxPage;
            if (Cache != null)
                Cache.Cachesize(get_CacheSize());
        }

        public void Shrink()
        {
            if (Cache != null)
                Cache.Shrink();
        }

#if CHECK_PAGES || DEBUG
        // For all dirty pages currently in the cache, invoke the specified callback. This is only used if the SQLITE_CHECK_PAGES macro is defined.
        public void IterateDirty(Action<PgHdr> iter)
        {
            for (var dirty = Dirty; dirty != null; dirty = dirty.DirtyNext)
                iter(dirty);
        }
#endif

        #endregion

        #region Test
#if TEST
#endif
        #endregion

        #region FromPCache1

        public static void PageBufferSetup(object buffer, int size, int n)
        {
            PCache1.BufferSetup(buffer, size, n);
        }

        public static PgHdr PageAlloc(int size)
        {
            return PCache1.Alloc(size);
        }

        public static byte[] PageAlloc2(int size)
        {
            return C._alloc(size);
        }

        public static void PageFree(ref PgHdr p)
        {
            PCache1.Free(ref p);
        }

        public static void PageFree2(ref byte[] p)
        {
            C._free(ref p);
        }

        #endregion
    }
}
