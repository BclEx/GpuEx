using Pid = System.UInt32;
using IPage = Core.PgHdr;
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    #region Struct

    public class PGroup
    {
        public MutexEx Mutex;           // MUTEX_STATIC_LRU or NULL
        public uint MaxPages;           // Sum of nMax for purgeable caches
        public uint MinPages;           // Sum of nMin for purgeable caches
        public uint MaxPinned;          // nMaxpage + 10 - nMinPage
        public uint CurrentPages;       // Number of purgeable pages allocated
        public PgHdr1 LruHead, LruTail; // LRU list of unpinned pages
    }

    public partial class PCache1 : IPCache
    {
        // Cache configuration parameters. Page size (szPage) and the purgeable flag (bPurgeable) are set when the cache is created. nMax may be 
        // modified at any time by a call to the pcache1CacheSize() method. The PGroup mutex must be held when accessing nMax.
        public PGroup Group;        // PGroup this cache belongs to
        public int SizePage;        // Size of allocated pages in bytes
        public int SizeExtra;       // Size of extra space in bytes
        public bool Purgeable;      // True if cache is purgeable
        public uint Min;            // Minimum number of pages reserved
        public uint Max;            // Configured "cache_size" value
        public uint N90pct;         // nMax*9/10
        public Pid MaxID;          // Largest key seen since xTruncate()
        // Hash table of all pages. The following variables may only be accessed when the accessor is holding the PGroup mutex.
        public uint Recyclables;    // Number of pages in the LRU list
        public uint Pages;          // Total number of pages in apHash
        public PgHdr1[] Hash;       // Hash table for fast lookup by key
        // For C#
        public void memset()
        {
            Recyclables = 0;
            Pages = 0;
            Hash = null;
            MaxID = 0;
        }
    }

    public class PgHdr1
    {
        public ICachePage Page;
        public Pid ID;             // Key value (page number)
        public PgHdr1 Next;         // Next in hash table chain
        public PCache1 Cache;       // Cache that currently owns this page
        public PgHdr1 LruNext;      // Next in LRU list of unpinned pages
        public PgHdr1 LruPrev;      // Previous in LRU list of unpinned pages
        // For C#
        public PgHdr _PgHdr = new PgHdr();
        public void memset()
        {
            ID = 0;
            Next = null;
            Cache = null;
            LruNext = LruPrev = null;
            // For C#
            _PgHdr.memset();
        }
    }

    public class PgFreeslot
    {
        public PgFreeslot Next;     // Next free slot
        // For C#
        internal PgHdr _PgHdr;
    }

    public struct PCacheGlobal
    {
        public PGroup Group;   // The global PGroup for mode (2)
        // Variables related to CONFIG_PAGECACHE settings.  The szSlot, nSlot, pStart, pEnd, nReserve, and isInit values are all
        // fixed at sqlite3_initialize() time and do not require mutex protection. The nFreeSlot and pFree values do require mutex protection.
        public bool IsInit;         // True if initialized
        public int SizeSlot;        // Size of each free slot
        public int Slots;           // The number of pcache slots
        public int Reserves;        // Try to keep nFreeSlot above this
        public object Start, End;   // Bounds of pagecache malloc range
        // Above requires no mutex.  Use mutex below for variable that follow.
        public MutexEx Mutex;       // Mutex for accessing the following:
        public PgFreeslot Free;     // Free page blocks
        public int FreeSlots;       // Number of unused pcache slots
        // The following value requires a mutex to change.  We skip the mutex on reading because (1) most platforms read a 32-bit integer atomically and
        // (2) even if an incorrect value is read, no great harm is done since this is really just an optimization.
        public bool UnderPressure;  // True if low on PAGECACHE memory
    }

    #endregion

    public partial class PCache1
    {
        private static PCacheGlobal _pcache1;
        private static bool _config_coreMutex = false;

        #region Page Allocation

        internal static void BufferSetup(object buffer, int size, int n)
        {
            if (_pcache1.IsInit)
            {
                size = SysEx.ROUNDDOWN8(size);
                _pcache1.SizeSlot = size;
                _pcache1.Slots = _pcache1.FreeSlots = n;
                _pcache1.Reserves = (n > 90 ? 10 : (n / 10 + 1));
                _pcache1.Start = buffer;
                _pcache1.End = null;
                _pcache1.Free = null;
                _pcache1.UnderPressure = false;
                while (n-- > 0)
                {
                    var p = new PgFreeslot { _PgHdr = new PgHdr() }; //(PgFreeslot *)buffer;
                    p.Next = _pcache1.Free;
                    _pcache1.Free = p;
                    //buffer = (void*)&((char*)buffer)[size];
                }
                _pcache1.End = buffer;
            }
        }

        internal static PgHdr Alloc(int bytes)
        {
            Debug.Assert(MutexEx.NotHeld(_pcache1.Mutex));
            StatusEx.StatusSet(StatusEx.STATUS.PAGECACHE_SIZE, bytes);
            PgHdr p = null;
            if (bytes <= _pcache1.SizeSlot)
            {
                MutexEx.Enter(_pcache1.Mutex);
                p = (PgHdr)_pcache1.Free._PgHdr;
                if (p != null)
                {
                    _pcache1.Free = _pcache1.Free.Next;
                    _pcache1.FreeSlots--;
                    _pcache1.UnderPressure = (_pcache1.FreeSlots < _pcache1.Reserves);
                    Debug.Assert(_pcache1.FreeSlots >= 0);
                    StatusEx.StatusAdd(StatusEx.STATUS.PAGECACHE_USED, 1);
                }
                MutexEx.Leave(_pcache1.Mutex);
            }
            if (p == null)
            {
                // Memory is not available in the SQLITE_CONFIG_PAGECACHE pool.  Get it from sqlite3Malloc instead.
                p = new PgHdr(); //_alloc(bytes);
                //if (p != null)
                {
                    var size = bytes; //_allocsize(p);
                    MutexEx.Enter(_pcache1.Mutex);
                    StatusEx.StatusAdd(StatusEx.STATUS.PAGECACHE_OVERFLOW, size);
                    MutexEx.Leave(_pcache1.Mutex);
                }
                SysEx.MemdebugSetType(p, SysEx.MEMTYPE.PCACHE);
            }
            return p;
        }

        internal static int Free(ref PgHdr p)
        {
            int freed = 0;
            if (p == null)
                return 0;
            if (p._CacheAllocated) //if (p >= _pcache1.Start && p < _pcache1.End)
            {
                MutexEx.Enter(_pcache1.Mutex);
                StatusEx.StatusAdd(StatusEx.STATUS.PAGECACHE_USED, -1);
                var slot = new PgFreeslot { _PgHdr = p }; //var slot = (PgFreeslot *)p;
                slot.Next = _pcache1.Free;
                _pcache1.Free = slot;
                _pcache1.FreeSlots++;
                _pcache1.UnderPressure = (_pcache1.FreeSlots < _pcache1.Reserves);
                Debug.Assert(_pcache1.FreeSlots <= _pcache1.Slots);
                MutexEx.Leave(_pcache1.Mutex);
            }
            else
            {
                Debug.Assert(SysEx.MemdebugHasType(p, SysEx.MEMTYPE.PCACHE));
                SysEx.MemdebugSetType(p, SysEx.MEMTYPE.HEAP);
                freed = SysEx.AllocSize(p.Data);
#if !DISABLE_PAGECACHE_OVERFLOW_STATS
                MutexEx.Enter(_pcache1.Mutex);
                StatusEx.StatusAdd(StatusEx.STATUS.PAGECACHE_OVERFLOW, -freed);
                MutexEx.Leave(_pcache1.Mutex);
#endif
                C._free(ref p.Data);
            }
            return freed;
        }

#if ENABLE_MEMORY_MANAGEMENT
        private static int MemSize(PgHdr p)
        {
            if (p._CacheAllocated)
                return _pcache1.SizeSlot;
            Debug.Assert(SysEx.MemdebugHasType(p, SysEx.MEMTYPE.PCACHE));
            SysEx.MemdebugSetType(p, SysEx.MEMTYPE.HEAP);
            var size = SysEx.AllocSize(p.Data);
            SysEx.MemdebugSetType(p, SysEx.MEMTYPE.PCACHE);
            return size;
        }
#endif

        private static PgHdr1 AllocPage(PCache1 cache)
        {
            Debug.Assert(MutexEx.Held(cache.Group.Mutex));
            MutexEx.Leave(cache.Group.Mutex);
            PgHdr pg;
            PgHdr1 p = null;
#if true || PCACHE_SEPARATE_HEADER
            pg = Alloc(cache.SizePage);
            p = new PgHdr1();
#else
		    //pg = Alloc(sizeof(PgHdr1) + t.SizePage + t.SizeExtra);
		    //p = (PgHdr1 *)&((uint8 *)pg)[t.SizePage];
#endif
            MutexEx.Enter(cache.Group.Mutex);
            //if (pg != null)
            {
                //p.Page.Buffer = pg;
                //p.Page.Extra = null;
                if (cache.Purgeable)
                    cache.Group.CurrentPages++;
                return p;
            }
        }

        private static void FreePage(ref PgHdr1 p)
        {
            if (C._ALWAYS(p != null))
            {
                var cache = p.Cache;
                Debug.Assert(MutexEx.Held(p.Cache.Group.Mutex));
#if true || PCACHE_SEPARATE_HEADER
                Free(ref p._PgHdr);
#endif
                if (cache.Purgeable)
                    cache.Group.CurrentPages--;
            }
        }

        private bool UnderMemoryPressure()
        {
            return (_pcache1.Slots != 0 && SizePage <= _pcache1.SizeSlot ? _pcache1.UnderPressure : SysEx.HeapNearlyFull());
        }

        #endregion

        #region General

        private static RC ResizeHash(PCache1 p)
        {
            Debug.Assert(MutexEx.Held(p.Group.Mutex));
            var newLength = p.Hash.Length * 2;
            if (newLength < 256)
                newLength = 256;
            MutexEx.Leave(p.Group.Mutex);
            if (p.Hash.Length != 0)
                C._benignalloc_begin();
            var newHash = new PgHdr1[newLength];
            if (p.Hash.Length != 0)
                C._benignalloc_end();
            MutexEx.Enter(p.Group.Mutex);
            if (newHash != null)
            {
                for (var i = 0U; i < p.Hash.Length; i++)
                {
                    PgHdr1 page;
                    var next = p.Hash[i];
                    while ((page = next) != null)
                    {
                        var h = (uint)(page.ID % newLength);
                        next = page.Next;
                        page.Next = newHash[h];
                        newHash[h] = page;
                    }
                }
                p.Hash = newHash;
            }
            return (p.Hash != null ? RC.OK : RC.NOMEM);
        }

        private static void PinPage(PgHdr1 page)
        {
            if (page == null)
                return;
            var cache = page.Cache;
            var group = cache.Group;
            Debug.Assert(MutexEx.Held(group.Mutex));
            if (page.LruNext != null || page == group.LruTail)
            {
                if (page.LruPrev != null)
                    page.LruPrev.LruNext = page.LruNext;
                if (page.LruNext != null)
                    page.LruNext.LruPrev = page.LruPrev;
                if (group.LruHead == page)
                    group.LruHead = page.LruNext;
                if (group.LruTail == page)
                    group.LruTail = page.LruPrev;
                page.LruNext = null;
                page.LruPrev = null;
                page.Cache.Recyclables--;
            }
        }

        private static void RemoveFromHash(PgHdr1 page)
        {
            var cache = page.Cache;
            Debug.Assert(MutexEx.Held(cache.Group.Mutex));
            var h = (uint)(page.ID % cache.Hash.Length);
            PgHdr1 pp;
            PgHdr1 prev = null;
            for (pp = cache.Hash[h]; pp != page; prev = pp, pp = pp.Next) ;
            if (prev == null)
                cache.Hash[h] = pp.Next;
            else
                prev.Next = pp.Next;
            cache.Pages--;
        }

        private static void EnforceMaxPage(PGroup group)
        {
            Debug.Assert(MutexEx.Held(group.Mutex));
            while (group.CurrentPages > group.MaxPages && group.LruTail != null)
            {
                PgHdr1 p = group.LruTail;
                Debug.Assert(p.Cache.Group == group);
                PinPage(p);
                RemoveFromHash(p);
                FreePage(ref p);
            }
        }

        private static void TruncateUnsafe(PCache1 p, Pid limit)
        {
#if DEBUG
            uint pages = 0;
#endif
            Debug.Assert(MutexEx.Held(p.Group.Mutex));
            for (uint h = 0; h < p.Hash.Length; h++)
            {
                var pp = p.Hash[h];
                PgHdr1 page;
                PgHdr1 prev = null;
                while ((page = pp) != null)
                {
                    if (page.ID >= limit)
                    {
                        p.Pages--;
                        pp = page.Next;
                        PinPage(page);
                        if (p.Hash[h] == page)
                            p.Hash[h] = page.Next;
                        else
                            prev.Next = pp;
                        FreePage(ref page);
                    }
                    else
                    {
                        pp = page.Next;
#if DEBUG
                        pages++;
#endif
                    }
                    prev = page;
                }
            }
#if DEBUG
            Debug.Assert(p.Pages == pages);
#endif
        }

        #endregion

        #region Interface

        public RC Init()
        {
            Debug.Assert(!_pcache1.IsInit);
            _pcache1 = new PCacheGlobal { Group = new PGroup() };
            if (_config_coreMutex)
            {
                _pcache1.Group.Mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_LRU);
                _pcache1.Mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_PMEM);
            }
            _pcache1.Group.MaxPinned = 10;
            _pcache1.IsInit = true;
            return RC.OK;
        }

        public void Shutdown()
        {
            Debug.Assert(_pcache1.IsInit);
            _pcache1 = new PCacheGlobal { Group = new PGroup() };
        }

        public IPCache Create(int sizePage, int sizeExtra, bool purgeable)
        {
            // The seperateCache variable is true if each PCache has its own private PGroup.  In other words, separateCache is true for mode (1) where no
            // mutexing is required.
            // *  Always use a unified cache (mode-2) if ENABLE_MEMORY_MANAGEMENT
            // *  Always use a unified cache in single-threaded applications
            // *  Otherwise (if multi-threaded and ENABLE_MEMORY_MANAGEMENT is off) use separate caches (mode-1)
#if ENABLE_MEMORY_MANAGEMENT || !THREADSAFE
            const bool separateCache = false;
#else
            bool separateCache = _config_coreMutex > 0;
#endif
            Debug.Assert((sizePage & (sizePage - 1)) == 0 && sizePage >= 512 && sizePage <= 65536);
            Debug.Assert(sizeExtra < 300);
            //int size = sizeof(PCache1) + sizeof(PGroup) * (int)separateCache;
            var cache = new PCache1();
            if (cache != null)
            {
                PGroup group;
                if (separateCache)
                {
                    //group = new PGroup();
                    //group.MaxPinned = 10;
                }
                else
                    group = _pcache1.Group;
                cache.Group = group;
                cache.SizePage = sizePage;
                cache.SizeExtra = sizeExtra;
                cache.Purgeable = purgeable;
                if (purgeable)
                {
                    cache.Min = 10;
                    MutexEx.Enter(group.Mutex);
                    group.MinPages += cache.Min;
                    group.MaxPinned = group.MaxPages + 10 - group.MinPages;
                    MutexEx.Leave(group.Mutex);
                }
            }
            return (IPCache)cache;
        }

        public void Cachesize(uint max)
        {
            if (Purgeable)
            {
                var group = Group;
                MutexEx.Enter(group.Mutex);
                group.MaxPages += (max - Max);
                group.MaxPinned = group.MaxPages + 10 - group.MinPages;
                Max = max;
                N90pct = Max * 9 / 10;
                EnforceMaxPage(group);
                MutexEx.Leave(group.Mutex);
            }
        }

        public void Shrink()
        {
            if (Purgeable)
            {
                var group = Group;
                MutexEx.Enter(group.Mutex);
                uint savedMaxPages = group.MaxPages;
                group.MaxPages = 0;
                EnforceMaxPage(group);
                group.MaxPages = savedMaxPages;
                MutexEx.Leave(group.Mutex);
            }
        }

        public int get_Pages()
        {
            MutexEx.Enter(Group.Mutex);
            var pages = (int)Pages;
            MutexEx.Leave(Group.Mutex);
            return pages;
        }

        public ICachePage Fetch(Pid id, bool createFlag)
        {
            Debug.Assert(Purgeable || !createFlag);
            Debug.Assert(Purgeable || Min == 0);
            Debug.Assert(!Purgeable || Min == 10);
            PGroup group;
            MutexEx.Enter((group = Group).Mutex);

            // Step 1: Search the hash table for an existing entry.
            PgHdr1 page = null;
            if (Hash.Length > 0)
            {
                var h = (int)(id % Hash.Length);
                for (page = Hash[h]; page != null && page.ID != id; page = page.Next) ;
            }

            // Step 2: Abort if no existing page is found and createFlag is 0
            if (page != null || !createFlag)
            {
                PinPage(page);
                goto fetch_out;
            }

            // The pGroup local variable will normally be initialized by the pcache1EnterMutex() macro above.  But if SQLITE_MUTEX_OMIT is defined,
            // then pcache1EnterMutex() is a no-op, so we have to initialize the local variable here.  Delaying the initialization of pGroup is an
            // optimization:  The common case is to exit the module before reaching this point.
#if MUTEX_OMIT
            group = cache.Group;
#endif

            // Step 3: Abort if createFlag is 1 but the cache is nearly full
            Debug.Assert(Pages - Recyclables >= 0);
            var pinned = Pages - Recyclables;
            Debug.Assert(group.MaxPinned == group.MaxPages + 10 - group.MinPages);
            Debug.Assert(N90pct == Max * 9 / 10);
            if (createFlag && (pinned >= group.MaxPinned || pinned >= (int)N90pct || UnderMemoryPressure()))
                goto fetch_out;
            if (Pages >= Hash.Length && ResizeHash(this) != 0)
                goto fetch_out;

            // Step 4. Try to recycle a page.
            if (Purgeable && group.LruTail != null && ((Pages + 1 >= Max) || group.CurrentPages >= group.MaxPages || UnderMemoryPressure()))
            {
                page = group.LruTail;
                RemoveFromHash(page);
                PinPage(page);
                PCache1 other = page.Cache;

                // We want to verify that szPage and szExtra are the same for pOther and pCache.  Assert that we can verify this by comparing sums.                
                Debug.Assert((SizePage & (SizePage - 1)) == 0 && SizePage >= 512);
                Debug.Assert(SizeExtra < 512);
                Debug.Assert((other.SizePage & (other.SizePage - 1)) == 0 && other.SizePage >= 512);
                Debug.Assert(other.SizeExtra < 512);

                if (other.SizePage + other.SizeExtra != SizePage + SizeExtra)
                {
                    FreePage(ref page);
                    page = null;
                }
                else
                    group.CurrentPages -= (other.Purgeable ? 1U : 0U) - (Purgeable ? 1U : 0U);
            }

            // Step 5. If a usable page buffer has still not been found, attempt to allocate a new one. 
            if (page == null)
            {
                if (createFlag) C._benignalloc_begin();
                page = AllocPage(this);
                if (createFlag) C._benignalloc_end();
            }
            if (page != null)
            {
                var h = (uint)(id % Hash.Length);
                Pages++;
                page.ID = id;
                page.Next = Hash[h];
                page.Cache = this;
                page.LruPrev = null;
                page.LruNext = null;
                page.Page.Extra = null;
                Hash[h] = page;
                //PGHDR1_TO_PAGE(page).ClearState();
                //page.Page.PgHdr1 = page;            
            }

        fetch_out:
            if (page != null && id > MaxID)
                MaxID = id;
            MutexEx.Leave(group.Mutex);
            return page.Page;
        }

        public void Unpin(ICachePage pg, bool reuseUnlikely)
        {
            var page = (PgHdr1)pg._PgHdr1;
            var group = Group;
            Debug.Assert(page.Cache == this);
            MutexEx.Enter(Group.Mutex);
            // It is an error to call this function if the page is already  part of the PGroup LRU list.
            Debug.Assert(page.LruPrev == null && page.LruNext == null);
            Debug.Assert(group.LruHead != page && group.LruTail != page);
            if (reuseUnlikely || group.CurrentPages > group.MaxPages)
            {
                RemoveFromHash(page);
                FreePage(ref page);
            }
            else
            {
                // Add the page to the PGroup LRU list. 
                if (group.LruHead != null)
                {
                    group.LruHead.LruPrev = page;
                    page.LruNext = group.LruHead;
                    group.LruHead = page;
                }
                else
                {
                    group.LruTail = page;
                    group.LruHead = page;
                }
                Recyclables++;
            }
            MutexEx.Leave(group.Mutex);
        }

        public void Rekey(ICachePage pg, Pid old, Pid new_)
        {
            var page = (PgHdr1)pg._PgHdr1;
            Debug.Assert(page.ID == old);
            Debug.Assert(page.Cache == this);
            MutexEx.Enter(Group.Mutex);
            var h = (uint)(old % Hash.Length);
            var pp = Hash[h];
            while (pp != page)
                pp = pp.Next;
            if (pp == Hash[h])
                Hash[h] = pp.Next;
            else
                pp.Next = page.Next;
            h = (uint)(new_ % Hash.Length);
            page.ID = new_;
            page.Next = Hash[h];
            Hash[h] = page;
            if (new_ > MaxID)
                MaxID = new_;
            MutexEx.Leave(Group.Mutex);
        }

        public void Truncate(Pid limit)
        {
            MutexEx.Enter(Group.Mutex);
            if (limit <= MaxID)
            {
                TruncateUnsafe(this, limit);
                MaxID = limit - 1;
            }
            MutexEx.Leave(Group.Mutex);
        }

        public void Destroy(ref IPCache p)
        {
            var cache = (PCache1)p;
            var group = cache.Group;
            Debug.Assert(cache.Purgeable || (cache.Max == 0 && cache.Min == 0));
            MutexEx.Enter(group.Mutex);
            TruncateUnsafe(cache, 0);
            Debug.Assert(group.MaxPages >= cache.Max);
            group.MaxPages -= cache.Max;
            Debug.Assert(group.MinPages >= cache.Min);
            group.MinPages -= cache.Min;
            group.MaxPinned = group.MaxPages + 10 - group.MinPages;
            EnforceMaxPage(group);
            MutexEx.Leave(group.Mutex);
            //C._free(ref cache.Hash);
            //C._free(ref cache);
            cache = null;
        }

#if ENABLE_MEMORY_MANAGEMENT
        int ReleaseMemory(int required)
        {
            Debug.Assert(MutexEx.NotHeld(_pcache1.Group.Mutex));
            Debug.Assert(MutexEx.NotHeld(_pcache1.Mutex));
            int free = 0;
            if (_pcache1.Start == null)
            {
                PgHdr1 p;
                MutexEx.Enter(_pcache1.Group.Mutex);
                while ((required < 0 || free < required) && ((p = _pcache1.Group.LruTail) != null))
                {
                    free += MemSize(p.Page);
#if PCACHE_SEPARATE_HEADER
                    free += MemSize(p);
#endif
                    PinPage(p);
                    RemoveFromHash(p);
                    FreePage(ref p);
                }
                MutexEx.Leave(_pcache1.Group.Mutex);
            }
            return free;
        }
#endif

        #endregion

        #region Tests
#if TEST

        void PCache1_testStats(out uint current, out uint max, out uint min, out uint recyclables)
        {
            uint recyclables2 = 0;
            for (PgHdr1 p = _pcache1.Group.LruHead; p != null; p = p.LruNext)
                recyclables2++;
            current = _pcache1.Group.CurrentPages;
            max = _pcache1.Group.MaxPages;
            min = _pcache1.Group.MinPages;
            recyclables = recyclables2;
        }

#endif
        #endregion
    }
}
