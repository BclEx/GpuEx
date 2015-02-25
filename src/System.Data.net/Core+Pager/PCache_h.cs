using Pid = System.UInt32;
using System;

namespace Core
{
    public partial class Pager { }

    public class ICachePage : PgHdr { }

    public interface IPCache
    {
        RC Init();
        void Shutdown();
        IPCache Create(int sizePage, int sizeExtra, bool purgeable);
        void Cachesize(uint max);
        void Shrink();
        int get_Pages();
        ICachePage Fetch(Pid id, bool createFlag);
        void Unpin(ICachePage pg, bool reuseUnlikely);
        void Rekey(ICachePage pg, Pid old, Pid new_);
        void Truncate(Pid limit);
        void Destroy(ref IPCache p);
    };

    public class PgHdr
    {
        public enum PGHDR : ushort
        {
            DIRTY = 0x002,          // Page has changed
            NEED_SYNC = 0x004,      // Fsync the rollback journal before writing this page to the database
            NEED_READ = 0x008,      // Content is unread
            REUSE_UNLIKELY = 0x010, // A hint that reuse is unlikely
            DONT_WRITE = 0x020,     // Do not write content to disk
        }
        public ICachePage Page;
        public byte[] Data;
        public object Extra;        // Extra content
        public PgHdr Dirty;       // Transient list of dirty pages
        public Pager Pager;         // The pager to which this page belongs
        public Pid ID;              // The page number for this page
#if CHECK_PAGES
        public uint PageHash;       // Hash of page content
#endif
        public PGHDR Flags;         // PGHDR flags defined below
        // Elements above are public.  All that follows is private to pcache.c and should not be accessed by other modules.
        internal int Refs;          // Number of users of this page
        internal PCache Cache;      // Cache that owns this page
        internal PgHdr DirtyNext;   // Next element in list of dirty pages
        internal PgHdr DirtyPrev;   // Previous element in list of dirty pages

        // For C#
        public bool _CacheAllocated { get; set; }
        public PgHdr1 _PgHdr1 { get; set; }
        //public static implicit operator bool(PgHdr b) { return (b != null); }

        public void memset()
        {
            //Page = null;
            Data = null;
            Extra = null;
            Dirty = null;
            Pager = null;
            ID = 0;
#if CHECK_PAGES
            PageHash = 0;
#endif
            Flags = 0;
            Refs = 0;
            Cache = null;
            DirtyNext = null;
            DirtyPrev = null;
        }

        internal void memsetData(int sizePage)
        {
            Data = C._alloc(sizePage);
        }
    }

    public partial class PCache
    {
        public PgHdr Dirty, DirtyTail;  // List of dirty pages in LRU order
        public PgHdr Synced;        // Last synced page in dirty page list
        public int Refs;            // Number of referenced pages
        public int SizeCache;       // Configured cache size
        public int SizePage;        // Size of every page in this cache
        public int SizeExtra;       // Size of extra space for each page
        public bool Purgeable;      // True if pages are on backing store
        public Func<object, PgHdr, RC> Stress;   // Call to try make a page clean
        public object StressArg;    // Argument to xStress
        public IPCache Cache;       // Pluggable cache module
        public PgHdr Page1;         // Reference to page 1

        public void memset()
        {
            Dirty = DirtyTail = null;
            Synced = null;
            Refs = 0;
        }
    }
}