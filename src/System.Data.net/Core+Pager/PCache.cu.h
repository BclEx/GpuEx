// pcache.h
namespace Core
{
	typedef struct PgHdr PgHdr;
	typedef struct PCache PCache;

	struct ICachePage
	{
		void *Buffer;	// The content of the page
		void *Extra;	// Extra information associated with the page
	};

	class IPCache
	{
	public:
		__device__ virtual RC Init() = 0;
		__device__ virtual void Shutdown() = 0;
		__device__ virtual IPCache *Create(int sizePage, int sizeExtra, bool purgeable) = 0;
		__device__ virtual void Cachesize(uint max) = 0;
		__device__ virtual void Shrink() = 0;
		__device__ virtual int get_Pages() = 0;
		__device__ virtual ICachePage *Fetch(Pid id, bool createFlag) = 0;
		__device__ virtual void Unpin(ICachePage *pg, bool reuseUnlikely) = 0;
		__device__ virtual void Rekey(ICachePage *pg, Pid old, Pid new_) = 0;
		__device__ virtual void Truncate(Pid limit) = 0;
		__device__ virtual void Destroy(IPCache *p) = 0;
	};

	struct PgHdr
	{
		enum PGHDR : uint16
		{
			PGHDR_DIRTY = 0x002,			// Page has changed
			PGHDR_NEED_SYNC = 0x004,		// Fsync the rollback journal before writing this page to the database
			PGHDR_NEED_READ = 0x008,		// Content is unread
			PGHDR_REUSE_UNLIKELY = 0x010, // A hint that reuse is unlikely
			PGHDR_DONT_WRITE = 0x020		// Do not write content to disk 
		};
		ICachePage *Page;			// Pcache object page handle
		void *Data;					// Page data
		void *Extra;				// Extra content
		PgHdr *Dirty;				// Transient list of dirty pages
		Pager *Pager;				// The pager this page is part of
		Pid ID;						// Page number for this page
#ifdef CHECK_PAGES
		uint32 PageHash;            // Hash of page content
#endif
		uint16 Flags;                // PGHDR flags defined below
		// Elements above are public.  All that follows is private to pcache.c and should not be accessed by other modules.
		int16 Refs;					// Number of users of this page
		PCache *Cache;              // Cache that owns this page
		PgHdr *DirtyNext;           // Next element in list of dirty pages
		PgHdr *DirtyPrev;           // Previous element in list of dirty pages
	};

	struct PCache
	{
		PgHdr *Dirty, *DirtyTail;   // List of dirty pages in LRU order
		PgHdr *Synced;              // Last synced page in dirty page list
		int Refs;                   // Number of referenced pages
		int SizeCache;              // Configured cache size
		int SizePage;               // Size of every page in this cache
		int SizeExtra;              // Size of extra space for each page
		bool Purgeable;             // True if pages are on backing store
		RC (*Stress)(void *, PgHdr *);// Call to try make a page clean
		void *StressArg;            // Argument to xStress
		IPCache *Cache;				// Pluggable cache module
		PgHdr *Page1;				// Reference to page 1
	public:
		__device__ static RC Initialize();
		__device__ static void Shutdown();
		__device__ static int SizeOf();
		__device__ static void Open(int sizePage, int sizeExtra, bool purgeable, RC (*stress)(void *, PgHdr *), void *stressArg, PCache *p);
		__device__ void SetPageSize(int sizePage);
		__device__ RC Fetch(Pid id, bool createFlag, PgHdr **pageOut);
		__device__ static void Release(PgHdr *p);
		__device__ static void Ref(PgHdr *p);
		__device__ static void Drop(PgHdr *p);			// Remove page from cache
		__device__ static void MakeDirty(PgHdr *p);	// Make sure page is marked dirty
		__device__ static void MakeClean(PgHdr *p);	// Mark a single page as clean
		__device__ void CleanAll();					// Mark all dirty list pages as clean
		__device__ void ClearSyncFlags();
		__device__ static void Move(PgHdr *p, Pid newID);
		__device__ void Truncate(Pid id);
		__device__ void Close();
		__device__ void Clear();
		__device__ PgHdr *DirtyList();
		__device__ int get_Refs();
		__device__ static int get_PageRefs(PgHdr *p);
		__device__ int get_Pages();
		__device__ uint get_CacheSize();
		__device__ void set_CacheSize(int maxPage);
		__device__ void Shrink();
#if defined(CHECK_PAGES) || defined(_DEBUG)
		__device__ void IterateDirty(void (*iter)(PgHdr *));
#endif
#ifdef ENABLE_MEMORY_MANAGEMENT
		__device__ static int ReleaseMemory(int required);
#endif
#ifdef TEST
		__device__ void PCache1_testStats(uint *current, uint *max, uint *min, uint *recyclables);
#endif
		// from pcache1
		__device__ static void PageBufferSetup(void *buffer, int size, int n);
		__device__ static void *PageAlloc(int size);
		__device__ static void PageFree(void *p);
	};
}
