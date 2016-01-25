// This file contains code used for testing the SQLite system. None of the code in this file goes into a deliverable build.
// 
// This file contains an application-defined pager cache implementation that can be plugged in in place of the
// default pcache.  This alternative pager cache will throw some errors that the default cache does not.
//
// This pagecache implementation is designed for simplicity not speed.  
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <new.h>

// Global data used by this test implementation.  There is no mutexing, which means this page cache will not work in a multi-threaded test.
typedef struct testpcacheGlobalType testpcacheGlobalType;
struct testpcacheGlobalType
{
	void *Dummy;			// Dummy allocation to simulate failures
	int Instances;			// Number of current instances
	unsigned DiscardChance;	// Chance of discarding on an unpin (0-100)
	unsigned PrngSeed;		// Seed for the PRNG
	unsigned HighStress;	// Call xStress agressively
};
__device__ static testpcacheGlobalType _testpcacheGlobal;

// Number of pages in a cache.
// The number of pages is a hard upper bound in this test module. If more pages are requested, sqlite3PcacheFetch() returns NULL.
// If testing with in-memory temp tables, provide a larger pcache. Some of the test cases need this.
#if defined(TEMP_STORE) && TEMP_STORE>=2
#define TESTPCACHE_NPAGE 499
#else
#define TESTPCACHE_NPAGE 217
#endif
#define TESTPCACHE_RESERVE 17

// Magic numbers used to determine validity of the page cache.
#define TESTPCACHE_VALID 0x364585fd
#define TESTPCACHE_CLEAR 0xd42670d4

class TestPCache : public IPCache
{
public:
	int SizePage;				// Size of each page.  Multiple of 8.
	int SizeExtra;				// Size of extra data that accompanies each page
	bool Purgeable;				// True if the page cache is purgeable
	int Free;					// Number of unused slots in a[]
	int Pinned;					// Number of pinned slots in a[]
	unsigned Rand;				// State of the PRNG
	unsigned Magic;				// Magic number for sanity checking
	struct testpcachePage
	{
		ICachePage page;		// Base class
		unsigned Key;			// The key for this page. 0 means unallocated
		bool IsPinned;			// True if the page is pinned
	} As[TESTPCACHE_NPAGE];		// All pages in the cache
public:
	// Initializer
	// Verify that the initializer is only called when the system is uninitialized.  Allocate some memory and report SQLITE_NOMEM if
	// the allocation fails.  This provides a means to test the recovery from a failed initialization attempt.  It also verifies that the
	// the destructor always gets call - otherwise there would be a memory leak.
	__device__ virtual RC Init()
	{
		_assert(!_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances == 0);
		_testpcacheGlobal.Dummy = _alloc(10);
		return (!_testpcacheGlobal.Dummy ? RC_NOMEM : RC_OK);
	}

	// Destructor
	// Verify that this is only called after initialization. Free the memory allocated by the initializer.
	__device__ virtual void Shutdown()
	{
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances == 0);
		_free(_testpcacheGlobal.Dummy);
		_testpcacheGlobal.Dummy = nullptr;
	}

	// Get a random number using the PRNG in the given page cache.
	__device__ unsigned Random()
	{
		unsigned x = 0;
		for (int i = 0; i < 4; i++)
		{
			Rand = (Rand*69069 + 5);
			x = (x<<8) | ((Rand>>16)&0xff);
		}
		return x;
	}

	// Allocate a new page cache instance.
	__device__ virtual IPCache *Create(int sizePage, int sizeExtra, bool purgeable)
	{
		_assert(_testpcacheGlobal.Dummy);
		sizePage = (sizePage+7)&~7;
		int mem = sizeof(TestPCache) + TESTPCACHE_NPAGE*(sizePage+sizeExtra);
		TestPCache *cache = (TestPCache *)_alloc(mem);
		if (!cache) return nullptr;
		cache = new (cache) TestPCache();
		char *x = (char *)&cache[1];
		cache->SizePage = sizePage;
		cache->SizeExtra = sizeExtra;
		cache->Free = TESTPCACHE_NPAGE;
		cache->Pinned = 0;
		cache->Rand = _testpcacheGlobal.PrngSeed;
		cache->Purgeable = purgeable;
		cache->Magic = TESTPCACHE_VALID;
		for (int i = 0; i < TESTPCACHE_NPAGE; i++, x += (sizePage+sizeExtra))
		{
			cache->As[i].Key = 0;
			cache->As[i].IsPinned = false;
			cache->As[i].page.Buffer = (void *)x;
			cache->As[i].page.Extra = (void *)&x[sizePage];
		}
		_testpcacheGlobal.Instances++;
		return (IPCache *)cache;
	}

	// Set the cache size
	__device__ virtual void Cachesize(uint newSize)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
	}

	// Return the number of pages in the cache that are being used. This includes both pinned and unpinned pages.
	__device__ virtual int get_Pages()
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		return TESTPCACHE_NPAGE - Free;
	}

	// Fetch a page.
	__device__ virtual ICachePage *Fetch(Pid key, int createFlag)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		// See if the page is already in cache.  Return immediately if it is
		int i, j;
		for (i = 0; i < TESTPCACHE_NPAGE; i++)
		{
			if (As[i].Key == key)
			{
				if (!As[i].IsPinned)
				{
					Pinned++;
					_assert(Pinned <= TESTPCACHE_NPAGE - Free);
					As[i].IsPinned = true;
				}
				return &As[i].page;
			}
		}
		// If createFlag is 0, never allocate a new page
		if (!createFlag)
			return nullptr;
		// If no pages are available, always fail
		if (Pinned == TESTPCACHE_NPAGE)
			return nullptr;
		// Do not allocate the last TESTPCACHE_RESERVE pages unless createFlag is 2
		if (Pinned >= TESTPCACHE_NPAGE-TESTPCACHE_RESERVE && createFlag < 2)
			return nullptr;
		// Do not allocate if highStress is enabled and createFlag is not 2.  
		// The highStress setting causes pagerStress() to be called much more often, which exercises the pager logic more intensely.
		if (_testpcacheGlobal.HighStress && createFlag < 2)
			return nullptr;
		// Find a free page to allocate if there are any free pages. Withhold TESTPCACHE_RESERVE free pages until createFlag is 2.
		if (Free > TESTPCACHE_RESERVE || (createFlag == 2 && Free > 0))
		{
			j = Random() % TESTPCACHE_NPAGE;
			for (i = 0; i < TESTPCACHE_NPAGE; i++, j = (j+1)%TESTPCACHE_NPAGE)
			{
				if (As[j].Key == 0)
				{
					As[j].Key = key;
					As[j].IsPinned = true;
					_memset(As[j].page.Buffer, 0, SizePage);
					_memset(As[j].page.Extra, 0, SizeExtra);
					Pinned++;
					Free--;
					_assert(Pinned <= TESTPCACHE_NPAGE - Free);
					return &As[j].page;
				}
			}
			// The prior loop always finds a freepage to allocate
			_assert(false);
		}
		// If this cache is not purgeable then we have to fail.
		if (!Purgeable)
			return nullptr;
		// If there are no free pages, recycle a page.  The page to recycle is selected at random from all unpinned pages.
		j = Random() % TESTPCACHE_NPAGE;
		for (i = 0; i < TESTPCACHE_NPAGE; i++, j = (j+1)%TESTPCACHE_NPAGE)
		{
			if (As[j].Key > 0 && !As[j].IsPinned)
			{
				As[j].Key = key;
				As[j].IsPinned = true;
				_memset(As[j].page.Buffer, 0, SizePage);
				_memset(As[j].page.Extra, 0, SizeExtra);
				Pinned++;
				_assert(Pinned <= TESTPCACHE_NPAGE - Free);
				return &As[j].page;
			}
		}
		// The previous loop always finds a page to recycle.
		_assert(false);
		return nullptr;
	}

	// Unpin a page.
	__device__ virtual void Unpin(ICachePage *oldPage, bool discard)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		// Randomly discard pages as they are unpinned according to the discardChance setting.  If discardChance is 0, the random discard
		// never happens.  If discardChance is 100, it always happens.
		if (Purgeable && (100-_testpcacheGlobal.DiscardChance) <= (Random()%100))
			discard = true;
		for (int i = 0; i < TESTPCACHE_NPAGE; i++)
		{
			if (&As[i].page == oldPage)
			{
				// The pOldPage pointer always points to a pinned page
				_assert(As[i].IsPinned);
				As[i].IsPinned = false;
				Pinned--;
				_assert(Pinned >= 0);
				if (discard)
				{
					As[i].Key = 0;
					Free++;
					_assert(Free <= TESTPCACHE_NPAGE);
				}
				return;
			}
		}
		// The pOldPage pointer always points to a valid page
		_assert(false);
	}

	// Rekey a single page.
	__device__ virtual void Rekey(ICachePage *oldPage, Pid oldKey, Pid newKey)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		// If there already exists another page at newKey, verify that the other page is unpinned and discard it.
		int i;
		for (i = 0; i < TESTPCACHE_NPAGE; i++)
		{
			if (As[i].Key == newKey)
			{
				// The new key is never a page that is already pinned
				_assert(!As[i].IsPinned);
				As[i].Key = 0;
				Free++;
				_assert(Free <= TESTPCACHE_NPAGE);
				break;
			}
		}
		// Find the page to be rekeyed and rekey it.
		for (i = 0; i < TESTPCACHE_NPAGE; i++)
		{
			if (As[i].Key == oldKey)
			{
				// The oldKey and pOldPage parameters match
				_assert(&As[i].page == oldPage);
				// Page to be rekeyed must be pinned
				_assert(As[i].IsPinned);
				As[i].Key = newKey;
				return;
			}
		}
		// Rekey is always given a valid page to work with
		_assert(false);
	}

	// Truncate the page cache.  Every page with a key of iLimit or larger is discarded.
	__device__ virtual void Truncate(Pid limit)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		for (int i = 0; i < TESTPCACHE_NPAGE; i++)
			if (As[i].Key >= limit)
			{
				As[i].Key = 0;
				if (As[i].IsPinned)
				{
					Pinned--;
					_assert(Pinned >= 0);
				}
				Free++;
				_assert(Free <= TESTPCACHE_NPAGE);
			}
	}

	// Destroy a page cache.
	__device__ virtual void Destroy(IPCache *p)
	{
		_assert(Magic == TESTPCACHE_VALID);
		_assert(_testpcacheGlobal.Dummy);
		_assert(_testpcacheGlobal.Instances > 0);
		Magic = TESTPCACHE_CLEAR;
		_free(p);
		_testpcacheGlobal.Instances--;
	}

	__device__ virtual void Shrink() { }
};

// Invoke this routine to register or unregister the testing pager cache implemented by this file.
//
// Install the test pager cache if installFlag is 1 and uninstall it if installFlag is 0.
//
// When installing, discardChance is a number between 0 and 100 that indicates the probability of discarding a page when unpinning the
// page.  0 means never discard (unless the discard flag is set). 100 means always discard.
__device__ static TestPCache _testPCache;
__device__ static IPCache *_defaultPCache;
__device__ static int _isInstalled = false;
__device__ void installTestPCache(bool installFlag, unsigned discardChance, unsigned prngSeed, unsigned highStress)
{
	_assert(_testpcacheGlobal.Instances == 0);
	_assert(!_testpcacheGlobal.Dummy);
	_assert(discardChance <= 100);
	_testpcacheGlobal.DiscardChance = discardChance;
	_testpcacheGlobal.PrngSeed = prngSeed ^ (prngSeed<<16);
	_testpcacheGlobal.HighStress = highStress;
	if (installFlag != _isInstalled)
	{
		if (installFlag)
		{
			Main::Config(Main::CONFIG_GETPCACHE2, &_defaultPCache);
			_assert(_defaultPCache != &_testPCache);
			Main::Config(Main::CONFIG_PCACHE2, &_testPCache);
		}
		else
		{
			_assert(_defaultPCache);
			Main::Config(Main::CONFIG_PCACHE2, &_defaultPCache);
		}
		_isInstalled = installFlag;
	}
}
