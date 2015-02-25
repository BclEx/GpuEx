namespace Core
{
	struct MutexEx;
	__device__ extern MutexEx MutexEx_Empty;
	__device__ extern MutexEx MutexEx_NotEmpty;

	struct MutexEx
	{
	public:
		enum MUTEX
		{
			MUTEX_FAST = 0,
			MUTEX_RECURSIVE = 1,
			MUTEX_STATIC_MASTER = 2,
			MUTEX_STATIC_MEM = 3,  // sqlite3_malloc()
			MUTEX_STATIC_MEM2 = 4,  // NOT USED
			MUTEX_STATIC_OPEN = 4,  // sqlite3BtreeOpen()
			MUTEX_STATIC_PRNG = 5,  // sqlite3_random()
			MUTEX_STATIC_LRU = 6,   // lru page list
			MUTEX_STATIC_LRU2 = 7,  // NOT USED
			MUTEX_STATIC_PMEM = 7, // sqlite3PageMalloc()
		};

		void *Tag;

		__device__ static RC Init();
		__device__ static RC End();

		__device__ inline static MutexEx Alloc(MUTEX id) { return MutexEx_NotEmpty; }
		__device__ inline static void Enter(MutexEx mutex) { }
		__device__ inline static void Leave(MutexEx mutex) { }
		__device__ inline static bool Held(MutexEx mutex) { return true; }
		__device__ inline static bool NotHeld(MutexEx mutex) { return true; }
		__device__ inline static void Free(MutexEx mutex) { }
	};
}
