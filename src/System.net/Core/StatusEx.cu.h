// status.c
namespace Core
{
	class StatusEx
	{
	public:
		enum STATUS
		{
			STATUS_MEMORY_USED = 0,
			STATUS_PAGECACHE_USED = 1,
			STATUS_PAGECACHE_OVERFLOW = 2,
			STATUS_SCRATCH_USED = 3,
			STATUS_SCRATCH_OVERFLOW = 4,
			STATUS_MALLOC_SIZE = 5,
			STATUS_PARSER_STACK = 6,
			STATUS_PAGECACHE_SIZE = 7,
			STATUS_SCRATCH_SIZE = 8,
			STATUS_MALLOC_COUNT = 9,
		};

		__device__ static int StatusValue(STATUS op);
		__device__ static void StatusAdd(STATUS op, int n);
		__device__ static void StatusSet(StatusEx::STATUS op, int x);
		__device__ static int Status(StatusEx::STATUS op, int *current, int *highwater, int resetFlag);
	};
}
