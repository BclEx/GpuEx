// sqlite3.h
namespace Core
{
	class VAlloc
	{
	public:
		__device__ virtual void *Alloc(int bytes);			// Memory allocation function
		__device__ virtual void Free(void *prior);			// Free a prior allocation
		__device__ virtual void *Realloc(void *prior, int bytes);	// Resize an allocation
		__device__ virtual int Size(void *p);				// Return the size of an allocation
		__device__ virtual int Roundup(int bytes);			// Round up request size to allocation size
		__device__ virtual RC Init(void *appData);			// Initialize the memory allocator
		__device__ virtual void Shutdown(void *appData);	// Deinitialize the memory allocator
		void *AppData;										// Argument to xInit() and xShutdown()
	};
}