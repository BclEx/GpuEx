#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#include <string.h>
#include "FallocHost.h"
#include "cuda_runtime_api.h"

typedef struct __align__(8)
{
	unsigned short magic;		// magic number says we're valid
	unsigned short count;		// number of blocks in sequence
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} fallocBlockHeader;

typedef struct __align__(8)
{
	fallocBlockHeader *block;	// block reference
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} fallocBlockRef;

typedef struct __align__(8)
{
	void *reserved;
	size_t blockSize;
	size_t blocksLength;
	size_t blockRefsLength; // Size of circular buffer (set up by host)
	fallocBlockRef *blockRefs; // Start of circular buffer (set up by host)
	volatile fallocBlockRef *freeBlockPtr; // Current atomically-incremented non-wrapped offset
	volatile fallocBlockRef *retnBlockPtr; // Current atomically-incremented non-wrapped offset
	char *blocks;
} fallocHeap;

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

inline static void writeBlockRef(fallocBlockRef *ref, fallocBlockHeader *block)
{
	ref->block = block;
	ref->blockid = 0;
	ref->threadid = 0;
}

//
//  cudaDeviceFallocCreate
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cudaDeviceFalloc cudaDeviceFallocCreate(size_t blockSize, size_t length, cudaError_t *error, void *reserved)
{
	cudaError_t localError; if (error == nullptr) error = &localError;
	cudaDeviceFalloc host; memset(&host, 0, sizeof(cudaDeviceFalloc));
	// fix up blockSize to include fallocBlockHeader
	blockSize = (blockSize + sizeof(fallocBlockHeader) + 15) & ~15;
	// fix up length to be a multiple of blockSize
	if (!length || length % blockSize)
		length += blockSize - (length % blockSize);
	size_t blocksLength = length;
	size_t blocks = (size_t)(blocksLength / blockSize);
	if (!blocks)
		return host;
	// fix up length to include fallocHeap + freeblocks
	unsigned int blockRefsLength = (unsigned int)(blocks * sizeof(fallocBlockRef));
	length = (length + blockRefsLength + sizeof(fallocHeap) + 15) & ~15;
	// allocate a heap on the device and zero it
	fallocHeap *heap;
	if ((*error = cudaMalloc((void **)&heap, length)) != cudaSuccess || (*error = cudaMemset(heap, 0, length)) != cudaSuccess)
		return host;
	// transfer to heap
	fallocHeap hostHeap;
	hostHeap.reserved = reserved;
	hostHeap.blockSize = blockSize;
	hostHeap.blocksLength = blocksLength;
	hostHeap.blockRefsLength = blockRefsLength;
	hostHeap.blockRefs = (fallocBlockRef *)((char *)heap + sizeof(fallocHeap));
	hostHeap.freeBlockPtr = hostHeap.retnBlockPtr = (volatile fallocBlockRef *)hostHeap.blockRefs;
	hostHeap.blocks = (char *)hostHeap.blockRefs + blockRefsLength;
	if ((*error = cudaMemcpy(heap, &hostHeap, sizeof(fallocHeap), cudaMemcpyHostToDevice)) != cudaSuccess)
		return host;
	// initial blockrefs
	char *block = hostHeap.blocks;
	fallocBlockRef *hostBlockRefs = new fallocBlockRef[blocks];
	int i;
	fallocBlockRef *r;
	for (i = 0, r = hostBlockRefs; i < blocks; i++, r++, block += blockSize)
		writeBlockRef(r, (fallocBlockHeader *)block);
	// transfer to heap
	*error = cudaMemcpy(hostHeap.blockRefs, hostBlockRefs, sizeof(fallocBlockRef) * blocks, cudaMemcpyHostToDevice);
	delete hostBlockRefs;
	if (*error != cudaSuccess)
		return host;
	// return the heap
	host.reserved = reserved;
	host.heap = heap;
	host.blockSize = blockSize;
	host.blocksLength = blocksLength;
	host.length = length;
	return host;
}

//
//  cudaDeviceFallocDestroy
//
//  Frees up the memory which we allocated
//
extern "C" void cudaDeviceFallocDestroy(cudaDeviceFalloc &host) {
	if (!host.heap)
		return;
	cudaFree(host.heap); host.heap = nullptr;
}
