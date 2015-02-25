#if __CUDA_ARCH__ == 100 
# error Atomics only used with > sm_10 architecture
#endif
#include "cuda_runtime_api.h"
#include <malloc.h>
#include <string.h>

typedef struct __align__(8) _cuFallocHeapChunk {
	unsigned short magic;
	unsigned short count;
	volatile struct _cuFallocHeapChunk* next;
	void* reserved;
} fallocHeapChunk;

typedef struct __align__(8) _cuFallocHeap {
	size_t chunkSize;
	size_t chunks;
	size_t fallocHeapLength;
	size_t freeChunksLength;					// Size of circular buffer (set up by host)
	fallocHeapChunk** freeChunks;				// Start of circular buffer (set up by host)
	volatile fallocHeapChunk** freeChunksPtr;	// Current atomically-incremented non-wrapped offset
	void* reserved;
} fallocHeap;


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE

const static int FALLOCNODE_SLACK = 0x10;

typedef struct _cuFallocNode {
	struct _cuFallocNode* next;
	struct _cuFallocNode* nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} fallocNode;

typedef struct _cuFallocContext {
	fallocNode node;
	fallocNode* nodes;
	fallocNode* availableNodes;
	fallocHeap* heap;
} fallocContext;

// All our headers are prefixed with a magic number so we know they're ready
#define FALLOC_MAGIC (unsigned short)0x3412        // Not a valid ascii character
#define FALLOCNODE_MAGIC (unsigned short)0x7856

__device__ void fallocInit(fallocHeap* heap) {
	if (threadIdx.x || threadIdx.y || threadIdx.z)
		return;
	size_t chunks = heap->chunks;
	if (!chunks)
		__THROW;
	fallocHeapChunk** freechunks = heap->freeChunks;
	size_t chunkSize = heap->chunkSize;
	// preset all chunks
	fallocHeapChunk* chunk = (fallocHeapChunk*)((__int8*)heap + heap->fallocHeapLength);
	chunk->magic = FALLOC_MAGIC;
	chunk->count = 1;
	chunk->reserved = nullptr;
	while (chunks-- > 1) {
		chunk = *freechunks++ = chunk->next = (fallocHeapChunk*)((__int8*)chunk + chunkSize);
		chunk->magic = FALLOC_MAGIC;
		chunk->count = 1;
		chunk->reserved = nullptr;
	}
	chunk->next = nullptr;
	heap->freeChunksPtr = freechunks;
}

__device__ void* fallocGetChunk(fallocHeap* heap) {
	if (threadIdx.x || threadIdx.y || threadIdx.z)
		__THROW;
	volatile fallocHeapChunk* chunk = heap->freeChunks;
	if (!chunk)
		return nullptr;
	{ // critical
		heap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(fallocHeapChunk));
}

__device__ void* fallocGetChunks(fallocHeap* heap, size_t length, size_t* allocLength = nullptr) {
	size_t chunkSize = heap->chunkSize;
	// fix up length to be a multiple of chunkSize
	length = (length < chunkSize ? chunkSize : length);
	if (length % chunkSize)
		length += chunkSize - (length % chunkSize);
	// set length, if requested
	if (allocLength)
		*allocLength = length - sizeof(fallocHeapChunk);
	size_t chunks = (size_t)(length / chunkSize);
	if (chunks > heap->chunks)
		__THROW;
	// single, equals: fallocGetChunk
	if (chunks == 1)
		return fallocGetChunk(heap);
	// multiple, find a contiguous chuck
	size_t index = chunks;
	volatile fallocHeapChunk* chunk;
	volatile fallocHeapChunk* endChunk = (fallocHeapChunk*)((__int8*)heap + sizeof(fallocHeap) + (chunkSize * heap->chunks));
	{ // critical
		for (chunk = (fallocHeapChunk*)((__int8*)heap + sizeof(fallocHeap)); index && chunk < endChunk; chunk = (fallocHeapChunk*)((__int8*)chunk + (chunkSize * chunk->count))) {
			if (chunk->magic != FALLOC_MAGIC)
				__THROW;
			index = (chunk->next ? index - 1 : chunks);
		}
		if (index)
			return nullptr;
		// found chuck, remove from freeChunks
		endChunk = chunk;
		chunk = (fallocHeapChunk*)((__int8*)chunk - (chunkSize * chunks));
		for (volatile fallocHeapChunk* chunk2 = heap->freeChunks; chunk2; chunk2 = chunk2->next)
			if (chunk2 >= chunk && chunk2 <= endChunk)
				chunk2->next = (chunk2->next ? chunk2->next->next : nullptr);
		chunk->count = chunks;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(fallocHeapChunk));
}

__device__ void fallocFreeChunk(fallocHeap* heap, void* obj) {
	if (threadIdx.x || threadIdx.y || threadIdx.z)
		__THROW;
	volatile fallocHeapChunk* chunk = (fallocHeapChunk*)((__int8*)obj - sizeof(fallocHeapChunk));
	if (chunk->magic != FALLOC_MAGIC || chunk->count > 1)
		__THROW;
	{ // critical
		chunk->next = heap->freeChunks;
		heap->freeChunks = chunk;
	}
}

__device__ void fallocFreeChunks(fallocHeap* heap, void* obj) {
	volatile fallocHeapChunk* chunk = (fallocHeapChunk*)((__int8*)obj - sizeof(fallocHeapChunk));
	if (chunk->magic != FALLOC_MAGIC)
		__THROW;
	size_t chunks = chunk->count;
	// single, equals: fallocFreeChunk
	if (chunks == 1) {
		{ // critical
			chunk->next = heap->freeChunks;
			heap->freeChunks = chunk;
		}
		return;
	}
	// retag chunks
	size_t chunkSize = heap->chunkSize;
	chunk->count = 1;
	while (chunks-- > 1) {
		chunk = chunk->next = (fallocHeapChunk*)((__int8*)chunk + sizeof(fallocHeapChunk) + chunkSize);
		chunk->magic = FALLOC_MAGIC;
		chunk->count = 1;
		chunk->reserved = nullptr;
	}
	{ // critical
		chunk->next = heap->freeChunks;
		heap->freeChunks = chunk;
	}
}


//////////////////////
// ALLOC

__device__ static fallocContext* fallocCreateCtx(fallocHeap* heap) {
	size_t chunkSize = heap->chunkSize;
	if (sizeof(fallocContext) > chunkSize)
		__THROW;
	fallocContext* ctx = (fallocContext*)fallocGetChunk(heap);
	if (!ctx)
		return nullptr;
	ctx->heap = heap;
	unsigned short freeOffset = ctx->node.freeOffset = sizeof(fallocContext);
	ctx->node.magic = FALLOCNODE_MAGIC;
	ctx->node.next = nullptr; ctx->nodes = (fallocNode*)ctx;
	ctx->node.nextAvailable = nullptr; ctx->availableNodes = (fallocNode*)ctx;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > chunkSize)
		ctx->availableNodes = nullptr;
	return ctx;
}

__device__ static void fallocDisposeCtx(fallocContext* ctx) {
	fallocHeap* heap = ctx->heap;
	for (fallocNode* node = ctx->nodes; node; node = node->next)
		fallocFreeChunk(heap, node);
}

__device__ static void* falloc(fallocContext* ctx, unsigned short bytes, bool alloc) {
	if (bytes > (HEAPCHUNK_SIZE - sizeof(fallocContext)))
		__THROW;
	// find or add available node
	fallocNode* node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	fallocNode* lastNode;
	for (lastNode = (fallocNode*)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			break;
	if (!node || !hasFreeSpace) {
		// add node
		node = (fallocNode*)fallocGetChunk(ctx->heap);
		if (!node)
			__THROW;
		freeOffset = node->freeOffset = sizeof(fallocNode); 
		freeOffset += bytes;
		node->magic = FALLOCNODE_MAGIC;
		node->next = ctx->nodes; ctx->nodes = node;
		node->nextAvailable = (alloc ? ctx->availableNodes : nullptr); ctx->availableNodes = node;
	}
	//
	void* obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if (alloc && ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)) {
		if (lastNode == (fallocNode*)ctx)
			ctx->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
}

__device__ static void* fallocRetract(fallocContext* ctx, unsigned short bytes) {
	fallocNode* node = ctx->availableNodes;
	int freeOffset = (int)node->freeOffset - bytes;
	// multi node, retract node
	if (node != &ctx->node && freeOffset < sizeof(fallocNode)) {
		node->freeOffset = sizeof(fallocNode);
		// search for previous node
		fallocNode* lastNode;
		for (lastNode = (fallocNode*)ctx, node = ctx->nodes; node; lastNode = node, node = node->next)
			if (node == ctx->availableNodes)
				break;
		node = ctx->availableNodes = lastNode;
		freeOffset = (int)node->freeOffset - bytes;
	}
	// first node && !overflow
	if (node == &ctx->node && freeOffset < sizeof(fallocContext))
		__THROW;
	node->freeOffset = (unsigned short)freeOffset;
	return (__int8*)node + freeOffset;
}

__device__ static void fallocMark(fallocContext* ctx, void* &mark, unsigned short &mark2) { mark = ctx->availableNodes; mark2 = ctx->availableNodes->freeOffset; }
__device__ static bool fallocAtMark(fallocContext* ctx, void* mark, unsigned short mark2) { return (mark == ctx->availableNodes && mark2 == ctx->availableNodes->freeOffset); }
