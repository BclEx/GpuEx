#ifndef __FALLOC_CU_H__
#define __FALLOC_CU_H__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#include "FallocHost.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

#if defined(__EMBED__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#ifndef __EMBED__
#define __static__ static
#endif
#include "Falloc.cu+native.h"
#else

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE

// Heap
extern "C" __device__ void _fallocSetHeap(void *heap);

// External function definitions for device-side code
typedef struct _cuFallocHeap fallocHeap;
extern "C" __device__ void *fallocGetBlock(fallocHeap *heap);
extern "C" __device__ void fallocFreeBlock(fallocHeap *heap, void *obj);
#if MULTIBLOCK
extern "C" __device__ void *fallocGetBlocks(fallocHeap *heap, size_t length, size_t *allocLength = nullptr);
extern "C" __device__ void fallocFreeBlocks(fallocHeap *heap, void *obj);
#endif

// CONTEXT
typedef struct _cuFallocContext fallocCtx;
extern "C" __device__ fallocCtx *fallocCreateCtx(fallocHeap *heap);
extern "C" __device__ void fallocDisposeCtx(fallocCtx *ctx);
extern "C" __device__ void *falloc(fallocCtx *ctx, unsigned short bytes, bool alloc = true);
extern "C" __device__ void *fallocRetract(fallocCtx *ctx, unsigned short bytes);
extern "C" __inline__ __device__ void fallocMark(fallocCtx *ctx, void *&mark, unsigned short &mark2);
extern "C" __inline__ __device__ bool fallocAtMark(fallocCtx *ctx, void *mark, unsigned short mark2);

#endif // __CUDA_ARCH__

// CONTEXT
template <typename T> __inline__ __device__ T* falloc(fallocCtx *ctx) { return (T *)falloc(ctx, sizeof(T), true); }
template <typename T> __inline__ __device__ void fallocPush(fallocCtx *ctx, T t) { *((T *)falloc(ctx, sizeof(T), false)) = t; }
template <typename T> __inline__ __device__ T fallocPop(fallocCtx *ctx) { return *((T *)fallocRetract(ctx, sizeof(T))); }

#endif // __FALLOC_CU_H__