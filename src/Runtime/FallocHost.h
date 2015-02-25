#ifndef __FALLOC_H__
#define __FALLOC_H__
#include <cuda_runtime.h>

/*
*	This is the header file supporting cuFalloc.cu and defining both the host and device-side interfaces. See that file for some more
*	explanation and sample use code. See also below for details of the host-side interfaces.
*
*  Quick sample code:
*
#include "cuFalloc.cu"

__global__ void TestFalloc(fallocHeap* heap)
{
// create/free heap
void* obj = fallocGetChunk(heap);
fallocFreeChunk(heap, obj);

// create/free alloc
fallocContext* ctx = fallocCreateCtx(heap);
char* testString = (char*)falloc(ctx, 10);
int* testInteger = falloc<int>(ctx);
fallocDisposeCtx(ctx);
}

int main()
{
cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(1);

// test
TestFalloc<<<1, 1>>>(fallocHost.heap);

// free and exit
cudaDeviceFallocDestroy(fallocHost);
printf("\ndone.\n"); // char c; scanf("%c", &c);
return 0;
}
*/

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code
#pragma region HOST SIDE

typedef struct
{
	void *reserved;
	void *heap;
	size_t blockSize;
	size_t blocksLength;
	size_t length;
} cudaDeviceFalloc;

//
//	cudaFallocSetHeap
//
extern "C" cudaError_t cudaFallocSetHeap(void *heap);

//
//	cudaDeviceFallocCreate
//
//	Call this to initialize a falloc heap. If the buffer size needs to be changed, call cudaDeviceFallocDestroy()
//	before re-calling cudaDeviceFallocCreate().
//
//	The default size for the buffer is 1 megabyte. The buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		length - Length, in bytes, of total space to reserve (in device global memory) for output.
//
//	Returns:
//		cudaDeviceFalloc if all is well.
//
// default 2k blocks, 1-meg heap
extern "C" cudaDeviceFalloc cudaDeviceFallocCreate(size_t blockSize = 2046, size_t length = 1048576, cudaError_t *error = nullptr, void *reserved = nullptr);

//
//	cudaDeviceFallocDestroy
//
//	Cleans up all memories allocated by cudaDeviceFallocCreate() for a heap.
//	Call this at exit, or before calling cudaDeviceFallocCreate() again.
//
extern "C" void cudaDeviceFallocDestroy(cudaDeviceFalloc &host);
#pragma endregion


///////////////////////////////////////////////////////////////////////////////
// VISUAL
// Visual render for host-side code
#pragma region VISUAL
#ifdef VISUAL
#include "VisualHost.h"

class FallocVisualRender : public IVisualRender
{
private:
	cudaDeviceFalloc _fallocHost;
public:
	FallocVisualRender(cudaDeviceFalloc fallocHost)
		: _fallocHost(fallocHost) { }
	virtual void Dispose();
	virtual void Keyboard(unsigned char key);
	virtual void Display();
	virtual void Initialize();
};

#endif
#pragma endregion

#endif // __FALLOC_H__