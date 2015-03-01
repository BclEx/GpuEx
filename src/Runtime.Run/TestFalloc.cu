#include <Falloc.cu.h>

// FALLOC
__global__ static void falloc0(void *f)
{
	void *b = fallocGetBlock((fallocHeap *)f);
	fallocFreeBlock((fallocHeap *)f, b);
}

void __testFalloc(cudaDeviceFalloc &f)
{
	falloc0<<<1, 1>>>(f.heap);
}