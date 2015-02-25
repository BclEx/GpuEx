#include <Falloc.cu.h>

__global__ static void fallocExample(void *f)
{
	void *b = fallocGetBlock((fallocHeap *)f);
	fallocFreeBlock((fallocHeap *)f, b);
}

void __fallocExample(cudaDeviceFalloc &f)
{
	fallocExample<<<1, 1>>>(f.heap);
}