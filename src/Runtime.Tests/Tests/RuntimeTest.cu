#include <Runtime.cu.h>

#define _TEST(id) \
	__global__ void runtimeTest##id(void *r); \
	void runtimeTest##id##_host(cudaDeviceHeap &r) { cudaDeviceHeapSelect(r); runtimeTest##id<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); } \
	__global__ void runtimeTest##id(void *r) \
{ \
	_runtimeSetHeap(r);

//////////////////////////////////////////////////

// printf outputs
_TEST(0) {
	_printf("test");
}}
