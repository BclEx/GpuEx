#include <Core\Core.cu.h>

#pragma region Preamble

#if __CUDACC__
#define TEST(id) \
	__global__ void pagerTest##id(void *r); \
	void pagerTest##id##_host(cudaRuntimeHost &r) { pagerTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void pagerTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();
#else
#define TEST(id) \
	__global__ void pagerTest##id(void *r); \
	void pagerTest##id##_host(cudaRuntimeHost &r) { pagerTest##id(r.heap); cudaRuntimeExecute(r); } \
	__global__ void pagerTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();
#endif

#pragma endregion

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}
