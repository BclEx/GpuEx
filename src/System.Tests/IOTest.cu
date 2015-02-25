#include <Core\Core.cu.h>

#pragma region Preamble

#if __CUDACC__
#define TEST(id) \
	__global__ void ioTest##id(void *r); \
	void ioTest##id##_host(cudaDeviceHeap &r) { ioTest##id<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); } \
	__global__ void ioTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();
#else
#define TEST(id) \
	__global__ void ioTest##id(void *r); \
	void ioTest##id##_host(cudaDeviceHeap &r) { ioTest##id(r.heap); cudaDeviceHeapSynchronize(r); } \
	__global__ void ioTest##id(void *r) \
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

// printf outputs
TEST(1) {
	auto vfs = VSystem::FindVfs("gpu");
	auto file = (VFile *)_alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, (VSystem::OPEN)((int)VSystem::OPEN_CREATE | (int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_MAIN_DB), nullptr);
	_printf("%d\n", rc);
	file->Write4(0, 123145);
	file->Close();
	_free(file);
}}
