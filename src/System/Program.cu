#ifndef _LIB

//#define VISUAL
#include "..\System.net\Core\Core.cu.h"
#include <string.h>

#pragma region TESTS

__device__ static void TestVFS()
{
	auto vfs = VSystem::FindVfs(nullptr);
	auto file = (VFile *)_alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, VSystem::OPEN_CREATE | VSystem::OPEN_READWRITE | VSystem::OPEN_MAIN_DB, nullptr);
	file->Write4(0, 123145);
	file->Close();
}

//void TestBitvec()
//{
//	int ops[] = { 5, 1, 1, 1, 0 };
//	Core::Bitvec_BuiltinTest(400, ops);
//}

__global__ static void Tests(void *r)
{
	_runtimeSetHeap(r);
	MutexEx masterMutex;
	RC rc = SysEx::PreInitialize(masterMutex);
	SysEx::PostInitialize(masterMutex);
	//
	TestVFS();
	//
	SysEx::Shutdown();
}

#if __CUDACC__
void __tests(cudaDeviceHeap &r)
{
	Tests<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __tests(cudaDeviceHeap &r)
{
	Tests(r.heap);
}
#endif

#pragma endregion

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
	__tests(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(runtimeHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// run
	__main(deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(deviceHeap);

	cudaDeviceReset();
	//printf("End.");
	//char c; scanf("%c", &c);
	return 0;
}
#endif

#endif