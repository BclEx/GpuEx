#ifndef _LIB

//#define VISUAL
#include "..\System.net\Core\Core.cu.h"
#include <string.h>

#pragma region TestSystem0

__device__ static void TestVFS()
{
	auto vfs = VSystem::FindVfs(nullptr);
	auto file = (VFile *)_allocZero(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, (VSystem::OPEN)(VSystem::OPEN_CREATE|VSystem::OPEN_READWRITE|VSystem::OPEN_MAIN_DB), nullptr);
	file->Write4(0, 123145);
	file->Close();
}

// NATIVE: assert
__global__ static void testSystem0(void *r)
{
	_runtimeSetHeap(r);
	MutexEx masterMutex;
	RC rc = SysEx::Initialize(masterMutex);
	//
	TestVFS();
	//
	SysEx::Shutdown();
	printf("System: 0\n");
}

#if __CUDACC__
void __testSystem(cudaDeviceHeap &r)
{
	testSystem0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystem(cudaDeviceHeap &r)
{
	testSystem0(r.heap);
}
#endif

#pragma endregion

#pragma region Main

//void __testSystem(cudaDeviceHeap &r);
#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
__testSystem(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

cudaDeviceHeap _deviceHeap
int main(int argc, char **argv)
{
	//cudaThreadSetLimit(cudaLimitStackSize, 1024*6);
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));

	_deviceHeap = cudaDeviceHeapCreate(256, 4096);
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(runtimeHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

	// run
	__main(_deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(_deviceHeap);

	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}
#endif

#pragma endregion

#endif