#ifndef _LIB

#define VISUAL
#include "..\SystemL.net\Core\Core.cu.h"
#include <stdio.h>
#include <string.h>
using namespace Core;
using namespace Core::IO;

static void TestVFS();

#if __CUDACC__
void __main(cudaRuntimeHost &r);
void main(int argc, char **argv)
{
	cudaRuntimeHost runtimeHost = cudaRuntimeInit(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender* render = new RuntimeVisualRender(runtimeHost); if (!Visual::InitGL(render, &argc, argv)) return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// test
	__main(runtimeHost);
	cudaRuntimeExecute(runtimeHost);

	// run
	//Visual::Main(); atexit(Visual::Dispose);

	cudaRuntimeEnd(runtimeHost);

	cudaDeviceReset();
	//printf("End.");
	//char c; scanf("%c", &c);
}

void __main(cudaRuntimeHost &r)
{	
	cudaRuntimeSetHeap(r.heap);
	MainTest<<<1, 1>>>(r.heap);
}

__global__ void main(int argc, char **argv)
#else
__global__ void main(int argc, char **argv)
#endif
{
	MutexEx masterMutex;
	RC rc = SysEx::PreInitialize(masterMutex);
	SysEx::PostInitialize(masterMutex);
	//
	TestVFS();
	//
	SysEx::Shutdown();
}

__device__ static void TestVFS()
{
	auto vfs = VSystem::FindVfs("win32");
	auto file = (VFile *)_alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, VSystem::OPEN_CREATE | VSystem::OPEN_READWRITE | VSystem::OPEN_MAIN_DB, nullptr);
	file->Write4(0, 123145);
	file->Close();
}

//namespace Core { int Bitvec_BuiltinTest(int size, int *ops); }
//void TestBitvec()
//{
//	int ops[] = { 5, 1, 1, 1, 0 };
//	Core::Bitvec_BuiltinTest(400, ops);
//}

#endif