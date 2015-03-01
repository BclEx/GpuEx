#ifndef _LIB

#define VISUAL
#include "..\System.Data.net\Core+Vdbe\Core+Vdbe.cu.h"
#include <stdio.h>
#include <string.h>

__device__ static bool MyCallback(void *args, int argsLength, char **args2, char **cols)
{
	return true;
}

__device__ static void TestDB()
{
	//SysEx_LOG(RC_OK, "START\n");
	//ParserTrace(stderr, "p: ");

	// open
	Context *ctx;
	Main::Open("C:\\T_\\Test2.db", &ctx);

	// run query
	char *errMsg = nullptr;
	//Main::Exec(ctx, "Select * From MyTable;", MyCallback, nullptr, &errMsg);
	Main::Exec(ctx, "PRAGMA database_list;", MyCallback, nullptr, &errMsg);
	if (errMsg)
	{
		_printf("Error: %s\n", errMsg);
		_free(errMsg);
	}

	// close
	Main::Close(ctx);
}

//__device__ static void TestVFS()
//{
//	auto vfs = VSystem::FindVfs(nullptr);
//	auto file = (VFile *)_alloc(vfs->SizeOsFile);
//	auto rc = vfs->Open("C:\\T_\\Test.db", file, VSystem::OPEN_CREATE | VSystem::OPEN_READWRITE | VSystem::OPEN_MAIN_DB, nullptr);
//	file->Write4(0, 123145);
//	file->Close();
//}

//namespace Core { int Bitvec_BuiltinTest(int size, int *ops); }
//void TestBitvec()
//{
//	int ops[] = { 5, 1, 1, 1, 0 };
//	Core::Bitvec_BuiltinTest(400, ops);
//}

#if __CUDACC__
__global__ void GMain(void *r) { _runtimeSetHeap(r);
#else
__global__ void main(int argc, char **argv) {
#endif
	Main::Initialize();
	//
	//TestVFS();
	TestDB();
	//
	Main::Shutdown();
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	
}

int main(int argc, char **argv)
{
	cudaDeviceHeap runtimeHost = cudaDeviceHeapCreate(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(runtimeHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// test
	__main(runtimeHost);
	//cudaRuntimeExecute(runtimeHost);

	// run
	//Visual::Main();

	cudaDeviceHeapDestroy(runtimeHost);

	cudaDeviceReset();
	//printf("End.");
	//char c; scanf("%c", &c);
	return 0;
}
#endif

#endif