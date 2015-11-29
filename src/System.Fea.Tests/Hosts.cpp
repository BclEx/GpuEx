#include <stdio.h>
#include <string.h>
#include <Runtime.h>
using namespace System;
using namespace Xunit;

#pragma region TESTCLASS
#define TESTCLASS(name, body) \
	public ref class name##Tests \
{ \
public: \
	name##Tests() \
{ \
	cudaErrorCheck2(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?")); \
	_deviceHeap = cudaDeviceHeapCreate(256, 4096); \
	cudaErrorCheck2(cudaDeviceHeapSelect(_deviceHeap), throw gcnew Exception("cudaDeviceHeapSelect failed!")); \
} \
	~name##Tests() \
{ \
	/*static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;*/ \
	/*FILE *f = fopen(path, "w");*/ \
	cudaDeviceHeapSynchronize(_deviceHeap); \
	/*fclose(f);*/ \
	cudaErrorCheck2(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!")); \
} \
	body \
};

#define FACT(name) \
	[Fact] \
	void name() { extern void name##_host(cudaDeviceHeap &r); name##_host(_deviceHeap); }
#pragma endregion

namespace Tests
{
	static cudaDeviceHeap _deviceHeap;

	TESTCLASS(select2,
		FACT(select2)
	)
}
