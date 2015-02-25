#include <stdio.h>
#include <string.h>
#include <RuntimeHost.h>
#include <FallocHost.h>
using namespace System;
using namespace Xunit;

extern void fallocTestA_host(cudaDeviceHeap &r, void *h);
extern void fallocTest0_host(cudaDeviceHeap &r, void *h);
extern void fallocTest1_host(cudaDeviceHeap &r, void *h);
extern void fallocTest2_host(cudaDeviceHeap &r, void *h);
extern void fallocTest3_host(cudaDeviceHeap &r, void *h);

namespace Tests
{
	static cudaDeviceHeap _deviceHeap;
	static cudaDeviceFalloc _deviceFalloc;

	public ref class FallocITest
	{
	public:
		void AssertHandler()
		{
			throw gcnew Exception();
		}

		FallocITest()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			//checkCudaErrors(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
			_deviceHeap = cudaDeviceHeapCreate(256, 4096);
			//cudaRuntimeSetHandler(&Tests::FallocITest::AssertHandler);
			TestInitialize();
		}
		~FallocITest()
		{
			TestCleanup();
			//static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			//FILE *f = fopen(path, "w");
			//cudaRuntimeExecute(f, true);
			cudaDeviceHeapDestroy(_deviceHeap);
			//fclose(f);
			//checkCudaErrors(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!"));
		}

		void TestInitialize() { _deviceFalloc = cudaDeviceFallocCreate(1024, 4098); }
		void TestCleanup() { cudaDeviceFallocDestroy(_deviceFalloc); }

		[Fact]
		void Initialize_Returns_FallocHost()
		{
			Assert::True(_deviceFalloc.heap != nullptr);
		}

		[Fact]
		void lauched_cuda_kernel()
		{
			fallocTestA_host(_deviceHeap, _deviceFalloc.heap);
			Assert::Equal(1, 1);
		}

		[Fact]
		void alloc_with_getblock() { fallocTest0_host(_deviceHeap, _deviceFalloc.heap); }

		[Fact]
		void alloc_with_getblocks() { fallocTest1_host(_deviceHeap, _deviceFalloc.heap); }

		[Fact]
		void alloc_with_context() { fallocTest2_host(_deviceHeap, _deviceFalloc.heap); }

		[Fact]
		void alloc_with_context_as_stack() { fallocTest3_host(_deviceHeap, _deviceFalloc.heap); }
	};
}
