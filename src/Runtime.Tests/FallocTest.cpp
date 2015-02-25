#include <stdio.h>
#include <string.h>
using namespace System;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

#define TRACE
#include <cuda_runtime.h>
#include <System.cu\cuFalloc.h>
#include <System.cu\cuPrintf.h>

void _testLinkedList(fallocHeap* heap);
char* _path = "C:\\T_\\test.txt";

namespace Tests
{
	public ref class FallocTest
	{
	public:
		TestContext^ Ctx;

		void TestInitialize()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			/*cudaError_t cudaStatus = cudaSetDevice(0);
			if (cudaStatus != cudaSuccess)
				throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");*/
			
			_fallocHost = cudaFallocWTraceInit(2048);
			cudaPrintfInit(256000);
		}

		void TestCleanup()
		{	
			static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			FILE *f = fopen(path, "w");
			cudaPrintfDisplay(f, true); cudaPrintfEnd();
			fclose(f);
			cudaFallocWTraceEnd(_fallocHost);

			//cudaError_t cudaStatus = cudaDeviceReset();
			//if (cudaStatus != cudaSuccess)
			//	throw gcnew Exception("cudaDeviceReset failed!");
		}

		[Fact]
		void cuFallocInit_Valid()
		{
			Assert::AreNotSame(nullptr, (int)_fallocHost.heap);
			Assert::AreNotSame(0, _fallocHost.length);
		}

		[Fact]
		void cuFallocInit_Valid2()
		{
			_testLinkedList(_fallocHost.heap);
		}


		//[TestMethod]
		//void x()
		//{
		//	void* obj = fallocGetChunk(heap);
		//	fallocFreeChunk(heap, obj);
		//}
	};
}
