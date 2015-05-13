#include <stdio.h>
#include <string.h>
#include <RuntimeHost.h>
using namespace System;
using namespace Xunit;

extern void ioTest0_host(cudaDeviceHeap &r);

namespace Tests
{
	static cudaDeviceHeap _deviceHeap;

	public ref class IOTest
	{
	public:
		IOTest()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			//checkCudaErrors(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
			_deviceHeap = cudaDeviceHeapCreate(256, 4096); cudaDeviceHeapSelect(_deviceHeap);
		}
		~IOTest()
		{
			//static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			//FILE *f = fopen(path, "w");
			//cudaRuntimeExecute(f, true);
			cudaDeviceHeapDestroy(_deviceHeap);
			//fclose(f);
			//checkCudaErrors(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!"));
		}

		[Fact]
		void printf_outputs() { ioTest0_host(_deviceHeap); }
	};
}
