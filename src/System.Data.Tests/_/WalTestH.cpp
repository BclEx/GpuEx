#include <stdio.h>
#include <string.h>
#include <Runtime.h>
using namespace System;
using namespace Xunit;

extern void walTest0_host(cudaRuntimeHost &r);

namespace Tests
{
	static cudaRuntimeHost _runtimeHost;

	public ref class WalTest
	{
	public:
		WalTest()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			//checkCudaErrors(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
			_runtimeHost = cudaRuntimeInit(256, 4096);
			cudaRuntimeSetHeap(_runtimeHost.heap);
		}
		~WalTest()
		{
			//static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			//FILE *f = fopen(path, "w");
			//cudaRuntimeExecute(f, true);
			cudaRuntimeEnd(_runtimeHost);
			//fclose(f);
			//checkCudaErrors(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!"));
		}

		[Fact]
		void first() { walTest0_host(_runtimeHost); }
	};
}
