#include "Runtime.h"
#include "RuntimeOS.h"
//#include <stdio.h>
//#include <stdlib.h>
//
//#define SENTINEL_VERSION (unsigned char)1
//#define SENTINEL_FILE ".gpu"
//#define SENTINEL_MAGIC (unsigned short)0xC811
//Sentinel _s;
//__host__ void osCreateSentinel(int fileSize)
//{
//	char b[1024];
//	Sentinel h_s;
//	//: Find file
//	_snprintf(b, sizeof(b), "%s\\%s", getenv("APPDATA"), SENTINEL_FILE);
//	FILE *f = fopen(b, "r");
//	//: if (file && magic/version check)
//	if (f && fread(&_s, sizeof(_s), 1, f) && _s.Magic == SENTINEL_MAGIC && _s.Version == SENTINEL_VERSION)
//	{
//		//: pull memory location
//		cudaError rc = cudaMemcpy(&h_s, _s.D_, sizeof(_s), cudaMemcpyDeviceToHost);
//		if (rc != cudaSuccess) 
//			fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(rc));
//		//: if (magic/version check)
//		if (!rc && h_s.Magic == SENTINEL_MAGIC && h_s.Version == SENTINEL_VERSION)
//		{
//			//: return
//			fclose(f);
//			return;
//		}
//	}
//	//: create file
//	f = (f ? freopen(b, "w+", f) : fopen(b, "w+"));
//	if (!f)
//	{
//		printf("unable to open/create sentinel file: %s\n", SENTINEL_FILE);
//		exit(1);
//	}
//	//: initalize
//	memset(&_s, 0, sizeof(_s));
//	_s.Magic = SENTINEL_MAGIC;
//	_s.Version = SENTINEL_VERSION;
//	//: write memory
//	cudaErrorCheck(cudaMalloc((void**)&_s.D_, sizeof(_s)));
//	cudaErrorCheck(cudaMemcpy(_s.D_, &_s, sizeof(_s), cudaMemcpyHostToDevice));
//	//: write file
//	fwrite(&_s, sizeof(_s), 1, f);
//	fflush(f);
//	fclose(f);
//}
//
//__host__ void oReleaseSentinel()
//{
//}

__device__ void osSleep(DWORD milliseconds)
{
#if __CUDACC__
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;)
	{
		clock_t now = clock();
		clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
		if (cycles >= end) break;
	}
#endif
}

__device__ DWORD osGetLastError()
{
	return 0;	
}

__device__ DWORD osWaitForSingleObject(HANDLE hHandle, DWORD dwMilliseconds)
{
	return 0;
}

__device__ HANDLE osCreateMutexA(void *dummy1, bool dummy2, const char *name)
{
	return 0;
}

__device__ void osReleaseMutex(HANDLE h)
{
}