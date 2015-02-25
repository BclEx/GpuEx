#ifndef __RUNTIME_H__
#define __RUNTIME_H__
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code
#pragma region HOST SIDE

typedef void (*cudaAssertHandler)();

typedef struct
{
	void *reserved;
	void *heap;
	char *blocks;
	char *blockStart;
	size_t blockSize;
	size_t blocksLength;
	size_t length;
	cudaAssertHandler assertHandler;
} cudaDeviceHeap;

//
//	cudaDeviceHeapSelect
//
extern "C" cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host);

//
//	cudaDeviceHeapCreate
//
//	Call this to initialize a runtime heap. If the buffer size needs to be changed, call cudaDeviceHeapDestroy()
//	before re-calling cudaDeviceHeapCreate().
//
//	The default size for the buffer is 1 megabyte. The buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		length - Length, in bytes, of total space to reserve (in device global memory) for output.
//
// default 2k blocks, 1-meg heap
extern "C" cudaDeviceHeap cudaDeviceHeapCreate(size_t blockSize = 256, size_t length = 1048576, cudaError_t *error = nullptr, void *reserved = nullptr);

//
//	cudaDeviceHeapDestroy
//
//	Cleans up all memory allocated by cudaDeviceHeapCreate() for a heap.
//	Call this at exit, or before calling cudaDeviceHeapCreate() again.
//
extern "C" void cudaDeviceHeapDestroy(cudaDeviceHeap &host);

//	cudaDeviceHeapSetHandler
//
//	Sets runtime handler for assert
//
extern "C" void cudaDeviceHeapSetHandler(cudaDeviceHeap &host, cudaAssertHandler handler);

//
//	cudaDeviceHeapSynchronize
//
//	Dumps all printfs in the output buffer to the specified file pointer. If the output pointer is not specified,
//	the default "stdout" is used.
//
//	Arguments:
//		output       - A file pointer to an output stream.
//		showThreadID - If "true", output strings are prefixed by "[blockid, threadid] " at output.
//
//	Returns:
//		cudaSuccess if all is well.
//
extern "C" cudaError_t cudaDeviceHeapSynchronize(cudaDeviceHeap &host, void *stream = nullptr, bool showThreadID = true);

#pragma endregion


///////////////////////////////////////////////////////////////////////////////
// EX
// Extra methods for host-side code
#pragma region EX
extern cudaError __cudaLastError;
#define cudaCheckErrors(action, failure) if ((__cudaLastError = action) != cudaSuccess) { printf("CudaError: %s \"%s\"\n", #action, cudaGetErrorString(__cudaLastError)); failure; }

inline int __convertSMVer2Cores(int major, int minor)
{
	typedef struct // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} SMToCores;
	SMToCores gpuArchCoresPerSM[] = {
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};
	int index = 0;
	while (gpuArchCoresPerSM[index].SM != -1)
	{
		if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor))
			return gpuArchCoresPerSM[index].Cores;
		index++;
	}
	//printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

inline int gpuGetMaxGflopsDeviceId()
{
	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	// Find the best major SM Architecture GPU device
	int bestMajor = 0;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited && deviceProp.major > 0 && deviceProp.major < 9999)
			bestMajor = (bestMajor > deviceProp.major ? bestMajor : deviceProp.major);
	}
	// Find the best CUDA capable GPU device
	int bestDevice = 0;
	int basePerformace = 0;
	for (int i = 0; i < deviceCount; i++ )
	{
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			int sm_per_multiproc = (deviceProp.major == 9999 && deviceProp.minor == 9999 ? 1 : __convertSMVer2Cores(deviceProp.major, deviceProp.minor));
			int performace = (deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate);
			if (performace > basePerformace)
			{
				basePerformace = performace;
				bestDevice = i;
			}
		}
	}
	return bestDevice;
}

#pragma endregion


///////////////////////////////////////////////////////////////////////////////
// VISUAL
// Visual render for host-side code
#pragma region VISUAL
#ifdef VISUAL
#include "VisualHost.h"

class RuntimeVisualRender : public IVisualRender
{
private:
	cudaDeviceHeap _runtimeHost;
public:
	RuntimeVisualRender(cudaDeviceHeap runtimeHost)
		: _runtimeHost(runtimeHost) { }
	virtual void Dispose();
	virtual void Keyboard(unsigned char key);
	virtual void Display();
	virtual void Initialize();
};

#endif
#pragma endregion

#endif // __RUNTIME_H__
