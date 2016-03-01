#pragma warning(disable: 4996)
#ifndef _LIB

#include <stdio.h>
#include "RuntimeHost.h"
#include "..\Runtime.Tests\TestRuntime.cu"

cudaDeviceHeap _deviceHeap;
int main(int argc, char **argv)
{
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));

	_deviceHeap = cudaDeviceHeapCreate(256, 100);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender* render = new RuntimeVisualRender(deviceHeap);
	//if (!Visual::InitGL(render, &argc, argv))
	//	return 0;
	//cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	//Visual::DataEx();
	//Visual::Dispose();
	//cudaErrorCheck(cudaDeviceHeapSynchronize(deviceHeap));

	RuntimeSentinel::Initialize();

	//cudaDeviceHeapSelect(_deviceHeap);
	__testRuntime(_deviceHeap);

	RuntimeSentinel::Shutdown();

	cudaDeviceHeapDestroy(_deviceHeap);
	return 0;
}

#endif