#pragma warning(disable: 4996)
#ifndef _LIB

#include "RuntimeHost.h"
#include "RuntimeOS.h"
#include "RuntimeSentinel.h"
#include <stdio.h>

//void __testRuntime(cudaDeviceHeap &r);

int main(int argc, char **argv)
{
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));

	InitializeSentinel();
	//osFindSentinel();

	//cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);

#if VISUAL
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender* render = new RuntimeVisualRender(deviceHeap);
	//if (!Visual::InitGL(render, &argc, argv))
	//	return 0;
	//cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	//Visual::Main();
	//Visual::Dispose();
	//cudaErrorCheck(cudaDeviceHeapSynchronize(deviceHeap));
#endif

	//cudaDeviceHeapSelect(deviceHeap);
	//__testRuntime(deviceHeap);

	ShutdownSentinel();

	//cudaDeviceHeapDestroy(deviceHeap);
	return 0;
}

#endif