#pragma warning(disable: 4996)
#ifndef _LIB
#include "RuntimeHost.h"
#include "FallocHost.h"
#include <stdio.h>

void main(int argc, char **argv)
{
	cudaDeviceHeap runtimeHost = cudaDeviceHeapCreate(); //256, 4096);
	cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(100, 1024);

#if VISUAL
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender* render = new RuntimeVisualRender(runtimeHost);
	//IVisualRender* render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv))
		return;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	Visual::Dispose();
#endif
	cudaCheckErrors(cudaDeviceHeapSynchronize(runtimeHost), 0);

	cudaDeviceHeapDestroy(runtimeHost);
	cudaDeviceFallocDestroy(fallocHost);
}

#endif