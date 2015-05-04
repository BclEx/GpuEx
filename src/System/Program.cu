#ifndef _LIB

//#define VISUAL
#include "..\System.net\Core\Core.cu.h"
#include <string.h>

//void __testSystem(cudaDeviceHeap &r);

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
__testSystem(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	//cudaThreadSetLimit(cudaLimitStackSize, 1024*6);
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));

	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(runtimeHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

	// run
	__main(deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(deviceHeap);

	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}
#endif

#endif