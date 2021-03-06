#ifndef _LIB

//#define VISUAL
#include <RuntimeHost.h>
#include <string.h>

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
//extern void select2_host(cudaDeviceHeap &r);
//select2_host(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaErrorCheck(cudaDeviceHeapSelect(r));
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*3));

	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

	// run
	__main(deviceHeap);
	//Visual::DataEx();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(deviceHeap);

	cudaDeviceReset();
	printf("\nEnd."); char c; scanf("%c", &c);
	return 0;
}
#endif

#endif