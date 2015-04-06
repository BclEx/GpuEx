#ifndef _LIB

//#define VISUAL
#include <RuntimeHost.h>

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
extern void select2_host(cudaDeviceHeap &r);
select2_host(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaCheckErrors(cudaDeviceHeapSelect(r), );
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaCheckErrors(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax), return -1);
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaCheckErrors(cudaSetDevice(deviceId), return -2);
	cudaCheckErrors(cudaDeviceSetLimit(cudaLimitStackSize, 1024*3), return -2);

	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaCheckErrors(cudaGLSetGLDevice(deviceId), return -3);

	// run
	__main(deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(deviceHeap);

	cudaDeviceReset();
	printf("\nEnd."); char c; scanf("%c", &c);
	return 0;
}
#endif

#endif