#include <RuntimeHost.h>
#include <FallocHost.h>

void __testSystem(cudaDeviceHeap &r);
void __testSystemIO(cudaDeviceHeap &r);

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
//__testSystem(r);
__testSystemIO(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaCheckErrors(cudaSetDeviceFlags(cudaDeviceMapHost), return -1);
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaCheckErrors(cudaSetDevice(deviceId), return -2);
	cudaDeviceReset();

	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);
	//cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//IVisualRender *render = new FallocVisualRender(fallocHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaCheckErrors(cudaGLSetGLDevice(deviceId), return -3);

	// run
	__main(deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(deviceHeap);
	//cudaDeviceFallocDestroy(fallocHost);

	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}
#endif