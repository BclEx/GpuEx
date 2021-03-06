#include <FallocHost.h>
#include <string.h>

void __testFalloc(cudaDeviceFalloc &f);

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) {
#endif
//__testFalloc(f);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*4));

	cudaDeviceFalloc deviceFalloc = cudaDeviceFallocCreate(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new FallocVisualRender(deviceFalloc);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

	// run
	__main(deviceHeap);
	//Visual::DataEx();
	//Visual::Dispose();

	cudaDeviceFallocDestroy(deviceFalloc);

	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}
#endif