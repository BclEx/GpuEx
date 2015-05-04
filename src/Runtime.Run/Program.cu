#include <RuntimeHost.h>
#include <FallocHost.h>

void __testFalloc(cudaDeviceFalloc &f);
void __testRuntime(cudaDeviceHeap &r);
void __testRegex1(cudaDeviceHeap &r);

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
//__testFalloc(f);
__testRuntime(r);
__testRegex1(r);
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

	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);
	//cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//IVisualRender *render = new FallocVisualRender(fallocHost);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

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