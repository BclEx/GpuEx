#include <RuntimeHost.h>
#include <FallocHost.h>

//void __fallocExample(cudaDeviceFalloc &f);
void __runtimeExample(cudaDeviceHeap &r);

#if __CUDACC__
__global__ void GMain(void *r) { _runtimeSetHeap(r);
#else
__global__ void main(int argc, char **argv) { cudaDeviceHeap r;
#endif
	//__fallocExample(fallocHost);
	__runtimeExample(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r.heap); cudaDeviceHeapSynchronize(r);
}

int main(int argc, char **argv)
{
	cudaDeviceHeap deviceHeap = cudaDeviceHeapCreate(256, 4096);
	//cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//IVisualRender *render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv)) return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// run
	__main(deviceHeap);
	//Visual::Main();
	//Visual::Dispose();
	
	cudaDeviceHeapDestroy(deviceHeap);
	//cudaDeviceFallocDestroy(fallocHost);

	cudaDeviceReset();
	//printf("End.");
	//char c; scanf("%c", &c);
	return 0;
}
#endif