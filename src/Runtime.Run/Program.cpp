#include <RuntimeHost.h>
#include <FallocHost.h>

void __fallocExample(cudaDeviceFalloc &f);
void __runtimeExample(cudaDeviceHeap &r);

int main(int argc, char **argv)
{
	cudaDeviceHeap runtimeHost = cudaDeviceHeapCreate(256, 4096);
	//cudaDeviceFalloc fallocHost = cudaDeviceFallocCreate(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender *render = new RuntimeVisualRender(runtimeHost);
	//IVisualRender *render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv)) return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// test
	//__fallocExample(fallocHost);
	__runtimeExample(runtimeHost);

	// run
	Visual::Main();
	Visual::Dispose();
	
	cudaDeviceHeapDestroy(runtimeHost);
	//cudaDeviceFallocDestroy(fallocHost);

	cudaDeviceReset();
	printf("End.");
	char c; scanf("%c", &c);
	return 1;
}
