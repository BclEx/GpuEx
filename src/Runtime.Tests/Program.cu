#include <string.h>
#include <RuntimeHost.h>
//http://thelegendofrandom.com/blog/archives/2231

void __testRuntime(cudaDeviceHeap &r);
void __testRuntimeEx(cudaDeviceHeap &r);
//void __testRegex1(cudaDeviceHeap &r);

#if __CUDACC__
void GMain(cudaDeviceHeap &r) {
#else
void main(int argc, char **argv) { cudaDeviceHeap r; memset(&r, 0, sizeof(r));
#endif
__testRuntime(r);
__testRuntimeEx(r);
//__testRegex1(r);
}

#if __CUDACC__
void __main(cudaDeviceHeap &r)
{	
	cudaDeviceHeapSelect(r);
	GMain(r); cudaDeviceHeapSynchronize(r);
}

cudaDeviceHeap _deviceHeap;
int main(int argc, char **argv)
{
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	//cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*4));

	_deviceHeap = cudaDeviceHeapCreate(256, 4096);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender *render = new RuntimeVisualRender(deviceHeap);
	//if (!Visual::InitGL(render, &argc, argv)) return 0;
	//cudaErrorCheck(cudaGLSetGLDevice(deviceId));

	// run
	__main(_deviceHeap);
	//Visual::Main();
	//Visual::Dispose();

	cudaDeviceHeapDestroy(_deviceHeap);

	cudaDeviceReset();
	printf("\nEnd"); char c; scanf("%c", &c);
	return 0;
}
#endif