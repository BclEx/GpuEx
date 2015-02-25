#include <RuntimeHost.h>

extern void coreTest0_host(cudaDeviceHeap &r);
static cudaDeviceHeap _deviceHeap;

void main(int argc, char **argv)
{
	_deviceHeap = cudaDeviceHeapCreate(256, 4096); cudaDeviceHeapSelect(_deviceHeap);

	coreTest0_host(_deviceHeap);

	cudaDeviceHeapDestroy(_deviceHeap); cudaDeviceReset();
	printf("\nEnd\n"); char c; scanf("%c", &c);
}
