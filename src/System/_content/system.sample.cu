#include <RuntimeHost.h>
#include <Core\Core.cu.h>

void fileAccessWithCuda(const char *path);

__global__ void fileAccessKernel(const char *path)
{
	auto vfs = VSystem::FindVfs(nullptr);
	auto file = (VFile *)_allocZero(vfs->SizeOsFile);
	auto rc = vfs->Open(path, file, (VSystem::OPEN)(VSystem::OPEN_CREATE|VSystem::OPEN_READWRITE|VSystem::OPEN_MAIN_DB), nullptr);
	file->Write4(0, 123145);
	file->Close();
}

int main()
{	
	// Write values to filesytem.
	fileAccessWithCuda("\\T_\\Test.db");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaErrorCheck(cudaDeviceReset());

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void fileAccessWithCuda(const char *path)
{
	cudaDeviceHeap deviceHeap; memset(&deviceHeap, 0, sizeof(deviceHeap));

	// Set DeviceFlags for cudaDeviceMapHost
	cudaErrorCheckF(cudaSetDeviceFlags(cudaDeviceMapHost), goto Error);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheckF(cudaSetDevice(0), goto Error);

	deviceHeap = cudaDeviceHeapCreate();
	cudaDeviceHeapSelect(deviceHeap);

	// Initialize Sentinel if required.
#if OS_MAP
	CoreS::VSystemSentinel::Initialize();
#endif

	// Launch a kernel on the GPU with one thread for each element.
	fileAccessKernel<<<1, 1>>>(path);

	// Shutdown Sentinel if required.
#if OS_MAP
	CoreS::VSystemSentinel::Shutdown();
#endif

	// Check for any errors launching the kernel
	cudaErrorCheckF(cudaGetLastError(), goto Error);

	// cudaDeviceHeapSynchronize..
	cudaErrorCheckF(cudaDeviceHeapSynchronize(deviceHeap), goto Error);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrorCheckF(cudaDeviceSynchronize(), goto Error);

Error:
	cudaDeviceHeapDestroy(deviceHeap);
}
