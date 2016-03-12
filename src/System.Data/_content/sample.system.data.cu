#if 0

//https://www.raywenderlich.com/123579/sqlite-tutorial-swift
#include <RuntimeHost.h>
#include <Core+Vdbe\Core+Vdbe.cu.h>

__device__ Context *_ctx = nullptr;
__global__ void initializeKernel(const char *path)
{
	DataEx::Initialize();
	RC rc = DataEx::Open(path, &_ctx);
	if (rc != RC_OK)
	{
		_printf("Can't open database: %s\n", DataEx::ErrMsg(_ctx));
		DataEx::Close(_ctx);
		return;
	}
}

__global__ void shutdownKernel()
{
	if (!_ctx)
		DataEx::Close(_ctx);
	DataEx::Shutdown();
}

__device__ static bool callback(void *notUsed, int argc, char **args, char **colNames)
{
	for (int i = 0; i < argc; i++)
		_printf("%s = %s\n", colNames[i], (args[i] ? args[i] : "NULL"));
	_printf("\n");
	return false;
}

__global__ void queryKernel(const char *sql)
{
	char *errMsg = nullptr;
	RC rc = DataEx::Exec(_ctx, sql, callback, 0, &errMsg);
	if (rc != RC_OK)
	{
		_printf("SQL error: %s\n", errMsg);
		_free(errMsg);
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaDeviceHeap _deviceHeap; 
void initialize(const char *path)
{
	memset(&_deviceHeap, 0, sizeof(_deviceHeap));
	// Set DeviceFlags for cudaDeviceMapHost
	cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheck(cudaSetDevice(0));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*15));
	_deviceHeap = cudaDeviceHeapCreate();
	cudaDeviceHeapSelect(_deviceHeap);
	// Initialize Sentinel if required.
#if OS_MAP
	CoreS::VSystemSentinel::Initialize();
#endif

	// Allocate GPU buffers for three vectors (two input, one output)    .
	char *dev_path = nullptr;
	unsigned int pathLength = strlen(path) + 1;
	cudaErrorCheck(cudaMalloc((void**)&dev_path, pathLength));
	cudaErrorCheck(cudaMemcpy(dev_path, path, pathLength, cudaMemcpyHostToDevice));
	// Launch a kernel on the GPU.
	initializeKernel<<<1, 1>>>(dev_path);
	cudaErrorCheck(cudaFree(dev_path));
}

void query(const char *sql)
{
	// Launch a kernel on the GPU.
	char *dev_sql = nullptr;
	unsigned int sqlLength = strlen(sql) + 1;
	cudaErrorCheck(cudaMalloc((void**)&dev_sql, sqlLength));
	cudaErrorCheck(cudaMemcpy(dev_sql, sql, sqlLength, cudaMemcpyHostToDevice));
	queryKernel<<<1, 1>>>(dev_sql);
	cudaErrorCheck(cudaFree(dev_sql));
}

void shutdown()
{
	// Launch a kernel on the GPU.
	shutdownKernel<<<1, 1>>>();
	// Shutdown Sentinel if required.
#if OS_MAP
	CoreS::VSystemSentinel::Shutdown();
#endif
	// Check for any errors launching the kernel
	cudaErrorCheck(cudaGetLastError());
	// cudaDeviceHeapSynchronize..
	cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap));
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrorCheck(cudaDeviceSynchronize());
	cudaDeviceHeapDestroy(_deviceHeap);
}

int main()
{	
	// Write values to filesytem.
	initialize(":memory:");
	query("CREATE TABLE Contact(" \
		"Id INT PRIMARY KEY NOT NULL," \
		"Name CHAR(255));");
	query("INSERT INTO Contact (Id, Name) VALUES (1, 'NAME');");
	query("SELECT * FROM Contact;");
	shutdown();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaErrorCheck(cudaDeviceReset());

	return 0;
}

#endif