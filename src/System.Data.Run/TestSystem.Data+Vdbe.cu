#include "..\System.Data.net\Core+Vdbe\Core+Vdbe.cu.h"

__device__ static bool MyCallback(void *args, int argsLength, char **args2, char **cols)
{
	return true;
}

__device__ static void TestDB()
{
	//SysEx_LOG(RC_OK, "START\n");
	ParserTrace(stderr, "p: ");

	// open
	Context *ctx;
	Main::Open("C:\\T_\\Test2.db", &ctx);

	// run query
	char *errMsg = nullptr;
	//Main::Exec(ctx, "Select * From MyTable;", MyCallback, nullptr, &errMsg);
	Main::Exec(ctx, "PRAGMA database_list;", MyCallback, nullptr, &errMsg);
	if (errMsg)
	{
		_printf("Error: %s\n", errMsg);
		_free(errMsg);
	}

	// close
	Main::Close(ctx);
}

// NATIVE: assert
__global__ static void testSystemDataVdbe0(void *r)
{
	_runtimeSetHeap(r);
	Main::Initialize();
	//
	TestDB();
	//
	Main::Shutdown();
	printf("System.Data+Vdbe: 0\n");
}

#if __CUDACC__
void __testSystemDataVdbe(cudaDeviceHeap &r)
{
	testSystemDataVdbe0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testSystemDataVdbe(cudaDeviceHeap &r)
{
	testSystemDataVdbe0(r.heap);
}
#endif