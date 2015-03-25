#include "..\System.Data.net\Core+Vdbe\Core+Vdbe.cu.h"

__device__ static bool MyCallback(void *args, int argsLength, char **args2, char **cols)
{
	if (args2)
	{
		int w = 5;
		int i;
		for (i = 0; i < argsLength; i++)
		{
			int len = _strlen30(cols[i] ? cols[i] : "");
			if (len > w) w = len;
		}
		_printf("\n");
		for (i = 0; i < argsLength; i++)
			_printf("%*s = %s\n", w, cols[i], args2[i] ? args2[i] : "null");
	}
	return false;
}

//__device__ static void Trace(void *args, const char *text)
//{
//	printf(text);
//}

__device__ static void TestDB()
{
	SysEx_LOG(RC_OK, "START\n");
#if _DEBUG
	//ParserTrace(stderr, "p: ");
#endif

	// open
	Context *ctx;
	//RC rc = Main::Open("C:\\T_\\Test2.db", &ctx);
	RC rc = Main::Open(":memory:", &ctx);
	if (rc != RC_OK)
	{
		_printf("Error: %s\n", Main::ErrMsg(ctx));
		goto close;
	}
	
	//ctx->Trace = Trace;

	// run query
	char *errMsg = nullptr;
	//Main::Exec(ctx, "Select * From sqlite_master;", MyCallback, nullptr, &errMsg);
	//Main::Exec(ctx, "Create Table Test (Name); Insert Test(Name) Values('Sky');", MyCallback, nullptr, &errMsg);
	Main::Exec(ctx, "PRAGMA database_list;", MyCallback, nullptr, &errMsg);
	if (errMsg)
	{
		_printf("Error: %s\n", errMsg);
		_free(errMsg);
	}

	// close
close:
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