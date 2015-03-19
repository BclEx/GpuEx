#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <Core+Test\TestCtx.cu.h>

#pragma region Preamble

#if __CUDACC__
#define DEVICE(name) \
class name##Class : public Tester { public: __device__ void Test(); }; \
	__global__ void name##Launcher(void *r) { _runtimeSetHeap(r); name##Class c; c.Initialize(); c.Test(); } \
	void name##_host(cudaDeviceHeap &r) { name##Launcher<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); } \
	__device__ void name##Class::Test()
#else
#define DEVICE(name) \
class name##Class : public Tester { public: __device__ void Test(); }; \
	__global__ void name##Launcher(void *r) { _runtimeSetHeap(r); name##Class c; c.Initialize(); c.Test(); } \
	void name##_host(cudaDeviceHeap &r) { name##Launcher(r.heap); cudaDeviceHeapSynchronize(r); } \
	__device__ void name##Class::Test()
#endif

#pragma endregion

enum PLATFORM
{
	PLATFORM_WINDOWS = 1,
};

class Tester
{
public:
	PLATFORM _tcl_platform;
	Hash G;
	bool _do_not_use_codec;
	bool _SLAVE;
	TestCtx *db;
	TestCtx *db2;
	TestCtx *db3;

	__device__ void sqlite3(TestCtx *ctx, array_t<void *> args);
	__device__ int GetFileRetries();
	__device__ int GetFileRetryDelay();
	__device__ char *GetPwd();
	__device__ void copy_file(char *from, char *to);
	__device__ void forcecopy(char *from, char *to);
	__device__ void do_copy_file(bool force, char *from, char *to);
	__device__ bool is_relative_file(char *file);
	__device__ bool test_pwd(array_t<char *> args);
	__device__ void delete_file(const char *args[], int argsLength);
	__device__ void forcedelete(const char *args[], int argsLength);
	__device__ void do_delete_file(bool force, const char *args[], int argsLength);
	__device__ void execpresql(TestCtx *handle, void *args);
	__device__ void do_not_use_codec();
	__device__ void Initialize();
	__device__ void reset_db();
	__device__ void finish_test();
	__device__ void finalize_testing();
	__device__ void show_memstats();
	__device__ void execsql(const char *sql, TestCtx *db = nullptr);
};
