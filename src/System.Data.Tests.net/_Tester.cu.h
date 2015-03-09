#include <Core+Vdbe\Core+Vdbe.cu.h>

#pragma region Preamble

#if __CUDACC__
#define DEVICE(name, body) \
	__global__ void name(void *r) { _runtimeSetHeap(r); body }; \
	void name##_host(cudaDeviceHeap &r) { name<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); }
#else
#define DEVICE(name, body) \
	__global__ void name(void *r) { _runtimeSetHeap(r); body }; \
	void name##_host(cudaDeviceHeap &r) { name(r.heap); cudaDeviceHeapSynchronize(r); }
#endif

#pragma endregion

enum PLATFORM
{
	PLATFORM_WINDOWS = 1,
};

class Tester
{
	PLATFORM _tcl_platform;
	Hash G;
	bool _do_not_use_codec;
	__device__ void sqlite3(Context *ctx, array_t<char *> args);
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
	__device__ void execpresql(void *handle, void *args);
	__device__ void do_not_use_codec();
	__device__ void reset_db();
	__device__ void finish_test();
	__device__ void execsql(const char *sql, Context *db = nullptr);
};
