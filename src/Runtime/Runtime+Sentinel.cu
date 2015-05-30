#include <windows.h>
#include <process.h>
#include <assert.h>
#ifdef __device__
#undef __device__
#define __device__
#endif
#define RUNTIME_NAME RuntimeS
#include "Runtime.h"
#include "RuntimeHost.h"

#if OS_MAP
#pragma region OS_MAP
#include "Runtime+Alloc.cu"
#include "Runtime+BenignAlloc.cu"
#include "Runtime+Mem0.cu"
#include "Runtime+Mem1.cu"
#include "Runtime+TagBase.cu"

static RuntimeSentinelContext _ctx;

static bool Executor(void *tag, RuntimeSentinelMessage *data, int length)
{
	switch (data->OP)
	{
	case 1: {
		Messages::Stdio_fprintf *msg = (Messages::Stdio_fprintf *)data;
		msg->RC = fprintf(msg->File, msg->Format);
		return true; }
	case 2: {
		Messages::Stdio_fopen *msg = (Messages::Stdio_fopen *)data;
		msg->RC = fopen(msg->Filename, msg->Mode);
		return true; }
	case 3: {
		Messages::Stdio_fflush *msg = (Messages::Stdio_fflush *)data;
		msg->RC = fflush(msg->File);
		return true; }
	case 4: {
		Messages::Stdio_fclose *msg = (Messages::Stdio_fclose *)data;
		msg->RC = fclose(msg->File);
		return true; }
	case 5: {
		Messages::Stdio_fputc *msg = (Messages::Stdio_fputc *)data;
		msg->RC = fputc(msg->Ch, msg->File);
		return true; }
	case 6: {
		Messages::Stdio_fputs *msg = (Messages::Stdio_fputs *)data;
		msg->RC = fputs(msg->Str, msg->File);
		return true; }
	case 7: {
		Messages::Stdio_fread *msg = (Messages::Stdio_fread *)data;
		msg->RC = fread(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	case 8: {
		Messages::Stdio_fwrite *msg = (Messages::Stdio_fwrite *)data;
		msg->RC = fwrite(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	}
	return false;
}

static HANDLE _threadHostHandle = NULL;
static unsigned int __stdcall SentinelHostThread(void *data) 
{
	RuntimeSentinelContext *ctx = &_ctx;
	RuntimeSentinelMap *map = ctx->Map;
	while (map)
	{
		long id = map->GetId;
		RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadHostHandle && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadHostHandle) return 0;
		if (cmd->Magic != SENTINEL_MAGIC)
		{
			printf("Bad Sentinel Magic");
			exit(1);
		}
		//map->Dump();
		cmd->Dump();
		RuntimeSentinelMessage *msg = (RuntimeSentinelMessage *)cmd->Data;
		for (RuntimeSentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		//printf(".");
		*status = (!msg->Async ? 4 : 0);
		map->GetId += SENTINEL_SIZE;
	}
	return 0;
}

static HANDLE _threadDeviceHandle = NULL;
static unsigned int __stdcall SentinelDeviceThread(void *data) 
{
	RuntimeSentinelContext *ctx = &_ctx; //(RuntimeSentinelContext *)data;
	RuntimeSentinelMap *map = ctx->Map;
	while (map)
	{
		long id = map->GetId;
		RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadDeviceHandle && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadDeviceHandle) return 0;
		if (cmd->Magic != SENTINEL_MAGIC)
		{
			printf("Bad Sentinel Magic");
			exit(1);
		}
		//map->Dump();
		cmd->Dump();
		RuntimeSentinelMessage *msg = (RuntimeSentinelMessage *)cmd->Data;
		for (RuntimeSentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		//printf(".");
		*status = (!msg->Async ? 4 : 0);
		map->GetId += SENTINEL_SIZE;
	}
	return 0;
}

static RuntimeSentinelExecutor _baseExecutor;
#if HAS_HOST
static HANDLE _mapHostHandle = NULL;
static int *_mapHost = nullptr;
#endif
static HANDLE _mapDeviceHandle = NULL;
static int *_mapDevice = nullptr;

void RuntimeSentinel::Initialize(RuntimeSentinelExecutor *executor)
{
	//https://msdn.microsoft.com/en-us/library/windows/desktop/aa366551(v=vs.85).aspx
	//https://github.com/pathscale/nvidia_sdk_samples/blob/master/simpleStreams/0_Simple/simpleStreams/simpleStreams.cu
	// host
#if HAS_HOST
	_mapForHostHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT, "MyName"); //"Global\\MyFileMappingObject"
	if (!_mapForHostHandle)
	{
		printf("Could not create file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_mapForHost = (int *)MapViewOfFile(_mapForHostHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	if (!_mapForHost)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_mapForHostHandle);
		exit(1);
	}
#endif
	// device
#ifdef _GPU
	cudaErrorCheckF(cudaHostAlloc(&_mapDevice, sizeof(RuntimeSentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
	_ctx.Map = _mapDevice;
	RuntimeSentinelContext *d_mapDevice;
	cudaErrorCheckF(cudaHostGetDevicePointer(&d_mapDevice, _ctx.Map, 0), goto initialize_error);
	cudaErrorCheckF(cudaMemcpyToSymbol(_runtimeSentinelMap, &d_mapDevice, sizeof(_ctx.Map)), goto initialize_error);
#else
	//_mapDevice = (int *)VirtualAlloc(NULL, (sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
	_mapDevice = (int *)malloc(sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	_ctx.Map = (RuntimeSentinelMap *)_ROUNDN(_mapDevice, MEMORY_ALIGNMENT);
	_runtimeSentinelMap = _ctx.Map;
#endif
	if (!_ctx.Map)
	{
		printf("Could not create map.\n");
		goto initialize_error;
	}
	memset(_ctx.Map, 0, sizeof(RuntimeSentinelMap));
	_baseExecutor.Name = "base";
	_baseExecutor.Executor = Executor;
	_baseExecutor.Tag = nullptr;
	RegisterExecutor(&_baseExecutor, true);
	if (executor)
		RegisterExecutor(executor, true);
#if HAS_HOST
	_threadHostHandle = (HANDLE)_beginthreadex(0, 0, SentinelHostThread, nullptr, 0, 0);
#endif
	_threadDeviceHandle = (HANDLE)_beginthreadex(0, 0, SentinelDeviceThread, nullptr, 0, 0);
initialize_error:
	Shutdown();
	exit(1);
}

void RuntimeSentinel::Shutdown()
{
	// host
#if HAS_HOST
	if (_threadHostHandle) { CloseHandle(_threadHostHandle); _threadHostHandle = NULL; }
	if (_mapHost) { _mapForHost(_map); _mapHost = nullptr; }
	if (_mapHostHandle) { CloseHandle(_mapHostHandle); _mapHostHandle = NULL; }
#endif
	// device
	if (_threadDeviceHandle) { CloseHandle(_threadDeviceHandle); _threadDeviceHandle = NULL; }
#ifdef _GPU
	if (_mapDevice) { cudaErrorCheckA(cudaFreeHost(_mapDevice)); _mapDevice = nullptr; }
#else
	if (_mapDevice) { free(_mapDevice); /*VirtualFree(_mapDevice, 0, MEM_RELEASE);*/ _mapDevice = nullptr; }
#endif
}

RuntimeSentinelExecutor *RuntimeSentinel::FindExecutor(const char *name)
{
	RuntimeSentinelExecutor *exec = nullptr;
	for (exec = _ctx.List; exec && name && strcmp(name, exec->Name); exec = exec->Next) { }
	return exec;
}

static void UnlinkExecutor(RuntimeSentinelExecutor *exec)
{
	if (!exec) { }
	else if (_ctx.List == exec)
		_ctx.List = exec->Next;
	else if (_ctx.List)
	{
		RuntimeSentinelExecutor *p = _ctx.List;
		while (p->Next && p->Next != exec)
			p = p->Next;
		if (p->Next == exec)
			p->Next = exec->Next;
	}
}

void RuntimeSentinel::RegisterExecutor(RuntimeSentinelExecutor *exec, bool default_)
{
	UnlinkExecutor(exec);
	if (default_ || !_ctx.List)
	{
		exec->Next = _ctx.List;
		_ctx.List = exec;
	}
	else
	{
		exec->Next = _ctx.List->Next;
		_ctx.List->Next = exec;
	}
	assert(_ctx.List != nullptr);
}

void RuntimeSentinel::UnregisterExecutor(RuntimeSentinelExecutor *exec)
{
	UnlinkExecutor(exec);
}

#pragma endregion
#endif