#include <windows.h>
#include <process.h>
#include <assert.h>
#ifdef __device__
#undef __device__
#define __device__
#endif
#define RUNTIME_NAME RuntimeS
#include "Runtime.h"

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
	}
	return false;
}

static HANDLE _threadHandle = NULL;
static unsigned int __stdcall SentinelThread(void *data) 
{
	RuntimeSentinelContext *ctx = &_ctx; //(RuntimeSentinelContext *)data;
	RuntimeSentinelMap *map = ctx->Map;
	while (map)
	{
		long id = map->GetId;
		RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadHandle && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadHandle) return 0;
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
static HANDLE _mapHandle = NULL;
static int *_map = nullptr;
//https://msdn.microsoft.com/en-us/library/windows/desktop/aa366551(v=vs.85).aspx
//https://github.com/pathscale/nvidia_sdk_samples/blob/master/simpleStreams/0_Simple/simpleStreams/simpleStreams.cu
void RuntimeSentinel::Initialize(RuntimeSentinelExecutor *executor)
{
	_mapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT, "MyName"); //"Global\\MyFileMappingObject"
	if (!_mapHandle)
	{
		printf("Could not create file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_map = (int *)MapViewOfFile(_mapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	//_map = (int *)malloc(sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	//_map = (int *)VirtualAlloc(NULL, (sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
	if (!_map)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_mapHandle);
		exit(1);
	}
	_ctx.Map = (RuntimeSentinelMap *)_ROUNDN(_map, MEMORY_ALIGNMENT);
#ifdef _GPU
	cudaErrorCheckF(cudaHostRegister(&_ctx.Map, sizeof(RuntimeSentinelMap), cudaHostRegisterPortable | cudaHostRegisterMapped), goto initialize_error);
	//cudaErrorCheckF(cudaHostAlloc(&_ctx.Map, sizeof(RuntimeSentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
	RuntimeSentinelContext *d_map;
	cudaErrorCheckF(cudaHostGetDevicePointer(&d_map, _ctx.Map, 0), goto initialize_error);
	cudaErrorCheckF(cudaMemcpyToSymbol(_runtimeSentinelMap, &d_map, sizeof(_ctx.Map)), goto initialize_error);
#else
	_runtimeSentinelMap = _ctx.Map; //= (RuntimeSentinelMap *)malloc(sizeof(RuntimeSentinelMap));
	if (!_runtimeSentinelMap)
	{
		printf("Could not create map.\n");
		goto initialize_error;
	}
#endif
	memset(_ctx.Map, 0, sizeof(RuntimeSentinelMap));
	_baseExecutor.Name = "base";
	_baseExecutor.Executor = Executor;
	_baseExecutor.Tag = nullptr;
	RegisterExecutor(&_baseExecutor, true);
	if (executor)
		RegisterExecutor(executor, true);
	_threadHandle = (HANDLE)_beginthreadex(0, 0, SentinelThread, nullptr, 0, 0);
	return;
initialize_error:
	Shutdown();
	exit(1);
}

void RuntimeSentinel::Shutdown()
{
	if (_threadHandle) { CloseHandle(_threadHandle); _threadHandle = NULL; }
#ifdef _GPU
	if (_ctx.Map) { cudaErrorCheckA(cudaHostUnregister(_ctx.Map)); _ctx.Map = nullptr; }
	//if (_ctx.Map) { cudaErrorCheckA(cudaFreeHost(_ctx.Map)); _ctx.Map = nullptr; }
#else
	//free(_ctx.Map);
#endif
	if (_map)
	{
		//VirtualFree(_map, 0, MEM_RELEASE);
		//free(_map);
		UnmapViewOfFile(_map);
		_map = nullptr;
	}
	if (_mapHandle) { CloseHandle(_mapHandle); _mapHandle = NULL; }
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