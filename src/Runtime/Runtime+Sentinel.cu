#include <windows.h>
#include <process.h>
#include <assert.h>
#include <io.h>
#ifdef __device__
#undef __device__
#define __device__
#endif
#ifdef __constant__
#undef __constant__
#define __constant__
#endif
#define SENTINEL
#define RUNTIME_NAME RuntimeS
#include "RuntimeHost.h"
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
	case 0: {
		Messages::Stdio_fprintf *msg = (Messages::Stdio_fprintf *)data;
		msg->RC = fprintf(msg->File, msg->Format);
		return true; }
	case 1: {
		Messages::Stdio_setvbuf *msg = (Messages::Stdio_setvbuf *)data;
		msg->RC = setvbuf(msg->File, msg->Buffer, msg->Mode, msg->Size);
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
		Messages::Stdio_fgetc *msg = (Messages::Stdio_fgetc *)data;
		msg->RC = fgetc(msg->File);
		return true; }
	case 6: {
		Messages::Stdio_fgets *msg = (Messages::Stdio_fgets *)data;
		msg->RC = fgets(msg->Str, msg->Num, msg->File);
		return true; }
	case 7: {
		Messages::Stdio_fputc *msg = (Messages::Stdio_fputc *)data;
		msg->RC = fputc(msg->Ch, msg->File);
		return true; }
	case 8: {
		Messages::Stdio_fputs *msg = (Messages::Stdio_fputs *)data;
		msg->RC = fputs(msg->Str, msg->File);
		return true; }
	case 9: {
		Messages::Stdio_fread *msg = (Messages::Stdio_fread *)data;
		msg->RC = fread(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	case 10: {
		Messages::Stdio_fwrite *msg = (Messages::Stdio_fwrite *)data;
		msg->RC = fwrite(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	case 11: {
		Messages::Stdio_fseek *msg = (Messages::Stdio_fseek *)data;
		msg->RC = fseek(msg->File, msg->Offset, msg->Origin);
		return true; }
	case 12: {
		Messages::Stdio_ftell *msg = (Messages::Stdio_ftell *)data;
		msg->RC = ftell(msg->File);
		return true; }
	case 13: {
		Messages::Stdio_feof *msg = (Messages::Stdio_feof *)data;
		msg->RC = feof(msg->File);
		return true; }
	case 14: {
		Messages::Stdio_ferror *msg = (Messages::Stdio_ferror *)data;
		msg->RC = ferror(msg->File);
		return true; }
	case 15: {
		Messages::Stdio_clearerr *msg = (Messages::Stdio_clearerr *)data;
		clearerr(msg->File);
		return true; }
	case 16: {
		Messages::Stdio_rename *msg = (Messages::Stdio_rename *)data;
		msg->RC = rename(msg->Oldname, msg->Newname);
		return true; }
	case 17: {
		Messages::Stdio_unlink *msg = (Messages::Stdio_unlink *)data;
		msg->RC = unlink(msg->Str);
		return true; }
	case 18: {
		Messages::Stdio_close *msg = (Messages::Stdio_close *)data;
		msg->RC = close(msg->Handle);
		return true; }
	case 19: {
		Messages::Stdio_system *msg = (Messages::Stdio_system *)data;
		msg->RC = system(msg->Str);
		return true; }
	}
	return false;
}

#if HAS_HOSTSENTINEL
static HANDLE _threadHostHandle = NULL;
static unsigned int __stdcall SentinelHostThread(void *data) 
{
	RuntimeSentinelContext *ctx = &_ctx;
	RuntimeSentinelMap *map = ctx->HostMap;
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
		//cmd->Dump();
		RuntimeSentinelMessage *msg = (RuntimeSentinelMessage *)cmd->Data;
		for (RuntimeSentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		//printf(".");
		*status = (!msg->Async ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}
#endif

static HANDLE _threadDeviceHandle[SENTINEL_DEVICEMAPS];
static unsigned int __stdcall SentinelDeviceThread(void *data) 
{
	int threadId = (int)data;
	RuntimeSentinelContext *ctx = &_ctx; 
	RuntimeSentinelMap *map = ctx->DeviceMap[threadId];
	while (map)
	{
		long id = map->GetId;
		RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadDeviceHandle[threadId] && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadDeviceHandle[threadId]) return 0;
		if (cmd->Magic != SENTINEL_MAGIC)
		{
			printf("Bad Sentinel Magic");
			exit(1);
		}
		//map->Dump();
		cmd->Dump();
		RuntimeSentinelMessage *msg = (RuntimeSentinelMessage *)cmd->Data;
		for (RuntimeSentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		printf(".");
		*status = (!msg->Async ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}

static RuntimeSentinelExecutor _baseExecutor;
#if HAS_HOSTSENTINEL
static RuntimeSentinelMap *_runtimeSentinelHostMap = nullptr;
static HANDLE _hostMapHandle = NULL;
static int *_hostMap = nullptr;
#endif
static int *_deviceMap[SENTINEL_DEVICEMAPS];

#if 0
// https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918(v=vs.85).aspx
// http://www.dreamincode.net/forums/topic/171917-how-to-setread-registry-key-in-c/
DWORD _savedTdrDelay = -2;
void TdrInitialize()
{
	HKEY key;
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"System\\CurrentControlSet\\Control\\GraphicsDrivers", 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
		return;
	DWORD dwBufSize = sizeof(DWORD);
	if (RegQueryValueExW(key, L"TdrDelay", 0, 0, (LPBYTE)&_savedTdrDelay, &dwBufSize) != ERROR_SUCCESS)
		_savedTdrDelay = -1;
	else
		printf("Key value is: %d\n", _savedTdrDelay);
	DWORD newTdrDelay = 10;
	if (RegSetValueExW(key, L"TdrDelay", 0, REG_DWORD, (const BYTE *)&newTdrDelay, sizeof(newTdrDelay)) != ERROR_SUCCESS)
		_savedTdrDelay = -1;
	RegCloseKey(key);
}

void TdrShutdown()
{
	HKEY key;
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"System\\CurrentControlSet\\Control\\GraphicsDrivers", 0, KEY_ALL_ACCESS, &key) != ERROR_SUCCESS)
		return;
	if (_savedTdrDelay < 0)
	{
		if (RegDeleteValueW(key, L"TdrDelay") != ERROR_SUCCESS)
			_savedTdrDelay = -1;
	}
	else
	{
		if (RegSetValueExW(key, L"TdrDelay", 0, REG_DWORD, (const BYTE *)&_savedTdrDelay, sizeof(_savedTdrDelay)) != ERROR_SUCCESS)
			_savedTdrDelay = -1;
	}
	RegCloseKey(key);
}
#endif

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa366551(v=vs.85).aspx
// https://github.com/pathscale/nvidia_sdk_samples/blob/master/simpleStreams/0_Simple/simpleStreams/simpleStreams.cu
void RuntimeSentinel::ServerInitialize(RuntimeSentinelExecutor *executor, char *mapHostName)
{
	// create host map
#if HAS_HOSTSENTINEL
	_hostMapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT, mapHostName);
	if (!_hostMapHandle)
	{
		printf("Could not create file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	if (!_hostMap)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_hostMapHandle);
		exit(1);
	}
	_runtimeSentinelHostMap = _ctx.HostMap = (RuntimeSentinelMap *)_ROUNDN(_hostMap, MEMORY_ALIGNMENT);
#endif

	// create device maps
#ifdef _GPU
	RuntimeSentinelMap *d_deviceMap[SENTINEL_DEVICEMAPS];
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		cudaErrorCheckF(cudaHostAlloc(&_deviceMap[i], sizeof(RuntimeSentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
		d_deviceMap[i] = _ctx.DeviceMap[i] = (RuntimeSentinelMap *)_deviceMap[i];
		cudaErrorCheckF(cudaHostGetDevicePointer(&d_deviceMap[i], _ctx.DeviceMap[i], 0), goto initialize_error);
	}
	cudaErrorCheckF(cudaMemcpyToSymbol(_runtimeSentinelDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
#else
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		//_deviceMap[i] = (int *)VirtualAlloc(NULL, (sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
		_deviceMap[i] = (int *)malloc(sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
		_runtimeSentinelDeviceMap[i] = _ctx.DeviceMap[i] = (RuntimeSentinelMap *)_ROUNDN(_deviceMap[i], MEMORY_ALIGNMENT);
		if (!_runtimeSentinelDeviceMap[i])
		{
			printf("Could not create map.\n");
			goto initialize_error;
		}
		memset(_runtimeSentinelDeviceMap[i], 0, sizeof(RuntimeSentinelMap));
	}
#endif

	// register executor
	_baseExecutor.Name = "base";
	_baseExecutor.Executor = Executor;
	_baseExecutor.Tag = nullptr;
	RegisterExecutor(&_baseExecutor, true);
	if (executor)
		RegisterExecutor(executor, true);

	// launch threads
#if HAS_HOSTSENTINEL
	_threadHostHandle = (HANDLE)_beginthreadex(0, 0, SentinelHostThread, nullptr, 0, 0);
#endif
	memset(_threadDeviceHandle, 0, sizeof(_threadDeviceHandle));
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
		_threadDeviceHandle[i] = (HANDLE)_beginthreadex(0, 0, SentinelDeviceThread, (void *)i, 0, 0);
	return;
initialize_error:
	ServerShutdown();
	exit(1);
}

void RuntimeSentinel::ServerShutdown()
{
	// close host map
#if HAS_HOSTSENTINEL
	if (_threadHostHandle) { CloseHandle(_threadHostHandle); _threadHostHandle = NULL; }
	if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
	if (_hostMapHandle) { CloseHandle(_hostMapHandle); _hostMapHandle = NULL; }
#endif
	// close device maps
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		if (_threadDeviceHandle[i]) { CloseHandle(_threadDeviceHandle[i]); _threadDeviceHandle[i] = NULL; }
#ifdef _GPU
		if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
#else
		if (_deviceMap[i]) { free(_deviceMap[i]); /*VirtualFree(_deviceMap[i], 0, MEM_RELEASE);*/ _deviceMap[i] = nullptr; }
#endif
	}
}

void RuntimeSentinel::ClientInitialize(char *mapHostName)
{
#if HAS_HOSTSENTINEL
	_hostMapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mapHostName);
	if (!_hostMapHandle)
	{
		printf("Could not open file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(RuntimeSentinelMap) + MEMORY_ALIGNMENT);
	if (!_hostMap)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_hostMapHandle);
		exit(1);
	}
	_runtimeSentinelHostMap = _ctx.HostMap = (RuntimeSentinelMap *)_ROUNDN(_hostMap, MEMORY_ALIGNMENT);
#endif
}

void RuntimeSentinel::ClientShutdown()
{
#if HAS_HOSTSENTINEL
	if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
	if (_hostMapHandle) { CloseHandle(_hostMapHandle); _hostMapHandle = NULL; }
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