#include <windows.h>
#include <process.h>
#include <assert.h>
#include "RuntimeSentinel.h"

static RuntimeSentinelContext _ctx;

static bool Executor(void *tag, RuntimeSentinelMessage *data, int length)
{
	switch (data->OP)
	{
	case 1: {
		Messages::Stdio_fprintf *msg = (Messages::Stdio_fprintf *)data;
		fprintf(msg->File, msg->Format);
		return true; }
	case 2: {
		Messages::Stdio_fopen *msg = (Messages::Stdio_fopen *)data;
		fopen(msg->Filename, msg->Mode);
		return true; }
	case 3: {
		Messages::Stdio_fflush *msg = (Messages::Stdio_fflush *)data;
		fflush(msg->File);
		return true; }
	case 4: {
		Messages::Stdio_fclose *msg = (Messages::Stdio_fclose *)data;
		fclose(msg->File);
		return true; }
	case 5: {
		Messages::Stdio_fputc *msg = (Messages::Stdio_fputc *)data;
		fputc(msg->Ch, msg->File);
		return true; }
	case 6: {
		Messages::Stdio_fputs *msg = (Messages::Stdio_fputs *)data;
		fputs(msg->Str, msg->File);
		return true; }
	}
	return false;
}

static unsigned int __stdcall SentinelThread(void *data) 
{
	RuntimeSentinelContext *ctx = &_ctx; //(RuntimeSentinelContext *)data;
	RuntimeSentinelMap *map = ctx->Map;
	while (true)
	{
		int id = InterlockedAdd((LONG *)&map->RunId, 1);
		RuntimeSentinelCommand *cmd = &map->Commands[(id-1)%_lengthof(map->Commands)];
		while (InterlockedCompareExchange((LONG *)&cmd->Status, 3, 2) != 2) { Sleep(10); }
		for (RuntimeSentinelExecutor *exec = _ctx.List; exec && !exec->Executor(exec->Tag, (RuntimeSentinelMessage *)cmd->Data, cmd->Length); exec = exec->Next) { }
		cmd->Status = 4;
	}
}

static HANDLE _thread;
static RuntimeSentinelExecutor _baseExecutor;
void RuntimeSentinel::Initialize(RuntimeSentinelExecutor *executor)
{
#if _CPU
	_ctx.Map = (RuntimeSentinelMap *)malloc(sizeof(*_ctx.Map));
#else
	cudaErrorCheck(cudaHostAlloc(&_ctx.Map, sizeof(*_ctx.Map), cudaHostAllocPortable));
#endif
	memset(_ctx.Map, 0, sizeof(*_ctx.Map));
	_baseExecutor.Name = "base";
	_baseExecutor.Executor = RUNTIMESENTINELEXECUTOR(Executor);
	_baseExecutor.Tag = nullptr;
	RegisterExecutor(&_baseExecutor, true);
	if (executor)
		RegisterExecutor(executor, true);
	_thread = (HANDLE)_beginthreadex(0, 0, SentinelThread, &_ctx, 0, 0);
}

void RuntimeSentinel::Shutdown()
{
	CloseHandle(_thread);
#if _CPU
	free(_ctx.Map);
#else
	cudaFreeHost(_ctx.Map);
#endif
}

RuntimeSentinelMap *RuntimeSentinel::GetMap() { return _ctx.Map; }

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