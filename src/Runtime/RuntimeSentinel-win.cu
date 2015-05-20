#include <windows.h>
#include <process.h>
#include <stdio.h>
#include "RuntimeSentinel.h"

#if OS_WIN
#pragma region OS_WIN

__device__ void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = GetMap();
	int id = InterlockedAdd((LONG *)&map->AddId, 1);
	RuntimeSentinelCommand *cmd = &map->Commands[(id-1)%_lengthof(map->Commands)];
	while (InterlockedCompareExchange((LONG *)&cmd->Status, 1, 0) != 0) { _sleep(10); }
	cmd->Length = msgLength;
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	if (msg2->Prepare)
		msg2->Prepare(msg, cmd->Data, sizeof(cmd->Data));
	memcpy(cmd->Data, msg, msgLength);
	cmd->Status = 2;
	while (InterlockedCompareExchange((LONG *)&cmd->Status, 5, 4) != 4) { _sleep(10); }
	memcpy(msg, cmd->Data, msgLength);
	cmd->Status = 0;
}

#pragma endregion
#endif