#include <windows.h>
#include <process.h>
#include <stdio.h>
#include "Runtime.h"

#if OS_MAP && OS_WIN
#pragma region OS_WIN

RuntimeSentinelMap *_runtimeSentinelMap = nullptr;
void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = _runtimeSentinelMap;
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = (InterlockedAdd((long *)&map->SetId, SENTINEL_SIZE) - SENTINEL_SIZE);
	RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	while (InterlockedCompareExchange((long *)status, 1, 0) != 0) { }
	cmd->Data = (char *)cmd + _ROUND8(sizeof(RuntimeSentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg2->Prepare && !msg2->Prepare(msg, &cmd->Data[0], &cmd->Data[length]))
	{
		printf("msg too long");
		exit(0);
	}
	memcpy(cmd->Data, msg, msgLength);
	*status = 2;
	if (!msg2->Async)
	{
		while (InterlockedCompareExchange((long *)status, 5, 4) != 4) { }
		memcpy(msg, cmd->Data, msgLength);
		*status = 0;
	}
}

#pragma endregion
#endif