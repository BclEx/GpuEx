#include "RuntimeSentinel.h"

#if OS_GPU
#pragma region OS_GPU

__device__ void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = 0;
	int id = atomicAdd((unsigned int *)&map->AddId, 1);
	RuntimeSentinelCommand *cmd = &map->Commands[(id-1)%_lengthof(map->Commands)];
	while (atomicCAS((unsigned int *)&cmd->Status, 1, 0) != 0);
	cmd->Length = msgLength;
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	if (msg2->Prepare)
		msg2->Prepare(msg, cmd->Data, sizeof(cmd->Data));
	memcpy(cmd->Data, msg, msgLength);
	cmd->Status = 2;
	while (atomicCAS((unsigned int *)&cmd->Status, 5, 4) != 4);
	memcpy(msg, cmd->Data, msgLength);
	cmd->Status = 0;
}

#pragma endregion
#endif