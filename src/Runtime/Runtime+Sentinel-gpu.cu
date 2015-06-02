#include "Runtime.h"

#if OS_MAP && OS_GPU
#pragma region OS_GPU

__device__ volatile unsigned int _runtimeSentinelMapId;
__constant__ RuntimeSentinelMap *_runtimeSentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = _runtimeSentinelDeviceMap[_runtimeSentinelMapId++ % SENTINEL_DEVICEMAPS];
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = __iAtomicAdd((int *)&map->SetId, SENTINEL_MSGSIZE);
	RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	//while (atomicCAS((unsigned int *)status, 1, 0) != 0) { __syncthreads(); }
	cmd->Data = (char *)cmd + _ROUND8(sizeof(RuntimeSentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg2->Prepare && !msg2->Prepare(msg, cmd->Data, cmd->Data+length))
	{
		printf("msg too long");
		asm("trap;"); //_trap();
	}
	memcpy(cmd->Data, msg, msgLength);
	*status = 2;
	if (!msg2->Async)
	{
		unsigned int s_; do { s_ = *status; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
		memcpy(msg, cmd->Data, msgLength);
		*status = 0;
	}
}

#pragma endregion
#endif