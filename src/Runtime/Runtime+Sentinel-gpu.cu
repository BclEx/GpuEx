#include "Runtime.h"

#if OS_MAP && OS_GPU
#pragma region OS_GPU

__device__ void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = _runtimeSentinelMap;
	int id = atomicAdd((unsigned int *)&map->AddId, 1);
	RuntimeSentinelCommand *cmd = &map->Commands[(id-1)%_lengthof(map->Commands)];
	volatile int *status = (volatile int *)&cmd->Status;
	//while (atomicCAS((unsigned int *)status, 1, 0) != 0) { __syncthreads(); }
	cmd->Length = msgLength;
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	if (msg2->Prepare)
		msg2->Prepare(msg, cmd->Data, sizeof(cmd->Data));
	memcpy(cmd->Data, msg, msgLength);
	*status = 2; //: asm("st.global.wt.u32 [%0], %1;" : "+l"(status) : "r"(2));
	// while (atomicCAS((unsigned int *)status, 5, 4)) != 4);
	unsigned int s_;
	do
	{
		s_ = *status;
		//asm("ld.global.cv.u32 %0, [%1];" : "=r"(s_) : "l"(status));
		printf("%d ", s_);
		__syncthreads();
	} while (s_ != 4);
	__syncthreads();
	memcpy(msg, cmd->Data, msgLength);
	*status = 0; //: asm("st.global.wt.u32 [%0], %1;" : "+l"(status) : "r"(0));	
}

#pragma endregion
#endif