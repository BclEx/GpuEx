#include "Runtime.h"

#if OS_MAP && OS_GPU
#pragma region OS_GPU

__constant__ RuntimeSentinelMap *_runtimeSentinelMap = nullptr;
__device__ void RuntimeSentinel::Send(void *msg, int msgLength)
{
	RuntimeSentinelMap *map = _runtimeSentinelMap;
	RuntimeSentinelMessage *msg2 = (RuntimeSentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = __iAtomicAdd((int *)&map->SetId, SENTINEL_SIZE);
	RuntimeSentinelCommand *cmd = (RuntimeSentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	//while (atomicCAS((unsigned int *)status, 1, 0) != 0) { __syncthreads(); }
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg2->Prepare && !msg2->Prepare(msg, cmd->Data, cmd->Data + length))
	{
		printf("msg too long");
		asm("trap;"); //_trap();
	}
	memcpy(cmd->Data, msg, msgLength);
	*status = 2; //: asm("st.global.wt.u32 [%0], %1;" : "+l"(status) : "r"(2));
	if (!msg2->Async)
	{
		// while (atomicCAS((unsigned int *)status, 5, 4)) != 4);
		unsigned int s_;
		do
		{
			s_ = *status; //: asm("ld.global.cv.u32 %0, [%1];" : "=r"(s_) : "l"(status));
			printf("%d ", s_);
			__syncthreads();
		} while (s_ != 4);
		__syncthreads();
		memcpy(msg, cmd->Data, msgLength);
		*status = 0; //: asm("st.global.wt.u32 [%0], %1;" : "+l"(status) : "r"(0));	
	}
}

#pragma endregion
#endif