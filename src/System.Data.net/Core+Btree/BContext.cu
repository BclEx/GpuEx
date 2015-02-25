#include "Core+Btree.cu.h"

namespace Core
{
#pragma region Busy Handler

	__device__ int BContext::InvokeBusyHandler()
	{
		if (BusyHandler.Func == nullptr || BusyHandler.Busys < 0) return 0;
		int rc = BusyHandler.Func(BusyHandler.Arg, BusyHandler.Busys);
		if (rc == 0)
			BusyHandler.Busys = -1;
		else
			BusyHandler.Busys++;
		return rc; 
	}

#pragma endregion
}
