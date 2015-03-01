#ifndef __CORE_PAGER_CU_H__
#define __CORE_PAGER_CU_H__

#include <Core/Core.cu.h>

#define Pid uint32

#pragma region IBackup

class IBackup
{
public:
	__device__ virtual void Update(Pid id, byte data[]);
	__device__ virtual void Restart();
};

#pragma endregion

#include "Pager.cu.h"
#include "PCache.cu.h"
#include "Wal.cu.h"

#endif // __CORE_PAGER_CU_H__