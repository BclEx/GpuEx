// mutex.c
#include "Core.cu.h"

namespace Core
{
	__device__ MutexEx MutexEx_Empty = { nullptr };
	__device__ MutexEx MutexEx_NotEmpty = { (void *)1 };

	__device__ RC MutexEx::Init()
	{ 
		//MutexEx_Empty.Tag = nullptr;
		//MutexEx_NotEmpty.Tag = (void *)1;
		return RC_OK;
	}

	__device__ RC MutexEx::End()
	{
		return RC_OK;
	}
}
