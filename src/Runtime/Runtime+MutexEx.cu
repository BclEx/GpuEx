// mutex.c
#include "Runtime.h"

__device__ MutexEx MutexEx_Empty = { nullptr };
__device__ MutexEx MutexEx_NotEmpty = { (void *)1 };

__device__ int MutexEx::Init()
{ 
	//MutexEx_Empty.Tag = nullptr;
	//MutexEx_NotEmpty.Tag = (void *)1;
	return 0;
}

__device__ int MutexEx::Shutdown()
{
	return 0;
}
