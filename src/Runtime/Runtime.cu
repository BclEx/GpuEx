#define __EMBED__ 1
#include "Runtime.h"
#include "RuntimeTypes.h"

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
__device__ unsigned char __one;

//////////////////////
// FPRINTF
#pragma region FPRINTF

__constant__ FILE *__iob_file[3] = { (FILE *)1, (FILE *)2, (FILE *)3 };
//__constant__ FILE __iob_file[3] = { {nullptr, 0, nullptr, 0, 0, 0, 0, nullptr}, {nullptr, 0, nullptr, 0, 0, 0, 0, nullptr}, {nullptr, 0, nullptr, 0, 0, 0, 0, nullptr} };

//#include "cuda_runtime_api.h"
//extern "C" cudaError_t cudaIobSelect()
//{
//	FILE *file0;
//	//FILE *file1;
//	//FILE *file2;
//	cudaError_t error;
//	if ((error = cudaGetSymbolAddress((void **)&file0, "__iob_file0")) != cudaSuccess)
//		return error;
//	printf("HERE");
//	return error;
//}

#pragma endregion
