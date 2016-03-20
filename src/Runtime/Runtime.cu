#define __EMBED__ 1
#include "Runtime.h"
#include "RuntimeTypes.h"

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
__device__ unsigned char __one;

//////////////////////
// FPRINTF
#pragma region FPRINTF

__constant__ FILE __iob_file[3] = { {nullptr, 0, nullptr, 0, 0, 0, 0, nullptr}, {nullptr, 1, nullptr, 0, 0, 0, 0, nullptr}, {nullptr, 2, nullptr, 0, 0, 0, 0, nullptr} };

#pragma endregion
