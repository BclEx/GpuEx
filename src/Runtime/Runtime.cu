#define __EMBED__ 1
#include "Runtime.h"
#include "RuntimeTypes.h"

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
__device__ unsigned char __one;

//////////////////////
// FPRINTF
#pragma region FPRINTF

__constant__ FILE _stdin_file  = {nullptr, 0, nullptr, 0, 0, 0, 0, nullptr};
__constant__ FILE _stdout_file = {nullptr, 1, nullptr, 0, 0, 0, 0, nullptr};
__constant__ FILE _stderr_file = {nullptr, 2, nullptr, 0, 0, 0, 0, nullptr};

#pragma endregion
