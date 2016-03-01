#define __EMBED__ 1
#include "Runtime.h"
#include "RuntimeTypes.h"

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
__device__ unsigned char __one;

//////////////////////
// FPRINTF
#pragma region FPRINTF

__constant__ FILE _stdin_file = {0, 0, 0};
__constant__ FILE _stdout_file = {0, 1, 0};
__constant__ FILE _stderr_file = {0, 2, 0};

#pragma endregion
