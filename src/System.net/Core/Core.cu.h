﻿#ifndef __CORE_CU_H__
#define __CORE_CU_H__
#define OS_MAP 1

//#if __CUDACC__
//#define OMIT_AUTOINIT
//#endif

#include <Runtime.h>
#include <RuntimeTypes.h>

#if defined(__GNUC__) && 0
#define likely(X) __builtin_expect((X),1)
#define unlikely(X) __builtin_expect((X),0)
#else
#define likely(X) !!(X)
#define unlikely(X) !!(X)
#endif

#define _dprintf printf
#include "RC.cu.h"
#include "SysEx.cu.h"
#include "Bitvec.cu.h"
#include "VSystem.cu.h"
#include "IO\IO.VFile.cu.h"
using namespace Core;
using namespace Core::IO;

#endif // __CORE_CU_H__