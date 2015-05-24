#ifndef __CORE_CU_H__
#define __CORE_CU_H__

#include <Runtime.h>
#include <RuntimeTypes.h>

#ifndef CORE_NAME
#define CORE_NAME			Core
#endif
#define CORE_VERSION		"--VERS--"
#define CORE_VERSION_NUMBER	3007016
#define CORE_SOURCE_ID		"--SOURCE-ID--"

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
#include "VSystem.cu.h"
#include "VFile.cu.h"
using namespace CORE_NAME;
#if OS_MAP
#include "VSystem+Sentinel.cu.h"
#endif

#endif // __CORE_CU_H__