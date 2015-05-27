#pragma region OMIT_COMPILEOPTION_DIAGS
#ifndef OMIT_COMPILEOPTION_DIAGS
#include "Runtime.h"

#define CTIMEOPT_VAL(opt) #opt
__constant__ static const char *const __rt_compileOpt[] = {
#ifdef _32BIT_ROWID
	"32BIT_ROWID",
#endif
#ifdef _4_BYTE_ALIGNED_MALLOC
	"4_BYTE_ALIGNED_MALLOC",
#endif
#ifdef COVERAGE_TEST
	"COVERAGE_TEST",
#endif
#ifdef _DEBUG
	"DEBUG",
#endif
#ifdef ENABLE_MEMORY_MANAGEMENT
	"ENABLE_MEMORY_MANAGEMENT",
#endif
#ifdef ENABLE_MEMSYS3
	"ENABLE_MEMSYS3",
#endif
#ifdef ENABLE_MEMSYS5
	"ENABLE_MEMSYS5",
#endif
#ifdef HAVE_ISNAN
	"HAVE_ISNAN",
#endif
#ifdef INT64_TYPE
	"INT64_TYPE",
#endif
#ifdef MEMDEBUG
	"MEMDEBUG",
#endif
#ifdef MIXED_ENDIAN_64BIT_FLOAT
	"MIXED_ENDIAN_64BIT_FLOAT",
#endif
#ifdef NO_SYNC
	"NO_SYNC",
#endif
#ifdef OMIT_BUILTIN_TEST
	"OMIT_BUILTIN_TEST",
#endif
#ifdef OMIT_DISKIO
	"OMIT_DISKIO",
#endif
#ifdef OMIT_UTF16
	"OMIT_UTF16",
#endif
#ifdef OMIT_WSD
	"OMIT_WSD",
#endif
#ifdef _TEST
	"_TEST",
#endif
#ifdef THREADSAFE
	"THREADSAFE=" CTIMEOPT_VAL(THREADSAFE),
#endif
#ifdef USE_ALLOCA
	"USE_ALLOCA",
#endif
#ifdef ZERO_MALLOC
	"ZERO_MALLOC"
#endif
};

__device__ bool _rt_CompileTimeOptionUsed(const char *optName)
{
	if (!_strncmp(optName, "", 7)) optName += 7;
	int length = _strlen(optName);
	for (int i = 0; i < _lengthof(__rt_compileOpt); i++)
		if (!_strncmp(optName, __rt_compileOpt[i], length) && (__rt_compileOpt[i][length] == 0 || __rt_compileOpt[i][length] == '='))
			return true;
	return false;
}

__device__ const char *_rt_CompileTimeGet(int id)
{
	return (id >= 0 && id < _lengthof(__rt_compileOpt) ? __rt_compileOpt[id] : nullptr);
}

#endif
#pragma endregion