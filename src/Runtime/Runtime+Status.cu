// status.c
#include "Runtime.h"

__device__ static _WSD struct Status
{
	int NowValue[10]; // Current value
	int MaxValue[10]; // Maximum value
} g_status = { {0,}, {0,} };
#ifndef OMIT_WSD
#define _status_Init
#define _stat2 g_status
#else
#define _status_Init Status *x = &_GLOBAL(Status, g_status)
#define _stat2 x[0]
#endif

__device__ int _status_value(STATUS op)
{
	_status_Init;
	_assert(op < _lengthof(_stat2.NowValue));
	return _stat2.NowValue[op];
}

__device__ void _status_add(STATUS op, int n)
{
	_status_Init;
	_assert(op < _lengthof(_stat2.NowValue));
	_stat2.NowValue[op] += n;
	if (_stat2.NowValue[op] > _stat2.MaxValue[op])
		_stat2.MaxValue[op] = _stat2.NowValue[op];
}

__device__ void _status_set(STATUS op, int x)
{
	_status_Init;
	_assert(op < _lengthof(_stat2.NowValue));
	_stat2.NowValue[op] = x;
	if (_stat2.NowValue[op] > _stat2.MaxValue[op])
		_stat2.MaxValue[op] = _stat2.NowValue[op];
}

__device__ bool _status(STATUS op, int *current, int *highwater, bool resetFlag)
{
	_status_Init;
	if (op >= _lengthof(_stat2.NowValue))
		return false;
	*current = _stat2.NowValue[op];
	*highwater = _stat2.MaxValue[op];
	if (resetFlag)
		_stat2.MaxValue[op] = _stat2.NowValue[op];
	return true;
}
