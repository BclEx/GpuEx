// status.c
#include "Runtime.h"

__device__ static _WSD struct Status
{
	int NowValue[10]; // Current value
	int MaxValue[10]; // Maximum value
} g_status = { {0,}, {0,} };
#ifndef OMIT_WSD
#define _status_Init
#define _stat g_status
#else
#define _status_Init Status *x = &_GLOBAL(Status, g_status)
#define _stat x[0]
#endif

__device__ int _status_value(STATUS op)
{
	_status_Init;
	_assert(op < _lengthof(_stat.NowValue));
	return _stat.NowValue[op];
}

__device__ void _status_add(STATUS op, int n)
{
	_status_Init;
	_assert(op < _lengthof(_stat.NowValue));
	_stat.NowValue[op] += n;
	if (_stat.NowValue[op] > _stat.MaxValue[op])
		_stat.MaxValue[op] = _stat.NowValue[op];
}

__device__ void _status_set(STATUS op, int x)
{
	_status_Init;
	_assert(op < _lengthof(_stat.NowValue));
	_stat.NowValue[op] = x;
	if (_stat.NowValue[op] > _stat.MaxValue[op])
		_stat.MaxValue[op] = _stat.NowValue[op];
}

__device__ bool _status(STATUS op, int *current, int *highwater, bool resetFlag)
{
	_status_Init;
	if (op >= _lengthof(_stat.NowValue))
		return false;
	*current = _stat.NowValue[op];
	*highwater = _stat.MaxValue[op];
	if (resetFlag)
		_stat.MaxValue[op] = _stat.NowValue[op];
	return true;
}
