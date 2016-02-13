#include "Runtime.h"
RUNTIME_NAMEBEGIN

#ifndef RUNTIME_DEFAULT_MEMSTATUS
#define RUNTIME_DEFAULT_MEMSTATUS false
#endif

	// The following singleton contains the global configuration for the SQLite library.
	__device__ _WSD TagBase::RuntimeStatics g_RuntimeStatics =
{
	false,						// CoreMutex
	THREADSAFE == 1,			// FullMutex
	{ nullptr, nullptr },		// AppendFormat
	//
	RUNTIME_DEFAULT_MEMSTATUS,	// Memstat
	true,						// RuntimeMutex
	nullptr,					// Mem
	nullptr,					// Mutex
	128,						// LookasideSize
	500,						// Lookasides
	(void *)nullptr,			// Scratch
	0,							// ScratchSize
	0,							// Scratchs
};

__device__ bool TagBase::SetupLookaside(void *buf, int size, int count)
{
	if (Lookaside.Outs)
		return false;
	// Free any existing lookaside buffer for this handle before allocating a new one so we don't have to have space for both at the same time.
	if (Lookaside.Malloced)
		_free(Lookaside.Start);
	// The size of a lookaside slot after ROUNDDOWN8 needs to be larger than a pointer to be useful.
	size = _ROUNDDOWN8(size); // IMP: R-33038-09382
	if (size <= (int)sizeof(TagBase::LookasideSlot *)) size = 0;
	if (count < 0) count = 0;
	void *start;
	if (size == 0 || count == 0)
	{
		size = 0;
		start = nullptr;
	}
	else if (!buf)
	{
		_benignalloc_begin();
		start = _alloc(size * count); // IMP: R-61949-35727
		_benignalloc_end();
		if (start) count = (int)_allocsize(start) / size;
	}
	else
		start = buf;
	Lookaside.Start = start;
	Lookaside.Free = nullptr;
	Lookaside.Size = (uint16)size;
	if (start)
	{
		_assert(size > (int)sizeof(TagBase::LookasideSlot *));
		TagBase::LookasideSlot *p = (TagBase::LookasideSlot *)start;
		for (int i = count - 1; i >= 0; i--)
		{
			p->Next = Lookaside.Free;
			Lookaside.Free = p;
			p = (TagBase::LookasideSlot *)&((uint8 *)p)[size];
		}
		Lookaside.End = p;
		Lookaside.Enabled = true;
		Lookaside.Malloced = (!buf);
	}
	else
	{
		Lookaside.End = nullptr;
		Lookaside.Enabled = false;
		Lookaside.Malloced = false;
	}
	return true;
}

#ifndef OMIT_BLOB_LITERAL
__device__ void *_taghextoblob(TagBase *tag, const char *z, size_t size)
{
	char *b = (char *)_tagalloc(tag, size / 2 + 1);
	size--;
	if (b)
	{
		int bIdx = 0;
		for (int i = 0; i < size; i += 2, bIdx++)
			b[bIdx] = (_hextobyte(z[i]) << 4) | _hextobyte(z[i + 1]);
		b[bIdx] = 0;
	}
	return b;
}
#endif

RUNTIME_NAMEEND
