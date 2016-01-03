// The code in this file contains sample implementations of the sqlite3_wsd_init() and sqlite3_wsd_find() functions required if the
// SQLITE_OMIT_WSD symbol is defined at build time.

#if defined(OMIT_WSD) && defined(_TEST)
#include <Core+Vdbe\VdbeInt.cu.h>

#define PLS_HASHSIZE 43

typedef struct ProcessLocalStorage ProcessLocalStorage;
typedef struct ProcessLocalVar ProcessLocalVar;

struct ProcessLocalStorage
{
	ProcessLocalVar *Datas[PLS_HASHSIZE];
	int FreeLength;
	uint8 *Free;
};

struct ProcessLocalVar
{
	void *Key;
	ProcessLocalVar *Next;
};

__device__ static ProcessLocalStorage *_global = nullptr;

__device__ int sqlite3_wsd_init(int n, int j)
{
	if (!_global)
	{
		int malloc = n + sizeof(ProcessLocalStorage) + j*sizeof(ProcessLocalVar);
		_global = (ProcessLocalStorage *)_alloc(malloc);
		if (_global)
		{
			_memset(_global, 0, sizeof(ProcessLocalStorage));
			_global->FreeLength = malloc - sizeof(ProcessLocalStorage);
			_global->Free = (uint8 *)&_global[1];
		}
	}

	return (_global ? RC_OK : RC_NOMEM);
}

__device__ void *sqlite3_wsd_find(void *k, int l)
{
	// Calculate a hash of K
	int hash = 0;
	for (int i = 0; i < sizeof(void *); i++)
		hash = (hash<<3) + ((unsigned char *)&k)[i];
	hash = hash%PLS_HASHSIZE;

	// Search the hash table for K.
	ProcessLocalVar *var;
	for (var = _global->Datas[hash]; var && var->Key != k; var = var->Next);
	if (!var) // If no entry for K was found, create and populate a new one.
	{
		int bytes = _ROUND8(sizeof(ProcessLocalVar) + l);
		_assert(_global->FreeLength >= bytes);
		var = (ProcessLocalVar *)_global->Free;
		var->Key = k;
		var->Next = _global->Datas[hash];
		_global->Datas[hash] = var;
		_global->FreeLength -= bytes;
		_global->Free += bytes;
		_memcpy(&var[1], k, l);
	}
	return (void *)&var[1];
}

#endif
