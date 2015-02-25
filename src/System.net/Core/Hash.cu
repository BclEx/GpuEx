//hash.c
#include "Core.cu.h"

namespace Core
{
	__device__ Hash::Hash()
	{
		Init();
	}

	__device__ void Hash::Init()
	{
		First = nullptr;
		Count = 0;
		TableSize = 0;
		Table = nullptr;
	}

	__device__ void Hash::Clear()
	{
		HashElem *elem = First; // For looping over all elements of the table
		First = nullptr;
		_free(Table); Table = nullptr;
		TableSize = 0;
		while (elem)
		{
			HashElem *nextElem = elem->Next;
			_free(elem);
			elem = nextElem;
		}
		Count = 0;
	}

	__device__ static unsigned int GetHashCode(const char *key, int keyLength)
	{
		_assert(keyLength >= 0);
		int h = 0;
		while (keyLength > 0) { h = (h<<3) ^ h ^ _tolower(*key++); keyLength--; }
		return (unsigned int)h;
	}

	__device__ static void InsertElement(Hash *hash, Hash::HTable *entry, HashElem *newElem)
	{
		HashElem *headElem; // First element already in entry
		if (entry)
		{
			headElem = (entry->Count ? entry->Chain : nullptr);
			entry->Count++;
			entry->Chain = newElem;
		}
		else
			headElem = nullptr;
		if (headElem)
		{
			newElem->Next = headElem;
			newElem->Prev = headElem->Prev;
			if (headElem->Prev) headElem->Prev->Next = newElem;
			else hash->First = newElem;
			headElem->Prev = newElem;
		}
		else
		{
			newElem->Next = hash->First;
			if (hash->First) hash->First->Prev = newElem;
			newElem->Prev = nullptr;
			hash->First = newElem;
		}
	}

	__device__ static bool Rehash(Hash *hash, unsigned int newSize)
	{
#if MALLOC_SOFT_LIMIT > 0
		if (newSize * sizeof(Hash::HTable) > MALLOC_SOFT_LIMIT)
			newSize = MALLOC_SOFT_LIMIT / sizeof(Hash::HTable);
		if (newSize == hash->TableSize) return false;
#endif

		// The inability to allocates space for a larger hash table is a performance hit but it is not a fatal error.  So mark the
		// allocation as a benign. Use sqlite3Malloc()/memset(0) instead of sqlite3MallocZero() to make the allocation, as sqlite3MallocZero()
		// only zeroes the requested number of bytes whereas this module will use the actual amount of space allocated for the hash table (which
		// may be larger than the requested amount).
		_benignalloc_begin();
		Hash::HTable *newTable = (Hash::HTable *)_alloc(newSize * sizeof(Hash::HTable)); // The new hash table
		_benignalloc_end();
		if (!newTable)
			return false;
		_free(hash->Table);
		hash->Table = newTable;
		hash->TableSize = newSize = (int)_allocsize(newTable) / sizeof(Hash::HTable);
		_memset(newTable, 0, newSize * sizeof(Hash::HTable));
		HashElem *elem, *nextElem;
		for (elem = hash->First, hash->First = nullptr; elem; elem = nextElem)
		{
			unsigned int h = GetHashCode(elem->Key, elem->KeyLength) % newSize;
			nextElem = elem->Next;
			InsertElement(hash, &newTable[h], elem);
		}
		return true;
	}

	__device__ static HashElem *FindElementGivenHash(const Hash *hash, const char *key, int keyLength, unsigned int h)
	{
		HashElem *elem; // Used to loop thru the element list
		int count; // Number of elements left to test
		if (hash->Table)
		{
			Hash::HTable *entry = &hash->Table[h];
			elem = entry->Chain;
			count = entry->Count;
		}
		else
		{
			elem = hash->First;
			count = hash->Count;
		}
		while (count-- && _ALWAYS(elem))
		{
			if (elem->KeyLength == keyLength && !_strncmp(elem->Key, key, keyLength))
				return elem;
			elem = elem->Next;
		}
		return nullptr;
	}

	__device__ static void RemoveElementGivenHash(Hash *hash, HashElem *elem,  unsigned int h)
	{
		if (elem->Prev)
			elem->Prev->Next = elem->Next; 
		else
			hash->First = elem->Next;
		if (elem->Next)
			elem->Next->Prev = elem->Prev;
		if (hash->Table)
		{
			Hash::HTable *entry = &hash->Table[h];
			if (entry->Chain == elem)
				entry->Chain = elem->Next;
			entry->Count--;
			_assert(entry->Count >= 0);
		}
		_free(elem);
		hash->Count--;
		if (hash->Count == 0)
		{
			_assert(hash->First == nullptr);
			_assert(hash->Count == 0);
			hash->Clear();
		}
	}

	__device__ void *Hash::Find(const char *key, int keyLength)
	{
		_assert(key != nullptr);
		_assert(keyLength >= 0);
		unsigned int h = (Table ? GetHashCode(key, keyLength) % TableSize : 0);
		HashElem *elem = FindElementGivenHash(this, key, keyLength, h);
		return (elem ? elem->Data : nullptr);
	}

	__device__ void *Hash::Insert(const char *key, int keyLength, void *data)
	{
		_assert(key != nullptr);
		_assert(keyLength >= 0);
		unsigned int h = (TableSize ? GetHashCode(key, keyLength) % TableSize : 0); // the hash of the key modulo hash table size
		HashElem *elem = FindElementGivenHash(this, key, keyLength, h); // Used to loop thru the element list
		if (elem)
		{
			void *oldData = elem->Data;
			if (data == nullptr)
				RemoveElementGivenHash(this, elem, h);
			else
			{
				elem->Data = data;
				elem->Key = key;
				_assert(keyLength == elem->KeyLength);
			}
			return oldData;
		}
		if (data == nullptr)
			return nullptr;
		HashElem *newElem = (HashElem *)_alloc(sizeof(HashElem));
		if (newElem == nullptr)
			return nullptr;
		newElem->Key = key;
		newElem->KeyLength = keyLength;
		newElem->Data = data;
		Count++;
		if (Count >= 10 && Count > 2 * TableSize)
		{
			if (Rehash(this, Count * 2))
			{
				_assert(TableSize > 0);
				h = GetHashCode(key, keyLength) % TableSize;
			}
		}
		InsertElement(this, (Table ? &Table[h] : nullptr), newElem);
		return nullptr;
	}
}
