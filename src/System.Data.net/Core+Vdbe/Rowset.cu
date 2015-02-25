#include "Core+Vdbe.cu.h"

namespace Core
{
	struct RowSetEntry
	{            
		int64 V;                    // ROWID value for this entry
		RowSetEntry *Right;	// Right subtree (larger entries) or list
		RowSetEntry *Left;	// Left subtree (smaller entries)
	};

#define ROWSET_ALLOCATION_SIZE 1024
#define ROWSET_ENTRY_PER_CHUNK ((ROWSET_ALLOCATION_SIZE-8)/sizeof(RowSetEntry))

	struct RowSetChunk
	{
		RowSetChunk *NextChunk; // Next chunk on list of them all
		RowSetEntry Entrys[ROWSET_ENTRY_PER_CHUNK]; // Allocated entries
	};

	enum ROWSET : uint8
	{
		ROWSET_SORTED = 0x01,   // True if RowSet.pEntry is sorted
		ROWSET_NEXT = 0x02,		// True if sqlite3RowSetNext() has been called
	};
	__device__ inline void operator|=(ROWSET &a, int b) { a = (ROWSET)(a | b); }
	__device__ inline void operator&=(ROWSET &a, int b) { a = (ROWSET)(a & b); }

	struct RowSet
	{
		RowSetChunk *Chunk;			// List of all chunk allocations
		Context *Ctx;				// The database connection
		RowSetEntry *Entry;			// List of entries using pRight
		RowSetEntry *Last;			// Last entry on the pEntry list
		array_t<RowSetEntry> Fresh; // Source of new entry objects
		RowSetEntry *Forest;		// List of binary trees of entries
		ROWSET Flags;               // Various flags
		uint8 Batch;                // Current insert batch
	};

	__device__ RowSet *RowSet_Init(Context *ctx, void *space, unsigned int n)
	{
		_assert(n >= _ROUND8(sizeof(RowSet **)));
		RowSet *p = (RowSet *)space;
		p->Chunk = nullptr;
		p->Ctx = ctx;
		p->Entry = nullptr;
		p->Last = nullptr;
		p->Forest = nullptr;
		p->Fresh = (struct RowSetEntry *)(_ROUND8(sizeof(*p)) + (char *)p);
		p->Fresh.length = (uint16)((n - _ROUND8(sizeof(*p))) / sizeof(RowSetEntry));
		p->Flags = ROWSET_SORTED;
		p->Batch = 0;
		return p;
	}

	__device__ void RowSet_Clear(RowSet *p)
	{
		struct RowSetChunk *chunk, *nextChunk;
		for (chunk = p->Chunk; chunk; chunk = nextChunk)
		{
			nextChunk = chunk->NextChunk;
			_tagfree(p->Ctx, chunk);
		}
		p->Chunk = nullptr;
		p->Fresh.length = 0;
		p->Entry = nullptr;
		p->Last = nullptr;
		p->Forest = nullptr;
		p->Flags = ROWSET_SORTED;
	}

	__device__ static RowSetEntry *RowSetEntryAlloc(RowSet *p)
	{
		_assert(p);
		if (!p->Fresh.length)
		{
			RowSetChunk *newChunk = (RowSetChunk *)_tagalloc(p->Ctx, sizeof(*newChunk));
			if (!newChunk)
				return nullptr;
			newChunk->NextChunk = p->Chunk;
			p->Chunk = newChunk;
			p->Fresh = newChunk->Entrys;
			p->Fresh.length = ROWSET_ENTRY_PER_CHUNK;
		}
		p->Fresh.length--;
		return p->Fresh.data++;
	}

	__device__ void RowSet_Insert(RowSet *p, int64 rowid)
	{
		// This routine is never called after sqlite3RowSetNext()
		_assert(p && (p->Flags & ROWSET_NEXT) == 0);

		RowSetEntry *entry = RowSetEntryAlloc(p); // The new entry
		if (!entry) return;
		entry->V = rowid;
		entry->Right = nullptr;
		RowSetEntry *last = p->Last; // The last prior entry
		if (last)
		{
			if ((p->Flags & ROWSET_SORTED) != 0 && rowid <= last->V)
				p->Flags &= ~ROWSET_SORTED;
			last->Right = entry;
		}
		else
			p->Entry = entry;
		p->Last = entry;
	}

	__device__ static RowSetEntry *RowSetEntryMerge(RowSetEntry *a, RowSetEntry *b)
	{
		RowSetEntry head;
		RowSetEntry *tail = &head;
		while (a && b)
		{
			_assert(!a->Right || a->V <= a->Right->V);
			_assert(!b->Right || b->V <= b->Right->V);
			if (a->V < b->V)
			{
				tail->Right = a;
				a = a->Right;
				tail = tail->Right;
			}
			else if (b->V < a->V)
			{
				tail->Right = b;
				b = b->Right;
				tail = tail->Right;
			}
			else
				a = a->Right;
		}
		if (a)
		{
			_assert(!a->Right || a->V <= a->Right->V);
			tail->Right = a;
		}
		else
		{
			_assert(!b || !b->Right || b->V <= b->Right->V);
			tail->Right = b;
		}
		return head.Right;
	}

	__device__ static RowSetEntry *RowSetEntrySort(RowSetEntry *p)
	{
		unsigned int i;
		RowSetEntry *next, *buckets[40];
		_memset(buckets, 0, sizeof(buckets));
		while (p)
		{
			next = p->Right;
			p->Right = nullptr;
			for (i = 0; buckets[i]; i++)
			{
				p = RowSetEntryMerge(buckets[i], p);
				buckets[i] = nullptr;
			}
			buckets[i] = p;
			p = next;
		}
		p = nullptr;
		for (i = 0; i < _lengthof(buckets); i++)
			p = RowSetEntryMerge(p, buckets[i]);
		return p;
	}

	__device__ static void RowSetTreeToList(RowSetEntry *parent, RowSetEntry **first, RowSetEntry **last)
	{
		_assert(parent);
		if (parent->Left)
		{
			RowSetEntry *p;
			RowSetTreeToList(parent->Left, first, &p);
			p->Right = parent;
		}
		else
			*first = parent;
		if (parent->Right)
			RowSetTreeToList(parent->Right, &parent->Right, last);
		else
			*last = parent;
		_assert(!(*last)->Right);
	}

	__device__ static RowSetEntry *RowSetNDeepTree(RowSetEntry **list, int depth)
	{
		if (!*list)
			return nullptr;
		RowSetEntry *p; // Root of the new tree
		RowSetEntry *left; // Left subtree
		if (depth == 1)
		{
			p = *list;
			*list = p->Right;
			p->Left = p->Right = nullptr;
			return p;
		}
		left = RowSetNDeepTree(list, depth - 1);
		p = *list;
		if (!p)
			return left;
		p->Left = left;
		*list = p->Right;
		p->Right = RowSetNDeepTree(list, depth - 1);
		return p;
	}

	__device__ static RowSetEntry *RowSetListToTree(RowSetEntry *list)
	{
		_assert(list);
		RowSetEntry *p = list; // Current tree root
		list = p->Right;
		p->Left = p->Right = nullptr;
		for (int depth = 1; list; depth++)
		{
			RowSetEntry *left = p; // Left subtree
			p = list;
			list = p->Right;
			p->Left = left;
			p->Right = RowSetNDeepTree(&list, depth);
		}
		return p;
	}

	__device__ static void RowSetToList(RowSet *p)
	{
		// This routine is called only once
		_assert(p && (p->Flags & ROWSET_NEXT) == 0);
		if ((p->Flags & ROWSET_SORTED) == 0)
			p->Entry = RowSetEntrySort(p->Entry);
		// While this module could theoretically support it, sqlite3RowSetNext() is never called after sqlite3RowSetText() for the same RowSet.  So
		// there is never a forest to deal with.  Should this change, simply remove the assert() and the #if 0.
		_assert(!p->Forest);
#if 0
		while (p->Forest)
		{
			RowSetEntry *tree = p->Forest->Left;
			if (tree)
			{
				RowSetEntry *head, *tail;
				RowSetTreeToList(tree, &head, &tail);
				p->Entry = RowSetEntryMerge(p->Entry, head);
			}
			p->Forest = p->Forest->Right;
		}
#endif
		p->Flags |= ROWSET_NEXT; // Verify this routine is never called again
	}

	__device__ bool RowSet_Next(RowSet *p, int64 *rowid)
	{
		_assert(p);

		// Merge the forest into a single sorted list on first call
		if ((p->Flags & ROWSET_NEXT) == 0) RowSetToList(p);

		// Return the next entry on the list
		if (p->Entry)
		{
			*rowid = p->Entry->V;
			p->Entry = p->Entry->Right;
			if (!p->Entry)
				RowSet_Clear(p);
			return true;
		}
		return false;
	}

	__device__ bool RowSet_Test(RowSet *rowSet, uint8 batch, int64 rowid)
	{
		// This routine is never called after sqlite3RowSetNext()
		_assert(rowSet && (rowSet->Flags & ROWSET_NEXT) == 0);

		// Sort entries into the forest on the first test of a new batch 
		RowSetEntry *p, *tree;
		if (batch != rowSet->Batch)
		{
			p = rowSet->Entry;
			if (p)
			{
				RowSetEntry **prevTree = &rowSet->Forest;
				if ((rowSet->Flags & ROWSET_SORTED) == 0)
					p = RowSetEntrySort(p);
				for (tree = rowSet->Forest; tree; tree = tree->Right)
				{
					prevTree = &tree->Right;
					if (!tree->Left)
					{
						tree->Left = RowSetListToTree(p);
						break;
					}
					else
					{
						RowSetEntry *aux, *tail;
						RowSetTreeToList(tree->Left, &aux, &tail);
						tree->Left = nullptr;
						p = RowSetEntryMerge(aux, p);
					}
				}
				if (!tree)
				{
					*prevTree = tree = RowSetEntryAlloc(rowSet);
					if (tree)
					{
						tree->V = 0;
						tree->Right = nullptr;
						tree->Left = RowSetListToTree(p);
					}
				}
				rowSet->Entry = nullptr;
				rowSet->Last = nullptr;
				rowSet->Flags |= ROWSET_SORTED;
			}
			rowSet->Batch = batch;
		}

		// Test to see if the iRowid value appears anywhere in the forest. Return 1 if it does and 0 if not.
		for (tree = rowSet->Forest; tree; tree = tree->Right)
		{
			p = tree->Left;
			while (p)
			{
				if (p->V < rowid) p = p->Right;
				else if( p->V > rowid) p = p->Left;
				else return true;
			}
		}
		return false;
	}

}