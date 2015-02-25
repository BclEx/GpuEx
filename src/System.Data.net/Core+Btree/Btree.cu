// btree.cBtreeOpen
#include "Core+Btree.cu.h"
#include "BtreeInt.cu.h"
#include <stddef.h>

namespace Core
{
#if _DEBUG
	__device__ bool BtreeTrace = true;
#define TRACE(X, ...) if (BtreeTrace) { _fprintf(nullptr, X, __VA_ARGS__); }
#else
#define TRACE(X, ...)
#endif

#pragma region Struct

	__device__ static const char _magicHeader[] = FILE_HEADER;
	__device__ static IVdbe *_vdbe;

	enum BTALLOC : uint8
	{
		ANY = 0,        // Allocate any page
		EXACT = 1,      // Allocate exact page if possible
		LE = 2,         // Allocate any page <= the parameter
	};

#ifndef OMIT_AUTOVACUUM
#define IFAUTOVACUUM(x) (x)
#else
#define IFAUTOVACUUM(x) false
#endif

#ifndef OMIT_SHARED_CACHE
	__device__ BtShared *_WSD g_sharedCacheList = nullptr;
#define _sharedCacheList _GLOBAL(BtShared *, g_sharedCacheList)
	__device__ int enable_shared_cache(bool enable)
	{
		SysEx_GlobalStatics.SharedCacheEnabled = enable;
		return RC_OK;
	}
#else
#define QuerySharedCacheTableLock(a,b,c) RC_OK
#define SetSharedCacheTableLock(a,b,c) RC_OK
#define ClearAllSharedCacheTableLocks(a)
#define DowngradeAllSharedCacheTableLocks(a)
#define HasSharedCacheTableLock(a,b,c,d) 1
#define HasReadConflicts(a, b) 0
#endif

#pragma endregion

#pragma region Shared Code1
#ifndef OMIT_SHARED_CACHE

#ifdef _DEBUG
	__device__ static bool HasSharedCacheTableLock(Btree *btree, Pid root, bool isIndex, LOCK lockType)
	{
		// If this database is not shareable, or if the client is reading and has the read-uncommitted flag set, then no lock is required. 
		// Return true immediately.
		if (!btree->Sharable_ || (lockType == LOCK_READ && (btree->Ctx->Flags & BContext::FLAG::FLAG_ReadUncommitted)))
			return true;

		// If the client is reading  or writing an index and the schema is not loaded, then it is too difficult to actually check to see if
		// the correct locks are held.  So do not bother - just return true. This case does not come up very often anyhow.
		Schema *schema = btree->Bt->Schema;
		if (isIndex && (!schema || (schema->Flags & SCHEMA_SchemaLoaded) == 0))
			return true;

		// Figure out the root-page that the lock should be held on. For table b-trees, this is just the root page of the b-tree being read or
		// written. For index b-trees, it is the root page of the associated table.
		Pid table = 0;
		if (isIndex)
			return false;
		//for (HashElem *p = sqliteHashFirst(&schema->IdxHash); p; p = sqliteHashNext(p))
		//{
		//	Index *idx = (Index *)sqliteHashData(p);
		//	if (idx->TID == (int)root)
		//		table = idx->Table->TID;
		//}
		else
			table = root;

		// Search for the required lock. Either a write-lock on root-page iTab, a write-lock on the schema table, or (if the client is reading) a
		// read-lock on iTab will suffice. Return 1 if any of these are found.
		for (BtLock *lock = btree->Bt->Lock; lock; lock = lock->Next)
			if (lock->Btree == btree && 
				(lock->Table == table || (lock->Lock == LOCK_WRITE && lock->Table == 1)) &&
				lock->Lock >= lockType)
				return true;

		// Failed to find the required lock.
		return false;
	}

	__device__ static bool HasReadConflicts(Btree *btree, Pid root)
	{
		for (BtCursor *p = btree->Bt->Cursor; p; p = p->Next)
			if (p->RootID == root &&
				p->Btree != btree &&
				(p->Btree->Ctx->Flags & BContext::FLAG::FLAG_ReadUncommitted) == 0)
				return true;
		return false;
	}
#endif

	__device__ static RC QuerySharedCacheTableLock(Btree *p, Pid table, LOCK lock)
	{
		_assert(p->HoldsMutex());
		_assert(lock == LOCK_READ || lock == LOCK_WRITE);
		_assert(p->Ctx != nullptr);
		_assert(!(p->Ctx->Flags & BContext::FLAG::FLAG_ReadUncommitted) || lock == LOCK_WRITE || table == 1);

		// If requesting a write-lock, then the Btree must have an open write transaction on this file. And, obviously, for this to be so there 
		// must be an open write transaction on the file itself.
		BtShared *bt = p->Bt;
		_assert(lock == LOCK_READ || (p == bt->Writer && p->InTrans == TRANS_WRITE));
		_assert(lock == LOCK_READ || bt->InTransaction == TRANS_WRITE);

		// This routine is a no-op if the shared-cache is not enabled
		if (!p->Sharable_)
			return RC_OK;

		// If some other connection is holding an exclusive lock, the requested lock may not be obtained.
		if (bt->Writer != p && (bt->BtsFlags & BTS_EXCLUSIVE) != 0)
		{
			p->Ctx->ConnectionBlocked(bt->Writer->Ctx);
			return RC_LOCKED_SHAREDCACHE;
		}

		for (BtLock *iter = bt->Lock; iter; iter = iter->Next)
		{
			// The condition (pIter->eLock!=eLock) in the following if(...) statement is a simplification of:
			//
			//   (eLock==WRITE_LOCK || pIter->eLock==WRITE_LOCK)
			//
			// since we know that if eLock==WRITE_LOCK, then no other connection may hold a WRITE_LOCK on any table in this file (since there can
			// only be a single writer).
			_assert(iter->Lock == LOCK_READ || iter->Lock == LOCK_WRITE);
			_assert(lock == LOCK_READ || iter->Btree == p || iter->Lock == LOCK_READ);
			if (iter->Btree != p && iter->Table == table && iter->Lock != lock)
			{
				p->Ctx->ConnectionBlocked(iter->Btree->Ctx);
				if (lock == LOCK_WRITE)
				{
					_assert(p == bt->Writer);
					bt->BtsFlags |= BTS_PENDING;
				}
				return RC_LOCKED_SHAREDCACHE;
			}
		}
		return RC_OK;
	}

	__device__ static RC SetSharedCacheTableLock(Btree *p, Pid table, LOCK lock)
	{
		_assert(p->HoldsMutex());
		_assert(lock == LOCK_READ || lock == LOCK_WRITE);
		_assert(p->Ctx != nullptr);

		// A connection with the read-uncommitted flag set will never try to obtain a read-lock using this function. The only read-lock obtained
		// by a connection in read-uncommitted mode is on the sqlite_master table, and that lock is obtained in BtreeBeginTrans().
		_assert((p->Ctx->Flags & BContext::FLAG::FLAG_ReadUncommitted) == 0 || lock == LOCK_WRITE);

		// This function should only be called on a sharable b-tree after it has been determined that no other b-tree holds a conflicting lock.
		_assert(p->Sharable_);
		_assert(QuerySharedCacheTableLock(p, table, lock) == RC_OK);

		// First search the list for an existing lock on this table.
		BtShared *bt = p->Bt;
		BtLock *newLock = nullptr;
		for (BtLock *iter = bt->Lock; iter; iter = iter->Next)
		{
			if (iter->Table == table && iter->Btree == p)
			{
				newLock = iter;
				break;
			}
		}

		// If the above search did not find a BtLock struct associating Btree p with table iTable, allocate one and link it into the list.
		if (!newLock)
		{
			newLock = (BtLock *)_alloc2(sizeof(BtLock), true);
			if (!newLock)
				return RC_NOMEM;
			newLock->Table = table;
			newLock->Btree = p;
			newLock->Next = bt->Lock;
			bt->Lock = newLock;
		}

		// Set the BtLock.eLock variable to the maximum of the current lock and the requested lock. This means if a write-lock was already held
		// and a read-lock requested, we don't incorrectly downgrade the lock.
		_assert(LOCK_WRITE > LOCK_READ);
		if (lock > newLock->Lock)
			newLock->Lock = lock;

		return RC_OK;
	}

	__device__ static void ClearAllSharedCacheTableLocks(Btree *p)
	{
		BtShared *bt = p->Bt;
		BtLock **iter = &bt->Lock;

		_assert(p->HoldsMutex());
		_assert(p->Sharable_ || iter == nullptr);
		_assert((int)p->InTrans > 0);

		while (*iter)
		{
			BtLock *lock = *iter;
			_assert((bt->BtsFlags & BTS_EXCLUSIVE) == 0 || bt->Writer == lock->Btree);
			_assert((int)lock->Btree->InTrans >= (int)lock->Lock);
			if (lock->Btree == p)
			{
				*iter = lock->Next;
				_assert(lock->Table != 1 || lock == &p->Lock);
				if (lock->Table != 1)
					_free(lock);
			}
			else
				iter = &lock->Next;
		}

		_assert((bt->BtsFlags & BTS_PENDING) == 0 || bt->Writer);
		if (bt->Writer == p)
		{
			bt->Writer = 0;
			bt->BtsFlags &= ~(BTS_EXCLUSIVE | BTS_PENDING);
		}
		else if (bt->Transactions == 2)
		{
			// This function is called when Btree p is concluding its transaction. If there currently exists a writer, and p is not
			// that writer, then the number of locks held by connections other than the writer must be about to drop to zero. In this case
			// set the BTS_PENDING flag to 0.
			//
			// If there is not currently a writer, then BTS_PENDING must be zero already. So this next line is harmless in that case.
			bt->BtsFlags &= ~BTS_PENDING;
		}
	}

	__device__ static void DowngradeAllSharedCacheTableLocks(Btree *p)
	{
		BtShared *bt = p->Bt;
		if (bt->Writer == p)
		{
			bt->Writer = nullptr;
			bt->BtsFlags &= ~(BTS_EXCLUSIVE | BTS_PENDING);
			for (BtLock *lock = bt->Lock; lock; lock = lock->Next)
			{
				_assert(lock->Lock==LOCK_READ || lock->Btree == p);
				lock->Lock = LOCK_READ;
			}
		}
	}

#endif
#pragma endregion

#pragma region Name2

	__device__ static void ReleasePage(MemPage *page);
#ifdef _DEBUG
	__device__ static bool CursorHoldsMutex(BtCursor *p)
	{
		return MutexEx::Held(p->Bt->Mutex);
	}
#endif

#ifndef OMIT_INCRBLOB
	__device__ static void InvalidateOverflowCache(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_free(cur->Overflows);
		cur->Overflows = nullptr;
	}

	__device__ static void InvalidateAllOverflowCache(BtShared *bt)
	{
		_assert(MutexEx::Held(bt->Mutex));
		for (BtCursor *p = bt->Cursor; p; p = p->Next)
			InvalidateOverflowCache(p);
	}

	__device__ static void InvalidateIncrblobCursors(Btree *btree, int64 rowid, bool isClearTable)
	{
		BtShared *bt = btree->Bt;
		_assert(btree->HoldsMutex());
		for (BtCursor *p = bt->Cursor; p; p = p->Next)
			if (p->IsIncrblobHandle && (isClearTable || p->Info.Key == rowid))
				p->State = CURSOR_INVALID;
	}
#else
#define InvalidateOverflowCache(x)
#define InvalidateAllOverflowCache(x)
#define InvalidateIncrblobCursors(x,y,z)
#endif

#pragma endregion

#pragma region Name3

	__device__ static RC BtreeSetHasContent(BtShared *bt, Pid id)
	{
		RC rc = RC_OK;
		if (!bt->HasContent)
		{
			_assert(id <= bt->Pages);
			bt->HasContent = new Bitvec(bt->Pages);
			if (!bt->HasContent)
				rc = RC_NOMEM;
		}
		if (rc == RC_OK && id <= bt->HasContent->get_Length())
			rc = bt->HasContent->Set(id);
		return rc;
	}

	__device__ static bool BtreeGetHasContent(BtShared *bt, Pid id)
	{
		Bitvec *p = bt->HasContent;
		return (p && (id > p->get_Length() || p->Get(id)));
	}

	__device__ static void BtreeClearHasContent(BtShared *bt)
	{
		Bitvec::Destroy(bt->HasContent);
		bt->HasContent = nullptr;
	}

	__device__ static void BtreeReleaseAllCursorPages(BtCursor *cur)
	{
		for (int i = 0; i <= cur->ID; i++)
		{
			ReleasePage(cur->Pages[i]);
			cur->Pages[i] = nullptr;
		}
		cur->ID = -1;
	}

	__device__ static RC SaveCursorPosition(BtCursor *cur)
	{
		_assert(cur->State == CURSOR_VALID);
		_assert(cur->Key == nullptr);
		_assert(CursorHoldsMutex(cur));

		RC rc = Btree::KeySize(cur, &cur->KeyLength);
		_assert(rc == RC_OK);  // KeySize() cannot fail

		// If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur->nKey. In this case this value is
		// all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur->nKey bytes of key data.
		if (!cur->Pages[0]->IntKey)
		{
			void *key = _alloc((int)cur->KeyLength);
			if (key)
			{
				rc = Btree::Key(cur, 0, (int)cur->KeyLength, key);
				if (rc == RC_OK)
					cur->Key = key;
				else
					_free(key);
			}
			else
				rc = RC_NOMEM;
		}
		_assert(!cur->Pages[0]->IntKey || !cur->Key);

		if (rc == RC_OK)
		{
			BtreeReleaseAllCursorPages(cur);
			cur->State = CURSOR_REQUIRESEEK;
		}

		InvalidateOverflowCache(cur);
		return rc;
	}

	__device__ static RC SaveAllCursors(BtShared *bt, Pid root, BtCursor *except)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(except == nullptr || except->Bt == bt);
		for (BtCursor *p = bt->Cursor; p; p = p->Next)
		{
			if (p != except && (root == 0 || p->RootID == root))
				if (p->State == CURSOR_VALID)
				{
					RC rc = SaveCursorPosition(p);
					if (rc != RC_OK)
						return rc;
				}
				else
				{
					ASSERTCOVERAGE(p->Pages > 0);
					BtreeReleaseAllCursorPages(p);
				}
		}
		return RC_OK;
	}

	__device__ void Btree::ClearCursor(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_free(cur->Key);
		cur->Key = nullptr;
		cur->State = CURSOR_INVALID;
	}

	__device__ static RC BtreeMoveto(BtCursor *cur, const void *key, int64 keyLength, int bias, int *eof)
	{
		UnpackedRecord *idxKey; // Unpacked index key
		char space[150]; // Temp space for pIdxKey - to avoid a malloc
		char *free = nullptr;
		if (key)
		{
			_assert(keyLength == (int64)(int)keyLength);
			idxKey = _vdbe->AllocUnpackedRecord(cur->KeyInfo, space, sizeof(space), &free);
			if (idxKey == nullptr) return RC_NOMEM;
			_vdbe->RecordUnpack(cur->KeyInfo, (int)keyLength, (uint8 *)key, idxKey);
		}
		else
			idxKey = nullptr;
		RC rc = Btree::MovetoUnpacked(cur, idxKey, keyLength, bias, eof);
		if (free)
			_tagfree(cur->KeyInfo->Ctx, free);
		return rc;
	}

	__device__ static RC BtreeRestoreCursorPosition(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State >= CURSOR_REQUIRESEEK);
		if (cur->State == CURSOR_FAULT)
			return (RC)cur->SkipNext;
		cur->State = CURSOR_INVALID;
		RC rc = BtreeMoveto(cur, cur->Key, cur->KeyLength, 0, &cur->SkipNext);
		if (rc == RC_OK)
		{
			_free(cur->Key);
			cur->Key = nullptr;
			_assert(cur->State == CURSOR_VALID || cur->State == CURSOR_INVALID);
		}
		return rc;
	}

#define restoreCursorPosition(p) (p->State >= CURSOR_REQUIRESEEK ? BtreeRestoreCursorPosition(p) : RC_OK)

	__device__ RC Btree::CursorHasMoved(BtCursor *cur, bool *hasMoved)
	{
		RC rc = restoreCursorPosition(cur);
		if (rc)
		{
			*hasMoved = true;
			return rc;
		}
		*hasMoved = (cur->State != CURSOR_VALID || cur->SkipNext != 0);
		return RC_OK;
	}

#pragma endregion

#pragma region Parse Cell

#ifndef OMIT_AUTOVACUUM
	__device__ static Pid PtrmapPageId(BtShared *bt, Pid id)
	{
		_assert(MutexEx::Held(bt->Mutex));
		if (id < 2) return 0;
		int pagesPerMapPage = (bt->UsableSize / 5) + 1;
		Pid ptrMap = (id - 2) / pagesPerMapPage;
		Pid ret = (ptrMap * pagesPerMapPage) + 2; 
		if (ret == PENDING_BYTE_PAGE(bt))
			ret++;
		return ret;
	}

	__device__ static void PtrmapPut(BtShared *bt, Pid key, PTRMAP type, Pid parent, RC *rcRef)
	{
		if (*rcRef != RC_OK) return;

		_assert(MutexEx::Held(bt->Mutex));
		// The master-journal page number must never be used as a pointer map page
		_assert(!PTRMAP_ISPAGE(bt, PENDING_BYTE_PAGE(bt)));

		_assert(bt->AutoVacuum);
		if (key == 0)
		{
			*rcRef = SysEx_CORRUPT_BKPT;
			return;
		}
		Pid ptrmapIdx = PTRMAP_PAGENO(bt, key); // The pointer map page number
		IPage *page; // The pointer map page
		RC rc = bt->Pager->Acquire(ptrmapIdx, &page, false);
		if (rc != RC_OK)
		{
			*rcRef = rc;
			return;
		}
		uint8 *ptrmap;
		int offset = PTRMAP_PTROFFSET(ptrmapIdx, key); // Offset in pointer map page
		if (offset < 0)
		{
			*rcRef = SysEx_CORRUPT_BKPT;
			goto ptrmap_exit;
		}
		_assert(offset <= (int)bt->UsableSize - 5);
		ptrmap = (uint8 *)Pager::GetData(page);// The pointer map data

		if (type != (PTRMAP)ptrmap[offset] || ConvertEx::Get4(&ptrmap[offset + 1]) != parent)
		{
			TRACE("PTRMAP_UPDATE: %d->(%d,%d)\n", key, type, parent);
			*rcRef = rc = Pager::Write(page);
			if (rc == RC_OK)
			{
				ptrmap[offset] = (uint8)type;
				ConvertEx::Put4(&ptrmap[offset + 1], parent);
			}
		}

ptrmap_exit:
		Pager::Unref(page);
	}

	__device__ static RC PtrmapGet(BtShared *bt, Pid key, PTRMAP *type, Pid *id)
	{
		_assert(MutexEx::Held(bt->Mutex));

		IPage *page; // The pointer map page
		uint8 ptrmapIdx = PTRMAP_PAGENO(bt, key); // Pointer map page index
		RC rc = bt->Pager->Acquire(ptrmapIdx, &page, false);
		if (rc != RC_OK)
			return rc;
		uint8 *ptrmap = (uint8 *)Pager::GetData(page); // Pointer map page data

		int offset = PTRMAP_PTROFFSET(ptrmapIdx, key); // Offset of entry in pointer map
		if (offset < 0)
		{
			Pager::Unref(page);
			return SysEx_CORRUPT_BKPT;
		}
		_assert(offset <= (int)bt->UsableSize - 5);
		_assert(type != 0);
		*type = (PTRMAP)ptrmap[offset];
		if (id) *id = ConvertEx::Get4(&ptrmap[offset + 1]);

		Pager::Unref(page);
		if ((uint8)*type < 1 || (uint8)*type > 5) return SysEx_CORRUPT_BKPT;
		return RC_OK;
	}

#else
#define PtrmapPut(w,x,y,z,rc)
#define PtrmapGet(w,x,y,z) RC_OK
#define PtrmapPutOvflPtr(x, y, rc)
#endif

#define FindCell(P,I) ((P)->Data + ((P)->MaskPage & ConvertEx::Get2(&(P)->CellIdx[2*(I)])))
#define FindCellv2(D,M,O,I) (D + (M & ConvertEx::Get2(D+(O+2*(I)))))

	__device__ static uint8 *FindOverflowCell(MemPage *page, uint cell)
	{
		_assert(MutexEx::Held(page->Bt->Mutex));
		for (int i = page->Overflows - 1; i >= 0; i--)
		{
			uint16 k = page->OvflIdxs[i];
			if (k <= cell)
			{
				if (k == cell)
					return page->Ovfls[i];
				cell--;
			}
		}
		return FindCell(page, cell);
	}

	__device__ static void BtreeParseCellPtr(MemPage *page, uint8 *cell, CellInfo *info)
	{
		_assert(MutexEx::Held(page->Bt->Mutex));

		info->Cell = cell;
		uint16 n = page->ChildPtrSize; // Number bytes in cell content header
		_assert(n == 4 - 4 * page->Leaf);
		uint32 payloadLength; // Number of bytes of cell payload
		if (page->IntKey)
		{
			if (page->HasData)
				n += ConvertEx::GetVarint32(&cell[n], &payloadLength);
			else
				payloadLength = 0;
			n += ConvertEx::GetVarint(&cell[n], (uint64 *)&info->Key);
			info->Data = payloadLength;
		}
		else
		{
			info->Data = 0;
			n += ConvertEx::GetVarint32(&cell[n], &payloadLength);
			info->Key = payloadLength;
		}
		info->Payload = payloadLength;
		info->Header = n;
		ASSERTCOVERAGE(payloadLength == page->MaxLocal);
		ASSERTCOVERAGE(payloadLength == page->MaxLocal + 1);
		if (likely(payloadLength <= page->MaxLocal))
		{
			// This is the (easy) common case where the entire payload fits on the local page.  No overflow is required.
			if ((info->Size = (uint16)(n + payloadLength)) < 4) info->Size = 4;
			info->Local = (uint16)payloadLength;
			info->Overflow = 0;
		}
		else
		{
			// If the payload will not fit completely on the local page, we have to decide how much to store locally and how much to spill onto
			// overflow pages.  The strategy is to minimize the amount of unused space on overflow pages while keeping the amount of local storage
			// in between minLocal and maxLocal.
			//
			// Warning:  changing the way overflow payload is distributed in any way will result in an incompatible file format.
			int minLocal = page->MinLocal; // Minimum amount of payload held locally
			int maxLocal = page->MaxLocal; // Maximum amount of payload held locally
			int surplus = minLocal + (payloadLength - minLocal) % (page->Bt->UsableSize - 4); // Overflow payload available for local storage
			ASSERTCOVERAGE(surplus == maxLocal);
			ASSERTCOVERAGE(surplus == maxLocal + 1);
			if (surplus <= maxLocal)
				info->Local = (uint16)surplus;
			else
				info->Local = (uint16)minLocal;
			info->Overflow = (uint16)(info->Local + n);
			info->Size = info->Overflow + 4;
		}
	}

#define ParseCell(page, cell, info) BtreeParseCellPtr((page), FindCell((page), (cell)), (info))
	__device__ static void BtreeParseCell(MemPage *page, uint cell, CellInfo *info) { ParseCell(page, cell, info); }

	__device__ static uint16 CellSizePtr(MemPage *page, uint8 *cell)
	{
#ifdef _DEBUG
		// The value returned by this function should always be the same as the (CellInfo.nSize) value found by doing a full parse of the
		// cell. If SQLITE_DEBUG is defined, an assert() at the bottom of this function verifies that this invariant is not violated.
		CellInfo debuginfo;
		BtreeParseCellPtr(page, cell, &debuginfo);
#endif
		uint32 size;
		uint8 *iter = &cell[page->ChildPtrSize];
		if (page->IntKey)
		{
			if (page->HasData)
				iter += ConvertEx::GetVarint32(iter, &size);
			else
				size = 0;

			// pIter now points at the 64-bit integer key value, a variable length integer. The following block moves pIter to point at the first byte
			// past the end of the key value. */
			uint8 *end = &iter[9];
			while ((*iter++) & 0x80 && iter < end) { }
		}
		else
			iter += ConvertEx::GetVarint32(iter, &size);

		ASSERTCOVERAGE(size == page->MaxLocal);
		ASSERTCOVERAGE(size == page->MaxLocal + 1);
		if (size > page->MaxLocal)
		{
			int minLocal = page->MinLocal;
			size = minLocal + (size - minLocal) % (page->Bt->UsableSize - 4);
			ASSERTCOVERAGE(size == page->MaxLocal);
			ASSERTCOVERAGE(size == page->MaxLocal + 1);
			if (size > page->MaxLocal)
				size = minLocal;
			size += 4;
		}
		size += (uint32)(iter - cell);

		// The minimum size of any cell is 4 bytes.
		if (size < 4)
			size = 4;

		_assert(size == debuginfo.Size);
		return (uint16)size;
	}

#ifdef _DEBUG
	__device__ static uint16 CellSize(MemPage *page, int cell)
	{
		return CellSizePtr(page, FindCell(page, cell));
	}
#endif

#ifndef OMIT_AUTOVACUUM
	__device__ static void PtrmapPutOvflPtr(MemPage *page, uint8 *cell, RC *rcRef)
	{
		if (*rcRef != RC_OK) return;
		_assert(cell != nullptr);
		CellInfo info;
		BtreeParseCellPtr(page, cell, &info);
		_assert((info.Data + (page->IntKey ? 0 : info.Key)) == info.Payload);
		if (info.Overflow)
		{
			Pid ovfl = ConvertEx::Get4(&cell[info.Overflow]);
			PtrmapPut(page->Bt, ovfl, PTRMAP_OVERFLOW1, page->ID, rcRef);
		}
	}
#endif

#pragma endregion

#pragma region Allocate / Defragment

	__device__ static RC DefragmentPage(MemPage *page)
	{
		_assert(Pager::Iswriteable(page->DBPage));
		_assert(page->Bt != nullptr);
		_assert(page->Bt->UsableSize <= MAX_PAGE_SIZE);
		_assert(page->Overflows == 0);
		_assert(MutexEx::Held(page->Bt->Mutex) );
		unsigned char *temp = (unsigned char *)page->Bt->Pager->get_TempSpace(); // Temp area for cell content
		unsigned char *data = page->Data; // The page data
		int hdr = page->HdrOffset; // Offset to the page header
		int cellOffset = page->CellOffset; // Offset to the cell pointer array
		int cells = page->Cells; // Number of cells on the page
		_assert(cells == ConvertEx::Get2(&data[hdr+3]) );
		uint usableSize = page->Bt->UsableSize; // Number of usable bytes on a page
		uint cbrk = (uint)ConvertEx::Get2(&data[hdr+5]); // Offset to the cell content area
		_memcpy(&temp[cbrk], &data[cbrk], usableSize - cbrk);
		cbrk = usableSize;
		uint16 cellFirst = cellOffset + 2 * cells; // First allowable cell index
		uint16 cellLast = usableSize - 4; // Last possible cell index
		for (int i = 0; i < cells; i++)
		{
			uint8 *addr = &data[cellOffset + i*2]; // The i-th cell pointer
			uint pc = ConvertEx::Get2(addr); // Address of a i-th cell
			ASSERTCOVERAGE(pc == cellFirst);
			ASSERTCOVERAGE(pc == cellLast);
#if !defined(ENABLE_OVERSIZE_CELL_CHECK)
			// These conditions have already been verified in BtreeInitPage() if ENABLE_OVERSIZE_CELL_CHECK is defined
			if (pc < cellFirst || pc > cellLast)
				return SysEx_CORRUPT_BKPT;
#endif
			_assert(pc >= cellFirst && pc <= cellLast);
			uint size = CellSizePtr(page, &temp[pc]); // Size of a cell
			cbrk -= size;
#if defined(ENABLE_OVERSIZE_CELL_CHECK)
			if (cbrk < cellFirst)
				return SysEx_CORRUPT_BKPT;
#else
			if (cbrk < cellFirst || pc + size > usableSize)
				return SysEx_CORRUPT_BKPT;
#endif
			_assert(cbrk + size <= usableSize && cbrk >= cellFirst);
			ASSERTCOVERAGE(cbrk + size == usableSize);
			ASSERTCOVERAGE(pc + size == usableSize);
			_memcpy(&data[cbrk], &temp[pc], size);
			ConvertEx::Put2(addr, cbrk);
		}
		_assert(cbrk >= cellFirst);
		ConvertEx::Put2(&data[hdr + 5], cbrk);
		data[hdr + 1] = 0;
		data[hdr + 2] = 0;
		data[hdr + 7] = 0;
		_memset(&data[cellFirst], 0, cbrk - cellFirst);
		_assert(Pager::Iswriteable(page->DBPage));
		if (cbrk - cellFirst != page->Frees)
			return SysEx_CORRUPT_BKPT;
		return RC_OK;
	}

	__device__ static RC AllocateSpace(MemPage *page, int bytes, uint *idx)
	{
		_assert(Pager::Iswriteable(page->DBPage));
		_assert(page->Bt != nullptr);
		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(bytes >= 0);  // Minimum cell size is 4
		_assert(page->Frees >= bytes);
		_assert(page->Overflows == 0);
		int usableSize = page->Bt->UsableSize; // Usable size of the page
		_assert(bytes < usableSize - 8);

		int hdr = page->HdrOffset;  // Local cache of pPage.hdrOffset
		uint8 *const data = page->Data; // Local cache of pPage->aData
		int frags = data[hdr + 7]; // Number of fragmented bytes on pPage
		_assert(page->CellOffset == hdr + 12 - 4 * page->Leaf);
		uint gap = page->CellOffset + 2 * page->Cells; // First byte of gap between cell pointers and cell content
		uint top = (uint)ConvertEx::Get2nz(&data[hdr + 5]); // First byte of cell content area
		if (gap > top) return SysEx_CORRUPT_BKPT;
		ASSERTCOVERAGE(gap + 2 == top);
		ASSERTCOVERAGE(gap + 1 == top);
		ASSERTCOVERAGE(gap == top);

		RC rc;
		if (frags >= 60)
		{
			// Always defragment highly fragmented pages
			rc = DefragmentPage(page);
			if (rc) return rc;
			top = ConvertEx::Get2nz(&data[hdr + 5]);
		}
		else if (gap + 2 <= top)
		{
			// Search the freelist looking for a free slot big enough to satisfy the request. The allocation is made from the first free slot in 
			// the list that is large enough to accomadate it.
			int pc;
			for (int addr = hdr + 1; (pc = ConvertEx::Get2(&data[addr])) > 0; addr = pc)
			{
				if (pc > usableSize - 4 || pc < addr + 4)
					return SysEx_CORRUPT_BKPT;
				int size = ConvertEx::Get2(&data[pc+2]); // Size of the free slot
				if (size >= bytes)
				{
					int x = size - bytes;
					ASSERTCOVERAGE(x == 4);
					ASSERTCOVERAGE(x == 3);
					if (x < 4)
					{
						// Remove the slot from the free-list. Update the number of fragmented bytes within the page.
						_memcpy(&data[addr], &data[pc], 2);
						data[hdr + 7] = (uint8)(frags + x);
					}
					else if (size + pc > usableSize)
						return SysEx_CORRUPT_BKPT;
					else // The slot remains on the free-list. Reduce its size to account for the portion used by the new allocation.
						ConvertEx::Put2(&data[pc+2], x);
					*idx = pc + x;
					return RC_OK;
				}
			}
		}

		// Check to make sure there is enough space in the gap to satisfy the allocation.  If not, defragment.
		ASSERTCOVERAGE(gap + 2 + bytes == top);
		if (gap + 2 + bytes > top)
		{
			rc = DefragmentPage(page);
			if (rc) return rc;
			top = ConvertEx::Get2nz(&data[hdr + 5]);
			_assert(gap + bytes <= top);
		}

		// Allocate memory from the gap in between the cell pointer array and the cell content area.  The BtreeInitPage() call has already
		// validated the freelist.  Given that the freelist is valid, there is no way that the allocation can extend off the end of the page.
		// The assert() below verifies the previous sentence.
		top -= bytes;
		ConvertEx::Put2(&data[hdr+5], top);
		_assert(top+bytes <= (int)page->Bt->UsableSize);
		*idx = top;
		return RC_OK;
	}

	__device__ static RC FreeSpace(MemPage *page, int start, int size)
	{
		_assert(page->Bt != nullptr);
		_assert(Pager::Iswriteable(page->DBPage));
		_assert(start >= page->HdrOffset + 6 + page->ChildPtrSize);
		_assert((start + size) <= (int)page->Bt->UsableSize);
		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(size >= 0); // Minimum cell size is 4

		unsigned char *data = page->Data;
		if (page->Bt->BtsFlags & BTS_SECURE_DELETE) // Overwrite deleted information with zeros when the secure_delete option is enabled
			_memset(&data[start], 0, size);

		// Add the space back into the linked list of freeblocks.  Note that even though the freeblock list was checked by BtreeInitPage(),
		// BtreeInitPage() did not detect overlapping cells or freeblocks that overlapped cells.   Nor does it detect when the
		// cell content area exceeds the value in the page header.  If these situations arise, then subsequent insert operations might corrupt
		// the freelist.  So we do need to check for corruption while scanning the freelist.
		int hdr = page->HdrOffset;
		int addr = hdr + 1;
		int last = page->Bt->UsableSize - 4; // Largest possible freeblock offset
		_assert(start <= last);
		int pbegin;
		while ((pbegin = ConvertEx::Get2(&data[addr])) < start && pbegin > 0)
		{
			if (pbegin < addr + 4)
				return SysEx_CORRUPT_BKPT;
			addr = pbegin;
		}
		if (pbegin > last)
			return SysEx_CORRUPT_BKPT;
		_assert(pbegin > addr || pbegin == 0);
		ConvertEx::Put2(&data[addr], start);
		ConvertEx::Put2(&data[start], pbegin);
		ConvertEx::Put2(&data[start + 2], size);
		page->Frees = page->Frees + (uint16)size;

		// Coalesce adjacent free blocks
		addr = hdr + 1;
		while ((pbegin = ConvertEx::Get2(&data[addr])) > 0)
		{
			_assert(pbegin > addr);
			_assert(pbegin <= (int)page->Bt->UsableSize - 4);
			int pnext = ConvertEx::Get2(&data[pbegin]);
			int psize = ConvertEx::Get2(&data[pbegin + 2]);
			if (pbegin + psize + 3 >= pnext && pnext > 0)
			{
				int frag = pnext - (pbegin + psize);
				if (frag < 0 || frag > (int)data[hdr + 7])
					return SysEx_CORRUPT_BKPT;
				data[hdr + 7] -= (uint8)frag;
				int x = ConvertEx::Get2(&data[pnext]);
				ConvertEx::Put2(&data[pbegin], x);
				x = pnext + ConvertEx::Get2(&data[pnext+2]) - pbegin;
				ConvertEx::Put2(&data[pbegin+2], x);
			}
			else
				addr = pbegin;
		}

		// If the cell content area begins with a freeblock, remove it.
		if (data[hdr + 1] == data[hdr + 5] && data[hdr + 2] == data[hdr + 6])
		{
			pbegin = ConvertEx::Get2(&data[hdr + 1]);
			_memcpy(&data[hdr + 1], &data[pbegin], 2);
			int top = ConvertEx::Get2(&data[hdr + 5]) + ConvertEx::Get2(&data[pbegin + 2]);
			ConvertEx::Put2(&data[hdr + 5], top);
		}
		_assert(Pager::Iswriteable(page->DBPage));
		return RC_OK;
	}

	__device__ static RC DecodeFlags(MemPage *page, int flagByte)
	{
		_assert(page->HdrOffset == (page->ID == 1 ? 100 : 0));
		_assert(MutexEx::Held(page->Bt->Mutex));
		page->Leaf = (flagByte >> 3 ? true : false); _assert(PTF_LEAF == 1 << 3);
		flagByte &= ~PTF_LEAF;
		page->ChildPtrSize = 4 - 4 * page->Leaf;
		BtShared *bt = page->Bt; // A copy of pPage->pBt
		if (flagByte == (PTF_LEAFDATA | PTF_INTKEY))
		{
			page->IntKey = true;
			page->HasData = page->Leaf;
			page->MaxLocal = bt->MaxLeaf;
			page->MinLocal = bt->MinLeaf;
		}
		else if (flagByte == PTF_ZERODATA)
		{
			page->IntKey = false;
			page->HasData = false;
			page->MaxLocal = bt->MaxLocal;
			page->MinLocal = bt->MinLocal;
		}
		else
			return SysEx_CORRUPT_BKPT;
		page->Max1bytePayload = bt->Max1bytePayload;
		return RC_OK;
	}

	__device__ static RC BtreeInitPage(MemPage *page)
	{
		_assert(page->Bt != nullptr);
		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(page->ID == Pager::get_PageID(page->DBPage));
		_assert(page == Pager::GetExtra(page->DBPage));
		_assert(page->Data == Pager::GetData(page->DBPage));

		if (!page->IsInit)
		{
			BtShared *bt = page->Bt; // The main btree structure

			uint8 hdr = page->HdrOffset; // Offset to beginning of page header
			uint8 *data = page->Data; // Equal to pPage->aData
			if (DecodeFlags(page, data[hdr])) return SysEx_CORRUPT_BKPT;
			_assert(bt->PageSize >= 512 && bt->PageSize <= 65536);
			page->MaskPage = (uint16)(bt->PageSize - 1);
			page->Overflows = 0;
			int usableSize = bt->UsableSize; // Amount of usable space on each page
			uint16 cellOffset; // Offset from start of page to first cell pointer
			page->CellOffset = cellOffset = hdr + 12 - 4 * page->Leaf;
			page->DataEnd = &data[usableSize];
			page->CellIdx = &data[cellOffset];
			int top = ConvertEx::Get2nz(&data[hdr + 5]); // First byte of the cell content area
			page->Cells = ConvertEx::Get2(&data[hdr + 3]);
			if (page->Cells > MX_CELL(bt)) // To many cells for a single page.  The page must be corrupt
				return SysEx_CORRUPT_BKPT;
			ASSERTCOVERAGE(page->Cells == MX_CELL(bt));

			// A malformed database page might cause us to read past the end of page when parsing a cell.  
			//
			// The following block of code checks early to see if a cell extends past the end of a page boundary and causes SQLITE_CORRUPT to be 
			// returned if it does.
			int cellFirst = cellOffset + 2 * page->Cells; // First allowable cell or freeblock offset
			int cellLast = usableSize - 4; // Last possible cell or freeblock offset
			uint16 pc;  // Address of a freeblock within pPage->aData[]
#if defined(ENABLE_OVERSIZE_CELL_CHECK)
			{
				if (!page->Leaf) cellLast--;
				for (int i = 0; i < page->Cells; i++)
				{
					pc = ConvertEx::Get2(&data[cellOffset + i * 2]);
					ASSERTCOVERAGE(pc == cellFirst);
					ASSERTCOVERAGE(pc == cellLast);
					if (pc < cellFirst || pc > cellLast)
						return SysEx_CORRUPT_BKPT;
					int sz = CellSizePtr(page, &data[pc]); // Size of a cell
					ASSERTCOVERAGE(pc + sz == usableSize);
					if (pc + sz > usableSize)
						return SysEx_CORRUPT_BKPT;
				}
				if (!page->Leaf) cellLast++;
			}  
#endif

			// Compute the total free space on the page
			pc = ConvertEx::Get2(&data[hdr + 1]);
			int free = data[hdr + 7] + top; // Number of unused bytes on the page
			while (pc > 0)
			{
				if (pc < cellFirst || pc > cellLast)
					// Start of free block is off the page
						return SysEx_CORRUPT_BKPT; 
				uint16 next = ConvertEx::Get2(&data[pc]);
				uint16 size = ConvertEx::Get2(&data[pc + 2]);
				if ((next > 0 && next <= pc + size + 3) || pc + size > usableSize)
					// Free blocks must be in ascending order. And the last byte of the free-block must lie on the database page.
						return SysEx_CORRUPT_BKPT; 
				free = free + size;
				pc = next;
			}

			// At this point, nFree contains the sum of the offset to the start of the cell-content area plus the number of free bytes within
			// the cell-content area. If this is greater than the usable-size of the page, then the page must be corrupted. This check also
			// serves to verify that the offset to the start of the cell-content area, according to the page header, lies within the page.
			if (free > usableSize)
				return SysEx_CORRUPT_BKPT; 
			page->Frees = (uint16)(free - cellFirst);
			page->IsInit = true;
		}
		return RC_OK;
	}

	__device__ static void ZeroPage(MemPage *page, int flags)
	{
		BtShared *bt = page->Bt;
		unsigned char *data = page->Data;
		_assert(Pager::get_PageID(page->DBPage) == page->ID);
		_assert(Pager::GetExtra(page->DBPage) == (void *)page);
		_assert(Pager::GetData(page->DBPage) == data);
		_assert(Pager::Iswriteable(page->DBPage));
		_assert(MutexEx::Held(bt->Mutex));
		uint8 hdr = page->HdrOffset;
		if (bt->BtsFlags & BTS_SECURE_DELETE)
			_memset(&data[hdr], 0, bt->UsableSize - hdr);
		data[hdr] = (char)flags;
		uint16 first = hdr + 8 + 4*((flags & PTF_LEAF) == 0 ? 1 : 0);
		_memset(&data[hdr + 1], 0, 4);
		data[hdr + 7] = 0;
		ConvertEx::Put2(&data[hdr + 5], bt->UsableSize);
		page->Frees = (uint16)(bt->UsableSize - first);
		DecodeFlags(page, flags);
		page->HdrOffset = hdr;
		page->CellOffset = first;
		page->DataEnd = &data[bt->UsableSize];
		page->CellIdx = &data[first];
		page->Overflows = 0;
		_assert(bt->PageSize >= 512 && bt->PageSize <= 65536);
		page->MaskPage = (uint16)(bt->PageSize - 1);
		page->Cells = 0;
		page->IsInit = true;
	}

#pragma endregion

#pragma region Page

	__device__ static MemPage *BtreePageFromDbPage(IPage *dbPage, Pid id, BtShared *bt)
	{
		MemPage *page = (MemPage *)Pager::GetExtra(dbPage);
		page->Data = (uint8 *)Pager::GetData(dbPage);
		page->DBPage = dbPage;
		page->Bt = bt;
		page->ID = id;
		page->HdrOffset = (page->ID == 1 ? 100 : 0);
		return page; 
	}

	__device__ static RC BtreeGetPage(BtShared *bt, Pid id, MemPage **page, bool noContent)
	{
		_assert(MutexEx::Held(bt->Mutex));
		IPage *dbPage;
		RC rc = bt->Pager->Acquire(id, (IPage **)&dbPage, noContent);
		if (rc) return rc;
		*page = BtreePageFromDbPage(dbPage, id, bt);
		return RC_OK;
	}

	__device__ static MemPage *BtreePageLookup(BtShared *bt, Pid id)
	{
		IPage *dbPage;
		_assert(MutexEx::Held(bt->Mutex));
		dbPage = bt->Pager->Lookup(id);
		return (dbPage ? BtreePageFromDbPage(dbPage, id, bt) : nullptr);
	}

	__device__ static Pid BtreePagecount(BtShared *bt)
	{
		return bt->Pages;
	}

	__device__ Pid Btree::LastPage()
	{
		_assert(HoldsMutex());
		_assert(((Bt->Pages) & 0x8000000) == 0);
		return BtreePagecount(Bt);
	}

	__device__ static RC GetAndInitPage(BtShared *bt, Pid id, MemPage **page)
	{
		_assert(MutexEx::Held(bt->Mutex));

		RC rc;
		if (id > BtreePagecount(bt))
			rc = SysEx_CORRUPT_BKPT;
		else
		{
			rc = BtreeGetPage(bt, id, page, false);
			if (rc == RC_OK)
			{
				rc = BtreeInitPage(*page);
				if (rc != RC_OK)
					ReleasePage(*page);
			}
		}

		ASSERTCOVERAGE(id == 0);
		_assert(id != 0 || rc == RC_CORRUPT);
		return rc;
	}

	__device__ static void ReleasePage(MemPage *page)
	{
		if (page)
		{
			_assert(page->Data != nullptr);
			_assert(page->Bt != nullptr);
			_assert(Pager::GetExtra(page->DBPage) == (void*)page);
			_assert(Pager::GetData(page->DBPage) == page->Data);
			_assert(MutexEx::Held(page->Bt->Mutex));
			Pager::Unref(page->DBPage);
		}
	}

	__device__ static void PageReinit(IPage *dbPage)
	{
		MemPage *page = (MemPage *)Pager::GetExtra(dbPage);
		_assert(Pager::get_PageRefs(dbPage) > 0);
		if (page->IsInit)
		{
			_assert(MutexEx::Held(page->Bt->Mutex));
			page->IsInit = false;
			if (Pager::get_PageRefs(dbPage) > 1)
			{
				// pPage might not be a btree page;  it might be an overflow page or ptrmap page or a free page.  In those cases, the following
				// call to BtreeInitPage() will likely return SQLITE_CORRUPT. But no harm is done by this.  And it is very important that
				// BtreeInitPage() be called on every btree page so we make the call for every page that comes in for re-initing.
				BtreeInitPage(page);
			}
		}
	}

#pragma endregion

#pragma region Open / Close

	__device__ static int BtreeInvokeBusyHandler(void *arg)
	{
		BtShared *bt = (BtShared *)arg;
		_assert(bt->Ctx != nullptr);
		_assert(MutexEx::Held(bt->Ctx->Mutex));
		return bt->Ctx->InvokeBusyHandler();
	}

	__device__ RC Btree::Open(VSystem *vfs, const char *filename, BContext *ctx, Btree **btree, OPEN flags, VSystem::OPEN vfsFlags)
	{
		// True if opening an ephemeral, temporary database
		const bool tempDB = (filename == nullptr || filename[0] == 0);

		// Set the variable isMemdb to true for an in-memory database, or false for a file-based database.
		const bool memoryDB = ((filename && _strcmp(filename, ":memory:") == 0) ||
			(tempDB && ctx->TempInMemory()) ||
			(vfsFlags & VSystem::OPEN_MEMORY) != 0);

		_assert(ctx != nullptr);
		_assert(vfs != nullptr);
		_assert(MutexEx::Held(ctx->Mutex));
		_assert(((uint)flags & 0xff) == (uint)flags); // flags fit in 8 bits

		// Only a BTREE_SINGLE database can be BTREE_UNORDERED
		_assert((flags & OPEN_UNORDERED) == 0 || (flags & OPEN_SINGLE) != 0);

		// A BTREE_SINGLE database is always a temporary and/or ephemeral
		_assert((flags & OPEN_SINGLE) == 0 || tempDB);

		if (memoryDB)
			flags |= OPEN_MEMORY;
		if ((vfsFlags & VSystem::OPEN_MAIN_DB) != 0 && (memoryDB || tempDB))
			vfsFlags = (VSystem::OPEN)((vfsFlags & ~VSystem::OPEN_MAIN_DB) | VSystem::OPEN_TEMP_DB);
		Btree *p = (Btree *)_alloc2(sizeof(Btree), true); // Handle to return
		if (!p)
			return RC_NOMEM;
		p->InTrans = TRANS_NONE;
		p->Ctx = ctx;
#ifndef OMIT_SHARED_CACHE
		p->Lock.Btree = p;
		p->Lock.Table = 1;
#endif

		RC rc = RC_OK; // Result code from this function
		BtShared *bt = nullptr; // Shared part of btree structure
		MutexEx mutexOpen = MutexEx_Empty;
#if !defined(OMIT_SHARED_CACHE) && !defined(OMIT_DISKIO)
		// If this Btree is a candidate for shared cache, try to find an existing BtShared object that we can share with
		if (!tempDB && (!memoryDB || (vfsFlags & VSystem::OPEN_URI) != 0))
			if (vfsFlags & VSystem::OPEN_SHAREDCACHE)
			{
				int fullPathnameLength = vfs->MaxPathname + 1;
				char *fullPathname = (char *)_alloc(fullPathnameLength);
				p->Sharable_ = true;
				if (!fullPathname)
				{
					_free(p);
					return RC_NOMEM;
				}
				if (memoryDB)
					_memcpy(fullPathname, filename, _strlen30(filename) + 1);
				else
				{
					rc = vfs->FullPathname(filename, fullPathnameLength, fullPathname);
					if (rc)
					{
						_free(fullPathname);
						_free(p);
						return rc;
					}
				}
				MutexEx mutexShared;
#if THREADSAFE
				mutexOpen = MutexEx::Alloc(MutexEx::MUTEX_STATIC_OPEN); // Prevents a race condition. Ticket #3537
				MutexEx::Enter(mutexOpen);
				mutexShared = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
				MutexEx::Enter(mutexShared);
#endif
				for (bt = _sharedCacheList; bt; bt = bt->Next)
				{
					_assert(bt->Refs > 0);
					if (_strcmp(fullPathname, bt->Pager->get_Filename(false)) == 0 && bt->Pager->get_Vfs() == vfs)
					{
						for (int i = ctx->DBs.length - 1; i >= 0; i--)
						{
							Btree *existing = ctx->DBs[i].Bt;
							if (existing && existing->Bt == bt)
							{
								MutexEx::Leave(mutexShared);
								MutexEx::Leave(mutexOpen);
								_free(fullPathname);
								_free(p);
								return RC_CONSTRAINT;
							}
						}
						p->Bt = bt;
						bt->Refs++;
						break;
					}
				}
				MutexEx::Leave(mutexShared);
				_free(fullPathname);
			}
#ifdef _DEBUG
			else
				// In debug mode, we mark all persistent databases as sharable even when they are not.  This exercises the locking code and
				// gives more opportunity for asserts(sqlite3_mutex_held()) statements to find locking problems.
				p->Sharable_ = true;
#endif
#endif

		int8 reserves; // Byte of unused space on each page
		unsigned char dbHeader[100]; // Database header content
		if (bt == nullptr)
		{
			// The following asserts make sure that structures used by the btree are the right size.  This is to guard against size changes that result
			// when compiling on a different architecture.
			_assert(sizeof(int64) == 8 || sizeof(int64) == 4);
			_assert(sizeof(uint64) == 8 || sizeof(uint64) == 4);
			_assert(sizeof(uint32) == 4);
			_assert(sizeof(uint16) == 2);
			_assert(sizeof(Pid) == 4);

			bt = (BtShared *)_alloc2(sizeof(*bt), true);
			if (bt == nullptr)
			{
				rc = RC_NOMEM;
				goto btree_open_out;
			}
			rc = Pager::Open(vfs, &bt->Pager, filename, EXTRA_SIZE, (IPager::PAGEROPEN)flags, vfsFlags, PageReinit);
			if (rc == RC_OK)
				rc = bt->Pager->ReadFileheader(sizeof(dbHeader), dbHeader);
			if (rc != RC_OK)
				goto btree_open_out;
			bt->OpenFlags = flags;
			bt->Ctx = ctx;
			bt->Pager->SetBusyhandler(BtreeInvokeBusyHandler, bt);
			p->Bt = bt;

			bt->Cursor = nullptr;
			bt->Page1 = nullptr;
			if (bt->Pager->get_Readonly()) bt->BtsFlags |= BTS_READ_ONLY;
#ifdef SECURE_DELETE
			bt->BtsFlags |= BTS_SECURE_DELETE;
#endif
			bt->PageSize = (dbHeader[16] << 8) | (dbHeader[17] << 16);
			if (bt->PageSize < 512 || bt->PageSize > MAX_PAGE_SIZE || ((bt->PageSize - 1) & bt->PageSize) != 0)
			{
				bt->PageSize = 0;
#ifndef OMIT_AUTOVACUUM
				// If the magic name ":memory:" will create an in-memory database, then leave the autoVacuum mode at 0 (do not auto-vacuum), even if
				// SQLITE_DEFAULT_AUTOVACUUM is true. On the other hand, if SQLITE_OMIT_MEMORYDB has been defined, then ":memory:" is just a
				// regular file-name. In this case the auto-vacuum applies as per normal.
				if (filename && !memoryDB)
				{
					bt->AutoVacuum = (DEFAULT_AUTOVACUUM != (AUTOVACUUM)0);
					bt->IncrVacuum = (DEFAULT_AUTOVACUUM == AUTOVACUUM_INCR);
				}
#endif
				reserves = 0;
			}
			else
			{
				reserves = dbHeader[20];
				bt->BtsFlags |= BTS_PAGESIZE_FIXED;
#ifndef OMIT_AUTOVACUUM
				bt->AutoVacuum = (ConvertEx::Get4(&dbHeader[36 + 4 * 4]) ? true : false);
				bt->IncrVacuum = (ConvertEx::Get4(&dbHeader[36 + 7 * 4]) ? true : false);
#endif
			}
			rc = bt->Pager->SetPageSize(&bt->PageSize, reserves);
			if (rc) goto btree_open_out;
			bt->UsableSize = bt->PageSize - reserves;
			_assert((bt->PageSize & 7) == 0); // 8-byte alignment of pageSize

#if !defined(OMIT_SHARED_CACHE) && !defined(OMIT_DISKIO)
			// Add the new BtShared object to the linked list sharable BtShareds.
			if (p->Sharable_)
			{
				bt->Refs = 1;
				MutexEx mutexShared;
#if THREADSAFE
				mutexShared = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
				bt->Mutex = MutexEx::Alloc(MutexEx::MUTEX_FAST);
#endif
				MutexEx::Enter(mutexShared);
				bt->Next = _sharedCacheList;
				_sharedCacheList = bt;
				MutexEx::Leave(mutexShared);
			}
#endif
		}

#if !defined(OMIT_SHARED_CACHE) && !defined(OMIT_DISKIO)
		// If the new Btree uses a sharable pBtShared, then link the new Btree into the list of all sharable Btrees for the same connection.
		// The list is kept in ascending order by pBt address.
		if (p->Sharable_)
		{
			Btree *sib;
			for (int i = 0; i < ctx->DBs.length; i++)
				if ((sib = ctx->DBs[i].Bt) != nullptr && sib->Sharable_)
				{
					while (sib->Prev) { sib = sib->Prev; }
					if (p->Bt < sib->Bt)
					{
						p->Next = sib;
						p->Prev = nullptr;
						sib->Prev = p;
					}
					else
					{
						while (sib->Next && sib->Next->Bt < p->Bt)
							sib = sib->Next;
						p->Next = sib->Next;
						p->Prev = sib;
						if (p->Next)
							p->Next->Prev = p;
						sib->Next = p;
					}
					break;
				}
		}
#endif
		*btree = p;

btree_open_out:
		if (rc != RC_OK)
		{
			if (bt && bt->Pager)
				bt->Pager->Close();
			_free(bt);
			_free(p);
			*btree = 0;
		}
		else
			// If the B-Tree was successfully opened, set the pager-cache size to the default value. Except, when opening on an existing shared pager-cache,
			// do not change the pager-cache size.
			if (p->Schema(0, nullptr) == nullptr)
				p->Bt->Pager->SetCacheSize(DEFAULT_CACHE_SIZE);
		if (mutexOpen.Tag)
		{
			_assert(MutexEx::Held(mutexOpen));
			MutexEx::Leave(mutexOpen);
		}
		return rc;
	}

	__device__ static bool RemoveFromSharingList(BtShared *bt)
	{
#ifndef OMIT_SHARED_CACHE
		_assert(MutexEx::Held(bt->Mutex));
		MutexEx master;
#if THREADSAFE
		master = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
#endif
		bool removed = false;
		MutexEx::Enter(master);
		bt->Refs--;
		if (bt->Refs <= 0)
		{
			if (_sharedCacheList == bt)
				_sharedCacheList = bt->Next;
			else
			{
				BtShared *list = _sharedCacheList;
				while (_ALWAYS(list) && list->Next != bt)
					list = list->Next;
				if (_ALWAYS(list))
					list->Next = bt->Next;
			}
#if THREADSAFE
			MutexEx::Free(bt->Mutex);
#endif
			removed = true;
		}
		MutexEx::Leave(master);
		return removed;
#else
		return true;
#endif
	}

	__device__ static void AllocateTempSpace(BtShared *bt)
	{
		if (!bt->TmpSpace)
			bt->TmpSpace = (uint8 *)PCache::PageAlloc(bt->PageSize);
	}

	__device__ static void FreeTempSpace(BtShared *bt)
	{
		PCache::PageFree(bt->TmpSpace);
		bt->TmpSpace = nullptr;
	}

	__device__ RC Btree::Close()
	{
		// Close all cursors opened via this handle.
		_assert(MutexEx::Held(Ctx->Mutex));
		Enter();
		BtShared *bt = Bt;
		BtCursor *cur = bt->Cursor;
		while (cur)
		{
			BtCursor *tmp = cur;
			cur = cur->Next;
			if (tmp->Btree == this)
				CloseCursor(tmp);
		}

		// Rollback any active transaction and free the handle structure. The call to sqlite3BtreeRollback() drops any table-locks held by this handle.
		Rollback(RC_OK);
		Leave();

		// If there are still other outstanding references to the shared-btree structure, return now. The remainder of this procedure cleans up the shared-btree.
		_assert(WantToLock == 0 && Locked == 0);
		if (!Sharable_ || RemoveFromSharingList(bt))
		{
			// The pBt is no longer on the sharing list, so we can access it without having to hold the mutex.
			//
			// Clean out and delete the BtShared object.
			_assert(!bt->Cursor);
			bt->Pager->Close();
			if (bt->FreeSchema && bt->Schema)
				bt->FreeSchema(bt->Schema);
			_tagfree(nullptr, bt->Schema);
			FreeTempSpace(bt);
			_free(bt);
		}

#ifndef OMIT_SHARED_CACHE
		_assert(WantToLock == 0 && !Locked);
		if (Prev) Prev->Next = Next;
		if (Next) Next->Prev = Prev;
#endif

		_free(this);
		return RC_OK;
	}

#pragma endregion

#pragma region Settings1

	__device__ RC Btree::SetCacheSize(int maxPage)
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		Enter();
		Bt->Pager->SetCacheSize(maxPage);
		Leave();
		return RC_OK;
	}

#ifndef OMIT_PAGER_PRAGMAS
	__device__ RC Btree::SetSafetyLevel(int level, bool fullSync, bool ckptFullSync)
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		_assert(level >= 1 && level <= 3);
		Enter();
		Bt->Pager->SetSafetyLevel(level, fullSync, ckptFullSync);
		Leave();
		return RC_OK;
	}
#endif

	__device__ bool Btree::SyncDisabled()
	{
		_assert(MutexEx::Held(Ctx->Mutex));  
		Enter();
		_assert(Bt && Bt->Pager);
		bool rc = Bt->Pager->get_NoSync();
		Leave();
		return rc;
	}

	__device__ RC Btree::SetPageSize(int pageSize, int reserves, bool fix)
	{
		_assert(reserves >= -1 && reserves <= 255);
		Enter();
		BtShared *bt = Bt;
		if (bt->BtsFlags & BTS_PAGESIZE_FIXED)
		{
			Leave();
			return RC_READONLY;
		}
		if (reserves < 0)
			reserves = bt->PageSize - bt->UsableSize;
		_assert(reserves >= 0 && reserves <= 255);
		if (pageSize >= 512 && pageSize <= MAX_PAGE_SIZE && ((pageSize - 1) & pageSize) == 0)
		{
			_assert((pageSize & 7) == 0);
			_assert(!bt->Page1 && !bt->Cursor);
			bt->PageSize = (uint32)pageSize;
			FreeTempSpace(bt);
		}
		RC rc = bt->Pager->SetPageSize(&bt->PageSize, reserves);
		bt->UsableSize = bt->PageSize - (uint16)reserves;
		if (fix) bt->BtsFlags |= BTS_PAGESIZE_FIXED;
		Leave();
		return rc;
	}

	__device__ int Btree::GetPageSize()
	{
		return Bt->PageSize;
	}

#if defined(HAS_CODEC) || defined(_DEBUG)
	__device__ int Btree::GetReserveNoMutex()
	{
		_assert(MutexEx::Held(Bt->Mutex));
		return Bt->PageSize - Bt->UsableSize;
	}
#endif

#if !defined(OMIT_PAGER_PRAGMAS) || !defined(OMIT_VACUUM)
	__device__ int Btree::GetReserve()
	{
		Enter();
		int n = Bt->PageSize - Bt->UsableSize;
		Leave();
		return n;
	}

	__device__ int Btree::MaxPageCount(int maxPage)
	{
		Enter();
		int n = Bt->Pager->MaxPages(maxPage);
		Leave();
		return n;
	}

	__device__ bool Btree::SecureDelete(bool newFlag)
	{
		Enter();
		Bt->BtsFlags &= ~BTS_SECURE_DELETE;
		if (newFlag) Bt->BtsFlags |= BTS_SECURE_DELETE;
		bool b = (Bt->BtsFlags & BTS_SECURE_DELETE) != 0;
		Leave();
		return b;
	}
#endif

	__device__ RC Btree::SetAutoVacuum(AUTOVACUUM autoVacuum)
	{
#ifdef OMIT_AUTOVACUUM
		return RC_READONLY;
#else
		RC rc = RC_OK;
		Enter();
		BtShared *bt = Bt;
		if ((bt->BtsFlags & BTS_PAGESIZE_FIXED) != 0 && (autoVacuum != (AUTOVACUUM)0) != bt->AutoVacuum)
			rc = RC_READONLY;
		else
		{
			bt->AutoVacuum = (autoVacuum != (AUTOVACUUM)0);
			bt->IncrVacuum = (autoVacuum == AUTOVACUUM_INCR);
		}
		Leave();
		return rc;
#endif
	}

	__device__ Btree::AUTOVACUUM Btree::GetAutoVacuum()
	{
#ifdef OMIT_AUTOVACUUM
		return AUTOVACUUM_NONE;
#else
		Enter();
		BtShared *bt = Bt;
		AUTOVACUUM rc = (!bt->AutoVacuum ? AUTOVACUUM_NONE :
			!bt->IncrVacuum ? AUTOVACUUM_FULL :
			AUTOVACUUM_INCR);
		Leave();
		return rc;
#endif
	}

#pragma endregion

#pragma region Lock / Unlock

	__device__ static RC LockBtree(BtShared *bt)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(bt->Page1 == nullptr);
		RC rc = bt->Pager->SharedLock();
		if (rc != RC_OK) return rc;
		MemPage *page1; // Page 1 of the database file
		rc = BtreeGetPage(bt, 1, &page1, false);
		if (rc != RC_OK) return rc;

		// Do some checking to help insure the file we opened really is a valid database file. 
		Pid pages = ConvertEx::Get4(28 + (uint8 *)page1->Data); // Number of pages in the database
		//Pid pagesHeader = pages; // Number of pages in the database according to hdr
		Pid pagesFile = 0; // Number of pages in the database file
		bt->Pager->Pages(&pagesFile);
		if (pages == 0 || _memcmp(24 + (uint8 *)page1->Data, 92 + (uint8 *)page1->Data, 4) != 0)
			pages = pagesFile;
		if (pages > 0)
		{
			uint8 *page1Data = page1->Data;
			rc = RC_NOTADB;
			if (_memcmp(page1Data, _magicHeader, 16) != 0)
				goto page1_init_failed;

#ifdef OMIT_WAL
			if (page1Data[18] > 1)
				bt->BtsFlags |= BTS_READ_ONLY;
			if (page1Data[19] > 1)
				goto page1_init_failed;
#else
			if (page1Data[18] > 2)
				bt->BtsFlags |= BTS_READ_ONLY;
			if (page1Data[19] > 2)
				goto page1_init_failed;

			// If the write version is set to 2, this database should be accessed in WAL mode. If the log is not already open, open it now. Then 
			// return SQLITE_OK and return without populating BtShared.pPage1. The caller detects this and calls this function again. This is
			// required as the version of page 1 currently in the page1 buffer may not be the latest version - there may be a newer one in the log file.
			if (page1Data[19] == 2 && (bt->BtsFlags & BTS_NO_WAL) == 0)
			{
				int isOpen = 0;
				rc = bt->Pager->OpenWal(&isOpen);
				if (rc != RC_OK)
					goto page1_init_failed;
				else if (isOpen == 0)
				{
					ReleasePage(page1);
					return RC_OK;
				}
				rc = RC_NOTADB;
			}
#endif

			// The maximum embedded fraction must be exactly 25%.  And the minimum embedded fraction must be 12.5% for both leaf-data and non-leaf-data.
			// The original design allowed these amounts to vary, but as of version 3.6.0, we require them to be fixed.
			if (_memcmp(&page1Data[21], "\100\040\040", 3) != 0)
				goto page1_init_failed;
			uint32 pageSize = (uint32)((page1Data[16] << 8) | (page1Data[17] << 16));
			if (((pageSize - 1) & pageSize) != 0 ||
				pageSize > MAX_PAGE_SIZE ||
				pageSize <= 256)
				goto page1_init_failed;
			_assert((pageSize & 7) == 0);
			uint32 usableSize = pageSize - page1Data[20];
			if (pageSize != bt->PageSize)
			{
				// After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
				// actually pageSize. Unlock the database, leave pBt->pPage1 at zero and return SQLITE_OK. The caller will call this function
				// again with the correct page-size.
				ReleasePage(page1);
				bt->UsableSize = usableSize;
				bt->PageSize = pageSize;
				FreeTempSpace(bt);
				rc = bt->Pager->SetPageSize(&bt->PageSize, pageSize - usableSize);
				return rc;
			}
			if ((bt->Ctx->Flags & BContext::FLAG::FLAG_RecoveryMode) == 0 && pages > pagesFile)
			{
				rc = SysEx_CORRUPT_BKPT;
				goto page1_init_failed;
			}
			if (usableSize < 480)
				goto page1_init_failed;
			bt->PageSize = pageSize;
			bt->UsableSize = usableSize;
#ifndef OMIT_AUTOVACUUM
			bt->AutoVacuum = (ConvertEx::Get4(&page1Data[36 + 4 * 4]) ? true : false);
			bt->IncrVacuum = (ConvertEx::Get4(&page1Data[36 + 7 * 4]) ? true : false);
#endif
		}

		// maxLocal is the maximum amount of payload to store locally for a cell.  Make sure it is small enough so that at least minFanout
		// cells can will fit on one page.  We assume a 10-byte page header. Besides the payload, the cell must store:
		//     2-byte pointer to the cell
		//     4-byte child pointer
		//     9-byte nKey value
		//     4-byte nData value
		//     4-byte overflow page pointer
		// So a cell consists of a 2-byte pointer, a header which is as much as 17 bytes long, 0 to N bytes of payload, and an optional 4 byte overflow
		// page pointer.
		bt->MaxLocal = (uint16)((bt->UsableSize - 12) * 64 / 255 - 23);
		bt->MinLocal = (uint16)((bt->UsableSize - 12) * 32 / 255 - 23);
		bt->MaxLeaf = (uint16)(bt->UsableSize - 35);
		bt->MinLeaf = (uint16)((bt->UsableSize - 12) * 32 / 255 - 23);
		if (bt->MaxLocal > 127)
			bt->Max1bytePayload = 127;
		else
			bt->Max1bytePayload = (uint8)bt->MaxLocal;
		_assert(bt->MaxLeaf + 23 <= MX_CELL_SIZE(bt));
		bt->Page1 = page1;
		bt->Pages = pages;
		return RC_OK;

page1_init_failed:
		ReleasePage(page1);
		bt->Page1 = nullptr;
		return rc;
	}

	__device__ static void UnlockBtreeIfUnused(BtShared *bt)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(bt->Cursor == nullptr || bt->InTransaction > TRANS_NONE);
		if (bt->InTransaction == TRANS_NONE && bt->Page1 != nullptr)
		{
			_assert(bt->Page1->Data != nullptr);
			_assert(bt->Pager->get_Refs() == 1);
			ReleasePage(bt->Page1);
			bt->Page1 = nullptr;
		}
	}

#pragma endregion

#pragma region NewDB

	__device__ static RC NewDatabase(BtShared *bt)
	{
		_assert(MutexEx::Held(bt->Mutex));
		if (bt->Pages > 0)
			return RC_OK;
		MemPage *p1 = bt->Page1;
		_assert(p1 != nullptr);
		uint8 *data = p1->Data;
		RC rc = Pager::Write(p1->DBPage);
		if (rc) return rc;
		_memcpy<char>((char *)data, _magicHeader, sizeof(_magicHeader));
		_assert(sizeof(_magicHeader) == 16);
		data[16] = (uint8)((bt->PageSize >> 8) & 0xff);
		data[17] = (uint8)((bt->PageSize >> 16) & 0xff);
		data[18] = 1;
		data[19] = 1;
		_assert(bt->UsableSize <= bt->PageSize && bt->UsableSize + 255 >= bt->PageSize);
		data[20] = (uint8)(bt->PageSize - bt->UsableSize);
		data[21] = 64;
		data[22] = 32;
		data[23] = 32;
		_memset(&data[24], 0, 100 - 24);
		ZeroPage(p1, PTF_INTKEY | PTF_LEAF | PTF_LEAFDATA);
		bt->BtsFlags |= BTS_PAGESIZE_FIXED;
#ifndef OMIT_AUTOVACUUM
		ConvertEx::Put4(&data[36 + 4 * 4], bt->AutoVacuum ? 1 : 0 );
		ConvertEx::Put4(&data[36 + 7 * 4], bt->IncrVacuum ? 1 : 0 );
#endif
		bt->Pages = 1;
		data[31] = 1;
		return RC_OK;
	}

	__device__ RC Btree::NewDb()
	{
		Enter();
		Bt->Pages = 0;
		RC rc = NewDatabase(Bt);
		Leave();
		return rc;
	}

#pragma endregion

#pragma region Transactions

	__device__ RC Btree::BeginTrans(int wrflag)
	{
		Enter();
		BtreeIntegrity(this);

#ifndef OMIT_SHARED_CACHE
		// If another database handle has already opened a write transaction on this shared-btree structure and a second write transaction is
		// requested, return SQLITE_LOCKED.
		BContext *blockingCtx = nullptr;
#endif

		// If the btree is already in a write-transaction, or it is already in a read-transaction and a read-transaction
		// is requested, this is a no-op.
		BtShared *bt = Bt;
		RC rc = RC_OK;
		if (InTrans == TRANS_WRITE || (InTrans == TRANS_READ && !wrflag))
			goto trans_begun;

		_assert(!IFAUTOVACUUM(bt->DoTruncate));

		// Write transactions are not possible on a read-only database
		if ((bt->BtsFlags & BTS_READ_ONLY) != 0 && wrflag)
		{
			rc = RC_READONLY;
			goto trans_begun;
		}

#ifndef OMIT_SHARED_CACHE
		// If another database handle has already opened a write transaction on this shared-btree structure and a second write transaction is
		// requested, return SQLITE_LOCKED.
		if ((wrflag && bt->InTransaction == TRANS_WRITE) || (bt->BtsFlags & BTS_PENDING) != 0)
			blockingCtx = bt->Writer->Ctx;
		else if (wrflag > 1)
		{
			for (BtLock *iter = bt->Lock; iter; iter = iter->Next)
				if (iter->Btree != this)
				{
					blockingCtx = iter->Btree->Ctx;
					break;
				}
		}

		if (blockingCtx)
		{
			Ctx->ConnectionBlocked(blockingCtx);
			rc = RC_LOCKED_SHAREDCACHE;
			goto trans_begun;
		}
#endif

		// Any read-only or read-write transaction implies a read-lock on page 1. So if some other shared-cache client already has a write-lock 
		// on page 1, the transaction cannot be opened. */
		rc = QuerySharedCacheTableLock(this, MASTER_ROOT, LOCK_READ);
		if (rc != RC_OK) goto trans_begun;

		bt->BtsFlags &= ~BTS_INITIALLY_EMPTY;
		if (bt->Pages == 0) bt->BtsFlags |= BTS_INITIALLY_EMPTY;
		do
		{
			// Call LockBtree() until either pBt->pPage1 is populated or LockBtree() returns something other than SQLITE_OK. LockBtree()
			// may return SQLITE_OK but leave pBt->pPage1 set to 0 if after reading page 1 it discovers that the page-size of the database 
			// file is not pBt->pageSize. In this case LockBtree() will update pBt->pageSize to the page-size of the file on disk.
			while (bt->Page1 == nullptr && (rc = LockBtree(bt)) == RC_OK);

			if (rc == RC_OK && wrflag)
			{
				if ((bt->BtsFlags & BTS_READ_ONLY) != 0)
					rc = RC_READONLY;
				else
				{
					rc = bt->Pager->Begin(wrflag > 1, Ctx->TempInMemory());
					if (rc == RC_OK)
						rc = NewDatabase(bt);
				}
			}

			if (rc != RC_OK)
				UnlockBtreeIfUnused(bt);
		} while ((rc & 0xFF) == RC_BUSY && bt->InTransaction == TRANS_NONE && BtreeInvokeBusyHandler(bt));

		if (rc == RC_OK)
		{
			if (InTrans == TRANS_NONE)
			{
				bt->Transactions++;
#ifndef OMIT_SHARED_CACHE
				if (Sharable_)
				{
					_assert(Lock.Btree == this && Lock.Table == 1);
					Lock.Lock = LOCK_READ;
					Lock.Next = bt->Lock;
					bt->Lock = &Lock;
				}
#endif
			}
			InTrans = (wrflag ? TRANS_WRITE : TRANS_READ);
			if (InTrans > bt->InTransaction)
				bt->InTransaction = InTrans;
			if (wrflag)
			{
				MemPage *page1 = bt->Page1;
#ifndef OMIT_SHARED_CACHE
				_assert(!bt->Writer);
				bt->Writer = this;
				bt->BtsFlags &= ~BTS_EXCLUSIVE;
				if (wrflag > 1) bt->BtsFlags |= BTS_EXCLUSIVE;
#endif

				// If the db-size header field is incorrect (as it may be if an old client has been writing the database file), update it now. Doing
				// this sooner rather than later means the database size can safely re-read the database size from page 1 if a savepoint or transaction
				// rollback occurs within the transaction.
				if (bt->Pages != ConvertEx::Get4(&page1->Data[28]))
				{
					rc = Pager::Write(page1->DBPage);
					if (rc == RC_OK)
						ConvertEx::Put4(&page1->Data[28], bt->Pages);
				}
			}
		}

trans_begun:
		if (rc == RC_OK && wrflag)
		{
			// This call makes sure that the pager has the correct number of open savepoints. If the second parameter is greater than 0 and
			// the sub-journal is not already open, then it will be opened here.
			rc = bt->Pager->OpenSavepoint(Ctx->SavepointsLength);
		}

		BtreeIntegrity(this);
		Leave();
		return rc;
	}

#pragma endregion

#pragma region Autovacuum
#ifndef OMIT_AUTOVACUUM

	__device__ static RC SetChildPtrmaps(MemPage *page)
	{
		bool isInitOrig = page->IsInit;
		BtShared *bt = page->Bt;
		_assert(MutexEx::Held(bt->Mutex));
		RC rc = BtreeInitPage(page);
		int cells;
		Pid id;
		if (rc != RC_OK)
			goto set_child_ptrmaps_out;
		cells = page->Cells; // Number of cells in page pPage

		id = page->ID;
		for (uint i = 0U; i < cells; i++)
		{
			uint8 *cell = FindCell(page, i);
			PtrmapPutOvflPtr(page, cell, &rc);
			if (!page->Leaf)
			{
				Pid childID = ConvertEx::Get4(cell);
				PtrmapPut(bt, childID, PTRMAP_BTREE, id, &rc);
			}
		}

		if (!page->Leaf)
		{
			Pid childID = ConvertEx::Get4(&page->Data[page->HdrOffset + 8]);
			PtrmapPut(bt, childID, PTRMAP_BTREE, id, &rc);
		}

set_child_ptrmaps_out:
		page->IsInit = isInitOrig;
		return rc;
	}

	__device__ static RC ModifyPagePointer(MemPage *page, Pid from, Pid to, PTRMAP type)
	{
		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(Pager::Iswriteable(page->DBPage));
		if (type == PTRMAP_OVERFLOW2)
		{
			// The pointer is always the first 4 bytes of the page in this case.
			if (ConvertEx::Get4(page->Data) != from)
				return SysEx_CORRUPT_BKPT;
			ConvertEx::Put4(page->Data, to);
		}
		else
		{
			bool isInitOrig = page->IsInit;

			BtreeInitPage(page);
			uint16 cells = page->Cells;

			uint i;
			for (i = 0; i < cells; i++)
			{
				uint8 *cell = FindCell(page, i);
				if (type == PTRMAP_OVERFLOW1)
				{
					CellInfo info;
					BtreeParseCellPtr(page, cell, &info);
					if (info.Overflow &&
						cell + info.Overflow + 3 <= page->Data + page->MaskPage &&
						from == ConvertEx::Get4(&cell[info.Overflow]))
					{
						ConvertEx::Put4(&cell[info.Overflow], to);
						break;
					}
				}
				else
					if (ConvertEx::Get4(cell) == from)
					{
						ConvertEx::Put4(cell, to);
						break;
					}
			}

			if (i == cells)
			{
				if (type != PTRMAP_BTREE || ConvertEx::Get4(&page->Data[page->HdrOffset + 8]) != from)
					return SysEx_CORRUPT_BKPT;
				ConvertEx::Put4(&page->Data[page->HdrOffset + 8], to);
			}

			page->IsInit = isInitOrig;
		}
		return RC_OK;
	}

	__device__ static RC RelocatePage(BtShared *bt, MemPage *page, PTRMAP type, Pid ptrPageID, Pid freePageID, bool isCommit)
	{
		_assert(type == PTRMAP_OVERFLOW2 || type == PTRMAP_OVERFLOW1 || type == PTRMAP_BTREE || type == PTRMAP_ROOTPAGE);
		_assert(MutexEx::Held(bt->Mutex));
		_assert(page->Bt == bt);

		// Move page iDbPage from its current location to page number iFreePage
		Pid lastID = page->ID;
		TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", lastID, freePageID, ptrPageID, type);
		Pager *pager = bt->Pager;
		RC rc = pager->Movepage(page->DBPage, freePageID, isCommit);
		if (rc != RC_OK)
			return rc;
		page->ID = freePageID;

		// If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
		// pages need to be changed.
		//
		// If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
		// the pointer map needs to be updated for the subsequent overflow page.
		if (type == PTRMAP_BTREE || type == PTRMAP_ROOTPAGE)
		{
			rc = SetChildPtrmaps(page);
			if (rc != RC_OK)
				return rc;
		}
		else
		{
			Pid nextOvfl = ConvertEx::Get4(page->Data);
			if (nextOvfl != 0)
			{
				PtrmapPut(bt, nextOvfl, PTRMAP_OVERFLOW2, freePageID, &rc);
				if (rc != RC_OK)
					return rc;
			}
		}

		// Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
		if (type != PTRMAP_ROOTPAGE)
		{
			MemPage *ptrPage; // The page that contains a pointer to pDbPage
			rc = BtreeGetPage(bt, ptrPageID, &ptrPage, false);
			if (rc != RC_OK)
				return rc;
			rc = Pager::Write(ptrPage->DBPage);
			if (rc != RC_OK)
			{
				ReleasePage(ptrPage);
				return rc;
			}
			rc = ModifyPagePointer(ptrPage, lastID, freePageID, type);
			ReleasePage(ptrPage);
			if (rc == RC_OK)
				PtrmapPut(bt, freePageID, type, ptrPageID, &rc);
		}
		return rc;
	}

	__device__ static RC AllocateBtreePage(BtShared *bt, MemPage **page, Pid *id, Pid nearby, BTALLOC mode);
	__device__ static RC IncrVacuumStep(BtShared *bt, Pid fins, Pid lastPageID, bool commit)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(lastPageID > fins);

		if (!PTRMAP_ISPAGE(bt, lastPageID) && lastPageID != PENDING_BYTE_PAGE(bt))
		{
			Pid freesList = ConvertEx::Get4(&bt->Page1->Data[36]); // Number of pages still on the free-list
			if (freesList == 0)
				return RC_DONE;

			PTRMAP type;
			Pid ptrPageID;
			RC rc = PtrmapGet(bt, lastPageID, &type, &ptrPageID);
			if (rc != RC_OK)
				return rc;
			if (type == PTRMAP_ROOTPAGE)
				return SysEx_CORRUPT_BKPT;

			if (type == PTRMAP_FREEPAGE)
			{
				if (!commit)
				{
					// Remove the page from the files free-list. This is not required if bCommit is non-zero. In that case, the free-list will be
					// truncated to zero after this function returns, so it doesn't matter if it still contains some garbage entries.
					Pid freePageID;
					MemPage *FreePage;
					rc = AllocateBtreePage(bt, &FreePage, &freePageID, lastPageID, BTALLOC::EXACT);
					if (rc != RC_OK)
						return rc;
					_assert(freePageID == lastPageID);
					ReleasePage(FreePage);
				}
			} 
			else 
			{
				MemPage *lastPage;
				rc = BtreeGetPage(bt, lastPageID, &lastPage, false);
				if (rc != RC_OK)
					return rc;

				// If bCommit is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
				//
				// On the other hand, if bCommit is greater than zero, then keep looping until a free-page located within the first nFin pages
				// of the file is found.
				BTALLOC mode = BTALLOC::ANY; // Mode parameter for AllocateBtreePage()
				Pid nearID = 0; // nearby parameter for AllocateBtreePage()
				if (!commit)
				{
					mode = BTALLOC::LE;
					nearID = fins;
				}
				Pid freePageID; // Index of free page to move pLastPg to
				do
				{
					MemPage *FreePage;
					rc = AllocateBtreePage(bt, &FreePage, &freePageID, nearID, mode);
					if (rc != RC_OK)
					{
						ReleasePage(lastPage);
						return rc;
					}
					ReleasePage(FreePage);
				} while (commit && freePageID > fins);
				_assert(freePageID < lastPageID);

				rc = RelocatePage(bt, lastPage, type, ptrPageID, freePageID, commit);
				ReleasePage(lastPage);
				if (rc != RC_OK)
					return rc;
			}
		}

		if (!commit)
		{
			do
			{
				lastPageID--;
			} while (lastPageID == PENDING_BYTE_PAGE(bt) || PTRMAP_ISPAGE(bt, lastPageID));
			bt->DoTruncate = true;
			bt->Pages = lastPageID;
		}
		return RC_OK;
	}

	__device__ static Pid FinalDbSize(BtShared *bt, Pid origs, Pid frees)
	{
		int entrys = bt->UsableSize / 5; // Number of entries on one ptrmap page
		Pid ptrmaps = (frees - origs + PTRMAP_PAGENO(bt, origs) + entrys) / entrys; // Number of PtrMap pages to be freed
		Pid fins = origs - frees - ptrmaps; // Return value
		if (origs > PENDING_BYTE_PAGE(bt) && fins < PENDING_BYTE_PAGE(bt))
			fins--;
		while (PTRMAP_ISPAGE(bt, fins) || fins == PENDING_BYTE_PAGE(bt))
			fins--;
		return fins;
	}

	__device__ RC Btree::IncrVacuum()
	{
		BtShared *bt = Bt;

		Enter();
		_assert(bt->InTransaction == TRANS_WRITE && InTrans == TRANS_WRITE);
		RC rc;
		if (!bt->AutoVacuum)
			rc = RC_DONE;
		else
		{
			Pid origs = BtreePagecount(bt);
			Pid frees = ConvertEx::Get4(&bt->Page1->Data[36]);
			Pid fins = FinalDbSize(bt, origs, frees);

			if (origs < fins)
				rc = SysEx_CORRUPT_BKPT;
			else if (frees > 0)
			{
				InvalidateAllOverflowCache(bt);
				rc = IncrVacuumStep(bt, fins, origs, false);
				if (rc == RC_OK)
				{
					rc = Pager::Write(bt->Page1->DBPage);
					ConvertEx::Put4(&bt->Page1->Data[28], bt->Pages);
				}
			}
			else
				rc = RC_DONE;
		}
		Leave();
		return rc;
	}

	__device__ static RC AutoVacuumCommit(BtShared *bt)
	{
		Pager *pager = bt->Pager;
		ASSERTONLY(int refs = pager->get_Refs());

		_assert(MutexEx::Held(bt->Mutex));
		InvalidateAllOverflowCache(bt);
		_assert(bt->AutoVacuum);
		RC rc = RC_OK;
		if (!bt->IncrVacuum)
		{
			Pid origs = BtreePagecount(bt); // Database size before freeing
			if (PTRMAP_ISPAGE(bt, origs) || origs == PENDING_BYTE_PAGE(bt))
			{
				// It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
				// is encountered, this indicates corruption.
				return SysEx_CORRUPT_BKPT;
			}

			Pid frees = ConvertEx::Get4(&bt->Page1->Data[36]); // Number of pages on the freelist initially
			Pid fins = FinalDbSize(bt, origs, frees); // Number of pages in database after autovacuuming
			if (fins > origs) return SysEx_CORRUPT_BKPT;

			for (Pid freeID = origs; freeID > fins && rc == RC_OK; freeID--) // The next page to be freed
				rc = IncrVacuumStep(bt, fins, freeID, true);
			if ((rc == RC_DONE || rc == RC_OK) && frees > 0)
			{
				rc = Pager::Write(bt->Page1->DBPage);
				ConvertEx::Put4(&bt->Page1->Data[32], 0);
				ConvertEx::Put4(&bt->Page1->Data[36], 0);
				ConvertEx::Put4(&bt->Page1->Data[28], fins);
				bt->DoTruncate = true;
				bt->Pages = fins;
			}
			if (rc != RC_OK)
				pager->Rollback();
		}

		_assert(refs == pager->get_Refs());
		return rc;
	}

#else
	//#define SetChildPtrmaps(x) RC_OK
#endif
#pragma endregion

#pragma region Commit / Rollback

	__device__ RC Btree::CommitPhaseOne(const char *master)
	{
		RC rc = RC_OK;
		if (InTrans == TRANS_WRITE)
		{
			BtShared *bt = Bt;
			Enter();
#ifndef OMIT_AUTOVACUUM
			if (bt->AutoVacuum)
			{
				rc = AutoVacuumCommit(bt);
				if (rc != RC_OK)
				{
					Leave();
					return rc;
				}
			}
			if (bt->DoTruncate)
				bt->Pager->TruncateImage(bt->Pages);
#endif
			rc = bt->Pager->CommitPhaseOne(master, false);
			Leave();
		}
		return rc;
	}

	__device__ static void BtreeEndTransaction(Btree *p)
	{
		BtShared *bt = p->Bt;
		_assert(p->HoldsMutex());

#ifndef OMIT_AUTOVACUUM
		bt->DoTruncate = false;
#endif
		BtreeClearHasContent(bt);
		if (p->InTrans > TRANS_NONE && p->Ctx->ActiveVdbeCnt > 1)
		{
			// If there are other active statements that belong to this database handle, downgrade to a read-only transaction. The other statements
			// may still be reading from the database.
			DowngradeAllSharedCacheTableLocks(p);
			p->InTrans = TRANS_READ;
		}
		else
		{
			// If the handle had any kind of transaction open, decrement the transaction count of the shared btree. If the transaction count 
			// reaches 0, set the shared state to TRANS_NONE. The UnlockBtreeIfUnused() call below will unlock the pager.
			if (p->InTrans != TRANS_NONE)
			{
				ClearAllSharedCacheTableLocks(p);
				bt->Transactions--;
				if (bt->Transactions == 0)
					bt->InTransaction = TRANS_NONE;
			}

			// Set the current transaction state to TRANS_NONE and unlock the  pager if this call closed the only read or write transaction.
			p->InTrans = TRANS_NONE;
			UnlockBtreeIfUnused(bt);
		}

		BtreeIntegrity(p);
	}

	__device__ RC Btree::CommitPhaseTwo(bool cleanup)
	{
		if (InTrans == TRANS_NONE) return RC_OK;
		Enter();
		BtreeIntegrity(this);

		// If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
		if (InTrans == TRANS_WRITE)
		{
			BtShared *bt = Bt;
			_assert(bt->InTransaction == TRANS_WRITE);
			_assert(bt->Transactions > 0);
			RC rc = bt->Pager->CommitPhaseTwo();
			if (rc != RC_OK && !cleanup)
			{
				Leave();
				return rc;
			}
			bt->InTransaction = TRANS_READ;
		}

		BtreeEndTransaction(this);
		Leave();
		return RC_OK;
	}

	__device__ RC Btree::Commit()
	{
		Enter();
		RC rc = CommitPhaseOne(nullptr);
		if (rc == RC_OK)
			rc = CommitPhaseTwo(false);
		Leave();
		return rc;
	}


#if _DEBUG
	__device__ static int CountWriteCursors(BtShared *bt)
	{
		int r = 0;
		for (BtCursor *cur = bt->Cursor; cur; cur = cur->Next)
			if (cur->WrFlag && cur->State != CURSOR_FAULT) r++; 
		return r;
	}
#endif

	__device__ void Btree::TripAllCursors(RC errCode)
	{
		Enter();
		for (BtCursor *p = Bt->Cursor; p; p = p->Next)
		{
			ClearCursor(p);
			p->State = CURSOR_FAULT;
			p->SkipNext = errCode;
			for (int i = 0; i <= p->ID; i++)
			{
				ReleasePage(p->Pages[i]);
				p->Pages[i] = nullptr;
			}
		}
		Leave();
	}

	__device__ RC Btree::Rollback(RC tripCode)
	{
		BtShared *bt = Bt;

		Enter();
		RC rc;
		if (tripCode == RC_OK)
			rc = tripCode = SaveAllCursors(bt, 0, nullptr);
		else
			rc = RC_OK;
		if (tripCode)
			TripAllCursors(tripCode);
		BtreeIntegrity(this);

		if (InTrans == TRANS_WRITE)
		{
			_assert(bt->InTransaction == TRANS_WRITE);
			RC rc2 = bt->Pager->Rollback();
			if (rc2 != RC_OK)
				rc = rc2;

			// The rollback may have destroyed the pPage1->aData value. So call BtreeGetPage() on page 1 again to make
			// sure pPage1->aData is set correctly.
			MemPage *page1;
			if (BtreeGetPage(bt, 1, &page1, false) == RC_OK)
			{
				Pid pages = ConvertEx::Get4(28 + (uint8 *)page1->Data);
				ASSERTCOVERAGE(pages == 0);
				if (pages == 0) bt->Pager->Pages(&pages);
				ASSERTCOVERAGE(bt->Pages != pages);
				bt->Pages = pages;
				ReleasePage(page1);
			}
			_assert(CountWriteCursors(bt) == 0);
			bt->InTransaction = TRANS_READ;
		}

		BtreeEndTransaction(this);
		Leave();
		return rc;
	}

	__device__ RC Btree::BeginStmt(int statements)
	{
		BtShared *bt = Bt;
		Enter();
		_assert(InTrans == TRANS_WRITE);
		_assert((bt->BtsFlags & BTS_READ_ONLY) == 0);
		_assert(statements > 0);
		_assert(statements > Ctx->SavepointsLength);
		_assert(bt->InTransaction == TRANS_WRITE);
		// At the pager level, a statement transaction is a savepoint with an index greater than all savepoints created explicitly using
		// SQL statements. It is illegal to open, release or rollback any such savepoints while the statement transaction savepoint is active.
		RC rc = bt->Pager->OpenSavepoint(statements);
		Leave();
		return rc;
	}

	__device__ RC Btree::Savepoint(IPager::SAVEPOINT op, int savepoints)
	{
		RC rc = RC_OK;
		if (InTrans == TRANS_WRITE)
		{
			BtShared *bt = Bt;
			_assert(op == IPager::SAVEPOINT_RELEASE || op == IPager::SAVEPOINT_ROLLBACK);
			_assert(savepoints >= 0 || (savepoints == -1 && op == IPager::SAVEPOINT_ROLLBACK));
			Enter();
			rc = bt->Pager->Savepoint(op, savepoints);
			if (rc == RC_OK)
			{
				if (savepoints < 0 && (bt->BtsFlags & BTS_INITIALLY_EMPTY) != 0)
					bt->Pages = 0;
				rc = NewDatabase(bt);
				bt->Pages = ConvertEx::Get4(28 + bt->Page1->Data);

				// The database size was written into the offset 28 of the header when the transaction started, so we know that the value at offset
				// 28 is nonzero.
				_assert(bt->Pages > 0);
			}
			Leave();
		}
		return rc;
	}

#pragma endregion

#pragma region Cursors

	__device__ static RC BtreeCursor(Btree *p, Pid tableID, bool wrFlag, KeyInfo *keyInfo, BtCursor *cur)
	{
		BtShared *bt = p->Bt; // Shared b-tree handle

		_assert(p->HoldsMutex());

		// The following assert statements verify that if this is a sharable b-tree database, the connection is holding the required table locks, 
		// and that no other connection has any open cursor that conflicts with this lock.
		_assert(HasSharedCacheTableLock(p, tableID, keyInfo != nullptr, (LOCK)(wrFlag + 1)));
		_assert(!wrFlag || !HasReadConflicts(p, tableID));

		// Assert that the caller has opened the required transaction.
		_assert(p->InTrans > TRANS_NONE);
		_assert(!wrFlag || p->InTrans == TRANS_WRITE);
		_assert(bt->Page1 && bt->Page1->Data);

		if (_NEVER(wrFlag && (bt->BtsFlags & BTS_READ_ONLY) != 0))
			return RC_READONLY;
		if (tableID == 1 && BtreePagecount(bt) == 0)
		{
			_assert(!wrFlag);
			tableID = 0;
		}

		// Now that no other errors can occur, finish filling in the BtCursor variables and link the cursor into the BtShared list.
		cur->RootID = tableID;
		cur->ID = -1;
		cur->KeyInfo = keyInfo;
		cur->Btree = p;
		cur->Bt = bt;
		cur->WrFlag = wrFlag;
		cur->Next = bt->Cursor;
		if (cur->Next)
			cur->Next->Prev = cur;
		bt->Cursor = cur;
		cur->State = CURSOR_INVALID;
		cur->CachedRowID = 0;
		return RC_OK;
	}

	__device__ RC Btree::Cursor(Pid tableID, bool wrFlag, KeyInfo *keyInfo, BtCursor *cur)
	{
		Enter();
		RC rc = BtreeCursor(this, tableID, wrFlag, keyInfo, cur);
		Leave();
		return rc;
	}

	__device__ int Btree::CursorSize()
	{
		return _ROUND8(sizeof(BtCursor));
	}

	__device__ void Btree::CursorZero(BtCursor *p)
	{
		_memset(p, 0, offsetof(BtCursor, Pages));
	}

	__device__ void Btree::SetCachedRowID(BtCursor *cur, int64 rowid)
	{
		for (BtCursor *p = cur->Bt->Cursor; p; p = p->Next)
			if (p->RootID == cur->RootID) p->CachedRowID = rowid;
		_assert(cur->CachedRowID == rowid);
	}

	__device__ int64 Btree::GetCachedRowID(BtCursor *cur)
	{
		return cur->CachedRowID;
	}

	__device__ RC Btree::CloseCursor(BtCursor *cur)
	{
		Btree *btree = cur->Btree;
		if (btree)
		{
			BtShared *bt = cur->Bt;
			btree->Enter();
			ClearCursor(cur);
			if (cur->Prev)
				cur->Prev->Next = cur->Next;
			else
				bt->Cursor = cur->Next;
			if (cur->Next)
				cur->Next->Prev = cur->Prev;
			for (int i = 0; i <= cur->ID; i++)
				ReleasePage(cur->Pages[i]);
			UnlockBtreeIfUnused(bt);
			InvalidateOverflowCache(cur);
			// _free(cur);
			btree->Leave();
		}
		return RC_OK;
	}

#ifdef _DEBUG
	__device__ static void AssertCellInfo(BtCursor *cur)
	{
		int id = cur->ID;
		CellInfo info;
		_memset(&info, 0, sizeof(info));
		BtreeParseCell(cur->Pages[id], cur->Idxs[id], &info);
		_assert(_memcmp(&info, &cur->Info, sizeof(info)) == 0);
	}
#else
#define AssertCellInfo(x)
#endif

#ifdef _MSC_VER
	__device__ static void GetCellInfo(BtCursor *cur)
	{
		if (cur->Info.Size == 0)
		{
			int id = cur->ID;
			BtreeParseCell(cur->Pages[id], cur->Idxs[id], &cur->Info);
			cur->ValidNKey = 1;
		}
		else
			AssertCellInfo(cur);
	}
#else
#define GetCellInfo(cur) \
	if (cur->info.Size == 0) { \
	int page = cur->Page; \
	BtreeParseCell(cur->Pages[page], cur->Idxs[page], &cur->info); \
	cur->ValidNKey = 1; \
	} else AssertCellInfo(cur);
#endif

#ifdef _DEBUG
	__device__ bool Btree::CursorIsValid(BtCursor *cur)
	{
		return cur && cur->State == CURSOR_VALID;
	}
#endif

	__device__ RC Btree::KeySize(BtCursor *cur, int64 *size)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_INVALID || cur->State == CURSOR_VALID);
		if (cur->State != CURSOR_VALID)
			*size = 0;
		else
		{
			GetCellInfo(cur);
			*size = cur->Info.Key;
		}
		return RC_OK;
	}

	__device__ RC Btree::DataSize(BtCursor *cur, uint32 *size)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		GetCellInfo(cur);
		*size = cur->Info.Data;
		return RC_OK;
	}

#pragma endregion

#pragma region Payload / Overflow

	__device__ static RC GetOverflowPage(BtShared *bt, Pid ovfl, MemPage **pageOut, Pid *idNextOut)
	{
		Pid next = 0;
		MemPage *page = nullptr;
		RC rc = RC_OK;

		_assert(MutexEx::Held(bt->Mutex));
		_assert(idNextOut != nullptr);

#ifndef OMIT_AUTOVACUUM
		// Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in 
		// the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page 
		// number ovfl to determine the next page number.
		if (bt->AutoVacuum)
		{
			Pid guess = ovfl + 1;
			while (PTRMAP_ISPAGE(bt, guess) || guess == PENDING_BYTE_PAGE(bt))
				guess++;
			if (guess <= BtreePagecount(bt))
			{
				Pid id;
				PTRMAP type;
				rc = PtrmapGet(bt, guess, &type, &id);
				if (rc == RC_OK && type == PTRMAP_OVERFLOW2 && id == ovfl)
				{
					next = guess;
					rc = RC_DONE;
				}
			}
		}
#endif

		_assert(next == 0 || rc == RC_DONE);
		if (rc == RC_OK)
		{
			rc = BtreeGetPage(bt, ovfl, &page, false);
			_assert(rc == RC_OK || page == 0);
			if (rc == RC_OK)
				next = ConvertEx::Get4(page->Data);
		}

		*idNextOut = next;
		if (pageOut)
			*pageOut = page;
		else
			ReleasePage(page);
		return (rc == RC_DONE ? RC_OK : rc);
	}

	__device__ static RC CopyPayload(void *payload, void *buf, int bytes, int op, IPage *dbPage)
	{
		if (op)
		{
			// Copy data from buffer to page (a write operation)
			RC rc = Pager::Write(dbPage);
			if (rc != RC_OK)
				return rc;
			_memcpy(payload, buf, bytes);
		}
		else
			// Copy data from page to buffer (a read operation)
			_memcpy(buf, payload, bytes);
		return RC_OK;
	}

	__device__ static RC AccessPayload(BtCursor *cur, uint32 offset, uint32 amount, uint8 *buf, int op)
	{
		MemPage *page = cur->Pages[cur->ID]; // Btree page of current entry

		_assert(page != nullptr);
		_assert(cur->State == CURSOR_VALID);
		_assert(cur->Idxs[cur->ID] < page->Cells);
		_assert(CursorHoldsMutex(cur));

		GetCellInfo(cur);
		uint8 *payload = cur->Info.Cell + cur->Info.Header;
		uint32 key = (page->IntKey ? 0 : (int)cur->Info.Key);

		BtShared *bt = cur->Bt; // Btree this cursor belongs to
		if (_NEVER(offset + amount > key + cur->Info.Data) || &payload[cur->Info.Local] > &page->Data[bt->UsableSize])
			// Trying to read or write past the end of the data is an error
				return SysEx_CORRUPT_BKPT;

		// Check if data must be read/written to/from the btree page itself.
		uint idx = 0U;
		RC rc = RC_OK;
		if (offset < cur->Info.Local)
		{
			int a = amount;
			if (a + offset > cur->Info.Local)
				a = cur->Info.Local - offset;
			rc = CopyPayload(&payload[offset], buf, a, op, page->DBPage);
			offset = 0;
			buf += a;
			amount -= a;
		}
		else
			offset -= cur->Info.Local;

		if (rc == RC_OK && amount > 0)
		{
			const uint32 ovflSize = bt->UsableSize - 4; // Bytes content per ovfl page
			Pid nextPage = ConvertEx::Get4(&payload[cur->Info.Local]);

#ifndef OMIT_INCRBLOB
			// If the isIncrblobHandle flag is set and the BtCursor.aOverflow[] has not been allocated, allocate it now. The array is sized at
			// one entry for each overflow page in the overflow chain. The page number of the first overflow page is stored in aOverflow[0],
			// etc. A value of 0 in the aOverflow[] array means "not yet known" (the cache is lazily populated).
			if (cur->IsIncrblobHandle && !cur->Overflows)
			{
				uint ovfl = (cur->Info.Payload - cur->Info.Local + ovflSize - 1) / ovflSize;
				cur->Overflows = (Pid *)_alloc2(sizeof(Pid) * ovfl, true);
				// nOvfl is always positive.  If it were zero, FetchPayload would have been used instead of this routine.
				if (_ALWAYS(ovfl) && !cur->Overflows)
					rc = RC_NOMEM;
			}

			// If the overflow page-list cache has been allocated and the entry for the first required overflow page is valid, skip
			// directly to it.
			if (cur->Overflows && cur->Overflows[offset / ovflSize])
			{
				idx = (offset / ovflSize);
				nextPage = cur->Overflows[idx];
				offset = (offset % ovflSize);
			}
#endif

			for ( ; rc == RC_OK && amount > 0 && nextPage; idx++)
			{
#ifndef OMIT_INCRBLOB
				// If required, populate the overflow page-list cache.
				if (cur->Overflows)
				{
					_assert(!cur->Overflows[idx] || cur->Overflows[idx] == nextPage);
					cur->Overflows[idx] = nextPage;
				}
#endif

				if (offset >= ovflSize)
				{
					// The only reason to read this page is to obtain the page number for the next page in the overflow chain. The page
					// data is not required. So first try to lookup the overflow page-list cache, if any, then fall back to the GetOverflowPage() function.
#ifndef OMIT_INCRBLOB
					if (cur->Overflows && cur->Overflows[idx + 1])
						nextPage = cur->Overflows[idx + 1];
					else 
#endif
						rc = GetOverflowPage(bt, nextPage, nullptr, &nextPage);
					offset -= ovflSize;
				}
				else
				{
					// Need to read this page properly. It contains some of the range of data that is being read (eOp==0) or written (eOp!=0).
					int a = amount;
					if (a + offset > ovflSize)
						a = ovflSize - offset;

#ifdef DIRECT_OVERFLOW_READ
					// If all the following are true:
					//
					//   1) this is a read operation, and 
					//   2) data is required from the start of this overflow page, and
					//   3) the database is file-backed, and
					//   4) there is no open write-transaction, and
					//   5) the database is not a WAL database,
					//
					// then data can be read directly from the database file into the output buffer, bypassing the page-cache altogether. This speeds
					// up loading large records that span many overflow pages.
					VFile *fd;
					if (op == 0 && // (1)
						offset == 0 && // (2)
						bt->InTransaction == TRANS_READ && // (4)
						(fd = bt->Pager->File())->Methods && // (3)
						bt->Page1->Data[19] == 0x01) // (5)
					{
						uint8 save[4];
						uint8 *write = &buf[-4];
						_memcpy(save, write, 4);
						rc = fd->Read(write, a + 4, (int64)bt->PageSize * (nextPage - 1));
						nextPage = ConvertEx::Get4(write);
						_memcpy(write, save, 4);
					}
					else
#endif

					{
						IPage *dbPage;
						rc = bt->Pager->Acquire(nextPage, &dbPage, false);
						if (rc == RC_OK)
						{
							payload = (uint8 *)Pager::GetData(dbPage);
							nextPage = ConvertEx::Get4(payload);
							rc = CopyPayload(&payload[offset + 4], buf, a, op, dbPage);
							Pager::Unref(dbPage);
							offset = 0;
						}
					}
					amount -= a;
					buf += a;
				}
			}
		}

		if (rc == RC_OK && amount > 0)
			return SysEx_CORRUPT_BKPT;
		return rc;
	}

	__device__ RC Btree::Key(BtCursor *cur, uint32 offset, uint32 amount, void *buf)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		_assert(cur->ID >= 0 && cur->Pages[cur->ID]);
		_assert(cur->Idxs[cur->ID] < cur->Pages[cur->ID]->Cells);
		return AccessPayload(cur, offset, amount, (uint8 *)buf, 0);
	}

	__device__ RC Btree::Data(BtCursor *cur, uint32 offset, uint32 amount, void *buf)
	{
#ifndef OMIT_INCRBLOB
		if (cur->State == CURSOR_INVALID)
			return RC_ABORT;
#endif

		_assert(CursorHoldsMutex(cur));
		RC rc = restoreCursorPosition(cur);
		if (rc == RC_OK)
		{
			_assert(cur->State == CURSOR_VALID);
			_assert(cur->ID >= 0 && cur->Pages[cur->ID]);
			_assert(cur->Idxs[cur->ID] < cur->Pages[cur->ID]->Cells);
			rc = AccessPayload(cur, offset, amount, (uint8 *)buf, 0);
		}
		return rc;
	}

	__device__ static const unsigned char *FetchPayload(BtCursor *cur, int *amount, bool skipKey)
	{
		_assert(cur != nullptr && cur->ID >= 0 && cur->Pages[cur->ID]);
		_assert(cur->State == CURSOR_VALID);
		_assert(CursorHoldsMutex(cur) );
		MemPage *page = cur->Pages[cur->ID];
		_assert(cur->Idxs[cur->ID] < page->Cells);
		if (_NEVER(cur->Info.Size == 0))
			BtreeParseCell(cur->Pages[cur->ID], cur->Idxs[cur->ID], &cur->Info);
		uint8 *payload = cur->Info.Cell + cur->Info.Header;
		uint32 key = (page->IntKey ? 0U : (int)cur->Info.Key);
		int local;
		if (skipKey)
		{
			payload += key;
			local = cur->Info.Local - key;
		}
		else
		{
			local = cur->Info.Local;
			_assert(local <= key);
		}
		*amount = local;
		return payload;
	}

	__device__ const void *Btree::KeyFetch(BtCursor *cur, int *amount)
	{
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		_assert(CursorHoldsMutex(cur));
		const void *p = nullptr;
		if (_ALWAYS(cur->State == CURSOR_VALID))
			p = (const void*)FetchPayload(cur, amount, false);
		return p;
	}

	__device__ const void *Btree::DataFetch(BtCursor *cur, int *amount)
	{
		const void *p = 0;
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		_assert(CursorHoldsMutex(cur));
		if (_ALWAYS(cur->State == CURSOR_VALID))
			p = (const void*)FetchPayload(cur, amount, true);
		return p;
	}

#pragma endregion

#pragma region Move Cursor

	__device__ static RC MoveToChild(BtCursor *cur, uint32 newID)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		_assert(cur->ID < BTCURSOR_MAX_DEPTH );
		if (cur->ID >= (BTCURSOR_MAX_DEPTH - 1))
			return SysEx_CORRUPT_BKPT;
		BtShared *bt = cur->Bt;
		MemPage *newPage;
		RC rc = GetAndInitPage(bt, newID, &newPage);
		if (rc) return rc;
		int i = cur->ID;
		cur->Pages[i + 1] = newPage;
		cur->Idxs[i + 1] = 0;
		cur->ID++;

		cur->Info.Size = 0;
		cur->ValidNKey = 0;
		if (newPage->Cells < 1 || newPage->IntKey != cur->Pages[i]->IntKey)
			return SysEx_CORRUPT_BKPT;
		return RC_OK;
	}

#if 0
	__device__ static void AssertParentIndex(MemPage *parent, int idx, Pid child)
	{
		_assert(idx <= parent->Cells);
		if (idx == parent->Cells)
			_assert(ConvertEx::Get4(&parent->Data[parent->HdrOffset + 8]) == child);
		else
			_assert(ConvertEx::Get4(FindCell(parent, idx)) == child);
	}
#else
#define AssertParentIndex(x,y,z) 
#endif

	__device__ static void MoveToParent(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		_assert(cur->ID > 0);
		_assert(cur->Pages[cur->ID] != nullptr);

		// UPDATE: It is actually possible for the condition tested by the assert below to be untrue if the database file is corrupt. This can occur if
		// one cursor has modified page pParent while a reference to it is held by a second cursor. Which can only happen if a single page is linked
		// into more than one b-tree structure in a corrupt database.
#if 0
		AssertParentIndex(cur->Pages[cur->Page - 1], cur->Idxs[cur->Page - 1], cur->Pages[cur->Page]->ID);
#endif
		ASSERTCOVERAGE(cur->Idxs[cur->ID - 1] > cur->Pages[cur->ID - 1]->Cells);

		ReleasePage(cur->Pages[cur->ID]);
		cur->ID--;
		cur->Info.Size = 0;
		cur->ValidNKey = 0;
	}

	__device__ static RC MoveToRoot(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(CURSOR_INVALID < CURSOR_REQUIRESEEK);
		_assert(CURSOR_VALID < CURSOR_REQUIRESEEK);
		_assert(CURSOR_FAULT > CURSOR_REQUIRESEEK);
		if (cur->State >= CURSOR_REQUIRESEEK)
		{
			if (cur->State == CURSOR_FAULT)
			{
				_assert(cur->SkipNext != 0);
				return (RC)cur->SkipNext;
			}
			Btree::ClearCursor(cur);
		}

		RC rc = RC_OK;
		BtShared *bt = cur->Btree->Bt;
		if (cur->ID >= 0)
		{
			for (int i = 1; i <= cur->ID; i++)
				ReleasePage(cur->Pages[i]);
			cur->ID = 0;
		}
		else if (cur->RootID == 0)
		{
			cur->State = CURSOR_INVALID;
			return RC_OK;
		}
		else
		{
			rc = GetAndInitPage(bt, cur->RootID, &cur->Pages[0]);
			if (rc != RC_OK)
			{
				cur->State = CURSOR_INVALID;
				return rc;
			}
			cur->ID = 0;

			// If pCur->pKeyInfo is not NULL, then the caller that opened this cursor expected to open it on an index b-tree. Otherwise, if pKeyInfo is
			// NULL, the caller expects a table b-tree. If this is not the case, return an SQLITE_CORRUPT error.
			if ((cur->KeyInfo == nullptr) != cur->Pages[0]->IntKey)
				return SysEx_CORRUPT_BKPT;
		}

		// Assert that the root page is of the correct type. This must be the case as the call to this function that loaded the root-page (either
		// this call or a previous invocation) would have detected corruption if the assumption were not true, and it is not possible for the flags 
		// byte to have been modified while this cursor is holding a reference to the page.
		MemPage *root = cur->Pages[0];
		_assert(root->ID == cur->RootID);
		_assert(root->IsInit && (cur->KeyInfo == nullptr) == root->IntKey);

		cur->Idxs[0] = 0;
		cur->Info.Size = 0;
		cur->AtLast = 0;
		cur->ValidNKey = 0;

		if (root->Cells == 0 && !root->Leaf)
		{
			if (root->ID != 1) return SysEx_CORRUPT_BKPT;
			Pid subpage = ConvertEx::Get4(&root->Data[root->HdrOffset + 8]);
			cur->State = CURSOR_VALID;
			rc = MoveToChild(cur, subpage);
		}
		else
			cur->State = (root->Cells > 0 ? CURSOR_VALID : CURSOR_INVALID);
		return rc;
	}

	__device__ static RC MoveToLeftmost(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		MemPage *page;
		RC rc = RC_OK;
		while (rc == RC_OK && !(page = cur->Pages[cur->ID])->Leaf)
		{
			_assert(cur->Idxs[cur->ID] < page->Cells);
			Pid id = ConvertEx::Get4(FindCell(page, cur->Idxs[cur->ID]));
			rc = MoveToChild(cur, id);
		}
		return rc;
	}

	__device__ static RC MoveToRightmost(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(cur->State == CURSOR_VALID);
		MemPage *page = nullptr;
		RC rc = RC_OK;
		while (rc == RC_OK && !(page = cur->Pages[cur->ID])->Leaf)
		{
			Pid id = ConvertEx::Get4(&page->Data[page->HdrOffset + 8]);
			cur->Idxs[cur->ID] = page->Cells;
			rc = MoveToChild(cur, id);
		}
		if (rc == RC_OK)
		{
			cur->Idxs[cur->ID] = page->Cells - 1;
			cur->Info.Size = 0;
			cur->ValidNKey = 0;
		}
		return rc;
	}

	__device__ RC Btree::First(BtCursor *cur, int *eof)
	{
		_assert(CursorHoldsMutex(cur) );
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		RC rc = MoveToRoot(cur);
		if (rc == RC_OK)
		{
			if (cur->State == CURSOR_INVALID)
			{
				_assert(cur->RootID == 0 || cur->Pages[cur->ID]->Cells == 0);
				*eof = 1;
			}
			else
			{
				_assert(cur->Pages[cur->ID]->Cells > 0);
				*eof = 0;
				rc = MoveToLeftmost(cur);
			}
		}
		return rc;
	}

	__device__ RC Btree::Last(BtCursor *cur, int *eof)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));

		// If the cursor already points to the last entry, this is a no-op.
		if (cur->State == CURSOR_VALID && cur->AtLast)
		{
#ifdef _DEBUG
			// This block serves to assert() that the cursor really does point to the last entry in the b-tree.
			for (int ii = 0; ii < cur->ID; ii++)
				_assert(cur->Idxs[ii] == cur->Pages[ii]->Cells);
			_assert(cur->Idxs[cur->ID] == cur->Pages[cur->ID]->Cells - 1);
			_assert(cur->Pages[cur->ID]->Leaf);
#endif
			return RC_OK;
		}

		RC rc = MoveToRoot(cur);
		if (rc == RC_OK)
		{
			if (cur->State == CURSOR_INVALID)
			{
				_assert(cur->RootID == 0 || cur->Pages[cur->ID]->Cells == 0);
				*eof = 1;
			}
			else
			{
				_assert(cur->State == CURSOR_VALID);
				*eof = 0;
				rc = MoveToRightmost(cur);
				cur->AtLast = (rc == RC_OK ? 1 : 0);
			}
		}
		return rc;
	}

	__device__ RC Btree::MovetoUnpacked(BtCursor *cur, UnpackedRecord *idxKey, int64 intKey, int biasRight, int *eof)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		_assert(eof != nullptr);
		_assert((idxKey == nullptr) == (cur->KeyInfo == nullptr));

		// If the cursor is already positioned at the point we are trying to move to, then just return without doing any work
		if (cur->State == CURSOR_VALID && cur->ValidNKey && cur->Pages[0]->IntKey)
		{
			if (cur->Info.Key == intKey)
			{
				*eof = 0;
				return RC_OK;
			}
			if (cur->AtLast && cur->Info.Key < intKey)
			{
				*eof = -1;
				return RC_OK;
			}
		}

		*eof = -1;
		RC rc = MoveToRoot(cur);
		if (rc)
			return rc;
		_assert(cur->RootID == 0 || cur->Pages[cur->ID]);
		_assert(cur->RootID == 0 || cur->Pages[cur->ID]->IsInit);
		_assert(cur->State == CURSOR_INVALID || cur->Pages[cur->ID]->Cells > 0);
		if (cur->State == CURSOR_INVALID)
		{
			*eof = -1;
			_assert(cur->RootID == 0 || cur->Pages[cur->ID]->Cells == 0);
			return RC_OK;
		}
		_assert(cur->Pages[0]->IntKey || idxKey);
		for (; ; )
		{
			// pPage->nCell must be greater than zero. If this is the root-page the cursor would have been INVALID above and this for(;;) loop
			// not run. If this is not the root-page, then the MoveToChild() routine would have already detected db corruption. Similarly, pPage must
			// be the right kind (index or table) of b-tree page. Otherwise a MoveToChild() or MoveToRoot() call would have detected corruption.
			MemPage *page = cur->Pages[cur->ID];
			_assert(page->Cells > 0);
			_assert(page->IntKey == (idxKey == 0));
			uint idx;
			uint lwr = 0;
			uint upr = page->Cells - 1;
			if (biasRight)
				cur->Idxs[cur->ID] = (uint16)(idx = upr);
			else
				cur->Idxs[cur->ID] = (uint16)(idx = (upr + lwr) / 2);
			int c;
			for (; ; )
			{
				_assert(idx == cur->Idxs[cur->ID]);
				cur->Info.Size = 0;
				uint8 *cell = FindCell(page, idx) + page->ChildPtrSize; // Pointer to current cell in pPage
				if (page->IntKey)
				{
					if (page->HasData)
					{
						uint32 dummy;
						cell += ConvertEx::GetVarint32(cell, &dummy);
					}
					int64 cellKeyLength;
					ConvertEx::GetVarint(cell, (uint64 *)&cellKeyLength);
					if (cellKeyLength == intKey)
						c = 0;
					else if (cellKeyLength < intKey)
						c = -1;
					else
						c = +1;
					cur->ValidNKey = 1;
					cur->Info.Key = cellKeyLength;
				}
				else
				{
					// The maximum supported page-size is 65536 bytes. This means that the maximum number of record bytes stored on an index B-Tree
					// page is less than 16384 bytes and may be stored as a 2-byte varint. This information is used to attempt to avoid parsing 
					// the entire cell by checking for the cases where the record is stored entirely within the b-tree page by inspecting the first 
					// 2 bytes of the cell.
					int cellLength = cell[0];
					if (cellLength <= page->Max1bytePayload /* && (cell + cellLength) < page->DataEnd */)
					{
						// This branch runs if the record-size field of the cell is a single byte varint and the record fits entirely on the main b-tree page.
						ASSERTCOVERAGE(cell + cellLength + 1 == page->DataEnd);
						c = _vdbe->RecordCompare(cellLength, (void *)&cell[1], idxKey);
					}
					else if (!(cell[1] & 0x80) && (cellLength = ((cellLength & 0x7f) << 7) + cell[1]) <= page->MaxLocal /* && (cell + cellLength + 2) <= page->DataEnd */)
					{
						// The record-size field is a 2 byte varint and the record fits entirely on the main b-tree page.
						ASSERTCOVERAGE(cell + cellLength + 2 == page->DataEnd);
						c = _vdbe->RecordCompare(cellLength, (void *)&cell[2], idxKey);
					}
					else
					{
						// The record flows over onto one or more overflow pages. In this case the whole cell needs to be parsed, a buffer allocated
						// and AccessPayload() used to retrieve the record into the buffer before VdbeRecordCompare() can be called.
						uint8 *const cellBody = cell - page->ChildPtrSize;
						BtreeParseCellPtr(page, cellBody, &cur->Info);
						cellLength = (int)cur->Info.Key;
						void *cellKey = _alloc(cellLength);
						if (cellKey == nullptr)
						{
							rc = RC_NOMEM;
							goto moveto_finish;
						}
						rc = AccessPayload(cur, 0, cellLength, (unsigned char *)cellKey, 0);
						if (rc)
						{
							_free(cellKey);
							goto moveto_finish;
						}
						c = _vdbe->RecordCompare(cellLength, cellKey, idxKey);
						_free(cellKey);
					}
				}
				if (c == 0)
				{
					if (page->IntKey && !page->Leaf)
					{
						lwr = idx;
						break;
					}
					else
					{
						*eof = 0;
						rc = RC_OK;
						goto moveto_finish;
					}
				}
				if (c < 0)
					lwr = idx + 1;
				else
					upr = idx - 1;
				if (lwr > upr)
					break;
				cur->Idxs[cur->ID] = (uint16)(idx = (lwr + upr) / 2);
			}
			_assert(lwr == upr + 1 || (page->IntKey && !page->Leaf));
			_assert(page->IsInit);
			Pid childID;
			if (page->Leaf)
				childID = 0;
			else if (lwr >= page->Cells)
				childID = ConvertEx::Get4(&page->Data[page->HdrOffset + 8]);
			else
				childID = ConvertEx::Get4(FindCell(page, lwr));
			if (childID == 0)
			{
				_assert(cur->Idxs[cur->ID] < cur->Pages[cur->ID]->Cells);
				*eof = c;
				rc = RC_OK;
				goto moveto_finish;
			}
			cur->Idxs[cur->ID] = (uint16)lwr;
			cur->Info.Size = 0;
			cur->ValidNKey = 0;
			rc = MoveToChild(cur, childID);
			if (rc) goto moveto_finish;
		}
moveto_finish:
		return rc;
	}

	__device__ bool Btree::Eof(BtCursor *cur)
	{
		// TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries have been deleted? This API will need to change to return an error code
		// as well as the boolean result value.
		return (cur->State != CURSOR_VALID);
	}

	__device__ RC Btree::Next_(BtCursor *cur, int *eof)
	{
		_assert(CursorHoldsMutex(cur));
		RC rc = restoreCursorPosition(cur);
		if (rc != RC_OK)
			return rc;
		_assert(eof != nullptr);
		if (cur->State == CURSOR_INVALID)
		{
			*eof = 1;
			return RC_OK;
		}
		if (cur->SkipNext > 0)
		{
			cur->SkipNext = 0;
			*eof = 0;
			return RC_OK;
		}
		cur->SkipNext = 0;

		MemPage *page = cur->Pages[cur->ID];
		uint idx = ++cur->Idxs[cur->ID];
		_assert(page->IsInit);

		// If the database file is corrupt, it is possible for the value of idx to be invalid here. This can only occur if a second cursor modifies
		// the page while cursor pCur is holding a reference to it. Which can only happen if the database is corrupt in such a way as to link the
		// page into more than one b-tree structure. */
		ASSERTCOVERAGE(idx > page->Cells);

		cur->Info.Size = 0;
		cur->ValidNKey = 0;
		if (idx >= page->Cells)
		{
			if (!page->Leaf)
			{
				rc = MoveToChild(cur, ConvertEx::Get4(&page->Data[page->HdrOffset + 8]));
				if (rc) return rc;
				rc = MoveToLeftmost(cur);
				*eof = 0;
				return rc;
			}
			do
			{
				if (cur->ID == 0)
				{
					*eof = 1;
					cur->State = CURSOR_INVALID;
					return RC_OK;
				}
				MoveToParent(cur);
				page = cur->Pages[cur->ID];
			} while (cur->Idxs[cur->ID] >= page->Cells);
			*eof = 0;
			if (page->IntKey)
				rc = Next_(cur, eof);
			else
				rc = RC_OK;
			return rc;
		}
		*eof = 0;
		if (page->Leaf)
			return RC_OK;
		rc = MoveToLeftmost(cur);
		return rc;
	}

	__device__ RC Btree::Previous(BtCursor *cur, int *eof)
	{
		_assert(CursorHoldsMutex(cur));
		RC rc = restoreCursorPosition(cur);
		if (rc != RC_OK)
			return rc;
		cur->AtLast = 0;
		if (cur->State == CURSOR_INVALID)
		{
			*eof = 1;
			return RC_OK;
		}
		if (cur->SkipNext < 0)
		{
			cur->SkipNext = 0;
			*eof = 0;
			return RC_OK;
		}
		cur->SkipNext = 0;

		MemPage *page = cur->Pages[cur->ID];
		_assert(page->IsInit);
		if (!page->Leaf)
		{
			uint idx = cur->Idxs[cur->ID];
			rc = MoveToChild(cur, ConvertEx::Get4(FindCell(page, idx)));
			if (rc)
				return rc;
			rc = MoveToRightmost(cur);
		}
		else
		{
			while (cur->Idxs[cur->ID] == 0)
			{
				if (cur->ID == 0)
				{
					cur->State = CURSOR_INVALID;
					*eof = 1;
					return RC_OK;
				}
				MoveToParent(cur);
			}
			cur->Info.Size = 0;
			cur->ValidNKey = 0;

			cur->Idxs[cur->ID]--;
			page = cur->Pages[cur->ID];
			if (page->IntKey && !page->Leaf)
				rc = Previous(cur, eof);
			else
				rc = RC_OK;
		}
		*eof = 0;
		return rc;
	}

#pragma endregion

#pragma region Allocate Page

	__device__ static RC AllocateBtreePage(BtShared *bt, MemPage **page, Pid *id, Pid nearby, BTALLOC mode)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(mode == BTALLOC::ANY || (nearby > 0 && IFAUTOVACUUM(bt->AutoVacuum)));
		MemPage *page1 = bt->Page1;
		Pid maxPage = BtreePagecount(bt); // Total size of the database file
		uint32 n = ConvertEx::Get4(&page1->Data[36]); // Number of pages on the freelist
		ASSERTCOVERAGE(n == maxPage - 1);
		if (n >= maxPage)
			return SysEx_CORRUPT_BKPT;
		RC rc;
		MemPage *trunk = nullptr;
		MemPage *prevTrunk = nullptr;
		if (n > 0)
		{
			// There are pages on the freelist.  Reuse one of those pages.
			bool searchList = false; // If the free-list must be searched for 'nearby'

			// If eMode==BTALLOC_EXACT and a query of the pointer-map shows that the page 'nearby' is somewhere on the free-list, then
			// the entire-list will be searched for that page.
#ifndef OMIT_AUTOVACUUM
			if (mode == BTALLOC::EXACT)
			{
				if (nearby <= maxPage)
				{
					_assert(nearby > 0);
					_assert(bt->AutoVacuum);
					PTRMAP type;
					rc = PtrmapGet(bt, nearby, &type, nullptr);
					if (rc) return rc;
					if (type == PTRMAP_FREEPAGE)
						searchList = true;
				}
			}
			else if (mode == BTALLOC::LE)
				searchList = true;
#endif

			// Decrement the free-list count by 1. Set iTrunk to the index of the first free-list trunk page. iPrevTrunk is initially 1.
			rc = Pager::Write(page1->DBPage);
			if (rc) return rc;
			ConvertEx::Put4(&page1->Data[36], n - 1);

			// The code within this loop is run only once if the 'searchList' variable is not true. Otherwise, it runs once for each trunk-page on the
			// free-list until the page 'nearby' is located (eMode==BTALLOC_EXACT) or until a page less than 'nearby' is located (eMode==BTALLOC_LT)
			Pid trunkID;
			do
			{
				prevTrunk = trunk;
				if (prevTrunk)
					trunkID = ConvertEx::Get4(&prevTrunk->Data[0]);
				else
					trunkID = ConvertEx::Get4(&page1->Data[32]);

				ASSERTCOVERAGE(trunkID == maxPage);
				if (trunkID > maxPage)
					rc = SysEx_CORRUPT_BKPT;
				else
					rc = BtreeGetPage(bt, trunkID, &trunk, false);
				if (rc)
				{
					trunk = nullptr;
					goto end_allocate_page;
				}
				_assert(trunk != nullptr);
				_assert(trunk->Data != nullptr);

				uint32 k = ConvertEx::Get4(&trunk->Data[4]); // # of leaves on this trunk page, Number of leaves on the trunk of the freelist
				if (k == 0 && !searchList)
				{
					// The trunk has no leaves and the list is not being searched. So extract the trunk page itself and use it as the newly allocated page
					_assert(prevTrunk == nullptr);
					rc = Pager::Write(trunk->DBPage);
					if (rc)
						goto end_allocate_page;
					*id = trunkID;
					_memcpy(&page1->Data[32], &trunk->Data[0], 4);
					*page = trunk;
					trunk = nullptr;
					TRACE("ALLOCATE: %d trunk - %d free pages left\n", *id, n - 1);
				}
				else if (k > (uint32)(bt->UsableSize / 4 - 2))
				{
					// Value of k is out of range.  Database corruption
					rc = SysEx_CORRUPT_BKPT;
					goto end_allocate_page;
#ifndef OMIT_AUTOVACUUM
				}
				else if (searchList && (nearby == trunkID || (trunkID < nearby && mode == BTALLOC::LE)))
				{
					// The list is being searched and this trunk page is the page to allocate, regardless of whether it has leaves.
					*id = trunkID;
					*page = trunk;
					searchList = false;
					rc = Pager::Write(trunk->DBPage);
					if (rc)
						goto end_allocate_page;
					if (k == 0)
					{
						if (!prevTrunk)
							_memcpy(&page1->Data[32], &trunk->Data[0], 4);
						else
						{
							rc = Pager::Write(prevTrunk->DBPage);
							if (rc != RC_OK)
								goto end_allocate_page;
							_memcpy(&prevTrunk->Data[0], &trunk->Data[0], 4);
						}
					}
					else
					{
						// The trunk page is required by the caller but it contains pointers to free-list leaves. The first leaf becomes a trunk
						// page in this case.
						Pid newTrunkID = ConvertEx::Get4(&trunk->Data[8]);
						if (newTrunkID > maxPage)
						{ 
							rc = SysEx_CORRUPT_BKPT;
							goto end_allocate_page;
						}
						ASSERTCOVERAGE(newTrunkID == maxPage);
						MemPage *newTrunk;
						rc = BtreeGetPage(bt, newTrunkID, &newTrunk, false);
						if (rc != RC_OK)
							goto end_allocate_page;
						rc = Pager::Write(newTrunk->DBPage);
						if (rc != RC_OK)
						{
							ReleasePage(newTrunk);
							goto end_allocate_page;
						}
						_memcpy(&newTrunk->Data[0], &trunk->Data[0], 4);
						ConvertEx::Put4(&newTrunk->Data[4], k - 1);
						_memcpy(&newTrunk->Data[8], &trunk->Data[12], (k - 1) * 4);
						ReleasePage(newTrunk);
						if (!prevTrunk)
						{
							_assert(Pager::Iswriteable(page1->DBPage));
							ConvertEx::Put4(&page1->Data[32], newTrunkID);
						}
						else
						{
							rc = Pager::Write(prevTrunk->DBPage);
							if (rc)
								goto end_allocate_page;
							ConvertEx::Put4(&prevTrunk->Data[0], newTrunkID);
						}
					}
					trunk = nullptr;
					TRACE("ALLOCATE: %d trunk - %d free pages left\n", *id, n - 1);
#endif
				}
				else if (k > 0)
				{
					// Extract a leaf from the trunk
					unsigned char *data = trunk->Data;
					Pid pageID;
					uint32 closest;
					if (nearby > 0)
					{
						closest = 0;
						if (mode == BTALLOC::LE)
						{
							for (uint32 i = 0U; i < k; i++)
							{
								pageID = ConvertEx::Get4(&data[8 + i * 4]);
								if (pageID <= nearby)
								{
									closest = i;
									break;
								}
							}
						}
						else
						{
							int dist = MathEx::Abs(ConvertEx::Get4(&data[8]) - nearby);
							for (uint32 i = 1U; i < k; i++)
							{
								int d2 = MathEx::Abs(ConvertEx::Get4(&data[8 + i * 4]) - nearby);
								if (d2 < dist)
								{
									closest = i;
									dist = d2;
								}
							}
						}
					}
					else
						closest = 0;

					pageID = ConvertEx::Get4(&data[8 + closest * 4]);
					ASSERTCOVERAGE(pageID == maxPage);
					if (pageID > maxPage)
					{
						rc = SysEx_CORRUPT_BKPT;
						goto end_allocate_page;
					}
					ASSERTCOVERAGE(pageID == maxPage);
					if (!searchList || (pageID == nearby || (pageID < nearby && mode == BTALLOC::LE)))
					{
						*id = pageID;
						TRACE("ALLOCATE: %d was leaf %d of %d on trunk %d: %d more free pages\n", *id, closest + 1, k, trunk->ID, n - 1);
						rc = Pager::Write(trunk->DBPage);
						if (rc) goto end_allocate_page;
						if (closest < k - 1)
							_memcpy(&data[8 + closest * 4], &data[4 + k * 4], 4);
						ConvertEx::Put4(&data[4], k - 1);
						bool noContent = !BtreeGetHasContent(bt, *id);
						rc = BtreeGetPage(bt, *id, page, noContent);
						if (rc == RC_OK)
						{
							rc = Pager::Write((*page)->DBPage);
							if (rc != RC_OK)
								ReleasePage(*page);
						}
						searchList = false;
					}
				}
				ReleasePage(prevTrunk);
				prevTrunk = nullptr;
			} while (searchList);
		}
		else
		{
			// Normally, new pages allocated by this block can be requested from the pager layer with the 'no-content' flag set. This prevents the pager
			// from trying to read the pages content from disk. However, if the current transaction has already run one or more incremental-vacuum
			// steps, then the page we are about to allocate may contain content that is required in the event of a rollback. In this case, do
			// not set the no-content flag. This causes the pager to load and journal the current page content before overwriting it.
			//
			// Note that the pager will not actually attempt to load or journal content for any page that really does lie past the end of the database
			// file on disk. So the effects of disabling the no-content optimization here are confined to those pages that lie between the end of the
			// database image and the end of the database file.
			bool noContent = !IFAUTOVACUUM(bt->DoTruncate);

			// There are no pages on the freelist, so append a new page to the database image.
			rc = Pager::Write(bt->Page1->DBPage);
			if (rc) return rc;
			bt->Pages++;
			if (bt->Pages == PENDING_BYTE_PAGE(bt)) bt->Pages++;

#ifndef OMIT_AUTOVACUUM
			if (bt->AutoVacuum && PTRMAP_ISPAGE(bt, bt->Pages))
			{
				// If *pPgno refers to a pointer-map page, allocate two new pages at the end of the file instead of one. The first allocated page
				// becomes a new pointer-map page, the second is used by the caller.
				TRACE("ALLOCATE: %d from end of file (pointer-map page)\n", bt->Pages);
				_assert(bt->Pages != PENDING_BYTE_PAGE(bt));
				MemPage *pg = nullptr;
				rc = BtreeGetPage(bt, bt->Pages, &pg, noContent);
				if (rc == RC_OK)
				{
					rc = Pager::Write(pg->DBPage);
					ReleasePage(pg);
				}
				if (rc) return rc;
				bt->Pages++;
				if (bt->Pages == PENDING_BYTE_PAGE(bt)) bt->Pages++;
			}
#endif
			ConvertEx::Put4(&bt->Page1->Data[28], bt->Pages);
			*id = bt->Pages;

			_assert(*id != PENDING_BYTE_PAGE(bt));
			rc = BtreeGetPage(bt, *id, page, noContent);
			if (rc) return rc;
			rc = Pager::Write((*page)->DBPage);
			if (rc != RC_OK)
				ReleasePage(*page);
			TRACE("ALLOCATE: %d from end of file\n", *id);
		}

		_assert(*id != PENDING_BYTE_PAGE(bt));

end_allocate_page:
		ReleasePage(trunk);
		ReleasePage(prevTrunk);
		if (rc == RC_OK)
		{
			if (Pager::get_PageRefs((*page)->DBPage) > 1)
			{
				ReleasePage(*page);
				return SysEx_CORRUPT_BKPT;
			}
			(*page)->IsInit = false;
		}
		else
			*page = nullptr;
		_assert(rc != RC_OK || Pager::Iswriteable((*page)->DBPage));
		return rc;
	}

	__device__ static RC FreePage2(BtShared *bt, MemPage *memPage, Pid pageID)
	{
		_assert(MutexEx::Held(bt->Mutex));
		_assert(pageID > 1);
		_assert(!memPage || memPage->ID == pageID);

		MemPage *page; // Page being freed. May be NULL.
		MemPage *trunk = nullptr; // Free-list trunk page
		if (memPage)
		{
			page = memPage;
			Pager::get_PageRefs(page->DBPage);
		}
		else
			page = BtreePageLookup(bt, pageID);

		// Increment the free page count on pPage1
		MemPage *page1 = bt->Page1; // Local reference to page 1
		RC rc = Pager::Write(page1->DBPage);
		int frees;
		Pid trunkID;
		if (rc) goto freepage_out;
		frees = ConvertEx::Get4(&page1->Data[36]); // Initial number of pages on free-list
		ConvertEx::Put4(&page1->Data[36], frees + 1);

		if (bt->BtsFlags & BTS_SECURE_DELETE)
		{
			// If the secure_delete option is enabled, then always fully overwrite deleted information with zeros.
			if ((!page && (rc = BtreeGetPage(bt, pageID, &page, false)) != RC_OK) || (rc = Pager::Write(page->DBPage)) != RC_OK)
				goto freepage_out;
			_memset(page->Data, 0, page->Bt->PageSize);
		}

		// If the database supports auto-vacuum, write an entry in the pointer-map to indicate that the page is free.
#if !OMIT_AUTOVACUUM
		if (bt->AutoVacuum)
		{
			PtrmapPut(bt, pageID, PTRMAP_FREEPAGE, 0, &rc);
			if (rc) goto freepage_out;
		}
#endif

		// Now manipulate the actual database free-list structure. There are two possibilities. If the free-list is currently empty, or if the first
		// trunk page in the free-list is full, then this page will become a new free-list trunk page. Otherwise, it will become a leaf of the
		// first trunk page in the current free-list. This block tests if it is possible to add the page as a new free-list leaf.
		trunkID = 0; // Page number of free-list trunk page
		if (frees != 0)
		{
			trunkID = ConvertEx::Get4(&page1->Data[32]);
			rc = BtreeGetPage(bt, trunkID, &trunk, false);
			if (rc != RC_OK)
				goto freepage_out;

			uint32 leafs = ConvertEx::Get4(&trunk->Data[4]); // Initial number of leaf cells on trunk page
			_assert(bt->UsableSize > 32);
			if (leafs > (uint32)bt->UsableSize / 4 - 2)
			{
				rc = SysEx_CORRUPT_BKPT;
				goto freepage_out;
			}
			if (leafs < (uint32)bt->UsableSize / 4 - 8)
			{
				// In this case there is room on the trunk page to insert the page being freed as a new leaf.
				//
				// Note that the trunk page is not really full until it contains usableSize/4 - 2 entries, not usableSize/4 - 8 entries as we have
				// coded.  But due to a coding error in versions of SQLite prior to 3.6.0, databases with freelist trunk pages holding more than
				// usableSize/4 - 8 entries will be reported as corrupt.  In order to maintain backwards compatibility with older versions of SQLite,
				// we will continue to restrict the number of entries to usableSize/4 - 8 for now.  At some point in the future (once everyone has upgraded
				// to 3.6.0 or later) we should consider fixing the conditional above to read "usableSize/4-2" instead of "usableSize/4-8".
				rc = Pager::Write(trunk->DBPage);
				if (rc == RC_OK)
				{
					ConvertEx::Put4(&trunk->Data[4], leafs + 1);
					ConvertEx::Put4(&trunk->Data[8 + leafs * 4], pageID);
					if (page && (bt->BtsFlags & BTS_SECURE_DELETE) == 0)
						Pager::DontWrite(page->DBPage);
					rc = BtreeSetHasContent(bt, pageID);
				}
				TRACE("FREE-PAGE: %d leaf on trunk page %d\n", page->ID, trunk->ID);
				goto freepage_out;
			}
		}

		// If control flows to this point, then it was not possible to add the the page being freed as a leaf page of the first trunk in the free-list.
		// Possibly because the free-list is empty, or possibly because the first trunk in the free-list is full. Either way, the page being freed
		// will become the new first trunk page in the free-list.
		if (page == nullptr && (rc = BtreeGetPage(bt, pageID, &page, false)) != RC_OK)
			goto freepage_out;
		rc = Pager::Write(page->DBPage);
		if (rc != RC_OK)
			goto freepage_out;
		ConvertEx::Put4(page->Data, trunkID);
		ConvertEx::Put4(&page->Data[4], 0);
		ConvertEx::Put4(&page1->Data[32], pageID);
		TRACE("FREE-PAGE: %d new trunk page replacing %d\n", page->ID, trunkID);

freepage_out:
		if (page)
			page->IsInit = false;
		ReleasePage(page);
		ReleasePage(trunk);
		return rc;
	}

	__device__ static void FreePage(MemPage *page, RC *rc)
	{
		if ((*rc) == RC_OK)
			*rc = FreePage2(page->Bt, page, page->ID);
	}

	__device__ static RC ClearCell(MemPage *page, unsigned char *cell)
	{
		_assert(MutexEx::Held(page->Bt->Mutex));
		CellInfo info;
		BtreeParseCellPtr(page, cell, &info);
		if (info.Overflow == 0)
			return RC_OK; // No overflow pages. Return without doing anything 
		if (cell + info.Overflow + 3 > page->Data + page->MaskPage)
			return SysEx_CORRUPT_BKPT; // Cell extends past end of page
		Pid ovflID = ConvertEx::Get4(&cell[info.Overflow]);
		BtShared *bt = page->Bt;
		_assert(bt->UsableSize > 4);
		uint32 ovflPageSize = bt->UsableSize - 4;
		int ovfls = (info.Payload - info.Local + ovflPageSize - 1) / ovflPageSize;
		_assert(ovflID == 0 || ovfls > 0);
		RC rc;
		while (ovfls--)
		{
			if (ovflID < 2 || ovflID > BtreePagecount(bt))
			{
				// 0 is not a legal page number and page 1 cannot be an overflow page. Therefore if ovflPgno<2 or past the end of the 
				// file the database must be corrupt.
				return SysEx_CORRUPT_BKPT;
			}
			Pid nextID = 0;
			MemPage *ovfl = nullptr;
			if (ovfls)
			{
				rc = GetOverflowPage(bt, ovflID, &ovfl, &nextID);
				if (rc) return rc;
			}

			if ((ovfl || (ovfl = BtreePageLookup(bt, ovflID)) != nullptr) && Pager::get_PageRefs(ovfl->DBPage) != 1)
			{
				// There is no reason any cursor should have an outstanding reference to an overflow page belonging to a cell that is being deleted/updated.
				// So if there exists more than one reference to this page, then it must not really be an overflow page and the database must be corrupt. 
				// It is helpful to detect this before calling FreePage2(), as FreePage2() may zero the page contents if secure-delete mode is
				// enabled. If this 'overflow' page happens to be a page that the caller is iterating through or using in some other way, this can be problematic.
				rc = SysEx_CORRUPT_BKPT;
			}
			else
				rc = FreePage2(bt, ovfl, ovflID);

			if (ovfl)
				Pager::Unref(ovfl->DBPage);
			if (rc) return rc;
			ovflID = nextID;
		}
		return RC_OK;
	}

	__device__ static RC FillInCell(MemPage *page, unsigned char *cell, const void *key, int64 keyLength, const void *data, int dataLength, int zeros, uint16 *sizeOut)
	{
		BtShared *bt = page->Bt;
		_assert(MutexEx::Held(bt->Mutex));

		// pPage is not necessarily writeable since pCell might be auxiliary buffer space that is separate from the pPage buffer area
		_assert(cell < page->Data || cell >= &page->Data[bt->PageSize] || Pager::Iswriteable(page->DBPage));

		// Fill in the header.
		uint header = 0U;
		if (!page->Leaf)
			header += 4;
		if (page->HasData)
			header += ConvertEx::PutVarint(&cell[header], dataLength + zeros);
		else
			dataLength = zeros = 0;
		header += ConvertEx::PutVarint(&cell[header], keyLength);
		CellInfo info;
		BtreeParseCellPtr(page, cell, &info);
		_assert(info.Header == header);
		_assert(info.Key == keyLength);
		_assert(info.Data == (uint32)(dataLength + zeros));

		// Fill in the payload
		const uint8 *src;
		int srcLength;
		int payloadLength = dataLength + zeros;
		if (page->IntKey)
		{
			src = (uint8 *)data;
			srcLength = dataLength;
			dataLength = 0;
		}
		else
		{ 
			if (_NEVER(keyLength > 0x7fffffff || key == nullptr))
				return SysEx_CORRUPT_BKPT;
			payloadLength += (int)keyLength;
			src = (uint8 *)key;
			srcLength = (int)keyLength;
		}
		*sizeOut = info.Size;
		int spaceLeft = info.Local;
		unsigned char *payload = &cell[header];
		unsigned char *prior = &cell[info.Overflow];

		RC rc;
		MemPage *toRelease = nullptr;
		while (payloadLength > 0)
		{
			if (spaceLeft == 0)
			{
				Pid idOvfl = 0;
#ifndef OMIT_AUTOVACUUM
				Pid idPtrmap = idOvfl; // Overflow page pointer-map entry page
				if (bt->AutoVacuum)
				{
					do
					{
						idOvfl++;
					} while (PTRMAP_ISPAGE(bt, idOvfl) || idOvfl == PENDING_BYTE_PAGE(bt));
				}
#endif
				MemPage *ovfl = nullptr;
				rc = AllocateBtreePage(bt, &ovfl, &idOvfl, idOvfl, BTALLOC::ANY);
#ifndef OMIT_AUTOVACUUM
				// If the database supports auto-vacuum, and the second or subsequent overflow page is being allocated, add an entry to the pointer-map
				// for that page now. 
				//
				// If this is the first overflow page, then write a partial entry to the pointer-map. If we write nothing to this pointer-map slot,
				// then the optimistic overflow chain processing in ClearCell() may misinterpret the uninitialized values and delete the
				// wrong pages from the database.
				if (bt->AutoVacuum && rc == RC_OK)
				{
					PTRMAP type = (idPtrmap ? PTRMAP_OVERFLOW2 : PTRMAP_OVERFLOW1);
					PtrmapPut(bt, idOvfl, type, idPtrmap, &rc);
					if (rc)
						ReleasePage(ovfl);
				}
#endif
				if (rc)
				{
					ReleasePage(toRelease);
					return rc;
				}

				// If pToRelease is not zero than pPrior points into the data area of pToRelease.  Make sure pToRelease is still writeable.
				_assert(toRelease == nullptr || Pager::Iswriteable(toRelease->DBPage));

				// If pPrior is part of the data area of pPage, then make sure pPage is still writeable
				_assert(prior < page->Data || prior >= &page->Data[bt->PageSize] || Pager::Iswriteable(page->DBPage));

				ConvertEx::Put4(prior, idOvfl);
				ReleasePage(toRelease);
				toRelease = ovfl;
				prior = ovfl->Data;
				ConvertEx::Put4(prior, 0);
				payload = &ovfl->Data[4];
				spaceLeft = bt->UsableSize - 4;
			}
			int n = payloadLength;
			if (n > spaceLeft) n = spaceLeft;

			// If pToRelease is not zero than pPayload points into the data area of pToRelease.  Make sure pToRelease is still writeable.
			_assert(toRelease == nullptr || Pager::Iswriteable(toRelease->DBPage));

			// If pPayload is part of the data area of pPage, then make sure pPage is still writeable
			_assert(payload < page->Data || payload >= &page->Data[bt->PageSize] || Pager::Iswriteable(page->DBPage));

			if (srcLength > 0)
			{
				if (n > srcLength) n = srcLength;
				_assert(src != nullptr);
				_memcpy(payload, src, n);
			}
			else
				_memset(payload, 0, n);
			payloadLength -= n;
			payload += n;
			src += n;
			srcLength -= n;
			spaceLeft -= n;
			if (srcLength == 0)
			{
				srcLength = dataLength;
				src = (uint8 *)data;
			}
		}
		ReleasePage(toRelease);
		return RC_OK;
	}

	__device__ static void DropCell(MemPage *page, uint idx, uint16 size, RC *rcRef)
	{
		if (*rcRef) return;

		_assert(idx < page->Cells);
		_assert(size == CellSize(page, idx));
		_assert(Pager::Iswriteable(page->DBPage));
		_assert(MutexEx::Held(page->Bt->Mutex));
		uint8 *data = page->Data;
		uint8 *ptr = &page->CellIdx[2 * idx]; // Used to move bytes around within data[]
		uint32 pc = ConvertEx::Get2(ptr); // Offset to cell content of cell being deleted
		int hdr = page->HdrOffset; // Beginning of the header.  0 most pages.  100 page 1
		ASSERTCOVERAGE(pc == ConvertEx::Get2(&data[hdr + 5]));
		ASSERTCOVERAGE(pc + size == page->Bt->UsableSize);
		if (pc < (uint32)ConvertEx::Get2(&data[hdr + 5]) || pc + size > page->Bt->UsableSize)
		{
			*rcRef = SysEx_CORRUPT_BKPT;
			return;
		}
		RC rc = FreeSpace(page, pc, size);
		if (rc)
		{
			*rcRef = rc;
			return;
		}
		uint8 *endPtr = &page->CellIdx[2 * page->Cells - 2]; // End of loop
		_assert((PTR_TO_INT(ptr) & 1) == 0); // ptr is always 2-byte aligned
		while (ptr < endPtr)
		{
			*(uint16 *)ptr = *(uint16 *)&ptr[2];
			ptr += 2;
		}
		page->Cells--;
		ConvertEx::Put2(&data[hdr + 3], page->Cells);
		page->Frees += 2;
	}

	__device__ static void InsertCell(MemPage *page, uint32 i, uint8 *cell, uint16 size, uint8 *temp, Pid childID, RC *rcRef)
	{
		if (*rcRef) return;

		_assert(i <= page->Cells + page->Overflows);
		_assert(page->Cells <= MX_CELL(page->Bt) && MX_CELL(page->Bt) <= 10921);
		_assert(page->Overflows <= _lengthof(page->Ovfls));
		_assert(_lengthof(page->Ovfls) == _lengthof(page->OvflIdxs));
		_assert(MutexEx::Held(page->Bt->Mutex));
		// The cell should normally be sized correctly.  However, when moving a malformed cell from a leaf page to an interior page, if the cell size
		// wanted to be less than 4 but got rounded up to 4 on the leaf, then size might be less than 8 (leaf-size + pointer) on the interior node.  Hence
		// the term after the || in the following assert().
		_assert(size == CellSizePtr(page, cell) || (size == 8 && childID > 0));
		int skip = (childID ? 4 : 0);
		if (page->Overflows || size + 2 > page->Frees)
		{
			if (temp)
			{
				_memcpy(temp + skip, cell + skip, size - skip);
				cell = temp;
			}
			if (childID)
				ConvertEx::Put4(cell, childID);
			int j = page->Overflows++;
			_assert(j < (int)(sizeof(page->Ovfls) / sizeof(page->Ovfls[0])));
			page->Ovfls[j] = cell;
			page->OvflIdxs[j] = (uint16)i;
		}
		else
		{
			RC rc = Pager::Write(page->DBPage);
			if (rc != RC_OK)
			{
				*rcRef = rc;
				return;
			}
			_assert(Pager::Iswriteable(page->DBPage));
			uint8 *data = page->Data;  // The content of the whole page
			uint cellOffset = page->CellOffset; // Address of first cell pointer in data[]
			uint end = cellOffset + 2 * page->Cells; // First byte past the last cell pointer in data[]
			uint ins = cellOffset + 2 * i; // Index in data[] where new cell pointer is inserted
			uint idx = 0; // Where to write new cell content in data[]
			rc = AllocateSpace(page, size, &idx);
			if (rc) { *rcRef = rc; return; }
			// The AllocateSpace() routine guarantees the following two properties if it returns success
			_assert(idx >= end + 2);
			_assert(idx + size <= (int)page->Bt->UsableSize);
			page->Cells++;
			page->Frees -= (uint16)(2 + size);
			_memcpy(&data[idx + skip], cell + skip, size - skip);
			if (childID)
				ConvertEx::Put4(&data[idx], childID);
			uint8 *ptr = &data[end]; // Used for moving information around in data[]
			uint8 *endPtr = &data[ins]; // End of the loop
			_assert((PTR_TO_INT(ptr) & 1) == 0); // ptr is always 2-byte aligned
			while (ptr > endPtr)
			{
				*(uint16 *)ptr = *(uint16 *)&ptr[-2];
				ptr -= 2;
			}
			ConvertEx::Put2(&data[ins], idx);
			ConvertEx::Put2(&data[page->HdrOffset + 3], page->Cells);
#ifndef OMIT_AUTOVACUUM
			if (page->Bt->AutoVacuum)
			{
				// The cell may contain a pointer to an overflow page. If so, write the entry for the overflow page into the pointer map.
				PtrmapPutOvflPtr(page, cell, rcRef);
			}
#endif
		}
	}

	__device__ static void AssemblePage(MemPage *page, int cells, uint8 **cellSet, uint16 *sizes)
	{
		_assert(page->Overflows == 0);
		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(cells >= 0 && cells <= (int)MX_CELL(page->Bt) && (int)MX_CELL(page->Bt) <= 10921);
		_assert(Pager::Iswriteable(page->DBPage));

		// Check that the page has just been zeroed by ZeroPage()
		uint8 *const data = page->Data; // Pointer to data for pPage
		const int hdr = page->HdrOffset; // Offset of header on pPage
		const int usable = page->Bt->UsableSize; // Usable size of page	_assert(page->Cells == 0);
		_assert(ConvertEx::Get2nz(&data[hdr + 5]) == usable);

		uint8 *cellptr = &page->CellIdx[cells * 2]; // Address of next cell pointer
		int cellbody = usable; // Address of next cell body
		for (int i = cells - 1; i >= 0; i--)
		{
			uint16 sz = sizes[i];
			cellptr -= 2;
			cellbody -= sz;
			ConvertEx::Put2(cellptr, cellbody);
			_memcpy(&data[cellbody], cellSet[i], sz);
		}
		ConvertEx::Put2(&data[hdr + 3], cells);
		ConvertEx::Put2(&data[hdr + 5], cellbody);
		page->Frees -= (cells * 2 + usable - cellbody);
		page->Cells = (uint16)cells;
	}

#pragma endregion

#pragma region Balance

#define NN 1				// Number of neighbors on either side of pPage
#define NB (NN * 2 + 1)		// Total pages involved in the Balance

#ifndef OMIT_QUICKBALANCE
	__device__ static RC Balance_Quick(MemPage *parent, MemPage *page, uint8 *space)
	{
		BtShared *const bt = page->Bt; // B-Tree Database

		_assert(MutexEx::Held(page->Bt->Mutex));
		_assert(Pager::Iswriteable(parent->DBPage));
		_assert(page->Overflows == 1);

		// This error condition is now caught prior to reaching this function
		if (page->Cells == 0) return SysEx_CORRUPT_BKPT;

		// Allocate a new page. This page will become the right-sibling of pPage. Make the parent page writable, so that the new divider cell
		// may be inserted. If both these operations are successful, proceed.
		MemPage *newPage; // Newly allocated page
		Pid newPageID; // Page number of pNew
		RC rc = AllocateBtreePage(bt, &newPage, &newPageID, 0, BTALLOC::ANY);

		if (rc == RC_OK)
		{
			uint8 *out = &space[4];
			uint8 *cell = page->Ovfls[0];
			uint16 sizeCell = CellSizePtr(page, cell);

			_assert(Pager::Iswriteable(newPage->DBPage));
			_assert(page->Data[0] == (PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF));
			ZeroPage(newPage, PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF);
			AssemblePage(newPage, 1, &cell, &sizeCell);

			// If this is an auto-vacuum database, update the pointer map with entries for the new page, and any pointer from the 
			// cell on the page to an overflow page. If either of these operations fails, the return code is set, but the contents
			// of the parent page are still manipulated by thh code below. That is Ok, at this point the parent page is guaranteed to
			// be marked as dirty. Returning an error code will cause a rollback, undoing any changes made to the parent page.
#if !OMIT_AUTOVACUUM
			if (bt->AutoVacuum)
			{
				PtrmapPut(bt, newPageID, PTRMAP_BTREE, parent->ID, &rc);
				if (sizeCell > newPage->MinLocal)
					PtrmapPutOvflPtr(newPage, cell, &rc);
			}
#endif

			// Create a divider cell to insert into pParent. The divider cell consists of a 4-byte page number (the page number of pPage) and
			// a variable length key value (which must be the same value as the largest key on pPage).
			//
			// To find the largest key value on pPage, first find the right-most cell on pPage. The first two fields of this cell are the 
			// record-length (a variable length integer at most 32-bits in size) and the key value (a variable length integer, may have any value).
			// The first of the while(...) loops below skips over the record-length field. The second while(...) loop copies the key value from the
			// cell on pPage into the pSpace buffer.
			cell = FindCell(page, page->Cells - 1);
			uint8 *stop = &cell[9];
			while ((*(cell++) & 0x80) && cell < stop);
			stop = &cell[9];
			while (((*(out++) = *(cell++)) & 0x80) && cell < stop);

			// Insert the new divider cell into pParent.
			InsertCell(parent, parent->Cells, space, (int)(out - space), 0, page->ID, &rc);

			// Set the right-child pointer of pParent to point to the new page. */
			ConvertEx::Put4(&parent->Data[parent->HdrOffset + 8], newPageID);

			// Release the reference to the new page.
			ReleasePage(newPage);
		}

		return rc;
	}
#endif

#if 0
	__device__ static int PtrmapCheckPages(MemPage **pageSet, int pages)
	{
		for (int i = 0; i < pages; i++)
		{
			MemPage *page = pageSet[i];
			BtShared *bt = page->Bt;
			_assert(page->IsInit);

			Pid n;
			PTRMAP e;
			for (uint j = 0U; j < page->Cells; j++)
			{
				uint8 *z = FindCell(page, j);
				CellInfo info;
				BtreeParseCellPtr(page, z, &info);
				if (info.Overflow)
				{
					Pid ovfl = ConvertEx::Get4(&z[info.Overflow]);
					PtrmapGet(bt, ovfl, &e, &n);
					_assert(n == page->ID && e == PTRMAP_OVERFLOW1);
				}
				if (!page->Leaf)
				{
					Pid child = ConvertEx::Get4(z);
					PtrmapGet(bt, child, &e, &n);
					_assert(n == page->ID && e == PTRMAP_BTREE);
				}
			}
			if (!page->Leaf)
			{
				Pid child = ConvertEx::Get4(&page->Data[page->HdrOffset + 8]);
				PtrmapGet(bt, child, &e, &n);
				_assert(n == page->ID && e == PTRMAP_BTREE);
			}
		}
		return 1;
	}
#endif

	__device__ static void CopyNodeContent(MemPage *from, MemPage *to, RC *rcRef)
	{
		if (*rcRef == RC_OK)
		{
			BtShared *const bt = from->Bt;
			uint8 *const fromData = from->Data;
			uint8 *const toData = to->Data;
			int const fromHdr = from->HdrOffset;
			int const toHdr = (to->ID == 1 ? 100 : 0);

			_assert(from->IsInit);
			_assert(from->Frees >= toHdr);
			_assert(ConvertEx::Get2(&toData[fromHdr + 5]) <= (int)bt->UsableSize);

			// Copy the b-tree node content from page pFrom to page pTo.
			int data = ConvertEx::Get2(&fromData[fromHdr + 5]);
			_memcpy(&toData[data], &fromData[data], bt->UsableSize - data);
			_memcpy(&toData[toHdr], &fromData[fromHdr], from->CellOffset + 2 * from->Cells);

			// Reinitialize page pTo so that the contents of the MemPage structure match the new data. The initialization of pTo can actually fail under
			// fairly obscure circumstances, even though it is a copy of initialized page pFrom.
			to->IsInit = false;
			RC rc = BtreeInitPage(to);
			if (rc != RC_OK)
			{
				*rcRef = rc;
				return;
			}

			// If this is an auto-vacuum database, update the pointer-map entries for any b-tree or overflow pages that pTo now contains the pointers to.
#if !OMIT_AUTOVACUUM
			if (bt->AutoVacuum)
				*rcRef = SetChildPtrmaps(to);
#endif
		}
	}

#if defined(_MSC_VER) && _MSC_VER >= 1700 && defined(_M_ARM)
#pragma optimize("", off)
#endif
	__device__ static RC Balance_Nonroot(MemPage *parent, int parentIdx, uint8 *ovflSpace, bool isRoot, bool bulk)
	{
		BtShared *bt = parent->Bt; // The whole database
		_assert(MutexEx::Held(bt->Mutex));
		_assert(Pager::Iswriteable(parent->DBPage));

#if 0
		TRACE("BALANCE: begin page %d child of %d\n", page->ID, parent->ID);
#endif

		// At this point pParent may have at most one overflow cell. And if this overflow cell is present, it must be the cell with 
		// index iParentIdx. This scenario comes about when this function is called (indirectly) from sqlite3BtreeDelete().
		_assert(parent->Overflows == 0 || parent->Overflows == 1);
		_assert(parent->Overflows == 0 || parent->OvflIdxs[0] == parentIdx);

		if (!ovflSpace)
			return RC_NOMEM;

		// Find the sibling pages to Balance. Also locate the cells in pParent that divide the siblings. An attempt is made to find NN siblings on 
		// either side of pPage. More siblings are taken from one side, however, if there are fewer than NN siblings on the other side. If pParent
		// has NB or fewer children then all children of pParent are taken.  
		//
		// This loop also drops the divider cells from the parent page. This way, the remainder of the function does not have to deal with any
		// overflow cells in the parent page, since if any existed they will have already been removed.
		uint i = parent->Overflows + parent->Cells;
		uint nxDiv; // Next divider slot in pParent->aCell[]
		if (i < 2)
			nxDiv = 0;
		else
		{
			if (parentIdx == 0)
				nxDiv = 0;
			else if (parentIdx == i)
				nxDiv = i - 2 + (bulk ? 1U : 0U);
			else
			{
				_assert(!bulk);
				nxDiv = parentIdx - 1;
			}
			i = 2 - (bulk ? 1U : 0U);
		}
		uint8 *right; // Location in parent of right-sibling pointer
		if ((i + nxDiv - parent->Overflows) == parent->Cells)
			right = &parent->Data[parent->HdrOffset + 8U];
		else
			right = FindCell(parent, i + nxDiv - parent->Overflows);
		Pid id = ConvertEx::Get4(right); // Temp var to store a page number in
		RC rc;
		MemPage *oldPages[NB]; // pPage and up to two siblings
		MemPage *copyPages[NB]; // Private copies of apOld[] pages
		MemPage *newPages[NB + 2]; // pPage and up to NB siblings after balancing
		uint oldPagesUsed = i + 1; // Number of pages in apOld[]
		uint newPagesUsed = 0; // Number of pages in apNew[]
		uint maxCells = 0; // Allocated size of apCell, szCell, aFrom.
		uint8 *divs[NB - 1]; // Divider cells in pParent
		Pid countNew[NB + 2]; // Index in aCell[] of cell after i-th page
		uint16 sizeNew[NB + 2]; // Combined size of cells place on i-th page
		uint8 **cell = nullptr;
		uint j, k;
		Pid cells;
		uint16 *sizeCell;
		int space1Idx;
		uint8 *space1;
		uint16 leafCorrection;
		uint leafData;
		uint subtotal; // Subtotal of bytes in cells on one page
		uint usableSpace;
		int pageFlags;
		int ovflSpaceID;
		int sizeScratch;
		while (true)
		{
			rc = GetAndInitPage(bt, id, &oldPages[i]);
			if (rc)
			{
				_memset(oldPages, 0, (i + 1) * sizeof(MemPage *));
				goto balance_cleanup;
			}
			maxCells += 1U + oldPages[i]->Cells + oldPages[i]->Overflows;
			if (i-- == 0) break;

			if (i + nxDiv == parent->OvflIdxs[0] && parent->Overflows)
			{
				divs[i] = parent->Ovfls[0];
				id = ConvertEx::Get4(divs[i]);
				sizeNew[i] = CellSizePtr(parent, divs[i]);
				parent->Overflows = 0;
			}
			else
			{
				divs[i] = FindCell(parent, i + nxDiv - parent->Overflows);
				id = ConvertEx::Get4(divs[i]);
				sizeNew[i] = CellSizePtr(parent, divs[i]);

				// Drop the cell from the parent page. apDiv[i] still points to the cell within the parent, even though it has been dropped.
				// This is safe because dropping a cell only overwrites the first four bytes of it, and this function does not need the first
				// four bytes of the divider cell. So the pointer is safe to use later on.
				//
				// But not if we are in secure-delete mode. In secure-delete mode, the DropCell() routine will overwrite the entire cell with zeroes.
				// In this case, temporarily copy the cell into the aOvflSpace[] buffer. It will be copied out again as soon as the aSpace[] buffer is allocated.
				if (bt->BtsFlags & BTS_SECURE_DELETE)
				{
					int off = PTR_TO_INT(divs[i]) - PTR_TO_INT(parent->Data);
					if ((off + PTR_TO_INT(newPages[i])) > (int)bt->UsableSize)
					{
						rc = SysEx_CORRUPT_BKPT;
						_memset(oldPages, 0, (i + 1) * sizeof(MemPage *));
						goto balance_cleanup;
					}
					else
					{
						_memcpy(&ovflSpace[off], divs[i], sizeNew[i]);
						divs[i] = &ovflSpace[divs[i] - parent->Data];
					}
				}
				DropCell(parent, i + nxDiv - parent->Overflows, sizeNew[i], &rc);
			}
		}

		// Make nMaxCells a multiple of 4 in order to preserve 8-byte alignment
		maxCells = (maxCells + 3) & ~3;

		// Allocate space for memory structures
		k = bt->PageSize + _ROUND8(sizeof(MemPage));
		sizeScratch = // Size of scratch memory requested
			maxCells * sizeof(uint8 *) // apCell
			+ maxCells * sizeof(uint16) // szCell
			+ bt->PageSize // aSpace1
			+ k * oldPagesUsed; // Page copies (apCopy)
		cells = 0; // Number of cells in apCell[]
		cell = (uint8 **)_stackalloc(nullptr, sizeScratch, false); // All cells begin balanced
		if (cell == nullptr)
		{
			rc = RC_NOMEM;
			goto balance_cleanup;
		}

		sizeCell = (uint16 *)&cell[maxCells]; // Local size of all cells in apCell[]
		space1Idx = 0; // First unused byte of aSpace1[]
		space1 = (uint8 *)&sizeCell[maxCells]; // Space for copies of dividers cells
		_assert(_HASALIGNMENT8(space1));

		// Load pointers to all cells on sibling pages and the divider cells into the local apCell[] array.  Make copies of the divider cells
		// into space obtained from aSpace1[] and remove the divider cells from pParent.
		//
		// If the siblings are on leaf pages, then the child pointers of the divider cells are stripped from the cells before they are copied
		// into aSpace1[].  In this way, all cells in apCell[] are without child pointers.  If siblings are not leaves, then all cell in
		// apCell[] include child pointers.  Either way, all cells in apCell[] are alike.
		//
		// leafCorrection:  4 if pPage is a leaf.  0 if pPage is not a leaf.
		// leafData:  1 if pPage holds key+data and pParent holds only keys.
		leafCorrection = oldPages[0]->Leaf * 4; // 4 if pPage is a leaf.  0 if not
		leafData = oldPages[0]->HasData; // True if pPage is a leaf of a LEAFDATA tree
		for (i = 0U; i < oldPagesUsed; i++)
		{
			// Before doing anything else, take a copy of the i'th original sibling The rest of this function will use data from the copies rather
			// that the original pages since the original pages will be in the process of being overwritten.
			MemPage *oldPage = copyPages[i] = (MemPage *)&space1[bt->PageSize + k * i];
			_memcpy(oldPage, oldPages[i], sizeof(MemPage));
			oldPage->Data = (uint8 *)&oldPages[1];
			_memcpy(oldPage->Data, oldPages[i]->Data, bt->PageSize);

			int limit = oldPage->Cells + oldPage->Overflows;
			if (oldPage->Overflows > 0)
			{
				for (int j = 0; j < limit; j++)
				{
					_assert(cells < maxCells);
					cell[cells] = FindOverflowCell(oldPage, j);
					sizeCell[cells] = CellSizePtr(oldPage, cell[cells]);
					cells++;
				}
			}
			else
			{
				uint8 *data = oldPage->Data;
				uint16 maskPage = oldPage->MaskPage;
				uint16 cellOffset = oldPage->CellOffset;
				for (int j = 0; j < limit; j++)
				{
					_assert(cells < maxCells);
					cell[cells] = FindCellv2(data, maskPage, cellOffset, j);
					sizeCell[cells] = CellSizePtr(oldPage, cell[cells]);
					cells++;
				}
			}       
			if (i < oldPagesUsed - 1 && !leafData)
			{
				uint size = (uint)sizeNew[i];
				_assert(cells < maxCells);
				sizeCell[cells] = size;
				uint8 *temp = &space1[space1Idx];
				space1Idx += size;
				_assert(size <= bt->MaxLocal + 23);
				_assert(space1Idx <= (int)bt->PageSize);
				_memcpy(temp, divs[i], size);
				cell[cells] = temp + leafCorrection;
				_assert(leafCorrection == 0 || leafCorrection == 4);
				sizeCell[cells] = sizeCell[cells] - leafCorrection;
				if (!oldPage->Leaf)
				{
					_assert(leafCorrection == 0);
					_assert(oldPage->HdrOffset == 0);
					// The right pointer of the child page pOld becomes the left pointer of the divider cell
					_memcpy(cell[cells], &oldPage->Data[8], 4);
				}
				else
				{
					_assert(leafCorrection == 4);
					if (sizeCell[cells] < 4) // Do not allow any cells smaller than 4 bytes.
						sizeCell[cells] = 4;
				}
				cells++;
			}
		}

		// Figure out the number of pages needed to hold all nCell cells. Store this number in "k".  Also compute szNew[] which is the total
		// size of all cells on the i-th page and cntNew[] which is the index in apCell[] of the cell that divides page i from page i+1.  
		// cntNew[k] should equal nCell.
		//
		// Values computed by this block:
		//           k: The total number of sibling pages
		//    szNew[i]: Spaced used on the i-th sibling page.
		//   cntNew[i]: Index in apCell[] and szCell[] for the first cell to the right of the i-th sibling page.
		// usableSpace: Number of bytes of space available on each sibling.
		usableSpace = bt->UsableSize - 12 + leafCorrection; // Bytes in pPage beyond the header
		for (subtotal = 0, k = i = 0; i < cells; i++)
		{
			_assert(i < maxCells);
			subtotal += sizeCell[i] + 2;
			if (subtotal > usableSpace)
			{
				sizeNew[k] = subtotal - sizeCell[i];
				countNew[k] = i;
				if (leafData) i--;
				subtotal = 0;
				k++;
				if (k > NB + 1) { rc = SysEx_CORRUPT_BKPT; goto balance_cleanup; }
			}
		}
		sizeNew[k] = subtotal;
		countNew[k] = cells;
		k++;

		// The packing computed by the previous block is biased toward the siblings on the left side.  The left siblings are always nearly full, while the
		// right-most sibling might be nearly empty.  This block of code attempts to adjust the packing of siblings to get a better Balance.
		//
		// This adjustment is more than an optimization.  The packing above might be so out of Balance as to be illegal.  For example, the right-most
		// sibling might be completely empty.  This adjustment is not optional.
		for (i = k - 1; i > 0; i--)
		{
			uint16 sizeRight = sizeNew[i]; // Size of sibling on the right
			uint16 sizeLeft = sizeNew[i - 1]; // Size of sibling on the left
			Pid r = countNew[i-1] - 1; // Index of right-most cell in left sibling
			uint d = r + 1 - leafData; // Index of first cell to the left of right sibling
			_assert(d < maxCells);
			_assert(r < maxCells);
			while (sizeRight == 0 || (!bulk && sizeRight + sizeCell[d] + 2 <= sizeLeft - (sizeCell[r] + 2)))
			{
				sizeRight += sizeCell[d] + 2;
				sizeLeft -= sizeCell[r] + 2;
				countNew[i - 1]--;
				r = countNew[i - 1] - 1;
				d = r + 1 - leafData;
			}
			sizeNew[i] = sizeRight;
			sizeNew[i - 1] = sizeLeft;
		}

		// Either we found one or more cells (cntnew[0])>0) or pPage is a virtual root page.  A virtual root page is when the real root
		// page is page 1 and we are the only child of that page.
		//
		// UPDATE:  The assert() below is not necessarily true if the database file is corrupt.  The corruption will be detected and reported later
		// in this procedure so there is no need to act upon it now.
#if 0
		_assert(countNew[0] > 0 || (parent->ID == 1 && parent->Cells == 0));
#endif

		TRACE("BALANCE: old: %d %d %d  ",
			oldPages[0]->ID, 
			oldPagesUsed >= 2 ? oldPages[1]->ID : 0,
			oldPagesUsed >= 3 ? oldPages[2]->ID : 0);

		// Allocate k new pages.  Reuse old pages where possible.
		if (oldPages[0]->ID <= 1)
		{
			rc = SysEx_CORRUPT_BKPT;
			goto balance_cleanup;
		}
		pageFlags = oldPages[0]->Data[0]; // Value of pPage->aData[0]
		for (i = 0; i < k; i++)
		{
			MemPage *newPage;
			if (i < oldPagesUsed)
			{
				newPage = newPages[i] = oldPages[i];
				oldPages[i] = nullptr;
				rc = Pager::Write(newPage->DBPage);
				newPagesUsed++;
				if (rc) goto balance_cleanup;
			}
			else
			{
				_assert(i > 0);
				rc = AllocateBtreePage(bt, &newPage, &id, (bulk ? 1 : id), BTALLOC::ANY);
				if (rc) goto balance_cleanup;
				newPages[i] = newPage;
				newPagesUsed++;

				// Set the pointer-map entry for the new sibling page.
#if !OMIT_AUTOVACUUM
				if (bt->AutoVacuum)
				{
					PtrmapPut(bt, newPage->ID, PTRMAP_BTREE, parent->ID, &rc);
					if (rc != RC_OK)
						goto balance_cleanup;
				}
#endif
			}
		}

		// Free any old pages that were not reused as new pages.
		while (i < oldPagesUsed)
		{
			FreePage(oldPages[i], &rc);
			if (rc) goto balance_cleanup;
			ReleasePage(oldPages[i]);
			oldPages[i] = nullptr;
			i++;
		}

		// Put the new pages in accending order.  This helps to keep entries in the disk file in order so that a scan
		// of the table is a linear scan through the file.  That in turn helps the operating system to deliver pages
		// from the disk more rapidly.
		//
		// An O(n^2) insertion sort algorithm is used, but since n is never more than NB (a small constant), that should
		// not be a problem.
		//
		// When NB==3, this one optimization makes the database about 25% faster for large insertions and deletions.
		for (i = 0; i < k - 1; i++)
		{
			Pid minV = newPages[i]->ID;
			Pid minI = i;
			for (j = i + 1; j < k; j++)
			{
				if (newPages[j]->ID < (Pid)minV)
				{
					minI = j;
					minV = newPages[j]->ID;
				}
			}
			if (minI > i)
			{
				MemPage *t = newPages[i];
				newPages[i] = newPages[minI];
				newPages[minI] = t;
			}
		}
		TRACE("new: %d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n",
			newPages[0]->ID, sizeNew[0],
			newPagesUsed >= 2 ? newPages[1]->ID : 0, newPagesUsed >= 2 ? sizeNew[1] : 0,
			newPagesUsed >= 3 ? newPages[2]->ID : 0, newPagesUsed >= 3 ? sizeNew[2] : 0,
			newPagesUsed >= 4 ? newPages[3]->ID : 0, newPagesUsed >= 4 ? sizeNew[3] : 0,
			newPagesUsed >= 5 ? newPages[4]->ID : 0, newPagesUsed >= 5 ? sizeNew[4] : 0);

		_assert(Pager::Iswriteable(parent->DBPage));
		ConvertEx::Put4(right, newPages[newPagesUsed - 1]->ID);

		// Evenly distribute the data in apCell[] across the new pages. Insert divider cells into pParent as necessary.
		ovflSpaceID = 0; // First unused byte of aOvflSpace[]
		j = 0;
		for (i = 0; i < newPagesUsed; i++)
		{
			// Assemble the new sibling page.
			MemPage *newPage = newPages[i];
			_assert(j < maxCells);
			ZeroPage(newPage, pageFlags);
			AssemblePage(newPage, countNew[i] - j, &cell[j], &sizeCell[j]);
			_assert(newPage->Cells > 0 || (newPagesUsed == 1 && countNew[0] == 0));
			_assert(newPage->Overflows == 0);

			j = countNew[i];

			// If the sibling page assembled above was not the right-most sibling, insert a divider cell into the parent page.
			_assert(i < newPagesUsed - 1 || j == cells);
			if (j < cells)
			{
				_assert(j < maxCells);
				uint8 *pCell = cell[j];
				int sz = sizeCell[j] + leafCorrection;
				uint8 *pTemp = &ovflSpace[ovflSpaceID];
				if (!newPage->Leaf)
					_memcpy(&newPage->Data[8], pCell, 4);
				else if (leafData)
				{
					// If the tree is a leaf-data tree, and the siblings are leaves, then there is no divider cell in apCell[]. Instead, the divider 
					// cell consists of the integer key for the right-most cell of the sibling-page assembled above only.
					j--;
					CellInfo info;
					BtreeParseCellPtr(newPage, cell[j], &info);
					pCell = pTemp;
					sz = 4 + ConvertEx::PutVarint(&pCell[4], info.Key);
					pTemp = nullptr;
				}
				else
				{
					pCell -= 4;
					// Obscure case for non-leaf-data trees: If the cell at pCell was previously stored on a leaf node, and its reported size was 4
					// bytes, then it may actually be smaller than this (see BtreeParseCellPtr(), 4 bytes is the minimum size of
					// any cell). But it is important to pass the correct size to InsertCell(), so reparse the cell now.
					//
					// Note that this can never happen in an SQLite data file, as all cells are at least 4 bytes. It only happens in b-trees used
					// to evaluate "IN (SELECT ...)" and similar clauses.
					if (sizeCell[j] == 4)
					{
						_assert(leafCorrection == 4);
						sz = CellSizePtr(parent, pCell);
					}
				}
				ovflSpaceID += sz;
				_assert(sz <= bt->MaxLocal + 23);
				_assert(ovflSpaceID <= (int)bt->PageSize);
				InsertCell(parent, nxDiv, pCell, sz, pTemp, newPage->ID, &rc);
				if (rc != RC_OK) goto balance_cleanup;
				_assert(Pager::Iswriteable(parent->DBPage));

				j++;
				nxDiv++;
			}
		}
		_assert(j == cells);
		_assert(oldPagesUsed > 0);
		_assert(newPagesUsed > 0);
		if ((pageFlags & PTF_LEAF) == 0)
		{
			uint8 *child = &copyPages[oldPagesUsed - 1]->Data[8];
			_memcpy(&newPages[newPagesUsed - 1]->Data[8], child, 4);
		}

		if (isRoot && parent->Cells == 0 && parent->HdrOffset <= newPages[0]->Frees)
		{
			// The root page of the b-tree now contains no cells. The only sibling page is the right-child of the parent. Copy the contents of the
			// child page into the parent, decreasing the overall height of the b-tree structure by one. This is described as the "Balance-shallower"
			// sub-algorithm in some documentation.
			//
			// If this is an auto-vacuum database, the call to CopyNodeContent() sets all pointer-map entries corresponding to database image pages 
			// for which the pointer is stored within the content being copied.
			//
			// The second assert below verifies that the child page is defragmented (it must be, as it was just reconstructed using AssemblePage()). This
			// is important if the parent page happens to be page 1 of the database image.
			_assert(newPagesUsed == 1);
			_assert(newPages[0]->Frees == (ConvertEx::Get2(&newPages[0]->Data[5]) - newPages[0]->CellOffset - newPages[0]->Cells * 2));
			CopyNodeContent(newPages[0], parent, &rc);
			FreePage(newPages[0], &rc);
		}
#if !OMIT_AUTOVACUUM
		else if (bt->AutoVacuum)
		{
			// Fix the pointer-map entries for all the cells that were shifted around. There are several different types of pointer-map entries that need to
			// be dealt with by this routine. Some of these have been set already, but many have not. The following is a summary:
			//
			//   1) The entries associated with new sibling pages that were not siblings when this function was called. These have already
			//      been set. We don't need to worry about old siblings that were moved to the free-list - the FreePage() code has taken care
			//      of those.
			//
			//   2) The pointer-map entries associated with the first overflow page in any overflow chains used by new divider cells. These 
			//      have also already been taken care of by the InsertCell() code.
			//
			//   3) If the sibling pages are not leaves, then the child pages of cells stored on the sibling pages may need to be updated.
			//
			//   4) If the sibling pages are not internal intkey nodes, then any overflow pages used by these cells may need to be updated
			//      (internal intkey nodes never contain pointers to overflow pages).
			//
			//   5) If the sibling pages are not leaves, then the pointer-map entries for the right-child pages of each sibling may need
			//      to be updated.
			//
			// Cases 1 and 2 are dealt with above by other code. The next block deals with cases 3 and 4 and the one after that, case 5. Since
			// setting a pointer map entry is a relatively expensive operation, this code only sets pointer map entries for child or overflow pages that have
			// actually moved between pages.
			MemPage *newPage = newPages[0];
			MemPage *oldPage = copyPages[0];
			uint overflows = oldPage->Overflows;
			Pid nextOldID = oldPage->Cells + overflows;
			int overflowID = (overflows ? oldPage->OvflIdxs[0] : -1);
			j = 0; // Current 'old' sibling page
			k = 0; // Current 'new' sibling page
			for (i = 0; i < cells; i++)
			{
				bool isDivider = false;
				while (i == nextOldID)
				{
					// Cell i is the cell immediately following the last cell on old sibling page j. If the siblings are not leaf pages of an
					// intkey b-tree, then cell i was a divider cell.
					_assert(j + 1 < _lengthof(copyPages));
					_assert(j + 1 < oldPagesUsed);
					oldPage = copyPages[++j];
					nextOldID = i + !leafData + oldPage->Cells + oldPage->Overflows;
					if (oldPage->Overflows)
					{
						overflows = oldPage->Overflows;
						overflowID = i + !leafData + oldPage->OvflIdxs[0];
					}
					isDivider = !leafData;
				}

				_assert(overflows > 0 || overflowID < i);
				_assert(overflows < 2 || oldPage->OvflIdxs[0] == oldPage->OvflIdxs[1] - 1);
				_assert(overflows < 3 || oldPage->OvflIdxs[1] == oldPage->OvflIdxs[2] - 1);
				if (i == overflowID)
				{
					isDivider = true;
					if (--overflows > 0)
						overflowID++;
				}

				if (i == countNew[k])
				{
					// Cell i is the cell immediately following the last cell on new sibling page k. If the siblings are not leaf pages of an
					// intkey b-tree, then cell i is a divider cell.
					newPage = newPages[++k];
					if (!leafData) continue;
				}
				_assert(j < oldPagesUsed);
				_assert(k < newPagesUsed);

				// If the cell was originally divider cell (and is not now) or an overflow cell, or if the cell was located on a different sibling
				// page before the balancing, then the pointer map entries associated with any child or overflow pages need to be updated.
				if (isDivider || oldPage->ID != newPage->ID)
				{
					if (leafCorrection == 0)
						PtrmapPut(bt, ConvertEx::Get4(cell[i]), PTRMAP_BTREE, newPage->ID, &rc);
					if (sizeCell[i] > newPage->MinLocal)
						PtrmapPutOvflPtr(newPage, cell[i], &rc);
				}
			}
#endif

			if (leafCorrection == 0)
			{
				for (i = 0; i < newPagesUsed; i++)
				{
					uint32 key = ConvertEx::Get4(&newPages[i]->Data[8]);
					PtrmapPut(bt, key, PTRMAP_BTREE, newPages[i]->ID, &rc);
				}
			}

#if 0
			// The PtrmapCheckPages() contains assert() statements that verify that all pointer map pages are set correctly. This is helpful while 
			// debugging. This is usually disabled because a corrupt database may cause an assert() statement to fail.
			PtrmapCheckPages(newPages, newPagesUsed);
			PtrmapCheckPages(&parent, 1);
#endif
		}

		_assert(parent->IsInit);
		TRACE("BALANCE: finished: old=%d new=%d cells=%d\n", oldPagesUsed, newPagesUsed, cells);

		// Cleanup before returning.
balance_cleanup:
		_stackfree(nullptr, cell);
		for (i = 0; i < oldPagesUsed; i++)
			ReleasePage(oldPages[i]);
		for (i = 0; i < newPagesUsed; i++)
			ReleasePage(newPages[i]);

		return rc;
	}
#if defined(_MSC_VER) && _MSC_VER >= 1700 && defined(_M_ARM)
#pragma optimize("", on)
#endif

	__device__ static RC Balance_Deeper(MemPage *root, MemPage **childOut)
	{
		BtShared *bt = root->Bt; // The BTree

		_assert(root->Overflows > 0);
		_assert(MutexEx::Held(bt->Mutex));

		// Make pRoot, the root page of the b-tree, writable. Allocate a new page that will become the new right-child of pPage. Copy the contents
		// of the node stored on pRoot into the new child page.
		MemPage *child = nullptr; // Pointer to a new child page
		Pid childID = 0; // Page number of the new child page 
		RC rc = Pager::Write(root->DBPage);
		if (rc == RC_OK)
		{
			rc = AllocateBtreePage(bt, &child, &childID, root->ID, BTALLOC::ANY);
			CopyNodeContent(root, child, &rc);
#if !OMIT_AUTOVACUUM
			if (bt->AutoVacuum)
				PtrmapPut(bt, childID, PTRMAP_BTREE, root->ID, &rc);
#endif
		}
		if (rc)
		{
			*childOut = nullptr;
			ReleasePage(child);
			return rc;
		}
		_assert(Pager::Iswriteable(child->DBPage));
		_assert(Pager::Iswriteable(root->DBPage));
		_assert(child->Cells == root->Cells);

		TRACE("BALANCE: copy root %d into %d\n", root->ID, child->ID);

		// Copy the overflow cells from pRoot to pChild
		_memcpy(child->OvflIdxs, root->OvflIdxs, root->Overflows * sizeof(root->OvflIdxs[0]));
		_memcpy(child->Ovfls, root->Ovfls, root->Overflows * sizeof(root->Ovfls[0]));
		child->Overflows = root->Overflows;

		// Zero the contents of pRoot. Then install pChild as the right-child.
		ZeroPage(root, child->Data[0] & ~PTF_LEAF);
		ConvertEx::Put4(&root->Data[root->HdrOffset + 8], childID);

		*childOut = child;
		return RC_OK;
	}

	__device__ static RC Balance(BtCursor *cur)
	{
		ASSERTONLY(int balance_quick_called = 0);
		ASSERTONLY(int balance_deeper_called = 0);

		uint8 *free = nullptr;
		const int min = cur->Bt->UsableSize * 2 / 3;
		RC rc = RC_OK;
		do
		{
			int pageID = cur->ID;
			MemPage *page = cur->Pages[pageID];

			if (pageID == 0)
			{
				if (page->Overflows)
				{
					// The root page of the b-tree is overfull. In this case call the Balance_Deeper() function to create a new child for the root-page
					// and copy the current contents of the root-page to it. The next iteration of the do-loop will Balance the child page.
					_assert(balance_deeper_called++ == 0);
					rc = Balance_Deeper(page, &cur->Pages[1]);
					if (rc == RC_OK)
					{
						cur->ID = 1;
						cur->Idxs[0] = 0;
						cur->Idxs[1] = 0;
						_assert(cur->Pages[1]->Overflows);
					}
				}
				else
					break;
			}
			else if (page->Overflows == 0 && page->Frees <= min)
				break;
			else
			{
				MemPage *const parent = cur->Pages[pageID - 1];
				uint16 const idx = cur->Idxs[pageID - 1];

				rc = Pager::Write(parent->DBPage);
				if (rc == RC_OK)
				{
#ifndef OMIT_QUICKBALANCE
					if (page->HasData && page->Overflows == 1 && page->OvflIdxs[0] == page->Cells && parent->ID != 1 && parent->Cells == idx)
					{
						// Call Balance_Quick() to create a new sibling of pPage on which to store the overflow cell. Balance_Quick() inserts a new cell
						// into pParent, which may cause pParent overflow. If this happens, the next interation of the do-loop will Balance pParent 
						// use either Balance_Nonroot() or Balance_Deeper(). Until this happens, the overflow cell is stored in the aBalanceQuickSpace[] buffer. 
						//
						// The purpose of the following assert() is to check that only a single call to Balance_Quick() is made for each call to this
						// function. If this were not verified, a subtle bug involving reuse of the aBalanceQuickSpace[] might sneak in.
						_assert(balance_quick_called++ == 0);
						uint8 balanceQuickSpace[13];
						rc = Balance_Quick(parent, page, balanceQuickSpace);
					}
					else
#endif
					{
						// In this case, call Balance_Nonroot() to redistribute cells between pPage and up to 2 of its sibling pages. This involves
						// modifying the contents of pParent, which may cause pParent to become overfull or underfull. The next iteration of the do-loop
						// will Balance the parent page to correct this.
						// 
						// If the parent page becomes overfull, the overflow cell or cells are stored in the pSpace buffer allocated immediately below. 
						// A subsequent iteration of the do-loop will deal with this by calling Balance_Nonroot() (Balance_Deeper() may be called first,
						// but it doesn't deal with overflow cells - just moves them to a different page). Once this subsequent call to Balance_Nonroot() 
						// has completed, it is safe to release the pSpace buffer used by the previous call, as the overflow cell data will have been 
						// copied either into the body of a database page or into the new pSpace buffer passed to the latter call to Balance_Nonroot().
						uint8 *space = (uint8 *)PCache::PageAlloc(cur->Bt->PageSize);
						rc = Balance_Nonroot(parent, idx, space, pageID == 1, (cur->Hints ? true : false));
						if (free)
						{
							// If pFree is not NULL, it points to the pSpace buffer used  by a previous call to Balance_Nonroot(). Its contents are
							// now stored either on real database pages or within the new pSpace buffer, so it may be safely freed here.
							PCache::PageFree(free);
						}

						// The pSpace buffer will be freed after the next call to Balance_Nonroot(), or just before this function returns, whichever comes first.
						free = space;
					}
				}

				page->Overflows = 0;

				// The next iteration of the do-loop balances the parent page.
				ReleasePage(page);
				cur->ID--;
			}
		} while (rc == RC_OK);

		if (free)
			PCache::PageFree(free);
		return rc;
	}

#pragma endregion

#pragma region Insert / Delete / CreateTable / DropTable

	__device__ RC Btree::Insert(BtCursor *cur, const void *key, int64 keyLength, const void *data, int dataLength, int zero, int appendBias, int seekResult)
	{
		if (cur->State == CURSOR_FAULT)
		{
			_assert(cur->SkipNext != 0);
			return (RC)cur->SkipNext;
		}

		Btree *p = cur->Btree;
		BtShared *bt = p->Bt;
		_assert(CursorHoldsMutex(cur));
		_assert(cur->WrFlag && bt->InTransaction == TRANS_WRITE && (bt->BtsFlags & BTS_READ_ONLY) == 0);
		_assert(HasSharedCacheTableLock(p, cur->RootID, cur->KeyInfo != nullptr, LOCK_WRITE));

		// Assert that the caller has been consistent. If this cursor was opened expecting an index b-tree, then the caller should be inserting blob
		// keys with no associated data. If the cursor was opened expecting an intkey table, the caller should be inserting integer keys with a
		// blob of associated data.
		_assert((key == nullptr) == (cur->KeyInfo == nullptr));

		// Save the positions of any other cursors open on this table.
		//
		// In some cases, the call to BtreeMoveto() below is a no-op. For example, when inserting data into a table with auto-generated integer
		// keys, the VDBE layer invokes sqlite3BtreeLast() to figure out the integer key to use. It then calls this function to actually insert the 
		// data into the intkey B-Tree. In this case BtreeMoveto() recognizes that the cursor is already where it needs to be and returns without
		// doing any work. To avoid thwarting these optimizations, it is important not to clear the cursor here.
		RC rc = SaveAllCursors(bt, cur->RootID, cur);
		if (rc) return rc;

		// If this is an insert into a table b-tree, invalidate any incrblob cursors open on the row being replaced (assuming this is a replace
		// operation - if it is not, the following is a no-op).
		if (cur->KeyInfo == nullptr)
			InvalidateIncrblobCursors(p, keyLength, false);

		int loc = seekResult; // -1: before desired location  +1: after
		if (!loc)
		{
			rc = BtreeMoveto(cur, key, keyLength, appendBias, &loc);
			if (rc) return rc;
		}
		_assert(cur->State == CURSOR_VALID || (cur->State == CURSOR_INVALID && loc));

		MemPage *page = cur->Pages[cur->ID];
		_assert(page->IntKey || keyLength >= 0);
		_assert(page->Leaf || !page->IntKey);

		TRACE("INSERT: table=%d nkey=%lld ndata=%d page=%d %s\n", cur->RootID, keyLength, dataLength, page->ID, loc == 0 ? "overwrite" : "new entry");
		_assert(page->IsInit);
		AllocateTempSpace(bt);
		uint8 *newCell = bt->TmpSpace;
		if (newCell == nullptr) return RC_NOMEM;
		uint16 sizeNew = 0;
		rc = FillInCell(page, newCell, key, keyLength, data, dataLength, zero, &sizeNew);
		uint idx;
		if (rc) goto end_insert;
		_assert(sizeNew == CellSizePtr(page, newCell));
		_assert(sizeNew <= MX_CELL_SIZE(bt));
		idx = cur->Idxs[cur->ID];
		if (loc == 0)
		{
			_assert(idx < page->Cells);
			rc = Pager::Write(page->DBPage);
			if (rc)
				goto end_insert;
			uint8 *oldCell = FindCell(page, idx);
			if (!page->Leaf)
				_memcpy(newCell, oldCell, 4);
			uint16 sizeOld = CellSizePtr(page, oldCell);
			rc = ClearCell(page, oldCell);
			DropCell(page, idx, sizeOld, &rc);
			if (rc) goto end_insert;
		}
		else if (loc < 0 && page->Cells > 0)
		{
			_assert(page->Leaf);
			idx = ++cur->Idxs[cur->ID];
		}
		else
			_assert(page->Leaf);
		InsertCell(page, idx, newCell, sizeNew, 0, 0, &rc);
		_assert(rc != RC_OK || page->Cells > 0 || page->Overflows > 0);

		// If no error has occurred and pPage has an overflow cell, call Balance() to redistribute the cells within the tree. Since Balance() may move
		// the cursor, zero the BtCursor.info.nSize and BtCursor.validNKey variables.
		//
		// Previous versions of SQLite called MoveToRoot() to move the cursor back to the root page as Balance() used to invalidate the contents
		// of BtCursor.apPage[] and BtCursor.aiIdx[]. Instead of doing that, set the cursor state to "invalid". This makes common insert operations
		// slightly faster.
		//
		// There is a subtle but important optimization here too. When inserting multiple records into an intkey b-tree using a single cursor (as can
		// happen while processing an "INSERT INTO ... SELECT" statement), it is advantageous to leave the cursor pointing to the last entry in
		// the b-tree if possible. If the cursor is left pointing to the last entry in the table, and the next row inserted has an integer key
		// larger than the largest existing key, it is possible to insert the row without seeking the cursor. This can be a big performance boost.
		cur->Info.Size = 0;
		cur->ValidNKey = 0;
		if (rc == RC_OK && page->Overflows)
		{
			rc = Balance(cur);

			// Must make sure nOverflow is reset to zero even if the Balance() fails. Internal data structure corruption will result otherwise. 
			// Also, set the cursor state to invalid. This stops SaveCursorPosition() from trying to save the current position of the cursor.
			cur->Pages[cur->ID]->Overflows = 0;
			cur->State = CURSOR_INVALID;
		}
		_assert(cur->Pages[cur->ID]->Overflows == 0);

end_insert:
		return rc;
	}

	__device__ RC Btree::Delete(BtCursor *cur)
	{
		Btree *p = cur->Btree;
		BtShared *bt = p->Bt;
		_assert(CursorHoldsMutex(cur));
		_assert(bt->InTransaction == TRANS_WRITE);
		_assert((bt->BtsFlags & BTS_READ_ONLY) == 0);
		_assert(cur->WrFlag);
		_assert(HasSharedCacheTableLock(p, cur->RootID, cur->KeyInfo != nullptr, LOCK_WRITE));
		_assert(!HasReadConflicts(p, cur->RootID));

		if (_NEVER(cur->Idxs[cur->ID] >= cur->Pages[cur->ID]->Cells) || _NEVER(cur->State != CURSOR_VALID))
			return RC_ERROR; // Something has gone awry.

		int cellDepth = cur->ID; // Depth of node containing pCell
		uint cellIdx = cur->Idxs[cellDepth]; // Index of cell to delete
		MemPage *page = cur->Pages[cellDepth]; // Page to delete cell from
		unsigned char *cell = FindCell(page, cellIdx); // Pointer to cell to delete

		// If the page containing the entry to delete is not a leaf page, move the cursor to the largest entry in the tree that is smaller than
		// the entry being deleted. This cell will replace the cell being deleted from the internal node. The 'previous' entry is used for this instead
		// of the 'next' entry, as the previous entry is always a part of the sub-tree headed by the child page of the cell being deleted. This makes
		// balancing the tree following the delete operation easier.
		RC rc;
		if (!page->Leaf)
		{
			int notUsed;
			rc = Previous(cur, &notUsed);
			if (rc) return rc;
		}

		// Save the positions of any other cursors open on this table before making any modifications. Make the page containing the entry to be 
		// deleted writable. Then free any overflow pages associated with the entry and finally remove the cell itself from within the page.  
		rc = SaveAllCursors(bt, cur->RootID, cur);
		if (rc) return rc;

		// If this is a delete operation to remove a row from a table b-tree, invalidate any incrblob cursors open on the row being deleted.
		if (cur->KeyInfo == 0)
			InvalidateIncrblobCursors(p, cur->Info.Key, false);

		rc = Pager::Write(page->DBPage);
		if (rc) return rc;
		rc = ClearCell(page, cell);
		DropCell(page, cellIdx, CellSizePtr(page, cell), &rc);
		if (rc) return rc;

		// If the cell deleted was not located on a leaf page, then the cursor is currently pointing to the largest entry in the sub-tree headed
		// by the child-page of the cell that was just deleted from an internal node. The cell from the leaf node needs to be moved to the internal
		// node to replace the deleted cell.
		if (!page->Leaf)
		{
			MemPage *leaf = cur->Pages[cur->ID];
			Pid n = cur->Pages[cellDepth + 1]->ID;

			cell = FindCell(leaf, leaf->Cells - 1);
			uint16 sizeCell = CellSizePtr(leaf, cell);
			_assert(MX_CELL_SIZE(bt) >= sizeCell);

			AllocateTempSpace(bt);
			uint8 *tmp = bt->TmpSpace;

			rc = Pager::Write(leaf->DBPage);
			InsertCell(page, cellIdx, cell - 4, sizeCell + 4, tmp, n, &rc);
			DropCell(leaf, leaf->Cells - 1, sizeCell, &rc);
			if (rc) return rc;
		}

		// Balance the tree. If the entry deleted was located on a leaf page, then the cursor still points to that page. In this case the first
		// call to Balance() repairs the tree, and the if(...) condition is never true.
		//
		// Otherwise, if the entry deleted was on an internal node page, then pCur is pointing to the leaf page from which a cell was removed to
		// replace the cell deleted from the internal node. This is slightly tricky as the leaf node may be underfull, and the internal node may
		// be either under or overfull. In this case run the balancing algorithm on the leaf node first. If the Balance proceeds far enough up the
		// tree that we can be sure that any problem in the internal node has been corrected, so be it. Otherwise, after balancing the leaf node,
		// walk the cursor up the tree to the internal node and Balance it as well.
		rc = Balance(cur);
		if (rc == RC_OK && cur->ID > cellDepth)
		{
			while (cur->ID > cellDepth)
				ReleasePage(cur->Pages[cur->ID--]);
			rc = Balance(cur);
		}

		if (rc == RC_OK)
			MoveToRoot(cur);
		return rc;
	}

	__device__ static RC BtreeCreateTable(Btree *p, int *tableID, int createTabFlags)
	{
		BtShared *bt = p->Bt;
		_assert(p->HoldsMutex());
		_assert(bt->InTransaction == TRANS_WRITE);
		_assert((bt->BtsFlags & BTS_READ_ONLY) == 0);

		RC rc;
		MemPage *root;
		Pid rootID;
#ifdef OMIT_AUTOVACUUM
		rc = AllocateBtreePage(bt, &root, &rootID, 1, BTALLOC::ANY);
		if (rc)
			return rc;
#else
		if (bt->AutoVacuum)
		{
			// Creating a new table may probably require moving an existing database to make room for the new tables root page. In case this page turns
			// out to be an overflow page, delete all overflow page-map caches held by open cursors.
			InvalidateAllOverflowCache(bt);

			// Read the value of meta[3] from the database to determine where the root page of the new table should go. meta[3] is the largest root-page
			// created so far, so the new root-page is (meta[3]+1).
			p->GetMeta(Btree::META_LARGEST_ROOT_PAGE, &rootID);
			rootID++;

			// The new root-page may not be allocated on a pointer-map page, or the PENDING_BYTE page.
			while (rootID == PTRMAP_PAGENO(bt, rootID) || rootID == PENDING_BYTE_PAGE(bt))
				rootID++;
			_assert(rootID >= 3);

			// Allocate a page. The page that currently resides at pgnoRoot will be moved to the allocated page (unless the allocated page happens
			// to reside at pgnoRoot).
			Pid moveID; // Move a page here to make room for the root-page
			MemPage *pageMove; // The page to move to.
			rc = AllocateBtreePage(bt, &pageMove, &moveID, rootID, BTALLOC::EXACT);
			if (rc != RC_OK)
				return rc;

			if (moveID != rootID)
			{
				ReleasePage(pageMove);

				// Move the page currently at pgnoRoot to pgnoMove.
				rc = BtreeGetPage(bt, rootID, &root, false);
				if (rc != RC_OK)
					return rc;

				// pgnoRoot is the page that will be used for the root-page of the new table (assuming an error did not occur). But we were
				// allocated pgnoMove. If required (i.e. if it was not allocated by extending the file), the current page at position pgnoMove
				// is already journaled.
				PTRMAP type = (PTRMAP)0;
				Pid ptrPageID = 0;
				rc = PtrmapGet(bt, rootID, &type, &ptrPageID);
				if (type == PTRMAP_ROOTPAGE || type == PTRMAP_FREEPAGE)
					rc = SysEx_CORRUPT_BKPT;
				if (rc != RC_OK)
				{
					ReleasePage(root);
					return rc;
				}
				_assert(type != PTRMAP_ROOTPAGE);
				_assert(type != PTRMAP_FREEPAGE);
				rc = RelocatePage(bt, root, type, ptrPageID, moveID, false);
				ReleasePage(root);

				// Obtain the page at pgnoRoot
				if (rc != RC_OK)
					return rc;
				rc = BtreeGetPage(bt, rootID, &root, false);
				if (rc != RC_OK)
					return rc;
				rc = Pager::Write(root->DBPage);
				if (rc != RC_OK)
				{
					ReleasePage(root);
					return rc;
				}
			}
			else
				root = pageMove;

			// Update the pointer-map and meta-data with the new root-page number.
			PtrmapPut(bt, rootID, PTRMAP_ROOTPAGE, 0, &rc);
			if (rc)
			{
				ReleasePage(root);
				return rc;
			}

			// When the new root page was allocated, page 1 was made writable in order either to increase the database filesize, or to decrement the
			// freelist count.  Hence, the sqlite3BtreeUpdateMeta() call cannot fail.
			_assert(Pager::Iswriteable(bt->Page1->DBPage));
			rc = p->UpdateMeta(Btree::META_LARGEST_ROOT_PAGE, rootID);
			if (_NEVER(rc))
			{
				ReleasePage(root);
				return rc;
			}
		}
		else
		{
			rc = AllocateBtreePage(bt, &root, &rootID, 1, BTALLOC::ANY);
			if (rc) return rc;
		}
#endif
		_assert(Pager::Iswriteable(root->DBPage));
		int ptfFlags; // Page-type flage for the root page of new table
		if (createTabFlags & BTREE_INTKEY)
			ptfFlags = PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF;
		else
			ptfFlags = PTF_ZERODATA | PTF_LEAF;
		ZeroPage(root, ptfFlags);
		Pager::Unref(root->DBPage);
		_assert((bt->OpenFlags & Btree::OPEN_SINGLE) == 0 || rootID == 2);
		*tableID = (int)rootID;
		return RC_OK;
	}

	__device__ RC Btree::CreateTable(int *tableID, int flags)
	{
		Enter();
		RC rc = BtreeCreateTable(this, tableID, flags);
		Leave();
		return rc;
	}

	__device__ static RC ClearDatabasePage(BtShared *bt, Pid id, bool freePageFlag, int *changes)
	{
		_assert(MutexEx::Held(bt->Mutex));
		if (id > BtreePagecount(bt))
			return SysEx_CORRUPT_BKPT;

		MemPage *page;
		RC rc = GetAndInitPage(bt, id, &page);
		if (rc) return rc;
		for (int i = 0; i < page->Cells; i++)
		{
			unsigned char *cell = FindCell(page, i);
			if (!page->Leaf)
			{
				rc = ClearDatabasePage(bt, ConvertEx::Get4(cell), true, changes);
				if (rc) goto cleardatabasepage_out;
			}
			rc = ClearCell(page, cell);
			if (rc) goto cleardatabasepage_out;
		}
		if (!page->Leaf)
		{
			rc = ClearDatabasePage(bt, ConvertEx::Get4(&page->Data[8]), true, changes);
			if (rc) goto cleardatabasepage_out;
		}
		else if (changes)
		{
			_assert(page->IntKey);
			*changes += page->Cells;
		}
		if (freePageFlag)
			FreePage(page, &rc);
		else if ((rc = Pager::Write(page->DBPage)) == 0)
			ZeroPage(page, page->Data[0] | PTF_LEAF);

cleardatabasepage_out:
		ReleasePage(page);
		return rc;
	}

	__device__ RC Btree::ClearTable(int tableID, int *changes)
	{
		BtShared *bt = Bt;
		Enter();
		_assert(InTrans == TRANS_WRITE);
		RC rc = SaveAllCursors(bt, (Pid)tableID, 0);
		if (rc == RC_OK)
		{
			// Invalidate all incrblob cursors open on table iTable (assuming iTable is the root of a table b-tree - if it is not, the following call is a no-op).
			InvalidateIncrblobCursors(this, 0, true);
			rc = ClearDatabasePage(bt, (Pid)tableID, false, changes);
		}
		Leave();
		return rc;
	}

	__device__ static RC BtreeDropTable(Btree *p, Pid tableID, int *movedID)
	{
		BtShared *bt = p->Bt;
		_assert(p->HoldsMutex());
		_assert(p->InTrans == TRANS_WRITE );

		// It is illegal to drop a table if any cursors are open on the database. This is because in auto-vacuum mode the backend may
		// need to move another root-page to fill a gap left by the deleted root page. If an open cursor was using this page a problem would occur.
		//
		// This error is caught long before control reaches this point.
		if (_NEVER(bt->Cursor))
		{
			p->Ctx->ConnectionBlocked(bt->Cursor->Btree->Ctx);
			return RC_LOCKED_SHAREDCACHE;
		}

		MemPage *page = nullptr;
		RC rc = BtreeGetPage(bt, (Pid)tableID, &page, false);
		if (rc) return rc;
		rc = p->ClearTable(tableID, nullptr);
		if (rc)
		{
			ReleasePage(page);
			return rc;
		}

		*movedID = 0;

		if (tableID > 1)
		{
#ifdef OMIT_AUTOVACUUM
			FreePage(page, &rc);
			ReleasePage(page);
#else
			if (bt->AutoVacuum)
			{
				Pid maxRootID;
				p->GetMeta(Btree::META_LARGEST_ROOT_PAGE, &maxRootID);

				if (tableID == maxRootID)
				{
					// If the table being dropped is the table with the largest root-page number in the database, put the root page on the free list. 
					FreePage(page, &rc);
					ReleasePage(page);
					if (rc != RC_OK)
						return rc;
				}
				else
				{
					// The table being dropped does not have the largest root-page number in the database. So move the page that does into the 
					// gap left by the deleted root-page.
					ReleasePage(page);
					MemPage *move;
					rc = BtreeGetPage(bt, maxRootID, &move, false);
					if (rc != RC_OK)
						return rc;
					rc = RelocatePage(bt, move, PTRMAP_ROOTPAGE, 0, tableID, false);
					ReleasePage(move);
					if (rc != RC_OK)
						return rc;
					move = nullptr;
					rc = BtreeGetPage(bt, maxRootID, &move, 0);
					FreePage(move, &rc);
					ReleasePage(move);
					if (rc != RC_OK)
						return rc;
					*movedID = maxRootID;
				}

				// Set the new 'max-root-page' value in the database header. This is the old value less one, less one more if that happens to
				// be a root-page number, less one again if that is the PENDING_BYTE_PAGE.
				maxRootID--;
				while (maxRootID == PENDING_BYTE_PAGE(bt) || PTRMAP_ISPAGE(bt, maxRootID))
					maxRootID--;
				_assert(maxRootID != PENDING_BYTE_PAGE(bt));

				rc = p->UpdateMeta(Btree::META_LARGEST_ROOT_PAGE, maxRootID);
			}
			else
			{
				FreePage(page, &rc);
				ReleasePage(page);
			}
#endif
		}
		else
		{
			// If sqlite3BtreeDropTable was called on page 1. This really never should happen except in a corrupt database. 
			ZeroPage(page, PTF_INTKEY | PTF_LEAF);
			ReleasePage(page);
		}
		return rc;  
	}

	__device__ RC Btree::DropTable(int tableID, int *movedID)
	{
		Enter();
		RC rc = BtreeDropTable(this, (Pid)tableID, movedID);
		Leave();
		return rc;
	}

#pragma endregion

#pragma region Meta / Count

	__device__ void Btree::GetMeta(META id, uint32 *meta)
	{
		BtShared *bt = Bt;
		Enter();
		_assert(InTrans > TRANS_NONE);
		_assert(QuerySharedCacheTableLock(this, MASTER_ROOT, LOCK_READ) == RC_OK);
		_assert(bt->Page1 != nullptr);
		_assert((int)id >= 0 && (int)id <= 15);
		*meta = ConvertEx::Get4(&bt->Page1->Data[36 + (int)id * 4]);
		// If auto-vacuum is disabled in this build and this is an auto-vacuum database, mark the database as read-only.
#ifdef OMIT_AUTOVACUUM
		if (idx == BTREE_LARGEST_ROOT_PAGE && *meta > 0)
			bt->BtsFlags |= BTS_READ_ONLY;
#endif
		Leave();
	}

	__device__ RC Btree::UpdateMeta(META id, uint32 meta)
	{
		BtShared *bt = Bt;
		_assert((int)id >= 1 && (int)id <= 15);
		Enter();
		_assert(InTrans == TRANS_WRITE);
		_assert(bt->Page1 != 0);
		unsigned char *p1 = bt->Page1->Data;
		RC rc = Pager::Write(bt->Page1->DBPage);
		if (rc == RC_OK)
		{
			ConvertEx::Put4(&p1[36 + (int)id * 4], meta);
#ifndef OMIT_AUTOVACUUM
			if (id == META_INCR_VACUUM)
			{
				_assert(bt->AutoVacuum || meta == 0);
				_assert(meta == 0 || meta == 1);
				bt->IncrVacuum = (meta != 0);
			}
#endif
		}
		Leave();
		return rc;
	}

#ifndef OMIT_BTREECOUNT
	__device__ RC Btree::Count(BtCursor *cur, int64 *entrysRef)
	{
		if (cur->RootID == 0)
		{
			*entrysRef = 0;
			return RC_OK;
		}
		RC rc = MoveToRoot(cur);

		// Unless an error occurs, the following loop runs one iteration for each page in the B-Tree structure (not including overflow pages). 
		int64 entrys = 0; // Value to return in *pnEntry
		while (rc == RC_OK)
		{
			// If this is a leaf page or the tree is not an int-key tree, then this page contains countable entries. Increment the entry counter accordingly.
			MemPage *page = cur->Pages[cur->ID]; // Current page of the b-tree
			if (page->Leaf || !page->IntKey)
				entrys += page->Cells;

			// pPage is a leaf node. This loop navigates the cursor so that it points to the first interior cell that it points to the parent of
			// the next page in the tree that has not yet been visited. The pCur->aiIdx[pCur->iPage] value is set to the index of the parent cell
			// of the page, or to the number of cells in the page if the next page to visit is the right-child of its parent.
			//
			// If all pages in the tree have been visited, return SQLITE_OK to the caller.
			if (page->Leaf)
			{
				do
				{
					if (cur->ID == 0)
					{
						// All pages of the b-tree have been visited. Return successfully.
						*entrysRef = entrys;
						return RC_OK;
					}
					MoveToParent(cur);
				} while (cur->Idxs[cur->ID] >= cur->Pages[cur->ID]->Cells);

				cur->Idxs[cur->ID]++;
				page = cur->Pages[cur->ID];
			}

			// Descend to the child node of the cell that the cursor currently points at. This is the right-child if (iIdx==pPage->nCell).
			uint idx = cur->Idxs[cur->ID]; // Index of child node in parent
			if (idx == page->Cells)
				rc = MoveToChild(cur, ConvertEx::Get4(&page->Data[page->HdrOffset + 8]));
			else
				rc = MoveToChild(cur, ConvertEx::Get4(FindCell(page, idx)));
		}

		// An error has occurred. Return an error code.
		return rc;
	}
#endif

	__device__ Pager *Btree::get_Pager()
	{
		return Bt->Pager;
	}

#pragma endregion

#pragma region Integrity Check
#ifndef OMIT_INTEGRITY_CHECK

	__device__ static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, va_list *args)
	{
		//va_list ap;
		if (!check->MaxErrors) return;
		check->MaxErrors--;
		check->Errors++;
		if (check->ErrMsg.Index)
			check->ErrMsg.Append("\n", 1);
		if (msg1)
			check->ErrMsg.Append(msg1, -1);
		check->ErrMsg.AppendFormat(true, fmt, args);
		if (check->ErrMsg.AllocFailed)
			check->MallocFailed = true;
	}
#if __CUDACC__
	__device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt) { va_list args; va_start(args, nullptr); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
	template <typename T1> __device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
	template <typename T1, typename T2> __device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
	template <typename T1, typename T2, typename T3> __device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); CheckAppendMsg(check, msg1, fmt, &args); va_end(args); }
#else
	__device__ inline static void CheckAppendMsg(IntegrityCk *check, char *msg1, const char *fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		CheckAppendMsg(check, msg1, fmt, &args);
		va_end(args);
	}
#endif

	__device__ static bool GetPageReferenced(IntegrityCk *check, Pid pageID)
	{
		_assert(pageID <= check->Pages && sizeof(check->PgRefs[0]) == 1);
		return (check->PgRefs[pageID / 8] & (1 << (pageID & 0x07)) ? true : false);
	}

	__device__ static void SetPageReferenced(IntegrityCk *check, Pid pageID)
	{
		_assert(pageID <= check->Pages && sizeof(check->PgRefs[0]) == 1);
		check->PgRefs[pageID / 8] |= (1 << (pageID & 0x07));
	}

	__device__ static bool CheckRef(IntegrityCk *check, Pid pageID, char *context)
	{
		if (pageID == 0) return true;
		if (pageID > check->Pages)
		{
			CheckAppendMsg(check, context, "invalid page number %d", pageID);
			return true;
		}
		if (GetPageReferenced(check, pageID))
		{
			CheckAppendMsg(check, context, "2nd reference to page %d", pageID);
			return true;
		}
		SetPageReferenced(check, pageID);
		return false;
	}

#ifndef OMIT_AUTOVACUUM
	__device__ static void CheckPtrmap(IntegrityCk *check, Pid childID, PTRMAP type, Pid parentID, char *context)
	{
		PTRMAP ptrmapType;
		Pid ptrmapParentID;
		RC rc = PtrmapGet(check->Bt, childID, &ptrmapType, &ptrmapParentID);
		if (rc != RC_OK)
		{
			if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM) check->MallocFailed = true;
			CheckAppendMsg(check, context, "Failed to read ptrmap key=%d", childID);
			return;
		}

		if (ptrmapType != type || ptrmapParentID != parentID)
			CheckAppendMsg(check, context, "Bad ptr map entry key=%d expected=(%d,%d) got=(%d,%d)", childID, type, parentID, ptrmapType, ptrmapParentID);
	}
#endif

	__device__ static void CheckList(IntegrityCk *check, bool isFreeList, Pid pageID, int length, char *context)
	{
		int expected = length;
		Pid firstID = pageID;
		while (length-- > 0 && check->MaxErrors)
		{
			if (pageID < 1)
			{
				CheckAppendMsg(check, context, "%d of %d pages missing from overflow list starting at %d", length + 1, expected, firstID);
				break;
			}
			if (CheckRef(check, pageID, context)) break;
			IPage *ovflPage;
			if (check->Pager->Acquire((Pid)pageID, &ovflPage, false))
			{
				CheckAppendMsg(check, context, "failed to get page %d", pageID);
				break;
			}
			uint8 *ovflData = (uint8 *)Pager::GetData(ovflPage);
			if (isFreeList)
			{
				int n = ConvertEx::Get4(&ovflData[4]);
#ifndef OMIT_AUTOVACUUM
				if (check->Bt->AutoVacuum)
					CheckPtrmap(check, pageID, PTRMAP_FREEPAGE, 0, context);
#endif
				if (n > (int)check->Bt->UsableSize / 4 - 2)
				{
					CheckAppendMsg(check, context, "freelist leaf count too big on page %d", pageID);
					length--;
				}
				else
				{
					for (int i = 0; i < n; i++)
					{
						Pid freePageID = ConvertEx::Get4(&ovflData[8 + i * 4]);
#ifndef OMIT_AUTOVACUUM
						if (check->Bt->AutoVacuum)
							CheckPtrmap(check, freePageID, PTRMAP_FREEPAGE, 0, context);
#endif
						CheckRef(check, freePageID, context);
					}
					length -= n;
				}
			}
#ifndef OMIT_AUTOVACUUM
			else
			{
				// If this database supports auto-vacuum and iPage is not the last page in this overflow list, check that the pointer-map entry for
				// the following page matches iPage.
				if (check->Bt->AutoVacuum && length > 0)
				{
					int i = ConvertEx::Get4(ovflData);
					CheckPtrmap(check, i, PTRMAP_OVERFLOW2, pageID, context);
				}
			}
#endif
			pageID = ConvertEx::Get4(ovflData);
			Pager::Unref(ovflPage);
		}
	}

	__device__ static int CheckTreePage(IntegrityCk *check, Pid pageID, char *parentContext, int64 *parentMinKey, int64 *parentMaxKey)
	{
		char msg[100];
		__snprintf(msg, sizeof(msg), "Page %d: ", pageID);

		// Check that the page exists
		BtShared *bt = check->Bt;
		int usableSize = bt->UsableSize;
		if (pageID == 0) return 0;
		if (CheckRef(check, pageID, parentContext)) return 0;
		RC rc;
		MemPage *page;
		if ((rc = BtreeGetPage(bt, pageID, &page, false)) != RC_OK)
		{
			CheckAppendMsg(check, msg, "unable to get the page. error code=%d", rc);
			return 0;
		}

		// Clear MemPage.isInit to make sure the corruption detection code in BtreeInitPage() is executed.
		page->IsInit = false;
		if ((rc = BtreeInitPage(page)) != RC_OK)
		{
			_assert(rc == RC_CORRUPT); // The only possible error from InitPage
			CheckAppendMsg(check, msg, "BtreeInitPage() returns error code %d", rc);
			ReleasePage(page);
			return 0;
		}

		// Check out all the cells.
		Pid id;
		uint i;
		int  depth = 0;
		int64 minKey = 0;
		int64 maxKey = 0;
		for (i = 0U; i < page->Cells && check->MaxErrors; i++)
		{
			// Check payload overflow pages
			__snprintf(msg, sizeof(msg), "On tree page %d cell %d: ", pageID, i);
			uint8 *cell = FindCell(page,i);
			CellInfo info;
			BtreeParseCellPtr(page, cell, &info);
			uint32 sizeCell = info.Data;
			if (!page->IntKey) sizeCell += (int)info.Key;
			// For intKey pages, check that the keys are in order.
			else if (i == 0) minKey = maxKey = info.Key;
			else
			{
				if (info.Key <= maxKey)
					CheckAppendMsg(check, msg, "Rowid %lld out of order (previous was %lld)", info.Key, maxKey);
				maxKey = info.Key;
			}
			_assert(sizeCell == info.Payload);
			if (sizeCell > info.Local && &cell[info.Overflow] <= &page->Data[bt->UsableSize])
			{
				int pages = (sizeCell - info.Local + usableSize - 5) / (usableSize - 4);
				Pid ovflID = ConvertEx::Get4(&cell[info.Overflow]);
#ifndef OMIT_AUTOVACUUM
				if (bt->AutoVacuum)
					CheckPtrmap(check, ovflID, PTRMAP_OVERFLOW1, pageID, msg);
#endif
				CheckList(check, false, ovflID, pages, msg);
			}

			// Check sanity of left child page.
			if (!page->Leaf)
			{
				id = ConvertEx::Get4(cell);
#ifndef OMIT_AUTOVACUUM
				if (bt->AutoVacuum)
					CheckPtrmap(check, id, PTRMAP_BTREE, pageID, msg);
#endif
				int depth2 = CheckTreePage(check, id, msg, &minKey, i == 0 ? nullptr : &maxKey);
				if (i > 0 && depth2 != depth)
					CheckAppendMsg(check, msg, "Child page depth differs");
				depth = depth2;
			}
		}

		if (!page->Leaf)
		{
			id = ConvertEx::Get4(&page->Data[page->HdrOffset + 8]);
			__snprintf(msg, sizeof(msg), "On page %d at right child: ", pageID);
#ifndef OMIT_AUTOVACUUM
			if (bt->AutoVacuum)
				CheckPtrmap(check, id, PTRMAP_BTREE, pageID, msg);
#endif
			CheckTreePage(check, id, msg, nullptr, (!page->Cells ? nullptr : &maxKey));
		}

		// For intKey leaf pages, check that the min/max keys are in order with any left/parent/right pages.
		if (page->Leaf && page->IntKey)
		{
			// if we are a left child page
			if (parentMinKey)
			{
				// if we are the left most child page
				if (!parentMaxKey)
				{
					if (maxKey > *parentMinKey)
						CheckAppendMsg(check, msg, "Rowid %lld out of order (max larger than parent min of %lld)", maxKey, *parentMinKey);
				}
				else
				{
					if (minKey <= *parentMinKey)
						CheckAppendMsg(check, msg, "Rowid %lld out of order (min less than parent min of %lld)", minKey, *parentMinKey);
					if (maxKey > *parentMaxKey)
						CheckAppendMsg(check, msg, "Rowid %lld out of order (max larger than parent max of %lld)", maxKey, *parentMaxKey);
					*parentMinKey = maxKey;
				}
			}
			// else if we're a right child page
			else if (parentMaxKey)
				if (minKey <= *parentMaxKey)
					CheckAppendMsg(check, msg, "Rowid %lld out of order (min less than parent max of %lld)", minKey, *parentMaxKey);
		}

		// Check for complete coverage of the page
		uint8 *data = page->Data;
		uint hdr = page->HdrOffset;
		uint8 *hit = (uint8 *)PCache::PageAlloc(bt->PageSize);
		if (hit == nullptr)
			check->MallocFailed = true;
		else
		{
			uint contentOffset = ConvertEx::Get2nz(&data[hdr + 5]);
			_assert(contentOffset <= usableSize); // Enforced by BtreeInitPage()
			_memset(hit + contentOffset, 0, usableSize - contentOffset);
			_memset(hit, 1, contentOffset);
			uint cells = ConvertEx::Get2(&data[hdr + 3]);
			uint cellStart = hdr + 12 - 4 * page->Leaf;
			for (i = 0; i < cells; i++)
			{
				uint32 sizeCell = 65536U;
				uint pc = ConvertEx::Get2(&data[cellStart + i * 2]);
				if (pc <= usableSize - 4)
					sizeCell = CellSizePtr(page, &data[pc]);
				if ((int)(pc + sizeCell - 1) >= usableSize)
					CheckAppendMsg(check, nullptr, "Corruption detected in cell %d on page %d", i, pageID);
				else
					for (int j = pc + sizeCell - 1; j >= pc; j--) hit[j]++;
			}
			i = ConvertEx::Get2(&data[hdr + 1]);
			while (i > 0)
			{
				_assert(i <= usableSize - 4); // Enforced by BtreeInitPage()
				uint size = ConvertEx::Get2(&data[i + 2]);
				_assert(i + size <= usableSize); // Enforced by BtreeInitPage()
				uint j;
				for (j = i + size - 1; j >= i; j--) hit[j]++;
				j = ConvertEx::Get2(&data[i]);
				_assert(j == 0 || j > i + size); // Enforced by BtreeInitPage()
				_assert(j <= usableSize - 4); // Enforced by BtreeInitPage()
				i = j;
			}
			uint cnt;
			for (i = cnt = 0; i < usableSize; i++)
			{
				if (hit[i] == 0)
					cnt++;
				else if (hit[i] > 1)
				{
					CheckAppendMsg(check, nullptr, "Multiple uses for byte %d of page %d", i, pageID);
					break;
				}
			}
			if (cnt != data[hdr + 7])
				CheckAppendMsg(check, nullptr, "Fragmentation of %d bytes reported as %d on page %d", cnt, data[hdr + 7], pageID);
		}
		PCache::PageFree(hit);
		ReleasePage(page);
		return depth + 1;
	}

	__device__ char *Btree::IntegrityCheck(Pid *roots, int rootsLength, int maxErrors, int *errors)
	{
		BtShared *bt = Bt;
		Enter();
		_assert(InTrans > TRANS_NONE && bt->InTransaction > TRANS_NONE);
		int refs = bt->Pager->get_Refs();
		IntegrityCk check;
		check.Bt = bt;
		check.Pager = bt->Pager;
		check.Pages = BtreePagecount(bt);
		check.MaxErrors = maxErrors;
		check.Errors = 0;
		check.MallocFailed = false;
		*errors = 0;
		if (check.Pages == 0)
		{
			Leave();
			return nullptr;
		}

		check.PgRefs = (uint8 *)_alloc2((check.Pages / 8) + 1, true);
		if (!check.PgRefs)
		{
			*errors = 1;
			Leave();
			return nullptr;
		}
		Pid i = PENDING_BYTE_PAGE(bt);
		if (i <= check.Pages) SetPageReferenced(&check, i);
		char err[100];
		TextBuilder::Init(&check.ErrMsg, err, sizeof(err), 2000);
		check.ErrMsg.AllocType = 2;

		// Check the integrity of the freelist
		CheckList(&check, true, (Pid)ConvertEx::Get4(&bt->Page1->Data[32]), (int)ConvertEx::Get4(&bt->Page1->Data[36]), "Main freelist: ");

		// Check all the tables.
		for (i = 0; (int)i < rootsLength && check.MaxErrors; i++)
		{
			if (roots[i] == 0) continue;
#ifndef OMIT_AUTOVACUUM
			if (bt->AutoVacuum && roots[i] > 1)
				CheckPtrmap(&check, roots[i], PTRMAP_ROOTPAGE, 0, nullptr);
#endif
			CheckTreePage(&check, roots[i], "List of tree roots: ", nullptr, nullptr);
		}

		// Make sure every page in the file is referenced
		for (i = 1; i <= check.Pages && check.MaxErrors; i++)
		{
#ifdef OMIT_AUTOVACUUM
			if (!GetPageReferenced(&check, i))
				CheckAppendMsg(&check, nullptr, "Page %d is never used", i);
#else
			// If the database supports auto-vacuum, make sure no tables contain references to pointer-map pages.
			if (!GetPageReferenced(&check, i) && (PTRMAP_PAGENO(bt, i) != i || !bt->AutoVacuum))
				CheckAppendMsg(&check, nullptr, "Page %d is never used", i);
			if (GetPageReferenced(&check, i) && (PTRMAP_PAGENO(bt, i) == i && bt->AutoVacuum))
				CheckAppendMsg(&check, 0, "Pointer map page %d is referenced", i);
#endif
		}

		// Make sure this analysis did not leave any unref() pages. This is an internal consistency check; an integrity check
		// of the integrity check.
		if (_NEVER(refs != bt->Pager->get_Refs()))
			CheckAppendMsg(&check, nullptr, "Outstanding page count goes from %d to %d during this analysis", refs, bt->Pager->get_Refs());

		// Clean  up and report errors.
		Leave();
		_free(check.PgRefs);
		if (check.MallocFailed)
		{
			check.ErrMsg.Reset();
			*errors = check.Errors + 1;
			return nullptr;
		}
		*errors = check.Errors;
		if (check.Errors == 0) check.ErrMsg.Reset();
		return check.ErrMsg.ToString();
	}
#endif
#pragma endregion

#pragma region Settings2

	__device__ const char *Btree::get_Filename()
	{
		_assert(Bt->Pager != nullptr);
		return Bt->Pager->get_Filename(true);
	}

	__device__ const char *Btree::get_Journalname()
	{
		_assert(Bt->Pager != nullptr);
		return Bt->Pager->get_Journalname();
	}

	__device__ bool Btree::IsInTrans()
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		return (InTrans == TRANS_WRITE);
	}

#ifndef OMIT_WAL
	__device__ RC Btree::Checkpoint(int mode, int *logs, int *checkpoints)
	{
		BtShared *bt = Bt;
		Enter();
		RC rc;
		if (bt->InTransaction != TRANS_NONE)
			rc = RC_LOCKED;
		else
			rc = bt->Pager->Checkpoint(mode, logs, checkpoints);
		Leave();
		return rc;
	}
#endif

	__device__ bool Btree::IsInReadTrans()
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		return (InTrans != TRANS_NONE);
	}

	__device__ bool Btree::IsInBackup()
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		return (Backups != 0);
	}

	__device__ Schema *Btree::Schema(int bytes, void (*free)(void *))
	{
		BtShared *bt = Bt;
		Enter();
		if (!bt->Schema && bytes)
		{
			bt->Schema = (Core::Schema *)_tagalloc2(nullptr, bytes, true);
			bt->FreeSchema = free;
		}
		Leave();
		return bt->Schema;
	}

	__device__ RC Btree::SchemaLocked()
	{
		_assert(MutexEx::Held(Ctx->Mutex));
		Enter();
		RC rc = QuerySharedCacheTableLock(this, MASTER_ROOT, LOCK_READ);
		_assert(rc == RC_OK || rc == RC_LOCKED_SHAREDCACHE);
		Leave();
		return rc;
	}


#ifndef OMIT_SHARED_CACHE
	__device__ RC Btree::LockTable(Pid tableID, bool isWriteLock)
	{
		_assert(InTrans != TRANS_NONE);
		RC rc = RC_OK;
		if (Sharable_)
		{
			LOCK lockType = (isWriteLock ? LOCK_READ : LOCK_WRITE);
			Enter();
			rc = QuerySharedCacheTableLock(this, tableID, lockType);
			if (rc == RC_OK)
				rc = SetSharedCacheTableLock(this, tableID, lockType);
			Leave();
		}
		return rc;
	}
#endif

#ifndef OMIT_INCRBLOB
	__device__ RC Btree::PutData(BtCursor *cur, uint32 offset, uint32 amount, void *z)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		_assert(cur->IsIncrblobHandle);

		RC rc = restoreCursorPosition(cur);
		if (rc != RC_OK)
			return rc;
		_assert(cur->State != CURSOR_REQUIRESEEK);
		if (cur->State != CURSOR_VALID)
			return RC_ABORT;

		// Check some assumptions: 
		//   (a) the cursor is open for writing,
		//   (b) there is a read/write transaction open,
		//   (c) the connection holds a write-lock on the table (if required),
		//   (d) there are no conflicting read-locks, and
		//   (e) the cursor points at a valid row of an intKey table.
		if (!cur->WrFlag)
			return RC_READONLY;
		_assert((cur->Bt->BtsFlags & BTS_READ_ONLY) == 0 && cur->Bt->InTransaction == TRANS_WRITE);
		_assert(HasSharedCacheTableLock(cur->Btree, cur->RootID, false, LOCK_WRITE));
		_assert(!HasReadConflicts(cur->Btree, cur->RootID));
		_assert(cur->Pages[cur->ID]->IntKey);

		return AccessPayload(cur, offset, amount, (unsigned char *)z, 1);
	}

	__device__ void Btree::CacheOverflow(BtCursor *cur)
	{
		_assert(CursorHoldsMutex(cur));
		_assert(MutexEx::Held(cur->Btree->Ctx->Mutex));
		InvalidateOverflowCache(cur);
		cur->IsIncrblobHandle = true;
	}
#endif

	__device__ RC Btree::SetVersion(int version)
	{
		_assert(version == 1 || version == 2);

		// If setting the version fields to 1, do not automatically open the WAL connection, even if the version fields are currently set to 2.
		BtShared *bt = Bt;
		bt->BtsFlags &= ~BTS_NO_WAL;
		if (version == 1) bt->BtsFlags |= BTS_NO_WAL;

		RC rc = BeginTrans(0);
		if (rc == RC_OK)
		{
			uint8 *data = bt->Page1->Data;
			if (data[18] != (uint8)version || data[19] != (uint8)version)
			{
				rc = BeginTrans(2);
				if (rc == RC_OK)
				{
					rc = Pager::Write(bt->Page1->DBPage);
					if (rc == RC_OK)
					{
						data[18] = (uint8)version;
						data[19] = (uint8)version;
					}
				}
			}
		}

		bt->BtsFlags &= ~BTS_NO_WAL;
		return rc;
	}

	__device__ void Btree::CursorHints(BtCursor *cur, unsigned int mask)
	{
		_assert(mask == BTREE_BULKLOAD || mask == 0);
		cur->Hints = (uint8)mask;
	}

#pragma endregion
}