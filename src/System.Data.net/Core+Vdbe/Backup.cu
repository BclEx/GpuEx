#include "Core+Vdbe.cu.h"
#include "..\Core+Btree\BtreeInt.cu.h"

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

namespace Core
{
	__device__ static Btree *FindBtree(Context *errorCtx, Context *ctx, const char *dbName)
	{
		int i = Parse::FindDbName(ctx, dbName);
		if (i == 1)
		{
			RC rc = (RC)0;
			Parse *parse = (Parse *)_stackalloc(errorCtx, sizeof(*parse), true);
			if (!parse)
			{
				Main::Error(errorCtx, RC_NOMEM, "out of memory");
				rc = RC_NOMEM;
			}
			else
			{
				parse->Ctx = ctx;
				if (parse->OpenTempDatabase())
				{
					Main::Error(errorCtx, parse->RC, "%s", parse->ErrMsg);
					rc = RC_ERROR;
				}
				_tagfree(errorCtx, parse->ErrMsg);
				_stackfree(errorCtx, parse);
			}
			if (rc)
				return nullptr;
		}

		if (i < 0)
		{
			Main::Error(errorCtx, RC_ERROR, "unknown database %s", dbName);
			return nullptr;
		}
		return ctx->DBs[i].Bt;
	}


	__device__ static RC SetDestPgsz(Backup *p)
	{
		RC rc = p->Dest->SetPageSize(p->Src->GetPageSize(), -1, false);
		return rc;
	}

	__device__ Backup *Backup::Init(Context *destCtx, const char *destDbName, Context *srcCtx, const char *srcDbName)
	{
		// Lock the source database handle. The destination database handle is not locked in this routine, but it is locked in
		// sqlite3_backup_step(). The user is required to ensure that no other thread accesses the destination handle for the duration
		// of the backup operation.  Any attempt to use the destination database connection while a backup is in progress may cause
		// a malfunction or a deadlock.
		MutexEx::Enter(srcCtx->Mutex);
		MutexEx::Enter(destCtx->Mutex);

		Backup *p;
		if (srcCtx == destCtx)
		{
			Main::Error(destCtx, RC_ERROR, "source and destination must be distinct");
			p = nullptr;
		}
		else
		{
			// Allocate space for a new sqlite3_backup object... EVIDENCE-OF: R-64852-21591 The sqlite3_backup object is created by a
			// call to sqlite3_backup_init() and is destroyed by a call to sqlite3_backup_finish().
			p = (Backup *)_alloc2(sizeof(Backup), true);
			if (!p)
				Main::Error(destCtx, RC_NOMEM, nullptr);
		}

		// If the allocation succeeded, populate the new object.
		if (p)
		{
			p->Src = FindBtree(destCtx, srcCtx, srcDbName);
			p->Dest = FindBtree(destCtx, destCtx, destDbName);
			p->DestCtx = destCtx;
			p->SrcCtx = srcCtx;
			p->NextId = 1;
			p->IsAttached = false;

			if (!p->Src || !p->Dest || SetDestPgsz(p) == RC_NOMEM)
			{
				// One (or both) of the named databases did not exist or an OOM error was hit.  The error has already been written into the
				// pDestDb handle.  All that is left to do here is free the sqlite3_backup structure.
				_free(p);
				p = nullptr;
			}
		}
		if (p)
			p->Src->Backups++;

		MutexEx::Leave(destCtx->Mutex);
		MutexEx::Leave(srcCtx->Mutex);
		return p;
	}

	__device__ static bool IsFatalError(RC rc)
	{
		return (rc != RC_OK && rc != RC_BUSY && _ALWAYS(rc != RC_LOCKED));
	}

	__device__ static RC BackupOnePage(Backup *p, Pid srcPg, const uint8 *srcData, bool update)
	{
		Pager *const destPager = p->Dest->get_Pager();
		const int srcPgsz = p->Src->GetPageSize();
		int destPgsz = p->Dest->GetPageSize();
		const int copy = MIN(srcPgsz, destPgsz);
		const int64 end = (int64)srcPg * (int64)srcPgsz;
		RC rc = RC_OK;

		_assert(p->Src->GetReserveNoMutex() >= 0);
		_assert(p->DestLocked);
		_assert(!IsFatalError(p->RC_));
		_assert(srcPg != PENDING_BYTE_PAGE(p->Src->Bt));
		_assert(srcData);

		// Catch the case where the destination is an in-memory database and the page sizes of the source and destination differ. 
		if (srcPgsz != destPgsz && destPager->get_MemoryDB())
			rc = RC_READONLY;

#ifdef HAS_CODEC
		// Use BtreeGetReserveNoMutex() for the source b-tree, as although it is guaranteed that the shared-mutex is held by this thread, handle
		// p->pSrc may not actually be the owner.  */
		int srcReserve = p->Src->GetReserveNoMutex();
		int destReserve = p->Dest->GetReserve();
		// Backup is not possible if the page size of the destination is changing and a codec is in use.
		if (srcPgsz != destPgsz && destPager->GetCodec() != nullptr)
			rc = RC_READONLY;

		// Backup is not possible if the number of bytes of reserve space differ between source and destination.  If there is a difference, try to
		// fix the destination to agree with the source.  If that is not possible, then the backup cannot proceed.
		if (srcReserve != destReserve)
		{
			uint32 newPgsz = srcPgsz;
			rc = destPager->SetPageSize(&newPgsz, srcReserve);
			if (rc == RC_OK && newPgsz != srcPgsz) rc = RC_READONLY;
		}
#endif
		// This loop runs once for each destination page spanned by the source page. For each iteration, variable iOff is set to the byte offset
		// of the destination page.
		for (int64 off = end-(int64)srcPgsz; rc == RC_OK && off < end; off += destPgsz)
		{
			IPage *destPg = nullptr;
			Pid dest = (Pid)(off / destPgsz)+1;
			if (dest == PENDING_BYTE_PAGE(p->Dest->Bt)) continue;
			if ((rc = destPager->Acquire(dest, &destPg, false)) == RC_OK && (rc = Pager::Write(destPg)) == RC_OK)
			{
				const uint8 *in_ = &srcData[off % srcPgsz];
				uint8 *destData = (uint8 *)Pager::GetData(destPg);
				uint8 *out_ = &destData[off % destPgsz];

				// Copy the data from the source page into the destination page. Then clear the Btree layer MemPage.isInit flag. Both this module
				// and the pager code use this trick (clearing the first byte of the page 'extra' space to invalidate the Btree layers
				// cached parse of the page). MemPage.isInit is marked "MUST BE FIRST" for this purpose.
				_memcpy(out_, in_, copy);
				((uint8 *)Pager::GetExtra(destPg))[0] = 0;
				if (off == 0 && !update)
					ConvertEx::Put4(&out_[28], p->Src->LastPage());
			}
			Pager::Unref(destPg);
		}
		return rc;
	}

	__device__ static RC BackupTruncateFile(VFile *file, int64 size)
	{
		int64 current;
		RC rc = file->get_FileSize(current);
		if (rc == RC_OK && current > size)
			rc = file->Truncate(size);
		return rc;
	}

	__device__ static void AttachBackupObject(Backup *p)
	{
		_assert(p->Src->HoldsMutex());
		Backup **pp = (Backup **)&p->Src->get_Pager()->Backup;
		p->Next = *pp;
		*pp = p;
		p->IsAttached = true;
	}

	__device__ RC Backup::Step(int pages)
	{
		MutexEx::Enter(SrcCtx->Mutex);
		Src->Enter();
		if (DestCtx)
			MutexEx::Enter(DestCtx->Mutex);

		RC rc = RC_;
		if (!IsFatalError(rc))
		{
			Pager *const srcPager = Src->get_Pager(); // Source pager
			Pager *const destPager = Dest->get_Pager(); // Dest pager
			Pid srcPage = 0; // Size of source db in pages
			bool closeTrans = false; // True if src db requires unlocking

			// If the source pager is currently in a write-transaction, return SQLITE_BUSY immediately.
			rc = (DestCtx && Src->Bt->InTransaction == TRANS_WRITE ? RC_BUSY : RC_OK);

			// Lock the destination database, if it is not locked already.
			if (rc == RC_OK && !DestLocked && (rc = Dest->BeginTrans(2)) == RC_OK) 
			{
				DestLocked = true;
				Dest->GetMeta(Btree::META_SCHEMA_VERSION, &DestSchema);
			}

			// If there is no open read-transaction on the source database, open one now. If a transaction is opened here, then it will be closed
			// before this function exits.
			if (rc == RC_OK && !Src->IsInReadTrans())
			{
				rc = Src->BeginTrans(0);
				closeTrans = true;
			}

			// Do not allow backup if the destination database is in WAL mode and the page sizes are different between source and destination
			int pgszSrc = Src->GetPageSize(); // Source page size
			int pgszDest = Dest->GetPageSize(); // Destination page size
			IPager::JOURNALMODE destMode = Dest->get_Pager()->GetJournalMode(); // Destination journal mode
			if (rc == RC_OK && destMode == IPager::JOURNALMODE_WAL && pgszSrc != pgszDest)
				rc = RC_READONLY;

			// Now that there is a read-lock on the source database, query the source pager for the number of pages in the database.
			srcPage = (int)Src->LastPage();
			_assert(srcPage >= 0);
			for (int ii = 0; (pages < 0 || ii < pages) && NextId <= (Pid)srcPage && !rc; ii++)
			{
				const Pid srcPg = NextId; // Source page number
				if (srcPg != PENDING_BYTE_PAGE(Src->Bt))
				{
					IPage *srcPgAsObj; // Source page object
					rc = srcPager->Acquire(srcPg, &srcPgAsObj, false);
					if (rc == RC_OK)
					{
						rc = BackupOnePage(this, srcPg, (const uint8 *)Pager::GetData(srcPgAsObj), false);
						Pager::Unref(srcPgAsObj);
					}
				}
				Next++;
			}
			if (rc == RC_OK)
			{
				Pagecount = srcPage;
				Remaining = srcPage+1 - NextId;
				if (NextId > (Pid)srcPage)
					rc = RC_DONE;
				else if (!IsAttached)
					AttachBackupObject(this);
			}

			// Update the schema version field in the destination database. This is to make sure that the schema-version really does change in
			// the case where the source and destination databases have the same schema version.
			if (rc == RC_DONE)
			{
				if (!srcPage)
				{
					rc = Dest->NewDb();
					srcPage = 1;
				}
				if (rc == RC_OK || rc == RC_DONE)
					rc = Dest->UpdateMeta(Btree::META_SCHEMA_VERSION, DestSchema + 1);
				if (rc == RC_OK)
				{
					if (DestCtx)
						Parse::ResetAllSchemasOfConnection(DestCtx);
					if (destMode == IPager::JOURNALMODE_WAL)
						rc = Dest->SetVersion(2);
				}
				if (rc == RC_OK)
				{
					// Set nDestTruncate to the final number of pages in the destination database. The complication here is that the destination page
					// size may be different to the source page size. 
					//
					// If the source page size is smaller than the destination page size, round up. In this case the call to sqlite3OsTruncate() below will
					// fix the size of the file. However it is important to call sqlite3PagerTruncateImage() here so that any pages in the 
					// destination file that lie beyond the nDestTruncate page mark are journalled by PagerCommitPhaseOne() before they are destroyed
					// by the file truncation.
					_assert(pgszSrc == Src->GetPageSize());
					_assert(pgszDest == Dest->GetPageSize());
					Pid destTruncate;
					if (pgszSrc < pgszDest)
					{
						int ratio = pgszDest/pgszSrc;
						destTruncate = (srcPage+ratio-1)/ratio;
						if (destTruncate == PENDING_BYTE_PAGE(Dest->Bt))
							destTruncate--;
					}
					else
						destTruncate = srcPage * (pgszSrc/pgszDest);
					_assert(destTruncate > 0);

					if (pgszSrc < pgszDest)
					{
						// If the source page-size is smaller than the destination page-size, two extra things may need to happen:
						//
						//   * The destination may need to be truncated, and
						//
						//   * Data stored on the pages immediately following the pending-byte page in the source database may need to be
						//     copied into the destination database.
						const int64 size = (int64)pgszSrc * (int64)srcPage;
						VFile *const file = destPager->get_File();
						_assert(file);
						_assert(destTruncate == 0 || (int64)destTruncate*(int64)pgszDest >= size || (destTruncate == (int)(PENDING_BYTE_PAGE(Dest->Bt)-1) && size >= PENDING_BYTE && size <= PENDING_BYTE+pgszDest));

						// This block ensures that all data required to recreate the original database has been stored in the journal for pDestPager and the
						// journal synced to disk. So at this point we may safely modify the database file in any way, knowing that if a power failure
						// occurs, the original database will be reconstructed from the journal file.
						uint32 dstPage;
						destPager->Pages(&dstPage);
						for (Pid pg = destTruncate; rc == RC_OK && pg <= (Pid)dstPage; pg++)
						{
							if (pg != PENDING_BYTE_PAGE(Dest->Bt))
							{
								IPage *pgAsObj;
								rc = destPager->Acquire(pg, &pgAsObj, false);
								if (rc == RC_OK)
								{
									rc = Pager::Write(pgAsObj);
									Pager::Unref(pgAsObj);
								}
							}
						}
						if (rc == RC_OK)
							rc = destPager->CommitPhaseOne(nullptr, true);

						// Write the extra pages and truncate the database file as required
						int64 end = MIN(PENDING_BYTE + pgszDest, size);
						for (int64 off = PENDING_BYTE+pgszSrc; rc == RC_OK && off < end; off += pgszSrc)
						{
							const Pid srcPg = (Pid)((off/pgszSrc)+1);
							PgHdr *srcPgAsObj = nullptr;
							rc = srcPager->Acquire(srcPg, &srcPgAsObj, false);
							if (rc == RC_OK)
							{
								uint8 *data = (uint8 *)Pager::GetData(srcPgAsObj);
								rc = file->Write(data, pgszSrc, off);
							}
							Pager::Unref(srcPgAsObj);
						}
						if (rc == RC_OK)
							rc = BackupTruncateFile(file, size);

						// Sync the database file to disk.
						if (rc == RC_OK)
							rc = destPager->Sync();
					}
					else
					{
						destPager->TruncateImage(destTruncate);
						rc = destPager->CommitPhaseOne(nullptr, false);
					}

					// Finish committing the transaction to the destination database.
					if (rc == RC_OK && (rc = Dest->CommitPhaseTwo(false)) == RC_OK)
						rc = RC_DONE;
				}
			}

			// If bCloseTrans is true, then this function opened a read transaction on the source database. Close the read transaction here. There is
			// no need to check the return values of the btree methods here, as "committing" a read-only transaction cannot fail.
			if (closeTrans)
			{
				ASSERTONLY(RC rc2);
				ASSERTONLY(rc2 = )Src->CommitPhaseOne(nullptr);
				ASSERTONLY(rc2 |= )Src->CommitPhaseTwo(nullptr);
				_assert(rc2 == RC_OK);
			}

			if (rc == RC_IOERR_NOMEM)
				rc = RC_NOMEM;
			RC_ = rc;
		}
		if (DestCtx)
			MutexEx::Leave(DestCtx->Mutex);
		Src->Leave();
		MutexEx::Leave(SrcCtx->Mutex);
		return rc;
	}

	__device__ RC Backup::Finish(Backup *p)
	{
		// Enter the mutexes
		if (!p) return RC_OK;
		Context *srcCtx = p->SrcCtx; // Source database connection
		MutexEx::Enter(srcCtx->Mutex);
		p->Src->Enter();
		if (p->DestCtx)
			MutexEx::Enter(p->DestCtx->Mutex);

		// Detach this backup from the source pager.
		if (p->DestCtx)
			p->Src->Backups--;
		if (p->IsAttached)
		{
			Backup **pp = (Backup **)&p->Src->get_Pager()->Backup; // Ptr to head of pagers backup list
			while (*pp != p)
				pp = &(*pp)->Next;
			*pp = p->Next;
		}

		// If a transaction is still open on the Btree, roll it back.
		p->Dest->Rollback(RC_OK);

		// Set the error code of the destination database handle.
		RC rc = (p->RC_ == RC_DONE ? RC_OK : p->RC_);
		Main::Error(p->DestCtx, rc, nullptr);

		// Exit the mutexes and free the backup context structure.
		if (p->DestCtx)
			Main::LeaveMutexAndCloseZombie(p->DestCtx);
		p->Src->Leave();
		// EVIDENCE-OF: R-64852-21591 The sqlite3_backup object is created by a call to sqlite3_backup_init() and is destroyed by a call to sqlite3_backup_finish().
		if (p->DestCtx)
			_free(p);
		Main::LeaveMutexAndCloseZombie(srcCtx);
		return rc;
	}

	//__device__ int Backup::Remaining()
	//{
	//	return p->Remaining;
	//}

	//__device__ int Backup::Pagecount()
	//{
	//	return p->Pagecount;
	//}

	__device__ void Backup::Update(Pid page, const uint8 *data)
	{
		for (Backup *p = this; p; p = p->Next)
		{
			_assert(MutexEx::Held(p->Src->Bt->Mutex));
			if (!IsFatalError(p->RC_) && page < p->NextId)
			{
				// The backup process p has already copied page iPage. But now it has been modified by a transaction on the source pager. Copy
				// the new data into the backup.
				_assert(p->DestCtx);
				MutexEx::Enter(p->DestCtx->Mutex);
				RC rc = BackupOnePage(p, page, data, true);
				MutexEx::Leave(p->DestCtx->Mutex);
				_assert(rc != RC_BUSY && rc != RC_LOCKED);
				if (rc != RC_OK ){
					p->RC_ = rc;
				}
			}
		}
	}

	__device__ void Backup::Restart()
	{
		for (Backup *p = this; p; p = p->Next)
		{
			_assert(MutexEx::Held(p->Src->Bt->Mutex));
			p->NextId = 1;
		}
	}

#ifndef OMIT_VACUUM
	__device__ RC Backup::BtreeCopyFile(Btree *to, Btree *from)
	{
		to->Enter();
		from->Enter();

		_assert(to->IsInTrans());
		RC rc;
		VFile *fd = to->get_Pager()->get_File(); // File descriptor for database pTo
		if (fd->Type)
		{
			int64 bytes = from->GetPageSize()*(int64)from->LastPage();
			rc = fd->FileControl(VFile::FCNTL_OVERWRITE, &bytes);
			if (rc == RC_NOTFOUND) rc = RC_OK;
			if (rc) goto copy_finished;
		}

		// Set up an sqlite3_backup object. sqlite3_backup.pDestDb must be set to 0. This is used by the implementations of sqlite3_backup_step()
		// and sqlite3_backup_finish() to detect that they are being called from this function, not directly by the user.
		Backup b;
		_memset(&b, 0, sizeof(b));
		b.SrcCtx = (Context *)from->Ctx;
		b.Src = from;
		b.Dest = to;
		b.NextId = 1;

		// 0x7FFFFFFF is the hard limit for the number of pages in a database file. By passing this as the number of pages to copy to
		// sqlite3_backup_step(), we can guarantee that the copy finishes within a single call (unless an error occurs). The assert() statement
		// checks this assumption - (p->rc) should be set to either SQLITE_DONE or an error code.
		b.Step(0x7FFFFFFF);
		_assert(b.RC_ != RC_OK);
		rc = Finish(&b);
		if (rc == RC_OK)
			to->Bt->BtsFlags &= ~BTS_PAGESIZE_FIXED;
		else
			b.Dest->get_Pager()->ClearCache();

		_assert(!to->IsInTrans());
copy_finished:
		from->Leave();
		to->Leave();
		return rc;
	}
#endif
}
