// pager.c
#include "Core+Pager.cu.h"
#define PAGER Pager::PAGER

namespace Core
{
#if _DEBUG
	__device__ bool PagerTrace = true;
#define PAGERTRACE(X, ...) if (PagerTrace) { _printf("PG: "X, __VA_ARGS__); }
#else
#define PAGERTRACE(X, ...)
#endif
#define PAGERID(p) ((int)(long long)p->File)
#define FILEHANDLEID(fd) ((int)(long long)fd)

#pragma region Struct

	__device__ static const unsigned char _journalMagic[] = { 0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7 };

	// sqliteLimit.h
#define DEFAULT_PAGE_SIZE 1024
#define MAX_DEFAULT_PAGE_SIZE 8192
#define MAX_PAGE_COUNT 1073741823
	// pager.h
#define DEFAULT_JOURNAL_SIZE_LIMIT -1

#ifdef HAS_CODEC
#define CODEC1(p, d, n, x, e) \
	if (p->Codec && p->Codec(p->CodecArg, d, id, x) == nullptr) { e; }
#define CODEC2(t, o, p, d, id, x, e) \
	if (p->Codec == nullptr) { o = (t *)d; } else if ((o = (t *)(p->Codec(p->CodecArg, d, id, x))) == nullptr) { e; }
#else
#define CODEC1(p, d, id, x, e)
#define CODEC2(t, o, p, d, id, x, e) o = (t *)d
#endif

#define MAX_SECTOR_SIZE 0x10000

	struct PagerSavepoint
	{
		int64 Offset;           // Starting offset in main journal
		int64 HdrOffset;        // See above
		Bitvec *InSavepoint;	// Set of pages in this savepoint
		Pid Orig;               // Original number of pages in file
		Pid SubRecords;         // Index of first record in sub-journal
#ifndef OMIT_WAL
		uint32 WalData[WAL_SAVEPOINT_NDATA];  // WAL savepoint context
#else
		uint32 *WalData;
#endif
	};

	enum STAT : char
	{
		STAT_HIT = 0,
		STAT_MISS = 1,
		STAT_WRITE = 2,
	};

#ifdef TEST
	__device__ int _readdb_count = 0;    // Number of full pages read from DB
	__device__ int _writedb_count = 0;   // Number of full pages written to DB
	__device__ int _writej_count = 0;    // Number of pages written to journal
#define PAGER_INCR(v) v++
#else
#define PAGER_INCR(v)
#endif

#define JOURNAL_PG_SZ(pager) ((pager->PageSize) + 8)
#define JOURNAL_HDR_SZ(pager) (pager->SectorSize)
#define MJ_PID(x) ((Pid)((PENDING_BYTE / ((x)->PageSize)) + 1))

#define MAX_PID 2147483647

#ifndef OMIT_WAL
	__device__ static int UseWal(Pager *pager) { return (pager->Wal != nullptr); }
#else
#define UseWal(x) false
#define PagerRollbackWal(x) RC_OK
#define PagerWalFrames(v,w,x,y) RC_OK
#define PagerOpenWalIfPresent(z) RC_OK
#define PagerBeginReadTransaction(z) RC_OK
#endif

#pragma endregion

#pragma region Debug
#if _DEBUG 

	__device__ static int AssertPagerState(Pager *p)
	{
		// State must be valid.
		_assert(p->State == Pager::PAGER_OPEN ||
			p->State == Pager::PAGER_READER ||
			p->State == Pager::PAGER_WRITER_LOCKED ||
			p->State == Pager::PAGER_WRITER_CACHEMOD ||
			p->State == Pager::PAGER_WRITER_DBMOD ||
			p->State == Pager::PAGER_WRITER_FINISHED ||
			p->State == Pager::PAGER_ERROR);

		// Regardless of the current state, a temp-file connection always behaves as if it has an exclusive lock on the database file. It never updates
		// the change-counter field, so the changeCountDone flag is always set.
		_assert(p->TempFile == 0 || p->Lock == VFile::LOCK_EXCLUSIVE);
		_assert(p->TempFile == 0 || p->ChangeCountDone);

		// If the useJournal flag is clear, the journal-mode must be "OFF". And if the journal-mode is "OFF", the journal file must not be open.
		_assert(p->JournalMode == IPager::JOURNALMODE_OFF || p->UseJournal);
		_assert(p->JournalMode != IPager::JOURNALMODE_OFF || !p->JournalFile->Opened);

		// Check that MEMDB implies noSync. And an in-memory journal. Since this means an in-memory pager performs no IO at all, it cannot encounter 
		// either SQLITE_IOERR or SQLITE_FULL during rollback or while finalizing a journal file. (although the in-memory journal implementation may 
		// return SQLITE_IOERR_NOMEM while the journal file is being written). It is therefore not possible for an in-memory pager to enter the ERROR state.
		if (p->MemoryDB)
		{
			_assert(p->NoSync);
			_assert(p->JournalMode == IPager::JOURNALMODE_OFF || p->JournalMode == IPager::JOURNALMODE_JMEMORY);
			_assert(p->State != Pager::PAGER_ERROR && p->State != Pager::PAGER_OPEN);
			_assert(!UseWal(p));
		}

		// If changeCountDone is set, a RESERVED lock or greater must be held on the file.
		_assert(p->ChangeCountDone == 0 || p->Lock >= VFile::LOCK_RESERVED);
		_assert(p->Lock != VFile::LOCK_PENDING);

		switch (p->State)
		{
		case Pager::PAGER_OPEN:
			_assert(!p->MemoryDB);
			_assert(p->ErrorCode == RC_OK);
			_assert(p->PCache->get_Refs() == 0 || p->TempFile);
			break;

		case Pager::PAGER_READER:
			_assert(p->ErrorCode == RC_OK);
			_assert(p->Lock != VFile::LOCK_UNKNOWN);
			_assert(p->Lock >= VFile::LOCK_SHARED);
			break;

		case Pager::PAGER_WRITER_LOCKED:
			_assert(p->Lock != VFile::LOCK_UNKNOWN);
			_assert(p->ErrorCode == RC_OK);
			if (!UseWal(p))
				_assert(p->Lock >= VFile::LOCK_RESERVED);
			_assert(p->DBSize == p->DBOrigSize);
			_assert(p->DBOrigSize == p->DBFileSize);
			_assert(p->DBOrigSize == p->DBHintSize);
			_assert(!p->SetMaster);
			break;

		case Pager::PAGER_WRITER_CACHEMOD:
			_assert(p->Lock != VFile::LOCK_UNKNOWN);
			_assert(p->ErrorCode == RC_OK);
			if (!UseWal(p))
			{
				// It is possible that if journal_mode=wal here that neither the journal file nor the WAL file are open. This happens during
				// a rollback transaction that switches from journal_mode=off to journal_mode=wal.
				_assert(p->Lock >= VFile::LOCK_RESERVED);
				_assert(p->JournalFile->Opened || p->JournalMode == IPager::JOURNALMODE_OFF || p->JournalMode == IPager::JOURNALMODE_WAL);
			}
			_assert(p->DBOrigSize == p->DBFileSize);
			_assert(p->DBOrigSize == p->DBHintSize);
			break;

		case Pager::PAGER_WRITER_DBMOD:
			_assert(p->Lock == VFile::LOCK_EXCLUSIVE);
			_assert(p->ErrorCode == RC_OK);
			_assert(!UseWal(p));
			_assert(p->Lock >= VFile::LOCK_EXCLUSIVE);
			_assert(p->JournalFile->Opened || p->JournalMode == IPager::JOURNALMODE_OFF || p->JournalMode == IPager::JOURNALMODE_WAL);
			_assert(p->DBOrigSize <= p->DBHintSize);
			break;

		case Pager::PAGER_WRITER_FINISHED:
			_assert(p->Lock == VFile::LOCK_EXCLUSIVE);
			_assert(p->ErrorCode == RC_OK);
			_assert(!UseWal(p));
			_assert(p->JournalFile->Opened || p->JournalMode == IPager::JOURNALMODE_OFF || p->JournalMode == IPager::JOURNALMODE_WAL);
			break;

		case Pager::PAGER_ERROR:
			// There must be at least one outstanding reference to the pager if in ERROR state. Otherwise the pager should have already dropped back to OPEN state.
			_assert(p->ErrorCode != RC_OK);
			_assert(p->PCache->get_Refs() > 0);
			break;
		}

		return true;
	}

	__device__ static const char *PrintPagerState(Pager *p)
	{
		__shared__ static const char r[1024];
		__snprintf(r, 1024,
			"Filename:      %s\n"
			"State:         %s errCode=%d\n"
			"Lock:          %s\n"
			"Locking mode:  locking_mode=%s\n"
			"Journal mode:  journal_mode=%s\n"
			"Backing store: tempFile=%d memDb=%d useJournal=%d\n"
			"Journal:       journalOff=%lld journalHdr=%lld\n"
			"Size:          dbsize=%d dbOrigSize=%d dbFileSize=%d\n"
			"Journal:       journalOff=%lld journalHdr=%lld\n"
			"Size:          dbsize=%d dbOrigSize=%d dbFileSize=%d\n"
			, p->Filename
			, (p->State == Pager::PAGER_OPEN ? "OPEN" :
			p->State == Pager::PAGER_READER ? "READER" :
			p->State == Pager::PAGER_WRITER_LOCKED ? "WRITER_LOCKED" :
			p->State == Pager::PAGER_WRITER_CACHEMOD ? "WRITER_CACHEMOD" :
			p->State == Pager::PAGER_WRITER_DBMOD ? "WRITER_DBMOD" :
			p->State == Pager::PAGER_WRITER_FINISHED ? "WRITER_FINISHED" :
			p->State == Pager::PAGER_ERROR ? "ERROR" : "?error?")
			, (int)p->ErrorCode
			, (p->Lock == VFile::LOCK_NO ? "NO_LOCK" :
			p->Lock == VFile::LOCK_RESERVED ? "RESERVED" :
			p->Lock == VFile::LOCK_EXCLUSIVE ? "EXCLUSIVE" :
			p->Lock == VFile::LOCK_SHARED ? "SHARED" :
			p->Lock == VFile::LOCK_UNKNOWN ? "UNKNOWN" : "?error?")
			, (p->ExclusiveMode ? "exclusive" : "normal")
			, (p->JournalMode == IPager::JOURNALMODE_JMEMORY ? "memory" :
			p->JournalMode == IPager::JOURNALMODE_OFF ? "off" :
			p->JournalMode == IPager::JOURNALMODE_DELETE ? "delete" :
			p->JournalMode == IPager::JOURNALMODE_PERSIST ? "persist" :
			p->JournalMode == IPager::JOURNALMODE_TRUNCATE ? "truncate" :
			p->JournalMode == IPager::JOURNALMODE_WAL ? "wal" : "?error?")
			, (int)p->TempFile, (int)p->MemoryDB, (int)p->UseJournal
			, p->JournalOffset, p->JournalHeader
			, (int)p->DBSize, (int)p->DBOrigSize, (int)p->DBFileSize);
		return r;
	}

#endif
#pragma endregion

#pragma region Name1

	__device__ static bool SubjRequiresPage(PgHdr *pg)
	{
		Pid id = pg->ID;
		Pager *pager = pg->Pager;
		for (int i = 0; i < pager->Savepoints.length; i++)
		{
			PagerSavepoint *p = &pager->Savepoints[i];
			if (p->Orig >= id && !p->InSavepoint->Get(id))
				return true;
		}
		return false;
	}

	__device__ static bool PageInJournal(PgHdr *pg)
	{
		return pg->Pager->InJournal->Get(pg->ID);
	}

	__device__ static RC PagerUnlockDb(Pager *pager, VFile::LOCK lock)
	{
		_assert(!pager->ExclusiveMode || pager->Lock == lock);
		_assert(lock == VFile::LOCK_NO || lock == VFile::LOCK_SHARED);
		_assert(lock != VFile::LOCK_NO || !UseWal(pager));
		RC rc = RC_OK;
		if (pager->File->Opened)
		{
			_assert(pager->Lock >= lock);
			rc = pager->File->Unlock(lock);
			if (pager->Lock != VFile::LOCK_UNKNOWN)
				pager->Lock = lock;
			SysEx_IOTRACE("UNLOCK %p %d\n", pager, lock);
		}
		return rc;
	}

	__device__ static RC PagerLockDb(Pager *pager, VFile::LOCK lock)
	{
		_assert(lock == VFile::LOCK_SHARED || lock == VFile::LOCK_RESERVED || lock == VFile::LOCK_EXCLUSIVE);
		RC rc = RC_OK;
		if (pager->Lock < lock || pager->Lock == VFile::LOCK_UNKNOWN)
		{
			rc = pager->File->Lock(lock);
			if (rc == RC_OK && (pager->Lock != VFile::LOCK_UNKNOWN || lock == VFile::LOCK_EXCLUSIVE))
			{
				pager->Lock = lock;
				SysEx_IOTRACE("LOCK %p %d\n", pager, lock);
			}
		}
		return rc;
	}

#ifdef ENABLE_ATOMIC_WRITE
	__device__ static int JrnlBufferSize(Pager *pager)
	{
		_assert(!pager->MemoryDB);
		if (!pager->TempFile)
		{
			_assert(pager->File->Opened);
			int dc = pager->File->get_DeviceCharacteristics();
			int sectorSize = pager->SectorSize;
			int sizePage = pager->PageSize;
			_assert(VFile::IOCAP_ATOMIC512 == (512 >> 8));
			_assert(VFile::IOCAP_ATOMIC64K == (65536 >> 8));
			if (!(dc & (VFile::IOCAP_ATOMIC | (sizePage >> 8)) || sectorSize > sizePage))
				return 0;
		}
		return JOURNAL_HDR_SZ(pager) + JOURNAL_PG_SZ(pager);
	}
#endif

#ifdef CHECK_PAGES
	__device__ static uint32 Pager_DataHash(int bytes, unsigned char *data)
	{
		uint32 hash = 0;
		for (int i = 0; i < bytes; i++)
			hash = (hash * 1039) + data[i];
		return hash;
	}
	__device__ static uint32 Pager_PageHash(PgHdr *page) { return Pager_DataHash(page->Pager->PageSize, (unsigned char *)page->Data); }
	__device__ static void Pager_SetPageHash(PgHdr *page) { page->PageHash = Pager_PageHash(page); }
#define CHECK_PAGE(x) CheckPage(x)
	__device__ static void CheckPage(PgHdr *page)
	{
		Pager *pager = page->Pager;
		_assert(pager->State != Pager::PAGER_ERROR);
		_assert((page->Flags & PgHdr::PGHDR_DIRTY) || page->PageHash == Pager_PageHash(page));
	}
#else
#define Pager_DataHash(X, Y) 0
#define Pager_PageHash(X) 0
#define Pager_SetPageHash(X)
#define CHECK_PAGE(x)
#endif

#pragma endregion

#pragma region Journal1

	__device__ static RC ReadMasterJournal(VFile *journalFile, char *master, uint32 masterLength)
	{
		uint32 nameLength; // Length in bytes of master journal name
		int64 fileSize; // Total size in bytes of journal file pJrnl
		uint32 checksum; // MJ checksum value read from journal
		unsigned char magic[8]; // A buffer to hold the magic header
		master[0] = '\0';
		RC rc;
		if ((rc = journalFile->get_FileSize(fileSize)) != RC_OK ||
			fileSize < 16 ||
			(rc = journalFile->Read4(fileSize - 16, &nameLength)) != RC_OK ||
			nameLength >= masterLength ||
			(rc = journalFile->Read4(fileSize - 12, &checksum)) != RC_OK ||
			(rc = journalFile->Read(magic, 8, fileSize - 8)) != RC_OK ||
			_memcmp(magic, _journalMagic, 8) ||
			(rc = journalFile->Read(master, nameLength, fileSize - 16 - nameLength)) != RC_OK)
			return rc;
		// See if the checksum matches the master journal name
		for (uint32 u = 0; u < nameLength; u++)
			checksum -= master[u];
		if (checksum)
		{
			// If the checksum doesn't add up, then one or more of the disk sectors containing the master journal filename is corrupted. This means
			// definitely roll back, so just return SQLITE_OK and report a (nul) master-journal filename.
			nameLength = 0;
		}
		master[nameLength] = '\0';
		return RC_OK;
	}

	__device__ static int64 JournalHdrOffset(Pager *pager)
	{
		int64 offset = 0;
		int64 c = pager->JournalOffset;
		if (c)
			offset = ((c - 1) / JOURNAL_HDR_SZ(pager) + 1) * JOURNAL_HDR_SZ(pager);
		_assert(offset % JOURNAL_HDR_SZ(pager) == 0);
		_assert(offset >= c);
		_assert((offset - c) < JOURNAL_HDR_SZ(pager));
		return offset;
	}

	__device__ static RC ZeroJournalHdr(Pager *pager, bool doTruncate)
	{
		_assert(pager->JournalFile->Opened);
		RC rc = RC_OK;
		if (pager->JournalOffset)
		{
#if __CUDACC__
			const char zeroHeader[28] = { 0 };
#else
			static const char zeroHeader[28] = { 0 };
#endif
			const int64 limit = pager->JournalSizeLimit; // Local cache of jsl
			SysEx_IOTRACE("JZEROHDR %p\n", pager);
			if (doTruncate || limit == 0)
				rc = pager->JournalFile->Truncate(0);
			else
				rc = pager->JournalFile->Write(zeroHeader, sizeof(zeroHeader), 0);
			if (rc == RC_OK && !pager->NoSync)
				rc = pager->JournalFile->Sync(VFile::SYNC_DATAONLY | pager->SyncFlags);
			// At this point the transaction is committed but the write lock is still held on the file. If there is a size limit configured for 
			// the persistent journal and the journal file currently consumes more space than that limit allows for, truncate it now. There is no need
			// to sync the file following this operation.
			if (rc == RC_OK && limit > 0)
			{
				int64 fileSize;
				rc = pager->JournalFile->get_FileSize(fileSize);
				if (rc == RC_OK && fileSize > limit)
					rc = pager->JournalFile->Truncate(limit);
			}
		}
		return rc;
	}

	__device__ static RC WriteJournalHdr(Pager *pager)
	{
		_assert(pager->JournalFile->Opened); 
		unsigned char *header = (unsigned char *)pager->TmpSpace;		// Temporary space used to build header
		uint32 headerSize = (uint32)pager->PageSize;	// Size of buffer pointed to by zHeader
		if (headerSize > JOURNAL_HDR_SZ(pager))
			headerSize = JOURNAL_HDR_SZ(pager);

		// If there are active savepoints and any of them were created since the most recent journal header was written, update the
		// PagerSavepoint.iHdrOffset fields now.
		for (int ii = 0; ii < pager->Savepoints.length; ii++)
			if (pager->Savepoints[ii].HdrOffset == 0)
				pager->Savepoints[ii].HdrOffset = pager->JournalOffset;
		pager->JournalHeader = pager->JournalOffset = JournalHdrOffset(pager);

		// Write the nRec Field - the number of page records that follow this journal header. Normally, zero is written to this value at this time.
		// After the records are added to the journal (and the journal synced, if in full-sync mode), the zero is overwritten with the true number
		// of records (see SyncJournal()).
		//
		// A faster alternative is to write 0xFFFFFFFF to the nRec field. When reading the journal this value tells SQLite to assume that the
		// rest of the journal file contains valid page records. This assumption is dangerous, as if a failure occurred whilst writing to the journal
		// file it may contain some garbage data. There are two scenarios where this risk can be ignored:
		//   * When the pager is in no-sync mode. Corruption can follow a power failure in this case anyway.
		//   * When the SQLITE_IOCAP_SAFE_APPEND flag is set. This guarantees that garbage data is never appended to the journal file.
		_assert(pager->File->Opened || pager->NoSync);
		if (pager->NoSync || (pager->JournalMode == IPager::JOURNALMODE_JMEMORY) || (pager->File->get_DeviceCharacteristics() & VFile::IOCAP_SAFE_APPEND) != 0)
		{
			_memcpy(header, _journalMagic, sizeof(_journalMagic));
			ConvertEx::Put4(&header[sizeof(header)], 0xffffffff);
		}
		else
			_memset(header, 0, sizeof(_journalMagic) + 4);
		SysEx::PutRandom(sizeof(pager->ChecksumInit), &pager->ChecksumInit);
		ConvertEx::Put4(&header[sizeof(_journalMagic) + 4], pager->ChecksumInit);	// The random check-hash initializer
		ConvertEx::Put4(&header[sizeof(_journalMagic) + 8], pager->DBOrigSize);		// The initial database size
		ConvertEx::Put4(&header[sizeof(_journalMagic) + 12], pager->SectorSize);	// The assumed sector size for this process
		ConvertEx::Put4(&header[sizeof(_journalMagic) + 16], pager->PageSize);		// The page size
		// Initializing the tail of the buffer is not necessary.  Everything works find if the following memset() is omitted.  But initializing
		// the memory prevents valgrind from complaining, so we are willing to take the performance hit.
		_memset(&header[sizeof(_journalMagic) + 20], 0, headerSize - (sizeof(_journalMagic) + 20));

		// In theory, it is only necessary to write the 28 bytes that the journal header consumes to the journal file here. Then increment the 
		// Pager.journalOff variable by JOURNAL_HDR_SZ so that the next record is written to the following sector (leaving a gap in the file
		// that will be implicitly filled in by the OS).
		//
		// However it has been discovered that on some systems this pattern can be significantly slower than contiguously writing data to the file,
		// even if that means explicitly writing data to the block of (JOURNAL_HDR_SZ - 28) bytes that will not be used. So that is what is done. 
		//
		// The loop is required here in case the sector-size is larger than the database page size. Since the zHeader buffer is only Pager.pageSize
		// bytes in size, more than one call to sqlite3OsWrite() may be required to populate the entire journal header sector. 
		RC rc = RC_OK;
		for (uint32 headerWritten = 0; rc == RC_OK && headerWritten < JOURNAL_HDR_SZ(pager); headerWritten += headerSize)
		{
			SysEx_IOTRACE("JHDR %p %lld %d\n", pager, pager->JournalHeader, headerSize);
			rc = pager->JournalFile->Write(header, headerSize, pager->JournalOffset);
			_assert(pager->JournalHeader <= pager->JournalOffset);
			pager->JournalOffset += headerSize;
		}
		return rc;
	}

	__device__ static RC ReadJournalHdr(Pager *pager, bool isHot, int64 journalSize, uint32 *recordsOut, uint32 *dbSizeOut)
	{
		_assert(pager->JournalFile->Opened);

		// Advance Pager.journalOff to the start of the next sector. If the journal file is too small for there to be a header stored at this
		// point, return SQLITE_DONE.
		pager->JournalOffset = JournalHdrOffset(pager);
		if (pager->JournalOffset + JOURNAL_HDR_SZ(pager) > journalSize)
			return RC_DONE;
		int64 headerOffset = pager->JournalOffset;

		// Read in the first 8 bytes of the journal header. If they do not match the  magic string found at the start of each journal header, return
		// SQLITE_DONE. If an IO error occurs, return an error code. Otherwise, proceed.
		RC rc;
		unsigned char magic[8];
		if (isHot || headerOffset != pager->JournalHeader)
		{
			rc = pager->JournalFile->Read(magic, sizeof(magic), headerOffset);
			if (rc)
				return rc;
			if (_memcmp(magic, _journalMagic, sizeof(magic)) != 0)
				return RC_DONE;
		}

		// Read the first three 32-bit fields of the journal header: The nRec field, the checksum-initializer and the database size at the start
		// of the transaction. Return an error code if anything goes wrong.
		if ((rc = pager->JournalFile->Read4(headerOffset + 8, recordsOut)) != RC_OK ||
			(rc = pager->JournalFile->Read4(headerOffset + 12, &pager->ChecksumInit)) != RC_OK ||
			(rc = pager->JournalFile->Read4(headerOffset + 16, dbSizeOut)) != RC_OK)
			return rc;

		if (pager->JournalOffset == 0)
		{
			uint32 pageSize; // Page-size field of journal header
			uint32 sectorSize; // Sector-size field of journal header
			// Read the page-size and sector-size journal header fields.
			if ((rc = pager->JournalFile->Read4(headerOffset + 20, &sectorSize)) != RC_OK ||
				(rc = pager->JournalFile->Read4(headerOffset + 24, &pageSize)) != RC_OK)
				return rc;

			// Versions of SQLite prior to 3.5.8 set the page-size field of the journal header to zero. In this case, assume that the Pager.pageSize
			// variable is already set to the correct page size.
			if (pageSize == 0)
				pageSize = pager->PageSize;

			// Check that the values read from the page-size and sector-size fields are within range. To be 'in range', both values need to be a power
			// of two greater than or equal to 512 or 32, and not greater than their respective compile time maximum limits.
			if (pageSize < 512 || sectorSize < 32 ||
				pageSize > MAX_PAGE_SIZE || sectorSize > MAX_SECTOR_SIZE ||
				((pageSize - 1) & pageSize) != 0 || ((sectorSize - 1) & sectorSize) != 0)
				// If the either the page-size or sector-size in the journal-header is invalid, then the process that wrote the journal-header must have 
				// crashed before the header was synced. In this case stop reading the journal file here.
				return RC_DONE;

			// Update the page-size to match the value read from the journal. Use a testcase() macro to make sure that malloc failure within PagerSetPagesize() is tested.
			rc = pager->SetPageSize(&pageSize, -1);
			ASSERTCOVERAGE(rc != RC_OK);

			// Update the assumed sector-size to match the value used by the process that created this journal. If this journal was
			// created by a process other than this one, then this routine is being called from within Pager_Playback(). The local value
			// of Pager.sectorSize is restored at the end of that routine.
			pager->SectorSize = sectorSize;
		}

		pager->JournalOffset += JOURNAL_HDR_SZ(pager);
		return rc;
	}

	__device__ static RC WriteMasterJournal(Pager *pager, const char *master)
	{
		_assert(!pager->SetMaster);
		_assert(!UseWal(pager));

		if (!master ||
			pager->JournalMode == IPager::JOURNALMODE_JMEMORY ||
			pager->JournalMode == IPager::JOURNALMODE_OFF)
			return RC_OK;
		pager->SetMaster = true;
		_assert(pager->JournalFile->Opened);
		_assert(pager->JournalHeader <= pager->JournalOffset);

		// Calculate the length in bytes and the checksum of zMaster
		uint32 checksum = 0; // Checksum of string zMaster
		int masterLength; // Length of string zMaster
		for (masterLength = 0; master[masterLength]; masterLength++)
			checksum += master[masterLength];

		// If in full-sync mode, advance to the next disk sector before writing the master journal name. This is in case the previous page written to
		// the journal has already been synced.
		if (pager->FullSync)
			pager->JournalOffset = JournalHdrOffset(pager);
		int64 headerOffset = pager->JournalOffset; // Offset of header in journal file

		// Write the master journal data to the end of the journal file. If an error occurs, return the error code to the caller.
		RC rc;
		if ((rc = pager->JournalFile->Write4(headerOffset, MJ_PID(pager))) != RC_OK ||
			(rc = pager->JournalFile->Write(master, masterLength, headerOffset+4)) != RC_OK ||
			(rc = pager->JournalFile->Write4(headerOffset + 4 + masterLength, masterLength)) != RC_OK ||
			(rc = pager->JournalFile->Write4(headerOffset + 4 + masterLength + 4, checksum)) != RC_OK ||
			(rc = pager->JournalFile->Write(_journalMagic, 8, headerOffset + 4 + masterLength + 8)) != RC_OK)
			return rc;
		pager->JournalOffset += (masterLength + 20);

		// If the pager is in peristent-journal mode, then the physical journal-file may extend past the end of the master-journal name
		// and 8 bytes of magic data just written to the file. This is dangerous because the code to rollback a hot-journal file
		// will not be able to find the master-journal name to determine whether or not the journal is hot. 
		//
		// Easiest thing to do in this scenario is to truncate the journal file to the required size.
		int64 journalSize;	// Size of journal file on disk
		if ((rc = pager->JournalFile->get_FileSize(journalSize)) == RC_OK && journalSize > pager->JournalOffset)
			rc = pager->JournalFile->Truncate(pager->JournalOffset);
		return rc;
	}

#pragma endregion

#pragma region Name2

	__device__ static PgHdr *Pager_Lookup(Pager *pager, Pid id)
	{
		// It is not possible for a call to PcacheFetch() with createFlag==0 to fail, since no attempt to allocate dynamic memory will be made.
		PgHdr *p;
		pager->PCache->Fetch(id, 0, &p);
		return p;
	}

	__device__ static void Pager_Reset(Pager *pager)
	{
		if (pager->Backup != nullptr)
			pager->Backup->Restart();
		pager->PCache->Clear();
	}

	__device__ static void ReleaseAllSavepoints(Pager *pager)
	{
		for (int ii = 0; ii < pager->Savepoints.length; ii++)
			Bitvec::Destroy(pager->Savepoints[ii].InSavepoint);
		if (!pager->ExclusiveMode || VFile::HasMemoryVFile(pager->SubJournalFile))
			pager->SubJournalFile->Close();
		_free(pager->Savepoints);
		pager->Savepoints = nullptr;
		pager->Savepoints.length = 0;
		pager->SubRecords = 0;
	}

	__device__ static RC AddToSavepointBitvecs(Pager *pager, Pid id)
	{
		RC rc = RC_OK;
		for (int ii = 0; ii < pager->Savepoints.length; ii++)
		{
			PagerSavepoint *p = &pager->Savepoints[ii];
			if (id <= p->Orig)
			{
				rc |= p->InSavepoint->Set(id);
				ASSERTCOVERAGE(rc == RC_NOMEM);
				_assert(rc == RC_OK || rc == RC_NOMEM);
			}
		}
		return rc;
	}

	__device__ static void Pager_Unlock(Pager *pager)
	{
		_assert(pager->State == Pager::PAGER_READER ||
			pager->State == Pager::PAGER_OPEN ||
			pager->State == Pager::PAGER_ERROR);

		Bitvec::Destroy(pager->InJournal);
		pager->InJournal = nullptr;
		ReleaseAllSavepoints(pager);

		if (UseWal(pager))
		{
			_assert(!pager->JournalFile->Opened);
			pager->Wal->EndReadTransaction();
			pager->State = Pager::PAGER_OPEN;
		}
		else if (!pager->ExclusiveMode)
		{
			// If the operating system support deletion of open files, then close the journal file when dropping the database lock.  Otherwise
			// another connection with journal_mode=delete might delete the file out from under us.
			_assert((IPager::JOURNALMODE_JMEMORY & 5) != 1);
			_assert((IPager::JOURNALMODE_OFF & 5) != 1);
			_assert((IPager::JOURNALMODE_WAL & 5) != 1);
			_assert((IPager::JOURNALMODE_DELETE & 5) != 1);
			_assert((IPager::JOURNALMODE_TRUNCATE & 5) == 1);
			_assert((IPager::JOURNALMODE_PERSIST & 5) == 1);
			int dc = (pager->File->Opened ? pager->File->get_DeviceCharacteristics() : 0);
			if ((dc & VFile::IOCAP_UNDELETABLE_WHEN_OPEN) == 0 || (pager->JournalMode & 5) != 1)
				pager->JournalFile->Close();

			// If the pager is in the ERROR state and the call to unlock the database file fails, set the current lock to UNKNOWN_LOCK. See the comment
			// above the #define for UNKNOWN_LOCK for an explanation of why this is necessary.
			RC rc = PagerUnlockDb(pager, VFile::LOCK_NO);
			if (rc != RC_OK && pager->State == Pager::PAGER_ERROR)
				pager->Lock = VFile::LOCK_UNKNOWN;

			// The pager state may be changed from PAGER_ERROR to PAGER_OPEN here without clearing the error code. This is intentional - the error
			// code is cleared and the cache reset in the block below.
			_assert(pager->ErrorCode || pager->State != Pager::PAGER_ERROR);
			pager->ChangeCountDone = 0;
			pager->State = Pager::PAGER_OPEN;
		}

		// If Pager.errCode is set, the contents of the pager cache cannot be trusted. Now that there are no outstanding references to the pager,
		// it can safely move back to PAGER_OPEN state. This happens in both normal and exclusive-locking mode.
		if (pager->ErrorCode)
		{
			_assert(!pager->MemoryDB);
			Pager_Reset(pager);
			pager->ChangeCountDone = pager->TempFile;
			pager->State = Pager::PAGER_OPEN;
			pager->ErrorCode = RC_OK;
		}

		pager->JournalOffset = 0;
		pager->JournalHeader = 0;
		pager->SetMaster = false;
	}

	__device__ static RC Pager_Error(Pager *pager, RC rc)
	{
		RC rc2 = (RC)(rc & 0xff);
		_assert(rc == RC_OK || !pager->MemoryDB);
		_assert(pager->ErrorCode == RC_FULL ||
			pager->ErrorCode == RC_OK ||
			(pager->ErrorCode & 0xff) == RC_IOERR);
		if (rc2 == RC_FULL || rc2 == RC_IOERR)
		{
			pager->ErrorCode = rc;
			pager->State = Pager::PAGER_ERROR;
		}
		return rc;
	}

#pragma endregion

#pragma region Transaction1

	__device__ static RC Pager_Truncate(Pager *pager, Pid pages);

	__device__ static RC Pager_EndTransaction(Pager *pager, bool hasMaster, bool commit)
	{
		// Do nothing if the pager does not have an open write transaction or at least a RESERVED lock. This function may be called when there
		// is no write-transaction active but a RESERVED or greater lock is held under two circumstances:
		//
		//   1. After a successful hot-journal rollback, it is called with eState==PAGER_NONE and eLock==EXCLUSIVE_LOCK.
		//
		//   2. If a connection with locking_mode=exclusive holding an EXCLUSIVE lock switches back to locking_mode=normal and then executes a
		//      read-transaction, this function is called with eState==PAGER_READER and eLock==EXCLUSIVE_LOCK when the read-transaction is closed.
		_assert(AssertPagerState(pager));
		_assert(pager->State != Pager::PAGER_ERROR);
		if (pager->State < Pager::PAGER_WRITER_LOCKED && pager->Lock < VFile::LOCK_RESERVED)
			return RC_OK;

		ReleaseAllSavepoints(pager);
		_assert(pager->JournalFile->Opened || pager->InJournal == nullptr);
		RC rc = RC_OK;
		if (pager->JournalFile->Opened)
		{
			_assert(!UseWal(pager));

			// Finalize the journal file.
			if (VFile::HasMemoryVFile(pager->JournalFile))
			{
				_assert(pager->JournalMode == IPager::JOURNALMODE_JMEMORY);
				pager->JournalFile->Close();
			}
			else if (pager->JournalMode == IPager::JOURNALMODE_TRUNCATE)
			{
				rc = (pager->JournalOffset == 0 ? RC_OK : pager->JournalFile->Truncate(0));
				pager->JournalOffset = 0;
			}
			else if (pager->JournalMode == IPager::JOURNALMODE_PERSIST || (pager->ExclusiveMode && pager->JournalMode != IPager::JOURNALMODE_WAL))
			{
				rc = ZeroJournalHdr(pager, hasMaster);
				pager->JournalOffset = 0;
			}
			else
			{
				// This branch may be executed with Pager.journalMode==MEMORY if a hot-journal was just rolled back. In this case the journal
				// file should be closed and deleted. If this connection writes to the database file, it will do so using an in-memory journal.
				bool delete_ = (!pager->TempFile && VFile::HasJournalVFile(pager->JournalFile));
				_assert(pager->JournalMode == IPager::JOURNALMODE_DELETE ||
					pager->JournalMode == IPager::JOURNALMODE_JMEMORY ||
					pager->JournalMode == IPager::JOURNALMODE_WAL);
				pager->JournalFile->Close();
				if (delete_)
					pager->Vfs->Delete(pager->Journal, false);
			}
		}

#ifdef CHECK_PAGES
		pager->PCache->IterateDirty(Pager_SetPageHash);
		if (pager->DBSize == 0 && pager->PCache->get_Refs() > 0)
		{
			PgHdr *p = Pager_Lookup(pager, 1);
			if (p)
			{
				p->PageHash = 0;
				Pager::Unref(p);
			}
		}
#endif

		Bitvec::Destroy(pager->InJournal); pager->InJournal = nullptr;
		pager->Records = 0;
		pager->PCache->CleanAll();
		pager->PCache->Truncate(pager->DBSize);

		RC rc2 = RC_OK;
		if (UseWal(pager))
		{
			// Drop the WAL write-lock, if any. Also, if the connection was in locking_mode=exclusive mode but is no longer, drop the EXCLUSIVE 
			// lock held on the database file.
			rc2 = pager->Wal->EndWriteTransaction();
			_assert(rc2 == RC_OK);
		}
		else if (rc == RC_OK && commit && pager->DBFileSize > pager->DBSize)
		{
			// This branch is taken when committing a transaction in rollback-journal mode if the database file on disk is larger than the database image.
			// At this point the journal has been finalized and the transaction successfully committed, but the EXCLUSIVE lock is still held on the
			// file. So it is safe to truncate the database file to its minimum required size.
			_assert(pager->Lock == VFile::LOCK_EXCLUSIVE);
			rc = Pager_Truncate(pager, pager->DBSize);
		}

		if (!pager->ExclusiveMode && (!UseWal(pager) || pager->Wal->ExclusiveMode(0)))
		{
			rc2 = PagerUnlockDb(pager, VFile::LOCK_SHARED);
			pager->ChangeCountDone = 0;
		}
		pager->State = Pager::PAGER_READER;
		pager->SetMaster = false;

		return (rc == RC_OK ? rc2 : rc);
	}

	__device__ static void PagerUnlockAndRollback(Pager *pager)
	{
		if (pager->State != Pager::PAGER_ERROR && pager->State != Pager::PAGER_OPEN)
		{
			_assert(AssertPagerState(pager));
			if (pager->State >= Pager::PAGER_WRITER_LOCKED)
			{
				_benignalloc_begin();
				pager->Rollback();
				_benignalloc_end();
			}
			else if (!pager->ExclusiveMode)
			{
				_assert(pager->State == Pager::PAGER_READER);
				Pager_EndTransaction(pager, false, false);
			}
		}
		Pager_Unlock(pager);
	}

	__device__ static uint32 Pager_Cksum(Pager *pager, const uint8 *data)
	{
		uint32 checksum = pager->ChecksumInit;
		int i = pager->PageSize - 200;
		while (i > 0)
		{
			checksum += data[i];
			i -= 200;
		}
		return checksum;
	}

#ifdef HAS_CODEC
	__device__ static void pagerReportSize(Pager *pager)
	{
		if (pager->CodecSizeChange)
			pager->CodecSizeChange(pager->Codec, pager->PageSize, (int)pager->ReserveBytes);
	}
#else
#define pagerReportSize(X)
#endif

	__device__ static RC Pager_PlaybackOnePage(Pager *pager, int64 *offset, Bitvec *done, bool isMainJournal, bool isSavepoint)
	{
		_assert(isMainJournal || done);			// pDone always used on sub-journals
		_assert(isSavepoint || done == 0);		// pDone never used on non-savepoint

		uint8 *data = (uint8 *)pager->TmpSpace; // Temporary storage for the page
		_assert(data != nullptr); // Temp storage must have already been allocated
		_assert(!UseWal(pager) || (!isMainJournal && isSavepoint));

		// Either the state is greater than PAGER_WRITER_CACHEMOD (a transaction or savepoint rollback done at the request of the caller) or this is
		// a hot-journal rollback. If it is a hot-journal rollback, the pager is in state OPEN and holds an EXCLUSIVE lock. Hot-journal rollback
		// only reads from the main journal, not the sub-journal.
		_assert(pager->State >= Pager::PAGER_WRITER_CACHEMOD || (pager->State == Pager::PAGER_OPEN && pager->Lock == VFile::LOCK_EXCLUSIVE));
		_assert(pager->State >= Pager::PAGER_WRITER_CACHEMOD || isMainJournal);

		// Read the page number and page data from the journal or sub-journal file. Return an error code to the caller if an IO error occurs.
		VFile *journalFile = (isMainJournal ? pager->JournalFile : pager->SubJournalFile); // The file descriptor for the journal file
		Pid id; // The page number of a page in journal
		RC rc = journalFile->Read4(*offset, &id);
		if (rc != RC_OK) return rc;
		rc = journalFile->Read(data, pager->PageSize, (*offset) + 4);
		if (rc != RC_OK) return rc;
		*offset += pager->PageSize + 4 + (isMainJournal ? 4 : 0); //TODO: CHECK THIS

		// Sanity checking on the page.  This is more important that I originally thought.  If a power failure occurs while the journal is being written,
		// it could cause invalid data to be written into the journal.  We need to detect this invalid data (with high probability) and ignore it.
		if (id == 0 || id == MJ_PID(pager))
		{
			_assert(!isSavepoint);
			return RC_DONE;
		}
		if (id > (Pid)pager->DBSize || done->Get(id))
			return RC_OK;
		if (isMainJournal)
		{
			uint32 checksum; // Checksum used for sanity checking
			rc = journalFile->Read4((*offset) - 4, &checksum);
			if (rc) return rc;
			if (!isSavepoint && Pager_Cksum(pager, data) != checksum)
				return RC_DONE;
		}

		// If this page has already been played by before during the current rollback, then don't bother to play it back again.
		if (done && (rc = done->Set(id)) != RC_OK)
			return rc;

		// When playing back page 1, restore the nReserve setting
		if (id == 1 && pager->ReserveBytes != data[20])
		{
			pager->ReserveBytes = data[20];
			pagerReportSize(pager);
		}

		// If the pager is in CACHEMOD state, then there must be a copy of this page in the pager cache. In this case just update the pager cache,
		// not the database file. The page is left marked dirty in this case.
		//
		// An exception to the above rule: If the database is in no-sync mode and a page is moved during an incremental vacuum then the page may
		// not be in the pager cache. Later: if a malloc() or IO error occurs during a Movepage() call, then the page may not be in the cache
		// either. So the condition described in the above paragraph is not assert()able.
		//
		// If in WRITER_DBMOD, WRITER_FINISHED or OPEN state, then we update the pager cache if it exists and the main file. The page is then marked 
		// not dirty. Since this code is only executed in PAGER_OPEN state for a hot-journal rollback, it is guaranteed that the page-cache is empty
		// if the pager is in OPEN state.
		//
		// Ticket #1171:  The statement journal might contain page content that is different from the page content at the start of the transaction.
		// This occurs when a page is changed prior to the start of a statement then changed again within the statement.  When rolling back such a
		// statement we must not write to the original database unless we know for certain that original page contents are synced into the main rollback
		// journal.  Otherwise, a power loss might leave modified data in the database file without an entry in the rollback journal that can
		// restore the database to its original form.  Two conditions must be met before writing to the database files. (1) the database must be
		// locked.  (2) we know that the original page content is fully synced in the main journal either because the page is not in cache or else
		// the page is marked as needSync==0.
		//
		// 2008-04-14:  When attempting to vacuum a corrupt database file, it is possible to fail a statement on a database that does not yet exist.
		// Do not attempt to write if database file has never been opened.
		PgHdr *pg = (UseWal(pager) ? nullptr : Pager_Lookup(pager, id)); // An existing page in the cache
		_assert(pg || !pager->MemoryDB);
		_assert(pager->State != Pager::PAGER_OPEN || pg == 0);
		PAGERTRACE("PLAYBACK %d page %d hash(%08x) %s\n", PAGERID(pager), id, Pager_DataHash(pager->PageSize, data), isMainJournal ? "main-journal" : "sub-journal");
		bool isSynced; // True if journal page is synced
		if (isMainJournal)
			isSynced = pager->NoSync || (*offset <= pager->JournalHeader);
		else
			isSynced = (pg == nullptr || (pg->Flags & PgHdr::PGHDR_NEED_SYNC) == 0);
		if (pager->File->Opened && (pager->State >= Pager::PAGER_WRITER_DBMOD || pager->State == Pager::PAGER_OPEN) && isSynced)
		{
			int64 offset = (id - 1) * (int64)pager->PageSize;
			ASSERTCOVERAGE(!isSavepoint && pg != nullptr && (pg->Flags & PgHdr::PGHDR_NEED_SYNC) != 0);
			_assert(!UseWal(pager));
			rc = pager->File->Write(data, pager->PageSize, offset);
			if (id > pager->DBFileSize)
				pager->DBFileSize = id;
			if (pager->Backup)
			{
				CODEC1(pager, data, id, 3, rc = RC_NOMEM);
				pager->Backup->Update(id, data);
				CODEC2(uint8, data, pager, data, id, 7, rc = RC_NOMEM);
			}
		}
		else if (!isMainJournal && pg == nullptr)
		{
			// If this is a rollback of a savepoint and data was not written to the database and the page is not in-memory, there is a potential
			// problem. When the page is next fetched by the b-tree layer, it will be read from the database file, which may or may not be 
			// current. 
			//
			// There are a couple of different ways this can happen. All are quite obscure. When running in synchronous mode, this can only happen 
			// if the page is on the free-list at the start of the transaction, then populated, then moved using sqlite3PagerMovepage().
			//
			// The solution is to add an in-memory page to the cache containing the data just read from the sub-journal. Mark the page as dirty 
			// and if the pager requires a journal-sync, then mark the page as requiring a journal-sync before it is written.
			_assert(isSavepoint);
			_assert(pager->DoNotSpill == 0);
			pager->DoNotSpill++;
			rc = pager->Acquire(id, &pg, true);
			_assert(pager->DoNotSpill == 1);
			pager->DoNotSpill--;
			if (rc != RC_OK) return rc;
			pg->Flags &= ~PgHdr::PGHDR_NEED_READ;
			PCache::MakeDirty(pg);
		}
		if (pg)
		{
			// No page should ever be explicitly rolled back that is in use, except for page 1 which is held in use in order to keep the lock on the
			// database active. However such a page may be rolled back as a result of an internal error resulting in an automatic call to
			// sqlite3PagerRollback().
			uint8 *pageData = (uint8 *)pg->Data;
			_memcpy(pageData, data, pager->PageSize);
			pager->Reiniter(pg);
			if (isMainJournal && (!isSavepoint || *offset <= pager->JournalHeader))
			{
				// If the contents of this page were just restored from the main journal file, then its content must be as they were when the 
				// transaction was first opened. In this case we can mark the page as clean, since there will be no need to write it out to the
				// database.
				//
				// There is one exception to this rule. If the page is being rolled back as part of a savepoint (or statement) rollback from an 
				// unsynced portion of the main journal file, then it is not safe to mark the page as clean. This is because marking the page as
				// clean will clear the PGHDR_NEED_SYNC flag. Since the page is already in the journal file (recorded in Pager.pInJournal) and
				// the PGHDR_NEED_SYNC flag is cleared, if the page is written to again within this transaction, it will be marked as dirty but
				// the PGHDR_NEED_SYNC flag will not be set. It could then potentially be written out into the database file before its journal file
				// segment is synced. If a crash occurs during or following this, database corruption may ensue.
				_assert(!UseWal(pager));
				PCache::MakeClean(pg);
			}
			Pager_SetPageHash(pg);

			// If this was page 1, then restore the value of Pager.dbFileVers. Do this before any decoding.
			if (id == 1)
				_memcpy((char *)&pager->DBFileVersion, &((char *)pageData)[24], sizeof(pager->DBFileVersion));

			// Decode the page just read from disk
			CODEC1(pager, pageData, pg->ID, 3, rc = RC_NOMEM);
			PCache::Release(pg);
		}
		return rc;
	}

	__device__ static RC Pager_Delmaster(Pager *pager, const char *master)
	{
		VSystem *vfs = pager->Vfs;

		// Allocate space for both the pJournal and pMaster file descriptors. If successful, open the master journal file for reading.         
		VFile *masterFile = (VFile *)_alloc2(vfs->SizeOsFile * 2, true); // Malloc'd master-journal file descriptor
		VFile *journalFile = (VFile *)(((uint8 *)masterFile) + vfs->SizeOsFile); // Malloc'd child-journal file descriptor
		RC rc;
		if (!masterFile)
			rc = RC_NOMEM;
		else
			rc = vfs->Open(master, masterFile, (VSystem::OPEN)(VSystem::OPEN_READONLY | VSystem::OPEN_MASTER_JOURNAL), 0);
		int masterPtrSize;
		char *masterJournal;
		char *masterPtr;
		char *journal;
		if (rc != RC_OK) goto delmaster_out;

		// Load the entire master journal file into space obtained from sqlite3_malloc() and pointed to by zMasterJournal.   Also obtain
		// sufficient space (in zMasterPtr) to hold the names of master journal files extracted from regular rollback-journals.
		int64 masterJournalSize; // Size of master journal file
		rc = masterFile->get_FileSize(masterJournalSize);
		if (rc != RC_OK) goto delmaster_out;
		masterPtrSize = vfs->MaxPathname + 1; // Amount of space allocated to zMasterPtr[]
		masterJournal = (char *)_alloc((int)masterJournalSize + masterPtrSize + 1); // Contents of master journal file
		if (!masterJournal)
		{
			rc = RC_NOMEM;
			goto delmaster_out;
		}
		masterPtr = &masterJournal[masterJournalSize + 1]; // Space to hold MJ filename from a journal file
		rc = masterFile->Read(masterJournal, (int)masterJournalSize, 0);
		if (rc != RC_OK) goto delmaster_out;
		masterJournal[masterJournalSize] = 0;

		journal = masterJournal; // Pointer to one journal within MJ file
		while ((journal - masterJournal) < masterJournalSize)
		{
			int exists;
			rc = vfs->Access(journal, VSystem::ACCESS_EXISTS, &exists);
			if (rc != RC_OK)
				goto delmaster_out;
			if (exists)
			{
				// One of the journals pointed to by the master journal exists. Open it and check if it points at the master journal. If
				// so, return without deleting the master journal file.
				rc = vfs->Open(journal, journalFile, (VSystem::OPEN)(VSystem::OPEN_READONLY | VSystem::OPEN_MAIN_JOURNAL), 0);
				if (rc != RC_OK)
					goto delmaster_out;

				rc = ReadMasterJournal(journalFile, masterPtr, masterPtrSize);
				journalFile->Close();
				if (rc != RC_OK)
					goto delmaster_out;

				int c = masterPtr[0] != 0 && _strcmp(masterPtr, master) == 0;
				if (c) // We have a match. Do not delete the master journal file.
					goto delmaster_out;
			}
			journal += (_strlen30(journal) + 1);
		}

		masterFile->Close();
		rc = vfs->Delete(master, false);

delmaster_out:
		_free(masterJournal);
		if (masterFile != nullptr)
		{
			masterFile->Close();
			_assert(!journalFile->Opened);
			_free(journalFile);
		}
		return rc;
	}

	__device__ static RC Pager_Truncate(Pager *pager, Pid pages)
	{
		_assert(pager->State != Pager::PAGER_ERROR);
		_assert(pager->State != Pager::PAGER_READER);

		RC rc = RC_OK;
		if (pager->File->Opened && (pager->State >= Pager::PAGER_WRITER_DBMOD || pager->State == Pager::PAGER_OPEN))
		{
			int sizePage = pager->PageSize;
			_assert(pager->Lock == VFile::LOCK_EXCLUSIVE);
			// TODO: Is it safe to use Pager.dbFileSize here?
			int64 currentSize;
			rc = pager->File->get_FileSize(currentSize);
			int64 newSize = sizePage * (int64)pages;
			if (rc == RC_OK && currentSize != newSize)
			{
				if (currentSize > newSize)
					rc = pager->File->Truncate(newSize);
				else if ((currentSize + sizePage) <= newSize)
				{
					uint8 *tmp = (uint8 *)pager->TmpSpace;
					_memset(tmp, 0, sizePage);
					ASSERTCOVERAGE((newSize - sizePage) == currentSize);
					ASSERTCOVERAGE((newSize - sizePage) > currentSize);
					rc = pager->File->Write(tmp, sizePage, newSize - sizePage);
				}
				if (rc == RC_OK)
					pager->DBFileSize = pages;
			}
		}
		return rc;
	}

#pragma endregion

#pragma region Transaction2

	__device__ int Pager::get_SectorSize(VFile *file)
	{
		int ret = file->get_SectorSize();
		if (ret < 32)
			ret = 512;
		else if (ret > MAX_SECTOR_SIZE)
		{
			_assert(MAX_SECTOR_SIZE >= 512);
			ret = MAX_SECTOR_SIZE;
		}
		return ret;
	}

	__device__ static void set_SectorSize(Pager *pager)
	{
		_assert(pager->File->Opened || pager->TempFile);
		if (pager->TempFile || (pager->File->get_DeviceCharacteristics() & VFile::IOCAP_POWERSAFE_OVERWRITE) != 0)
			pager->SectorSize = 512; // Sector size doesn't matter for temporary files. Also, the file may not have been opened yet, in which case the OsSectorSize() call will segfault.
		else
			pager->SectorSize = pager->File->get_SectorSize();
	}

	__device__ static RC Pager_Playback(Pager *pager, bool isHot)
	{
		int res = 1;
		char *master = nullptr;

		// Figure out how many records are in the journal.  Abort early if the journal is empty.
		_assert(pager->JournalFile->Opened);
		int64 sizeJournal; // Size of the journal file in bytes
		RC rc = pager->JournalFile->get_FileSize(sizeJournal);
		VSystem *vfs;
		bool needPagerReset;
		if (rc != RC_OK)
			goto end_playback;

		// Read the master journal name from the journal, if it is present. If a master journal file name is specified, but the file is not
		// present on disk, then the journal is not hot and does not need to be played back.
		//
		// TODO: Technically the following is an error because it assumes that buffer Pager.pTmpSpace is (mxPathname+1) bytes or larger. i.e. that
		// (pPager->pageSize >= pPager->pVfs->mxPathname+1). Using os_unix.c, mxPathname is 512, which is the same as the minimum allowable value
		// for pageSize.
		vfs = pager->Vfs;
		master = (char *)pager->TmpSpace; // Name of master journal file if any
		rc = ReadMasterJournal(pager->JournalFile, master, vfs->MaxPathname + 1);
		if (rc == RC_OK && master[0])
			rc = vfs->Access(master, VSystem::ACCESS_EXISTS, &res);
		master = nullptr;
		if (rc != RC_OK || !res)
			goto end_playback;
		pager->JournalOffset = 0;
		needPagerReset = isHot; // True to reset page prior to first page rollback

		// This loop terminates either when a ReadJournalHdr() or Pager_PlaybackOnePage() call returns SQLITE_DONE or an IO error occurs. 
		while (true)
		{
			// Read the next journal header from the journal file.  If there are not enough bytes left in the journal file for a complete header, or
			// it is corrupted, then a process must have failed while writing it. This indicates nothing more needs to be rolled back.
			uint32 records; // Number of Records in the journal
			Pid maxPage = 0; // Size of the original file in pages
			rc = ReadJournalHdr(pager, isHot, sizeJournal, &records, &maxPage);
			if (rc != RC_OK)
			{ 
				if (rc == RC_DONE)
					rc = RC_OK;
				goto end_playback;
			}

			// If nRec is 0xffffffff, then this journal was created by a process working in no-sync mode. This means that the rest of the journal
			// file consists of pages, there are no more journal headers. Compute the value of nRec based on this assumption.
			if (records == 0xffffffff)
			{
				_assert(pager->JournalOffset == JOURNAL_HDR_SZ(pager));
				records = (uint32)((sizeJournal - JOURNAL_HDR_SZ(pager)) / JOURNAL_PG_SZ(pager));
			}

			// If nRec is 0 and this rollback is of a transaction created by this process and if this is the final header in the journal, then it means
			// that this part of the journal was being filled but has not yet been synced to disk.  Compute the number of pages based on the remaining
			// size of the file.
			//
			// The third term of the test was added to fix ticket #2565. When rolling back a hot journal, nRec==0 always means that the next
			// chunk of the journal contains zero pages to be rolled back.  But when doing a ROLLBACK and the nRec==0 chunk is the last chunk in
			// the journal, it means that the journal might contain additional pages that need to be rolled back and that the number of pages 
			// should be computed based on the journal file size.
			if (records == 0 && !isHot && pager->JournalHeader + JOURNAL_HDR_SZ(pager) == pager->JournalOffset)
				records = (uint)((sizeJournal - pager->JournalOffset) / JOURNAL_PG_SZ(pager));

			// If this is the first header read from the journal, truncate the database file back to its original size.
			if (pager->JournalOffset == JOURNAL_HDR_SZ(pager))
			{
				rc = Pager_Truncate(pager, maxPage);
				if (rc != RC_OK)
					goto end_playback;
				pager->DBSize = maxPage;
			}

			// Copy original pages out of the journal and back into the database file and/or page cache.
			for (uint32 u = 0U; u < records; u++)
			{
				if (needPagerReset)
				{
					Pager_Reset(pager);
					needPagerReset = false;
				}
				rc = Pager_PlaybackOnePage(pager, &pager->JournalOffset, nullptr, true, false);
				if (rc != RC_OK)
					if (rc == RC_DONE)
					{
						pager->JournalOffset = sizeJournal;
						break;
					}
					else if (rc == RC_IOERR_SHORT_READ)
					{
						// If the journal has been truncated, simply stop reading and processing the journal. This might happen if the journal was
						// not completely written and synced prior to a crash.  In that case, the database should have never been written in the
						// first place so it is OK to simply abandon the rollback.
						rc = RC_OK;
						goto end_playback;
					}
					else
					{
						// If we are unable to rollback, quit and return the error code.  This will cause the pager to enter the error state
						// so that no further harm will be done.  Perhaps the next process to come along will be able to rollback the database.
						goto end_playback;
					}
			}
		}

end_playback:
		// Following a rollback, the database file should be back in its original state prior to the start of the transaction, so invoke the
		// SQLITE_FCNTL_DB_UNCHANGED file-control method to disable the assertion that the transaction counter was modified.
#ifdef _DEBUG
		pager->File->FileControl(VFile::FCNTL_DB_UNCHANGED, 0);
#endif

		// If this playback is happening automatically as a result of an IO or malloc error that occurred after the change-counter was updated but 
		// before the transaction was committed, then the change-counter modification may just have been reverted. If this happens in exclusive 
		// mode, then subsequent transactions performed by the connection will not update the change-counter at all. This may lead to cache inconsistency
		// problems for other processes at some point in the future. So, just in case this has happened, clear the changeCountDone flag now.
		pager->ChangeCountDone = pager->TempFile;

		if (rc == RC_OK)
		{
			master = (char *)pager->TmpSpace;
			rc = ReadMasterJournal(pager->JournalFile, master, vfs->MaxPathname + 1);
			ASSERTCOVERAGE(rc != RC_OK);
		}
		if (rc == RC_OK && (pager->State >= Pager::PAGER_WRITER_DBMOD || pager->State == Pager::PAGER_OPEN))
			rc = pager->Sync();
		if (rc == RC_OK)
		{
			rc = Pager_EndTransaction(pager, master[0] != '\0', false);
			ASSERTCOVERAGE(rc != RC_OK);
		}
		if (rc == RC_OK && master[0] && res)
		{
			// If there was a master journal and this routine will return success, see if it is possible to delete the master journal.
			rc = Pager_Delmaster(pager, master);
			ASSERTCOVERAGE(rc != RC_OK);
		}

		// The Pager.sectorSize variable may have been updated while rolling back a journal created by a process with a different sector size
		// value. Reset it to the correct value for this process.
		set_SectorSize(pager);
		return rc;
	}

	__device__ static RC ReadDbPage(PgHdr *pg)
	{
		Pager *pager = pg->Pager; // Pager object associated with page pPg

		_assert(pager->State >= Pager::PAGER_READER && !pager->MemoryDB);
		_assert(pager->File->Opened);

		if (_NEVER(!pager->File->Opened))
		{
			_assert(pager->TempFile);
			_memset(pg->Data, 0, pager->PageSize);
			return RC_OK;
		}

		RC rc = RC_OK;
		Pid id = pg->ID; // Page number to read
		bool isInWal = 0; // True if page is in log file
		int pageSize = pager->PageSize; // Number of bytes to read
		if (UseWal(pager)) // Try to pull the page from the write-ahead log.
			rc = pager->Wal->Read(id, &isInWal, pageSize, (uint8 *)pg->Data);
		if (rc == RC_OK && !isInWal)
		{
			int64 offset = (id - 1) * (int64)pager->PageSize;
			rc = pager->File->Read(pg->Data, pageSize, offset);
			if (rc == RC_IOERR_SHORT_READ)
				rc = RC_OK;
		}

		if (id == 1)
		{
			// If the read is unsuccessful, set the dbFileVers[] to something that will never be a valid file version.  dbFileVers[] is a copy
			// of bytes 24..39 of the database.  Bytes 28..31 should always be zero or the size of the database in page. Bytes 32..35 and 35..39
			// should be page numbers which are never 0xffffffff.  So filling pPager->dbFileVers[] with all 0xff bytes should suffice.
			//
			// For an encrypted database, the situation is more complex:  bytes 24..39 of the database are white noise.  But the probability of
			// white noising equaling 16 bytes of 0xff is vanishingly small so we should still be ok.
			if (rc)
				_memset(pager->DBFileVersion, 0xff, sizeof(pager->DBFileVersion));
			else
				_memcpy(pager->DBFileVersion, &((char *)pg->Data)[24], sizeof(pager->DBFileVersion));
		}
		CODEC1(pager, pg->Data, id, 3, rc = RC_NOMEM);

		PAGER_INCR(_readdb_count);
		PAGER_INCR(pager->Reads);
		SysEx_IOTRACE("PGIN %p %d\n", pager, id);
		PAGERTRACE("FETCH %d page %d hash(%08x)\n", PAGERID(pager), id, Pager_PageHash(pg));

		return rc;
	}

	__device__ static void Pager_WriteChangecounter(PgHdr *pg)
	{
		// Increment the value just read and write it back to byte 24.
		uint32 change_counter = ConvertEx::Get4((uint8 *)pg->Pager->DBFileVersion) + 1;
		ConvertEx::Put4(((uint8 *)pg->Data) + 24, change_counter);

		// Also store the SQLite version number in bytes 96..99 and in bytes 92..95 store the change counter for which the version number is valid.
		ConvertEx::Put4(((uint8 *)pg->Data) + 92, change_counter);
		ConvertEx::Put4(((uint8 *)pg->Data) + 96, CORE_VERSION_NUMBER);
	}

#ifndef OMIT_WAL
	__device__ static RC PagerUndoCallback(void *ctx, Pid id)
	{
		RC rc = RC_OK;
		Pager *pager = (Pager *)ctx;
		PgHdr *pg = pager->Lookup(id);
		if (pg)
		{
			if (PCache::get_PageRefs(pg) == 1)
				PCache::Drop(pg);
			else
			{
				rc = ReadDbPage(pg);
				if (rc == RC_OK)
					pager->Reiniter(pg);
				Pager::Unref(pg);
			}
		}

		// Normally, if a transaction is rolled back, any backup processes are updated as data is copied out of the rollback journal and into the
		// database. This is not generally possible with a WAL database, as rollback involves simply truncating the log file. Therefore, if one
		// or more frames have already been written to the log (and therefore also copied into the backup databases) as part of this transaction,
		// the backups must be restarted.
		if (pager->Backup != nullptr)
			pager->Backup->Restart();

		return rc;
	}

	__device__ static RC PagerRollbackWal(Pager *pager)
	{
		// For all pages in the cache that are currently dirty or have already been written (but not committed) to the log file, do one of the following:
		//
		//   + Discard the cached page (if refcount==0), or
		//   + Reload page content from the database (if refcount>0).
		pager->DBSize = pager->DBOrigSize;
		RC rc = pager->Wal->Undo(PagerUndoCallback, (void *)pager);
		PgHdr *list = pager->PCache->DirtyList(); // List of dirty pages to revert
		while (list && rc == RC_OK)
		{
			PgHdr *next = list->Dirty;
			rc = PagerUndoCallback((void *)pager, list->ID);
			list = next;
		}

		return rc;
	}

	__device__ static RC PagerWalFrames(Pager *pager, PgHdr *list, Pid truncate, bool isCommit)
	{
		_assert(pager->Wal != nullptr);
		_assert(list != nullptr);
		PgHdr *p; // For looping over pages
#ifdef _DEBUG
		// Verify that the page list is in accending order
		for (p = list; p && p->Dirty; p = p->Dirty)
			_assert(p->ID < p->Dirty->ID);
#endif
		int listPages; // Number of pages in pList
		_assert(list->Dirty == nullptr || isCommit);
		if (isCommit)
		{
			// If a WAL transaction is being committed, there is no point in writing any pages with page numbers greater than nTruncate into the WAL file.
			// They will never be read by any client. So remove them from the pDirty list here.
			PgHdr **next = &list;
			listPages = 0;
			for (p = list; (*next = p) != nullptr; p = p->Dirty)
				if (p->ID <= truncate)
				{
					next = &p->Dirty;
					listPages++;
				}
				_assert(list != nullptr);
		}
		else
			listPages = 1;
		pager->Stats[STAT_WRITE] += listPages;

		if (list->ID == 1) Pager_WriteChangecounter(list);
		RC rc = pager->Wal->Frames(pager->PageSize, list, truncate, isCommit, pager->WalSyncFlags);
		if (rc == RC_OK && pager->Backup)
			for (p = list; p; p = p->Dirty)
				pager->Backup->Update(p->ID, (uint8 *)p->Data);

#ifdef CHECK_PAGES
		list = pager->PCache->DirtyList();
		for (p = list; p; p = p->Dirty)
			Pager_SetPageHash(p);
#endif

		return rc;
	}

	__device__ static RC PagerBeginReadTransaction(Pager *pager)
	{
		_assert(UseWal(pager));
		_assert(pager->State == Pager::PAGER_OPEN || pager->State == Pager::PAGER_READER);

		// sqlite3WalEndReadTransaction() was not called for the previous transaction in locking_mode=EXCLUSIVE.  So call it now.  If we
		// are in locking_mode=NORMAL and EndRead() was previously called, the duplicate call is harmless.
		pager->Wal->EndReadTransaction();

		int changed = 0; // True if cache must be reset
		RC rc = pager->Wal->BeginReadTransaction(&changed);
		if (rc != RC_OK || changed)
			Pager_Reset(pager);

		return rc;
	}
#endif

	__device__ static RC PagerPagecount(Pager *pager, Pid *pagesRef)
	{
		// Query the WAL sub-system for the database size. The WalDbsize() function returns zero if the WAL is not open (i.e. Pager.pWal==0), or
		// if the database size is not available. The database size is not available from the WAL sub-system if the log file is empty or
		// contains no valid committed transactions.
		_assert(pager->State == Pager::PAGER_OPEN);
		_assert(pager->Lock >= VFile::LOCK_SHARED);
		Pid pages = pager->Wal->DBSize();

		// If the database size was not available from the WAL sub-system, determine it based on the size of the database file. If the size
		// of the database file is not an integer multiple of the page-size, round down to the nearest page. Except, any file larger than 0
		// bytes in size is considered to contain at least one page.
		if (pages == 0)
		{
			_assert(pager->File->Opened || pager->TempFile);
			int64 n = 0; // Size of db file in bytes
			if (pager->File->Opened)
			{
				RC rc = pager->File->get_FileSize(n);
				if (rc != RC_OK)
					return rc;
			}
			pages = (Pid)((n + pager->PageSize - 1) / pager->PageSize);
		}

		// If the current number of pages in the file is greater than the configured maximum pager number, increase the allowed limit so
		// that the file can be read.
		if (pages > pager->MaxPid)
			pager->MaxPid = (Pid)pages;

		*pagesRef = pages;
		return RC_OK;
	}

#ifndef OMIT_WAL
	__device__ static RC PagerOpenWalIfPresent(Pager *pager)
	{
		_assert(pager->State == Pager::PAGER_OPEN);
		_assert(pager->Lock >= VFile::LOCK_SHARED);
		RC rc = RC_OK;

		if (!pager->TempFile)
		{
			Pid pages; // Size of the database file
			rc = PagerPagecount(pager, &pages);
			if (rc) return rc;
			int isWal; // True if WAL file exists
			if (pages == 0)
			{
				rc = pager->Vfs->Delete(pager->WalName, false);
				if (rc == RC_IOERR_DELETE_NOENT) rc = RC_OK;
				isWal = 0;
			}
			else
				rc = pager->Vfs->Access(pager->WalName, VSystem::ACCESS_EXISTS, &isWal);
			if (rc == RC_OK)
			{
				if (isWal)
				{
					ASSERTCOVERAGE(pager->PCache->get_Pages() == 0);
					rc = pager->OpenWal(0);
				}
				else if (pager->JournalMode == IPager::JOURNALMODE_WAL)
					pager->JournalMode = IPager::JOURNALMODE_DELETE;
			}
		}
		return rc;
	}
#endif

	__device__ static RC PagerPlaybackSavepoint(Pager *pager, PagerSavepoint *savepoint)
	{
		_assert(pager->State != Pager::PAGER_ERROR);
		_assert(pager->State >= Pager::PAGER_WRITER_LOCKED);

		// Allocate a bitvec to use to store the set of pages rolled back
		Bitvec *done = nullptr; // Bitvec to ensure pages played back only once
		if (savepoint)
		{
			done = new Bitvec(savepoint->Orig);
			if (!done)
				return RC_NOMEM;
		}

		// Set the database size back to the value it was before the savepoint  being reverted was opened.
		pager->DBSize = (savepoint ? savepoint->Orig : pager->DBOrigSize);
		pager->ChangeCountDone = pager->TempFile;

		if (!savepoint && UseWal(pager))
			return PagerRollbackWal(pager);

		// Use pPager->journalOff as the effective size of the main rollback journal.  The actual file might be larger than this in
		// PAGER_JOURNALMODE_TRUNCATE or PAGER_JOURNALMODE_PERSIST.  But anything past pPager->journalOff is off-limits to us.
		int64 sizeJournal = pager->JournalOffset; // Effective size of the main journal
		_assert(!UseWal(pager) || sizeJournal == 0);

		// Begin by rolling back records from the main journal starting at PagerSavepoint.iOffset and continuing to the next journal header.
		// There might be records in the main journal that have a page number greater than the current database size (pPager->dbSize) but those
		// will be skipped automatically.  Pages are added to pDone as they are played back.
		RC rc = RC_OK;
		if (savepoint && !UseWal(pager))
		{
			int64 hdrOffset = (savepoint->HdrOffset ? savepoint->HdrOffset : sizeJournal); // End of first segment of main-journal records
			pager->JournalOffset = savepoint->Offset;
			while (rc == RC_OK && pager->JournalOffset < hdrOffset)
				rc = Pager_PlaybackOnePage(pager, &pager->JournalOffset, done, true, true);
			_assert(rc != RC_DONE);
		}
		else
			pager->JournalOffset = 0;

		// Continue rolling back records out of the main journal starting at the first journal header seen and continuing until the effective end
		// of the main journal file.  Continue to skip out-of-range pages and continue adding pages rolled back to pDone.
		while (rc == RC_OK && pager->JournalOffset < sizeJournal)
		{
			uint32 records = 0; // Number of Journal Records
			uint32 dummy;
			rc = ReadJournalHdr(pager, false, sizeJournal, &records, &dummy);
			_assert(rc != RC_DONE);

			// The "pPager->journalHdr+JOURNAL_HDR_SZ(pPager)==pPager->journalOff" test is related to ticket #2565.  See the discussion in the
			// Pager_Playback() function for additional information.
			if (records == 0 && pager->JournalHeader + JOURNAL_HDR_SZ(pager) == pager->JournalOffset)
				records = (uint32)((sizeJournal - pager->JournalOffset) / JOURNAL_PG_SZ(pager));
			for (uint32 ii = 0U; rc == RC_OK && ii < records && pager->JournalOffset < sizeJournal; ii++)
				rc = Pager_PlaybackOnePage(pager, &pager->JournalOffset, done, true, true);
			_assert(rc != RC_DONE);
		}
		_assert(rc != RC_OK || pager->JournalOffset >= sizeJournal);

		// Finally,  rollback pages from the sub-journal.  Page that were previously rolled back out of the main journal (and are hence in pDone)
		// will be skipped.  Out-of-range pages are also skipped.
		if (savepoint)
		{
			int64 offset = (int64)savepoint->SubRecords * (4 + pager->PageSize);

			if (UseWal(pager))
				rc = pager->Wal->SavepointUndo(savepoint->WalData);
			for (uint32 ii = savepoint->SubRecords; rc == RC_OK && ii < pager->SubRecords; ii++)
			{
				_assert(offset == (int64)ii * (4 + pager->PageSize));
				rc = Pager_PlaybackOnePage(pager, &offset, done, false, true);
			}
			_assert(rc != RC_DONE);
		}

		Bitvec::Destroy(done);
		if (rc == RC_OK)
			pager->JournalOffset = sizeJournal;

		return rc;
	}

#pragma endregion

#pragma region Name3

	__device__ void Pager::SetCacheSize(int maxPages)
	{
		PCache->set_CacheSize(maxPages);
	}

	__device__ void Pager::Shrink()
	{
		PCache->Shrink();
	}

#ifndef OMIT_PAGER_PRAGMAS
	__device__ void Pager::SetSafetyLevel(int level, bool fullFsync, bool checkpointFullFsync)
	{
		_assert(level >= 1 && level <= 3);
		NoSync =  (level == 1 || TempFile);
		FullSync = (level == 3 && !TempFile);
		if (NoSync)
		{
			SyncFlags = (VFile::SYNC)0;
			CheckpointSyncFlags = (VFile::SYNC)0;
		}
		else if (fullFsync)
		{
			SyncFlags = VFile::SYNC_FULL;
			CheckpointSyncFlags = VFile::SYNC_FULL;
		}
		else if (checkpointFullFsync)
		{
			SyncFlags = VFile::SYNC_NORMAL;
			CheckpointSyncFlags = VFile::SYNC_FULL;
		}
		else
		{
			SyncFlags = VFile::SYNC_NORMAL;
			CheckpointSyncFlags = VFile::SYNC_NORMAL;
		}
		WalSyncFlags = SyncFlags;
		if (FullSync)
			WalSyncFlags |= VFile::SYNC_WAL_TRANSACTIONS;
	}
#endif

#ifdef TEST
	// The following global variable is incremented whenever the library attempts to open a temporary file.  This information is used for testing and analysis only.  
	__device__ int g_opentemp_count = 0;
#endif

	__device__ static RC pagerOpentemp(Pager *pager, VFile *file, VSystem::OPEN vfsFlags)
	{
#ifdef TEST
		g_opentemp_count++; // Used for testing and analysis only
#endif
		vfsFlags |= (VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_EXCLUSIVE | VSystem::OPEN_DELETEONCLOSE);
		RC rc = pager->Vfs->Open(nullptr, file, vfsFlags, nullptr);
		_assert(rc != RC_OK || file->Opened);
		return rc;
	}

	__device__ void Pager::SetBusyhandler(int (*busyHandler)(void *), void *busyHandlerArg)
	{
		BusyHandler = busyHandler;
		BusyHandlerArg = busyHandlerArg;

		if (File->Opened)
		{
			void **ap = (void **)&BusyHandler;
			_assert(((int(*)(void *))(ap[0])) == busyHandler);
			_assert(ap[1] == busyHandlerArg);
			File->FileControl(VFile::FCNTL_BUSYHANDLER, (void *)ap);
		}
	}

	__device__ RC Pager::SetPageSize(uint32 *pageSizeRef, int reserveBytes)
	{
		// It is not possible to do a full AssertPagerState() here, as this function may be called from within PagerOpen(), before the state
		// of the Pager object is internally consistent.
		//
		// At one point this function returned an error if the pager was in PAGER_ERROR state. But since PAGER_ERROR state guarantees that
		// there is at least one outstanding page reference, this function is a no-op for that case anyhow.
		uint32 pageSize = *pageSizeRef;
		_assert(pageSize == 0 || (pageSize >= 512 && pageSize <= MAX_PAGE_SIZE));
		RC rc = RC_OK;
		if ((!MemoryDB || DBSize == 0) &&
			PCache->get_Refs() == 0 &&
			pageSize && pageSize != (uint32)PageSize)
		{
			int64 bytes = 0;
			if (State > Pager::PAGER_OPEN && File->Opened)
				rc = File->get_FileSize(bytes);
			char *tempSpace = nullptr; // New temp space
			if (rc == RC_OK)
			{
				tempSpace = (char *)PCache::PageAlloc(pageSize);
				if (!tempSpace) rc = RC_NOMEM;
			}
			if (rc == RC_OK)
			{
				Pager_Reset(this);
				DBSize = (Pid)((bytes + pageSize - 1) / pageSize);
				PageSize = (int)pageSize;
				PCache::PageFree(TmpSpace);
				TmpSpace = tempSpace;
				PCache->SetPageSize(pageSize);
			}
		}
		*pageSizeRef = (uint)PageSize;
		if (rc == RC_OK)
		{
			if (reserveBytes < 0) reserveBytes = ReserveBytes;
			_assert(reserveBytes >= 0 && reserveBytes < 1000);
			ReserveBytes = (int16)reserveBytes;
			pagerReportSize(this);
		}
		return rc;
	}

	__device__ void *Pager::get_TempSpace()
	{
		return TmpSpace;
	}

	__device__ int Pager::MaxPages(int maxPages)
	{
		if (maxPages > 0)
			MaxPid = maxPages;
		_assert(State != Pager::PAGER_OPEN); // Called only by OP_MaxPgcnt
		_assert(MaxPid >= DBSize); // OP_MaxPgcnt enforces this
		return MaxPid;
	}

#ifdef TEST
	__device__ int _io_error_pending;
	__device__ int _io_error_hit;
	__device__ static int saved_cnt;

	__device__ void disable_simulated_io_errors()
	{
		saved_cnt = _io_error_pending;
		_io_error_pending = -1;
	}

	__device__ void enable_simulated_io_errors()
	{
		_io_error_pending = saved_cnt;
	}
#else
#define disable_simulated_io_errors()
#define enable_simulated_io_errors()
#endif

	__device__ RC Pager::ReadFileheader(int n, unsigned char *dest)
	{
		_memset(dest, 0, n);
		_assert(File->Opened || TempFile);

		// This routine is only called by btree immediately after creating the Pager object.  There has not been an opportunity to transition to WAL mode yet.
		_assert(!UseWal(this));

		RC rc = RC_OK;
		if (File->Opened)
		{
			SysEx_IOTRACE("DBHDR %p 0 %d\n", this, n);
			rc = File->Read(dest, n, 0);
			if (rc == RC_IOERR_SHORT_READ)
				rc = RC_OK;
		}
		return rc;
	}

	__device__ void Pager::Pages(Pid *pagesOut)
	{
		_assert(State >= Pager::PAGER_READER);
		_assert(State != Pager::PAGER_WRITER_FINISHED);
		*pagesOut = DBSize;
	}

	__device__ static RC pager_wait_on_lock(Pager *pager, VFile::LOCK locktype)
	{
		// Check that this is either a no-op (because the requested lock is already held, or one of the transistions that the busy-handler
		// may be invoked during, according to the comment above sqlite3PagerSetBusyhandler().
		_assert((pager->Lock >= locktype) ||
			(pager->Lock == VFile::LOCK_NO && locktype == VFile::LOCK_SHARED) ||
			(pager->Lock == VFile::LOCK_RESERVED && locktype == VFile::LOCK_EXCLUSIVE));

		RC rc;
		do
		{
			rc = PagerLockDb(pager, locktype);
		} while(rc == RC_BUSY && pager->BusyHandler(pager->BusyHandlerArg));
		return rc;
	}

#if defined(_DEBUG)
	__device__ static void AssertTruncateConstraintCb(PgHdr *pg)
	{
		_assert(pg->Flags & PgHdr::PGHDR_DIRTY);
		_assert(!SubjRequiresPage(pg) || pg->ID <= pg->Pager->DBSize);
	}

	__device__ static void AssertTruncateConstraint(Pager *pager)
	{
		pager->PCache->IterateDirty(AssertTruncateConstraintCb);
	}
#else
#define AssertTruncateConstraint(pager)
#endif

	__device__ void Pager::TruncateImage(Pid pages)
	{
		_assert(DBSize >= pages);
		_assert(State >= Pager::PAGER_WRITER_CACHEMOD);
		DBSize = pages;

		// At one point the code here called AssertTruncateConstraint() to ensure that all pages being truncated away by this operation are,
		// if one or more savepoints are open, present in the savepoint journal so that they can be restored if the savepoint is rolled
		// back. This is no longer necessary as this function is now only called right before committing a transaction. So although the 
		// Pager object may still have open savepoints (Pager.nSavepoint!=0), they cannot be rolled back. So the AssertTruncateConstraint() call
		// is no longer correct.
	}

	__device__ static RC pagerSyncHotJournal(Pager *pager)
	{
		RC rc = RC_OK;
		if (!pager->NoSync)
			rc = pager->JournalFile->Sync(VFile::SYNC_NORMAL);
		if (rc == RC_OK)
			rc = pager->JournalFile->get_FileSize(pager->JournalHeader);
		return rc;
	}

	__device__ RC Pager::Close()
	{
		_assert(AssertPagerState(this));
		disable_simulated_io_errors();
		_benignalloc_begin();
		ErrorCode = RC_OK;
		ExclusiveMode = (IPager::LOCKINGMODE)0;
		uint8 *tmp = (uint8 *)TmpSpace;
#ifndef OMIT_WAL
		Wal->Close(CheckpointSyncFlags, PageSize, tmp);
		Wal = nullptr;
#endif
		Pager_Reset(this);
		if (MemoryDB)
			Pager_Unlock(this);
		else
		{
			// If it is open, sync the journal file before calling UnlockAndRollback. If this is not done, then an unsynced portion of the open journal 
			// file may be played back into the database. If a power failure occurs while this is happening, the database could become corrupt.
			//
			// If an error occurs while trying to sync the journal, shift the pager into the ERROR state. This causes UnlockAndRollback to unlock the
			// database and close the journal file without attempting to roll it back or finalize it. The next database user will have to do hot-journal
			// rollback before accessing the database file.
			if (JournalFile->Opened)
				Pager_Error(this, pagerSyncHotJournal(this));
			PagerUnlockAndRollback(this);
		}
		_benignalloc_end();
		enable_simulated_io_errors();
		PAGERTRACE("CLOSE %d\n", PAGERID(this));
		SysEx_IOTRACE("CLOSE %p\n", this);
		JournalFile->Close();
		File->Close();
		PCache::PageFree(tmp);
		PCache->Close();

#ifdef HAS_CODEC
		if (CodecFree) CodecFree(Codec);
#endif

		_assert(!Savepoints && !InJournal);
		_assert(!JournalFile->Opened && !SubJournalFile->Opened);

		_free(this);
		return RC_OK;
	}

#if !defined(_DEBUG) || defined(TEST)
	__device__ Pid Pager::get_PageID(IPage *pg)
	{
		return ((PgHdr *)pg)->ID;
	}
#endif

	__device__ void Pager::Ref(IPage *pg)
	{
		PCache::Ref((PgHdr *)pg);
	}

#pragma endregion

#pragma region Main

//#if __CUDACC__
//	__constant__ uint8 _zerobyte = 0;
//#else
//	static const uint8 _zerobyte = 0;
//#endif
	__constant__ uint8 _zerobyte = 0;
	__device__ static RC SyncJournal(Pager *pager, bool newHeader)
	{
		_assert(pager->State == Pager::PAGER_WRITER_CACHEMOD || pager->State == Pager::PAGER_WRITER_DBMOD);
		_assert(AssertPagerState(pager));
		_assert(!UseWal(pager));

		RC rc = pager->ExclusiveLock();
		if (rc != RC_OK) return rc;

		if (!pager->NoSync)
		{
			_assert(!pager->TempFile);
			if (pager->JournalFile->Opened && pager->JournalMode != IPager::JOURNALMODE_JMEMORY)
			{
				const int dc = pager->File->get_DeviceCharacteristics();
				_assert(pager->JournalFile->Opened);

				if ((dc & VFile::IOCAP_SAFE_APPEND) == 0)
				{
					// This block deals with an obscure problem. If the last connection that wrote to this database was operating in persistent-journal
					// mode, then the journal file may at this point actually be larger than Pager.journalOff bytes. If the next thing in the journal
					// file happens to be a journal-header (written as part of the previous connection's transaction), and a crash or power-failure 
					// occurs after nRec is updated but before this connection writes anything else to the journal file (or commits/rolls back its 
					// transaction), then SQLite may become confused when doing the hot-journal rollback following recovery. It may roll back all
					// of this connections data, then proceed to rolling back the old, out-of-date data that follows it. Database corruption.
					//
					// To work around this, if the journal file does appear to contain a valid header following Pager.journalOff, then write a 0x00
					// byte to the start of it to prevent it from being recognized.
					//
					// Variable iNextHdrOffset is set to the offset at which this problematic header will occur, if it exists. aMagic is used 
					// as a temporary buffer to inspect the first couple of bytes of the potential journal header.
					uint8 header[sizeof(_journalMagic) + 4];
					_memcpy(header, _journalMagic, sizeof(_journalMagic));
					ConvertEx::Put4(&header[sizeof(_journalMagic)], pager->Records);

					uint8 magic[8];
					int64 nextHdrOffset = JournalHdrOffset(pager);
					rc = pager->JournalFile->Read(magic, 8, nextHdrOffset);
					if (rc== RC_OK && _memcmp(magic, _journalMagic, 8) == 0)
						rc = pager->JournalFile->Write(&_zerobyte, 1, nextHdrOffset);
					if (rc != RC_OK && rc != RC_IOERR_SHORT_READ)
						return rc;

					// Write the nRec value into the journal file header. If in full-synchronous mode, sync the journal first. This ensures that
					// all data has really hit the disk before nRec is updated to mark it as a candidate for rollback.
					//
					// This is not required if the persistent media supports the SAFE_APPEND property. Because in this case it is not possible 
					// for garbage data to be appended to the file, the nRec field is populated with 0xFFFFFFFF when the journal header is written
					// and never needs to be updated.
					if (pager->FullSync && (dc & VFile::IOCAP_SEQUENTIAL) == 0)
					{
						PAGERTRACE("SYNC journal of %d\n", PAGERID(pager));
						SysEx_IOTRACE("JSYNC %p\n", pager);
						rc = pager->JournalFile->Sync(pager->SyncFlags);
						if (rc != RC_OK) return rc;
					}
					SysEx_IOTRACE("JHDR %p %lld\n", pager, pager->JournalHeader);
					rc = pager->JournalFile->Write(header, sizeof(header), pager->JournalHeader);
					if (rc != RC_OK) return rc;
				}
				if ((dc & VFile::IOCAP_SEQUENTIAL) == 0)
				{
					PAGERTRACE("SYNC journal of %d\n", PAGERID(pager));
					SysEx_IOTRACE("JSYNC %p\n", pager);
					rc = pager->JournalFile->Sync(pager->SyncFlags | (pager->SyncFlags == VFile::SYNC_FULL ? VFile::SYNC_DATAONLY : (VFile::SYNC)0));
					if (rc != RC_OK) return rc;
				}

				pager->JournalHeader = pager->JournalOffset;
				if (newHeader && (dc & VFile::IOCAP_SAFE_APPEND) == 0)
				{
					pager->Records = 0;
					rc = WriteJournalHdr(pager);
					if (rc != RC_OK) return rc;
				}
			}
			else
				pager->JournalHeader = pager->JournalOffset;
		}

		// Unless the pager is in noSync mode, the journal file was just successfully synced. Either way, clear the PGHDR_NEED_SYNC flag on all pages.
		pager->PCache->ClearSyncFlags();
		pager->State = Pager::PAGER_WRITER_DBMOD;
		_assert(AssertPagerState(pager));
		return RC_OK;
	}

	__device__ static RC Pager_WritePagelist(Pager *pager, PgHdr *list)
	{
		// This function is only called for rollback pagers in WRITER_DBMOD state.
		_assert(!UseWal(pager));
		_assert(pager->State == Pager::PAGER_WRITER_DBMOD);
		_assert(pager->Lock == VFile::LOCK_EXCLUSIVE);

		// If the file is a temp-file has not yet been opened, open it now. It is not possible for rc to be other than SQLITE_OK if this branch
		// is taken, as pager_wait_on_lock() is a no-op for temp-files.
		RC rc = RC_OK;
		if (!pager->File->Opened)
		{
			_assert(pager->TempFile && rc == RC_OK);
			rc = pagerOpentemp(pager, pager->File, pager->VfsFlags);
		}

		// Before the first write, give the VFS a hint of what the final file size will be.
		_assert(rc != RC_OK || pager->File->Opened);
		if (rc == RC_OK && pager->DBSize > pager->DBHintSize)
		{
			int64 sizeFile = pager->PageSize * (int64)pager->DBSize;
			pager->File->FileControl(VFile::FCNTL_SIZE_HINT, &sizeFile);
			pager->DBHintSize = pager->DBSize;
		}

		while (rc == RC_OK && list)
		{
			Pid id = list->ID;

			// If there are dirty pages in the page cache with page numbers greater than Pager.dbSize, this means sqlite3PagerTruncateImage() was called to
			// make the file smaller (presumably by auto-vacuum code). Do not write any such pages to the file.
			//
			// Also, do not write out any page that has the PGHDR_DONT_WRITE flag set (set by sqlite3PagerDontWrite()).
			if (id <= pager->DBSize && (list->Flags & PgHdr::PGHDR_DONT_WRITE) == 0)
			{
				_assert((list->Flags & PgHdr::PGHDR_NEED_SYNC) == 0);
				if (id == 1) Pager_WriteChangecounter(list);

				// Encode the database
				char *data; // Data to write
				CODEC2(char, data, pager, list->Data, id, 6, return RC_NOMEM);

				// Write out the page data.
				int64 offset = (id - 1) * (int64)pager->PageSize; // Offset to write
				rc = pager->File->Write(data, pager->PageSize, offset);

				// If page 1 was just written, update Pager.dbFileVers to match the value now stored in the database file. If writing this 
				// page caused the database file to grow, update dbFileSize. 
				if (id == 1)
					_memcpy(((char *)&pager->DBFileVersion), &data[24], sizeof(pager->DBFileVersion));
				if (id > pager->DBFileSize)
					pager->DBFileSize = id;
				pager->Stats[STAT_WRITE]++;

				// Update any backup objects copying the contents of this pager.
				if (pager->Backup != nullptr) pager->Backup->Update(id, (uint8 *)list->Data);

				PAGERTRACE("STORE %d page %d hash(%08x)\n", PAGERID(pager), id, Pager_PageHash(list));
				SysEx_IOTRACE("PGOUT %p %d\n", pager, id);
				PAGER_INCR(_writedb_count);
			}
			else
				PAGERTRACE("NOSTORE %d page %d\n", PAGERID(pager), id);
			Pager_SetPageHash(list);
			list = list->Dirty;
		}

		return rc;
	}

	__device__ static RC OpenSubJournal(Pager *pager)
	{
		RC rc = RC_OK;
		if (!pager->SubJournalFile->Opened)
		{
			if (pager->JournalMode == IPager::JOURNALMODE_JMEMORY || pager->SubjInMemory)
				VFile::MemoryVFileOpen(pager->SubJournalFile);
			else
				rc = pagerOpentemp(pager, pager->SubJournalFile, VSystem::OPEN_SUBJOURNAL);
		}
		return rc;
	}

	__device__ static RC SubjournalPage(PgHdr *pg)
	{
		RC rc = RC_OK;
		Pager *pager = pg->Pager;
		if (pager->JournalMode != IPager::JOURNALMODE_OFF)
		{
			// Open the sub-journal, if it has not already been opened
			_assert(pager->UseJournal);
			_assert(pager->JournalFile->Opened || UseWal(pager));
			_assert(pager->SubJournalFile->Opened || pager->SubRecords == 0);
			_assert(UseWal(Pager) ||
				PageInJournal(pg) ||
				pg->ID > pager->DBOrigSize);
			rc = OpenSubJournal(pager);

			// If the sub-journal was opened successfully (or was already open), write the journal record into the file.
			if (rc == RC_OK)
			{
				void *data = pg->Data;
				int64 offset = (int64)pager->SubRecords * (4 + pager->PageSize);
				char *data2;
				CODEC2(char, data2, pager, data, pg->ID, 7, return RC_NOMEM);
				PAGERTRACE("STMT-JOURNAL %d page %d\n", PAGERID(pager), pg->ID);
				rc = pager->SubJournalFile->Write4(offset, pg->ID);
				if (rc == RC_OK)
					rc = pager->SubJournalFile->Write(data2, pager->PageSize, offset + 4);
			}
		}
		if (rc == RC_OK)
		{
			pager->SubRecords++;
			_assert(pager->Savepoints.length > 0);
			rc = AddToSavepointBitvecs(pager, pg->ID);
		}
		return rc;
	}

	__device__ static RC PagerStress(void *p, IPage *pg)
	{
		Pager *pager = (Pager *)p;
		_assert(pg->Pager == pager);
		_assert(pg->Flags & PgHdr::PGHDR_DIRTY);

		// The doNotSyncSpill flag is set during times when doing a sync of journal (and adding a new header) is not allowed. This occurs
		// during calls to sqlite3PagerWrite() while trying to journal multiple pages belonging to the same sector.
		//
		// The doNotSpill flag inhibits all cache spilling regardless of whether or not a sync is required.  This is set during a rollback.
		//
		// Spilling is also prohibited when in an error state since that could lead to database corruption.   In the current implementaton it 
		// is impossible for sqlite3PcacheFetch() to be called with createFlag==1 while in the error state, hence it is impossible for this routine to
		// be called in the error state.  Nevertheless, we include a NEVER() test for the error state as a safeguard against future changes.
		if (_NEVER(pager->ErrorCode)) return RC_OK;
		if (pager->DoNotSpill) return RC_OK;
		if (pager->DoNotSyncSpill && (pg->Flags & PgHdr::PGHDR_NEED_SYNC) != 0)
			return RC_OK;

		pg->Dirty = nullptr;
		RC rc = RC_OK;
		if (UseWal(pager))
		{
			// Write a single frame for this page to the log.
			if (SubjRequiresPage(pg))
				rc = SubjournalPage(pg); 
			if (rc == RC_OK)
				rc = PagerWalFrames(pager, pg, 0, false);
		}
		else
		{
			// Sync the journal file if required.
			if (pg->Flags & PgHdr::PGHDR_NEED_SYNC || pager->State == Pager::PAGER_WRITER_CACHEMOD)
				rc = SyncJournal(pager, true);

			// If the page number of this page is larger than the current size of the database image, it may need to be written to the sub-journal.
			// This is because the call to Pager_WritePagelist() below will not actually write data to the file in this case.
			//
			// Consider the following sequence of events:
			//
			//   BEGIN;
			//     <journal page X>
			//     <modify page X>
			//     SAVEPOINT sp;
			//       <shrink database file to Y pages>
			//       PagerStress(page X)
			//     ROLLBACK TO sp;
			//
			// If (X>Y), then when PagerStress is called page X will not be written out to the database file, but will be dropped from the cache. Then,
			// following the "ROLLBACK TO sp" statement, reading page X will read data from the database file. This will be the copy of page X as it
			// was when the transaction started, not as it was when "SAVEPOINT sp" was executed.
			//
			// The solution is to write the current data for page X into the sub-journal file now (if it is not already there), so that it will
			// be restored to its current value when the "ROLLBACK TO sp" is executed.
			if (_NEVER(rc == RC_OK && pg->ID > pager->DBSize && SubjRequiresPage(pg)))
				rc = SubjournalPage(pg);

			// Write the contents of the page out to the database file.
			if (rc == RC_OK)
			{
				_assert((pg->Flags & PgHdr::PGHDR_NEED_SYNC) == 0);
				rc = Pager_WritePagelist(pager, pg);
			}
		}

		// Mark the page as clean.
		if (rc == RC_OK)
		{
			PAGERTRACE("STRESS %d page %d\n", PAGERID(pager), pg->ID);
			PCache::MakeClean(pg);
		}

		return Pager_Error(pager, rc); 
	}

	__device__ RC Pager::Open(VSystem *vfs, Pager **pagerOut, const char *filename, int extraBytes, IPager::PAGEROPEN flags, VSystem::OPEN vfsFlags, void (*reinit)(IPage *))
	{
		// Figure out how much space is required for each journal file-handle (there are two of them, the main journal and the sub-journal). This
		// is the maximum space required for an in-memory journal file handle and a regular journal file-handle. Note that a "regular journal-handle"
		// may be a wrapper capable of caching the first portion of the journal file in memory to implement the atomic-write optimization (see 
		// source file journal.c).
		int journalFileSize = _ROUND8(VFile::JournalVFileSize(vfs) > VFile::MemoryVFileSize() ? VFile::JournalVFileSize(vfs) : VFile::MemoryVFileSize()); // Bytes to allocate for each journal fd

		// Set the output variable to NULL in case an error occurs.
		*pagerOut = nullptr;

		bool memoryDB = false;		// True if this is an in-memory file
		char *pathname = nullptr;	// Full path to database file
		int pathnameLength = 0;		// Number of bytes in zPathname
		const char *uri = nullptr;	// URI args to copy
		int uriLength = 0;			// Number of bytes of URI args at *zUri
#ifndef OMIT_MEMORYDB
		if (flags & IPager::PAGEROPEN_MEMORY)
		{
			memoryDB = true;
			if (filename && filename[0])
			{
				pathname = _tagstrdup(nullptr, filename);
				if (pathname == nullptr) return RC_NOMEM;
				pathnameLength = _strlen30(pathname);
				filename = nullptr;
			}
		}
#endif

		// Compute and store the full pathname in an allocated buffer pointed to by zPathname, length nPathname. Or, if this is a temporary file,
		// leave both nPathname and zPathname set to 0.
		RC rc = RC_OK;
		if (filename && filename[0])
		{
			pathnameLength = vfs->MaxPathname + 1;
			pathname = (char *)_tagalloc(nullptr, pathnameLength * 2);
			if (!pathname) return RC_NOMEM;
			pathname[0] = 0; // Make sure initialized even if FullPathname() fails
			rc = vfs->FullPathname(filename, pathnameLength, pathname);
			pathnameLength = _strlen30(pathname);
			const char *z = uri = &filename[_strlen30(filename) + 1];
			while (*z)
			{
				z += _strlen30(z) + 1;
				z += _strlen30(z) + 1;
			}
			uriLength = (int)(&z[1] - uri);
			_assert(uriLength >= 0);
			// This branch is taken when the journal path required by the database being opened will be more than pVfs->mxPathname
			// bytes in length. This means the database cannot be opened, as it will not be possible to open the journal file or even
			// check for a hot-journal before reading.
			if (rc == RC_OK && pathnameLength + 8 > vfs->MaxPathname)
				rc = SysEx_CANTOPEN_BKPT;
			if (rc != RC_OK)
			{
				_tagfree(nullptr, pathname);
				return rc;
			}
		}

		// Allocate memory for the Pager structure, PCache object, the three file descriptors, the database file name and the journal file name.
		int pcacheSizeOf = PCache::SizeOf();	// Bytes to allocate for PCache
		uint8 *ptr = (uint8 *)_alloc2(
			_ROUND8(sizeof(Pager)) +		// Pager structure
			_ROUND8(pcacheSizeOf) +        // PCache object
			_ROUND8(vfs->SizeOsFile) +     // The main db file
			journalFileSize * 2 +				// The two journal files
			pathnameLength + 1 + uriLength +    // zFilename
			pathnameLength + 8 + 2              // zJournal
#ifndef OMIT_WAL
			+ pathnameLength + 4 + 2			// zWal
#endif
			, true);
		_assert(_HASALIGNMENT8(INT_TO_PTR(journalFileSize)));
		if (!ptr)
		{
			_tagfree(nullptr, pathname);
			return RC_NOMEM;
		}
		Pager *pager = (Pager *)(ptr);
		pager->PCache = (Core::PCache *)(ptr += _ROUND8(sizeof(Pager)));
		pager->File = (VFile *)(ptr += _ROUND8(pcacheSizeOf));
		pager->SubJournalFile = vfs->_AttachFile(ptr += _ROUND8(vfs->SizeOsFile));
		pager->JournalFile = vfs->_AttachFile(ptr += journalFileSize);
		pager->Filename = (char *)(ptr += journalFileSize);
		_assert(_HASALIGNMENT8(pager->JournalFile));

		// Fill in the Pager.zFilename and Pager.zJournal buffers, if required.
		if (pathname)
		{
			_assert(pathnameLength > 0);
			pager->Journal = (char *)(ptr += pathnameLength + 1 + uriLength);
			_memcpy(pager->Filename, pathname, pathnameLength);
			if (uriLength) _memcpy(&pager->Filename[pathnameLength + 1], uri, uriLength);
			_memcpy(pager->Journal, pathname, pathnameLength);
			_memcpy(&pager->Journal[pathnameLength], "-journal\000", 8 + 2);
#ifndef OMIT_WAL
			pager->WalName = &pager->Journal[pathnameLength + 8 + 1];
			_memcpy(pager->WalName, pathname, pathnameLength);
			_memcpy(&pager->WalName[pathnameLength], "-wal\000", 4 + 1);
#endif
			_tagfree(nullptr, pathname);
		}
		pager->Vfs = vfs;
		pager->VfsFlags = vfsFlags;

		// Open the pager file.
		bool tempFile = false; // True for temp files (incl. in-memory files)
		bool readOnly = false; // True if this is a read-only file
		uint32 sizePage = DEFAULT_PAGE_SIZE;  // Default page size
		if (filename && filename[0])
		{
			VSystem::OPEN fout = (VSystem::OPEN)0; // VFS flags returned by xOpen()
			rc = vfs->Open(pager->Filename, pager->File, vfsFlags, &fout);
			_assert(!memoryDB);
			readOnly = ((fout & VSystem::OPEN_READONLY) != 0);

			// If the file was successfully opened for read/write access, choose a default page size in case we have to create the
			// database file. The default page size is the maximum of:
			//
			//    + SQLITE_DEFAULT_PAGE_SIZE,
			//    + The value returned by sqlite3OsSectorSize()
			//    + The largest page size that can be written atomically.
			if (rc == RC_OK && !readOnly)
			{
				set_SectorSize(pager);
				_assert(DEFAULT_PAGE_SIZE <= MAX_DEFAULT_PAGE_SIZE);
				if (sizePage < pager->SectorSize)
					sizePage = (pager->SectorSize > MAX_DEFAULT_PAGE_SIZE ? MAX_DEFAULT_PAGE_SIZE : (uint32)pager->SectorSize);
#ifdef ENABLE_ATOMIC_WRITE
				{
					_assert(VFile::IOCAP_ATOMIC512 == (512 >> 8));
					_assert(VFile::IOCAP_ATOMIC64K == (65536 >> 8));
					_assert(MAX_DEFAULT_PAGE_SIZE <= 65536);
					int dc = pager->File->get_DeviceCharacteristics();
					for (int ii = sizePage; ii <= MAX_DEFAULT_PAGE_SIZE; ii = ii * 2)
						if (dc & (VFile::IOCAP_ATOMIC | (ii >> 8)))
							sizePage = ii;
				}
#endif
			}
		}
		else
		{
			// If a temporary file is requested, it is not opened immediately. In this case we accept the default page size and delay actually
			// opening the file until the first call to OsWrite().
			//
			// This branch is also run for an in-memory database. An in-memory database is the same as a temp-file that is never written out to
			// disk and uses an in-memory rollback journal.
			tempFile = true;
			pager->State = Pager::PAGER_READER;
			pager->Lock = VFile::LOCK_EXCLUSIVE;
			readOnly = ((vfsFlags & VSystem::OPEN_READONLY) != 0);
		}

		// The following call to PagerSetPagesize() serves to set the value of Pager.pageSize and to allocate the Pager.pTmpSpace buffer.
		if (rc == RC_OK)
		{
			_assert(!pager->MemoryDB);
			rc = pager->SetPageSize(&sizePage, -1);
			ASSERTCOVERAGE(rc != RC_OK);
		}

		// If an error occurred in either of the blocks above, free the Pager structure and close the file.
		if (rc != RC_OK)
		{
			_assert(!pager->TmpSpace);
			pager->File->Close();
			_free(pager);
			return rc;
		}

		// Initialize the PCache object.
		_assert(extraBytes < 1000);
		extraBytes = _ROUND8(extraBytes);
		PCache::Open(sizePage, extraBytes, !memoryDB, (!memoryDB ? PagerStress : nullptr), (void *)pager, pager->PCache);

		PAGERTRACE("OPEN %d %s\n", FILEHANDLEID(pager->File), pager->Filename);
		SysEx_IOTRACE("OPEN %p %s\n", pager, pager->Filename);

		bool useJournal = (flags & IPager::PAGEROPEN_OMIT_JOURNAL) == 0; // False to omit journal
		pager->UseJournal = useJournal;
		pager->MaxPid = MAX_PAGE_COUNT;
		pager->TempFile = tempFile;
		//_assert(tempFile == IPager::LOCKINGMODE_NORMAL || tempFile == IPager::LOCKINGMODE_EXCLUSIVE);
		//_assert(IPager::LOCKINGMODE_EXCLUSIVE == 1);
		pager->ExclusiveMode = (IPager::LOCKINGMODE)(tempFile ? 1 : 0); 
		pager->ChangeCountDone = tempFile;
		pager->MemoryDB = memoryDB;
		pager->ReadOnly = readOnly;
		_assert(useJournal || tempFile);
		pager->NoSync = tempFile;
		if (pager->NoSync)
		{
			_assert(!pager->FullSync);
			_assert(pager->SyncFlags == 0);
			_assert(pager->WalSyncFlags == 0);
			_assert(pager->CheckpointSyncFlags == 0);
		}
		else
		{
			pager->FullSync = true;
			pager->SyncFlags = VFile::SYNC_NORMAL;
			pager->WalSyncFlags = (VFile::SYNC)(VFile::SYNC_NORMAL | VFile::SYNC_WAL_TRANSACTIONS);
			pager->CheckpointSyncFlags = VFile::SYNC_NORMAL;
		}
		pager->ExtraBytes = (uint16)extraBytes;
		pager->JournalSizeLimit = DEFAULT_JOURNAL_SIZE_LIMIT;
		_assert(pager->File->Opened || tempFile);
		set_SectorSize(pager);
		if (!useJournal)
			pager->JournalMode = IPager::JOURNALMODE_OFF;
		else if (memoryDB)
			pager->JournalMode = IPager::JOURNALMODE_JMEMORY;
		pager->Reiniter = reinit;

		*pagerOut = pager;
		return RC_OK;
	}

	__device__ static RC HasHotJournal(Pager *pager, bool *existsOut)
	{
		_assert(pager->UseJournal);
		_assert(pager->File->Opened);
		_assert(pager->State == Pager::PAGER_OPEN);
		bool journalOpened = pager->JournalFile->Opened;
		_assert(!journalOpened || (pager->JournalFile->get_DeviceCharacteristics() & VFile::IOCAP_UNDELETABLE_WHEN_OPEN));

		*existsOut = false;
		VSystem *const vfs = pager->Vfs;
		RC rc = RC_OK;
		int exists = 1; // True if a journal file is present
		if (!journalOpened)
			rc = vfs->Access(pager->Journal, VSystem::ACCESS_EXISTS, &exists);
		if (rc == RC_OK && exists)
		{
			// Race condition here:  Another process might have been holding the the RESERVED lock and have a journal open at the sqlite3OsAccess() 
			// call above, but then delete the journal and drop the lock before we get to the following sqlite3OsCheckReservedLock() call.  If that
			// is the case, this routine might think there is a hot journal when in fact there is none.  This results in a false-positive which will
			// be dealt with by the playback routine.  Ticket #3883.
			int locked = 0; // True if some process holds a RESERVED lock
			rc = pager->File->CheckReservedLock(locked);
			if (rc == RC_OK && !locked)
			{
				// Check the size of the database file. If it consists of 0 pages, then delete the journal file. See the header comment above for 
				// the reasoning here.  Delete the obsolete journal file under a RESERVED lock to avoid race conditions and to avoid violating [H33020].
				Pid pages; // Number of pages in database file
				rc = PagerPagecount(pager, &pages);
				if (rc == RC_OK)
					if (pages == 0)
					{
						_benignalloc_begin();
						if (PagerLockDb(pager, VFile::LOCK_RESERVED) == RC_OK)
						{
							vfs->Delete(pager->Journal, false);
							if (!pager->ExclusiveMode) PagerUnlockDb(pager, VFile::LOCK_SHARED);
						}
						_benignalloc_end();
					}
					else
					{
						// The journal file exists and no other connection has a reserved or greater lock on the database file. Now check that there is
						// at least one non-zero bytes at the start of the journal file. If there is, then we consider this journal to be hot. If not, 
						// it can be ignored.
						if (!journalOpened)
						{
							VSystem::OPEN f = (VSystem::OPEN)(VSystem::OPEN_READONLY | VSystem::OPEN_MAIN_JOURNAL);
							rc = vfs->Open(pager->Journal, pager->JournalFile, f, &f);
						}
						if (rc == RC_OK)
						{
							uint8 first = 0;
							rc = pager->JournalFile->Read((void *)&first, 1, 0);
							if (rc == RC_IOERR_SHORT_READ)
								rc = RC_OK;
							if (!journalOpened)
								pager->JournalFile->Close();
							*existsOut = (first != 0);
						}
						else if (rc == RC_CANTOPEN)
						{
							// If we cannot open the rollback journal file in order to see if its has a zero header, that might be due to an I/O error, or
							// it might be due to the race condition described above and in ticket #3883.  Either way, assume that the journal is hot.
							// This might be a false positive.  But if it is, then the automatic journal playback and recovery mechanism will deal
							// with it under an EXCLUSIVE lock where we do not need to worry so much with race conditions.
							*existsOut = true;
							rc = RC_OK;
						}
					}
			}
		}

		return rc;
	}

	__device__ RC Pager::SharedLock()
	{
		// This routine is only called from b-tree and only when there are no outstanding pages. This implies that the pager state should either
		// be OPEN or READER. READER is only possible if the pager is or was in  exclusive access mode.
		_assert(PCache->get_Refs() == 0);
		_assert(AssertPagerState(this));
		_assert(State == Pager::PAGER_OPEN || State == Pager::PAGER_READER);
		if (_NEVER(MemoryDB && ErrorCode)) return ErrorCode;

		RC rc = RC_OK;
		if (!UseWal(this) && State == Pager::PAGER_OPEN)
		{
			_assert(!MemoryDB);

			rc = pager_wait_on_lock(this, VFile::LOCK_SHARED);
			if (rc != RC_OK)
			{
				_assert(Lock == VFile::LOCK_NO || Lock == VFile::LOCK_UNKNOWN);
				goto failed;
			}

			// If a journal file exists, and there is no RESERVED lock on the database file, then it either needs to be played back or deleted.
			bool hotJournal = true; // True if there exists a hot journal-file
			if (Lock <= VFile::LOCK_SHARED)
				rc = HasHotJournal(this, &hotJournal);
			if (rc != RC_OK)
				goto failed;
			if (hotJournal)
			{
				if (ReadOnly)
				{
					rc = RC_READONLY;
					goto failed;
				}

				// Get an EXCLUSIVE lock on the database file. At this point it is important that a RESERVED lock is not obtained on the way to the
				// EXCLUSIVE lock. If it were, another process might open the database file, detect the RESERVED lock, and conclude that the
				// database is safe to read while this process is still rolling the hot-journal back.
				// 
				// Because the intermediate RESERVED lock is not requested, any other process attempting to access the database file will get to 
				// this point in the code and fail to obtain its own EXCLUSIVE lock on the database file.
				//
				// Unless the pager is in locking_mode=exclusive mode, the lock is downgraded to SHARED_LOCK before this function returns.
				rc = PagerLockDb(this, VFile::LOCK_EXCLUSIVE);
				if (rc != RC_OK)
					goto failed;

				// If it is not already open and the file exists on disk, open the journal for read/write access. Write access is required because 
				// in exclusive-access mode the file descriptor will be kept open and possibly used for a transaction later on. Also, write-access 
				// is usually required to finalize the journal in journal_mode=persist mode (and also for journal_mode=truncate on some systems).
				//
				// If the journal does not exist, it usually means that some other connection managed to get in and roll it back before 
				// this connection obtained the exclusive lock above. Or, it may mean that the pager was in the error-state when this
				// function was called and the journal file does not exist.
				if (!JournalFile->Opened)
				{
					VSystem *const vfs = Vfs;
					int exists; // True if journal file exists
					rc = vfs->Access(Journal, VSystem::ACCESS_EXISTS, &exists);
					if (rc == RC_OK && exists)
					{
						_assert(!TempFile);
						VSystem::OPEN fout = (VSystem::OPEN)0;
						rc = vfs->Open(Journal, JournalFile, (VSystem::OPEN)(VSystem::OPEN_READWRITE | VSystem::OPEN_MAIN_JOURNAL), &fout);
						_assert(rc != RC_OK || JournalFile->Opened);
						if (rc == RC_OK && fout & VSystem::OPEN_READONLY)
						{
							rc = SysEx_CANTOPEN_BKPT;
							JournalFile->Close();
						}
					}
				}

				// Playback and delete the journal.  Drop the database write lock and reacquire the read lock. Purge the cache before
				// playing back the hot-journal so that we don't end up with an inconsistent cache.  Sync the hot journal before playing
				// it back since the process that crashed and left the hot journal probably did not sync it and we are required to always sync
				// the journal before playing it back.
				if (JournalFile->Opened)
				{
					_assert(rc == RC_OK);
					rc = pagerSyncHotJournal(this);
					if (rc == RC_OK)
					{
						rc = Pager_Playback(this, true);
						State = Pager::PAGER_OPEN;
					}
				}
				else if (!ExclusiveMode)
					PagerUnlockDb(this, VFile::LOCK_SHARED);

				if (rc != RC_OK)
				{
					// This branch is taken if an error occurs while trying to open or roll back a hot-journal while holding an EXCLUSIVE lock. The
					// Pager_Unlock() routine will be called before returning to unlock the file. If the unlock attempt fails, then Pager.eLock must be
					// set to UNKNOWN_LOCK (see the comment above the #define for UNKNOWN_LOCK above for an explanation). 
					//
					// In order to get Pager_Unlock() to do this, set Pager.eState to PAGER_ERROR now. This is not actually counted as a transition
					// to ERROR state in the state diagram at the top of this file, since we know that the same call to Pager_Unlock() will very
					// shortly transition the pager object to the OPEN state. Calling AssertPagerState() would fail now, as it should not be possible
					// to be in ERROR state when there are zero outstanding page references.
					Pager_Error(this, rc);
					goto failed;
				}

				_assert(State == Pager::PAGER_OPEN);
				_assert((Lock == VFile::LOCK_SHARED) || (ExclusiveMode && Lock > VFile::LOCK_SHARED));
			}

			if (!TempFile && (Backup || PCache->get_Pages() > 0))
			{
				// The shared-lock has just been acquired on the database file and there are already pages in the cache (from a previous
				// read or write transaction).  Check to see if the database has been modified.  If the database has changed, flush the cache.
				//
				// Database changes is detected by looking at 15 bytes beginning at offset 24 into the file.  The first 4 of these 16 bytes are
				// a 32-bit counter that is incremented with each change.  The other bytes change randomly with each file change when
				// a codec is in use.
				// 
				// There is a vanishingly small chance that a change will not be detected.  The chance of an undetected change is so small that
				// it can be neglected.
				Pid pages = 0;
				char dbFileVersion[sizeof(DBFileVersion)];

				rc = PagerPagecount(this, &pages);
				if (rc != RC_OK)
					goto failed;

				if (pages > 0)
				{
					SysEx_IOTRACE("CKVERS %p %d\n", this, sizeof(dbFileVersion));
					rc = File->Read(&dbFileVersion, sizeof(dbFileVersion), 24);
					if (rc != RC_OK)
						goto failed;
				}
				else
					_memset(dbFileVersion, 0, sizeof(dbFileVersion));

				if (_memcmp(DBFileVersion, dbFileVersion, sizeof(dbFileVersion)) != 0)
					Pager_Reset(this);
			}

			// If there is a WAL file in the file-system, open this database in WAL mode. Otherwise, the following function call is a no-op.
			rc = PagerOpenWalIfPresent(this);
#ifndef OMIT_WAL
			_assert(pager->Wal == nullptr || rc == RC_OK);
#endif
		}

		if (UseWal(pager))
		{
			_assert(rc == RC_OK);
			rc = PagerBeginReadTransaction(this);
		}

		if (State == Pager::PAGER_OPEN && rc == RC_OK)
			rc = PagerPagecount(this, &DBSize);

failed:
		if (rc != RC_OK)
		{
			_assert(!MemoryDB);
			Pager_Unlock(this);
			_assert(State == Pager::PAGER_OPEN);
		}
		else
			State = Pager::PAGER_READER;
		return rc;
	}

	__device__ static void PagerUnlockIfUnused(Pager *pager)
	{
		if (pager->PCache->get_Refs() == 0)
			PagerUnlockAndRollback(pager);
	}

	__device__ RC Pager::Acquire(Pid id, IPage **pageOut, bool noContent)
	{
		_assert(State >= Pager::PAGER_READER);
		_assert(AssertPagerState(this));

		if (id == 0)
			return SysEx_CORRUPT_BKPT;

		// If the pager is in the error state, return an error immediately. Otherwise, request the page from the PCache layer.
		RC rc;
		if (ErrorCode != RC_OK) 
			rc = ErrorCode;
		else 
			rc = PCache->Fetch(id, true, pageOut);

		PgHdr *pg;
		if (rc != RC_OK)
		{
			// Either the call to sqlite3PcacheFetch() returned an error or the pager was already in the error-state when this function was called.
			// Set pPg to 0 and jump to the exception handler.
			pg = nullptr;
			goto pager_acquire_err;
		}
		_assert((*pageOut)->ID == id);
		_assert((*pageOut)->Pager == this || (*pageOut)->Pager == nullptr);

		if ((*pageOut)->Pager && !noContent)
		{
			// In this case the pcache already contains an initialized copy of the page. Return without further ado.
			_assert(id <= MAX_PID && id != MJ_PID(this));
			Stats[STAT_HIT]++;
			return RC_OK;
		}
		// The pager cache has created a new page. Its content needs to be initialized.
		pg = *pageOut;
		pg->Pager = this;

		// The maximum page number is 2^31. Return SQLITE_CORRUPT if a page number greater than this, or the unused locking-page, is requested.
		if (id > MAX_PID || id == MJ_PID(this))
		{
			rc = SysEx_CORRUPT_BKPT;
			goto pager_acquire_err;
		}

		if (MemoryDB || DBSize < id || noContent || !File->Opened)
		{
			if (id > MaxPid)
			{
				rc = RC_FULL;
				goto pager_acquire_err;
			}
			if (noContent)
			{
				// Failure to set the bits in the InJournal bit-vectors is benign. It merely means that we might do some extra work to journal a 
				// page that does not need to be journaled.  Nevertheless, be sure to test the case where a malloc error occurs while trying to set 
				// a bit in a bit vector.
				_benignalloc_begin();
				if (id <= DBOrigSize)
				{
					ASSERTONLY(rc =) InJournal->Set(id);
					ASSERTCOVERAGE(rc == RC_NOMEM);
				}
				ASSERTONLY(rc =) AddToSavepointBitvecs(this, id);
				ASSERTCOVERAGE(rc == RC_NOMEM);
				_benignalloc_end();
			}
			_memset(pg->Data, 0, PageSize);
			SysEx_IOTRACE("ZERO %p %d\n", this, id);
		}
		else
		{
			_assert(pg->Pager == this);
			Stats[STAT_MISS]++;
			rc = ReadDbPage(pg);
			if (rc != RC_OK)
				goto pager_acquire_err;
		}
		Pager_SetPageHash(pg);
		return RC_OK;

pager_acquire_err:
		_assert(rc != RC_OK);
		if (pg)
			PCache::Drop(pg);
		PagerUnlockIfUnused(this);

		*pageOut = nullptr;
		return rc;
	}

	__device__ IPage *Pager::Lookup(Pid id)
	{
		_assert(id != 0);
		_assert(PCache != nullptr);
		_assert(State >= Pager::PAGER_READER && State != Pager::PAGER_ERROR);
		PgHdr *pg;
		PCache->Fetch(id, false, &pg);
		return pg;
	}

	__device__ void Pager::Unref(IPage *pg)
	{
		if (pg)
		{
			Pager *pager = pg->Pager;
			PCache::Release(pg);
			PagerUnlockIfUnused(pager);
		}
	}

	__device__ static RC Pager_OpenJournal(Pager *pager)
	{
		_assert(pager->State == Pager::PAGER_WRITER_LOCKED);
		_assert(AssertPagerState(pager));
		_assert(pager->InJournal == nullptr);

		// If already in the error state, this function is a no-op.  But on the other hand, this routine is never called if we are already in
		// an error state.
		if (_NEVER(pager->ErrorCode)) return pager->ErrorCode;

		RC rc = RC_OK;
		if (!UseWal(pager) && pager->JournalMode != IPager::JOURNALMODE_OFF)
		{
			pager->InJournal = new Bitvec(pager->DBSize);
			if (pager->InJournal == nullptr)
				return RC_NOMEM;

			// Open the journal file if it is not already open.
			if (!pager->JournalFile->Opened)
			{
				if (pager->JournalMode == IPager::JOURNALMODE_JMEMORY)
					VFile::MemoryVFileOpen(pager->JournalFile);
				else
				{
					const VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | (pager->TempFile ? VSystem::OPEN_DELETEONCLOSE | VSystem::OPEN_TEMP_JOURNAL : VSystem::OPEN_MAIN_JOURNAL));
#ifdef ENABLE_ATOMIC_WRITE
					rc = VFile::JournalVFileOpen(pager->Vfs, pager->Journal, pager->JournalFile, flags, JrnlBufferSize(pager));
#else
					rc = pager->Vfs->Open(pager->Journal, pager->JournalFile, flags, 0);
#endif
				}
				_assert(rc != RC_OK || pager->JournalFile->Opened);
			}

			// Write the first journal header to the journal file and open the sub-journal if necessary.
			if (rc == RC_OK)
			{
				// TODO: Check if all of these are really required.
				pager->Records = 0;
				pager->JournalOffset = 0;
				pager->SetMaster = false;
				pager->JournalHeader = 0;
				rc = WriteJournalHdr(pager);
			}
		}

		if (rc != RC_OK)
		{
			Bitvec::Destroy(pager->InJournal);
			pager->InJournal = nullptr;
		}
		else
		{
			_assert(pager->State == Pager::PAGER_WRITER_LOCKED);
			pager->State = Pager::PAGER_WRITER_CACHEMOD;
		}

		return rc;
	}

	__device__ RC Pager::Begin(bool exFlag, bool subjInMemory)
	{
		if (ErrorCode) return ErrorCode;
		_assert(State >= Pager::PAGER_READER && State < Pager::PAGER_ERROR);
		SubjInMemory = subjInMemory;

		RC rc = RC_OK;
		if (_ALWAYS(State == Pager::PAGER_READER))
		{
			_assert(InJournal == nullptr);

			if (UseWal(this))
			{
				// If the pager is configured to use locking_mode=exclusive, and an exclusive lock on the database is not already held, obtain it now.
				if (ExclusiveMode && Wal->ExclusiveMode(-1))
				{
					rc = PagerLockDb(this, VFile::LOCK_EXCLUSIVE);
					if (rc != RC_OK)
						return rc;
					Wal->ExclusiveMode(1);
				}

				// Grab the write lock on the log file. If successful, upgrade to PAGER_RESERVED state. Otherwise, return an error code to the caller.
				// The busy-handler is not invoked if another connection already holds the write-lock. If possible, the upper layer will call it.
				rc = Wal->BeginWriteTransaction();
			}
			else
			{
				// Obtain a RESERVED lock on the database file. If the exFlag parameter is true, then immediately upgrade this to an EXCLUSIVE lock. The
				// busy-handler callback can be used when upgrading to the EXCLUSIVE lock, but not when obtaining the RESERVED lock.
				rc = PagerLockDb(this, VFile::LOCK_RESERVED);
				if (rc == RC_OK && exFlag)
					rc = pager_wait_on_lock(this, VFile::LOCK_EXCLUSIVE);
			}

			if (rc == RC_OK)
			{
				// Change to WRITER_LOCKED state.
				//
				// WAL mode sets Pager.eState to PAGER_WRITER_LOCKED or CACHEMOD when it has an open transaction, but never to DBMOD or FINISHED.
				// This is because in those states the code to roll back savepoint transactions may copy data from the sub-journal into the database 
				// file as well as into the page cache. Which would be incorrect in WAL mode.
				State = Pager::PAGER_WRITER_LOCKED;
				DBHintSize = DBSize;
				DBFileSize = DBSize;
				DBOrigSize = DBSize;
				JournalOffset = 0;
			}

			_assert(rc == RC_OK || State == Pager::PAGER_READER);
			_assert(rc != RC_OK || State == Pager::PAGER_WRITER_LOCKED);
			_assert(AssertPagerState(this));
		}

		PAGERTRACE("TRANSACTION %d\n", PAGERID(this));
		return rc;
	}

	__device__ static RC Pager_Write(PgHdr *pg)
	{
		void *data = pg->Data;
		Pager *pager = pg->Pager;

		// This routine is not called unless a write-transaction has already been started. The journal file may or may not be open at this point. It is never called in the ERROR state.
		_assert(pager->State == Pager::PAGER_WRITER_LOCKED ||
			pager->State == Pager::PAGER_WRITER_CACHEMOD ||
			pager->State == Pager::PAGER_WRITER_DBMOD);
		_assert(AssertPagerState(pager));

		// If an error has been previously detected, report the same error again. This should not happen, but the check provides robustness.
		if (_NEVER(pager->ErrorCode)) return pager->ErrorCode;

		// Higher-level routines never call this function if database is not writable.  But check anyway, just for robustness.
		if (_NEVER(pager->ReadOnly)) return RC_PERM;

		CHECK_PAGE(pg);

		// The journal file needs to be opened. Higher level routines have already obtained the necessary locks to begin the write-transaction, but the
		// rollback journal might not yet be open. Open it now if this is the case.
		//
		// This is done before calling sqlite3PcacheMakeDirty() on the page. Otherwise, if it were done after calling sqlite3PcacheMakeDirty(), then
		// an error might occur and the pager would end up in WRITER_LOCKED state with pages marked as dirty in the cache.
		RC rc = RC_OK;
		if (pager->State == Pager::PAGER_WRITER_LOCKED)
		{
			rc = Pager_OpenJournal(pager);
			if (rc != RC_OK) return rc;
		}
		_assert(pager->State >= Pager::PAGER_WRITER_CACHEMOD);
		_assert(AssertPagerState(pager));

		// Mark the page as dirty.  If the page has already been written to the journal then we can return right away.
		PCache::MakeDirty(pg);
		if (PageInJournal(pg) && !SubjRequiresPage(pg))
			_assert(!UseWal(pager));
		else
		{
			// The transaction journal now exists and we have a RESERVED or an EXCLUSIVE lock on the main database file.  Write the current page to the transaction journal if it is not there already.
			if (!PageInJournal(pg) && !UseWal(pager))
			{
				_assert(UseWal(pager) == 0);
				if (pg->ID <= pager->DBOrigSize && pager->JournalFile->Opened)
				{
					// We should never write to the journal file the page that contains the database locks.  The following assert verifies that we do not.
					_assert(pg->ID != MJ_PID(pager));

					_assert(pager->JournalHeader <= pager->JournalOffset);
					char *data2;
					CODEC2(char, data2, pager, data, pg->ID, 7, return RC_NOMEM);
					uint32 checksum = Pager_Cksum(pager, (uint8 *)data2);

					// Even if an IO or diskfull error occurs while journalling the page in the block above, set the need-sync flag for the page.
					// Otherwise, when the transaction is rolled back, the logic in playback_one_page() will think that the page needs to be restored
					// in the database file. And if an IO error occurs while doing so, then corruption may follow.
					pg->Flags |= PgHdr::PGHDR_NEED_SYNC;

					int64 offset = pager->JournalOffset;
					rc = pager->JournalFile->Write4(offset, pg->ID);
					if (rc != RC_OK) return rc;
					rc = pager->JournalFile->Write(data2, pager->PageSize, offset + 4);
					if (rc != RC_OK) return rc;
					rc = pager->JournalFile->Write4(offset + pager->PageSize + 4, checksum);
					if (rc != RC_OK) return rc;

					SysEx_IOTRACE("JOUT %p %d %lld %d\n", pager, pg->ID, pager->JournalOffset, pager->PageSize);
					PAGER_INCR(_writej_count);
					PAGERTRACE("JOURNAL %d page %d needSync=%d hash(%08x)\n", PAGERID(pager), pg->ID, ((pg->Flags & PgHdr::PGHDR_NEED_SYNC) ? 1 : 0), Pager_PageHash(pg));

					pager->JournalOffset += 8 + pager->PageSize;
					pager->Records++;
					_assert(pager->InJournal != nullptr);
					rc = pager->InJournal->Set(pg->ID);
					ASSERTCOVERAGE(rc == RC_NOMEM);
					_assert(rc == RC_OK || rc == RC_NOMEM);
					rc |= AddToSavepointBitvecs(pager, pg->ID);
					if (rc != RC_OK)
					{
						_assert(rc == RC_NOMEM);
						return rc;
					}
				}
				else
				{
					if (pager->State != Pager::PAGER_WRITER_DBMOD)
						pg->Flags |= PgHdr::PGHDR_NEED_SYNC;
					PAGERTRACE("APPEND %d page %d needSync=%d\n", PAGERID(pager), pg->ID, ((pg->Flags & PgHdr::PGHDR_NEED_SYNC) ? 1 : 0));
				}
			}

			// If the statement journal is open and the page is not in it, then write the current page to the statement journal.  Note that
			// the statement journal format differs from the standard journal format in that it omits the checksums and the header.
			if (SubjRequiresPage(pg))
				rc = SubjournalPage(pg);
		}

		// Update the database size and return.
		if (pager->DBSize < pg->ID)
			pager->DBSize = pg->ID;
		return rc;
	}

	__device__ RC Pager::Write(IPage *page)
	{
		PgHdr *pg = page;
		Pager *pager = pg->Pager;
		Pid pagePerSector = (pager->SectorSize / pager->PageSize);

		_assert(pager->State >= Pager::PAGER_WRITER_LOCKED);
		_assert(pager->State != Pager::PAGER_ERROR);
		_assert(AssertPagerState(pager));

		RC rc = RC_OK;
		if (pagePerSector > 1)
		{
			// Set the doNotSyncSpill flag to 1. This is because we cannot allow a journal header to be written between the pages journaled by this function.
			_assert(!pager->MemoryDB);
			_assert(pager->DoNotSyncSpill == 0);
			pager->DoNotSyncSpill++;

			// This trick assumes that both the page-size and sector-size are an integer power of 2. It sets variable pg1 to the identifier of the first page of the sector pPg is located on.
			Pid pg1 = ((pg->ID - 1) & ~(pagePerSector - 1)) + 1; // First page of the sector pPg is located on.

			int pages = 0; // Number of pages starting at pg1 to journal
			Pid pageCount = pager->DBSize; // Total number of pages in database file
			if (pg->ID > pageCount)
				pages = (pg->ID - pg1) + 1;
			else if ((pg1 + pagePerSector - 1) > pageCount)
				pages = pageCount + 1 - pg1;
			else
				pages = pagePerSector;
			_assert(pages > 0);
			_assert(pg1 <= pg->ID);
			_assert((pg1 + pages) > pg->ID);

			bool needSync = false; // True if any page has PGHDR_NEED_SYNC
			for (int ii = 0; ii < pages && rc == RC_OK; ii++)
			{
				Pid id = pg1 + ii;
				PgHdr *page2;
				if (id == pg->ID || !pager->InJournal->Get(id))
				{
					if (id != MJ_PID(pager))
					{
						rc = pager->Acquire(id, &page2, false);
						if (rc == RC_OK)
						{
							rc = Pager_Write(page2);
							if (page2->Flags & PgHdr::PGHDR_NEED_SYNC)
								needSync = true;
							Unref(page2);
						}
					}
				}
				else if ((page2 = Pager_Lookup(pager, id)) != 0)
				{
					if (page2->Flags & PgHdr::PGHDR_NEED_SYNC)
						needSync = true;
					Unref(page2);
				}
			}

			// If the PGHDR_NEED_SYNC flag is set for any of the nPage pages starting at pg1, then it needs to be set for all of them. Because
			// writing to any of these nPage pages may damage the others, the journal file must contain sync()ed copies of all of them
			// before any of them can be written out to the database file.
			if (rc == RC_OK && needSync)
			{
				_assert(!pager->MemoryDB);
				for (int ii = 0; ii < pages; ii++)
				{
					PgHdr *page2 = Pager_Lookup(pager, pg1 + ii);
					if (page2)
					{
						page2->Flags |= PgHdr::PGHDR_NEED_SYNC;
						Unref(page2);
					}
				}
			}

			_assert(pager->DoNotSyncSpill == 1);
			pager->DoNotSyncSpill--;
		}
		else
			rc = Pager_Write(page);
		return rc;
	}

#ifndef DEBUG
	__device__ bool Pager::Iswriteable(IPage *pg)
	{
		return ((pg->Flags & PgHdr::PGHDR_DIRTY) != 0);
	}
#endif

	__device__ void Pager::DontWrite(IPage *pg)
	{
		Pager *pager = pg->Pager;
		if ((pg->Flags & PgHdr::PGHDR_DIRTY) && pager->Savepoints.length == 0)
		{
			PAGERTRACE("DONT_WRITE page %d of %d\n", pg->ID, PAGERID(pager));
			SysEx_IOTRACE("CLEAN %p %d\n", pager, pg->ID);
			pg->Flags |= PgHdr::PGHDR_DONT_WRITE;
			Pager_SetPageHash(pg);
		}
	}

	__device__ static RC pager_incr_changecounter(Pager *pager, int isDirectMode)
	{
		_assert(pager->State == Pager::PAGER_WRITER_CACHEMOD
			|| pager->State == Pager::PAGER_WRITER_DBMOD);
		_assert(AssertPagerState(pager));

		// Declare and initialize constant integer 'isDirect'. If the atomic-write optimization is enabled in this build, then isDirect
		// is initialized to the value passed as the isDirectMode parameter to this function. Otherwise, it is always set to zero.
		//
		// The idea is that if the atomic-write optimization is not enabled at compile time, the compiler can omit the tests of
		// 'isDirect' below, as well as the block enclosed in the "if( isDirect )" condition.
#ifndef ENABLE_ATOMIC_WRITE
#define DIRECT_MODE 0
		_assert(isDirectMode == 0);
#else
#define DIRECT_MODE isDirectMode
#endif
		RC rc = RC_OK;
		if (!pager->ChangeCountDone && _ALWAYS(pager->DBSize > 0))
		{
			_assert(!pager->TempFile && pager->File->Opened);

			// Open page 1 of the file for writing.
			PgHdr *pgHdr; // Reference to page 1
			rc = pager->Acquire(1, &pgHdr, false);
			_assert(pgHdr == nullptr || rc == RC_OK);

			// If page one was fetched successfully, and this function is not operating in direct-mode, make page 1 writable.  When not in 
			// direct mode, page 1 is always held in cache and hence the PagerGet() above is always successful - hence the ALWAYS on rc==SQLITE_OK.
			if (!DIRECT_MODE && _ALWAYS(rc == RC_OK))
				rc = Pager::Write(pgHdr);

			if (rc == RC_OK)
			{
				// Actually do the update of the change counter
				Pager_WriteChangecounter(pgHdr);

				// If running in direct mode, write the contents of page 1 to the file.
				if (DIRECT_MODE)
				{
					const void *buf;
					_assert(pager->DBFileSize > 0);
					CODEC2(void, buf, pager, pgHdr->Data, 1, 6, rc = RC_NOMEM);
					if (rc == RC_OK)
					{
						rc = pager->File->Write(buf, pager->PageSize, 0);
						pager->Stats[STAT_WRITE]++;
					}
					if (rc == RC_OK)
						pager->ChangeCountDone = true;
				}
				else
					pager->ChangeCountDone = true;
			}

			// Release the page reference.
			Pager::Unref(pgHdr);
		}
		return rc;
	}

	__device__ RC Pager::Sync()
	{
		RC rc = RC_OK;
		if (!NoSync)
		{
			_assert(!MemoryDB);
			rc = File->Sync(SyncFlags);
		}
		else if (File->Opened)
		{
			_assert(!MemoryDB);
			rc = File->FileControl(VFile::FCNTL_SYNC_OMITTED, 0);
			if (rc == RC_NOTFOUND)
				rc = RC_OK;
		}
		return rc;
	}

#pragma endregion

#pragma region Commit

	__device__ RC Pager::ExclusiveLock()
	{
		_assert(State == Pager::PAGER_WRITER_CACHEMOD || 
			State == Pager::PAGER_WRITER_DBMOD ||
			State == Pager::PAGER_WRITER_LOCKED);
		_assert(AssertPagerState(this));
		RC rc = RC_OK;
		if (!UseWal(this))
			rc = pager_wait_on_lock(this, VFile::LOCK_EXCLUSIVE);
		return rc;
	}

	__device__ RC Pager::CommitPhaseOne(const char *master, bool noSync)
	{
		_assert(State == Pager::PAGER_WRITER_LOCKED ||
			State == Pager::PAGER_WRITER_CACHEMOD ||
			State == Pager::PAGER_WRITER_DBMOD ||
			State == Pager::PAGER_ERROR);
		_assert(AssertPagerState(this));

		// If a prior error occurred, report that error again.
		if (_NEVER(ErrorCode)) return ErrorCode;

		PAGERTRACE("DATABASE SYNC: File=%s zMaster=%s nSize=%d\n", Filename, master, DBSize);

		// If no database changes have been made, return early.
		if (State < Pager::PAGER_WRITER_CACHEMOD) return RC_OK;

		RC rc = RC_OK;
		if (MemoryDB)
		{
			// If this is an in-memory db, or no pages have been written to, or this function has already been called, it is mostly a no-op.  However, any
			// backup in progress needs to be restarted.
			if (Backup) Backup->Restart();
		}
		else
		{
			if (UseWal(this))
			{
				PgHdr *list = PCache->DirtyList();
				PgHdr *pageOne = nullptr;
				if (list == nullptr)
				{
					// Must have at least one page for the WAL commit flag. Ticket [2d1a5c67dfc2363e44f29d9bbd57f] 2011-05-18
					rc = Acquire(1, &pageOne, false);
					list = pageOne;
					list->Dirty = nullptr;
				}
				_assert(rc == RC_OK);
				if (_ALWAYS(list))
					rc = PagerWalFrames(this, list, DBSize, 1);
				Unref(pageOne);
				if (rc == RC_OK)
					PCache->CleanAll();
			}
			else
			{
				// The following block updates the change-counter. Exactly how it does this depends on whether or not the atomic-update optimization
				// was enabled at compile time, and if this transaction meets the runtime criteria to use the operation: 
				//
				//    * The file-system supports the atomic-write property for blocks of size page-size, and 
				//    * This commit is not part of a multi-file transaction, and
				//    * Exactly one page has been modified and store in the journal file.
				//
				// If the optimization was not enabled at compile time, then the pager_incr_changecounter() function is called to update the change
				// counter in 'indirect-mode'. If the optimization is compiled in but is not applicable to this transaction, call sqlite3JournalCreate()
				// to make sure the journal file has actually been created, then call pager_incr_changecounter() to update the change-counter in indirect mode. 
				//
				// Otherwise, if the optimization is both enabled and applicable, then call pager_incr_changecounter() to update the change-counter
				// in 'direct' mode. In this case the journal file will never be created for this transaction.
#ifdef ENABLE_ATOMIC_WRITE
				PgHdr *pg;
				_assert(JournalFile->Opened ||
					JournalMode == IPager::JOURNALMODE_OFF ||
					JournalMode == IPager::JOURNALMODE_WAL);
				if (!master && JournalFile->Opened &&
					JournalOffset == JrnlBufferSize(this) &&
					DBSize >= DBOrigSize &&
					((pg = PCache->DirtyList()) == nullptr || pg->Dirty == nullptr))
				{
					// Update the db file change counter via the direct-write method. The following call will modify the in-memory representation of page 1 
					// to include the updated change counter and then write page 1 directly to the database file. Because of the atomic-write 
					// property of the host file-system, this is safe.
					rc = pager_incr_changecounter(this, true);
				}
				else
				{
					rc = VFile::JournalVFileCreate(JournalFile);
					if (rc == RC_OK)
						rc = pager_incr_changecounter(this, false);
				}
#else
				rc = pager_incr_changecounter(this, false);
#endif
				if (rc != RC_OK) goto commit_phase_one_exit;

				// Write the master journal name into the journal file. If a master journal file name has already been written to the journal file, 
				// or if zMaster is NULL (no master journal), then this call is a no-op.
				rc = WriteMasterJournal(this, master);
				if (rc != RC_OK) goto commit_phase_one_exit;

				// Sync the journal file and write all dirty pages to the database. If the atomic-update optimization is being used, this sync will not 
				// create the journal file or perform any real IO.
				//
				// Because the change-counter page was just modified, unless the atomic-update optimization is used it is almost certain that the
				// journal requires a sync here. However, in locking_mode=exclusive on a system under memory pressure it is just possible that this is 
				// not the case. In this case it is likely enough that the redundant xSync() call will be changed to a no-op by the OS anyhow. 
				rc = SyncJournal(this, false);
				if (rc != RC_OK) goto commit_phase_one_exit;

				rc = Pager_WritePagelist(this, PCache->DirtyList());
				if (rc != RC_OK)
				{
					_assert(rc != RC_IOERR_BLOCKED);
					goto commit_phase_one_exit;
				}
				PCache->CleanAll();

				// If the file on disk is smaller than the database image, use Pager_Truncate to grow the file here. This can happen if the database
				// image was extended as part of the current transaction and then the last page in the db image moved to the free-list. In this case the
				// last page is never written out to disk, leaving the database file undersized. Fix this now if it is the case.
				if (DBSize > DBFileSize)
				{
					Pid newID = DBSize - (DBSize == MJ_PID(this) ? 1 : 0);
					_assert(State == Pager::PAGER_WRITER_DBMOD);
					rc = Pager_Truncate(this, newID);
					if (rc != RC_OK) goto commit_phase_one_exit;
				}

				// Finally, sync the database file.
				if (!noSync)
					rc = Sync();
				SysEx_IOTRACE("DBSYNC %p\n", this);
			}
		}

commit_phase_one_exit:
		if (rc == RC_OK && !UseWal(this))
			State = Pager::PAGER_WRITER_FINISHED;
		return rc;
	}

	__device__ RC Pager::CommitPhaseTwo()
	{
		// This routine should not be called if a prior error has occurred. But if (due to a coding error elsewhere in the system) it does get
		// called, just return the same error code without doing anything.
		if (_NEVER(ErrorCode)) return ErrorCode;
		_assert(State == Pager::PAGER_WRITER_LOCKED ||
			State == Pager::PAGER_WRITER_FINISHED ||
			(UseWal(this) && State == Pager::PAGER_WRITER_CACHEMOD));
		_assert(AssertPagerState(this));

		// An optimization. If the database was not actually modified during this transaction, the pager is running in exclusive-mode and is
		// using persistent journals, then this function is a no-op.
		//
		// The start of the journal file currently contains a single journal header with the nRec field set to 0. If such a journal is used as
		// a hot-journal during hot-journal rollback, 0 changes will be made to the database file. So there is no need to zero the journal 
		// header. Since the pager is in exclusive mode, there is no need to drop any locks either.
		if (State == Pager::PAGER_WRITER_LOCKED &&
			ExclusiveMode &&
			JournalMode == IPager::JOURNALMODE_PERSIST)
		{
			_assert(JournalOffset == JOURNAL_HDR_SZ(this) || !JournalOffset);
			State = Pager::PAGER_READER;
			return RC_OK;
		}

		PAGERTRACE("COMMIT %d\n", PAGERID(this));
		RC rc = Pager_EndTransaction(this, SetMaster, true);
		return Pager_Error(this, rc);
	}

	__device__ RC Pager::Rollback()
	{
		PAGERTRACE("ROLLBACK %d\n", PAGERID(this));
		// PagerRollback() is a no-op if called in READER or OPEN state. If the pager is already in the ERROR state, the rollback is not attempted here. Instead, the error code is returned to the caller.
		_assert(AssertPagerState(this));
		if (State == Pager::PAGER_ERROR) return ErrorCode;
		if (State <= Pager::PAGER_READER) return RC_OK;

		RC rc = RC_OK;
		if (UseWal(this))
		{
			rc = Savepoint(IPager::SAVEPOINT_ROLLBACK, -1);
			RC rc2 = Pager_EndTransaction(this, SetMaster, false);
			if (rc == RC_OK) rc = rc2;
		}
		else if (!JournalFile->Opened || State == Pager::PAGER_WRITER_LOCKED)
		{
			PAGER state = State;
			rc = Pager_EndTransaction(this, false, false);
			if (!MemoryDB && state > Pager::PAGER_WRITER_LOCKED)
			{
				// This can happen using journal_mode=off. Move the pager to the error state to indicate that the contents of the cache may not be trusted. Any active readers will get SQLITE_ABORT.
				ErrorCode = RC_ABORT;
				State = Pager::PAGER_ERROR;
				return rc;
			}
		}
		else
			rc = Pager_Playback(this, false);

		_assert(State == Pager::PAGER_READER || rc != RC_OK);
		_assert(rc == RC_OK || rc == RC_FULL || rc == RC_NOMEM || (rc & 0xFF) == RC_IOERR);

		// If an error occurs during a ROLLBACK, we can no longer trust the pager cache. So call Pager_Error() on the way out to make any error persistent.
		return Pager_Error(this, rc);
	}

#pragma endregion

#pragma region Name4

	__device__ bool Pager::get_Readonly()
	{
		return ReadOnly;
	}

	__device__ int Pager::get_Refs()
	{
		return PCache->get_Refs();
	}

	__device__ int Pager::get_MemUsed()
	{
		int perPageSize = PageSize + ExtraBytes + sizeof(PgHdr) + 5 * sizeof(void *);
		return perPageSize * PCache->get_Pages() + _allocsize(this) + PageSize;
	}

	__device__ int Pager::get_PageRefs(IPage *page)
	{
		return PCache::get_PageRefs(page);
	}

#ifdef TEST
	__device__ int *Pager::get_Stats()
	{
		__shared__ static int a[11];
		a[0] = PCache->get_Refs();
		a[1] = PCache->get_Pages();
		a[2] = PCache->get_CacheSize();
		a[3] = (State == Pager::PAGER_OPEN ? -1 : (int)DBSize);
		a[4] = State;
		a[5] = ErrorCode;
		a[6] = Stats[STAT_HIT];
		a[7] = Stats[STAT_MISS];
		a[8] = 0;  // Used to be pager->nOvfl
		a[9] = Reads;
		a[10] = Stats[STAT_WRITE];
		return a;
	}
#endif

	__device__ void Pager::CacheStat(int dbStatus, bool reset, int *value)
	{
		//_assert(dbStatus == DBSTATUS_CACHE_HIT ||
		//	dbStatus == DBSTATUS_CACHE_MISS ||
		//	dbStatus == DBSTATUS_CACHE_WRITE);
		//_assert(DBSTATUS_CACHE_HIT + 1 == DBSTATUS_CACHE_MISS);
		//_assert(DBSTATUS_CACHE_HIT + 2 == DBSTATUS_CACHE_WRITE);
		//_assert(STAT_HIT == 0 &&
		//	STAT_MISS == 1 &&
		//	STAT_WRITE == 2);

		//*value += Stats[dbStatus - DBSTATUS_CACHE_HIT];
		//if (reset)
		//	Stats[dbStatus - DBSTATUS_CACHE_HIT] = 0;
	}

	__device__ bool Pager::get_MemoryDB()
	{
		return MemoryDB;
	}

	__device__ RC Pager::OpenSavepoint(int savepoints)
	{
		_assert(State >= Pager::PAGER_WRITER_LOCKED);
		_assert(AssertPagerState(this));

		RC rc = RC_OK;
		int currentSavepoints = Savepoints.length; // Current number of savepoints
		if (savepoints > currentSavepoints && UseJournal)
		{
			// Grow the Pager.aSavepoint array using realloc(). Return SQLITE_NOMEM if the allocation fails. Otherwise, zero the new portion in case a 
			// malloc failure occurs while populating it in the for(...) loop below.
			PagerSavepoint *newSavepoints = (PagerSavepoint *)_realloc(Savepoints, sizeof(PagerSavepoint) * savepoints); // New Pager.Savepoints array
			if (!newSavepoints)
				return RC_NOMEM;
			_memset(&newSavepoints[currentSavepoints], 0, (savepoints - currentSavepoints) * sizeof(PagerSavepoint));
			Savepoints = newSavepoints;

			// Populate the PagerSavepoint structures just allocated.
			for (int ii = currentSavepoints; ii < savepoints; ii++)
			{
				newSavepoints[ii].Orig = DBSize;
				newSavepoints[ii].Offset = (JournalFile->Opened && JournalOffset > 0 ? JournalOffset : JOURNAL_HDR_SZ(this));
				newSavepoints[ii].SubRecords = SubRecords;
				newSavepoints[ii].InSavepoint = new Bitvec(DBSize);
				if (!newSavepoints[ii].InSavepoint)
					return RC_NOMEM;
				if (UseWal(this))
					Wal->Savepoint(newSavepoints[ii].WalData);
				Savepoints.length = ii + 1;
			}
			_assert(Savepoints.length == savepoints);
			AssertTruncateConstraint(this);
		}
		return rc;
	}

	__device__ RC Pager::Savepoint(IPager::SAVEPOINT op, int savepoints)
	{
		_assert(op == IPager::SAVEPOINT_RELEASE || op == IPager::SAVEPOINT_ROLLBACK);
		_assert(savepoints >= 0 || op == IPager::SAVEPOINT_ROLLBACK);
		RC rc = ErrorCode;
		if (rc == RC_OK && savepoints < Savepoints.length)
		{
			// Figure out how many savepoints will still be active after this operation. Store this value in nNew. Then free resources associated 
			// with any savepoints that are destroyed by this operation.
			int newLength = savepoints + (op == IPager::SAVEPOINT_RELEASE ? 0 : 1); // Number of remaining savepoints after this op.
			for (int ii = newLength; ii < Savepoints.length; ii++)
				Bitvec::Destroy(Savepoints[ii].InSavepoint);
			Savepoints.length = newLength;

			// If this is a release of the outermost savepoint, truncate the sub-journal to zero bytes in size.
			if (op == IPager::SAVEPOINT_RELEASE)
			{
				if (newLength == 0 && SubJournalFile->Opened)
				{
					// Only truncate if it is an in-memory sub-journal.
					if (VFile::HasMemoryVFile(SubJournalFile))
					{
						rc = SubJournalFile->Truncate(0);
						_assert(rc == RC_OK);
					}
					SubRecords = 0;
				}
			}
			// Else this is a rollback operation, playback the specified savepoint. If this is a temp-file, it is possible that the journal file has
			// not yet been opened. In this case there have been no changes to the database file, so the playback operation can be skipped.
			else if (UseWal(this) || JournalFile->Opened)
			{
				PagerSavepoint *savepoint = (newLength == 0 ? nullptr : &Savepoints[newLength - 1]);
				rc = PagerPlaybackSavepoint(this, savepoint);
				_assert(rc != RC_DONE);
			}
		}
		return rc;
	}

	__device__ const char *Pager::get_Filename(bool nullIfMemDb)
	{
		return (nullIfMemDb && MemoryDB ? "" : Filename);
	}

	__device__ const VSystem *Pager::get_Vfs()
	{
		return Vfs;
	}

	__device__ VFile *Pager::get_File()
	{
		return File;
	}

	__device__ const char *Pager::get_Journalname()
	{
		return Journal;
	}

	__device__ bool Pager::get_NoSync()
	{
		return NoSync;
	}

#ifdef HAS_CODEC
	__device__ void Pager::SetCodec(void *(*codec)(void *,void *, Pid, int), void (*codecSizeChange)(void *, int, int), void (*codecFree)(void *), void *codecArg)
	{
		if (CodecFree) CodecFree(Codec);
		Codec = (MemoryDB ? nullptr : codec);
		CodecSizeChange = codecSizeChange;
		CodecFree = codecFree;
		CodecArg = codecArg;
		pagerReportSize(this);
	}

	__device__ void *Pager::GetCodec()
	{
		return Codec;
	}
#endif

#ifndef OMIT_AUTOVACUUM
	__device__ RC Pager::Movepage(IPage *pg, Pid id, bool isCommit)
	{
		_assert(pg->Refs > 0);
		_assert(State == Pager::PAGER_WRITER_CACHEMOD ||
			State == Pager::PAGER_WRITER_DBMOD);
		_assert(AssertPagerState(this));

		// In order to be able to rollback, an in-memory database must journal the page we are moving from.
		RC rc;
		if (MemoryDB)
		{
			rc = Pager::Write(pg);
			if (rc) return rc;
		}

		// If the page being moved is dirty and has not been saved by the latest savepoint, then save the current contents of the page into the 
		// sub-journal now. This is required to handle the following scenario:
		//
		//   BEGIN;
		//     <journal page X, then modify it in memory>
		//     SAVEPOINT one;
		//       <Move page X to location Y>
		//     ROLLBACK TO one;
		//
		// If page X were not written to the sub-journal here, it would not be possible to restore its contents when the "ROLLBACK TO one"
		// statement were is processed.
		//
		// SubjournalPage() may need to allocate space to store pPg->pgno into one or more savepoint bitvecs. This is the reason this function
		// may return SQLITE_NOMEM.
		if (pg->Flags & PgHdr::PGHDR_DIRTY &&
			SubjRequiresPage(pg) &&
			(rc = SubjournalPage(pg)) != RC_OK)
			return rc;

		PAGERTRACE("MOVE %d page %d (needSync=%d) moves to %d\n", PAGERID(this), pg->ID, (pg->Flags & PgHdr::PGHDR_NEED_SYNC ? 1 : 0), id);
		SysEx_IOTRACE("MOVE %p %d %d\n", this, pg->ID, id);

		// If the journal needs to be sync()ed before page pPg->pgno can be written to, store pPg->pgno in local variable needSyncPgno.
		//
		// If the isCommit flag is set, there is no need to remember that the journal needs to be sync()ed before database page pPg->pgno 
		// can be written to. The caller has already promised not to write to it.
		Pid needSyncID = 0; // Old value of pPg->pgno, if sync is required
		if ((pg->Flags & PgHdr::PGHDR_NEED_SYNC) && !isCommit)
		{
			needSyncID = pg->ID;
			_assert(JournalMode == IPager::JOURNALMODE_OFF || PageInJournal(pg) || pg->ID > DBOrigSize);
			_assert(pg->Flags & PgHdr::PGHDR_DIRTY);
		}

		// If the cache contains a page with page-number pgno, remove it from its hash chain. Also, if the PGHDR_NEED_SYNC flag was set for 
		// page pgno before the 'move' operation, it needs to be retained for the page moved there.
		pg->Flags &= ~PgHdr::PGHDR_NEED_SYNC;
		PgHdr *pgOld = Pager_Lookup(this, id); // The page being overwritten.
		_assert(!pgOld || pgOld->Refs == 1);
		if (pgOld)
		{
			pg->Flags |= (pgOld->Flags & PgHdr::PGHDR_NEED_SYNC);
			if (MemoryDB)
			{
				// Do not discard pages from an in-memory database since we might need to rollback later.  Just move the page out of the way.
				PCache::Move(pgOld, DBSize + 1);
			}
			else
				PCache::Drop(pgOld);
		}

		Pid origID = pg->ID; // The original page number
		PCache::Move(pg, id);
		PCache::MakeDirty(pg);

		// For an in-memory database, make sure the original page continues to exist, in case the transaction needs to roll back.  Use pPgOld
		// as the original page since it has already been allocated.
		if (MemoryDB)
		{
			_assert(pgOld != nullptr);
			PCache::Move(pgOld, origID);
			Unref(pgOld);
		}

		if (needSyncID)
		{
			// If needSyncPgno is non-zero, then the journal file needs to be sync()ed before any data is written to database file page needSyncPgno.
			// Currently, no such page exists in the page-cache and the "is journaled" bitvec flag has been set. This needs to be remedied by
			// loading the page into the pager-cache and setting the PGHDR_NEED_SYNC flag.
			//
			// If the attempt to load the page into the page-cache fails, (due to a malloc() or IO failure), clear the bit in the pInJournal[]
			// array. Otherwise, if the page is loaded and written again in this transaction, it may be written to the database file before
			// it is synced into the journal file. This way, it may end up in the journal file twice, but that is not a problem.
			PgHdr *pgHdr;
			rc = Acquire(needSyncID, &pgHdr, false);
			if (rc != RC_OK)
			{
				if (needSyncID <= DBOrigSize)
				{
					_assert(TmpSpace != nullptr);
					InJournal->Clear(needSyncID, TmpSpace);
				}
				return rc;
			}
			pgHdr->Flags |= PgHdr::PGHDR_NEED_SYNC;
			PCache::MakeDirty(pgHdr);
			Unref(pgHdr);
		}
		return RC_OK;
	}
#endif

	__device__ void *Pager::GetData(IPage *pg)
	{
		_assert(pg->Refs > 0 || pg->Pager->MemoryDB);
		return pg->Data;
	}

	__device__ void *Pager::GetExtra(IPage *pg)
	{
		return pg->Extra;
	}

	__device__ IPager::LOCKINGMODE Pager::LockingMode(IPager::LOCKINGMODE mode)
	{
		_assert(mode == IPager::LOCKINGMODE_QUERY ||
			mode == IPager::LOCKINGMODE_NORMAL ||
			mode == IPager::LOCKINGMODE_EXCLUSIVE);
		_assert(IPager::LOCKINGMODE_QUERY < 0);
		_assert(IPager::LOCKINGMODE_NORMAL >= 0 && IPager::LOCKINGMODE_EXCLUSIVE >= 0);
		_assert(ExclusiveMode || Wal->get_HeapMemory() == 0);
		if (mode >= 0 && !TempFile && !Wal->get_HeapMemory())
			ExclusiveMode = mode;
		return ExclusiveMode;
	}

	__device__ IPager::JOURNALMODE Pager::SetJournalMode(IPager::JOURNALMODE mode)
	{
#ifdef _DEBUG
		// The PrintPagerState() routine is intended to be used by the debugger only.  We invoke it once here to suppress a compiler warning.
		PrintPagerState(this);
#endif

		// The mode parameter is always valid
		_assert(mode == IPager::JOURNALMODE_DELETE ||
			mode == IPager::JOURNALMODE_TRUNCATE ||
			mode == IPager::JOURNALMODE_PERSIST ||
			mode == IPager::JOURNALMODE_OFF ||
			mode == IPager::JOURNALMODE_WAL ||
			mode == IPager::JOURNALMODE_JMEMORY);

		// This routine is only called from the OP_JournalMode opcode, and the logic there will never allow a temporary file to be changed to WAL mode.
		_assert(!TempFile || mode != IPager::JOURNALMODE_WAL);

		// Do allow the journalmode of an in-memory database to be set to anything other than MEMORY or OFF
		IPager::JOURNALMODE old = JournalMode; // Prior journalmode
		if (MemoryDB)
		{
			_assert(old == IPager::JOURNALMODE_JMEMORY || old == IPager::JOURNALMODE_OFF);
			if (mode != IPager::JOURNALMODE_JMEMORY && mode != IPager::JOURNALMODE_OFF)
				mode = old;
		}

		if (mode != old)
		{
			// Change the journal mode
			_assert(State != Pager::PAGER_ERROR);
			JournalMode = mode;

			// When transistioning from TRUNCATE or PERSIST to any other journal mode except WAL, unless the pager is in locking_mode=exclusive mode,
			// delete the journal file.
			_assert((IPager::JOURNALMODE_TRUNCATE & 5) == 1);
			_assert((IPager::JOURNALMODE_PERSIST & 5) == 1);
			_assert((IPager::JOURNALMODE_DELETE & 5) == 0);
			_assert((IPager::JOURNALMODE_JMEMORY & 5) == 4);
			_assert((IPager::JOURNALMODE_OFF & 5) == 0);
			_assert((IPager::JOURNALMODE_WAL & 5) == 5);

			_assert(File->Opened || ExclusiveMode);
			if (!ExclusiveMode && (old & 5) == 1 && (mode & 1) == 0)
			{
				// In this case we would like to delete the journal file. If it is not possible, then that is not a problem. Deleting the journal file
				// here is an optimization only.
				//
				// Before deleting the journal file, obtain a RESERVED lock on the database file. This ensures that the journal file is not deleted
				// while it is in use by some other client.
				JournalFile->Close();
				if (Lock >= VFile::LOCK_RESERVED)
					Vfs->Delete(Journal, false);
				else
				{
					RC rc = RC_OK;
					PAGER state = State;
					_assert(state == Pager::PAGER_OPEN || state == Pager::PAGER_READER);
					if (state == Pager::PAGER_OPEN)
						rc = SharedLock();
					if (State == Pager::PAGER_READER)
					{
						_assert(rc == RC_OK);
						rc = PagerLockDb(this, VFile::LOCK_RESERVED);
					}
					if (rc == RC_OK)
						Vfs->Delete(Journal, false);
					if (rc == RC_OK && state == Pager::PAGER_READER)
						PagerUnlockDb(this, VFile::LOCK_SHARED);
					else if (state == Pager::PAGER_OPEN)
						Pager_Unlock(this);
					_assert(state == State);
				}
			}
		}

		// Return the new journal mode
		return JournalMode;
	}

	__device__ IPager::JOURNALMODE Pager::GetJournalMode()
	{
		return JournalMode;
	}

	__device__ bool Pager::OkToChangeJournalMode()
	{
		_assert(AssertPagerState(this));
		if (State >= Pager::PAGER_WRITER_CACHEMOD) return false;
		return (_NEVER(JournalFile->Opened && JournalOffset > 0) ? false : true);
	}

	__device__ int64 Pager::SetJournalSizeLimit(int64 limit)
	{
		if (limit >= -1)
		{
			JournalSizeLimit = limit;
			Wal->Limit(limit);
		}
		return JournalSizeLimit;
	}

	//__device__ IBackup **Pager::BackupPtr()
	//{
	//	return &Backup;
	//}

#ifndef OMIT_VACUUM
	__device__ void Pager::ClearCache()
	{
		if (!MemoryDB && !TempFile)
			Pager_Reset(this);
	}
#endif

#pragma endregion

#pragma region Wal
#ifndef OMIT_WAL

	__device__ RC Pager::Checkpoint(int mode, int *logs, int *checkpoints)
	{
		RC rc = RC_OK;
		if (Wal)
			rc = Wal->Checkpoint(mode, 
			BusyHandler, BusyHandlerArg, 
			CheckpointSyncFlags, PageSize, (uint8 *)TmpSpace, 
			logs, checkpoints);
		return rc;
	}

	__device__ RC Pager::WalCallback()
	{
		return Wal->Callback();
	}

	__device__ bool Pager::WalSupported()
	{
		return ExclusiveMode || false;
		//const sqlite3_io_methods *pMethods = pager->File->xShmMap;
		//return ExclusiveMode || (pMethods->iVersion>=2 && pMethods->xShmMap);
	}

	__device__ static RC PagerExclusiveLock(Pager *pager)
	{
		_assert(pager->Lock == VFile::LOCK_SHARED || pager->Lock == VFile::LOCK_EXCLUSIVE);
		RC rc = PagerLockDb(pager, VFile::LOCK_EXCLUSIVE);
		if (rc != RC_OK) // If the attempt to grab the exclusive lock failed, release the pending lock that may have been obtained instead.
			PagerUnlockDb(pager, VFile::LOCK_SHARED);
		return rc;
	}

	__device__ static RC pagerOpenWal(Pager *pager)
	{
		_assert(pager->Wal == nullptr && !pager->TempFile);
		_assert(pager->Lock == VFile::LOCK_SHARED || pager->Lock == VFile::LOCK_EXCLUSIVE);

		// If the pager is already in exclusive-mode, the WAL module will use heap-memory for the wal-index instead of the VFS shared-memory 
		// implementation. Take the exclusive lock now, before opening the WAL file, to make sure this is safe.
		RC rc = RC_OK;
		if (pager->ExclusiveMode)
			rc = PagerExclusiveLock(pager);

		// Open the connection to the log file. If this operation fails, (e.g. due to malloc() failure), return an error code.
		if (rc == RC_OK)
			rc = pager->Vfs->Open(pager->File, pager->WalName, pager->ExclusiveMode, pager->JournalSizeLimit, &pager->Wal);

		return rc;
	}

	__device__ RC Pager::OpenWal(bool *opened)
	{
		_assert(AssertPagerState(this));
		_assert(State == Pager::PAGER_OPEN || opened);
		_assert(State == Pager::PAGER_READER || !opened);
		_assert(opened == nullptr || *opened == false);
		_assert(opened != nullptr || (!TempFile && !Wal));

		RC rc = RC_OK;
		if (!TempFile && !Wal)
		{
			if (!WalSupported()) return RC_CANTOPEN;

			// Close any rollback journal previously open
			JournalFile->Close();

			rc = pagerOpenWal(this);
			if (rc == RC_OK)
			{
				JournalMode = IPager::JOURNALMODE_WAL;
				State = Pager::PAGER_OPEN;
			}
		}
		else
			*opened = true;

		return rc;
	}

	__device__ RC Pager::CloseWal()
	{
		_assert(JournalMode == IPager::JOURNALMODE_WAL);

		// If the log file is not already open, but does exist in the file-system, it may need to be checkpointed before the connection can switch to
		// rollback mode. Open it now so this can happen.
		RC rc = RC_OK;
		if (!Wal)
		{
			rc = PagerLockDb(this, VFile::LOCK_SHARED);
			int logexists = 0;
			if (rc == RC_OK)
				rc = Vfs->Access(WalName, VSystem::ACCESS::EXISTS, &logexists);
			if (rc == RC_OK && logexists)
				rc = pagerOpenWal(this);
		}

		// Checkpoint and close the log. Because an EXCLUSIVE lock is held on the database file, the log and log-summary files will be deleted.
		if (rc == RC_OK && Wal)
		{
			rc = PagerExclusiveLock(this);
			if (rc == RC_OK)
			{
				rc = Wal->Close(CheckpointSyncFlags, PageSize, (uint8 *)TmpSpace);
				Wal = nullptr;
			}
		}
		return rc;
	}

#endif
#pragma endregion

#pragma region Misc

#ifdef ENABLE_ZIPVFS
	__device__ int Pager::WalFramesize()
	{
		_assert(State == Pager::PAGER_READER);
		return Wal->Frames();
	}
#endif

#if defined(HAS_CODEC) && !defined(OMIT_WAL)
	__device__ void *Pager::get_Codec(IPage *pg)
	{
		void *data = nullptr;
		CODEC2(void, data, pg->Pager, pg->Data, pg->ID, 6, return nullptr);
		return data;
	}
#endif

#pragma endregion
}