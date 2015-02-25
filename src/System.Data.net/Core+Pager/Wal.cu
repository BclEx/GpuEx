// wal.c
#include "Core+Pager.cu.h"
#include <stddef.h> 

#ifndef OMIT_WAL
namespace Core
{

#ifdef _DEBUG
	bool WalTrace = false;
#define WALTRACE(X) if (WalTrace) printf(X)
#else
#define WALTRACE(X)
#endif

#pragma region Struct

	//typedef struct WalIterator WalIterator;
	//typedef struct WalCheckpointInfo WalCheckpointInfo;

#define WAL_MAX_VERSION 3007000
#define WALINDEX_MAX_VERSION 3007000
#define WAL_WRITE_LOCK 0
#define WAL_ALL_BUT_WRITE 1
#define WAL_CKPT_LOCK 1
#define WAL_RECOVER_LOCK 2
#define WAL_READ_LOCK(I) (3 + (I))
#define WAL_NREADER (VFile::SHM_MAX - 3)

	struct WalCheckpointInfo
	{
		uint32 Backfills;               // Number of WAL frames backfilled into DB
		uint32 ReadMarks[WAL_NREADER];  // Reader marks
	};

#define READMARK_NOT_USED 0xffffffff
#define WALINDEX_LOCK_OFFSET (sizeof(Wal::IndexHeader)*2 + sizeof(WalCheckpointInfo))
#define WALINDEX_LOCK_RESERVED 16
#define WALINDEX_HDR_SIZE (WALINDEX_LOCK_OFFSET+WALINDEX_LOCK_RESERVED)
#define WAL_FRAME_HDRSIZE 24
#define WAL_HDRSIZE 32
#define WAL_MAGIC 0x377f0682
#define walFrameOffset(frame, sizePage) (WAL_HDRSIZE + ((frame) - 1) * (int64)((sizePage) + WAL_FRAME_HDRSIZE))

	typedef uint16 ht_slot;

	struct WalIterator
	{
		int Prior;						// Last result returned from the iterator
		int SegmentsLength;             // Number of entries in aSegment[]
		struct Segment
		{
			int Next;					// Next slot in aIndex[] not yet returned
			ht_slot *Indexs;            // i0, i1, i2... such that aPgno[iN] ascend
			uint32 *IDs;				// Array of page numbers.
			int Entrys;                 // Nr. of entries in aPgno[] and aIndex[]
			int Zero;					// Frame number associated with aPgno[0]
		} Segments[1];			// One for every 32KB page in the wal-index
	};

#define HASHTABLE_NPAGE      4096                 // Must be power of 2
#define HASHTABLE_HASH_1     383                  // Should be prime
#define HASHTABLE_NSLOT      (HASHTABLE_NPAGE*2)  // Must be a power of 2
#define HASHTABLE_NPAGE_ONE (HASHTABLE_NPAGE - (WALINDEX_HDR_SIZE / sizeof(uint32)))

#define WALINDEX_PGSZ   (sizeof(ht_slot) * HASHTABLE_NSLOT + HASHTABLE_NPAGE * sizeof(uint32))

#pragma endregion

#pragma region Alloc

	__device__ static RC walIndexPage(Wal *wal, Pid id, volatile Pid **idOut)
	{
		// Enlarge the pWal->apWiData[] array if required
		if (wal->WiData.length <= id)
		{
			int bytes = sizeof(uint32 *) * (id + 1);
			volatile uint32 **newWiData = (volatile uint32 **)SysEx::Realloc((void *)wal->WiData, bytes);
			if (!newWiData)
			{
				*idOut = nullptr;
				return RC_NOMEM;
			}
			_memset((void *)&newWiData[wal->WiData.length], 0, sizeof(uint32 *) * (id + 1 - wal->WiData.length));
			__arraySet(wal->WiData, newWiData, id + 1);
		}

		// Request a pointer to the required page from the VFS
		RC rc = RC_OK;
		if (wal->WiData[id] == 0)
		{
			if (wal->ExclusiveMode_ == Wal::MODE_HEAPMEMORY)
			{
				wal->WiData[id] = (uint32 volatile *)_alloc(WALINDEX_PGSZ, true);
				if (!wal->WiData[id]) rc = RC_NOMEM;
			}
			else
			{
				rc = wal->DBFile->ShmMap(id, WALINDEX_PGSZ, wal->WriteLock, (void volatile **)&wal->WiData[id]);
				if (rc == RC_READONLY)
				{
					wal->ReadOnly |= Wal::RDONLY_RDONLY;
					rc = RC_OK;
				}
			}
		}

		*idOut = wal->WiData[id];
		_assert(id == 0 || *idOut || rc != RC_OK);
		return rc;
	}

	__device__ static volatile WalCheckpointInfo *walCkptInfo(Wal *wal)
	{
		_assert(wal->WiData.length > 0 && wal->WiData[0]);
		return (volatile WalCheckpointInfo *) & (wal->WiData[0][sizeof(Wal::IndexHeader) / 2]);
	}

	__device__ static volatile Wal::IndexHeader *walIndexHeader(Wal *wal)
	{
		_assert(wal->WiData.length > 0 && wal->WiData[0]);
		return (volatile Wal::IndexHeader *)wal->WiData[0];
	}

#pragma endregion

#pragma region Checksum

#define BYTESWAP32(x) ((((x)&0x000000FF)<<24) + (((x)&0x0000FF00)<<8) + (((x)&0x00FF0000)>>8) + (((x)&0xFF000000)>>24))

	__device__ static void walChecksumBytes(bool nativeChecksum, uint8 *b, int length, const uint32 *checksum, uint32 *checksumOut)
	{
		uint32 s1, s2;
		if (checksum)
		{
			s1 = checksum[0];
			s2 = checksum[1];
		}
		else
			s1 = s2 = 0;

		_assert(length >= 8);
		_assert((length & 0x00000007) == 0);

		uint32 *data = (uint32 *)b;
		uint32 *end = (uint32 *)&b[length];
		if (nativeChecksum)
			do
			{
				s1 += *data++ + s2;
				s2 += *data++ + s1;
			} while (data < end);
		else
			do
			{
				s1 += BYTESWAP32(data[0]) + s2;
				s2 += BYTESWAP32(data[1]) + s1;
				data += 2;
			} while (data < end);

			checksumOut[0] = s1;
			checksumOut[1] = s2;
	}

	__device__ static void walShmBarrier(Wal *wal)
	{
		if (wal->ExclusiveMode_ != Wal::MODE_HEAPMEMORY)
			wal->DBFile->ShmBarrier();
	}

	__device__ static void walIndexWriteHdr(Wal *wal)
	{
		volatile Wal::IndexHeader *header = walIndexHeader(wal);
		const int checksumIdx = offsetof(Wal::IndexHeader, Checksum);

		_assert(wal->WriteLock);
		wal->Header.IsInit = true;
		wal->Header.Version = WALINDEX_MAX_VERSION;
		walChecksumBytes(1, (uint8 *)&wal->Header, checksumIdx, 0, wal->Header.Checksum);
		_memcpy((void *)&header[1], (void *)&wal->Header, sizeof(Wal::IndexHeader));
		walShmBarrier(wal);
		_memcpy((void *)&header[0], (void *)&wal->Header, sizeof(Wal::IndexHeader));
	}

	__device__ static void walEncodeFrame(Wal *wal, Pid id, uint32 truncate, uint8 *data, uint8 *frame)
	{
		uint32 *checksum = wal->Header.FrameChecksum;
		_assert(WAL_FRAME_HDRSIZE == 24);
		ConvertEx::Put4(&frame[0], id);
		ConvertEx::Put4(&frame[4], truncate);
		_memcpy(&frame[8], (uint8 *)wal->Header.Salt, 8);

		bool nativeChecksum = (wal->Header.BigEndianChecksum == TYPE_BIGENDIAN); // True for native byte-order checksums
		walChecksumBytes(nativeChecksum, frame, 8, checksum, checksum);
		walChecksumBytes(nativeChecksum, data, wal->SizePage, checksum, checksum);

		ConvertEx::Put4(&frame[16], checksum[0]);
		ConvertEx::Put4(&frame[20], checksum[1]);
	}

	__device__ static int walDecodeFrame(Wal *wal, Pid *idOut, uint32 *truncateOut, uint8 *data, uint8 *frame)
	{
		uint32 *checksum = wal->Header.FrameChecksum;
		_assert(WAL_FRAME_HDRSIZE == 24);

		// A frame is only valid if the salt values in the frame-header match the salt values in the wal-header. 
		if (_memcmp(&wal->Header.Salt, &frame[8], 8) != 0)
			return false;

		// A frame is only valid if the page number is creater than zero.
		Pid id = ConvertEx::Get4(&frame[0]); // Page number of the frame
		if (id == 0)
			return false;

		// A frame is only valid if a checksum of the WAL header, all prior frams, the first 16 bytes of this frame-header, 
		// and the frame-data matches the checksum in the last 8 bytes of this frame-header.
		bool nativeChecksum = (wal->Header.BigEndianChecksum == TYPE_BIGENDIAN); // True for native byte-order checksums
		walChecksumBytes(nativeChecksum, frame, 8, checksum, checksum);
		walChecksumBytes(nativeChecksum, data, wal->SizePage, checksum, checksum);
		if (checksum[0] != ConvertEx::Get4(&frame[16]) || checksum[1]!=ConvertEx::Get4(&frame[20])) // Checksum failed.
			return false;

		// If we reach this point, the frame is valid.  Return the page number and the new database size.
		*idOut = id;
		*truncateOut = ConvertEx::Get4(&frame[4]);
		return true;
	}

#pragma endregion

#pragma region Lock

#if defined(TEST) && defined(_DEBUG)
	__device__ static const char *walLockName(int lockIdx)
	{
		if (lockIdx == WAL_WRITE_LOCK)
			return "WRITE-LOCK";
		else if (lockIdx == WAL_CKPT_LOCK)
			return "CKPT-LOCK";
		else if (lockIdx == WAL_RECOVER_LOCK)
			return "RECOVER-LOCK";
		else
		{
			static char name[15];
			__snprintf(name, sizeof(name), "READ-LOCK[%d]", lockIdx - WAL_READ_LOCK(0));
			return name;
		}
	}
#endif

	__device__ static RC walLockShared(Wal *wal, int lockIdx)
	{
		if (wal->ExclusiveMode_) return RC_OK;
		RC rc = wal->DBFile->ShmLock(lockIdx, 1, (VFile::SHM)(VFile::SHM_LOCK | VFile::SHM_SHARED));
		WALTRACE("WAL%p: acquire SHARED-%s %s\n", wal, walLockName(lockIdx), rc ? "failed" : "ok");
		return rc;
	}

	__device__ static void walUnlockShared(Wal *wal, int lockIdx)
	{
		if (wal->ExclusiveMode_) return;
		wal->DBFile->ShmLock(lockIdx, 1, (VFile::SHM)(VFile::SHM_UNLOCK | VFile::SHM_SHARED));
		WALTRACE("WAL%p: release SHARED-%s\n", wal, walLockName(lockIdx));
	}

	__device__ static RC walLockExclusive(Wal *wal, int lockIdx, int n)
	{
		if (wal->ExclusiveMode_) return RC_OK;
		RC rc = wal->DBFile->ShmLock(lockIdx, n, (VFile::SHM)(VFile::SHM_LOCK | VFile::SHM_EXCLUSIVE));
		WALTRACE("WAL%p: acquire EXCLUSIVE-%s cnt=%d %s\n", wal, walLockName(lockIdx), n, rc ? "failed" : "ok");
		return rc;
	}

	__device__ static void walUnlockExclusive(Wal *wal, int lockIdx, int n)
	{
		if (wal->ExclusiveMode_) return;
		wal->DBFile->ShmLock(lockIdx, n, (VFile::SHM)(VFile::SHM_UNLOCK | VFile::SHM_EXCLUSIVE));
		WALTRACE("WAL%p: release EXCLUSIVE-%s cnt=%d\n", wal, walLockName(lockIdx), n);
	}

#pragma endregion

#pragma region Hash

	__device__ static int walHash(uint id)
	{
		_assert(id > 0);
		_assert((HASHTABLE_NSLOT & (HASHTABLE_NSLOT-1)) == 0);
		return (id * HASHTABLE_HASH_1) & (HASHTABLE_NSLOT-1);
	}

	__device__ static int walNextHash(int priorHash)
	{
		return (priorHash + 1) & (HASHTABLE_NSLOT - 1);
	}

	__device__ static RC walHashGet(Wal *wal, Pid id, volatile ht_slot **hashOut, volatile Pid **idsOut, uint32 *zeroOut)
	{
		volatile Pid *ids;
		RC rc = walIndexPage(wal, id, &ids);
		_assert(rc == RC_OK || id > 0);

		if (rc == RC_OK)
		{
			Pid zero;
			volatile ht_slot *hash = (volatile ht_slot *)&ids[HASHTABLE_NPAGE];
			if (id == 0)
			{
				ids = &ids[WALINDEX_HDR_SIZE / sizeof(Pid)];
				zero = 0;
			}
			else
				zero = HASHTABLE_NPAGE_ONE + (id - 1) * HASHTABLE_NPAGE;

			*idsOut = &ids[-1];
			*hashOut = hash;
			*zeroOut = zero;
		}
		return rc;
	}

	__device__ static int walFramePage(uint32 frame)
	{
		int hash = (frame + HASHTABLE_NPAGE-HASHTABLE_NPAGE_ONE - 1) / HASHTABLE_NPAGE;
		_assert((hash == 0 || frame > HASHTABLE_NPAGE_ONE) && 
			(hash >= 1 || frame <= HASHTABLE_NPAGE_ONE) && 
			(hash <= 1 || frame > (HASHTABLE_NPAGE_ONE + HASHTABLE_NPAGE)) && 
			(hash >= 2 || frame <= HASHTABLE_NPAGE_ONE + HASHTABLE_NPAGE) && 
			(hash <= 2 || frame > (HASHTABLE_NPAGE_ONE + 2 * HASHTABLE_NPAGE)));
		return hash;
	}

	__device__ static uint32 walFramePgno(Wal *wal, uint32 frame)
	{
		int hash = walFramePage(frame);
		if (hash == 0)
			return wal->WiData[0][WALINDEX_HDR_SIZE / sizeof(uint32) + frame - 1];
		return wal->WiData[hash][(frame - 1 - HASHTABLE_NPAGE_ONE) % HASHTABLE_NPAGE];
	}

	__device__ static void walCleanupHash(Wal *wal)
	{
		_assert(wal->WriteLock);
		ASSERTCOVERAGE(wal->Header.MaxFrame == HASHTABLE_NPAGE_ONE - 1);
		ASSERTCOVERAGE(wal->Header.MaxFrame == HASHTABLE_NPAGE_ONE);
		ASSERTCOVERAGE(wal->Header.MaxFrame == HASHTABLE_NPAGE_ONE + 1);

		if (wal->Header.MaxFrame == 0) return;

		// Obtain pointers to the hash-table and page-number array containing the entry that corresponds to frame pWal->hdr.mxFrame. It is guaranteed
		// that the page said hash-table and array reside on is already mapped.
		_assert(wal->WiData.length > walFramePage(wal->Header.MaxFrame));
		_assert(wal->WiData[walFramePage(wal->Header.MaxFrame)] != 0);
		volatile ht_slot *hash = nullptr; // Pointer to hash table to clear
		volatile Pid *ids = nullptr; // Page number array for hash table
		int zero = 0; // frame == (aHash[x]+iZero)
		walHashGet(wal, walFramePage(wal->Header.MaxFrame), &hash, &ids, &zero);

		// Zero all hash-table entries that correspond to frame numbers greater than pWal->hdr.mxFrame.
		int limit = wal->Header.MaxFrame - zero; // Zero values greater than this
		_assert(limit > 0);
		for (int i = 0; i < HASHTABLE_NSLOT; i++)
			if (hash[i] > limit)
				hash[i] = 0;

		// Zero the entries in the aPgno array that correspond to frames with frame numbers greater than pWal->hdr.mxFrame. 
		int bytes = (int)((char *)hash - (char *)&ids[limit + 1]); // Number of bytes to zero in aPgno[]
		_memset((void *)&ids[limit + 1], 0, bytes);

#ifdef ENABLE_EXPENSIVE_ASSERT
		// Verify that the every entry in the mapping region is still reachable via the hash table even after the cleanup.
		int key; // Hash key
		if (limit)
			for (int i = 1; i <= limit; i++)
			{
				for (key = walHash(ids[i]); hash[key]; key = walNextHash(key))
					if (hash[key] == i) break;
				_assert(hash[key] == i);
			}
#endif
	}

#pragma endregion

#pragma region Index

	__device__ static RC walIndexAppend(Wal *wal, uint32 frame, Pid id)
	{
		volatile ht_slot *hash = nullptr; // Hash table
		volatile Pid *ids = nullptr; // Page number array
		uint zero = 0; // One less than frame number of aPgno[1]
		RC rc = walHashGet(wal, walFramePage(frame), &hash, &ids, &zero);

		// Assuming the wal-index file was successfully mapped, populate the page number array and hash table entry.
		if (rc == RC_OK)
		{
			int idx = frame - zero; // Value to write to hash-table slot
			_assert(idx <= HASHTABLE_NSLOT / 2 + 1 );

			// If this is the first entry to be added to this hash-table, zero the entire hash table and aPgno[] array before proceding. 
			if (idx == 1)
			{
				int bytes = (int)((uint8 *)&hash[HASHTABLE_NSLOT] - (uint8 *)&ids[1]);
				_memset((void*)&ids[1], 0, bytes);
			}

			// If the entry in aPgno[] is already set, then the previous writer must have exited unexpectedly in the middle of a transaction (after
			// writing one or more dirty pages to the WAL to free up memory). Remove the remnants of that writers uncommitted transaction from 
			// the hash-table before writing any new entries.
			if (ids[idx])
			{
				walCleanupHash(wal);
				_assert(!ids[idx]);
			}

			// Write the aPgno[] array entry and the hash-table slot.
			int collide = idx; // Number of hash collisions
			int key; // Hash table key
			for (key = walHash(id); hash[key]; key = walNextHash(key))
				if ((collide--) == 0) return SysEx_CORRUPT_BKPT;
			ids[idx] = id;
			hash[key] = (ht_slot)idx;

#ifdef ENABLE_EXPENSIVE_ASSERT
			// Verify that the number of entries in the hash table exactly equals the number of entries in the mapping region.
			{
				int entry = 0; // Number of entries in the hash table
				for (int i = 0; i < HASHTABLE_NSLOT; i++) { if (hash[i]) entry++; }
				_assert(entry == idx);
			}

			// Verify that the every entry in the mapping region is reachable via the hash table.  This turns out to be a really, really expensive
			// thing to check, so only do this occasionally - not on every iteration.
			if ((idx & 0x3ff) == 0)
				for (int i = 1; i <= idx; i++)
				{
					for (key = walHash(ids[i]); hash[key]; key = walNextHash(key))
						if (hash[key] == i) break;
					_assert(hash[key] == i);
				}
#endif
		}

		return rc;
	}

	__device__ static int walIndexRecover(Wal *wal)
	{
		uint32 frameChecksum[2] = {0, 0};

		// Obtain an exclusive lock on all byte in the locking range not already locked by the caller. The caller is guaranteed to have locked the
		// WAL_WRITE_LOCK byte, and may have also locked the WAL_CKPT_LOCK byte. If successful, the same bytes that are locked here are unlocked before
		// this function returns.
		_assert(wal->CheckpointLock == 1 || wal->CheckpointLock == 0);
		_assert(WAL_ALL_BUT_WRITE == WAL_WRITE_LOCK + 1);
		_assert(WAL_CKPT_LOCK == WAL_ALL_BUT_WRITE);
		_assert(wal->WriteLock);
		int lockIdx = WAL_ALL_BUT_WRITE + wal->CheckpointLock; // Lock offset to lock for checkpoint
		int locks = VFile::SHM_MAX - lockIdx; // Number of locks to hold
		RC rc = walLockExclusive(wal, lockIdx, locks);
		if (rc)
			return rc;
		WALTRACE("WAL%p: recovery begin...\n", wal);

		_memset(&wal->Header, 0, sizeof(Wal::IndexHeader));

		int64 size; // Size of log file
		rc = wal->WalFile->get_FileSize(size);
		if (rc != RC_OK)
			goto recovery_error;

		if (size > WAL_HDRSIZE)
		{
			// Read in the WAL header.
			uint8 buf[WAL_HDRSIZE]; // Buffer to load WAL header into
			rc = wal->WalFile->Read(buf, WAL_HDRSIZE, 0);
			if (rc != RC_OK)
				goto recovery_error;

			// If the database page size is not a power of two, or is greater than SQLITE_MAX_PAGE_SIZE, conclude that the WAL file contains no valid 
			// data. Similarly, if the 'magic' value is invalid, ignore the whole WAL file.
			uint32 magic = ConvertEx::Get4(&buf[0]); // Magic value read from WAL header
			int sizePage = ConvertEx::Get4(&buf[8]); // Page size according to the log
			if ((magic & 0xFFFFFFFE) != WAL_MAGIC ||
				sizePage & (sizePage - 1) ||
				sizePage > MAX_PAGE_SIZE ||
				sizePage < 512)
				goto finished;
			wal->Header.BigEndianChecksum = (uint8)(magic & 0x00000001);
			wal->SizePage = sizePage;
			wal->Checkpoints = ConvertEx::Get4(&buf[12]);
			_memcpy((uint8 *)&wal->Header.Salt, &buf[16], 8);

			// Verify that the WAL header checksum is correct
			walChecksumBytes(wal->Header.BigEndianChecksum == TYPE_BIGENDIAN, buf, WAL_HDRSIZE - 2 * 4, 0, wal->Header.FrameChecksum);
			if (wal->Header.FrameChecksum[0] != ConvertEx::Get4(&buf[24]) || wal->Header.FrameChecksum[1] != ConvertEx::Get4(&buf[28]))
				goto finished;

			// Verify that the version number on the WAL format is one that are able to understand
			uint32 version = ConvertEx::Get4(&buf[4]); // Magic value read from WAL header
			if (version != WAL_MAX_VERSION)
			{
				rc = SysEx_CANTOPEN_BKPT;
				goto finished;
			}

			// Malloc a buffer to read frames into.
			int sizeFrame = sizePage + WAL_FRAME_HDRSIZE; // Number of bytes in buffer aFrame[]
			uint8 *frames = (uint8 *)_alloc(sizeFrame); // Malloc'd buffer to load entire frame
			if (!frames)
			{
				rc = RC_NOMEM;
				goto recovery_error;
			}
			uint8 *data = &frames[WAL_FRAME_HDRSIZE]; // Pointer to data part of aFrame buffer

			// Read all frames from the log file.
			int frameIdx = 0; // Index of last frame read
			for (int64 offset = WAL_HDRSIZE; (offset + sizeFrame) <= size; offset += sizeFrame) // Next offset to read from log file
			{ 
				// Read and decode the next log frame.
				frameIdx++;
				rc = wal->WalFile->Read(frames, sizeFrame, offset);
				if (rc != RC_OK) break;
				Pid id; // Database page number for frame
				uint32 truncate; // dbsize field from frame header
				bool isValid = walDecodeFrame(wal, &id, &truncate, data, frames); // True if this frame is valid
				if (!isValid) break;
				rc = walIndexAppend(wal, frameIdx, id);
				if (rc != RC_OK) break;

				// If nTruncate is non-zero, this is a commit record.
				if (truncate)
				{
					wal->Header.MaxFrame = frameIdx;
					wal->Header.Pages = truncate;
					wal->Header.SizePage = (uint16)((sizePage & 0xff00) | (sizePage >> 16));
					ASSERTCOVERAGE(sizePage <= 32768);
					ASSERTCOVERAGE(sizePage >= 65536);
					frameChecksum[0] = wal->Header.FrameChecksum[0];
					frameChecksum[1] = wal->Header.FrameChecksum[1];
				}
			}

			_free(frames);
		}

finished:
		if (rc == RC_OK)
		{
			volatile WalCheckpointInfo *info;
			wal->Header.FrameChecksum[0] = frameChecksum[0];
			wal->Header.FrameChecksum[1] = frameChecksum[1];
			walIndexWriteHdr(wal);

			// Reset the checkpoint-header. This is safe because this thread is currently holding locks that exclude all other readers, writers and checkpointers.
			info = walCkptInfo(wal);
			info->Backfills = 0;
			info->ReadMarks[0] = 0;
			for (int i = 1; i < WAL_NREADER; i++) info->ReadMarks[i] = READMARK_NOT_USED;
			if (wal->Header.MaxFrame) info->ReadMarks[1] = wal->Header.MaxFrame;

			// If more than one frame was recovered from the log file, report an event via sqlite3_log(). This is to help with identifying performance
			// problems caused by applications routinely shutting down without checkpointing the log file.
			if (wal->Header.Pages)
				SysEx_LOG(RC_OK, "Recovered %d frames from WAL file %s", wal->Header.Pages, wal->WalName);
		}

recovery_error:
		WALTRACE("WAL%p: recovery %s\n", wal, rc ? "failed" : "ok");
		walUnlockExclusive(wal, lockIdx, locks);
		return rc;
	}

	__device__ static void walIndexClose(Wal *wal, int isDelete)
	{
		if (wal->ExclusiveMode_ == Wal::MODE_HEAPMEMORY)
			for (int i = 0; i < wal->WiData.length; i++)
			{
				_free((void *)wal->WiData[i]);
				wal->WiData[i] = nullptr;
			}
		else
			wal->DBFile->ShmUnmap(isDelete);
	}

#pragma endregion

#pragma region Interface

	__device__ RC Wal::Open(VSystem *vfs, VFile *dbFile, const char *walName, bool noShm, int64 maxWalSize, Wal **walOut)
	{
		_assert(walName && walName[0]);
		_assert(dbFile != nullptr);

		// In the amalgamation, the os_unix.c and os_win.c source files come before this source file.  Verify that the #defines of the locking byte offsets
		// in os_unix.c and os_win.c agree with the WALINDEX_LOCK_OFFSET value.
#ifdef WIN_SHM_BASE
		_assert(WIN_SHM_BASE == WALINDEX_LOCK_OFFSET);
#endif
#ifdef UNIX_SHM_BASE
		_assert(UNIX_SHM_BASE == WALINDEX_LOCK_OFFSET);
#endif

		// Allocate an instance of struct Wal to return.
		*walOut = nullptr;
		Wal *r = (Wal *)_alloc(sizeof(Wal) + vfs->SizeOsFile, true); // Object to allocate and return
		if (!r)
			return RC_NOMEM;

		r->Vfs = vfs;
		r->WalFile = (VFile *)&r[1];
		r->DBFile = dbFile;
		r->ReadLock = -1;
		r->MaxWalSize = maxWalSize;
		r->WalName = walName;
		r->SyncHeader = 1;
		r->PadToSectorBoundary = 1;
		r->ExclusiveMode_ = (noShm ? MODE_HEAPMEMORY : MODE_NORMAL);

		// Open file handle on the write-ahead log file.
		VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_WAL);
		RC rc = vfs->Open(walName, r->WalFile, flags, &flags);
		if (rc == RC_OK && flags & VSystem::OPEN_READONLY)
			r->ReadOnly = RDONLY_RDONLY;

		if (rc != RC_OK)
		{
			walIndexClose(r, 0);
			r->WalFile->Close();
			_free(r);
		}
		else
		{
			int dc = r->WalFile->get_DeviceCharacteristics();
			if (dc & VFile::IOCAP_SEQUENTIAL) { r->SyncHeader = 0; }
			if (dc & VFile::IOCAP_POWERSAFE_OVERWRITE)
				r->PadToSectorBoundary = 0;
			*walOut = r;
			WALTRACE("WAL%d: opened\n", r);
		}
		return rc;
	}

	__device__ void Wal::Limit(int64 limit)
	{
		MaxWalSize = limit;
	}

#pragma endregion

#pragma region Name2

	__device__ static int walIteratorNext(WalIterator *p, uint32 *page, uint32 *frame)
	{
		uint32 r = 0xFFFFFFFF; // 0xffffffff is never a valid page number
		uint32 min = p->Prior; // Result pgno must be greater than iMin
		_assert(min < 0xffffffff);
		for (int i = p->Segments.length - 1; i >= 0; i--)
		{
			WalSegment *segment = &p->Segments[i];
			while (segment->Next < segment->Entrys)
			{
				uint32 id = segment->IDs[segment->Indexs[segment->Next]];
				if (id > min)
				{
					if (id < r)
					{
						r = id;
						*frame = segment->Zero + segment->Indexs[segment->Next];
					}
					break;
				}
				segment->Next++;
			}
		}

		*page = p->Prior = r;
		return (r == 0xFFFFFFFF);
	}

	__device__ static void walMerge(const uint32 *content, ht_slot *lefts, int leftsLength, ht_slot **rightsOut, int *rightsLengthOut, ht_slot *tmp)
	{
		int left = 0; // Current index in aLeft
		int right = 0; // Current index in aRight
		int out = 0; // Current index in output buffer */
		int rightsLength = *rightsLengthOut;
		ht_slot *rights = *rightsOut;

		_assert(leftsLength > 0 && rightsLength > 0);
		while (right < rightsLength || left < leftsLength)
		{
			ht_slot logpage;
			if (left < leftsLength && (right >= rightsLength || content[lefts[left]] < content[rights[right]]))
				logpage = lefts[left++];
			else
				logpage = rights[right++];
			Pid dbpage = content[logpage];

			tmp[out++] = logpage;
			if (left < leftsLength && content[lefts[left]] == dbpage) left++;

			_assert(left >= leftsLength || content[lefts[left]] > dbpage);
			_assert(right >= rightsLength || content[rights[right]] > dbpage);
		}

		*rightsOut = lefts;
		*rightsLengthOut = out;
		_memcpy(lefts, tmp, sizeof(tmp[0]) * out);
	}

	__device__ static void walMergesort(const uint32 *content, ht_slot *buffer, ht_slot *list, int *listLengthRef)
	{
		struct Sublist
		{
			int ListLength; // Number of elements in aList
			ht_slot *List; // Pointer to sub-list content
		};

		const int listLength = *listLengthRef; // Size of input list
		int mergeLength = 0; // Number of elements in list aMerge
		ht_slot *merge = nullptr; // List to be merged
		int subIdx = 0; // Index into aSub array
		struct Sublist subs[13]; // Array of sub-lists

		_memset(subs, 0, sizeof(subs));
		_assert(listLength <= HASHTABLE_NPAGE && listLength > 0);
		_assert(HASHTABLE_NPAGE == (1 << (_lengthof(subs) - 1)));

		for (int listIdx = 0; listIdx < listLength; listIdx++) // Index into input list
		{
			mergeLength = 1;
			merge = &list[listIdx];
			for (subIdx = 0; listIdx & (1 << subIdx); subIdx++)
			{
				struct Sublist *p = &subs[subIdx];
				_assert(p->List && p->ListLength <= (1 << subIdx));
				_assert(p->List == &list[listIdx & ~((2 << subIdx) - 1)]);
				walMerge(content, p->List, p->ListLength, &merge, &mergeLength, buffer);
			}
			subs[subIdx].List = merge;
			subs[subIdx].ListLength = mergeLength;
		}

		for (subIdx++; subIdx < _lengthof(subs); subIdx++)
		{
			if (listLength & (1 << subIdx))
			{
				struct Sublist *p = &subs[subIdx];
				_assert(p->ListLength <= (1 << subIdx));
				_assert(p->List == &list[listLength & ~((2 << subIdx) - 1)]);
				walMerge(content, p->List, p->ListLength, &merge, &mergeLength, buffer);
			}
		}
		_assert(merge == list);
		*listLengthRef = mergeLength;

#ifdef _DEBUG
		for (int i = 1; i < *listLengthRef; i++)
			_assert(content[list[i]] > content[list[i - 1]]);
#endif
	}

	__device__ static void walIteratorFree(WalIterator *p)
	{
		_stackfree(p);
	}

	__device__ static RC walIteratorInit(Wal *wal, WalIterator **iteratorOut)
	{
		// This routine only runs while holding the checkpoint lock. And it only runs if there is actually content in the log (mxFrame>0).
		_assert(wal->CheckpointLock && wal->Header.MaxFrame > 0);
		uint32 lastFrame = wal->Header.MaxFrame; // Last frame in log

		// Allocate space for the WalIterator object.
		int segments = walFramePage(lastFrame) + 1; // Number of segments to merge
		int bytes = sizeof(WalIterator) + (segments - 1) * sizeof(WalSegment) + lastFrame * sizeof(ht_slot); // Number of bytes to allocate
		WalIterator *p = (WalIterator *)_stackalloc(bytes); // Return value
		if (!p)
			return RC_NOMEM;
		_memset(p, 0, bytes);
		p->SegmentsLength = segments;

		// Allocate temporary space used by the merge-sort routine. This block of memory will be freed before this function returns.
		RC rc = RC_OK;
		ht_slot *tmp = (ht_slot *)_stackalloc(sizeof(ht_slot) * (lastFrame > HASHTABLE_NPAGE ? HASHTABLE_NPAGE : lastFrame)); // Temp space used by merge-sort
		if (!tmp)
			rc = RC_NOMEM;

		for (int i = 0; rc == RC_OK && i < segments; i++)
		{
			volatile ht_slot *hash;
			volatile uint32 *ids;
			uint32 zero;
			rc = walHashGet(wal, i, &hash, &ids, &zero);
			if (rc == RC_OK)
			{
				ids++;
				int entrys; // Number of entries in this segment
				if ((i + 1) == segments)
					entrys = (int)(lastFrame - zero);
				else
					entrys = (int)((uint32 *)hash - (Pid *)ids);
				ht_slot *indexs = &((ht_slot *)&p->Segments[p->SegmentsLength])[zero]; // Sorted index for this segment
				zero++;

				for (int j = 0; j < entrys; j++)
					indexs[j] = (ht_slot)j;
				walMergesort((Pid *)ids, tmp, indexs, &entrys);
				p->Segments[i].Zero = zero;
				p->Segments[i].Entrys = entrys;
				p->Segments[i].Indexs = indexs;
				p->Segments[i].IDs = (Pid *)ids;
			}
		}
		_stackfree(tmp);

		if (rc != RC_OK)
			walIteratorFree(p);
		*iteratorOut = p;
		return rc;
	}

	__device__ static RC walBusyLock(Wal *wal, int (*busy)(void *), void *busyArg, int lockIdx, int n)
	{
		RC rc;
		do
		{
			rc = walLockExclusive(wal, lockIdx, n);
		} while (busy && rc == RC_BUSY && busy(busyArg));
		return rc;
	}

	__device__ static int walPagesize(Wal *wal)
	{
		return (wal->Header.SizePage & 0xfe00) + ((wal->Header.SizePage & 0x0001) << 16);
	}

	__device__ static int walCheckpoint(Wal *wal, IPager::CHECKPOINT mode, int (*busyCall)(void *), void *busyArg, VFile::SYNC sync_flags, uint8 *buf)
	{
		int sizePage = walPagesize(wal); // Database page-size
		ASSERTCOVERAGE(sizePage <= 32768);
		ASSERTCOVERAGE(sizePage >= 65536);
		volatile WalCheckpointInfo *info = walCheckpointInfo(wal); // The checkpoint status information
		if (info->Backfills >= wal->Header.MaxFrame) return RC_OK;

		// Allocate the iterator
		WalIterator *iter = nullptr; // Wal iterator context
		RC rc = walIteratorInit(wal, &iter);
		if (rc != RC_OK)
			return rc;
		_assert(iter != nullptr);

		int (*busy)(void *) = nullptr; // Function to call when waiting for locks
		if (mode != IPager::CHECKPOINT_PASSIVE) busy = busyCall;

		// Compute in mxSafeFrame the index of the last frame of the WAL that is safe to write into the database.  Frames beyond mxSafeFrame might
		// overwrite database pages that are in use by active readers and thus cannot be backfilled from the WAL.
		uint32 maxSafeFrame = wal->Header.MaxFrame; // Max frame that can be backfilled
		uint32 maxPage = wal->Header.Pages; // Max database page to write 
		for (int i = 1; i < WAL_NREADER; i++)
		{
			uint32 y = info->ReadMarks[i];
			if (maxSafeFrame > y)
			{
				_assert(y <= wal->Header.MaxFrame);
				rc = walBusyLock(wal, busy, busyArg, WAL_READ_LOCK(i), 1);
				if (rc == RC_OK)
				{
					info->ReadMarks[i] = (i == 1 ? maxSafeFrame : READMARK_NOT_USED);
					walUnlockExclusive(wal, WAL_READ_LOCK(i), 1);
				}
				else if (rc == RC_BUSY)
				{
					maxSafeFrame = y;
					busy = nullptr;
				}
				else
					goto walcheckpoint_out;
			}
		}

		uint32 dbpage = 0; // Next database page to write
		uint32 frame = 0; // Wal frame containing data for iDbpage

		if (info->Backfills < maxSafeFrame && (rc = walBusyLock(wal, busy, busyArg, WAL_READ_LOCK(0), 1)) = RC_OK)
		{
			// Sync the WAL to disk
			if (sync_flags)
				rc = wal->WalFile->Sync(sync_flags);

			// If the database file may grow as a result of this checkpoint, hint about the eventual size of the db file to the VFS layer. 
			if (rc == RC_OK)
			{
				int64 size; // Current size of database file
				rc = wal->DBFile->get_FileSize(size);
				int64 requiredSize = ((int64)maxPage * sizePage);
				if (rc == RC_OK && size < requiredSize)
					wal->DBFile->FileControl(VFile::FCNTL_SIZE_HINT, &requiredSize);
			}

			uint32 backfills = info->Backfills;
			// Iterate through the contents of the WAL, copying data to the db file.
			while (rc == RC_OK && !walIteratorNext(iter, &dbpage, &frame))
			{
				_assert(walFramePgno(wal, frame) == dbpage);
				if (frame <= backfills || frame> maxSafeFrame || dbpage > maxPage) continue;
				int offset = walFrameOffset(frame, sizePage) + WAL_FRAME_HDRSIZE;
				// ASSERTCOVERAGE(IS_BIG_INT(offset)); // requires a 4GiB WAL file
				rc = wal->WalFile->Read(buf, sizePage, offset);
				if (rc != RC_OK) break;
				offset = (dbpage - 1) * (int64)sizePage;
				ASSERTCOVERAGE(IS_BIG_INT(offset));
				rc = wal->DBFile->Write(buf, sizePage, offset);
				if (rc != RC_OK) break;
			}

			// If work was actually accomplished...
			if (rc == RC_OK)
			{
				if (maxSafeFrame == walIndexHdr(wal)->MaxFrame)
				{
					int64 sizeDB = wal->Header.Pages * (int64)sizePage;
					ASSERTCOVERAGE(IS_BIG_INT(sizeDB));
					rc = wal->DBFile->Truncate(sizeDB);
					if (rc == RC_OK && sync_flags)
						rc = wal->DBFile->Sync(sync_flags);
				}
				if (rc == RC_OK)
					info->Backfills = maxSafeFrame;
			}

			// Release the reader lock held while backfilling
			walUnlockExclusive(wal, WAL_READ_LOCK(0), 1);
		}

		if (rc == RC_BUSY) // Reset the return code so as not to report a checkpoint failure just because there are active readers.
			rc = RC_OK;

		// If this is an SQLITE_CHECKPOINT_RESTART operation, and the entire wal file has been copied into the database file, then block until all
		// readers have finished using the wal file. This ensures that the next process to write to the database restarts the wal file.
		if (rc == RC_OK && mode != IPager::CHECKPOINT_PASSIVE)
		{
			_assert(wal->WriteLock);
			if (info->Backfills < wal->Header.MaxFrame)
				rc = RC_BUSY;
			else if (mode == IPager::CHECKPOINT_RESTART)
			{
				_assert(maxSafeFrame == wal->Header.MaxFrame);
				rc = walBusyLock(wal, busy, busyArg, WAL_READ_LOCK(1), WAL_NREADER - 1);
				if (rc == RC_OK)
					walUnlockExclusive(wal, WAL_READ_LOCK(1), WAL_NREADER-1);
			}
		}

walcheckpoint_out:
		walIteratorFree(iter);
		return rc;
	}

	__device__ static void walLimitSize(Wal *wal, int64 max)
	{
		_benignalloc_begin();
		int64 size;
		RC rc = wal->WalFile->get_FileSize(size);
		if (rc == RC_OK && size > max)
			rc = wal->WalFile->Truncate(max);
		_benignalloc_end();
		if (rc != RC_OK)
			SysEx_LOG(rc, "cannot limit WAL size: %s", wal->WalName);
	}

	__device__ RC Wal::Close(VFile::SYNC sync_flags, int bufLength, uint8 *buf)
	{
		// If an EXCLUSIVE lock can be obtained on the database file (using the ordinary, rollback-mode locking methods, this guarantees that the
		// connection associated with this log file is the only connection to the database. In this case checkpoint the database and unlink both
		// the wal and wal-index files.
		//
		// The EXCLUSIVE lock is not released before returning.
		bool isDelete = 0; // True to unlink wal and wal-index files
		RC rc = DBFile->Lock(VFile::LOCK_EXCLUSIVE);
		if (rc == RC_OK)
		{
			if (ExclusiveMode_ == MODE_NORMAL)
				ExclusiveMode_ = MODE_EXCLUSIVE;
			rc = Checkpoint(IPager::CHECKPOINT_PASSIVE, 0, 0, sync_flags, bufLength, buf, 0, 0);
			if (rc == RC_OK)
			{
				int persist = -1;
				DBFile->FileControl(VFile::FCNTL_PERSIST_WAL, &persist);
				if (persist != 1) // Try to delete the WAL file if the checkpoint completed and fsyned (rc==SQLITE_OK) and if we are not in persistent-wal mode (!bPersist)
					isDelete = 1;
				else if (MaxWalSize >= 0)
				{
					// Try to truncate the WAL file to zero bytes if the checkpoint completed and fsynced (rc==SQLITE_OK) and we are in persistent
					// WAL mode (bPersist) and if the PRAGMA journal_size_limit is a non-negative value (pWal->mxWalSize>=0).  Note that we truncate
					// to zero bytes as truncating to the journal_size_limit might leave a corrupt WAL file on disk. */
					walLimitSize(this, 0);
				}
			}
		}

		walIndexClose(this, isDelete);
		WalFile->Close();
		if (isDelete)
		{
			_benignalloc_begin();
			Vfs->Delete(WalName, 0);
			_benignalloc_end();
		}
		WALTRACE("WAL%p: closed\n", this);
		_free((void *)WiData);
		_free(this);
		return rc;
	}

	__device__ static bool walIndexTryHdr(Wal *wal, bool *changed)
	{
		// The first page of the wal-index must be mapped at this point.
		_assert(wal->WiData.length > 0 && wal->WiData[0]);

		// Read the header. This might happen concurrently with a write to the same area of shared memory on a different CPU in a SMP,
		// meaning it is possible that an inconsistent snapshot is read from the file. If this happens, return non-zero.
		//
		// There are two copies of the header at the beginning of the wal-index. When reading, read [0] first then [1].  Writes are in the reverse order.
		// Memory barriers are used to prevent the compiler or the hardware from reordering the reads and writes.
		Wal::IndexHeader h1, h2; // Two copies of the header content
		Wal::IndexHeader volatile *header = walIndexHeader(wal); // Header in shared memory
		_memcpy((void *)&h1, (void *)&header[0], sizeof(h1));
		walShmBarrier(wal);
		_memcpy((void *)&h2, (void *)&header[1], sizeof(h2));

		if (_memcmp(&h1, &h2, sizeof(h1)) != 0)
			return true; // Dirty read
		if (h1.IsInit == 0)
			return true; // Malformed header - probably all zeros
		uint32 checksum[2]; // Checksum on the header content
		walChecksumBytes(1, (uint8 *)&h1, sizeof(h1) - sizeof(h1.Checksum), 0, checksum);
		if (checksum[0] != h1.Checksum[0] || checksum[1] != h1.Checksum[1])
			return true; // Checksum does not match

		if (_memcmp(&wal->Header, &h1, sizeof(Wal::IndexHeader)))
		{
			*changed = true;
			_memcpy(&wal->Header, &h1, sizeof(Wal::IndexHeader));
			wal->SizePage = (wal->Header.SizePage & 0xfe00) + ((wal->Header.SizePage & 0x0001) << 16);
			ASSERTCOVERAGE(wal->SizePage <= 32768);
			ASSERTCOVERAGE(wal->SizePage >= 65536);
		}

		// The header was successfully read. Return zero.
		return false;
	}

	__device__ static RC walIndexReadHdr(Wal *wal, bool *changed)
	{
		// Ensure that page 0 of the wal-index (the page that contains the wal-index header) is mapped. Return early if an error occurs here.
		_assert(*changed);
		volatile uint32 *page0; // Chunk of wal-index containing header
		RC rc = walIndexPage(wal, 0, &page0);
		if (rc != RC_OK)
			return rc;
		_assert(page0 || !wal->WriteLock);

		// If the first page of the wal-index has been mapped, try to read the wal-index header immediately, without holding any lock. This usually
		// works, but may fail if the wal-index header is corrupt or currently being modified by another thread or process.
		bool badHdr = (page0 ? walIndexTryHdr(wal, changed) : true); // True if a header read failed

		// If the first attempt failed, it might have been due to a race with a writer.  So get a WRITE lock and try again.
		_assert(!badHdr || !wal->WriteLock);
		if (badHdr)
		{
			if (wal->ReadOnly & Wal::RDONLY_SHM_RDONLY)
			{
				if ((rc = walLockShared(wal, WAL_WRITE_LOCK)) == RC_OK)
				{
					walUnlockShared(wal, WAL_WRITE_LOCK);
					rc = RC_READONLY_RECOVERY;
				}
			}
			else if ((rc = walLockExclusive(wal, WAL_WRITE_LOCK, 1)) == RC_OK)
			{
				wal->WriteLock = true;
				if ((rc = walIndexPage(wal, 0, &page0)) == RC_OK)
				{
					badHdr = walIndexTryHdr(wal, changed);
					if (badHdr)
					{
						// If the wal-index header is still malformed even while holding a WRITE lock, it can only mean that the header is corrupted and
						// needs to be reconstructed.  So run recovery to do exactly that.
						rc = walIndexRecover(wal);
						*changed = true;
					}
				}
				wal->WriteLock = false;
				walUnlockExclusive(wal, WAL_WRITE_LOCK, 1);
			}
		}

		// If the header is read successfully, check the version number to make sure the wal-index was not constructed with some future format that
		// this version of SQLite cannot understand.
		if (!badHdr && wal->Header.Version != WALINDEX_MAX_VERSION)
			rc = SysEx_CANTOPEN_BKPT;

		return rc;
	}

	__device__ static RC walTryBeginRead(Wal *wal, bool *changed, bool useWal, int count)
	{
		_assert(wal->ReadLock < 0); // Not currently locked

		// Take steps to avoid spinning forever if there is a protocol error.
		//
		// Circumstances that cause a RETRY should only last for the briefest instances of time.  No I/O or other system calls are done while the
		// locks are held, so the locks should not be held for very long. But if we are unlucky, another process that is holding a lock might get
		// paged out or take a page-fault that is time-consuming to resolve, during the few nanoseconds that it is holding the lock.  In that case,
		// it might take longer than normal for the lock to free.
		//
		// After 5 RETRYs, we begin calling sqlite3OsSleep().  The first few calls to sqlite3OsSleep() have a delay of 1 microsecond.  Really this
		// is more of a scheduler yield than an actual delay.  But on the 10th an subsequent retries, the delays start becoming longer and longer, 
		// so that on the 100th (and last) RETRY we delay for 21 milliseconds. The total delay time before giving up is less than 1 second.
		if (count > 5)
		{
			int delay = 1; // Pause time in microseconds
			if (count > 100)
			{
				//VVA_ONLY(wal->LockError = 1;)
				return RC_PROTOCOL;
			}
			if (count >= 10) delay = (count - 9) * 238; // Max delay 21ms. Total delay 996ms
			wal->Vfs->Sleep(delay);
		}

		RC rc = RC_OK;
		if (!useWal)
		{
			rc = walIndexReadHdr(wal, changed);
			if (rc == RC_BUSY)
			{
				// If there is not a recovery running in another thread or process then convert BUSY errors to WAL_RETRY.  If recovery is known to
				// be running, convert BUSY to BUSY_RECOVERY.  There is a race here which might cause WAL_RETRY to be returned even if BUSY_RECOVERY
				// would be technically correct.  But the race is benign since with WAL_RETRY this routine will be called again and will probably be
				// right on the second iteration.
				if (wal->WiData[0] == 0)
				{
					// This branch is taken when the xShmMap() method returns SQLITE_BUSY. We assume this is a transient condition, so return WAL_RETRY. The
					// xShmMap() implementation used by the default unix and win32 VFS modules may return SQLITE_BUSY due to a race condition in the 
					// code that determines whether or not the shared-memory region must be zeroed before the requested page is returned.
					rc = RC_INVALID;
				}
				else if ((rc = walLockShared(wal, WAL_RECOVER_LOCK)) == RC_OK)
				{
					walUnlockShared(wal, WAL_RECOVER_LOCK);
					rc = RC_INVALID;
				}
				else if (rc == RC_BUSY)
					rc = RC_BUSY_RECOVERY;
			}
			if (rc != RC_OK)
				return rc;
		}

		volatile WalCheckpointInfo *info = walCheckpointInfo(wal); // Checkpoint information in wal-index
		if (!useWal && info->Backfills == wal->Header.MaxFrame)
		{
			// The WAL has been completely backfilled (or it is empty). and can be safely ignored.
			rc = walLockShared(wal, WAL_READ_LOCK(0));
			walShmBarrier(wal);
			if (rc == RC_OK)
			{
				if (_memcmp((void *)walIndexHdr(wal), (void *)&wal->Header, sizeof(Wal::IndexHeader)))
				{
					// It is not safe to allow the reader to continue here if frames may have been appended to the log before READ_LOCK(0) was obtained.
					// When holding READ_LOCK(0), the reader ignores the entire log file, which implies that the database file contains a trustworthy
					// snapshoT. Since holding READ_LOCK(0) prevents a checkpoint from happening, this is usually correct.
					//
					// However, if frames have been appended to the log (or if the log is wrapped and written for that matter) before the READ_LOCK(0)
					// is obtained, that is not necessarily true. A checkpointer may have started to backfill the appended frames but crashed before
					// it finished. Leaving a corrupt image in the database file.
					walUnlockShared(wal, WAL_READ_LOCK(0));
					return RC_INVALID;
				}
				wal->ReadLock = 0;
				return RC_OK;
			}
			else if (rc != RC_BUSY)
				return rc;
		}

		// If we get this far, it means that the reader will want to use the WAL to get at content from recent commits.  The job now is
		// to select one of the aReadMark[] entries that is closest to but not exceeding pWal->hdr.mxFrame and lock that entry.
		uint32 maxReadMark = 0; // Largest aReadMark[] value
		int mxI = 0; // Index of largest aReadMark[] value
		for (int i = 1; i < WAL_NREADER; i++)
		{
			uint32 thisMark = info->ReadMarks[i];
			if (maxReadMark <= thisMark && thisMark <= wal->Header.MaxFrame)
			{
				_assert(thisMark != READMARK_NOT_USED);
				maxReadMark = thisMark;
				mxI = i;
			}
		}
		if ((wal->ReadOnly & Wal::RDONLY_SHM_RDONLY) == 0 && (maxReadMark < wal->Header.MaxFrame || mxI == 0))
		{
			for (int i = 1; i < WAL_NREADER; i++)
			{
				rc = walLockExclusive(wal, WAL_READ_LOCK(i), 1);
				if (rc == RC_OK)
				{
					maxReadMark = info->ReadMarks[i] = wal->Header.MaxFrame;
					mxI = i;
					walUnlockExclusive(wal, WAL_READ_LOCK(i), 1);
					break;
				}
				else if(rc != RC_BUSY)
					return rc;
			}
		}
		if (mxI == 0)
		{
			_assert(rc == RC_BUSY || (wal->ReadOnly & Wal::RDONLY_SHM_RDONLY) != 0);
			return (rc == RC_BUSY ? RC_INVALID : RC_READONLY_CANTLOCK);
		}

		rc = walLockShared(wal, WAL_READ_LOCK(mxI));
		if (rc)
			return (rc == RC_BUSY ? RC_INVALID : rc);
		// Now that the read-lock has been obtained, check that neither the value in the aReadMark[] array or the contents of the wal-index
		// header have changed.
		//
		// It is necessary to check that the wal-index header did not change between the time it was read and when the shared-lock was obtained
		// on WAL_READ_LOCK(mxI) was obtained to account for the possibility that the log file may have been wrapped by a writer, or that frames
		// that occur later in the log than pWal->hdr.mxFrame may have been copied into the database by a checkpointer. If either of these things
		// happened, then reading the database with the current value of pWal->hdr.mxFrame risks reading a corrupted snapshot. So, retry instead.
		//
		// This does not guarantee that the copy of the wal-index header is up to date before proceeding. That would not be possible without somehow
		// blocking writers. It only guarantees that a dangerous checkpoint or log-wrap (either of which would require an exclusive lock on
		// WAL_READ_LOCK(mxI)) has not occurred since the snapshot was valid.
		walShmBarrier(wal);
		if (info->ReadMarks[mxI] != maxReadMark || _memcmp((void *)walIndexHeader(wal), &wal->Header, sizeof(Wal::IndexHeader)))
		{
			walUnlockShared(wal, WAL_READ_LOCK(mxI));
			return RC_INVALID;
		}
		else
		{
			_assert(maxReadMark <= wal->Header.MaxFrame);
			wal->ReadLock = (int16)mxI;
		}
		return rc;
	}

#pragma endregion

#pragma region Interface2

	__device__ RC Wal::BeginReadTransaction(bool *changed)
	{
		int count = 0; // Number of TryBeginRead attempts
		RC rc;
		do
		{
			rc = walTryBeginRead(this, changed, 0, ++count);
		} while (rc == RC_INVALID);
		ASSERTCOVERAGE((rc & 0xff) == RC_BUSY);
		ASSERTCOVERAGE((rc & 0xff) == RC_IOERR);
		ASSERTCOVERAGE(rc == RC_PROTOCOL);
		ASSERTCOVERAGE(rc == RC_OK);
		return rc;
	}

	__device__ void Wal::EndReadTransaction()
	{
		EndWriteTransaction();
		if (ReadLock >= 0)
		{
			walUnlockShared(this, WAL_READ_LOCK(ReadLock));
			ReadLock = -1;
		}
	}

	__device__ RC Wal::Read(Pid id, bool *inWal, int bufLength, uint8 *buf)
	{
		// This routine is only be called from within a read transaction.
		_assert(ReadLock >= 0 || LockError);

		// If the "last page" field of the wal-index header snapshot is 0, then no data will be read from the wal under any circumstances. Return early
		// in this case as an optimization.  Likewise, if pWal->readLock==0, then the WAL is ignored by the reader so return early, as if the WAL were empty.
		uint32 last = Header.MaxFrame; // Last page in WAL for this reader
		if (last == 0 || ReadLock == 0)
		{
			*inWal = false;
			return RC_OK;
		}

		// Search the hash table or tables for an entry matching page number pgno. Each iteration of the following for() loop searches one
		// hash table (each hash table indexes up to HASHTABLE_NPAGE frames).
		//
		// This code might run concurrently to the code in walIndexAppend() that adds entries to the wal-index (and possibly to this hash 
		// table). This means the value just read from the hash slot (aHash[iKey]) may have been added before or after the 
		// current read transaction was opened. Values added after the read transaction was opened may have been written incorrectly -
		// i.e. these slots may contain garbage data. However, we assume that any slots written before the current read transaction was
		// opened remain unmodified.
		//
		// For the reasons above, the if(...) condition featured in the inner loop of the following block is more stringent that would be required 
		// if we had exclusive access to the hash-table:
		//
		//   (aPgno[iFrame]==pgno): 
		//     This condition filters out normal hash-table collisions.
		//
		//   (iFrame<=iLast): 
		//     This condition filters out entries that were added to the hash table after the current read-transaction had started.
		uint32 read = 0; // If !=0, WAL frame to return data from
		for (int hash = walFramePage(last); hash >= 0 && read == 0; hash--)
		{
			volatile ht_slot *hashs; // Pointer to hash table
			volatile Pid *ids; // Pointer to array of page numbers
			uint32 zero; // Frame number corresponding to aPgno[0]
			RC rc = walHashGet(this, hash, &hashs, &ids, &zero);
			if (rc != RC_OK)
				return rc;
			int collides = HASHTABLE_NSLOT; // Number of hash collisions remaining
			for (int key = walHash(id); hashs[key]; key = walNextHash(key)) // Hash slot index
			{
				uint32 frame = hashs[key] + zero;
				if (frame <= last && ids[hashs[key]] == id)
					read = frame;
				if ((collides--) == 0)
					return SysEx_CORRUPT_BKPT;
			}
		}

#ifdef ENABLE_EXPENSIVE_ASSERT
		// If expensive assert() statements are available, do a linear search of the wal-index file content. Make sure the results agree with the
		// result obtained using the hash indexes above.  */
		{
			uint32 read2 = 0;
			for(uint32 test = last; test > 0; test--)
				if (walFramePgno(this, test) == id)
				{
					read2 = test;
					break;
				}
				_assert(read == read2);
		}
#endif

		// If iRead is non-zero, then it is the log frame number that contains the required page. Read and return data from the log file.
		if (read)
		{
			int size = Header.SizePage;
			size = (size & 0xfe00) + ((size & 0x0001) << 16);
			ASSERTCOVERAGE(size <= 32768);
			ASSERTCOVERAGE(size >= 65536);
			int64 offset = walFrameOffset(read, size) + WAL_FRAME_HDRSIZE;
			*inWal = true;
			// ASSERTCOVERAGE(IS_BIG_INT(offset)); // requires a 4GiB WAL */
			return WalFile->Read(buf, (bufLength > size ? size : bufLength), offset);
		}

		*inWal = false;
		return RC_OK;
	}

	__device__ Pid Wal::DBSize()
	{
		return (_ALWAYS(ReadLock >= 0) ? Header.Pages : 0);
	}

	__device__ RC Wal::BeginWriteTransaction()
	{
		// Cannot start a write transaction without first holding a read transaction.
		_assert(ReadLock >= 0);

		if (ReadOnly)
			return RC_READONLY;

		// Only one writer allowed at a time.  Get the write lock.  Return SQLITE_BUSY if unable.
		RC rc = walLockExclusive(this, WAL_WRITE_LOCK, 1);
		if (rc)
			return rc;
		WriteLock = 1;

		// If another connection has written to the database file since the time the read transaction on this connection was started, then
		// the write is disallowed.
		if (_memcmp(&Header, (void *)walIndexHeader(this), sizeof(Wal::IndexHeader)) != 0)
		{
			walUnlockExclusive(this, WAL_WRITE_LOCK, 1);
			WriteLock = 0;
			rc = RC_BUSY;
		}

		return rc;
	}

	__device__ RC Wal::EndWriteTransaction()
	{
		if (WriteLock)
		{
			walUnlockExclusive(this, WAL_WRITE_LOCK, 1);
			WriteLock = 0;
			TruncateOnCommit = false;
		}
		return RC_OK;
	}

	__device__ RC Wal::Undo(int (*undo)(void *, Pid), void *undoCtx)
	{
		RC rc = RC_OK;
		if (_ALWAYS(WriteLock))
		{
			// Restore the clients cache of the wal-index header to the state it was in before the client began writing to the database. 
			_memcpy((void *)&Header, (void *)walIndexHeader(this), sizeof(Wal::IndexHeader));
			Pid max = Header.MaxFrame;
			for (Pid frame = Header.MaxFrame + 1;  _ALWAYS(rc == RC_OK) && frame <= max; frame++)
			{
				// This call cannot fail. Unless the page for which the page number is passed as the second argument is (a) in the cache and 
				// (b) has an outstanding reference, then xUndo is either a no-op (if (a) is false) or simply expels the page from the cache (if (b) is false).
				//
				// If the upper layer is doing a rollback, it is guaranteed that there are no outstanding references to any page other than page 1. And
				// page 1 is never written to the log until the transaction is committed. As a result, the call to xUndo may not fail.
				_assert(walFramePgno(this, frame) != 1);
				rc = undo(undoCtx, walFramePgno(this, frame));
			}
			if (max != Header.MaxFrame) walCleanupHash(this);
		}
		_assert(rc == RC_OK);
		return rc;
	}

	__device__ void Wal::Savepoint(uint32 *walData)
	{
		_assert(WriteLock);
		walData[0] = Header.MaxFrame;
		walData[1] = Header.FrameChecksum[0];
		walData[2] = Header.FrameChecksum[1];
		walData[3] = __arrrayLength(Checkpoints);
	}

	__device__ RC Wal::SavepointUndo(uint32 *walData)
	{
		_assert(WriteLock );
		_assert(walData[3] != Checkpoints.length || walData[0] <= Header.MaxFrame);

		if (walData[3] != Checkpoints.length)
		{
			// This savepoint was opened immediately after the write-transaction was started. Right after that, the writer decided to wrap around
			// to the start of the log. Update the savepoint values to match.
			walData[0] = 0;
			walData[3] = Checkpoints.length;
		}

		if (walData[0] < Header.MaxFrame)
		{
			Header.MaxFrame = walData[0];
			Header.FrameChecksum[0] = walData[1];
			Header.FrameChecksum[1] = walData[2];
			walCleanupHash(this);
		}

		return RC_OK;
	}

	__device__ static int walRestartLog(Wal *wal)
	{
		RC rc = RC_OK;
		if (wal->ReadLock == 0)
		{
			volatile WalCheckpointInfo *info = walCheckpointInfo(wal);
			_assert(info->Backfills == wal->Header.MaxFrame);
			if (info->Backfills > 0)
			{
				uint32 salt1;
				SysEx::PutRandom(4, &salt1);
				rc = walLockExclusive(wal, WAL_READ_LOCK(1), WAL_NREADER - 1);
				if (rc == RC_OK)
				{
					// If all readers are using WAL_READ_LOCK(0) (in other words if no readers are currently using the WAL), then the transactions
					// frames will overwrite the start of the existing log. Update the wal-index header to reflect this.
					//
					// In theory it would be Ok to update the cache of the header only at this point. But updating the actual wal-index header is also
					// safe and means there is no special case for sqlite3WalUndo() to handle if this transaction is rolled back.
					int i;                    /* Loop counter */
					uint32 *salt = wal->Header.Salt; // Big-endian salt values

					wal->Checkounts++;
					wal->Header.MaxFrame = 0;
					ConvertEx::Put4((uint8 *)&salt[0], 1 + ConvertEx::Get4((uint8 *)&salt[0]));
					salt[1] = salt1;
					walIndexWriteHeader(wal);
					info->Backfills = 0;
					info->ReadMarks[1] = 0;
					for (int i = 2; i < WAL_NREADER; i++) info->ReadMarks[i] = READMARK_NOT_USED;
					_assert(info->ReadMarks[0]==0 );
					walUnlockExclusive(wal, WAL_READ_LOCK(1), WAL_NREADER - 1);
				}
				else if (rc != RC_BUSY)
					return rc;
			}
			walUnlockShared(wal, WAL_READ_LOCK(0));
			wal->ReadLock = -1;
			int count = 0;
			do
			{
				int notUsed;
				rc = walTryBeginRead(wal, &notUsed, 1, ++count);
			} while (rc == RC_INVALID);
			_assert((rc & 0xff) != RC_BUSY); // BUSY not possible when useWal==1
			ASSERTCOVERAGE((rc & 0xff) == RC_IOERR);
			ASSERTCOVERAGE(rc == RC_PROTOCOL);
			ASSERTCOVERAGE(rc == RC_OK);
		}
		return rc;
	}

	typedef struct WalWriter
	{
		Wal *Wal;               // The complete WAL information
		VFile *File;			// The WAL file to which we write
		int64 SyncPoint;		// Fsync at this offset
		int SyncFlags;          // Flags for the fsync
		int SizePage;           // Size of one page
	} WalWriter;

	__device__ static RC walWriteToLog(WalWriter *p, void *content, int amount, int64 offset)
	{
		RC rc;
		if (offset < p->SyncPoint && offset + amount >= p->SyncPoint)
		{
			int firstAmount = (int)(p->SyncPoint - offset);
			rc = p->File->Write(content, firstAmount, offset);
			if (rc) return rc;
			offset += firstAmount;
			amount -= firstAmount;
			content = (void *)(firstAmount + (char *)content);
			_assert(p->SyncFlags & (VFile::SYNC_NORMAL | VFile::SYNC_FULL));
			rc = p->File->Sync(p->SyncFlags);
			if (amount == 0 || rc) return rc;
		}
		rc = p->File->Write(content, amount, offset);
		return rc;
	}

	__device__ static RC walWriteOneFrame(WalWriter *p, PgHdr *page, int truncate, int64 offset)
	{
		void *data; // Data actually written
#if defined(HAS_CODEC)
		if ((data = sqlite3PagerCodec(page)) == nullptr) return RC_NOMEM;
#else
		data = page->Data;
#endif
		uint8 frame[WAL_FRAME_HDRSIZE]; // Buffer to assemble frame-header in
		walEncodeFrame(p->Wal, page->ID, truncate, data, frame);
		RC rc = walWriteToLog(p, frame, sizeof(frame), offset);
		if (rc) return rc;
		// Write the page data
		rc = walWriteToLog(p, data, p->SizePage, offset + sizeof(frame));
		return rc;
	}

	__device__ RC Wal::Frames(int sizePage, PgHdr *list, Pid truncate, bool isCommit, VFile::SYNC sync_flags)
	{
		_assert(list);
		_assert(WriteLock);

		// If this frame set completes a transaction, then nTruncate>0.  If nTruncate==0 then this frame set does not complete the transaction.
		_assert((isCommit != 0) == (truncate != 0));

#if defined(TEST) && defined(_DEBUG)
		{ 
			int count;
			for (count = 0, p = list; p; p = p->Dirty, count++) { }
			WALTRACE("WAL%p: frame write begin. %d frames. mxFrame=%d. %s\n", this, count, Header.MaxFrame, isCommit ? "Commit" : "Spill");
		}
#endif

		// See if it is possible to write these frames into the start of the log file, instead of appending to it at pWal->hdr.mxFrame.
		RC rc;
		if ((rc = walRestartLog(this)) != RC_OK)
			return rc;

		// If this is the first frame written into the log, write the WAL header to the start of the WAL file. See comments at the top of
		// this source file for a description of the WAL header format.
		uint32 frame = Header.MaxFrame; // Next frame address
		if (frame == 0)
		{
			uint8 walHdr[WAL_HDRSIZE]; // Buffer to assemble wal-header in
			uint32 checksum[2]; // Checksum for wal-header

			ConvertEx::Put4(&walHdr[0], (WAL_MAGIC | TYPE_BIGENDIAN));
			ConvertEx::Put4(&walHdr[4], WAL_MAX_VERSION);
			ConvertEx::Put4(&walHdr[8], sizePage);
			ConvertEx::Put4(&walHdr[12], Checkpoints);
			if (Checkpoints == 0) SysEx::PutRandom(8, Header.Salt);
			_memcpy(&walHdr[16], (uint8 *)Header.Salt, 8);
			walChecksumBytes(1, walHdr, WAL_HDRSIZE - 2 * 4, 0, checksum);
			ConvertEx::Put4(&walHdr[24], checksum[0]);
			ConvertEx::Put4(&walHdr[28], checksum[1]);

			SizePage = sizePage;
			Header.BigEndianChecksum = TYPE_BIGENDIAN;
			Header.FrameChecksum[0] = checksum[0];
			Header.FrameChecksum[1] = checksum[1];
			TruncateOnCommit = true;

			rc = WalFile->Write(walHdr, sizeof(walHdr), 0);
			WALTRACE("WAL%p: wal-header write %s\n", this, rc ? "failed" : "ok");
			if (rc != RC_OK)
				return rc;

			// Sync the header (unless SQLITE_IOCAP_SEQUENTIAL is true or unless all syncing is turned off by PRAGMA synchronous=OFF).  Otherwise
			// an out-of-order write following a WAL restart could result in database corruption.
			if (SyncHeader && sync_flags)
			{
				rc = WalFile->Sync(sync_flags & VFile::SYNC_WAL_MASK);
				if (rc) return rc;
			}
		}
		_assert((int)SizePage == sizePage);

		// Setup information needed to write frames into the WAL 
		WalWriter w; // The writer
		w.Wal = this;
		w.File = WalFile;
		w.SyncPoint = 0;
		w.SyncFlags = sync_flags;
		w.SizePage = sizePage;
		int64 offset = walFrameOffset(frame + 1, sizePage); // Next byte to write in WAL file
		int sizeFrame = sizePage + WAL_FRAME_HDRSIZE; // The size of a single frame

		// Write all frames into the log file exactly once
		PgHdr *last = nullptr; // Last frame in list
		for (PgHdr *p = list; p; p = p->Dirty)
		{
			int dbSize; // 0 normally.  Positive == commit flag
			frame++;
			_assert(offset == walFrameOffset(frame, sizePage));
			dbSize = (isCommit && p->Dirty == nullptr ? truncate : 0);
			rc = walWriteOneFrame(&w, p, dbSize, offset);
			if (rc) return rc;
			last = p;
			offset += sizeFrame;
		}

		// If this is the end of a transaction, then we might need to pad the transaction and/or sync the WAL file.
		//
		// Padding and syncing only occur if this set of frames complete a transaction and if PRAGMA synchronous=FULL.  If synchronous==NORMAL
		// or synchonous==OFF, then no padding or syncing are needed.
		//
		// If SQLITE_IOCAP_POWERSAFE_OVERWRITE is defined, then padding is not needed and only the sync is done.  If padding is needed, then the
		// final frame is repeated (with its commit mark) until the next sector boundary is crossed.  Only the part of the WAL prior to the last
		// sector boundary is synced; the part of the last frame that extends past the sector boundary is written after the sync.
		int extras = 0; // Number of extra copies of last page
		if (isCommit && (sync_flags & VFile::SYNC_WAL_TRANSACTIONS) != 0)
		{
			if (PadToSectorBoundary)
			{
				int sectorSize = WalFile->get_SectorSize();
				w.SyncPoint = ((offset + sectorSize - 1) / sectorSize) * sectorSize;
				while (offset < w.SyncPoint)
				{
					rc = walWriteOneFrame(&w, last, truncate, offset);
					if (rc) return rc;
					offset += sizeFrame;
					extras++;
				}
			}
			else
				rc = w.File->Sync(sync_flags & VFile::SYNC_WAL_MASK);
		}

		// If this frame set completes the first transaction in the WAL and if PRAGMA journal_size_limit is set, then truncate the WAL to the
		// journal size limit, if possible.
		if (isCommit && TruncateOnCommit && MaxWalSize >= 0)
		{
			int64 size = MaxWalSize;
			if (walFrameOffset(frame + extras + 1, sizePage) > MaxWalSize)
				size = walFrameOffset(frame + extras + 1, sizePage);
			walLimitSize(this, size);
			TruncateOnCommit = false;
		}

		// Append data to the wal-index. It is not necessary to lock the wal-index to do this as the SQLITE_SHM_WRITE lock held on the wal-index
		// guarantees that there are no other writers, and no data that may be in use by existing readers is being overwritten.
		frame = Header.MaxFrame;
		for (PgHdr *p = list; p && rc == RC_OK; p = p->Dirty)
		{
			frame++;
			rc = walIndexAppend(this, frame, p->ID);
		}
		while (rc == RC_OK && extras > 0)
		{
			frame++;
			extras--;
			rc = walIndexAppend(this, frame, last->ID);
		}

		if (rc == RC_OK)
		{
			// Update the private copy of the header.
			Header.SizePage = (uint16)((sizePage & 0xff00) | (sizePage >> 16));
			ASSERTCOVERAGE(sizePage <= 32768);
			ASSERTCOVERAGE(sizePage >= 65536);
			Header.MaxFrame = frame;
			if (isCommit)
			{
				Header.Change++;
				Header.Pages = truncate;
			}
			// If this is a commit, update the wal-index header too.
			if (isCommit)
			{
				walIndexWriteHdr(this);
				Callback = frame;
			}
		}

		WALTRACE("WAL%p: frame write %s\n", this, rc ? "failed" : "ok");
		return rc;
	}

	__device__ RC Wal::Checkpoint(IPager::CHECKPOINT mode, int (*busy)(void*), void *busyArg, VFile::SYNC sync_flags, int bufLength, uint8 *buf, int *logs, int *checkpoints)
	{
		_assert(CheckpointLock == 0);
		_assert(WriteLock == 0);

		if (ReadOnly) return RC_READONLY;
		WALTRACE("WAL%p: checkpoint begins\n", this);
		RC rc = walLockExclusive(this, WAL_CKPT_LOCK, 1);
		if (rc) // Usually this is SQLITE_BUSY meaning that another thread or process is already running a checkpoint, or maybe a recovery.  But it might also be SQLITE_IOERR.
			return rc;
		CheckpointLock = true;

		// If this is a blocking-checkpoint, then obtain the write-lock as well to prevent any writers from running while the checkpoint is underway.
		// This has to be done before the call to walIndexReadHdr() below.
		//
		// If the writer lock cannot be obtained, then a passive checkpoint is run instead. Since the checkpointer is not holding the writer lock,
		// there is no point in blocking waiting for any readers. Assuming no other error occurs, this function will return SQLITE_BUSY to the caller.
		IPager::CHECKPOINT mode2 = mode; // Mode to pass to walCheckpoint()
		if (mode != IPager::CHECKPOINT_PASSIVE)
		{
			rc = walBusyLock(this, busy, busyArg, WAL_WRITE_LOCK, 1);
			if (rc == RC_OK)
				WriteLock = true;
			else if (rc == RC_BUSY)
			{
				mode2 = IPager::CHECKPOINT_PASSIVE;
				rc = RC_OK;
			}
		}

		// Read the wal-index header.
		bool isChanged = false; // True if a new wal-index header is loaded
		if (rc == RC_OK)
			rc = walIndexReadHdr(this, &isChanged);

		// Copy data from the log to the database file.
		if (rc == RC_OK)
		{
			if (Header.MaxFrame && walPagesize(this) != bufLength)
				rc = SysEx_CORRUPT_BKPT;
			else
				rc = walCheckpoint(this, mode2, busy, busyArg, sync_flags, buf);

			// If no error occurred, set the output variables.
			if (rc == RC_OK || rc == RC_BUSY)
			{
				if (logs) *logs = (int)Header.MaxFrame;
				if (checkpoints) *checkpoints = (int)(walCheckpointInfo(this)->Backfills);
			}
		}

		if (isChanged)
		{
			// If a new wal-index header was loaded before the checkpoint was performed, then the pager-cache associated with pWal is now
			// out of date. So zero the cached wal-index header to ensure that next time the pager opens a snapshot on this database it knows that
			// the cache needs to be reset.
			_memset(&Header, 0, sizeof(Wal::IndexHeader));
		}

		// Release the locks.
		EndWriteTransaction();
		walUnlockExclusive(this, WAL_CKPT_LOCK, 1);
		CheckpointLock = 0;
		WALTRACE("WAL%p: checkpoint %s\n", wal, rc ? "failed" : "ok");
		return (rc == RC_OK && mode != mode2 ? RC_BUSY : rc);
	}

	__device__ int Wal::get_Callback()
	{
		uint32 r = Callback;
		Callback = 0;
		return (int)r;
	}

	__device__ bool Wal::ExclusiveMode(int op)
	{
		_assert(WriteLock == 0);
		_assert(ExclusiveMode_ != MODE_HEAPMEMORY || op == -1);

		// pWal->readLock is usually set, but might be -1 if there was a prior error while attempting to acquire are read-lock. This cannot 
		// happen if the connection is actually in exclusive mode (as no xShmLock locks are taken in this case). Nor should the pager attempt to
		// upgrade to exclusive-mode following such an error.
		_assert(ReadLock >= 0 || LockError);
		_assert(ReadLock >= 0 || (op <= 0 && ExclusiveMode_ == false));

		bool rc;
		if (op == 0)
		{
			if (ExclusiveMode_)
			{
				ExclusiveMode_ = false;
				if (walLockShared(this, WAL_READ_LOCK(ReadLock)) != RC_OK)
					ExclusiveMode_ = true;
				rc = !ExclusiveMode_;
			}
			else // Already in locking_mode=NORMAL
				rc = false;
		}
		else if (op > 0)
		{
			_assert(!ExclusiveMode_);
			_assert(ReadLock >= 0);
			walUnlockShared(this, WAL_READ_LOCK(ReadLock));
			ExclusiveMode_ = true;
			rc = true;
		}
		else
			rc = !ExclusiveMode_;
		return rc;
	}

	__device__ bool Wal::get_HeapMemory()
	{
		return (ExclusiveMode_ == MODE_HEAPMEMORY);
	}

#ifdef ENABLE_ZIPVFS
	__device__ int Wal::get_Framesize()
	{
		_assert(wal == nullptr || wal->ReadLock >= 0);
		return (wal ? wal->SizePage : 0);
	}
#endif

#pragma endregion

}
#endif