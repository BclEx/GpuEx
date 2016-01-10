// This file contains code for a VFS layer that acts as a wrapper around an existing VFS. The code in this file attempts to verify that SQLite
// correctly populates and syncs a journal file before writing to a corresponding database file.
//
// INTERFACE
//   The public interface to this wrapper VFS is two functions:
//     jt_register()
//     jt_unregister()
//
//   See header comments associated with those two functions below for details.
//
// LIMITATIONS
//   This wrapper will not work if "PRAGMA synchronous = off" is used.
//
// OPERATION
//  Starting a Transaction:
//   When a write-transaction is started, the contents of the database is inspected and the following data stored as part of the database file 
//   handle (type struct jt_file):
//
//     a) The page-size of the database file.
//     b) The number of pages that are in the database file.
//     c) The set of page numbers corresponding to free-list leaf pages.
//     d) A check-sum for every page in the database file.
//
//   The start of a write-transaction is deemed to have occurred when a 28-byte journal header is written to byte offset 0 of the journal file.
//
//  Syncing the Journal File:
//   Whenever the xSync method is invoked to sync a journal-file, the contents of the journal file are read. For each page written to
//   the journal file, a check-sum is calculated and compared to the check-sum calculated for the corresponding database page when the
//   write-transaction was initialized. The success of the comparison is assert()ed. So if SQLite has written something other than the
//   original content to the database file, an assert() will fail.
//
//   Additionally, the set of page numbers for which records exist in the journal file is added to (unioned with) the set of page numbers
//   corresponding to free-list leaf pages collected when the write-transaction was initialized. This set comprises the page-numbers 
//   corresponding to those pages that SQLite may now safely modify.
//
//  Writing to the Database File:
//   When a block of data is written to a database file, the following invariants are asserted:
//
//     a) That the block of data is an aligned block of page-size bytes.
//     b) That if the page being written did not exist when the transaction was started (i.e. the database file is growing), then
//        the journal-file must have been synced at least once since the start of the transaction.
//     c) That if the page being written did exist when the transaction was started, then the page must have either been a free-list
//        leaf page at the start of the transaction, or else must have been stored in the journal file prior to the most recent sync.
//
//  Closing a Transaction:
//   When a transaction is closed, all data collected at the start of the transaction, or following an xSync of a journal-file, is 
//   discarded. The end of a transaction is recognized when any one of the following occur:
//
//     a) A block of zeroes (or anything else that is not a valid journal-header) is written to the start of the journal file.
//     b) A journal file is truncated to zero bytes in size using xTruncate.
//     c) The journal file is deleted using xDelete.
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <new.h>
#define pid uint32

// Maximum pathname length supported by the jt backend.
#define JT_MAX_PATHNAME 512
// Name used to identify this VFS.
#define JT_VFS_NAME "jt"

class JtVFile;
struct JtGlobal
{
	VSystem *Vfs;				// Parent VFS
	JtVFile *List;              // List of all open files
};
__device__ static struct JtGlobal _g = { nullptr, nullptr };

// Functions to obtain and relinquish a mutex to protect g.pList. The STATIC_PRNG mutex is reused, purely for the sake of convenience.
__device__ static void _enterJtMutex() { _mutex_enter(_mutex_alloc(MUTEX_STATIC_PRNG)); }
__device__ static void _leaveJtMutex() { _mutex_leave(_mutex_alloc(MUTEX_STATIC_PRNG)); }

namespace CORE_NAME { extern __device__ int g_io_error_pending; extern __device__ int g_io_error_hit; }
__device__ static void stop_ioerr_simulation(int *save, int *save2)
{
	*save = g_io_error_pending;
	*save2 = g_io_error_hit;
	g_io_error_pending = -1;
	g_io_error_hit = 0;
}
__device__ static void start_ioerr_simulation(int save, int save2)
{
	g_io_error_pending = save;
	g_io_error_hit = save2;
}

// Parameter z points to a buffer of 4 bytes in size containing a unsigned 32-bit integer stored in big-endian format. Decode the integer and return its value.
__device__ static uint32 DecodeUint32(const unsigned char *z) { return (z[0]<<24) + (z[1]<<16) + (z[2]<<8) + z[3]; }
// Calculate a checksum from the buffer of length n bytes pointed to by parameter z.
__device__ static uint32 GenChecksum(const unsigned char *z, int n)
{
	uint32 checksum = 0;
	for (int i = 0; i < n; i++)
		checksum = checksum + z[i] + (checksum<<3);
	return checksum;
}

class JtVFile : public VFile
{
public:
	const char *Path;       // Name of open file
	VSystem::OPEN Flags;    // Flags the file was opened with
	// The following are only used by database file file handles
	int Lock_;				// Current lock held on the file
	pid Pages;				// Size of file in pages when transaction started
	uint32 Pagesize;		// Page size when transaction started
	Bitvec *Writable;		// Bitvec of pages that may be written to the file
	uint32 *Checksum;       // Checksum for first nPage pages
	int Syncs;              // Number of times journal file has been synced
	// Only used by journal file-handles
	int64 MaxOff;		// Maximum offset written to this transaction
	//
	JtVFile *Next;      // All files are stored in a linked list
	VFile *Real;		// The file handle for the underlying vfs
public:
	// The jt_file pointed to by the argument may or may not be a file-handle open on a main database file. If it is, and a transaction is currently
	// opened on the file, then discard all transaction related data.
	__device__ void CloseTransaction()
	{
		Bitvec::Destroy(Writable);
		_free(Checksum);
		Writable = nullptr;
		Checksum = nullptr;
		Syncs = 0;
	}

	// Parameter zJournal is the name of a journal file that is currently open. This function locates and returns the handle opened on the
	// corresponding database file by the pager that currently has the journal file opened. This file-handle is identified by the 
	// following properties:
	//   a) SQLITE_OPEN_MAIN_DB was specified when the file was opened.
	//   b) The file-name specified when the file was opened matches all but the final 8 characters of the journal file name.
	//   c) There is currently a reserved lock on the file.
	__device__ static JtVFile *LocateDatabaseHandle(const char *journal)
	{
		JtVFile *main = nullptr;
		_enterJtMutex();
		for (main = _g.List; main; main = main->Next)
		{
			int pathLength = (int)(_strlen(journal) - _strlen("-journal"));
			if ((main->Flags & VSystem::OPEN_MAIN_DB) && (int)_strlen(main->Path) == pathLength && !_memcmp(main->Path, journal, pathLength) && main->Lock_ >= VFile::LOCK_RESERVED)
				break;
		}
		_leaveJtMutex();
		return main;
	}

	// The first argument, zBuf, points to a buffer containing a 28 byte serialized journal header. This function deserializes four of the
	// integer fields contained in the journal header and writes their values to the output variables.
	//
	// RC_OK is returned if the journal-header is successfully decoded. Otherwise, RC_ERROR.
	__device__ static RC DecodeJournalHdr(const unsigned char *buf, uint32 *recs, uint32 *pages, uint32 *sectorLength, uint32 *pagesize)
	{
		unsigned char magic[] = { 0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7 };
		if (_memcmp(magic, buf, 8)) return RC_ERROR;
		if (recs) *recs = DecodeUint32(&buf[8]);
		if (pages) *pages = DecodeUint32(&buf[16]);
		if (sectorLength) *sectorLength = DecodeUint32(&buf[20]);
		if (pagesize) *pagesize = DecodeUint32(&buf[24]);
		return RC_OK;
	}

	// This function is called when a new transaction is opened, just after the first journal-header is written to the journal file.
	__device__ static RC OpenTransaction(JtVFile *main, JtVFile *journal)
	{
		main->CloseTransaction();
		unsigned char *data = (unsigned char *)_alloc(main->Pagesize);
		main->Writable = Bitvec::New(main->Pages);
		main->Checksum = (uint32 *)_alloc(sizeof(uint32) * (main->Pages + 1));
		journal->MaxOff = 0;

		VFile *p = main->Real;
		RC rc = RC_OK;
		if (!main->Writable || !main->Checksum || !data)
			rc = RC_IOERR_NOMEM;
		else if (main->Pages > 0)
		{
			int save, save2;
			stop_ioerr_simulation(&save, &save2);
			// Read the database free-list. Add the page-number for each free-list leaf to the jt_file.pWritable bitvec.
			rc = p->Read(data, main->Pagesize, 0);
			if (rc == RC_OK)
			{
				uint32 dbsize = DecodeUint32(&data[28]);
				if (dbsize > 0 && !_memcmp(&data[24], &data[92], 4))
					for (pid pgid = dbsize+1; pgid <= main->Pages; pgid++)
						main->Writable->Set(pgid);
			}
			uint32 trunk = DecodeUint32(&data[32]);
			while (rc == RC_OK && trunk > 0)
			{
				int64 offset = (int64)(trunk-1)*main->Pagesize;
				rc = p->Read(data, main->Pagesize, offset);
				int leafs = DecodeUint32(&data[4]);
				for (uint32 leaf = 0; rc == RC_OK && leaf < leafs; leaf++)
				{
					pid pgid = DecodeUint32(&data[8+4*leaf]);
					main->Writable->Set(pgid);
				}
				trunk = DecodeUint32(data);
			}
			// Calculate and store a checksum for each page in the database file.
			if (rc == RC_OK)
			{
				for (int ii = 0; rc == RC_OK && ii < (int)main->Pages; ii++)
				{
					int64 offset = (int64)(main->Pagesize) * (int64)ii;
					if (offset == PENDING_BYTE) continue;
					rc = p->Read(data, main->Pagesize, offset);
					main->Checksum[ii] = GenChecksum(data, main->Pagesize);
					if (ii+1 == main->Pages && rc == RC_IOERR_SHORT_READ) rc = RC_OK;
				}
			}
			start_ioerr_simulation(save, save2);
		}
		_free(data);
		return rc;
	}

	// The first argument to this function is a handle open on a journal file. This function reads the journal file and adds the page number for each page in the journal to the Bitvec object passed as the second argument.
	__device__ RC ReadJournalFile(JtVFile *main)
	{
		unsigned char *page = (unsigned char *)_alloc(main->Pagesize);
		if (!page)
			return RC_IOERR_NOMEM;
		int save, save2;
		stop_ioerr_simulation(&save, &save2);

		RC rc = RC_OK;
		int64 offset = 0;
		int64 size = MaxOff;
		unsigned char buf[28];
		VFile *real = Real;
		while (rc == RC_OK && offset < size)
		{
			// Read and decode the next journal-header from the journal file.
			rc = real->Read(buf, 28, offset);
			uint32 recs, pages, sectorLength, pagesize;
			if (rc != RC_OK || DecodeJournalHdr(buf, &recs, &pages, &sectorLength, &pagesize))
				goto finish_rjf;
			offset += sectorLength;
			if (recs == 0)
			{
				// A trick. There might be another journal-header immediately following this one. In this case, 0 records means 0 records, not "read until the end of the file". See also ticket #2565.
				if (size >= (offset+sectorLength))
				{
					rc = real->Read(buf, 28, offset);
					if (rc != RC_OK || !DecodeJournalHdr(buf, nullptr, nullptr, nullptr, nullptr))
						continue;
				}
				recs = (uint32)((size-offset) / (main->Pagesize+8));
			}
			// Read all the records that follow the journal-header just read.
			for (uint32 ii = 0; rc == RC_OK && ii < recs && offset < size; ii++)
			{
				pid pgid;
				rc = real->Read(buf, 4, offset);
				if (rc == RC_OK)
				{
					pgid = DecodeUint32(buf);
					if (pgid > 0 && pgid <= main->Pages)
					{
						if (!main->Writable->Get(pgid))
						{
							rc = real->Read(page, main->Pagesize, offset+4);
							if (rc == RC_OK)
							{
								uint32 checksum = GenChecksum(page, main->Pagesize);
								_assert(checksum == main->Checksum[pgid-1]);
							}
						}
						main->Writable->Set(pgid);
					}
					offset += (8 + main->Pagesize);
				}
			}
			offset = ((offset + (sectorLength-1)) / sectorLength) * sectorLength;
		}
finish_rjf:
		start_ioerr_simulation(save, save2);
		_free(page);
		if (rc == RC_IOERR_SHORT_READ)
			rc = RC_OK;
		return rc;
	}

	__device__ virtual RC Close_()
	{
		CloseTransaction();
		_enterJtMutex();
		if (Path)
		{
			JtVFile **pp;
			for (pp = &_g.List; *pp != this; pp = &(*pp)->Next);
			*pp = Next;
		}
		_leaveJtMutex();
		return Real->Close();
	}
	__device__ virtual RC Read(void *buffer, int amount, int64 offset) { return Real->Read(buffer, amount, offset); }
	__device__ virtual RC Write(const void *buffer, int amount, int64 offset)
	{
		RC rc;
		if (Flags & VSystem::OPEN_MAIN_JOURNAL)
		{
			if (offset == 0)
			{
				JtVFile *main = LocateDatabaseHandle(Path);
				_assert(main);
				// Zeroing the first journal-file header. This is the end of a transaction.
				if (amount == 28)
					main->CloseTransaction();
				else if (amount != 12)
				{
					// Writing the first journal header to a journal file. This happens when a transaction is first started.
					uint8 *z = (uint8 *)buffer;
					main->Pages = DecodeUint32(&z[16]);
					main->Pagesize = DecodeUint32(&z[24]);
					if ((rc = OpenTransaction(main, this)) != RC_OK)
						return rc;
				}
			}
			if (MaxOff < (offset + amount))
				MaxOff = offset + amount;
		}
		if (Flags & VSystem::OPEN_MAIN_DB && Writable)
		{
			if (amount < (int)Pagesize && Pagesize%amount == 0 && offset >= (PENDING_BYTE+512) && offset+amount <= PENDING_BYTE+Pagesize)
			{
				// No-op. This special case is hit when the backup code is copying a to a database with a larger page-size than the source database and
				// it needs to fill in the non-locking-region part of the original pending-byte page.
			}
			else
			{
				pid pgid = (pid)(offset/Pagesize + 1);
				_assert((amount == 1||amount == Pagesize) && ((offset+amount)%Pagesize) == 0);
				_assert(pgid <= Pages || Syncs > 0);
				_assert(pgid > Pages || Writable->Get(pgid));
			}
		}
		rc = Real->Write(buffer, amount, offset);
		if ((Flags & VSystem::OPEN_MAIN_JOURNAL) && amount == 12)
		{
			JtVFile *main = LocateDatabaseHandle(Path);
			RC rc2 = ReadJournalFile(main);
			if (rc == RC_OK) rc = rc2;
		}
		return rc;
	}
	__device__ virtual RC Truncate(int64 size)
	{
		if ((Flags & VSystem::OPEN_MAIN_JOURNAL) && size == 0)
		{
			// Truncating a journal file. This is the end of a transaction.
			JtVFile *main = LocateDatabaseHandle(Path);
			main->CloseTransaction();
		}
		if ((Flags & VSystem::OPEN_MAIN_DB) && Writable)
		{
			uint32 locking_pageid = (uint32)(PENDING_BYTE/Pagesize+1);
			for (pid pgid = (pid)(size/Pagesize+1); pgid <= Pages; pgid++)
				_assert(pgid == locking_pageid || Writable->Get(pgid));
		}
		return Real->Truncate(size);
	}
	__device__ virtual RC Sync(SYNC flags)
	{
		if (Flags & VSystem::OPEN_MAIN_JOURNAL)
		{
			// The journal file is being synced. At this point, we inspect the contents of the file up to this point and set each bit in the 
			// jt_file.pWritable bitvec of the main database file associated with this journal file.
			JtVFile *main = LocateDatabaseHandle(Path); // The associated database file
			_assert(main);
			// Set the bitvec values
			if (Writable)
			{
				main->Syncs++;
				RC rc = ReadJournalFile(main);
				if (rc != RC_OK)
					return rc;
			}
		}
		return Real->Sync(flags);
	}
	__device__ virtual RC get_FileSize(int64 &size) { return Real->get_FileSize(size); }

	__device__ virtual RC Lock(LOCK lock)
	{ 
		RC rc = Real->Lock(lock);
		if (rc == RC_OK && lock > Lock_)
			Lock_ = lock;
		return rc;
	}
	__device__ virtual RC Unlock(LOCK lock)
	{ 
		RC rc = Real->Unlock(lock);
		if (rc == RC_OK && lock < Lock_)
			Lock_ = lock;
		return rc;
	}
	__device__ virtual RC CheckReservedLock(int &lock) { return Real->CheckReservedLock(lock); }
	__device__ virtual RC FileControl(FCNTL op, void *arg) { return Real->FileControl(op, arg); }

	__device__ virtual uint get_SectorSize() { return Real->get_SectorSize(); }
	__device__ virtual IOCAP get_DeviceCharacteristics() { return Real->get_DeviceCharacteristics(); }
};

class JtVSystem : public VSystem
{
public:
	VFile *Real; // The "real" underlying file descriptor
public:
	__device__ virtual VFile *_AttachFile(void *buffer) { return new (buffer) JtVFile(); }
	__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
	{
		JtVFile *p = (JtVFile *)file;
		p->Opened = false;
		p->Real = (VFile *)&p[1];
		p->Real->Opened = false;
		RC rc = _g.Vfs->Open(path, p->Real, flags, outFlags);
		_assert(rc == RC_OK || !p->Real->Opened);
		if (rc == RC_OK)
		{
			p->Opened = true;
			p->Lock_ = 0;
			p->Path = path;
			p->Flags = flags;
			p->Next = nullptr;
			p->Writable = 0;
			p->Checksum = 0;
			_enterJtMutex();
			if (path)
			{
				p->Next = _g.List;
				_g.List = p;
			}
			_leaveJtMutex();
		}
		return rc;
	}
	__device__ virtual RC Delete(const char *path, bool syncDirectory)
	{ 
		int pathLength = (int)_strlen(path);
		if (pathLength > 8 && !_strcmp("-journal", &path[pathLength-8]))
		{
			// Deleting a journal file. The end of a transaction.
			JtVFile *main = JtVFile::LocateDatabaseHandle(path);
			if (main)
				main->CloseTransaction();
		}
		return _g.Vfs->Delete(path, syncDirectory);
	}
	__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC) { return _g.Vfs->Access(path, flags, outRC); }
	__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut) { return _g.Vfs->FullPathname(path, pathOutLength, pathOut); }

#ifndef OMIT_LOAD_EXTENSION
	__device__ virtual void *DlOpen(const char *filename) { return _g.Vfs->DlOpen(filename); }
	__device__ virtual void DlError(int bufLength, char *buf) { return _g.Vfs->DlError(bufLength, buf); }
	__device__ virtual void (*DlSym(void *handle, const char *symbol))() { return _g.Vfs->DlSym(handle, symbol); }
	__device__ virtual void DlClose(void *handle) { return _g.Vfs->DlClose(handle); }
#endif
	__device__ virtual int Randomness(int bufLength, char *buf) { return _g.Vfs->Randomness(bufLength, buf); }
	__device__ virtual int Sleep(int microseconds) { return _g.Vfs->Sleep(microseconds); }
	__device__ virtual RC CurrentTimeInt64(int64 *now) { return _g.Vfs->CurrentTimeInt64(now); }
	__device__ virtual RC CurrentTime(double *now) { return _g.Vfs->CurrentTime(now); }
	//__device__ virtual RC GetLastError(int bufLength, char *buf);
};

// Configure the jt VFS as a wrapper around the VFS named by parameter zWrap. If the isDefault parameter is true, then the jt VFS is installed
// as the new default VFS for SQLite connections. If isDefault is not true, then the jt VFS is installed as non-default. In this case it
// is available via its name, "jt".
__device__ static unsigned char _jtVfsBuf[sizeof(JtVSystem)];
__device__ static JtVSystem *_jtVfs;
__device__ int jt_register(char *wrap, bool isDefault)
{
	_g.Vfs = VSystem::FindVfs(wrap);
	if (!_g.Vfs)
		return RC_ERROR;
	_jtVfs = new (_jtVfsBuf) JtVSystem();
	_jtVfs->SizeOsFile = sizeof(JtVFile) + _g.Vfs->SizeOsFile;
	_jtVfs->MaxPathname = JT_MAX_PATHNAME;
	_jtVfs->Name = JT_VFS_NAME;
	VSystem::RegisterVfs(_jtVfs, isDefault);
	return RC_OK;
}

// Uninstall the jt VFS, if it is installed.
__device__ void jt_unregister()
{
	VSystem::UnregisterVfs(_jtVfs);
}
