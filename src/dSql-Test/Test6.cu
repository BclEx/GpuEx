#include "TclContext.cu.h"

#ifndef OMIT_DISKIO  // This file is a no-op if disk I/O is disabled

//#define TRACE_CRASHTEST

typedef struct CrashFile CrashFile;
typedef struct CrashGlobal CrashGlobal;
typedef struct WriteBuffer WriteBuffer;

// Method:
//
//   This layer is implemented as a wrapper around the "real" sqlite3_file object for the host system. Each time data is 
//   written to the file object, instead of being written to the underlying file, the write operation is stored in an in-memory 
//   structure (type WriteBuffer). This structure is placed at the end of a global ordered list (the write-list).
//
//   When data is read from a file object, the requested region is first retrieved from the real file. The write-list is then 
//   traversed and data copied from any overlapping WriteBuffer structures to the output buffer. i.e. a read() operation following
//   one or more write() operations works as expected, even if no data has actually been written out to the real file.
//
//   When a fsync() operation is performed, an operating system crash may be simulated, in which case exit(-1) is called (the call to 
//   xSync() never returns). Whether or not a crash is simulated, the data associated with a subset of the WriteBuffer structures 
//   stored in the write-list is written to the real underlying files and the entries removed from the write-list. If a crash is simulated,
//   a subset of the buffers may be corrupted before the data is written.
//
//   The exact subset of the write-list written and/or corrupted is determined by the simulated device characteristics and sector-size.
//
// "Normal" mode:
//
//   Normal mode is used when the simulated device has none of the SQLITE_IOCAP_XXX flags set.
//
//   In normal mode, if the fsync() is not a simulated crash, the write-list is traversed from beginning to end. Each WriteBuffer
//   structure associated with the file handle used to call xSync() is written to the real file and removed from the write-list.
//
//   If a crash is simulated, one of the following takes place for each WriteBuffer in the write-list, regardless of which 
//   file-handle it is associated with:
//
//     1. The buffer is correctly written to the file, just as if a crash were not being simulated.
//
//     2. Nothing is done.
//
//     3. Garbage data is written to all sectors of the file that overlap the region specified by the WriteBuffer. Or garbage
//        data is written to some contiguous section within the overlapped sectors.
//
// Device Characteristic flag handling:
//
//   If the IOCAP_ATOMIC flag is set, then option (3) above is never selected.
//
//   If the IOCAP_ATOMIC512 flag is set, and the WriteBuffer represents an aligned write() of an integer number of 512 byte regions, then
//   option (3) above is never selected. Instead, each 512 byte region is either correctly written or left completely untouched. Similar
//   logic governs the behavior if any of the other ATOMICXXX flags is set.
//
//   If either the IOCAP_SAFEAPPEND or IOCAP_SEQUENTIAL flags are set and a crash is being simulated, then an entry of the write-list is
//   selected at random. Everything in the list after the selected entry is discarded before processing begins.
//
//   If IOCAP_SEQUENTIAL is set and a crash is being simulated, option (1) is selected for all write-list entries except the last. If a 
//   crash is not being simulated, then all entries in the write-list that occur before at least one write() on the file-handle specified
//   as part of the xSync() are written to their associated real files.
//
//   If IOCAP_SAFEAPPEND is set and the first byte written by the write() operation is one byte past the current end of the file, then option
//   (1) is always selected.

// Each write operation in the write-list is represented by an instance of the following structure.
//
// If zBuf is 0, then this structure represents a call to xTruncate(), not xWrite(). In that case, iOffset is the size that the file is
// truncated to.

struct WriteBuffer
{
	i64 Offset;                 // Byte offset of the start of this write()
	int BufLength;              // Number of bytes written
	uint8 *Buf;                 // Pointer to copy of written data
	CrashFile *File;            // File this write() applies to

	WriteBuffer *Next;          // Next in CrashGlobal.pWriteList
};

struct CrashFile
{
	//const sqlite3_io_methods *pMethod;   // Must be first
	VFile *RealFile;				// Underlying "real" file handle
	char *Name;
	VSystem::OPEN Flags;                      // Flags the file was opened with

	// Cache of the entire file. This is used to speed up OsRead() and OsFileSize() calls. Although both could be done by traversing the
	// write-list, in practice this is impractically slow.
	int Size;                      // Size of file in bytes
	int DataLength;                      // Size of buffer allocated at zData
	uint8 *Data;                   // Buffer containing file contents
};

struct CrashGlobal
{
	WriteBuffer *WriteList;     // Head of write-list
	WriteBuffer *WriteListEnd;  // End of write-list

	int SectorSize;             // Value of simulated sector size
	VFile::IOCAP DeviceCharacteristics;  // Value of simulated device characteristics

	int CrashAt;                // Crash on the iCrash'th call to xSync()
	char CrashFile[500];        // Crash during an xSync() on this file
};

static CrashGlobal _g = {0, 0, CORE_DEFAULT_SECTOR_SIZE, 0, 0};

// Set this global variable to 1 to enable crash testing.
__device__ static bool _crashTestEnable = false;

__device__ static void *crash_malloc(int bytes)
{
	return (void *)Tcl_Alloc((size_t)bytes);
}
__device__ static void crash_free(void *p)
{
	Tcl_Free(p);
}
__device__ static void *crash_realloc(void *p, int n)
{
	return (void *)Tcl_Realloc(p, (size_t)n);
}

// Wrapper around the sqlite3OsWrite() function that avoids writing to the 512 byte block begining at offset PENDING_BYTE.
__device__ static RC writeDbFile(CrashFile *p, uint8 *z, int64 amount, int64 offset)
{
	RC rc = RC_OK;
	int skip = (offset == PENDING_BYTE && (p->Flags & VSystem::OPEN_MAIN_DB) ? 512 : 0);
	if ((amount-skip) > 0)
		rc = p->RealFile->Write(&z[skip], (int)(amount-skip), offset+skip);
	return rc;
}

// Flush the write-list as if xSync() had been called on file handle pFile. If isCrash is true, simulate a crash.
__device__ static int writeListSync(CrashFile *file, bool isCrash)
{
	RC rc = RC_OK;
	VFile::IOCAP dc = _g.DeviceCharacteristics;

	// If this is not a crash simulation, set pFinal to point to the last element of the write-list that is associated with file handle pFile.
	//
	// If this is a crash simulation, set pFinal to an arbitrarily selected element of the write-list.
	WriteBuffer *write;
	WriteBuffer *final_ = nullptr;
	if (!isCrash)
	{
		for (write = _g.WriteList; write; write = write->Next)
			if (write->File == file)
				final_ = write;
	}
	else if (dc & (VFile::IOCAP_SEQUENTIAL|VFile::IOCAP_SAFE_APPEND))
	{
		int writes = 0;
		for (write = _g.WriteList; write; write = write->Next) writes++;
		int finalId;
		SysEx::PutRandom(sizeof(int), &finalId);
		finalId = (finalId < 0?-1*finalId:finalId)%writes;
		for (write = _g.WriteList; finalId > 0; write = write->Next) finalId--;
		final_ = write;
	}

#ifdef TRACE_CRASHTEST
	_printf("Sync %s (is %s crash)\n", file->Name, (isCrash?"a":"not a"));
#endif

	WriteBuffer **ptr = &_g.WriteList;
	for (write = *ptr; rc == RC_OK && write; write = *ptr)
	{
		VFile *realFile = write->File->RealFile;
		// (eAction == 1)      -> write block out normally,
		// (eAction == 2)      -> do nothing,
		// (eAction == 3)      -> trash sectors.
		int eAction = 0;
		if (!isCrash)
			eAction = (write->File == file || dc & VFile::IOCAP_SEQUENTIAL ? 1 : 2);
		else
		{
			char random;
			SysEx::PutRandom(1, &random);
			// Do not select option 3 (sector trashing) if the IOCAP_ATOMIC flag is set or this is an OsTruncate(), not an Oswrite().
			if ((dc & VFile::IOCAP_ATOMIC) || !write->Buf)
				random &= 0x01;
			// If IOCAP_SEQUENTIAL is set and this is not the final entry in the truncated write-list, always select option 1 (write out correctly).
			if (dc & VFile::IOCAP_SEQUENTIAL && write != final_)
				random = 0;
			// If IOCAP_SAFE_APPEND is set and this OsWrite() operation is an append (first byte of the written region is 1 byte past the
			// current EOF), always select option 1 (write out correctly).
			if (dc & VFile::IOCAP_SAFE_APPEND && write->Buf)
			{
				int64 size;
				realFile->get_FileSize(size);
				if (size == write->Offset)
					random = 0;
			}
			if ((random&0x06) == 0x06)
				eAction = 3;
			else
				eAction = ((random & 0x01) ? 2 : 1);
		}

		switch (eAction)
		{
		case 1: { // Write out correctly
			if (write->Buf)
				rc = writeDbFile(write->File, write->Buf, write->BufLength, write->Offset);
			else
				rc = realFile->Truncate(write->Offset);
			*ptr = write->Next;
#ifdef TRACE_CRASHTEST
			if (isCrash)
				_printf("Writing %d bytes @ %d (%s)\n", write->BufLength, (int)write->Offset, write->File->Name);
#endif
			crash_free(write);
			break; }
		case 2: { // Do nothing
			ptr = &write->Next;
#ifdef TRACE_CRASHTEST
			if (isCrash)
				_printf("Omiting %d bytes @ %d (%s)\n", write->BufLength, (int)write->Offset, write->File->Name);
#endif
			break; }
		case 3: { // Trash sectors
			int firstId = (int)(write->Offset/_g.SectorSize);
			int lastId = (int)((write->Offset+write->BufLength-1)/_g.SectorSize);
			_assert(write->Buf);
#ifdef TRACE_CRASHTEST
			_printf("Trashing %d sectors @ %lld (sector %d) (%s)\n", 1+lastId-firstId, write->Offset, firstId, write->File->Name);
#endif
			uint8 *garbage = crash_malloc(g.iSectorSize);
			if (garbage)
			{
				int64 i;
				for (i = firstId; rc == RC_OK && i <= lastId; i++)
				{
					SysEx::PutRandom(_g.SectorSize, garbage); 
					rc = writeDbFile(write->File, garbage, _g.SectorSize, i*_g.SectorSize);
				}
				crash_free(garbage);
			}
			else
				rc = RC_NOMEM;

			ptr = &write->Next;
			break; }
		default:
			_assert(!"Cannot happen");
		}
		if (write == final_) break;
	}

	if (rc == RC_OK && isCrash)
		exit(-1);

	for (write = _g.WriteList; write && write->Next; write = write->Next);
	_g.WriteListEnd = write;
	return rc;
}

// Add an entry to the end of the write-list.
__device__ static RC writeListAppend(VFile *file, int64 offset, const uint8 *buf, int bufLength)
{
	_assert((buf && bufLength) || (!bufLength && !buf));
	WriteBuffer *new_ = (WriteBuffer *)crash_malloc(sizeof(WriteBuffer) + bufLength);
	if (!new_)
		_fprintf(stderr, "out of memory in the crash simulator\n");
	_memset(new_, 0, sizeof(WriteBuffer) + bufLength);
	new_->Offset = offset;
	new_->BufLength = bufLength;
	new_->File = (CrashFile *)file;
	if (buf)
	{
		new_->Buf = (uint8 *)&new_[1];
		_memcpy(new_->Buf, buf, bufLength);
	}
	if (_g.WriteList)
	{
		_assert(_g.WriteListEnd);
		_g.WriteListEnd->Next = new_;
	}
	else
		_g.WriteList = new_;
	_g.WriteListEnd = new_;
	return RC_OK;
}

// Close a crash-file.
__device__ static RC cfClose(VFile *file)
{
	CrashFile *crashFile = (CrashFile *)file;
	writeListSync(crashFile, false);
	crashFile->RealFile->Close();
	return RC_OK;
}

// Read data from a crash-file.
__device__ static RC cfRead(VFile *file, void *buf, int amount, int64 offset)
{
	CrashFile *crashFile = (CrashFile *)file;
	// Check the file-size to see if this is a short-read
	if (crashFile->Size < (offset+amount))
		return RC_IOERR_SHORT_READ;
	_memcpy(buf, &crashFile->Data[offset], amount);
	return RC_OK;
}

// Write data to a crash-file.
__device__ static RC cfWrite(VFile *file, const void *buf, int amount, int64 offset)
{
	CrashFile *crashFile = (CrashFile *)file;
	if (amount+offset > crashFile->Size)
		crashFile->Size = (int)(amount+offset);
	while (crashFile->Size > crashFile->DataLength)
	{
		int newLength = (crashFile->DataLength*2) + 4096;
		uint8 *new_ = (uint8 *)crash_realloc(crashFile->Data, newLength);
		if (!new_)
			return RC_NOMEM;
		_memset(&new_[crashFile->DataLength], 0, newLength-crashFile->DataLength);
		crashFile->DataLength = newLength;
		crashFile->Data = new_;
	}
	_memcpy(&crashFile->Data[offset], buf, amount);
	return writeListAppend(file, offset, buf, amount);
}

/*
** Truncate a crash-file.
*/
static int cfTruncate(sqlite3_file *pFile, sqlite_int64 size){
	CrashFile *pCrash = (CrashFile *)pFile;
	assert(size>=0);
	if( pCrash->iSize>size ){
		pCrash->iSize = (int)size;
	}
	return writeListAppend(pFile, size, 0, 0);
}

/*
** Sync a crash-file.
*/
static int cfSync(sqlite3_file *pFile, int flags){
	CrashFile *pCrash = (CrashFile *)pFile;
	int isCrash = 0;

	const char *zName = pCrash->zName;
	const char *zCrashFile = g.zCrashFile;
	int nName = (int)strlen(zName);
	int nCrashFile = (int)strlen(zCrashFile);

	if( nCrashFile>0 && zCrashFile[nCrashFile-1]=='*' ){
		nCrashFile--;
		if( nName>nCrashFile ) nName = nCrashFile;
	}

#ifdef TRACE_CRASHTEST
	printf("cfSync(): nName = %d, nCrashFile = %d, zName = %s, zCrashFile = %s\n",
		nName, nCrashFile, zName, zCrashFile);
#endif

	if( nName==nCrashFile && 0==memcmp(zName, zCrashFile, nName) ){
#ifdef TRACE_CRASHTEST
		printf("cfSync(): name matched, g.iCrash = %d\n", g.iCrash);
#endif
		if( (--g.iCrash)==0 ) isCrash = 1;
	}

	return writeListSync(pCrash, isCrash);
}

/*
** Return the current file-size of the crash-file.
*/
static int cfFileSize(sqlite3_file *pFile, sqlite_int64 *pSize){
	CrashFile *pCrash = (CrashFile *)pFile;
	*pSize = (i64)pCrash->iSize;
	return SQLITE_OK;
}

/*
** Calls related to file-locks are passed on to the real file handle.
*/
static int cfLock(sqlite3_file *pFile, int eLock){
	return sqlite3OsLock(((CrashFile *)pFile)->pRealFile, eLock);
}
static int cfUnlock(sqlite3_file *pFile, int eLock){
	return sqlite3OsUnlock(((CrashFile *)pFile)->pRealFile, eLock);
}
static int cfCheckReservedLock(sqlite3_file *pFile, int *pResOut){
	return sqlite3OsCheckReservedLock(((CrashFile *)pFile)->pRealFile, pResOut);
}
static int cfFileControl(sqlite3_file *pFile, int op, void *pArg){
	if( op==SQLITE_FCNTL_SIZE_HINT ){
		CrashFile *pCrash = (CrashFile *)pFile;
		i64 nByte = *(i64 *)pArg;
		if( nByte>pCrash->iSize ){
			if( SQLITE_OK==writeListAppend(pFile, nByte, 0, 0) ){
				pCrash->iSize = (int)nByte;
			}
		}
		return SQLITE_OK;
	}
	return sqlite3OsFileControl(((CrashFile *)pFile)->pRealFile, op, pArg);
}

/*
** The xSectorSize() and xDeviceCharacteristics() functions return
** the global values configured by the [sqlite_crashparams] tcl
*  interface.
*/
static int cfSectorSize(sqlite3_file *pFile){
	return g.iSectorSize;
}
static int cfDeviceCharacteristics(sqlite3_file *pFile){
	return g.iDeviceCharacteristics;
}

/*
** Pass-throughs for WAL support.
*/
static int cfShmLock(sqlite3_file *pFile, int ofst, int n, int flags){
	return sqlite3OsShmLock(((CrashFile*)pFile)->pRealFile, ofst, n, flags);
}
static void cfShmBarrier(sqlite3_file *pFile){
	sqlite3OsShmBarrier(((CrashFile*)pFile)->pRealFile);
}
static int cfShmUnmap(sqlite3_file *pFile, int delFlag){
	return sqlite3OsShmUnmap(((CrashFile*)pFile)->pRealFile, delFlag);
}
static int cfShmMap(
	sqlite3_file *pFile,            /* Handle open on database file */
	int iRegion,                    /* Region to retrieve */
	int sz,                         /* Size of regions */
	int w,                          /* True to extend file if necessary */
	void volatile **pp              /* OUT: Mapped memory */
	){
		return sqlite3OsShmMap(((CrashFile*)pFile)->pRealFile, iRegion, sz, w, pp);
}

static const sqlite3_io_methods CrashFileVtab = {
	2,                            /* iVersion */
	cfClose,                      /* xClose */
	cfRead,                       /* xRead */
	cfWrite,                      /* xWrite */
	cfTruncate,                   /* xTruncate */
	cfSync,                       /* xSync */
	cfFileSize,                   /* xFileSize */
	cfLock,                       /* xLock */
	cfUnlock,                     /* xUnlock */
	cfCheckReservedLock,          /* xCheckReservedLock */
	cfFileControl,                /* xFileControl */
	cfSectorSize,                 /* xSectorSize */
	cfDeviceCharacteristics,      /* xDeviceCharacteristics */
	cfShmMap,                     /* xShmMap */
	cfShmLock,                    /* xShmLock */
	cfShmBarrier,                 /* xShmBarrier */
	cfShmUnmap                    /* xShmUnmap */
};

/*
** Application data for the crash VFS
*/
struct crashAppData {
	sqlite3_vfs *pOrig;                   /* Wrapped vfs structure */
};

/*
** Open a crash-file file handle.
**
** The caller will have allocated pVfs->szOsFile bytes of space
** at pFile. This file uses this space for the CrashFile structure
** and allocates space for the "real" file structure using 
** sqlite3_malloc(). The assumption here is (pVfs->szOsFile) is
** equal or greater than sizeof(CrashFile).
*/
static int cfOpen(
	sqlite3_vfs *pCfVfs,
	const char *zName,
	sqlite3_file *pFile,
	int flags,
	int *pOutFlags
	){
		sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
		int rc;
		CrashFile *pWrapper = (CrashFile *)pFile;
		sqlite3_file *pReal = (sqlite3_file*)&pWrapper[1];

		memset(pWrapper, 0, sizeof(CrashFile));
		rc = sqlite3OsOpen(pVfs, zName, pReal, flags, pOutFlags);

		if( rc==SQLITE_OK ){
			i64 iSize;
			pWrapper->pMethod = &CrashFileVtab;
			pWrapper->zName = (char *)zName;
			pWrapper->pRealFile = pReal;
			rc = sqlite3OsFileSize(pReal, &iSize);
			pWrapper->iSize = (int)iSize;
			pWrapper->flags = flags;
		}
		if( rc==SQLITE_OK ){
			pWrapper->nData = (4096 + pWrapper->iSize);
			pWrapper->zData = crash_malloc(pWrapper->nData);
			if( pWrapper->zData ){
				/* os_unix.c contains an assert() that fails if the caller attempts
				** to read data from the 512-byte locking region of a file opened
				** with the SQLITE_OPEN_MAIN_DB flag. This region of a database file
				** never contains valid data anyhow. So avoid doing such a read here.
				**
				** UPDATE: It also contains an assert() verifying that each call
				** to the xRead() method reads less than 128KB of data.
				*/
				const int isDb = (flags&SQLITE_OPEN_MAIN_DB);
				i64 iOff;

				memset(pWrapper->zData, 0, pWrapper->nData);
				for(iOff=0; iOff<pWrapper->iSize; iOff += 512){
					int nRead = pWrapper->iSize - (int)iOff;
					if( nRead>512 ) nRead = 512;
					if( isDb && iOff==PENDING_BYTE ) continue;
					rc = sqlite3OsRead(pReal, &pWrapper->zData[iOff], nRead, iOff);
				}
			}else{
				rc = SQLITE_NOMEM;
			}
		}
		if( rc!=SQLITE_OK && pWrapper->pMethod ){
			sqlite3OsClose(pFile);
		}
		return rc;
}

static int cfDelete(sqlite3_vfs *pCfVfs, const char *zPath, int dirSync){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xDelete(pVfs, zPath, dirSync);
}
static int cfAccess(
	sqlite3_vfs *pCfVfs, 
	const char *zPath, 
	int flags, 
	int *pResOut
	){
		sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
		return pVfs->xAccess(pVfs, zPath, flags, pResOut);
}
static int cfFullPathname(
	sqlite3_vfs *pCfVfs, 
	const char *zPath, 
	int nPathOut,
	char *zPathOut
	){
		sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
		return pVfs->xFullPathname(pVfs, zPath, nPathOut, zPathOut);
}
static void *cfDlOpen(sqlite3_vfs *pCfVfs, const char *zPath){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xDlOpen(pVfs, zPath);
}
static void cfDlError(sqlite3_vfs *pCfVfs, int nByte, char *zErrMsg){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	pVfs->xDlError(pVfs, nByte, zErrMsg);
}
static void (*cfDlSym(sqlite3_vfs *pCfVfs, void *pH, const char *zSym))(void){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xDlSym(pVfs, pH, zSym);
}
static void cfDlClose(sqlite3_vfs *pCfVfs, void *pHandle){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	pVfs->xDlClose(pVfs, pHandle);
}
static int cfRandomness(sqlite3_vfs *pCfVfs, int nByte, char *zBufOut){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xRandomness(pVfs, nByte, zBufOut);
}
static int cfSleep(sqlite3_vfs *pCfVfs, int nMicro){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xSleep(pVfs, nMicro);
}
static int cfCurrentTime(sqlite3_vfs *pCfVfs, double *pTimeOut){
	sqlite3_vfs *pVfs = (sqlite3_vfs *)pCfVfs->pAppData;
	return pVfs->xCurrentTime(pVfs, pTimeOut);
}

static int processDevSymArgs(
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[],
	int *piDeviceChar,
	int *piSectorSize
	){
		struct DeviceFlag {
			char *zName;
			int iValue;
		} aFlag[] = {
			{ "atomic",              SQLITE_IOCAP_ATOMIC                },
			{ "atomic512",           SQLITE_IOCAP_ATOMIC512             },
			{ "atomic1k",            SQLITE_IOCAP_ATOMIC1K              },
			{ "atomic2k",            SQLITE_IOCAP_ATOMIC2K              },
			{ "atomic4k",            SQLITE_IOCAP_ATOMIC4K              },
			{ "atomic8k",            SQLITE_IOCAP_ATOMIC8K              },
			{ "atomic16k",           SQLITE_IOCAP_ATOMIC16K             },
			{ "atomic32k",           SQLITE_IOCAP_ATOMIC32K             },
			{ "atomic64k",           SQLITE_IOCAP_ATOMIC64K             },
			{ "sequential",          SQLITE_IOCAP_SEQUENTIAL            },
			{ "safe_append",         SQLITE_IOCAP_SAFE_APPEND           },
			{ "powersafe_overwrite", SQLITE_IOCAP_POWERSAFE_OVERWRITE   },
			{ 0, 0 }
		};

		int i;
		int iDc = 0;
		int iSectorSize = 0;
		int setSectorsize = 0;
		int setDeviceChar = 0;

		for(i=0; i<objc; i+=2){
			int nOpt;
			char *zOpt = Tcl_GetStringFromObj(objv[i], &nOpt);

			if( (nOpt>11 || nOpt<2 || strncmp("-sectorsize", zOpt, nOpt)) 
				&& (nOpt>16 || nOpt<2 || strncmp("-characteristics", zOpt, nOpt))
				){
					Tcl_AppendResult(interp, 
						"Bad option: \"", zOpt, 
						"\" - must be \"-characteristics\" or \"-sectorsize\"", 0
						);
					return TCL_ERROR;
			}
			if( i==objc-1 ){
				Tcl_AppendResult(interp, "Option requires an argument: \"", zOpt, "\"",0);
				return TCL_ERROR;
			}

			if( zOpt[1]=='s' ){
				if( Tcl_GetIntFromObj(interp, objv[i+1], &iSectorSize) ){
					return TCL_ERROR;
				}
				setSectorsize = 1;
			}else{
				int j;
				Tcl_Obj **apObj;
				int nObj;
				if( Tcl_ListObjGetElements(interp, objv[i+1], &nObj, &apObj) ){
					return TCL_ERROR;
				}
				for(j=0; j<nObj; j++){
					int rc;
					int iChoice;
					Tcl_Obj *pFlag = Tcl_DuplicateObj(apObj[j]);
					Tcl_IncrRefCount(pFlag);
					Tcl_UtfToLower(Tcl_GetString(pFlag));

					rc = Tcl_GetIndexFromObjStruct(
						interp, pFlag, aFlag, sizeof(aFlag[0]), "no such flag", 0, &iChoice
						);
					Tcl_DecrRefCount(pFlag);
					if( rc ){
						return TCL_ERROR;
					}

					iDc |= aFlag[iChoice].iValue;
				}
				setDeviceChar = 1;
			}
		}

		if( setDeviceChar ){
			*piDeviceChar = iDc;
		}
		if( setSectorsize ){
			*piSectorSize = iSectorSize;
		}

		return TCL_OK;
}

/*
** tclcmd:   sqlite_crash_enable ENABLE
**
** Parameter ENABLE must be a boolean value. If true, then the "crash"
** vfs is added to the system. If false, it is removed.
*/
static int crashEnableCmd(
	void * clientData,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	int isEnable;
	static sqlite3_vfs crashVfs = {
		2,                  /* iVersion */
		0,                  /* szOsFile */
		0,                  /* mxPathname */
		0,                  /* pNext */
		"crash",            /* zName */
		0,                  /* pAppData */

		cfOpen,               /* xOpen */
		cfDelete,             /* xDelete */
		cfAccess,             /* xAccess */
		cfFullPathname,       /* xFullPathname */
		cfDlOpen,             /* xDlOpen */
		cfDlError,            /* xDlError */
		cfDlSym,              /* xDlSym */
		cfDlClose,            /* xDlClose */
		cfRandomness,         /* xRandomness */
		cfSleep,              /* xSleep */
		cfCurrentTime,        /* xCurrentTime */
		0,                    /* xGetlastError */
		0,                    /* xCurrentTimeInt64 */
	};

	if( objc!=2 ){
		Tcl_WrongNumArgs(interp, 1, objv, "ENABLE");
		return TCL_ERROR;
	}

	if( Tcl_GetBooleanFromObj(interp, objv[1], &isEnable) ){
		return TCL_ERROR;
	}

	if( (isEnable && crashVfs.pAppData) || (!isEnable && !crashVfs.pAppData) ){
		return TCL_OK;
	}

	if( crashVfs.pAppData==0 ){
		sqlite3_vfs *pOriginalVfs = sqlite3_vfs_find(0);
		crashVfs.mxPathname = pOriginalVfs->mxPathname;
		crashVfs.pAppData = (void *)pOriginalVfs;
		crashVfs.szOsFile = sizeof(CrashFile) + pOriginalVfs->szOsFile;
		sqlite3_vfs_register(&crashVfs, 0);
	}else{
		crashVfs.pAppData = 0;
		sqlite3_vfs_unregister(&crashVfs);
	}

	return TCL_OK;
}

/*
** tclcmd:   sqlite_crashparams ?OPTIONS? DELAY CRASHFILE
**
** This procedure implements a TCL command that enables crash testing
** in testfixture.  Once enabled, crash testing cannot be disabled.
**
** Available options are "-characteristics" and "-sectorsize". Both require
** an argument. For -sectorsize, this is the simulated sector size in
** bytes. For -characteristics, the argument must be a list of io-capability
** flags to simulate. Valid flags are "atomic", "atomic512", "atomic1K",
** "atomic2K", "atomic4K", "atomic8K", "atomic16K", "atomic32K", 
** "atomic64K", "sequential" and "safe_append".
**
** Example:
**
**   sqlite_crashparams -sect 1024 -char {atomic sequential} ./test.db 1
**
*/
static int crashParamsObjCmd(
	void * clientData,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	int iDelay;
	const char *zCrashFile;
	int nCrashFile, iDc, iSectorSize;

	iDc = -1;
	iSectorSize = -1;

	if( objc<3 ){
		Tcl_WrongNumArgs(interp, 1, objv, "?OPTIONS? DELAY CRASHFILE");
		goto error;
	}

	zCrashFile = Tcl_GetStringFromObj(objv[objc-1], &nCrashFile);
	if( nCrashFile>=sizeof(g.zCrashFile) ){
		Tcl_AppendResult(interp, "Filename is too long: \"", zCrashFile, "\"", 0);
		goto error;
	}
	if( Tcl_GetIntFromObj(interp, objv[objc-2], &iDelay) ){
		goto error;
	}

	if( processDevSymArgs(interp, objc-3, &objv[1], &iDc, &iSectorSize) ){
		return TCL_ERROR;
	}

	if( iDc>=0 ){
		g.iDeviceCharacteristics = iDc;
	}
	if( iSectorSize>=0 ){
		g.iSectorSize = iSectorSize;
	}

	g.iCrash = iDelay;
	memcpy(g.zCrashFile, zCrashFile, nCrashFile+1);
	sqlite3CrashTestEnable = 1;
	return TCL_OK;

error:
	return TCL_ERROR;
}

static int devSymObjCmd(
	void * clientData,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	void devsym_register(int iDeviceChar, int iSectorSize);

	int iDc = -1;
	int iSectorSize = -1;

	if( processDevSymArgs(interp, objc-1, &objv[1], &iDc, &iSectorSize) ){
		return TCL_ERROR;
	}
	devsym_register(iDc, iSectorSize);

	return TCL_OK;
}

/*
** tclcmd: register_jt_vfs ?-default? PARENT-VFS
*/
static int jtObjCmd(
	void * clientData,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	int jt_register(char *, int);
	char *zParent = 0;

	if( objc!=2 && objc!=3 ){
		Tcl_WrongNumArgs(interp, 1, objv, "?-default? PARENT-VFS");
		return TCL_ERROR;
	}
	zParent = Tcl_GetString(objv[1]);
	if( objc==3 ){
		if( strcmp(zParent, "-default") ){
			Tcl_AppendResult(interp, 
				"bad option \"", zParent, "\": must be -default", 0
				);
			return TCL_ERROR;
		}
		zParent = Tcl_GetString(objv[2]);
	}

	if( !(*zParent) ){
		zParent = 0;
	}
	if( jt_register(zParent, objc==3) ){
		Tcl_AppendResult(interp, "Error in jt_register", 0);
		return TCL_ERROR;
	}

	return TCL_OK;
}

/*
** tclcmd: unregister_jt_vfs
*/
static int jtUnregisterObjCmd(
	void * clientData,
	Tcl_Interp *interp,
	int objc,
	Tcl_Obj *CONST objv[]
){
	void jt_unregister(void);

	if( objc!=1 ){
		Tcl_WrongNumArgs(interp, 1, objv, "");
		return TCL_ERROR;
	}

	jt_unregister();
	return TCL_OK;
}

#endif /* SQLITE_OMIT_DISKIO */

/*
** This procedure registers the TCL procedures defined in this file.
*/
int Sqlitetest6_Init(Tcl_Interp *interp){
#ifndef SQLITE_OMIT_DISKIO
	Tcl_CreateObjCommand(interp, "sqlite3_crash_enable", crashEnableCmd, 0, 0);
	Tcl_CreateObjCommand(interp, "sqlite3_crashparams", crashParamsObjCmd, 0, 0);
	Tcl_CreateObjCommand(interp, "sqlite3_simulate_device", devSymObjCmd, 0, 0);
	Tcl_CreateObjCommand(interp, "register_jt_vfs", jtObjCmd, 0, 0);
	Tcl_CreateObjCommand(interp, "unregister_jt_vfs", jtUnregisterObjCmd, 0, 0);
#endif
	return TCL_OK;
}
