// This file contains the implementation of the Tcl [testvfs] command, used to create SQLite VFS implementations with various properties and
// instrumentation to support testing SQLite.
//
//   testvfs VFSNAME ?OPTIONS?
//
// Available options are:
//
//   -noshm      BOOLEAN        (True to omit shm methods. Default false)
//   -default    BOOLEAN        (True to make the vfs default. Default false)
//   -szosfile   INTEGER        (Value for sqlite3_vfs.szOsFile)
//   -mxpathname INTEGER        (Value for sqlite3_vfs.mxPathname)
//   -iversion   INTEGER        (Value for sqlite3_vfs.iVersion)
#if _TEST

#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>
#include <Jim+EventLoop.h>
#include <new.h>

struct TestvfsBuffer;
struct TestvfsFd
{
	VSystem *Vfs;				// The VFS
	const char *Filename;       // Filename as passed to xOpen()
	VFile *Real;				// The real, underlying file descriptor
	Jim_Obj *ShmId;             // Shared memory id for Tcl callbacks
	TestvfsBuffer *Shm;         // Shared memory buffer
	uint32 excllock;            // Mask of exclusive locks
	uint32 sharedlock;          // Mask of shared locks
	TestvfsFd *Next;            // Next handle opened on the same file
};

// The Testvfs.mask variable is set to a combination of the following. If a bit is clear in Testvfs.mask, then calls made by SQLite to the 
// corresponding VFS method is ignored for purposes of:
//   + Simulating IO errors, and
//   + Invoking the Tcl callback script.
#define TESTVFS_SHMOPEN_MASK      0x00000001
#define TESTVFS_SHMLOCK_MASK      0x00000010
#define TESTVFS_SHMMAP_MASK       0x00000020
#define TESTVFS_SHMBARRIER_MASK   0x00000040
#define TESTVFS_SHMCLOSE_MASK     0x00000080
#define TESTVFS_OPEN_MASK         0x00000100
#define TESTVFS_SYNC_MASK         0x00000200
#define TESTVFS_DELETE_MASK       0x00000400
#define TESTVFS_CLOSE_MASK        0x00000800
#define TESTVFS_WRITE_MASK        0x00001000
#define TESTVFS_TRUNCATE_MASK     0x00002000
#define TESTVFS_ACCESS_MASK       0x00004000
#define TESTVFS_FULLPATHNAME_MASK 0x00008000
#define TESTVFS_READ_MASK         0x00010000
#define TESTVFS_ALL_MASK          0x0001FFFF
#define TESTVFS_MAX_PAGES 1024
#define TESTVFS_MAX_ARGS 12

// A shared-memory buffer. There is one of these objects for each shared memory region opened by clients. If two clients open the same file,
// there are two TestvfsFile structures but only one TestvfsBuffer structure.
struct TestvfsBuffer
{
	char *FileName;					// Associated file name
	int Pagesize;                   // Page size
	uint8 *Pages[TESTVFS_MAX_PAGES]; // Array of ckalloc'd pages
	TestvfsFd *Files;				// List of open handles
	TestvfsBuffer *Next;			// Next in linked list of all buffers
};
__device__ static void AllocPage(TestvfsBuffer *p, int page, int pagesize)
{
	_assert(page < TESTVFS_MAX_PAGES);
	if (!p->Pages[page])
	{
		p->Pages[page] = (uint8 *)_alloc(pagesize);
		memset(p->Pages[page], 0, pagesize);
		p->Pagesize = pagesize;
	}
}

class TestVfsVSystem;
__device__ static bool tvfsResultCode(TestVfsVSystem *p, RC *rc);

#define FAULT_INJECT_NONE       0
#define FAULT_INJECT_TRANSIENT  1
#define FAULT_INJECT_PERSISTENT 2
typedef struct TestFaultInject TestFaultInject;
struct TestFaultInject
{
	int Cnts;           // Remaining calls before fault injection
	int Fault;         // A FAULT_INJECT_* value
	int Fails;          // Number of faults injected
};

class TestVfsVFile;
class TestVfsVSystem : public VSystem
{
public:
	char *Name;                // Name of this VFS
	VSystem *Parent;			// The VFS to use for file IO
	VSystem *Vfs;              // The testvfs registered with SQLite
	Jim_Interp *interp;         // Interpreter to run script in
	Jim_Obj *Script;           // Script to execute
	TestvfsBuffer *Buffer;     // List of shared buffers
	int IsNoshm;
	int IsFullshm;
	int Mask;                   // Mask controlling [script] and [ioerr]
	TestFaultInject ioerr_err;
	TestFaultInject full_err;
	TestFaultInject cantopen_err;
	VFile::IOCAP Devchar;
	int Sectorsize;
public:
	__device__ static bool InjectFault(TestFaultInject *p)
	{
		bool ret = false;
		if (p->Fault)
		{
			p->Cnts--;
			if (p->Cnts == 0 || (p->Cnts < 0 && p->Fault == FAULT_INJECT_PERSISTENT)) { ret = true; p->Fails++; }
		}
		return ret;
	}
	__device__ _inline int InjectIoerr() { return InjectFault(&ioerr_err); }
	__device__ _inline int InjectFullerr() { return InjectFault(&full_err); }
	__device__ _inline int InjectCantopenerr() { return InjectFault(&cantopen_err); }

	__device__ void ExecTcl(const char *method, Jim_Obj *arg1, Jim_Obj *arg2, Jim_Obj *arg3, Jim_Obj *arg4)
	{
		_assert(Script);
		_assert(method);
		_assert(arg2 == 0 || arg1 != 0);
		_assert(arg3 == 0 || arg2 != 0);
		Jim_Obj *eval = Jim_DuplicateObj(interp, Script);
		Jim_IncrRefCount(Script);
		Jim_ListAppendElement(interp, eval, Jim_NewStringObj(interp, method, -1));
		if (arg1) Jim_ListAppendElement(interp, eval, arg1);
		if (arg2) Jim_ListAppendElement(interp, eval, arg2);
		if (arg3) Jim_ListAppendElement(interp, eval, arg3);
		if (arg4) Jim_ListAppendElement(interp, eval, arg4);
		int rc = Jim_EvalGlobal(interp, Jim_String(eval));
		if (rc != JIM_OK)
		{
			Jim_BackgroundError(interp);
			Jim_ResetResult(interp);
		}
	}

	__device__ virtual VFile *_AttachFile(void *buffer);
	__device__ virtual RC Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags);
	__device__ virtual RC Delete(const char *path, bool syncDirectory)
	{
		RC rc = RC_OK;
		if (Script && (Mask&TESTVFS_DELETE_MASK))
		{
			ExecTcl("xDelete", Jim_NewStringObj(interp, path, -1), Jim_NewIntObj(interp, syncDirectory), nullptr, nullptr);
			tvfsResultCode(this, &rc);
		}
		if (rc == RC_OK)
			rc = Parent->Delete(path, syncDirectory);
		return rc;
	}
	__device__ virtual RC Access(const char *path, ACCESS flags, int *outRC)
	{
		if (Script && (Mask&TESTVFS_ACCESS_MASK))
		{
			char *arg = nullptr;
			if (flags == ACCESS_EXISTS) arg = "SQLITE_ACCESS_EXISTS";
			if (flags == ACCESS_READWRITE) arg = "SQLITE_ACCESS_READWRITE";
			if (flags == ACCESS_READ) arg = "SQLITE_ACCESS_READ";
			ExecTcl("xAccess", Jim_NewStringObj(interp, path, -1), Jim_NewStringObj(interp, arg, -1), nullptr, nullptr);
			RC rc;
			if (tvfsResultCode(this, &rc))
			{
				if (rc != RC_OK) return rc;
			}
			else
			{
				if (Jim_GetInt(interp, Jim_GetResult(interp), outRC) == JIM_OK)
					return RC_OK;
			}
		}
		return Parent->Access(path, flags, outRC);
	}
	__device__ virtual RC FullPathname(const char *path, int pathOutLength, char *pathOut)
	{
		if (Script && (Mask&TESTVFS_FULLPATHNAME_MASK))
		{
			ExecTcl("xFullPathname", Jim_NewStringObj(interp, path, -1), nullptr, nullptr, nullptr);
			RC rc;
			if (tvfsResultCode(this, &rc))
				if (rc != RC_OK) return rc;
		}
		return Parent->FullPathname(path, pathOutLength, pathOut);
	}

#ifndef OMIT_LOAD_EXTENSION
	__device__ virtual void *DlOpen(const char *filename) { return Parent->DlOpen(filename); }
	__device__ virtual void DlError(int bufLength, char *buf) { Parent->DlError(bufLength, buf); }
	__device__ virtual void (*DlSym(void *handle, const char *symbol))() { return Parent->DlSym(handle, symbol); }
	__device__ virtual void DlClose(void *handle) { Parent->DlClose(handle); }
#endif

	__device__ virtual int Randomness(int bufLength, char *buf) { return Parent->Randomness(bufLength, buf); }
	__device__ virtual int Sleep(int microseconds) { return Parent->Sleep(microseconds); }
	__device__ virtual RC CurrentTime(double *now) { return Parent->CurrentTime(now); }
};

struct errcode {
	RC Code;
	const char *Name;
} _codes[] = {
	{ RC_OK,     "SQLITE_OK"     },
	{ RC_ERROR,  "SQLITE_ERROR"  },
	{ RC_IOERR,  "SQLITE_IOERR"  },
	{ RC_LOCKED, "SQLITE_LOCKED" },
	{ RC_BUSY,   "SQLITE_BUSY"   },
};
__device__ static bool tvfsResultCode(TestVfsVSystem *p, RC *rc)
{
	const char *z = Jim_String(Jim_GetResult(p->interp));
	for (int i = 0; i < _lengthof(_codes); i++)
	{
		if (!_strcmp(z, _codes[i].Name))
		{
			*rc = _codes[i].Code;
			return true;
		}
	}
	return false;
}

class TestVfsVFile : public VFile
{
public:
	TestvfsFd *Fd; // File data
public:
	__device__ virtual RC Close_()
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		if (p->Script && (p->Mask&TESTVFS_CLOSE_MASK))
			p->ExecTcl("xClose", Jim_NewStringObj(interp, fd->Filename, -1), fd->ShmId, nullptr, nullptr);
		if (fd->ShmId)
		{
			Jim_DecrRefCount(interp, fd->ShmId);
			fd->ShmId = nullptr;
		}
		if (Opened)
		{
			//ckfree((char *)pFile->pMethods);
			Opened = false;
		}
		RC rc = fd->Real->Close();
		_free((char *)fd);
		Fd = nullptr;
		return rc;
	}
	__device__ virtual RC Read(void *buffer, int amount, int64 offset)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		RC rc = RC_OK;
		if (p->Script && (p->Mask&TESTVFS_READ_MASK))
		{
			p->ExecTcl("xRead", Jim_NewStringObj(interp, fd->Filename, -1), fd->ShmId, nullptr, nullptr);
			tvfsResultCode(p, &rc);
		}
		if (rc == RC_OK && (p->Mask&TESTVFS_READ_MASK) && p->InjectIoerr())
			rc = RC_IOERR;
		if (rc == RC_OK)
			rc = fd->Real->Read(buffer, amount, offset);
		return rc;
	}
	__device__ virtual RC Write(const void *buffer, int amount, int64 offset)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		RC rc = RC_OK;
		if (p->Script && (p->Mask&TESTVFS_WRITE_MASK))
		{
			p->ExecTcl("xWrite", Jim_NewStringObj(interp, fd->Filename, -1), fd->ShmId, Jim_NewWideObj(interp, offset), Jim_NewIntObj(interp, amount));
			tvfsResultCode(p, &rc);
		}
		if (rc == RC_OK && p->InjectFullerr())
			rc = RC_FULL;
		if (rc == RC_OK && (p->Mask&TESTVFS_WRITE_MASK) && p->InjectIoerr())
			rc = RC_IOERR;
		if (rc == RC_OK)
			rc = fd->Real->Write(buffer, amount, offset);
		return rc;
	}
	__device__ virtual RC Truncate(int64 size)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		RC rc = RC_OK;
		if (p->Script && (p->Mask&TESTVFS_TRUNCATE_MASK))
		{
			p->ExecTcl("xTruncate", Jim_NewStringObj(interp, fd->Filename, -1), fd->ShmId, nullptr, nullptr);
			tvfsResultCode(p, &rc);
		}
		if (rc == RC_OK)
			rc = fd->Real->Truncate(size);
		return rc;
	}
	__device__ virtual RC Sync(SYNC flags)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		RC rc = RC_OK;
		if (p->Script && (p->Mask&TESTVFS_SYNC_MASK))
		{
			char *flagsName;
			switch (flags){
			case SYNC_NORMAL: flagsName = "normal"; break;
			case SYNC_FULL: flagsName = "full"; break;
			case SYNC_NORMAL|SYNC_DATAONLY: flagsName = "normal|dataonly"; break;
			case SYNC_FULL|SYNC_DATAONLY: flagsName = "full|dataonly"; break;
			default: _assert(false);
			}
			p->ExecTcl("xSync", Jim_NewStringObj(interp, fd->Filename, -1), fd->ShmId, Jim_NewStringObj(interp, flagsName, -1), nullptr);
			tvfsResultCode(p, &rc);
		}
		if (rc == RC_OK && p->InjectFullerr())
			rc = RC_FULL;
		if (rc == RC_OK)
			rc = fd->Real->Sync(flags);
		return rc;
	}
	__device__ virtual RC get_FileSize(int64 &size) { return Fd->Real->get_FileSize(size); }
	__device__ virtual RC Lock(LOCK lock) { return Fd->Real->Lock(lock); }
	__device__ virtual RC Unlock(LOCK lock) { return Fd->Real->Unlock(lock); } 
	__device__ virtual RC CheckReservedLock(int &lock) { return Fd->Real->CheckReservedLock(lock); }
	__device__ virtual RC FileControl(FCNTL op, void *arg)
	{
		if (op == FCNTL_PRAGMA)
		{
			char **argv = (char **)arg;
			if (!_stricmp(argv[1], "error"))
			{
				RC rc = RC_ERROR;
				if (argv[2])
				{
					const char *z = argv[2];
					int x = _atoi(z);
					if (x)
					{
						rc = (RC)x;
						while (_isdigit(z[0])) { z++; }
						while (_isspace(z[0])) { z++; }
					}
					if (z[0]) argv[0] = _mprintf("%s", z);
				}
				return rc;
			}
			if (!_stricmp(argv[1], "filename"))
			{
				argv[0] = _mprintf("%s", Fd->Filename);
				return RC_OK;
			}
		}
		return Fd->Real->FileControl(op, arg);
	}

	__device__ virtual uint get_SectorSize()
	{
		TestVfsVSystem *p = (TestVfsVSystem *)Fd->Vfs;
		return (p->Sectorsize >= 0 ? p->Sectorsize : Fd->Real->get_SectorSize());
	}
	__device__ virtual IOCAP get_DeviceCharacteristics()
	{
		TestVfsVSystem *p = (TestVfsVSystem *)Fd->Vfs;
		return (p->Devchar >= 0 ? p->Devchar : Fd->Real->get_DeviceCharacteristics());
	}

	__device__ RC ShmOpen()
	{
		TestvfsFd *fd = Fd; // The testvfs file structure
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		_assert(!p->IsFullshm);
		_assert(fd->ShmId && !fd->Shm && !fd->Next);
		// Evaluate the Tcl script: 
		//   SCRIPT xShmOpen FILENAME
		RC rc = RC_OK;
		Jim_ResetResult(interp);
		if (p->Script && (p->Mask&TESTVFS_SHMOPEN_MASK))
		{
			p->ExecTcl("xShmOpen", Jim_NewStringObj(interp, fd->Filename, -1), nullptr, nullptr, nullptr);
			if (tvfsResultCode(p, &rc))
			{
				if (rc != RC_OK) return rc;
			}
		}
		_assert(rc == RC_OK);
		if ((p->Mask&TESTVFS_SHMOPEN_MASK) && p->InjectIoerr())
			return RC_IOERR;
		// Search for a TestvfsBuffer. Create a new one if required.
		TestvfsBuffer *buffer; // Buffer to open connection to
		for (buffer = p->Buffer; buffer; buffer = buffer->Next)
			if (!_strcmp(fd->Filename, buffer->FileName))
				break;
		if (!buffer)
		{
			int bytes = sizeof(TestvfsBuffer) + (int)_strlen(fd->Filename) + 1;
			buffer = (TestvfsBuffer *)_alloc(bytes);
			memset(buffer, 0, bytes);
			buffer->FileName = (char *)&buffer[1];
			strcpy(buffer->FileName, fd->Filename);
			buffer->Next = p->Buffer;
			p->Buffer = buffer;
		}
		// Connect the TestvfsBuffer to the new TestvfsShm handle and return.
		fd->Next = buffer->Files;
		buffer->Files = fd;
		fd->Shm = buffer;
		return RC_OK;
	}
	__device__ virtual RC ShmLock(int offset, int n, SHM flags)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		if (p->IsFullshm)
			return fd->Real->ShmLock(offset, n, flags);
		RC rc = RC_OK;
		if (p->Script && (p->Mask&TESTVFS_SHMLOCK_MASK))
		{
			char lock[80];
			_snprintf(lock, sizeof(lock), "%d %d", offset, n);
			int lockLength = (int)_strlen(lock);
			_strcpy(&lock[lockLength], (flags&SHM_LOCK ? " lock" : " unlock"));
			lockLength += (int)strlen(&lock[lockLength]);
			_strcpy(&lock[lockLength], (flags&SHM_SHARED ? " shared" : " exclusive"));
			p->ExecTcl("xShmLock", Jim_NewStringObj(interp, fd->Shm->FileName, -1), fd->ShmId, Jim_NewStringObj(interp, lock, -1), nullptr);
			tvfsResultCode(p, &rc);
		}
		if (rc == RC_OK && (p->Mask&TESTVFS_SHMLOCK_MASK) && p->InjectIoerr())
			rc = RC_IOERR;
		if (rc == RC_OK)
		{
			int isLock = (flags&SHM_LOCK);
			int isExcl = (flags&SHM_EXCLUSIVE);
			uint32 mask = (((1<<n)-1) << offset);
			if (isLock)
			{
				for (TestvfsFd *p2 = fd->Shm->Files; p2; p2 = p2->Next)
				{
					if (p2 == fd) continue;
					if ((p2->excllock&mask) || (isExcl && p2->sharedlock&mask))
					{
						rc = RC_BUSY;
						break;
					}
				}
				if (rc == RC_OK)
				{
					if (isExcl) fd->excllock |= mask;
					if (!isExcl) fd->sharedlock |= mask;
				}
			}
			else
			{
				if (isExcl) fd->excllock &= (~mask);
				if (!isExcl) fd->sharedlock &= (~mask);
			}
		}
		return rc;
	}
	__device__ virtual void ShmBarrier()
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		if (p->IsFullshm)
		{
			fd->Real->ShmBarrier();
			return;
		}
		if (p->Script && (p->Mask&TESTVFS_SHMBARRIER_MASK))
			p->ExecTcl("xShmBarrier", Jim_NewStringObj(interp, fd->Shm->FileName, -1), fd->ShmId, nullptr, nullptr);
	}
	__device__ virtual RC ShmUnmap(bool deleteFlag)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		if (p->IsFullshm)
			return fd->Real->ShmUnmap(deleteFlag);
		RC rc = RC_OK;
		TestvfsBuffer *buffer = fd->Shm;
		if (!buffer) return RC_OK;
		_assert(fd->ShmId && fd->Shm);
		if (p->Script && (p->Mask&TESTVFS_SHMCLOSE_MASK))
		{
			p->ExecTcl("xShmUnmap", Jim_NewStringObj(interp, fd->Shm->FileName, -1), fd->ShmId, nullptr, nullptr);
			tvfsResultCode(p, &rc);
		}
		TestvfsFd **fd_;
		for (fd_ = &buffer->Files; *fd_ != fd; fd_ = &((*fd_)->Next));
		_assert(*fd_ == fd);
		*fd_ = fd->Next;
		fd->Next = nullptr;
		if (!buffer->Files)
		{
			TestvfsBuffer **pp;
			for (pp = &p->Buffer; *pp != buffer; pp = &((*pp)->Next));
			*pp = (*pp)->Next;
			for (int i = 0; buffer->Pages[i]; i++)
				_free((char *)buffer->Pages[i]);
			_free((char *)buffer);
		}
		fd->Shm = nullptr;
		return rc;
	}
	__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp)
	{
		TestvfsFd *fd = Fd;
		TestVfsVSystem *p = (TestVfsVSystem *)fd->Vfs;
		Jim_Interp *interp = p->interp;
		if (p->IsFullshm)
			return fd->Real->ShmMap(region, sizeRegion, isWrite, pp);
		RC rc = RC_OK;
		if (!fd->Shm)
		{
			rc = ShmOpen();
			if (rc != RC_OK)
				return rc;
		}
		if (p->Script && (p->Mask&TESTVFS_SHMMAP_MASK))
		{
			Jim_Obj *arg = Jim_NewListObj(interp, nullptr, 0);
			Jim_IncrRefCount(arg);
			Jim_ListAppendElement(interp, arg, Jim_NewIntObj(interp, region));
			Jim_ListAppendElement(interp, arg, Jim_NewIntObj(interp, sizeRegion));
			Jim_ListAppendElement(interp, arg, Jim_NewIntObj(interp, isWrite));
			p->ExecTcl("xShmMap", Jim_NewStringObj(interp, fd->Shm->FileName, -1), fd->ShmId, arg, nullptr);
			tvfsResultCode(p, &rc);
			Jim_DecrRefCount(interp, arg);
		}
		if (rc == RC_OK && (p->Mask&TESTVFS_SHMMAP_MASK) && p->InjectIoerr())
			rc = RC_IOERR;
		if (rc == RC_OK && isWrite && !fd->Shm->Pages[region])
			AllocPage(fd->Shm, region, sizeRegion);
		*pp = (void volatile *)fd->Shm->Pages[region];
		return rc;
	}
};

__device__ VFile *TestVfsVSystem::_AttachFile(void *buffer) { return new (buffer) TestVfsVFile(); }
__device__ RC TestVfsVSystem::Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags)
{
	TestVfsVFile *testfile = (TestVfsVFile *)file;
	TestvfsFd *fd = (TestvfsFd *)Jim_Alloc(sizeof(TestvfsFd) + SizeOsFile);
	memset(fd, 0, sizeof(TestvfsFd) + SizeOsFile);
	fd->Shm = nullptr;
	fd->ShmId = nullptr;
	fd->Filename = path;
	fd->Vfs = this;
	fd->Real = (VFile *)&fd[1];
	memset(testfile, 0, sizeof(TestVfsVFile));
	testfile->Fd = fd;

	// Evaluate the Tcl script: 
	//   SCRIPT xOpen FILENAME KEY-VALUE-ARGS
	//
	// If the script returns an SQLite error code other than SQLITE_OK, an error is returned to the caller. If it returns SQLITE_OK, the new
	// connection is named "anon". Otherwise, the value returned by the script is used as the connection name.
	RC rc;
	Jim_Obj *id = nullptr;
	Jim_ResetResult(interp);
	if (Script && (Mask&TESTVFS_OPEN_MASK))
	{
		Jim_Obj *arg = Jim_NewListObj(interp, nullptr, 0);
		Jim_IncrRefCount(arg);
		if (flags&VSystem::OPEN_MAIN_DB)
		{
			const char *z = &path[_strlen(path)+1];
			while (*z)
			{
				Jim_ListAppendElement(nullptr, arg, Jim_NewStringObj(interp, z, -1));
				z += _strlen(z) + 1;
				Jim_ListAppendElement(nullptr, arg, Jim_NewStringObj(interp, z, -1));
				z += _strlen(z) + 1;
			}
		}
		ExecTcl("xOpen", Jim_NewStringObj(interp, fd->Filename, -1), arg, nullptr, nullptr);
		Jim_DecrRefCount(interp, arg);
		if (tvfsResultCode(this, &rc))
		{
			if (rc != RC_OK) return rc;
		}
		else
			id = Jim_GetResult(interp);
	}

	if ((Mask&TESTVFS_OPEN_MASK) && InjectIoerr()) return RC_IOERR;
	if (InjectCantopenerr()) return RC_CANTOPEN;
	if (InjectFullerr()) return RC_FULL;

	if (!id)
		id = Jim_NewStringObj(interp, "anon", -1);
	Jim_IncrRefCount(id);
	fd->ShmId = id;
	Jim_ResetResult(interp);

	rc = Parent->Open(path, fd->Real, flags, outFlags);
	if (fd->Real->Opened)
	{
		file->Opened = true;
	}
	return rc;
}


__device__ static int testvfs_obj_cmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	TestVfsVSystem *p = (TestVfsVSystem *)clientData;
	enum DB_enum { 
		CMD_SHM, CMD_DELETE, CMD_FILTER, CMD_IOERR, CMD_SCRIPT, 
		CMD_DEVCHAR, CMD_SECTORSIZE, CMD_FULLERR, CMD_CANTOPENERR
	};
	struct TestvfsSubcmd {
		char *Name;
		enum DB_enum Cmd;
	} _subcmds[] = {
		{ "shm",         CMD_SHM         },
		{ "delete",      CMD_DELETE      },
		{ "filter",      CMD_FILTER      },
		{ "ioerr",       CMD_IOERR       },
		{ "fullerr",     CMD_FULLERR     },
		{ "cantopenerr", CMD_CANTOPENERR },
		{ "script",      CMD_SCRIPT      },
		{ "devchar",     CMD_DEVCHAR     },
		{ "sectorsize",  CMD_SECTORSIZE  },
		{ nullptr, (DB_enum)0 }
	};
	if (argc < 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "SUBCOMMAND ...");
		return JIM_ERROR;
	}
	int i;
	if (Jim_GetEnumFromStruct(interp, args[1], (const void **)_subcmds, sizeof(_subcmds[0]), &i, "subcommand", 0))
		return JIM_ERROR;
	Jim_ResetResult(interp);
	switch (_subcmds[i].Cmd) {
	case CMD_SHM: {
		if (argc != 3 && argc != 4)
		{
			Jim_WrongNumArgs(interp, 2, args, "FILE ?VALUE?");
			return JIM_ERROR;
		}
		char *name = (char *)_alloc(p->Parent->MaxPathname);
		RC rc = p->Parent->FullPathname(Jim_String(args[2]), p->Parent->MaxPathname, name);
		if (rc != RC_OK)
		{
			Jim_AppendResult(interp, "failed to get full path: ", Jim_String(args[2]), nullptr);
			_free(name);
			return JIM_ERROR;
		}
		TestvfsBuffer *buffer;
		for (buffer = p->Buffer; buffer; buffer = buffer->Next)
			if (!_strcmp(buffer->FileName, name))
				break;
		_free(name);
		if (!buffer)
		{
			Jim_AppendResult(interp, "no such file: ", Jim_String(args[2]), nullptr);
			return JIM_ERROR;
		}
		if (argc == 4)
		{
			int n;
			uint8 *a = (uint8 *)Jim_GetByteArray(args[3], &n);
			int pagesize = buffer->Pagesize;
			if (pagesize == 0) pagesize = 65536;
			for (i = 0; i*pagesize < n; i++)
			{
				int bytes = pagesize;
				AllocPage(buffer, i, pagesize);
				if (n-i*pagesize < pagesize)
					bytes = n;
				memcpy(buffer->Pages[i], &a[i*pagesize], bytes);
			}
		}
		Jim_Obj *obj = Jim_NewEmptyStringObj(interp);
		for (i = 0; buffer->Pages[i]; i++)
		{
			int pagesize = buffer->Pagesize;
			if (pagesize == 0) pagesize = 65536;
			Jim_AppendObj(interp, obj, Jim_NewByteArrayObj(interp, buffer->Pages[i], pagesize));
		}
		Jim_SetResult(interp, obj);
		break; }
	case CMD_FILTER: {
		static struct VfsMethod {
			char *Name;
			int Mask;
		} _vfsmethods [] = {
			{ "xShmOpen",      TESTVFS_SHMOPEN_MASK },
			{ "xShmLock",      TESTVFS_SHMLOCK_MASK },
			{ "xShmBarrier",   TESTVFS_SHMBARRIER_MASK },
			{ "xShmUnmap",     TESTVFS_SHMCLOSE_MASK },
			{ "xShmMap",       TESTVFS_SHMMAP_MASK },
			{ "xSync",         TESTVFS_SYNC_MASK },
			{ "xDelete",       TESTVFS_DELETE_MASK },
			{ "xWrite",        TESTVFS_WRITE_MASK },
			{ "xRead",         TESTVFS_READ_MASK },
			{ "xTruncate",     TESTVFS_TRUNCATE_MASK },
			{ "xOpen",         TESTVFS_OPEN_MASK },
			{ "xClose",        TESTVFS_CLOSE_MASK },
			{ "xAccess",       TESTVFS_ACCESS_MASK },
			{ "xFullPathname", TESTVFS_FULLPATHNAME_MASK },
		};
		int mask = 0;
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "LIST");
			return JIM_ERROR;
		}
		Jim_Obj **elems = nullptr;
		int elemLength = 0;
		if (Jim_ListGetElements(interp, args[2], &elemLength, &elems))
			return JIM_ERROR;
		Jim_ResetResult(interp);
		for (i = 0; i < elemLength; i++)
		{
			const char *elem = Jim_String(elems[i]);
			int method;
			for (method = 0; method < _lengthof(_vfsmethods); method++)
			{
				if (!_strcmp(elem, _vfsmethods[method].Name))
				{
					mask |= _vfsmethods[method].Mask;
					break;
				}
			}
			if (method == _lengthof(_vfsmethods))
			{
				Jim_AppendResult(interp, "unknown method: ", elem, nullptr);
				return JIM_ERROR;
			}
		}
		p->Mask = mask;
		break; }
	case CMD_SCRIPT: {
		if (argc == 3)
		{
			if (p->Script)
			{
				Jim_DecrRefCount(interp, p->Script);
				p->Script = nullptr;
			}
			int bytes;
			Jim_GetString(args[2], &bytes);
			if (bytes > 0)
			{
				p->Script = Jim_DuplicateObj(interp, args[2]);
				Jim_IncrRefCount(p->Script);
			}
		}
		else if (argc != 2)
		{
			Jim_WrongNumArgs(interp, 2, args, "?SCRIPT?");
			return JIM_ERROR;
		}
		Jim_ResetResult(interp);
		if (p->Script)
			Jim_SetResult(interp, p->Script);
		break; }
	case CMD_CANTOPENERR:
	case CMD_IOERR:
	case CMD_FULLERR: {
		// TESTVFS ioerr ?IFAIL PERSIST?
		//
		//   Where IFAIL is an integer and PERSIST is boolean.
		TestFaultInject *test;
		switch (_subcmds[i].Cmd) {
		case CMD_IOERR: test = &p->ioerr_err; break;
		case CMD_FULLERR: test = &p->full_err; break;
		case CMD_CANTOPENERR: test = &p->cantopen_err; break;
		default: _assert(false);
		}
		int r = test->Fails;
		test->Fails = 0;
		test->Fault = 0;
		test->Cnts = 0;
		if (argc == 4)
		{
			int cnts;
			bool persist;
			if (Jim_GetInt(interp, args[2], &cnts) != JIM_OK || Jim_GetBoolean(interp, args[3], &persist) != JIM_OK)
				return JIM_ERROR;
			test->Fault = (persist ? FAULT_INJECT_PERSISTENT : FAULT_INJECT_TRANSIENT);
			test->Cnts = cnts;
		}
		else if (argc != 2)
		{
			Jim_WrongNumArgs(interp, 2, args, "?CNT PERSIST?");
			return JIM_ERROR;
		}
		Jim_SetResultInt(interp, r);
		break; }
	case CMD_DELETE: {
		Jim_DeleteCommand(interp, Jim_String(args[0]));
		break; }
	case CMD_DEVCHAR: {
		struct DeviceFlag {
			char *Name;
			int Value;
		} _flags[] = {
			{ "default",               (VFile::IOCAP)-1 },
			{ "atomic",                VFile::IOCAP_ATOMIC                },
			{ "atomic512",             VFile::IOCAP_ATOMIC512             },
			{ "atomic1k",              VFile::IOCAP_ATOMIC1K              },
			{ "atomic2k",              VFile::IOCAP_ATOMIC2K              },
			{ "atomic4k",              VFile::IOCAP_ATOMIC4K              },
			{ "atomic8k",              VFile::IOCAP_ATOMIC8K              },
			{ "atomic16k",             VFile::IOCAP_ATOMIC16K             },
			{ "atomic32k",             VFile::IOCAP_ATOMIC32K             },
			{ "atomic64k",             VFile::IOCAP_ATOMIC64K             },
			{ "sequential",            VFile::IOCAP_SEQUENTIAL            },
			{ "safe_append",           VFile::IOCAP_SAFE_APPEND           },
			{ "undeletable_when_open", VFile::IOCAP_UNDELETABLE_WHEN_OPEN },
			{ "powersafe_overwrite",   VFile::IOCAP_POWERSAFE_OVERWRITE   },
			{ nullptr, (VFile::IOCAP)0 }
		};
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?ATTR-LIST?");
			return JIM_ERROR;
		}
		if (argc == 3)
		{
			Jim_Obj **flags = nullptr;
			int flagsLength = 0;
			if (Jim_ListGetElements(interp, args[2], &flagsLength, &flags))
				return JIM_ERROR;
			int new_ = 0;
			for (int j = 0; j < flagsLength; j++)
			{
				i = 0;
				if (Jim_GetEnumFromStruct(interp, flags[j], (const void **)_flags, sizeof(_flags[0]), &i, "flag", 0))
					return JIM_ERROR;
				if (_flags[i].Value < 0 && flagsLength > 1)
				{
					Jim_AppendResult(interp, "bad flags: ", Jim_String(args[2]), nullptr);
					return JIM_ERROR;
				}
				new_ |= _flags[i].Value;
			}
			p->Devchar = (VFile::IOCAP)(new_ | 0x10000000);
		}
		Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
		for (i = 0; i < _lengthof(_flags); i++)
			if (p->Devchar & _flags[i].Value)
				Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, _flags[i].Name, -1));
		Jim_SetResult(interp, r);
		break; }
	case CMD_SECTORSIZE: {
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?VALUE?");
			return JIM_ERROR;
		}
		if (argc == 3)
		{
			int new_ = 0;
			if (Jim_GetInt(interp, args[2], &new_))
				return JIM_ERROR;
			p->Sectorsize = new_;
		}
		Jim_SetResultInt(interp, p->Sectorsize);
		break; }
	}
	return JIM_OK;
}

__device__ static void testvfs_obj_del(ClientData data, Jim_Interp *interp)
{
	TestVfsVSystem *p = (TestVfsVSystem *)data;
	if (p->Script) Jim_DecrRefCount(interp, p->Script);
	VSystem::UnregisterVfs(p->Vfs);
	_free((char *)p->Vfs);
	_free((char *)p);
}

// Usage:  testvfs VFSNAME ?SWITCHES?
//
// Switches are:
//   -noshm   BOOLEAN             (True to omit shm methods. Default false)
//   -default BOOLEAN             (True to make the vfs default. Default false)
//
// This command creates two things when it is invoked: an SQLite VFS, and a Tcl command. Both are named VFSNAME. The VFS is installed. It is not
// installed as the default VFS.
//
// The VFS passes all file I/O calls through to the underlying VFS.
//
// Whenever the xShmMap method of the VFS is invoked, the SCRIPT is executed as follows:
//   SCRIPT xShmMap    FILENAME ID
//
// The value returned by the invocation of SCRIPT above is interpreted as an SQLite error code and returned to SQLite. Either a symbolic 
// "SQLITE_OK" or numeric "0" value may be returned.
//
// The contents of the shared-memory buffer associated with a given file may be read and set using the following command:
//   VFSNAME shm FILENAME ?NEWVALUE?
//
// When the xShmLock method is invoked by SQLite, the following script is run:
//   SCRIPT xShmLock    FILENAME ID LOCK
//
// where LOCK is of the form "OFFSET NBYTE lock/unlock shared/exclusive"
__device__ static int testvfs_cmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	bool isNoshm = false;           // True if -noshm is passed
	bool isFullshm = false;         // True if -fullshm is passed
	bool isDefault = false;         // True if -default is passed
	int sizeOsFile = 0;             // Value passed to -szosfile
	int maxPathname = -1;           // Value passed to -mxpathname
	int iversion = 2;               // Value passed to -iversion
	if (argc < 2 || (argc%2) != 0)
		goto bad_args;
	for (int i = 2; i < argc; i += 2)
	{
		int switchLength;
		const char *switch_ = Jim_GetString(args[i], &switchLength); 
		if (switchLength > 2 && !_strncmp("-noshm", switch_, switchLength))
		{
			if (Jim_GetBoolean(interp, args[i+1], &isNoshm))
				return JIM_ERROR;
			if (isNoshm) isFullshm = false;
		}
		else if (switchLength > 2 && !_strncmp("-default", switch_, switchLength))
		{
			if (Jim_GetBoolean(interp, args[i+1], &isDefault))
				return JIM_ERROR;
		}
		else if (switchLength > 2 && !_strncmp("-szosfile", switch_, switchLength))
		{
			if (Jim_GetInt(interp, args[i+1], &sizeOsFile))
				return JIM_ERROR;
		}
		else if (switchLength > 2 && !_strncmp("-mxpathname", switch_, switchLength))
		{
			if (Jim_GetInt(interp, args[i+1], &maxPathname))
				return JIM_ERROR;
		}
		else if (switchLength > 2 && !_strncmp("-iversion", switch_, switchLength))
		{
			if (Jim_GetInt(interp, args[i+1], &iversion))
				return JIM_ERROR;
		}
		else if (switchLength > 2 && !_strncmp("-fullshm", switch_, switchLength))
		{
			if (Jim_GetBoolean(interp, args[i+1], &isFullshm))
				return JIM_ERROR;
			if (isFullshm) isNoshm = false;
		}
		else
			goto bad_args;
	}
	if (sizeOsFile < sizeof(TestVfsVFile))
		sizeOsFile = sizeof(TestVfsVFile);
	const char *vfsName = Jim_String(args[1]);
	int bytes = sizeof(TestVfsVSystem) + (int)_strlen(vfsName)+1;
	TestVfsVSystem *p = (TestVfsVSystem *)_alloc(bytes);
	memset(p, 0, bytes);
	p->Devchar = (VFile::IOCAP)-1;
	p->Sectorsize = -1;

	// Create the new object command before querying SQLite for a default VFS to use for 'real' IO operations. This is because creating the new VFS
	// may delete an existing [testvfs] VFS of the same name. If such a VFS is currently the default, the new [testvfs] may end up calling the methods of a deleted object.
	Jim_CreateCommand(interp, vfsName, testvfs_obj_cmd, p, testvfs_obj_del);
	p->Parent = VSystem::FindVfs(nullptr);
	p->interp = interp;
	p->Name = (char *)&p[1];
	memcpy(p->Name, vfsName, _strlen(vfsName)+1);
	p->MaxPathname = p->Parent->MaxPathname;
	if (maxPathname >= 0 && maxPathname < p->MaxPathname)
		p->MaxPathname = maxPathname;
	p->SizeOsFile = sizeOsFile;
	p->IsNoshm = isNoshm;
	p->IsFullshm = isFullshm;
	p->Mask = TESTVFS_ALL_MASK;
	VSystem::RegisterVfs(p, isDefault);
	return JIM_OK;
bad_args:
	Jim_WrongNumArgs(interp, 1, args, "VFSNAME ?-noshm BOOL? ?-default BOOL? ?-mxpathname INT? ?-szosfile INT? ?-iversion INT?");
	return JIM_ERROR;
}

int Sqlitetestvfs_Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "testvfs", testvfs_cmd, nullptr, nullptr);
	return JIM_OK;
}

#endif
