// sqlite3.h
namespace Core { typedef class VSystem VSystem; }
namespace Core { namespace IO
{
#define PENDING_BYTE 0x40000000
#define RESERVED_BYTE (PENDING_BYTE+1)
#define SHARED_FIRST (PENDING_BYTE+2)
#define SHARED_SIZE 510

	// sqliteInt.h
	typedef class VFile VFile;

	class VFile
	{
	public:
		enum LOCK : char
		{
			LOCK_NO = 0,
			LOCK_SHARED = 1,
			LOCK_RESERVED = 2,
			LOCK_PENDING = 3,
			LOCK_EXCLUSIVE = 4,
			LOCK_UNKNOWN = 5,
		};

		// sqlite3.h
		enum SYNC : char
		{
			SYNC_NORMAL = 0x00002,
			SYNC_FULL = 0x00003,
			SYNC_DATAONLY = 0x00010,
			// wal.h
			SYNC_WAL_TRANSACTIONS = 0x20,    // Sync at the end of each transaction
			SYNC_WAL_MASK = 0x13,            // Mask off the SQLITE_SYNC_* values
		};

		// sqlite3.h
		enum FCNTL : uint
		{
			FCNTL_LOCKSTATE = 1,
			FCNTL_GET_LOCKPROXYFILE = 2,
			FCNTL_SET_LOCKPROXYFILE = 3,
			FCNTL_LAST_ERRNO = 4,
			FCNTL_SIZE_HINT = 5,
			FCNTL_CHUNK_SIZE = 6,
			FCNTL_FILE_POINTER = 7,
			FCNTL_SYNC_OMITTED = 8,
			FCNTL_WIN32_AV_RETRY = 9,
			FCNTL_PERSIST_WAL = 10,
			FCNTL_OVERWRITE = 11,
			FCNTL_VFSNAME = 12,
			FCNTL_POWERSAFE_OVERWRITE = 13,
			FCNTL_PRAGMA = 14,
			FCNTL_BUSYHANDLER = 15,
			FCNTL_TEMPFILENAME = 16,
			FCNTL_MMAP_SIZE = 18,
			// os.h
			FCNTL_DB_UNCHANGED = 0xca093fa0,
		};

		// sqlite3.h
		enum IOCAP : uint
		{
			IOCAP_ATOMIC = 0x00000001,
			IOCAP_ATOMIC512 = 0x00000002,
			IOCAP_ATOMIC1K = 0x00000004,
			IOCAP_ATOMIC2K = 0x00000008,
			IOCAP_ATOMIC4K = 0x00000010,
			IOCAP_ATOMIC8K = 0x00000020,
			IOCAP_ATOMIC16K = 0x00000040,
			IOCAP_ATOMIC32K = 0x00000080,
			IOCAP_ATOMIC64K = 0x00000100,
			IOCAP_SAFE_APPEND = 0x00000200,
			IOCAP_SEQUENTIAL = 0x00000400,
			IOCAP_UNDELETABLE_WHEN_OPEN = 0x00000800,
			IOCAP_POWERSAFE_OVERWRITE = 0x00001000,
		};

		// sqlite3.h
		enum SHM : char
		{
			SHM_UNLOCK = 1,
			SHM_LOCK = 2,
			SHM_SHARED = 4,
			SHM_EXCLUSIVE = 8,
			SHM_MAX = 8,
		};

		uint8 Type;
		bool Opened;

		__device__ virtual RC Read(void *buffer, int amount, int64 offset) = 0;
		__device__ virtual RC Write(const void *buffer, int amount, int64 offset) = 0;
		__device__ virtual RC Truncate(int64 size) = 0;
		__device__ inline RC Close() { if (!Opened) return RC_OK; Opened = false; return Close_(); }
		__device__ virtual RC Close_() = 0;
		__device__ virtual RC Sync(SYNC flags) = 0;
		__device__ virtual RC get_FileSize(int64 &size) = 0;

		__device__ virtual RC Lock(LOCK lock);
		__device__ virtual RC Unlock(LOCK lock);
		__device__ virtual RC CheckReservedLock(int &lock);
		__device__ virtual RC FileControl(FCNTL op, void *arg);

		__device__ virtual uint get_SectorSize();
		__device__ virtual IOCAP get_DeviceCharacteristics();

		__device__ virtual RC ShmLock(int offset, int n, SHM flags);
		__device__ virtual void ShmBarrier();
		__device__ virtual RC ShmUnmap(bool deleteFlag);
		__device__ virtual RC ShmMap(int region, int sizeRegion, bool isWrite, void volatile **pp);

		__device__ inline RC Read4(int64 offset, uint32 *valueOut)
		{
			unsigned char ac[4];
			RC rc = Read(ac, sizeof(ac), offset);
			if (rc == RC_OK)
				*valueOut = ConvertEx::Get4(ac);
			return rc;
		}

		__device__ inline RC Write4(int64 offset, uint32 value)
		{
			char ac[4];
			ConvertEx::Put4((uint8 *)ac, value);
			return Write(ac, 4, offset);
		}

		__device__ inline RC CloseAndFree()
		{
			RC rc = Close();
			_free(this);
			return rc;
		}

		// extensions
#ifdef ENABLE_ATOMIC_WRITE
		__device__ static RC JournalVFileOpen(VSystem *vfs, const char *name, VFile *file, VSystem::OPEN flags, int bufferLength);
		__device__ static int JournalVFileSize(VSystem *vfs);
		__device__ static RC JournalVFileCreate(VFile *file);
		__device__ static bool HasJournalVFile(VFile *file);
#else
		__device__ inline static int JournalVFileSize(VSystem *vfs) { return vfs->SizeOsFile; }
		__device__ inline bool HasJournalVFile(VFile *file) { return true; }
		//#define JournalSize(vfs) ((vfs)->SizeOsFile)
		//#define HasJournal(file) true
#endif
		__device__ static void MemoryVFileOpen(VFile *file);
		__device__ static bool HasMemoryVFile(VFile *file);
		__device__ static int MemoryVFileSize() ;
	};

	__device__ inline void operator|=(VFile::SYNC &a, int b) { a = (VFile::SYNC)(a | b); }
	__device__ inline VFile::SYNC operator|(VFile::SYNC a, VFile::SYNC b) { return (VFile::SYNC)((int)a | (int)b); }
}}