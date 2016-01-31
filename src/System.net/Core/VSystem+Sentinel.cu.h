#if OS_MAP

namespace CoreS
{
	struct VSystemSentinel
	{
	public:
		static void Initialize();
		static void Shutdown();
	};
}

namespace CORE_NAME { namespace Messages
{
#pragma region File

	struct File_Close
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_Close(VFile *f)
			: Base(false, 10, 0, nullptr), F(f) { RuntimeSentinel::Send(this, sizeof(File_Close)); }
		RC RC;
	};

	struct File_Read
	{
		__device__ inline static char *Prepare(File_Read *t, char *data, char *dataEnd)
		{
			t->Buffer  = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			return end;
		}
		RuntimeSentinelMessage Base;
		VFile *F; int Amount; int64 Offset;
		__device__ File_Read(VFile *f, int amount, int64 offset)
			: Base(false, 11, 1024, RUNTIMESENTINELPREPARE(Prepare)), F(f), Amount(amount), Offset(offset) { RuntimeSentinel::Send(this, sizeof(File_Read)); }
		RC RC;
		char *Buffer;
	};

	struct File_Write
	{
		__device__ inline static char *Prepare(File_Write *t, char *data, char *dataEnd)
		{
			char *buffer = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			memcpy(buffer, t->Buffer, t->Amount);
			t->Buffer = buffer;
			return end;
		}
		RuntimeSentinelMessage Base;
		VFile *F; const void *Buffer; int Amount; int64 Offset;
		__device__ File_Write(VFile *f, const void *buffer, int amount, int64 offset)
			: Base(false, 12, 1024, RUNTIMESENTINELPREPARE(Prepare)), F(f), Buffer(buffer), Amount(amount), Offset(offset) { RuntimeSentinel::Send(this, sizeof(File_Write)); }
		RC RC;
	};

	struct File_Truncate
	{
		RuntimeSentinelMessage Base;
		VFile *F; int64 Size;
		__device__ File_Truncate(VFile *f, int64 size)
			: Base(false, 13, 0, nullptr), F(f), Size(size) { RuntimeSentinel::Send(this, sizeof(File_Truncate)); }
		RC RC;
	};

	struct File_Sync
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::SYNC Flags;
		__device__ File_Sync(VFile *f, VFile::SYNC flags)
			: Base(false, 14, 0, nullptr), F(f), Flags(flags) { RuntimeSentinel::Send(this, sizeof(File_Sync)); }
		RC RC;
	};

	struct File_get_FileSize
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_get_FileSize(VFile *f)
			: Base(false, 15, 0, nullptr), F(f) { RuntimeSentinel::Send(this, sizeof(File_get_FileSize)); }
		int64 Size;
		RC RC;
	};

	struct File_Lock
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::LOCK Lock;
		__device__ File_Lock(VFile *f, VFile::LOCK lock)
			: Base(false, 16, 0, nullptr), F(f), Lock(lock) { RuntimeSentinel::Send(this, sizeof(File_Lock)); }
		RC RC;
	};

	struct File_CheckReservedLock
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_CheckReservedLock(VFile *f)
			: Base(false, 17, 0, nullptr), F(f) { RuntimeSentinel::Send(this, sizeof(File_CheckReservedLock)); }
		int Lock;
		RC RC;
	};

	struct File_Unlock
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::LOCK Lock;
		__device__ File_Unlock(VFile *f, VFile::LOCK lock)
			: Base(false, 18, 0, nullptr), F(f), Lock(lock) { RuntimeSentinel::Send(this, sizeof(File_Unlock)); }
		RC RC;
	};

#pragma endregion

#pragma region System

	struct System_Open
	{
		__device__ inline static char *Prepare(System_Open *t, char *data, char *dataEnd)
		{
			// filenames are double-zero terminated if they are not URIs with parameters.
			int nameLength;
			if (t->Name) { nameLength = _strlen(t->Name) + 1; nameLength += _strlen((const char *)t->Name[nameLength]) + 1; }
			else nameLength = 0;
			char *name = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += nameLength);
			if (end > dataEnd) return nullptr;
			memcpy(name, t->Name, nameLength);
			t->Name = name;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Name; VSystem::OPEN Flags;
		__device__ System_Open(const char *name, VSystem::OPEN flags)
			: Base(false, 21, 1024, RUNTIMESENTINELPREPARE(Prepare)), Name(name), Flags(flags) { RuntimeSentinel::Send(this, sizeof(System_Open)); }
		VFile *F;
		VSystem::OPEN OutFlags;
		RC RC;
	};

	struct System_Delete
	{
		__device__ inline static char *Prepare(System_Delete *t, char *data, char *dataEnd)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += filenameLength);
			if (end > dataEnd) return nullptr;
			memcpy(filename, t->Filename, filenameLength);
			t->Filename = filename;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; bool SyncDir;
		__device__ System_Delete(const char *filename, bool syncDir)
			: Base(false, 22, 1024, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), SyncDir(syncDir) { RuntimeSentinel::Send(this, sizeof(System_Delete)); }
		RC RC;
	};

	struct System_Access
	{
		__device__ inline static char *Prepare(System_Access *t, char *data, char *dataEnd)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += filenameLength);
			if (end > dataEnd) return nullptr;
			memcpy(filename, t->Filename, filenameLength);
			t->Filename = filename;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; VSystem::ACCESS Flags;
		__device__ System_Access(const char *filename, VSystem::ACCESS flags)
			: Base(false, 23, 1024, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), Flags(flags) { RuntimeSentinel::Send(this, sizeof(System_Access)); }
		int ResOut;
		RC RC;
	};

	struct System_FullPathname
	{
		__device__ inline static char *Prepare(System_FullPathname *t, char *data, char *dataEnd)
		{
			int relativeLength = (t->Relative ? _strlen(t->Relative) + 1 : 0);
			char *relative = (char *)(data += _ROUND8(sizeof(*t)));
			char *full = (char *)(data += relativeLength);
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			memcpy(relative, t->Relative, relativeLength);
			t->Relative = relative;
			t->Full = full;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Relative; int FullLength;
		__device__ System_FullPathname(const char *relative, int fullLength)
			: Base(false, 24, 2024, RUNTIMESENTINELPREPARE(Prepare)), Relative(relative), FullLength(fullLength) { RuntimeSentinel::Send(this, sizeof(System_FullPathname)); }
		char *Full;
		RC RC;
	};

	struct System_GetLastError
	{
		__device__ inline static char *Prepare(System_GetLastError *t, char *data, char *dataEnd)
		{
			t->Buf = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			return end;
		}
		RuntimeSentinelMessage Base;
		int BufLength;
		__device__ System_GetLastError(int bufLength)
			: Base(false, 25, 2024, RUNTIMESENTINELPREPARE(Prepare)), BufLength(bufLength) { RuntimeSentinel::Send(this, sizeof(System_GetLastError)); }
		char *Buf;
		RC RC;
	};

#pragma endregion
} }

#endif
;