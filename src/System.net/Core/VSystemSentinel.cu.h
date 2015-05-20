#include <RuntimeSentinel.h>
#include "Core.cu.h"

namespace Core
{
	struct VSystemSentinel
	{
	public:
		static void Initialize();
		static void Shutdown();
	};
}

namespace Core { namespace Messages
{
#pragma region File

	struct File_Close
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_Close(VFile *f)
			: Base(10, nullptr), F(f) { }
		RC RC;
	};

	struct File_Read
	{
		__device__ inline static void Prepare(File_Read *t, char *data, int length)
		{
			t->Buffer = (char *)(data += _ROUND8(sizeof(*t)));
		}
		RuntimeSentinelMessage Base;
		VFile *F; int Amount; int64 Offset;
		__device__ File_Read(VFile *f, int amount, int64 offset)
			: Base(11, RUNTIMESENTINELPREPARE(Prepare)), F(f), Amount(amount), Offset(offset) { }
		RC RC;
		char *Buffer;
	};

	struct File_Write
	{
		__device__ inline static void Prepare(File_Write *t, char *data, int length)
		{
			char *buffer = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(buffer, t->Buffer, t->Amount);
			t->Buffer = buffer;
		}
		RuntimeSentinelMessage Base;
		VFile *F; const void *Buffer; int Amount; int64 Offset;
		__device__ File_Write(VFile *f, const void *buffer, int amount, int64 offset)
			: Base(12, RUNTIMESENTINELPREPARE(Prepare)), F(f), Buffer(buffer), Amount(amount), Offset(offset) { }
		RC RC;
	};

	struct File_Truncate
	{
		RuntimeSentinelMessage Base;
		VFile *F; int64 Size;
		__device__ File_Truncate(VFile *f, int64 size)
			: Base(13, nullptr), F(f), Size(size) { }
		RC RC;
	};

	struct File_Sync
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::SYNC Flags;
		__device__ File_Sync(VFile *f, VFile::SYNC flags)
			: Base(14, nullptr), F(f), Flags(flags) { }
		RC RC;
	};

	struct File_get_FileSize
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_get_FileSize(VFile *f)
			: Base(15, nullptr), F(f) { }
		int64 Size;
		RC RC;
	};

	struct File_Lock
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::LOCK Lock;
		__device__ File_Lock(VFile *f, VFile::LOCK lock)
			: Base(16, nullptr), F(f), Lock(lock) { }
		RC RC;
	};

	struct File_CheckReservedLock
	{
		RuntimeSentinelMessage Base;
		VFile *F;
		__device__ File_CheckReservedLock(VFile *f)
			: Base(17, nullptr), F(f) { }
		int Lock;
		RC RC;
	};

	struct File_Unlock
	{
		RuntimeSentinelMessage Base;
		VFile *F; VFile::LOCK Lock;
		__device__ File_Unlock(VFile *f, VFile::LOCK lock)
			: Base(18, nullptr), F(f), Lock(lock) { }
		RC RC;
	};

#pragma endregion

#pragma region System

	struct System_Open
	{
		__device__ inline static void Prepare(System_Open *t, char *data, int length)
		{
			int nameLength = (t->Name ? _strlen(t->Name) + 1 : 0);
			char *name = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(name, t->Name, nameLength);
			t->Name = name;
		}
		RuntimeSentinelMessage Base;
		const char *Name; VSystem::OPEN Flags;
		__device__ System_Open(const char *name, VSystem::OPEN flags)
			: Base(21, RUNTIMESENTINELPREPARE(Prepare)), Name(name), Flags(flags) { }
		VFile *F;
		VSystem::OPEN OutFlags;
		RC RC;
	};

	struct System_Delete
	{
		__device__ inline static void Prepare(System_Delete *t, char *data, int length)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(filename, t->Filename, filenameLength);
			t->Filename = filename;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; bool SyncDir;
		__device__ System_Delete(const char *filename, bool syncDir)
			: Base(22, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), SyncDir(syncDir) { }
		RC RC;
	};

	struct System_Access
	{
		__device__ inline static void Prepare(System_Access *t, char *data, int length)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(filename, t->Filename, filenameLength);
			t->Filename = filename;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; VSystem::ACCESS Flags;
		__device__ System_Access(const char *filename, VSystem::ACCESS flags)
			: Base(23, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), Flags(flags) { }
		int ResOut;
		RC RC;
	};

	struct System_FullPathname
	{
		__device__ inline static void Prepare(System_FullPathname *t, char *data, int length)
		{
			int relativeLength = (t->Relative ? _strlen(t->Relative) + 1 : 0);
			char *relative = (char *)(data += _ROUND8(sizeof(*t)));
			char *full = (char *)(data += relativeLength);
			_memcpy(relative, t->Relative, relativeLength);
			t->Relative = relative;
			t->Full = full;
		}
		RuntimeSentinelMessage Base;
		const char *Relative; int FullLength;
		__device__ System_FullPathname(const char *relative, int fullLength)
			: Base(24, RUNTIMESENTINELPREPARE(Prepare)), Relative(relative), FullLength(fullLength) { }
		char *Full;
		RC RC;
	};

	struct System_GetLastError
	{
		__device__ inline static void Prepare(System_GetLastError *t, char *data, int length)
		{
			t->Buf = (char *)(data += _ROUND8(sizeof(*t)));
		}
		RuntimeSentinelMessage Base;
		int BufLength;
		__device__ System_GetLastError(int bufLength)
			: Base(25, RUNTIMESENTINELPREPARE(Prepare)), BufLength(bufLength) { }
		char *Buf;
		RC RC;
	};

#pragma endregion
} }