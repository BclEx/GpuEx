#include "Core.cu.h"
namespace Core
{
	typedef struct
	{
		int Length;
		char Data[50];
	} SentinelCommand;

	typedef struct
	{
		volatile long Id;
		SentinelCommand Commands[1];
	} SentinelMap;

	namespace Messages
	{

#pragma region File

		struct File_Close
		{
			char OP;
			VFile *F;
			File_Close(VFile *f)
				: OP(10), F(f) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_Read
		{
			char OP;
			VFile *F;
			int Amount;
			int64 Offset;
			File_Read(VFile *f, int amount, int64 offset)
				: OP(11), F(f), Amount(amount), Offset(offset) { }
			RC RC;
			char *Buffer;
			inline void Prepare(char *data, int length)
			{
				Buffer = (char *)(data += _ROUND8(sizeof(File_Read)));
			}
		};

		struct File_Write
		{
			char OP;
			VFile *F;
			const void *Buffer;
			int Amount;
			int64 Offset;
			File_Write(VFile *f, const void *buffer, int amount, int64 offset)
				: OP(12), F(f), Buffer(buffer), Amount(amount), Offset(offset) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
				char *buffer = (char *)(data += _ROUND8(sizeof(File_Write)));
				_memcpy(buffer, Buffer, Amount);
				Buffer = buffer;
			}
		};

		struct File_Truncate
		{
			char OP;
			VFile *F;
			int64 Size;
			File_Truncate(VFile *f, int64 size)
				: OP(13), F(f), Size(size) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_Sync
		{
			char OP;
			VFile *F;
			VFile::SYNC Flags;
			File_Sync(VFile *f, VFile::SYNC flags)
				: OP(14), F(f), Flags(flags) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_get_FileSize
		{
			char OP;
			VFile *F;
			File_get_FileSize(VFile *f)
				: OP(15), F(f) { }
			int64 Size;
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_Lock
		{
			char OP;
			VFile *F;
			VFile::LOCK Lock;
			File_Lock(VFile *f, VFile::LOCK lock)
				: OP(16), F(f), Lock(lock) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_CheckReservedLock
		{
			char OP;
			VFile *F;
			File_CheckReservedLock(VFile *f)
				: OP(17), F(f) { }
			int Lock;
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

		struct File_Unlock
		{
			char OP;
			VFile *F;
			VFile::LOCK Lock;
			File_Unlock(VFile *f, VFile::LOCK lock)
				: OP(18), F(f), Lock(lock) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
			}
		};

#pragma endregion

#pragma region System

		struct System_Open
		{
			char OP;
			const char *Name;
			VSystem::OPEN Flags;
			System_Open(const char *name, VSystem::OPEN flags)
				: OP(1), Name(name), Flags(flags) { }
			VFile *F;
			VSystem::OPEN OutFlags;
			RC RC;
			inline void Prepare(char *data, int length)
			{
				int nameLength = (Name ? _strlen(Name) + 1 : 0);
				char *name = (char *)(data += _ROUND8(sizeof(System_Open)));
				_memcpy(name, Name, nameLength);
				Name = name;
			}
		};

		struct System_Delete
		{
			char OP;
			const char *Filename;
			bool SyncDir;
			System_Delete(const char *filename, bool syncDir)
				: OP(2), Filename(filename), SyncDir(syncDir) { }
			RC RC;
			inline void Prepare(char *data, int length)
			{
				int filenameLength = (Filename ? _strlen(Filename) + 1 : 0);
				char *filename = (char *)(data += _ROUND8(sizeof(System_Delete)));
				_memcpy(filename, Filename, filenameLength);
				Filename = filename;
			}
		};

		struct System_Access
		{
			char OP;
			const char *Filename;
			VSystem::ACCESS Flags;
			System_Access(const char *filename, VSystem::ACCESS flags)
				: OP(3), Filename(filename), Flags(flags) { }
			int ResOut;
			RC RC;
			inline void Prepare(char *data, int length)
			{
				int filenameLength = (Filename ? _strlen(Filename) + 1 : 0);
				char *filename = (char *)(data += _ROUND8(sizeof(System_Access)));
				_memcpy(filename, Filename, filenameLength);
				Filename = filename;
			}
		};

		struct System_FullPathname
		{
			char OP;
			const char *Relative;
			int FullLength;
			System_FullPathname(const char *relative, int fullLength)
				: OP(4), Relative(relative), FullLength(fullLength) { }
			char *Full;
			RC RC;
			inline void Prepare(char *data, int length)
			{
				int relativeLength = (Relative ? _strlen(Relative) + 1 : 0);
				char *relative = (char *)(data += _ROUND8(sizeof(System_Delete)));
				char *full = (char *)(data += relativeLength);
				_memcpy(relative, Relative, relativeLength);
				Relative = relative;
				Full = full;
			}
		};

		struct System_GetLastError
		{
			char OP;
			int BufLength;
			System_GetLastError(int bufLength)
				: OP(5), BufLength(bufLength) { }
			char *Buf;
			RC RC;
			inline void Prepare(char *data, int length)
			{
				Buf = (char *)(data += _ROUND8(sizeof(System_GetLastError)));
			}
		};

#pragma endregion
	}

	struct Sentinel
	{
	public:
		static void Initialize();
		static void Shutdown();
	};

	void Sentinel_Send(void *data, int length);
}
