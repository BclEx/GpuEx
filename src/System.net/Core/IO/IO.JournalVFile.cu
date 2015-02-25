// journal.c
#include "../Core.cu.h"
#include <new.h>
#ifdef ENABLE_ATOMIC_WRITE

namespace Core { namespace IO
{
	class JournalVFile : public VFile
	{
	public:
		int BufferLength;               // Size of zBuf[] in bytes
		char *Buffer;					// Space to buffer journal writes
		int Size;						// Amount of zBuf[] currently used
		VSystem::OPEN Flags;        // xOpen flags
		VSystem *Vfs;				// The "real" underlying VFS
		VFile *Real;					// The "real" underlying file descriptor
		const char *Journal;			// Name of the journal file
		__device__ RC CreateFile();
	public:
		//bool Opened;
		__device__ virtual RC Close_();
		__device__ virtual RC Read(void *buffer, int amount, int64 offset);
		__device__ virtual RC Write(const void *buffer, int amount, int64 offset);
		__device__ virtual RC Truncate(int64 size);
		__device__ virtual RC Sync(SYNC flags);
		__device__ virtual RC get_FileSize(int64 &size);
	};

	__device__ RC JournalVFile::CreateFile()
	{
		RC rc = RC_OK;
		if (!Real)
		{
			VFile *real = (VFile *)&this[1];
			rc = Vfs->Open(Journal, Real, Flags, 0);
			if (rc == RC_OK)
			{
				Real = real;
				if (Size > 0)
				{
					_assert(Size <= BufferLength);
					rc = Real->Write(Buffer, Size, 0);
				}
				if (rc != RC_OK)
				{
					// If an error occurred while writing to the file, close it before returning. This way, SQLite uses the in-memory journal data to 
					// roll back changes made to the internal page-cache before this function was called.
					Real->Close_();
					Real = nullptr;
				}
			}
		}
		return rc;
	}

	__device__ RC JournalVFile::Close_()
	{
		if (Real)
			Real->Close_();
		_free(Buffer);
		return RC_OK;
	}

	__device__ RC JournalVFile::Read(void *buffer, int amount, int64 offset)
	{
		if (Real)
			return Real->Read(buffer, amount, offset);
		if ((amount + offset) > Size)
			return RC_IOERR_SHORT_READ;
		_memcpy((char *)buffer, &Buffer[offset], amount);
		return RC_OK;
	}

	__device__ RC JournalVFile::Write(const void *buffer, int amount, int64 offset)
	{
		RC rc = RC_OK;
		if (!Real && (offset + amount) > BufferLength)
			rc = CreateFile();
		if (rc == RC_OK)
		{
			if (Real)
				return Real->Write(buffer, amount, offset);
			_memcpy(&Buffer[offset], (char *)buffer, amount);
			if (Size < (offset + amount))
				Size = (int)(offset + amount);
		}
		return rc;
	}

	__device__ RC JournalVFile::Truncate(int64 size)
	{
		if (Real)
			return Real->Truncate(size);
		if (size < Size)
			Size = (int)size;
		return RC_OK;
	}

	__device__ RC JournalVFile::Sync(SYNC flags)
	{
		if (Real)
			return Real->Sync(flags);
		return RC_OK;
	}

	__device__ RC JournalVFile::get_FileSize(int64 &size)
	{
		if (Real)
			return Real->get_FileSize(size);
		size = (int64)Size;
		return RC_OK;
	}

	// extensions
	__device__ RC VFile::JournalVFileOpen(VSystem *vfs, const char *name, VFile *file, VSystem::OPEN flags, int bufferLength)
	{
		_memset(file, 0, JournalVFileSize(vfs));
		file = new (file) JournalVFile();
		JournalVFile *p = (JournalVFile *)file;
		if (bufferLength > 0)
		{
			p->Buffer = (char *)_alloc2(bufferLength, true);
			if (!p->Buffer)
				return RC_NOMEM;
		}
		else
			return vfs->Open(name, file, flags, 0);
		p->Type = 2;
		p->BufferLength = bufferLength;
		p->Flags = flags;
		p->Journal = name;
		p->Vfs = vfs;
		return RC_OK;
	}

	__device__ RC VFile::JournalVFileCreate(VFile *file)
	{
		if (file->Type != 2)
			return RC_OK;
		return ((JournalVFile *)file)->CreateFile();
	}

	__device__ bool VFile::HasJournalVFile(VFile *file)
	{
		return (file->Type != 2 || ((JournalVFile *)file)->Real != nullptr);
	}

	__device__ int VFile::JournalVFileSize(VSystem *vfs)
	{
		return (vfs->SizeOsFile + sizeof(JournalVFile));
	}
}}

#endif