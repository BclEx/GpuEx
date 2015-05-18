#include <windows.h>
#include <process.h>
#include <stdio.h>
#include "Core.cu.h"
#include "Sentinel.cu.h"

namespace Core
{
	unsigned int __stdcall SentinelThread(void *data) 
	{
		VSystem *vfs = (VSystem *)data;
		while (true)
		{
			//printf("Thread inside %d \n", GetCurrentThreadId());
			_sleep(1000);
		}
		return 0;
	}

	HANDLE _thread;
	SentinelMap _map;
	VSystem *_vfs;

	extern RC MapVSystem_Initialize();
	void Sentinel::Initialize()
	{
		memset(&_map, 0, sizeof(_map));
		MutexEx masterMutex;
		SysEx::Initialize(masterMutex);
		VSystem *vfs = _vfs = VSystem::FindVfs(nullptr);
		VSystem::UnregisterVfs(vfs);
		MapVSystem_Initialize();
		_thread = (HANDLE)_beginthreadex(0, 0, SentinelThread, vfs, 0, 0);
	}

	void Sentinel::Shutdown()
	{
		SysEx::Shutdown();
		CloseHandle(_thread);
	}

	void OneCommand(VSystem *vfs, char *data, int length)
	{
		switch (data[0])
		{
#pragma region File
		case 10: {
			Messages::File_Close *msg = (Messages::File_Close *)data;
			msg->RC = msg->F->Close_();
			break; }
		case 11: {
			Messages::File_Read *msg = (Messages::File_Read *)data;
			msg->RC = msg->F->Read(msg->Buffer, msg->Amount, msg->Offset);
			break; }
		case 12: {
			Messages::File_Write *msg = (Messages::File_Write *)data;
			msg->RC = msg->F->Write(msg->Buffer, msg->Amount, msg->Offset);
			break; }
		case 13: {
			Messages::File_Truncate *msg = (Messages::File_Truncate *)data;
			msg->RC = msg->F->Truncate(msg->Size);
			break; }
		case 14: {
			Messages::File_Sync *msg = (Messages::File_Sync *)data;
			msg->RC = msg->F->Sync(msg->Flags);
			break; }
		case 15: {
			Messages::File_get_FileSize *msg = (Messages::File_get_FileSize *)data;
			msg->RC = msg->F->get_FileSize(msg->Size);
			break; }
		case 16: {
			Messages::File_Lock *msg = (Messages::File_Lock *)data;
			msg->RC = msg->F->Lock(msg->Lock);
			break; }
		case 17: {
			Messages::File_CheckReservedLock *msg = (Messages::File_CheckReservedLock *)data;
			msg->RC = msg->F->CheckReservedLock(msg->Lock);
			break; }
		case 18: {
			Messages::File_Unlock *msg = (Messages::File_Unlock *)data;
			msg->RC = msg->F->Unlock(msg->Lock);
			break; }
#pragma endregion
#pragma region System
		case 1: {
			Messages::System_Open *msg = (Messages::System_Open *)data;
			VFile *f = (VFile *)_allocZero(vfs->SizeOsFile);
			msg->RC = vfs->Open(msg->Name, f, msg->Flags, &msg->OutFlags);
			msg->F = f;
			break; }
		case 2: {
			Messages::System_Delete *msg = (Messages::System_Delete *)data;
			msg->RC = vfs->Delete(msg->Filename, msg->SyncDir);
			break; }
		case 3: {
			Messages::System_Access *msg = (Messages::System_Access *)data;
			msg->RC = vfs->Access(msg->Filename, msg->Flags, &msg->ResOut);
			break; }
		case 4: {
			Messages::System_FullPathname *msg = (Messages::System_FullPathname *)data;
			msg->RC = vfs->FullPathname(msg->Relative, msg->FullLength, msg->Full);
			break; }
		case 5: {
			Messages::System_GetLastError *msg = (Messages::System_GetLastError *)data;
			msg->RC = vfs->GetLastError(msg->BufLength, msg->Buf);
			break; }
#pragma endregion
		}
	}

	void Sentinel_Send(void *data, int length)
	{
		long id = InterlockedAdd(&_map.Id, 1);
		SentinelCommand *cmd = &_map.Commands[id % _lengthof(_map.Commands)];
		cmd->Length = length;
		switch (((char *)data)[0])
		{
#pragma region File
		case 10: {
			Messages::File_Close *msg = (Messages::File_Close *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 11: {
			Messages::File_Read *msg = (Messages::File_Read *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 12: {
			Messages::File_Write *msg = (Messages::File_Write *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 13: {
			Messages::File_Truncate *msg = (Messages::File_Truncate *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 14: {
			Messages::File_Sync *msg = (Messages::File_Sync *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 15: {
			Messages::File_get_FileSize *msg = (Messages::File_get_FileSize *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 16: {
			Messages::File_Lock *msg = (Messages::File_Lock *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 17: {
			Messages::File_CheckReservedLock *msg = (Messages::File_CheckReservedLock *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 18: {
			Messages::File_Unlock *msg = (Messages::File_Unlock *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
#pragma endregion
#pragma region System
		case 1: {
			Messages::System_Open *msg = (Messages::System_Open *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 2: {
			Messages::System_Delete *msg = (Messages::System_Delete *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 3: {
			Messages::System_Access *msg = (Messages::System_Access *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 4: {
			Messages::System_FullPathname *msg = (Messages::System_FullPathname *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
		case 5: {
			Messages::System_GetLastError *msg = (Messages::System_GetLastError *)data;
			msg->Prepare(cmd->Data, sizeof(cmd->Data));
			break; }
#pragma endregion
		}
		memcpy(cmd->Data, data, length);
		OneCommand(_vfs, cmd->Data, cmd->Length);
		memcpy(data, cmd->Data, length);
	}
}
