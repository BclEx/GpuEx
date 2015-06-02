#define RUNTIME_NAME RuntimeS
#include <Runtime.h>

#if OS_MAP
#pragma region OS_MAP

#define CORE_NAME CoreS
#include "Core.cu.h"
#include "SysEx.cu"
#include "SysEx+Random.cu"
#include "VFile.cu"
#include "VSystem.cu"
#include "VSystem-win.cu"

namespace CORE_NAME
{
	static bool Executor(void *tag, RuntimeSentinelMessage *data, int length)
	{
		VSystem *vfs = (VSystem *)tag;
		switch (data->OP)
		{
#pragma region File
		case 10: {
			Messages::File_Close *msg = (Messages::File_Close *)data;
			msg->RC = msg->F->Close_();
			return true; }
		case 11: {
			Messages::File_Read *msg = (Messages::File_Read *)data;
			msg->RC = msg->F->Read(msg->Buffer, msg->Amount, msg->Offset);
			return true; }
		case 12: {
			Messages::File_Write *msg = (Messages::File_Write *)data;
			msg->RC = msg->F->Write(msg->Buffer, msg->Amount, msg->Offset);
			return true; }
		case 13: {
			Messages::File_Truncate *msg = (Messages::File_Truncate *)data;
			msg->RC = msg->F->Truncate(msg->Size);
			return true; }
		case 14: {
			Messages::File_Sync *msg = (Messages::File_Sync *)data;
			msg->RC = msg->F->Sync(msg->Flags);
			return true; }
		case 15: {
			Messages::File_get_FileSize *msg = (Messages::File_get_FileSize *)data;
			msg->RC = msg->F->get_FileSize(msg->Size);
			return true; }
		case 16: {
			Messages::File_Lock *msg = (Messages::File_Lock *)data;
			msg->RC = msg->F->Lock(msg->Lock);
			return true; }
		case 17: {
			Messages::File_CheckReservedLock *msg = (Messages::File_CheckReservedLock *)data;
			msg->RC = msg->F->CheckReservedLock(msg->Lock);
			return true; }
		case 18: {
			Messages::File_Unlock *msg = (Messages::File_Unlock *)data;
			msg->RC = msg->F->Unlock(msg->Lock);
			return true; }
#pragma endregion
#pragma region System
		case 21: {
			Messages::System_Open *msg = (Messages::System_Open *)data;
			VFile *f = (VFile *)_allocZero(vfs->SizeOsFile);
			msg->RC = vfs->Open(msg->Name, f, msg->Flags, &msg->OutFlags);
			msg->F = f;
			return true; }
		case 22: {
			Messages::System_Delete *msg = (Messages::System_Delete *)data;
			msg->RC = vfs->Delete(msg->Filename, msg->SyncDir);
			return true; }
		case 23: {
			Messages::System_Access *msg = (Messages::System_Access *)data;
			msg->RC = vfs->Access(msg->Filename, msg->Flags, &msg->ResOut);
			return true; }
		case 24: {
			Messages::System_FullPathname *msg = (Messages::System_FullPathname *)data;
			msg->RC = vfs->FullPathname(msg->Relative, msg->FullLength, msg->Full);
			return true; }
		case 25: {
			Messages::System_GetLastError *msg = (Messages::System_GetLastError *)data;
			msg->RC = vfs->GetLastError(msg->BufLength, msg->Buf);
			return true; }
#pragma endregion
		}
		return false;
	}

	static RuntimeSentinelExecutor _sysExecutor;
	void VSystemSentinel::Initialize()
	{
		MutexEx masterMutex;
		SysEx::Initialize(masterMutex);
		VSystem *vfs = VSystem::FindVfs(nullptr);
		_sysExecutor.Name = "sys";
		_sysExecutor.Executor = Executor;
		_sysExecutor.Tag = vfs;
		RuntimeSentinel::ServerInitialize(&_sysExecutor);
	}

	void VSystemSentinel::Shutdown()
	{
		RuntimeSentinel::ServerShutdown();
		SysEx::Shutdown();
	}
}

#pragma region
#endif