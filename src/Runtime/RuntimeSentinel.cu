//#include <windows.h>
//#include <process.h>
//#include <stdio.h>
//#include "Runtime.h"
//#include "RuntimeSentinel.h"
//
//enum OP
//{
//	OP_open,
//};
//
//typedef struct
//{
//	OP OP;
//	int DataLength;
//	unsigned char Data[1024];
//} SentinelCommand;
//
//namespace SentinelCommands
//{
//}
//
//typedef struct
//{
//	SentinelCommand Commands[4];
//} Sentinel;
//
//
////DWORD WINAPI myThread(LPVOID lpParameter)
//unsigned int __stdcall mythread(void *data) 
//{
//	while (true)
//	{
//		printf("Thread inside %d \n", GetCurrentThreadId());
//		_sleep(1000);
//	}
//	return 0;
//}
//
//HANDLE _thread;
//void InitializeSentinel()
//{
//	_thread = (HANDLE)_beginthreadex(0, 0, mythread, nullptr, 0, 0);
//	getchar();
//}
//
//void ShutdownSentinel()
//{
//	CloseHandle(_thread);
//}
//
//
//void ProcessCommand(SentinelCommand c)
//{
//}