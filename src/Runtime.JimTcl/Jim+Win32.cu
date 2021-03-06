#pragma region License
/*
* WIN32 extension
*
* Copyright (C) 2005 Pat Thoyts <patthoyts@users.sourceforge.net>
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above
*    copyright notice, this list of conditions and the following
*    disclaimer in the documentation and/or other materials
*    provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE JIM TCL PROJECT ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* JIM TCL PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation
* are those of the authors and should not be interpreted as representing
* official policies, either expressed or implied, of the Jim Tcl Project.
*/
#pragma endregion

#include "Jim.h"
// Apparently windows.h and cygwin don't mix, but we seem to get away with it here. Use at your own risk under cygwin
//#if defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
//#endif
#include <shellapi.h>
#include <lmcons.h>
#include <psapi.h>
#include <ctype.h>
#if _MSC_VER >= 1000
#pragma comment(lib, "shell32")
#pragma comment(lib, "user32")
#pragma comment(lib, "advapi32")
#pragma comment(lib, "psapi")
#endif

__device__ static Jim_Obj *Win32ErrorObj(Jim_Interp *interp, const char *szPrefix, DWORD dwError)
{
	char *lpBuffer = NULL;
	DWORD dwLen = 0;
	dwLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, dwError, LANG_NEUTRAL, (char *)&lpBuffer, 0, NULL);
	if (dwLen < 1)
		dwLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_STRING | FORMAT_MESSAGE_ARGUMENT_ARRAY, "code 0x%1!08X!%n", 0, LANG_NEUTRAL, (char *)&lpBuffer, 0, (va_list *)&dwError);
	Jim_Obj *msgObj = Jim_NewStringObj(interp, szPrefix, -1);
	if (dwLen > 0) {
		char *p = lpBuffer + dwLen - 1; // remove cr-lf at end
		for (; p && *p && isspace(*p); p--) { }
		*++p = 0;
		Jim_AppendString(interp, msgObj, ": ", 2);
		Jim_AppendString(interp, msgObj, lpBuffer, -1);
	}
	LocalFree((HLOCAL)lpBuffer);
	return msgObj;
}

// win32.ShellExecute verb file args
__device__ static int Win32_ShellExecute(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc < 3 || objc > 4) {
		Jim_WrongNumArgs(interp, 1, objv, "verb path ?parameters?");
		return JIM_ERROR;
	}
	const char *verb = Jim_String(objv[1]);
	const char *file = Jim_String(objv[2]);
	char cwd[MAX_PATH + 1];
	GetCurrentDirectoryA(MAX_PATH + 1, cwd);
	const char *parm = (objc == 4 ? Jim_String(objv[3]) : NULL);
	int r = (int)ShellExecuteA(NULL, verb, file, parm, cwd, SW_SHOWNORMAL);
	if (r < 33)
		Jim_SetResult(interp, Win32ErrorObj(interp, "ShellExecute", GetLastError()));
	return (r < 33 ? JIM_ERROR : JIM_OK);
}

// win32.FindWindow title ?class?
__device__ static int Win32_FindWindow(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc < 2 || objc > 3) {
		Jim_WrongNumArgs(interp, 1, objv, "title ?class?");
		return JIM_ERROR;
	}
	int r = JIM_OK;
	const char *title = Jim_String(objv[1]);
	const char *class_ = (objc == 3 ? Jim_String(objv[2]) : NULL);
	HWND hwnd = hwnd = FindWindowA(class_, title);
	if (hwnd == NULL) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "FindWindow", GetLastError()));
		r = JIM_ERROR;
	} else
		Jim_SetResult(interp, Jim_NewIntObj(interp, (long)hwnd));
	return r;
}

// win32.CloseWindow windowHandle
__device__ static int Win32_CloseWindow(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "?windowHandle?");
		return JIM_ERROR;
	}
	long hwnd;
	if (Jim_GetLong(interp, objv[1], &hwnd) != JIM_OK)
		return JIM_ERROR;
	if (!CloseWindow((HWND)hwnd)) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "CloseWindow", GetLastError()));
		return JIM_ERROR;
	}
	return JIM_OK;
}

__device__ static int Win32_GetActiveWindow(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	Jim_SetResult(interp, Jim_NewIntObj(interp, (DWORD)GetActiveWindow()));
	return JIM_OK;
}

__device__ static int Win32_SetActiveWindow(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "windowHandle");
		return JIM_ERROR;
	}
	HWND hwnd;
	int r = r = Jim_GetLong(interp, objv[1], (long *)&hwnd);
	if (r == JIM_OK) {
		HWND old = SetActiveWindow(hwnd);
		if (old == NULL) {
			Jim_SetResult(interp, Win32ErrorObj(interp, "SetActiveWindow", GetLastError()));
			r = JIM_ERROR;
		} else 
			Jim_SetResult(interp, Jim_NewIntObj(interp, (long)old));
	}
	return r;
}

__device__ static int Win32_SetForegroundWindow(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "windowHandle");
		return JIM_ERROR;
	}
	HWND hwnd;
	int r = Jim_GetLong(interp, objv[1], (long *)&hwnd);
	if (r == JIM_OK) {
		if (!SetForegroundWindow(hwnd)) {
			Jim_SetResult(interp, Win32ErrorObj(interp, "SetForegroundWindow", GetLastError()));
			r = JIM_ERROR;
		}
	}
	return r;
}

static int Win32_Beep(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 3) {
		Jim_WrongNumArgs(interp, 1, objv, "freq duration");
		return JIM_ERROR;
	}
	long freq, duration;
	int r = Jim_GetLong(interp, objv[1], &freq);
	if (r == JIM_OK)
		r = Jim_GetLong(interp, objv[2], &duration);
	if (freq < 0x25) freq = 0x25;
	if (freq > 0x7fff) freq = 0x7fff;
	if (r == JIM_OK) {
		if (!Beep(freq, duration)) {
			Jim_SetResult(interp, Win32ErrorObj(interp, "Beep", GetLastError()));
			r = JIM_ERROR;
		}
	}
	return r;
}

__device__ static int Win32_GetComputerName(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 1) {
		Jim_WrongNumArgs(interp, 1, objv, "");
		return JIM_ERROR;
	}
	int r = JIM_OK;
	char name[MAX_COMPUTERNAME_LENGTH + 1];
	DWORD size = MAX_COMPUTERNAME_LENGTH;
	if (GetComputerNameA(name, &size)) {
		Jim_Obj *nameObj = Jim_NewStringObj(interp, name, size);
		Jim_SetResult(interp, nameObj);
	} else {
		Jim_SetResult(interp, Win32ErrorObj(interp, "GetComputerName", GetLastError()));
		r = JIM_ERROR;
	}
	return r;
}

__device__ static int Win32_GetUserName(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 1) {
		Jim_WrongNumArgs(interp, 1, objv, "");
		return JIM_ERROR;
	}
	int r = JIM_OK;
	char name[UNLEN + 1];
	DWORD size = UNLEN;
	if (GetUserNameA(name, &size)) {
		Jim_Obj *nameObj = Jim_NewStringObj(interp, name, size);
		Jim_SetResult(interp, nameObj);
	} else {
		Jim_SetResult(interp, Win32ErrorObj(interp, "GetUserName", GetLastError()));
		r = JIM_ERROR;
	}
	return r;
}

__device__ static int Win32_GetModuleFileName(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc > 2) {
		Jim_WrongNumArgs(interp, 1, objv, "?moduleid?");
		return JIM_ERROR;
	}
	HMODULE hModule = NULL;
	if (objc == 2)
		if (Jim_GetLong(interp, objv[1], (long *)&hModule) != JIM_OK)
			return JIM_ERROR;
	char path[MAX_PATH];
	DWORD len = GetModuleFileNameA(hModule, path, MAX_PATH);
	if (len != 0) {
		Jim_Obj *pathObj = Jim_NewStringObj(interp, path, len);
		Jim_SetResult(interp, pathObj);
	} else {
		Jim_SetResult(interp, Win32ErrorObj(interp, "GetModuleFileName", GetLastError()));
		return JIM_ERROR;
	}
	return JIM_OK;
}

__device__ static int Win32_GetVersion(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	Jim_SetResult(interp, Jim_NewIntObj(interp, GetVersion()));
	return JIM_OK;
}

__device__ static int Win32_GetTickCount(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	Jim_SetResult(interp, Jim_NewIntObj(interp, GetTickCount()));
	return JIM_OK;
}

__device__ static int Win32_GetSystemTime(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	Jim_Obj *a[16];
	size_t n = 0;
	SYSTEMTIME t;
	GetSystemTime(&t);
#define JIMADD(name) a[n++] = Jim_NewStringObj(interp, #name, -1); a[n++] = Jim_NewIntObj(interp, t.w ## name)
	JIMADD(Year);
	JIMADD(Month);
	JIMADD(DayOfWeek);
	JIMADD(Day);
	JIMADD(Hour);
	JIMADD(Minute);
	JIMADD(Second);
	JIMADD(Milliseconds);
#undef JIMADD
	Jim_SetResult(interp, Jim_NewListObj(interp, a, (int)n));
	return JIM_OK;
}

// function not available on mingw or cygwin
#if !defined(__MINGW32__) && !defined(__CYGWIN__)
// FIX ME: win2k+ so should do version checks really.
__device__ static int Win32_GetPerformanceInfo(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	Jim_Obj *a[26];
	size_t n = 0;
	PERFORMANCE_INFORMATION pi;
	if (!GetPerformanceInfo(&pi, sizeof(pi))) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "GetPerformanceInfo", GetLastError()));
		return JIM_ERROR;
	}
#define JIMADD(name) a[n++] = Jim_NewStringObj(interp, #name, -1); a[n++] = Jim_NewIntObj(interp, pi. name)
	JIMADD(CommitTotal);
	JIMADD(CommitLimit);
	JIMADD(CommitPeak);
	JIMADD(PhysicalTotal);
	JIMADD(PhysicalAvailable);
	JIMADD(SystemCache);
	JIMADD(KernelTotal);
	JIMADD(KernelPaged);
	JIMADD(KernelNonpaged);
	JIMADD(PageSize);
	JIMADD(HandleCount);
	JIMADD(ProcessCount);
	JIMADD(ThreadCount);
#undef JIMADD
	Jim_SetResult(interp, Jim_NewListObj(interp, a, (int)n));
	return JIM_OK;
}
#endif

__device__ static int Win32_SetComputerName(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "computername");
		return JIM_ERROR;
	}
	int r = JIM_OK;
	const char *name = Jim_String(objv[1]);
	if (!SetComputerNameA(name)) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "SetComputerName", GetLastError()));
		r = JIM_ERROR;
	}
	return r;
}

__device__ static int Win32_GetModuleHandle(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc < 1 || objc >  2) {
		Jim_WrongNumArgs(interp, 1, objv, "?name?");
		return JIM_ERROR;
	}
	const char *name = (objc == 2 ? Jim_String(objv[1]) : NULL);
	HMODULE hModule = GetModuleHandleA(name);
	if (hModule == NULL) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "GetModuleHandle", GetLastError()));
		return JIM_ERROR;
	}
	Jim_SetResult(interp, Jim_NewIntObj(interp, (unsigned long)hModule));
	return JIM_OK;
}

__device__ static int Win32_LoadLibrary(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "path");
		return JIM_ERROR;
	}
	HMODULE hLib = LoadLibraryA(Jim_String(objv[1]));
	if (hLib == NULL) {
		Jim_SetResult(interp, Win32ErrorObj(interp, "LoadLibrary", GetLastError()));
		return JIM_ERROR;
	}
	Jim_SetResult(interp, Jim_NewIntObj(interp, (unsigned long)hLib));
	return JIM_OK;
}

__device__ static int Win32_FreeLibrary(ClientData dummy, Jim_Interp *interp, int objc, Jim_Obj *const *objv)
{
	if (objc != 2) {
		Jim_WrongNumArgs(interp, 1, objv, "hmodule");
		return JIM_ERROR;
	}
	HMODULE hModule = NULL;
	int r = Jim_GetLong(interp, objv[1], (long *)&hModule);
	if (r == JIM_OK) {
		if (!FreeLibrary(hModule)) {
			Jim_SetResult(interp, Win32ErrorObj(interp, "FreeLibrary", GetLastError()));
			r = JIM_ERROR;
		}
	}
	return r;
}

__device__ int Jim_win32Init(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "win32", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
#define CMD(name) Jim_CreateCommand(interp, "win32." #name, Win32_ ## name, NULL, NULL)
	CMD(ShellExecute);
	CMD(FindWindow);
	CMD(CloseWindow);
	CMD(GetActiveWindow);
	CMD(SetActiveWindow);
	CMD(SetForegroundWindow);
	CMD(Beep);
	CMD(GetComputerName);
	CMD(SetComputerName);
	CMD(GetUserName);
	CMD(GetModuleFileName);
	CMD(GetVersion);
	CMD(GetTickCount);
	CMD(GetSystemTime);
#if !defined(__MINGW32__) && !defined(__CYGWIN__)
	CMD(GetPerformanceInfo);
#endif
	CMD(GetModuleHandle);
	CMD(LoadLibrary);
	CMD(FreeLibrary);
	return JIM_OK;
}
