#include <stdio.h>
#include <stdlib.h>
#include <RuntimeHost.h>
#include <Tcl.h>
#include <TclEx+Int.h>
#ifdef DEBUGGER
#include <TclEx+Dbg.h>
#endif

__device__ void TclEx_InitExtensions(Tcl_Interp *interp);

__device__ bool _quitFlag = false;
__constant__ char _initCmd[] = "puts stdout \"Tiny Tcl 6.8.0\n\"";

#ifdef TCL_MEM_DEBUG
__device__ char _dumpFile[100];
__device__ int cmdCheckmem(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " fileName\"", (char *)NULL);
		return TCL_ERROR;
	}
	_strcpy(_dumpFile, args[1]);
	_quitFlag = true;
	return TCL_OK;
}
#endif

// SAMPLE COMMAND
#if 0
__device__ int SampleCommand(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " Msg\"", (char *)NULL);
		return TCL_ERROR;
	}
	printf("%s\n", args[1]);
	return TCL_OK;
}
#endif

#pragma region Startup + Shutdown

struct PrimaryData {
	Tcl_Interp *interp;
	Tcl_CmdBuf buffer;
	int noninteractive;
	bool gotPartial;
	bool quitFlag;
	int retcode;
};
struct PrimaryData h_dataP;

// MAIN-INIT
#if __CUDACC__
static cudaDeviceHeap _deviceHeap;

__device__ struct PrimaryData d_dataP;
void D_DATAP() { cudaErrorCheck(cudaMemcpyToSymbol(d_dataP, &h_dataP, sizeof(h_dataP))); }
void H_DATAP() { cudaErrorCheck(cudaMemcpyFromSymbol(&h_dataP, d_dataP, sizeof(h_dataP))); }

__global__ void g_MainInit(int argc, char *const argv[]);
static int MainInit(int argc, char *const argv[]) {
	memset(&h_dataP, 0, sizeof(h_dataP));
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
	int deviceId = gpuGetMaxGflopsDeviceId();
	cudaErrorCheck(cudaSetDevice(deviceId));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*5));
	_deviceHeap = cudaDeviceHeapCreate(256, 100);
	cudaErrorCheck(cudaDeviceHeapSelect(_deviceHeap));
	//
	char **d_argv = cudaDeviceTransferStringArray(argc, argv);
	D_DATAP(); g_MainInit<<<1,1>>>(argc, d_argv); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATAP();
	cudaFree(d_argv);
	return h_dataP.retcode;
}

//#define _exit(v) _dataP.quitFlag = true; _dataP.retcode = 1; return
#define _dataP d_dataP
__global__ void g_MainInit(int argc, char *const argv[]) {
#else
#define _dataP h_dataP
static void MainInit(int argc, char *const argv[]) {
	memset(&h_dataP, 0, sizeof(h_dataP));
#endif
	Tcl_Interp *interp = _dataP.interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
	Tcl_InitMemory(interp);
#endif
	TclEx_InitDebug(interp);
	TclEx_InitGeneral(interp);
#ifdef DEBUGGER
	TclEx_InitDebug(interp);
#endif

	// Init any static extensions
	TclEx_InitExtensions(interp);
#ifdef TCL_MEM_DEBUG
	Tcl_CreateCommand(interp, "checkmem", cmdCheckmem, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	// SAMPLE COMMAND
#if 0
	Tcl_CreateCommand(interp, "sample", SampleCommand, nullptr, nullptr);
#endif
	_dataP.buffer = Tcl_CreateCmdBuf();

	int result;
	if (argc > 1 && _strcmp(argv[1], "-"))
	{
		char *filename = (char *)argv[1]+1;

		// Before we eval the file, create an args global containing the remaining arguments
		char *args = Tcl_Merge(argc - 2, (const char **)argv + 2);
		Tcl_SetVar(interp, "argv", args, TCL_GLOBAL_ONLY);
		_freeFast(args);

		result = Tcl_EvalFile(interp, filename);
		if (result != TCL_OK)
		{
			// And make sure we print an informative error if something goes wrong
			Tcl_AddErrorInfo(interp, "");
			printf("%s\n", Tcl_GetVar(interp, "errorInfo", TCL_LEAVE_ERR_MSG));
			_exit(1);
		}
		_exit(0);
	}
	else
	{
		// Are we in interactive mode or script from stdin mode?
		_dataP.noninteractive = (argc > 1);
#ifndef TCL_GENERIC_ONLY
		if (!_dataP.noninteractive)
		{
			result = Tcl_Eval(interp, _initCmd, 0, (char **)NULL);
			if (result != TCL_OK)
			{
				printf("%s\n", interp->result);
				_exit(1);
			}
		}
#endif
		_dataP.retcode = -1;
		return;
	}
}

// INTERACTIVE-PROMPT
static void InteractiveExecute(char *line);
void InteractivePrompt() {
	FILE *in = stdin;
	FILE *out = stdout;
	h_dataP.gotPartial = false;
	while (true) {
		clearerr(in);
		if (!h_dataP.gotPartial) {
			if (!h_dataP.noninteractive) fputs("% ", out);
			fflush(out);
		}
		char line[1000];
		if (fgets(line, 1000, in) == NULL) {
			if (!h_dataP.gotPartial) exit(0);
			line[0] = 0;
		}
		InteractiveExecute(line);
	}
}

// INTERACTIVE-EXEC
#if __CUDACC__
__global__ void g_InteractiveExecute(char *line);
static void InteractiveExecute(char *line) {
	char *d_line;
	int size = strlen(line) + 1;
	cudaErrorCheck(cudaMalloc((void **)&d_line, size));
	cudaErrorCheck(cudaMemcpy(d_line, line, size, cudaMemcpyHostToDevice));
	D_DATAP(); g_InteractiveExecute<<<1,1>>>(d_line); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATAP();
	cudaFree(d_line);
}

__global__ void g_InteractiveExecute(char *line) {
#else
static void InteractiveExecute(char *line) {
#endif
	Tcl_Interp *interp = _dataP.interp;
	Tcl_CmdBuf buffer = _dataP.buffer;
	char *cmd = Tcl_AssembleCmd(buffer, line);
	if (cmd == NULL) {
		_dataP.gotPartial = true;
		return;
	}

	_dataP.gotPartial = false;
#ifdef TCL_NO_HISTORY
	int result = Tcl_Eval(interp, cmd, 0, (char **)NULL);
#else
	int result = Tcl_RecordAndEval(interp, cmd, 0);
#endif
	if (result == TCL_OK) {
		if (*interp->result != 0 && !_dataP.noninteractive) printf("%s\n", interp->result);
		if (_quitFlag) {
			Tcl_DeleteInterp(interp);
			Tcl_DeleteCmdBuf(buffer);
#ifdef TCL_MEM_DEBUG
			Tcl_DumpActiveMemory(_dumpFile);
#endif
			_exit(0);
		}
	}
	else {
		if (result == TCL_ERROR) printf("Error");
		else printf("Error %d", result);
		if (*interp->result != 0) printf(": %s\n", interp->result);
		else printf("\n");
	}
}

// MAIN-SHUTDOWN
#if __CUDACC__
__global__ void g_MainShutdown();
static void MainShutdown() {
	D_DATAP(); g_MainShutdown<<<1,1>>>(); cudaErrorCheck(cudaDeviceHeapSynchronize(_deviceHeap)); H_DATAP();
	cudaDeviceHeapDestroy(_deviceHeap);
	cudaDeviceReset();
	//return h_dataP.retcode;
}

__global__ void g_MainShutdown() {
#else
static void MainShutdown() {
#endif
}

#pragma endregion

int main(int argc, char *const argv[])
{
	MainInit(argc, argv);
	if (h_dataP.quitFlag)
		exit(h_dataP.retcode);
	if (h_dataP.retcode == -1)
		InteractivePrompt();
	MainShutdown();
}