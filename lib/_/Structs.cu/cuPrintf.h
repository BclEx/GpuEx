#ifndef CUPRINTF_H
#define CUPRINTF_H

/*
 *	This is the header file supporting cuPrintf.cu and defining both
 *	the host and device-side interfaces. See that file for some more
 *	explanation and sample use code. See also below for details of the
 *	host-side interfaces.
 *
 *  Quick sample code:
 *
	#include "cuPrintf.cu"
 	
	__global__ void testKernel(int val)
	{
		cuPrintf("Value is: %d\n", val);
	}

	int main()
	{
		cudaPrintfInit();
		testKernel<<< 2, 3 >>>(10);
		cudaPrintfDisplay(stdout, true);
		cudaPrintfEnd();
        return 0;
	}
*/

#define CUPRINTF_UNRESTRICTED	-1

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

//
//	cudaPrintfInit
//
//	Call this once to initialise the printf system. If the output
//	file or buffer size needs to be changed, call cudaPrintfEnd()
//	before re-calling cudaPrintfInit().
//
//	The default size for the buffer is 1 megabyte. For CUDA
//	architecture 1.1 and above, the buffer is filled linearly and
//	is completely used;	however for architecture 1.0, the buffer
//	is divided into as many segments are there are threads, even
//	if some threads do not call cuPrintf().
//
//	Arguments:
//		bufferLen - Length, in bytes, of total space to reserve
//		            (in device global memory) for output.
//
//	Returns:
//		cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfInit(size_t bufferLen=1048576);   // 1-meg - that's enough for 4096 printfs by all threads put together

//
//	cudaPrintfEnd
//
//	Cleans up all memories allocated by cudaPrintfInit().
//	Call this at exit, or before calling cudaPrintfInit() again.
//
extern "C" void cudaPrintfEnd();

//
//	cudaPrintfDisplay
//
//	Dumps the contents of the output buffer to the specified
//	file pointer. If the output pointer is not specified,
//	the default "stdout" is used.
//
//	Arguments:
//		outputFP     - A file pointer to an output stream.
//		showThreadID - If "true", output strings are prefixed
//		               by "[blockid, threadid] " at output.
//
//	Returns:
//		cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfDisplay(void *outputFP=NULL, bool showThreadID=false);

#endif  // CUPRINTF_H

/*
Limitations / Known Issues

Currently, the following limitations and restrictions apply to cuPrintf:

Buffer size is rounded up to the nearest factor of 256
Arguments associated with “%s” string format specifiers must be of type (const char *)
To print the value of a (const char *) pointer, it must first be converted to (char *). All (const char *) arguments are interpreted as strings
Non-zero return code does not match standard C printf()
Cannot asynchronously output the printf buffer (i.e. while kernel is running)
Calling cudaPrintfDisplay implicitly issues a cudaDeviceSynchronize()
Restrictions applied by cuPrintfRestrict persist between launches. To clear these from the host-side, you must call cudaPrintfEnd() then cudaPrintfInit() again
cuPrintf output is undefined if multiple modules are loaded into a single context
Compile with “-arch=sm_11” or better when possible. Buffer usage is far more efficient and register use is lower
Supported format specifiers are: “cdiouxXeEfgGaAs”
Behaviour of format specifiers, especially justification/size specifiers, are dependent on the host machine’s implementation of printf
cuPrintf requires applications to be built using the CUDA runtime API

 */