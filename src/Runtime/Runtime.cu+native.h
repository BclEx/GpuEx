#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#ifndef __static__
#define __static__
#endif
#include "RuntimeHost.h"

///////////////////////////////////////////////////////////////////////////////
// STRUCT
#pragma region STRUCT

typedef struct __align__(8)
{
	int threadid; // RUNTIME_UNRESTRICTED for unrestricted
	int blockid;  // RUNTIME_UNRESTRICTED for unrestricted
} runtimeRestriction;

typedef struct __align__(8) runtimeHeap_s
{
	volatile char *blockPtr; // current atomically-incremented non-wrapped offset
	void *reserved;
	runtimeRestriction restriction;
	size_t blockSize;
	size_t blocksLength; // size of circular buffer (set up by host)
	char *blocks; // start of circular buffer (set up by host)
} runtimeHeap;

typedef struct __align__(8)
{
	unsigned short magic;		// magic number says we're valid
	unsigned short type;		// type of block
	unsigned short fmtoffset;	// offset of fmt string into buffer
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} runtimeBlockHeader;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
__shared__ static runtimeHeap *__runtimeHeap;
extern "C" __device__ static void _runtimeSetHeap(void *heap) { __runtimeHeap = (runtimeHeap *)heap; }
extern "C" cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host) { return cudaSuccess; }
#else
__device__ runtimeHeap *__runtimeHeap;
extern "C" __device__ void _runtimeSetHeap(void *heap) { }
extern "C" cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host) { return cudaMemcpyToSymbol(__runtimeHeap, &host.heap, sizeof(__runtimeHeap)); }
#endif

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define RUNTIME_MAGIC (unsigned short)0xC811

extern "C" __device__ __static__ char *__heap_movenext(char *&end, char *&bufptr)
{
	if (!__runtimeHeap) __THROW;
	// thread/block restriction check
	runtimeRestriction restriction = __runtimeHeap->restriction;
	if (restriction.blockid != CURT_UNRESTRICTED && restriction.blockid != (blockIdx.x + gridDim.x*blockIdx.y))
		return nullptr;
	if (restriction.threadid != CURT_UNRESTRICTED && restriction.threadid != (threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z))
		return nullptr;
	// advance pointer
	char *start = __runtimeHeap->blocks;
	size_t offset = atomicAdd((unsigned int *)&__runtimeHeap->blockPtr, __runtimeHeap->blockSize) - (size_t)start;
	offset %= __runtimeHeap->blocksLength;
	start += offset;
	end = start + __runtimeHeap->blockSize;
	bufptr = start + sizeof(runtimeBlockHeader);
	return start;
}

extern "C" __device__ __static__ void runtimeRestrict(int threadid, int blockid)
{
	int threadMax = blockDim.x * blockDim.y * blockDim.z;
	if ((threadid < threadMax && threadid >= 0) || threadid == CURT_UNRESTRICTED)
		__runtimeHeap->restriction.threadid = threadid;
	int blockMax = gridDim.x * gridDim.y;
	if ((blockid < blockMax && blockid >= 0) || blockid == CURT_UNRESTRICTED)
		__runtimeHeap->restriction.blockid = blockid;
}

extern "C" __device__ __static__ void __heap_writeheader(unsigned short type, char *ptr, char *fmtptr)
{
	runtimeBlockHeader header;
	header.magic = RUNTIME_MAGIC;
	header.type = type;
	header.fmtoffset = (unsigned short)(fmtptr - ptr);
	header.blockid = gridDim.x*blockIdx.y + blockIdx.x;
	header.threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	*(runtimeBlockHeader *)(void *)ptr = header;
}

extern "C" __device__ __static__ char *__heap_write(char *dest, const char *src, int maxLength, char *end)
{
	// initialization and overflow check
	if (!dest || dest >= end) //|| !src)
		return nullptr;
	// prepare to write the length specifier. We're guaranteed to have at least "RUNTIME_ALIGNSIZE" bytes left because we only write out in
	// chunks that size, and blockSize is aligned with RUNTIME_ALIGNSIZE.
	int *lenptr = (int *)(void *)dest;
	int len = 0;
	dest += __HEAP_ALIGNSIZE;
	// now copy the string
	if (maxLength == 0)
		maxLength = __runtimeHeap->blockSize;
	while (maxLength--)
	{
		if (dest >= end) // overflow check
			break;
		len++;
		*dest++ = *src;
		if (*src++ == '\0')
			break;
	}
	// now write out the padding bytes, and we have our length.
	while (dest < end && ((unsigned long long)dest & (__HEAP_ALIGNSIZE - 1)) != 0)
	{
		len++;
		*dest++ = 0;
	}
	*lenptr = len;
	return (dest < end ? dest : nullptr); // overflow means return nullptr
}

#pragma endregion

//////////////////////
// ASSERT
#pragma region ASSERT

#define ASSERT_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __heap_movenext(end, bufptr)) == nullptr) return;
#define ASSERT_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define ASSERT_POSTAMBLE \
	fmtstart = bufptr; end = __heap_write(bufptr, fmt, 0, end); \
	__heap_writeheader(__HEAP_HEADER_ASSERT, start, (end ? fmtstart : nullptr));

extern "C" __device__ __static__ void __assertWrite(const char *fmt, const char *file, unsigned int line)
{
	ASSERT_PREAMBLE;
	ASSERT_POSTAMBLE;
}

#undef ASSERT_PREAMBLE
#undef ASSERT_ARG
#undef ASSERT_POSTAMBLE

#pragma endregion

//////////////////////
// EMBED
#pragma region EMBED
//#ifdef __EMBED__

__constant__ unsigned char __curtUpperToLower[256] = {
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
	18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
	36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
	54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 97, 98, 99,100,101,102,103,
	104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,
	122, 91, 92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,104,105,106,107,
	108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,
	126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
	144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,
	162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,
	180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,
	198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
	216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,
	234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,
	252,253,254,255
};

__constant__ unsigned char __curtCtypeMap[256] = {
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 00..07    ........ */
	0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,  /* 08..0f    ........ */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 10..17    ........ */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 18..1f    ........ */
	0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,  /* 20..27     !"#$%&' */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 28..2f    ()*+,-./ */
	0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,  /* 30..37    01234567 */
	0x0c, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 38..3f    89:;<=>? */

	0x00, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x02,  /* 40..47    @ABCDEFG */
	0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  /* 48..4f    HIJKLMNO */
	0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  /* 50..57    PQRSTUVW */
	0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x40,  /* 58..5f    XYZ[\]^_ */
	0x00, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x22,  /* 60..67    `abcdefg */
	0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,  /* 68..6f    hijklmno */
	0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,  /* 70..77    pqrstuvw */
	0x22, 0x22, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 78..7f    xyz{|}~. */

	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 80..87    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 88..8f    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 90..97    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 98..9f    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* a0..a7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* a8..af    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* b0..b7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* b8..bf    ........ */

	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* c0..c7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* c8..cf    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* d0..d7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* d8..df    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* e0..e7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* e8..ef    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* f0..f7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40   /* f8..ff    ........ */
};

//#endif
#pragma endregion