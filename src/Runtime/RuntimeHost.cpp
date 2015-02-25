#pragma warning(disable: 4996)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#include <string.h>
#include <stdio.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <alloca.h>
#else
#include <malloc.h>
#endif
#include "RuntimeHost.h"
#include "cuda_runtime_api.h"

#define CURT_UNRESTRICTED -1

typedef struct __align__(8)
{
	int threadid; // RUNTIME_UNRESTRICTED for unrestricted
	int blockid;  // RUNTIME_UNRESTRICTED for unrestricted
} runtimeRestriction;

typedef struct __align__(8)
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

static FILE *_stream;
cudaError __cudaLastError;

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

#define RUNTIME_MAGIC (unsigned short)0xC811
#define __HEAP_ALIGNSIZE sizeof(long long)
#define	__HEAP_HEADER_RAW 0
#define __HEAP_HEADER_PRINTF 1
#define __HEAP_HEADER_TRANSFER 2
#define __HEAP_HEADER_ASSERT 4
#define __HEAP_HEADER_THROW 5

extern "C" cudaDeviceHeap cudaDeviceHeapCreate(size_t blockSize, size_t length, cudaError_t *error, void *reserved)
{
	cudaError_t localError; if (error == nullptr) error = &localError;
	cudaDeviceHeap host; memset(&host, 0, sizeof(cudaDeviceHeap));
	// fix up blockSize to include fallocBlockHeader
	blockSize = (blockSize + (__HEAP_ALIGNSIZE - 1)) & ~(__HEAP_ALIGNSIZE - 1);
	// fix up length to be a multiple of blockSize
	if (!length || length % blockSize)
		length += blockSize - (length % blockSize);
	size_t blocksLength = length;
	length = (length + sizeof(runtimeHeap) + 15) & ~15;
	// allocate a heap on the device and zero it
	char *heap;
#ifndef _LCcpu
	if ((*error = cudaMalloc((void **)&heap, length)) != cudaSuccess || (*error = cudaMemset(heap, 0, length)) != cudaSuccess)
		return host;
#else
	heap = nullptr;
#endif
	char *blocks = heap + length - blocksLength;
	// no restrictions to begin with
	runtimeRestriction restriction;
	restriction.threadid = restriction.blockid = CURT_UNRESTRICTED;
	// transfer to heap
	runtimeHeap hostHeap;
	hostHeap.blockPtr = (volatile char *)blocks;
	hostHeap.blocks = blocks;
	hostHeap.reserved = reserved;
	hostHeap.restriction = restriction;
	hostHeap.blockSize = blockSize;
	hostHeap.blocksLength = blocksLength;
#ifndef _LCcpu
	if ((*error = cudaMemcpy(heap, &hostHeap, sizeof(runtimeHeap), cudaMemcpyHostToDevice)) != cudaSuccess)
		return host;
#endif
	// return the heap
	host.reserved = reserved;
	host.heap = heap;
	host.blocks = host.blockStart = blocks;
	host.blockSize = blockSize;
	host.blocksLength = blocksLength;
	host.length = length;
	return host;
}

extern "C" void cudaDeviceHeapDestroy(cudaDeviceHeap &host)
{
	if (!host.heap)
		return;
	#ifndef _LCcpu
	cudaFree(host.heap); host.heap = nullptr;
#endif
}

extern "C" void cudaDeviceHeapSetHandler(cudaDeviceHeap &host, cudaAssertHandler handler)
{
	host.assertHandler = handler;
}

static bool outputPrintfData(FILE *stream, size_t blockSize, char *fmt, char *data);
static int executeRuntime(cudaAssertHandler assertHandler, size_t blockSize, char *heap, int headings, int clear, char *bufstart, char *bufend, char *bufptr, char *endptr)
{
	// grab, piece-by-piece, each output element until we catch up with the circular buffer end pointer
	int count = 0;
	char *b = (char *)alloca(blockSize + 1);
	b[blockSize] = '\0';
	while (bufptr != endptr)
	{
		// wrap ourselves at the end-of-buffer
		if (bufptr == bufend)
			bufptr = bufstart;
		// adjust our start pointer to within the circular buffer and copy a block.
#ifndef _LCcpu
		if (cudaMemcpy(b, bufptr, blockSize, cudaMemcpyDeviceToHost) != cudaSuccess)
			break;
#endif
		// if the magic number isn't valid, then this write hasn't gone through yet and we'll wait until it does (or we're past the end for non-async printfs).
		runtimeBlockHeader *hdr = (runtimeBlockHeader *)b;
		//if (hdr->magic != RUNTIME_MAGIC || hdr->fmtoffset >= blockSize)
		//{
		//	fprintf(_stream, "Bad magic number in runtime header\n");
		//	break;
		//}
		// Extract all the info and get this printf done
		bool error = false;
		switch (hdr->type)
		{
		case __HEAP_HEADER_PRINTF:
			if (headings)
				fprintf(_stream, "[%d, %d]: ", hdr->blockid, hdr->threadid);
			if (hdr->fmtoffset == 0)
				fprintf(_stream, "printf buffer overflow\n");
			else
				error = !outputPrintfData(_stream, blockSize, b + hdr->fmtoffset, b + sizeof(runtimeBlockHeader));
			break;
		case __HEAP_HEADER_TRANSFER:
			break;
		case __HEAP_HEADER_ASSERT:
			if (headings)
				fprintf(_stream, "[%d, %d]: ", hdr->blockid, hdr->threadid);
			fprintf(_stream, "ASSERT: ");
			if (hdr->fmtoffset == 0)
				fprintf(_stream, "printf buffer overflow\n");
			else
				error = !outputPrintfData(_stream, blockSize, b + hdr->fmtoffset, b + sizeof(runtimeBlockHeader));
			fprintf(_stream, "\n");
			//assertHandler();
			break;
		case __HEAP_HEADER_THROW:
			if (headings)
				fprintf(_stream, "[%d, %d]: ", hdr->blockid, hdr->threadid);
			fprintf(_stream, "THROW: ");
			if (hdr->fmtoffset == 0)
				fprintf(_stream, "printf buffer overflow\n");
			else
				error = !outputPrintfData(_stream, blockSize, b + hdr->fmtoffset, b + sizeof(runtimeBlockHeader));
			fprintf(_stream, "\n");
			break;
		}
		if (error)
			break;
		count++;
		// clear if asked
#ifndef _LCcpu
		if (clear)
			cudaMemset(bufptr, 0, blockSize);
#endif
		// now advance our start location, because we're done, and keep copying
		bufptr += blockSize;
	}
	return count;
}

extern "C" cudaError_t cudaDeviceHeapSynchronize(cudaDeviceHeap &host, void *stream, bool showThreadID)
{
	// for now, we force "synchronous" mode which means we're not concurrent with kernel execution. This also means we don't need clearOnPrint.
	// if you're patching it for async operation, here's where you want it.
	bool sync = true;
	//
	_stream = (FILE *)(stream == nullptr ? stdout : stream);
	// initialisation check
	if (!host.blockStart || !host.heap || !_stream)
		return cudaErrorMissingConfiguration;
	size_t blocksLength = host.blocksLength;
	char *blocks = host.blocks;
	// grab the current "end of circular buffer" pointer.
	char *blockEnd = nullptr;
#ifndef _LCcpu
	cudaMemcpy(&blockEnd, host.heap, sizeof(char *), cudaMemcpyDeviceToHost);
#endif
	// adjust our starting and ending pointers to within the block
	char *bufptr = ((host.blockStart - blocks) % blocksLength) + blocks;
	char *endptr = ((blockEnd - blocks) % blocksLength) + blocks;
	// for synchronous (i.e. after-kernel-exit) printf display, we have to handle circular buffer wrap carefully because we could miss those past "end".
	if (sync)
		executeRuntime(host.assertHandler, host.blockSize, blocks, showThreadID, false, blocks, blocks + blocksLength, endptr, blocks + blocksLength);
	executeRuntime(host.assertHandler, host.blockSize, blocks, showThreadID, false, blocks, blocks + blocksLength, bufptr, endptr);
	host.blockStart = blockEnd;
	// if we were synchronous, then we must ensure that the memory is cleared on exit otherwise another kernel launch with a different grid size could conflict.
#ifndef _LCcpu
	if (sync)
		cudaMemset(blocks, 0, blocksLength);
#endif
	return cudaSuccess;
}

//////////////////////
// PRINTF
#pragma region PRINTF

static bool outputPrintfData(FILE *stream, size_t blockSize, char *fmt, char *data)
{
	// Format string is prefixed by a length that we don't need
	fmt += __HEAP_ALIGNSIZE;
	// now run through it, printing everything we can. We must run to every % character, extract only that, and use printf to format it.
	char *p = strchr(fmt, '%');
	while (p != nullptr)
	{
		// print up to the % character
		*p = '\0'; fputs(fmt, stream); *p = '%'; // Put back the %
		// now handle the format specifier
		char *format = p++; // Points to the '%'
		p += strcspn(p, "%cdiouxXeEfgGaAnps");
		if (*p == '\0') // If no format specifier, print the whole thing
		{
			fmt = format;
			break;
		}
		// cut out the format bit and use printf to print it. It's prefixed by its length.
		int arglen = *(int *)data;
		if (arglen > blockSize)
		{
			fputs("Corrupt printf buffer data - aborting\n", stream);
			return false;
		}
		data += __HEAP_ALIGNSIZE;
		//
		char specifier = *p++;
		char c = *p; *p = '\0'; // store for later and clip
		switch (specifier)
		{
			// these all take integer arguments
		case 'c': case 'd': case 'i': case 'o': case 'u': case 'x': case 'X': case 'p': fprintf(stream, format, *((int *)data)); break;
			// these all take float/double arguments, float vs. double thing
		case 'e': case 'E': case 'f': case 'g': case 'G': case 'a': case 'A': if (arglen == 4) fprintf(stream, format, *((float *)data)); else fprintf(stream, format, *((double *)data)); break;
			// Strings are handled in a special way
		case 's': fprintf(stream, format, (char *)data); break;
			// % is special
		case '%': fprintf(stream, "%%"); break;
			// everything else is just printed out as-is
		default: fprintf(stream, format); break;
		}
		data += __HEAP_ALIGNSIZE; // move on to next argument
		*p = c; fmt = p; // restore what we removed, and adjust fmt string to be past the specifier
		p = strchr(fmt, '%'); // and get the next specifier
	}
	// print out the last of the string
	fputs(fmt, stream);
	return true;
}

static bool outputPrintfData(char *stream, size_t blockSize, char *fmt, char *data)
{
	// Format string is prefixed by a length that we don't need
	fmt += __HEAP_ALIGNSIZE;
	// now run through it, printing everything we can. We must run to every % character, extract only that, and use printf to format it.
	char *p = strchr(fmt, '%');
	while (p != nullptr)
	{
		// print up to the % character
		*p = '\0'; strcpy(stream, fmt); *p = '%'; // Put back the %
		// now handle the format specifier
		char *format = p++; // Points to the '%'
		p += strcspn(p, "%cdiouxXeEfgGaAnps");
		if (*p == '\0') // If no format specifier, print the whole thing
		{
			fmt = format;
			break;
		}
		// cut out the format bit and use printf to print it. It's prefixed by its length.
		int arglen = *(int *)data;
		if (arglen > blockSize)
		{
			strcpy(stream, "Corrupt printf buffer data - aborting\n");
			return false;
		}
		data += __HEAP_ALIGNSIZE;
		//
		char specifier = *p++;
		char c = *p; *p = '\0'; // store for later and clip
		switch (specifier)
		{
			// these all take integer arguments
		case 'c': case 'd': case 'i': case 'o': case 'u': case 'x': case 'X': case 'p': sprintf(stream, format, *((int *)data)); break;
			// these all take float/double arguments, float vs. double thing
		case 'e': case 'E': case 'f': case 'g': case 'G': case 'a': case 'A': if (arglen == 4) sprintf(stream, format, *((float *)data)); else sprintf(stream, format, *((double *)data)); break;
			// Strings are handled in a special way
		case 's': sprintf(stream, format, (char *)data); break;
			// % is special
		case '%': sprintf(stream, "%%"); break;
			// everything else is just printed out as-is
		default: sprintf(stream, format); break;
		}
		data += __HEAP_ALIGNSIZE; // move on to next argument
		*p = c; fmt = p; // restore what we removed, and adjust fmt string to be past the specifier
		p = strchr(fmt, '%'); // and get the next specifier
	}
	// print out the last of the string
	strcpy(stream, fmt);
	return true;
}

#pragma endregion

//////////////////////
// EMBED
#pragma region EMBED
//#ifdef _LCcpu

extern "C" const unsigned char __curtUpperToLower[256] = {
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

extern "C" const unsigned char __curtCtypeMap[256] = {
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
