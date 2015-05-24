#ifndef __RUNTIME_H__
#define __RUNTIME_H__
#include "RuntimeTypes.h"

//////////////////////
// NATIVE
#pragma region NATIVE

#include <stdio.h>
#if !defined(_DEBUG) && !defined(NDEBUG)
#define NDEBUG
#endif
#if __CUDACC__
#define __forceinline __forceinline__
#if __CUDA_ARCH__
#define __host_constant__ __constant__
#else
#define __host_constant__
#endif
#include "Runtime.cu.h"
#else
#define __host_constant__
#include <string.h>
#include <malloc.h>
#include "Runtime.cpu.h"
#endif

#ifndef _API
#define _API extern
#endif

#if defined(_GPU) || defined(_SENTINEL)
#define OS_MAP 1
#define OMIT_AUTOINIT
#else
#define OS_MAP 0
#endif

#if __CUDACC__
#define OS_WIN 0
#define OS_UNIX 0
#define OS_GPU 1
#elif defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN 1
#define OS_UNIX 0
#define OS_GPU 0
#else
#define OS_WIN 0
#define OS_UNIX 1
#define OS_GPU 0
#endif

#pragma endregion

//////////////////////
// LIMITS
#pragma region LIMITS

// The maximum length of a TEXT or BLOB in bytes.   This also limits the size of a row in a table or index.
#ifndef CORE_MAX_LENGTH
#define CORE_MAX_LENGTH 1000000000 // The hard limit is the ability of a 32-bit signed integer to count the size: 2^31-1 or 2147483647.
#endif

// Macros to determine whether the machine is big or little endian, evaluated at runtime.
#if defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(__x86_64) || defined(__x86_64__)
#define TYPE_BIGENDIAN 0
#define TYPE_LITTLEENDIAN 1
#define TEXTENCODE_UTF16NATIVE TEXTENCODE_UTF16LE
#else
__device__ extern unsigned char __one;
#define TYPE_BIGENDIAN (*(char *)(&__one) == 0)
#define TYPE_LITTLEENDIAN (*(char *)(&__one) == 1)
#define TEXTENCODE_UTF16NATIVE (TYPE_BIGENDIAN ? TEXTENCODE_UTF16BE : TEXTENCODE_UTF16LE)
#endif

#pragma endregion

//////////////////////
// WSD
#pragma region WSD

// When OMIT_WSD is defined, it means that the target platform does not support Writable Static Data (WSD) such as global and static variables.
// All variables must either be on the stack or dynamically allocated from the heap.  When WSD is unsupported, the variable declarations scattered
// throughout the SQLite code must become constants instead.  The _WSD macro is used for this purpose.  And instead of referencing the variable
// directly, we use its constant as a key to lookup the run-time allocated buffer that holds real variable.  The constant is also the initializer
// for the run-time allocated buffer.
//
// In the usual case where WSD is supported, the _WSD and _GLOBAL macros become no-ops and have zero performance impact.
#ifdef OMIT_WSD
int __wsdinit(int n, int j);
void *__wsdfind(void *k, int l);
#define _WSD const
#define _GLOBAL(t, v) (*(t*)__wsdfind((void *)&(v), sizeof(v)))
#else
#define _WSD
#define _GLOBAL(t, v) v
#endif
#define UNUSED_PARAMETER(x) (void)(x)
#define UNUSED_PARAMETER2(x,y) (void)(x),(void)(y)

#pragma endregion

//////////////////////
// UTF
#pragma region UTF

#define _strskiputf8(z) { if ((*(z++)) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }
//template <typename T> __device__ inline void _strskiputf8(const T *z)
//{
//	if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; }
//}
extern "C" __device__ unsigned int _utf8read(const unsigned char **z);
extern "C" __device__ int _utf8charlength(const char *z, int bytes);
#if _DEBUG
extern "C" __device__ int _utf8to8(unsigned char *z);
#endif
#ifndef OMIT_UTF16
extern "C" __device__ int _utf16bytelength(const void *z, int chars);
#ifdef _TEST
extern "C" __device__ void _runtime_utfselftest();
#endif
#endif

#ifdef _UNICODE
#define char_t unsigned short
#define MAX_CHAR 0xFFFF
#define _L(c) L##c
//#define _isprint iswprint
//#define _strlen wcslen
//#define _printf wprintf
#else
#define char_t char
#define MAX_CHAR 0xFF
#define _L(c) (c) 
//#define _isprint isprint
//#define _strlen strlen
//#define _printf printf
#endif

#pragma endregion

//////////////////////
// CONVERT
#pragma region CONVERT

enum TEXTENCODE : uint8
{
	TEXTENCODE_UTF8 = 1,
	TEXTENCODE_UTF16LE = 2,
	TEXTENCODE_UTF16BE = 3,
	TEXTENCODE_UTF16 = 4, // Use native byte order
	TEXTENCODE_ANY = 5, // sqlite3_create_function only
	TEXTENCODE_UTF16_ALIGNED = 8, // sqlite3_create_collation only
};
__device__ __forceinline void operator|=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a | b); }
__device__ __forceinline void operator&=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a & b); }

#define _convert_getvarint32(A,B) \
	(uint8)((*(A)<(uint8)0x80)?((B)=(uint32)*(A)),1:\
	_convert_getvarint32_((A),(uint32 *)&(B)))
#define _convert_putvarint32(A,B) \
	(uint8)(((uint32)(B)<(uint32)0x80)?(*(A)=(unsigned char)(B)),1:\
	_convert_putvarint32_((A),(B)))

#pragma region Varint
extern "C" __device__ int _convert_putvarint(unsigned char *p, uint64 v);
extern "C" __device__ int _convert_putvarint32_(unsigned char *p, uint32 v);
extern "C" __device__ uint8 _convert_getvarint(const unsigned char *p, uint64 *v);
extern "C" __device__ uint8 _convert_getvarint32_(const unsigned char *p, uint32 *v);
extern "C" __device__ int _convert_getvarintLength(uint64 v);
#pragma endregion
#pragma region AtoX
extern "C" __device__ bool _atof(const char *z, double *out, int length, TEXTENCODE encode);
extern "C" __device__ int __atoi64(const char *z, int64 *out, int length, TEXTENCODE encode);
extern "C" __device__ bool _atoi(const char *z, int *out);
__device__ __forceinline int _atoi(const char *z) { int out = 0; if (z) _atoi(z, &out); return out; }
#pragma endregion

extern "C" __device__ inline uint16 _convert_get2nz(const uint8 *p) { return ((( (int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
extern "C" __device__ inline uint16 _convert_get2(const uint8 *p) { return (p[0]<<8) | p[1]; }
extern "C" __device__ inline void _convert_put2(unsigned char *p, uint32 v)
{
	p[0] = (uint8)(v>>8);
	p[1] = (uint8)v;
}
extern "C" __device__ inline uint32 _convert_get4(const uint8 *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
extern "C" __device__ inline void _convert_put4(unsigned char *p, uint32 v)
{
	p[0] = (uint8)(v>>24);
	p[1] = (uint8)(v>>16);
	p[2] = (uint8)(v>>8);
	p[3] = (uint8)v;
}

#pragma region From: Pragma_c
extern "C" __device__ uint8 __atolevel(const char *z, int omitFull, uint8 dflt);
extern "C" __device__ bool __atob(const char *z, uint8 dflt);
#pragma endregion

#pragma endregion

//////////////////////
// HASH
#pragma region HASH

struct HashElem
{
	HashElem *Next, *Prev;       // Next and previous elements in the table
	void *Data;                  // Data associated with this element
	const char *Key; int KeyLength;  // Key associated with this element
};

struct Hash
{
	unsigned int TableSize;     // Number of buckets in the hash table
	unsigned int Count;			// Number of entries in this table
	HashElem *First;			// The first element of the array
	struct HTable
	{              
		int Count;              // Number of entries with this hash
		HashElem *Chain;        // Pointer to first entry with this hash
	} *Table; // the hash table

	__device__ Hash();
	__device__ void Init();
	__device__ void *Insert(const char *key, int keyLength, void *data);
	__device__ void *Find(const char *key, int keyLength);
	__device__ void Clear();
};

#pragma endregion

//////////////////////
// MATH
#pragma region MATH

extern "C" __device__ bool _math_add(int64 *aRef, int64 b);
extern "C" __device__ bool _math_sub(int64 *aRef, int64 b);
extern "C" __device__ bool _math_mul(int64 *aRef, int64 b);
//__device__ int _math_abs(int x);
extern "C" __device__ inline int _math_abs(int x)
{
	if (x >= 0) return x;
	if (x == (int)0x8000000) return 0x7fffffff;
	return -x;
}

#pragma endregion

//////////////////////
// MUTEX
#pragma region MUTEX

#if !defined(THREADSAFE)
#if defined(__THREADSAFE__)
#define THREADSAFE __THREADSAFE__
#else
#define THREADSAFE 0 // IMP: R-07272-22309
#endif
#endif

// Figure out what version of the code to use.  The choices are
//   MUTEX_OMIT         No mutex logic.  Not even stubs.  The mutexes implemention cannot be overridden at start-time.
//   MUTEX_NOOP         For single-threaded applications.  No mutual exclusion is provided.  But this implementation can be overridden at start-time.
//   MUTEX_PTHREADS     For multi-threaded applications on Unix.
//   MUTEX_W32          For multi-threaded applications on Win32.
#if THREADSAFE == 0
#define MUTEX_OMIT
#else
#if OS_GPU
#define MUTEX_NOOP
#elif OS_UNIX
#define MUTEX_PTHREADS
#elif OS_WIN
#define MUTEX_WIN
#else
#define MUTEX_NOOP
#endif
#endif

enum MUTEX : unsigned char
{
	MUTEX_FAST = 0,
	MUTEX_RECURSIVE = 1,
	MUTEX_STATIC_MASTER = 2,
	MUTEX_STATIC_MEM = 3,  // sqlite3_malloc()
	MUTEX_STATIC_OPEN = 4,  // sqlite3BtreeOpen()
	MUTEX_STATIC_PRNG = 5,  // sqlite3_random()
	MUTEX_STATIC_LRU = 6,   // lru page list
	MUTEX_STATIC_PMEM = 7, // sqlite3PageMalloc()
};

#ifdef MUTEX_OMIT
typedef void *MutexEx;
#define _mutex_held(X) ((void)(X), 1)
#define _mutex_notheld(X) ((void)(X), 1)
#define _mutex_init() 0
#define _mutex_shutdown()
#define _mutex_alloc(X) ((MutexEx)1)
#define _mutex_enter(X)
#define MutexEx_TryEnter(X) 0
#define _mutex_leave(X)
#define _mutex_free(X)
#define MUTEX_LOGIC(X)
#else
struct _mutex_obj;
typedef _mutex_obj *MutexEx;
#ifdef _DEBUG
extern "C" __device__ bool _mutex_held(MutexEx p);
extern "C" __device__ bool _mutex_notheld(MutexEx p);
#endif
extern "C" __device__ int _mutex_init();
extern "C" __device__ void _mutex_shutdown();
extern "C" __device__ MutexEx _mutex_alloc(MUTEX id);
extern "C" __device__ void _mutex_free(MutexEx p);
extern "C" __device__ void _mutex_enter(MutexEx p);
extern "C" __device__ bool _mutex_tryenter(MutexEx p);
extern "C" __device__ void _mutex_leave(MutexEx p);
#define MUTEX_LOGIC(X) X
#endif

#pragma endregion

//////////////////////
// STATUS
#pragma region STATUS

enum STATUS : unsigned char
{
	STATUS_MEMORY_USED = 0,
	STATUS_PAGECACHE_USED = 1,
	STATUS_PAGECACHE_OVERFLOW = 2,
	STATUS_SCRATCH_USED = 3,
	STATUS_SCRATCH_OVERFLOW = 4,
	STATUS_MALLOC_SIZE = 5,
	STATUS_PARSER_STACK = 6,
	STATUS_PAGECACHE_SIZE = 7,
	STATUS_SCRATCH_SIZE = 8,
	STATUS_MALLOC_COUNT = 9,
};

extern "C" __device__ int _status_value(STATUS op);
extern "C" __device__ void _status_add(STATUS op, int n);
extern "C" __device__ void _status_set(STATUS op, int x);
extern "C" __device__ bool _status(STATUS op, int *current, int *highwater, bool resetFlag);

#pragma endregion

//////////////////////
// TAGBASE
#pragma region TAGBASE

class TextBuilder;
class TagBase
{
public:
	struct RuntimeStatics
	{
		bool CoreMutex;			// True to enable core mutexing
		bool FullMutex;			// True to enable full mutexing
		void (*AppendFormat[2])(TextBuilder *b, va_list &args); // Formatter
		//
		bool Memstat;						// True to enable memory status
		bool RuntimeMutex;					// True to enable core mutexing
		size_t LookasideSize;				// Default lookaside buffer size
		int Lookasides;						// Default lookaside buffer count
		void *Scratch;						// Scratch memory
		size_t ScratchSize;					// Size of each scratch buffer
		int Scratchs;						// Number of scratch buffers
		//Main::void *Page;					// Page cache memory
		//Main::int PageSize;				// Size of each page in pPage[]
		//Main::int Pages;					// Number of pages in pPage[]
		//Main::int MaxParserStack;			// maximum depth of the parser stack
	};

	struct LookasideSlot
	{
		LookasideSlot *Next;    // Next buffer in the list of free buffers
	};

	struct Lookaside
	{
		unsigned short Size;    // Size of each buffer in bytes
		bool Enabled;           // False to disable new lookaside allocations
		bool Malloced;          // True if pStart obtained from sqlite3_malloc()
		int Outs;               // Number of buffers currently checked out
		int MaxOuts;            // Highwater mark for nOut
		int Stats[3];			// 0: hits.  1: size misses.  2: full misses
		LookasideSlot *Free;	// List of available buffers
		void *Start;			// First byte of available memory space
		void *End;				// First byte past end of available space
	};

	MutexEx Mutex;			// Connection mutex 
	bool MallocFailed;		// True if we have seen a malloc failure
	int ErrCode;			// Most recent error code (RC_*)
	int ErrMask;			// & result codes with this before returning
	Lookaside Lookaside;	// Lookaside malloc configuration
	int *BytesFreed;		// If not NULL, increment this in DbFree()

	__device__ bool SetupLookaside(void *buf, int size, int count);
};

__device__ extern _WSD TagBase::RuntimeStatics g_RuntimeStatics;
#define TagBase_RuntimeStatics _GLOBAL(TagBase::RuntimeStatics, g_RuntimeStatics)


#pragma endregion

//////////////////////
// FUNC
#pragma region FUNC

#define __toupper(x) ((x)&~(__curtCtypeMap[(unsigned char)(x)]&0x20))
#define _isupper(x) (((x)&~(__curtCtypeMap[(unsigned char)(x)]&0x20))==x)
#define _isspace(x) ((__curtCtypeMap[(unsigned char)(x)]&0x01)!=0)
#define _isalnum(x) ((__curtCtypeMap[(unsigned char)(x)]&0x06)!=0)
#define _isalpha(x) ((__curtCtypeMap[(unsigned char)(x)]&0x02)!=0)
#define _isdigit(x) ((__curtCtypeMap[(unsigned char)(x)]&0x04)!=0)
#define _isxdigit(x) ((__curtCtypeMap[(unsigned char)(x)]&0x08)!=0)
#define _isidchar(x) ((__curtCtypeMap[(unsigned char)(x)]&0x46)!=0)
#define __tolower(x) (__curtUpperToLower[(unsigned char)(x)])
#define _islower(x) (__curtUpperToLower[(unsigned char)(x)]==x)
#define _ispoweroftwo(x) (((x)&((x)-1))==0)
__device__ inline static bool _isalpha2(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

#define _isprint(x) ((unsigned char)c>0x1f&&(unsigned char)c!=0x7f)
#define _iscntrl(x) ((unsigned char)c<=0x1f||(unsigned char)c==0x7f)

// array
template <typename T> struct array_t { int length; T *data; __device__ __forceinline array_t() { data = nullptr; length = 0; } __device__ __forceinline array_t(T *a) { data = a; length = 0; } __device__ __forceinline array_t(T *a, int b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; __device__ __forceinline array_t2() { data = nullptr; length = 0; } __device__ __forceinline array_t2(T *a) { data = a; length = 0; } __device__ __forceinline array_t2(T *a, size_t b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T, size_t size> struct array_t3 { TLength length; T data[size]; __forceinline array_t3() { length = 0; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
#define _lengthof(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// strcpy
template <typename T> __device__ __forceinline void _strcpy(const T *__restrict__ dest, const T *__restrict__ src)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	while (*b) { *a++ = *b++; } *a = *b;
}

// strncpy
template <typename T> __device__ __forceinline void _strncpy(T *__restrict__ dest, const T *__restrict__ src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	size_t i = 0;
	for (; i < length && *b; ++i, ++a, ++b)
		*a = *b;
	for (; i < length; ++i, ++a, ++b)
		*a = 0;
}
template <typename T> __device__ __forceinline void _strncpy(T *__restrict__ dest, T *__restrict__ src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	size_t i = 0;
	for (; i < length && *b; ++i, ++a, ++b)
		*a = *b;
	for (; i < length; ++i, ++a, ++b)
		*a = 0;
}

// strchr
template <typename T> __device__ __forceinline const T *_strchr(const T *src, char character)
{
	register unsigned char *a, b;
	a = (unsigned char *)src;
	b = (unsigned char)__curtUpperToLower[character];
	while (*a != 0 && __curtUpperToLower[*a] != b) { a++; }
	return (const T *)*a;
}

// strstr
template <typename T> __device__ __forceinline const T *_strstr(const T *__restrict__ src, const T *__restrict__ str)
{
	return nullptr;
	//http://articles.leetcode.com/2010/10/implement-strstr-to-find-substring-in.html
	//register unsigned char *a, b;
	//a = (unsigned char *)src;
	//b = (unsigned char)__curtUpperToLower[character];
	//while (*a != 0 && __curtUpperToLower[*a] != b) { a++; }
	//return (const T *)*a;
}

// strcmp
template <typename T> __device__ __forceinline int _strcmp(const T *__restrict__ left, const T *__restrict__ right)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

// strncmp
#undef _fstrncmp
template <typename T> __device__ __forceinline int _strncmp(const T *__restrict__ left, const T *__restrict__ right, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b]);
}
#define _fstrncmp(x, y) (__tolower(*(unsigned char *)(x))==__tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// memcpy
#if __CUDACC__
#define _memcpy(dest, src, length) if (length) memcpy(dest, src, length)
#else
#define _memcpy(dest, src, length) memcpy(dest, src, length)
#endif
#if 0
template <typename T> __device__ __forceinline void _memcpy(T *__restrict__ dest, const T *__restrict__ src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}
template <typename T> __device__ __forceinline void _memcpy(T *__restrict__ dest, T *__restrict__ src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}
#endif

// memset
#if __CUDACC__
#define _memset(dest, value, length) if (length) memset(dest, value, length)
#else
#define _memset(dest, value, length) memset(dest, value, length)
#endif
#if 0
template <typename T> __device__ __forceinline void _memset(T *dest, const char value, size_t length)
{
	register unsigned char *a;
	a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}
#endif

// memchr
template <typename T> __device__ __forceinline const T *_memchr(const T *src, char character)
{
	register unsigned char *a, b;
	a = (unsigned char *)src;
	b = (unsigned char)character;
	while (*a != 0 && *a != b) { a++; }
	return (const T *)*a;
}

// memcmp
template <typename T, typename Y> __device__ __forceinline int _memcmp(T *__restrict__ left, Y *__restrict__ right, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (--length > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// memmove
template <typename T, typename Y> __device__ __forceinline void _memmove(T *__restrict__ left, Y *__restrict__ right, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	if (a == b) return; // No need to do that thing.
	if (a < b && b < a + length) // Check for destructive overlap.
	{
		a += length; b += length; // Destructive overlap ...
		while (length-- > 0) { *--a= *--b; } // have to copy backwards.
		return;
	}
	while (length-- > 0) { *a++ = *b++; } // Do an ascending copy.
}

// strlen30
__device__ __forceinline int _strlen(const char *z)
{
	register const char *z2 = z;
	if (z == nullptr) return 0;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}

// hextobyte
__device__ __forceinline unsigned char _hextobyte(char h)
{
	_assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
	return (unsigned char)((h + 9*(1&(h>>6))) & 0xf);
}
#ifndef OMIT_BLOB_LITERAL
__device__ void *_taghextoblob(TagBase *tag, const char *z, size_t size);
#endif

#ifndef OMIT_FLOATING_POINT
__device__ inline bool _isnan(double x)
{
#if !defined(HAVE_ISNAN)
	// Systems that support the isnan() library function should probably make use of it by compiling with -DHAVE_ISNAN.  But we have
	// found that many systems do not have a working isnan() function so this implementation is provided as an alternative.
	//
	// This NaN test sometimes fails if compiled on GCC with -ffast-math. On the other hand, the use of -ffast-math comes with the following
	// warning:
	//
	//      This option [-ffast-math] should never be turned on by any -O option since it can result in incorrect output for programs
	//      which depend on an exact implementation of IEEE or ISO rules/specifications for math functions.
	//
	// Under MSVC, this NaN test may fail if compiled with a floating-point precision mode other than /fp:precise.  From the MSDN 
	// documentation:
	//
	//      The compiler [with /fp:precise] will properly handle comparisons involving NaN. For example, x != x evaluates to true if x is NaN 
#ifdef __FAST_MATH__
#error Runtime will not work correctly with the -ffast-math option of GCC.
#endif
	volatile double y = x;
	volatile double z = y;
	return (y != z);
#else
	return isnan(x);
#endif
}
#endif

#pragma endregion

//////////////////////
// MEMORY ALLOCATION
#pragma region MEMORY ALLOCATION

#define _ROUNDT(t, x)	(((x)+sizeof(t)-1)&~(sizeof(t)-1))
#define _ROUND8(x)		(((x)+7)&~7)
#define _ROUNDDOWN8(x)	((x)&~7)
#ifdef BYTEALIGNED4
#define _HASALIGNMENT8(X) ((((char *)(X) - (char *)0)&3) == 0)
#else
#define _HASALIGNMENT8(X) ((((char *)(X) - (char *)0)&7) == 0)
#endif

enum MEMTYPE : unsigned char
{
	MEMTYPE_HEAP = 0x01,         // General heap allocations
	MEMTYPE_LOOKASIDE = 0x02,    // Might have been lookaside memory
	MEMTYPE_LRATCH = 0x04,      // Scratch allocations
	MEMTYPE_PCACHE = 0x08,       // Page cache allocations
	MEMTYPE_DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
};

#if MEMDEBUG
extern "C" __device__ inline static void _memdbg_settype(void *p, MEMTYPE memType) { }
extern "C" __device__ inline static bool _memdbg_hastype(void *p, MEMTYPE memType) { return true; }
extern "C" __device__ inline static bool _memdbg_nottype(void *p, MEMTYPE memType) { return true; }
#else
#define _memdbg_settype(X,Y) // no-op
#define _memdbg_hastype(X,Y) true
#define _memdbg_nottype(X,Y) true
#endif

// BenignMallocHooks
#ifndef OMIT_BUILTIN_TEST
extern "C" __device__ void _benignalloc_hook(void (*benignBegin)(), void (*benignEnd)());
extern "C" __device__ void _benignalloc_begin();
extern "C" __device__ void _benignalloc_end();
#else
#define _benignalloc_begin()
#define _benignalloc_end()
#endif
//
extern "C" __device__ void *__allocsystem_alloc(size_t size);
extern "C" __device__ void __allocsystem_free(void *prior);
extern "C" __device__ void *__allocsystem_realloc(void *prior, size_t size);
extern "C" __device__ size_t __allocsystem_size(void *prior);
extern "C" __device__ size_t __allocsystem_roundup(size_t size);
extern "C" __device__ int __allocsystem_init(void *p);
extern "C" __device__ void __allocsystem_shutdown(void *p);
//
//__device__ void __alloc_setmemoryalarm(int (*callback)(void*,long long,int), void *arg, long long threshold);
//__device__ long long __alloc_softheaplimit64(long long n);
//__device__ void __alloc_softheaplimit(int n);
extern "C" __device__ int _alloc_init();
extern "C" __device__ bool _alloc_heapnearlyfull();
extern "C" __device__ void _alloc_shutdown();
//__device__ long long __alloc_memoryused();
//__device__ long long __alloc_memoryhighwater(bool resetFlag);
extern "C" __device__ void *_alloc(size_t size);
extern "C" __device__ void *_scratchalloc(size_t size);
extern "C" __device__ void _scratchfree(void *p);
extern "C" __device__ size_t _allocsize(void *p);
extern "C" __device__ size_t _tagallocsize(TagBase *tag, void *p);
extern "C" __device__ void _free(void *p);
extern "C" __device__ void _tagfree(TagBase *tag, void *p);
extern "C" __device__ void *_realloc(void *old, size_t newSize);
extern "C" __device__ void *_allocZero(size_t size);
extern "C" __device__ void *_tagallocZero(TagBase *tag, size_t size);
extern "C" __device__ void *_tagalloc(TagBase *tag, size_t size);
extern "C" __device__ void *_tagrealloc(TagBase *tag, void *old, size_t size);
//__device__ void *_tagrealloc_or_free(TagBase *tag, void *old, size_t newSize);
extern "C" __device__ __forceinline void *_tagrealloc_or_free(TagBase *tag, void *old, size_t newSize)
{
	void *p = _tagrealloc(tag, old, newSize);
	if (!p) _tagfree(tag, old);
	return p;
}

//__device__ char *_tagstrdup(TagBase *tag, const char *z);
extern "C" __device__ __forceinline char *_tagstrdup(TagBase *tag, const char *z)
{
	if (z == nullptr) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagalloc(tag, (int)n);
	if (newZ) _memcpy(newZ, (char *)z, n);
	return newZ;
}
//__device__ char *_tagstrndup(TagBase *tag, const char *z, int n);
extern "C" __device__ __forceinline char *_tagstrndup(TagBase *tag, const char *z, int n)
{
	if (z == nullptr) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagalloc(tag, n + 1);
	if (newZ) { _memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}

// On systems with ample stack space and that support alloca(), make use of alloca() to obtain space for large automatic objects.  By default,
// obtain space from malloc().
//
// The alloca() routine never returns NULL.  This will cause code paths that deal with sqlite3StackAlloc() failures to be unreachable.
#ifdef USE_ALLOCA
#define _stackalloc(D,N) alloca(N)
#define _stackallocZero(D,N) _memset(alloca(N), 0, N)
#define _stackfree(D,P)       
#else
#define _stackalloc(D,N) _tagalloc(D,N)
#define _stackallocZero(D,N) _tagallocZero(D,N)
#define _stackfree(D,P) _tagfree(D,P)
#endif

typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)_allocsize)

#pragma endregion

//////////////////////
// BITVEC
#pragma region BITVEC

#define BITVEC_SZ 512
#define BITVEC_USIZE (((BITVEC_SZ - (3 * sizeof(uint32))) / sizeof(Bitvec *)) * sizeof(Bitvec *))
#define BITVEC_SZELEM 8
#define BITVEC_NELEM (BITVEC_USIZE / sizeof(uint8))
#define BITVEC_NBIT (BITVEC_NELEM * BITVEC_SZELEM)
#define BITVEC_NINT (BITVEC_USIZE / sizeof(uint))
#define BITVEC_MXHASH (BITVEC_NINT / 2)
#define BITVEC_HASH(X) (((X) * 1) % BITVEC_NINT)
#define BITVEC_NPTR (BITVEC_USIZE / sizeof(Bitvec *))

struct Bitvec
{
private:
	uint32 _size;      // Maximum bit index.  Max iSize is 4,294,967,296.
	uint32 _set;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
	uint32 _divisor;   // Number of bits handled by each apSub[] entry.
	// Should >=0 for apSub element. */
	// Max iDivisor is max(uint32) / BITVEC_NPTR + 1.
	// For a BITVEC_SZ of 512, this would be 34,359,739.
	union
	{
		uint8 Bitmap[BITVEC_NELEM]; // Bitmap representation
		uint32 Hash[BITVEC_NINT];	// Hash table representation
		Bitvec *Sub[BITVEC_NPTR];	// Recursive representation
	} u;
public:
	__device__ static Bitvec *New(uint32 size);
	__device__ bool Get(uint32 index);
	__device__ bool Set(uint32 index);
	__device__ void Clear(uint32 index, void *buffer);
	__device__ static inline void Destroy(Bitvec *p)
	{
		if (!p)
			return;
		if (p->_divisor)
			for (unsigned int index = 0; index < BITVEC_NPTR; index++)
				Destroy(p->u.Sub[index]);
		_free(p);
	}
	__device__ inline uint32 get_Length() { return _size; }
};

#pragma endregion

//////////////////////
// SENTINEL
#pragma region SENTINEL
#if OS_MAP

struct RuntimeSentinelMessage
{
	char OP;
	void (*Prepare)(void*,char*,int);
	__device__ RuntimeSentinelMessage(char op, void (*prepare)(void*,char*,int))
		: OP(op), Prepare(prepare) { }
public:
};
#define RUNTIMESENTINELPREPARE(P) ((void (*)(void*,char*,int))&P)

typedef struct
{
	volatile int Status;
	int Length;
	char Data[1024];
} RuntimeSentinelCommand;

typedef struct
{
	volatile unsigned int AddId;
	volatile unsigned int RunId;
	RuntimeSentinelCommand Commands[1];
} RuntimeSentinelMap;

typedef struct RuntimeSentinelExecutor
{
	RuntimeSentinelExecutor *Next;
	const char *Name;
	bool (*Executor)(void*,RuntimeSentinelMessage*,int);
	void *Tag;
} RuntimeSentinelExecutor;
#define RUNTIMESENTINELEXECUTOR(E) ((bool (*)(void*,RuntimeSentinelMessage*,int))&E)

typedef struct RuntimeSentinelContext
{
	RuntimeSentinelMap *Map;
	RuntimeSentinelExecutor *List;
} RuntimeSentinelContext;

extern __constant__ RuntimeSentinelMap *_runtimeSentinelMap;
struct RuntimeSentinel
{
public:
	static void Initialize(RuntimeSentinelExecutor *executor = nullptr);
	static void Shutdown();
	__device__ static void SetDeviceMap(RuntimeSentinelMap *map);
	__device__ static RuntimeSentinelMap *GetDeviceMap();
	//
	static RuntimeSentinelExecutor *FindExecutor(const char *name);
	static void RegisterExecutor(RuntimeSentinelExecutor *exec, bool _default = false);
	static void UnregisterExecutor(RuntimeSentinelExecutor *exec);
	//
	__device__ static void Send(void *msg, int msgLength);
};

namespace Messages
{
	struct Stdio_fprintf
	{
		__device__ inline static void Prepare(Stdio_fprintf *t, char *data, int length)
		{
			int formatLength = (t->Format ? _strlen(t->Format) + 1 : 0);
			char *format = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(format, t->Format, formatLength);
			t->Format = format;
		}
		RuntimeSentinelMessage Base;
		FILE *File; const char *Format;
		__device__ Stdio_fprintf(FILE *file, const char *format)
			: Base(1, RUNTIMESENTINELPREPARE(Prepare)), File(file), Format(format) { RuntimeSentinel::Send(this, sizeof(Stdio_fprintf)); }
		int RC; 
	};

	struct Stdio_fopen
	{
		__device__ inline static void Prepare(Stdio_fopen *t, char *data, int length)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			int modeLength = (t->Mode ? _strlen(t->Mode) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			char *mode = (char *)(data += filenameLength);
			_memcpy(filename, t->Filename, filenameLength);
			_memcpy(mode, t->Mode, modeLength);
			t->Filename = filename;
			t->Mode = mode;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; const char *Mode;
		__device__ Stdio_fopen(const char *filename, const char *mode)
			: Base(2, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), Mode(mode) { RuntimeSentinel::Send(this, sizeof(Stdio_fopen)); }
		FILE *RC; 
	};

	struct Stdio_fflush
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fflush(FILE *file)
			: Base(3, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fflush)); }
		int RC; 
	};

	struct Stdio_fclose
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fclose(FILE *file)
			: Base(4, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fclose)); }
		int RC; 
	};

	struct Stdio_fputc
	{
		RuntimeSentinelMessage Base;
		int Ch; FILE *File;
		__device__ Stdio_fputc(int ch, FILE *file)
			: Base(5, nullptr), Ch(ch), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fputc)); }
		int RC; 
	};

	struct Stdio_fputs
	{
		__device__ inline static void Prepare(Stdio_fputs *t, char *data, int length)
		{
			int strLength = (t->Str ? _strlen(t->Str) + 1 : 0);
			char *str = (char *)(data += _ROUND8(sizeof(*t)));
			_memcpy(str, t->Str, strLength);
			t->Str = str;
		}
		RuntimeSentinelMessage Base;
		const char *Str; FILE *File;
		__device__ Stdio_fputs(const char *str, FILE *file)
			: Base(6, RUNTIMESENTINELPREPARE(Prepare)), Str(str), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fputs)); }
		int RC; 
	};
}

#endif
#pragma endregion

//////////////////////
// PRINT
#pragma region PRINT

class TextBuilder
{
public:
	TagBase *Tag;		// Optional database for lookaside.  Can be NULL
	char *Base;			// A base allocation.  Not from malloc.
	char *Text;			// The string collected so far
	int Index;			// Length of the string so far
	int Size;			// Amount of space allocated in zText
	int MaxSize;		// Maximum allowed string length
	bool AllocFailed;  // Becomes true if any memory allocation fails
	unsigned char AllocType; // 0: none,  1: _tagalloc,  2: _alloc
	bool Overflowed;    // Becomes true if string size exceeds limits

	__device__ void AppendSpace(int length);
	__device__ void AppendFormat_(bool useExtended, const char *fmt, va_list &args);
	__device__ void Append(const char *z, int length);
	__device__ char *ToString();
	__device__ void Reset();
	__device__ static void Init(TextBuilder *b, char *text, int capacity, int maxAlloc);
	//
#if __CUDACC__
	__device__ __forceinline void AppendFormat(const char *fmt) { va_list args; va_start(args); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); AppendFormat_(true, fmt, args); va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); AppendFormat_(true, fmt, args); va_end(args); }
#else
	__device__ inline void AppendFormat(const char *fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		AppendFormat_(true, fmt, args);
		va_end(args);
	}
#endif
};

#pragma endregion

//////////////////////
// SNPRINTF
#pragma region SNPRINTF

extern "C" __device__ char *__vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list *args);
#if __CUDACC__
__device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt) { va_list args; va_start(args); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); char *z = __vsnprintf(buf, bufLen, fmt, &args); va_end(args); return z; }
#else
extern "C" __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = __vsnprintf(buf, bufLen, fmt, &args);
	va_end(args);
	return z;
}
#endif
#define _sprintf(buf, fmt, ...) __snprintf(buf, sizeof(buf), fmt, __VA_ARGS__)

#pragma endregion

//////////////////////
// FPRINTF
#pragma region FPRINTF

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 200
#undef stdin
#undef stdout
#undef stderr
extern __constant__ FILE _stdin_file;
extern __constant__ FILE _stdout_file;
extern __constant__ FILE _stderr_file;
#define stdin &_stdin_file
#define stdout &_stdout_file
#define stderr &_stderr_file
#endif

#if 0 && OS_MAP
#define _fprintf(f, ...) _fprintf_(f, _mprintf("%s", __VA_ARGS__))
extern "C" __device__ inline int _fprintf_(FILE *f, const char *v) { Messages::Stdio_fprintf msg(f, v); return msg.RC; }
extern "C" __device__ inline FILE *_fopen(const char *f, const char *m) { Messages::Stdio_fopen msg(f, m); return msg.RC; }
extern "C" __device__ inline int _fflush(FILE *f) { Messages::Stdio_fflush msg(f); return msg.RC; }
extern "C" __device__ inline int _fclose(FILE *f) { Messages::Stdio_fclose msg(f); return msg.RC; }
extern "C" __device__ inline int _fputc(int c, FILE *f) { Messages::Stdio_fputc msg(c, f); return msg.RC; }
extern "C" __device__ inline int _fputs(const char *s, FILE *f) { Messages::Stdio_fputs msg(s, f); return msg.RC; }
#else
#if __CUDACC__
#define _fprintf(f, ...) printf(__VA_ARGS__)
#define _fopen(f, m) 0
#define _fflush(f)
#define _fclose(f)
#define _fputc(c, f) printf("%c", c)
#define _fputs(s, f) printf("%s\n", c)
#else
#define _fprintf(f, ...) fprintf(f, __VA_ARGS__)
#define _fopen(f, m) fopen(f, m)
#define _fflush(f) fflush(f)
#define _fclose(f) fclose(f)
#define _fputc(c, f) fputc(c, f)
#define _fputs(s, f) fputs(s, f)
#endif
#endif

#pragma endregion

//////////////////////
// MPRINTF
#pragma region MPRINTF

extern "C" __device__ char *_vmprintf(const char *fmt, va_list *args);
extern "C" __device__ char *_vmtagprintf(TagBase *tag, const char *fmt, va_list *args);
#if __CUDACC__
__device__ __forceinline char *_mprintf(const char *fmt) { va_list args; va_start(args); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmprintf(fmt, &args); va_end(args); return z; }
//
__device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); return z; }
//
__device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, src); return z; }

__device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); va_end(args); _tagfree(tag, *src); *src = z; }
#else
extern "C" __device__ inline static char *_mprintf(const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmprintf(fmt, &args);
	va_end(args);
	return z;
}
//
extern "C" __device__ inline static char *_mtagprintf(TagBase *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args);
	va_end(args);
	return z;
}
//
extern "C" __device__ inline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args);
	va_end(args);
	_tagfree(tag, src);
	return z;
}
extern "C" __device__ inline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args);
	va_end(args);
	_tagfree(tag, *src);
	*src = z;
}
#endif

#pragma endregion

#endif // __RUNTIME_H__