#ifndef __RUNTIME_H__
#define __RUNTIME_H__
#include "RuntimeTypes.h"

#define OMIT_INLINECONVERT
#define OMIT_INLINEMATH
#define OMIT_INLINEFUNC
#define OMIT_INLINEMEM

//////////////////////
// NATIVE
#pragma region NATIVE

#include <stdio.h>
#pragma warning(disable:4996)
#if !defined(_DEBUG) && !defined(NDEBUG)
#define NDEBUG
#endif

#ifndef __RUNTIMEHOST_H__
typedef struct
{
	void *reserved;
	void *heap;
	char *blocks;
	char *blockStart;
	size_t blockSize;
	size_t blocksLength;
	size_t length;
	void *cudaAssertHandler;
} cudaDeviceHeap;
#endif

//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 100
//#define _GPU
//#endif

#if __CUDACC__
#define __forceinline __forceinline__
#define __host_device__ __host__ __device__
#if __CUDA_ARCH__
#define __host_constant__ __constant__
#else
#define __host_constant__
#endif
#include "Runtime.cu.h"
// panic
#define _exit(v) asm("trap;")
#define _abort() asm("trap;")
#define _panic(fmt, ...) printf(fmt, __VA_ARGS__); asm("trap;")
#else
#define __host_device__
#define __host_constant__
#include <string.h>
#include <malloc.h>
#include "Runtime.cpu.h"
// panic
#define _exit(v) exit(v)
#define _abort() abort()
#define _panic(fmt, ...) printf(fmt, __VA_ARGS__)
#endif

//// Macro to use instead of "void" for arguments that must have type "void *" in ANSI C;  maps them to type "char *" in non-ANSI systems.
//#ifndef VOID
//# ifdef __STDC__
//# define VOID void
//# else
//# define VOID char
//# endif
//#endif
//
//// Miscellaneous declarations (to allow Tcl to be used stand-alone, without the rest of Sprite).
//#ifndef NULL
//#define NULL 0
//#endif

#ifndef _API
#define _API extern
#endif

#define OMIT_AUTOINIT
#if defined(_GPU) || defined(_SENTINEL)
//#pragma message("OS_MAP:HAS_HOSTSENTINEL")
#define OS_MAP 1
#define HAS_HOSTSENTINEL 0 // 1 for IPC
#else
//#pragma message("OS_MAP:HAS_HOSTSENTINEL")
#define OS_MAP 0
#define HAS_HOSTSENTINEL 0 // 1 for IPC
#endif

#if __CUDACC__
//#pragma message("OS_GPU:")
#define OS_WIN 0
#define OS_UNIX 0
#define OS_GPU 1
#elif defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
//#pragma message("OS_WIN:")
#define OS_WIN 1
#define OS_UNIX 0
#define OS_GPU 0
#else
//#pragma message("OS_UNIX:")
#define OS_WIN 0
#define OS_UNIX 1
#define OS_GPU 0
#endif

#ifdef RUNTIME_NAME
#define RUNTIME_NAMEBEGIN namespace RUNTIME_NAME {
#define RUNTIME_NAMEEND }
#else
#define RUNTIME_NAMEBEGIN
#define RUNTIME_NAMEEND
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
// GANGING
#pragma region GANGING

#if defined(GANGING) && (!defined(__CUDACC__) || defined(SENTINEL))
#undef GANGING
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
//template <typename T> __device__ inline void _strskiputf8(const T *z) { if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }
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

// No utf-8 support. 1 byte = 1 char
#define utf8_strlen(S, B) ((B) < 0 ? _strlen(S) : (B))
#define utf8_tounicode(S, CP) (*(CP) = (unsigned char)*(S), 1)
#define utf8_getchars(CP, C) (*(CP) = (C), 1)
#define utf8_upper(C) __toupper(C)
#define utf8_title(C) __toupper(C)
#define utf8_lower(C) __tolower(C)
#define utf8_index(C, I) (I)
#define utf8_charlen(C) 1
#define utf8_prev_len(S, L) 1

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
__device__ __forceinline double _atof(const char *z) { double out = 0; if (z) _atof(z, &out, -1, TEXTENCODE_UTF8); return out; }
extern "C" __device__ int __atoi64(const char *z, int64 *out, int length, TEXTENCODE encode);
extern "C" __device__ bool _atoi(const char *z, int *out);
__device__ __forceinline int _atoi(const char *z) { int out = 0; if (z) _atoi(z, &out); return out; }
#define _itoa(i, b) _itoa64((int64)i, b)
extern "C" __device__ char *_itoa64(int64 i, char *b);
#pragma endregion
#ifndef OMIT_INLINECONVERT
__device__ __forceinline uint16 _convert_get2nz(const uint8 *p) { return ((( (int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
__device__ __forceinline uint16 _convert_get2(const uint8 *p) { return (p[0]<<8) | p[1]; }
__device__ __forceinline void _convert_put2(unsigned char *p, uint32 v)
{
	p[0] = (uint8)(v>>8);
	p[1] = (uint8)v;
}
__device__ __forceinline uint32 _convert_get4(const uint8 *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
__device__ __forceinline void _convert_put4(unsigned char *p, uint32 v)
{
	p[0] = (uint8)(v>>24);
	p[1] = (uint8)(v>>16);
	p[2] = (uint8)(v>>8);
	p[3] = (uint8)v;
}
#else
extern "C" __device__ uint16 _convert_get2nz(const uint8 *p);
extern "C" __device__ uint16 _convert_get2(const uint8 *p);
extern "C" __device__ void _convert_put2(unsigned char *p, uint32 v);
extern "C" __device__ uint32 _convert_get4(const uint8 *p);
extern "C" __device__ void _convert_put4(unsigned char *p, uint32 v);
#endif

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

#ifdef GANGING
// NOT GANG-SAFE
#endif

#pragma endregion

//////////////////////
// MATH
#pragma region MATH

extern "C" __device__ bool _math_add(int64 *aRef, int64 b);
extern "C" __device__ bool _math_sub(int64 *aRef, int64 b);
extern "C" __device__ bool _math_mul(int64 *aRef, int64 b);
#ifndef OMIT_INLINEMATH
__device__ __forceinline int _math_abs(int x)
{
	if (x >= 0) return x;
	if (x == (int)0x8000000) return 0x7fffffff;
	return -x;
}
#else
extern "C" __device__ int _math_abs(int x);
#endif

#pragma endregion

//////////////////////
// MUTEX
#pragma region MUTEX
RUNTIME_NAMEBEGIN

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
#define _mutex_free(X)
#define _mutex_enter(X)
#define _mutex_tryenter(X) 0
#define _mutex_leave(X)
#define MUTEX_LOGIC(X)
#else
struct _mutex_obj;
typedef _mutex_obj *MutexEx;
#ifdef _DEBUG
__device__ bool _mutex_held(MutexEx p);
__device__ bool _mutex_notheld(MutexEx p);
#endif
__device__ int _mutex_init();
__device__ void _mutex_shutdown();
__device__ MutexEx _mutex_alloc(MUTEX id);
__device__ void _mutex_free(MutexEx p);
__device__ void _mutex_enter(MutexEx p);
__device__ bool _mutex_tryenter(MutexEx p);
__device__ void _mutex_leave(MutexEx p);
#define MUTEX_LOGIC(X) X
#endif

#ifdef GANGING
// NOT GANG-SAFE
#endif

RUNTIME_NAMEEND
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
RUNTIME_NAMEBEGIN

	// AllocSystem
struct _mem_methods
{
	void *(*Alloc)(size_t);        // Memory allocation function
	void (*Free)(void*);			// Free a prior allocation
	void *(*Realloc)(void*,size_t); // Resize an allocation
	size_t (*Size)(void*);          // Return the size of an allocation
	size_t (*Roundup)(size_t);      // Round up request size to allocation size
	int (*Init)(void*);				// Initialize the memory allocator
	void (*Shutdown)(void*);		// Deinitialize the memory allocator
	void *AppData;					// Argument to xInit() and xShutdown()
};

__device__ void __allocsystem_setdefault();		// Default mem interface
#define __allocsystem g_RuntimeStatics.AllocSystem

// MutexSystem
struct _mutex_methods
{
	int (*Init)();
	void (*Shutdown)();
	MutexEx (*Alloc)(MUTEX);
	void (*Free)(MutexEx);
	void (*Enter)(MutexEx);
	bool (*TryEnter)(MutexEx);
	void (*Leave)(MutexEx);
	bool (*Held)(MutexEx);
	bool (*NotHeld)(MutexEx);
};
#define __mutexsystem g_RuntimeStatics.MutexSystem

class TextBuilder;
class TagBase
{
public:
	struct RuntimeStatics
	{
		bool CoreMutex;			// True to enable core mutexing
		bool FullMutex;			// True to enable full mutexing
		void (*AppendFormat[2])(TextBuilder *b, _va_list &args); // Formatter
		//
		bool Memstat;						// True to enable memory status
		bool RuntimeMutex;					// True to enable core mutexing
		_mem_methods AllocSystem;			// Low-level mem interface
		_mutex_methods MutexSystem;			// Low-level mutex interface
		size_t LookasideSize;				// Default lookaside buffer size
		int Lookasides;						// Default lookaside buffer count
		void *Scratch;						// Scratch memory
		size_t ScratchSize;					// Size of each scratch buffer
		int Scratchs;						// Number of scratch buffers
		//DataEx::void *Page;					// Page cache memory
		//DataEx::int PageSize;				// Size of each page in pPage[]
		//DataEx::int Pages;					// Number of pages in pPage[]
		//DataEx::int MaxParserStack;			// maximum depth of the parser stack
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

#ifndef OMIT_BLOB_LITERAL
__device__ void *_taghextoblob(TagBase *tag, const char *z, size_t size);
#endif

#ifdef GANGING
// NOT GANG-SAFE
#endif

RUNTIME_NAMEEND
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
	__device__ inline static bool _isalpha2(unsigned char x) { return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z'); }

#define _isprint(x) ((unsigned char)x>0x1f&&(unsigned char)x!=0x7f)
#define _iscntrl(x) ((unsigned char)x<=0x1f||(unsigned char)x==0x7f)

// array
template <typename T> struct array_t { int length; T *data; __device__ __forceinline array_t() { data = nullptr; length = 0; } __device__ __forceinline array_t(T *a) { data = a; length = 0; } __device__ __forceinline array_t(T *a, int b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; __device__ __forceinline array_t2() { data = nullptr; length = 0; } __device__ __forceinline array_t2(T *a) { data = a; length = 0; } __device__ __forceinline array_t2(T *a, size_t b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T, size_t size> struct array_t3 { TLength length; T data[size]; __forceinline array_t3() { length = 0; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
#define _lengthof(symbol) (sizeof(symbol) / sizeof(symbol[0]))

#ifndef OMIT_INLINEFUNC

// strcpy
#if 1
__device__ __forceinline void _strcpy(char *__restrict__ dest, const char *__restrict__ src)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
}
#else
__device__ __forceinline void _strcpy(register char *__restrict__ dest, register const char *__restrict__ src)
{
	register int i = 0;
	while (src[i]); { dest[i++] = src[i]; } dest[i] = src[i];
}
#endif

// strncpy
__device__ __forceinline void _strncpy(char *__restrict__ dest, const char *__restrict__ src, size_t length)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	size_t i = 0;
	for (; i < length && *s; ++i, ++d, ++s)
		*d = *s;
	for (; i < length; ++i, ++d, ++s)
		*d = 0;
}

//strcat
#if 1
__device__ __forceinline void _strcat(char *__restrict__ dest, const char *__restrict__ src)
{
	register unsigned char *d = (unsigned char *)dest;
	while (*d) d++;
	//_strcpy<T>(d, src);
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
}
#else
__device__ char *_strcat(register char *__restrict__ dest, register const char *__restrict__ src)
{
	register int i = 0;
	while (dest[i] != 0) i++;
	_strcpy(dest + i, src);
	return dest;
}
#endif

// strchr
__device__ __forceinline char *_strchr(const char *src, int ch)
{
	register unsigned char *s = (unsigned char *)src;
	register unsigned char l = (unsigned char)__curtUpperToLower[ch];
	while (*s && __curtUpperToLower[*s] != l) { s++; }
	return (char *)(*s ? s : nullptr);
}

// strstr
//http://articles.leetcode.com/2010/10/implement-strstr-to-find-substring-in.html
__device__ __forceinline const char *_strstr(const char *__restrict__ src, const char *__restrict__ str)
{
	if (!*str) return src;
	char *p1 = (char *)src, *p2 = (char *)str;
	char *p1Adv = (char *)src;
	while (*++p2)
		p1Adv++;
	while (*p1Adv)
	{
		char *p1Begin = p1;
		p2 = (char *)str;
		while (*p1 && *p2 && *p1 == *p2)
		{
			p1++;
			p2++;
		}
		if (!*p2)
			return p1Begin;
		p1 = p1Begin + 1;
		p1Adv++;
	}
	return nullptr;
}

// strcmp
__device__ __forceinline int _strcmp(const char *__restrict__ left, const char *__restrict__ right)
{
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
	while (*a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

// strncmp
#undef _fstrncmp
__device__ __forceinline int _strncmp(const char *__restrict__ left, const char *__restrict__ right, int n)
{
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
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
__device__ __forceinline void _memcpy(char *__restrict__ dest, const char *__restrict__ src, size_t length)
{
	a, *b;
	register unsigned char *a = (unsigned char *)dest;
	register unsigned char *b = (unsigned char *)src;
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
__device__ __forceinline void _memset(char *dest, const char value, size_t length)
{
	register unsigned char *a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}
#endif

// memchr
__device__ __forceinline const void *_memchr(const void *src, char ch, size_t length)
{
	if (length != 0) {
		register const unsigned char *p = (const unsigned char *)src;
		do {
			if (*p++ == ch)
				return (const void *)(p - 1);
		} while (--length != 0);
	}
	return nullptr;
	//register unsigned char *a = (unsigned char *)src;
	//register unsigned char b = (unsigned char)ch;
	//while (--length > 0 && *a && *a != b) { a++; }
	//return (const T *)*a;
}

// memcmp
__device__ __forceinline int _memcmp(const void *__restrict__ left, const void *__restrict__ right, size_t length)
{
	if (!length)
		return 0;
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
	while (--length > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// memmove
__device__ __forceinline void _memmove(void *__restrict__ left, const void *__restrict__ right, size_t length)
{
	if (!length)
		return;
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
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
	if (!z) return 0;
	register const char *z2 = z;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}
__device__ __forceinline int _strlen16(const void *z)
{
	if (!z) return 0;
	register const char *z2 = (const char *)z;
	int n;
	for (n = 0; z2[n] || z2[n+1]; n += 2) { }
	return n;
}

// hextobyte
__device__ __forceinline unsigned char _hextobyte(char h)
{
	_assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
	return (unsigned char)((h + 9*(1&(h>>6))) & 0xf);
}

#ifndef OMIT_FLOATING_POINT
__device__ __forceinline bool _isnan(double x)
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

#else

// strcpy
__device__ void _strcpy(char *__restrict__ dest, const char *__restrict__ src);

// strncpy
__device__ void _strncpy(char *__restrict__ dest, const char *__restrict__ src, size_t length);

// strcat
__device__ void _strcat(char *__restrict__ dest, const char *__restrict__ src);

// strchr
__device__ char *_strchr(const char *src, int ch);

// strstr
__device__ const char *_strstr(const char *__restrict__ src, const char *__restrict__ str);

// strcmp
__device__ int _strcmp(const char *__restrict__ left, const char *__restrict__ right);

// strncmp
#undef _fstrncmp
__device__ int _strncmp(const char *__restrict__ left, const char *__restrict__ right, int n);
#define _fstrncmp(x, y) (__tolower(*(unsigned char *)(x))==__tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// memcpy
#if __CUDACC__
#define _memcpy(dest, src, length) if (length) memcpy(dest, src, length)
#else
#define _memcpy(dest, src, length) memcpy(dest, src, length)
#endif

// memset
#if __CUDACC__
#define _memset(dest, value, length) if (length) memset(dest, value, length)
#else
#define _memset(dest, value, length) memset(dest, value, length)
#endif

// memchr
__device__ const void *_memchr(const void *src, char ch, size_t length);

// memcmp
__device__ int _memcmp(const void *__restrict__ left, const void *__restrict__ right, size_t length);

// memmove
__device__ void _memmove(void *__restrict__ left, const void *__restrict__ right, size_t length);

// strlen30
__host__ __device__ int _strlen(const char *z);
__device__ int _strlen16(const void *z);

// hextobyte
__device__ unsigned char _hextobyte(char h);

#ifndef OMIT_FLOATING_POINT
__host__ __device__ bool _isnan(double x);
#endif

#endif

#if __CUDACC__
#ifdef __cplusplus
extern "C++" {
	__device__ __forceinline char *_strchr(char *src, int ch) { return (char *)_strchr((const char *)src, ch); }
	__device__ __forceinline char *_strstr(char *src, const char *str) { return (char *)_strstr((const char *)src, str); }
	__device__ __forceinline void *_memchr(void *src, int ch, size_t length) { return (void *)_memchr((const void *)src, ch, length); }
}
#endif
#endif

#pragma endregion

//////////////////////
// MEMORY ALLOCATION
#pragma region MEMORY ALLOCATION
RUNTIME_NAMEBEGIN

#define MEMORY_ALIGNMENT 4096
#define _ROUNDT(t, x)		(((x)+sizeof(t)-1)&~(sizeof(t)-1))
#define _ROUND8(x)			(((x)+7)&~7)
#define _ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
#define _ROUNDDOWN8(x)		((x)&~7)
#define _ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
#ifdef BYTEALIGNED4
#define _HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define _HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
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
__device__ inline static void _memdbg_settype(void *p, MEMTYPE memType) { }
__device__ inline static bool _memdbg_hastype(void *p, MEMTYPE memType) { return true; }
__device__ inline static bool _memdbg_nottype(void *p, MEMTYPE memType) { return true; }
#else
#define _memdbg_settype(X,Y) // no-op
#define _memdbg_hastype(X,Y) true
#define _memdbg_nottype(X,Y) true
#endif

// BenignMallocHooks
#ifndef OMIT_BUILTIN_TEST
__device__ void _benignalloc_hook(void (*benignBegin)(), void (*benignEnd)());
__device__ void _benignalloc_begin();
__device__ void _benignalloc_end();
#else
#define _benignalloc_begin()
#define _benignalloc_end()
#endif
//
//__device__ void __alloc_setmemoryalarm(int (*callback)(void*,long long,int), void *arg, long long threshold);
//__device__ long long __alloc_softheaplimit64(long long n);
//__device__ void __alloc_softheaplimit(int n);
__device__ int _alloc_initG();
__device__ bool _alloc_heapnearlyfull();
__device__ void _alloc_shutdownG();
__device__ long long __alloc_memoryused();
__device__ long long __alloc_memoryhighwater(bool resetFlag);
__device__ void *_allocG(size_t size);
__device__ void *_scratchallocG(size_t size);
__device__ void _scratchfreeG(void *p);
__device__ size_t _allocsize(void *p);
__device__ size_t _tagallocsize(TagBase *tag, void *p);
__device__ void _freeG(void *p);
__device__ void _tagfreeG(TagBase *tag, void *p);
__device__ void *_reallocG(void *old, size_t newSize);
__device__ void *_allocZeroG(size_t size);
__device__ void *_tagallocZeroG(TagBase *tag, size_t size);
__device__ void *_tagallocG(TagBase *tag, size_t size);
__device__ void *_tagreallocG(TagBase *tag, void *old, size_t size);
__device__ __forceinline void *_tagrealloc_or_freeG(TagBase *tag, void *old, size_t newSize)
{
	void *p = _tagreallocG(tag, old, newSize);
	if (!p) _tagfreeG(tag, old);
	return p;
}

#ifndef OMIT_INLINEMEM
__device__ __forceinline char *__strdupG(const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG((int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ __forceinline char *_strndupG(const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG(n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}

__device__ __forceinline char *_tagstrdupG(TagBase *tag, const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, (int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ __forceinline char *_tagstrndupG(TagBase *tag, const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}
#else
__device__ char *__strdupG(const char *z);
__device__ char *_strndupG(const char *z, int n);
__device__ char *_tagstrdupG(TagBase *tag, const char *z);
__device__ char *_tagstrndupG(TagBase *tag, const char *z, int n);
#endif

// On systems with ample stack space and that support alloca(), make use of alloca() to obtain space for large automatic objects.  By default,
// obtain space from malloc().
//
// The alloca() routine never returns NULL.  This will cause code paths that deal with sqlite3StackAlloc() failures to be unreachable.
#ifdef USE_ALLOCA
#define _stackallocG(D,N) alloca(N)
#define _stackallocZeroG(D,N) _memset(alloca(N), 0, N)
#define _stackfreeG(D,P)       
#else
#define _stackallocG(D,N) _tagallocG(D,N)
#define _stackallocZeroG(D,N) _tagallocZeroG(D,N)
#define _stackfreeG(D,P) _tagfreeG(D,P)
#endif

typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)_allocsize)

#ifdef GANGING
__device__ __forceinline int _alloc_init() { __shared__ int v; if (!threadIdx.x) v = _alloc_initG(); __syncthreads(); return v; }
__device__ __forceinline void _alloc_shutdown() { if (!threadIdx.x) _alloc_shutdownG(); __syncthreads(); }
//
__device__ __forceinline void *_alloc(size_t size) { __shared__ void *v; if (!threadIdx.x) v = _allocG(size); __syncthreads(); return v; }
__device__ __forceinline void *_scratchalloc(size_t size) { __shared__ void *v; if (!threadIdx.x) v = _scratchallocG(size); __syncthreads(); return v; }
__device__ __forceinline void _scratchfree(void *p) { if (!threadIdx.x) _scratchfreeG(p); __syncthreads(); }
__device__ __forceinline void _free(void *p) { if (!threadIdx.x) _freeG(p); __syncthreads(); }
__device__ __forceinline void _tagfree(TagBase *tag, void *p) { if (!threadIdx.x) _tagfreeG(tag, p); __syncthreads(); }
__device__ __forceinline void *_realloc(void *old, size_t newSize) { __shared__ void *v; if (!threadIdx.x) v = _reallocG(old, newSize); __syncthreads(); return v; }
__device__ __forceinline void *_allocZero(size_t size) { __shared__ void *v; if (!threadIdx.x) v = _allocZeroG(size); __syncthreads(); return v; }
__device__ __forceinline void *_tagallocZero(TagBase *tag, size_t size) { __shared__ void *v; if (!threadIdx.x) v = _tagallocZeroG(tag, size); __syncthreads(); return v; }
__device__ __forceinline void *_tagalloc(TagBase *tag, size_t size) { __shared__ void *v; if (!threadIdx.x) v = _tagallocG(tag, size); __syncthreads(); return v; }
__device__ __forceinline void *_tagrealloc(TagBase *tag, void *old, size_t size) { __shared__ void *v; if (!threadIdx.x) v = _tagreallocG(tag, old, size); __syncthreads(); return v; }
__device__ __forceinline void *_tagrealloc_or_free(TagBase *tag, void *old, size_t newSize) { __shared__ void *v; if (!threadIdx.x) v = _tagrealloc_or_freeG(tag, old, newSize); __syncthreads(); return v; }
//
__device__ __forceinline char *__strdup(const char *z) { __shared__ char *v; if (!threadIdx.x) v = __strdupG(z); __syncthreads(); return v; }
__device__ __forceinline char *_strndup(const char *z, int n) { __shared__ char *v; if (!threadIdx.x) v = _strndupG(z, n); __syncthreads(); return v; }
__device__ __forceinline char *_tagstrdup(TagBase *tag, const char *z) { __shared__ char *v; if (!threadIdx.x) v = _tagstrdupG(tag, z); __syncthreads(); return v; }
__device__ __forceinline char *_tagstrndup(TagBase *tag, const char *z, int n) { __shared__ char *v; if (!threadIdx.x) v = _tagstrndup(tag, z, n); __syncthreads(); return v; }
//
__device__ __forceinline void *_stackalloc(TagBase *tag, size_t size) { __shared__ void *v; if (!threadIdx.x) v = _stackallocG(tag, size); __syncthreads(); return v; }
__device__ __forceinline void *_stackallocZero(TagBase *tag, size_t size) { __shared__ void *v; if (!threadIdx.x) v = _stackallocZeroG(tag, size); __syncthreads(); return v; }
__device__ __forceinline void _stackfree(TagBase *tag, void *p) { if (!threadIdx.x) _stackfreeG(tag, p); __syncthreads(); }
#else
#define _alloc_init _alloc_initG
#define _alloc_shutdown _alloc_shutdownG
//
#define _alloc _allocG
#define _scratchalloc _scratchallocG
#define _scratchfree _scratchfreeG
#define _free _freeG
#define _tagfree _tagfreeG
#define _realloc _reallocG
#define _allocZero _allocZeroG
#define _tagallocZero _tagallocZeroG
#define _tagalloc _tagallocG
#define _tagrealloc _tagreallocG
#define _tagrealloc_or_free _tagrealloc_or_freeG
//
#define __strdup __strdupG
#define _strndup _strndupG
#define _tagstrdup _tagstrdupG
#define _tagstrndup _tagstrndupG
//
#define _stackalloc _stackallocG
#define _stackallocZero _stackallocZeroG
#define _stackfree _stackfreeG
#endif

RUNTIME_NAMEEND
#ifdef RUNTIME_NAME
	using namespace RUNTIME_NAME;
#endif
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
	__device__ __forceinline static void Destroy(Bitvec *p)
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

#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_MSGSIZE 4096
#define SENTINEL_MSGCOUNT 1
//#define SENTINEL_NAME "Global\\RuntimeSentinel"
#define SENTINEL_NAME "RuntimeSentinel"
#define SENTINEL_DEVICEMAPS 1

struct RuntimeSentinelMessage
{
	bool Async;
	char OP;
	int Size;
	char *(*Prepare)(void*,char*,char*);
	__device__ RuntimeSentinelMessage(bool async, char op, int size, char *(*prepare)(void*,char*,char*))
		: Async(async), OP(op), Size(size), Prepare(prepare) { }
public:
};
#define RUNTIMESENTINELPREPARE(P) ((char *(*)(void*,char*,char*))&P)

typedef struct
{
	unsigned short Magic;
	volatile long Status;
	int Length;
	char *Data;
	inline void Dump()
	{
		register char *b = Data;
		register int l = Length;
		printf("Command: 0x%x[%d] '", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
	}
} RuntimeSentinelCommand;

typedef struct
{
	long GetId;
	volatile long SetId;
	char Data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT];
	inline void Dump()
	{
		register char *b = (char *)this;
		register int l = sizeof(RuntimeSentinelMap);
		printf("Map: 0x%x[%d] '", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
	}
} RuntimeSentinelMap;

typedef struct RuntimeSentinelExecutor
{
	RuntimeSentinelExecutor *Next;
	const char *Name;
	bool (*Executor)(void*,RuntimeSentinelMessage*,int);
	void *Tag;
} RuntimeSentinelExecutor;

typedef struct RuntimeSentinelContext
{
	RuntimeSentinelMap *DeviceMap[SENTINEL_DEVICEMAPS];
	RuntimeSentinelMap *HostMap;
	RuntimeSentinelExecutor *List;
} RuntimeSentinelContext;

#if HAS_HOSTSENTINEL
extern RuntimeSentinelMap *_runtimeSentinelHostMap;
#endif
extern __constant__ RuntimeSentinelMap *_runtimeSentinelDeviceMap[SENTINEL_DEVICEMAPS];
struct RuntimeSentinel
{
public:
	static void ServerInitialize(RuntimeSentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME); 
	static void ServerShutdown();
	static void ClientInitialize(char *mapHostName = SENTINEL_NAME);
	static void ClientShutdown();
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
		__device__ __forceinline static char *Prepare(Stdio_fprintf *t, char *data, char *dataEnd)
		{
			int formatLength = (t->Format ? _strlen(t->Format) + 1 : 0);
			char *format = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += formatLength);
			if (end > dataEnd) return nullptr;
			memcpy(format, t->Format, formatLength);
			t->Format = format;
			return end;
		}
		RuntimeSentinelMessage Base;
		FILE *File; const char *Format;
		__device__ Stdio_fprintf(bool async, FILE *file, const char *format)
			: Base(async, 0, 1024, RUNTIMESENTINELPREPARE(Prepare)), File(file), Format(format) { RuntimeSentinel::Send(this, sizeof(Stdio_fprintf)); }
		int RC;
	};

	struct Stdio_setvbuf
	{
		__device__ __forceinline static char *Prepare(Stdio_setvbuf *t, char *data, char *dataEnd)
		{
			int bufferLength = (t->Buffer ? _strlen(t->Buffer) + 1 : 0);
			char *buffer = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += bufferLength);
			if (end > dataEnd) return nullptr;
			memcpy(buffer, t->Buffer, bufferLength);
			t->Buffer = buffer;
			return end;
		}
		RuntimeSentinelMessage Base;
		FILE *File; char *Buffer; int Mode; size_t Size;
		__device__ Stdio_setvbuf(FILE *file, char *buffer, int mode, size_t size)
			: Base(false, 1, 1024, RUNTIMESENTINELPREPARE(Prepare)), File(file), Buffer(buffer), Mode(mode), Size(size) { RuntimeSentinel::Send(this, sizeof(Stdio_setvbuf)); }
		int RC;
	};

	struct Stdio_fopen
	{
		__device__ __forceinline static char *Prepare(Stdio_fopen *t, char *data, char *dataEnd)
		{
			int filenameLength = (t->Filename ? _strlen(t->Filename) + 1 : 0);
			int modeLength = (t->Mode ? _strlen(t->Mode) + 1 : 0);
			char *filename = (char *)(data += _ROUND8(sizeof(*t)));
			char *mode = (char *)(data += filenameLength);
			char *end = (char *)(data += modeLength);
			if (end > dataEnd) return nullptr;
			memcpy(filename, t->Filename, filenameLength);
			memcpy(mode, t->Mode, modeLength);
			t->Filename = filename;
			t->Mode = mode;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Filename; const char *Mode;
		__device__ Stdio_fopen(const char *filename, const char *mode)
			: Base(false, 2, 1024, RUNTIMESENTINELPREPARE(Prepare)), Filename(filename), Mode(mode) { RuntimeSentinel::Send(this, sizeof(Stdio_fopen)); }
		FILE *RC;
	};

	struct Stdio_fflush
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fflush(bool async, FILE *file)
			: Base(async, 3, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fflush)); }
		int RC;
	};

	struct Stdio_fclose
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fclose(bool async, FILE *file)
			: Base(async, 4, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fclose)); }
		int RC;
	};

	struct Stdio_fgetc
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_fgetc(FILE *file)
			: Base(false, 5, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fgetc)); }
		int RC;
	};

	struct Stdio_fgets
	{
		__device__ __forceinline static char *Prepare(Stdio_fgets *t, char *data, char *dataEnd)
		{
			t->Str = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			return end;
		}
		RuntimeSentinelMessage Base;
		int Num; FILE *File;
		__device__ Stdio_fgets(char *str, int num, FILE *file)
			: Base(false, 6, 1024, RUNTIMESENTINELPREPARE(Prepare)), Str(str), Num(num), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fgets)); }
		char *Str; 
		char *RC;
	};

	struct Stdio_fputc
	{
		RuntimeSentinelMessage Base;
		int Ch; FILE *File;
		__device__ Stdio_fputc(bool async, int ch, FILE *file)
			: Base(async, 7, 0, nullptr), Ch(ch), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fputc)); }
		int RC;
	};

	struct Stdio_fputs
	{
		__device__ __forceinline static char *Prepare(Stdio_fputs *t, char *data, char *dataEnd)
		{
			int strLength = (t->Str ? _strlen(t->Str) + 1 : 0);
			char *str = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += strLength);
			if (end > dataEnd) return nullptr;
			memcpy(str, t->Str, strLength);
			t->Str = str;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Str; FILE *File;
		__device__ Stdio_fputs(bool async, const char *str, FILE *file)
			: Base(async, 8, 1024, RUNTIMESENTINELPREPARE(Prepare)), Str(str), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fputs)); }
		int RC;
	};

	struct Stdio_fread
	{
		__device__ __forceinline static char *Prepare(Stdio_fread *t, char *data, char *dataEnd)
		{
			t->Ptr = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += 1024);
			if (end > dataEnd) return nullptr;
			return end;
		}
		RuntimeSentinelMessage Base;
		size_t Size; size_t Num; FILE *File;
		__device__ Stdio_fread(bool async, size_t size, size_t num, FILE *file)
			: Base(async, 9, 1024, RUNTIMESENTINELPREPARE(Prepare)), Size(size), Num(num), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fread)); }
		size_t RC;
		void *Ptr;
	};

	struct Stdio_fwrite
	{
		__device__ __forceinline static char *Prepare(Stdio_fwrite *t, char *data, char *dataEnd)
		{
			size_t size = t->Size * t->Num;
			char *ptr = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += size);
			if (end > dataEnd) return nullptr;
			memcpy(ptr, t->Ptr, size);
			t->Ptr = ptr;
			return end;
		}
		RuntimeSentinelMessage Base;
		const void *Ptr; size_t Size; size_t Num; FILE *File;
		__device__ Stdio_fwrite(bool async, const void *ptr, size_t size, size_t num, FILE *file)
			: Base(async, 10, 1024, RUNTIMESENTINELPREPARE(Prepare)), Ptr(ptr), Size(size), Num(num), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_fwrite)); }
		size_t RC;
	};

	struct Stdio_fseek
	{
		RuntimeSentinelMessage Base;
		FILE *File; long int Offset; int Origin;
		__device__ Stdio_fseek(bool async, FILE *file, long int offset, int origin)
			: Base(async, 11, 0, nullptr), File(file), Offset(offset), Origin(origin) { RuntimeSentinel::Send(this, sizeof(Stdio_fseek)); }
		int RC;
	};

	struct Stdio_ftell
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_ftell(FILE *file)
			: Base(false, 12, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_ftell)); }
		int RC;
	};

	struct Stdio_feof
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_feof(FILE *file)
			: Base(false, 13, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_feof)); }
		int RC;
	};

	struct Stdio_ferror
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_ferror(FILE *file)
			: Base(false, 14, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_ferror)); }
		int RC;
	};

	struct Stdio_clearerr
	{
		RuntimeSentinelMessage Base;
		FILE *File;
		__device__ Stdio_clearerr(FILE *file)
			: Base(true, 15, 0, nullptr), File(file) { RuntimeSentinel::Send(this, sizeof(Stdio_clearerr)); }
	};

	struct Stdio_rename
	{
		__device__ __forceinline static char *Prepare(Stdio_rename *t, char *data, char *dataEnd)
		{
			int oldnameLength = (t->Oldname ? _strlen(t->Oldname) + 1 : 0);
			int newnameLength = (t->Newname ? _strlen(t->Newname) + 1 : 0);
			char *oldname = (char *)(data += _ROUND8(sizeof(*t)));
			char *newname = (char *)(data += oldnameLength);
			char *end = (char *)(data += newnameLength);
			if (end > dataEnd) return nullptr;
			memcpy(oldname, t->Oldname, oldnameLength);
			memcpy(newname, t->Newname, newnameLength);
			t->Oldname = oldname;
			t->Newname = newname;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Oldname; const char *Newname;
		__device__ Stdio_rename(const char *oldname, const char *newname)
			: Base(false, 16, 1024, RUNTIMESENTINELPREPARE(Prepare)), Oldname(oldname), Newname(newname) { RuntimeSentinel::Send(this, sizeof(Stdio_rename)); }
		int RC;
	};

	struct Stdio_unlink
	{
		__device__ __forceinline static char *Prepare(Stdio_unlink *t, char *data, char *dataEnd)
		{
			int strLength = (t->Str ? _strlen(t->Str) + 1 : 0);
			char *str = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += strLength);
			if (end > dataEnd) return nullptr;
			memcpy(str, t->Str, strLength);
			t->Str = str;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Str;
		__device__ Stdio_unlink(const char *str)
			: Base(false, 17, 1024, RUNTIMESENTINELPREPARE(Prepare)), Str(str) { RuntimeSentinel::Send(this, sizeof(Stdio_unlink)); }
		int RC;
	};

	struct Stdio_close
	{
		RuntimeSentinelMessage Base;
		int Handle;
		__device__ Stdio_close(int handle)
			: Base(false, 18, 0, nullptr), Handle(handle) { RuntimeSentinel::Send(this, sizeof(Stdio_close)); }
		int RC;
	};

	struct Stdio_system
	{
		__device__ __forceinline static char *Prepare(Stdio_system *t, char *data, char *dataEnd)
		{
			int strLength = (t->Str ? _strlen(t->Str) + 1 : 0);
			char *str = (char *)(data += _ROUND8(sizeof(*t)));
			char *end = (char *)(data += strLength);
			if (end > dataEnd) return nullptr;
			memcpy(str, t->Str, strLength);
			t->Str = str;
			return end;
		}
		RuntimeSentinelMessage Base;
		const char *Str;
		__device__ Stdio_system(const char *str)
			: Base(false, 19, 1024, RUNTIMESENTINELPREPARE(Prepare)), Str(str) { RuntimeSentinel::Send(this, sizeof(Stdio_system)); }
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
	size_t Size;		// Amount of space allocated in zText
	int MaxSize;		// Maximum allowed string length
	bool AllocFailed;	// Becomes true if any memory allocation fails
	unsigned char AllocType; // 0: none,  1: _tagalloc,  2: _alloc
	bool Overflowed;    // Becomes true if string size exceeds limits

	__device__ void AppendSpace(int length);
	__device__ void AppendFormat_(bool useExtended, const char *fmt, _va_list &args);
	__device__ void Append(const char *z, int length);
	__device__ __forceinline void AppendElement(const char *z) { Append(", ", 2); Append(z, _strlen(z)); }
	__device__ char *ToString();
	__device__ void Reset();
	__device__ static void Init(TextBuilder *b, char *text = nullptr, int capacity = -1, int maxAlloc = -1);
	//
#if __CUDACC__
	__device__ __forceinline void AppendFormat(const char *fmt) { _va_list args; _va_start(args); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); AppendFormat_(true, fmt, args); _va_end(args); }
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void AppendFormat(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); AppendFormat_(true, fmt, args); _va_end(args); }
#else
	__device__ __forceinline void AppendFormat(const char *fmt, ...) { _va_list args; _va_start(args, fmt); AppendFormat_(true, fmt, args); _va_end(args); }
#endif
};

#pragma endregion

//////////////////////
// SNPRINTF
#pragma region SNPRINTF

extern "C" __device__ int __vsnprintf(const char *buf, size_t bufLen, const char *fmt, _va_list *args);
#if __CUDACC__
__device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt) { _va_list args; _va_start(args); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static int __snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
#else
extern "C" __device__ __forceinline int __snprintf(const char *buf, size_t bufLen, const char *fmt, ...) { _va_list args; _va_start(args, fmt); int n = __vsnprintf(buf, bufLen, fmt, &args); _va_end(args); return n; }
#endif
#define _sprintf(buf, fmt, ...) __snprintf(buf, -1, fmt, __VA_ARGS__)

#pragma endregion

//////////////////////
// FPRINTF
#pragma region FPRINTF

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
extern __constant__ FILE *__iob_file[3];
#define _stdin  (__iob_file[0])
#define _stdout (__iob_file[1])
#define _stderr (__iob_file[2])
#else
#define _stdin stdin
#define _stdout stdout
#define _stderr stderr
#endif

#if OS_MAP
//#define _fprintfR _fprintf
//#define _fprintf(f, ...) printf(__VA_ARGS__)
extern "C" __device__ inline int __fileno(FILE *f) { return (f == _stdin ? 0 : f == _stdout ? 1 : f == _stderr ? 2 : -1); }
#define _fprintf(f, ...) (f == _stdout || f == _stderr ? printf(__VA_ARGS__) : _fprintf_(f, _mprintf("%s", __VA_ARGS__)))
#define _fprintfR(f, ...) (f == _stdout || f == _stderr ? printf(__VA_ARGS__) : _fprintfR_(f, _mprintf("%s", __VA_ARGS__)))
extern "C" __device__ inline void _fprintf_(FILE *f, const char *v) { Messages::Stdio_fprintf msg(true, f, v); _free((void *)v); }
extern "C" __device__ inline int _fprintfR_(FILE *f, const char *v) { Messages::Stdio_fprintf msg(false, f, v); _free((void *)v); return msg.RC; }
extern "C" __device__ inline int _setvbuf(FILE *f, char *b, int m, size_t s) { Messages::Stdio_setvbuf msg(f, b, m, s); return msg.RC; }
extern "C" __device__ inline FILE *_fopen(const char *f, const char *m) { Messages::Stdio_fopen msg(f, m); return msg.RC; }
extern "C" __device__ inline void _fflush(FILE *f) { if (f == _stdout || f == _stderr) return; Messages::Stdio_fflush msg(true, f); }
extern "C" __device__ inline int _fflushR(FILE *f) { if (f == _stdout || f == _stderr) return 0; Messages::Stdio_fflush msg(false, f); return msg.RC; }
extern "C" __device__ inline void _fclose(FILE *f) { if (f == _stdout || f == _stderr) return; Messages::Stdio_fclose msg(true, f); }
extern "C" __device__ inline int _fcloseR(FILE *f) { if (f == _stdout || f == _stderr) return 0; Messages::Stdio_fclose msg(false, f); return msg.RC; }
extern "C" __device__ inline int _fgetc(FILE *f) { Messages::Stdio_fgetc msg(f); return msg.RC; }
extern "C" __device__ inline char *_fgets(char *c, int n, FILE *f) { Messages::Stdio_fgets msg(c, n, f); return msg.RC; }
extern "C" __device__ inline void _fputc(int c, FILE *f) { if (f == _stdout || f == _stderr) { printf("%c", c); return; } Messages::Stdio_fputc msg(true, c, f); }
extern "C" __device__ inline int _fputcR(int c, FILE *f) { if (f == _stdout || f == _stderr) { printf("%c", c); return 0; } Messages::Stdio_fputc msg(false, c, f); return msg.RC; }
extern "C" __device__ inline void _fputs(const char *s, FILE *f) { if (f == _stdout || f == _stderr) { printf(s); return; } Messages::Stdio_fputs msg(true, s, f); }
extern "C" __device__ inline int _fputsR(const char *s, FILE *f) { if (f == _stdout || f == _stderr) { printf(s); return 0; } Messages::Stdio_fputs msg(false, s, f); return msg.RC; }
extern "C" __device__ inline size_t _fread(void *p, size_t s, size_t n, FILE *f) { Messages::Stdio_fread msg(false, s, n, f); memcpy(p, msg.Ptr, msg.RC); return msg.RC; }
extern "C" __device__ inline size_t _fwrite(const void *p, size_t s, size_t n, FILE *f) { Messages::Stdio_fwrite msg(false, p, s, n, f); return msg.RC; }
extern "C" __device__ inline int _fseek(FILE *f, long int o, int s) { Messages::Stdio_fseek msg(true, f, o, s); return msg.RC; }
extern "C" __device__ inline int _ftell(FILE *f) { Messages::Stdio_ftell msg(f); return msg.RC; }
extern "C" __device__ inline int _feof(FILE *f) { Messages::Stdio_feof msg(f); return msg.RC; }
extern "C" __device__ inline int _ferror(FILE *f) { if (f == _stdout || f == _stderr) return 0; Messages::Stdio_ferror msg(f); return msg.RC; }
extern "C" __device__ inline void _clearerr(FILE *f) { Messages::Stdio_clearerr msg(f); }
extern "C" __device__ inline int _rename(const char *a, const char *b) { Messages::Stdio_rename msg(a, b); return msg.RC; }
extern "C" __device__ inline int __unlink(const char *a) { Messages::Stdio_unlink msg(a); return msg.RC; }
extern "C" __device__ inline int __close(int a) { Messages::Stdio_close msg(a); return msg.RC; }
extern "C" __device__ inline int _system(const char *c) { Messages::Stdio_system msg(c); return msg.RC; }
extern "C" __device__ inline void _puts(const char *s) { printf("%s\n", s); }
#else
#define _fprintfR _fprintf
#define _fflushR _fflush
#define _fcloseR _fclose
#define _fputcR _fputc
#define _fputsR _fputs
#if __CUDACC__
#define __fileno(f) (int)(f == _stdin ? 0 : f == _stdout ? 1 : f == _stderr ? 2 : -1)
#define _fprintf(f, ...) printf(__VA_ARGS__)
#define _setvbuf(f, b, m, s) (int)0
#define _fopen(f, m) (FILE *)0
#define _fflush(f) (int)0
#define _fclose(f) (int)0
#define _fgetc(f) (int)0
#define _fgets(s, n, f) (int)0
#define _fputc(c, f) printf("%c", c)
#define _fputs(s, f) printf(s)
#define _fread(p, s, n, f) (size_t)0
#define _fwrite(p, s, n, f) (size_t)0
#define _fseek(f, o, s) (int)0
#define _ftell(f) (int)0
#define _feof(f) (int)0
#define _ferror(f) (int)0
#define _clearerr(f) (void)0
#define _rename(a, b) (int)0
#define __unlink(a) (int)0
#define __close(a) (int)0
#define _system(c) (int)0
#define _puts(s) printf("%s\n", s)
#else
#define __fileno(f) fileno(f)
#define _fprintf(f, ...) fprintf(f, __VA_ARGS__)
#define _setvbuf(f, b, m, s) setvbuf(f, b, m, s)
#define _fopen(f, m) fopen(f, m)
#define _fflush(f) fflush(f)
#define _fclose(f) fclose(f)
#define _fgetc(f) fgetc(f)
#define _fgets(s, n, f) fgets(s, n, f)
#define _fputc(c, f) fputc(c, f)
#define _fputs(s, f) fputs(s, f)
#define _fread(p, s, n, f) fread(p, s, n, f)
#define _fwrite(p, s, n, f) fread(p, s, n, f)
#define _fseek(f, o, s) fseek(f, o, s)
#define _ftell(f) ftell(f)
#define _feof(f) feof(f)
#define _ferror(f) ferror(f)
#define _clearerr(f) clearerr(f)
#define _rename(a, b) rename(a, b)
#define __unlink(a) remove(a)
#define __close(a) close(a)
#define _system(c) system(c)
#define _puts(s) puts(s)
#endif
#endif

#pragma endregion

//////////////////////
// MPRINTF
#pragma region MPRINTF

extern "C" __device__ char *_vmprintf(const char *fmt, _va_list *args);
extern "C" __device__ char *_vmtagprintf(TagBase *tag, const char *fmt, _va_list *args);
#if __CUDACC__
__device__ __forceinline char *_mprintf(const char *fmt) { _va_list args; _va_start(args); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
//
__device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt) { _va_list args; _va_start(args); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
//
__device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt) { _va_list args; _va_start(args); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }

__device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt) { _va_list args; _va_start(args); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
#else
extern "C" __device__ __forceinline char *_mprintf(const char *fmt, ...) { _va_list args; _va_start(args, fmt); char *z = _vmprintf(fmt, &args); _va_end(args); return z; }
extern "C" __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, ...) { _va_list args; _va_start(args, fmt); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); return z; }
extern "C" __device__ __forceinline char *_mtagappendf(TagBase *tag, char *src, const char *fmt, ...) { _va_list args; _va_start(args, fmt); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, src); return z; }
extern "C" __device__ __forceinline void _mtagassignf(char **src, TagBase *tag, const char *fmt, ...) { _va_list args; _va_start(args, fmt); char *z = _vmtagprintf(tag, fmt, &args); _va_end(args); _tagfree(tag, *src); *src = z; }
#endif

#pragma endregion

//////////////////////
// MNPRINTF
#pragma region MNPRINTF

extern "C" __device__ char *__vmnprintf(const char *buf, size_t bufLen, const char *fmt, _va_list *args);
#if __CUDACC__
__device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt) { _va_list args; _va_start(args); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list1<T1> args; _va_start(args, arg1); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; _va_start(args, arg1, arg2); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; _va_start(args, arg1, arg2, arg3); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; _va_start(args, arg1, arg2, arg3, arg4); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; _va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; _va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
#else
extern "C" __device__ __forceinline char *__mnprintf(const char *buf, size_t bufLen, const char *fmt, ...) { _va_list args; _va_start(args, fmt); char *z = __vmnprintf(buf, bufLen, fmt, &args); _va_end(args); return z; }
#endif
#define _mmprintf(buf, fmt, ...) __mnprintf(buf, sizeof(buf), fmt, __VA_ARGS__)

#pragma endregion

#endif // __RUNTIME_H__