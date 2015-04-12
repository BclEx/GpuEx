#ifndef __RUNTIME_H__
#define __RUNTIME_H__

//////////////////////
// NATIVE
#pragma region NATIVE

#include <stdio.h>
#if __CUDACC__
#define __forceinline __forceinline__
#include "Runtime.cu.h"
#else
#include <string.h>
#include <malloc.h>
#include "Runtime.cpu.h"
#endif

#ifndef _API
#define _API extern
#endif

#ifndef OS_WIN
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
#else
#define OS_UNIX 0
#define OS_GPU 0
#endif

#pragma endregion

#pragma region Limits

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
__device__ unsigned int _utf8read(const unsigned char **z);
__device__ int _utf8charlength(const char *z, int bytes);
#if _DEBUG
__device__ int _utf8to8(unsigned char *z);
#endif
#ifndef OMIT_UTF16
__device__ int _utf16bytelength(const void *z, int chars);
#ifdef TEST
__device__ void _runtime_utfselftest();
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
// MUTEX
#pragma region MUTEX

#if !defined(THREADSAFE)
#if defined(__THREADSAFE__)
#define THREADSAFE __THREADSAFE__
#else
#define THREADSAFE 1 // IMP: R-07272-22309
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
#if OS_UNIX
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

struct _mutex_obj;
typedef _mutex_obj *MutexEx;
#ifdef MUTEX_OMIT
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
#ifdef _DEBUG
__device__ extern bool _mutex_held(MutexEx p);
__device__ extern bool _mutex_notheld(MutexEx p);
#endif
__device__ extern int _mutex_init();
__device__ extern void _mutex_shutdown();
__device__ extern MutexEx _mutex_alloc(MUTEX id);
__device__ extern void _mutex_free(MutexEx p);
__device__ extern void _mutex_enter(MutexEx p);
__device__ extern bool _mutex_tryenter(MutexEx p);
__device__ extern void _mutex_leave(MutexEx p);
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
	STATUS_LRATCH_USED = 3,
	STATUS_LRATCH_OVERFLOW = 4,
	STATUS_MALLOC_SIZE = 5,
	STATUS_PARSER_STACK = 6,
	STATUS_PAGECACHE_SIZE = 7,
	STATUS_LRATCH_SIZE = 8,
	STATUS_MALLOC_COUNT = 9,
};

__device__ extern int _status_value(STATUS op);
__device__ extern void _status_add(STATUS op, int n);
__device__ extern void _status_set(STATUS op, int x);
__device__ extern bool _status(STATUS op, int *current, int *highwater, bool resetFlag);

#pragma endregion

//////////////////////
// TAGBASE
#pragma region TAGBASE

class TagBase
{
public:
	struct RuntimeStatics
	{
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
};

__device__ extern _WSD TagBase::RuntimeStatics g_RuntimeStatics;
#define TagBase_RuntimeStatics _GLOBAL(TagBase::RuntimeStatics, g_RuntimeStatics)

#pragma endregion

//////////////////////
// FUNC
#pragma region FUNC

#undef _toupper
#undef _tolower
#define _toupper(x) ((x)&~(__curtCtypeMap[(unsigned char)(x)]&0x20))
#define _isspace(x) ((__curtCtypeMap[(unsigned char)(x)]&0x01)!=0)
#define _isalnum(x) ((__curtCtypeMap[(unsigned char)(x)]&0x06)!=0)
#define _isalpha(x) ((__curtCtypeMap[(unsigned char)(x)]&0x02)!=0)
#define _isdigit(x) ((__curtCtypeMap[(unsigned char)(x)]&0x04)!=0)
#define _isxdigit(x) ((__curtCtypeMap[(unsigned char)(x)]&0x08)!=0)
#define _isidchar(x) ((__curtCtypeMap[(unsigned char)(x)]&0x46)!=0)
#define _tolower(x) (__curtUpperToLower[(unsigned char)(x)])
#define _ispoweroftwo(x) (((x)&((x)-1))==0)
__device__ inline static bool _isalpha2(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// array
template <typename T> struct array_t { int length; T *data; __device__ __forceinline array_t() { data = nullptr; length = 0; } __device__ __forceinline array_t(T *a) { data = a; length = 0; } __device__ __forceinline array_t(T *a, int b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; __device__ __forceinline array_t2() { data = nullptr; length = 0; } __device__ __forceinline array_t2(T *a) { data = a; length = 0; } __device__ __forceinline array_t2(T *a, size_t b) { data = a; length = b; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
template <typename TLength, typename T, size_t size> struct array_t3 { TLength length; T data[size]; __forceinline array_t3() { length = 0; } __device__ __forceinline void operator=(T *a) { data = a; } __device__ __forceinline operator T *() { return data; } };
#define _lengthof(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// strcpy
template <typename T> __device__ __forceinline void _strcpy(const T *dest, const T *src)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	while (*a) { *a++ = *b++; }
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

// strcmp
template <typename T> __device__ __forceinline int _strcmp(const T *left, const T *right)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

// strncmp
#undef _fstrncmp
template <typename T> __device__ __forceinline int _strncmp(const T *left, const T *right, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b]);
}
#define _fstrncmp(x, y) (_tolower(*(unsigned char *)(x))==_tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// memcpy
#define _memcpy(dest, src, length) if (length) memcpy(dest, src, length)
#if 0
template <typename T> __device__ __forceinline void _memcpy(T *dest, const T *src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}
template <typename T> __device__ __forceinline void _memcpy(T *dest, T *src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}
#endif

// memset
#define _memset(dest, value, length) if (length) memset(dest, value, length)
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
template <typename T, typename Y> __device__ __forceinline int _memcmp(T *left, Y *right, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (--length > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// memmove
template <typename T, typename Y> __device__ __forceinline void _memmove(T *left, Y *right, size_t length)
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
__device__ void *__allocsystem_alloc(size_t size);
__device__ void __allocsystem_free(void *prior);
__device__ void *__allocsystem_realloc(void *prior, size_t size);
__device__ size_t __allocsystem_size(void *prior);
__device__ size_t __allocsystem_roundup(size_t size);
__device__ int __allocsystem_init(void *p);
__device__ void __allocsystem_shutdown(void *p);
//
//__device__ void __alloc_setmemoryalarm(int (*callback)(void*,long long,int), void *arg, long long threshold);
//__device__ long long __alloc_softheaplimit64(long long n);
//__device__ void __alloc_softheaplimit(int n);
__device__ int _alloc_init();
__device__ bool _alloc_heapnearlyfull();
__device__ void _alloc_shutdown();
//__device__ long long __alloc_memoryused();
//__device__ long long __alloc_memoryhighwater(bool resetFlag);
__device__ void *_alloc(size_t size);
__device__ void *_scratchalloc(size_t size);
__device__ void _scratchfree(void *p);
__device__ size_t _allocsize(void *p);
__device__ size_t _tagallocsize(TagBase *tag, void *p);
__device__ void _free(void *p);
__device__ void _tagfree(TagBase *tag, void *p);
__device__ void *_realloc(void *old, size_t newSize);
__device__ void *_allocZero(size_t size);
__device__ void *_tagallocZero(TagBase *tag, size_t size);
__device__ void *_tagalloc(TagBase *tag, size_t size);
__device__ void *_tagrealloc(TagBase *tag, void *old, size_t size);
//__device__ void *_tagrealloc_or_free(TagBase *tag, void *old, size_t newSize);
__device__ __forceinline void *_tagrealloc_or_free(TagBase *tag, void *old, size_t newSize)
{
	void *p = _tagrealloc(tag, old, newSize);
	if (!p) _tagfree(tag, old);
	return p;
}

//__device__ char *_tagstrdup(TagBase *tag, const char *z);
__device__ __forceinline char *_tagstrdup(TagBase *tag, const char *z)
{
	if (z == nullptr) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagalloc(tag, (int)n);
	if (newZ) _memcpy(newZ, (char *)z, n);
	return newZ;
}
//__device__ char *_tagstrndup(TagBase *tag, const char *z, int n);
__device__ __forceinline char *_tagstrndup(TagBase *tag, const char *z, int n)
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

__device__ char *__vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list *args, int *length);
#if __CUDACC__
__device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt) { va_list args; va_start(args); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr); va_end(args); return z; }
#else
__device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = __vsnprintf(buf, bufLen, fmt, &args, nullptr);
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
#define stdin nullptr
#define stdout nullptr
#define stderr nullptr
#endif

#if __CUDACC__
#define _fprintf(f, ...) printf(__VA_ARGS__)
#define _fopen(f, b) 0
#define _fflush(f)
#define _fclose(f)
#else
#define _fprintf(f, ...) fprintf(f, __VA_ARGS__)
#define _fopen(f, b) fopen(f, b)
#define _fflush(f) fflush(f)
#define _fclose(f) fclose(f)
#endif

#pragma endregion

//////////////////////
// MPRINTF
#pragma region MPRINTF

__device__ char *_vmprintf(const char *fmt, va_list *args, int *length);
__device__ char *_vmtagprintf(TagBase *tag, const char *fmt, va_list *args, int *length);
#if __CUDACC__
__device__ __forceinline char *_mprintf(const char *fmt) { va_list args; va_start(args); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmprintf(fmt, &args, nullptr); va_end(args); return z; }
//
__device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline char *_mtagprintf(TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); return z; }
//
__device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, src); return z; }

__device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt) { va_list args; va_start(args); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1) { va_list1<T1> args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2) { va_list2<T1,T2> args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list3<T1,T2,T3> args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list4<T1,T2,T3,T4> args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list5<T1,T2,T3,T4,T5> args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list6<T1,T2,T3,T4,T5,T6> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list7<T1,T2,T3,T4,T5,T6,T7> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list8<T1,T2,T3,T4,T5,T6,T7,T8> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, &args, nullptr); va_end(args); _tagfree(tag, *src); *src = z; }
#else
__device__ inline static char *_mprintf(const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmprintf(fmt, &args, nullptr);
	va_end(args);
	return z;
}
//
__device__ inline static char *_mtagprintf(TagBase *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args, nullptr);
	va_end(args);
	return z;
}
//
__device__ inline static char *_mtagappendf(TagBase *tag, char *src, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args, nullptr);
	va_end(args);
	_tagfree(tag, src);
	return z;
}
__device__ inline static void _mtagassignf(char **src, TagBase *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, &args, nullptr);
	va_end(args);
	_tagfree(tag, *src);
	*src = z;
}
#endif

#pragma endregion

#endif // __RUNTIME_H__