#include "Runtime.h"

#ifdef OMIT_INLINEFUNC

// strcpy
#if 1
__device__ void _strcpy(char *__restrict__ dest, const char *__restrict__ src)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
}
#else
__device__ void _strcpy(register char *__restrict__ dest, register const char *__restrict__ src)
{
	register int i = 0;
	while (src[i]); { dest[i++] = src[i]; } dest[i] = src[i];
}
#endif

// strncpy
__device__ void _strncpy(char *__restrict__ dest, const char *__restrict__ src, size_t length)
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
__device__ void _strcat(char *__restrict__ dest, const char *__restrict__ src)
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
__device__ char *_strchr(const char *src, int ch)
{
	register unsigned char *s = (unsigned char *)src;
	register unsigned char l = (unsigned char)__curtUpperToLower[ch];
	while (*s && __curtUpperToLower[*s] != l) { s++; }
	return (char *)(*s ? s : nullptr);
}

// strstr
//http://articles.leetcode.com/2010/10/implement-strstr-to-find-substring-in.html
__device__ const char *_strstr(const char *__restrict__ src, const char *__restrict__ str)
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
__device__ int _strcmp(const char *__restrict__ left, const char *__restrict__ right)
{
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
	while (*a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

// strncmp
__device__ int _strncmp(const char *__restrict__ left, const char *__restrict__ right, int n)
{
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b]);
}

// memcpy
#if 0
__device__ void _memcpy(char *__restrict__ dest, const char *__restrict__ src, size_t length)
{
	register unsigned char *a = (unsigned char *)dest;
	register unsigned char *b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}
#endif

// memset
#if 0
__device__ void _memset(char *dest, const char value, size_t length)
{
	register unsigned char *a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}
#endif

// memchr
__device__ const void *_memchr(const void *src, char ch, size_t length)
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
__device__ int _memcmp(const void *__restrict__ left, const void *__restrict__ right, size_t length)
{
	if (!length)
		return 0;
	register unsigned char *a = (unsigned char *)left;
	register unsigned char *b = (unsigned char *)right;
	while (--length > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// memmove
#define	wsize sizeof(word)
#define	wmask (wsize - 1)
#define	TLOOP(s) do { s; } while (--t)
__device__ void _memmove(void *__restrict__ dst0, const void *__restrict__ src0, size_t length)
{
	if (!length || dst0 == src0) return; // nothing to do
	register char *dst = (char *)dst0;
	register const char *src = (const char *)src0;
	register size_t t;
	// Copy forward.
	if ((unsigned long)dst < (unsigned long)src)
	{
		t = (int)src; // only need low bits
		if ((t | (int)dst) & wmask)
		{
			// Try to align operands.  This cannot be done unless the low bits match.
			t = ((t ^ (int)dst) & wmask || length < wsize ? length : wsize - (t & wmask));
			length -= t;
			TLOOP(*dst++ = *src++);
		}
		// Copy whole words, then mop up any trailing bytes.
		t = length / wsize;
		if (t) TLOOP(*(word *)dst = *(word *)src; src += wsize; dst += wsize);
		t = length & wmask;
		if (t) TLOOP(*dst++ = *src++);
	}
	// Copy backwards.  Otherwise essentially the same. Alignment works as before, except that it takes (t&wmask) bytes to align, not wsize-(t&wmask).
	else
	{
		src += length;
		dst += length;
		t = (int)src;
		if ((t | (int)dst) & wmask)
		{
			t = ((t ^ (int)dst) & wmask || length <= wsize ? length : t & wmask);
			length -= t;
			TLOOP(*--dst = *--src);
		}
		t = length / wsize;
		if (t) TLOOP(src -= wsize; dst -= wsize; *(word *)dst = *(word *)src);
		t = length & wmask;
		if (t) TLOOP(*--dst = *--src);
	}
}

// strlen30
__host__ __device__ int _strlen(const char *z)
{
	if (!z) return 0;
	register const char *z2 = z;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}
__device__ int _strlen16(const void *z)
{
	if (!z) return 0;
	register const char *z2 = (const char *)z;
	int n;
	for (n = 0; z2[n] || z2[n+1]; n += 2) { }
	return n;
}

// hextobyte
__device__ unsigned char _hextobyte(char h)
{
	_assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
	return (unsigned char)((h + 9*(1&(h>>6))) & 0xf);
}

#ifndef OMIT_FLOATING_POINT
__host__ __device__ bool _isnan(double x)
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

#endif