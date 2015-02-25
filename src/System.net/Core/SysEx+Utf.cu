// utf.c
#include "Core.cu.h"

namespace Core
{

#pragma region UTF Macros

	__device__ static const unsigned char _utf8Trans1[] =
	{
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
		0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x00, 0x00,
	};

#define WRITE_UTF8(z, c) { \
	if (c < 0x00080) { \
	*z++ = (uint8)(c&0xFF); \
	} else if (c < 0x00800) { \
	*z++ = 0xC0 + (uint8)((c>>6)&0x1F); \
	*z++ = 0x80 + (uint8)(c&0x3F); \
	} else if (c < 0x10000) { \
	*z++ = 0xE0 + (uint8)((c>>12)&0x0F); \
	*z++ = 0x80 + (uint8)((c>>6)&0x3F); \
	*z++ = 0x80 + (uint8)(c&0x3F); \
		} else { \
		*z++ = 0xF0 + (uint8)((c>>18)&0x07); \
		*z++ = 0x80 + (uint8)((c>>12)&0x3F); \
		*z++ = 0x80 + (uint8)((c>>6)&0x3F); \
		*z++ = 0x80 + (uint8)(c&0x3F); \
		} \
		}

#define WRITE_UTF16LE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (uint8)(c&0x00FF); \
	*z++ = (uint8)((c>>8)&0x00FF); \
	} else { \
	*z++ = (uint8)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (uint8)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (uint8)(c&0x00FF); \
	*z++ = (uint8)(0x00DC + ((c>>8)&0x03)); \
	} \
		}

#define WRITE_UTF16BE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (uint8)((c>>8)&0x00FF); \
	*z++ = (uint8)(c&0x00FF); \
	} else { \
	*z++ = (uint8)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (uint8)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (uint8)(0x00DC + ((c>>8)&0x03)); \
	*z++ = (uint8)(c&0x00FF); \
	} \
		}

#define READ_UTF16LE(z, TERM, c) { \
	c = (*z++); \
	c += ((*z++)<<8); \
	if (c >= 0xD800 && c < 0xE000 && TERM) { \
	int c2 = (*z++); \
	c2 += ((*z++)<<8); \
	c = (c2&0x03FF) + ((c&0x003F)<<10) + (((c&0x03C0)+0x0040)<<10); \
	} \
		}

#define READ_UTF16BE(z, TERM, c) { \
	c = ((*z++)<<8); \
	c += (*z++); \
	if (c >= 0xD800 && c < 0xE000 && TERM) { \
	int c2 = ((*z++)<<8); \
	c2 += (*z++); \
	c = (c2&0x03FF) + ((c&0x003F)<<10) + (((c&0x03C0)+0x0040)<<10); \
	} \
		}

#define READ_UTF8(z, term, c) \
	c = *(z++); \
	if (c >= 0xc0) { \
	c = _utf8Trans1[c-0xc0]; \
	while (z != term && (*z & 0xc0) == 0x80) { \
	c = (c<<6) + (0x3f & *(z++)); \
	} \
	if (c < 0x80 || (c&0xFFFFF800) == 0xD800 || (c&0xFFFFFFFE) == 0xFFFE) c = 0xFFFD; \
	}

#pragma endregion

	__device__ uint32 SysEx::Utf8Read(const unsigned char **z)
	{
		// Same as READ_UTF8() above but without the zTerm parameter. For this routine, we assume the UTF8 string is always zero-terminated.
		unsigned int c = *((*z)++);
		if (c >= 0xc0)
		{
			c = _utf8Trans1[c-0xc0];
			while ((*(*z) & 0xc0) == 0x80)
				c = (c<<6) + (0x3f & *((*z)++));
			if (c < 0x80 || (c&0xFFFFF800) == 0xD800 || (c&0xFFFFFFFE) == 0xFFFE ) c = 0xFFFD;
		}
		return c;
	}

	__device__ int SysEx::Utf8CharLen(const char *z, int bytes)
	{
		int r = 0;
		const uint8 *z2 = (const uint8 *)z;
		const uint8 *zTerm = (bytes >= 0 ? &z2[bytes] : (const uint8 *)-1);
		_assert(z2 <= zTerm);
		while (*z2 != 0 && z2 < zTerm)
		{
			_strskiputf8(z2);
			r++;
		}
		return r;
	}

#if defined(TEST) && defined(_DEBUG)
	__device__ int SysEx::Utf8To8(unsigned char *z)
	{
		unsigned char *zOut = z;
		unsigned char *zStart = z;
		uint32 c;
		while (z[0] && zOut <= z)
		{
			c = Utf8Read((const uint8 **)&z);
			if (c != 0xfffd)
				WRITE_UTF8(zOut, c);
		}
		*zOut = 0;
		return (int)(zOut - zStart);
	}
#endif

#ifndef OMIT_UTF16
	__device__ int SysEx::Utf16ByteLen(const void *z, int chars)
	{
		int c;
		unsigned char const *z2 = (unsigned char const *)z;
		int n = 0;
		if (TEXTENCODE_UTF16NATIVE == TEXTENCODE_UTF16BE)
		{
			while (n < chars)
			{
				READ_UTF16BE(z2, 1, c);
				n++;
			}
		}
		else
		{
			while (n < chars)
			{
				READ_UTF16LE(z2, 1, c);
				n++;
			}
		}
		return (int)(z2 - (unsigned char const *)z);
	}

#if defined(TEST)
	__device__ void SysEx::UtfSelfTest()
	{
		unsigned int i, t;
		unsigned char buf[20];
		unsigned char *z;
		int n;
		unsigned int c;
		for (i = 0; i < 0x00110000; i++)
		{
			z = buf;
			WRITE_UTF8(z, i);
			n = (int)(z - buf);
			_assert(n > 0 && n <= 4);
			z[0] = 0;
			z = buf;
			c = Utf8Read((const uint8 **)&z);
			t = i;
			if (i >= 0xD800 && i <= 0xDFFF) t = 0xFFFD;
			if ((i&0xFFFFFFFE) == 0xFFFE) t = 0xFFFD;
			_assert(c == t);
			_assert((z - buf) == n);
		}
		for (i = 0; i < 0x00110000; i++)
		{
			if (i >= 0xD800 && i < 0xE000) continue;
			z = buf;
			WRITE_UTF16LE(z, i);
			n = (int)(z - buf);
			_assert(n > 0 && n <= 4);
			z[0] = 0;
			z = buf;
			READ_UTF16LE(z, 1, c);
			_assert(c == i);
			_assert((z - buf) == n);
		}
		for (i = 0; i < 0x00110000; i++)
		{
			if (i >= 0xD800 && i < 0xE000) continue;
			z = buf;
			WRITE_UTF16BE(z, i);
			n = (int)(z-buf);
			_assert(n > 0 && n <= 4);
			z[0] = 0;
			z = buf;
			READ_UTF16BE(z, 1, c);
			_assert(c == i);
			_assert((z - buf) == n);
		}
	}
#endif
#endif
}