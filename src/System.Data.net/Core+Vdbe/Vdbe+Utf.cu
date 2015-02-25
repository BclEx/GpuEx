// utf.c
#ifndef OMIT_UTF16
#include "VdbeInt.cu.h"
#define TRANSLATE_TRACE

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

	__device__ RC Vdbe::MemTranslate(Mem *mem, TEXTENCODE desiredEncode)
	{
		int len; // Maximum length of output string in bytes
		unsigned char *out_; // Output buffer
		unsigned char *in_; // Input iterator
		unsigned char *term_; // End of input
		unsigned char *z; // Output iterator
		unsigned int c;

		_assert(mem->Ctx == nullptr || MutexEx::Held(mem->Ctx->Mutex));
		_assert(mem->Flags & MEM_Str);
		_assert(mem->Encode != desiredEncode);
		_assert(mem->Encode != 0);
		_assert(mem->N >= 0);

#if defined(TRANSLATE_TRACE) && defined(_DEBUG)
		{
			char buf[100];
			MemPrettyPrint(mem, buf);
			fprintf(stderr, "INPUT:  %s\n", buf);
		}
#endif

		// If the translation is between UTF-16 little and big endian, then all that is required is to swap the byte order. This case is handled differently from the others.
		if (mem->Encode != TEXTENCODE_UTF8 && desiredEncode != TEXTENCODE_UTF8)
		{
			RC rc = MemMakeWriteable(mem);
			if (rc != RC_OK)
			{
				_assert(rc == RC_NOMEM);
				return RC_NOMEM;
			}
			in_ = (uint8 *)mem->Z;
			term_ = &in_[mem->N&~1];
			while (in_ < term_)
			{
				c = *in_;
				*in_ = *(in_+1);
				in_++;
				*in_++ = c;
			}
			mem->Encode = desiredEncode;
			goto translate_out;
		}

		// Set len to the maximum number of bytes required in the output buffer.
		if (desiredEncode == TEXTENCODE_UTF8)
		{
			// When converting from UTF-16, the maximum growth results from translating a 2-byte character to a 4-byte UTF-8 character.
			// A single byte is required for the output string nul-terminator.
			mem->N &= ~1;
			len = mem->N * 2 + 1;
		}
		else
		{
			// When converting from UTF-8 to UTF-16 the maximum growth is caused when a 1-byte UTF-8 character is translated into a 2-byte UTF-16
			// character. Two bytes are required in the output buffer for the nul-terminator.
			len = mem->N * 2 + 2;
		}

		// Set zIn to point at the start of the input buffer and zTerm to point 1 byte past the end.
		// Variable zOut is set to point at the output buffer, space obtained from sqlite3_malloc().
		in_ = (uint8 *)mem->Z; // Input iterator
		term_ = &in_[mem->N]; // End of input
		out_ = (unsigned char *)_tagalloc(mem->Ctx, len); // Output buffer
		if (!out_)
			return RC_NOMEM;
		z = out_; // Output iterator

		if (mem->Encode == TEXTENCODE_UTF8)
		{
			if (desiredEncode == TEXTENCODE_UTF16LE)
			{
				// UTF-8 -> UTF-16 Little-endian
				while (in_ < term_)
				{
					READ_UTF8(in_, term_, c);
					WRITE_UTF16LE(z, c);
				}
			}
			else
			{
				_assert(desiredEncode == TEXTENCODE_UTF16BE);
				// UTF-8 -> UTF-16 Big-endian
				while (in_ < term_)
				{
					READ_UTF8(in_, term_, c);
					WRITE_UTF16BE(z, c);
				}
			}
			mem->N = (int)(z - out_);
			*z++ = 0;
		}
		else
		{
			_assert(desiredEncode == TEXTENCODE_UTF8);
			if (mem->Encode == TEXTENCODE_UTF16LE)
			{
				// UTF-16 Little-endian -> UTF-8
				while (in_ < term_)
				{
					READ_UTF16LE(in_, in_ < term_, c); 
					WRITE_UTF8(z, c);
				}
			}
			else
			{
				// UTF-16 Big-endian -> UTF-8 
				while (in_ < term_)
				{
					READ_UTF16BE(in_, in_ < term_, c); 
					WRITE_UTF8(z, c);
				}
			}
			mem->N = (int)(z - out_);
		}
		*z = 0;
		_assert((mem->N+(desiredEncode == TEXTENCODE_UTF8 ? 1 : 2)) <= len);

		MemRelease(mem);
		mem->Flags &= ~(MEM_Static|MEM_Dyn|MEM_Ephem);
		mem->Encode = desiredEncode;
		mem->Flags |= (MEM_Term|MEM_Dyn);
		mem->Z = (char *)out_;
		mem->Malloc = mem->Z;

translate_out:
#if defined(TRANSLATE_TRACE) && defined(_DEBUG)
		{
			char buf[100];
			MemPrettyPrint(mem, buf);
			fprintf(stderr, "OUTPUT: %s\n", buf);
		}
#endif
		return RC_OK;
	}

	__device__ int Vdbe::MemHandleBom(Mem *mem)
	{
		int rc = RC_OK;
		TEXTENCODE bom = (TEXTENCODE)0;
		_assert(mem->N >= 0);
		if (mem->N > 1)
		{
			uint8 b1 = *(uint8 *)mem->Z;
			uint8 b2 = *(((uint8 *)mem->Z) + 1);
			if (b1 == 0xFE && b2 == 0xFF)
				bom = TEXTENCODE_UTF16BE;
			if (b1 == 0xFF && b2 == 0xFE)
				bom = TEXTENCODE_UTF16LE;
		}
		if (bom)
		{
			rc = MemMakeWriteable(mem);
			if (rc == RC_OK)
			{
				mem->N -= 2;
				_memmove(mem->Z, &mem->Z[2], mem->N);
				mem->Z[mem->N] = '\0';
				mem->Z[mem->N+1] = '\0';
				mem->Flags |= MEM_Term;
				mem->Encode = bom;
			}
		}
		return rc;
	}


	__device__ char *Vdbe::Utf16to8(Context *ctx, const void *z, int bytes, TEXTENCODE encode)
	{
		Mem m;
		_memset(&m, 0, sizeof(m));
		m.Ctx = ctx;
		MemSetStr(&m, (const char *)z, bytes, encode, DESTRUCTOR_STATIC);
		ChangeEncoding(&m, TEXTENCODE_UTF8);
		if (ctx->MallocFailed)
		{
			MemRelease(&m);
			m.Z = nullptr;
		}
		_assert((m.Flags & MEM_Term) != 0 || ctx->MallocFailed);
		_assert((m.Flags & MEM_Str) != 0 || ctx->MallocFailed);
		_assert((m.Flags & MEM_Dyn) != 0 || ctx->MallocFailed);
		_assert(m.Z || ctx->MallocFailed);
		return m.Z;
	}

#ifdef ENABLE_STAT3
	__device__ char *Vdbe::Utf8to16(Context *ctx, TEXTENCODE encode, char *z, int n, int *out_)
	{
		Mem m;
		_memset(&m, 0, sizeof(m));
		m.Ctx = ctx;
		MemSetStr(&m, z, n, TEXTENCODE_UTF8, DESTRUCTOR_STATIC);
		if (MemTranslate(&m, encode))
		{
			_assert(ctx->MallocFailed);
			return nullptr;
		}
		_assert(m.Z == m.Malloc);
		*out_ = m.N;
		return m.Z;
	}
#endif

}
#endif