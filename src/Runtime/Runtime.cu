#define __EMBED__ 1
#include "Runtime.h"
#include "RuntimeTypes.h"
#define _NEVER(X) (X)

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
//__device__ static cudaRuntime __curt = { -2 };
//__device__ static int (*cudaRuntimeInitialize)(cudaRuntime *curt) = nullptr;
__device__ unsigned char __one;

//////////////////////
// UTF
#pragma region UTF

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
	*z++ = (unsigned char)(c&0xFF); \
	} else if (c < 0x00800) { \
	*z++ = 0xC0 + (unsigned char)((c>>6)&0x1F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} else if (c < 0x10000) { \
	*z++ = 0xE0 + (unsigned char)((c>>12)&0x0F); \
	*z++ = 0x80 + (unsigned char)((c>>6)&0x3F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} else { \
	*z++ = 0xF0 + (unsigned char)((c>>18)&0x07); \
	*z++ = 0x80 + (unsigned char)((c>>12)&0x3F); \
	*z++ = 0x80 + (unsigned char)((c>>6)&0x3F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} \
	}

#define WRITE_UTF16LE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (unsigned char)(c&0x00FF); \
	*z++ = (unsigned char)((c>>8)&0x00FF); \
	} else { \
	*z++ = (unsigned char)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (unsigned char)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (unsigned char)(c&0x00FF); \
	*z++ = (unsigned char)(0x00DC + ((c>>8)&0x03)); \
	} \
	}

#define WRITE_UTF16BE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (unsigned char)((c>>8)&0x00FF); \
	*z++ = (unsigned char)(c&0x00FF); \
	} else { \
	*z++ = (unsigned char)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (unsigned char)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (unsigned char)(0x00DC + ((c>>8)&0x03)); \
	*z++ = (unsigned char)(c&0x00FF); \
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

__device__ unsigned int _utf8read(const unsigned char **z)
{
	// Same as READ_UTF8() above but without the zTerm parameter. For this routine, we assume the UTF8 string is always zero-terminated.
	unsigned int c = *((*z)++);
	if (c >= 0xc0)
	{
		c = _utf8Trans1[c-0xc0];
		while ((*(*z) & 0xc0) == 0x80)
			c = (c<<6) + (0x3f & *((*z)++));
		if (c < 0x80 || (c&0xFFFFF800) == 0xD800 || (c&0xFFFFFFFE) == 0xFFFE) c = 0xFFFD;
	}
	return c;
}

__device__ int _utf8charlength(const char *z, int bytes)
{
	const char *term = (bytes >= 0 ? &z[bytes] : (const char *)-1);
	_assert(z <= term);
	int r = 0;
	while (*z != 0 && z < term)
	{
		_strskiputf8(z);
		r++;
	}
	return r;
}

#if _DEBUG
__device__ int _utf8to8(unsigned char *z)
{
	unsigned char *z2 = z;
	unsigned char *start = z;
	while (z[0] && z2 <= z)
	{
		unsigned int c = _utf8read((const unsigned char **)&z);
		if (c != 0xfffd)
			WRITE_UTF8(z2, c);
	}
	*z2 = 0;
	return (int)(z2 - start);
}
#endif

#ifndef OMIT_UTF16
__device__ int _utf16bytelength(const void *z, int chars)
{
	int c;
	unsigned char const *z2 = (unsigned char const *)z;
	int n = 0;
	if (TYPE_BIGENDIAN)
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

#ifdef TEST
__device__ void _runtime_utfselftest()
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
		c = _utf8read((const char **)&z);
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
		n = (int)(z - buf);
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

#pragma endregion

//////////////////////
// FUNC
#pragma region FUNC

#ifndef OMIT_BLOB_LITERAL
__device__ void *_taghextoblob(void *tag, const char *z, size_t size)
{
	char *b = (char *)_tagalloc(tag, size / 2 + 1);
	size--;
	if (b)
	{
		int bIdx = 0;
		for (int i = 0; i < size; i += 2, bIdx++)
			b[bIdx] = (_hextobyte(z[i]) << 4) | _hextobyte(z[i + 1]);
		b[bIdx] = 0;
	}
	return b;
}
#endif

#pragma endregion

//////////////////////
// STDARG

//////////////////////
// PRINT
#pragma region PRINT

#ifndef PRINT_BUF_SIZE
#define PRINT_BUF_SIZE 70
#endif
#define BUFSIZE PRINT_BUF_SIZE  // Size of the output buffer

enum TYPE : unsigned char
{
	TYPE_RADIX = 1,			// Integer types.  %d, %x, %o, and so forth
	TYPE_FLOAT = 2,			// Floating point.  %f
	TYPE_EXP = 3,			// Exponentional notation. %e and %E
	TYPE_GENERIC = 4,		// Floating or exponential, depending on exponent. %g
	TYPE_SIZE = 5,			// Return number of characters processed so far. %n
	TYPE_STRING = 6,		// Strings. %s
	TYPE_DYNSTRING = 7,		// Dynamically allocated strings. %z
	TYPE_PERCENT = 8,		// Percent symbol. %%
	TYPE_CHARX = 9,			// Characters. %c
	// The rest are extensions, not normally found in printf()
	TYPE_SQLESCAPE = 10,	// Strings with '\'' doubled.  %q
	TYPE_SQLESCAPE2 = 11,	// Strings with '\'' doubled and enclosed in '', NULL pointers replaced by SQL NULL.  %Q
	TYPE_TOKEN = 12,		// a pointer to a Token structure
	TYPE_SRCLIST = 13,		// a pointer to a SrcList
	TYPE_POINTER = 14,		// The %p conversion
	TYPE_SQLESCAPE3 = 15,	// %w -> Strings with '\"' doubled
	TYPE_ORDINAL = 16,		// %r -> 1st, 2nd, 3rd, 4th, etc.  English only
	//
	TYPE_INVALID = 0,		// Any unrecognized conversion type
};

enum FLAG : unsigned char
{
	FLAG_SIGNED = 1,	// True if the value to convert is signed
	FLAG_INTERN = 2,	// True if for internal use only
	FLAG_STRING = 4,	// Allow infinity precision
};

// Each builtin conversion character (ex: the 'd' in "%d") is described by an instance of the following structure
struct Info
{   // Information about each format field
	char Fmttype; // The format field code letter
	unsigned char Base; // The base for radix conversion
	FLAG Flags; // One or more of FLAG_ constants below
	TYPE Type; // Conversion paradigm
	unsigned char Charset; // Offset into aDigits[] of the digits string
	unsigned char Prefix; // Offset into aPrefix[] of the prefix string
};

// The following table is searched linearly, so it is good to put the most frequently used conversion types first.
__device__ static const char _digits[] = "0123456789ABCDEF0123456789abcdef";
__device__ static const char _prefix[] = "-x0\000X0";
__device__ static const Info _info[] = {
	{ 'd', 10, (FLAG)1, TYPE_RADIX,      0,  0 },
	{ 's',  0, (FLAG)4, TYPE_STRING,     0,  0 },
	{ 'g',  0, (FLAG)1, TYPE_GENERIC,    30, 0 },
	{ 'z',  0, (FLAG)4, TYPE_DYNSTRING,  0,  0 },
	{ 'q',  0, (FLAG)4, TYPE_SQLESCAPE,  0,  0 },
	{ 'Q',  0, (FLAG)4, TYPE_SQLESCAPE2, 0,  0 },
	{ 'w',  0, (FLAG)4, TYPE_SQLESCAPE3, 0,  0 },
	{ 'c',  0, (FLAG)0, TYPE_CHARX,      0,  0 },
	{ 'o',  8, (FLAG)0, TYPE_RADIX,      0,  2 },
	{ 'u', 10, (FLAG)0, TYPE_RADIX,      0,  0 },
	{ 'x', 16, (FLAG)0, TYPE_RADIX,      16, 1 },
	{ 'X', 16, (FLAG)0, TYPE_RADIX,      0,  4 },
#ifndef OMIT_FLOATING_POINT
	{ 'f',  0, (FLAG)1, TYPE_FLOAT,      0,  0 },
	{ 'e',  0, (FLAG)1, TYPE_EXP,        30, 0 },
	{ 'E',  0, (FLAG)1, TYPE_EXP,        14, 0 },
	{ 'G',  0, (FLAG)1, TYPE_GENERIC,    14, 0 },
#endif
	{ 'i', 10, (FLAG)1, TYPE_RADIX,      0,  0 },
	{ 'n',  0, (FLAG)0, TYPE_SIZE,       0,  0 },
	{ '%',  0, (FLAG)0, TYPE_PERCENT,    0,  0 },
	{ 'p', 16, (FLAG)0, TYPE_POINTER,    0,  1 },
	// All the rest have the FLAG_INTERN bit set and are thus for internal use only
	{ 'T',  0, (FLAG)2, TYPE_TOKEN,      0,  0 },
	{ 'S',  0, (FLAG)2, TYPE_SRCLIST,    0,  0 },
	{ 'r', 10, (FLAG)3, TYPE_ORDINAL,    0,  0 },
};

#ifndef OMIT_FLOATING_POINT
__device__ static char GetDigit(double64 *val, int *cnt)
{
	if ((*cnt) <= 0) return '0';
	(*cnt)--;
	int digit = (int)*val;
	double64 d = digit;
	digit += '0';
	*val = (*val - d)*10.0;
	return (char)digit;
}
#endif

__constant__ static const char _spaces[] = "                             ";
__device__ void TextBuilder::AppendSpace(int length)
{
	while (length >= (int)sizeof(_spaces)-1)
	{
		Append(_spaces, sizeof(_spaces)-1);
		length -= sizeof(_spaces)-1;
	}
	if (length > 0)
		Append(_spaces, length);
}

__constant__ static const char _ord[] = "thstndrd";
__device__ void TextBuilder::AppendFormat(bool useExtended, const char *fmt, va_list *args) //: was: vxprintf
{
	char buf[BUFSIZE]; // Conversion buffer
	char *bufpt = nullptr; // Pointer to the conversion buffer
	int c; // Next character in the format string
	bool flag_leftjustify = false; // True if "-" flag is present
	int width = 0; // Width of the current field
	int length = 0; // Length of the field
	for (; (c = (*fmt)) != 0; ++fmt)
	{
		if (c != '%')
		{
			bufpt = (char *)fmt;
			int amt = 1;
			while ((c = (*++fmt)) != '%' && c != 0) amt++;
			Append(bufpt, amt);
			if (c == 0) break;
		}
		if ((c = (*++fmt)) == 0)
		{
			Append("%", 1);
			break;
		}
		// Find out what flags are present
		flag_leftjustify = false; // True if "-" flag is present
		bool flag_plussign = false; // True if "+" flag is present
		bool flag_blanksign = false; // True if " " flag is present
		bool flag_alternateform = false; // True if "#" flag is present
		bool flag_altform2 = false; // True if "!" flag is present
		bool flag_zeropad = false; // True if field width constant starts with zero
		bool done = false; // Loop termination flag
		do
		{
			switch (c)
			{
			case '-': flag_leftjustify = true; break;
			case '+': flag_plussign = true; break;
			case ' ': flag_blanksign = true; break;
			case '#': flag_alternateform = true; break;
			case '!': flag_altform2 = true; break;
			case '0': flag_zeropad = true; break;
			default: done = true; break;
			}
		} while (!done && (c = (*++fmt)) != 0);
		// Get the field width
		width = 0; // Width of the current field
		if (c == '*')
		{
			width = va_arg(args, int);
			if (width < 0)
			{
				flag_leftjustify = true;
				width = -width;
			}
			c = *++fmt;
		}
		else
		{
			while (c >= '0' && c <= '9')
			{
				width = width*10 + c - '0';
				c = *++fmt;
			}
		}
		// Get the precision
		int precision; // Precision of the current field
		if (c == '.')
		{
			precision = 0;
			c = *++fmt;
			if (c == '*')
			{
				precision = va_arg(args, int);
				if (precision < 0) precision = -precision;
				c = *++fmt;
			}
			else
			{
				while (c >= '0' && c <= '9')
				{
					precision = precision*10 + c - '0';
					c = *++fmt;
				}
			}
		}
		else
			precision = -1;
		// Get the conversion type modifier
		bool flag_long; // True if "l" flag is present
		bool flag_longlong; // True if the "ll" flag is present
		if (c == 'l')
		{
			flag_long = true;
			c = *++fmt;
			if (c == 'l')
			{
				flag_longlong = true;
				c = *++fmt;
			}
			else
				flag_longlong = false;
		}
		else
			flag_long = flag_longlong = false;
		// Fetch the info entry for the field
		const Info *info = &_info[0]; // Pointer to the appropriate info structure
		TYPE type = TYPE_INVALID; // Conversion paradigm
		int i;
		for (i = 0; i < _lengthof(_info); i++)
		{
			if (c == _info[i].Fmttype)
			{
				info = &_info[i];
				if (useExtended || (info->Flags & FLAG_INTERN) == 0) type = info->Type;
				else return;
				break;
			}
		}

		char prefix; // Prefix character.  "+" or "-" or " " or '\0'.
		unsigned long long longvalue; // Value for integer types
		double64 realvalue; // Value for real types
#ifndef OMIT_FLOATING_POINT
		int exp, e2; // exponent of real numbers
		int nsd; // Number of significant digits returned
		double rounder; // Used for rounding floating point values
		bool flag_dp; // True if decimal point should be shown
		bool flag_rtz; // True if trailing zeros should be removed
#endif

		// At this point, variables are initialized as follows:
		//   flag_alternateform          TRUE if a '#' is present.
		//   flag_altform2               TRUE if a '!' is present.
		//   flag_plussign               TRUE if a '+' is present.
		//   flag_leftjustify            TRUE if a '-' is present or if the field width was negative.
		//   flag_zeropad                TRUE if the width began with 0.
		//   flag_long                   TRUE if the letter 'l' (ell) prefixed the conversion character.
		//   flag_longlong               TRUE if the letter 'll' (ell ell) prefixed the conversion character.
		//   flag_blanksign              TRUE if a ' ' is present.
		//   width                       The specified field width.  This is always non-negative.  Zero is the default.
		//   precision                   The specified precision.  The default is -1.
		//   type                        The class of the conversion.
		//   info                        Pointer to the appropriate info struct.
		char *extra = nullptr; // Malloced memory used by some conversion
		char *out_; // Rendering buffer
		int outLength; // Size of the rendering buffer
		switch (type)
		{
		case TYPE_POINTER:
			flag_longlong = (sizeof(char *) == sizeof(long long));
			flag_long = (sizeof(char *) == sizeof(long int));
			// Fall through into the next case
		case TYPE_ORDINAL:
		case TYPE_RADIX:
			if (info->Flags & FLAG_SIGNED)
			{
				long long v;
				if (flag_longlong) v = va_arg(args, long long);
				else if (flag_long) v = va_arg(args, long int);
				else v = va_arg(args, int);
				if (v < 0)
				{
					longvalue = (v == SMALLEST_INT64 ? ((unsigned long long)1)<<63 : -v);
					prefix = '-';
				}
				else
				{
					longvalue = v;
					if (flag_plussign) prefix = '+';
					else if (flag_blanksign) prefix = ' ';
					else prefix = '\0';
				}
			}
			else
			{
				if (flag_longlong) longvalue = va_arg(args, unsigned long long);
				else if (flag_long) longvalue = va_arg(args, unsigned long int);
				else longvalue = va_arg(args, unsigned int);
				prefix = 0;
			}
			if (longvalue == 0) flag_alternateform = false;
			if (flag_zeropad && precision < width - (prefix != '\0'))
				precision = width-(prefix!=0);
			if (precision < BUFSIZE-10)
			{
				outLength = BUFSIZE;
				out_ = buf;
			}
			else
			{
				outLength = precision + 10;
				out_ = extra = (char *)_alloc(outLength);
				if (!out_)
				{
					AllocFailed = true;
					return;
				}
			}
			bufpt = &out_[outLength-1];
			if (type == TYPE_ORDINAL)
			{
				int x = (int)(longvalue % 10);
				if (x >= 4 || (longvalue/10)%10 == 1) x = 0;
				*(--bufpt) = _ord[x*2+1];
				*(--bufpt) = _ord[x*2];
			}
			{
				register const char *cset = &_digits[info->Charset]; // Use registers for speed
				register int base = info->Base;
				do // Convert to ascii
				{                                           
					*(--bufpt) = cset[longvalue % base];
					longvalue = longvalue / base;
				} while(longvalue > 0);
			}
			length = (int)(&out_[outLength-1]-bufpt);
			for (i = precision - length; i > 0; i--) *(--bufpt) = '0'; // Zero pad
			if (prefix) *(--bufpt) = prefix; // Add sign
			if (flag_alternateform && info->Prefix) // Add "0" or "0x"
			{
				char x;
				const char *pre = &_prefix[info->Prefix];
				for (; (x = (*pre)) != 0; pre++) *(--bufpt) = x;
			}
			length = (int)(&out_[outLength-1]-bufpt);
			break;
		case TYPE_FLOAT:
		case TYPE_EXP:
		case TYPE_GENERIC:
			realvalue = va_arg(args, double);
#ifdef OMIT_FLOATING_POINT
			length = 0;
#else
			if (precision < 0) precision = 6; // Set default precision
			if (realvalue < 0.0)
			{
				realvalue = -realvalue;
				prefix = '-';
			}
			else
			{
				if (flag_plussign) prefix = '+';
				else if (flag_blanksign) prefix = ' ';
				else prefix = 0;
			}
			if (type == TYPE_GENERIC && precision > 0) precision--;
#if 0
			// Rounding works like BSD when the constant 0.4999 is used.  Wierd!
			for (i = precision, rounder = 0.4999; i > 0; i--, rounder *= 0.1);
#else
			// It makes more sense to use 0.5
			for (i = precision, rounder = 0.5; i > 0; i--, rounder *= 0.1) { }
#endif
			if (type == TYPE_FLOAT) realvalue += rounder;
			// Normalize realvalue to within 10.0 > realvalue >= 1.0
			exp = 0;
			if (_isnan((double)realvalue))
			{
				bufpt = "NaN";
				length = 3;
				break;
			}
			if (realvalue > 0.0)
			{
				double64 scale = 1.0;
				while (realvalue >= 1e100*scale && exp <= 350) { scale *= 1e100;exp += 100; }
				while (realvalue >= 1e64*scale && exp <= 350) { scale *= 1e64; exp += 64; }
				while (realvalue >= 1e8*scale && exp <= 350) { scale *= 1e8; exp += 8; }
				while (realvalue >= 10.0*scale && exp <= 350) { scale *= 10.0; exp++; }
				realvalue /= scale;
				while (realvalue < 1e-8) { realvalue *= 1e8; exp -= 8; }
				while (realvalue < 1.0) { realvalue *= 10.0; exp--; }
				if (exp > 350)
				{
					if (prefix == '-') bufpt = "-Inf";
					else if (prefix == '+') bufpt = "+Inf";
					else bufpt = "Inf";
					length = _strlen30(bufpt);
					break;
				}
			}
			bufpt = buf;
			// If the field type is etGENERIC, then convert to either etEXP or etFLOAT, as appropriate.
			if (type != TYPE_FLOAT)
			{
				realvalue += rounder;
				if (realvalue >= 10.0) { realvalue *= 0.1; exp++; }
			}
			if (type == TYPE_GENERIC)
			{
				flag_rtz = !flag_alternateform;
				if (exp < -4 || exp > precision) type = TYPE_EXP;
				else { precision = precision - exp; type = TYPE_FLOAT; }
			}
			else
				flag_rtz = flag_altform2;
			e2 = (type == TYPE_EXP ? 0 : exp);
			if (e2+precision+width > BUFSIZE - 15)
			{
				bufpt = extra = (char *)_alloc(e2+precision+width+15);
				if (!bufpt)
				{
					AllocFailed = true;
					return;
				}
			}
			out_ = bufpt;
			nsd = 16 + flag_altform2*10;
			flag_dp = (precision > 0) | flag_alternateform | flag_altform2;
			// The sign in front of the number
			if (prefix) *(bufpt++) = prefix;
			// Digits prior to the decimal point
			if (e2 < 0) *(bufpt++) = '0';
			else for (; e2 >= 0; e2--) *(bufpt++) = GetDigit(&realvalue, &nsd);
			// The decimal point
			if (flag_dp) *(bufpt++) = '.';
			// "0" digits after the decimal point but before the first significant digit of the number
			for (e2++; e2 < 0; precision--, e2++) { _assert(precision > 0); *(bufpt++) = '0'; }
			// Significant digits after the decimal point
			while ((precision--) > 0) *(bufpt++) = GetDigit(&realvalue, &nsd);
			// Remove trailing zeros and the "." if no digits follow the "."
			if (flag_rtz && flag_dp)
			{
				while (bufpt[-1] == '0') *(--bufpt) = 0;
				_assert(bufpt > out_);
				if (bufpt[-1] == '.')
				{
					if (flag_altform2) *(bufpt++) = '0';
					else *(--bufpt) = 0;
				}
			}
			// Add the "eNNN" suffix
			if (type == TYPE_EXP)
			{
				*(bufpt++) = _digits[info->Charset];
				if (exp < 0) { *(bufpt++) = '-'; exp = -exp; }
				else *(bufpt++) = '+';
				if (exp >= 100) { *(bufpt++) = (char)((exp/100)+'0'); exp %= 100; } // 100's digit
				*(bufpt++) = (char)(exp/10+'0'); // 10's digit
				*(bufpt++) = (char)(exp%10+'0'); // 1's digit
			}
			*bufpt = 0;

			// The converted number is in buf[] and zero terminated. Output it. Note that the number is in the usual order, not reversed as with integer conversions.
			length = (int)(bufpt-out_);
			bufpt = out_;

			// Special case:  Add leading zeros if the flag_zeropad flag is set and we are not left justified
			if (flag_zeropad && !flag_leftjustify && length < width)
			{
				int pad = width - length;
				for (i = width; i >= pad; i--) bufpt[i] = bufpt[i-pad];
				i = (prefix != '\0');
				while (pad--) bufpt[i++] = '0';
				length = width;
			}
#endif
			break;
		case TYPE_SIZE:
			*(va_arg(args, int*)) = Size;
			length = width = 0;
			break;
		case TYPE_PERCENT:
			buf[0] = '%';
			bufpt = buf;
			length = 1;
			break;
		case TYPE_CHARX:
			c = va_arg(args, int);
			buf[0] = (char)c;
			if (precision >= 0)
			{
				for (i = 1; i < precision; i++) buf[i] = (char)c;
				length = precision;
			}
			else length =1;
			bufpt = buf;
			break;
		case TYPE_STRING:
		case TYPE_DYNSTRING:
			bufpt = va_arg(args, char*);
			if (bufpt == 0) bufpt = "";
			else if (type == TYPE_DYNSTRING) extra = bufpt;
			if (precision >= 0) for (length = 0; length < precision && bufpt[length]; length++) { }
			else length = _strlen30(bufpt);
			break;
		case TYPE_SQLESCAPE:
		case TYPE_SQLESCAPE2:
		case TYPE_SQLESCAPE3: {
			char q = (type == TYPE_SQLESCAPE3 ? '"' : '\''); // Quote character
			char *escarg = va_arg(args, char*);
			bool isnull = (escarg == 0);
			if (isnull) escarg = (type == TYPE_SQLESCAPE2 ? "NULL" : "(NULL)");
			int k = precision;
			int j, n;
			char ch;
			for (i = n = 0; k != 0 && (ch = escarg[i]) != 0; i++, k--)
				if (ch == q) n++;
			bool needQuote = (!isnull && type == TYPE_SQLESCAPE2);
			n += i + 1 + needQuote*2;
			if (n > BUFSIZE)
			{
				bufpt = extra = (char *)_alloc(n);
				if (!bufpt)
				{
					AllocFailed = true;
					return;
				}
			}
			else
				bufpt = buf;
			j = 0;
			if (needQuote) bufpt[j++] = q;
			k = i;
			for (i = 0; i < k; i++)
			{
				bufpt[j++] = ch = escarg[i];
				if (ch == q) bufpt[j++] = ch;
			}
			if (needQuote) bufpt[j++] = q;
			bufpt[j] = 0;
			length = j;
			// The precision in %q and %Q means how many input characters to consume, not the length of the output...
			// if (precision>=0 && precision<length) length = precision;
			break; }

							  //case TYPE_TOKEN: {
							  //	Token *token = va_arg(args, Token*);
							  //	if (token) Append((const char *)token->z, token->n);
							  //	length = width = 0;
							  //	break; }
							  //case TYPE_SRCLIST: {
							  //	SrcList *src = va_arg(args, SrcList*);
							  //	int k = va_arg(args, int);
							  //	SrcList::SrcListItem *item = &src->Ids[k];
							  //	_assert(k >= 0 && k < src->Srcs);
							  //	if (item->DatabaseName)
							  //	{
							  //		Append(item->DatabaseName, -1);
							  //		Append(".", 1);
							  //	}
							  //	Append(item->Name, -1);
							  //	length = width = 0;
							  //	break; }
		default: {
			_assert(type == TYPE_INVALID);
			return; }
		}
		// The text of the conversion is pointed to by "bufpt" and is "length" characters long.  The field width is "width".  Do the output.
		if (!flag_leftjustify)
		{
			register int nspace = width-length;
			if (nspace > 0) AppendSpace(nspace);
		}
		if (length > 0) Append(bufpt, length);
		if (flag_leftjustify)
		{
			register int nspace = width-length;
			if (nspace > 0) AppendSpace(nspace);
		}
		if (extra != nullptr)
			_free(extra);
	}
}

__device__ void TextBuilder::Append(const char *z, int length)
{
	_assert(z != nullptr || length == 0);
	if (Overflowed | AllocFailed)
	{
		ASSERTCOVERAGE(Overflowed);
		ASSERTCOVERAGE(AllocFailed);
		return;
	}
	_assert(Text != nullptr || Index == 0);
	if (length < 0)
		length = _strlen30(z);
	if (length == 0 || _NEVER(z == nullptr))
		return;
	if (Index + length >= Size)
	{
		char *newText;
		if (!AllocType)
		{
			Overflowed = true;
			length = Size - Index - 1;
			if (length <= 0)
				return;
		}
		else
		{
			char *oldText = (Text == Base ? nullptr : Text);
			long long newSize = Index;
			newSize += length + 1;
			if (newSize > MaxSize)
			{
				Reset();
				Overflowed = true;
				return;
			}
			else
				Size = (int)newSize;
			if (AllocType == 1)
				newText = (char *)_tagrealloc(Tag, oldText, Size);
			else
				newText = (char *)_realloc(oldText, Size);
			if (newText)
			{
				if (oldText == nullptr && Index > 0) _memcpy(newText, Text, Index);
				Text = newText;
			}
			else
			{
				AllocFailed = true;
				Reset();
				return;
			}
		}
	}
	_assert(Text != nullptr);
	_memcpy(&Text[Index], z, length);
	Index += length;
}

__device__ char *TextBuilder::ToString()
{
	if (Text)
	{
		Text[Index] = 0;
		if (AllocType && Text == Base)
		{
			if (AllocType == 1)
				Text = (char *)_tagalloc(Tag, Index + 1);
			else
				Text = (char *)_alloc(Index + 1);
			if (Text)
				_memcpy(Text, Base, Index + 1);
			else
				AllocFailed = true;
		}
	}
	return Text;
}

__device__ void TextBuilder::Reset()
{
	if (Text != Base)
	{
		if (AllocType == 1)
			_tagfree(Tag, Text);
		else
			_free(Text);
	}
	Text = nullptr;
}

__device__ void TextBuilder::Init(TextBuilder *b, char *text, int capacity, int maxSize)
{
	b->Text = b->Base = text;
	b->Tag = nullptr;
	b->Index = 0;
	b->Size = capacity;
	b->MaxSize = maxSize;
	b->AllocType = 1;
	b->Overflowed = false;
	b->AllocFailed = false;
}

__device__ char *_vmtagprintf(void *tag, const char *fmt, va_list *args, int *length)
{
	//if (!RuntimeInitialize()) return nullptr;
	_assert(tag != nullptr);
	char base[PRINT_BUF_SIZE];
	TextBuilder b;
	TextBuilder::Init(&b, base, sizeof(base), 0); //? tag->Limit[LIMIT_LENGTH]);
	b.Tag = tag;
	b.AppendFormat(true, fmt, args);
	if (length) *length = b.Index;
	char *z = b.ToString();
	//? if (b.AllocFailed) _tagallocfailed(tag);
	return z;
}

__device__ char *_vmprintf(const char *fmt, va_list *args, int *length)
{
	//if (!RuntimeInitialize()) return nullptr;
	char base[PRINT_BUF_SIZE];
	TextBuilder b;
	TextBuilder::Init(&b, base, sizeof(base), CORE_MAX_LENGTH);
	b.AllocType = 2;
	b.AppendFormat(false, fmt, args);
	if (length) *length = b.Index;
	return b.ToString();
}

__device__ char *__vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list *args, int *length)
{
	if (bufLen <= 0) return (char *)buf;
	TextBuilder b;
	TextBuilder::Init(&b, (char *)buf, (int)bufLen, 0);
	b.AllocType = 0;
	b.AppendFormat(false, fmt, args);
	if (length) *length = b.Index;
	return b.ToString();
}

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL
#ifdef VISUAL
#if __CUDACC__
#include "RuntimeHost.h"

//#define MAX(a,b) (a > b ? a : b)
#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, .7, 1)
#define BLOCK2COLOR make_float4(0, 0, 1, 1)
#define MARKERCOLOR make_float4(1, 1, 0, 1)

__global__ static void RenderHeap(struct runtimeHeap_s *heap, quad4 *b, unsigned int offset)
{
	int index = offset;
	// heap
	b[index] = make_quad4(
		make_float4(00, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 0, 1, 1), HEADERCOLOR,
		make_float4(00, 0, 1, 1), HEADERCOLOR);
	// free
	float x1, y1;
	if (heap->blockPtr)
	{
		size_t offset = ((char *)heap->blockPtr - (char *)heap->blocks);
		offset %= heap->blocksLength;
		offset /= heap->blockSize;
		//
		unsigned int x = offset % BLOCKPITCH;
		unsigned int y = offset / BLOCKPITCH;
		x1 = x * 10; y1 = y * 20 + 2;
		b[index + 1] = make_quad4(
			make_float4(x1 + 0, y1 + 1, 1, 1), MARKERCOLOR,
			make_float4(x1 + 1, y1 + 1, 1, 1), MARKERCOLOR,
			make_float4(x1 + 1, y1 + 0, 1, 1), MARKERCOLOR,
			make_float4(x1 + 0, y1 + 0, 1, 1), MARKERCOLOR);
	}
}

__global__ static void RenderBlock(struct runtimeHeap_s *heap, quad4 *b, size_t blocks, unsigned int blocksY, unsigned int offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int blockIndex = y * BLOCKPITCH + x;
	if (blockIndex >= blocks)
		return;
	runtimeBlockHeader *hdr = (runtimeBlockHeader *)(heap->blocks + blockIndex * heap->blockSize);
	int index = blockIndex * 2 + offset;
	// block
	float x2 = x * 10; float y2 = y * 20 + 2;
	if (hdr->magic != RUNTIME_MAGIC || hdr->fmtoffset >= heap->blockSize)
	{
		b[index] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCKCOLOR);
	}
	else
	{
		b[index] = make_quad4(
			make_float4(x2 + 0, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 0, 1, 1), HEADERCOLOR,
			make_float4(x2 + 0, y2 + 0, 1, 1), HEADERCOLOR);
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK2COLOR);
	}
}

__device__ static int _printf(const char *fmt);
__global__ static void Keypress(struct runtimeHeap_s *heap, unsigned char key)
{
	_runtimeSetHeap(heap);
	switch (key)
	{
	case 'a': _printf("Test\n"); break;
	case 'A': printf("Test\n"); break;
	case 'b': _printf("Test %d\n", threadIdx.x); break;
	case 'B': printf("Test %d\n", threadIdx.x); break;
	case 'c': _assert(true); break;
	case 'C': assert(true); break;
	case 'd': _assert(false); break;
	case 'D': assert(false); break;
	case 'e': _transfer("test", 1); break;
	case 'f': _throw("test", 1); break;
	case 'g': {
		va_list args;
		va_start(args, 1, "2");
		int a1 = va_arg(args, int);
		char *a2 = va_arg(args, char*);
		va_end(args);
		break; }
	}
}

inline size_t GetRuntimeRenderQuads(size_t blocks)
{ 
	return 2 + (blocks * 2);
}

static void LaunchRuntimeRender(cudaDeviceHeap &host, float4 *b, size_t blocks)
{
	cudaCheckErrors(cudaDeviceHeapSelect(host), exit(0));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	RenderHeap<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, (quad4 *)b, 0);
	//
	dim3 blockBlock(16, 16, 1);
	dim3 blockGrid((unsigned int)MAX(BLOCKPITCH / 16, 1), (unsigned int)MAX(blocks / BLOCKPITCH / 16, 1), 1);
	RenderBlock<<<blockGrid, blockBlock>>>((runtimeHeap *)host.heap, (quad4 *)b, blocks, (unsigned int)blocks / BLOCKPITCH, 2);
}

static void LaunchRuntimeKeypress(cudaDeviceHeap &host, unsigned char key)
{
	if (key == 'z')
	{
		cudaDeviceHeapSynchronize(host);
		return;
	}
	cudaCheckErrors(cudaDeviceHeapSelect(host), exit(0));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, key);
}

// _vbo variables
static GLuint _runtimeVbo;
static GLsizei _runtimeVboSize;
static struct cudaGraphicsResource *_runtimeVboResource;

static void RuntimeRunCuda(cudaDeviceHeap &host, size_t blocks, struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	cudaCheckErrors(cudaGraphicsMapResources(1, resource, nullptr), exit(0));
	float4 *b;
	size_t size;
	cudaCheckErrors(cudaGraphicsResourceGetMappedPointer((void **)&b, &size, *resource), exit(0));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	LaunchRuntimeRender(host, b, blocks);
	// unmap buffer object
	cudaCheckErrors(cudaGraphicsUnmapResources(1, resource, nullptr), exit(0));
}

static void RuntimeCreateVBO(size_t blocks, GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer object
	_runtimeVboSize = (GLsizei)GetRuntimeRenderQuads(blocks) * 4;
	unsigned int size = _runtimeVboSize * 2 * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// register this buffer object with CUDA
	cudaCheckErrors(cudaGraphicsGLRegisterBuffer(resource, *vbo, vbo_res_flags), exit(0));
	SDK_CHECK_ERROR_GL();
}

static void RuntimeDeleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
{
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void RuntimeVisualRender::Dispose()
{
	if (_runtimeVbo)
		RuntimeDeleteVBO(&_runtimeVbo, _runtimeVboResource);
}

void RuntimeVisualRender::Keyboard(unsigned char key)
{
	LaunchRuntimeKeypress(_runtimeHost, key);
}

void RuntimeVisualRender::Display()
{
	size_t blocks = _runtimeHost.blocksLength / _runtimeHost.blockSize;
	// run CUDA kernel to generate vertex positions
	RuntimeRunCuda(_runtimeHost, blocks, &_runtimeVboResource);

	//gluLookAt(0, 0, 200, 0, 0, 0, 0, 1, 0);
	//glScalef(.02, .02, .02);
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(Visual::TranslateX, Visual::TranslateY, Visual::TranslateZ);
	glRotatef(Visual::RotateX, 1.0, 0.0, 0.0);
	glRotatef(Visual::RotateY, 0.0, 1.0, 0.0);

	// render from the _vbo
	glBindBuffer(GL_ARRAY_BUFFER, _runtimeVbo);
	glVertexPointer(4, GL_FLOAT, sizeof(float4) * 2, 0);
	glColorPointer(4, GL_FLOAT, sizeof(float4) * 2, (GLvoid*)sizeof(float4));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_QUADS, 0, _runtimeVboSize);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void RuntimeVisualRender::Initialize()
{
	size_t blocks = _runtimeHost.blocksLength / _runtimeHost.blockSize;
	// create VBO
	RuntimeCreateVBO(blocks, &_runtimeVbo, &_runtimeVboResource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	RuntimeRunCuda(_runtimeHost, blocks, &_runtimeVboResource);
}

#undef MAX
#undef BLOCKPITCH
#undef HEADERPITCH
#undef BLOCKREFCOLOR
#undef HEADERCOLOR
#undef BLOCKCOLOR
#undef BLOCK2COLOR
#undef MARKERCOLOR

#else
void RuntimeVisualRender::Dispose() { }
void RuntimeVisualRender::Keyboard(unsigned char key) { }
void RuntimeVisualRender::Display() { }
void RuntimeVisualRender::Initialize() { }
#endif
#endif
#pragma endregion
