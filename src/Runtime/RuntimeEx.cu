#include "RuntimeEx.h"
#if OS_WIN
#include <errno.h>
#include <io.h> // _findfirst and _findnext set errno iff they return -1
#endif
//http://www.opensource.apple.com/source/Libc/Libc-167/

//////////////////////
// FUNC
#pragma region FUNC

// setjmp/longjmp
#if __CUDACC__
// TODO: BUILD
__device__ int _setjmp(jmp_buf xxenv)
{
	return 0;
}

// TODO: BUILD
__device__ void _longjmp(jmp_buf yyenv, int zzval)
{
}
#endif

// rand
#if __CUDACC__
// TODO: BUILD
__device__ int _rand()
{
	return 0;
}
#endif

// time
#if __CUDACC__
// TODO: BUILD
__device__ time_t _time(time_t *timer)
{
	//clock_t start = clock();
	time_t epoch = 0;
	return epoch;
}
#endif

// gettimeofday
#if __CUDACC__
// TODO: BUILD
__device__ int _gettimeofday(struct timeval *tp, void *tz)
{
	time_t seconds = _time(nullptr);
	tp->tv_usec = 0;
	tp->tv_sec = seconds;
	return 0;
	//if (tz)
	//	_abort();
	//tp->tv_usec = 0;
	//return (_time(&tp->tv_sec) == (time_t)-1 ? -1 : 0);
}
#else
#ifdef _MSC_VER
#include <sys/timeb.h>
__device__ int _gettimeofday(struct timeval *tp, void *unused)
{
	struct _timeb tb;
	_ftime(&tb);
	tp->tv_sec = (long)tb.time;
	tp->tv_usec = tb.millitm * 1000;
	return 0;
}
#endif
#endif

// sleep
#if __CUDACC__
__device__ void __sleep(unsigned long milliseconds)
{
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;)
	{
		clock_t now = clock();
		clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
		if (cycles >= end) break;
	}
}
#endif

// errno
#if __CUDACC__
__device__ int __errno;
// TODO: BUILD
__device__ char *__strerror(int errno_)
{
	return "ERROR";
}
#endif

// strtol_, strtoll_, strtoq_
#if __CUDACC__
#pragma warning(disable : 4146)
// Convert a string to a long integer.
//
// Ignores 'locale' stuff.  Assumes that the upper and lower case alphabets and digits are each contiguous.
__device__ unsigned long _strtol_(const char *str, char **endptr, register int base, bool signed_)
{
	register const char *s = str;
	// Skip white space and pick up leading +/- sign if any. If base is 0, allow 0x for hex and 0 for octal, else assume decimal; if base is already 16, allow 0x.
	register int neg = 0;
	register int c;
	do {
		c = *s++;
	} while (_isspace(c));
	if (c == '-') {
		neg = 1;
		c = *s++;
	} else if (c == '+')
		c = *s++;
	if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
		c = s[1];
		s += 2;
		base = 16;
	} else if ((base == 0 || base == 2) && c == '0' && (*s == 'b' || *s == 'B')) {
		c = s[1];
		s += 2;
		base = 2;
	}
	if (base == 0)
		base = (c == '0' ? 8 : 10);

	// Compute the cutoff value between legal numbers and illegal numbers.  That is the largest legal value, divided by the
	// base.  An input number that is greater than this value, if followed by a legal input character, is too big.  One that
	// is equal to this value may be valid or not; the limit between valid and invalid numbers is then based on the last
	// digit.  For instance, if the range for longs is [-2147483648..2147483647] and the input base is 10,
	// cutoff will be set to 214748364 and cutlim to either 7 (neg==0) or 8 (neg==1), meaning that if we have accumulated
	// a value > 214748364, or equal but the next digit is > 7 (or 8), the number is too big, and we will return a range error.
	//
	// Set any if any `digits' consumed; make it negative to indicate overflow.
	register unsigned long cutoff;
	register int cutlim;
	if (signed_)
	{
		cutoff = (neg ? -(unsigned long)LONG_MIN : LONG_MAX);
		cutlim = cutoff % (unsigned long)base;
		cutoff /= (unsigned long)base;
	} else {
		cutoff = (unsigned long)ULONG_MAX / (unsigned long)base;
		cutlim = (unsigned long)ULONG_MAX % (unsigned long)base;
	}

	register unsigned long acc;
	register int any;
	for (acc = 0, any = 0;; c = *s++) {
		if (_isdigit(c))
			c -= '0';
		else if (_isalpha(c))
			c -= (_isupper(c) ? 'A' - 10 : 'a' - 10);
		else
			break;
		if (c >= base)
			break;
		if (any < 0 || acc > cutoff || acc == cutoff && c > cutlim)
			any = -1;
		else {
			any = 1;
			acc *= base;
			acc += c;
		}
	}
	if (any < 0) {
		if (signed_)
			acc = (neg ? LONG_MIN : LONG_MAX);
		else {
			acc = ULONG_MAX;
			__errno = ERANGE;
		}
	} else if (neg)
		acc = -acc;
	if (endptr != 0)
		*endptr = (char *)(any ? s - 1 : str);
	return acc;
}

// Convert a string to a long long integer.
//
// Ignores `locale' stuff.  Assumes that the upper and lower case alphabets and digits are each contiguous.
__device__ unsigned long long _strtoll_(const char *str, char **endptr, register int base, bool signed_)
{
	register const char *s = str;
	// Skip white space and pick up leading +/- sign if any. If base is 0, allow 0x for hex and 0 for octal, else assume decimal; if base is already 16, allow 0x.
	register int neg = 0;
	register int c;
	do {
		c = *s++;
	} while (_isspace(c));
	if (c == '-') {
		neg = 1;
		c = *s++;
	} else if (c == '+')
		c = *s++;
	if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
		c = s[1];
		s += 2;
		base = 16;
	} else if ((base == 0 || base == 2) && c == '0' && (*s == 'b' || *s == 'B')) {
		c = s[1];
		s += 2;
		base = 2;
	}
	if (base == 0)
		base = (c == '0' ? 8 : 10);

	// Compute the cutoff value between legal numbers and illegal numbers.  That is the largest legal value, divided by the
	// base.  An input number that is greater than this value, if followed by a legal input character, is too big.  One that
	// is equal to this value may be valid or not; the limit between valid and invalid numbers is then based on the last
	// digit.  For instance, if the range for long longs is [-9223372036854775808..9223372036854775807] and the input base
	// is 10, cutoff will be set to 922337203685477580 and cutlim to either 7 (neg==0) or 8 (neg==1), meaning that if we have
	// accumulated a value > 922337203685477580, or equal but the next digit is > 7 (or 8), the number is too big, and we will return a range error.
	//
	// Set any if any `digits' consumed; make it negative to indicate overflow.
	register unsigned long long cutoff;
	register int cutlim;
	if (signed_)
	{
		cutoff = (neg ? -(unsigned long long)LLONG_MIN : LLONG_MAX);
		cutlim = cutoff % (unsigned long long)base;
		cutoff /= (unsigned long long)base;
	} else {
		cutoff = (unsigned long long)ULLONG_MAX / (unsigned long long)base;
		cutlim = (unsigned long long)ULLONG_MAX % (unsigned long long)base;
	}

	if (neg) {
		if (cutlim > 0) {
			cutlim -= base;
			cutoff += 1;
		}
		cutlim = -cutlim;
	}

	register unsigned long long acc;
	register int any;
	for (acc = 0, any = 0;; c = *s++) {
		if (_isdigit(c))
			c -= '0';
		else if (_isalpha(c))
			c -= (_isupper(c) ? 'A' - 10 : 'a' - 10);
		else
			break;
		if (c >= base)
			break;
		if (any < 0)
			continue;
		if (neg) {
			if (acc < cutoff || (acc == cutoff && c > cutlim)) {
				any = -1;
				acc = LLONG_MIN;
				__errno = ERANGE;
			} else {
				any = 1;
				acc *= base;
				acc -= c;
			}
		} else {
			if (acc > cutoff || (acc == cutoff && c > cutlim)) {
				any = -1;
				acc = LLONG_MAX;
				__errno = ERANGE;
			} else {
				any = 1;
				acc *= base;
				acc += c;
			}
		}
	}
	if (endptr != 0)
		*endptr = (char *)(any ? s - 1 : str);
	return acc;
}

// Convert a string to an unsigned quad integer.
//
// Ignores 'locale' stuff.  Assumes that the upper and lower case alphabets and digits are each contiguous.
__device__ u_quad_t _strtoq_(const char *str, char **endptr, register int base, bool signed_)
{
	register const char *s = str;
	// Skip white space and pick up leading +/- sign if any. If base is 0, allow 0x for hex and 0 for octal, else assume decimal; if base is already 16, allow 0x.
	register int neg = 0;
	register int c;
	do {
		c = *s++;
	} while (_isspace(c));
	if (c == '-')
	{
		neg = 1;
		c = *s++;
	}
	else if (c == '+')
		c = *s++;
	if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X'))
	{
		c = s[1];
		s += 2;
		base = 16;
	}
	if (base == 0)
		base = (c == '0' ? 8 : 10);

	// Compute the cutoff value between legal numbers and illegal numbers.  That is the largest legal value, divided by the
	// base.  An input number that is greater than this value, if followed by a legal input character, is too big.  One that
	// is equal to this value may be valid or not; the limit between valid and invalid numbers is then based on the last
	// digit.  For instance, if the range for quads is [-9223372036854775808..9223372036854775807] and the input base
	// is 10, cutoff will be set to 922337203685477580 and cutlim to either 7 (neg==0) or 8 (neg==1), meaning that if we have
	// accumulated a value > 922337203685477580, or equal but the next digit is > 7 (or 8), the number is too big, and we will
	// return a range error.
	//
	// Set any if any `digits' consumed; make it negative to indicate overflow.
	register u_quad_t qbase = (unsigned)base;
	register u_quad_t cutoff;
	register int cutlim;
	if (signed_)
	{
		cutoff = (neg ? -(u_quad_t)QUAD_MIN : QUAD_MAX);
		cutlim = (int)(cutoff % qbase);
		cutoff /= qbase;
	} else {
		cutoff = (u_quad_t)UQUAD_MAX / qbase;
		cutlim = (u_quad_t)UQUAD_MAX % qbase;
	}

	register u_quad_t acc;
	register int any;
	for (acc = 0, any = 0;; c = *s++) {
		if (_isdigit(c))
			c -= '0';
		else if (_isalpha(c))
			c -= (_isupper(c) ? 'A' - 10 : 'a' - 10);
		else
			break;
		if (c >= base)
			break;
		if (any < 0 || acc > cutoff || acc == cutoff && c > cutlim)
			any = -1;
		else {
			any = 1;
			acc *= qbase;
			acc += c;
		}
	}
	if (any < 0)
	{
		if (signed_) acc = (neg ? QUAD_MIN : QUAD_MAX);
		else acc = UQUAD_MAX;
		__errno = ERANGE;
	} else if (neg)
		acc = -acc;
	if (endptr != 0)
		*endptr = (char *)(any ? s - 1 : str);
	return acc;
}
#endif

// strtod
#if __CUDACC__
#define DBL_MAX_EXP 1024                    /* max binary exponent */
#define DBL_MIN_EXP (-1021)                 /* min binary exponent */
__device__ double _strtod(const char *str, char **endptr)
{
	// Skip leading whitespace
	char *p = (char *)str;
	while (_isspace(*p)) p++;

	// Handle optional sign
	int negative = 0;
	switch (*p)
	{
	case '-': negative = 1; // Fall through to increment position
	case '+': p++;
	}

	double number = 0.;
	int exponent = 0;
	int num_digits = 0;
	int num_decimals = 0;

	// Process string of digits
	while (_isdigit(*p))
	{
		number = number * 10. + (*p - '0');
		p++;
		num_digits++;
	}

	// Process decimal part
	if (*p == '.')
	{
		p++;
		while (_isdigit(*p))
		{
			number = number * 10. + (*p - '0');
			p++;
			num_digits++;
			num_decimals++;
		}
		exponent -= num_decimals;
	}
	if (num_digits == 0)
	{
		__errno = ERANGE;
		return 0.0;
	}

	// Correct for sign
	if (negative) number = -number;

	// Process an exponent string
	int n;
	if (*p == 'e' || *p == 'E')
	{
		// Handle optional sign
		negative = 0;
		switch (*++p)
		{
		case '-': negative = 1;   // Fall through to increment pos
		case '+': p++;
		}
		// Process string of digits
		n = 0;
		while (_isdigit(*p))
		{
			n = n * 10 + (*p - '0');
			p++;
		}
		if (negative) exponent -= n;
		else exponent += n;
	}

	if (exponent < DBL_MIN_EXP  || exponent > DBL_MAX_EXP)
	{
		__errno = ERANGE;
		return _HUGE_VAL;
	}

	// Scale the result
	double p10 = 10.;
	n = exponent;
	if (n < 0) n = -n;
	while (n)
	{
		if (n & 1)
		{
			if (exponent < 0) number /= p10;
			else number *= p10;
		}
		n >>= 1;
		p10 *= p10;
	}

	if (number == _HUGE_VAL) __errno = ERANGE;
	if (endptr) *endptr = p;
	return number;
}
#endif

// strrchr
#if __CUDACC__
__device__ char *_strrchr(const char *str, int ch)
{
	char *save;
	char c;
	for (save = (char *)0; (c = *str); str++)
		if (c == ch)
			save = (char *)str;
	return save;
}
#endif

// strpbrk
#if __CUDACC__
// Find the first occurrence in s1 of a character in s2 (excluding NUL).
__device__ char *_strpbrk(register const char *s1, register const char *s2)
{
	register const char *scanp;
	register int c, sc;
	while ((c = *s1++) != 0) {
		for (scanp = s2; (sc = *scanp++) != 0;)
			if (sc == c)
				return ((char *)(s1 - 1));
	}
	return nullptr;
}
#endif

// sscanf
#if __CUDACC__
#define	BUF		32 	// Maximum length of numeric string.

// Flags used during conversion.
#define	LONG		0x01	// l: long or double
#define	SHORT		0x04	// h: short
#define	SUPPRESS	0x08	// *: suppress assignment
#define	POINTER		0x10	// p: void * (as hex)
#define	NOSKIP		0x20	// [ or c: do not skip blanks
#define	LONGLONG	0x400	// ll: long long (+ deprecated q: quad)
#define	SHORTSHORT	0x4000	// hh: char
#define	UNSIGNED	0x8000	// %[oupxX] conversions

// The following are used in numeric conversions only:
// SIGNOK, NDIGITS, DPTOK, and EXPOK are for floating point;
// SIGNOK, NDIGITS, PFXOK, and NZDIGITS are for integral.
#define	SIGNOK		0x40	// +/- is (still) legal
#define	NDIGITS		0x80	// no digits detected
#define	DPTOK		0x100	// (float) decimal point is still legal
#define	EXPOK		0x200	// (float) exponent (e+3, etc) still legal
#define	PFXOK		0x100	// 0x prefix is (still) legal
#define	NZDIGITS	0x200	// no zero digits detected

// Conversion types.
#define	CT_CHAR		0	// %c conversion
#define	CT_CCL		1	// %[...] conversion
#define	CT_STRING	2	// %s conversion
#define	CT_INT		3	// %[dioupxX] conversion

__device__ static const char *__sccl(char *, const char *);

__constant__ static short _basefix[17] = { 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 'basefix' is used to avoid 'if' tests in the integer scanner
__device__ int _sscanf_(const char *str, const char *fmt, _va_list &args)
{
	int c; // character from format, or conversion
	size_t width; // field width, or 0
	char *p; // points into all kinds of strings
	int n; // handy integer
	int flags; // flags as defined above
	char *p0; // saves original value of p when necessary
	char ccltab[256]; // character class table for %[...]
	char buf[BUF]; // buffer for numeric conversions

	int nassigned = 0; // number of fields assigned
	int nconversions = 0; // number of conversions
	int nread = 0; // number of characters consumed from fp
	int base = 0; // base argument to conversion function

	int inr = _strlen(str);
	for (;;)
	{
		c = *fmt++;
		if (c == 0)
			return nassigned;
		if (_isspace(c))
		{
			while (inr > 0 && _isspace(*str)) nread++, inr--, str++;
			continue;
		}
		if (c != '%')
			goto literal_;
		width = 0;
		flags = 0;
		// switch on the format.  continue if done; break once format type is derived.
again:	c = *fmt++;
		switch (c) {
		case '%':
literal_:
			if (inr <= 0)
				goto input_failure;
			if (*str != c)
				goto match_failure;
			inr--, str++;
			nread++;
			continue;
		case '*':
			flags |= SUPPRESS;
			goto again;
		case 'l':
			if (flags & LONG) {
				flags &= ~LONG;
				flags |= LONGLONG;
			} else
				flags |= LONG;
			goto again;
		case 'q':
			flags |= LONGLONG; // not quite
			goto again;
		case 'h':
			if (flags & SHORT) {
				flags &= ~SHORT;
				flags |= SHORTSHORT;
			} else
				flags |= SHORT;
			goto again;

		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			width = width * 10 + c - '0';
			goto again;

			// Conversions.
		case 'd':
			c = CT_INT;
			base = 10;
			break;
		case 'i':
			c = CT_INT;
			base = 0;
			break;
		case 'o':
			c = CT_INT;
			flags |= UNSIGNED;
			base = 8;
			break;
		case 'u':
			c = CT_INT;
			flags |= UNSIGNED;
			base = 10;
			break;
		case 'X':
		case 'x':
			flags |= PFXOK;	// enable 0x prefixing
			c = CT_INT;
			flags |= UNSIGNED;
			base = 16;
			break;
		case 's':
			c = CT_STRING;
			break;
		case '[':
			fmt = __sccl(ccltab, fmt);
			flags |= NOSKIP;
			c = CT_CCL;
			break;
		case 'c':
			flags |= NOSKIP;
			c = CT_CHAR;
			break;
		case 'p': // pointer format is like hex
			flags |= POINTER | PFXOK;
			c = CT_INT;
			flags |= UNSIGNED;
			base = 16;
			break;
		case 'n':
			nconversions++;
			if (flags & SUPPRESS) continue; // ??? 
			if (flags & SHORTSHORT) *_va_arg(args, char *) = nread;
			else if (flags & SHORT) *_va_arg(args, short *) = nread;
			else if (flags & LONG) *_va_arg(args, long *) = nread;
			else if (flags & LONGLONG) *_va_arg(args, long long *) = nread;
			else *_va_arg(args, int *) = nread;
			continue;
		}

		// We have a conversion that requires input.
		if (inr <= 0)
			goto input_failure;

		// Consume leading white space, except for formats that suppress this.
		if ((flags & NOSKIP) == 0)
		{
			while (_isspace(*str))
			{
				nread++;
				if (--inr > 0) str++;
				else goto input_failure;
			}
			// Note that there is at least one character in the buffer, so conversions that do not set NOSKIP
			// can no longer result in an input failure.
		}

		// Do the conversion.
		switch (c) {
		case CT_CHAR: // scan arbitrary characters (sets NOSKIP)
			if (width == 0)
				width = 1;
			if (flags & SUPPRESS) {
				size_t sum = 0;
				for (;;)
				{
					if ((n = inr) < (int)width)
					{
						sum += n;
						width -= n;
						str += n;
						if (sum == 0)
							goto input_failure;
						break;
					}
					else
					{
						sum += width;
						inr -= width;
						str += width;
						break;
					}
				}
				nread += sum;
			}
			else
			{
				_memcpy(_va_arg(args, char *), str, width);
				inr -= width;
				str += width;
				nread += width;
				nassigned++;
			}
			nconversions++;
			break;
		case CT_CCL: // scan a (nonempty) character class (sets NOSKIP)
			if (width == 0)
				width = (size_t)~0;	// 'infinity'
			// take only those things in the class
			if (flags & SUPPRESS)
			{
				n = 0;
				while (ccltab[(unsigned char)*str])
				{
					n++, inr--, str++;
					if (--width == 0) break;
					if (inr <= 0) {
						if (n == 0)
							goto input_failure;
						break;
					}
				}
				if (n == 0)
					goto match_failure;
			}
			else
			{
				p0 = p = _va_arg(args, char *);
				while (ccltab[(unsigned char)*str])
				{
					inr--;
					*p++ = *str++;
					if (--width == 0) break;
					if (inr <= 0) {
						if (p == p0)
							goto input_failure;
						break;
					}
				}
				n = p - p0;
				if (n == 0)
					goto match_failure;
				*p = 0;
				nassigned++;
			}
			nread += n;
			nconversions++;
			break;
		case CT_STRING: // like CCL, but zero-length string OK, & no NOSKIP
			if (width == 0)
				width = (size_t)~0;
			if (flags & SUPPRESS) {
				n = 0;
				while (!_isspace(*str))
				{
					n++, inr--, str++;
					if (--width == 0) break;
					if (inr <= 0) break;
				}
				nread += n;
			} else {
				p0 = p = _va_arg(args, char *);
				while (!_isspace(*str))
				{
					inr--;
					*p++ = *str++;
					if (--width == 0) break;
					if (inr <= 0) break;
				}
				*p = 0;
				nread += p - p0;
				nassigned++;
			}
			nconversions++;
			continue;
		case CT_INT: // scan an integer as if by the conversion function
#ifdef hardway
			if (width == 0 || width > sizeof(buf) - 1)
				width = sizeof(buf) - 1;
#else
			// size_t is unsigned, hence this optimisation
			if (--width > sizeof(buf) - 2)
				width = sizeof(buf) - 2;
			width++;
#endif
			flags |= SIGNOK | NDIGITS | NZDIGITS;
			for (p = buf; width; width--) {
				c = *str;
				// Switch on the character; `goto ok' if we accept it as a part of number.
				switch (c) {
				case '0':
					// The digit 0 is always legal, but is special.  For %i conversions, if no digits (zero or nonzero) have been
					// scanned (only signs), we will have base==0.  In that case, we should set it to 8 and enable 0x prefixing.
					// Also, if we have not scanned zero digits before this, do not turn off prefixing (someone else will turn it off if we
					// have scanned any nonzero digits).
					if (base == 0) {
						base = 8;
						flags |= PFXOK;
					}
					if (flags & NZDIGITS) flags &= ~(SIGNOK|NZDIGITS|NDIGITS);
					else flags &= ~(SIGNOK|PFXOK|NDIGITS);
					goto ok;
				case '1': case '2': case '3': // 1 through 7 always legal
				case '4': case '5': case '6': case '7':
					base = _basefix[base];
					flags &= ~(SIGNOK | PFXOK | NDIGITS);
					goto ok;
				case '8': case '9': // digits 8 and 9 ok iff decimal or hex
					base = _basefix[base];
					if (base <= 8) break; // not legal here
					flags &= ~(SIGNOK | PFXOK | NDIGITS);
					goto ok;
				case 'A': case 'B': case 'C': // letters ok iff hex
				case 'D': case 'E': case 'F':
				case 'a': case 'b': case 'c':
				case 'd': case 'e': case 'f':
					// no need to fix base here
					if (base <= 10) break; // not legal here
					flags &= ~(SIGNOK | PFXOK | NDIGITS);
					goto ok;
				case '+': case '-': // sign ok only as first character
					if (flags & SIGNOK) {
						flags &= ~SIGNOK;
						goto ok;
					}
					break;
				case 'x': case 'X': // x ok iff flag still set & 2nd char
					if (flags & PFXOK && p == buf + 1) {
						base = 16; // if %i
						flags &= ~PFXOK;
						goto ok;
					}
					break;
				}
				// If we got here, c is not a legal character for a number.  Stop accumulating digits.
				break;
ok:
				// c is legal: store it and look at the next.
				*p++ = c;
				if (--inr > 0)
					str++;
				else 
					break; // end of input
			}
			// If we had only a sign, it is no good; push back the sign.  If the number ends in `x',
			// it was [sign] '0' 'x', so push back the x and treat it as [sign] '0'.
			if (flags & NDIGITS) {
				if (p > buf) {
					str--;
					inr++;
				}
				goto match_failure;
			}
			c = ((char *)p)[-1];
			if (c == 'x' || c == 'X') {
				--p;
				str--;
				inr++;
			}
			if ((flags & SUPPRESS) == 0) {
				quad_t res;
				*p = 0;
				if ((flags & UNSIGNED) == 0) res = _strtoq_(buf, (char **)NULL, base, true);
				else res = _strtoq_(buf, (char **)NULL, base, false);
				if (flags & POINTER) *_va_arg(args, void **) = (void *)(intptr_t)res;
				else if (flags & SHORTSHORT) *_va_arg(args, char *) = res;
				else if (flags & SHORT) *_va_arg(args, short *) = res;
				else if (flags & LONG) *_va_arg(args, long *) = res;
				else if (flags & LONGLONG) *_va_arg(args, long long *) = res;
				else *_va_arg(args, int *) = res;
				nassigned++;
			}
			nread += p - buf;
			nconversions++;
			break;
		}
	}
input_failure:
	return (nconversions != 0 ? nassigned : -1);
match_failure:
	return nassigned;
}

__device__ static const char *__sccl(char *tab, const char *fmt)
{
	// first 'clear' the whole table
	int c, n, v;
	c = *fmt++; // first char hat => negated scanset
	if (c == '^')
	{
		v = 1; // default => accept
		c = *fmt++; // get new first char
	} else
		v = 0; // default => reject 
	_memset(tab, v, 256); // XXX: Will not work if sizeof(tab*) > sizeof(char)
	if (c == 0)
		return (fmt - 1); // format ended before closing ]
	// Now set the entries corresponding to the actual scanset to the opposite of the above.
	// The first character may be ']' (or '-') without being special; the last character may be '-'.
	v = 1 - v;
	for (;;)
	{
		tab[c] = v; // take character c
doswitch:
		n = *fmt++; // and examine the next
		switch (n)
		{
		case 0: // format ended too soon
			return (fmt - 1);
		case '-':
			// A scanset of the form [01+-]
			// is defined as `the digit 0, the digit 1, the character +, the character -', but
			// the effect of a scanset such as [a-zA-Z0-9]
			// is implementation defined.  The V7 Unix scanf treats `a-z' as `the letters a through
			// z', but treats `a-a' as `the letter a, the character -, and the letter a'.
			//
			// For compatibility, the `-' is not considerd to define a range if the character following
			// it is either a close bracket (required by ANSI) or is not numerically greater than the character
			// we just stored in the table (c).
			n = *fmt;
			if (n == ']' || n < c)
			{
				c = '-';
				break; // resume the for(;;)
			}
			fmt++;
			// fill in the range
			do
			{
				tab[++c] = v;
			} while (c < n);
			c = n;
			// Alas, the V7 Unix scanf also treats formats such as [a-c-e] as `the letters a through e'. This too is permitted by the standard....
			goto doswitch;
			//break;
		case ']': // end of scanset
			return fmt;
		default:
			// just another character
			c = n;
			break;
		}
	}
}

#endif

// qsort
#if __CUDACC__

#define min(a, b) ((a) < (b) ? a : b)
#define swapcode(TYPE, parmi, parmj, n) {\
	long i = (n) / sizeof (TYPE);\
	register TYPE *pi = (TYPE *)(parmi);\
	register TYPE *pj = (TYPE *)(parmj);\
	do { register TYPE t = *pi; *pi++ = *pj; *pj++ = t; } while (--i > 0);\
}
#define SWAPINIT(a, es) swaptype = (((char*)a-(char*)0)%sizeof(long)||es%sizeof(long)?2:(es==sizeof(long)?0:1));
__device__ static inline void swapfunc(char *a, char *b, int n, int swaptype)
{
	if (swaptype <= 1) swapcode(long, a, b, n)
	else swapcode(char, a, b, n)
}
#define swap(a, b)\
	if (swaptype == 0) { long t = *(long *)(a); *(long *)(a) = *(long *)(b); *(long *)(b) = t; }\
	else swapfunc(a, b, es, swaptype)
#define vecswap(a, b, n) if ((n) > 0) swapfunc(a, b, n, swaptype)
__device__ static inline char *med3(char *a, char *b, char *c, int (*cmp)(const void*,const void*))
{
	return (cmp(a, b)<0 ? (cmp(b, c)<0?b:(cmp(a, c)<0?c:a)) : (cmp(b, c)>0?b:(cmp(a, c)<0?a:c)));
}

__device__ void _qsort(void *base, size_t n, size_t es, int (*cmp)(const void*,const void*))
{
	char *a = (char *)base;
	char *pa, *pb, *pc, *pd, *pl, *pm, *pn;
	int d, r, swaptype, swap_cnt;
loop:
	SWAPINIT(a, es);
	swap_cnt = 0;
	if (n < 7)
	{
		for (pm = a + es; pm < (char *)a + n * es; pm += es)
			for (pl = pm; pl > (char *)a && cmp(pl - es, pl) > 0; pl -= es)
				swap(pl, pl - es);
		return;
	}
	pm = a + (n / 2) * es;
	if (n > 7)
	{
		pl = a;
		pn = a + (n - 1) * es;
		if (n > 40)
		{
			d = (n / 8) * es;
			pl = med3(pl, pl + d, pl + 2 * d, cmp);
			pm = med3(pm - d, pm, pm + d, cmp);
			pn = med3(pn - 2 * d, pn - d, pn, cmp);
		}
		pm = med3(pl, pm, pn, cmp);
	}
	swap(a, pm);
	pa = pb = a + es;
	//
	pc = pd = a + (n - 1) * es;
	for (;;)
	{
		while (pb <= pc && (r = cmp(pb, a)) <= 0)
		{
			if (r == 0)
			{
				swap_cnt = 1;
				swap(pa, pb);
				pa += es;
			}
			pb += es;
		}
		while (pb <= pc && (r = cmp(pc, a)) >= 0)
		{
			if (r == 0)
			{
				swap_cnt = 1;
				swap(pc, pd);
				pd -= es;
			}
			pc -= es;
		}
		if (pb > pc)
			break;
		swap(pb, pc);
		swap_cnt = 1;
		pb += es;
		pc -= es;
	}
	if (swap_cnt == 0) // Switch to insertion sort
	{  
		for (pm = a + es; pm < (char *)a + n * es; pm += es)
			for (pl = pm; pl > (char *)a && cmp(pl - es, pl) > 0; pl -= es)
				swap(pl, pl - es);
		return;
	}
	//
	pn = a + n * es;
	r = min(pa - (char *)a, pb - pa);
	vecswap(a, pb - r, r);
	r = min(pd - pc, pn - pd - es);
	vecswap(pb, pn - r, r);
	if ((r = pb - pa) > es)
		_qsort(a, r / es, es, cmp);
	if ((r = pd - pc) > es)
	{
		// Iterate rather than recurse to save stack space
		a = pn - r;
		n = r / es;
		goto loop;
	}
	/*qsort(pn - r, r / es, es, cmp);*/
}

#endif

// div
#if __CUDACC__
__device__ div_t _div(int num, int denom)
{
	div_t r;
	r.quot = num / denom;
	r.rem = num % denom;
	if (num >= 0 && r.rem < 0)
	{
		r.quot++;
		r.rem -= denom;
	}
	return r;
}
#endif

#pragma endregion

//////////////////////
// OS
#pragma region OS

#if __CUDACC__
__device__ char *__environ[2] = { "HOME=", "PATH=" }; // pointer to environment table
__device__ char *_getenv(const char *name)
{
	//if (!_strcmp(name, "HOME")) return "gpu:\\";
	//if (!_strcmp(name, "PATH")) return "gpu:\\";
	return nullptr;
}

__device__ void _setenv(const char *name, const char *value)
{
}
#endif

#if __CUDACC__
__device__ int __chmod(const char *a, mode_t m)
{
	return 0;
}
__device__ int __rmdir(const char *a)
{
	return 0;
}
__device__ int __mkdir(const char *a, mode_t m)
{
	return 0;
}
__device__ int __mkfifo(const char *a, mode_t m)
{
	return 0;
}
__device__ int __stat(const char *a, struct stat *b)
{
	return 0;
}
__device__ char *__getcwd(char *b, int l)
{
	return "GPU";
}
__device__ int __chdir(const char *p)
{
	return 0;
}
__device__ int __access(const char *p, int flags)
{
	return 0;
}
#endif

#if OS_GPU

__device__ DIR *_opendir(const char *)
{
	return nullptr;
}
__device__ int _closedir(DIR *)
{
	return 0;
}
__device__ struct _dirent *_readdir(DIR *)
{
	return nullptr;
}
__device__ void _rewinddir(DIR *)
{
}

#elif OS_WIN

typedef ptrdiff_t handle_type; // C99's intptr_t not sufficiently portable

struct DIR
{
	handle_type handle; // -1 for failed rewind
	struct _finddata_t info;
	struct _dirent result; // d_name null iff first time
	char *name;  // null-terminated char string
};

DIR *_opendir(const char *name)
{
	DIR *dir = 0;
	if (name && name[0])
	{
		size_t base_length = strlen(name);
		const char *all = (strchr("/\\", name[base_length - 1]) ? "*" : "/*"); // search pattern must end with suitable wildcard
		if ((dir = (DIR *)malloc(sizeof *dir)) != 0 && (dir->name = (char *)malloc(base_length + strlen(all) + 1)) != 0)
		{
			strcat(strcpy(dir->name, name), all);
			if ((dir->handle = (handle_type)_findfirst(dir->name, &dir->info)) != -1)
				dir->result.d_name = 0;
			else // rollback
			{
				free(dir->name);
				free(dir);
				dir = 0;
			}
		}
		else // rollback
		{
			free(dir);
			dir = 0;
			errno = ENOMEM;
		}
	}
	else
		errno = EINVAL;
	return dir;
}

int _closedir(DIR *dir)
{
	int result = -1;
	if (dir)
	{
		if (dir->handle != -1)
			result = _findclose(dir->handle);
		free(dir->name);
		free(dir);
	}
	if (result == -1) // map all errors to EBADF
		errno = EBADF;
	return result;
}

struct _dirent *_readdir(DIR *dir)
{
	struct _dirent *result = 0;
	if (dir && dir->handle != -1)
	{
		if (!dir->result.d_name || _findnext(dir->handle, &dir->info) != -1)
		{
			result = &dir->result;
			result->d_name = dir->info.name;
		}
	}
	else
		errno = EBADF;
	return result;
}

void _rewinddir(DIR *dir)
{
	if (dir && dir->handle != -1)
	{
		_findclose(dir->handle);
		dir->handle = (handle_type)_findfirst(dir->name, &dir->info);
		dir->result.d_name = 0;
	}
	else
		errno = EBADF;
}

#endif
#pragma endregion


