#include "RuntimeEx.h"
#if OS_WIN
#include <errno.h>
#include <io.h> // _findfirst and _findnext set errno iff they return -1
#endif

//////////////////////
// FUNC
#pragma region FUNC
#if __CUDACC__

/*
* Convert a string to a long integer.
*
* Ignores `locale' stuff.  Assumes that the upper and lower case alphabets and digits are each contiguous.
*/
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
		else
			acc = ULONG_MAX;
		//		errno = ERANGE;
	} else if (neg)
		acc = -acc;
	if (endptr != 0)
		*endptr = (char *)(any ? s - 1 : str);
	return acc;
}

//__device__ double _strtod(const char *str, char **endptr)
//{
//	return 0;
//}

#endif
#pragma endregion

//////////////////////
// OS
#pragma region OS
#if OS_GPU

__device__ DIR *_opendir(const char *)
{
	return nullptr;
}
__device__ int _closedir(DIR *)
{
	return 0;
}
__device__ struct dirent *_readdir(DIR *)
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
	struct dirent result; // d_name null iff first time
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

struct dirent *_readdir(DIR *dir)
{
	struct dirent *result = 0;
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
