#if 0
#ifndef __REGEX_H__
#define __REGEX_H__
typedef struct
{
} regex_t;

typedef int regoff_t;
struct regmatch_t
{
	regoff_t rm_so;    // byte offset from start of string to start of substring
	regoff_t rm_eo;    // byte offset from start of string of the first character after the end of substring

	const char *ptr;
	int len;
};

__device__ int reg_match(const char *regexp, const char *buf, int buf_len, struct regmatch_t *caps, int num_caps, int flags);
__device__ char *reg_replace(const char *regex, const char *buf, const char *sub);

#define REG_IGNORE_CASE				1
//#define REG_NOMATCH				-1
#define REG_UNEXPECTED_QUANTIFIER  -2
#define REG_UNBALANCED_BRACKETS    -3
#define REG_INTERNAL_ERROR         -4
#define REG_INVALID_CHARACTER_SET  -5
#define REG_INVALID_METACHARACTER  -6
#define REG_CAPS_ARRAY_TOO_SMALL   -7
#define REG_TOO_MANY_BRANCHES      -8
#define REG_TOO_MANY_BRACKETS      -9





// Values for the cflags parameter to the regcomp() function:
#define REG_EXTENDED	91 // Use Extended Regular Expressions.
#define REG_ICASE		92 // Ignore case in match.
#define REG_NOSUB		93 // Report only success or fail in regexec().
#define REG_NEWLINE		94 // Change the handling of newline.

// Values for the eflags parameter to the regexec() function:
#define REG_NOTBOL		91 // The circumflex character (^), when taken as a special character, will not match the beginning of string.
#define REG_NOTEOL		92 // The dollar sign ($), when taken as a special character, will not match the end of string.

// The following constants are defined as error return values:
#define REG_NOMATCH		91 // regexec() failed to match.
#define REG_BADPAT		92 // Invalid regular expression.
#define REG_ECOLLATE	93 // Invalid collating element referenced.
#define REG_ECTYPE		94 // Invalid character class type referenced.
#define REG_EESCAPE		95 // Trailing \ in pattern.
#define REG_ESUBREG		96 // Number in \digit invalid or in error.
#define REG_EBRACK		97 // [ ] imbalance.
#define REG_EPAREN		98 // \( \) or ( ) imbalance.
#define REG_EBRACE		99 // \{ \} imbalance.
#define REG_BADBR		100 // Content of \{ \} invalid: not a number, number too large, more than two numbers, first larger than second.
#define REG_ERANGE		101 // Invalid endpoint in range expression.
#define REG_ESPACE		102 // Out of memory.
#define REG_BADRPT		103 // ?, * or + not preceded by valid regular expression.
#define REG_ENOSYS		104 // The implementation does not support the function.

__device__ int regcomp(regex_t *, const char *, int);
__device__ int regexec(const regex_t *, const char *, size_t, regmatch_t[], int);
__device__ size_t regerror(int, const regex_t *, char *, size_t);
__device__ void regfree(regex_t *);

#endif
#endif