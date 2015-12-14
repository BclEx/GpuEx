#ifndef REGEX_H
#define REGEX_H
#include "Runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
	{
		int rm_so;
		int rm_eo;
	} regmatch_t;

	typedef struct regexp
	{
		int re_nsub;		// number of parenthesized subexpressions

		// private
		int cflags;			// Flags used when compiling
		int err;			// Any error which occurred during compile
		int regstart;		// Internal use only
		int reganch;		// Internal use only
		int regmust;		// Internal use only
		int regmlen;		// Internal use only
		int *program;		// Allocated
		// working state - compile
		const char *regparse;	// Input-scan pointer
		int p;					// Current output pos in program
		int proglen;			// Allocated program size
		// working state - exec
		int eflags;				// Flags used when executing
		const char *start;		// Initial string pointer
		const char *reginput;	// Current input pointer
		const char *regbol;		// Beginning of input, for ^ check
		// Input to regexec()
		regmatch_t *pmatch;		// submatches will be stored here
		int nmatch;				// size of pmatch[]
	} regex_t;

#define REG_EXTENDED 0
#define REG_NEWLINE 1
#define REG_ICASE 2
#define REG_NOTBOL 16

	enum
	{
		REG_NOERROR,      // Success
		REG_NOMATCH,      // Didn't find a match (for regexec)
		REG_BADPAT,		  // >= REG_BADPAT is an error
		REG_ERR_NULL_ARGUMENT,
		REG_ERR_UNKNOWN,
		REG_ERR_TOO_BIG,
		REG_ERR_NOMEM,
		REG_ERR_TOO_MANY_PAREN,
		REG_ERR_UNMATCHED_PAREN,
		REG_ERR_UNMATCHED_BRACES,
		REG_ERR_BAD_COUNT,
		REG_ERR_JUNK_ON_END,
		REG_ERR_OPERAND_COULD_BE_EMPTY,
		REG_ERR_NESTED_COUNT,
		REG_ERR_INTERNAL,
		REG_ERR_COUNT_FOLLOWS_NOTHING,
		REG_ERR_TRAILING_BACKSLASH,
		REG_ERR_CORRUPTED,
		REG_ERR_NULL_CHAR,
		REG_ERR_NUM
	};

	__device__ int regcomp(regex_t *preg, const char *regex, int cflags);
	__device__ int regexec(regex_t *preg,  const  char *string, size_t nmatch, regmatch_t pmatch[], int eflags);
	__device__ size_t regerror(int errcode, const regex_t *preg, char *errbuf, size_t errbuf_size);
	__device__ void regfree(regex_t *preg);

#ifdef __cplusplus
}
#endif
#endif
