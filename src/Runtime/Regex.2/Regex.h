#if 0
#ifndef __REGEX_H__
#define __REGEX_H__

typedef struct regex_t regex_t;

typedef struct
{
	const char_t *begin;
	int len;
} regmatch_t;

_API __device__ regex_t *reg_compile(const char_t *pattern, const char_t **error);
_API __device__ void reg_free(regex_t *exp);
_API __device__ bool reg_match(regex_t* exp, const char_t* text);
_API __device__ bool reg_search(regex_t* exp, const char_t* text, const char_t** out_begin, const char_t** out_end);
_API __device__ bool reg_searchrange(regex_t* exp, const char_t* text_begin, const char_t* text_end, const char_t** out_begin, const char_t** out_end);
_API __device__ int reg_getsubexpcount(regex_t* exp);
_API __device__ bool reg_getsubexp(regex_t* exp, int n, regmatch_t *subexp);

#endif
#endif