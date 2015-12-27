#if 0
#include "Runtime.h"
#include "Regex.h"

__device__ int regcomp(regex_t *a, const char *b, int c)
{
	return 0;
}

__device__ int regexec(const regex_t *a, const char *b, size_t c, regmatch_t *d, int e)
{
	return 0;
}

__device__ size_t regerror(int a, const regex_t * b, char * c, size_t d)
{
	return 0;
}

__device__ void regfree(regex_t * a)
{
}
#endif