#ifndef __REGEX_H__
#define __REGEX_H__

typedef struct
{
} regex_t;

struct regmatch_t
{
  const char *ptr;
  int len;
};

__device__ int reg_match(const char *regexp, const char *buf, int buf_len, struct regmatch_t *caps, int num_caps, int flags);
__device__ char *reg_replace(const char *regex, const char *buf, const char *sub);

#define REG_IGNORE_CASE				1
#define REG_NOMATCH				   -1
#define REG_UNEXPECTED_QUANTIFIER  -2
#define REG_UNBALANCED_BRACKETS    -3
#define REG_INTERNAL_ERROR         -4
#define REG_INVALID_CHARACTER_SET  -5
#define REG_INVALID_METACHARACTER  -6
#define REG_CAPS_ARRAY_TOO_SMALL   -7
#define REG_TOO_MANY_BRANCHES      -8
#define REG_TOO_MANY_BRACKETS      -9

#endif
