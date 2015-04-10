#ifndef REG_HEADER_DEFINED
#define REG_HEADER_DEFINED

//http://pubs.opengroup.org/stage7tc1/basedefs/regex.h.html

struct slre_cap {
  const char *ptr;
  int len;
};


int slre_match(const char *regexp, const char *buf, int buf_len, struct slre_cap *caps, int num_caps, int flags);

enum { REG_IGNORE_CASE = 1 };

#define REG_NO_MATCH               -1
#define REG_UNEXPECTED_QUANTIFIER  -2
#define REG_UNBALANCED_BRACKETS    -3
#define REG_INTERNAL_ERROR         -4
#define REG_INVALID_CHARACTER_SET  -5
#define REG_INVALID_METACHARACTER  -6
#define REG_CAPS_ARRAY_TOO_SMALL   -7
#define REG_TOO_MANY_BRANCHES      -8
#define REG_TOO_MANY_BRACKETS      -9

#endif
