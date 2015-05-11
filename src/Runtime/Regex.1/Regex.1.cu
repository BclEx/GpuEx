//#include <stdio.h>
//#include <ctype.h>
//#include <string.h>
#include "Runtime.h"
#include "Regex.h"

#define MAX_BRANCHES 100
#define MAX_BRACKETS 100
#define FAILIF(condition, error_code) if (condition) return (error_code)

#ifdef REG_DEBUG
#define DBG(fmt, ...) _printf(x, __VA_ARGS__)
#else
#define DBG(fmt, ...)
#endif

struct bracket_pair
{
	const char *ptr;  // Points to the first char after '(' in regex
	int len;          // Length of the text between '(' and ')'
	int branches;     // Index in the branches array for this pair
	int num_branches; // Number of '|' in this bracket pair
};

struct branch
{
	int bracket_index;    //* index for 'struct bracket_pair brackets' array defined below
	const char *schlong;  //* points to the '|' character in the regex
};

struct regex_info
{
	// Describes all bracket pairs in the regular expression. First entry is always present, and grabs the whole regex.
	struct bracket_pair brackets[MAX_BRACKETS];
	int num_brackets;

	// Describes alternations ('|' operators) in the regular expression. Each branch falls into a specific branch pair.
	struct branch branches[MAX_BRANCHES];
	int num_branches;

	// Array of captures provided by the user
	struct regmatch_t *caps;
	int num_caps;

	// E.g. REG_IGNORE_CASE. See enum below
	int flags;
};

__constant__ static const char *_metacharacters = "^$().[]*+?|\\Ssdbfnrtv";
__device__ static int is_metacharacter(const char *s)
{
	return (!_strchr(_metacharacters, *s));
}

__device__ static int op_len(const char *re)
{
	return (re[0] == '\\' && re[1] == 'x' ? 4 : re[0] == '\\' ? 2 : 1);
}

__device__ static int set_len(const char *re, int re_len)
{
	int len = 0;
	while (len < re_len && re[len] != ']')
		len += op_len(re + len);
	return (len <= re_len ? len + 1 : -1);
}

__device__ static int get_op_len(const char *re, int re_len)
{
	return (re[0] == '[' ? set_len(re + 1, re_len - 1) + 1 : op_len(re));
}

__device__ static int is_quantifier(const char *re)
{
	return (re[0] == '*' || re[0] == '+' || re[0] == '?');
}

__device__ static int toi(int x)
{
	return (_isdigit(x) ? x - '0' : x - 'W');
}

__device__ static int hextoi(const unsigned char *s)
{
	return (toi(_tolower(s[0])) << 4) | toi(_tolower(s[1]));
}

__device__ static int match_op(const unsigned char *re, const unsigned char *s, struct regex_info *info) {
	int result = 0;
	switch (*re)
	{
	case '\\':
		// Metacharacters
		switch (re[1])
		{
		case 'S': FAILIF(_isspace(*s), REG_NOMATCH); result++; break;
		case 's': FAILIF(!_isspace(*s), REG_NOMATCH); result++; break;
		case 'd': FAILIF(!_isdigit(*s), REG_NOMATCH); result++; break;
		case 'b': FAILIF(*s != '\b', REG_NOMATCH); result++; break;
		case 'f': FAILIF(*s != '\f', REG_NOMATCH); result++; break;
		case 'n': FAILIF(*s != '\n', REG_NOMATCH); result++; break;
		case 'r': FAILIF(*s != '\r', REG_NOMATCH); result++; break;
		case 't': FAILIF(*s != '\t', REG_NOMATCH); result++; break;
		case 'v': FAILIF(*s != '\v', REG_NOMATCH); result++; break;

		case 'x':
			// Match byte, \xHH where HH is hexadecimal byte representaion
			FAILIF(hextoi(re + 2) != *s, REG_NOMATCH);
			result++;
			break;

		default:
			// Valid metacharacter check is done in bar()
			FAILIF(re[1] != s[0], REG_NOMATCH);
			result++;
			break;
		}
		break;

	case '|': FAILIF(1, REG_INTERNAL_ERROR); break;
	case '$': FAILIF(1, REG_NOMATCH); break;
	case '.': result++; break;

	default:
		if (info->flags & REG_IGNORE_CASE)
			FAILIF(_tolower(*re) != _tolower(*s), REG_NOMATCH);
		else
			FAILIF(*re != *s, REG_NOMATCH);
		result++;
		break;
	}

	return result;
}

__device__ static int match_set(const char *re, int re_len, const char *s, struct regex_info *info)
{
	int len = 0, result = -1, invert = re[0] == '^';
	if (invert) re++, re_len--;

	while (len <= re_len && re[len] != ']' && result <= 0)
	{
		// Support character range
		if (re[len] != '-' && re[len + 1] == '-' && re[len + 2] != ']' && re[len + 2] != '\0')
		{
			result = (info->flags && REG_IGNORE_CASE ? *s >= re[len] && *s <= re[len + 2] : _tolower(*s) >= _tolower(re[len]) && _tolower(*s) <= _tolower(re[len + 2]));
			len += 3;
		} 
		else
		{
			result = match_op((unsigned char *)re + len, (unsigned char *)s, info);
			len += op_len(re + len);
		}
	}
	return ((!invert && result > 0) || (invert && result <= 0) ? 1 : -1);
}

__device__ static int doh(const char *s, int s_len, struct regex_info *info, int bi);

__device__ static int bar(const char *re, int re_len, const char *s, int s_len, struct regex_info *info, int bi)
{
	// i is offset in re, j is offset in s, bi is brackets index
	int i, j, n, step;
	for (i = j = 0; i < re_len && j <= s_len; i += step)
	{

		// Handle quantifiers. Get the length of the chunk.
		step = (re[i] == '(' ? info->brackets[bi + 1].len + 2 : get_op_len(re + i, re_len - i));

		DBG("%s [%.*s] [%.*s] re_len=%d step=%d i=%d j=%d\n", __func__, re_len - i, re + i, s_len - j, s + j, re_len, step, i, j);

		FAILIF(is_quantifier(&re[i]), REG_UNEXPECTED_QUANTIFIER);
		FAILIF(step <= 0, REG_INVALID_CHARACTER_SET);

		if (i + step < re_len && is_quantifier(re + i + step))
		{
			DBG("QUANTIFIER: [%.*s]%c [%.*s]\n", step, re + i, re[i + step], s_len - j, s + j);
			if (re[i + step] == '?')
			{
				int result = bar(re + i, step, s + j, s_len - j, info, bi);
				j += (result > 0 ? result : 0);
				i++;
			}
			else if (re[i + step] == '+' || re[i + step] == '*')
			{
				int j2 = j, nj = j, n1, n2 = -1, ni, non_greedy = 0;

				// Points to the regexp code after the quantifier
				ni = i + step + 1;
				if (ni < re_len && re[ni] == '?')
				{
					non_greedy = 1;
					ni++;
				}

				do {
					if ((n1 = bar(re + i, step, s + j2, s_len - j2, info, bi)) > 0)
						j2 += n1;
					if (re[i + step] == '+' && n1 < 0) break;

					if (ni >= re_len)
						nj = j2; // After quantifier, there is nothing
					else if ((n2 = bar(re + ni, re_len - ni, s + j2, s_len - j2, info, bi)) >= 0)
						nj = j2 + n2; // Regex after quantifier matched
					if (nj > j && non_greedy) break;
				} while (n1 > 0);

				if (n1 < 0 && re[i + step] == '*' && (n2 = bar(re + ni, re_len - ni, s + j, s_len - j, info, bi)) > 0)
					nj = j + n2;

				DBG("STAR/PLUS END: %d %d %d %d %d\n", j, nj, re_len - ni, n1, n2);
				FAILIF(re[i + step] == '+' && nj == j, REG_NOMATCH);

				// If while loop body above was not executed for the * quantifier, make sure the rest of the regex matches
				FAILIF(nj == j && ni < re_len && n2 < 0, REG_NOMATCH);

				// Returning here cause we've matched the rest of RE already
				return nj;
			}
			continue;
		}

		if (re[i] == '[')
		{
			n = match_set(re + i + 1, re_len - (i + 2), s + j, info);
			DBG("SET %.*s [%.*s] -> %d\n", step, re + i, s_len - j, s + j, n);
			FAILIF(n <= 0, REG_NOMATCH);
			j += n;
		}
		else if (re[i] == '(')
		{
			n = REG_NOMATCH;
			bi++;
			FAILIF(bi >= info->num_brackets, REG_INTERNAL_ERROR);
			DBG("CAPTURING [%.*s] [%.*s] [%s]\n", step, re + i, s_len - j, s + j, re + i + step);

			if (re_len - (i + step) <= 0)
				n = doh(s + j, s_len - j, info, bi); // Nothing follows brackets
			else
				for (int j2 = 0; j2 <= s_len - j; j2++)
					if ((n = doh(s + j, s_len - (j + j2), info, bi)) >= 0 && bar(re + i + step, re_len - (i + step), s + j + n, s_len - (j + n), info, bi) >= 0) break;

			DBG("CAPTURED [%.*s] [%.*s]:%d\n", step, re + i, s_len - j, s + j, n);
			FAILIF(n < 0, n);
			if (info->caps != nullptr)
			{
				info->caps[bi - 1].ptr = s + j;
				info->caps[bi - 1].len = n;
			}
			j += n;
		}
		else if (re[i] == '^')
			FAILIF(j != 0, REG_NOMATCH);
		else if (re[i] == '$')
			FAILIF(j != s_len, REG_NOMATCH);
		else
		{
			FAILIF(j >= s_len, REG_NOMATCH);
			n = match_op((unsigned char *) (re + i), (unsigned char *) (s + j), info);
			FAILIF(n <= 0, n);
			j += n;
		}
	}
	return j;
}

// Process branch points
__device__ static int doh(const char *s, int s_len, struct regex_info *info, int bi)
{
	const struct bracket_pair *b = &info->brackets[bi];
	int i = 0, len, result;
	const char *p;
	do
	{
		p = (i == 0 ? b->ptr : info->branches[b->branches + i - 1].schlong + 1);
		len = b->num_branches == 0 ? b->len :
			i == b->num_branches ? (int) (b->ptr + b->len - p) :
			(int) (info->branches[b->branches + i].schlong - p);
		DBG("%s %d %d [%.*s] [%.*s]\n", __func__, bi, i, len, p, s_len, s);
		result = bar(p, len, s, s_len, info, bi);
		DBG("%s <- %d\n", __func__, result);
	} while (result <= 0 && i++ < b->num_branches); // At least 1 iteration
	return result;
}

__device__ static int baz(const char *s, int s_len, struct regex_info *info)
{
	int i, result = -1, is_anchored = (info->brackets[0].ptr[0] == '^');
	for (i = 0; i <= s_len; i++)
	{
		result = doh(s + i, s_len - i, info, 0);
		if (result >= 0)
		{
			result += i;
			break;
		}
		if (is_anchored) break;
	}
	return result;
}

__device__ static void setup_branch_points(struct regex_info *info)
{
	int i, j;
	struct branch tmp;

	// First, sort branches. Must be stable, no qsort. Use bubble algo.
	for (i = 0; i < info->num_branches; i++)
	{
		for (j = i + 1; j < info->num_branches; j++)
		{
			if (info->branches[i].bracket_index > info->branches[j].bracket_index)
			{
				tmp = info->branches[i];
				info->branches[i] = info->branches[j];
				info->branches[j] = tmp;
			}
		}
	}

	// For each bracket, set their branch points. This way, for every bracket (i.e. every chunk of regex) we know all branch points before matching.
	for (i = j = 0; i < info->num_brackets; i++)
	{
		info->brackets[i].num_branches = 0;
		info->brackets[i].branches = j;
		while (j < info->num_branches && info->branches[j].bracket_index == i)
		{
			info->brackets[i].num_branches++;
			j++;
		}
	}
}

__device__ int foo(const char *re, int re_len, const char *s, int s_len, struct regex_info *info)
{
	int i, step, depth = 0;

	// First bracket captures everything
	info->brackets[0].ptr = re;
	info->brackets[0].len = re_len;
	info->num_brackets = 1;

	// Make a single pass over regex string, memorize brackets and branches
	for (i = 0; i < re_len; i += step)
	{
		step = get_op_len(re + i, re_len - i);

		if (re[i] == '|')
		{
			FAILIF(info->num_branches >= (int)_lengthof(info->branches), REG_TOO_MANY_BRANCHES);
			info->branches[info->num_branches].bracket_index = (info->brackets[info->num_brackets - 1].len == -1 ? info->num_brackets - 1 : depth);
			info->branches[info->num_branches].schlong = &re[i];
			info->num_branches++;
		}
		else if (re[i] == '\\')
		{
			FAILIF(i >= re_len - 1, REG_INVALID_METACHARACTER);
			if (re[i + 1] == 'x')
			{
				// Hex digit specification must follow
				FAILIF(re[i + 1] == 'x' && i >= re_len - 3, REG_INVALID_METACHARACTER);
				FAILIF(re[i + 1] ==  'x' && !(_isxdigit(re[i + 2]) && _isxdigit(re[i + 3])), REG_INVALID_METACHARACTER);
			}
			else
				FAILIF(!is_metacharacter((char *)re + i + 1), REG_INVALID_METACHARACTER);
		} else if (re[i] == '(')
		{
			FAILIF(info->num_brackets >= (int)_lengthof(info->brackets), REG_TOO_MANY_BRACKETS);
			depth++;  // Order is important here. Depth increments first.
			info->brackets[info->num_brackets].ptr = re + i + 1;
			info->brackets[info->num_brackets].len = -1;
			info->num_brackets++;
			FAILIF(info->num_caps > 0 && info->num_brackets - 1 > info->num_caps, REG_CAPS_ARRAY_TOO_SMALL);
		}
		else if (re[i] == ')')
		{
			int ind = (info->brackets[info->num_brackets - 1].len == -1 ? info->num_brackets - 1 : depth);
			info->brackets[ind].len = (int) (&re[i] - info->brackets[ind].ptr);
			DBG("SETTING BRACKET %d [%.*s]\n", ind, info->brackets[ind].len, info->brackets[ind].ptr);
			depth--;
			FAILIF(depth < 0, REG_UNBALANCED_BRACKETS);
			FAILIF(i > 0 && re[i - 1] == '(', REG_NOMATCH);
		}
	}

	FAILIF(depth != 0, REG_UNBALANCED_BRACKETS);
	setup_branch_points(info);

	return baz(s, s_len, info);
}

__device__ int reg_match(const char *regexp, const char *s, int s_len, struct regmatch_t *caps, int num_caps, int flags)
{
	// Initialize info structure
	struct regex_info info;
	info.flags = flags;
	info.num_brackets = info.num_branches = 0;
	info.num_caps = num_caps;
	info.caps = caps;

	DBG("========================> [%s] [%.*s]\n", regexp, s_len, s);
	return foo(regexp, (int) _strlen(regexp), s, s_len, &info);
}

// Regex must have exactly one bracket pair
__device__ char *reg_replace(const char *regex, const char *buf, const char *sub)
{
	char *s = nullptr;
	int n, n1, n2, n3, s_len, len = _strlen(buf);
	struct regmatch_t cap = { 0, 0, nullptr, 0 };
	do
	{
		s_len = (s == nullptr ? 0 : _strlen(s));
		if ((n = reg_match(regex, buf, len, &cap, 1, 0)) > 0)
			n1 = (int)(cap.ptr - buf), n2 = _strlen(sub), n3 = (int)(&buf[n] - &cap.ptr[cap.len]);
		else
			n1 = len, n2 = 0, n3 = 0;
		s = (char *)_realloc(s, s_len + n1 + n2 + n3 + 1);
		_memcpy(s + s_len, buf, n1);
		_memcpy(s + s_len + n1, sub, n2);
		_memcpy(s + s_len + n1 + n2, cap.ptr + cap.len, n3);
		s[s_len + n1 + n2 + n3] = '\0';

		buf += (n > 0 ? n : len);
		len -= (n > 0 ? n : len);
	} while (len > 0);
	return s;
}