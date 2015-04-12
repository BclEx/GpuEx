#include <Runtime.h>
#include <Regex.h>

// NATIVE: assert
__global__ static void regex0(void *r)
{
	_runtimeSetHeap(r);
	struct regmatch_t caps[10];

	// Metacharacters
	_assert(reg_match("$", "abcd", 4, nullptr, 0, 0) == 4);
	_assert(reg_match("^", "abcd", 4, nullptr, 0, 0) == 0);
	_assert(reg_match("x|^", "abcd", 4, nullptr, 0, 0) == 0);
	_assert(reg_match("x|$", "abcd", 4, nullptr, 0, 0) == 4);
	_assert(reg_match("x", "abcd", 4, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match(".", "abcd", 4, nullptr, 0, 0) == 1);
	_assert(reg_match("^.*\\\\.*$", "c:\\Tools", 8, nullptr, 0, REG_IGNORE_CASE) == 8);
	_assert(reg_match("\\", "a", 1, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("\\x", "a", 1, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("\\x1", "a", 1, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("\\x20", " ", 1, nullptr, 0, 0) == 1);

	_assert(reg_match("^.+$", "", 0, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("^(.+)$", "", 0, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("^([\\+-]?)([\\d]+)$", "+", 1, caps, 10, REG_IGNORE_CASE) == REG_NOMATCH);
	_assert(reg_match("^([\\+-]?)([\\d]+)$", "+27", 3, caps, 10, REG_IGNORE_CASE) == 3);
	_assert(caps[0].len == 1);
	_assert(caps[0].ptr[0] == '+');
	_assert(caps[1].len == 2);
	_assert(_memcmp(caps[1].ptr, "27", 2) == 0);

	_assert(reg_match("tel:\\+(\\d+[\\d-]+\\d)", "tel:+1-201-555-0123;a=b", 23, caps, 10, 0) == 19);
	_assert(caps[0].len == 14);
	_assert(_memcmp(caps[0].ptr, "1-201-555-0123", 14) == 0);

	// Character sets
	_assert(reg_match("[abc]", "1c2", 3, nullptr, 0, 0) == 2);
	_assert(reg_match("[abc]", "1C2", 3, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("[abc]", "1C2", 3, nullptr, 0, REG_IGNORE_CASE) == 2);
	_assert(reg_match("[.2]", "1C2", 3, nullptr, 0, 0) == 1);
	_assert(reg_match("[\\S]+", "ab cd", 5, nullptr, 0, 0) == 2);
	_assert(reg_match("[\\S]+\\s+[tyc]*", "ab cd", 5, nullptr, 0, 0) == 4);
	_assert(reg_match("[\\d]", "ab cd", 5, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("[^\\d]", "ab cd", 5, nullptr, 0, 0) == 1);
	_assert(reg_match("[^\\d]+", "abc123", 6, nullptr, 0, 0) == 3);
	_assert(reg_match("[1-5]+", "123456789", 9, nullptr, 0, 0) == 5);
	_assert(reg_match("[1-5a-c]+", "123abcdef", 9, nullptr, 0, 0) == 6);
	_assert(reg_match("[1-5a-]+", "123abcdef", 9, nullptr, 0, 0) == 4);
	_assert(reg_match("[1-5a-]+", "123a--2oo", 9, nullptr, 0, 0) == 7);
	_assert(reg_match("[htps]+://", "https://", 8, nullptr, 0, 0) == 8);
	_assert(reg_match("[^\\s]+", "abc def", 7, nullptr, 0, 0) == 3);
	_assert(reg_match("[^fc]+", "abc def", 7, nullptr, 0, 0) == 2);
	_assert(reg_match("[^d\\sf]+", "abc def", 7, nullptr, 0, 0) == 3);

	// Flags - case sensitivity
	_assert(reg_match("FO", "foo", 3, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("FO", "foo", 3, nullptr, 0, REG_IGNORE_CASE) == 2);
	_assert(reg_match("(?m)FO", "foo", 3, nullptr, 0, 0) == REG_UNEXPECTED_QUANTIFIER);
	_assert(reg_match("(?m)x", "foo", 3, nullptr, 0, 0) == REG_UNEXPECTED_QUANTIFIER);

	_assert(reg_match("fo", "foo", 3, nullptr, 0, 0) == 2);
	_assert(reg_match(".+", "foo", 3, nullptr, 0, 0) == 3);
	_assert(reg_match(".+k", "fooklmn", 7, nullptr, 0, 0) == 4);
	_assert(reg_match(".+k.", "fooklmn", 7, nullptr, 0, 0) == 5);
	_assert(reg_match("p+", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("ok", "fooklmn", 7, nullptr, 0, 0) == 4);
	_assert(reg_match("lmno", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("mn.", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("o", "fooklmn", 7, nullptr, 0, 0) == 2);
	_assert(reg_match("^o", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("^", "fooklmn", 7, nullptr, 0, 0) == 0);
	_assert(reg_match("n$", "fooklmn", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("n$k", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("l$", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match(".$", "fooklmn", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("a?", "fooklmn", 7, nullptr, 0, 0) == 0);
	_assert(reg_match("^a*CONTROL", "CONTROL", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("^[a]*CONTROL", "CONTROL", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("^(a*)CONTROL", "CONTROL", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("^(a*)?CONTROL", "CONTROL", 7, nullptr, 0, 0) == 7);

	_assert(reg_match("\\_", "abc", 3, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("+", "fooklmn", 7, nullptr, 0, 0) == REG_UNEXPECTED_QUANTIFIER);
	_assert(reg_match("()+", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);
	_assert(reg_match("\\x", "12", 2, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("\\xhi", "12", 2, nullptr, 0, 0) == REG_INVALID_METACHARACTER);
	_assert(reg_match("\\x20", "_ J", 3, nullptr, 0, 0) == 2);
	_assert(reg_match("\\x4A", "_ J", 3, nullptr, 0, 0) == 3);
	_assert(reg_match("\\d+", "abc123def", 9, nullptr, 0, 0) == 6);

	// Balancing brackets
	_assert(reg_match("(x))", "fooklmn", 7, nullptr, 0, 0) == REG_UNBALANCED_BRACKETS);
	_assert(reg_match("(", "fooklmn", 7, nullptr, 0, 0) == REG_UNBALANCED_BRACKETS);

	_assert(reg_match("klz?mn", "fooklmn", 7, nullptr, 0, 0) == 7);
	_assert(reg_match("fa?b", "fooklmn", 7, nullptr, 0, 0) == REG_NOMATCH);

	// Brackets & capturing
	_assert(reg_match("^(te)", "tenacity subdues all", 20, caps, 10, 0) == 2);
	_assert(reg_match("(bc)", "abcdef", 6, caps, 10, 0) == 3);
	_assert(reg_match(".(d.)", "abcdef", 6, caps, 10, 0) == 5);
	_assert(reg_match(".(d.)\\)?", "abcdef", 6, caps, 10, 0) == 5);
	_assert(caps[0].len == 2);
	_assert(_memcmp(caps[0].ptr, "de", 2) == 0);
	_assert(reg_match("(.+)", "123", 3, caps, 10, 0) == 3);
	_assert(reg_match("(2.+)", "123", 3, caps, 10, 0) == 3);
	_assert(caps[0].len == 2);
	_assert(_memcmp(caps[0].ptr, "23", 2) == 0);
	_assert(reg_match("(.+2)", "123", 3, caps, 10, 0) == 2);
	_assert(caps[0].len == 2);
	_assert(_memcmp(caps[0].ptr, "12", 2) == 0);
	_assert(reg_match("(.*(2.))", "123", 3, caps, 10, 0) == 3);
	_assert(reg_match("(.)(.)", "123", 3, caps, 10, 0) == 2);
	_assert(reg_match("(\\d+)\\s+(\\S+)", "12 hi", 5, caps, 10, 0) == 5);
	_assert(reg_match("ab(cd)+ef", "abcdcdef", 8, nullptr, 0, 0) == 8);
	_assert(reg_match("ab(cd)*ef", "abcdcdef", 8, nullptr, 0, 0) == 8);
	_assert(reg_match("ab(cd)+?ef", "abcdcdef", 8, nullptr, 0, 0) == 8);
	_assert(reg_match("ab(cd)+?.", "abcdcdef", 8, nullptr, 0, 0) == 5);
	_assert(reg_match("ab(cd)?", "abcdcdef", 8, nullptr, 0, 0) == 4);
	_assert(reg_match("a(b)(cd)", "abcdcdef", 8, caps, 1, 0) == REG_CAPS_ARRAY_TOO_SMALL);
	_assert(reg_match("(.+/\\d+\\.\\d+)\\.jpg$", "/foo/bar/12.34.jpg", 18, caps, 1, 0) == 18);
	_assert(reg_match("(ab|cd).*\\.(xx|yy)", "ab.yy", 5, nullptr, 0, 0) == 5);
	_assert(reg_match(".*a", "abcdef", 6, nullptr, 0, 0) == 1);
	_assert(reg_match("(.+)c", "abcdef", 6, nullptr, 0, 0) == 3);
	_assert(reg_match("\\n", "abc\ndef", 7, nullptr, 0, 0) == 4);
	_assert(reg_match("b.\\s*\\n", "aa\r\nbb\r\ncc\r\n\r\n", 14, caps, 10, 0) == 8);

	// Greedy vs non-greedy
	_assert(reg_match(".+c", "abcabc", 6, nullptr, 0, 0) == 6);
	_assert(reg_match(".+?c", "abcabc", 6, nullptr, 0, 0) == 3);
	_assert(reg_match(".*?c", "abcabc", 6, nullptr, 0, 0) == 3);
	_assert(reg_match(".*c", "abcabc", 6, nullptr, 0, 0) == 6);
	_assert(reg_match("bc.d?k?b+", "abcabc", 6, nullptr, 0, 0) == 5);

	// Branching
	_assert(reg_match("|", "abc", 3, nullptr, 0, 0) == 0);
	_assert(reg_match("|.", "abc", 3, nullptr, 0, 0) == 1);
	_assert(reg_match("x|y|b", "abc", 3, nullptr, 0, 0) == 2);
	_assert(reg_match("k(xx|yy)|ca", "abcabc", 6, nullptr, 0, 0) == 4);
	_assert(reg_match("k(xx|yy)|ca|bc", "abcabc", 6, nullptr, 0, 0) == 3);
	_assert(reg_match("(|.c)", "abc", 3, caps, 10, 0) == 3);
	_assert(caps[0].len == 2);
	_assert(_memcmp(caps[0].ptr, "bc", 2) == 0);
	_assert(reg_match("a|b|c", "a", 1, nullptr, 0, 0) == 1);
	_assert(reg_match("a|b|c", "b", 1, nullptr, 0, 0) == 1);
	_assert(reg_match("a|b|c", "c", 1, nullptr, 0, 0) == 1);
	_assert(reg_match("a|b|c", "d", 1, nullptr, 0, 0) == REG_NOMATCH);

	// Optional match at the end of the string
	_assert(reg_match("^.*c.?$", "abc", 3, nullptr, 0, 0) == 3);
	_assert(reg_match("^.*C.?$", "abc", 3, nullptr, 0, REG_IGNORE_CASE) == 3);
	_assert(reg_match("bk?", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("b(k?)", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("b[k-z]*", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("ab(k|z|y)*", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("[b-z].*", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("(b|z|u).*", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("ab(k|z|y)?", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match(".*", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match(".*$", "ab", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("a+$", "aa", 2, nullptr, 0, 0) == 2);
	_assert(reg_match("a*$", "aa", 2, nullptr, 0, 0) == 2);
	_assert(reg_match( "a+$" ,"Xaa", 3, nullptr, 0, 0) == 3);
	_assert(reg_match( "a*$" ,"Xaa", 3, nullptr, 0, 0) == 3);

	printf("Example: 0\n");
}

// NATIVE: heap
__global__ static void regex1(void *r)
{
	_runtimeSetHeap(r);

	// Example: HTTP request
	const char *request = " GET /index.html HTTP/1.0\r\n\r\n";
	struct regmatch_t caps[4];

	if (reg_match("^\\s*(\\S+)\\s+(\\S+)\\s+HTTP/(\\d)\\.(\\d)", request, _strlen(request), caps, 4, 0) > 0)
		printf("Method: [%.*s], URI: [%.*s]\n", caps[0].len, caps[0].ptr, caps[1].len, caps[1].ptr);
	else
		printf("Error parsing [%s]\n", request);
	_assert(caps[1].len == 11);
	_assert(_memcmp(caps[1].ptr, "/index.html", caps[1].len) == 0);

	// Example: string replacement
	char *s = reg_replace("({{.+?}})", "Good morning, {{foo}}. How are you, {{bar}}?", "Bob");
	printf("%s\n", s);
	_assert(!_strcmp(s, "Good morning, Bob. How are you, Bob?"));
	_free(s);

	// Example: find all URLs in a string
	const char *str = "<img src=\"HTTPS://FOO.COM/x?b#c=tab1\"/> <a href=\"http://cesanta.com\">some link</a>";
	const char *regex = "(?i)((https?://)[^\\s/'\"<>]+/?[^\\s'\"<>]*)";
	//struct regmatch_t caps[2];
	int i, j = 0, str_len = (int)_strlen(str);

	while (j < str_len && (i = reg_match(regex, str + j, str_len - j, caps, 2, 0)) > 0)
	{
		printf("Found URL: [%.*s]\n", caps[0].len, caps[0].ptr);
		j += i;
	}

	printf("Example: 1\n");
}

#if __CUDACC__
void __testRegex1(cudaDeviceHeap &r)
{
	regex0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	regex1<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}
#else
void __testRegex1(cudaDeviceHeap &r)
{
	regex0(r.heap);
	regex1(r.heap);
}
#endif