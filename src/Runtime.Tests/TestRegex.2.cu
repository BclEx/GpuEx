//#include "trex.h"
//#include <stdio.h>
//#include <string.h>
//
//#ifdef _UNICODE
//#define reg_sprintf swprintf
//#else
//#define reg_sprintf sprintf
//#endif
//
///* test edited by Kai Uwe Jesussek for parsing html*/
//
//void reg_test(const char_t *regex, const char_t *text, const bool expected) {
//	const char_t *error = NULL;
//	regex_t *regex_compiled = reg_compile(_TREXC(regex), &error);
//
//	// Check for regex compilation errors
//	if (!regex_compiled) {
//		_printf(_TREXC("Compilation error [%s]!\n"), error ? error : _TREXC("undefined"));
//	}
//
//	bool result = reg_match(regex_compiled, text);
//
//	#ifdef _DEBUG
//		printf("DEBUG reg_test: result = %d\n", result);
//		printf("DEBUG reg_test: expected = %d\n", expected);
//	#endif
//
//	// Print matching outcome compared to expected outcome
//	if (result == expected) {
//		if (result == true) {
//			_printf("True positive: String '%s' matched.\n", regex);
//		} else {
//			_printf("True negative: No matches for '%s'.\n", regex);
//		}
//	} else {
//		if (result == true) {
//			_printf("FALSE POSITIVE: STRING '%s' SHOULD NOT HAVE MATCHED!\n", regex);
//		} else {
//			_printf("FALSE NEGATIVE: STRING '%s' SHOULD HAVE MATCHED.\n", regex);
//		}
//	}
//
//	reg_free(regex_compiled);
//}
//
//int main(int argc, char* argv[])
//{
//	const char_t *begin,*end;
//	char_t sTemp[200];
//	const char_t *error = NULL;
//	regex_t *x = reg_compile(_TREXC("<a href=[\"|'](.*)[\"|']>(.*)</a>"),&error);
//	if(x) {
//		reg_sprintf(sTemp,_TREXC("<html><head></head><body><a href='link.html'>link</a></body></html>"));
//		if(reg_search(x,sTemp,&begin,&end))
//		{
//			int i,n = reg_getsubexpcount(x);
//			regmatch_t match;
//			for(i = 0; i < n; i++)
//			{
//				char_t t[200];
//				reg_getsubexp(x,i,&match);
//				reg_sprintf(t,_TREXC("[%%d]%%.%ds\n"),match.len);
//				_printf(t,i,match.begin);
//			}
//			_printf(_TREXC("match! %d sub matches\n"),reg_getsubexpcount(x));
//		}
//		else {
//			_printf(_TREXC("no match!\n"));
//		}
//		reg_free(x);
//	}
//	else {
//		_printf(_TREXC("compilation error [%s]!\n"),error?error:_TREXC("undefined"));
//	}
//
//	// Negative testing for strings that shouldn't match
//	char_t regex_str[200];
//	char_t test_string[200];
//	reg_sprintf(test_string, _TREXC("Match some part of this string."));
//
//	_printf("=============================\n");
//	_printf("Simple matching string cases.\n");
//	_printf("Test string: %s\n", test_string);
//
//	_printf("Test 1: missing letter\n");
//	reg_sprintf(regex_str, _TREXC("soe"));
//	reg_test(regex_str, test_string, false);
//
//	_printf("Test 2: missing space\n");
//	reg_sprintf(regex_str, _TREXC("matchs"));
//	reg_test(regex_str, test_string, false);
//
//	_printf("Test 3a: case mismatch\n");
//	reg_sprintf(regex_str, _TREXC("match"));
//	reg_test(regex_str,test_string, false);
//	_printf("Test 3b: case match\n");
//	reg_sprintf(regex_str, _TREXC("Match"));
//	reg_test(regex_str,test_string, true);
//
//	_printf("Test 4: unused character\n");
//	reg_sprintf(regex_str, _TREXC("!"));
//	reg_test(regex_str, test_string, false);
//
//	_printf("Test 5: single character\n");
//	reg_sprintf(regex_str, _TREXC("."));
//	reg_test(regex_str, test_string, true);
//	// End negative testing
//
//	return 0;
//}
