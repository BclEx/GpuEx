#include <Runtime.h>

// NATIVE: assert
__global__ static void runtime0(void *r)
{
	_runtimeSetHeap(r);
	_assert(true);
	printf("Example: 0\n");
}

// NATIVE: heap
__global__ static void runtime1(void *r)
{
	_runtimeSetHeap(r);
	printf("Example: 1\n");
}

// NATIVE: stdargs
__global__ static void runtime2(void *r)
{
	_runtimeSetHeap(r);
#if __CUDACC__
	va_list2<const char*, int> args;
	_va_start(args, "Name", 4);
	char *a0 = _va_arg(args, char*); _assert(a0 == "Name");
	int a1 = _va_arg(args, int); _assert(a1 == 4);
	_va_end(args);
#endif
	printf("Example: 2\n");
}

// NATIVE: printf
__global__ static void runtime3(void *r)
{
	_runtimeSetHeap(r);
	_printf("t0\n");
	_printf("t1 %s\n", "1");
	_printf("t2 %s %d\n", "1", 2);
	_printf("t3 %s %d %d\n", "1", 2, 3);
	_printf("t4 %s %d %d %d\n", "1", 2, 3, 4);
	_printf("t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	_printf("t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	_printf("t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	_printf("t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	_printf("t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	_printf("ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
	printf("Example: 3\n");
}

// NATIVE: transfer
__global__ static void runtime4(void *r)
{
	_runtimeSetHeap(r);
	_transfer("t0\n");
	_transfer("t1 %s\n", "1");
	_transfer("t2 %s %d\n", "1", 2);
	_transfer("t3 %s %d %d\n", "1", 2, 3);
	_transfer("t4 %s %d %d %d\n", "1", 2, 3, 4);
	_transfer("t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	_transfer("t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	_transfer("t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	_transfer("t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	_transfer("t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	_transfer("ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
	printf("Example: 4\n");
}

// NATIVE: throw
__global__ static void runtime5(void *r)
{
	_runtimeSetHeap(r);
#if __CUDACC__
	_throw("t0\n");
	_throw("t1 %s\n", "1");
	_throw("t2 %s %d\n", "1", 2);
	_throw("t3 %s %d %d\n", "1", 2, 3);
	_throw("t4 %s %d %d %d\n", "1", 2, 3, 4);
#endif
	printf("Example: 5\n");
}

// UTF: tests
__global__ static void runtime6(void *r)
{
	_runtimeSetHeap(r);
	//	_strskiputf8();
	//	_utf8read();
	//	_utf8charlength();
	//#ifndef OMIT_UTF16
	//	__device__ int _utf16bytelength(const void *z, int chars);
	//#endif
	printf("Example: 6\n");
}

// FUNC: func
__global__ static void runtime7(void *r)
{
	_runtimeSetHeap(r);
	char a0 = __toupper('a'); char a0n = __toupper('A'); _assert(a0 == 'A' || a0n == 'A');
	bool b0 = _isupper('a'); bool b0n = _isupper('A'); _assert(!b0 && b0n);
	bool a1 = _isspace('a'); bool a1n = _isspace(' '); _assert(!a1 && a1n);
	bool a2 = _isalnum('a'); bool a2n = _isalnum('1'); _assert(a2 && a2n);
	bool a3 = _isalpha('a'); bool a3n = _isalpha('A'); _assert(a3 && a3n);
	bool a4 = _isdigit('a'); bool a4n = _isdigit('1'); _assert(!a4 && a4n);
	bool a5 = _isxdigit('a'); bool a5n = _isxdigit('A'); _assert(a5 && a5n);
	char a6 = __tolower('a'); char a6n = __tolower('A'); _assert(a6 == 'a' && a6n == 'a');
	bool b6 = _islower('a'); bool b6n = _islower('A'); _assert(b6 && !b6n);
	bool a7 = _ispoweroftwo(2); bool a7n = _ispoweroftwo(3); _assert(a7 && !a7n);
	bool a8 = _isalpha2('a'); bool a8n = _isalpha2('A'); _assert(a8 && a8n);
	printf("Example: 7\n");
}

// FUNC: array_t
__global__ static void runtime8(void *r)
{
	_runtimeSetHeap(r);
	array_t<char> a0 = "Name1"; _assert(!_strcmp(a0.data, "Name1"));
	a0.length = 5; _assert(a0.length == 5);
	a0 = "Name2"; _assert(!_strcmp(a0.data, "Name2"));
	printf("Example: 8\n");
}

// FUNC: templates
__global__ static void runtime9(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	_strcpy(buf, "Test"); int a0 = _memcmp(buf, "Test", 4); _assert(!a0);
	_strncpy(buf, "Test", 4); int b0 = _memcmp(buf, "Test", 4); _assert(!b0);
	const char *a1 = _strchr("Test", 'E'); _assert(a1);
	int a2 = _strcmp("Test", "Test"); _assert(!a2);
	int a3 = _strncmp("Tesa", "Tesb", 3); _assert(!a3);
	_memcpy(buf, "Test", 4); int a4 = _memcmp(buf, "Test", 4); _assert(!a4);
	_memset(buf, 0, sizeof(buf)); int a5 = _memcmp(buf, "\0\0\0\0", 4); _assert(!a5);
	const char *a6 = _memchr("Test", 'e'); _assert(a6);
	int a7 = _memcmp("Test", "Test", 4); _assert(!a7);
	_memmove(buf, "Test", 4); int a8 = _memcmp(buf, "Test", 4); _assert(!a8);
	int a9 = _strlen("Test"); _assert(a9 == 4);
	int a10 = _hextobyte('a'); _assert(a10 == 10);
#ifndef OMIT_BLOB_LITERAL
	void *a11 = _taghextoblob(nullptr, "z", 1); _assert(a11);
#endif
#ifndef OMIT_FLOATING_POINT
	bool a12 = _isnan(0.0); _assert(!a12);
#endif
	printf("Example: 9\n");
}

// MEMORY ALLOCATION
__global__ static void runtime10(void *r)
{
	_runtimeSetHeap(r);
	printf("Example: 10\n");
}

// PRINT
__global__ static void runtime11(void *r)
{
	_runtimeSetHeap(r);
	char base[100];
	TextBuilder b;
	TextBuilder::Init(&b, base, sizeof(base), CORE_MAX_LENGTH);
	b.AllocType = 0;
	//
	b.AppendSpace(3); _assert(b.Index == 3); _assert(!_strcmp(b.ToString(), "   "));
	b.Index = 0;
	b.Append("a+", 2); b.Append("b=", 2); _assert(b.Index == 4); _assert(!_strcmp(b.ToString(), "a+b="));
	b.Index = 0;
	b.AppendFormat("self: %s", "test"); _assert(!_strcmp(b.ToString(), "self: test"));
	printf("Example: 11\n");
}

// SNPRINTF
__global__ static void runtime12(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	char *a0 = __snprintf(buf, sizeof(buf), "t0\n"); _assert(!_strcmp(a0, "t0\n"));
	char *a1 = __snprintf(buf, sizeof(buf), "t1 %s\n", "1"); _assert(!_strcmp(a1, "t1 1\n"));
	char *a2 = __snprintf(buf, sizeof(buf), "t2 %s %d\n", "1", 2); _assert(!_strcmp(a2, "t2 1 2\n"));
	char *a3 = __snprintf(buf, sizeof(buf), "t3 %s %d %d\n", "1", 2, 3); _assert(!_strcmp(a3, "t3 1 2 3\n"));
	char *a4 = __snprintf(buf, sizeof(buf), "t4 %s %d %d %d\n", "1", 2, 3, 4); _assert(!_strcmp(a4, "t4 1 2 3 4\n"));
	char *a5 = __snprintf(buf, sizeof(buf), "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5); _assert(!_strcmp(a5, "t5 1 2 3 4 5\n"));
	char *a6 = __snprintf(buf, sizeof(buf), "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6); _assert(!_strcmp(a6, "t6 1 2 3 4 5 6\n"));
	char *a7 = __snprintf(buf, sizeof(buf), "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7); _assert(!_strcmp(a7, "t7 1 2 3 4 5 6 7\n"));
	char *a8 = __snprintf(buf, sizeof(buf), "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8); _assert(!_strcmp(a8, "t8 1 2 3 4 5 6 7 8\n"));
	char *a9 = __snprintf(buf, sizeof(buf), "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9"); _assert(!_strcmp(a9, "t9 1 2 3 4 5 6 7 8 9\n")); //: errors with %s
	char *aA = __snprintf(buf, sizeof(buf), "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A"); _assert(!_strcmp(aA, "ta 1 2 3 4 5 6 7 8 9 A\n"));
	// extended
	char *aB = __snprintf(buf, sizeof(buf), "tb %s %d %d %d %d %d %d %d %d %s %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A", "B"); _assert(!_strcmp(aB, "tb 1 2 3 4 5 6 7 8 9 A B\n"));
	char *aC = __snprintf(buf, sizeof(buf), "tc %s %d %d %d %d %d %d %d %d %s %s %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C"); _assert(!_strcmp(aC, "tc 1 2 3 4 5 6 7 8 9 A B C\n"));
	char *aD = __snprintf(buf, sizeof(buf), "td %s %d %d %d %d %d %d %d %d %s %s %s %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D"); _assert(!_strcmp(aD, "td 1 2 3 4 5 6 7 8 9 A B C D\n"));
	char *aE = __snprintf(buf, sizeof(buf), "te %s %d %d %d %d %d %d %d %d %s %s %s %s %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E"); _assert(!_strcmp(aE, "te 1 2 3 4 5 6 7 8 9 A B C D E\n"));
	char *aF = __snprintf(buf, sizeof(buf), "tf %s %d %d %d %d %d %d %d %d %s %s %s %s %s %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "F"); _assert(!_strcmp(aF, "tf 1 2 3 4 5 6 7 8 9 A B C D E F\n"));
	//
	char *b0 = _sprintf(buf, "t0\n"); _assert(!_strcmp(b0, "t0\n"));
	printf("Example: 12\n");
}

// FPRINTF
__global__ static void runtime13(void *r)
{
	_runtimeSetHeap(r);
	FILE *f = _fopen("C:\\T_\\fopen.txt", "w");
	_fprintfR(f, "The quick brown fox jumps over the lazy dog");
	_fflushR(f);
	_fcloseR(f);
	printf("Example: 13\n");
}

// MPRINTF
__global__ static void runtime14(void *r)
{
	TagBase *tag = new TagBase(); //"tag";
	_runtimeSetHeap(r);
	char *a0 = _mprintf("t0\n"); _assert(!_strcmp(a0, "t0\n")); _free(a0);
	char *a1 = _mprintf("t1 %s\n", "1"); _assert(!_strcmp(a1, "t1 1\n")); _free(a1);
	char *a2 = _mprintf("t2 %s %d\n", "1", 2); _assert(!_strcmp(a2, "t2 1 2\n")); _free(a2);
	char *a3 = _mprintf("t3 %s %d %d\n", "1", 2, 3); _assert(!_strcmp(a3, "t3 1 2 3\n")); _free(a3);
	char *a4 = _mprintf("t4 %s %d %d %d\n", "1", 2, 3, 4); _assert(!_strcmp(a4, "t4 1 2 3 4\n")); _free(a4);
	char *a5 = _mprintf("t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5); _assert(!_strcmp(a5, "t5 1 2 3 4 5\n")); _free(a5);
	char *a6 = _mprintf("t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6); _assert(!_strcmp(a6, "t6 1 2 3 4 5 6\n")); _free(a6);
	char *a7 = _mprintf("t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7); _assert(!_strcmp(a7, "t7 1 2 3 4 5 6 7\n")); _free(a7);
	char *a8 = _mprintf("t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8); _assert(!_strcmp(a8, "t8 1 2 3 4 5 6 7 8\n")); _free(a8);
	char *a9 = _mprintf("t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9"); _assert(!_strcmp(a9, "t9 1 2 3 4 5 6 7 8 9\n")); _free(a9);
	char *aA = _mprintf("ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A"); _assert(!_strcmp(aA, "ta 1 2 3 4 5 6 7 8 9 A\n")); _free(aA);
	//
	char *b0 = _mtagprintf(tag, "t0\n"); _assert(!_strcmp(b0, "t0\n")); _tagfree(tag, b0);
	char *b1 = _mtagprintf(tag, "t1 %s\n", "1"); _assert(!_strcmp(b1, "t1 1\n")); _tagfree(tag, b1);
	char *b2 = _mtagprintf(tag, "t2 %s %d\n", "1", 2); _assert(!_strcmp(b2, "t2 1 2\n")); _tagfree(tag, b2);
	char *b3 = _mtagprintf(tag, "t3 %s %d %d\n", "1", 2, 3); _assert(!_strcmp(b3, "t3 1 2 3\n")); _tagfree(tag, b3);
	char *b4 = _mtagprintf(tag, "t4 %s %d %d %d\n", "1", 2, 3, 4); _assert(!_strcmp(b4, "t4 1 2 3 4\n")); _tagfree(tag, b4);
	char *b5 = _mtagprintf(tag, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5); _assert(!_strcmp(b5, "t5 1 2 3 4 5\n")); _tagfree(tag, b5);
	char *b6 = _mtagprintf(tag, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6); _assert(!_strcmp(b6, "t6 1 2 3 4 5 6\n")); _tagfree(tag, b6);
	char *b7 = _mtagprintf(tag, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7); _assert(!_strcmp(b7, "t7 1 2 3 4 5 6 7\n")); _tagfree(tag, b7);
	char *b8 = _mtagprintf(tag, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8); _assert(!_strcmp(b8, "t8 1 2 3 4 5 6 7 8\n")); _tagfree(tag, b8);
	char *b9 = _mtagprintf(tag, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9"); _assert(!_strcmp(b9, "t9 1 2 3 4 5 6 7 8 9\n")); _tagfree(tag, b9);
	char *bA = _mtagprintf(tag, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A"); _assert(!_strcmp(bA, "ta 1 2 3 4 5 6 7 8 9 A\n")); _tagfree(tag, bA);
	//
	char *c0 = _mtagappendf(tag, nullptr, "t0\n"); _assert(!_strcmp(c0, "t0\n")); _tagfree(tag, c0);
	char *c1 = _mtagappendf(tag, nullptr, "t1 %s\n", "1"); _assert(!_strcmp(c1, "t1 1\n")); _tagfree(tag, c1);
	char *c2 = _mtagappendf(tag, nullptr, "t2 %s %d\n", "1", 2); _assert(!_strcmp(c2, "t2 1 2\n")); _tagfree(tag, c2);
	char *c3 = _mtagappendf(tag, nullptr, "t3 %s %d %d\n", "1", 2, 3); _assert(!_strcmp(c3, "t3 1 2 3\n")); _tagfree(tag, c3);
	char *c4 = _mtagappendf(tag, nullptr, "t4 %s %d %d %d\n", "1", 2, 3, 4); _assert(!_strcmp(c4, "t4 1 2 3 4\n")); _tagfree(tag, c4);
	char *c5 = _mtagappendf(tag, nullptr, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5); _assert(!_strcmp(c5, "t5 1 2 3 4 5\n")); _tagfree(tag, c5);
	char *c6 = _mtagappendf(tag, nullptr, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6); _assert(!_strcmp(c6, "t6 1 2 3 4 5 6\n")); _tagfree(tag, c6);
	char *c7 = _mtagappendf(tag, nullptr, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7); _assert(!_strcmp(c7, "t7 1 2 3 4 5 6 7\n")); _tagfree(tag, c7);
	char *c8 = _mtagappendf(tag, nullptr, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8); _assert(!_strcmp(c8, "t8 1 2 3 4 5 6 7 8\n")); _tagfree(tag, c8);
	char *c9 = _mtagappendf(tag, nullptr, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9"); _assert(!_strcmp(c9, "t9 1 2 3 4 5 6 7 8 9\n")); _tagfree(tag, c9);
	char *cA = _mtagappendf(tag, nullptr, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A"); _assert(!_strcmp(cA, "ta 1 2 3 4 5 6 7 8 9 A\n")); _tagfree(tag, cA);
	//
	char *d0 = (char *)_tagalloc(tag, 1);
	_mtagassignf(&d0, tag, "t0\n"); _assert(!_strcmp(d0, "t0\n"));
	_mtagassignf(&d0, tag, "t1 %s\n", "1"); _assert(!_strcmp(d0, "t1 1\n"));
	_mtagassignf(&d0, tag, "t2 %s %d\n", "1", 2); _assert(!_strcmp(d0, "t2 1 2\n"));
	_mtagassignf(&d0, tag, "t3 %s %d %d\n", "1", 2, 3); _assert(!_strcmp(d0, "t3 1 2 3\n"));
	_mtagassignf(&d0, tag, "t4 %s %d %d %d\n", "1", 2, 3, 4); _assert(!_strcmp(d0, "t4 1 2 3 4\n"));
	_mtagassignf(&d0, tag, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5); _assert(!_strcmp(d0, "t5 1 2 3 4 5\n"));
	_mtagassignf(&d0, tag, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6); _assert(!_strcmp(d0, "t6 1 2 3 4 5 6\n"));
	_mtagassignf(&d0, tag, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7); _assert(!_strcmp(d0, "t7 1 2 3 4 5 6 7\n"));
	_mtagassignf(&d0, tag, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8); _assert(!_strcmp(d0, "t8 1 2 3 4 5 6 7 8\n"));
	_mtagassignf(&d0, tag, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9"); _assert(!_strcmp(d0, "t9 1 2 3 4 5 6 7 8 9\n"));
	_mtagassignf(&d0, tag, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A"); _assert(!_strcmp(d0, "ta 1 2 3 4 5 6 7 8 9 A\n"));
	_tagfree(tag, d0);
	printf("Example: 14\n");
}

#if __CUDACC__
void __testRuntime(cudaDeviceHeap &r)
{
	RuntimeSentinel::Initialize();
	runtime0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime1<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime2<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime3<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime4<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	//runtime5<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime6<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime7<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime8<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime9<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime10<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime11<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime12<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime13<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtime14<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	RuntimeSentinel::Shutdown();
}
#else
void __testRuntime(cudaDeviceHeap &r)
{
#if OS_MAP
	RuntimeSentinel::Initialize();
#endif
	runtime0(r.heap);
	runtime1(r.heap);
	runtime2(r.heap);
	runtime3(r.heap);
	runtime4(r.heap);
	runtime5(r.heap);
	runtime6(r.heap);
	runtime7(r.heap);
	runtime8(r.heap);
	runtime9(r.heap);
	runtime10(r.heap);
	runtime11(r.heap);
	runtime12(r.heap);
	runtime13(r.heap);
	runtime14(r.heap);
#if OS_MAP
	RuntimeSentinel::Shutdown();
#endif
}
#endif