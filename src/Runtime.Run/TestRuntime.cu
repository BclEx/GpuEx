#include <Runtime.h>

__global__ static void runtimeExample0(void *r)
{
	_runtimeSetHeap(r);
	_assert(false);
}

__global__ static void runtimeExample1(void *r)
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
}

__global__ static void runtimeExample2(void *r)
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
}

__global__ static void runtimeExample3(void *r)
{
	_runtimeSetHeap(r);
	_throw("t0\n");
	_throw("t1 %s\n", "1");
	_throw("t2 %s %d\n", "1", 2);
	_throw("t3 %s %d %d\n", "1", 2, 3);
	_throw("t4 %s %d %d %d\n", "1", 2, 3, 4);
}

__global__ static void runtimeExample4(void *r)
{
	_runtimeSetHeap(r);
	char a0 = _toupper('a'); char a0n = _toupper('A'); _assert(a0 == 'A' || a0n == 'A');
	bool a1 = _isspace('a'); bool a1n = _isspace(' '); _assert(!a1 && a1n);
	bool a2 = _isalnum('a'); bool a2n = _isalnum('1'); _assert(a2 && a2n);
	bool a3 = _isalpha('a'); bool a3n = _isalpha('A'); _assert(a3 && a3n);
	bool a4 = _isdigit('a'); bool a4n = _isdigit('1'); _assert(!a4 && a4n);
	bool a5 = _isxdigit('a'); bool a5n = _isxdigit('A'); _assert(!a5 && !a5n);
	char a6 = _tolower('a'); char a6n = _tolower('A'); _assert(a6 == 'a' && a6n == 'a');
}

__global__ static void runtimeExample5(void *r)
{
	_runtimeSetHeap(r);
	array_t<char> name = "SkyM";
	name = "ScottP";
	char *a0 = name;
	size_t a0l = name.length; _assert(a0l == 6);
}

__global__ static void runtimeExample6(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	int a0 = _strcmp("Test", "Test"); _assert(!a0);
	int a1 = _strncmp("Tesa", "Tesb", 3); _assert(!a1);
	_memcpy(buf, "Test", 4);
	_memset(buf, 0, sizeof(buf));
	int a2 = _memcmp("Test", "Test", 4); _assert(!a2);
	int a3 = _strlen30("Test"); _assert(a3 == 4);
	int a4 = _hextobyte('a'); _assert(a4 == 10);
	bool a5 = _isnan(0.0); _assert(a5);
}

__global__ static void runtimeExample7(void *r)
{
	_runtimeSetHeap(r);
}

__global__ static void runtimeExample8(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	__snprintf(buf, sizeof(buf), "t0\n");
	__snprintf(buf, sizeof(buf), "t1 %s\n", "1");
	__snprintf(buf, sizeof(buf), "t2 %s %d\n", "1", 2);
	__snprintf(buf, sizeof(buf), "t3 %s %d %d\n", "1", 2, 3);
	__snprintf(buf, sizeof(buf), "t4 %s %d %d %d\n", "1", 2, 3, 4);
	__snprintf(buf, sizeof(buf), "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	__snprintf(buf, sizeof(buf), "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	__snprintf(buf, sizeof(buf), "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	__snprintf(buf, sizeof(buf), "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	__snprintf(buf, sizeof(buf), "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	__snprintf(buf, sizeof(buf), "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
}

__global__ static void runtimeExample9(void *r)
{
	_runtimeSetHeap(r);
	char *a0 = _mprintf("t0\n");
	char *a1 = _mprintf("t1 %s\n", "1");
	char *a2 = _mprintf("t2 %s %d\n", "1", 2);
	char *a3 = _mprintf("t3 %s %d %d\n", "1", 2, 3);
	char *a4 = _mprintf("t4 %s %d %d %d\n", "1", 2, 3, 4);
	char *a5 = _mprintf("t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	char *a6 = _mprintf("t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	char *a7 = _mprintf("t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	char *a8 = _mprintf("t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	char *a9 = _mprintf("t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	char *a10 = _mprintf("ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
	//
	char *b0 = _mtagprintf(nullptr, "t0\n");
	char *b1 = _mtagprintf(nullptr, "t1 %s\n", "1");
	char *b2 = _mtagprintf(nullptr, "t2 %s %d\n", "1", 2);
	char *b3 = _mtagprintf(nullptr, "t3 %s %d %d\n", "1", 2, 3);
	char *b4 = _mtagprintf(nullptr, "t4 %s %d %d %d\n", "1", 2, 3, 4);
	char *b5 = _mtagprintf(nullptr, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	char *b6 = _mtagprintf(nullptr, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	char *b7 = _mtagprintf(nullptr, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	char *b8 = _mtagprintf(nullptr, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	char *b9 = _mtagprintf(nullptr, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	char *b10 = _mtagprintf(nullptr, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
	//
	char *c0 = _mtagappendf(nullptr, nullptr, "t0\n");
	char *c1 = _mtagappendf(nullptr, nullptr, "t1 %s\n", "1");
	char *c2 = _mtagappendf(nullptr, nullptr, "t2 %s %d\n", "1", 2);
	char *c3 = _mtagappendf(nullptr, nullptr, "t3 %s %d %d\n", "1", 2, 3);
	char *c4 = _mtagappendf(nullptr, nullptr, "t4 %s %d %d %d\n", "1", 2, 3, 4);
	char *c5 = _mtagappendf(nullptr, nullptr, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	char *c6 = _mtagappendf(nullptr, nullptr, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	char *c7 = _mtagappendf(nullptr, nullptr, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	char *c8 = _mtagappendf(nullptr, nullptr, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	char *c9 = _mtagappendf(nullptr, nullptr, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	char *c10 = _mtagappendf(nullptr, nullptr, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
	//
	char *z;
	_mtagassignf(nullptr, &z, "t0\n");
	_mtagassignf(nullptr, &z, "t1 %s\n", "1");
	_mtagassignf(nullptr, &z, "t2 %s %d\n", "1", 2);
	_mtagassignf(nullptr, &z, "t3 %s %d %d\n", "1", 2, 3);
	_mtagassignf(nullptr, &z, "t4 %s %d %d %d\n", "1", 2, 3, 4);
	_mtagassignf(nullptr, &z, "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	_mtagassignf(nullptr, &z, "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	_mtagassignf(nullptr, &z, "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	_mtagassignf(nullptr, &z, "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	_mtagassignf(nullptr, &z, "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	_mtagassignf(nullptr, &z, "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
}

void __runtimeExample(cudaDeviceHeap &r)
{
	cudaDeviceHeapSelect(r);
	runtimeExample0<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample1<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample2<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample3<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample4<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample5<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample6<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample7<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample8<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
	runtimeExample9<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r);
}